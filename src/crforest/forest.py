"""Public CompetingRiskForest class."""

from __future__ import annotations

import functools
import warnings
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from crforest._binning import apply_bins, fit_bin_edges
from crforest._gpu_detect import detect_cuda
from crforest._hist_tree import build_tree_hist, predict_tree_hist, predict_tree_hist_chf
from crforest._importance import _compute_importance_impl
from crforest._sklearn_compat import (
    Surv,
    is_structured_survival_y,
    unpack_structured_y,
)
from crforest._time_grid import coarsen_time_grid, fit_time_grid
from crforest._tree import build_tree, predict_tree, predict_tree_chf
from crforest._validation import check_inputs
from crforest.metrics import concordance_index_cr

DEFAULT_SPLIT_NTIME = 10


@functools.lru_cache(maxsize=1)
def _detect_cuda_cached() -> tuple[bool, str]:
    """One-shot CUDA probe; cached for the lifetime of the process."""
    return detect_cuda()


class CompetingRiskForest(BaseEstimator):
    """Competing risks random forest.

    Both modes store compact per-cause event / at-risk counts at leaves
    and materialize CIF (Aalen-Johansen) / CHF (Nelson-Aalen) tables
    lazily on first predict. ``mode="default"`` uses histogram split
    search with uint8-binned features; ``mode="reference"`` uses
    pure-NumPy exact splitting with raw (unbinned) thresholds.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    max_depth : int, default=15
        Maximum tree depth.
    min_samples_split : int, default=6
        Minimum samples required to attempt a split at an internal node.
    min_samples_leaf : int, default=3
        Minimum samples in each child node after a split.
    max_features : {"sqrt", "log2", None}, int, or float, default="sqrt"
        Number of features considered at each split. "sqrt" and "log2"
        are rounded up; a float is interpreted as a fraction of
        ``n_features``; None uses all features.
    bootstrap : bool, default=True
        If True, each tree is built on a bootstrap sample drawn with
        replacement; otherwise each tree sees all training rows.
    random_state : int or None, default=None
        Seed for bootstrap sampling and mtry draws. If None, results are
        nondeterministic.
    mode : {"default", "reference"}, default="default"
        ``"default"`` uses the numba-jitted histogram split kernel with
        uint8-binned features (recommended for production fits).
        ``"reference"`` uses a pure-NumPy exact split search over raw
        feature values (slower; mainly used for equivalence checks).
    n_bins : int, default=256
        Number of histogram bins per feature; ignored in reference mode.
        Must be in [2, 256].
    time_grid : int, default=200
        Max points on the shared event-time grid for compact leaf storage;
        ignored in reference mode.
    n_jobs : int or None, default=-1
        Number of threads for parallel tree building. ``-1`` uses all
        available CPUs (as reported by joblib), ``1`` runs serially.
        ``None`` is accepted and treated as ``1`` (joblib convention).
        Output is bit-identical across ``n_jobs`` values for a fixed
        ``random_state``.

        Speedup applies only to ``mode="default"``, whose split kernel
        is numba-jitted and releases the GIL. ``mode="reference"`` is
        pure-Python recursion and GIL-bound; ``n_jobs > 1`` will not
        parallelize it.

        Joblib dispatch has sub-millisecond overhead per tree. For
        ``n_estimators`` under ~20 on small datasets, ``n_jobs=1`` may
        be faster than ``n_jobs=-1``.
    splitrule : {"logrankCR", "logrank"}, default="logrankCR"
        Split criterion. ``"logrankCR"`` is the composite competing-risks
        log-rank (pooled across causes with Lau-inclusive at-risk);
        matches rfSRC 3.6.1 ``SURV_CR_LAU``. ``"logrank"`` is the
        cause-specific log-rank (standard at-risk — competing-cause
        events remove the subject from the target-cause risk set);
        matches rfSRC 3.6.1 ``SURV_LR``.
    cause : int, default=1
        1-based cause index for ``splitrule="logrank"``. Ignored when
        ``splitrule="logrankCR"`` or when ``cause_weights`` is given.
    cause_weights : array-like of float or None, default=None
        Per-cause weight vector of length ``n_causes`` for
        ``splitrule="logrank"``. When supplied, uses the weighted
        combination ``(Σ_k w_k num_k)² / (Σ_k w_k² var_k)`` and
        ``cause`` is ignored. Supported in ``mode="reference"`` only;
        passing this in ``mode="default"`` raises
        ``NotImplementedError`` (see CHANGELOG note on weighted
        histogram kernel deferral).
    nsplit : int or None, default=None
        Number of random split-point draws per feature at each node.
        ``nsplit=0`` evaluates every candidate threshold exhaustively;
        ``nsplit>0`` samples that many distinct candidate split points
        without replacement from the observed unique values at the current
        node (excluding the maximum, which would leave an empty right
        child) — matches rfSRC's split-candidate sampling semantics.
        ``None`` resolves to ``10`` in ``mode="default"`` (rfSRC default)
        and ``0`` in ``mode="reference"`` (preserves exactness for CI
        ground truth).
    split_ntime : int or None, default=10
        Coarse time bins for split-search log-rank evaluation in
        ``mode="default"``. Leaves keep full ``time_grid_`` stats for
        CIF/CHF output. ``None`` disables coarsening (full time grid for
        splits). Ignored in ``mode="reference"``. For very small cohorts
        (n≲500) prefer ``50`` or ``None``: undersampling dominates
        coarsening loss when there are few unique event times.
    equivalence : {None, "rfsrc"}, default=None
        Preset for cross-library predictive alignment. ``"rfsrc"`` sets
        ``rng_mode="rfsrc_aligned"``, disables split-scoring time-grid
        coarsening (``split_ntime=None``), and removes the ``time_grid``
        cap so all unique event times are used — matching rfSRC's default
        behaviour. Costs ~2-3x fit time vs the default numpy RNG path.
        Requires an explicit ``random_state``. Conflicts with explicit
        ``rng_mode`` / ``split_ntime`` raise at fit time.

        To achieve **bit-identical trees** vs rfSRC under ``bootstrap=False``,
        use these parameter mappings::

            rfSRC parameter       crforest parameter
            -----------------     --------------------------------
            nodesize=K         -> min_samples_split=2*K, min_samples_leaf=1
            samp=matrix(1L,...) -> bootstrap=False
            ntime=0            -> handled internally (all event times used)
            (no max-depth)     -> max_depth=None

        rfSRC's ``nodesize`` is a parent-min-size constraint: both children
        must reach ``K`` observations.  crforest matches this with
        ``min_samples_split=2*K`` (guarantees each child can reach K) and
        ``min_samples_leaf=1`` (removes crforest's own child-size floor).

        Known limitation: under ``bootstrap=True`` a residual ~0.003 p95
        ΔCIF persists because rfSRC consumes an additional RNG stream during
        bootstrap book-keeping (SUN-44 tracks the Phase 1d fix). For
        bit-identity, set ``bootstrap=False``.
    device : {"auto", "cpu", "cuda"}, default="auto"
        Compute backend for the flat-tree path. In v0.1, ``"auto"`` resolves
        to ``"cpu"`` — the cuda backend is shipped as a preview and is
        faster only at low feature count (p ≲ 20); at typical clinical
        workloads (p ≈ 58) it is ~1.15x slower than cpu on the same machine
        because host orchestration dominates the single-tree wall. Pass
        ``device="cuda"`` explicitly to opt into the GPU path; the full
        GPU rewrite is scheduled for v1.1. ``"cuda"`` requires the optional
        ``crforest[gpu]`` install and is incompatible with
        ``equivalence="rfsrc"`` / ``rng_mode="rfsrc_aligned"``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: str | int | float | None = "sqrt",
        bootstrap: bool = True,
        random_state: int | None = None,
        mode: str = "default",
        n_bins: int = 256,
        time_grid: int = 200,
        n_jobs: int = -1,
        splitrule: str = "logrankCR",
        cause: int = 1,
        cause_weights: np.ndarray | None = None,
        nsplit: int | None = None,
        split_ntime: int | None = DEFAULT_SPLIT_NTIME,
        rng_mode: str = "numpy",
        equivalence: str | None = None,
        device: Literal["auto", "cpu", "cuda"] = "auto",
    ):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"device must be one of 'auto'/'cpu'/'cuda'; got {device!r}")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.mode = mode
        self.n_bins = n_bins
        self.time_grid = time_grid
        self.n_jobs = n_jobs
        self.splitrule = splitrule
        self.cause = cause
        self.cause_weights = cause_weights
        self.nsplit = nsplit
        self.split_ntime = split_ntime
        self.rng_mode = rng_mode
        self.equivalence = equivalence
        self.device = device

    def fit(self, X, time, event=None):
        # Dual signature for sklearn drop-in compatibility:
        # - fit(X, time, event): legacy three-positional, two 1-D arrays
        # - fit(X, y) where y is structured with 'time' and 'event' fields:
        #   sklearn-friendly form, used by cross_val_score / Pipeline / etc.
        if event is None:
            time, event = unpack_structured_y(time)
        if self.mode not in ("default", "reference"):
            raise ValueError(f"mode must be 'default' or 'reference'; got {self.mode!r}")
        if self.splitrule not in ("logrankCR", "logrank"):
            raise ValueError(f"splitrule must be 'logrankCR' or 'logrank'; got {self.splitrule!r}")
        # Resolve `equivalence` preset into the per-fit effective rng_mode /
        # split_ntime / time_grid_max. Validates against explicit conflicting
        # values; the public attributes stay untouched so a second .fit() with
        # the same constructor args is reproducible.
        (
            self._rng_mode_eff_,
            self._split_ntime_eff_,
            self._time_grid_max_eff_,
        ) = self._resolve_equivalence()
        if self.nsplit is None:
            self._resolved_nsplit_ = 10 if self.mode == "default" else 0
        else:
            if self.nsplit < 0:
                raise ValueError(f"nsplit must be >= 0; got {self.nsplit}")
            self._resolved_nsplit_ = int(self.nsplit)
        X, time, event, n_causes = check_inputs(X, time, event)
        self.n_causes_ = n_causes
        self.n_features_in_ = X.shape[1]
        max_features = self._resolve_max_features(self.n_features_in_)

        if self.cause_weights is not None:
            cw = np.asarray(self.cause_weights, dtype=np.float64)
            if cw.ndim != 1 or len(cw) != n_causes:
                raise ValueError(
                    f"cause_weights must be a 1-D vector of length n_causes={n_causes}; "
                    f"got shape {cw.shape}"
                )
            self._cause_weights_arr = cw
        else:
            self._cause_weights_arr = None

        if self.mode == "reference":
            self._fit_reference(X, time, event, max_features)
        else:
            self._fit_default(X, time, event, max_features)
        if self.bootstrap:
            self._X_train_oob_ = X
            self._y_train_oob_ = Surv.from_arrays(event=event, time=time)
        else:
            self._X_train_oob_ = None
            self._y_train_oob_ = None
        return self

    def _fit_reference(self, X, time, event, max_features):
        self._effective_device_ = "cpu"
        n = X.shape[0]
        self.unique_times_ = np.sort(np.unique(time))

        def build_one(idx, tree_rng):
            return build_tree(
                X[idx],
                time[idx],
                event[idx],
                n_causes=self.n_causes_,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                unique_times=self.unique_times_,
                max_features=max_features,
                rng=tree_rng,
                splitrule=self.splitrule,
                cause=self.cause,
                cause_weights=self._cause_weights_arr,
                nsplit=self._resolved_nsplit_,
            )

        self.trees_, self.oob_indices_ = self._build_ensemble(n, build_one)
        return self

    def _fit_default(self, X, time, event, max_features):
        if self.splitrule == "logrank" and self._cause_weights_arr is not None:
            raise NotImplementedError(
                "splitrule='logrank' with cause_weights is only available in "
                "mode='reference'; the weighted histogram kernel is not yet implemented."
            )
        if self.n_causes_ > 255:
            raise ValueError(
                f"crforest supports up to 255 competing causes in histogram mode; "
                f"got n_causes={self.n_causes_}. Please file an issue if you need higher."
            )
        if self.time_grid > 65_535:
            raise ValueError(
                f"time_grid={self.time_grid} exceeds maximum supported time bins "
                "(65,535). Reduce the `time_grid` parameter."
            )
        n = X.shape[0]
        self.bin_edges_ = fit_bin_edges(X, n_bins=self.n_bins)
        X_binned = apply_bins(X, self.bin_edges_)
        # Under equivalence='rfsrc', _time_grid_max_eff_ is None → no cap on the
        # event-time grid (rfSRC uses all unique event times). A cap here would
        # produce coarser split candidates → different best splits → non-identical
        # trees.  For the normal path, _time_grid_max_eff_ == self.time_grid (an int).
        _tg_max = self._time_grid_max_eff_ if self._time_grid_max_eff_ is not None else 2**31
        self.time_grid_ = fit_time_grid(time, event, max_points=_tg_max)
        self.unique_times_ = self.time_grid_
        n_time_bins_full = len(self.time_grid_)
        t_idx_full = np.clip(
            np.searchsorted(self.time_grid_, time, side="right") - 1,
            0,
            n_time_bins_full - 1,
        ).astype(np.int32)

        if self._split_ntime_eff_ is None or self._split_ntime_eff_ >= n_time_bins_full:
            t_idx_split = t_idx_full
            n_time_bins_split = n_time_bins_full
        else:
            full_to_split = coarsen_time_grid(self.time_grid_, self._split_ntime_eff_)
            t_idx_split = full_to_split[t_idx_full]
            n_time_bins_split = int(self._split_ntime_eff_)

        rng_mode = self._rng_mode_eff_

        # Flat-tree path is the default for rng_mode='numpy'. The rfsrc-
        # aligned path keeps the legacy HistTreeNode flow for bit-identity
        # with rfSRC.
        use_flat_path = rng_mode == "numpy"

        # Validate device against the resolved path BEFORE backend resolution
        # so the error message names the actual conflict (rng_mode/equivalence)
        # rather than later silently falling through to cpu on the legacy path.
        # Symmetric with how _select_backend raises when device='cuda' but cuda
        # is unavailable.
        if self.device == "cuda" and not use_flat_path:
            raise ValueError(
                "device='cuda' requires the flat-tree path "
                "(rng_mode='numpy' / no equivalence='rfsrc'); "
                f"got rng_mode={self._rng_mode_eff_!r}, equivalence={self.equivalence!r}"
            )

        if use_flat_path:
            backend = self._select_backend()
            self._effective_device_ = backend
            if backend == "cuda":
                from crforest._gpu_kernels import build_flat_tree_gpu as _flat_builder

                # Default n_jobs=-1 silently coerces to 1 (single-GPU is serial).
                # Only warn when the user *explicitly* asked for multi-core CPU
                # parallelism (>1) — that's a real intent mismatch worth surfacing.
                if self.n_jobs not in (None, 1, -1):
                    warnings.warn(
                        "n_jobs ignored on cuda backend (single GPU = serial trees)",
                        stacklevel=2,
                    )
                n_jobs_local = 1
            else:
                from crforest._flat_tree_builder import build_flat_tree as _flat_builder

                n_jobs_local = self.n_jobs

            splitrule_code = 0 if self.splitrule == "logrankCR" else 1
            # max_features=None means "use all features" in the flat-tree path
            # (numba can't type None). Hoist event-cast outside the closure so
            # the per-tree fancy index doesn't re-allocate the event array.
            _flat_max_features = X_binned.shape[1] if max_features is None else max_features
            _event_i32 = event if event.dtype == np.int32 else event.astype(np.int32, copy=False)

            def build_one_flat(idx, seed):
                # Split kernel uses the coarse grid (n_time_bins_split) for speed;
                # leaf accumulation uses the full grid (n_time_bins_full) so that
                # leaf CIFs align with time_grid_ for prediction.
                return _flat_builder(
                    X_binned[idx],
                    t_idx_split[idx],
                    t_idx_full[idx],
                    _event_i32[idx],
                    bootstrap_indices=np.arange(len(idx), dtype=np.int32),
                    n_bins=self.n_bins,
                    n_causes=self.n_causes_,
                    n_time_bins_split=n_time_bins_split,
                    n_time_bins_full=n_time_bins_full,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_depth=-1 if self.max_depth is None else self.max_depth,
                    max_features=_flat_max_features,
                    nsplit=self._resolved_nsplit_,
                    splitrule_code=splitrule_code,
                    cause=self.cause,
                    seed=int(seed),
                )

            builder = build_one_flat
        else:
            self._effective_device_ = "cpu"
            n_jobs_local = self.n_jobs
            # Legacy HistTreeNode path: split kernel can run batched (across
            # features in one njit call) when split_ntime is coarser than the
            # full grid. rfsrc_aligned mode requires per-feature interleaving
            # so it never uses the batched kernel.
            use_batched = n_time_bins_split < n_time_bins_full
            aligned_use_batched = False if rng_mode == "rfsrc_aligned" else use_batched

            def build_one(idx, tree_rng):
                return build_tree_hist(
                    X_binned[idx],
                    t_idx_split[idx],
                    event[idx],
                    n_causes=self.n_causes_,
                    n_bins=self.n_bins,
                    n_time_bins=n_time_bins_split,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=max_features,
                    rng=tree_rng,
                    splitrule=self.splitrule,
                    cause=self.cause,
                    nsplit=self._resolved_nsplit_,
                    time_indices_full=t_idx_full[idx],
                    n_time_bins_full=n_time_bins_full,
                    use_batched=aligned_use_batched,
                    rng_mode=rng_mode,
                )

            builder = build_one

        self.trees_, self.oob_indices_ = self._build_ensemble(
            n, builder, use_flat_path=use_flat_path, n_jobs=n_jobs_local
        )
        return self

    def _build_ensemble(
        self, n, build_one, *, use_flat_path: bool = False, n_jobs: int | None = None
    ):
        if self._rng_mode_eff_ not in ("numpy", "rfsrc_aligned"):
            raise ValueError(
                f"rng_mode must be 'numpy' or 'rfsrc_aligned'; got {self._rng_mode_eff_!r}"
            )
        rng = np.random.RandomState(self.random_state)
        # Pre-derive per-tree "stream B" RNGs for rfsrc_aligned mode. In numpy mode
        # the per-tree RNG is seeded inside the loop (interleaved with bootstrap
        # draw) to preserve the historical RNG consumption order; changing that
        # order would change bootstrap indices even when rng_mode is numpy.
        rfsrc_tree_rngs: list | None = None
        if self._rng_mode_eff_ == "rfsrc_aligned":
            if self.random_state is None:
                raise ValueError(
                    "rng_mode='rfsrc_aligned' requires an explicit random_state "
                    "(rfSRC derives per-tree seeds deterministically from the user seed)."
                )
            from crforest._aligned_rng import AlignedRng, derive_per_tree_seeds

            seeds_b = derive_per_tree_seeds(int(self.random_state), self.n_estimators)
            rfsrc_tree_rngs = [AlignedRng(int(s)) for s in seeds_b]

        prepared = []
        # Bootstrap draws always use numpy MT in both modes (stream A alignment is
        # empirically a ~1.5% effect on cross-lib gap; see bootstrap_aligned_spike).
        # Keeping bootstrap on numpy lets rng_mode='rfsrc_aligned' isolate the
        # stream-B contribution (mtry + nsplit candidate subsetting).
        for i in range(self.n_estimators):
            idx, oob = self._sample_indices(rng, n)
            if rfsrc_tree_rngs is not None:
                tree_rng = rfsrc_tree_rngs[i]
                tree_seed = int(rng.randint(0, 2**31))  # advance rng identically to numpy mode
            else:
                tree_seed = int(rng.randint(0, 2**31))
                tree_rng = np.random.RandomState(tree_seed)
            tree_arg = tree_seed if use_flat_path else tree_rng
            prepared.append((idx, oob, tree_arg))

        effective_n_jobs = self.n_jobs if n_jobs is None else n_jobs
        trees = Parallel(n_jobs=effective_n_jobs, prefer="threads")(
            delayed(build_one)(idx, tree_rng) for idx, _, tree_rng in prepared
        )
        oob_indices = [oob for _, oob, _ in prepared]
        # Per-tree inbag counts (n, ntree) int32 — feeds rfSRC's
        # bootstrap="by.user", samp=<this matrix> for cross-lib paired fits.
        # Only populated under equivalence="rfsrc" to keep pickle size bounded
        # for users who don't need cross-lib parity.
        if self.equivalence == "rfsrc" and self.bootstrap:
            self.inbag_ = np.column_stack(
                [np.bincount(idx, minlength=n).astype(np.int32) for idx, _, _ in prepared]
            )
        else:
            self.inbag_ = None
        return trees, oob_indices

    def _select_backend(self) -> str:
        """Resolve ``device`` to ``'cpu'`` or ``'cuda'``. v0.1: ``auto`` → cpu (see class docstring)."""
        if self.device == "cuda":
            available, reason = _detect_cuda_cached()
            if not available:
                raise RuntimeError(f"device='cuda' but {reason}")
            return "cuda"
        return "cpu"

    def _resolve_equivalence(self) -> tuple[str, int | None, int | None]:
        """Resolve the ``equivalence`` preset into effective rng_mode + split_ntime + time_grid_max.

        Returns
        -------
        rng_mode_eff : str
        split_ntime_eff : int or None
        time_grid_max_eff : int or None
            Maximum grid points passed to ``fit_time_grid``. ``None`` means no cap
            (use all unique event times), which rfSRC does by default.
        """
        if self.equivalence is None:
            return self.rng_mode, self.split_ntime, self.time_grid
        if self.equivalence != "rfsrc":
            raise ValueError(f"equivalence must be None or 'rfsrc'; got {self.equivalence!r}")
        if self.rng_mode not in ("numpy", "rfsrc_aligned"):
            raise ValueError(
                f"equivalence='rfsrc' is incompatible with rng_mode={self.rng_mode!r}; "
                "either drop equivalence or set rng_mode='rfsrc_aligned'"
            )
        if self.split_ntime not in (None, DEFAULT_SPLIT_NTIME):
            raise ValueError(
                f"equivalence='rfsrc' requires split_ntime in "
                f"{{None, {DEFAULT_SPLIT_NTIME} (default)}} "
                "so the preset can disable time-grid coarsening; "
                f"got split_ntime={self.split_ntime!r}"
            )
        # None = no cap: rfSRC uses all unique event times; cap to 200 would cause
        # coarser splits → different best-split decisions → non-identical trees.
        return "rfsrc_aligned", None, None

    def _sample_indices(self, rng: np.random.RandomState, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Draw bootstrap and out-of-bag indices for one tree.

        Returns (idx, oob). If ``bootstrap=False``, idx is ``arange(n)`` and
        oob is empty.
        """
        if self.bootstrap:
            idx = rng.choice(n, size=n, replace=True)
            oob = np.setdiff1d(np.arange(n), idx, assume_unique=False)
        else:
            idx = np.arange(n)
            oob = np.empty(0, dtype=np.int64)
        return idx, oob

    def predict_cif(self, X, times=None) -> np.ndarray:
        """Predict cause-specific cumulative incidence (Aalen-Johansen), averaged across trees.

        Returns
        -------
        cif : ndarray, shape (n_samples, n_causes, n_times), float64
            Ensemble mean of per-tree leaf Aalen-Johansen CIFs. When ``times``
            is provided, values are step-forward interpolated onto the
            requested grid (right-continuous step function; 0 before the
            first observed event time).
        """
        return self._predict_quantity(X, times, predict_tree, predict_tree_hist)

    def predict_chf(self, X, times=None) -> np.ndarray:
        """Predict cause-specific cumulative hazard (Nelson-Aalen), averaged across trees.

        Returns
        -------
        chf : ndarray, shape (n_samples, n_causes, n_times), float64
            Ensemble mean of per-tree leaf Nelson-Aalen CHFs. When ``times``
            is provided, values are step-forward interpolated onto the
            requested grid (right-continuous step function; 0 before the
            first observed event time). On the default flat-tree path the
            CHF leaf table is materialised lazily on first call from raw
            counts persisted at fit time, then cached on the tree.
        """
        return self._predict_quantity(X, times, predict_tree_chf, predict_tree_hist_chf)

    def _prepare_predict_inputs(self, X, ref_fn, hist_fn):
        """Validate X, bin if needed, and pick the per-tree predictor for ``mode``."""
        check_is_fitted(self, "trees_")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D; got ndim={X.ndim}")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has wrong n_features: expected {self.n_features_in_}, got {X.shape[1]}"
            )
        if self.mode == "default":
            return apply_bins(X, self.bin_edges_), hist_fn
        return X, ref_fn

    def _ensemble_mean(self, X_input, predict_fn, project) -> np.ndarray:
        """Apply ``project`` to each tree's prediction, sum, divide by ntree.

        ``project`` is invoked per tree so the accumulator only ever holds
        the projected shape — letting callers shrink ``(n, n_causes, n_time)``
        scratch to ``(n, n_causes, len(times))`` or ``(n,)`` without
        materialising the full ensemble tensor.
        """
        total = project(predict_fn(self.trees_[0], X_input)).astype(np.float64)
        for tree in self.trees_[1:]:
            total += project(predict_fn(tree, X_input))
        return total / len(self.trees_)

    def _predict_quantity(self, X, times, ref_predict_fn, hist_predict_fn) -> np.ndarray:
        """Tree-ensemble average for per-leaf quantities (CIF, CHF).

        When ``times`` is provided, the time-axis projection is applied
        per-tree before accumulating; bit-equivalent to accumulate-then-
        interpolate (linear indexing commutes with summation).
        """
        X_input, predict_fn = self._prepare_predict_inputs(X, ref_predict_fn, hist_predict_fn)
        project = (
            (lambda arr: arr)
            if times is None
            else self._make_time_projection(np.asarray(times, dtype=np.float64))
        )
        return self._ensemble_mean(X_input, predict_fn, project)

    def _make_time_projection(self, times: np.ndarray):
        """Build a closure that projects ``(n, n_causes, n_time_full)`` arrays
        onto ``times`` via right-continuous step interpolation. Indices are
        precomputed once so per-tree projection is a single fancy-index op."""
        idx = np.searchsorted(self.unique_times_, times, side="right") - 1
        take = np.clip(idx, 0, None)
        before = idx < 0
        before_any = bool(before.any())

        def _project(arr):
            out = arr[:, :, take]
            if before_any:
                out[:, :, before] = 0.0
            return out

        return _project

    def predict_risk(self, X, cause: int = 1, kind: str = "integrated_chf") -> np.ndarray:
        """Per-subject risk scalar for cause-specific concordance scoring.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        cause : int, default 1
            Cause of interest (1..n_causes_).
        kind : {"integrated_chf", "cif_last"}, default "integrated_chf"
            Risk scalar derived from the per-subject CIF/CHF curve:

            - ``"integrated_chf"`` (default) — sum of cause-specific CHF
              over the training time grid. Mirrors rfSRC's
              ``predict$predicted[, cause]`` convention. Better Harrell
              C-index discrimination than ``cif_last`` when many subjects
              share similar end-of-followup CIF but different hazard
              accumulation paths.
            - ``"cif_last"`` — CIF_cause at the last training time. Use
              when you want an absolute "probability of cause-k event by
              end of follow-up" instead of a ranking score.

            For Uno IPCW C-index neither scalar dominates.

        Returns
        -------
        risk : ndarray, shape (n_samples,), float64
        """
        check_is_fitted(self, "trees_")
        if cause < 1 or cause > self.n_causes_:
            raise ValueError(f"cause={cause} out of range [1, {self.n_causes_}]")
        if kind not in ("cif_last", "integrated_chf"):
            raise ValueError(f"kind must be 'cif_last' or 'integrated_chf'; got {kind!r}")
        if kind == "cif_last":
            return self.predict_cif(X, times=self.unique_times_[[-1]])[:, cause - 1, 0]
        # integrated_chf: per-tree time-axis sum keeps the accumulator at (n,)
        # instead of materialising the (n, n_causes, n_time) ensemble tensor.
        X_input, predict_fn = self._prepare_predict_inputs(
            X, predict_tree_chf, predict_tree_hist_chf
        )
        c = cause - 1
        return self._ensemble_mean(X_input, predict_fn, lambda arr: arr[:, c, :].sum(axis=-1))

    def predict_oob_risk(self, cause: int = 1) -> np.ndarray:
        """Per-row OOB ensemble integrated-CHF risk on the training set.

        For each training row, averages cause-specific integrated CHF over
        only the trees where that row was out-of-bag. Mirrors rfSRC's
        ``predict$predicted.oob[, cause]`` convention. Requires
        ``bootstrap=True`` at fit time.

        Rows in-bag for every tree (probability ~0.37**n_estimators, i.e.
        vanishingly small for n_estimators >= 100) get a risk of 0.
        """
        check_is_fitted(self, "trees_")
        if not self.bootstrap:
            raise ValueError("predict_oob_risk needs bootstrap=True at fit time")
        if cause < 1 or cause > self.n_causes_:
            raise ValueError(f"cause={cause} out of range [1, {self.n_causes_}]")
        X_train = getattr(self, "_X_train_oob_", None)
        if X_train is None:
            raise ValueError("training cache missing — refit the forest with bootstrap=True")
        from crforest._importance import _ensemble_oob_predictions

        bin_edges = getattr(self, "bin_edges_", None)
        pred, count = _ensemble_oob_predictions(
            self,
            np.asarray(X_train, dtype=np.float64),
            causes=[cause],
            bin_edges=bin_edges,
            time_grid=self.unique_times_,
        )
        return pred[0] / np.maximum(count, 1)

    def oob_score(self, cause: int = 1) -> float:
        """OOB Harrell C-index on the training set for ``cause``.

        Computed against the cached training outcomes using the OOB
        integrated-CHF risk from :meth:`predict_oob_risk`. Requires
        ``bootstrap=True`` at fit time.
        """
        risk = self.predict_oob_risk(cause=cause)
        time, event = unpack_structured_y(self._y_train_oob_)
        return float(concordance_index_cr(event, time, risk, cause=cause))

    def score(self, X, time, event=None, cause: int = 1, kind: str = "integrated_chf") -> float:
        """Cause-specific Harrell C-index. ``kind`` forwards to predict_risk.

        Accepts either the three-positional legacy form ``score(X, time, event)``
        or the sklearn-friendly ``score(X, y)`` where ``y`` is a structured
        array with ``time`` and ``event`` fields (see :class:`crforest.Surv`).
        """
        check_is_fitted(self, "trees_")
        if event is None:
            time, event = unpack_structured_y(time)
        risk = self.predict_risk(X, cause=cause, kind=kind)
        return concordance_index_cr(event, time, risk, cause=cause)

    def predict(self, X) -> np.ndarray:
        """sklearn-style alias for ``predict_risk(X, cause=1)``.

        Returned shape ``(n_samples,)``. The cause-1 default lets crforest
        slot into ``Pipeline`` / ``cross_val_predict`` without a wrapper;
        for cause-k risk or for CIF / CHF curves, call
        :meth:`predict_risk` / :meth:`predict_cif` / :meth:`predict_chf`
        directly.
        """
        return self.predict_risk(X, cause=1)

    def _resolve_max_features(self, n_features: int) -> int | None:
        mf = self.max_features
        if mf is None:
            return None
        if mf == "sqrt":
            return max(1, int(np.ceil(n_features**0.5)))
        if mf == "log2":
            return max(1, int(np.ceil(np.log2(n_features))))
        if isinstance(mf, int):
            return max(1, min(mf, n_features))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        raise ValueError(f"Invalid max_features={mf!r}. Use None, 'sqrt', 'log2', int, or float.")

    def compute_importance(
        self,
        X_eval=None,
        y_eval=None,
        *,
        causes: list[int] | None = None,
        n_repeats: int = 5,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        """Compute per-cause + composite permutation variable importance.

        Two flavours, dispatched by whether an evaluation set is supplied:

        * **OOB Breiman** (``X_eval=None`` and ``y_eval=None``): scored on
          the cached training data using the Uno IPCW C-index over each
          tree's out-of-bag rows. Requires ``bootstrap=True`` at fit time.
        * **Held-out**: standard sklearn permutation importance with a
          per-cause Wolbers-C-index scorer.

        Parameters
        ----------
        X_eval : array-like, shape (n_samples, n_features), optional
            Held-out features. If both ``X_eval`` and ``y_eval`` are None,
            OOB importance is computed instead.
        y_eval : structured array with fields ``time`` and ``event``, optional
            Held-out survival outcomes.
        causes : list of int, optional
            Causes to score. Defaults to ``range(1, n_causes_ + 1)``.
        n_repeats : int, default 5
            ``sklearn.inspection.permutation_importance`` n_repeats.
            Held-out mode only; ignored in OOB mode (single permutation
            per tree per feature).
        random_state : int, optional
            Seed for permutation draws (reproducibility).
        n_jobs : int, optional
            Override ``forest.n_jobs`` for the per-feature parallel layer
            in OOB mode.

        Returns
        -------
        pd.DataFrame with columns ``feature``, ``cause_{k}_vimp`` for each
        fitted cause in numeric order, and ``composite_vimp``.

        Raises
        ------
        ValueError
            OOB mode requires ``bootstrap=True`` at fit time.
        TypeError
            Held-out mode requires ``y_eval`` to be a structured array with
            ``time`` and ``event`` fields.

        Notes
        -----
        VIMP scales as ``n_features * n_repeats * n_causes`` calls to
        ``predict_risk`` in held-out mode (each call walks every tree in
        Python). For wide cohorts the wall time can be material; downsample
        ``X_eval`` or use OOB mode if cost is a concern.
        """
        check_is_fitted(self, "trees_")
        if X_eval is None and y_eval is None:
            from crforest._importance import _compute_importance_oob_impl

            resolved_causes = (
                list(causes) if causes is not None else list(range(1, self.n_causes_ + 1))
            )
            df = _compute_importance_oob_impl(
                self,
                causes=resolved_causes,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            self._feature_importances_cache = df
            return df
        if X_eval is None:
            raise ValueError("X_eval is None but y_eval is not; pass both or neither.")
        if y_eval is None:
            raise ValueError("y_eval is None but X_eval is not; pass both or neither.")
        y_arr = np.asarray(y_eval)
        if not is_structured_survival_y(y_arr):
            raise TypeError(
                "y_eval must be a structured array with 'time' and 'event' fields; "
                "see CompetingRiskForest.compute_importance docstring."
            )
        resolved_causes = list(causes) if causes is not None else list(range(1, self.n_causes_ + 1))
        feature_names = self._importance_feature_names()
        df = _compute_importance_impl(
            self,
            np.asarray(X_eval, dtype=np.float64),
            y_arr,
            feature_names=feature_names,
            causes=resolved_causes,
            cause_weights=self._cause_weights_arr,
            n_repeats=n_repeats,
            random_state=random_state,
        )
        self._feature_importances_cache = df
        return df

    def minimal_depth(
        self,
        threshold: str = "md",
        *,
        return_extra: bool = False,
    ):
        """Ishwaran-style minimal-depth variable selection.

        A variable's *minimal depth* in a tree is the depth of the highest
        split that uses it (root = depth 0). Variables never split on get
        a sentinel depth of ``D_T`` where ``D_T`` is the tree's max depth
        (Ishwaran et al. 2010, JASA, Eq. (2)). Smaller mean minimal depth
        across the forest indicates a more important variable.

        The selection threshold is E[D*v] computed once from the
        forest-averaged node-count vector l_bar*_d and average tree depth
        D_bar, per Ishwaran et al. (2010, JASA, Theorem 1 + Section 3).

        Parameters
        ----------
        threshold : {"md"}, default "md"
            Selection threshold method. Only forest-averaged ``"md"`` (the
            paper's recommendation, Ishwaran et al. 2010, JASA, Section 3)
            is supported in v0.3.0.
        return_extra : bool, default False
            If True, additionally include ``min_depth_q25``,
            ``min_depth_q75``, ``frac_trees_used`` columns.

        Returns
        -------
        pandas.DataFrame
            Sorted ascending by ``mean_min_depth``. Columns:
            ``feature``, ``mean_min_depth``, ``threshold``, ``selected``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the forest has not been fitted.
        ValueError
            If ``threshold`` is not ``"md"``.

        Notes
        -----
        The threshold uses the paper's recommended forest-averaging (Section 3),
        not tree-averaged E[md_T]. rfSRC defaults to tree-averaged aggregation,
        so numeric threshold values may differ from rfSRC even with
        ``equivalence='rfsrc'``.
        """
        check_is_fitted(self, "trees_")
        from crforest._minimal_depth import compute_minimal_depth

        return compute_minimal_depth(
            self,
            threshold=threshold,
            return_extra=return_extra,
        )

    @property
    def feature_importances_(self):
        """Cached result of the last ``compute_importance`` call.

        Raises
        ------
        AttributeError
            If ``compute_importance`` has not been called.
        """
        if not hasattr(self, "_feature_importances_cache"):
            raise AttributeError(
                "feature_importances_ requires a prior compute_importance(X_eval, y_eval) call."
            )
        return self._feature_importances_cache

    def _importance_feature_names(self) -> list[str]:
        """Feature names for VIMP output; positional fallback when unset."""
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            return [f"feature_{i}" for i in range(self.n_features_in_)]
        return list(names)
