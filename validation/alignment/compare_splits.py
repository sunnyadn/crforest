"""Per-feature split-threshold comparison harness: crforest vs rfSRC.

Covers both splitrules:
- ``splitrule="logrankCR"`` — composite log-rank (SURV_CR_LAU in rfSRC).
- ``splitrule="logrank"``   — cause-specific log-rank (SURV_LR in rfSRC).

Uses one rfSRC fit per feature (``nsplit=0``, ``bootstrap="none"``,
``nodedepth=1``) to read rfSRC's chosen root split threshold, and compares
against crforest's argmax across midpoint candidates.
Consumers: ``tests/test_rfsrc_split_equivalence.py`` and
``validation.alignment.gen_fixtures``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crforest._splits import (
    bin_times,
    cause_specific_log_rank_statistic,
    composite_log_rank_statistic,
)


def toy_input(
    seed: int,
    n: int = 30,
    n_features: int = 3,
    n_causes: int = 2,
) -> dict:
    """Generate a small, deterministic competing-risks dataset for alignment checks.

    Returns a dict with keys X (float64 array (n, n_features)), time (float64 (n,)),
    event (int64 (n,) with codes 0..n_causes), and n_causes (int).

    Construction: X ~ N(0, 1); baseline hazards h_k depend on a linear
    function of X; event time is the minimum of per-cause exponential draws;
    censoring times are exponential with rate equal to 1 / median event
    time.  A rejection loop retries up to 32 times if a draw produces no
    censoring or no events of some cause, so the regression guard in the
    tests is stable.
    """
    rng = np.random.default_rng(seed)
    for _ in range(32):  # retry up to 32 times to avoid pathological draws
        X = rng.standard_normal((n, n_features))
        beta = np.arange(1, n_causes + 1, dtype=np.float64)  # coefficient per cause
        cause_times = np.empty((n, n_causes), dtype=np.float64)
        for k in range(n_causes):
            rate = np.exp(0.5 * (X[:, k % n_features] * beta[k]))
            cause_times[:, k] = rng.exponential(scale=1.0 / rate)
        first_cause = np.argmin(cause_times, axis=1)
        event_time = cause_times[np.arange(n), first_cause]
        censor_time = rng.exponential(scale=np.median(event_time), size=n)
        observed = np.minimum(event_time, censor_time)
        is_event = event_time <= censor_time
        event = np.where(is_event, first_cause + 1, 0).astype(np.int64)
        if (event == 0).any() and all((event == k).any() for k in range(1, n_causes + 1)):
            return {
                "X": X.astype(np.float64),
                "time": observed.astype(np.float64),
                "event": event,
                "n_causes": int(n_causes),
            }
    raise RuntimeError(f"toy_input(seed={seed}) failed to produce a well-conditioned draw")


def rfsrc_per_feature_best_split(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    splitrule: str = "logrankCR",
    cause: int | None = None,
) -> pd.DataFrame:
    """Per-feature root split threshold chosen by rfSRC (via rpy2).

    For each feature column, fits a single-column depth-1 rfSRC with
    ``nsplit=0`` (exhaustive), ``bootstrap="none"``, and the given
    ``splitrule``, then reads the root split threshold from
    ``fit$forest$nativeArray$contPT``.  Returns a frame with columns
    ``(feature, best_threshold)``.

    rfSRC stores ``contPT`` as the lower sorted-unique-X value (``x[j]``);
    crforest reports the midpoint ``(x[j] + x[j+1]) / 2``.  Both describe
    the same sample partition — this function converts to the midpoint
    convention so frame-level comparisons match exactly.

    Parameters
    ----------
    X, time, event:
        Arrays as returned by ``toy_input``.
    splitrule:
        ``"logrankCR"`` (composite, default) or ``"logrank"`` (cause-specific).
    cause:
        Required when ``splitrule="logrank"``; the cause index (1-based) to
        optimise.  Ignored for ``"logrankCR"``.

    Requires rpy2 and the randomForestSRC R package.
    """
    if splitrule == "logrank" and cause is None:
        raise ValueError("cause must be provided when splitrule='logrank'")
    if splitrule not in ("logrankCR", "logrank"):
        raise ValueError(f"Unsupported splitrule: {splitrule!r}; use 'logrankCR' or 'logrank'")

    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr

    from validation.alignment import _rpy2_converter

    importr("randomForestSRC")  # ensure loaded in the R session
    converter = _rpy2_converter()

    rows = []
    for feat in range(X.shape[1]):
        try:
            r_df = pd.DataFrame(
                {
                    "x": X[:, feat],
                    "time": time,
                    "event": event.astype(np.int32),
                }
            )
            # Push the frame in via localconverter, but run the fit via a
            # plain ro.r string eval so the returned rfsrc object keeps its
            # R class attributes (nativeArray access needs them).
            with localconverter(converter):
                ro.globalenv["df"] = r_df

            if splitrule == "logrankCR":
                r_call = (
                    "fit__ <- rfsrc(Surv(time, event) ~ x, data = df, "
                    'ntree = 1, nsplit = 0, bootstrap = "none", '
                    'splitrule = "logrankCR", '
                    "nodedepth = 1, nodesize = 1, seed = -1)"
                )
            else:  # logrank
                r_call = (
                    f"fit__ <- rfsrc(Surv(time, event) ~ x, data = df, "
                    f'ntree = 1, nsplit = 0, bootstrap = "none", '
                    f'splitrule = "logrank", cause = {int(cause)}, '
                    f"nodedepth = 1, nodesize = 1, seed = -1)"
                )
            ro.r(r_call)

            # nativeArray columns: treeID, nodeID, nodeSZ, brnodeID, parmID,
            # contPT, mwcpSZ, fsrecID.  The row with parmID != 0 and a
            # non-null contPT is the root split node.
            with localconverter(converter):
                na_raw = ro.r("as.data.frame(fit__$forest$nativeArray)")
            na_df = pd.DataFrame(na_raw)

            internal = na_df[(na_df["parmID"] != 0) & na_df["contPT"].notna()]
            if len(internal) == 0:
                raise RuntimeError(
                    f"rfSRC produced no internal split for feature {feat}; nativeArray:\n{na_df}"
                )

            x_j = float(internal["contPT"].iloc[0])
            uniq = np.sort(np.unique(X[:, feat]))
            idx = int(np.searchsorted(uniq, x_j))
            if idx >= len(uniq) - 1 or uniq[idx] != x_j:
                raise RuntimeError(
                    f"rfSRC contPT={x_j!r} not found as a unique-X value with a "
                    f"successor for feature {feat}; unique values: {uniq!r}"
                )
            threshold = (uniq[idx] + uniq[idx + 1]) / 2.0
            rows.append({"feature": int(feat), "best_threshold": threshold})
        finally:
            with localconverter(converter):
                ro.r("if (exists('fit__')) rm(fit__); if (exists('df')) rm(df)")

    return pd.DataFrame(rows, columns=["feature", "best_threshold"])


def crforest_candidate_stats(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_causes: int,
    *,
    splitrule: str = "logrankCR",
    cause: int = 1,
) -> pd.DataFrame:
    """Compute crforest's log-rank statistic for every candidate split.

    Candidates are the midpoints between consecutive unique values per feature,
    matching find_best_split's enumeration. Returns a DataFrame with columns
    (feature, threshold, stat).

    Parameters
    ----------
    n_causes:
        Number of competing causes (used for composite log-rank).
    splitrule:
        ``"logrankCR"`` — composite log-rank statistic (default).
        ``"logrank"``   — cause-specific log-rank statistic.
    cause:
        Cause index (1-based) used when ``splitrule="logrank"``.
    """
    if splitrule not in ("logrankCR", "logrank"):
        raise ValueError(f"Unsupported splitrule: {splitrule!r}; use 'logrankCR' or 'logrank'")

    tb = bin_times(time, event)
    rows = []
    for feat in range(X.shape[1]):
        uniq = np.unique(X[:, feat])
        if len(uniq) < 2:
            continue
        mids = (uniq[:-1] + uniq[1:]) / 2.0
        for mid in mids:
            left_mask = X[:, feat] <= mid
            if left_mask.all() or (~left_mask).all():
                continue
            if splitrule == "logrankCR":
                stat = composite_log_rank_statistic(tb, left_mask, n_causes)
            else:  # logrank
                stat = cause_specific_log_rank_statistic(tb, left_mask, cause=cause)
            rows.append({"feature": int(feat), "threshold": float(mid), "stat": float(stat)})
    return pd.DataFrame(rows, columns=["feature", "threshold", "stat"])
