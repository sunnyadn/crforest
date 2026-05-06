"""Bit-exact alignment vs the canonical ``shap`` library.

For a fixed ``(cause, time_idx)`` slice, every leaf of a comprisk tree
collapses to a scalar — so the SHAP problem at that slice is identical to
TreeSHAP on a regression tree and ``shap.TreeExplainer`` (Lundberg's own
reference implementation) is the gold standard to validate against.

We construct a single-tree forest, project its FlatTree to the slice as a
shap-library raw-tree dict, and assert that per-sample, per-feature SHAP
values match to fp64 round-off.
"""

from __future__ import annotations

import numpy as np
import pytest

shap = pytest.importorskip("shap")

from comprisk._shap import _compute_node_covers, _get_flat_and_leaf_counts  # noqa: E402
from comprisk.forest import CompetingRiskForest  # noqa: E402


def _make_synthetic(seed=0, n=100, p=4):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, p))
    time = 5.0 - 2.0 * X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=n)
    time = np.clip(time, 0.1, None)
    event = rng.integers(0, 3, size=n)
    if not np.any(event == 1):
        event[0] = 1
    if not np.any(event == 2):
        event[1] = 2
    return X, time, event


def _flat_to_shap_dict(flat, covers, cause_idx, time_idx):
    """Project a comprisk FlatTree onto a fixed (cause, time) slice and
    convert to the dict format ``shap.TreeExplainer`` expects."""
    n_nodes = len(flat.is_leaf_flags)
    values = np.zeros((n_nodes, 1), dtype=np.float64)
    children_left = flat.left_children.astype(np.int32).copy()
    children_right = flat.right_children.astype(np.int32).copy()

    for j in range(n_nodes):
        if flat.is_leaf_flags[j]:
            li = flat.leaf_idx_of_node[j]
            values[j, 0] = flat.leaf_table[li, cause_idx, time_idx]
            children_left[j] = -1
            children_right[j] = -1

    return {
        "children_left": children_left,
        "children_right": children_right,
        "children_default": children_left.copy(),
        "features": flat.features.astype(np.int32).copy(),
        "thresholds": flat.split_values.astype(np.float64).copy(),
        "values": values,
        "node_sample_weight": covers.astype(np.float64).copy(),
    }


@pytest.mark.parametrize("mode", ["default", "reference"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shap_matches_canonical_shap_library(mode, seed):
    """Single-tree forest: comprisk SHAP at slice == shap.TreeExplainer SHAP."""
    X, time, event = _make_synthetic(seed=seed, n=120, p=4)
    f = CompetingRiskForest(n_estimators=1, random_state=seed, mode=mode, max_depth=5).fit(
        X, time, event
    )

    X_input = (
        np.asarray(X, dtype=np.float64)
        if mode == "reference"
        else __import__("comprisk._binning", fromlist=["apply_bins"]).apply_bins(
            np.asarray(X, dtype=np.float64), f.bin_edges_
        )
    )

    flat, leaf_counts = _get_flat_and_leaf_counts(f.trees_[0])
    covers = _compute_node_covers(
        flat.is_leaf_flags,
        flat.left_children,
        flat.right_children,
        flat.leaf_idx_of_node,
        leaf_counts,
    )

    cause_idx, time_idx = 0, len(f.unique_times_) - 1

    tree_dict = _flat_to_shap_dict(flat, covers, cause_idx, time_idx)
    explainer = shap.TreeExplainer({"trees": [tree_dict]})
    ref_shap = explainer.shap_values(X_input)
    ref_base = explainer.expected_value

    our_shap, our_base = f.shap_values(X)
    our_slice = our_shap[:, :, time_idx, cause_idx]
    our_base_slice = our_base[time_idx, cause_idx]

    max_abs_err = float(np.max(np.abs(our_slice - ref_shap)))
    base_err = float(abs(our_base_slice - float(np.atleast_1d(ref_base)[0])))

    assert max_abs_err < 1e-9, (
        f"comprisk vs shap.TreeExplainer per-feature max|Δ|={max_abs_err:.3e} "
        f"(mode={mode}, seed={seed})"
    )
    assert base_err < 1e-9, f"base value mismatch: {base_err:.3e}"


def test_shap_matches_canonical_multi_tree():
    """100-tree forest: forest-averaged SHAP at slice == averaged ref."""
    X, time, event = _make_synthetic(seed=42, n=200, p=5)
    f = CompetingRiskForest(n_estimators=10, random_state=42, max_depth=4).fit(X, time, event)

    from comprisk._binning import apply_bins

    X_input = apply_bins(np.asarray(X, dtype=np.float64), f.bin_edges_)

    cause_idx, time_idx = 1, len(f.unique_times_) // 2

    tree_dicts = []
    for tree in f.trees_:
        flat, leaf_counts = _get_flat_and_leaf_counts(tree)
        covers = _compute_node_covers(
            flat.is_leaf_flags,
            flat.left_children,
            flat.right_children,
            flat.leaf_idx_of_node,
            leaf_counts,
        )
        tree_dicts.append(_flat_to_shap_dict(flat, covers, cause_idx, time_idx))

    explainer = shap.TreeExplainer({"trees": tree_dicts})
    # shap.TreeExplainer sums across trees; comprisk averages.
    ref_shap = explainer.shap_values(X_input) / len(f.trees_)

    our_shap, _ = f.shap_values(X)
    our_slice = our_shap[:, :, time_idx, cause_idx]

    max_abs_err = float(np.max(np.abs(our_slice - ref_shap)))
    assert max_abs_err < 1e-9, (
        f"comprisk vs shap.TreeExplainer (10-tree forest) max|Δ|={max_abs_err:.3e}"
    )
