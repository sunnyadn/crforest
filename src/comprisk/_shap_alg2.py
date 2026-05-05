"""Algorithm 2 TreeSHAP for comprisk — numba-jitted O(L·D²)."""

from __future__ import annotations

import numpy as np
from numba import njit

from comprisk._tree_flat import FlatTree

# ---------------------------------------------------------------------------
# Path operations (EXTEND / UNWIND / unwound_path_sum)
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _extend_path(
    path_feature,
    path_z,
    path_o,
    path_w,
    unique_depth,
    zero_fraction,
    one_fraction,
    feature_index,
):
    """Extend decision path with a new feature.

    Operates on the slice starting at index 0 with logical depth ``unique_depth``
    before the call.  After the call the valid entries are ``0 .. unique_depth``.
    """
    path_feature[unique_depth] = feature_index
    path_z[unique_depth] = zero_fraction
    path_o[unique_depth] = one_fraction
    path_w[unique_depth] = 1.0 if unique_depth == 0 else 0.0
    for i in range(unique_depth - 1, -1, -1):
        path_w[i + 1] += one_fraction * path_w[i] * (i + 1) / (unique_depth + 1)
        path_w[i] = zero_fraction * path_w[i] * (unique_depth - i) / (unique_depth + 1)


@njit(cache=True, nogil=True)
def _unwound_path_sum(path_z, path_o, path_w, unique_depth, path_index):
    """Total permutation weight if ``path_index`` were unwound."""
    one_fraction = path_o[path_index]
    zero_fraction = path_z[path_index]
    next_one_portion = path_w[unique_depth]
    total = 0.0
    if one_fraction != 0.0:
        for i in range(unique_depth - 1, -1, -1):
            tmp = next_one_portion / ((i + 1) * one_fraction)
            total += tmp
            next_one_portion = path_w[i] - tmp * zero_fraction * (unique_depth - i)
    else:
        for i in range(unique_depth - 1, -1, -1):
            total += path_w[i] / (zero_fraction * (unique_depth - i))
    return total * (unique_depth + 1)


@njit(cache=True, nogil=True)
def _unwind_path(path_feature, path_z, path_o, path_w, unique_depth, path_index):
    """Undo a previous extension; remove ``path_index`` and shift left."""
    one_fraction = path_o[path_index]
    zero_fraction = path_z[path_index]
    next_one_portion = path_w[unique_depth]
    for i in range(unique_depth - 1, -1, -1):
        if one_fraction != 0.0:
            tmp = path_w[i]
            path_w[i] = next_one_portion * (unique_depth + 1) / ((i + 1) * one_fraction)
            next_one_portion = tmp - path_w[i] * zero_fraction * (unique_depth - i) / (
                unique_depth + 1
            )
        else:
            path_w[i] = path_w[i] * (unique_depth + 1) / (zero_fraction * (unique_depth - i))
    for i in range(path_index, unique_depth):
        path_feature[i] = path_feature[i + 1]
        path_z[i] = path_z[i + 1]
        path_o[i] = path_o[i + 1]


# ---------------------------------------------------------------------------
# Recursive Algorithm 2 core  (offset-based, C++-style pointer arithmetic)
# ---------------------------------------------------------------------------


@njit(cache=True, nogil=True)
def _tree_shap_recursive(
    x,
    features,
    split_values,
    left_children,
    right_children,
    is_leaf_flags,
    leaf_idx_of_node,
    leaf_table,
    covers,
    node,
    unique_depth,
    path_feature,
    path_z,
    path_o,
    path_w,
    path_offset,
    parent_z,
    parent_o,
    parent_feat,
    phi,
):
    """Algorithm 2 — recursive descent with EXTEND / UNWIND.

    ``path_offset`` points to the *parent* path in the shared arrays.
    This routine first copies the parent prefix into a new slice at
    ``my_offset = path_offset + unique_depth + 1`` (mirroring C++
    ``unique_path = parent_unique_path + unique_depth + 1``), then
    extends it.  Children receive ``my_offset`` as their parent offset.
    """
    my_offset = path_offset + unique_depth + 1

    # Copy parent path into our working slice (C++ std::copy equivalent)
    for i in range(unique_depth + 1):
        path_feature[my_offset + i] = path_feature[path_offset + i]
        path_z[my_offset + i] = path_z[path_offset + i]
        path_o[my_offset + i] = path_o[path_offset + i]
        path_w[my_offset + i] = path_w[path_offset + i]

    _extend_path(
        path_feature[my_offset:],
        path_z[my_offset:],
        path_o[my_offset:],
        path_w[my_offset:],
        unique_depth,
        parent_z,
        parent_o,
        parent_feat,
    )

    n_causes = leaf_table.shape[1]
    n_times = leaf_table.shape[2]

    if is_leaf_flags[node]:
        leaf_idx = leaf_idx_of_node[node]
        for i in range(1, unique_depth + 1):
            w = _unwound_path_sum(
                path_z[my_offset:],
                path_o[my_offset:],
                path_w[my_offset:],
                unique_depth,
                i,
            )
            feat = path_feature[my_offset + i]
            if feat < 0:
                continue
            scale = w * (path_o[my_offset + i] - path_z[my_offset + i])
            for c in range(n_causes):
                for t in range(n_times):
                    phi[feat, c, t] += scale * leaf_table[leaf_idx, c, t]
        return

    feat = features[node]
    threshold = split_values[node]
    if x[feat] <= threshold:
        hot = left_children[node]
        cold = right_children[node]
    else:
        hot = right_children[node]
        cold = left_children[node]

    cover_total = covers[node]
    cover_hot = covers[hot]
    cover_cold = covers[cold]

    hot_z_frac = cover_hot / cover_total if cover_total > 0 else 0.0
    cold_z_frac = cover_cold / cover_total if cover_total > 0 else 0.0

    # Check for repeated feature already on the path
    incoming_z = 1.0
    incoming_o = 1.0
    path_index = unique_depth + 1  # not-found sentinel
    for i in range(unique_depth + 1):
        if path_feature[my_offset + i] == feat:
            path_index = i
            break

    if path_index != unique_depth + 1:
        incoming_z = path_z[my_offset + path_index]
        incoming_o = path_o[my_offset + path_index]
        _unwind_path(
            path_feature[my_offset:],
            path_z[my_offset:],
            path_o[my_offset:],
            path_w[my_offset:],
            unique_depth,
            path_index,
        )
        unique_depth -= 1

    child_offset = my_offset
    child_depth = unique_depth + 1

    # Recurse hot child (branch followed by sample x)
    _tree_shap_recursive(
        x,
        features,
        split_values,
        left_children,
        right_children,
        is_leaf_flags,
        leaf_idx_of_node,
        leaf_table,
        covers,
        hot,
        child_depth,
        path_feature,
        path_z,
        path_o,
        path_w,
        child_offset,
        hot_z_frac * incoming_z,
        incoming_o,
        feat,
        phi,
    )

    # Recurse cold child (branch NOT followed by sample x)
    _tree_shap_recursive(
        x,
        features,
        split_values,
        left_children,
        right_children,
        is_leaf_flags,
        leaf_idx_of_node,
        leaf_table,
        covers,
        cold,
        child_depth,
        path_feature,
        path_z,
        path_o,
        path_w,
        child_offset,
        cold_z_frac * incoming_z,
        0.0,
        feat,
        phi,
    )


# ---------------------------------------------------------------------------
# Public wrapper (single tree, single sample)
# ---------------------------------------------------------------------------


def shap_tree_single_alg2(
    flat: FlatTree, x: np.ndarray, n_features: int, covers: np.ndarray
) -> np.ndarray:
    """TreeSHAP Algorithm 2 — O(L·D²) per sample."""
    n_causes, n_times = flat.leaf_table.shape[1:]
    phi = np.zeros((n_features, n_causes, n_times), dtype=np.float64)

    max_depth = len(flat.features)
    # Worst-case offset sequence: 0, 1, 3, 6, 10, ... (triangular numbers)
    # At depth d, offset = d*(d+1)/2.  Reserve enough for d = max_depth+1.
    max_offset = (max_depth + 2) * (max_depth + 3) // 2 + 4
    path_feature = np.full(max_offset, -1, dtype=np.int64)
    path_z = np.zeros(max_offset, dtype=np.float64)
    path_o = np.zeros(max_offset, dtype=np.float64)
    path_w = np.zeros(max_offset, dtype=np.float64)

    # Seed a dummy sentinel at index 0 so the initial copy has something valid
    path_feature[0] = -1
    path_z[0] = 1.0
    path_o[0] = 1.0
    path_w[0] = 1.0

    _tree_shap_recursive(
        x,
        flat.features,
        flat.split_values,
        flat.left_children,
        flat.right_children,
        flat.is_leaf_flags,
        flat.leaf_idx_of_node,
        flat.leaf_table,
        covers,
        0,
        0,
        path_feature,
        path_z,
        path_o,
        path_w,
        0,
        1.0,
        1.0,
        -1,
        phi,
    )
    return phi
