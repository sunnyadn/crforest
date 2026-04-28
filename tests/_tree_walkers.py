"""Shared tree-serialization helpers for tree-equivalence tests.

The crforest project has three tree representations that may need pre-order
DFS serialization in determinism / parallel-equivalence / regression tests:

- ``FlatTree`` (``crforest._tree_flat``) — array-of-records, default-mode
  flat-tree builder output.
- ``HistTreeNode`` (``crforest._hist_tree``) — recursive, used by the
  ``equivalence='rfsrc'`` aligned path.
- ``RefTreeNode`` (``crforest._tree``) — recursive, used by the float-mode
  reference path.

Each leaf format includes raw leaf content (not predicted CIF/CHF) so a
regression in the build pass is caught without being perturbed by predict-
path changes.

A self-comparison (two trees of the same type walked with the same
serializer) is bit-equivalent iff structure + leaf data match. Cross-type
comparisons are intentionally unequal — leaf tuples have different arity
and content semantics.
"""

from __future__ import annotations


def walk_tree(tree) -> list:
    """Pre-order DFS serialization of a tree, including leaf content.

    Splits encode as ``("split", feature, split_value)``. Leaves encode
    differently per tree type so the serializer captures the relevant
    invariant for that representation:

    - ``FlatTree``: ``("leaf", leaf_table_bytes)`` — the materialized CIF.
    - ``HistTreeNode``: ``("leaf", event_counts_bytes, at_risk_bytes)`` —
      raw dense counts (materialized from sparse leaf rep).
    - ``RefTreeNode``: ``("leaf", event_counts_bytes, at_risk_bytes)`` —
      raw counts.
    """
    from crforest._hist_tree import HistTreeNode
    from crforest._tree import RefTreeNode
    from crforest._tree_flat import FlatTree

    if isinstance(tree, FlatTree):
        out: list = []
        stack = [0]
        while stack:
            i = stack.pop()
            if tree.is_leaf_flags[i]:
                leaf_k = int(tree.leaf_idx_of_node[i])
                out.append(("leaf", tree.leaf_table[leaf_k].tobytes()))
            else:
                out.append(("split", int(tree.features[i]), int(tree.split_values[i])))
                stack.append(int(tree.right_children[i]))
                stack.append(int(tree.left_children[i]))
        return out

    if isinstance(tree, HistTreeNode):
        out = []
        stack = [tree]
        while stack:
            n = stack.pop()
            if n.is_leaf:
                ec = n.event_counts_dense.tobytes() if n.event_counts_sparse is not None else b""
                ar = n.at_risk_dense.tobytes() if n.at_risk_sparse is not None else b""
                out.append(("leaf", ec, ar))
            else:
                out.append(("split", int(n.feature), int(n.bin_idx)))
                stack.append(n.right)
                stack.append(n.left)
        return out

    if isinstance(tree, RefTreeNode):
        out = []
        stack = [tree]
        while stack:
            n = stack.pop()
            if n.is_leaf:
                ec = n.event_counts.tobytes() if n.event_counts is not None else b""
                ar = n.at_risk.tobytes() if n.at_risk is not None else b""
                out.append(("leaf", ec, ar))
            else:
                out.append(("split", int(n.feature), float(n.threshold)))
                stack.append(n.right)
                stack.append(n.left)
        return out

    raise TypeError(f"walk_tree: unsupported tree type {type(tree).__name__}")
