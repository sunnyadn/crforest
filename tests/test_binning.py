"""Tests for midpoint bin edges, bin assignment, and inverse lookup."""

import numpy as np
import pytest

from crforest._binning import apply_bins, bin_to_threshold, fit_bin_edges


def test_fit_bin_edges_midpoints_when_few_unique_values():
    X = np.array([[1.0], [2.0], [3.0]])
    edges = fit_bin_edges(X, n_bins=256)
    assert len(edges) == 1
    np.testing.assert_array_equal(edges[0], [1.5, 2.5])  # midpoints of [1,2,3]


def test_fit_bin_edges_quantile_when_many_unique_values():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(1000, 1))
    edges = fit_bin_edges(X, n_bins=256)
    assert edges[0].shape == (255,)  # n_bins - 1 interior cutpoints
    assert np.all(np.diff(edges[0]) > 0)
    # All interior — strictly greater than min, strictly less than max
    assert edges[0][0] > X.min()
    assert edges[0][-1] < X.max()


def test_fit_bin_edges_multiple_features():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    edges = fit_bin_edges(X, n_bins=256)
    assert len(edges) == 2
    np.testing.assert_array_equal(edges[0], [1.5, 2.5])
    np.testing.assert_array_equal(edges[1], [15.0, 25.0])


def test_fit_bin_edges_rejects_n_bins_over_256():
    X = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="n_bins"):
        fit_bin_edges(X, n_bins=257)


def test_fit_bin_edges_rejects_n_bins_below_2():
    X = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="n_bins"):
        fit_bin_edges(X, n_bins=1)


def test_apply_bins_maps_to_uint8():
    # Midpoint edges [1.5, 2.5, 3.5]; 4 bins: <1.5, [1.5,2.5), [2.5,3.5), [3.5,∞)
    edges = [np.array([1.5, 2.5, 3.5])]
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    bins = apply_bins(X, edges)
    assert bins.dtype == np.uint8
    assert bins.shape == (4, 1)
    np.testing.assert_array_equal(bins, np.array([[0], [1], [2], [3]], dtype=np.uint8))


def test_apply_bins_monotonic_per_column():
    rng = np.random.default_rng(1)
    X = rng.uniform(size=(200, 1))
    uniq = np.unique(X[:, 0])
    # 10 equal-spaced interior cutpoints
    edges = [np.quantile(uniq, np.linspace(0.0, 1.0, 11)[1:-1])]
    bins = apply_bins(X, edges)
    order = np.argsort(X[:, 0])
    sorted_bins = bins[order, 0]
    assert np.all(np.diff(sorted_bins.astype(np.int64)) >= 0)


def test_apply_bins_clips_values_above_training_range():
    edges = [np.array([1.5, 2.5])]  # 3 bins
    X = np.array([[5.0]])
    bins = apply_bins(X, edges)
    assert bins[0, 0] == 2  # last bin (3 - 1)


def test_apply_bins_clips_values_below_training_range():
    edges = [np.array([1.5, 2.5])]  # 3 bins
    X = np.array([[0.0]])
    bins = apply_bins(X, edges)
    assert bins[0, 0] == 0


def test_apply_bins_multiple_features():
    edges = [np.array([0.5]), np.array([15.0])]  # 2 bins each
    X = np.array([[0.0, 10.0], [1.0, 20.0]])
    bins = apply_bins(X, edges)
    np.testing.assert_array_equal(bins, np.array([[0, 0], [1, 1]], dtype=np.uint8))


def test_apply_bins_rejects_edges_with_too_many_bins():
    # 256 edges => 257 bins, out of range
    edges = [np.arange(256, dtype=np.float64)]
    X = np.zeros((1, 1))
    with pytest.raises(ValueError, match="must be ≤ 256"):
        apply_bins(X, edges)


def test_apply_bins_accepts_single_bin_feature():
    # Constant-feature column: edges is empty, producing 1 bin.
    # All samples should map to bin 0; no error.
    edges = [np.array([], dtype=np.float64)]
    X = np.array([[5.0], [5.0], [5.0]])
    bins = apply_bins(X, edges)
    assert bins.dtype == np.uint8
    assert bins.shape == (3, 1)
    np.testing.assert_array_equal(bins, np.zeros((3, 1), dtype=np.uint8))


def test_bin_to_threshold_roundtrip():
    edges = [np.array([1.5, 2.5, 3.5])]  # 4 bins
    X = np.array([[2.0]])  # midpoint-binned to bin 1 (between 1.5 and 2.5)
    bins = apply_bins(X, edges)
    assert bins[0, 0] == 1
    t = bin_to_threshold(edges, feature=0, bin_idx=int(bins[0, 0]))
    assert X[0, 0] <= t  # threshold == 2.5


def test_bin_to_threshold_uses_midpoint_upper_bound():
    edges = [np.array([1.5, 2.5])]  # 3 bins
    assert bin_to_threshold(edges, 0, 0) == 1.5
    assert bin_to_threshold(edges, 0, 1) == 2.5


def test_bin_to_threshold_out_of_range_returns_inf():
    edges = [np.array([1.5, 2.5])]  # 3 bins
    assert bin_to_threshold(edges, 0, 2) == np.inf
