"""Forest-level device dispatcher: auto / cpu / cuda + n_jobs interaction."""

import warnings

import numpy as np
import pytest

from crforest import CompetingRiskForest


def _toy(n=200, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(size=(n, p))
    time = rng.uniform(0.1, 5.0, n)
    event = rng.integers(0, 3, n)
    if not np.any(event == 1):
        event[0] = 1
    if not np.any(event == 2):
        event[1] = 2
    return X, time, event


def test_device_default_is_auto():
    f = CompetingRiskForest()
    assert f.device == "auto"


def test_device_cpu_explicit_skips_detection():
    X, t, e = _toy()
    f = CompetingRiskForest(n_estimators=3, device="cpu", random_state=0).fit(X, t, e)
    assert f._effective_device_ == "cpu"


def test_device_invalid_raises():
    with pytest.raises(ValueError, match="device"):
        CompetingRiskForest(device="rocm")


def test_device_cuda_with_n_jobs_warns():
    X, t, e = _toy()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            CompetingRiskForest(n_estimators=3, device="cuda", n_jobs=4, random_state=0).fit(
                X, t, e
            )
        except RuntimeError as exc:
            assert "cuda" in str(exc).lower()
            return
        assert any("n_jobs" in str(item.message) for item in w)


@pytest.mark.gpu
def test_device_cuda_uses_gpu_path():
    X, t, e = _toy()
    f = CompetingRiskForest(n_estimators=3, device="cuda", random_state=0).fit(X, t, e)
    assert f._effective_device_ == "cuda"


def test_device_cuda_with_rfsrc_alignment_raises():
    """device='cuda' is incompatible with equivalence='rfsrc' (legacy hist-tree path).

    Raise rather than silently fall back to cpu — symmetric with how
    cuda-unavailable raises on the flat-tree path.
    """
    X, t, e = _toy()
    with pytest.raises(ValueError, match="cuda"):
        CompetingRiskForest(
            n_estimators=3,
            device="cuda",
            equivalence="rfsrc",
            random_state=0,
        ).fit(X, t, e)


def test_device_auto_resolves_to_cpu_silently():
    """v0.1: device='auto' resolves to cpu on every host, no warning.

    cuda is opt-in via explicit device='cuda' in v0.1 (the cuda backend is
    a preview that's slower at typical clinical p; full GPU rewrite is
    deferred to v1.1). 'auto' should be silent on both cpu-only and
    cuda-equipped hosts.
    """
    import warnings as _warn

    X, t, e = _toy()
    with _warn.catch_warnings(record=True) as w:
        _warn.simplefilter("always")
        f = CompetingRiskForest(n_estimators=2, device="auto", random_state=0).fit(X, t, e)
    assert f._effective_device_ == "cpu"
    auto_warns = [ww for ww in w if "auto" in str(ww.message).lower()]
    assert auto_warns == [], (
        f"unexpected auto-related warning(s): {[str(x.message) for x in auto_warns]}"
    )


def test_device_cuda_default_njobs_no_warn():
    """device='cuda' with default n_jobs (=-1) should NOT emit the n_jobs-ignored warning.

    Only positive user-supplied n_jobs > 1 should warn.
    """
    import warnings as _warn

    X, t, e = _toy()
    with _warn.catch_warnings(record=True) as w:
        _warn.simplefilter("always")
        try:
            # default n_jobs=-1
            CompetingRiskForest(n_estimators=2, device="cuda", random_state=0).fit(X, t, e)
        except RuntimeError:
            # Mac: cuda unavailable — fine; the test is about the warning, not the run.
            return
    n_jobs_warns = [ww for ww in w if "n_jobs" in str(ww.message)]
    assert n_jobs_warns == [], (
        f"unexpected n_jobs warning(s) at default n_jobs=-1: "
        f"{[str(x.message) for x in n_jobs_warns]}"
    )
