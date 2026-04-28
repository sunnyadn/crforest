"""Tests for _gpu_detect.detect_cuda — must work whether or not cupy is installed."""

import builtins
import sys

from crforest import _gpu_detect


def test_detect_cuda_returns_bool_and_reason():
    available, reason = _gpu_detect.detect_cuda()
    assert isinstance(available, bool)
    assert isinstance(reason, str)
    assert reason  # non-empty


def test_detect_cuda_reason_mentions_cupy_when_unavailable():
    available, reason = _gpu_detect.detect_cuda()
    if not available:
        # Either cupy missing or no device; reason must explain.
        assert "cupy" in reason.lower() or "device" in reason.lower() or "cuda" in reason.lower()


def test_detect_cuda_handles_runtime_error_on_import(monkeypatch):
    """cupy's DynamicLibNotFoundError (NVRTC SO unfindable) is a RuntimeError, not
    ImportError — detect_cuda must still return (False, reason), not propagate.
    """
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cupy":
            raise RuntimeError('DynamicLibNotFoundError: Failure finding "libnvrtc.so"')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "cupy", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    available, reason = _gpu_detect.detect_cuda()
    assert available is False
    assert reason
