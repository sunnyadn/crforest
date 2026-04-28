"""Runtime CUDA detection. cupy is optional — never imported at module load."""

from __future__ import annotations


def detect_cuda() -> tuple[bool, str]:
    """Return (available, reason_string).

    available is True iff cupy is importable AND at least one CUDA device is
    visible. reason_string is human-readable for warnings/error messages
    ("cupy not installed", "no CUDA device", "cupy 14.x + 1 device", etc.).
    """
    try:
        import cupy as cp
    except Exception as exc:
        # cupy's DynamicLibNotFoundError (raised when NVRTC/runtime DLLs are
        # unfindable) is a RuntimeError, not an ImportError — hence the broad catch.
        return False, f"cupy unavailable ({type(exc).__name__}: {exc})"
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        return False, f"cupy import OK but no CUDA device ({exc})"
    if n < 1:
        return False, "no CUDA device"
    return True, f"cupy {cp.__version__} + {n} CUDA device(s)"
