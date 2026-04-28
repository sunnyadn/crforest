"""Tests for validation.alignment package scaffold."""

from validation.alignment import _rpy2_available


def test_rpy2_available_returns_bool():
    result = _rpy2_available()
    assert isinstance(result, bool)


def test_rpy2_available_is_true_when_rpy2_importable():
    # This test only asserts the positive branch when rpy2 is actually installed.
    # In environments without rpy2 it is a trivial True assertion.
    try:
        import rpy2  # noqa: F401
    except ImportError:
        return
    assert _rpy2_available() is True
