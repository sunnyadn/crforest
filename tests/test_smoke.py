"""Smoke tests — verify the package is importable and reports a version."""

import crforest


def test_package_imports():
    assert crforest is not None


def test_version_matches_expected():
    # Pinned to the current release version. Bump when releasing.
    assert crforest.__version__ == "0.1.0"
