"""Smoke tests — verify the package is importable and reports a version."""

import comprisk


def test_package_imports():
    assert comprisk is not None


def test_version_matches_expected():
    # Pinned to the current release version. Bump when releasing.
    assert comprisk.__version__ == "0.3.1"
