"""Maintainer-only alignment harness comparing crforest to randomForestSRC.

Not shipped in the public wheel (inherits validation/'s maintainer-only status).

Modules:
- compare_cif.py  - per-seed CIF diagnostic (maintainer).
- compare_splits.py - per-feature best-split comparison (used by CI fixture).
- equivalence_gate.py - maintainer-invoked audit; CLI at
  ``python -m validation.alignment.equivalence_gate``.
- report_equivalence.py - markdown writer for the audit.
"""

from __future__ import annotations


def _rpy2_available() -> bool:
    """Return True if rpy2 and an R interpreter are importable in this process."""
    try:
        import rpy2.robjects  # noqa: F401
    except ImportError:
        return False
    except Exception:
        # rpy2 can raise non-ImportError exceptions if R is missing; treat as unavailable.
        return False
    return True


def _rpy2_converter():
    """Return the standard rpy2 converter (pandas + numpy) used across this harness."""
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri, pandas2ri

    return ro.default_converter + pandas2ri.converter + numpy2ri.converter
