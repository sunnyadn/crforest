"""Reproducibility fingerprint for canonical benchmark scripts.

Every script in validation/comparisons/ (and validation/scaling/ when added)
should call dump_fingerprint(out_parquet_path) at start so the parquet has
a sidecar _fingerprint.json with: git SHA, dirty-tree flag, lib versions,
machine name, OS, CPU model, RAM, datetime, Python version. The fingerprint
is what tells a future reader whether the parquet's numbers are still
trustworthy after the repo has moved on.

Library version probe is best-effort; missing libs go in as None, not as an
error.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROBE_LIBS = (
    "comprisk",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "scikit-survival",
    "numba",
    "joblib",
    "cupy",
    "cupy-cuda12x",
)


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _lib_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in PROBE_LIBS:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _cpu_brand() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
            ).strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        return platform.processor()
    except Exception:
        return ""


def _ram_gb() -> float:
    try:
        if sys.platform == "darwin":
            return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()) / (1024**3)
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return 0.0


def fingerprint() -> dict:
    return {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "machine": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_brand": _cpu_brand(),
        "ram_gb": round(_ram_gb(), 1),
        "python": sys.version.split()[0],
        "lib_versions": _lib_versions(),
    }


def dump_fingerprint(parquet_out: str | Path) -> Path:
    """Write <parquet_out>.fingerprint.json sidecar; return its path."""
    parquet_out = Path(parquet_out)
    out = parquet_out.with_suffix(parquet_out.suffix + ".fingerprint.json")
    out.write_text(json.dumps(fingerprint(), indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(fingerprint(), indent=2))
