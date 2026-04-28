import subprocess
import sys


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "validation", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "calibrate" in result.stdout
    assert "run" in result.stdout
    assert "report" in result.stdout


def test_cli_run_help():
    result = subprocess.run(
        [sys.executable, "-m", "validation", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--seeds" in result.stdout
    assert "--dataset" in result.stdout
