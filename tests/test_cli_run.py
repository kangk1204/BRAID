"""CLI regression tests for BRAID subcommands.

Each test invokes ``python -m braid ...`` via subprocess and checks
exit codes and key strings in output.
"""

from __future__ import annotations

import subprocess
import sys


def _run_braid(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m braid <args>`` and capture output.

    Args:
        args: CLI arguments after ``braid``.

    Returns:
        Completed process with captured stdout and stderr.
    """
    return subprocess.run(
        [sys.executable, "-m", "braid", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_run_help() -> None:
    """``braid run -h`` exits 0 and mentions auto-detection."""
    result = _run_braid("run", "-h")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "auto-detects" in combined.lower(), (
        f"Expected 'auto-detects' in help output, got:\n{combined}"
    )


def test_psi_help() -> None:
    """``braid psi -h`` exits 0 and mentions --bam."""
    result = _run_braid("psi", "-h")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "--bam" in combined, (
        f"Expected '--bam' in help output, got:\n{combined}"
    )


def test_differential_help() -> None:
    """``braid differential -h`` exits 0 and mentions --ctrl."""
    result = _run_braid("differential", "-h")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "--ctrl" in combined, (
        f"Expected '--ctrl' in help output, got:\n{combined}"
    )


def test_run_no_args() -> None:
    """``braid run`` without arguments should print help or error, not crash."""
    result = _run_braid("run")
    # Accept either exit 0 (help printed) or exit 2 (argparse error) —
    # the key requirement is that it does NOT crash with a traceback.
    assert result.returncode in (0, 1, 2), (
        f"Unexpected exit code {result.returncode}; "
        f"stderr: {result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined, (
        f"Command crashed with traceback:\n{combined}"
    )


def test_unknown_command() -> None:
    """``braid nonexistent`` should give a clear error, not silently assemble."""
    result = _run_braid("nonexistent")
    assert result.returncode != 0, (
        "Expected non-zero exit for unknown command"
    )
    combined = result.stdout + result.stderr
    assert "Unknown command" in combined, (
        f"Expected 'Unknown command' error, got:\n{combined}"
    )


def test_version() -> None:
    """``braid --version`` prints the version string."""
    result = _run_braid("--version")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "0.1.0" in combined, (
        f"Expected '0.1.0' in version output, got:\n{combined}"
    )
