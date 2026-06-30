"""Tests for the zero-setup ``braid example`` demo command."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from braid.commands.example import run_example


def test_example_runs_and_writes_calibrated_tiers(tmp_path: Path, capsys) -> None:
    run_example(argparse.Namespace(output=str(tmp_path), keep=True))
    out = tmp_path / "braid_differential.tsv"
    assert out.exists()
    rows = {r["gene"]: r for r in csv.DictReader(open(out), delimiter="\t")}
    # the strong synthetic effects are high-confidence; the null is not
    assert rows["STRONG_UP"]["tier"] == "high-confidence"
    assert rows["STRONG_DOWN"]["tier"] == "high-confidence"
    assert rows["NO_CHANGE"]["tier"] == "not-significant"
    # every event carries a calibrated interval
    for r in rows.values():
        lo, hi = float(r["ci_low"]), float(r["ci_high"])
        assert -1.0 <= lo <= hi <= 1.0
    # the demo prints an interpreted summary
    captured = capsys.readouterr().out
    assert "calibrated" in captured.lower()
    assert "high-confidence" in captured


def test_example_default_tempdir(capsys) -> None:
    run_example(argparse.Namespace(output=None, keep=False))
    out = capsys.readouterr().out
    assert "braid_example_" in out  # used a temp dir
    assert "STRONG_UP" in out


def test_example_output_path_is_a_file_raises_clear_error(tmp_path: Path) -> None:
    """braid example -o <existing file>: makedirs(exist_ok=True) raises FileExistsError
    when the path is a FILE (exist_ok only forgives a directory). Must be a clear
    ValueError, not an 'Unexpected error' traceback for this new-user demo command."""
    import pytest

    not_a_dir = tmp_path / "notadir.txt"
    not_a_dir.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        run_example(argparse.Namespace(output=str(not_a_dir), keep=False))
