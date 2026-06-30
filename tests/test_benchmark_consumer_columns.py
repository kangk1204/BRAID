"""Regression: the differential output schema stays compatible with the DM1/ESRP
benchmark application consumers.

``braid differential`` and the application scripts under ``benchmarks/`` are coupled
by the differential TSV column names, but those consumers are not otherwise exercised
by the test suite -- so a rename of a differential output column would silently break
the paper's DM1/ESRP application figures. This test builds a *real* differential
output and feeds it to each consumer's column-derivation step, so the contract fails
loudly instead of drifting. It skips cleanly where pandas or the (gitignored)
benchmark scripts are absent.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from braid.commands.differential import run_differential
from tests.test_differential import _args, _write_se_table

pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]


def _differential_frame(tmp_path: Path):
    """A real ``braid differential`` output frame (synthetic rMATS -> run_differential)."""
    rmats_dir = tmp_path / "rmats"
    out = tmp_path / "diff.tsv"
    _write_se_table(rmats_dir)
    run_differential(_args(rmats_dir, out))
    return pd.read_csv(out, sep="\t")


def _load_consumer(rel_path: str, name: str):
    path = ROOT / rel_path
    if not path.exists():
        pytest.skip(f"{rel_path} not present (benchmarks excluded from this checkout)")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_esrp_consumer_reads_differential_schema(tmp_path: Path) -> None:
    esrp = _load_consumer(
        "benchmarks/application_esrp/run_esrp_application.py", "esrp_consumer_under_test"
    )
    out = esrp.add_columns(_differential_frame(tmp_path))
    for col in ("abs_dpsi", "ci_width", "total_support"):
        assert col in out.columns, f"ESRP consumer lost derived column {col!r}"


def test_dm1_consumer_reads_differential_schema(tmp_path: Path, monkeypatch) -> None:
    # The DM1 script does ``from docx import Document`` at module scope for its .docx
    # report; stub it so the column-derivation contract can be checked without that
    # optional dependency (python-docx is not a BRAID requirement). Must be set before
    # the module is imported below.
    docx = types.ModuleType("docx")
    docx.Document = object
    monkeypatch.setitem(sys.modules, "docx", docx)
    dm1 = _load_consumer(
        "benchmarks/application_dm1/run_dm1_application.py", "dm1_consumer_under_test"
    )
    out = dm1.add_application_columns(_differential_frame(tmp_path))
    for col in ("control_psi", "total_support", "disease_minus_control_dpsi", "abs_dpsi"):
        assert col in out.columns, f"DM1 consumer lost derived column {col!r}"
