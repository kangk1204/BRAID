"""Pipeline-robustness suite for the public ``braid filter`` path.

Stages: (1) parse caller output -> CallerEvent, (2) calibrate -> rows,
(3) report -> TSV/Excel/figure. Each test maps to one (stage, failure-mode)
cell of the robustness matrix and asserts something non-trivial (return schema
or an error ``match=``), never just "did not crash".
"""
from __future__ import annotations

import json
import math
import os

import pytest

from braid.adapters import PARSERS, calibrate_events
from braid.adapters.report import make_figure, write_excel, write_tsv
from braid.target.conformal import load_differential_conformal_calibrator


@pytest.fixture(scope="module")
def calibrator():
    return load_differential_conformal_calibrator()


def _write(tmp_path, name: str, content: str, *, encoding: str = "utf-8") -> str:
    p = tmp_path / name
    p.write_text(content, encoding=encoding)
    return str(p)


# ---- table-cell content per caller (ΔPSI-reading callers only) ----------------
# Minimal valid row + a templated bad-ΔPSI row for each caller.
_GOOD = {
    "majiq": "lsv_id\te(dpsi)\tprob\nG1\t0.50\t0.90\n",
    "suppa2": "event\tdpsi\tpval\nG1;SE\t0.50\t0.01\n",
    "betas": "gene\tevent_id\tdeltapsi\tlower\tupper\nG\tE1\t0.50\t0.40\t0.60\n",
}
_BAD = {  # {caller: (header, "{dpsi}"-templated bad row)}
    "majiq": ("lsv_id\te(dpsi)\tprob\n", "G1\t{dpsi}\t0.90\n"),
    "suppa2": ("event\tdpsi\tpval\n", "G1;SE\t{dpsi}\t0.01\n"),
    "betas": ("gene\tevent_id\tdeltapsi\tlower\tupper\n", "G\tE1\t{dpsi}\t0.40\t0.60\n"),
}
DPSI_CALLERS = ("majiq", "suppa2", "betas")


# ==============================================================================
# Stage 1 — Parse
# ==============================================================================
@pytest.mark.parametrize("caller", DPSI_CALLERS)
def test_parse_empty_file_returns_empty_list(tmp_path, caller):
    # A2: an empty file is not an error; it yields zero events.
    out = PARSERS[caller](_write(tmp_path, f"{caller}_empty.tsv", ""))
    assert out == []


@pytest.mark.parametrize("caller", DPSI_CALLERS)
def test_parse_missing_dpsi_column_raises(tmp_path, caller):
    # B4: a table without a ΔPSI column is a hard contract violation.
    with pytest.raises(ValueError, match="(?i)dpsi"):
        PARSERS[caller](_write(tmp_path, f"{caller}_nocol.tsv", "foo\tbar\nx\ty\n"))


def test_parse_nan_dpsi_is_skipped(tmp_path):
    # A1: a NaN ΔPSI row carries no usable estimate -> dropped.
    out = PARSERS["majiq"](
        _write(tmp_path, "nan.tsv", "lsv_id\te(dpsi)\tprob\nG1\tNaN\t0.9\n")
    )
    assert out == []


@pytest.mark.parametrize("caller", DPSI_CALLERS)
@pytest.mark.parametrize("bad", ["inf", "-inf", "Inf"])
def test_parse_nonfinite_dpsi_is_rejected(tmp_path, caller, bad):
    # A10: ΔPSI is a difference of fractions in [-1, 1]; inf is physically
    # impossible and must be dropped like NaN, not emitted downstream.
    header, row = _BAD[caller]
    out = PARSERS[caller](_write(tmp_path, f"{caller}_{bad}.tsv", header + row.format(dpsi=bad)))
    assert out == [], f"non-finite ΔPSI leaked through {caller} parser: {out}"


@pytest.mark.parametrize("caller", DPSI_CALLERS)
@pytest.mark.parametrize("bad", ["5.0", "-3.0", "1.5"])
def test_parse_out_of_range_dpsi_is_rejected(tmp_path, caller, bad):
    # Contract: |ΔPSI| <= 1 always. A value outside [-1, 1] is corrupt input.
    header, row = _BAD[caller]
    out = PARSERS[caller](_write(tmp_path, f"{caller}_oor{bad}.tsv", header + row.format(dpsi=bad)))
    assert out == [], f"out-of-range ΔPSI={bad} leaked through {caller} parser: {out}"


@pytest.mark.parametrize("caller", DPSI_CALLERS)
def test_parse_in_range_dpsi_is_kept(tmp_path, caller):
    # Guard against an over-aggressive fix: a valid in-range ΔPSI must survive.
    out = PARSERS[caller](_write(tmp_path, f"{caller}_good.tsv", _GOOD[caller]))
    assert len(out) == 1
    assert math.isfinite(out[0].dpsi) and -1.0 <= out[0].dpsi <= 1.0


def test_parse_bom_prefixed_file(tmp_path):
    # A7: a UTF-8 BOM (common from spreadsheet exports) must not corrupt the
    # first column / break parsing.
    out = PARSERS["majiq"](
        _write(tmp_path, "bom.tsv", "lsv_id\te(dpsi)\tprob\nG1\t0.5\t0.9\n",
               encoding="utf-8-sig")
    )
    assert len(out) == 1 and abs(out[0].dpsi - 0.5) < 1e-9


def test_parse_crlf_line_endings(tmp_path):
    # B5: Windows CRLF endings must parse identically to LF.
    out = PARSERS["majiq"](
        _write(tmp_path, "crlf.tsv", "lsv_id\te(dpsi)\tprob\r\nG1\t0.5\t0.9\r\n")
    )
    assert len(out) == 1 and abs(out[0].dpsi - 0.5) < 1e-9


# ==============================================================================
# Stage 2 — Calibrate
# ==============================================================================
def test_calibrate_empty_events_returns_empty(calibrator):
    # A2: no events -> no rows, no crash.
    assert calibrate_events([], calibrator, effect_cutoff=0.1, sig_threshold=0.05) == []


def test_corrupt_calibrator_json_errors_clearly(tmp_path):
    # B1: a truncated/garbage calibrator artifact must fail loudly, not silently
    # produce uncalibrated output.
    from braid.target.conformal import ConformalCalibrator

    bad = _write(tmp_path, "bad_cal.json", "{not valid json")
    with pytest.raises((ValueError, json.JSONDecodeError, KeyError, OSError)):
        ConformalCalibrator.from_json(bad)


# ==============================================================================
# Stage 3 — Report
# ==============================================================================
def _events_from(calibrator):  # helper: build one calibrated row deterministically
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "g.tsv")
    open(p, "w").write("lsv_id\te(dpsi)\tprob\nG1\t0.5\t0.9\n")
    return PARSERS["majiq"](p)


def test_report_empty_rows_do_not_crash(tmp_path):
    # A2: every writer must tolerate zero rows (return schema, not exception).
    assert write_tsv([], str(tmp_path / "e.tsv")) is None
    assert (tmp_path / "e.tsv").exists()
    assert write_excel([], str(tmp_path / "e.xlsx")) in (True, False)
    assert make_figure([], str(tmp_path / "e")) is False  # nothing to draw


def test_report_nan_interval_row_does_not_crash(tmp_path, calibrator):
    # A10: a NaN interval must not blow up the figure renderer.
    rows = calibrate_events(_events_from(calibrator), calibrator,
                            effect_cutoff=0.1, sig_threshold=0.05)
    rows[0] = {**rows[0], "dpsi": float("nan"),
               "ci_low": float("nan"), "ci_high": float("nan")}
    write_tsv(rows, str(tmp_path / "n.tsv"))
    assert (tmp_path / "n.tsv").exists()
    assert make_figure(rows, str(tmp_path / "n")) in (True, False)


# ==============================================================================
# End-to-end (CLI boundary)
# ==============================================================================
def test_filter_cli_missing_input_clear_error(tmp_path):
    # C1: a non-existent input path must produce a clear FileNotFoundError, not a
    # confusing downstream stack trace.
    import argparse

    from braid.commands.filter_cmd import run_filter

    args = argparse.Namespace(
        input=str(tmp_path / "does_not_exist.tsv"), caller="majiq", example=False,
        output=str(tmp_path / "o"), effect_cutoff=0.1, sig_threshold=0.05,
        min_support=20, strict=False, calibration=None, top_n=25,
        make_figure=False, verbose=False,
    )
    with pytest.raises(FileNotFoundError, match="(?i)not found"):
        run_filter(args)
