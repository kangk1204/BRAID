"""Pipeline-robustness suite for the `braid differential` ΔPSI chain.

Complements ``test_pipeline_robustness.py`` (which covers the ``braid filter``
parse->calibrate->report path). Here the unit is the rMATS two-group differential
pipeline: parse -> PSI/ΔPSI posterior -> conformal interval -> tier -> write.

Each test maps to one (stage, failure-mode) cell of the matrix in
``FAILURE_MATRIX.md`` and asserts something non-trivial (return schema or an error
``match=``), never just "did not crash".

Failure-mode IDs: A1 missing/NA · A2 empty · A4 type mismatch · A5 schema drift ·
A6 out-of-range · A9 extreme scale · A10 inf/NaN · B4 row/header misalignment ·
C1 file-not-found · F4 contract.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pytest

from braid.commands.differential import (
    _resolve_differential_calibrator,
    run_differential,
)

# rMATS SE table header (real rMATS-turbo JC schema).
_HEADER = [
    "ID", "GeneID", "geneSymbol", "chr", "strand", "exonStart_0base", "exonEnd",
    "upstreamES", "upstreamEE", "downstreamES", "downstreamEE",
    "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
]


def _row(eid, gene, es, inc1, sjc1, inc2, sjc2, fdr, dpsi,
         inc_fl="100", skip_fl="100"):
    psi1 = inc1 / (inc1 + sjc1) if (inc1 + sjc1) else 0.0
    psi2 = inc2 / (inc2 + sjc2) if (inc2 + sjc2) else 0.0
    return [
        str(eid), gene, gene, "chr1", "+", str(es), str(es + 100),
        str(es - 200), str(es - 100), str(es + 200), str(es + 300),
        str(inc1), str(sjc1), str(inc2), str(sjc2),
        inc_fl, skip_fl, "0.0", str(fdr),
        f"{psi1:.3f}", f"{psi2:.3f}", str(dpsi),
    ]


def _write_table(rmats_dir: Path, rows, header=_HEADER) -> None:
    rmats_dir.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)] + ["\t".join(r) for r in rows]
    (rmats_dir / "SE.MATS.JC.txt").write_text("\n".join(lines) + "\n")


def _args(rmats_dir: Path, out: Path, **kw) -> argparse.Namespace:
    base = dict(
        rmats_dir=str(rmats_dir), output=str(out), replicates=500, confidence=0.95,
        fdr=0.05, effect_cutoff=0.1, min_support=20, seed=42, verbose=False,
        ctrl=None, treat=None, strict=False, use_conformal=True, calibration=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ==============================================================================
# S3 — parse stage: count-vector & schema pathologies
# ==============================================================================
def test_S3_A1_trailing_comma_count_vector_is_kept() -> None:
    """A benign trailing comma / trailing NA (empty final replicate) is dropped
    quietly, not treated as corruption."""
    from braid.target.rmats_bootstrap import _parse_count_vector

    assert _parse_count_vector("10,20,") == (10, 20)
    assert _parse_count_vector("10,20,NA") == (10, 20)


def test_S3_A1_interior_gap_count_vector_is_rejected() -> None:
    """An interior empty/NA cell would shift replicate indices and desync the
    inclusion/exclusion pairing -> must hard-error, never silently re-index."""
    from braid.target.rmats_bootstrap import _parse_count_vector

    with pytest.raises(ValueError, match="interior empty/NA"):
        _parse_count_vector("10,,20")
    with pytest.raises(ValueError, match="interior empty/NA"):
        _parse_count_vector("10,NA,20")


def test_S3_A4_non_integer_count_skips_row_nonstrict(tmp_path: Path) -> None:
    """A non-integer junction count is malformed; non-strict skips the row (event
    ABSENT from output), never crashes or coerces."""
    bad = _row(1, "BAD", 1000, 180, 20, 60, 140, 0.001, 0.6)
    bad[11] = "18x0"  # IJC_SAMPLE_1 garbage
    good = _row(2, "GOOD", 5000, 100, 100, 100, 100, 0.5, 0.0)
    _write_table(tmp_path / "rmats", [bad, good])
    out = tmp_path / "out.tsv"
    run_differential(_args(tmp_path / "rmats", out))
    genes = {r["gene"] for r in _read_tsv(out)}
    assert "BAD" not in genes
    assert "GOOD" in genes


def test_S3_A4_non_integer_count_fails_fast_strict(tmp_path: Path) -> None:
    """--strict turns the same malformed row into a hard failure (data-integrity)."""
    bad = _row(1, "BAD", 1000, 180, 20, 60, 140, 0.001, 0.6)
    bad[11] = "18x0"
    _write_table(tmp_path / "rmats", [bad])
    with pytest.raises(ValueError, match="Malformed rMATS row"):
        run_differential(_args(tmp_path / "rmats", tmp_path / "out.tsv", strict=True))


def test_S3_A5_missing_required_count_column_fails_fast(tmp_path: Path) -> None:
    """A table missing a required count column (SJC_SAMPLE_2) is self-consistent
    (len(fields)==len(header)) so the row-truncation guard does NOT fire. Without a
    column-presence check, every treat exclusion count silently becomes 0,
    fabricating treat_psi=1.0 and a spurious large ΔPSI. The parser must fail fast."""
    header = [h for h in _HEADER if h != "SJC_SAMPLE_2"]
    full = _row(1, "BIG", 1000, 180, 20, 60, 140, 0.001, 0.6)
    row = [v for h, v in zip(_HEADER, full) if h != "SJC_SAMPLE_2"]
    _write_table(tmp_path / "rmats", [row], header=header)
    with pytest.raises(ValueError, match="missing required count column"):
        run_differential(_args(tmp_path / "rmats", tmp_path / "out.tsv"))


def test_S3_A2_empty_table_file_clear_error(tmp_path: Path) -> None:
    """A 0-byte / blank-header rMATS table must fail fast with an 'empty/no header'
    message, NOT be reported as every count column missing (clearer than the
    schema-drift path, and safer than silently returning zero events)."""
    from braid.target.rmats_bootstrap import parse_rmats_output

    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir(parents=True)
    (rmats_dir / "SE.MATS.JC.txt").write_text("")  # 0 bytes
    with pytest.raises(ValueError, match="empty or has no header"):
        parse_rmats_output(str(rmats_dir), min_total_count=10)


def test_S3_B4_truncated_row_skipped_nonstrict(tmp_path: Path) -> None:
    """A row with fewer fields than the header is skipped (non-strict)."""
    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir(parents=True)
    good = "\t".join(_row(1, "GOOD", 1000, 180, 20, 60, 140, 0.001, 0.6))
    truncated = "\t".join(["1", "T", "T", "chr1", "+"])  # 5 << 22 fields
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join(_HEADER) + "\n" + good + "\n" + truncated + "\n"
    )
    out = tmp_path / "out.tsv"
    run_differential(_args(rmats_dir, out))
    assert {r["gene"] for r in _read_tsv(out)} == {"GOOD"}


# ==============================================================================
# S4 — PSI/ΔPSI posterior stage
# ==============================================================================
def test_S4_A2_all_below_support_writes_header_only(tmp_path: Path) -> None:
    """When no event clears --min-support the file still exists with just a header
    (never a crash, never a silent missing file)."""
    rows = [_row(1, "TINY", 1000, 2, 1, 1, 2, 0.5, 0.0)]  # support 3 << 20
    _write_table(tmp_path / "rmats", rows)
    out = tmp_path / "out.tsv"
    run_differential(_args(tmp_path / "rmats", out))
    assert out.exists()
    assert _read_tsv(out) == []
    assert out.read_text(encoding="utf-8").startswith("event_id\t")


def test_S4_A9_extreme_form_len_ratio_stays_finite(tmp_path: Path) -> None:
    """A large IncFormLen/SkipFormLen ratio must not produce NaN/inf PSI or CI."""
    row = _row(1, "EXT", 1000, 100, 100, 100, 100, 0.01, 0.0,
               inc_fl="100000", skip_fl="1")
    _write_table(tmp_path / "rmats", [row])
    out = tmp_path / "out.tsv"
    run_differential(_args(tmp_path / "rmats", out, use_conformal=False))
    r = _read_tsv(out)[0]
    for col in ("dpsi", "ci_low", "ci_high", "ctrl_psi", "treat_psi"):
        assert np.isfinite(float(r[col]))
        assert -1.0 <= float(r[col]) <= 1.0


# ==============================================================================
# S5 — conformal stage
# ==============================================================================
def test_S5_A10_robust_interval_nonfinite_scale_returns_full_clip() -> None:
    """NaN/inf sampling spread (or point estimate) must yield the conservative full
    clip, never (nan, nan)."""
    from braid.target.conformal import ConformalCalibrator

    cal = ConformalCalibrator(alpha=0.05, q_global=0.3, q_by_bin={},
                              scale_kind="absolute_dpsi")
    assert cal.robust_interval(0.2, float("nan"), 100.0, clip=(-1.0, 1.0)) == (-1.0, 1.0)
    assert cal.robust_interval(0.2, float("inf"), 100.0, clip=(-1.0, 1.0)) == (-1.0, 1.0)
    assert cal.robust_interval(float("nan"), 0.01, 100.0, clip=(-1.0, 1.0)) == (-1.0, 1.0)


def test_S5_A6_negative_or_nan_support_routes_to_widest_bin() -> None:
    """Out-of-range support must get the widest (most conservative) bin, never the
    tightest 250+ bin (np.digitize would send NaN to the top bin)."""
    from braid.target.conformal import assign_support_bins

    labels = assign_support_bins(np.array([-5.0, np.nan, np.inf, 300.0]))
    assert list(labels[:3]) == ["<20", "<20", "<20"]
    assert labels[3] == "250+"


def test_S5_F4_scale_kind_mismatch_is_rejected(tmp_path: Path) -> None:
    """Loading a posterior_std calibrator into the absolute_dpsi differential mode
    silently changes interval units -> must fail fast with a clear message."""
    from braid.target.conformal import ConformalCalibrator

    wrong = ConformalCalibrator(alpha=0.05, q_global=0.3, q_by_bin={},
                                scale_kind="posterior_std")
    path = tmp_path / "wrong_scale.json"
    wrong.to_json(path)
    args = argparse.Namespace(use_conformal=True, calibration=str(path))
    with pytest.raises(ValueError, match="scale_kind"):
        _resolve_differential_calibrator(args)


def test_S5_C1_missing_calibration_file_clear_error(tmp_path: Path) -> None:
    """A user-supplied --calibration path that does not exist must produce a clear,
    calibration-scoped error, not a raw errno FileNotFoundError."""
    args = argparse.Namespace(
        use_conformal=True, calibration=str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError, match="--calibration"):
        _resolve_differential_calibrator(args)


def test_S5_C2_calibration_directory_clear_error(tmp_path: Path) -> None:
    """--calibration pointing at a directory (or other unreadable non-missing path)
    must produce a clear --calibration-scoped error, not a raw IsADirectoryError."""
    a_dir = tmp_path / "cal_dir"
    a_dir.mkdir()
    args = argparse.Namespace(use_conformal=True, calibration=str(a_dir))
    with pytest.raises(OSError, match="--calibration path could not be read"):
        _resolve_differential_calibrator(args)


def test_S5_F4_malformed_calibration_json_clear_error(tmp_path: Path) -> None:
    """A malformed/garbage --calibration JSON, or a valid JSON with the wrong schema,
    must produce a clear calibration-scoped error -- not a raw json.JSONDecodeError /
    KeyError leaking from the loader."""
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json ]")
    args = argparse.Namespace(use_conformal=True, calibration=str(bad))
    with pytest.raises(ValueError, match="not a valid"):
        _resolve_differential_calibrator(args)

    wrong_schema = tmp_path / "wrong_schema.json"
    wrong_schema.write_text('{"q_global": 0.3}')  # missing required 'alpha'
    args2 = argparse.Namespace(use_conformal=True, calibration=str(wrong_schema))
    with pytest.raises(ValueError, match="not a valid"):
        _resolve_differential_calibrator(args2)


# ==============================================================================
# S6 — tier stage
# ==============================================================================
def test_S6_A10_nan_fdr_yields_not_significant_caller_flag(tmp_path: Path) -> None:
    """An rMATS row with FDR=NA must keep the event (it still gets a CI) but the
    caller-significant flag must be False (np.isfinite guard), never crash."""
    row = _row(1, "NAFDR", 1000, 180, 20, 60, 140, "NA", "NA")
    _write_table(tmp_path / "rmats", [row])
    out = tmp_path / "out.tsv"
    run_differential(_args(tmp_path / "rmats", out, use_conformal=False))
    r = _read_tsv(out)[0]
    assert r["gene"] == "NAFDR"
    assert r["caller_significant"] == "no"
    assert r["rmats_fdr"].lower() in ("nan", "-nan")


# ==============================================================================
# S7 — output write stage
# ==============================================================================
def test_S7_A7_csv_formula_injection_in_gene_is_neutralized(tmp_path: Path) -> None:
    """A gene symbol beginning with a spreadsheet formula leader must be written as
    inert text (apostrophe-guarded), never as a live formula."""
    row = _row(1, "=cmd|'/c calc'!A1", 1000, 180, 20, 60, 140, 0.001, 0.6)
    _write_table(tmp_path / "rmats", [row])
    out = tmp_path / "out.tsv"
    run_differential(_args(tmp_path / "rmats", out, use_conformal=False))
    raw = out.read_text(encoding="utf-8").splitlines()[1]
    gene_cell = raw.split("\t")[2]
    assert gene_cell.startswith("'=")  # guarded
