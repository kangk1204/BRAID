"""Regression tests for three input/output contract fixes.

1. braid psi: --min-support is applied to the REPORTED --sample's own support, and
   --sample selects which condition's PSI is emitted (no evidence-free PSI rows).
2. braid filter --caller rmats: the sampling SD is length-normalized like the
   differential path, so both rMATS paths report the same interval width.
3. braid run --no-conformal: honoured in differential mode (was silently dropped).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest

_HEADER = [
    "ID", "GeneID", "geneSymbol", "chr", "strand", "exonStart_0base", "exonEnd",
    "upstreamES", "upstreamEE", "downstreamES", "downstreamEE",
    "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
]


def _row(eid, gene, inc1, sjc1, inc2, sjc2, fdr, dpsi, inc_fl=100, skip_fl=100):
    def psi(i, s):
        return i / (i + s) if (i + s) else 0.0
    return [
        str(eid), gene, gene, "chr1", "+", "1000", "1100", "800", "900",
        "1200", "1300", str(inc1), str(sjc1), str(inc2), str(sjc2),
        str(inc_fl), str(skip_fl), "0.0", str(fdr),
        f"{psi(inc1, sjc1):.3f}", f"{psi(inc2, sjc2):.3f}", str(dpsi),
    ]


def _write(rmats_dir: Path, rows) -> None:
    rmats_dir.mkdir(parents=True, exist_ok=True)
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\n".join(["\t".join(_HEADER)] + ["\t".join(r) for r in rows]) + "\n"
    )


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# --------------------------------------------------------------------------- #
# Fix 1: braid psi per-sample min-support + --sample
# --------------------------------------------------------------------------- #
def _psi_args(rmats_dir, out, sample="sample_1", min_support=10):
    return argparse.Namespace(
        bam=None, rmats_dir=str(rmats_dir), output=str(out), replicates=200,
        confidence=0.95, min_support=min_support, seed=42, verbose=False,
        use_conformal=True, calibration=None, allow_replicate_fallback=False,
        sample=sample,
    )


def test_psi_min_support_applies_to_reported_sample(tmp_path: Path):
    from braid.commands.psi import run_psi
    # No sample_1 evidence (0/0), large sample_2 (80/20): the two-group total (100)
    # passes min_support, but sample_1 has nothing to report.
    _write(tmp_path / "rmats", [_row(1, "ASYM", 0, 0, 80, 20, 0.01, -0.8)])

    out1 = tmp_path / "s1.tsv"
    run_psi(_psi_args(tmp_path / "rmats", out1, sample="sample_1"))
    assert len(_read(out1)) == 0   # filtered: no evidence-free PSI row for sample_1

    out2 = tmp_path / "s2.tsv"
    run_psi(_psi_args(tmp_path / "rmats", out2, sample="sample_2"))
    rows = _read(out2)
    assert len(rows) == 1
    assert abs(float(rows[0]["PSI"]) - 0.8) < 0.05   # sample_2 PSI = 80/100
    assert rows[0]["sample"] == "sample_2"           # output self-describes the condition


def test_psi_sample_2_rejected_in_multireplicate(tmp_path: Path):
    """--sample sample_2 must not be silently misrouted to sample_1 in multi-rep mode."""
    from braid.commands.psi import run_psi
    _write(tmp_path / "rmats", [_row(1, "G", 20, 80, 80, 20, 0.01, -0.6)])
    args = _psi_args(tmp_path / "rmats", tmp_path / "o.tsv", sample="sample_2")
    args.bam = ["r1.bam", "r2.bam"]   # two BAMs -> multi-replicate mode
    with pytest.raises(SystemExit):
        run_psi(args)


def test_rmats_strict_hard_fails_on_malformed_row(tmp_path: Path):
    """--strict raises on a malformed (interior-gap) rMATS row; default skips it."""
    from braid.target.rmats_bootstrap import parse_rmats_output
    rmats = tmp_path / "r"
    rmats.mkdir()
    bad = [
        "1", "A", "A", "chr1", "+", "1000", "1100", "800", "900", "1200", "1300",
        "10,NA,20", "5,7,9", "30", "10", "100", "100", "0", "0.01", "0.6", "0.5", "0.1",
    ]
    (rmats / "SE.MATS.JC.txt").write_text(
        "\t".join(_HEADER) + "\n" + "\t".join(bad) + "\n"
    )
    assert parse_rmats_output(str(rmats), min_total_count=0) == []   # default: skip
    with pytest.raises(ValueError, match="Malformed rMATS row"):
        parse_rmats_output(str(rmats), min_total_count=0, strict=True)


def test_rmats_strict_hard_fails_on_truncated_row(tmp_path: Path):
    """A truncated row (fewer columns than the header) must also hard-fail under
    --strict, not be silently dropped before the strict check."""
    from braid.target.rmats_bootstrap import parse_rmats_output
    rmats = tmp_path / "r"
    rmats.mkdir()
    (rmats / "SE.MATS.JC.txt").write_text(
        "\t".join(_HEADER) + "\n" + "1\tA\tA\n"   # only 3 of 22 columns
    )
    assert parse_rmats_output(str(rmats), min_total_count=0) == []   # default: skip
    with pytest.raises(ValueError, match="Truncated rMATS row"):
        parse_rmats_output(str(rmats), min_total_count=0, strict=True)


def test_csv_safe_neutralises_formula_leaders():
    """The shared output-safety helper used by every BRAID table writer."""
    from braid.output_safety import csv_restore, csv_safe
    for bad in ("=A", "+A", "-A", "@A", "\tA"):
        assert csv_safe(bad) == "'" + bad
    assert csv_safe("ENSG001") == "ENSG001"   # normal value unchanged
    assert csv_safe("") == ""
    # csv_safe/csv_restore are an exact bijection, incl. apostrophe-leading values
    for original in ("=REAL", "'=REAL", "@x", "'abc", "ENSG1", "0.75", ""):
        assert csv_restore(csv_safe(original)) == original


# --------------------------------------------------------------------------- #
# Fix 2: filter --caller rmats sampling SD is length-normalized
# --------------------------------------------------------------------------- #
def test_filter_rmats_sampling_sd_is_length_normalized(tmp_path: Path):
    from braid.adapters import parse_rmats
    from braid.adapters.base import beta_dpsi_std
    # IncFormLen=200, SkipFormLen=100 -> ratio 2.0; exclusion counts rescaled x2.
    _write(tmp_path / "rmats",
           [_row(1, "G", 40, 10, 10, 40, 0.01, 0.5, inc_fl=200, skip_fl=100)])
    ev = parse_rmats(str(tmp_path / "rmats"), min_support=0)[0]
    normalized = beta_dpsi_std(40, 10 * 2.0, 10, 40 * 2.0)
    raw = beta_dpsi_std(40, 10, 10, 40)
    assert abs(ev.sampling_std - normalized) < 1e-9   # matches differential's scale
    assert abs(ev.sampling_std - raw) > 1e-6          # genuinely not the raw scale


# --------------------------------------------------------------------------- #
# Fix 3: braid run --no-conformal honoured in differential mode
# --------------------------------------------------------------------------- #
def test_run_differential_honours_no_conformal(tmp_path: Path):
    from braid.commands.run import _run_differential_mode
    _write(tmp_path / "rmats", [_row(1, "BIG", 180, 20, 60, 140, 0.001, 0.6)])

    def width(use_conformal: bool, outdir: Path) -> float:
        outdir.mkdir()
        args = argparse.Namespace(
            ctrl=None, treat=None, rmats=str(tmp_path / "rmats"),
            replicates=2000, confidence=0.95, effect_cutoff=0.1, fdr=0.05,
            min_support=20, seed=42, verbose=False,
            use_conformal=use_conformal, calibration=None,
        )
        _run_differential_mode(args, str(outdir))
        big = {r["gene"]: r for r in _read(outdir / "differential.tsv")}["BIG"]
        return float(big["ci_high"]) - float(big["ci_low"])

    w_on = width(True, tmp_path / "on")
    w_off = width(False, tmp_path / "off")
    # conformal ON uses the RT-PCR residual quantile (~0.34 half-width) -> wider than
    # the raw Jeffreys posterior percentile interval. If --no-conformal were dropped,
    # the two would be identical.
    assert w_on > w_off + 0.05
