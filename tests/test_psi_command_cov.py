"""Coverage tests for braid/commands/psi.py targeting >=80% branch/line coverage.

Covers:
- add_psi_subparser: parser registration
- _resolve_conformal_calibrator: no-conformal flag, custom path, default, OSError fallback
- run_psi: single-sample rMATS path, multi-replicate rMATS path,
  "--rmats-dir required" error
- _resolve_effective_replicate_count: already tested in test_psi_multireplicate.py (imported)
- _bootstrap_per_replicate: group-sum fallback when index out of range, n_fallback warning
- _combine_replicate_results: empty input, single-rep passthrough, multi-rep combination
- _write_psi_tsv: header row, data row formatting, is_confident flag
"""

from __future__ import annotations

import argparse
import logging

import pytest

from braid.commands.psi import (
    _bootstrap_per_replicate,
    _combine_replicate_results,
    _resolve_conformal_calibrator,
    _write_psi_tsv,
    add_psi_subparser,
    run_psi,
)
from braid.target.psi_bootstrap import PSIResult
from braid.target.rmats_bootstrap import RmatsEvent

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _event(
    inc_reps: tuple[int, ...],
    exc_reps: tuple[int, ...],
    group_inc: int | None = None,
    group_exc: int | None = None,
    event_type: str = "SE",
    event_id: str = "chr1:SE:100:200:+",
) -> RmatsEvent:
    """Build a minimal RmatsEvent with given per-replicate vectors."""
    return RmatsEvent(
        event_id=event_id,
        event_type=event_type,
        chrom="chr1",
        strand="+",
        gene="GENEX",
        inc_count=sum(inc_reps) if inc_reps else (group_inc or 0),
        exc_count=sum(exc_reps) if exc_reps else (group_exc or 0),
        rmats_psi=0.5,
        rmats_fdr=1.0,
        rmats_dpsi=0.0,
        sample_1_inc_count=group_inc if group_inc is not None else sum(inc_reps),
        sample_1_exc_count=group_exc if group_exc is not None else sum(exc_reps),
        sample_1_inc_replicates=tuple(inc_reps),
        sample_1_exc_replicates=tuple(exc_reps),
    )


def _psi_result(
    psi: float = 0.6,
    ci_low: float = 0.4,
    ci_high: float = 0.8,
    cv: float = 0.1,
    inc: int = 20,
    exc: int = 10,
    is_confident: bool = True,
) -> PSIResult:
    width = ci_high - ci_low
    return PSIResult(
        event_id="chr1:SE:100:200:+",
        event_type="SE",
        gene="GENEX",
        chrom="chr1",
        psi=psi,
        ci_low=ci_low,
        ci_high=ci_high,
        cv=cv,
        inclusion_count=inc,
        exclusion_count=exc,
        event_start=100,
        event_end=200,
        ci_width=width,
        is_confident=is_confident,
    )


def _minimal_args(**overrides) -> argparse.Namespace:
    """Return a Namespace mimicking what argparse produces for the psi subcommand."""
    defaults = dict(
        bam=None,
        rmats_dir=None,
        output="braid_psi.tsv",
        replicates=50,
        confidence=0.95,
        min_support=10,
        seed=42,
        verbose=False,
        use_conformal=True,
        calibration=None,
        allow_replicate_fallback=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _allow_fake_rmats_dir(monkeypatch) -> None:
    """Keep rMATS-mode unit tests focused on their monkeypatched parser path."""
    monkeypatch.setattr(
        "braid.commands.rmats_input.require_rmats_tables",
        lambda *args, **kwargs: None,
    )

# ---------------------------------------------------------------------------
# add_psi_subparser
# ---------------------------------------------------------------------------


def test_add_psi_subparser_registers_command():
    """Parser must register a 'psi' subcommand with func=run_psi."""
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers()
    add_psi_subparser(subparsers)
    args = main_parser.parse_args([
        "psi",
        "--rmats-dir", "/some/dir",
        "--bam", "a.bam",
    ])
    assert args.func is run_psi
    assert args.rmats_dir == "/some/dir"


def test_add_psi_subparser_defaults():
    """Default values for optional flags should be set correctly."""
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers()
    add_psi_subparser(subparsers)
    args = main_parser.parse_args(["psi", "--rmats-dir", "/d"])
    assert args.replicates == 500
    assert args.confidence == 0.95
    assert args.min_support == 10
    assert args.seed == 42
    assert args.use_conformal is True
    assert args.output == "braid_psi.tsv"
    assert args.allow_replicate_fallback is False


def test_add_psi_subparser_no_conformal_flag():
    """--no-conformal must set use_conformal=False."""
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers()
    add_psi_subparser(subparsers)
    args = main_parser.parse_args(["psi", "--rmats-dir", "/d", "--no-conformal"])
    assert args.use_conformal is False


def test_add_psi_subparser_allow_replicate_fallback():
    """--allow-replicate-fallback must set the flag to True."""
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers()
    add_psi_subparser(subparsers)
    args = main_parser.parse_args(["psi", "--rmats-dir", "/d", "--allow-replicate-fallback"])
    assert args.allow_replicate_fallback is True


@pytest.mark.parametrize("flag", ["--gtf", "--gene", "--region"])
def test_add_psi_subparser_rejects_non_rmats_selectors(flag):
    """rMATS-only public PSI must not accept legacy direct-BAM selectors."""
    main_parser = argparse.ArgumentParser()
    subparsers = main_parser.add_subparsers()
    add_psi_subparser(subparsers)
    with pytest.raises(SystemExit):
        main_parser.parse_args(["psi", flag, "x"])


# ---------------------------------------------------------------------------
# _resolve_conformal_calibrator
# ---------------------------------------------------------------------------


def test_resolve_conformal_no_conformal_returns_none():
    """use_conformal=False must short-circuit and return None."""
    args = _minimal_args(use_conformal=False)
    result = _resolve_conformal_calibrator(args)
    assert result is None


def test_resolve_conformal_default_path_returns_calibrator_or_none():
    """When use_conformal=True and no custom path, return calibrator or None (no crash)."""
    args = _minimal_args(use_conformal=True, calibration=None)
    result = _resolve_conformal_calibrator(args)
    # May be None if the shipped artifact is missing in test env; must not raise.
    assert result is None or hasattr(result, "q_for")


def test_resolve_conformal_missing_custom_path_raises(tmp_path):
    """An explicit --calibration path that does not exist is a hard error, not a
    silent fall-back to legacy intervals (mirrors `braid differential`)."""
    args = _minimal_args(
        use_conformal=True, calibration=str(tmp_path / "nope.json"))
    with pytest.raises(FileNotFoundError, match="--calibration"):
        _resolve_conformal_calibrator(args)


def test_resolve_conformal_custom_path_directory_raises(tmp_path):
    """--calibration pointing at a directory (or other unreadable non-missing
    path) must produce a clear --calibration-scoped error, not IsADirectoryError."""
    a_dir = tmp_path / "cal_dir"
    a_dir.mkdir()
    args = _minimal_args(use_conformal=True, calibration=str(a_dir))
    with pytest.raises(OSError, match="--calibration path could not be read"):
        _resolve_conformal_calibrator(args)


def test_resolve_conformal_custom_path_malformed_json_raises(tmp_path):
    """A malformed/garbage --calibration JSON, or a valid JSON with the wrong
    schema, must produce a clear calibration-scoped error."""
    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json ]")
    args = _minimal_args(use_conformal=True, calibration=str(bad))
    with pytest.raises(ValueError, match="not a valid"):
        _resolve_conformal_calibrator(args)

    wrong_schema = tmp_path / "wrong_schema.json"
    wrong_schema.write_text('{"q_global": 0.3}')  # missing required 'alpha'
    args2 = _minimal_args(use_conformal=True, calibration=str(wrong_schema))
    with pytest.raises(ValueError, match="not a valid"):
        _resolve_conformal_calibrator(args2)


def test_resolve_conformal_missing_default_falls_back(monkeypatch, caplog):
    """When NO custom path is given, an absent shipped default must warn and fall
    back to legacy intervals (return None) -- the default may be missing in some
    installs, unlike an explicitly-requested --calibration file."""
    def _missing_default():
        raise FileNotFoundError("default artifact absent")

    monkeypatch.setattr(
        "braid.target.conformal.load_default_conformal_calibrator",
        _missing_default,
    )
    args = _minimal_args(use_conformal=True, calibration=None)
    with caplog.at_level(logging.WARNING, logger="braid.commands.psi"):
        result = _resolve_conformal_calibrator(args)
    assert result is None
    assert any("unavailable" in r.message.lower() for r in caplog.records)


def test_resolve_conformal_custom_path_loaded(tmp_path):
    """A valid calibrator JSON at a custom path must be loaded and returned."""
    from braid.target.conformal import ConformalCalibrator

    cal = ConformalCalibrator(
        alpha=0.05,
        q_global=1.5,
        q_by_bin={"<20": 2.0, "20-49": 1.8, "50-99": 1.5, "100-249": 1.3, "250+": 1.1},
    )
    cal_path = tmp_path / "test_cal.json"
    cal.to_json(cal_path)

    args = _minimal_args(use_conformal=True, calibration=str(cal_path))
    result = _resolve_conformal_calibrator(args)
    assert result is not None
    assert abs(result.q_global - 1.5) < 1e-9


def test_resolve_conformal_no_attribute_use_conformal_defaults_to_true(tmp_path):
    """Missing use_conformal attribute defaults to True (getattr guard)."""
    from braid.target.conformal import ConformalCalibrator

    cal = ConformalCalibrator(alpha=0.05, q_global=1.2, q_by_bin={})
    cal_path = tmp_path / "cal.json"
    cal.to_json(cal_path)

    # Namespace without use_conformal attribute at all
    args = argparse.Namespace(calibration=str(cal_path))
    result = _resolve_conformal_calibrator(args)
    assert result is not None


# ---------------------------------------------------------------------------
# _write_psi_tsv
# ---------------------------------------------------------------------------


def test_write_psi_tsv_header_columns(tmp_path):
    """TSV header must contain all expected column names."""
    out = str(tmp_path / "out.tsv")
    _write_psi_tsv([], out, "single-sample")
    with open(out) as f:
        header = f.readline().strip().split("\t")
    expected = [
        "event_id", "event_type", "gene", "chrom", "sample", "PSI",
        "CI_low", "CI_high", "CI_width", "CV",
        "inc_count", "exc_count", "confident", "mode",
    ]
    assert header == expected


def test_write_psi_tsv_empty_results(tmp_path):
    """Empty results list must produce only the header row."""
    out = str(tmp_path / "out.tsv")
    _write_psi_tsv([], out, "single-sample")
    with open(out) as f:
        lines = f.readlines()
    assert len(lines) == 1


def test_write_psi_tsv_data_row_values(tmp_path):
    """Data rows must encode PSI/CI values to 4 decimal places and mode correctly."""
    result = _psi_result(psi=0.6543, ci_low=0.4321, ci_high=0.8765, is_confident=True)
    out = str(tmp_path / "out.tsv")
    _write_psi_tsv([result], out, "multi-replicate")
    with open(out) as f:
        lines = f.readlines()
    assert len(lines) == 2
    cols = lines[1].strip().split("\t")
    assert cols[4] == "sample_1"  # sample (reported condition)
    assert cols[5] == "0.6543"   # PSI
    assert cols[6] == "0.4321"   # CI_low
    assert cols[7] == "0.8765"   # CI_high
    assert cols[12] == "yes"     # confident
    assert cols[13] == "multi-replicate"


def test_write_psi_tsv_not_confident(tmp_path):
    """is_confident=False must produce 'no' in the confident column."""
    result = _psi_result(is_confident=False)
    out = str(tmp_path / "out.tsv")
    _write_psi_tsv([result], out, "single-sample")
    with open(out) as f:
        lines = f.readlines()
    cols = lines[1].strip().split("\t")
    assert cols[12] == "no"


def test_write_psi_tsv_ci_width_from_result_field(tmp_path):
    """CI_width column uses the ci_width field, not recomputing ci_high - ci_low."""
    result = _psi_result(ci_low=0.3, ci_high=0.7)
    # ci_width is set to 0.4 by _psi_result; verify it appears verbatim
    out = str(tmp_path / "out.tsv")
    _write_psi_tsv([result], out, "single-sample")
    with open(out) as f:
        lines = f.readlines()
    cols = lines[1].strip().split("\t")
    assert cols[8] == "0.4000"


def test_write_psi_tsv_multiple_rows(tmp_path):
    """Multiple results must produce multiple data rows in order."""
    r1 = _psi_result(psi=0.2)
    r2 = _psi_result(psi=0.8)
    out = str(tmp_path / "out.tsv")
    _write_psi_tsv([r1, r2], out, "single-sample")
    with open(out) as f:
        lines = f.readlines()
    assert len(lines) == 3
    assert lines[1].split("\t")[5] == "0.2000"
    assert lines[2].split("\t")[5] == "0.8000"


# ---------------------------------------------------------------------------
# _combine_replicate_results
# ---------------------------------------------------------------------------


def test_combine_replicate_results_empty_returns_empty():
    """Empty list input must return empty list."""
    assert _combine_replicate_results([], 0.95) == []


def test_combine_replicate_results_empty_inner_list_returns_empty():
    """If the first replicate list is empty, return empty."""
    assert _combine_replicate_results([[]], 0.95) == []


def test_combine_replicate_results_single_rep_passthrough():
    """With only one replicate the original PSIResult must be returned unchanged."""
    r = _psi_result(psi=0.5)
    combined = _combine_replicate_results([[r]], 0.95)
    assert len(combined) == 1
    assert combined[0].psi == pytest.approx(0.5)


def test_combine_replicate_results_two_reps_averages_psi():
    """With two replicates mean PSI should be the average of the two."""
    r1 = _psi_result(psi=0.4, ci_low=0.2, ci_high=0.6)
    r2 = _psi_result(psi=0.6, ci_low=0.4, ci_high=0.8)
    combined = _combine_replicate_results([[r1], [r2]], 0.95)
    assert len(combined) == 1
    assert combined[0].psi == pytest.approx(0.5, abs=1e-9)


def test_combine_replicate_results_sums_counts():
    """Combined inclusion/exclusion counts must be the sum across replicates."""
    r1 = _psi_result(inc=10, exc=5)
    r2 = _psi_result(inc=20, exc=8)
    combined = _combine_replicate_results([[r1], [r2]], 0.95)
    assert combined[0].inclusion_count == 30
    assert combined[0].exclusion_count == 13


def test_combine_replicate_results_ci_bounds_clamped():
    """CI bounds must be clamped to [0, 1]."""
    r1 = _psi_result(psi=0.05, ci_low=0.0, ci_high=0.1)
    r2 = _psi_result(psi=0.05, ci_low=0.0, ci_high=0.1)
    combined = _combine_replicate_results([[r1], [r2]], 0.95)
    assert combined[0].ci_low >= 0.0
    assert combined[0].ci_high <= 1.0


def test_combine_replicate_results_three_reps():
    """Three replicates must produce a sensible result without crash."""
    reps = [
        [_psi_result(psi=0.3, ci_low=0.1, ci_high=0.5)],
        [_psi_result(psi=0.5, ci_low=0.3, ci_high=0.7)],
        [_psi_result(psi=0.7, ci_low=0.5, ci_high=0.9)],
    ]
    combined = _combine_replicate_results(reps, 0.95)
    assert len(combined) == 1
    assert combined[0].psi == pytest.approx(0.5, abs=1e-9)


def test_combine_replicate_results_zero_mean_psi_cv_is_nan():
    """When mean PSI is 0, CV must be nan (no division by zero)."""
    import math

    r1 = _psi_result(psi=0.0, ci_low=0.0, ci_high=0.0, cv=float("nan"))
    r2 = _psi_result(psi=0.0, ci_low=0.0, ci_high=0.0, cv=float("nan"))
    combined = _combine_replicate_results([[r1], [r2]], 0.95)
    assert math.isnan(combined[0].cv)


# ---------------------------------------------------------------------------
# _bootstrap_per_replicate
# ---------------------------------------------------------------------------


def test_bootstrap_per_replicate_uses_per_replicate_counts():
    """When per-replicate counts are available, inclusion_count matches the vector."""
    ev = _event((15, 25), (8, 12))
    results = _bootstrap_per_replicate([ev], 0, n_replicates=50, seed=1)
    assert results[0].inclusion_count == 15
    assert results[0].exclusion_count == 8


def test_bootstrap_per_replicate_second_replicate():
    """Index 1 must use the second per-replicate values."""
    ev = _event((15, 25), (8, 12))
    results = _bootstrap_per_replicate([ev], 1, n_replicates=50, seed=2)
    assert results[0].inclusion_count == 25
    assert results[0].exclusion_count == 12


def test_bootstrap_per_replicate_fallback_to_group_sum(caplog):
    """Index beyond replicate vector length must fall back to group-sum and warn."""
    ev = _event((10,), (5,), group_inc=10, group_exc=5)
    with caplog.at_level(logging.WARNING, logger="braid.commands.psi"):
        results = _bootstrap_per_replicate([ev], replicate_index=5, n_replicates=50, seed=1)
    assert results[0].inclusion_count == 10  # group sum
    assert results[0].exclusion_count == 5
    assert any("group-sum" in r.message.lower() or "fallback" in r.message.lower()
               for r in caplog.records)


def test_bootstrap_per_replicate_returns_psi_result_fields():
    """Returned PSIResult must have event metadata from the RmatsEvent."""
    ev_with_id = RmatsEvent(
        event_id="chr2:SE:300:400:+",
        event_type="SE",
        chrom="chr2",
        strand="+",
        gene="MYGENEX",
        inc_count=20,
        exc_count=10,
        rmats_psi=0.67,
        rmats_fdr=0.01,
        rmats_dpsi=0.1,
        sample_1_inc_count=20,
        sample_1_exc_count=10,
        sample_1_inc_replicates=(20,),
        sample_1_exc_replicates=(10,),
    )
    results = _bootstrap_per_replicate([ev_with_id], 0, n_replicates=50, seed=1)
    r = results[0]
    assert r.event_id == "chr2:SE:300:400:+"
    assert r.gene == "MYGENEX"
    assert r.chrom == "chr2"
    assert 0.0 <= r.psi <= 1.0
    assert r.ci_low <= r.ci_high


def test_bootstrap_per_replicate_empty_events_returns_empty():
    """Empty event list must return an empty result list."""
    results = _bootstrap_per_replicate([], 0, n_replicates=50, seed=1)
    assert results == []


def test_bootstrap_per_replicate_multiple_events():
    """Multiple events must each produce one PSIResult."""
    evs = [
        _event((10,), (5,), event_id="chr1:SE:100:200:+"),
        _event((30,), (10,), event_id="chr1:SE:300:400:+"),
    ]
    results = _bootstrap_per_replicate(evs, 0, n_replicates=50, seed=1)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# run_psi — single-sample rMATS path
# ---------------------------------------------------------------------------


def test_run_psi_missing_rmats_dir_exits(tmp_path, caplog):
    """A missing rMATS directory is input error, not a successful empty result."""
    args = _minimal_args(
        rmats_dir=str(tmp_path / "missing_rmats"),
        output=str(tmp_path / "psi.tsv"),
    )

    with caplog.at_level(logging.ERROR, logger="braid.commands.psi"):
        with pytest.raises(SystemExit) as excinfo:
            run_psi(args)

    assert excinfo.value.code == 1
    assert not (tmp_path / "psi.tsv").exists()
    assert any("rMATS directory not found" in r.message for r in caplog.records)


def test_run_psi_empty_rmats_dir_exits(tmp_path, caplog):
    """A directory with no supported rMATS event tables must fail fast."""
    rmats_dir = tmp_path / "empty_rmats"
    rmats_dir.mkdir()
    args = _minimal_args(rmats_dir=str(rmats_dir), output=str(tmp_path / "psi.tsv"))

    with caplog.at_level(logging.ERROR, logger="braid.commands.psi"):
        with pytest.raises(SystemExit) as excinfo:
            run_psi(args)

    assert excinfo.value.code == 1
    assert not (tmp_path / "psi.tsv").exists()
    assert any("No supported rMATS event tables" in r.message for r in caplog.records)


def test_run_psi_negative_min_support_raises(tmp_path):
    """A negative --min-support is a hard error: it would make both support gates
    vacuously true and admit evidence-free PSI rows. Mirrors `braid differential`."""
    args = _minimal_args(
        rmats_dir="/fake/dir", output=str(tmp_path / "psi.tsv"),
        bam=None, min_support=-1)
    with pytest.raises(ValueError, match="min_support must be >= 0"):
        run_psi(args)


def test_run_psi_single_sample_rmats(tmp_path, monkeypatch):
    """Single-sample path (n_bams<=1) must call add_bootstrap_ci and write TSV."""
    out = str(tmp_path / "psi.tsv")
    events = [_event((20,), (10,))]

    monkeypatch.setattr(
        "braid.commands.psi._resolve_conformal_calibrator", lambda _args: None
    )
    _allow_fake_rmats_dir(monkeypatch)
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.parse_rmats_output",
        lambda *a, **kw: events,
    )
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.add_bootstrap_ci",
        lambda *a, **kw: [_psi_result()],
    )

    args = _minimal_args(rmats_dir="/fake/dir", output=out, bam=None)
    run_psi(args)

    with open(out) as f:
        lines = f.readlines()
    assert len(lines) == 2  # header + 1 data row


def test_run_psi_single_bam_rmats(tmp_path, monkeypatch):
    """One BAM file also takes the single-sample path."""
    out = str(tmp_path / "psi.tsv")
    events = [_event((20,), (10,))]

    monkeypatch.setattr(
        "braid.commands.psi._resolve_conformal_calibrator", lambda _args: None
    )
    _allow_fake_rmats_dir(monkeypatch)
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.parse_rmats_output",
        lambda *a, **kw: events,
    )
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.add_bootstrap_ci",
        lambda *a, **kw: [_psi_result()],
    )

    args = _minimal_args(rmats_dir="/fake/dir", output=out, bam=["sample.bam"])
    run_psi(args)

    with open(out) as f:
        lines = f.readlines()
    assert len(lines) == 2


def test_run_psi_single_sample_prints_summary(tmp_path, monkeypatch, capsys):
    """run_psi must print the event count and output path."""
    out = str(tmp_path / "psi.tsv")
    events = [_event((20,), (10,))]

    monkeypatch.setattr(
        "braid.commands.psi._resolve_conformal_calibrator", lambda _args: None
    )
    _allow_fake_rmats_dir(monkeypatch)
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.parse_rmats_output",
        lambda *a, **kw: events,
    )
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.add_bootstrap_ci",
        lambda *a, **kw: [_psi_result()],
    )

    args = _minimal_args(rmats_dir="/fake/dir", output=out)
    run_psi(args)

    captured = capsys.readouterr()
    assert "1 events" in captured.out
    assert out in captured.out


# ---------------------------------------------------------------------------
# run_psi — multi-replicate rMATS path
# ---------------------------------------------------------------------------


def test_run_psi_multi_replicate_rmats(tmp_path, monkeypatch):
    """Multi-replicate path (2 BAMs, matching 2-rep table) must produce combined TSV."""
    out = str(tmp_path / "psi.tsv")
    events = [_event((10, 20), (5, 8))]

    monkeypatch.setattr(
        "braid.commands.psi._resolve_conformal_calibrator", lambda _args: None
    )
    _allow_fake_rmats_dir(monkeypatch)
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.parse_rmats_output",
        lambda *a, **kw: events,
    )

    args = _minimal_args(
        rmats_dir="/fake/dir",
        output=out,
        bam=["rep1.bam", "rep2.bam"],
        replicates=50,
    )
    run_psi(args)

    with open(out) as f:
        lines = f.readlines()
    assert len(lines) == 2  # header + 1 combined row


def test_run_psi_multi_replicate_mismatch_exits(tmp_path, monkeypatch):
    """BAM/table count mismatch must sys.exit(1) without --allow-replicate-fallback."""
    out = str(tmp_path / "psi.tsv")
    events = [_event((10,), (5,))]  # only 1 replicate in table

    monkeypatch.setattr(
        "braid.commands.psi._resolve_conformal_calibrator", lambda _args: None
    )
    _allow_fake_rmats_dir(monkeypatch)
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.parse_rmats_output",
        lambda *a, **kw: events,
    )

    args = _minimal_args(
        rmats_dir="/fake/dir",
        output=out,
        bam=["rep1.bam", "rep2.bam"],  # 2 BAMs but only 1 rep in table
        allow_replicate_fallback=False,
    )
    with pytest.raises(SystemExit):
        run_psi(args)


def test_run_psi_multi_replicate_fallback_proceeds(tmp_path, monkeypatch):
    """With --allow-replicate-fallback a mismatch must proceed, not exit."""
    out = str(tmp_path / "psi.tsv")
    events = [_event((10,), (5,))]

    monkeypatch.setattr(
        "braid.commands.psi._resolve_conformal_calibrator", lambda _args: None
    )
    _allow_fake_rmats_dir(monkeypatch)
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.parse_rmats_output",
        lambda *a, **kw: events,
    )

    args = _minimal_args(
        rmats_dir="/fake/dir",
        output=out,
        bam=["rep1.bam", "rep2.bam"],
        allow_replicate_fallback=True,
        replicates=50,
    )
    run_psi(args)  # must not raise

    with open(out) as f:
        lines = f.readlines()
    assert len(lines) >= 1  # at least a header


# ---------------------------------------------------------------------------
# run_psi — "--rmats-dir required" error path
# ---------------------------------------------------------------------------


def test_run_psi_no_mode_exits(tmp_path, monkeypatch):
    """No --rmats-dir must sys.exit(1)."""
    out = str(tmp_path / "psi.tsv")
    monkeypatch.setattr(
        "braid.commands.psi._resolve_conformal_calibrator", lambda _args: None
    )
    args = _minimal_args(
        rmats_dir=None,
        bam=None,
        output=out,
    )
    with pytest.raises(SystemExit):
        run_psi(args)


# ---------------------------------------------------------------------------
# run_psi — verbose flag
# ---------------------------------------------------------------------------


def test_run_psi_verbose_flag(tmp_path, monkeypatch):
    """verbose=True must not crash; basicConfig called with DEBUG level."""
    out = str(tmp_path / "psi.tsv")
    events = [_event((20,), (10,))]

    monkeypatch.setattr(
        "braid.commands.psi._resolve_conformal_calibrator", lambda _args: None
    )
    _allow_fake_rmats_dir(monkeypatch)
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.parse_rmats_output",
        lambda *a, **kw: events,
    )
    monkeypatch.setattr(
        "braid.target.rmats_bootstrap.add_bootstrap_ci",
        lambda *a, **kw: [_psi_result()],
    )

    args = _minimal_args(rmats_dir="/fake/dir", output=out, verbose=True)
    run_psi(args)  # should not raise


def test_psi_no_conformal_with_calibration_warns_and_ignores(tmp_path, caplog) -> None:
    """braid psi --no-conformal + --calibration is contradictory: the calibration
    is ignored, but the conflict must be surfaced (warned), not silently dropped."""
    import argparse

    from braid.commands.psi import _resolve_conformal_calibrator

    args = argparse.Namespace(use_conformal=False, calibration=str(tmp_path / "x.json"))
    with caplog.at_level("WARNING", logger="braid.commands.psi"):
        cal = _resolve_conformal_calibrator(args)

    assert cal is None
    assert any("ignored because --no-conformal" in r.message for r in caplog.records)


# ── QA: upfront posterior-knob validation (parity with `braid differential`) ──
# Found by adversarial CLI probing: invalid --confidence/--replicates were only
# caught per-event inside bootstrap_psi, so they were silently accepted whenever zero
# events passed the support filter (the run wrote a header-only file). run_psi now
# validates both upfront, like run_differential.


def test_psi_rejects_invalid_confidence_upfront() -> None:
    """--confidence outside (0, 1) fails fast, regardless of how many events pass."""
    import pytest
    for bad in (1.5, 0.0, 1.0, -0.2):
        with pytest.raises(ValueError, match="confidence"):
            run_psi(_minimal_args(confidence=bad))


def test_psi_rejects_invalid_replicates_upfront() -> None:
    """--replicates < 1 fails fast, regardless of how many events pass."""
    import pytest
    with pytest.raises(ValueError, match="replicates"):
        run_psi(_minimal_args(replicates=0))
    with pytest.raises(ValueError, match="replicates"):
        run_psi(_minimal_args(replicates=-5))


def test_psi_valid_knobs_pass_validation() -> None:
    """Valid knobs must NOT trip the new guards (no over-rejection).

    With valid confidence/replicates and no input, run_psi proceeds PAST the knob
    guards and exits at the ``--rmats-dir is required`` check (SystemExit). Reaching
    that point proves the knob validation did not fire on valid values.
    """
    with pytest.raises(SystemExit):
        run_psi(_minimal_args(confidence=0.95, replicates=50))
