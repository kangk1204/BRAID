"""Tests for the differential (two-group ΔPSI) command.

Locks two fixes: (1) ΔPSI uses concentrated two-group Jeffreys-Beta posteriors so
genuine large effects exclude zero (the overdispersed posterior made them
pathologically wide); (2) dpsi_mean carries the SAME sign as rMATS
IncLevelDifference (it was negated before).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest

from braid.commands.differential import run_differential

_HEADER = [
    "ID", "GeneID", "geneSymbol", "chr", "strand", "exonStart_0base", "exonEnd",
    "upstreamES", "upstreamEE", "downstreamES", "downstreamEE",
    "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
]


def _row(eid, gene, es, inc1, sjc1, inc2, sjc2, fdr, dpsi):
    psi1 = inc1 / (inc1 + sjc1)
    psi2 = inc2 / (inc2 + sjc2)
    return [
        str(eid), gene, gene, "chr1", "+", str(es), str(es + 100),
        str(es - 200), str(es - 100), str(es + 200), str(es + 300),
        str(inc1), str(sjc1), str(inc2), str(sjc2),
        "100", "100", "0.0", str(fdr),
        f"{psi1:.3f}", f"{psi2:.3f}", str(dpsi),
    ]


def _replicate_row(eid, gene, es, inc1, sjc1, inc2, sjc2, fdr):
    def _psi_parts(inc, exc):
        return [i / (i + j) if (i + j) > 0 else 0.0 for i, j in zip(inc, exc)]

    psi1 = _psi_parts(inc1, sjc1)
    psi2 = _psi_parts(inc2, sjc2)
    dpsi = sum(psi1) / len(psi1) - sum(psi2) / len(psi2)
    return [
        str(eid), gene, gene, "chr1", "+", str(es), str(es + 100),
        str(es - 200), str(es - 100), str(es + 200), str(es + 300),
        ",".join(str(x) for x in inc1),
        ",".join(str(x) for x in sjc1),
        ",".join(str(x) for x in inc2),
        ",".join(str(x) for x in sjc2),
        "100", "100", "0.0", str(fdr),
        ",".join(f"{x:.3f}" for x in psi1),
        ",".join(f"{x:.3f}" for x in psi2),
        f"{dpsi:.6f}",
    ]


def _write_se_table(rmats_dir: Path) -> None:
    rmats_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        # large positive effect: ctrl PSI 0.9, treat 0.3 -> rMATS dPSI +0.6, significant
        _row(1, "BIG", 1000, 180, 20, 60, 140, 0.001, 0.6),
        # null: both 0.5, not significant
        _row(2, "NULL", 5000, 100, 100, 100, 100, 0.8, 0.0),
    ]
    lines = ["\t".join(_HEADER)] + ["\t".join(r) for r in rows]
    (rmats_dir / "SE.MATS.JC.txt").write_text("\n".join(lines) + "\n")


def _args(
    rmats_dir: Path,
    out: Path,
    *,
    use_conformal: bool = True,
    calibration: str | None = None,
    confidence: float = 0.95,
    fdr: float = 0.05,
    effect_cutoff: float = 0.1,
    min_support: int = 20,
    seed: int = 42,
    replicates: int = 2000,
    differential_model: str = "auto",
) -> argparse.Namespace:
    return argparse.Namespace(
        rmats_dir=str(rmats_dir), output=str(out), replicates=replicates, confidence=confidence,
        fdr=fdr, effect_cutoff=effect_cutoff, min_support=min_support, seed=seed,
        verbose=False,
        use_conformal=use_conformal, calibration=calibration,
        differential_model=differential_model,
    )


def _read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_differential_large_effect_excludes_zero_and_sign_matches_rmats(tmp_path: Path) -> None:
    rmats_dir = tmp_path / "rmats"
    out = tmp_path / "diff.tsv"
    _write_se_table(rmats_dir)
    run_differential(_args(rmats_dir, out))
    rows = {r["gene"]: r for r in _read_tsv(out)}

    big = rows["BIG"]
    rmats_dpsi = float(big["rmats_dpsi"])
    dpsi_mean = float(big["dpsi"])
    # (2) sign convention matches rMATS
    assert (dpsi_mean > 0) == (rmats_dpsi > 0)
    assert abs(dpsi_mean - rmats_dpsi) < 0.1
    # (1) the genuine large effect now excludes zero and is high-confidence
    assert big["reliable"] == "yes"
    assert big["tier"] == "high-confidence"


def test_differential_null_event_does_not_exclude_zero(tmp_path: Path) -> None:
    rmats_dir = tmp_path / "rmats"
    out = tmp_path / "diff.tsv"
    _write_se_table(rmats_dir)
    run_differential(_args(rmats_dir, out))
    rows = {r["gene"]: r for r in _read_tsv(out)}

    null = rows["NULL"]
    assert null["reliable"] == "no"
    assert null["tier"] == "not-significant"


def test_differential_applies_event_type_conformal_quantile(tmp_path: Path) -> None:
    """braid differential must forward event_type into the conformal q_for cascade.

    Locks the differential<->conformal integration: a calibrator with a large SE
    quantile must widen SE events, while one keyed on a different type (A3SS) falls
    through to the tiny global quantile. If event_type were dropped (the pre-fix bug)
    both calibrators would produce identical widths.
    """
    from braid.target.conformal import ConformalCalibrator

    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)  # all events are SE type

    def _cal(et_map: dict) -> str:
        cal = ConformalCalibrator(
            alpha=0.05, q_global=0.01, q_by_bin={},
            scale_kind="absolute_dpsi", q_by_event_type=et_map,
        )
        key = next(iter(et_map))
        path = tmp_path / f"cal_{key}.json"
        cal.to_json(path)
        return str(path)

    out_se = tmp_path / "se.tsv"
    out_other = tmp_path / "other.tsv"
    run_differential(_args(rmats_dir, out_se, calibration=_cal({"SE": 0.5})))
    run_differential(_args(rmats_dir, out_other, calibration=_cal({"A3SS": 0.5})))

    def _width(path: Path) -> float:
        big = {r["gene"]: r for r in _read_tsv(path)}["BIG"]
        return float(big["ci_high"]) - float(big["ci_low"])

    w_se, w_other = _width(out_se), _width(out_other)
    assert w_se > 0.6              # SE event received the SE quantile (0.5) -> wide
    assert w_se > 4 * w_other      # A3SS-keyed calibrator fell through to q_global -> narrow


def test_differential_fdr_gate_is_independent_from_ci_confidence(tmp_path: Path) -> None:
    """rMATS FDR tiering and ΔPSI CI confidence are separate contracts."""
    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir(parents=True, exist_ok=True)
    row = _row(1, "ALPHA", 1000, 180, 20, 60, 140, 0.08, 0.6)
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join(_HEADER) + "\n" + "\t".join(row) + "\n",
        encoding="utf-8",
    )
    low_conf = tmp_path / "low_conf.tsv"
    high_conf = tmp_path / "high_conf.tsv"
    loose_fdr = tmp_path / "loose_fdr.tsv"

    run_differential(_args(rmats_dir, low_conf, use_conformal=False, confidence=0.90))
    run_differential(_args(rmats_dir, high_conf, use_conformal=False, confidence=0.95))
    run_differential(_args(rmats_dir, loose_fdr, use_conformal=False, fdr=0.10))

    # ALPHA has a large, reliable effect (ΔPSI~0.6, interval excludes 0) but rMATS
    # FDR 0.08 > 0.05, so it is BRAID-reliable without the caller's flag: "supported".
    # Changing the CI confidence (0.90 vs 0.95) does not change the tier; only the FDR
    # threshold (loose_fdr) promotes it to high-confidence -- the two axes stay
    # independent under the unified confidence_tier vocabulary.
    assert _read_tsv(low_conf)[0]["tier"] == "supported"
    assert _read_tsv(high_conf)[0]["tier"] == "supported"
    assert _read_tsv(loose_fdr)[0]["tier"] == "high-confidence"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("replicates", 0, "replicates must be >= 1"),
        ("confidence", 1.0, "confidence must be in"),
        ("fdr", -0.1, "fdr must be in"),
        ("min_support", -1, "min_support must be >= 0"),
        ("effect_cutoff", -0.1, "effect-cutoff must be >= 0"),
    ],
)
def test_differential_rejects_invalid_numeric_contracts(
    tmp_path: Path,
    field: str,
    value: float | int,
    message: str,
) -> None:
    """Invalid CLI numeric contracts must fail before any rMATS parsing work."""
    args = _args(tmp_path, tmp_path / "out.tsv")
    setattr(args, field, value)

    with pytest.raises(ValueError, match=message):
        run_differential(args)


def test_differential_missing_rmats_dir_exits(tmp_path: Path, caplog) -> None:
    """A typo in --rmats-dir must not produce a successful header-only TSV."""
    args = _args(tmp_path / "missing_rmats", tmp_path / "diff.tsv")

    with caplog.at_level("ERROR", logger="braid.commands.differential"):
        with pytest.raises(SystemExit) as excinfo:
            run_differential(args)

    assert excinfo.value.code == 1
    assert not (tmp_path / "diff.tsv").exists()
    assert any("rMATS directory not found" in r.message for r in caplog.records)


def test_differential_empty_rmats_dir_exits(tmp_path: Path, caplog) -> None:
    """An rMATS directory must contain at least one supported event table."""
    rmats_dir = tmp_path / "empty_rmats"
    rmats_dir.mkdir()
    args = _args(rmats_dir, tmp_path / "diff.tsv")

    with caplog.at_level("ERROR", logger="braid.commands.differential"):
        with pytest.raises(SystemExit) as excinfo:
            run_differential(args)

    assert excinfo.value.code == 1
    assert not (tmp_path / "diff.tsv").exists()
    assert any("No supported rMATS event tables" in r.message for r in caplog.records)


def test_differential_conformal_default_widens_ci_to_calibrated_halfwidth(
    tmp_path: Path,
) -> None:
    """The shipped default applies the real-data conformal calibrator: the ΔPSI CI
    half-width is the calibrator quantile (~0.34), much wider than the raw posterior
    percentile, so the stated 95% interval is honestly calibrated."""
    rmats_dir = tmp_path / "rmats"
    out_conf = tmp_path / "conf.tsv"
    out_raw = tmp_path / "raw.tsv"
    _write_se_table(rmats_dir)

    run_differential(_args(rmats_dir, out_conf, use_conformal=True))
    run_differential(_args(rmats_dir, out_raw, use_conformal=False))
    conf = {r["gene"]: r for r in _read_tsv(out_conf)}["BIG"]
    raw = {r["gene"]: r for r in _read_tsv(out_raw)}["BIG"]

    conf_w = float(conf["ci_high"]) - float(conf["ci_low"])
    raw_w = float(raw["ci_high"]) - float(raw["ci_low"])
    # conformal half-width ~0.34 -> width ~0.66; raw posterior is far tighter
    assert 0.5 < conf_w < 0.85
    assert raw_w < 0.25
    assert conf_w > 3 * raw_w
    # the genuine large effect (|ΔPSI|~0.6) still excludes zero under the wide CI
    assert conf["reliable"] == "yes"


def test_differential_length_normalization_shifts_psi(tmp_path: Path) -> None:
    """With IncFormLen != SkipFormLen, PSI is length-normalized to the rMATS IncLevel
    scale (raw inc/(inc+exc) would over-state inclusion)."""
    rmats_dir = tmp_path / "rmats"
    out = tmp_path / "diff.tsv"
    rmats_dir.mkdir(parents=True, exist_ok=True)
    # inc=exc=60 raw -> raw PSI 0.5; IncFormLen=200, SkipFormLen=100 -> normalized
    # PSI = (60/200)/((60/200)+(60/100)) = 0.333
    row = _row(1, "LN", 1000, 60, 60, 60, 60, 0.5, 0.0)
    row[15] = "200"  # IncFormLen
    row[16] = "100"  # SkipFormLen
    header = "\t".join(_HEADER)
    (rmats_dir / "SE.MATS.JC.txt").write_text(header + "\n" + "\t".join(row) + "\n")

    run_differential(_args(rmats_dir, out, use_conformal=False))
    r = {x["gene"]: x for x in _read_tsv(out)}["LN"]
    assert abs(float(r["ctrl_psi"]) - 1.0 / 3.0) < 0.02
    assert abs(float(r["treat_psi"]) - 1.0 / 3.0) < 0.02


def test_differential_replicate_aware_uses_rmats_sample_vectors(tmp_path: Path) -> None:
    """Multiple rMATS samples must affect the ΔPSI uncertainty, not only the sums."""
    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir(parents=True, exist_ok=True)
    row = _replicate_row(
        1, "HET", 1000,
        (95, 50, 95), (5, 50, 5),
        (5, 50, 5), (95, 50, 95),
        0.001,
    )
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join(_HEADER) + "\n" + "\t".join(row) + "\n",
        encoding="utf-8",
    )

    aware_out = tmp_path / "aware.tsv"
    pooled_out = tmp_path / "pooled.tsv"
    run_differential(
        _args(rmats_dir, aware_out, use_conformal=False, replicates=6000, seed=9)
    )
    run_differential(
        _args(
            rmats_dir, pooled_out, use_conformal=False, replicates=6000,
            seed=9, differential_model="sum",
        )
    )
    aware = _read_tsv(aware_out)[0]
    pooled = _read_tsv(pooled_out)[0]

    aware_width = float(aware["ci_high"]) - float(aware["ci_low"])
    pooled_width = float(pooled["ci_high"]) - float(pooled["ci_low"])
    assert aware["differential_model"] == "rep"
    assert aware["ctrl_replicates"] == "3"
    assert aware["treat_replicates"] == "3"
    assert pooled["differential_model"] == "sum"
    assert aware_width > 3 * pooled_width


def test_differential_model_rep_requires_replicate_vectors(tmp_path: Path) -> None:
    """Explicit rep mode must not silently fall back to pooled sums."""
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)

    with pytest.raises(ValueError, match="differential_model=rep requires"):
        run_differential(
            _args(
                rmats_dir, tmp_path / "diff.tsv",
                use_conformal=False, differential_model="rep",
            )
        )


def test_differential_model_auto_is_event_local_for_mixed_vectors(
    tmp_path: Path,
) -> None:
    """Auto mode must not leak one event's selected model into the next event."""
    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir(parents=True, exist_ok=True)
    replicate_row = _replicate_row(
        1, "HET", 1000,
        (95, 50, 95), (5, 50, 5),
        (5, 50, 5), (95, 50, 95),
        0.001,
    )
    scalar_row = _row(2, "SCALAR", 5000, 180, 20, 60, 140, 0.001, 0.6)
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join(_HEADER) + "\n"
        + "\t".join(replicate_row) + "\n"
        + "\t".join(scalar_row) + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "mixed.tsv"
    run_differential(_args(rmats_dir, out, use_conformal=False))
    rows = {r["gene"]: r for r in _read_tsv(out)}

    assert rows["HET"]["differential_model"] == "rep"
    assert rows["SCALAR"]["differential_model"] == "sum"


def test_differential_dpsi_equals_ctrl_minus_treat_psi_both_models(tmp_path: Path) -> None:
    """Regression: dpsi and the reported ctrl_psi/treat_psi must share one estimand.

    With depth-imbalanced replicates the pooled read-weighted PSI difference (~+0.67)
    and the equal-weight rep ΔPSI (~0) diverge in magnitude (and can flip sign), so
    the PSI columns must be the posterior means of whichever model produced dpsi --
    otherwise downstream `dpsi = ctrl_psi - treat_psi` (DM1/Esrp apps) breaks.
    """
    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir(parents=True, exist_ok=True)
    # ctrl rep1 high-depth high-PSI + rep2 low-depth low-PSI (mirror for treat):
    # pooled sees a large effect, equal-weight rep sees almost none.
    row = _replicate_row(1, "IMB", 1000, (200, 2), (20, 20), (20, 20), (200, 2), 0.001)
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join(_HEADER) + "\n" + "\t".join(row) + "\n", encoding="utf-8")

    for model in ("sum", "rep"):
        out = tmp_path / f"{model}.tsv"
        run_differential(
            _args(rmats_dir, out, use_conformal=False, differential_model=model))
        r = _read_tsv(out)[0]
        dpsi, cp, tp = float(r["dpsi"]), float(r["ctrl_psi"]), float(r["treat_psi"])
        assert abs(dpsi - (cp - tp)) < 1e-4, f"{model}: dpsi {dpsi} != ctrl-treat {cp - tp}"

    # the two estimands genuinely differ here (else the regression would be vacuous)
    assert abs(float(_read_tsv(tmp_path / "sum.tsv")[0]["dpsi"])) > 0.4
    assert abs(float(_read_tsv(tmp_path / "rep.tsv")[0]["dpsi"])) < 0.2


def test_differential_replicate_aware_falls_back_for_single_replicate(
    tmp_path: Path,
) -> None:
    """Scalar rMATS count columns keep the legacy pooled-count posterior."""
    rmats_dir = tmp_path / "rmats"
    out = tmp_path / "diff.tsv"
    _write_se_table(rmats_dir)

    run_differential(_args(rmats_dir, out, use_conformal=False))
    big = {r["gene"]: r for r in _read_tsv(out)}["BIG"]

    assert big["differential_model"] == "sum"
    assert big["ctrl_replicates"] == "1"
    assert big["treat_replicates"] == "1"


# ---------------------------------------------------------------------------
# Parameter-application regression tests
# ---------------------------------------------------------------------------
# Each production knob must demonstrably change the output, so a future
# "silently-ignored parameter" regression (a knob accepted but never wired into
# the computation) fails loudly here rather than shipping a misleading result.


def test_min_support_param_gates_events(tmp_path: Path) -> None:
    """--min-support must filter: both fixture events have 200 reads per group, so
    a threshold above that drops them all."""
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)

    out_low = tmp_path / "low.tsv"
    run_differential(_args(rmats_dir, out_low, min_support=20))
    assert len(_read_tsv(out_low)) == 2

    out_high = tmp_path / "high.tsv"
    run_differential(_args(rmats_dir, out_high, min_support=300))
    assert len(_read_tsv(out_high)) == 0


def test_effect_cutoff_param_changes_tier(tmp_path: Path) -> None:
    """--effect-cutoff must gate the supported tier: the BIG event (|ΔPSI|~0.6) is
    high-confidence at cutoff 0.1 but drops to caller-significant-only at 0.95 (no
    BRAID effect, the caller still flagged it)."""
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)

    out_lo = tmp_path / "lo.tsv"
    run_differential(_args(rmats_dir, out_lo, effect_cutoff=0.1))
    big_lo = {r["gene"]: r for r in _read_tsv(out_lo)}["BIG"]

    out_hi = tmp_path / "hi.tsv"
    run_differential(_args(rmats_dir, out_hi, effect_cutoff=0.95))
    big_hi = {r["gene"]: r for r in _read_tsv(out_hi)}["BIG"]

    assert big_lo["tier"] in {"supported", "high-confidence"}
    assert big_hi["tier"] == "caller-significant-only"


def test_fdr_param_changes_significance(tmp_path: Path) -> None:
    """--fdr must gate rMATS significance: the NULL event (FDR 0.8, ΔPSI~0) is
    not-significant at fdr 0.05 but caller-significant-only at fdr 0.9 (the caller
    flag flips on, but the interval still crosses 0)."""
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)

    out_strict = tmp_path / "strict.tsv"
    run_differential(_args(rmats_dir, out_strict, fdr=0.05))
    assert {r["gene"]: r for r in _read_tsv(out_strict)}["NULL"]["tier"] == "not-significant"

    out_loose = tmp_path / "loose.tsv"
    run_differential(_args(rmats_dir, out_loose, fdr=0.9))
    assert {r["gene"]: r for r in _read_tsv(out_loose)}["NULL"]["tier"] == "caller-significant-only"


def test_no_conformal_param_changes_interval(tmp_path: Path) -> None:
    """--no-conformal must switch the interval estimator: the BIG event's interval
    differs between the conformal calibrator and the percentile fallback."""
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)

    out_conf = tmp_path / "conf.tsv"
    run_differential(_args(rmats_dir, out_conf, use_conformal=True))
    big_conf = {r["gene"]: r for r in _read_tsv(out_conf)}["BIG"]

    out_raw = tmp_path / "raw.tsv"
    run_differential(_args(rmats_dir, out_raw, use_conformal=False))
    big_raw = {r["gene"]: r for r in _read_tsv(out_raw)}["BIG"]

    assert (float(big_conf["ci_low"]), float(big_conf["ci_high"])) != (
        float(big_raw["ci_low"]), float(big_raw["ci_high"])
    )


def test_seed_param_is_deterministic(tmp_path: Path) -> None:
    """The same --seed must reproduce the bootstrap exactly (identical estimate)."""
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)

    out_a = tmp_path / "a.tsv"
    run_differential(_args(rmats_dir, out_a, seed=7))
    out_b = tmp_path / "b.tsv"
    run_differential(_args(rmats_dir, out_b, seed=7))

    a = {r["gene"]: r for r in _read_tsv(out_a)}["BIG"]
    b = {r["gene"]: r for r in _read_tsv(out_b)}["BIG"]
    assert a["dpsi"] == b["dpsi"]


def test_assemble_max_paths_default_matches_config() -> None:
    """The assemble CLI --max-paths default must equal PipelineConfig and
    DecomposeConfig (2000), so a CLI run and a programmatic PipelineConfig
    enumerate the same number of source-to-sink paths. The CLI default was 500
    while every other site used 2000, silently under-enumerating CLI assemblies."""
    import dataclasses

    from braid.cli import _add_assembly_args
    from braid.flow.decompose import DecomposeConfig
    from braid.pipeline import PipelineConfig

    p = argparse.ArgumentParser()
    _add_assembly_args(p)
    arg = {a.dest: a.default for a in p._actions if a.dest == "max_paths"}["max_paths"]
    pc = {f.name: f.default for f in dataclasses.fields(PipelineConfig)}["max_paths"]
    dc = {f.name: f.default for f in dataclasses.fields(DecomposeConfig)}["max_paths"]
    assert arg == pc == dc == 2000


def test_no_conformal_with_calibration_warns_and_ignores(tmp_path: Path, caplog) -> None:
    """--no-conformal + --calibration is contradictory: the calibration is ignored,
    but the conflict must be surfaced (warned), not silently dropped."""
    from braid.commands.differential import _resolve_differential_calibrator

    args = argparse.Namespace(use_conformal=False, calibration=str(tmp_path / "x.json"))
    with caplog.at_level("WARNING", logger="braid.commands.differential"):
        cal = _resolve_differential_calibrator(args)

    assert cal is None
    assert any("ignored because --no-conformal" in r.message for r in caplog.records)


def test_differential_no_conformal_log_omits_calibrator_transfer_note(
    tmp_path: Path, caplog,
) -> None:
    """Under --no-conformal the model log must not claim a conformal calibrator is
    used (calibrator is None -> raw Jeffreys percentile), so the state is not
    misstated."""
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)
    with caplog.at_level("INFO", logger="braid.commands.differential"):
        run_differential(_args(rmats_dir, tmp_path / "o.tsv", use_conformal=False))
    log = "\n".join(r.message for r in caplog.records)
    assert "calibration OFF" in log
    assert "pooled-fit conformal" not in log
    assert "calibrator by transfer" not in log

    caplog.clear()
    with caplog.at_level("INFO", logger="braid.commands.differential"):
        run_differential(_args(rmats_dir, tmp_path / "conf.tsv", use_conformal=True))
    log = "\n".join(r.message for r in caplog.records)
    assert "calibration ON" in log
    assert "pooled-fit conformal" in log
    assert "calibrator by transfer" in log


def test_cli_input_value_error_surfaces_cleanly(tmp_path: Path, monkeypatch, capsys) -> None:
    """A CLI input/validation error must fail short and clear (exit 2, 'Error: ...'),
    not as an 'Unexpected error' traceback."""
    from braid import cli

    monkeypatch.setattr(
        "sys.argv",
        ["braid", "differential", "--rmats-dir", str(tmp_path / "nodir"),
         "--effect-cutoff", "-0.1"],
    )
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "Error: effect-cutoff must be >= 0" in err
    assert "Traceback" not in err and "Unexpected error" not in err
