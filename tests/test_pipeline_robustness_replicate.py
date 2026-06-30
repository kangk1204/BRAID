"""Pipeline-robustness suite for the replicate-aware differential model stage.

Complements ``test_pipeline_robustness_differential.py`` (the pooled differential
chain) by hardening the per-replicate machinery added with --differential-model
{auto,rep,sum}: the rMATS per-replicate count-vector helpers and the model resolver.

Each test maps to one (stage, failure-mode) cell of the matrix and asserts something
non-trivial (return schema or an error ``match=``), never just "did not crash".

Failure-mode IDs: A2 empty/degenerate · A3 single-row · A8 length mismatch ·
A10 inf/NaN · F1 contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from braid.commands.differential import (
    _complete_replicate_arrays,
    _draw_replicate_mean_psi,
    _resolve_differential_model,
    add_differential_subparser,
    run_differential,
)

_HEADER = [
    "ID", "GeneID", "geneSymbol", "chr", "strand", "exonStart_0base", "exonEnd",
    "upstreamES", "upstreamEE", "downstreamES", "downstreamEE",
    "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
]


def _rep_row(eid, gene, es, ijc1, sjc1, ijc2, sjc2, fdr="0.001", dpsi="0.6"):
    """A row whose count cells are comma-joined per-replicate vectors (strings)."""
    return [
        str(eid), gene, gene, "chr1", "+", str(es), str(es + 100),
        str(es - 200), str(es - 100), str(es + 200), str(es + 300),
        ijc1, sjc1, ijc2, sjc2, "100", "100", "0.0", fdr, "0.9", "0.3", dpsi,
    ]


def _write(rmats_dir: Path, rows) -> None:
    rmats_dir.mkdir(parents=True, exist_ok=True)
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join(_HEADER) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n"
    )


def _args(rmats_dir: Path, out: Path, **kw) -> argparse.Namespace:
    base = dict(
        rmats_dir=str(rmats_dir), output=str(out), replicates=400, confidence=0.95,
        fdr=0.05, effect_cutoff=0.1, min_support=20, seed=42, verbose=False,
        ctrl=None, treat=None, strict=False, use_conformal=False,
        calibration=None, differential_model="auto",
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _read(path: Path) -> list[dict]:
    import csv
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# ==============================================================================
# complete-array guard
# ==============================================================================
def test_complete_arrays_rejects_inc_exc_length_mismatch() -> None:
    """A8: inc/exc replicate vectors of different lengths cannot be paired -> None
    (the model must fall back / error, never index a desynced pair)."""
    ev = SimpleNamespace(
        sample_1_inc_replicates=(10, 20, 30),
        sample_1_exc_replicates=(5, 10),  # length 2 != 3
    )
    assert _complete_replicate_arrays(ev, "sample_1") is None


def test_complete_arrays_rejects_single_replicate() -> None:
    """A3: a single replicate carries no between-replicate variance -> None."""
    ev = SimpleNamespace(
        sample_1_inc_replicates=(10,), sample_1_exc_replicates=(5,),
    )
    assert _complete_replicate_arrays(ev, "sample_1") is None


def test_complete_arrays_accepts_matched_pairs() -> None:
    """Guard against over-rejection: matched pairs of length >= 2 are accepted."""
    ev = SimpleNamespace(
        sample_1_inc_replicates=(10, 20), sample_1_exc_replicates=(5, 10),
    )
    out = _complete_replicate_arrays(ev, "sample_1")
    assert out is not None
    inc, exc = out
    assert inc.tolist() == [10.0, 20.0] and exc.tolist() == [5.0, 10.0]


def test_complete_arrays_drops_zero_support_replicate() -> None:
    """A2: a replicate with no reads (inc+exc==0) carries no PSI info -> dropped, so
    the equal-weight rep mean is not polluted with a spurious Beta(0.5,0.5)~0.5 draw."""
    ev = SimpleNamespace(
        sample_1_inc_replicates=(100, 100, 0),
        sample_1_exc_replicates=(5, 5, 0),  # third replicate is 0/0 -> dropped
    )
    out = _complete_replicate_arrays(ev, "sample_1")
    assert out is not None
    inc, exc = out
    assert inc.tolist() == [100.0, 100.0] and exc.tolist() == [5.0, 5.0]


def test_complete_arrays_rejects_under_two_informative_replicates() -> None:
    """A2/A3: after dropping zero-support replicates, fewer than two informative ones
    remain -> None (caller falls back to sum / errors in rep)."""
    ev = SimpleNamespace(
        sample_1_inc_replicates=(100, 0),
        sample_1_exc_replicates=(5, 0),  # only one informative replicate left
    )
    assert _complete_replicate_arrays(ev, "sample_1") is None


# ==============================================================================
# replicate draw numerical safety
# ==============================================================================
def test_draw_replicate_mean_psi_all_zero_is_finite() -> None:
    """A2/A10: an all-zero replicate (inc=exc=0) gives Beta(0.5,0.5) draws, never
    NaN/inf; the group mean stays a valid PSI in [0,1]."""
    rng = np.random.default_rng(0)
    out = _draw_replicate_mean_psi(
        np.zeros(3), np.zeros(3), ratio=1.0, draws=500, rng=rng
    )
    assert out.shape == (500,)
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_draw_replicate_mean_psi_extreme_ratio_is_finite() -> None:
    """A10: a large length ratio scales exclusion but must keep PSI finite in [0,1]."""
    rng = np.random.default_rng(1)
    out = _draw_replicate_mean_psi(
        np.array([100.0, 50.0]), np.array([1.0, 2.0]), ratio=1000.0, draws=500, rng=rng
    )
    assert np.all(np.isfinite(out)) and np.all((out >= 0.0) & (out <= 1.0))


# ==============================================================================
# model resolver contract
# ==============================================================================
def test_resolve_model_rejects_invalid_value() -> None:
    """F1: an unknown model string must fail fast with a clear message."""
    args = argparse.Namespace(differential_model="glmm")
    with pytest.raises(ValueError, match="differential_model must be one of"):
        _resolve_differential_model(args)


def test_resolve_model_legacy_replicate_aware_flag() -> None:
    """F1: the old programmatic replicate_aware flag still maps cleanly when
    differential_model is absent (auto when True, sum when False)."""
    assert _resolve_differential_model(
        argparse.Namespace(replicate_aware=True)) == "auto"
    assert _resolve_differential_model(
        argparse.Namespace(replicate_aware=False)) == "sum"


def test_no_replicate_aware_alias_resolves_to_sum() -> None:
    """F1: the --no-replicate-aware CLI alias must set differential_model='sum'."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    add_differential_subparser(sub)
    ns = parser.parse_args(
        ["differential", "--rmats-dir", "x", "--no-replicate-aware"])
    assert ns.differential_model == "sum"
    # and the plain default is auto
    ns2 = parser.parse_args(["differential", "--rmats-dir", "x"])
    assert ns2.differential_model == "auto"


# ==============================================================================
# dispatch integration (auto fallback / rep fail-fast)
# ==============================================================================
def test_auto_falls_back_to_sum_on_vector_length_mismatch(tmp_path: Path) -> None:
    """A8 end-to-end: a row whose sample_1 inc has 3 reps but exc has 2 (trailing
    empty) cannot be paired; auto must fall back to the pooled sum model, not crash
    or index a desynced pair."""
    # IJC_SAMPLE_1 has 3 reps; SJC_SAMPLE_1 trailing-empty -> 2 reps after parse.
    row = _rep_row(1, "MM", 1000, "60,60,60", "20,20,", "60,55,65", "140,145,135")
    _write(tmp_path / "rmats", [row])
    out = tmp_path / "o.tsv"
    run_differential(_args(tmp_path / "rmats", out))
    r = _read(out)[0]
    assert r["differential_model"] == "sum"          # fell back
    assert np.isfinite(float(r["dpsi"]))             # produced a valid number


def test_rep_mode_writes_no_partial_output_on_error(tmp_path: Path) -> None:
    """rep mode must fail fast and leave NO output file (no half-written TSV) when an
    event lacks complete replicate vectors."""
    row = _rep_row(1, "ONE", 1000, "180", "20", "60", "140")  # single replicate
    _write(tmp_path / "rmats", [row])
    out = tmp_path / "o.tsv"
    with pytest.raises(ValueError, match="differential_model=rep requires"):
        run_differential(_args(tmp_path / "rmats", out, differential_model="rep"))
    assert not out.exists()
