"""Tests for assembly CLI option validation."""

from __future__ import annotations

from argparse import Namespace

import pytest

from rapidsplice.cli import _run_assemble


def _assemble_args(**overrides: object) -> Namespace:
    """Build a minimal assemble-namespace for option validation tests."""
    base = {
        "bam": "synthetic.bam",
        "reference": None,
        "output": "out.gtf",
        "output_format": "gtf",
        "backend": "cpu",
        "gpu": False,
        "threads": 1,
        "min_mapq": 0,
        "min_junction_support": 2,
        "min_phasing_support": 1,
        "min_coverage": 1.0,
        "min_score": 0.3,
        "max_intron_length": 500_000,
        "min_anchor_length": 8,
        "max_paths": 500,
        "no_adaptive_junction_filter": False,
        "no_motif_validation": False,
        "no_safe_paths": False,
        "no_ml_scoring": False,
        "model": None,
        "chromosomes": None,
        "verbose": False,
        "nmf": False,
        "diagnostics_dir": None,
        "decomposer": "legacy",
        "shadow_decomposer": None,
        "relaxed_pruning": False,
        "builder_profile": "default",
    }
    base.update(overrides)
    return Namespace(**base)


def test_run_assemble_rejects_nmf_with_nonlegacy_decomposer() -> None:
    """NMF mode must not be combined with the new decomposer selector."""
    args = _assemble_args(nmf=True, decomposer="iterative_v2")
    with pytest.raises(SystemExit, match="--nmf cannot be combined with --decomposer"):
        _run_assemble(args)


def test_run_assemble_rejects_nmf_with_shadow_decomposer() -> None:
    """NMF mode must not advertise shadow decomposer diagnostics."""
    args = _assemble_args(nmf=True, shadow_decomposer="iterative_v2")
    with pytest.raises(SystemExit, match="--nmf cannot be combined with --shadow-decomposer"):
        _run_assemble(args)


def test_run_assemble_forwards_builder_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI flags should populate the pipeline config without hidden defaults."""

    seen: dict[str, object] = {}

    class DummyPipeline:
        def __init__(self, config: object) -> None:
            seen["config"] = config

        def run(self) -> str:
            return "dummy.gtf"

    monkeypatch.setattr("rapidsplice.cli.AssemblyPipeline", DummyPipeline)

    _run_assemble(_assemble_args(
        min_anchor_length=12,
        max_paths=2048,
        no_adaptive_junction_filter=True,
        no_motif_validation=True,
        builder_profile="conservative_correctness",
    ))

    config = seen["config"]
    assert config.min_anchor_length == 12
    assert config.max_paths == 2048
    assert config.adaptive_junction_filter is False
    assert config.enable_motif_validation is False
    assert config.builder_profile == "conservative_correctness"
