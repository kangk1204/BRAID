"""Tests for assembly CLI option validation."""

from __future__ import annotations

import sys
from argparse import Namespace

import pytest

import braid.cli as cli_module
from braid.cli import _build_pipeline_config, _run_assemble


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
        "engine": "legacy",
        "shadow_engine": None,
        "candidate_budget": 256,
        "candidate_beam_width": 32,
        "complexity_penalty": 0.05,
        "relaxed_pruning": False,
        "builder_profile": "default",
    }
    base.update(overrides)
    return Namespace(**base)


def test_run_assemble_rejects_nmf_with_nonlegacy_engine() -> None:
    """NMF mode must not be combined with the new engine selector."""
    args = _assemble_args(nmf=True, engine="iterative_v2")
    with pytest.raises(SystemExit, match="--nmf cannot be combined with --engine"):
        _run_assemble(args)


def test_run_assemble_rejects_nmf_with_shadow_engine() -> None:
    """NMF mode must not advertise shadow engine diagnostics."""
    args = _assemble_args(nmf=True, shadow_engine="iterative_v2")
    with pytest.raises(SystemExit, match="--nmf cannot be combined with --shadow-engine"):
        _run_assemble(args)


def test_build_pipeline_config_coerces_nmf_with_braid_v2_engine() -> None:
    """NMF mode keeps the new default engine surface but forces the legacy backend."""
    args = _assemble_args(nmf=True, engine="braid_v2")
    config = _build_pipeline_config("synthetic.bam", args, "out.gtf")
    assert config.use_nmf_decomposition is True
    assert config.decomposer == "legacy"


def test_run_assemble_forwards_builder_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI flags should populate the pipeline config without hidden defaults."""

    seen: dict[str, object] = {}

    class DummyPipeline:
        def __init__(self, config: object) -> None:
            seen["config"] = config

        def run(self) -> str:
            return "dummy.gtf"

    monkeypatch.setattr("braid.cli.AssemblyPipeline", DummyPipeline)

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


def test_main_rejects_dotty_unknown_command(monkeypatch, capsys) -> None:
    """Mistyped subcommands such as run.py should not hit legacy BAM parsing."""
    monkeypatch.setattr(sys, "argv", ["braid", "run.py"])

    with pytest.raises(SystemExit) as exc:
        cli_module.main()

    assert exc.value.code == 1
    assert "Unknown command: run.py" in capsys.readouterr().err


def test_run_assemble_forwards_braid_v2_candidate_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new braid_v2 candidate knobs should propagate into pipeline config."""
    seen: dict[str, object] = {}

    class DummyPipeline:
        def __init__(self, config: object) -> None:
            seen["config"] = config

        def run(self) -> str:
            return "dummy.gtf"

    monkeypatch.setattr("braid.cli.AssemblyPipeline", DummyPipeline)

    _run_assemble(_assemble_args(
        engine="braid_v2",
        candidate_budget=96,
        candidate_beam_width=12,
        complexity_penalty=0.125,
    ))

    config = seen["config"]
    assert config.decomposer == "braid_v2"
    assert config.candidate_budget == 96
    assert config.candidate_beam_width == 12
    assert config.complexity_penalty == 0.125
