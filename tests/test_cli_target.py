"""Tests for targeted-assembly CLI contracts."""

from __future__ import annotations

from argparse import Namespace

import pytest

from braid.cli import _run_target
from braid.target.assembler import IsoformResult, TargetAssemblyResult


def _target_args(**overrides: object) -> Namespace:
    base = {
        "bam": "sample.bam",
        "gene": None,
        "region": "chr1:101-250",
        "gtf": None,
        "reference": None,
        "flank": 1000,
        "max_paths": 5000,
        "bootstrap_replicates": 200,
        "min_presence": 0.3,
        "strandedness": "none",
        "format": "text",
        "output": None,
        "verbose": False,
    }
    base.update(overrides)
    return Namespace(**base)


def test_run_target_rejects_gene_and_region_together() -> None:
    args = _target_args(gene="TP53", region="chr1:101-250", gtf="genes.gtf")

    with pytest.raises(SystemExit, match="mutually exclusive"):
        _run_target(args)


def test_run_target_gtf_preserves_zero_bootstrap_attributes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    out = tmp_path / "target.gtf"
    def fake_assemble_target(config):
        return TargetAssemblyResult(
            region=config.region,
            isoforms=[
                IsoformResult(
                    transcript_id="iso1",
                    exons=[(100, 150), (200, 250)],
                    strand="+",
                    weight=0.0,
                    score=0.0,
                    ci_low=0.0,
                    ci_high=0.0,
                    presence_rate=0.0,
                    cv=0.0,
                ),
            ],
        )

    monkeypatch.setattr("braid.target.assembler.assemble_target", fake_assemble_target)

    _run_target(_target_args(region="chr1:101-250", format="gtf", output=str(out)))

    text = out.read_text()
    assert 'bootstrap_ci_low "0.00"' in text
    assert 'bootstrap_ci_high "0.00"' in text
    assert 'bootstrap_presence "0.000"' in text
    assert 'bootstrap_cv "0.000"' in text
