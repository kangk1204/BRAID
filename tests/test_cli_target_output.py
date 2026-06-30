"""Regression tests for target CLI output formatting."""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

from braid.cli import _run_target
from braid.io.gtf_writer import GtfWriter, TranscriptRecord


def test_run_target_gtf_preserves_zero_bootstrap_fields(monkeypatch, tmp_path):
    region = SimpleNamespace(chrom="chr1", start=100, end=200, strand="+", gene_name="GENE")
    iso = SimpleNamespace(
        transcript_id="GENE.1",
        strand="+",
        exons=[(100, 200)],
        score=0.0,
        weight=0.0,
        ci_low=0.0,
        ci_high=0.0,
        presence_rate=0.0,
        cv=0.0,
        n_junctions=0,
    )
    result = SimpleNamespace(
        isoforms=[iso],
        n_isoforms=1,
        n_confident=0,
        assembly_time_seconds=0.0,
        bootstrap_time_seconds=0.0,
    )

    monkeypatch.setattr("braid.target.extractor.parse_region_string", lambda value: region)
    monkeypatch.setattr("braid.target.assembler.assemble_target", lambda config: result)

    out = tmp_path / "target.gtf"
    _run_target(Namespace(
        bam="reads.bam", reference=None, region="chr1:100-200", gene=None, gtf=None,
        flank=1000, max_paths=10, bootstrap_replicates=5, min_presence=0.0,
        strandedness="none", format="gtf", output=str(out), verbose=False,
    ))

    text = out.read_text()
    assert 'bootstrap_ci_low "0.00"' in text
    assert 'bootstrap_ci_high "0.00"' in text
    assert 'bootstrap_presence "0.000"' in text
    assert 'bootstrap_cv "0.000"' in text


def test_gtf_writer_allows_partial_bootstrap_fields(tmp_path):
    """Partial optional bootstrap metadata should not format None as a float."""
    out = tmp_path / "partial.gtf"
    record = TranscriptRecord(
        transcript_id="tx1",
        gene_id="gene1",
        chrom="chr1",
        strand="+",
        start=100,
        end=200,
        exons=[(100, 200)],
        bootstrap_ci_low=0.123,
    )

    GtfWriter(str(out)).write_transcripts([record])

    text = out.read_text()
    assert 'bootstrap_ci_low "0.12"' in text
    assert "bootstrap_ci_high" not in text
    assert "bootstrap_presence" not in text
    assert "bootstrap_cv" not in text
