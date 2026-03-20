"""Tests for focused decomposer benchmark automation."""

from __future__ import annotations

import sys
from pathlib import Path

from benchmarks.run_decomposer_comparison import (
    VariantSpec,
    _aggregate_metric_maps,
    _build_loss_accounting,
    _build_assemble_command,
    _build_delta_summary,
    _resolve_baseline_variant,
)


def test_build_assemble_command_enables_diagnostics_and_relaxed_pruning() -> None:
    """Comparison command should always emit diagnostics-rich recovery runs."""
    cmd = _build_assemble_command(
        Path("input.bam"),
        Path("reference.fa"),
        Path("out.gtf"),
        Path("diag"),
        VariantSpec(
            name="iterative_v2_relaxed",
            decomposer="iterative_v2",
            builder_profile="aggressive_recall",
            relaxed_pruning=True,
            shadow_decomposer="legacy",
        ),
        threads=2,
        backend="cpu",
        builder_profile="default",
        min_coverage=1.0,
        min_score=0.1,
        min_junction_support=3,
        min_phasing_support=2,
    )

    assert cmd[:4] == [sys.executable, "-m", "braid.cli", "assemble"]
    assert "--diagnostics-dir" in cmd
    assert "--decomposer" in cmd
    assert "--shadow-decomposer" in cmd
    assert "--builder-profile" in cmd
    assert "--relaxed-pruning" in cmd
    assert "--no-safe-paths" in cmd
    assert "--no-ml-scoring" in cmd
    assert cmd[cmd.index("--min-phasing-support") + 1] == "2"
    assert cmd[cmd.index("--builder-profile") + 1] == "aggressive_recall"


def test_build_assemble_command_can_disable_motif_validation() -> None:
    """Recovery variants should be able to benchmark with motif checks off."""
    cmd = _build_assemble_command(
        Path("input.bam"),
        Path("reference.fa"),
        Path("out.gtf"),
        Path("diag"),
        VariantSpec(
            name="iterative_v2_no_motif_validation",
            decomposer="iterative_v2",
            enable_motif_validation=False,
        ),
        threads=1,
        backend="cpu",
        builder_profile="default",
        min_coverage=1.0,
        min_score=0.1,
        min_junction_support=3,
        min_phasing_support=1,
    )

    assert "--no-motif-validation" in cmd


def test_aggregate_metric_maps_summarizes_numeric_locus_metrics() -> None:
    """Numeric decomposition metrics should be rolled up for reporting."""
    rows = [
        {
            "decomposition_metrics": {
                "accepted_paths": 3,
                "phasing_match_rate": 0.5,
                "max_paths_hit": 1,
            }
        },
        {
            "decomposition_metrics": {
                "accepted_paths": 5,
                "phasing_match_rate": 1.0,
                "max_paths_hit": 0,
            }
        },
        {"decomposition_metrics": {}},
    ]

    aggregate = _aggregate_metric_maps(rows, "decomposition_metrics")

    assert aggregate["loci_with_metrics"] == 2
    assert aggregate["sum"]["accepted_paths"] == 8.0
    assert aggregate["mean"]["phasing_match_rate"] == 0.75
    assert aggregate["max"]["max_paths_hit"] == 1.0


def test_aggregate_metric_maps_ignores_nonfinite_values() -> None:
    """NaN/inf decomposition metrics should not poison summary aggregates."""
    rows = [
        {"decomposition_metrics": {"accepted_paths": 3, "a_condition_number": float("nan")}},
        {"decomposition_metrics": {"accepted_paths": 5, "a_condition_number": float("inf")}},
    ]

    aggregate = _aggregate_metric_maps(rows, "decomposition_metrics")

    assert aggregate["sum"]["accepted_paths"] == 8.0
    assert "a_condition_number" not in aggregate["sum"]


def test_build_delta_summary_compares_against_legacy_baseline() -> None:
    """Comparison summary should expose deltas against the legacy run."""
    rows = [
        {
            "variant": "legacy",
            "status": "ok",
            "runtime_seconds": 10.0,
            "peak_memory_mb": 100.0,
            "transcript_count": 4,
            "metrics": {
                "transcript_sensitivity": 0.2,
                "transcript_precision": 0.3,
            },
            "diagnostics": {
                "summary": {
                    "graphs_built": 5,
                    "loci_with_candidates": 4,
                    "loci_with_survivors": 3,
                    "total_candidates_before_merge": 10,
                    "total_candidates_after_merge": 7,
                    "total_surviving_transcripts": 4,
                }
            },
        },
        {
            "variant": "iterative_v2",
            "status": "ok",
            "runtime_seconds": 12.5,
            "peak_memory_mb": 130.0,
            "transcript_count": 6,
            "metrics": {
                "transcript_sensitivity": 0.35,
                "transcript_precision": 0.25,
            },
            "diagnostics": {
                "summary": {
                    "graphs_built": 5,
                    "loci_with_candidates": 5,
                    "loci_with_survivors": 4,
                    "total_candidates_before_merge": 12,
                    "total_candidates_after_merge": 8,
                    "total_surviving_transcripts": 6,
                }
            },
        },
    ]

    delta = _build_delta_summary(rows, baseline_variant="legacy")

    iterative = delta["variants"]["iterative_v2"]
    assert delta["baseline"] == "legacy"
    assert iterative["runtime_seconds_delta"] == 2.5
    assert iterative["peak_memory_mb_delta"] == 30.0
    assert iterative["transcript_count_delta"] == 2
    assert iterative["metrics"]["transcript_sensitivity"] == 0.15
    assert iterative["metrics"]["transcript_precision"] == -0.05
    assert iterative["diagnostics_summary"]["loci_with_candidates"] == 1.0


def test_resolve_baseline_variant_prefers_legacy_no_motif_when_available() -> None:
    """Synthetic recovery comparisons should baseline against motif-off legacy."""
    rows = [
        {"variant": "iterative_v2_no_motif_validation", "status": "ok"},
        {"variant": "legacy_no_motif_validation", "status": "ok"},
        {"variant": "legacy", "status": "ok"},
    ]

    assert _resolve_baseline_variant(rows) == "legacy_no_motif_validation"


def test_build_loss_accounting_classifies_fragmentation_and_decomposition_miss(tmp_path) -> None:
    """Truth transcripts should distinguish locus fragmentation from downstream misses."""
    truth = tmp_path / "truth.gtf"
    pred = tmp_path / "pred.gtf"
    diagnostics_dir = tmp_path / "diag"
    diagnostics_dir.mkdir()
    loci = diagnostics_dir / "loci.jsonl"

    truth.write_text(
        "chr1\ttruth\texon\t101\t150\t.\t+\t.\tgene_id \"g1\"; transcript_id \"t1\";\n"
        "chr1\ttruth\texon\t201\t250\t.\t+\t.\tgene_id \"g1\"; transcript_id \"t1\";\n"
        "chr1\ttruth\texon\t401\t450\t.\t+\t.\tgene_id \"g2\"; transcript_id \"t2\";\n"
        "chr1\ttruth\texon\t601\t650\t.\t+\t.\tgene_id \"g2\"; transcript_id \"t2\";\n"
        "chr1\ttruth\texon\t801\t850\t.\t+\t.\tgene_id \"g3\"; transcript_id \"t3\";\n"
        "chr1\ttruth\texon\t901\t950\t.\t+\t.\tgene_id \"g3\"; transcript_id \"t3\";\n",
        encoding="utf-8",
    )
    pred.write_text(
        "chr1\tpred\texon\t101\t150\t.\t+\t.\tgene_id \"g1\"; transcript_id \"p1\";\n"
        "chr1\tpred\texon\t201\t250\t.\t+\t.\tgene_id \"g1\"; transcript_id \"p1\";\n",
        encoding="utf-8",
    )
    loci.write_text(
        "{\"chrom\":\"chr1\",\"start\":100,\"end\":250,\"strand\":\"+\",\"graph_built\":true,"
        "\"candidates_before_merge\":2,\"surviving_transcripts\":1}\n"
        "{\"chrom\":\"chr1\",\"start\":420,\"end\":620,\"strand\":\"+\",\"graph_built\":true,"
        "\"candidates_before_merge\":2,\"surviving_transcripts\":1}\n"
        "{\"chrom\":\"chr1\",\"start\":800,\"end\":950,\"strand\":\"+\",\"graph_built\":true,"
        "\"candidates_before_merge\":0,\"surviving_transcripts\":0}\n",
        encoding="utf-8",
    )

    accounting = _build_loss_accounting(truth, pred, diagnostics_dir)

    assert accounting["truth_multi_exon_transcripts"] == 3
    assert accounting["recovered_truth_transcripts"] == 1
    assert accounting["fragmented_locus_truth_transcripts"] == 1
    assert accounting["decomposition_miss_truth_transcripts"] == 1
    assert accounting["recovery_rate"] == 0.333333
