"""Tests for assembly diagnostics collection and reporting."""

from __future__ import annotations

import json

from braid.diagnostics import (
    ChromosomeDiagnostics,
    DiagnosticsCollector,
    LocusDiagnostics,
)
from braid.utils.stats import AssemblyStats


def test_diagnostics_collector_writes_summary_and_jsonl(tmp_path) -> None:
    """Collector writes chromosome/locus records and aggregate summary."""
    out_dir = tmp_path / "diagnostics"
    collector = DiagnosticsCollector(out_dir)

    collector.record_chromosome(
        ChromosomeDiagnostics(
            chrom="chr1",
            spliced_reads=12,
            raw_junctions=8,
            anchor_filtered_junctions=6,
            motif_filtered_junctions=5,
            filtered_junctions=5,
            n_loci=2,
            stage_timings={
                "junction_extraction": 1.5,
                "identify_loci": 0.5,
            },
        )
    )
    collector.record_locus(
        LocusDiagnostics(
            chrom="chr1",
            start=100,
            end=500,
            strand="+",
            raw_junctions=3,
            n_reads=15,
            graph_built=True,
            graph_nodes=6,
            graph_edges=7,
            phasing_paths=2,
            phasing_paths_dropped=3,
            fallback_source_edges_added=1,
            fallback_sink_edges_added=0,
            stage_timings={
                "read_fetch": 0.1,
                "graph_build": 0.2,
            },
            decomposition_method="legacy",
            scorer_mode="heuristic_fallback",
            decomposition_metrics={
                "accepted_paths": 4,
                "phasing_match_rate": 0.5,
            },
            candidates_before_merge=4,
            candidates_after_merge=3,
            surviving_transcripts=2,
            shadow_decomposition_method="iterative_v2",
            shadow_decomposition_metrics={
                "accepted_paths": 5,
                "phasing_match_rate": 1.0,
            },
            shadow_candidates_before_merge=5,
            shadow_candidates_after_merge=4,
            shadow_surviving_transcripts=3,
            filter_diagnostics={
                "initial": 4,
                "after_score": 3,
                "after_coverage": 3,
                "after_junction_support": 2,
                "after_length": 2,
                "after_redundancy": 2,
                "after_cap": 2,
            },
        )
    )
    collector.record_locus(
        LocusDiagnostics(
            chrom="chr1",
            start=600,
            end=900,
            strand="-",
            raw_junctions=2,
            n_reads=6,
            phasing_paths_dropped=1,
            fallback_source_edges_added=0,
            fallback_sink_edges_added=1,
            stage_timings={"read_fetch": 0.05},
            decomposition_method="legacy",
            scorer_mode="trained_model",
            skipped_reason="filtered_out",
            filter_diagnostics={
                "initial": 2,
                "after_score": 1,
                "after_coverage": 1,
                "after_junction_support": 0,
                "after_length": 0,
                "after_redundancy": 0,
                "after_cap": 0,
            },
        )
    )

    stats = AssemblyStats(total_reads=20, total_loci=2, assembled_transcripts=2)
    collector.finalize(stats)

    chromosomes_path = out_dir / "chromosomes.jsonl"
    loci_path = out_dir / "loci.jsonl"
    summary_path = out_dir / "summary.json"

    assert chromosomes_path.exists()
    assert loci_path.exists()
    assert summary_path.exists()

    with open(chromosomes_path, encoding="utf-8") as fh:
        chrom_rows = [json.loads(line) for line in fh]
    with open(loci_path, encoding="utf-8") as fh:
        locus_rows = [json.loads(line) for line in fh]
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)

    assert len(chrom_rows) == 1
    assert len(locus_rows) == 2
    assert locus_rows[0]["decomposition_metrics"]["accepted_paths"] == 4
    assert locus_rows[0]["shadow_decomposition_metrics"]["accepted_paths"] == 5
    assert summary["chromosomes"] == 1
    assert summary["loci"] == 2
    assert summary["graphs_built"] == 1
    assert summary["loci_with_reads"] == 2
    assert summary["loci_with_candidates"] == 1
    assert summary["loci_with_survivors"] == 1
    assert summary["loci_with_shadow"] == 1
    assert summary["total_candidates_before_merge"] == 4
    assert summary["total_candidates_after_merge"] == 3
    assert summary["total_surviving_transcripts"] == 2
    assert summary["total_shadow_candidates_before_merge"] == 5
    assert summary["total_shadow_candidates_after_merge"] == 4
    assert summary["total_shadow_surviving_transcripts"] == 3
    assert summary["total_phasing_paths_dropped"] == 4
    assert summary["fallback_source_edges_added"] == 1
    assert summary["fallback_sink_edges_added"] == 1
    assert summary["junction_counts"] == {
        "raw": 8,
        "after_anchor": 6,
        "after_motif": 5,
        "after_adaptive": 5,
    }
    assert summary["skipped_reasons"] == {"filtered_out": 1}
    assert summary["filter_totals"]["initial"] == 6
    assert summary["filter_totals"]["after_junction_support"] == 2
    assert summary["chromosome_stage_timing_seconds_total"]["junction_extraction"] == 1.5
    assert summary["chromosome_stage_timing_seconds_mean"]["identify_loci"] == 0.5
    assert summary["locus_stage_timing_seconds_total"]["read_fetch"] == 0.15
    assert summary["locus_stage_timing_seconds_mean"]["read_fetch"] == 0.075
    assert summary["assembly_stats"]["total_reads"] == 20
