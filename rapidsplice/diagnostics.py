"""Run diagnostics and stage-level instrumentation for assembly recovery.

This module records lightweight chromosome- and locus-level metrics so that
assembler failures can be localized to a specific stage: junction extraction,
graph construction, decomposition, or filtering.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rapidsplice.utils.stats import AssemblyStats


@dataclass(slots=True)
class ChromosomeDiagnostics:
    """High-level metrics for a chromosome assembly pass."""

    chrom: str
    spliced_reads: int
    raw_junctions: int
    anchor_filtered_junctions: int
    motif_filtered_junctions: int
    filtered_junctions: int
    n_loci: int
    stage_timings: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class LocusDiagnostics:
    """Stage metrics for a single locus."""

    chrom: str
    start: int
    end: int
    strand: str
    raw_junctions: int
    n_reads: int = 0
    graph_built: bool = False
    graph_nodes: int = 0
    graph_edges: int = 0
    phasing_paths: int = 0
    phasing_paths_dropped: int = 0
    fallback_source_edges_added: int = 0
    fallback_sink_edges_added: int = 0
    stage_timings: dict[str, float] = field(default_factory=dict)
    decomposition_method: str = "legacy"
    scorer_mode: str = "heuristic_fallback"
    decomposition_metrics: dict[str, float | int] = field(default_factory=dict)
    candidates_before_merge: int = 0
    candidates_after_merge: int = 0
    surviving_transcripts: int = 0
    shadow_decomposition_method: str | None = None
    shadow_decomposition_metrics: dict[str, float | int] = field(default_factory=dict)
    shadow_candidates_before_merge: int = 0
    shadow_candidates_after_merge: int = 0
    shadow_surviving_transcripts: int = 0
    skipped_reason: str | None = None
    filter_diagnostics: dict[str, int] = field(default_factory=dict)
    shadow_filter_diagnostics: dict[str, int] = field(default_factory=dict)


class DiagnosticsCollector:
    """Collect and write assembly diagnostics to disk."""

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._chrom_records: list[dict[str, Any]] = []
        self._locus_records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    @property
    def output_dir(self) -> Path:
        """Directory where diagnostics files are written."""
        return self._output_dir

    def record_chromosome(self, record: ChromosomeDiagnostics) -> None:
        """Append a chromosome-level record."""
        with self._lock:
            self._chrom_records.append(asdict(record))

    def record_locus(self, record: LocusDiagnostics) -> None:
        """Append a locus-level record."""
        with self._lock:
            self._locus_records.append(asdict(record))

    def finalize(self, stats: AssemblyStats | None = None) -> None:
        """Write JSONL and summary outputs to disk.

        Takes a snapshot of all records under the lock so that any
        concurrent ``record_*`` calls do not mutate lists mid-iteration.
        """
        with self._lock:
            chrom_snapshot = list(self._chrom_records)
            locus_snapshot = list(self._locus_records)

        self._write_jsonl(self._output_dir / "chromosomes.jsonl", chrom_snapshot)
        self._write_jsonl(self._output_dir / "loci.jsonl", locus_snapshot)

        summary = self._build_summary(chrom_snapshot, locus_snapshot, stats)
        with open(self._output_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)

    def _build_summary(
        self,
        chroms: list[dict[str, Any]],
        loci: list[dict[str, Any]],
        stats: AssemblyStats | None,
    ) -> dict[str, Any]:
        n_loci = len(loci)
        n_graphs = sum(1 for row in loci if row["graph_built"])
        n_with_reads = sum(1 for row in loci if row["n_reads"] > 0)
        n_with_candidates = sum(1 for row in loci if row["candidates_before_merge"] > 0)
        n_with_survivors = sum(1 for row in loci if row["surviving_transcripts"] > 0)
        n_with_shadow = sum(
            1 for row in loci if row.get("shadow_decomposition_method") is not None
        )

        skipped_reasons: dict[str, int] = {}
        for row in loci:
            reason = row.get("skipped_reason")
            if reason:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1

        filter_totals = {
            "initial": 0,
            "after_score": 0,
            "after_coverage": 0,
            "after_junction_support": 0,
            "after_length": 0,
            "after_redundancy": 0,
            "after_cap": 0,
        }
        for row in loci:
            diag = row.get("filter_diagnostics") or {}
            for key in filter_totals:
                filter_totals[key] += int(diag.get(key, 0))

        chrom_stage_totals, chrom_stage_means = _aggregate_stage_timings(
            chroms, "stage_timings",
        )
        locus_stage_totals, locus_stage_means = _aggregate_stage_timings(
            loci, "stage_timings",
        )

        summary: dict[str, Any] = {
            "chromosomes": len(chroms),
            "loci": n_loci,
            "loci_with_reads": n_with_reads,
            "graphs_built": n_graphs,
            "loci_with_candidates": n_with_candidates,
            "loci_with_survivors": n_with_survivors,
            "loci_with_shadow": n_with_shadow,
            "total_candidates_before_merge": sum(
                int(row["candidates_before_merge"]) for row in loci
            ),
            "total_candidates_after_merge": sum(
                int(row["candidates_after_merge"]) for row in loci
            ),
            "total_surviving_transcripts": sum(
                int(row["surviving_transcripts"]) for row in loci
            ),
            "total_shadow_candidates_before_merge": sum(
                int(row.get("shadow_candidates_before_merge", 0)) for row in loci
            ),
            "total_shadow_candidates_after_merge": sum(
                int(row.get("shadow_candidates_after_merge", 0)) for row in loci
            ),
            "total_shadow_surviving_transcripts": sum(
                int(row.get("shadow_surviving_transcripts", 0)) for row in loci
            ),
            "total_phasing_paths_dropped": sum(
                int(row.get("phasing_paths_dropped", 0)) for row in loci
            ),
            "fallback_source_edges_added": sum(
                int(row.get("fallback_source_edges_added", 0)) for row in loci
            ),
            "fallback_sink_edges_added": sum(
                int(row.get("fallback_sink_edges_added", 0)) for row in loci
            ),
            "junction_counts": {
                "raw": sum(int(row.get("raw_junctions", 0)) for row in chroms),
                "after_anchor": sum(
                    int(row.get("anchor_filtered_junctions", 0)) for row in chroms
                ),
                "after_motif": sum(
                    int(row.get("motif_filtered_junctions", 0)) for row in chroms
                ),
                "after_adaptive": sum(
                    int(row.get("filtered_junctions", 0)) for row in chroms
                ),
            },
            "skipped_reasons": skipped_reasons,
            "filter_totals": filter_totals,
            "chromosome_stage_timing_seconds_total": chrom_stage_totals,
            "chromosome_stage_timing_seconds_mean": chrom_stage_means,
            "locus_stage_timing_seconds_total": locus_stage_totals,
            "locus_stage_timing_seconds_mean": locus_stage_means,
        }
        if stats is not None:
            summary["assembly_stats"] = asdict(stats)
        return summary

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True))
                fh.write("\n")


def _aggregate_stage_timings(
    rows: list[dict[str, Any]],
    key: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate per-stage timings across records."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        timings = row.get(key) or {}
        for stage, value in timings.items():
            seconds = float(value)
            totals[stage] = totals.get(stage, 0.0) + seconds
            counts[stage] = counts.get(stage, 0) + 1
    means = {
        stage: round(total / counts[stage], 6)
        for stage, total in totals.items()
        if counts.get(stage, 0) > 0
    }
    rounded_totals = {stage: round(total, 6) for stage, total in totals.items()}
    return rounded_totals, means
