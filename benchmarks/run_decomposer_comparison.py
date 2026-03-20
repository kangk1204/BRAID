"""Focused benchmark runner for comparing RapidSplice decomposers.

This script targets the existing synthetic benchmark dataset under
``benchmark_results/synthetic_data`` and runs multiple RapidSplice
configuration variants on the same BAM. Each variant writes:

- assembled GTF output
- diagnostics JSONL/summary artifacts
- stdout/stderr logs
- evaluation metrics against the synthetic truth GTF

Results are consolidated into both JSON and TSV summaries so recovery work can
compare ``legacy`` against ``iterative_v2`` without reusing stale outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import psutil

from benchmarks.run_benchmark import BenchmarkConfig, BenchmarkRunner, _parse_gtf_transcripts
from rapidsplice.io.reference import ReferenceGenome

logger = logging.getLogger(__name__)

DEFAULT_SYNTHETIC_DIR = Path("benchmark_results") / "synthetic_data"
DEFAULT_OUTPUT_DIR = Path("benchmark_results") / "decomposer_compare"
SUMMARY_FIELDS = (
    "variant",
    "status",
    "decomposer",
    "shadow_decomposer",
    "builder_profile",
    "motif_validation",
    "relaxed_pruning",
    "runtime_seconds",
    "peak_memory_mb",
    "transcript_count",
    "transcript_sensitivity",
    "transcript_precision",
    "exon_sensitivity",
    "exon_precision",
    "intron_sensitivity",
    "intron_precision",
    "graphs_built",
    "loci",
    "loci_with_candidates",
    "loci_with_survivors",
    "total_candidates_before_merge",
    "total_candidates_after_merge",
    "total_surviving_transcripts",
    "truth_multi_exon_transcripts",
    "recovered_truth_transcripts",
    "fragmented_locus_truth_transcripts",
    "graph_miss_truth_transcripts",
    "decomposition_miss_truth_transcripts",
    "filter_miss_truth_transcripts",
    "chain_miss_truth_transcripts",
)


@dataclass(frozen=True)
class VariantSpec:
    """One RapidSplice decomposer configuration to benchmark."""

    name: str
    decomposer: str
    builder_profile: str = "default"
    enable_motif_validation: bool = True
    relaxed_pruning: bool = False
    shadow_decomposer: str | None = None


@dataclass
class CommandResult:
    """Execution result for one assemble command."""

    returncode: int
    elapsed_seconds: float
    peak_memory_mb: float
    timed_out: bool
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TranscriptStructure:
    """Transcript structure with coordinates needed for loss accounting."""

    transcript_id: str
    chrom: str
    strand: str
    exons: tuple[tuple[int, int], ...]

    @property
    def start(self) -> int:
        return self.exons[0][0]

    @property
    def end(self) -> int:
        return self.exons[-1][1]

    @property
    def intron_chain(self) -> tuple[tuple[int, int], ...]:
        if len(self.exons) < 2:
            return ()
        return tuple(
            (self.exons[i][1], self.exons[i + 1][0])
            for i in range(len(self.exons) - 1)
        )


def _variant_specs(
    variant_names: list[str],
    shadow_mode: str | None,
) -> list[VariantSpec]:
    variants: list[VariantSpec] = []
    for name in variant_names:
        if name == "legacy":
            variants.append(
                VariantSpec(
                    name="legacy",
                    decomposer="legacy",
                    builder_profile="default",
                    shadow_decomposer=shadow_mode if shadow_mode != "legacy" else None,
                )
            )
        elif name == "iterative_v2":
            variants.append(
                VariantSpec(
                    name="iterative_v2",
                    decomposer="iterative_v2",
                    builder_profile="default",
                    shadow_decomposer=shadow_mode
                    if shadow_mode != "iterative_v2"
                    else None,
                )
            )
        elif name == "iterative_v2_relaxed":
            variants.append(
                VariantSpec(
                    name="iterative_v2_relaxed",
                    decomposer="iterative_v2",
                    builder_profile="aggressive_recall",
                    relaxed_pruning=True,
                    shadow_decomposer=shadow_mode
                    if shadow_mode != "iterative_v2"
                    else None,
                )
            )
        elif name == "legacy_no_motif_validation":
            variants.append(
                VariantSpec(
                    name="legacy_no_motif_validation",
                    decomposer="legacy",
                    builder_profile="default",
                    enable_motif_validation=False,
                    shadow_decomposer=shadow_mode if shadow_mode != "legacy" else None,
                )
            )
        elif name == "iterative_v2_no_motif_validation":
            variants.append(
                VariantSpec(
                    name="iterative_v2_no_motif_validation",
                    decomposer="iterative_v2",
                    builder_profile="default",
                    enable_motif_validation=False,
                    shadow_decomposer=shadow_mode
                    if shadow_mode != "iterative_v2"
                    else None,
                )
            )
        elif name == "iterative_v2_relaxed_no_motif_validation":
            variants.append(
                VariantSpec(
                    name="iterative_v2_relaxed_no_motif_validation",
                    decomposer="iterative_v2",
                    builder_profile="aggressive_recall",
                    enable_motif_validation=False,
                    relaxed_pruning=True,
                    shadow_decomposer=shadow_mode
                    if shadow_mode != "iterative_v2"
                    else None,
                )
            )
        else:
            raise ValueError(f"Unsupported variant: {name}")
    return variants


def _default_dataset_path(filename: str) -> Path:
    return DEFAULT_SYNTHETIC_DIR / filename


def _resolve_existing_input(path_value: str | None, default_filename: str) -> Path:
    path = Path(path_value) if path_value else _default_dataset_path(default_filename)
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}. "
            "Generate the synthetic benchmark dataset first or pass an explicit path."
        )
    return path.resolve()


def _build_assemble_command(
    bam_path: Path,
    reference_path: Path,
    output_gtf: Path,
    diagnostics_dir: Path,
    variant: VariantSpec,
    *,
    threads: int,
    backend: str,
    builder_profile: str = "default",
    min_coverage: float,
    min_score: float,
    min_junction_support: int,
    min_phasing_support: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "rapidsplice.cli",
        "assemble",
        str(bam_path),
        "-o",
        str(output_gtf),
        "-r",
        str(reference_path),
        "--backend",
        backend,
        "-t",
        str(threads),
        "-c",
        str(min_coverage),
        "-s",
        str(min_score),
        "-j",
        str(min_junction_support),
        "--min-phasing-support",
        str(min_phasing_support),
        "--decomposer",
        variant.decomposer,
        "--builder-profile",
        variant.builder_profile if variant.builder_profile != "default" else builder_profile,
        "--diagnostics-dir",
        str(diagnostics_dir),
        "--no-safe-paths",
        "--no-ml-scoring",
    ]
    if variant.shadow_decomposer:
        cmd.extend(["--shadow-decomposer", variant.shadow_decomposer])
    if not variant.enable_motif_validation:
        cmd.append("--no-motif-validation")
    if variant.relaxed_pruning:
        cmd.append("--relaxed-pruning")
    return cmd


def _assess_truth_motif_compatibility(
    reference_path: Path,
    truth_gtf: Path,
) -> dict[str, float | int]:
    """Measure how compatible synthetic truth introns are with motif validation."""
    truth_tx = _parse_gtf_transcripts(str(truth_gtf))
    introns: set[tuple[int, int]] = set()
    chrom_name = "chr1"
    for _tx_id, exons in truth_tx.items():
        sorted_exons = sorted(exons)
        for i in range(len(sorted_exons) - 1):
            introns.add((sorted_exons[i][1], sorted_exons[i + 1][0]))
    if not introns:
        return {
            "truth_unique_introns": 0,
            "canonical_truth_introns": 0,
            "canonical_truth_fraction": 0.0,
        }

    starts = np.array([start for start, _end in sorted(introns)], dtype=np.int64)
    ends = np.array([end for _start, end in sorted(introns)], dtype=np.int64)
    reference = ReferenceGenome(str(reference_path))
    try:
        valid_mask = reference.validate_junctions(chrom_name, starts, ends)
    finally:
        reference.close()
    canonical_truth_introns = int(np.sum(valid_mask))
    return {
        "truth_unique_introns": len(introns),
        "canonical_truth_introns": canonical_truth_introns,
        "canonical_truth_fraction": _round_metric(
            canonical_truth_introns / max(len(introns), 1),
        ),
    }


def _parse_gtf_structures(gtf_path: str | Path) -> list[TranscriptStructure]:
    """Parse transcript exon structures including chromosome and strand."""
    transcripts: dict[str, dict[str, Any]] = {}
    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            attrs = parts[8]
            tx_id = _extract_gtf_attribute(attrs, "transcript_id")
            if tx_id is None:
                continue
            transcripts.setdefault(
                tx_id,
                {
                    "chrom": parts[0],
                    "strand": parts[6],
                    "exons": [],
                },
            )
            transcripts[tx_id]["exons"].append((int(parts[3]) - 1, int(parts[4])))

    structures: list[TranscriptStructure] = []
    for tx_id, info in transcripts.items():
        exons = tuple(sorted(info["exons"]))
        if not exons:
            continue
        structures.append(
            TranscriptStructure(
                transcript_id=tx_id,
                chrom=str(info["chrom"]),
                strand=str(info["strand"]),
                exons=exons,
            )
        )
    return structures


def _extract_gtf_attribute(attrs: str, key: str) -> str | None:
    """Extract a quoted or unquoted GTF attribute."""
    for item in attrs.split(";"):
        item = item.strip()
        if not item.startswith(key):
            continue
        quoted = item.split('"')
        if len(quoted) >= 2:
            return quoted[1]
        parts = item.split()
        if len(parts) >= 2:
            return parts[1].strip('"')
    return None


def _interval_overlap(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> int:
    """Return the overlap length between two half-open intervals."""
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _best_locus_index(
    transcript: TranscriptStructure,
    loci_rows: list[dict[str, Any]],
) -> int | None:
    """Find the best overlapping locus row for a transcript."""
    best_index: int | None = None
    best_overlap = 0
    for idx, row in enumerate(loci_rows):
        if row.get("chrom") != transcript.chrom:
            continue
        row_strand = row.get("strand")
        if row_strand not in {None, ".", transcript.strand}:
            continue
        overlap = _interval_overlap(
            transcript.start,
            transcript.end,
            int(row.get("start", 0)),
            int(row.get("end", 0)),
        )
        if overlap > best_overlap:
            best_index = idx
            best_overlap = overlap
    return best_index


def _covering_locus_indices(
    transcript: TranscriptStructure,
    loci_rows: list[dict[str, Any]],
) -> list[int]:
    """Return loci that fully contain a truth transcript span on the same strand."""
    covering: list[int] = []
    for idx, row in enumerate(loci_rows):
        if row.get("chrom") != transcript.chrom:
            continue
        row_strand = row.get("strand")
        if row_strand not in {None, ".", transcript.strand}:
            continue
        row_start = int(row.get("start", 0))
        row_end = int(row.get("end", 0))
        if row_start <= transcript.start and row_end >= transcript.end:
            covering.append(idx)
    return covering


def _build_loss_accounting(
    truth_gtf: Path,
    predicted_gtf: Path,
    diagnostics_dir: Path,
) -> dict[str, Any]:
    """Classify truth multi-exon transcripts by the stage where they were lost."""
    loci_path = diagnostics_dir / "loci.jsonl"
    if not loci_path.exists():
        return {}

    loci_rows = _read_jsonl(loci_path)
    truth_structures = [
        tx for tx in _parse_gtf_structures(truth_gtf) if len(tx.intron_chain) > 0
    ]
    predicted_structures = (
        _parse_gtf_structures(predicted_gtf) if predicted_gtf.exists() else []
    )
    predicted_by_locus: dict[int, set[tuple[tuple[int, int], ...]]] = {}
    for tx in predicted_structures:
        locus_index = _best_locus_index(tx, loci_rows)
        if locus_index is None:
            continue
        predicted_by_locus.setdefault(locus_index, set()).add(tx.intron_chain)

    accounting = {
        "truth_multi_exon_transcripts": len(truth_structures),
        "recovered_truth_transcripts": 0,
        "no_locus_truth_transcripts": 0,
        "fragmented_locus_truth_transcripts": 0,
        "graph_miss_truth_transcripts": 0,
        "decomposition_miss_truth_transcripts": 0,
        "filter_miss_truth_transcripts": 0,
        "chain_miss_truth_transcripts": 0,
    }

    for truth_tx in truth_structures:
        overlapping_index = _best_locus_index(truth_tx, loci_rows)
        if overlapping_index is None:
            accounting["no_locus_truth_transcripts"] += 1
            continue

        covering_indices = _covering_locus_indices(truth_tx, loci_rows)
        if not covering_indices:
            accounting["fragmented_locus_truth_transcripts"] += 1
            continue

        locus_index = max(
            covering_indices,
            key=lambda idx: _interval_overlap(
                truth_tx.start,
                truth_tx.end,
                int(loci_rows[idx].get("start", 0)),
                int(loci_rows[idx].get("end", 0)),
            ),
        )
        if truth_tx.intron_chain in predicted_by_locus.get(locus_index, set()):
            accounting["recovered_truth_transcripts"] += 1
            continue

        locus_row = loci_rows[locus_index]
        if not bool(locus_row.get("graph_built", False)):
            accounting["graph_miss_truth_transcripts"] += 1
        elif int(locus_row.get("candidates_before_merge", 0)) == 0:
            accounting["decomposition_miss_truth_transcripts"] += 1
        elif int(locus_row.get("surviving_transcripts", 0)) == 0:
            accounting["filter_miss_truth_transcripts"] += 1
        else:
            accounting["chain_miss_truth_transcripts"] += 1

    if accounting["truth_multi_exon_transcripts"] > 0:
        accounting["recovery_rate"] = _round_metric(
            accounting["recovered_truth_transcripts"]
            / accounting["truth_multi_exon_transcripts"],
        )
    else:
        accounting["recovery_rate"] = 0.0
    return accounting


def _run_command(cmd: list[str], cwd: Path, timeout: int) -> CommandResult:
    start_time = time.perf_counter()
    with (
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stdout_file,
        tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file,
    ):
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        proc_handle = psutil.Process(process.pid)
        peak_rss_bytes = 0

        while True:
            try:
                rss_bytes = proc_handle.memory_info().rss
                for child in proc_handle.children(recursive=True):
                    try:
                        rss_bytes += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                peak_rss_bytes = max(peak_rss_bytes, rss_bytes)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            if process.poll() is not None:
                break
            if time.perf_counter() - start_time >= timeout:
                process.kill()
                process.wait()
                stdout_file.seek(0)
                stderr_file.seek(0)
                elapsed = time.perf_counter() - start_time
                return CommandResult(
                    returncode=124,
                    elapsed_seconds=elapsed,
                    peak_memory_mb=_round_metric(peak_rss_bytes / (1024.0 * 1024.0)),
                    timed_out=True,
                    stdout=stdout_file.read(),
                    stderr=stderr_file.read() + f"\nTimed out after {timeout} seconds.\n",
                )
            time.sleep(0.05)

        process.wait()
        stdout_file.seek(0)
        stderr_file.seek(0)
        elapsed = time.perf_counter() - start_time
        return CommandResult(
            returncode=process.returncode,
            elapsed_seconds=elapsed,
            peak_memory_mb=_round_metric(peak_rss_bytes / (1024.0 * 1024.0)),
            timed_out=False,
            stdout=stdout_file.read(),
            stderr=stderr_file.read(),
        )


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _count_transcripts(gtf_path: Path) -> int:
    count = 0
    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2] == "transcript":
                count += 1
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _round_metric(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return value
    return round(value, 6)


def _aggregate_metric_maps(
    rows: list[dict[str, Any]],
    field_name: str,
) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    loci_with_metrics = 0
    for row in rows:
        metric_map = row.get(field_name) or {}
        if metric_map:
            loci_with_metrics += 1
        for key, value in metric_map.items():
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                if not math.isfinite(float(value)):
                    continue
                values.setdefault(key, []).append(float(value))

    aggregates = {
        "loci_with_metrics": loci_with_metrics,
        "sum": {},
        "mean": {},
        "max": {},
    }
    for key, key_values in values.items():
        aggregates["sum"][key] = _round_metric(sum(key_values))
        aggregates["mean"][key] = _round_metric(sum(key_values) / len(key_values))
        aggregates["max"][key] = _round_metric(max(key_values))
    return aggregates


def _load_diagnostics_snapshot(diagnostics_dir: Path) -> dict[str, Any]:
    summary_path = diagnostics_dir / "summary.json"
    loci_path = diagnostics_dir / "loci.jsonl"
    if not summary_path.exists():
        return {}

    loci_rows = _read_jsonl(loci_path)
    snapshot = {
        "summary": _read_json(summary_path),
        "decomposition_metrics": _aggregate_metric_maps(
            loci_rows, "decomposition_metrics"
        ),
    }
    shadow_aggregate = _aggregate_metric_maps(loci_rows, "shadow_decomposition_metrics")
    if shadow_aggregate["loci_with_metrics"] > 0:
        snapshot["shadow_decomposition_metrics"] = shadow_aggregate
    return snapshot


def _build_variant_row(
    *,
    variant: VariantSpec,
    builder_profile: str,
    gtf_path: Path,
    diagnostics_dir: Path,
    logs_dir: Path,
    command: list[str],
    command_result: CommandResult,
    evaluator: BenchmarkRunner,
    truth_gtf: Path,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "variant": variant.name,
        "status": (
            "timed_out"
            if command_result.timed_out
            else ("ok" if command_result.returncode == 0 else "failed")
        ),
        "decomposer": variant.decomposer,
        "shadow_decomposer": variant.shadow_decomposer,
        "builder_profile": builder_profile,
        "motif_validation": "enabled" if variant.enable_motif_validation else "disabled",
        "relaxed_pruning": variant.relaxed_pruning,
        "command": command,
        "gtf_path": str(gtf_path.resolve()),
        "diagnostics_dir": str(diagnostics_dir.resolve()),
        "stdout_log": str((logs_dir / "stdout.log").resolve()),
        "stderr_log": str((logs_dir / "stderr.log").resolve()),
        "runtime_seconds": _round_metric(command_result.elapsed_seconds),
        "peak_memory_mb": _round_metric(command_result.peak_memory_mb),
    }

    if row["status"] == "ok" and gtf_path.exists():
        row["transcript_count"] = _count_transcripts(gtf_path)
        row["metrics"] = {
            key: _round_metric(value)
            for key, value in evaluator.evaluate_gtf(str(gtf_path), str(truth_gtf)).items()
        }
    else:
        row["partial_output_present"] = gtf_path.exists()
        row["transcript_count"] = 0
        row["metrics"] = {}

    diagnostics_snapshot = _load_diagnostics_snapshot(diagnostics_dir)
    if diagnostics_snapshot:
        row["diagnostics"] = diagnostics_snapshot
    loss_accounting = _build_loss_accounting(truth_gtf, gtf_path, diagnostics_dir)
    if loss_accounting:
        row["loss_accounting"] = loss_accounting

    return row


def _build_delta_summary(rows: list[dict[str, Any]], baseline_variant: str) -> dict[str, Any]:
    baseline = next((row for row in rows if row["variant"] == baseline_variant), None)
    if baseline is None or baseline.get("status") != "ok":
        return {"baseline": baseline_variant, "variants": {}}

    baseline_metrics = baseline.get("metrics", {})
    baseline_diag = (baseline.get("diagnostics") or {}).get("summary", {})

    variants: dict[str, Any] = {}
    for row in rows:
        if row["variant"] == baseline_variant or row.get("status") != "ok":
            continue

        metric_delta = {
            key: _round_metric(
                row.get("metrics", {}).get(key, 0.0)
                - baseline_metrics.get(key, 0.0)
            )
            for key in baseline_metrics
            if key in row.get("metrics", {})
        }
        diag_delta: dict[str, float] = {}
        for key in (
            "graphs_built",
            "loci_with_candidates",
            "loci_with_survivors",
            "total_candidates_before_merge",
            "total_candidates_after_merge",
            "total_surviving_transcripts",
        ):
            if key in row.get("diagnostics", {}).get("summary", {}) and key in baseline_diag:
                diag_delta[key] = _round_metric(
                    float(row["diagnostics"]["summary"][key]) - float(baseline_diag[key])
                )

        variants[row["variant"]] = {
            "runtime_seconds_delta": _round_metric(
                float(row["runtime_seconds"]) - float(baseline["runtime_seconds"])
            ),
            "peak_memory_mb_delta": _round_metric(
                float(row["peak_memory_mb"]) - float(baseline["peak_memory_mb"])
            ),
            "transcript_count_delta": row.get("transcript_count", 0)
            - baseline.get("transcript_count", 0),
            "metrics": metric_delta,
            "diagnostics_summary": diag_delta,
        }

    return {"baseline": baseline_variant, "variants": variants}


def _resolve_baseline_variant(rows: list[dict[str, Any]]) -> str:
    """Pick the most appropriate baseline for the current comparison set."""
    ok_variants = {
        str(row["variant"])
        for row in rows
        if row.get("status") == "ok" and row.get("variant") is not None
    }
    for preferred in ("legacy_no_motif_validation", "legacy"):
        if preferred in ok_variants:
            return preferred
    if ok_variants:
        return next(
            str(row["variant"])
            for row in rows
            if row.get("status") == "ok" and row.get("variant") is not None
        )
    return "legacy"


def _write_summary_tsv(rows: list[dict[str, Any]], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SUMMARY_FIELDS), delimiter="\t")
        writer.writeheader()
        for row in rows:
            metrics = row.get("metrics", {})
            diag_summary = (row.get("diagnostics") or {}).get("summary", {})
            loss_accounting = row.get("loss_accounting") or {}
            writer.writerow(
                {
                    "variant": row["variant"],
                    "status": row["status"],
                    "decomposer": row["decomposer"],
                    "shadow_decomposer": row.get("shadow_decomposer") or "",
                    "builder_profile": row.get("builder_profile", "default"),
                    "motif_validation": row.get("motif_validation", "enabled"),
                    "relaxed_pruning": row["relaxed_pruning"],
                    "runtime_seconds": row["runtime_seconds"],
                    "peak_memory_mb": row["peak_memory_mb"],
                    "transcript_count": row.get("transcript_count", 0),
                    "transcript_sensitivity": metrics.get("transcript_sensitivity", ""),
                    "transcript_precision": metrics.get("transcript_precision", ""),
                    "exon_sensitivity": metrics.get("exon_sensitivity", ""),
                    "exon_precision": metrics.get("exon_precision", ""),
                    "intron_sensitivity": metrics.get("intron_sensitivity", ""),
                    "intron_precision": metrics.get("intron_precision", ""),
                    "graphs_built": diag_summary.get("graphs_built", ""),
                    "loci": diag_summary.get("loci", ""),
                    "loci_with_candidates": diag_summary.get("loci_with_candidates", ""),
                    "loci_with_survivors": diag_summary.get("loci_with_survivors", ""),
                    "total_candidates_before_merge": diag_summary.get(
                        "total_candidates_before_merge", ""
                    ),
                    "total_candidates_after_merge": diag_summary.get(
                        "total_candidates_after_merge", ""
                    ),
                    "total_surviving_transcripts": diag_summary.get(
                        "total_surviving_transcripts", ""
                    ),
                    "truth_multi_exon_transcripts": loss_accounting.get(
                        "truth_multi_exon_transcripts", ""
                    ),
                    "recovered_truth_transcripts": loss_accounting.get(
                        "recovered_truth_transcripts", ""
                    ),
                    "fragmented_locus_truth_transcripts": loss_accounting.get(
                        "fragmented_locus_truth_transcripts", ""
                    ),
                    "graph_miss_truth_transcripts": loss_accounting.get(
                        "graph_miss_truth_transcripts", ""
                    ),
                    "decomposition_miss_truth_transcripts": loss_accounting.get(
                        "decomposition_miss_truth_transcripts", ""
                    ),
                    "filter_miss_truth_transcripts": loss_accounting.get(
                        "filter_miss_truth_transcripts", ""
                    ),
                    "chain_miss_truth_transcripts": loss_accounting.get(
                        "chain_miss_truth_transcripts", ""
                    ),
                }
            )


def _print_console_summary(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'variant':<22} {'status':<8} {'tx_sn':>8} {'tx_pr':>8} "
        f"{'intron_sn':>10} {'runtime_s':>10} {'peak_mb':>9} {'survivors':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        metrics = row.get("metrics", {})
        diag_summary = (row.get("diagnostics") or {}).get("summary", {})
        print(
            f"{row['variant']:<22} "
            f"{row['status']:<8} "
            f"{metrics.get('transcript_sensitivity', 0.0):>8.4f} "
            f"{metrics.get('transcript_precision', 0.0):>8.4f} "
            f"{metrics.get('intron_sensitivity', 0.0):>10.4f} "
            f"{float(row['runtime_seconds']):>10.2f} "
            f"{float(row['peak_memory_mb']):>9.1f} "
            f"{int(diag_summary.get('total_surviving_transcripts', 0)):>10d}"
        )


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    bam_path = _resolve_existing_input(args.bam, "simulated.bam")
    reference_path = _resolve_existing_input(args.reference, "reference.fa")
    truth_gtf = _resolve_existing_input(args.truth_gtf, "truth.gtf")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    motif_compatibility = _assess_truth_motif_compatibility(reference_path, truth_gtf)
    if motif_compatibility["canonical_truth_fraction"] < 0.25:
        logger.warning(
            "Truth/reference motif compatibility is low: %d/%d truth introns canonical. "
            "Include no-motif-validation variants for synthetic comparisons.",
            motif_compatibility["canonical_truth_introns"],
            motif_compatibility["truth_unique_introns"],
        )

    evaluator = BenchmarkRunner(
        BenchmarkConfig(
            output_dir=str(output_dir),
            run_stringtie=False,
            run_scallop=False,
            threads=args.threads,
        )
    )
    if len(args.variants) != len(dict.fromkeys(args.variants)):
        raise ValueError("Duplicate variant names are not allowed in one comparison run.")
    rows: list[dict[str, Any]] = []

    for variant in _variant_specs(args.variants, args.shadow_decomposer):
        variant_dir = output_dir / variant.name
        shutil.rmtree(variant_dir, ignore_errors=True)
        diagnostics_dir = variant_dir / "diagnostics"
        logs_dir = variant_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        gtf_path = variant_dir / "assembled.gtf"

        effective_builder_profile = (
            variant.builder_profile
            if variant.builder_profile != "default"
            else args.builder_profile
        )
        command = _build_assemble_command(
            bam_path,
            reference_path,
            gtf_path,
            diagnostics_dir,
            variant,
            threads=args.threads,
            backend=args.backend,
            builder_profile=effective_builder_profile,
            min_coverage=args.min_coverage,
            min_score=args.min_score,
            min_junction_support=args.min_junction_support,
            min_phasing_support=args.min_phasing_support,
        )
        logger.info("Running %s: %s", variant.name, " ".join(command))
        result = _run_command(command, cwd=Path.cwd(), timeout=args.timeout)

        stdout_log = logs_dir / "stdout.log"
        stderr_log = logs_dir / "stderr.log"
        stdout_log.write_text(result.stdout, encoding="utf-8")
        stderr_log.write_text(result.stderr, encoding="utf-8")

        row = _build_variant_row(
            variant=variant,
            builder_profile=effective_builder_profile,
            gtf_path=gtf_path,
            diagnostics_dir=diagnostics_dir,
            logs_dir=logs_dir,
            command=command,
            command_result=result,
            evaluator=evaluator,
            truth_gtf=truth_gtf,
        )
        rows.append(row)

        if result.returncode != 0 and args.fail_fast:
            raise RuntimeError(
                f"Variant {variant.name} failed with exit code {result.returncode}. "
                f"See {stderr_log}."
            )

    baseline_variant = _resolve_baseline_variant(rows)
    comparison = {
        "dataset": {
            "bam_path": str(bam_path),
            "reference_path": str(reference_path),
            "truth_gtf": str(truth_gtf),
            "motif_compatibility": motif_compatibility,
        },
        "config": {
            "output_dir": str(output_dir),
            "variants": args.variants,
            "shadow_decomposer": args.shadow_decomposer,
            "backend": args.backend,
            "threads": args.threads,
            "builder_profile": args.builder_profile,
            "min_coverage": args.min_coverage,
            "min_score": args.min_score,
            "min_junction_support": args.min_junction_support,
            "min_phasing_support": args.min_phasing_support,
            "timeout": args.timeout,
            "fail_fast": args.fail_fast,
        },
        "baseline_variant": baseline_variant,
        "variants": rows,
        "delta_vs_baseline": _build_delta_summary(rows, baseline_variant=baseline_variant),
        "delta_vs_legacy": _build_delta_summary(rows, baseline_variant="legacy"),
    }

    json_path = output_dir / "comparison.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, sort_keys=True)

    _write_summary_tsv(rows, output_dir / "summary.tsv")
    _print_console_summary(rows)
    print(f"\nWrote comparison JSON to {json_path}")

    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run RapidSplice decomposer comparisons on the existing synthetic BAM "
            "and capture diagnostics-rich outputs."
        )
    )
    parser.add_argument(
        "--bam",
        help="Input BAM path. Defaults to benchmark_results/synthetic_data/simulated.bam.",
    )
    parser.add_argument(
        "--reference",
        help="Reference FASTA path. Defaults to benchmark_results/synthetic_data/reference.fa.",
    )
    parser.add_argument(
        "--truth-gtf",
        help="Truth GTF path. Defaults to benchmark_results/synthetic_data/truth.gtf.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where per-variant outputs and comparison summaries are written.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "legacy",
            "iterative_v2",
            "iterative_v2_relaxed",
            "legacy_no_motif_validation",
            "iterative_v2_no_motif_validation",
            "iterative_v2_relaxed_no_motif_validation",
        ],
        default=[
            "legacy_no_motif_validation",
            "iterative_v2_no_motif_validation",
        ],
        help=(
            "Variant set to run. Defaults target the synthetic recovery matrix "
            "with motif validation disabled, because the bundled synthetic "
            "reference is mostly non-canonical."
        ),
    )
    parser.add_argument(
        "--shadow-decomposer",
        choices=["legacy", "iterative_v2"],
        default=None,
        help="Optional shadow decomposer recorded in diagnostics for non-matching variants.",
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "gpu"],
        default="cpu",
        help="RapidSplice backend to use (default: cpu).",
    )
    parser.add_argument(
        "--builder-profile",
        choices=["default", "conservative_correctness", "aggressive_recall"],
        default="default",
        help=(
            "Builder pruning profile applied to variants unless the variant "
            "pins its own profile."
        ),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Thread count passed to RapidSplice (default: 1).",
    )
    parser.add_argument(
        "-c",
        "--min-coverage",
        type=float,
        default=1.0,
        help="Assembler minimum coverage threshold (default: 1.0).",
    )
    parser.add_argument(
        "-s",
        "--min-score",
        type=float,
        default=0.1,
        help="Assembler minimum transcript score threshold (default: 0.1).",
    )
    parser.add_argument(
        "-j",
        "--min-junction-support",
        type=int,
        default=3,
        help="Minimum junction support passed to assemble (default: 3).",
    )
    parser.add_argument(
        "--min-phasing-support",
        type=int,
        default=1,
        help="Minimum phasing support passed to assemble (default: 1).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-variant timeout in seconds (default: 3600).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when any variant returns a non-zero exit code.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_comparison(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
