#!/usr/bin/env python3
"""Run the BRAID QKI exon-skipping benchmark.

The shipped QKI target tables are not mutually exclusive: most entries in
``failed_events.tsv`` also appear in ``validated_events.tsv``. This script
records that overlap explicitly and reports the failed table as a separate
cohort/rescue subset instead of treating it as a negative class.

BRAID v2 reports either a single-sample benchmark or a multi-sample view
with per-sample results plus a "detected in either sample" aggregate.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from statistics import median

from rapidsplice.target.psi_bootstrap import (
    DEFAULT_MIN_MAPQ,
    PSIResult,
    compute_psi_from_junctions,
)


@dataclass(frozen=True)
class QKITarget:
    """One exon-skipping target from the QKI benchmark tables."""

    gene: str
    chrom: str
    start: int
    end: int
    cohort: str

    @property
    def key(self) -> tuple[str, str, int, int]:
        """Return a stable target identifier."""
        return (self.gene, self.chrom, self.start, self.end)


def _resolve_target_table(
    qki_dir: str,
    filename: str,
    *,
    prefer_lifted: bool = True,
) -> tuple[str, str]:
    """Prefer lifted hg38 target tables when they are available."""
    stem, ext = os.path.splitext(filename)
    lifted_path = os.path.join(qki_dir, f"{stem}.hg38{ext}")
    if prefer_lifted and os.path.exists(lifted_path):
        return lifted_path, "hg38_lifted_from_hg19"
    return os.path.join(qki_dir, filename), "source_table"


def _load_target_tables(
    qki_dir: str,
    *,
    prefer_lifted: bool = True,
) -> tuple[list[QKITarget], list[QKITarget], dict]:
    """Load validated and failed QKI targets with resolved source metadata."""
    validated_path, validated_build = _resolve_target_table(
        qki_dir,
        "validated_events.tsv",
        prefer_lifted=prefer_lifted,
    )
    failed_path, failed_build = _resolve_target_table(
        qki_dir,
        "failed_events.tsv",
        prefer_lifted=prefer_lifted,
    )
    metadata = {
        "validated_targets_path": validated_path,
        "validated_targets_build": validated_build,
        "failed_targets_path": failed_path,
        "failed_targets_build": failed_build,
        "prefer_lifted_targets": prefer_lifted,
    }
    return (
        _load_targets(validated_path, "validated"),
        _load_targets(failed_path, "failed"),
        metadata,
    )


def _load_targets(path: str, cohort: str) -> list[QKITarget]:
    """Load QKI targets from a TSV file."""
    targets: list[QKITarget] = []
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene = row.get("gene")
            chrom = row.get("chrom")
            start = row.get("exon_start")
            end = row.get("exon_end")
            if not gene or not chrom or start is None or end is None:
                continue
            targets.append(QKITarget(
                gene=gene,
                chrom=chrom,
                start=int(start),
                end=int(end),
                cohort=cohort,
            ))
    return targets


def _group_targets(
    targets: list[QKITarget],
) -> dict[tuple[str, str], list[QKITarget]]:
    """Group targets by gene and chromosome."""
    grouped: dict[tuple[str, str], list[QKITarget]] = defaultdict(list)
    for target in targets:
        grouped[(target.gene, target.chrom)].append(target)
    return grouped


def _find_matching_se(
    target: QKITarget,
    psi_results: list[PSIResult],
    tolerance: int,
) -> tuple[PSIResult | None, int]:
    """Return the best matching SE event and the number of matches."""
    candidates = [
        result
        for result in psi_results
        if result.event_type == "SE"
        and result.event_start is not None
        and result.event_end is not None
        and abs(result.event_start - target.start) <= tolerance
        and abs(result.event_end - target.end) <= tolerance
    ]
    if not candidates:
        return None, 0

    def _sort_key(result: PSIResult) -> tuple[int, int, float, str]:
        coord_delta = (
            abs((result.event_start or 0) - target.start)
            + abs((result.event_end or 0) - target.end)
        )
        support = result.inclusion_count + result.exclusion_count
        return (coord_delta, -support, result.ci_width, result.event_id)

    return min(candidates, key=_sort_key), len(candidates)


def _target_result(
    target: QKITarget,
    matched: PSIResult | None,
    n_matching_candidates: int,
    n_se_in_region: int,
    overlap_with_validated: bool,
) -> dict:
    """Serialize one target result row."""
    result = {
        "gene": target.gene,
        "chrom": target.chrom,
        "target_start": target.start,
        "target_end": target.end,
        "cohort": target.cohort,
        "overlap_with_validated": overlap_with_validated,
        "n_matching_candidates": n_matching_candidates,
        "n_se_events_in_region": n_se_in_region,
        "matched": matched is not None,
    }
    if matched is None:
        return result

    coord_delta = (
        abs((matched.event_start or 0) - target.start)
        + abs((matched.event_end or 0) - target.end)
    )
    result.update({
        "event_id": matched.event_id,
        "event_start": matched.event_start,
        "event_end": matched.event_end,
        "coord_delta": coord_delta,
        "psi": matched.psi,
        "ci_low": matched.ci_low,
        "ci_high": matched.ci_high,
        "ci_width": matched.ci_width,
        "cv": matched.cv,
        "confident": matched.is_confident,
        "inc": matched.inclusion_count,
        "exc": matched.exclusion_count,
        "support_total": matched.inclusion_count + matched.exclusion_count,
        "proposal_source": matched.proposal_source,
        "n_candidate_junctions": matched.n_candidate_junctions,
        "n_supported_junctions": matched.n_supported_junctions,
        "evidence_breakdown": matched.evidence_breakdown,
    })
    return result


def _median_or_none(values: list[float]) -> float | None:
    """Return the median as a float, or None when empty."""
    return float(median(values)) if values else None


def _summarize(total_targets: int, rows: list[dict]) -> dict:
    """Summarize a target cohort."""
    matched_rows = [row for row in rows if row.get("matched")]
    confident_rows = [row for row in matched_rows if row.get("confident")]
    ci_widths = [row["ci_width"] for row in matched_rows]
    cvs = [row["cv"] for row in matched_rows if row.get("cv") is not None]
    supports = [row["support_total"] for row in matched_rows]

    return {
        "total_targets": total_targets,
        "matched_targets": len(matched_rows),
        "match_rate": (
            len(matched_rows) / total_targets if total_targets else None
        ),
        "confident_matches": len(confident_rows),
        "confident_rate_over_total": (
            len(confident_rows) / total_targets if total_targets else None
        ),
        "confident_rate_over_matches": (
            len(confident_rows) / len(matched_rows) if matched_rows else None
        ),
        "median_ci_width": _median_or_none(ci_widths),
        "median_cv": _median_or_none(cvs),
        "median_support_total": _median_or_none(supports),
    }


def _build_metadata(
    *,
    gtf_path: str,
    window: int,
    tolerance: int,
    n_replicates: int,
    confidence_level: float,
    min_mapq: int,
    seed: int,
    validated_targets: list[QKITarget],
    failed_targets: list[QKITarget],
    overlap_keys: set[tuple[str, str, int, int]],
    target_table_metadata: dict,
) -> dict:
    """Build shared benchmark metadata."""
    metadata = {
        "gtf_path": gtf_path,
        "window": window,
        "match_tolerance": tolerance,
        "n_replicates": n_replicates,
        "confidence_level": confidence_level,
        "min_mapq": min_mapq,
        "seed": seed,
        "validated_total": len(validated_targets),
        "failed_total": len(failed_targets),
        "label_overlap_count": len(overlap_keys),
        "label_overlap_targets": [
            {
                "gene": gene,
                "chrom": chrom,
                "start": start,
                "end": end,
            }
            for gene, chrom, start, end in sorted(overlap_keys)
        ],
        "note": (
            "failed_events.tsv overlaps validated_events.tsv; failed "
            "targets are reported as a separate cohort, not as negatives."
        ),
    }
    metadata.update(target_table_metadata)
    return metadata


def _finalize_sample_result(
    *,
    metadata: dict,
    bam_path: str,
    validated_rows: list[dict],
    failed_rows: list[dict],
) -> dict:
    """Build final result object for one sample."""
    overlap_rows = [row for row in failed_rows if row["overlap_with_validated"]]
    exclusive_failed_rows = [
        row for row in failed_rows if not row["overlap_with_validated"]
    ]
    exclusive_validated_rows = [
        row for row in validated_rows if not row["overlap_with_validated"]
    ]

    sample_metadata = dict(metadata)
    sample_metadata["bam_path"] = bam_path

    return {
        "metadata": sample_metadata,
        "validated_summary": _summarize(
            metadata["validated_total"],
            validated_rows,
        ),
        "failed_summary": _summarize(
            metadata["failed_total"],
            failed_rows,
        ),
        "overlap_failed_summary": _summarize(len(overlap_rows), overlap_rows),
        "exclusive_failed_summary": _summarize(
            len(exclusive_failed_rows),
            exclusive_failed_rows,
        ),
        "exclusive_validated_summary": _summarize(
            len(exclusive_validated_rows),
            exclusive_validated_rows,
        ),
        "validated": validated_rows,
        "failed": failed_rows,
    }


def _run_qki_single_sample(
    *,
    bam_path: str,
    qki_dir: str,
    gtf_path: str,
    window: int,
    tolerance: int,
    n_replicates: int,
    confidence_level: float,
    min_mapq: int,
    seed: int,
    prefer_lifted_targets: bool,
) -> dict:
    """Execute QKI benchmark logic for one BAM."""
    (
        validated_targets,
        failed_targets,
        target_table_metadata,
    ) = _load_target_tables(
        qki_dir,
        prefer_lifted=prefer_lifted_targets,
    )
    overlap_keys = {target.key for target in validated_targets} & {
        target.key for target in failed_targets
    }
    metadata = _build_metadata(
        gtf_path=gtf_path,
        window=window,
        tolerance=tolerance,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        min_mapq=min_mapq,
        seed=seed,
        validated_targets=validated_targets,
        failed_targets=failed_targets,
        overlap_keys=overlap_keys,
        target_table_metadata=target_table_metadata,
    )

    all_targets = validated_targets + failed_targets
    grouped_targets = _group_targets(all_targets)

    psi_by_gene: dict[tuple[str, str], list[PSIResult]] = {}
    for (gene, chrom), group in grouped_targets.items():
        region_start = max(0, min(target.start for target in group) - window)
        region_end = max(target.end for target in group) + window
        psi_by_gene[(gene, chrom)] = compute_psi_from_junctions(
            bam_path,
            chrom,
            region_start,
            region_end,
            gene=gene,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            min_mapq=min_mapq,
            seed=seed,
            annotation_gtf=gtf_path,
            event_source="hybrid",
            uncertainty_model="overdispersed",
        )

    validated_rows: list[dict] = []
    failed_rows: list[dict] = []
    for cohort_targets, rows in (
        (validated_targets, validated_rows),
        (failed_targets, failed_rows),
    ):
        for target in cohort_targets:
            psi_results = psi_by_gene.get((target.gene, target.chrom), [])
            matched, n_matching_candidates = _find_matching_se(
                target,
                psi_results,
                tolerance,
            )
            n_se_in_region = sum(
                1 for result in psi_results if result.event_type == "SE"
            )
            rows.append(_target_result(
                target=target,
                matched=matched,
                n_matching_candidates=n_matching_candidates,
                n_se_in_region=n_se_in_region,
                overlap_with_validated=target.key in overlap_keys,
            ))

    return _finalize_sample_result(
        metadata=metadata,
        bam_path=bam_path,
        validated_rows=validated_rows,
        failed_rows=failed_rows,
    )


def _merge_target_rows(sample_name_to_row: dict[str, dict]) -> dict:
    """Aggregate one target across samples using the best matched sample."""
    sample_names = sorted(sample_name_to_row)
    exemplar = sample_name_to_row[sample_names[0]]
    matched_rows = [
        (sample_name, row)
        for sample_name, row in sample_name_to_row.items()
        if row.get("matched")
    ]
    aggregate = {
        "gene": exemplar["gene"],
        "chrom": exemplar["chrom"],
        "target_start": exemplar["target_start"],
        "target_end": exemplar["target_end"],
        "cohort": exemplar["cohort"],
        "overlap_with_validated": exemplar["overlap_with_validated"],
        "matched": bool(matched_rows),
        "matched_samples": [sample for sample, _row in matched_rows],
        "n_samples_checked": len(sample_name_to_row),
        "n_matching_candidates_by_sample": {
            sample: row.get("n_matching_candidates", 0)
            for sample, row in sample_name_to_row.items()
        },
        "n_se_events_in_region_by_sample": {
            sample: row.get("n_se_events_in_region", 0)
            for sample, row in sample_name_to_row.items()
        },
    }
    if not matched_rows:
        return aggregate

    def _sort_key(item: tuple[str, dict]) -> tuple[int, int, float, str]:
        sample_name, row = item
        return (
            row.get("coord_delta", 10**9),
            -row.get("support_total", 0),
            row.get("ci_width", 10.0),
            sample_name,
        )

    best_sample, best_row = min(matched_rows, key=_sort_key)
    aggregate.update({
        "best_sample": best_sample,
        "confident": any(row.get("confident") for _, row in matched_rows),
        "confident_samples": [
            sample for sample, row in matched_rows if row.get("confident")
        ],
        "event_id": best_row["event_id"],
        "event_start": best_row["event_start"],
        "event_end": best_row["event_end"],
        "coord_delta": best_row["coord_delta"],
        "psi": best_row["psi"],
        "ci_low": best_row["ci_low"],
        "ci_high": best_row["ci_high"],
        "ci_width": best_row["ci_width"],
        "cv": best_row["cv"],
        "inc": best_row["inc"],
        "exc": best_row["exc"],
        "support_total": best_row["support_total"],
        "proposal_source": best_row.get("proposal_source"),
        "n_candidate_junctions": best_row.get("n_candidate_junctions", 0),
        "n_supported_junctions": best_row.get("n_supported_junctions", 0),
        "evidence_breakdown": best_row.get("evidence_breakdown", {}),
    })
    return aggregate


def _aggregate_across_samples(
    *,
    sample_results: dict[str, dict],
) -> dict:
    """Build an either-sample aggregate from per-sample benchmark results."""
    first_result = next(iter(sample_results.values()))
    metadata = dict(first_result["metadata"])
    metadata["sample_names"] = sorted(sample_results)
    metadata["aggregation"] = "detected_in_either_sample"

    validated_rows: list[dict] = []
    failed_rows: list[dict] = []
    for cohort_name in ("validated", "failed"):
        exemplar_rows = first_result[cohort_name]
        for idx, exemplar in enumerate(exemplar_rows):
            sample_name_to_row = {
                sample_name: sample_result[cohort_name][idx]
                for sample_name, sample_result in sample_results.items()
            }
            merged = _merge_target_rows(sample_name_to_row)
            if cohort_name == "validated":
                validated_rows.append(merged)
            else:
                failed_rows.append(merged)

    return _finalize_sample_result(
        metadata=metadata,
        bam_path="multiple",
        validated_rows=validated_rows,
        failed_rows=failed_rows,
    )


def _write_json(output_path: str, payload: dict) -> None:
    """Write JSON output, creating the parent directory if needed."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_qki_benchmark(
    *,
    bam_path: str,
    qki_dir: str,
    gtf_path: str,
    output_path: str,
    window: int,
    tolerance: int,
    n_replicates: int,
    confidence_level: float,
    min_mapq: int,
    seed: int,
    prefer_lifted_targets: bool = True,
) -> dict:
    """Execute the single-sample QKI benchmark and persist the JSON result."""
    results = _run_qki_single_sample(
        bam_path=bam_path,
        qki_dir=qki_dir,
        gtf_path=gtf_path,
        window=window,
        tolerance=tolerance,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        min_mapq=min_mapq,
        seed=seed,
        prefer_lifted_targets=prefer_lifted_targets,
    )
    _write_json(output_path, results)
    return results


def run_qki_multi_sample_benchmark(
    *,
    sample_bams: dict[str, str],
    qki_dir: str,
    gtf_path: str,
    output_path: str,
    window: int,
    tolerance: int,
    n_replicates: int,
    confidence_level: float,
    min_mapq: int,
    seed: int,
    prefer_lifted_targets: bool = True,
) -> dict:
    """Execute the QKI benchmark for multiple samples plus an aggregate."""
    sample_results: dict[str, dict] = {}
    for sample_name, bam_path in sample_bams.items():
        sample_results[sample_name] = _run_qki_single_sample(
            bam_path=bam_path,
            qki_dir=qki_dir,
            gtf_path=gtf_path,
            window=window,
            tolerance=tolerance,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            min_mapq=min_mapq,
            seed=seed,
            prefer_lifted_targets=prefer_lifted_targets,
        )
        sample_results[sample_name]["metadata"]["sample_name"] = sample_name

    aggregate = _aggregate_across_samples(sample_results=sample_results)
    payload = {
        "metadata": {
            "sample_names": sorted(sample_bams),
            "gtf_path": gtf_path,
            "window": window,
            "match_tolerance": tolerance,
            "n_replicates": n_replicates,
            "confidence_level": confidence_level,
            "min_mapq": min_mapq,
            "seed": seed,
        },
        "samples": sample_results,
        "detected_in_either_sample": aggregate,
    }
    _write_json(output_path, payload)
    return payload


def _print_single_summary(results: dict, *, label: str | None = None) -> None:
    """Emit a concise console summary for one benchmark result block."""
    metadata = results["metadata"]
    validated = results["validated_summary"]
    failed = results["failed_summary"]
    overlap = results["overlap_failed_summary"]
    exclusive_failed = results["exclusive_failed_summary"]

    if label:
        print(f"[{label}]")
    print(f"Validated targets:      {metadata['validated_total']}")
    print(f"Failed targets:         {metadata['failed_total']}")
    print(f"Label overlap:          {metadata['label_overlap_count']}")
    print("Note: failed targets are not treated as negatives when overlap exists.")
    print("")
    print(
        "Validated matched:      "
        f"{validated['matched_targets']}/{validated['total_targets']}"
    )
    print(
        "Validated confident:    "
        f"{validated['confident_matches']}/{validated['total_targets']}"
    )
    print(
        "Failed matched:         "
        f"{failed['matched_targets']}/{failed['total_targets']}"
    )
    print(
        "Failed confident:       "
        f"{failed['confident_matches']}/{failed['total_targets']}"
    )
    print(
        "Overlap failed matched: "
        f"{overlap['matched_targets']}/{overlap['total_targets']}"
    )
    print(
        "Exclusive failed:       "
        f"{exclusive_failed['matched_targets']}/"
        f"{exclusive_failed['total_targets']}"
    )
    if validated["median_ci_width"] is not None:
        print(
            "Validated median CI:    "
            f"{validated['median_ci_width']:.3f}"
        )
    if failed["median_ci_width"] is not None:
        print(
            "Failed median CI:       "
            f"{failed['median_ci_width']:.3f}"
        )


def _print_summary(results: dict) -> None:
    """Emit console summary for single-sample or multi-sample outputs."""
    print("=" * 60)
    print("  QKI BRAID Benchmark")
    print("=" * 60)
    if "samples" not in results:
        _print_single_summary(results)
        return

    for sample_name in sorted(results["samples"]):
        _print_single_summary(results["samples"][sample_name], label=sample_name)
        print("")

    print("[detected_in_either_sample]")
    _print_single_summary(results["detected_in_either_sample"])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bam",
        help="Indexed BAM used for legacy single-sample BRAID PSI estimation.",
    )
    parser.add_argument(
        "--sample-bam",
        action="append",
        default=[],
        help="Repeated SAMPLE=PATH entries for multi-sample mode.",
    )
    parser.add_argument(
        "--qki-dir",
        default="real_benchmark/rtpcr_benchmark/qki",
        help="Directory containing validated_events.tsv and failed_events.tsv.",
    )
    parser.add_argument(
        "--gtf",
        default="real_benchmark/annotation/gencode.v38.nochr.gtf",
        help="Annotation GTF used for reporting metadata.",
    )
    parser.add_argument(
        "--output",
        default="real_benchmark/rtpcr_benchmark/qki/braid_benchmark_results.json",
        help="Path to the JSON output file.",
    )
    parser.add_argument("--window", type=int, default=50000)
    parser.add_argument("--tolerance", type=int, default=10)
    parser.add_argument("--n-replicates", type=int, default=200)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-mapq", type=int, default=DEFAULT_MIN_MAPQ)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-targets",
        action="store_true",
        help="Ignore validated_events.hg38.tsv / failed_events.hg38.tsv and use raw source tables.",
    )
    return parser.parse_args()


def _parse_sample_bams(sample_bam_args: list[str]) -> dict[str, str]:
    """Parse repeated SAMPLE=PATH arguments."""
    sample_bams: dict[str, str] = {}
    for item in sample_bam_args:
        if "=" not in item:
            raise ValueError(
                f"Invalid --sample-bam value {item!r}; expected SAMPLE=PATH",
            )
        sample_name, bam_path = item.split("=", 1)
        if not sample_name or not bam_path:
            raise ValueError(
                f"Invalid --sample-bam value {item!r}; expected SAMPLE=PATH",
            )
        sample_bams[sample_name] = bam_path
    return sample_bams


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    sample_bams = _parse_sample_bams(args.sample_bam)

    if sample_bams:
        results = run_qki_multi_sample_benchmark(
            sample_bams=sample_bams,
            qki_dir=args.qki_dir,
            gtf_path=args.gtf,
            output_path=args.output,
            window=args.window,
            tolerance=args.tolerance,
            n_replicates=args.n_replicates,
            confidence_level=args.confidence_level,
            min_mapq=args.min_mapq,
            seed=args.seed,
            prefer_lifted_targets=not args.raw_targets,
        )
    elif args.bam:
        results = run_qki_benchmark(
            bam_path=args.bam,
            qki_dir=args.qki_dir,
            gtf_path=args.gtf,
            output_path=args.output,
            window=args.window,
            tolerance=args.tolerance,
            n_replicates=args.n_replicates,
            confidence_level=args.confidence_level,
            min_mapq=args.min_mapq,
            seed=args.seed,
            prefer_lifted_targets=not args.raw_targets,
        )
    else:
        raise SystemExit("Provide --bam or at least one --sample-bam SAMPLE=PATH")

    _print_summary(results)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
