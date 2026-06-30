#!/usr/bin/env python3
"""Benchmark explicit junction-pair RT-PCR targets with BRAID delta PSI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import zlib
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, ".")

from braid.target.psi_bootstrap import (
    DEFAULT_MIN_MAPQ,
    SpliceEvent,
    bootstrap_psi,
    extract_event_evidence_from_bam,
    merge_event_evidence,
    sample_psi_posterior,
)

SUPPORT_BINS: tuple[tuple[int, int | None, str], ...] = (
    (0, 19, "<20"),
    (20, 49, "20-49"),
    (50, 99, "50-99"),
    (100, 249, "100-249"),
    (250, None, "250+"),
)


@dataclass(frozen=True)
class JunctionPairTarget:
    """One RT-PCR target defined by one inclusion/exclusion junction pair."""

    gene: str
    chrom: str
    strand: str
    inclusion_junction: tuple[int, int]
    exclusion_junction: tuple[int, int]
    delta_psi_rtpcr: float

    @property
    def event_type(self) -> str:
        inc_start, inc_end = self.inclusion_junction
        exc_start, exc_end = self.exclusion_junction
        if inc_start == exc_start:
            return "A3SS" if self.strand == "+" else "A5SS"
        if inc_end == exc_end:
            return "A5SS" if self.strand == "+" else "A3SS"
        raise ValueError(
            f"{self.gene} junctions do not share one splice site: "
            f"{self.inclusion_junction} vs {self.exclusion_junction}",
        )

    @property
    def event_id(self) -> str:
        inc = f"{self.inclusion_junction[0]}-{self.inclusion_junction[1]}"
        exc = f"{self.exclusion_junction[0]}-{self.exclusion_junction[1]}"
        return f"{self.event_type}:{self.chrom}:{inc}|{exc}"

    def to_event(self) -> SpliceEvent:
        coords = self.inclusion_junction + self.exclusion_junction
        return SpliceEvent(
            event_id=self.event_id,
            event_type=self.event_type,
            gene=self.gene,
            chrom=self.chrom,
            event_start=min(coords),
            event_end=max(coords),
            inclusion_junctions=(self.inclusion_junction,),
            exclusion_junctions=(self.exclusion_junction,),
            proposal_source="external",
            metadata={"n_candidate_junctions": 2},
        )


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int((base_seed + zlib.crc32(payload)) % (2**31 - 1))


def _classify_bin(value: int) -> str:
    for lower, upper, label in SUPPORT_BINS:
        if value < lower:
            continue
        if upper is None or value <= upper:
            return label
    return SUPPORT_BINS[-1][2]


def _load_bam_group_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        content = handle.read().strip()
    if not content:
        return []
    parts = [part.strip() for part in content.replace("\n", ",").split(",")]
    return [part for part in parts if part]


def _parse_junction(value: str) -> tuple[int, int]:
    left, right = value.split("-")
    # SUPPA2/GSE54651 tables store splice junction ends in a 1-based inclusive
    # convention. BAM junction extraction here uses 0-based, end-exclusive
    # coordinates, so normalize the right boundary down by one.
    return int(left), int(right) - 1


def _load_targets(path: str) -> list[JunctionPairTarget]:
    targets: list[JunctionPairTarget] = []
    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            targets.append(
                JunctionPairTarget(
                    gene=row["gene"],
                    chrom=row["chrom"],
                    strand=row["strand"],
                    inclusion_junction=_parse_junction(row["inclusion_junction"]),
                    exclusion_junction=_parse_junction(row["exclusion_junction"]),
                    delta_psi_rtpcr=float(row["delta_psi_rtpcr"]),
                ),
            )
    return targets


def _condition_metrics(
    *,
    inclusion_count: int,
    exclusion_count: int,
    event_type: str,
    n_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | int | None]:
    psi, ci_low, ci_high, cv = bootstrap_psi(
        inclusion_count,
        exclusion_count,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        seed=seed,
        event_type=event_type,
    )
    return {
        "psi": psi,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "cv": cv if math.isfinite(cv) else None,
        "inc": inclusion_count,
        "exc": exclusion_count,
        "support_total": inclusion_count + exclusion_count,
    }


def _evaluate_target(
    target: JunctionPairTarget,
    *,
    sample_1_bams: list[str],
    sample_2_bams: list[str],
    n_replicates: int,
    confidence_level: float,
    min_mapq: int,
    seed: int,
    effect_cutoff: float,
    supported_probability_threshold: float,
    min_total_support: int,
    high_confidence_min_total_support: int,
) -> dict:
    event = target.to_event()
    sample_1_evidence = [
        extract_event_evidence_from_bam(bam_path, event, min_mapq=min_mapq)
        for bam_path in sample_1_bams
    ]
    sample_2_evidence = [
        extract_event_evidence_from_bam(bam_path, event, min_mapq=min_mapq)
        for bam_path in sample_2_bams
    ]
    merged_1 = merge_event_evidence(sample_1_evidence)
    merged_2 = merge_event_evidence(sample_2_evidence)

    total_support = (
        merged_1.inclusion_count
        + merged_1.exclusion_count
        + merged_2.inclusion_count
        + merged_2.exclusion_count
    )
    measured = total_support > 0

    metrics_1 = _condition_metrics(
        inclusion_count=merged_1.inclusion_count,
        exclusion_count=merged_1.exclusion_count,
        event_type=event.event_type,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        seed=_stable_seed(seed, target.event_id, "sample_1"),
    )
    metrics_2 = _condition_metrics(
        inclusion_count=merged_2.inclusion_count,
        exclusion_count=merged_2.exclusion_count,
        event_type=event.event_type,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        seed=_stable_seed(seed, target.event_id, "sample_2"),
    )

    samples_1 = sample_psi_posterior(
        merged_1.inclusion_count,
        merged_1.exclusion_count,
        n_replicates=n_replicates,
        seed=_stable_seed(seed, target.event_id, "posterior_sample_1"),
        event_type=event.event_type,
    )
    samples_2 = sample_psi_posterior(
        merged_2.inclusion_count,
        merged_2.exclusion_count,
        n_replicates=n_replicates,
        seed=_stable_seed(seed, target.event_id, "posterior_sample_2"),
        event_type=event.event_type,
    )
    delta_samples = samples_2 - samples_1
    alpha = 1.0 - confidence_level
    braid_dpsi = float(np.mean(delta_samples))
    braid_dpsi_ci_low = float(np.percentile(delta_samples, 100 * alpha / 2))
    braid_dpsi_ci_high = float(np.percentile(delta_samples, 100 * (1 - alpha / 2)))
    braid_dpsi_prob_abs_ge_cutoff = float(
        np.mean(np.abs(delta_samples) >= effect_cutoff),
    )
    braid_dpsi_excludes_zero = braid_dpsi_ci_low > 0 or braid_dpsi_ci_high < 0
    supported = (
        measured
        and total_support >= min_total_support
        and braid_dpsi_prob_abs_ge_cutoff >= supported_probability_threshold
    )
    high_confidence = (
        supported
        and braid_dpsi_excludes_zero
        and total_support >= high_confidence_min_total_support
    )

    rtpcr_sign = (
        0 if abs(target.delta_psi_rtpcr) < 1e-9
        else (1 if target.delta_psi_rtpcr > 0 else -1)
    )
    braid_sign = 0 if abs(braid_dpsi) < 1e-9 else (1 if braid_dpsi > 0 else -1)
    direction_match = (
        None
        if rtpcr_sign == 0 or braid_sign == 0
        else braid_sign == rtpcr_sign
    )

    return {
        "gene": target.gene,
        "chrom": target.chrom,
        "strand": target.strand,
        "event_id": target.event_id,
        "event_type": event.event_type,
        "inclusion_junction": f"{target.inclusion_junction[0]}-{target.inclusion_junction[1]}",
        "exclusion_junction": f"{target.exclusion_junction[0]}-{target.exclusion_junction[1]}",
        "delta_psi_rtpcr": target.delta_psi_rtpcr,
        "measured": measured,
        "support_total": total_support,
        "support_bin": _classify_bin(total_support),
        "braid_dpsi": braid_dpsi,
        "braid_dpsi_ci_low": braid_dpsi_ci_low,
        "braid_dpsi_ci_high": braid_dpsi_ci_high,
        "braid_dpsi_ci_width": braid_dpsi_ci_high - braid_dpsi_ci_low,
        "braid_dpsi_excludes_zero": braid_dpsi_excludes_zero,
        "braid_dpsi_prob_abs_ge_cutoff": braid_dpsi_prob_abs_ge_cutoff,
        "supported": supported,
        "high_confidence": high_confidence,
        "direction_match": direction_match,
        "sample_1": metrics_1,
        "sample_2": metrics_2,
        "sample_1_bam_count": len(sample_1_bams),
        "sample_2_bam_count": len(sample_2_bams),
    }


def _median_or_none(values: list[float]) -> float | None:
    return float(np.median(np.array(values, dtype=float))) if values else None


def _pearson_or_none(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    if np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(np.array(xs), np.array(ys))[0, 1])


def _summarize(rows: list[dict]) -> dict:
    measured = [row for row in rows if row["measured"]]
    supported = [row for row in measured if row["supported"]]
    high_conf = [row for row in measured if row["high_confidence"]]
    direction_rows = [
        row for row in measured if row["direction_match"] is not None
    ]
    supported_direction_rows = [
        row for row in supported if row["direction_match"] is not None
    ]
    braid_dpsi = [row["braid_dpsi"] for row in measured]
    rtpcr_dpsi = [row["delta_psi_rtpcr"] for row in measured]
    supported_braid = [row["braid_dpsi"] for row in supported]
    supported_rtpcr = [row["delta_psi_rtpcr"] for row in supported]
    return {
        "total_targets": len(rows),
        "measured_targets": len(measured),
        "supported_targets": len(supported),
        "high_confidence_targets": len(high_conf),
        "measured_rate": len(measured) / len(rows) if rows else None,
        "supported_rate": len(supported) / len(rows) if rows else None,
        "high_confidence_rate": len(high_conf) / len(rows) if rows else None,
        "median_support_total": _median_or_none([row["support_total"] for row in measured]),
        "median_abs_braid_dpsi": _median_or_none([abs(row["braid_dpsi"]) for row in measured]),
        "median_abs_rtpcr_dpsi": _median_or_none([abs(row["delta_psi_rtpcr"]) for row in measured]),
        "direction_match_rate": (
            float(np.mean([row["direction_match"] for row in direction_rows]))
            if direction_rows
            else None
        ),
        "supported_direction_match_rate": (
            float(np.mean([row["direction_match"] for row in supported_direction_rows]))
            if supported_direction_rows
            else None
        ),
        "pearson_r_measured": _pearson_or_none(braid_dpsi, rtpcr_dpsi),
        "pearson_r_supported": _pearson_or_none(supported_braid, supported_rtpcr),
    }


def _write_json(path: str, payload: dict) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_benchmark(
    *,
    targets_path: str,
    b1_path: str,
    b2_path: str,
    output_path: str,
    n_replicates: int,
    confidence_level: float,
    min_mapq: int,
    seed: int,
    effect_cutoff: float,
    supported_probability_threshold: float,
    min_total_support: int,
    high_confidence_min_total_support: int,
) -> dict:
    targets = _load_targets(targets_path)
    sample_1_bams = _load_bam_group_file(b1_path)
    sample_2_bams = _load_bam_group_file(b2_path)
    rows = [
        _evaluate_target(
            target,
            sample_1_bams=sample_1_bams,
            sample_2_bams=sample_2_bams,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            min_mapq=min_mapq,
            seed=seed,
            effect_cutoff=effect_cutoff,
            supported_probability_threshold=supported_probability_threshold,
            min_total_support=min_total_support,
            high_confidence_min_total_support=high_confidence_min_total_support,
        )
        for target in targets
    ]
    payload = {
        "metadata": {
            "targets_path": targets_path,
            "b1_path": b1_path,
            "b2_path": b2_path,
            "n_replicates": n_replicates,
            "confidence_level": confidence_level,
            "min_mapq": min_mapq,
            "seed": seed,
            "effect_cutoff": effect_cutoff,
            "supported_probability_threshold": supported_probability_threshold,
            "min_total_support": min_total_support,
            "high_confidence_min_total_support": high_confidence_min_total_support,
            "sample_1_bams": sample_1_bams,
            "sample_2_bams": sample_2_bams,
        },
        "summary": _summarize(rows),
        "rows": rows,
    }
    _write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--b1", required=True, help="Comma-separated BAM group file for sample 1.")
    parser.add_argument("--b2", required=True, help="Comma-separated BAM group file for sample 2.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-replicates", type=int, default=200)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--min-mapq", type=int, default=DEFAULT_MIN_MAPQ)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--effect-cutoff", type=float, default=0.1)
    parser.add_argument("--supported-probability-threshold", type=float, default=0.85)
    parser.add_argument("--min-total-support", type=int, default=1)
    parser.add_argument("--high-confidence-min-total-support", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_benchmark(
        targets_path=args.targets,
        b1_path=args.b1,
        b2_path=args.b2,
        output_path=args.output,
        n_replicates=args.n_replicates,
        confidence_level=args.confidence_level,
        min_mapq=args.min_mapq,
        seed=args.seed,
        effect_cutoff=args.effect_cutoff,
        supported_probability_threshold=args.supported_probability_threshold,
        min_total_support=args.min_total_support,
        high_confidence_min_total_support=args.high_confidence_min_total_support,
    )
    summary = payload["summary"]
    print("=" * 60)
    print("Junction Pair BRAID Benchmark")
    print("=" * 60)
    print(f"Measured targets:         {summary['measured_targets']}/{summary['total_targets']}")
    print(f"Supported targets:        {summary['supported_targets']}/{summary['total_targets']}")
    print(
        f"High-confidence targets:  "
        f"{summary['high_confidence_targets']}/{summary['total_targets']}"
    )
    if summary["direction_match_rate"] is not None:
        print(f"Direction match rate:     {summary['direction_match_rate']:.3f}")
    if summary["pearson_r_measured"] is not None:
        print(f"Measured Pearson r:       {summary['pearson_r_measured']:.3f}")
    if summary["pearson_r_supported"] is not None:
        print(f"Supported Pearson r:      {summary['pearson_r_supported']:.3f}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
