#!/usr/bin/env python3
"""Benchmark QKI RT-PCR targets with rMATS event detection plus BRAID CI."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zlib
from collections import Counter, defaultdict
from statistics import median

import numpy as np

sys.path.insert(0, ".")

from benchmarks.qki_benchmark import QKITarget, _load_target_tables
from rapidsplice.target.psi_bootstrap import (
    CONFIDENT_CI_WIDTH_THRESHOLD,
    CONFIDENT_CV_THRESHOLD,
    DEFAULT_MIN_MAPQ,
    bootstrap_psi,
    build_se_splice_event,
    extract_event_evidence_from_bam,
    merge_event_evidence,
    sample_psi_posterior,
)
from rapidsplice.target.rmats_bootstrap import RmatsEvent, parse_rmats_output


SUPPORT_BINS: tuple[tuple[int, int | None, str], ...] = (
    (20, 49, "20-49"),
    (50, 99, "50-99"),
    (100, 249, "100-249"),
    (250, None, "250+"),
)
LENGTH_BINS: tuple[tuple[int, int | None, str], ...] = (
    (0, 99, "<100"),
    (100, 199, "100-199"),
    (200, 399, "200-399"),
    (400, None, "400+"),
)
NULL_CALIBRATION_SPARSE_BIN_MIN_EVENTS = 25
BODY_RESCUE_WEIGHT = 0.25
LOW_SUPPORT_SHRINK_THRESHOLD = 40.0
LOW_SUPPORT_SHRINK_MAX = 0.10
HIGH_SUPPORT_CONTROL_BINS = {"100-249", "250+"}
CASEBOOK_TOP_NEAR_MISS = 8
CASEBOOK_TOP_NULL = 8


def _normalize_chrom(chrom: str) -> str:
    """Normalize chromosome labels for mixed chr/no-chr inputs."""
    return chrom[3:] if chrom.startswith("chr") else chrom


def _overlap_bp(
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> int:
    """Return overlap in base pairs between two half-open intervals."""
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _classify_bin(
    value: int,
    bins: tuple[tuple[int, int | None, str], ...],
) -> str:
    """Assign one non-negative integer to a configured bin label."""
    for lower, upper, label in bins:
        if value < lower:
            continue
        if upper is None or value <= upper:
            return label
    return bins[-1][2]


def _median_or_none(values: list[float]) -> float | None:
    """Return a median value or None for empty lists."""
    return float(median(values)) if values else None


def _metric_quantiles(values: list[float]) -> dict[str, float]:
    """Return a compact quantile block for one numeric series."""
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def _stable_seed(base_seed: int, *parts: object) -> int:
    """Return a deterministic derived seed."""
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int((base_seed + zlib.crc32(payload)) % (2**31 - 1))


def _load_rmats_templates(
    rmats_dir: str,
    *,
    event_type: str = "SE",
) -> dict[tuple[str, str], list[tuple[int, int]]]:
    """Load annotation-derived rMATS event templates for target normalization."""
    path = os.path.join(rmats_dir, f"fromGTF.{event_type}.txt")
    templates: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    if not os.path.exists(path):
        return templates

    with open(path, encoding="utf-8") as handle:
        header = handle.readline().strip().split("\t")
        cols = {name: idx for idx, name in enumerate(header)}
        for line in handle:
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                continue
            gene = fields[cols.get("geneSymbol", 2)].strip('"')
            chrom = _normalize_chrom(fields[cols.get("chr", 3)])
            exon_start = int(fields[cols.get("exonStart_0base", 5)])
            exon_end = int(fields[cols.get("exonEnd", 6)])
            templates[(gene, chrom)].append((exon_start, exon_end))
    return templates


def _normalize_target_coords(
    target: QKITarget,
    templates: dict[tuple[str, str], list[tuple[int, int]]],
    *,
    max_delta: int,
) -> tuple[int, int, bool, int | None]:
    """Project one RT-PCR target onto the nearest rMATS annotation template."""
    best: tuple[int, int, int, int] | None = None
    for exon_start, exon_end in templates.get(
        (target.gene, _normalize_chrom(target.chrom)),
        [],
    ):
        delta = abs(exon_start - target.start) + abs(exon_end - target.end)
        overlap = _overlap_bp(target.start, target.end, exon_start, exon_end)
        candidate = (delta, -overlap, exon_start, exon_end)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        return target.start, target.end, False, None

    delta, _neg_overlap, exon_start, exon_end = best
    if delta <= max_delta:
        return exon_start, exon_end, delta > 0, delta
    return target.start, target.end, False, delta


def _event_total_support(event: RmatsEvent) -> int:
    """Return one event's total short-read support from rMATS count tables."""
    return (
        event.sample_1_inc_count
        + event.sample_1_exc_count
        + event.sample_2_inc_count
        + event.sample_2_exc_count
    )


def _event_matches(
    target: QKITarget,
    event: RmatsEvent,
    *,
    target_start: int,
    target_end: int,
    tolerance: int,
    min_overlap_fraction: float,
) -> tuple[bool, int, float, int]:
    """Return match status plus overlap diagnostics for one target/event pair."""
    if _normalize_chrom(target.chrom) != _normalize_chrom(event.chrom):
        return False, 0, 0.0, 10**9
    if event.gene and event.gene != target.gene:
        return False, 0, 0.0, 10**9

    start_delta = abs(event.exon_start - target_start)
    end_delta = abs(event.exon_end - target_end)
    coord_delta = start_delta + end_delta
    overlap = _overlap_bp(target_start, target_end, event.exon_start, event.exon_end)
    target_len = max(1, target_end - target_start)
    event_len = max(1, event.exon_end - event.exon_start)
    overlap_fraction = overlap / min(target_len, event_len)
    coord_match = start_delta <= tolerance and end_delta <= tolerance
    overlap_match = overlap_fraction >= min_overlap_fraction
    return coord_match or overlap_match, overlap, overlap_fraction, coord_delta


def _find_matching_event(
    target: QKITarget,
    events: list[RmatsEvent],
    *,
    target_start: int,
    target_end: int,
    tolerance: int,
    min_overlap_fraction: float,
) -> tuple[RmatsEvent | None, int, int, float, int]:
    """Select the best matching rMATS event for one RT-PCR target."""
    candidates: list[tuple[RmatsEvent, int, float, int]] = []
    for event in events:
        matched, overlap_bp, overlap_fraction, coord_delta = _event_matches(
            target,
            event,
            target_start=target_start,
            target_end=target_end,
            tolerance=tolerance,
            min_overlap_fraction=min_overlap_fraction,
        )
        if matched:
            candidates.append((event, overlap_bp, overlap_fraction, coord_delta))

    if not candidates:
        return None, 0, 0, 0.0, 10**9

    def _sort_key(item: tuple[RmatsEvent, int, float, int]) -> tuple[int, int, float, float, str]:
        event, overlap_bp, overlap_fraction, coord_delta = item
        fdr = event.rmats_fdr if math.isfinite(event.rmats_fdr) else 1.0
        return (
            coord_delta,
            -overlap_bp,
            -overlap_fraction,
            -float(_event_total_support(event)),
            fdr,
            event.event_id,
        )

    best_event, best_overlap_bp, best_overlap_fraction, best_coord_delta = min(
        candidates,
        key=_sort_key,
    )
    return (
        best_event,
        len(candidates),
        best_overlap_bp,
        best_overlap_fraction,
        best_coord_delta,
    )


def _confidence_flag(ci_width: float, cv: float) -> bool:
    """Apply the BRAID conservative confidence rule."""
    return (
        ci_width < CONFIDENT_CI_WIDTH_THRESHOLD
        and math.isfinite(cv)
        and cv <= CONFIDENT_CV_THRESHOLD
    )


def _build_condition_metrics(
    inclusion_count: int,
    exclusion_count: int,
    *,
    event_type: str,
    n_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float | bool | int]:
    """Return BRAID PSI metrics for one condition from counts."""
    psi, ci_low, ci_high, cv = bootstrap_psi(
        inclusion_count,
        exclusion_count,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        seed=seed,
        event_type=event_type,
    )
    ci_width = ci_high - ci_low
    return {
        "psi": psi,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_width,
        "cv": cv,
        "is_confident": _confidence_flag(ci_width, cv),
        "inclusion_count": inclusion_count,
        "exclusion_count": exclusion_count,
        "support_total": inclusion_count + exclusion_count,
    }


def _build_dpsi_summary_from_counts(
    inclusion_a: int,
    exclusion_a: int,
    inclusion_b: int,
    exclusion_b: int,
    *,
    event_type: str,
    n_replicates: int,
    confidence_level: float,
    effect_cutoff: float,
    seed: int,
) -> dict[str, float | bool]:
    """Estimate one differential PSI posterior from two count pairs."""
    alpha = 1.0 - confidence_level
    ctrl_samples = sample_psi_posterior(
        inclusion_a,
        exclusion_a,
        n_replicates=n_replicates,
        seed=seed,
        event_type=event_type,
    )
    kd_samples = sample_psi_posterior(
        inclusion_b,
        exclusion_b,
        n_replicates=n_replicates,
        seed=seed + 1,
        event_type=event_type,
    )
    dpsi_samples = ctrl_samples - kd_samples
    ci_low = float(np.percentile(dpsi_samples, 100 * alpha / 2))
    ci_high = float(np.percentile(dpsi_samples, 100 * (1 - alpha / 2)))
    return {
        "posterior_mean": float(np.mean(dpsi_samples)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "excludes_zero": ci_low > 0 or ci_high < 0,
        "prob_abs_ge_cutoff": float(np.mean(np.abs(dpsi_samples) >= effect_cutoff)),
    }


def _build_count_table_metrics_by_event(
    events: list[RmatsEvent],
    *,
    n_replicates: int,
    confidence_level: float,
    effect_cutoff: float,
    seed: int,
) -> dict[str, dict[str, object]]:
    """Precompute count-table BRAID metrics for each rMATS event."""
    payload: dict[str, dict[str, object]] = {}
    for event in events:
        event_seed = _stable_seed(seed, "count_table", event.event_id)
        ctrl = _build_condition_metrics(
            event.sample_1_inc_count,
            event.sample_1_exc_count,
            event_type=event.event_type,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            seed=event_seed,
        )
        kd = _build_condition_metrics(
            event.sample_2_inc_count,
            event.sample_2_exc_count,
            event_type=event.event_type,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            seed=event_seed + 11,
        )
        dpsi = _build_dpsi_summary_from_counts(
            event.sample_1_inc_count,
            event.sample_1_exc_count,
            event.sample_2_inc_count,
            event.sample_2_exc_count,
            event_type=event.event_type,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            effect_cutoff=effect_cutoff,
            seed=event_seed + 23,
        )
        payload[event.event_id] = {
            "source": "count_table",
            "ctrl": ctrl,
            "kd": kd,
            "dpsi": dpsi,
            "recount_available": False,
            "recount_unavailable_reason": "bam_recount_not_attempted",
        }
    return payload


def _extract_recount_channels(evidence_breakdown: dict[str, object]) -> dict[str, float]:
    """Normalize one evidence_breakdown map into channel counts."""
    left = float(evidence_breakdown.get("left_junction", 0) or 0)
    right = float(evidence_breakdown.get("right_junction", 0) or 0)
    skip = float(evidence_breakdown.get("skip_junction", 0) or 0)
    body = float(evidence_breakdown.get("body_count", 0) or 0)
    paired = min(left, right)
    body_capped = min(body, max(left, right)) if max(left, right) > 0 else 0.0
    body_effective = BODY_RESCUE_WEIGHT * body_capped
    effective_inclusion = 0.5 * left + 0.5 * right + body_effective
    effective_exclusion = skip
    return {
        "left_junction": left,
        "right_junction": right,
        "skip_junction": skip,
        "body_count": body,
        "paired_junction": paired,
        "body_capped": body_capped,
        "body_effective": body_effective,
        "effective_inclusion": effective_inclusion,
        "effective_exclusion": effective_exclusion,
        "effective_support_total": effective_inclusion + effective_exclusion,
    }


def _summarize_posterior_samples(
    samples: np.ndarray,
    *,
    effective_inclusion: float,
    effective_exclusion: float,
    confidence_level: float,
) -> dict[str, float | bool]:
    """Summarize one posterior sample array into BRAID-like metrics."""
    alpha = 1.0 - confidence_level
    psi = float(np.mean(samples)) if len(samples) else 0.0
    ci_low = float(np.percentile(samples, 100 * alpha / 2)) if len(samples) else 0.0
    ci_high = float(np.percentile(samples, 100 * (1 - alpha / 2))) if len(samples) else 0.0
    ci_width = ci_high - ci_low
    std = float(np.std(samples)) if len(samples) else 0.0
    cv = std / psi if psi > 0 else float("nan")
    return {
        "psi": psi,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_width,
        "cv": cv,
        "is_confident": _confidence_flag(ci_width, cv),
        "inclusion_count": effective_inclusion,
        "exclusion_count": effective_exclusion,
        "support_total": effective_inclusion + effective_exclusion,
    }


def _sample_channel_fused_replicate(
    channels: dict[str, float],
    *,
    n_replicates: int,
    confidence_level: float,
    seed: int,
    apply_single_sample_shrink: bool,
) -> dict[str, object]:
    """Sample one replicate-level fused PSI posterior from SE evidence channels."""
    channel_specs: list[tuple[str, float, np.ndarray]] = []
    skip = max(channels["skip_junction"], 0.0)

    left_weight = 0.5 * max(channels["left_junction"], 0.0)
    right_weight = 0.5 * max(channels["right_junction"], 0.0)
    body_weight = max(channels["body_effective"], 0.0)
    channel_inputs = (
        ("left", channels["left_junction"], left_weight),
        ("right", channels["right_junction"], right_weight),
        ("body", body_weight, body_weight),
    )
    for index, (label, alpha_count, weight) in enumerate(channel_inputs):
        if weight <= 0 or alpha_count <= 0:
            continue
        samples = sample_psi_posterior(
            alpha_count,
            skip,
            n_replicates=n_replicates,
            seed=seed + index,
            event_type="SE",
        )
        channel_specs.append((label, weight, samples))

    if not channel_specs:
        fused_samples = np.zeros(n_replicates, dtype=float)
    else:
        stacked = np.vstack([samples for _label, _weight, samples in channel_specs])
        weights = np.array([weight for _label, weight, _samples in channel_specs], dtype=float)
        fused_samples = np.average(stacked, axis=0, weights=weights)

    if apply_single_sample_shrink and channels["effective_support_total"] < LOW_SUPPORT_SHRINK_THRESHOLD:
        shrink = min(
            LOW_SUPPORT_SHRINK_MAX,
            (
                (LOW_SUPPORT_SHRINK_THRESHOLD - channels["effective_support_total"])
                / LOW_SUPPORT_SHRINK_THRESHOLD
            ) * LOW_SUPPORT_SHRINK_MAX,
        )
        fused_samples = (1.0 - shrink) * fused_samples + shrink * 0.5
    else:
        shrink = 0.0

    summary = _summarize_posterior_samples(
        fused_samples,
        effective_inclusion=channels["effective_inclusion"],
        effective_exclusion=channels["effective_exclusion"],
        confidence_level=confidence_level,
    )
    return {
        "samples": fused_samples,
        "summary": summary,
        "channels": channels,
        "channel_weights": {
            label: float(weight)
            for label, weight, _samples in channel_specs
        },
        "single_sample_shrink": shrink,
    }


def _build_condition_metrics_from_replicates(
    replicate_payloads: list[dict[str, object]],
    *,
    n_replicates: int,
    confidence_level: float,
) -> dict[str, object]:
    """Combine replicate-level channel-fused posteriors into one condition posterior."""
    if not replicate_payloads:
        zero_samples = np.zeros(n_replicates, dtype=float)
        summary = _summarize_posterior_samples(
            zero_samples,
            effective_inclusion=0.0,
            effective_exclusion=0.0,
            confidence_level=confidence_level,
        )
        return {
            "samples": zero_samples,
            "summary": summary,
            "replicates": [],
            "fused_channels": _extract_recount_channels({}),
        }

    condition_samples = np.mean(
        np.vstack([payload["samples"] for payload in replicate_payloads]),
        axis=0,
    )
    fused_channels = _extract_recount_channels({})
    for key in fused_channels:
        fused_channels[key] = sum(
            float(payload["channels"].get(key, 0.0))
            for payload in replicate_payloads
        )
    summary = _summarize_posterior_samples(
        condition_samples,
        effective_inclusion=fused_channels["effective_inclusion"],
        effective_exclusion=fused_channels["effective_exclusion"],
        confidence_level=confidence_level,
    )
    return {
        "samples": condition_samples,
        "summary": summary,
        "replicates": replicate_payloads,
        "fused_channels": fused_channels,
    }


def _build_dpsi_summary_from_samples(
    ctrl_samples: np.ndarray,
    kd_samples: np.ndarray,
    *,
    confidence_level: float,
    effect_cutoff: float,
) -> dict[str, float | bool]:
    """Estimate one differential posterior from two sampled condition posteriors."""
    alpha = 1.0 - confidence_level
    dpsi_samples = ctrl_samples - kd_samples
    ci_low = float(np.percentile(dpsi_samples, 100 * alpha / 2))
    ci_high = float(np.percentile(dpsi_samples, 100 * (1 - alpha / 2)))
    return {
        "posterior_mean": float(np.mean(dpsi_samples)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_high - ci_low,
        "excludes_zero": ci_low > 0 or ci_high < 0,
        "prob_abs_ge_cutoff": float(np.mean(np.abs(dpsi_samples) >= effect_cutoff)),
    }


def _resolve_path_list(raw_values: list[str], base_dir: str) -> list[str]:
    """Resolve raw BAM paths relative to CWD first, then the group-file directory."""
    resolved: list[str] = []
    for raw in raw_values:
        raw = raw.strip()
        if not raw:
            continue
        if os.path.isabs(raw) and os.path.exists(raw):
            resolved.append(raw)
            continue
        if os.path.exists(raw):
            resolved.append(raw)
            continue
        candidate = os.path.join(base_dir, raw)
        if os.path.exists(candidate):
            resolved.append(candidate)
    return resolved


def _load_bam_group_file(path: str) -> list[str]:
    """Load one rMATS BAM-group text file."""
    if not os.path.exists(path):
        return []
    base_dir = os.path.dirname(path)
    raw_values: list[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw_values.extend(value for value in line.split(",") if value)
    return _resolve_path_list(raw_values, base_dir)


def _resolve_recount_bam_groups(rmats_dir: str) -> dict[str, object]:
    """Resolve BAM paths from rMATS group files for recount-backed BRAID metrics."""
    ctrl_bams = _load_bam_group_file(os.path.join(rmats_dir, "b1.txt"))
    kd_bams = _load_bam_group_file(os.path.join(rmats_dir, "b2.txt"))
    available = bool(ctrl_bams and kd_bams)
    reason = None
    if not available:
        reason = "missing_or_empty_bam_group_files"
    return {
        "available": available,
        "reason": reason,
        "sample_1_bams": ctrl_bams,
        "sample_2_bams": kd_bams,
    }


def _build_rmats_se_splice_event(event: RmatsEvent):
    """Convert one rMATS SE event into a canonical BRAID SE event."""
    required = (
        event.upstream_ee,
        event.downstream_es,
        event.exon_start,
        event.exon_end,
    )
    if any(value in {None, 0} for value in required):
        return None
    return build_se_splice_event(
        event_id=event.event_id,
        gene=event.gene,
        chrom=event.chrom,
        exon_start=event.exon_start,
        exon_end=event.exon_end,
        upstream_ee=int(event.upstream_ee),
        downstream_es=int(event.downstream_es),
        upstream_es=event.upstream_es,
        downstream_ee=event.downstream_ee,
        proposal_source="rmats_template",
    )


def _build_recount_metrics_for_event(
    event: RmatsEvent,
    *,
    bam_groups: dict[str, object],
    n_replicates: int,
    confidence_level: float,
    effect_cutoff: float,
    seed: int,
    min_mapq: int,
) -> dict[str, object]:
    """Measure one rMATS SE event directly from the QKI BAMs."""
    if not bam_groups["available"]:
        return {
            "source": "count_table",
            "recount_available": False,
            "recount_unavailable_reason": bam_groups["reason"],
        }

    splice_event = _build_rmats_se_splice_event(event)
    if splice_event is None:
        return {
            "source": "count_table",
            "recount_available": False,
            "recount_unavailable_reason": "missing_rmats_flanking_coordinates",
        }

    ctrl_evidences = []
    kd_evidences = []
    for bam_path in bam_groups["sample_1_bams"]:
        ctrl_evidences.append(
            extract_event_evidence_from_bam(
                bam_path,
                splice_event,
                min_mapq=min_mapq,
            ),
        )
    for bam_path in bam_groups["sample_2_bams"]:
        kd_evidences.append(
            extract_event_evidence_from_bam(
                bam_path,
                splice_event,
                min_mapq=min_mapq,
            ),
        )

    event_seed = _stable_seed(seed, "recount", event.event_id)
    ctrl_replicates = [
        _sample_channel_fused_replicate(
            _extract_recount_channels(evidence.evidence_breakdown),
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            seed=event_seed + 100 + idx * 7,
            apply_single_sample_shrink=False,
        )
        for idx, evidence in enumerate(ctrl_evidences)
    ]
    kd_replicates = [
        _sample_channel_fused_replicate(
            _extract_recount_channels(evidence.evidence_breakdown),
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            seed=event_seed + 200 + idx * 7,
            apply_single_sample_shrink=len(kd_evidences) == 1,
        )
        for idx, evidence in enumerate(kd_evidences)
    ]
    ctrl_condition = _build_condition_metrics_from_replicates(
        ctrl_replicates,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
    )
    kd_condition = _build_condition_metrics_from_replicates(
        kd_replicates,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
    )
    dpsi = _build_dpsi_summary_from_samples(
        ctrl_condition["samples"],
        kd_condition["samples"],
        confidence_level=confidence_level,
        effect_cutoff=effect_cutoff,
    )
    return {
        "source": "recount",
        "recount_available": True,
        "recount_unavailable_reason": None,
        "ctrl": ctrl_condition["summary"],
        "kd": kd_condition["summary"],
        "dpsi": dpsi,
        "ctrl_replicate_channels": [
            payload["channels"]
            for payload in ctrl_replicates
        ],
        "kd_replicate_channels": [
            payload["channels"]
            for payload in kd_replicates
        ],
        "channel_fused_ctrl": ctrl_condition["fused_channels"],
        "channel_fused_kd": kd_condition["fused_channels"],
        "ctrl_channel_weights": [
            payload["channel_weights"]
            for payload in ctrl_replicates
        ],
        "kd_channel_weights": [
            payload["channel_weights"]
            for payload in kd_replicates
        ],
        "ctrl_replicate_shrink": [
            payload["single_sample_shrink"]
            for payload in ctrl_replicates
        ],
        "kd_replicate_shrink": [
            payload["single_sample_shrink"]
            for payload in kd_replicates
        ],
        "ctrl_bam_count": len(bam_groups["sample_1_bams"]),
        "kd_bam_count": len(bam_groups["sample_2_bams"]),
    }


def _get_recount_metrics_for_event(
    event: RmatsEvent,
    *,
    cache: dict[str, dict[str, object]],
    bam_groups: dict[str, object],
    n_replicates: int,
    confidence_level: float,
    effect_cutoff: float,
    seed: int,
    min_mapq: int,
) -> dict[str, object]:
    """Return cached recount metrics for one event, computing them lazily."""
    cached = cache.get(event.event_id)
    if cached is not None:
        return cached
    cache[event.event_id] = _build_recount_metrics_for_event(
        event,
        bam_groups=bam_groups,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        effect_cutoff=effect_cutoff,
        seed=seed,
        min_mapq=min_mapq,
    )
    return cache[event.event_id]


def _event_overlaps_targets(
    event: RmatsEvent,
    targets: list[QKITarget],
) -> bool:
    """Return True when one rMATS event overlaps any named QKI target interval."""
    event_chrom = _normalize_chrom(event.chrom)
    for target in targets:
        if event_chrom != _normalize_chrom(target.chrom):
            continue
        if _overlap_bp(event.exon_start, event.exon_end, target.start, target.end) > 0:
            return True
    return False


def _build_null_calibration(
    events: list[RmatsEvent],
    *,
    n_replicates: int,
    confidence_level: float,
    effect_cutoff: float,
    min_total_support: int,
    target_null_rate: float,
    minimum_threshold: float,
    sparse_bin_min_events: int,
    seed: int,
) -> dict:
    """Calibrate posterior-probability thresholds from ctrl-vs-ctrl null events."""
    records: list[dict[str, object]] = []
    for event in events:
        if (
            len(event.sample_1_inc_replicates) < 2
            or len(event.sample_1_exc_replicates) < 2
        ):
            continue
        inc_a = event.sample_1_inc_replicates[0]
        exc_a = event.sample_1_exc_replicates[0]
        inc_b = event.sample_1_inc_replicates[1]
        exc_b = event.sample_1_exc_replicates[1]
        total_support = inc_a + exc_a + inc_b + exc_b
        if total_support < min_total_support:
            continue

        dpsi_summary = _build_dpsi_summary_from_counts(
            inc_a,
            exc_a,
            inc_b,
            exc_b,
            event_type=event.event_type,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            effect_cutoff=effect_cutoff,
            seed=_stable_seed(seed, "null", event.event_id),
        )
        records.append({
            "event_id": event.event_id,
            "support_total": total_support,
            "support_bin": _classify_bin(total_support, SUPPORT_BINS),
            "probability": float(dpsi_summary["prob_abs_ge_cutoff"]),
        })

    probabilities = [float(record["probability"]) for record in records]
    payload: dict[str, object] = {
        "available": bool(records),
        "event_count": len(records),
        "target_null_rate": target_null_rate,
        "minimum_probability_floor": minimum_threshold,
        "effect_cutoff": effect_cutoff,
        "min_total_support": min_total_support,
        "sparse_bin_min_events": sparse_bin_min_events,
        "bins": {},
    }
    if not records:
        payload.update({
            "calibrated_probability_threshold": minimum_threshold,
            "observed_null_rate": None,
            "median_total_support": None,
            "global": {
                "event_count": 0,
                "calibrated_probability_threshold": minimum_threshold,
                "observed_null_rate": None,
                "null_probability_quantiles": {},
                "probabilities": [],
            },
        })
        return payload

    global_threshold = max(
        minimum_threshold,
        float(np.quantile(probabilities, 1 - target_null_rate, method="higher")),
    )
    global_observed = float(np.mean(np.array(probabilities) >= global_threshold))
    payload.update({
        "calibrated_probability_threshold": global_threshold,
        "observed_null_rate": global_observed,
        "median_total_support": _median_or_none(
            [float(record["support_total"]) for record in records],
        ),
        "null_probability_quantiles": _metric_quantiles(probabilities),
        "global": {
            "event_count": len(records),
            "calibrated_probability_threshold": global_threshold,
            "observed_null_rate": global_observed,
            "median_total_support": _median_or_none(
                [float(record["support_total"]) for record in records],
            ),
            "null_probability_quantiles": _metric_quantiles(probabilities),
            "probabilities": probabilities,
        },
    })

    records_by_bin: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        records_by_bin[str(record["support_bin"])].append(record)
    ordered_labels = [label for _lower, _upper, label in SUPPORT_BINS]

    for index, label in enumerate(ordered_labels):
        bin_records = records_by_bin.get(label, [])
        bin_probs = [float(record["probability"]) for record in bin_records]
        event_count = len(bin_probs)
        threshold_source = f"bin:{label}"
        if event_count >= sparse_bin_min_events:
            threshold = max(
                minimum_threshold,
                float(np.quantile(bin_probs, 1 - target_null_rate, method="higher")),
            )
        else:
            threshold = None
            for next_label in ordered_labels[index + 1:]:
                next_records = records_by_bin.get(next_label, [])
                next_probs = [
                    float(record["probability"])
                    for record in next_records
                ]
                if len(next_probs) >= sparse_bin_min_events:
                    threshold = max(
                        minimum_threshold,
                        float(np.quantile(next_probs, 1 - target_null_rate, method="higher")),
                    )
                    threshold_source = f"fallback:{next_label}"
                    break
            if threshold is None and probabilities:
                threshold = global_threshold
                threshold_source = "global"
            if threshold is None:
                threshold = minimum_threshold
                threshold_source = "minimum_floor"

        payload["bins"][label] = {
            "event_count": event_count,
            "is_sparse": event_count < sparse_bin_min_events,
            "calibrated_probability_threshold": threshold,
            "threshold_source": threshold_source,
            "observed_null_rate": (
                float(np.mean(np.array(bin_probs) >= threshold))
                if bin_probs
                else None
            ),
            "median_total_support": _median_or_none(
                [float(record["support_total"]) for record in bin_records],
            ),
            "null_probability_quantiles": _metric_quantiles(bin_probs),
            "probabilities": bin_probs,
        }
    return payload


def _resolve_supported_threshold(
    total_support: int,
    null_calibration: dict,
) -> tuple[str, float, str]:
    """Resolve the support bin and posterior threshold used for one event."""
    support_bin = _classify_bin(
        max(
            total_support,
            int(null_calibration.get("min_total_support", SUPPORT_BINS[0][0])),
        ),
        SUPPORT_BINS,
    )
    bin_meta = null_calibration.get("bins", {}).get(support_bin, {})
    threshold = float(
        bin_meta.get(
            "calibrated_probability_threshold",
            null_calibration.get("calibrated_probability_threshold", 0.5),
        ),
    )
    source = str(bin_meta.get("threshold_source", "global"))
    return support_bin, threshold, source


def _summarize(total_targets: int, rows: list[dict]) -> dict:
    """Summarize one QKI cohort."""
    matched_rows = [row for row in rows if row.get("matched")]
    significant_rows = [row for row in matched_rows if row.get("significant")]
    confident_rows = [row for row in matched_rows if row.get("either_confident")]
    supported_rows = [row for row in matched_rows if row.get("supported_differential")]
    near_strict_rows = [row for row in matched_rows if row.get("near_strict")]
    high_conf_rows = [row for row in matched_rows if row.get("high_confidence")]
    recount_rows = [row for row in matched_rows if row.get("recount_available")]
    supports = [
        float(row["total_support"])
        for row in matched_rows
        if row.get("total_support") is not None
    ]
    coord_deltas = [
        float(row["coord_delta"])
        for row in matched_rows
        if row.get("coord_delta") is not None
    ]

    return {
        "total_targets": total_targets,
        "matched_targets": len(matched_rows),
        "match_rate": len(matched_rows) / total_targets if total_targets else None,
        "significant_matches": len(significant_rows),
        "significant_rate_over_total": (
            len(significant_rows) / total_targets if total_targets else None
        ),
        "confident_matches": len(confident_rows),
        "confident_rate_over_total": (
            len(confident_rows) / total_targets if total_targets else None
        ),
        "supported_matches": len(supported_rows),
        "supported_rate_over_total": (
            len(supported_rows) / total_targets if total_targets else None
        ),
        "near_strict_matches": len(near_strict_rows),
        "near_strict_rate_over_total": (
            len(near_strict_rows) / total_targets if total_targets else None
        ),
        "high_confidence_matches": len(high_conf_rows),
        "high_confidence_rate_over_total": (
            len(high_conf_rows) / total_targets if total_targets else None
        ),
        "recount_backed_matches": len(recount_rows),
        "recount_backed_rate_over_total": (
            len(recount_rows) / total_targets if total_targets else None
        ),
        "median_total_support": _median_or_none(supports),
        "median_coord_delta": _median_or_none(coord_deltas),
    }


def _serialize_metrics(
    prefix: str,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Flatten one condition metrics block into row fields."""
    return {
        f"{prefix}_psi": metrics["psi"],
        f"{prefix}_ci_low": metrics["ci_low"],
        f"{prefix}_ci_high": metrics["ci_high"],
        f"{prefix}_ci_width": metrics["ci_width"],
        f"{prefix}_cv": metrics["cv"],
        f"{prefix}_confident": metrics["is_confident"],
        f"{prefix}_inc": metrics["inclusion_count"],
        f"{prefix}_exc": metrics["exclusion_count"],
        f"{prefix}_support_total": metrics["support_total"],
    }


def _target_result(
    target: QKITarget,
    matched: RmatsEvent | None,
    *,
    n_matching_candidates: int,
    overlap_with_validated: bool,
    overlap_bp: int,
    overlap_fraction: float,
    coord_delta: int | None,
    normalized_target_start: int,
    normalized_target_end: int,
    target_was_normalized: bool,
    normalization_delta: int | None,
    match_basis: str,
    count_metrics_by_event: dict[str, dict[str, object]],
    recount_metrics_by_event: dict[str, dict[str, object]],
    null_calibration: dict,
    fdr_threshold: float,
    high_confidence_dpsi: float,
    high_confidence_min_total_support: int,
) -> dict:
    """Serialize one target result row."""
    row = {
        "gene": target.gene,
        "chrom": target.chrom,
        "target_start": target.start,
        "target_end": target.end,
        "cohort": target.cohort,
        "overlap_with_validated": overlap_with_validated,
        "n_matching_candidates": n_matching_candidates,
        "normalized_target_start": normalized_target_start,
        "normalized_target_end": normalized_target_end,
        "target_was_normalized": target_was_normalized,
        "target_normalization_delta": normalization_delta,
        "match_basis": match_basis,
        "matched": matched is not None,
    }
    if matched is None:
        return row

    count_metrics = count_metrics_by_event[matched.event_id]
    recount_metrics = recount_metrics_by_event.get(matched.event_id, {
        "source": "count_table",
        "recount_available": False,
        "recount_unavailable_reason": "recount_metrics_missing",
    })
    active_metrics = (
        recount_metrics
        if recount_metrics.get("recount_available")
        else count_metrics
    )
    ctrl_metrics = active_metrics["ctrl"]
    kd_metrics = active_metrics["kd"]
    dpsi_summary = active_metrics["dpsi"]
    total_support = int(ctrl_metrics["support_total"]) + int(kd_metrics["support_total"])
    support_bin, supported_threshold, threshold_source = _resolve_supported_threshold(
        total_support,
        null_calibration,
    )
    braid_dpsi = float(ctrl_metrics["psi"]) - float(kd_metrics["psi"])
    significant = math.isfinite(matched.rmats_fdr) and matched.rmats_fdr <= fdr_threshold
    either_confident = bool(ctrl_metrics["is_confident"]) or bool(kd_metrics["is_confident"])
    high_confidence = (
        significant
        and abs(braid_dpsi) >= high_confidence_dpsi
        and bool(dpsi_summary["excludes_zero"])
        and total_support >= high_confidence_min_total_support
    )
    supported_differential = (
        significant
        and abs(braid_dpsi) >= high_confidence_dpsi
        and float(dpsi_summary["prob_abs_ge_cutoff"]) >= supported_threshold
        and total_support >= high_confidence_min_total_support
    )
    near_strict = supported_differential and not high_confidence
    tier = "matched_only"
    if significant:
        tier = "significant_only"
    if supported_differential:
        tier = "supported"
    if near_strict:
        tier = "near_strict"
    if high_confidence:
        tier = "high_confidence"

    row.update({
        "event_id": matched.event_id,
        "event_start": matched.exon_start,
        "event_end": matched.exon_end,
        "coord_delta": coord_delta,
        "overlap_bp": overlap_bp,
        "overlap_fraction": overlap_fraction,
        "significant": significant,
        "rmats_fdr": matched.rmats_fdr,
        "rmats_dpsi": matched.rmats_dpsi,
        "rmats_ctrl_psi": matched.sample_1_psi,
        "rmats_kd_psi": matched.sample_2_psi,
        "evidence_source": active_metrics.get("source", "count_table"),
        "recount_available": bool(recount_metrics.get("recount_available")),
        "recount_unavailable_reason": recount_metrics.get("recount_unavailable_reason"),
        "supported_support_bin": support_bin,
        "supported_probability_threshold": supported_threshold,
        "supported_threshold_source": threshold_source,
        "either_confident": either_confident,
        "braid_dpsi": braid_dpsi,
        "braid_dpsi_posterior_mean": dpsi_summary["posterior_mean"],
        "braid_dpsi_ci_low": dpsi_summary["ci_low"],
        "braid_dpsi_ci_high": dpsi_summary["ci_high"],
        "braid_dpsi_ci_width": dpsi_summary["ci_width"],
        "braid_dpsi_excludes_zero": dpsi_summary["excludes_zero"],
        "braid_dpsi_prob_abs_ge_cutoff": dpsi_summary["prob_abs_ge_cutoff"],
        "supported_differential": supported_differential,
        "near_strict": near_strict,
        "high_confidence": high_confidence,
        "tier": tier,
        "total_support": total_support,
        "count_table_total_support": (
            int(count_metrics["ctrl"]["support_total"]) + int(count_metrics["kd"]["support_total"])
        ),
        "count_table_braid_dpsi": (
            float(count_metrics["ctrl"]["psi"]) - float(count_metrics["kd"]["psi"])
        ),
        "count_table_braid_dpsi_prob_abs_ge_cutoff": (
            count_metrics["dpsi"]["prob_abs_ge_cutoff"]
        ),
        "count_table_braid_dpsi_ci_low": count_metrics["dpsi"]["ci_low"],
        "count_table_braid_dpsi_ci_high": count_metrics["dpsi"]["ci_high"],
    })
    row.update(_serialize_metrics("ctrl", ctrl_metrics))
    row.update(_serialize_metrics("kd", kd_metrics))
    row.update({
        "count_table_ctrl_psi": count_metrics["ctrl"]["psi"],
        "count_table_ctrl_inc": count_metrics["ctrl"]["inclusion_count"],
        "count_table_ctrl_exc": count_metrics["ctrl"]["exclusion_count"],
        "count_table_kd_psi": count_metrics["kd"]["psi"],
        "count_table_kd_inc": count_metrics["kd"]["inclusion_count"],
        "count_table_kd_exc": count_metrics["kd"]["exclusion_count"],
    })
    if recount_metrics.get("recount_available"):
        row.update({
            "recount_ctrl_inc": recount_metrics["ctrl"]["inclusion_count"],
            "recount_ctrl_exc": recount_metrics["ctrl"]["exclusion_count"],
            "recount_kd_inc": recount_metrics["kd"]["inclusion_count"],
            "recount_kd_exc": recount_metrics["kd"]["exclusion_count"],
            "recount_total_support": total_support,
            "recount_braid_dpsi": braid_dpsi,
            "recount_braid_dpsi_ci_low": dpsi_summary["ci_low"],
            "recount_braid_dpsi_ci_high": dpsi_summary["ci_high"],
            "recount_braid_dpsi_prob_abs_ge_cutoff": dpsi_summary["prob_abs_ge_cutoff"],
            "recount_ctrl_replicate_channels": recount_metrics.get("ctrl_replicate_channels", []),
            "recount_kd_replicate_channels": recount_metrics.get("kd_replicate_channels", []),
            "recount_channel_fused_ctrl": recount_metrics.get("channel_fused_ctrl", {}),
            "recount_channel_fused_kd": recount_metrics.get("channel_fused_kd", {}),
            "recount_ctrl_channel_weights": recount_metrics.get("ctrl_channel_weights", []),
            "recount_kd_channel_weights": recount_metrics.get("kd_channel_weights", []),
            "recount_ctrl_replicate_shrink": recount_metrics.get("ctrl_replicate_shrink", []),
            "recount_kd_replicate_shrink": recount_metrics.get("kd_replicate_shrink", []),
            "recount_ctrl_bam_count": recount_metrics.get("ctrl_bam_count"),
            "recount_kd_bam_count": recount_metrics.get("kd_bam_count"),
        })
    return row


def _build_matched_null_control_rows(
    *,
    events: list[RmatsEvent],
    reference_rows: list[dict],
    all_targets: list[QKITarget],
    count_metrics_by_event: dict[str, dict[str, object]],
    recount_metrics_by_event: dict[str, dict[str, object]],
    bam_groups: dict[str, object],
    null_calibration: dict,
    fdr_threshold: float,
    high_confidence_dpsi: float,
    high_confidence_min_total_support: int,
    n_replicates: int,
    confidence_level: float,
    min_mapq: int,
    seed: int,
    cohort_name: str,
    candidate_support_bins: set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Build one matched null-control cohort from non-target rMATS SE events."""
    priority = lambda event_id: _stable_seed(seed, cohort_name, event_id)
    candidates: list[RmatsEvent] = []
    for event in events:
        if not math.isfinite(event.rmats_fdr) or event.rmats_fdr < 0.5:
            continue
        if _event_overlaps_targets(event, all_targets):
            continue
        total_support = _event_total_support(event)
        if total_support < high_confidence_min_total_support:
            continue
        if candidate_support_bins is not None and _classify_bin(total_support, SUPPORT_BINS) not in candidate_support_bins:
            continue
        candidates.append(event)
    candidates.sort(key=lambda event: (priority(event.event_id), event.event_id))

    used_event_ids: set[str] = set()
    selection_sources: Counter[str] = Counter()
    length_bin_counts: Counter[str] = Counter()
    support_bin_counts: Counter[str] = Counter()
    selected_rows: list[dict] = []

    def _pick_candidate(
        *,
        desired_length_bin: str,
        desired_support_bin: str,
    ) -> tuple[RmatsEvent | None, str]:
        selection_specs = (
            ("exact_pair", lambda ev: (
                _classify_bin(max(1, ev.exon_end - ev.exon_start), LENGTH_BINS) == desired_length_bin
                and _classify_bin(_event_total_support(ev), SUPPORT_BINS) == desired_support_bin
            )),
            ("same_support", lambda ev: (
                _classify_bin(_event_total_support(ev), SUPPORT_BINS) == desired_support_bin
            )),
            ("same_length", lambda ev: (
                _classify_bin(max(1, ev.exon_end - ev.exon_start), LENGTH_BINS) == desired_length_bin
            )),
            ("global", lambda ev: True),
        )
        for source, predicate in selection_specs:
            for candidate in candidates:
                if candidate.event_id in used_event_ids:
                    continue
                if predicate(candidate):
                    return candidate, source
        return None, "unavailable"

    for row in reference_rows:
        desired_length_bin = _classify_bin(
            max(1, int(row["normalized_target_end"]) - int(row["normalized_target_start"])),
            LENGTH_BINS,
        )
        desired_support_bin = _classify_bin(
            int(row.get("total_support") or high_confidence_min_total_support),
            SUPPORT_BINS,
        )
        if candidate_support_bins is not None and desired_support_bin not in candidate_support_bins:
            continue
        candidate, selection_source = _pick_candidate(
            desired_length_bin=desired_length_bin,
            desired_support_bin=desired_support_bin,
        )
        if candidate is None:
            continue
        _get_recount_metrics_for_event(
            candidate,
            cache=recount_metrics_by_event,
            bam_groups=bam_groups,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            effect_cutoff=high_confidence_dpsi,
            seed=seed,
            min_mapq=min_mapq,
        )
        used_event_ids.add(candidate.event_id)
        selection_sources[selection_source] += 1
        length_bin_counts[desired_length_bin] += 1
        support_bin_counts[desired_support_bin] += 1
        control_target = QKITarget(
            gene=candidate.gene,
            chrom=candidate.chrom,
            start=candidate.exon_start,
            end=candidate.exon_end,
            cohort=cohort_name,
        )
        control_row = _target_result(
            control_target,
            candidate,
            n_matching_candidates=1,
            overlap_with_validated=False,
            overlap_bp=max(1, candidate.exon_end - candidate.exon_start),
            overlap_fraction=1.0,
            coord_delta=0,
            normalized_target_start=candidate.exon_start,
            normalized_target_end=candidate.exon_end,
            target_was_normalized=False,
            normalization_delta=None,
            match_basis="matched_null_control",
            count_metrics_by_event=count_metrics_by_event,
            recount_metrics_by_event=recount_metrics_by_event,
            null_calibration=null_calibration,
            fdr_threshold=fdr_threshold,
            high_confidence_dpsi=high_confidence_dpsi,
            high_confidence_min_total_support=high_confidence_min_total_support,
        )
        control_row.update({
            "matched_null_reference_gene": row["gene"],
            "matched_null_reference_length_bin": desired_length_bin,
            "matched_null_reference_support_bin": desired_support_bin,
            "matched_null_selection_source": selection_source,
        })
        selected_rows.append(control_row)

    metadata = {
        "requested_count": len(reference_rows),
        "selected_count": len(selected_rows),
        "candidate_pool_size": len(candidates),
        "selection_source_counts": dict(selection_sources),
        "reference_length_bin_counts": dict(length_bin_counts),
        "reference_support_bin_counts": dict(support_bin_counts),
        "seed": seed,
        "cohort_name": cohort_name,
    }
    return selected_rows, metadata


def _derive_casebook_output_path(output_path: str) -> str:
    """Return a sibling path for the casebook artifact."""
    if output_path.endswith("_results.json"):
        return output_path.replace("_results.json", "_casebook.json")
    if output_path.endswith(".json"):
        return output_path[:-5] + "_casebook.json"
    return f"{output_path}.casebook.json"


def _build_casebook(
    *,
    validated_rows: list[dict],
    matched_null_control_rows: list[dict],
    high_support_matched_null_rows: list[dict],
) -> dict:
    """Build a compact reviewer-facing casebook of near-miss and null examples."""
    near_miss = [
        row for row in validated_rows
        if row.get("matched")
        and row.get("significant")
        and not row.get("high_confidence")
    ]
    near_miss.sort(
        key=lambda row: (
            row.get("near_strict", False),
            row.get("supported_differential", False),
            abs(float(row.get("braid_dpsi", 0.0))),
            float(row.get("total_support", 0.0)),
        ),
        reverse=True,
    )
    high_support_null = sorted(
        high_support_matched_null_rows,
        key=lambda row: float(row.get("total_support", 0.0)),
        reverse=True,
    )
    matched_null = sorted(
        matched_null_control_rows,
        key=lambda row: float(row.get("total_support", 0.0)),
        reverse=True,
    )
    return {
        "validated_near_miss": near_miss[:CASEBOOK_TOP_NEAR_MISS],
        "high_support_matched_null": high_support_null[:CASEBOOK_TOP_NULL],
        "matched_null_examples": matched_null[:CASEBOOK_TOP_NULL],
    }


def _build_hybrid_comparison(results: dict) -> dict:
    """Summarize the rMATS-alone vs BRAID-layer comparison for paper/report use."""
    return {
        "validated": {
            "rmats_significant_matches": results["validated_summary"]["significant_matches"],
            "braid_supported_matches": results["validated_summary"]["supported_matches"],
            "braid_near_strict_matches": results["validated_summary"]["near_strict_matches"],
            "braid_high_confidence_matches": results["validated_summary"]["high_confidence_matches"],
        },
        "matched_null_control": {
            "rmats_significant_matches": results["matched_null_control_summary"]["significant_matches"],
            "braid_supported_matches": results["matched_null_control_summary"]["supported_matches"],
            "braid_near_strict_matches": results["matched_null_control_summary"]["near_strict_matches"],
            "braid_high_confidence_matches": results["matched_null_control_summary"]["high_confidence_matches"],
        },
    }


def _write_json(output_path: str, payload: dict) -> None:
    """Write one JSON payload."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_qki_rmats_benchmark(
    *,
    rmats_dir: str,
    qki_dir: str,
    output_path: str,
    tolerance: int,
    min_overlap_fraction: float,
    target_normalization_max_delta: int,
    min_total_count: int,
    n_replicates: int,
    confidence_level: float,
    fdr_threshold: float,
    high_confidence_dpsi: float,
    high_confidence_min_total_support: int,
    seed: int,
    supported_target_null_rate: float = 0.05,
    supported_min_probability: float = 0.5,
    prefer_lifted_targets: bool = True,
    min_mapq: int = DEFAULT_MIN_MAPQ,
) -> dict:
    """Benchmark QKI RT-PCR targets against rMATS SE event detection."""
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
    all_targets = validated_targets + failed_targets

    events = parse_rmats_output(
        rmats_dir,
        event_types=["SE"],
        min_total_count=min_total_count,
    )
    templates = _load_rmats_templates(rmats_dir, event_type="SE")
    count_metrics_by_event = _build_count_table_metrics_by_event(
        events,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        effect_cutoff=high_confidence_dpsi,
        seed=seed,
    )
    bam_groups = _resolve_recount_bam_groups(rmats_dir)
    recount_metrics_by_event: dict[str, dict[str, object]] = {}
    null_calibration = _build_null_calibration(
        events,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        effect_cutoff=high_confidence_dpsi,
        min_total_support=high_confidence_min_total_support,
        target_null_rate=supported_target_null_rate,
        minimum_threshold=supported_min_probability,
        sparse_bin_min_events=NULL_CALIBRATION_SPARSE_BIN_MIN_EVENTS,
        seed=seed + 101,
    )

    rows_by_cohort = {
        "validated": [],
        "failed": [],
    }
    matched_specs: list[dict[str, object]] = []
    for target in validated_targets + failed_targets:
        normalized_start, normalized_end, target_was_normalized, normalization_delta = (
            _normalize_target_coords(
                target,
                templates,
                max_delta=target_normalization_max_delta,
            )
        )

        matched, n_matches, overlap_bp, overlap_fraction, coord_delta = _find_matching_event(
            target,
            events,
            target_start=target.start,
            target_end=target.end,
            tolerance=tolerance,
            min_overlap_fraction=min_overlap_fraction,
        )
        match_basis = "original"
        if matched is None and (
            normalized_start != target.start or normalized_end != target.end
        ):
            matched, n_matches, overlap_bp, overlap_fraction, coord_delta = _find_matching_event(
                target,
                events,
                target_start=normalized_start,
                target_end=normalized_end,
                tolerance=tolerance,
                min_overlap_fraction=min_overlap_fraction,
            )
            if matched is not None:
                match_basis = "normalized"
        matched_specs.append({
            "target": target,
            "matched": matched,
            "n_matching_candidates": n_matches,
            "overlap_with_validated": target.key in overlap_keys,
            "overlap_bp": overlap_bp,
            "overlap_fraction": overlap_fraction,
            "coord_delta": coord_delta if matched is not None else None,
            "normalized_target_start": normalized_start,
            "normalized_target_end": normalized_end,
            "target_was_normalized": target_was_normalized,
            "normalization_delta": normalization_delta,
            "match_basis": match_basis,
        })
        if matched is not None:
            _get_recount_metrics_for_event(
                matched,
                cache=recount_metrics_by_event,
                bam_groups=bam_groups,
                n_replicates=n_replicates,
                confidence_level=confidence_level,
                effect_cutoff=high_confidence_dpsi,
                seed=seed,
                min_mapq=min_mapq,
            )

    for spec in matched_specs:
        target = spec["target"]
        row = _target_result(
            target,
            spec["matched"],
            n_matching_candidates=spec["n_matching_candidates"],
            overlap_with_validated=spec["overlap_with_validated"],
            overlap_bp=spec["overlap_bp"],
            overlap_fraction=spec["overlap_fraction"],
            coord_delta=spec["coord_delta"],
            normalized_target_start=spec["normalized_target_start"],
            normalized_target_end=spec["normalized_target_end"],
            target_was_normalized=spec["target_was_normalized"],
            normalization_delta=spec["normalization_delta"],
            match_basis=spec["match_basis"],
            count_metrics_by_event=count_metrics_by_event,
            recount_metrics_by_event=recount_metrics_by_event,
            null_calibration=null_calibration,
            fdr_threshold=fdr_threshold,
            high_confidence_dpsi=high_confidence_dpsi,
            high_confidence_min_total_support=high_confidence_min_total_support,
        )
        rows_by_cohort[target.cohort].append(row)

    validated_rows = rows_by_cohort["validated"]
    failed_rows = rows_by_cohort["failed"]
    overlap_failed_rows = [row for row in failed_rows if row["overlap_with_validated"]]
    exclusive_failed_rows = [row for row in failed_rows if not row["overlap_with_validated"]]
    matched_null_control_rows, matched_null_control_selection = _build_matched_null_control_rows(
        events=events,
        reference_rows=validated_rows,
        all_targets=all_targets,
        count_metrics_by_event=count_metrics_by_event,
        recount_metrics_by_event=recount_metrics_by_event,
        bam_groups=bam_groups,
        null_calibration=null_calibration,
        fdr_threshold=fdr_threshold,
        high_confidence_dpsi=high_confidence_dpsi,
        high_confidence_min_total_support=high_confidence_min_total_support,
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        min_mapq=min_mapq,
        seed=seed,
        cohort_name="matched_null_control",
    )
    high_support_reference_rows = [
        row for row in validated_rows
        if row.get("supported_support_bin") in HIGH_SUPPORT_CONTROL_BINS
    ]
    high_support_matched_null_rows, high_support_matched_null_selection = (
        _build_matched_null_control_rows(
            events=events,
            reference_rows=high_support_reference_rows,
            all_targets=all_targets,
            count_metrics_by_event=count_metrics_by_event,
            recount_metrics_by_event=recount_metrics_by_event,
            bam_groups=bam_groups,
            null_calibration=null_calibration,
            fdr_threshold=fdr_threshold,
            high_confidence_dpsi=high_confidence_dpsi,
            high_confidence_min_total_support=high_confidence_min_total_support,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            min_mapq=min_mapq,
            seed=seed + 17,
            cohort_name="high_support_matched_null_control",
            candidate_support_bins=HIGH_SUPPORT_CONTROL_BINS,
        )
    )

    recount_available_count = sum(
        1
        for row in (
            validated_rows
            + failed_rows
            + matched_null_control_rows
            + high_support_matched_null_rows
        )
        if row.get("recount_available")
    )
    casebook = _build_casebook(
        validated_rows=validated_rows,
        matched_null_control_rows=matched_null_control_rows,
        high_support_matched_null_rows=high_support_matched_null_rows,
    )
    casebook_output_path = _derive_casebook_output_path(output_path)
    payload = {
        "metadata": {
            "rmats_dir": rmats_dir,
            "qki_dir": qki_dir,
            "event_type": "SE",
            "rmats_event_count": len(events),
            "match_tolerance": tolerance,
            "min_overlap_fraction": min_overlap_fraction,
            "target_normalization_max_delta": target_normalization_max_delta,
            "min_total_count": min_total_count,
            "n_replicates": n_replicates,
            "confidence_level": confidence_level,
            "fdr_threshold": fdr_threshold,
            "high_confidence_dpsi": high_confidence_dpsi,
            "high_confidence_min_total_support": high_confidence_min_total_support,
            "supported_target_null_rate": supported_target_null_rate,
            "supported_min_probability": supported_min_probability,
            "supported_probability_threshold": null_calibration["calibrated_probability_threshold"],
            "supported_probability_threshold_bins": {
                label: block["calibrated_probability_threshold"]
                for label, block in null_calibration.get("bins", {}).items()
            },
            "null_calibration_sparse_bin_min_events": NULL_CALIBRATION_SPARSE_BIN_MIN_EVENTS,
            "min_mapq": min_mapq,
            "seed": seed,
            "validated_total": len(validated_targets),
            "failed_total": len(failed_targets),
            "label_overlap_count": len(overlap_keys),
            "template_group_count": len(templates),
            "recount_available": bam_groups["available"],
            "recount_reason": bam_groups["reason"],
            "recount_sample_1_bams": bam_groups["sample_1_bams"],
            "recount_sample_2_bams": bam_groups["sample_2_bams"],
            "recount_backed_row_count": recount_available_count,
            "casebook_output_path": casebook_output_path,
            **target_table_metadata,
        },
        "validated_summary": _summarize(len(validated_targets), validated_rows),
        "failed_summary": _summarize(len(failed_targets), failed_rows),
        "overlap_failed_summary": _summarize(len(overlap_failed_rows), overlap_failed_rows),
        "exclusive_failed_summary": _summarize(len(exclusive_failed_rows), exclusive_failed_rows),
        "matched_null_control_summary": _summarize(
            matched_null_control_selection["requested_count"],
            matched_null_control_rows,
        ),
        "high_support_matched_null_control_summary": _summarize(
            high_support_matched_null_selection["requested_count"],
            high_support_matched_null_rows,
        ),
        "matched_null_control_selection": matched_null_control_selection,
        "high_support_matched_null_control_selection": high_support_matched_null_selection,
        "null_calibration": null_calibration,
        "hybrid_comparison": {},
        "validated": validated_rows,
        "failed": failed_rows,
        "matched_null_control": matched_null_control_rows,
        "high_support_matched_null_control": high_support_matched_null_rows,
    }
    payload["hybrid_comparison"] = _build_hybrid_comparison(payload)
    _write_json(output_path, payload)
    _write_json(casebook_output_path, casebook)
    return payload


def _print_summary(results: dict) -> None:
    """Emit a concise console summary."""
    metadata = results["metadata"]
    validated = results["validated_summary"]
    failed = results["failed_summary"]
    overlap = results["overlap_failed_summary"]
    exclusive_failed = results["exclusive_failed_summary"]
    matched_null = results["matched_null_control_summary"]
    high_support_null = results["high_support_matched_null_control_summary"]
    null_cal = results["null_calibration"]

    print("=" * 60)
    print("  QKI rMATS + BRAID Benchmark")
    print("=" * 60)
    print(f"rMATS SE events:           {metadata['rmats_event_count']}")
    print(f"Validated matched:         {validated['matched_targets']}/{validated['total_targets']}")
    print(f"Validated significant:     {validated['significant_matches']}/{validated['total_targets']}")
    print(f"Validated supported:       {validated['supported_matches']}/{validated['total_targets']}")
    print(f"Validated near-strict:     {validated['near_strict_matches']}/{validated['total_targets']}")
    print(f"Validated high-conf:       {validated['high_confidence_matches']}/{validated['total_targets']}")
    print(f"Failed matched:            {failed['matched_targets']}/{failed['total_targets']}")
    print(f"Overlap failed supported:  {overlap['supported_matches']}/{overlap['total_targets']}")
    print(f"Exclusive failed supported:{exclusive_failed['supported_matches']}/{exclusive_failed['total_targets']}")
    print(f"Matched null supported:    {matched_null['supported_matches']}/{matched_null['total_targets']}")
    print(f"Matched null high-conf:    {matched_null['high_confidence_matches']}/{matched_null['total_targets']}")
    print(f"High-support null supp.:   {high_support_null['supported_matches']}/{high_support_null['total_targets']}")
    print(f"High-support null hconf:   {high_support_null['high_confidence_matches']}/{high_support_null['total_targets']}")
    if null_cal["available"]:
        print(
            "Global null prob thr:     "
            f"{null_cal['calibrated_probability_threshold']:.3f}"
        )
        for label in [label for _lower, _upper, label in SUPPORT_BINS]:
            block = null_cal["bins"][label]
            print(
                f"  Bin {label:>7}: thr={block['calibrated_probability_threshold']:.3f} "
                f"src={block['threshold_source']} n={block['event_count']}"
            )
    if validated["median_total_support"] is not None:
        print(f"Validated median support:  {validated['median_total_support']:.1f}")
    if validated["median_coord_delta"] is not None:
        print(f"Validated median Δcoord:   {validated['median_coord_delta']:.1f}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rmats-dir",
        required=True,
        help="Directory containing rMATS *.MATS.JunctionCountOnly.txt files.",
    )
    parser.add_argument(
        "--qki-dir",
        default="real_benchmark/rtpcr_benchmark/qki",
        help="Directory containing validated_events.tsv and failed_events.tsv.",
    )
    parser.add_argument(
        "--output",
        default="real_benchmark/rtpcr_benchmark/qki/qki_rmats_benchmark_results.json",
        help="Path to the benchmark JSON output file.",
    )
    parser.add_argument("--tolerance", type=int, default=50)
    parser.add_argument("--min-overlap-fraction", type=float, default=0.5)
    parser.add_argument("--target-normalization-max-delta", type=int, default=500)
    parser.add_argument("--min-total-count", type=int, default=1)
    parser.add_argument("--n-replicates", type=int, default=200)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--fdr-threshold", type=float, default=0.05)
    parser.add_argument("--high-confidence-dpsi", type=float, default=0.1)
    parser.add_argument("--high-confidence-min-total-support", type=int, default=20)
    parser.add_argument("--supported-target-null-rate", type=float, default=0.05)
    parser.add_argument("--supported-min-probability", type=float, default=0.5)
    parser.add_argument("--min-mapq", type=int, default=DEFAULT_MIN_MAPQ)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-targets",
        action="store_true",
        help="Ignore validated_events.hg38.tsv / failed_events.hg38.tsv and use raw source tables.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    results = run_qki_rmats_benchmark(
        rmats_dir=args.rmats_dir,
        qki_dir=args.qki_dir,
        output_path=args.output,
        tolerance=args.tolerance,
        min_overlap_fraction=args.min_overlap_fraction,
        target_normalization_max_delta=args.target_normalization_max_delta,
        min_total_count=args.min_total_count,
        n_replicates=args.n_replicates,
        confidence_level=args.confidence_level,
        fdr_threshold=args.fdr_threshold,
        high_confidence_dpsi=args.high_confidence_dpsi,
        high_confidence_min_total_support=args.high_confidence_min_total_support,
        seed=args.seed,
        supported_target_null_rate=args.supported_target_null_rate,
        supported_min_probability=args.supported_min_probability,
        prefer_lifted_targets=not args.raw_targets,
        min_mapq=args.min_mapq,
    )
    _print_summary(results)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
