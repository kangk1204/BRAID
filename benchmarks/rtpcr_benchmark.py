#!/usr/bin/env python3
"""Benchmark BRAID PSI bootstrap CI against external validation data.

Uses experimentally validated PSI data from:
1. VastDB RT-PCR compendium (Tapial 2017, R²=0.81)
2. MAJIQ RT-PCR (Vaquero-Garcia 2016, 200+ experiments)
3. PSI-Sigma RT-PCR (Lin 2019, 130 validations)
4. Our own PacBio long-read validation (K562 ENCFF652QLH)

Current implementation wires in the PacBio long-read validation path.
It benchmarks junction-centric A3SS/A5SS events only, because the
long-read proxy computed here is defined on junction usage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from statistics import median

import numpy as np

sys.path.insert(0, ".")

SUPPORT_BINS: tuple[tuple[int, int | None, str], ...] = (
    (0, 19, "<20"),
    (20, 49, "20-49"),
    (50, 99, "50-99"),
    (100, 249, "100-249"),
    (250, None, "250+"),
)
CALIBRATION_TARGET_COVERAGE = 0.95
CALIBRATION_SPARSE_BIN_MIN_EVENTS = 10


def _classify_support_bin(total_support: int) -> str:
    """Assign one event support total to a named bin."""
    for lower, upper, label in SUPPORT_BINS:
        if total_support < lower:
            continue
        if upper is None or total_support <= upper:
            return label
    return SUPPORT_BINS[-1][2]


def _required_interval_inflation_factor(
    *,
    psi: float,
    ci_low: float,
    ci_high: float,
    truth: float,
) -> float:
    """Return the minimum multiplicative interval inflation needed to include truth."""
    if ci_low <= truth <= ci_high:
        return 1.0

    low_span = max(abs(psi - ci_low), 1e-9)
    high_span = max(abs(ci_high - psi), 1e-9)
    if truth < ci_low:
        return max(1.0, abs(psi - truth) / low_span)
    return max(1.0, abs(truth - psi) / high_span)


def _contains_after_inflation(row: dict[str, float | bool], factor: float) -> bool:
    """Check whether a truth value falls inside an inflated interval."""
    psi = float(row["braid_psi"])
    ci_low = float(row["ci_low"])
    ci_high = float(row["ci_high"])
    truth = float(row["lr_psi"])
    low_span = abs(psi - ci_low)
    high_span = abs(ci_high - psi)
    return (psi - factor * low_span) <= truth <= (psi + factor * high_span)


def _summarize_rows(
    rows: list[dict[str, float | bool]],
    *,
    factor: float,
    confidence_width_threshold: float | dict[str, float],
    use_row_confident_flag: bool = False,
) -> dict[str, float | int | None]:
    """Summarize one support bin under a fixed interval inflation factor."""
    if not rows:
        return {}

    contains = [_contains_after_inflation(row, factor) for row in rows]
    widths = [float(row["ci_width"]) * factor for row in rows]
    def _threshold_for_row(row: dict[str, float | bool]) -> float:
        if "confidence_width_threshold" in row:
            return float(row["confidence_width_threshold"])
        if isinstance(confidence_width_threshold, dict):
            return float(confidence_width_threshold.get(str(row["support_bin"]), 0.2))
        return float(confidence_width_threshold)
    if use_row_confident_flag:
        calibrated_confident = [row for row in rows if bool(row["raw_confident"])]
    else:
        calibrated_confident = [
            row for row, width in zip(rows, widths)
            if bool(row["raw_confident"]) and width < _threshold_for_row(row)
        ]
    calibrated_confident_correct = [
        bool(row["confident_correct"])
        for row in calibrated_confident
    ]
    return {
        "n_events": len(rows),
        "ci_coverage": float(np.mean(contains)),
        "median_ci_width": float(median(widths)),
        "confident_count": len(calibrated_confident),
        "confident_accuracy": (
            float(np.mean(calibrated_confident_correct))
            if calibrated_confident_correct
            else None
        ),
    }


def _calibration_factor_for_rows(
    rows: list[dict[str, float | bool]],
    *,
    target_coverage: float,
) -> float:
    """Choose a conservative factor that meets the target empirical coverage."""
    if not rows:
        return 1.0
    required = sorted(float(row["required_inflation_factor"]) for row in rows)
    index = max(0, int(np.ceil(target_coverage * len(required))) - 1)
    return max(1.0, float(required[index]))


def _build_interval_calibration(
    support_bin_rows: dict[str, list[dict[str, float | bool]]],
    *,
    confidence_width_threshold: float,
    target_coverage: float = CALIBRATION_TARGET_COVERAGE,
    sparse_bin_min_events: int = CALIBRATION_SPARSE_BIN_MIN_EVENTS,
) -> dict:
    """Build a benchmark-layer support-binned interval calibration artifact."""
    ordered_labels = [label for _lower, _upper, label in SUPPORT_BINS]
    all_rows = [row for label in ordered_labels for row in support_bin_rows.get(label, [])]
    global_factor = _calibration_factor_for_rows(
        all_rows,
        target_coverage=target_coverage,
    )

    raw_support_bin_summary = {}
    calibrated_support_bin_summary = {}
    support_bin_factors = {}
    provisional_factors: dict[str, float] = {}
    factor_sources: dict[str, str] = {}

    for idx, label in enumerate(ordered_labels):
        rows = support_bin_rows.get(label, [])
        raw_support_bin_summary[label] = _summarize_rows(
            rows,
            factor=1.0,
            confidence_width_threshold=confidence_width_threshold,
            use_row_confident_flag=True,
        )
        if not rows:
            provisional_factors[label] = global_factor
            factor_sources[label] = "fallback:global_no_events"
            continue
        if len(rows) >= sparse_bin_min_events:
            provisional_factors[label] = _calibration_factor_for_rows(
                rows,
                target_coverage=target_coverage,
            )
            factor_sources[label] = "bin"
            continue

        fallback_label = None
        for next_label in ordered_labels[idx + 1:]:
            next_rows = support_bin_rows.get(next_label, [])
            if len(next_rows) >= sparse_bin_min_events:
                fallback_label = next_label
                provisional_factors[label] = _calibration_factor_for_rows(
                    next_rows,
                    target_coverage=target_coverage,
                )
                factor_sources[label] = f"fallback:{next_label}"
                break
        if fallback_label is None:
            provisional_factors[label] = global_factor
            factor_sources[label] = "fallback:global"

    running_factor = 1.0
    for label in ordered_labels:
        factor = max(running_factor, provisional_factors[label])
        if factor > provisional_factors[label]:
            factor_sources[label] = f"monotonic:{factor_sources[label]}"
        running_factor = factor
        rows = support_bin_rows.get(label, [])
        calibrated_support_bin_summary[label] = _summarize_rows(
            rows,
            factor=factor,
            confidence_width_threshold=confidence_width_threshold,
        )
        support_bin_factors[label] = {
            "inflation_factor": factor,
            "factor_source": factor_sources[label],
            "n_events": len(rows),
            "raw_ci_coverage": raw_support_bin_summary[label].get("ci_coverage"),
            "calibrated_ci_coverage": calibrated_support_bin_summary[label].get("ci_coverage"),
            "raw_median_ci_width": raw_support_bin_summary[label].get("median_ci_width"),
            "calibrated_median_ci_width": calibrated_support_bin_summary[label].get("median_ci_width"),
        }

    return {
        "target_coverage": target_coverage,
        "sparse_bin_min_events": sparse_bin_min_events,
        "support_bin_factors": support_bin_factors,
        "raw_overall_coverage": float(np.mean([_contains_after_inflation(row, 1.0) for row in all_rows]))
        if all_rows
        else None,
        "calibrated_overall_coverage": float(np.mean([
            _contains_after_inflation(
                row,
                float(support_bin_factors[str(row["support_bin"])]["inflation_factor"]),
            )
            for row in all_rows
        ]))
        if all_rows
        else None,
        "raw_confident_count": int(sum(1 for row in all_rows if row["raw_confident"])),
        "calibrated_confident_count": int(sum(
            1
            for row in all_rows
            if bool(row["raw_confident"])
            and float(row["ci_width"])
            * float(support_bin_factors[str(row["support_bin"])]["inflation_factor"])
            < (
                float(confidence_width_threshold.get(str(row["support_bin"]), 0.2))
                if isinstance(confidence_width_threshold, dict)
                else float(confidence_width_threshold)
            )
        )),
        "raw_support_bin_summary": raw_support_bin_summary,
        "calibrated_support_bin_summary": calibrated_support_bin_summary,
    }


def _evaluate_pacbio_row(
    row: dict[str, object],
    *,
    braid_bootstrap,
    base_scale: float,
    schedule_mode: str,
    calibration_schedule: dict[str, object] | None = None,
) -> dict[str, object]:
    """Evaluate one PacBio row under one posterior schedule."""
    psi, ci_low, ci_high, cv = braid_bootstrap.bootstrap_psi(
        int(row["inclusion_count"]),
        int(row["exclusion_count"]),
        n_replicates=500,
        confidence_level=0.95,
        seed=int(row["event_seed"]),
        model=braid_bootstrap.DEFAULT_UNCERTAINTY_MODEL,
        event_type=str(row["event_type"]),
        base_scale=base_scale,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    ci_width, is_confident = braid_bootstrap._is_confident_interval(
        ci_low,
        ci_high,
        cv,
        psi=psi,
        event_type=str(row["event_type"]),
        inclusion_count=int(row["inclusion_count"]),
        exclusion_count=int(row["exclusion_count"]),
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    effect_strength = abs(float(psi) - 0.5)
    effect_snr = effect_strength / max(ci_width, 1e-9)
    contains = ci_low <= float(row["lr_psi"]) <= ci_high
    confident_correct = abs(psi - float(row["lr_psi"])) < 0.2
    metadata = braid_bootstrap.effective_count_scale_metadata(
        int(row["inclusion_count"]),
        int(row["exclusion_count"]),
        event_type=str(row["event_type"]),
        base_scale=base_scale,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    return {
        **row,
        "braid_psi": psi,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": ci_width,
        "cv": cv,
        "effect_strength": effect_strength,
        "effect_snr": effect_snr,
        "contains": contains,
        "raw_confident": is_confident,
        "confident_correct": confident_correct,
        "confidence_success": contains and confident_correct,
        "required_inflation_factor": _required_interval_inflation_factor(
            psi=psi,
            ci_low=ci_low,
            ci_high=ci_high,
            truth=float(row["lr_psi"]),
        ),
        "effective_scale": float(metadata["effective_scale"]),
        "audit_interval_factor": float(metadata.get("audit_interval_factor", 1.0)),
        "confidence_width_threshold": float(
            braid_bootstrap.confidence_width_threshold(
                int(row["inclusion_count"]),
                int(row["exclusion_count"]),
                event_type=str(row["event_type"]),
                schedule_mode=schedule_mode,
                calibration_schedule=calibration_schedule,
            )
        ),
        "confidence_cv_threshold": float(
            braid_bootstrap.confidence_cv_threshold(
                int(row["inclusion_count"]),
                int(row["exclusion_count"]),
                event_type=str(row["event_type"]),
                schedule_mode=schedule_mode,
                calibration_schedule=calibration_schedule,
            )
        ),
        "confidence_effect_threshold": braid_bootstrap.confidence_effect_threshold(
            int(row["inclusion_count"]),
            int(row["exclusion_count"]),
            event_type=str(row["event_type"]),
            schedule_mode=schedule_mode,
            calibration_schedule=calibration_schedule,
        ),
        "confidence_effect_snr_threshold": braid_bootstrap.confidence_effect_snr_threshold(
            int(row["inclusion_count"]),
            int(row["exclusion_count"]),
            event_type=str(row["event_type"]),
            schedule_mode=schedule_mode,
            calibration_schedule=calibration_schedule,
        ),
        "confidence_effect_cv_threshold": braid_bootstrap.confidence_effect_cv_threshold(
            int(row["inclusion_count"]),
            int(row["exclusion_count"]),
            event_type=str(row["event_type"]),
            schedule_mode=schedule_mode,
            calibration_schedule=calibration_schedule,
        ),
        "interval_inflation_factor": float(
            braid_bootstrap.native_interval_inflation_factor(
                int(row["inclusion_count"]),
                int(row["exclusion_count"]),
                event_type=str(row["event_type"]),
                schedule_mode=schedule_mode,
                calibration_schedule=calibration_schedule,
            )
        ),
    }


def _evaluate_pacbio_rows(
    rows: list[dict[str, object]],
    *,
    braid_bootstrap,
    base_scale: float,
    schedule_mode: str,
    calibration_schedule: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Evaluate multiple PacBio rows under one posterior schedule."""
    return [
        _evaluate_pacbio_row(
            row,
            braid_bootstrap=braid_bootstrap,
            base_scale=base_scale,
            schedule_mode=schedule_mode,
            calibration_schedule=calibration_schedule,
        )
        for row in rows
    ]


def _rows_for_bin_with_fallback(
    rows_by_bin: dict[str, list[dict[str, object]]],
    label: str,
    *,
    sparse_bin_min_events: int,
) -> tuple[list[dict[str, object]], str]:
    """Return rows for one support bin or a deterministic fallback pool."""
    ordered_labels = [name for _lower, _upper, name in SUPPORT_BINS]
    rows = rows_by_bin.get(label, [])
    if len(rows) >= sparse_bin_min_events:
        return rows, f"bin:{label}"

    start_idx = ordered_labels.index(label)
    for next_label in ordered_labels[start_idx + 1:]:
        next_rows = rows_by_bin.get(next_label, [])
        if len(next_rows) >= sparse_bin_min_events:
            return next_rows, f"fallback:{next_label}"

    global_rows = [row for rows_here in rows_by_bin.values() for row in rows_here]
    return global_rows, "fallback:global"


def _confidence_threshold_for_rows(
    rows: list[dict[str, object]],
    *,
    target_accuracy: float,
    min_confident_events: int,
) -> float | None:
    """Choose the largest CI-width threshold that preserves confident accuracy."""
    if not rows:
        return None
    candidates = sorted({float(row["ci_width"]) for row in rows})
    best_threshold = None
    for threshold in candidates:
        selected = [row for row in rows if float(row["ci_width"]) <= threshold]
        if len(selected) < min_confident_events:
            continue
        success_rate = float(np.mean([bool(row["confidence_success"]) for row in selected]))
        if success_rate >= target_accuracy:
            best_threshold = threshold
    return best_threshold


def _confidence_gate_for_rows(
    rows: list[dict[str, object]],
    *,
    target_accuracy: float,
    min_confident_events: int,
    default_width: float,
    default_cv: float,
) -> tuple[float, float] | None:
    """Choose width and CV gates that maximize confident events under an accuracy target."""
    if not rows:
        return None
    widths = sorted({float(row["ci_width"]) for row in rows})
    cvs = sorted({
        float(row["cv"])
        for row in rows
        if np.isfinite(float(row["cv"]))
    })
    best: tuple[int, float, float, float] | None = None
    for width in widths:
        width_rows = [row for row in rows if float(row["ci_width"]) <= width]
        if len(width_rows) < min_confident_events:
            continue
        for cv in cvs:
            selected = [row for row in width_rows if float(row["cv"]) <= cv]
            if len(selected) < min_confident_events:
                continue
            accuracy = float(np.mean([bool(row["confidence_success"]) for row in selected]))
            if accuracy < target_accuracy:
                continue
            candidate = (len(selected), accuracy, -width, -cv)
            if best is None or candidate > (best[0], best[3], -best[1], -best[2]):
                best = (len(selected), width, cv, accuracy)
    if best is None:
        return None
    width, cv = best[1], best[2]
    return max(default_width, float(width)), min(default_cv, float(cv))


def _effect_gate_for_rows(
    rows: list[dict[str, object]],
    *,
    target_accuracy: float,
    min_confident_events: int,
) -> dict[str, float | bool] | None:
    """Choose an effect-aware override gate for mid-support bins."""
    if not rows:
        return None
    effect_values = sorted({
        float(row["effect_strength"])
        for row in rows
        if np.isfinite(float(row["effect_strength"]))
    })
    snr_values = sorted({
        float(row["effect_snr"])
        for row in rows
        if np.isfinite(float(row["effect_snr"]))
    })
    cv_values = sorted({
        float(row["cv"])
        for row in rows
        if np.isfinite(float(row["cv"]))
    })
    if not effect_values or not snr_values:
        return None

    best: dict[str, float | bool] | None = None

    def _consider(selected: list[dict[str, object]], *, effect_threshold: float, snr_threshold: float, cv_threshold: float | None) -> None:
        nonlocal best
        if len(selected) < min_confident_events:
            return
        accuracy = float(np.mean([bool(row["confidence_success"]) for row in selected]))
        if accuracy < target_accuracy:
            return
        candidate = {
            "selected_count": float(len(selected)),
            "accuracy": accuracy,
            "effect_threshold": float(effect_threshold),
            "snr_threshold": float(snr_threshold),
            "cv_threshold": None if cv_threshold is None else float(cv_threshold),
            "uses_cv": cv_threshold is not None,
        }
        if best is None:
            best = candidate
            return
        best_key = (
            int(best["selected_count"]),
            float(best["accuracy"]),
            1 if bool(best["uses_cv"]) else 0,
            float(best["effect_threshold"]),
            float(best["snr_threshold"]),
            -1.0 if best["cv_threshold"] is None else -float(best["cv_threshold"]),
        )
        cand_key = (
            int(candidate["selected_count"]),
            float(candidate["accuracy"]),
            1 if bool(candidate["uses_cv"]) else 0,
            float(candidate["effect_threshold"]),
            float(candidate["snr_threshold"]),
            -1.0 if candidate["cv_threshold"] is None else -float(candidate["cv_threshold"]),
        )
        if cand_key > best_key:
            best = candidate

    for effect_threshold in effect_values:
        for snr_threshold in snr_values:
            effect_selected = [
                row
                for row in rows
                if float(row["effect_strength"]) >= effect_threshold
                and float(row["effect_snr"]) >= snr_threshold
            ]
            _consider(
                effect_selected,
                effect_threshold=effect_threshold,
                snr_threshold=snr_threshold,
                cv_threshold=None,
            )
            for cv_threshold in cv_values:
                selected = [
                    row
                    for row in effect_selected
                    if float(row["cv"]) <= cv_threshold
                ]
                _consider(
                    selected,
                    effect_threshold=effect_threshold,
                    snr_threshold=snr_threshold,
                    cv_threshold=cv_threshold,
                )
    return best


def _derive_native_schedule_from_training_rows(
    training_rows: list[dict[str, object]],
    *,
    braid_bootstrap,
    base_scale: float,
    target_coverage: float = CALIBRATION_TARGET_COVERAGE,
    sparse_bin_min_events: int = CALIBRATION_SPARSE_BIN_MIN_EVENTS,
    target_confident_accuracy: float = 1.0,
    min_confident_events: int = 3,
) -> dict[str, object]:
    """Derive one support-aware native schedule from held-in PacBio rows."""
    ordered_labels = [label for _lower, _upper, label in SUPPORT_BINS]
    default_threshold = float(braid_bootstrap.CONFIDENT_CI_WIDTH_THRESHOLD)

    fixed_rows = _evaluate_pacbio_rows(
        training_rows,
        braid_bootstrap=braid_bootstrap,
        base_scale=base_scale,
        schedule_mode="fixed",
    )
    fixed_by_bin = {label: [] for label in ordered_labels}
    for row in fixed_rows:
        fixed_by_bin[str(row["support_bin"])].append(row)

    provisional_audit: dict[str, float] = {}
    audit_source: dict[str, str] = {}
    for label in ordered_labels:
        source_rows, source = _rows_for_bin_with_fallback(
            fixed_by_bin,
            label,
            sparse_bin_min_events=sparse_bin_min_events,
        )
        provisional_audit[label] = _calibration_factor_for_rows(
            source_rows,
            target_coverage=target_coverage,
        ) if source_rows else 1.0
        audit_source[label] = source

    audit_by_bin: dict[str, float] = {}
    running_audit = 1.0
    for label in ordered_labels:
        audit_factor = max(running_audit, provisional_audit[label])
        running_audit = audit_factor
        audit_by_bin[label] = audit_factor

    scale_by_bin = {
        label: min(1.0, base_scale / (audit_by_bin[label] ** 2))
        for label in ordered_labels
    }
    interval_placeholder = {label: 1.0 for label in ordered_labels}
    confidence_placeholder = {label: default_threshold for label in ordered_labels}
    scaled_schedule = {
        "mode": "leave_one_gene_out_native_schedule",
        "training_scope": "leave_one_gene_out",
        "base_scale": base_scale,
        "scale_by_bin": scale_by_bin,
        "audit_interval_factor_by_bin": audit_by_bin,
        "interval_inflation_by_bin": interval_placeholder,
        "confidence_width_by_bin": confidence_placeholder,
    }

    scaled_rows = _evaluate_pacbio_rows(
        training_rows,
        braid_bootstrap=braid_bootstrap,
        base_scale=base_scale,
        schedule_mode="native",
        calibration_schedule=scaled_schedule,
    )
    scaled_by_bin = {label: [] for label in ordered_labels}
    for row in scaled_rows:
        scaled_by_bin[str(row["support_bin"])].append(row)

    provisional_interval: dict[str, float] = {}
    interval_source: dict[str, str] = {}
    for label in ordered_labels:
        source_rows, source = _rows_for_bin_with_fallback(
            scaled_by_bin,
            label,
            sparse_bin_min_events=sparse_bin_min_events,
        )
        provisional_interval[label] = _calibration_factor_for_rows(
            source_rows,
            target_coverage=target_coverage,
        ) if source_rows else 1.0
        interval_source[label] = source

    interval_by_bin: dict[str, float] = {}
    running_interval = 1.0
    for label in ordered_labels:
        factor = max(running_interval, provisional_interval[label])
        running_interval = factor
        interval_by_bin[label] = factor

    interval_schedule = dict(scaled_schedule)
    interval_schedule["interval_inflation_by_bin"] = interval_by_bin
    interval_rows = _evaluate_pacbio_rows(
        training_rows,
        braid_bootstrap=braid_bootstrap,
        base_scale=base_scale,
        schedule_mode="native",
        calibration_schedule=interval_schedule,
    )
    interval_by_bin_rows = {label: [] for label in ordered_labels}
    for row in interval_rows:
        interval_by_bin_rows[str(row["support_bin"])].append(row)

    provisional_threshold: dict[str, float] = {}
    threshold_source: dict[str, str] = {}
    provisional_cv_threshold: dict[str, float] = {}
    cv_threshold_source: dict[str, str] = {}
    provisional_effect_threshold: dict[str, float] = {}
    effect_threshold_source: dict[str, str] = {}
    provisional_effect_snr_threshold: dict[str, float] = {}
    effect_snr_threshold_source: dict[str, str] = {}
    provisional_effect_cv_threshold: dict[str, float] = {}
    effect_cv_threshold_source: dict[str, str] = {}
    for label in ordered_labels:
        if label not in {"100-249", "250+"}:
            provisional_threshold[label] = default_threshold
            threshold_source[label] = "fixed_default"
            provisional_cv_threshold[label] = float(braid_bootstrap.CONFIDENT_CV_THRESHOLD)
            cv_threshold_source[label] = "fixed_default"
            continue
        source_rows, source = _rows_for_bin_with_fallback(
            interval_by_bin_rows,
            label,
            sparse_bin_min_events=sparse_bin_min_events,
        )
        gate = _confidence_gate_for_rows(
            source_rows,
            target_accuracy=target_confident_accuracy,
            min_confident_events=min_confident_events,
            default_width=default_threshold,
            default_cv=float(braid_bootstrap.CONFIDENT_CV_THRESHOLD),
        )
        if gate is None:
            provisional_threshold[label] = default_threshold
            threshold_source[label] = f"default:{source}"
            provisional_cv_threshold[label] = float(braid_bootstrap.CONFIDENT_CV_THRESHOLD)
            cv_threshold_source[label] = f"default:{source}"
        else:
            provisional_threshold[label] = float(gate[0])
            threshold_source[label] = source
            provisional_cv_threshold[label] = float(gate[1])
            cv_threshold_source[label] = source

    for label in ordered_labels:
        if label not in {"100-249"}:
            continue
        source_rows, source = _rows_for_bin_with_fallback(
            interval_by_bin_rows,
            label,
            sparse_bin_min_events=sparse_bin_min_events,
        )
        effect_gate = _effect_gate_for_rows(
            source_rows,
            target_accuracy=target_confident_accuracy,
            min_confident_events=1,
        )
        if effect_gate is None:
            continue
        provisional_effect_threshold[label] = float(effect_gate["effect_threshold"])
        effect_threshold_source[label] = source
        provisional_effect_snr_threshold[label] = float(effect_gate["snr_threshold"])
        effect_snr_threshold_source[label] = source
        cv_override = effect_gate["cv_threshold"]
        if cv_override is not None:
            provisional_effect_cv_threshold[label] = float(cv_override)
            effect_cv_threshold_source[label] = source

    confidence_by_bin: dict[str, float] = {}
    confidence_cv_by_bin: dict[str, float] = {}
    running_threshold = 0.0
    running_cv_threshold = float(braid_bootstrap.CONFIDENT_CV_THRESHOLD)
    for label in ordered_labels:
        threshold = max(running_threshold, provisional_threshold[label])
        running_threshold = threshold
        confidence_by_bin[label] = threshold
        cv_threshold = min(running_cv_threshold, provisional_cv_threshold[label])
        running_cv_threshold = cv_threshold
        confidence_cv_by_bin[label] = cv_threshold

    confidence_effect_by_bin: dict[str, float] = {}
    confidence_effect_snr_by_bin: dict[str, float] = {}
    confidence_effect_cv_by_bin: dict[str, float] = {}
    running_effect_threshold = float("inf")
    running_effect_snr_threshold = float("inf")
    running_effect_cv_threshold = float("inf")
    for label in ordered_labels:
        if label in provisional_effect_threshold:
            running_effect_threshold = min(
                running_effect_threshold,
                provisional_effect_threshold[label],
            )
            confidence_effect_by_bin[label] = running_effect_threshold
        if label in provisional_effect_snr_threshold:
            running_effect_snr_threshold = min(
                running_effect_snr_threshold,
                provisional_effect_snr_threshold[label],
            )
            confidence_effect_snr_by_bin[label] = running_effect_snr_threshold
        if label in provisional_effect_cv_threshold:
            running_effect_cv_threshold = min(
                running_effect_cv_threshold,
                provisional_effect_cv_threshold[label],
            )
            confidence_effect_cv_by_bin[label] = running_effect_cv_threshold

    return {
        "mode": "leave_one_gene_out_native_schedule",
        "training_scope": "leave_one_gene_out",
        "base_scale": base_scale,
        "scale_by_bin": scale_by_bin,
        "audit_interval_factor_by_bin": audit_by_bin,
        "interval_inflation_by_bin": interval_by_bin,
        "confidence_width_by_bin": confidence_by_bin,
        "confidence_cv_by_bin": confidence_cv_by_bin,
        "confidence_effect_by_bin": confidence_effect_by_bin,
        "confidence_effect_snr_by_bin": confidence_effect_snr_by_bin,
        "confidence_effect_cv_by_bin": confidence_effect_cv_by_bin,
        "scale_source_by_bin": audit_source,
        "interval_source_by_bin": interval_source,
        "confidence_source_by_bin": threshold_source,
        "confidence_cv_source_by_bin": cv_threshold_source,
        "confidence_effect_source_by_bin": effect_threshold_source,
        "confidence_effect_snr_source_by_bin": effect_snr_threshold_source,
        "confidence_effect_cv_source_by_bin": effect_cv_threshold_source,
    }


def _summarize_crossfit_schedule(
    fold_schedules: dict[str, dict[str, object]],
    *,
    value_key: str,
    summary_key: str,
    mode: str,
    extra_key: str | None = None,
    extra_summary_key: str | None = None,
    labels: list[str] | None = None,
) -> dict[str, dict[str, float | int | str]]:
    """Summarize one per-gene calibration schedule into a reviewer-facing block."""
    ordered_labels = labels or [label for _lower, _upper, label in SUPPORT_BINS]
    summary: dict[str, dict[str, float | int | str]] = {}
    for label in ordered_labels:
        values = [
            float(schedule.get(value_key, {}).get(label, 0.0))
            for schedule in fold_schedules.values()
            if label in (schedule.get(value_key, {}) or {})
        ]
        if not values:
            continue
        block: dict[str, float | int | str] = {
            summary_key: float(median(values)),
            f"{summary_key}_min": float(min(values)),
            f"{summary_key}_max": float(max(values)),
            "n_folds": len(values),
            "mode": mode,
            "training_scope": "leave_one_gene_out",
        }
        if extra_key and extra_summary_key:
            extra_values = [
                float(schedule.get(extra_key, {}).get(label, 0.0))
                for schedule in fold_schedules.values()
                if label in (schedule.get(extra_key, {}) or {})
            ]
            if extra_values:
                block[extra_summary_key] = float(median(extra_values))
                block[f"{extra_summary_key}_min"] = float(min(extra_values))
                block[f"{extra_summary_key}_max"] = float(max(extra_values))
        summary[label] = block
    return summary


def benchmark_pacbio_junction_validation(
    *,
    count_scale: float | None = None,
) -> dict:
    """Benchmark BRAID PSI against PacBio long-read junction evidence.

    For AS events where we have both BRAID PSI (from short reads)
    and PacBio junction counts, compare PSI values and check if
    BRAID CI contains the PacBio-estimated PSI.
    """
    from rapidsplice.target import psi_bootstrap as braid_bootstrap
    from rapidsplice.target.psi_bootstrap import compute_psi_from_junctions
    from rapidsplice.target.extractor import lookup_gene
    import pysam

    original_scale = braid_bootstrap.OVERDISPERSED_COUNT_SCALE
    if count_scale is not None:
        braid_bootstrap.OVERDISPERSED_COUNT_SCALE = count_scale

    print("=" * 60)
    print("  Benchmark: BRAID PSI vs PacBio Long-Read")
    print("=" * 60)

    GTF = "real_benchmark/annotation/gencode.v38.nochr.gtf"
    BAM = "real_benchmark/bam/SRR387661.bam"
    LR_GTF = "real_benchmark/longread/ENCFF652QLH.nochr.gtf"

    if count_scale is not None:
        print(f"  Overdispersed count scale: {count_scale:.3f}")

    try:
        if not os.path.exists(LR_GTF):
            print("  SKIP: PacBio GTF not found")
            return {}

        # Build long-read junction counts
        print("  Loading PacBio junctions...")
        lr_junctions: dict[tuple[str, int, int], int] = {}
        tx_exons: dict[str, list[tuple[str, int, int]]] = {}
        with open(LR_GTF) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                fields = line.strip().split("\t")
                if len(fields) < 9 or fields[2] != "exon":
                    continue
                chrom = fields[0]
                start = int(fields[3]) - 1
                end = int(fields[4])
                attrs = fields[8]
                tid = None
                for part in attrs.split(";"):
                    part = part.strip()
                    if part.startswith("transcript_id"):
                        tid = part.split('"')[1]
                if tid:
                    tx_exons.setdefault(tid, []).append((chrom, start, end))

        for tid, exons in tx_exons.items():
            exons.sort(key=lambda x: x[1])
            for i in range(len(exons) - 1):
                chrom = exons[i][0]
                donor = exons[i][2]
                acceptor = exons[i + 1][1]
                key = (chrom, donor, acceptor)
                lr_junctions[key] = lr_junctions.get(key, 0) + 1

        print(f"  PacBio: {len(lr_junctions):,} unique junctions")

        # Compare BRAID PSI vs PacBio PSI for shared junctions
        GENES = [
            "TP53", "BRCA1", "EZH2", "BRAF", "AKT1", "PTEN", "RUNX1",
            "STAT3", "KRAS", "MYC", "BCR", "NRAS", "VHL", "ABL1", "MDM2",
            "DNMT3A", "TET2", "BAX", "SUZ12", "GATA2",
        ]

        raw_rows: list[dict[str, object]] = []
        event_type_counts: dict[str, int] = {}

        for gene in GENES:
            region = lookup_gene(GTF, gene)
            if not region:
                continue

            try:
                braid_results = compute_psi_from_junctions(
                    BAM,
                    region.chrom,
                    region.start,
                    region.end,
                    gene=gene,
                    n_replicates=500,
                    confidence_level=0.95,
                    seed=42,
                    schedule_mode="fixed",
                )
            except Exception:
                continue

            for r in braid_results:
                if r.event_type not in {"A3SS", "A5SS"}:
                    continue
                if r.event_start is None or r.event_end is None:
                    continue
                donor = r.event_start
                acceptor = r.event_end

                key = (region.chrom, donor, acceptor)
                if key not in lr_junctions:
                    continue

                # Compute PacBio PSI for this junction
                # (fraction of long-read transcripts using this junction
                #  vs all junctions from same donor)
                lr_count = lr_junctions[key]
                same_donor = [
                    (c, d, a)
                    for (c, d, a), cnt in lr_junctions.items()
                    if c == region.chrom and d == donor
                    and a != acceptor
                    and region.start <= d <= region.end
                ]
                lr_total = lr_count + sum(
                    lr_junctions.get(k, 0) for k in same_donor
                )

                if lr_total < 5:
                    continue

                lr_psi = lr_count / lr_total
                total_support = r.inclusion_count + r.exclusion_count
                support_bin = _classify_support_bin(total_support)
                event_seed = zlib.crc32(f"{gene}:{r.event_id}".encode("utf-8")) % (2**31 - 1)
                raw_rows.append({
                    "gene": gene,
                    "event_id": r.event_id,
                    "event_type": r.event_type,
                    "chrom": region.chrom,
                    "event_start": donor,
                    "event_end": acceptor,
                    "inclusion_count": r.inclusion_count,
                    "exclusion_count": r.exclusion_count,
                    "lr_psi": lr_psi,
                    "total_support": total_support,
                    "support_bin": support_bin,
                    "event_seed": event_seed,
                })
                event_type_counts[r.event_type] = (
                    event_type_counts.get(r.event_type, 0) + 1
                )

        if not raw_rows:
            print("  No matching events found")
            return {}

        legacy_rows = _evaluate_pacbio_rows(
            raw_rows,
            braid_bootstrap=braid_bootstrap,
            base_scale=braid_bootstrap.OVERDISPERSED_COUNT_SCALE,
            schedule_mode="legacy",
        )
        fold_schedules: dict[str, dict[str, object]] = {}
        native_rows: list[dict[str, object]] = []
        for gene in sorted({str(row["gene"]) for row in raw_rows}):
            heldout_rows = [row for row in raw_rows if row["gene"] == gene]
            training_rows = [row for row in raw_rows if row["gene"] != gene]
            if not training_rows:
                continue
            schedule = _derive_native_schedule_from_training_rows(
                training_rows,
                braid_bootstrap=braid_bootstrap,
                base_scale=braid_bootstrap.OVERDISPERSED_COUNT_SCALE,
            )
            fold_schedules[gene] = schedule
            native_rows.extend(_evaluate_pacbio_rows(
                heldout_rows,
                braid_bootstrap=braid_bootstrap,
                base_scale=braid_bootstrap.OVERDISPERSED_COUNT_SCALE,
                schedule_mode="native",
                calibration_schedule=schedule,
            ))

        n_events = len(native_rows)
        if n_events == 0:
            print("  No cross-fit events found")
            return {}

        braid_arr = np.array([float(row["braid_psi"]) for row in native_rows])
        lr_arr = np.array([float(row["lr_psi"]) for row in native_rows])

        correlation = np.corrcoef(braid_arr, lr_arr)[0, 1]
        r_squared = correlation ** 2
        ci_coverage = float(np.mean([bool(row["contains"]) for row in native_rows]))
        legacy_ci_coverage = float(np.mean([bool(row["contains"]) for row in legacy_rows]))
        mae = float(np.mean(np.abs(braid_arr - lr_arr)))
        confident_correct = [
            bool(row["confident_correct"])
            for row in native_rows
            if bool(row["raw_confident"])
        ]
        legacy_confident_correct = [
            bool(row["confident_correct"])
            for row in legacy_rows
            if bool(row["raw_confident"])
        ]
        confident_count = len(confident_correct)
        legacy_confident_count = len(legacy_confident_correct)
        support_bin_rows: dict[str, list[dict[str, object]]] = {
            label: []
            for _lower, _upper, label in SUPPORT_BINS
        }
        legacy_support_bin_rows: dict[str, list[dict[str, object]]] = {
            label: []
            for _lower, _upper, label in SUPPORT_BINS
        }
        for row in native_rows:
            support_bin_rows[str(row["support_bin"])].append(row)
        for row in legacy_rows:
            legacy_support_bin_rows[str(row["support_bin"])].append(row)

        native_scale_schedule = _summarize_crossfit_schedule(
            fold_schedules,
            value_key="scale_by_bin",
            summary_key="effective_scale",
            mode="leave_one_gene_out_native_support_schedule",
            extra_key="audit_interval_factor_by_bin",
            extra_summary_key="audit_interval_factor",
        )
        native_interval_schedule = _summarize_crossfit_schedule(
            fold_schedules,
            value_key="interval_inflation_by_bin",
            summary_key="interval_inflation_factor",
            mode="leave_one_gene_out_native_interval_schedule",
        )
        native_confidence_schedule = _summarize_crossfit_schedule(
            fold_schedules,
            value_key="confidence_width_by_bin",
            summary_key="confidence_width_threshold",
            mode="leave_one_gene_out_native_confidence_schedule",
        )
        native_confidence_cv_schedule = _summarize_crossfit_schedule(
            fold_schedules,
            value_key="confidence_cv_by_bin",
            summary_key="confidence_cv_threshold",
            mode="leave_one_gene_out_native_confidence_cv_schedule",
        )
        native_confidence_effect_schedule = _summarize_crossfit_schedule(
            fold_schedules,
            value_key="confidence_effect_by_bin",
            summary_key="confidence_effect_threshold",
            mode="leave_one_gene_out_native_confidence_effect_schedule",
            labels=["100-249"],
        )
        native_confidence_effect_snr_schedule = _summarize_crossfit_schedule(
            fold_schedules,
            value_key="confidence_effect_snr_by_bin",
            summary_key="confidence_effect_snr_threshold",
            mode="leave_one_gene_out_native_confidence_effect_snr_schedule",
            labels=["100-249"],
        )
        native_confidence_effect_cv_schedule = _summarize_crossfit_schedule(
            fold_schedules,
            value_key="confidence_effect_cv_by_bin",
            summary_key="confidence_effect_cv_threshold",
            mode="leave_one_gene_out_native_confidence_effect_cv_schedule",
            labels=["100-249"],
        )
        legacy_confidence_width_threshold = float(braid_bootstrap.CONFIDENT_CI_WIDTH_THRESHOLD)
        support_bin_summary = {}
        for label, rows in support_bin_rows.items():
            if not rows:
                continue
            support_bin_summary[label] = _summarize_rows(
                rows,
                factor=1.0,
                confidence_width_threshold=0.2,
                use_row_confident_flag=True,
            )
        legacy_support_bin_summary = {}
        for label, rows in legacy_support_bin_rows.items():
            if not rows:
                continue
            legacy_support_bin_summary[label] = _summarize_rows(
                rows,
                factor=1.0,
                confidence_width_threshold=legacy_confidence_width_threshold,
                use_row_confident_flag=True,
            )
        audit_posthoc = _build_interval_calibration(
            support_bin_rows,
            confidence_width_threshold=0.2,
        )
        calibration_protocol = {
            "type": "leave_one_gene_out",
            "training_scope": "gene_holdout",
            "base_schedule_mode": "fixed_global_scale",
            "target_coverage": CALIBRATION_TARGET_COVERAGE,
            "target_confident_accuracy": 1.0,
            "min_confident_events": 3,
            "effect_confidence_bins": ["100-249"],
            "effect_min_confident_events_by_bin": {
                "100-249": 1,
            },
            "n_genes": len(fold_schedules),
            "genes": sorted(fold_schedules),
            "event_types": sorted(event_type_counts),
        }

        print(f"\n  Results ({n_events} matched events):")
        print(f"  PSI correlation (R²):           {r_squared:.3f}")
        print(f"  PSI correlation (r):            {correlation:.3f}")
        print(f"  Mean absolute error:            {mae:.3f}")
        print(f"  Legacy global-scale coverage:   {legacy_ci_coverage:.1%}")
        print(f"  Native cross-fit CI coverage:   {ci_coverage:.1%}")
        if audit_posthoc.get("calibrated_overall_coverage") is not None:
            print(
                "  Audit post-hoc coverage:       "
                f"{float(audit_posthoc['calibrated_overall_coverage']):.1%}"
            )
        if event_type_counts:
            counts = ", ".join(
                f"{k}={v}" for k, v in sorted(event_type_counts.items())
            )
            print(f"  Event types:                    {counts}")
        print(f"  Legacy confident events:        {legacy_confident_count}")
        print(f"  Native cross-fit confident:     {confident_count}")
        print(
            "  Audit-posthoc confident:       "
            f"{int(audit_posthoc.get('calibrated_confident_count', 0))}"
        )
        if confident_correct:
            conf_acc = np.mean(confident_correct)
            print(f"  Native confident accuracy:      {conf_acc:.1%}")
        if legacy_confident_correct:
            legacy_conf_acc = np.mean(legacy_confident_correct)
            print(f"  Legacy confident accuracy:      {legacy_conf_acc:.1%}")

        return {
            "count_scale": float(braid_bootstrap.OVERDISPERSED_COUNT_SCALE),
            "schedule_mode": "leave_one_gene_out_native_support_schedule",
            "calibration_protocol": calibration_protocol,
            "n_events": n_events,
            "r_squared": float(r_squared),
            "correlation": float(correlation),
            "mae": float(mae),
            "ci_coverage": float(ci_coverage),
            "native_ci_coverage": float(ci_coverage),
            "legacy_ci_coverage": float(legacy_ci_coverage),
            "event_type_counts": event_type_counts,
            "confident_count": confident_count,
            "confident_accuracy": float(np.mean(confident_correct))
            if confident_correct
            else None,
            "support_bin_summary": support_bin_summary,
            "native_support_bin_summary": support_bin_summary,
            "native_support_scale_schedule": native_scale_schedule,
            "native_interval_inflation_schedule": native_interval_schedule,
            "native_confidence_width_schedule": native_confidence_schedule,
            "native_confidence_cv_schedule": native_confidence_cv_schedule,
            "native_confidence_effect_schedule": native_confidence_effect_schedule,
            "native_confidence_effect_snr_schedule": native_confidence_effect_snr_schedule,
            "native_confidence_effect_cv_schedule": native_confidence_effect_cv_schedule,
            "legacy_global_scale_summary": {
                "schedule_mode": "legacy_support_heuristic",
                "ci_coverage": float(legacy_ci_coverage),
                "confident_count": legacy_confident_count,
                "confident_accuracy": float(np.mean(legacy_confident_correct))
                if legacy_confident_correct
                else None,
                "support_bin_summary": legacy_support_bin_summary,
            },
            "audit_posthoc_calibration": audit_posthoc,
            "calibration": audit_posthoc,
        }
    finally:
        braid_bootstrap.OVERDISPERSED_COUNT_SCALE = original_scale


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--count-scale",
        type=float,
        default=None,
        help="Override BRAID overdispersed count scale for this run only.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/rtpcr_benchmark.json",
        help="Path to the JSON output file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all RT-PCR benchmarks."""
    args = parse_args()
    results = {}

    # Benchmark 1: PacBio long-read PSI validation
    r = benchmark_pacbio_junction_validation(count_scale=args.count_scale)
    if r:
        results["pacbio_psi"] = r

    # Save
    output = args.output
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
