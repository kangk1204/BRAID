"""Tests for PacBio benchmark calibration helpers."""

from __future__ import annotations

import pytest

from benchmarks import rtpcr_benchmark


def _row(*, truth: float, raw_confident: bool = True) -> dict[str, float | bool | str]:
    return {
        "support_bin": "unused",
        "braid_psi": 0.5,
        "ci_low": 0.4,
        "ci_high": 0.6,
        "ci_width": 0.2,
        "lr_psi": truth,
        "required_inflation_factor": rtpcr_benchmark._required_interval_inflation_factor(
            psi=0.5,
            ci_low=0.4,
            ci_high=0.6,
            truth=truth,
        ),
        "raw_confident": raw_confident,
        "confident_correct": True,
    }


def test_required_interval_inflation_factor_is_one_when_truth_is_inside() -> None:
    assert rtpcr_benchmark._required_interval_inflation_factor(
        psi=0.5,
        ci_low=0.4,
        ci_high=0.6,
        truth=0.55,
    ) == pytest.approx(1.0)


def test_build_interval_calibration_uses_sparse_fallback_and_monotonic_factors() -> None:
    support_bin_rows = {
        "<20": [_row(truth=0.55) for _ in range(5)],
        "20-49": [_row(truth=0.65) for _ in range(10)],
        "50-99": [_row(truth=0.62) for _ in range(10)],
        "100-249": [],
        "250+": [],
    }
    for row in support_bin_rows["<20"]:
        row["support_bin"] = "<20"
    for row in support_bin_rows["20-49"]:
        row["support_bin"] = "20-49"
    for row in support_bin_rows["50-99"]:
        row["support_bin"] = "50-99"

    calibration = rtpcr_benchmark._build_interval_calibration(
        support_bin_rows,
        confidence_width_threshold=0.2,
        target_coverage=0.95,
        sparse_bin_min_events=10,
    )

    factors = calibration["support_bin_factors"]
    assert factors["<20"]["factor_source"] == "fallback:20-49"
    assert factors["<20"]["inflation_factor"] == pytest.approx(1.5)
    assert factors["50-99"]["inflation_factor"] == pytest.approx(1.5)
    assert factors["50-99"]["factor_source"].startswith("monotonic:")
    assert calibration["calibrated_overall_coverage"] >= calibration["raw_overall_coverage"]


def test_effect_gate_for_rows_prefers_high_effect_high_snr_subset() -> None:
    rows = [
        {
            "effect_strength": 0.49,
            "effect_snr": 0.55,
            "cv": 0.90,
            "confidence_success": True,
        },
        {
            "effect_strength": 0.48,
            "effect_snr": 0.52,
            "cv": 0.94,
            "confidence_success": True,
        },
        {
            "effect_strength": 0.47,
            "effect_snr": 0.46,
            "cv": 0.32,
            "confidence_success": True,
        },
        {
            "effect_strength": 0.20,
            "effect_snr": 0.20,
            "cv": 0.40,
            "confidence_success": False,
        },
    ]

    gate = rtpcr_benchmark._effect_gate_for_rows(
        rows,
        target_accuracy=1.0,
        min_confident_events=2,
    )

    assert gate is not None
    assert gate["selected_count"] >= 2
    assert gate["effect_threshold"] >= 0.47
    assert gate["snr_threshold"] >= 0.46
