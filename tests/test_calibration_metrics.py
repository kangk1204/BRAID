"""Tests for deconfounded calibration / sharpness metrics."""
from __future__ import annotations

import numpy as np

from braid.target.calibration_metrics import (
    empirical_coverage,
    estimate_centered_coverage,
    interval_score,
    random_centered_coverage,
    sharpness_report,
)


def test_coverage_basic() -> None:
    low = np.array([0.0, 0.4, 0.8])
    high = np.array([0.2, 0.6, 1.0])
    truth = np.array([0.1, 0.5, 0.5])  # first two covered, third not
    assert empirical_coverage(low, high, truth) == 2 / 3


def test_interval_score_rewards_sharp_covering_intervals() -> None:
    truth = np.array([0.5, 0.5, 0.5])
    # Tight interval that covers
    tight = interval_score(np.array([0.45, 0.45, 0.45]), np.array([0.55, 0.55, 0.55]), truth)
    # Trivially wide [0,1] interval (always covers but not sharp)
    wide = interval_score(np.zeros(3), np.ones(3), truth)
    assert tight < wide  # proper scoring rule favours the sharp covering interval

    # A tight interval that MISSES is heavily penalized vs the wide one
    missing = interval_score(np.array([0.0, 0.0, 0.0]), np.array([0.1, 0.1, 0.1]), truth)
    assert missing > wide


def test_estimate_centered_baseline() -> None:
    estimate = np.array([0.5, 0.5])
    truth = np.array([0.55, 0.9])  # within 0.1 of estimate? 0.05 yes, 0.4 no
    width = np.array([0.2, 0.2])  # half-width 0.1
    assert estimate_centered_coverage(estimate, width, truth) == 0.5


def test_deconfound_exposes_point_estimate_contribution() -> None:
    """A wide interval centred on a good estimate covers well, but the fair
    estimate-centred baseline covers just as well -> deconfounded lift ~ 0,
    revealing that the coverage comes from the estimate, not the width shape."""
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.2, 0.8, size=500)
    estimate = np.clip(truth + rng.normal(0, 0.03, size=500), 0, 1)
    width = np.full(500, 0.5)
    low = np.clip(estimate - width / 2, 0, 1)
    high = np.clip(estimate + width / 2, 0, 1)

    rep = sharpness_report(low, high, estimate, truth)
    # BRAID-like interval and the fair estimate-centred baseline are identical
    # here (symmetric), so the deconfounded lift is ~0...
    assert abs(rep.lift_over_estimate_centered) < 1e-9
    # ...while the lift over a *randomly* centred interval is large, which is the
    # misleading number the original analysis reported.
    assert rep.lift_over_random > 0.2


def test_random_centered_coverage_is_deterministic() -> None:
    width = np.full(200, 0.4)
    truth = np.random.default_rng(1).uniform(0, 1, size=200)
    a = random_centered_coverage(width, truth, seed=7)
    b = random_centered_coverage(width, truth, seed=7)
    assert a == b
