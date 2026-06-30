"""Defensive NaN-safety tests for the conformal interval constructors.

`half = q * max(float(sigma), 0.0)` does NOT sanitize a NaN sigma, because
`max(nan, 0.0)` returns nan — which would propagate to a `(nan, nan)` interval,
violating `ci_low <= ci_high` and the project's NaN-containment rule. Production
sigma is usually finite (Beta posterior std), so this is a latent gap; the guard
maps a non-finite point estimate or spread to the conservative full clip range and
must leave every finite-input result byte-identical.
"""

from __future__ import annotations

import numpy as np

from braid.target.conformal import (
    ConformalCalibrator,
    mondrian_conformal_intervals,
    split_conformal_intervals,
)


def _calib() -> ConformalCalibrator:
    return ConformalCalibrator(
        alpha=0.05, q_global=2.0, q_by_bin={"<20": 1.5},
        bin_edges=(20, 50, 100, 250), q_by_event_type={},
    )


def test_interval_nan_sigma_returns_full_clip_not_nan():
    lo, hi = _calib().interval(0.5, float("nan"), 10.0)
    assert (lo, hi) == (0.0, 1.0)


def test_interval_nan_psi_returns_full_clip_not_nan():
    lo, hi = _calib().interval(float("nan"), 0.1, 10.0)
    assert (lo, hi) == (0.0, 1.0)


def test_robust_interval_nan_sampling_std_returns_full_clip():
    # The differential ΔPSI path passes clip=(-1, 1); a NaN spread must yield
    # that full conservative range, not a NaN interval.
    lo, hi = _calib().robust_interval(0.0, float("nan"), 500.0, clip=(-1.0, 1.0))
    assert (lo, hi) == (-1.0, 1.0)


def test_robust_interval_nan_psi_returns_full_clip_not_nan():
    lo, hi = _calib().robust_interval(
        float("nan"), 0.1, 500.0, clip=(-1.0, 1.0)
    )
    assert (lo, hi) == (-1.0, 1.0)


def test_interval_finite_sigma_unchanged():
    """Finite inputs must be identical to the closed form (no behavior drift)."""
    c = _calib()
    lo, hi = c.interval(0.5, 0.1, 10.0)  # bin '<20' -> q=1.5; half=0.15
    assert abs(lo - 0.35) < 1e-9
    assert abs(hi - 0.65) < 1e-9


def test_interval_finite_always_ordered_and_in_unit():
    c = _calib()
    for psi in (0.0, 0.3, 0.5, 0.9, 1.0):
        for sig in (0.0, 0.05, 0.5, 5.0):
            lo, hi = c.interval(psi, sig, 30.0)
            assert 0.0 <= lo <= hi <= 1.0
            assert np.isfinite(lo) and np.isfinite(hi)


def test_split_conformal_nan_test_scale_returns_full_clip_not_nan():
    intervals = split_conformal_intervals(
        cal_estimates=np.array([0.4, 0.6, 0.5]),
        cal_truth=np.array([0.45, 0.55, 0.5]),
        cal_scale=np.array([0.1, 0.1, 0.1]),
        test_estimates=np.array([0.5]),
        test_scale=np.array([np.nan]),
        alpha=0.5,
    )
    assert intervals.low[0] == 0.0
    assert intervals.high[0] == 1.0
    assert np.isfinite(intervals.low[0]) and np.isfinite(intervals.high[0])


def test_mondrian_conformal_nan_test_scale_returns_full_clip_not_nan():
    intervals = mondrian_conformal_intervals(
        cal_estimates=np.array([0.4, 0.6, 0.5]),
        cal_truth=np.array([0.45, 0.55, 0.5]),
        cal_scale=np.array([0.1, 0.1, 0.1]),
        cal_bins=np.array(["<20", "<20", "<20"], dtype=object),
        test_estimates=np.array([0.5]),
        test_scale=np.array([np.nan]),
        test_bins=np.array(["<20"], dtype=object),
        alpha=0.5,
    )
    assert intervals.low[0] == 0.0
    assert intervals.high[0] == 1.0
    assert np.isfinite(intervals.low[0]) and np.isfinite(intervals.high[0])
