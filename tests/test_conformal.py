"""Tests for conformal PSI prediction intervals.

The headline property is the distribution-free, finite-sample marginal coverage
guarantee: averaged over calibration/test draws, a 1-alpha conformal interval
covers the truth with probability >= 1-alpha -- and does so without being
trivially wide. These tests verify both halves on synthetic data.
"""
from __future__ import annotations

import numpy as np
import pytest

from braid.target.conformal import (
    ConformalCalibrator,
    assign_support_bins,
    conformal_quantile,
    fit_conformal_calibrator,
    mondrian_conformal_intervals,
    nonconformity_scores,
    split_conformal_intervals,
)


def _synthetic(seed: int, n: int):
    """Generate (estimate, truth, scale, support) with a known per-event noise sd."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.1, 0.9, size=n)
    support = rng.integers(5, 600, size=n).astype(float)
    # smaller scale at higher support; bounded so intervals stay informative
    scale = np.clip(0.6 / np.sqrt(support), 0.02, 0.25)
    estimate = np.clip(truth + rng.normal(0.0, scale), 0.0, 1.0)
    return estimate, truth, scale, support


def test_conformal_quantile_rank_and_edges() -> None:
    scores = np.arange(1, 101, dtype=float)  # 1..100
    # ceil((100+1)*0.95)=ceil(95.95)=96 -> 96th smallest == 96.0
    assert conformal_quantile(scores, 0.05) == 96.0
    # too few points for the level -> unbounded
    assert conformal_quantile(np.array([1.0, 2.0]), 0.05) == float("inf")
    assert conformal_quantile(np.array([]), 0.1) == float("inf")
    with pytest.raises(ValueError):
        conformal_quantile(scores, 1.5)


def test_nonconformity_scores_shape_guard() -> None:
    with pytest.raises(ValueError):
        nonconformity_scores(np.zeros(3), np.zeros(2), np.ones(3))


def test_split_conformal_marginal_coverage_guarantee() -> None:
    alpha = 0.05
    coverages, widths = [], []
    for seed in range(25):
        est, truth, scale, _ = _synthetic(seed, 1500)
        n_cal = 600
        iv = split_conformal_intervals(
            est[:n_cal], truth[:n_cal], scale[:n_cal],
            est[n_cal:], scale[n_cal:], alpha=alpha,
        )
        coverages.append(iv.coverage(truth[n_cal:]))
        widths.append(float(np.mean(iv.width)))
    mean_cov = float(np.mean(coverages))
    mean_width = float(np.mean(widths))
    # Guarantee holds in expectation (allow small finite-sample slack).
    assert mean_cov >= 0.94, mean_cov
    # ...and is not achieved trivially by [0, 1] intervals.
    assert mean_width < 0.8, mean_width


def test_split_conformal_tighter_alpha_widens_intervals() -> None:
    est, truth, scale, _ = _synthetic(0, 1200)
    n_cal = 600
    iv95 = split_conformal_intervals(
        est[:n_cal], truth[:n_cal], scale[:n_cal], est[n_cal:], scale[n_cal:], alpha=0.05
    )
    iv99 = split_conformal_intervals(
        est[:n_cal], truth[:n_cal], scale[:n_cal], est[n_cal:], scale[n_cal:], alpha=0.01
    )
    assert iv99.q > iv95.q
    assert float(np.mean(iv99.width)) >= float(np.mean(iv95.width))


def test_assign_support_bins() -> None:
    bins = assign_support_bins(np.array([5, 20, 49, 99, 250, 1000]))
    assert list(bins) == ["<20", "20-49", "20-49", "50-99", "250+", "250+"]


def test_mondrian_coverage_and_small_bin_fallback() -> None:
    alpha = 0.05
    coverages = []
    for seed in range(15):
        est, truth, scale, support = _synthetic(seed, 1500)
        bins = assign_support_bins(support)
        n_cal = 700
        iv = mondrian_conformal_intervals(
            est[:n_cal], truth[:n_cal], scale[:n_cal], bins[:n_cal],
            est[n_cal:], scale[n_cal:], bins[n_cal:], alpha=alpha,
        )
        coverages.append(iv.coverage(truth[n_cal:]))
        # every test interval is finite (small bins fall back to the global q)
        assert np.all(np.isfinite(iv.low)) and np.all(np.isfinite(iv.high))
    assert float(np.mean(coverages)) >= 0.93


# --- defect fixes -----------------------------------------------------------


def test_assign_support_bins_routes_invalid_to_widest() -> None:
    # NaN/inf/negative must NOT land in the tightest 250+ bin (overconfident).
    bins = assign_support_bins(np.array([np.nan, np.inf, -5.0, 0.0, 1000.0]))
    assert list(bins[:4]) == ["<20", "<20", "<20", "<20"]
    assert bins[4] == "250+"


def test_mondrian_rejects_mismatched_test_shapes() -> None:
    est, truth, scale, support = _synthetic(0, 100)
    bins = assign_support_bins(support)
    with pytest.raises(ValueError):
        mondrian_conformal_intervals(
            est, truth, scale, bins,
            est[:50], scale[:50], bins[:40],  # test_bins shorter than estimates
            alpha=0.05,
        )


# --- ConformalCalibrator ----------------------------------------------------


def test_calibrator_json_round_trip_including_inf() -> None:
    cal = ConformalCalibrator(
        alpha=0.05, q_global=float("inf"),
        q_by_bin={"<20": float("inf"), "250+": 1.9}, training_scope="t",
    )
    back = ConformalCalibrator.from_dict(cal.to_dict())
    assert back.alpha == 0.05
    assert back.q_global == float("inf")
    assert back.q_by_bin["<20"] == float("inf")
    assert back.q_by_bin["250+"] == 1.9


def test_calibrator_interval_unbounded_q_gives_full_range() -> None:
    cal = ConformalCalibrator(alpha=0.05, q_global=float("inf"), q_by_bin={})
    assert cal.interval(0.5, 0.1, 10) == (0.0, 1.0)


def test_from_dict_rejects_negative_quantile() -> None:
    """A finite negative q in a loaded calibrator would make interval() emit a
    negative half-width and silently invert the interval (ci_low > ci_high); the
    JSON loader must reject it. inf/nan stay valid (unbounded -> full clip)."""
    with pytest.raises(ValueError, match="negative quantile"):
        ConformalCalibrator.from_dict(
            {"alpha": 0.05, "q_global": -0.2, "q_by_bin": {},
             "scale_kind": "posterior_std"}
        )
    with pytest.raises(ValueError, match="negative quantile"):
        ConformalCalibrator.from_dict(
            {"alpha": 0.05, "q_global": 0.3, "q_by_bin": {"<20": -1.0},
             "scale_kind": "posterior_std"}
        )
    # A negative quantile in any of the finer Mondrian maps is rejected too.
    with pytest.raises(ValueError, match="negative quantile"):
        ConformalCalibrator.from_dict(
            {"alpha": 0.05, "q_global": 0.3, "q_by_bin": {},
             "q_by_event_type": {"SE": -0.5}, "scale_kind": "posterior_std"}
        )
    # inf remains valid: it means "unbounded" and falls back to the full clip range.
    ok = ConformalCalibrator.from_dict(
        {"alpha": 0.05, "q_global": float("inf"), "q_by_bin": {},
         "scale_kind": "posterior_std"}
    )
    assert ok.q_global == float("inf")


def test_robust_interval_reduces_to_q_at_zero_std_and_widens_with_std() -> None:
    """Depth-robust interval = sqrt(q^2 + (z*std)^2): equals +/-q when std=0,
    strictly wider as the sampling std grows, and clips to the requested range."""
    cal = ConformalCalibrator(alpha=0.05, q_global=0.3, q_by_bin={},
                              scale_kind="absolute_dpsi")
    lo0, hi0 = cal.robust_interval(0.0, 0.0, 1000, clip=(-1.0, 1.0))
    assert (hi0 - lo0) == pytest.approx(0.6, abs=1e-9)  # 2*q
    lo1, hi1 = cal.robust_interval(0.0, 0.10, 1000, clip=(-1.0, 1.0))
    assert (hi1 - lo1) > (hi0 - lo0)  # sampling term widens it
    expected_half = float(np.hypot(0.3, 1.959963984540054 * 0.10))
    assert (hi1 - lo1) == pytest.approx(2 * expected_half, abs=1e-9)
    # clipping to the ΔPSI range
    lo2, hi2 = cal.robust_interval(0.95, 0.5, 1000, clip=(-1.0, 1.0))
    assert lo2 >= -1.0 and hi2 <= 1.0


def test_robust_interval_unbounded_q_gives_full_range() -> None:
    cal = ConformalCalibrator(alpha=0.05, q_global=float("inf"), q_by_bin={})
    assert cal.robust_interval(0.5, 0.1, 10, clip=(-1.0, 1.0)) == (-1.0, 1.0)


def test_check_applicability_flags_low_support_shift() -> None:
    cal = ConformalCalibrator(
        alpha=0.05, q_global=0.3, q_by_bin={},
        calibration_profile={"n": 162, "support_median": 600.0},
    )
    ok_hi, msg_hi = cal.check_applicability(np.array([800, 1000, 1500]))
    assert ok_hi is True and msg_hi == ""
    ok_lo, msg_lo = cal.check_applicability(np.array([20, 30, 25]))
    assert ok_lo is False and "below the calibration regime" in msg_lo
    # no profile -> cannot check, returns ok
    cal_np = ConformalCalibrator(alpha=0.05, q_global=0.3, q_by_bin={})
    assert cal_np.check_applicability(np.array([10, 20])) == (True, "")


def test_calibration_profile_survives_json_round_trip() -> None:
    cal = ConformalCalibrator(
        alpha=0.05, q_global=0.34, q_by_bin={"250+": 0.34},
        scale_kind="absolute_dpsi",
        calibration_profile={"n": 162.0, "support_median": 627.5},
    )
    back = ConformalCalibrator.from_dict(cal.to_dict())
    assert back.calibration_profile["support_median"] == 627.5
    assert back.calibration_profile["n"] == 162.0


def test_fit_calibrator_guarantees_coverage() -> None:
    alpha = 0.05
    covs = []
    for seed in range(25):
        est, truth, scale, support = _synthetic(seed, 1500)
        n_cal = 700
        cal = fit_conformal_calibrator(
            est[:n_cal], truth[:n_cal], scale[:n_cal], support[:n_cal], alpha=alpha
        )
        lo = np.empty(1500 - n_cal)
        hi = np.empty_like(lo)
        for i, j in enumerate(range(n_cal, 1500)):
            lo[i], hi[i] = cal.interval(est[j], scale[j], support[j])
        covs.append(float(np.mean((truth[n_cal:] >= lo) & (truth[n_cal:] <= hi))))
    assert float(np.mean(covs)) >= 0.94
    assert float(np.mean(hi - lo)) < 0.8  # sharp, not trivially [0,1]


def test_fit_calibrator_shape_guard() -> None:
    with pytest.raises(ValueError):
        fit_conformal_calibrator(np.zeros(5), np.zeros(5), np.ones(5), np.ones(4))


def test_q_for_event_type_precedence_and_fallback() -> None:
    """Event-type quantile wins when present; otherwise fall back to support bin/global.
    Passing event_type=None reproduces the support-only behaviour exactly."""
    cal = ConformalCalibrator(
        alpha=0.05, q_global=0.30, q_by_bin={"250+": 0.20},
        q_by_event_type={"MXE": 0.55},
    )
    assert cal.q_for(1000) == 0.20          # no event_type -> support-bin behaviour
    assert cal.q_for(1000, None) == 0.20
    assert cal.q_for(1000, "MXE") == 0.55   # matching type takes precedence
    assert cal.q_for(1000, "SE") == 0.20    # unknown type -> support bin
    cal2 = ConformalCalibrator(alpha=0.05, q_global=0.3, q_by_bin={},
                               q_by_event_type={"RI": float("inf")})
    assert cal2.q_for(10, "RI") == 0.30     # non-finite type quantile -> fall back


def test_interval_uses_event_type_quantile() -> None:
    cal = ConformalCalibrator(alpha=0.05, q_global=0.30, q_by_bin={},
                              scale_kind="absolute_dpsi", q_by_event_type={"MXE": 0.50})
    lo_se, hi_se = cal.robust_interval(0.0, 0.0, 1000, event_type="SE", clip=(-1.0, 1.0))
    lo_mx, hi_mx = cal.robust_interval(0.0, 0.0, 1000, event_type="MXE", clip=(-1.0, 1.0))
    assert (hi_se - lo_se) == pytest.approx(0.60, abs=1e-9)   # 2*q_global
    assert (hi_mx - lo_mx) == pytest.approx(1.00, abs=1e-9)   # 2*q_by_event_type[MXE]


def test_q_by_event_type_survives_json_round_trip() -> None:
    cal = ConformalCalibrator(
        alpha=0.05, q_global=0.34, q_by_bin={"250+": 0.34},
        q_by_event_type={"SE": 0.34, "MXE": 0.51, "RI": float("inf")},
    )
    back = ConformalCalibrator.from_dict(cal.to_dict())
    assert back.q_by_event_type["SE"] == 0.34
    assert back.q_by_event_type["MXE"] == 0.51
    assert back.q_by_event_type["RI"] == float("inf")


def test_fit_with_event_types_produces_per_type_quantiles() -> None:
    est, truth, scale, support = _synthetic(0, 1200)
    ets = np.array(["SE"] * 600 + ["MXE"] * 600)
    cal = fit_conformal_calibrator(est, truth, scale, support, alpha=0.05, event_types=ets)
    assert set(cal.q_by_event_type) == {"SE", "MXE"}
    assert all(np.isfinite(v) and v > 0 for v in cal.q_by_event_type.values())
    with pytest.raises(ValueError):
        fit_conformal_calibrator(est, truth, scale, support, event_types=ets[:10])


def test_default_calibrator_artifact_ships_and_loads() -> None:
    from braid.target.conformal import load_default_conformal_calibrator

    cal = load_default_conformal_calibrator()
    assert cal.alpha == 0.05
    assert np.isfinite(cal.q_global) and cal.q_global > 0
    assert len(cal.q_by_bin) >= 1
    assert cal.training_scope  # provenance recorded
    # produces a finite, ordered, in-range interval
    lo, hi = cal.interval(0.7, 0.2, 300)
    assert 0.0 <= lo <= 0.7 <= hi <= 1.0


def test_conformal_quantile_ignores_nonfinite_scores() -> None:
    """conformal_quantile must drop NaN/Inf nonconformity scores before ranking, so
    a single non-finite residual cannot corrupt the calibrated quantile."""
    import numpy as np

    from braid.target.conformal import conformal_quantile

    finite = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    polluted = np.concatenate([finite, [np.nan, np.inf, -np.inf]])

    q_clean = conformal_quantile(finite, 0.1)
    q_polluted = conformal_quantile(polluted, 0.1)

    assert np.isfinite(q_polluted)
    # Non-finite scores are dropped, so the result equals the finite-only quantile.
    assert q_polluted == q_clean
