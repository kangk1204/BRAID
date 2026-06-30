"""Calibration and sharpness metrics for PSI prediction intervals.

These functions support an honest, deconfounded assessment of interval quality.
The original BRAID sharpness control compared its intervals against random
intervals centred uniformly on [0, 1] and scored against a *uniform* truth, so
part of the reported "coverage lift" reflected the informativeness of the point
estimate rather than width calibration. The metrics here separate those effects:

- ``empirical_coverage`` / ``mean_width``: the basics.
- ``interval_score``: the Gneiting-Raftery (Winkler) interval score, a proper
  scoring rule that rewards coverage *and* sharpness, so optimizing it cannot be
  gamed by trivially wide intervals.
- ``estimate_centered_coverage``: the fair, deconfounded baseline -- a
  width-matched interval centred on the *same* point estimate, scored against the
  real truth. If BRAID's coverage does not exceed this, its interval shape adds
  nothing beyond "a width-w interval around a good estimate".
- ``random_centered_coverage``: a width-matched interval centred at random,
  scored against the real truth (a stricter "random" baseline than uniform truth).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def empirical_coverage(low: np.ndarray, high: np.ndarray, truth: np.ndarray) -> float:
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if low.size == 0:
        return float("nan")
    return float(np.mean((truth >= low) & (truth <= high)))


def mean_width(low: np.ndarray, high: np.ndarray) -> float:
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if low.size == 0:
        return float("nan")
    return float(np.mean(high - low))


def interval_score(
    low: np.ndarray,
    high: np.ndarray,
    truth: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Mean Gneiting-Raftery interval score for central (1 - alpha) intervals.

    ``IS = (high - low) + (2/alpha)(low - y) 1[y < low] + (2/alpha)(y - high) 1[y > high]``.
    Lower is better; it rewards narrow intervals but penalizes missing the truth,
    so it cannot be minimized by widening intervals to [0, 1]. This is the
    objective BRAID should select its calibration against, instead of fitting to a
    nominal coverage target.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if low.size == 0:
        return float("nan")
    width = high - low
    below = np.where(truth < low, (2.0 / alpha) * (low - truth), 0.0)
    above = np.where(truth > high, (2.0 / alpha) * (truth - high), 0.0)
    return float(np.mean(width + below + above))


def estimate_centered_coverage(
    estimate: np.ndarray,
    width: np.ndarray,
    truth: np.ndarray,
    clip: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Coverage of a width-matched interval centred on the point estimate.

    This is the fair, deconfounded baseline: it holds the interval width fixed and
    centres it on the same estimate BRAID uses, so any excess coverage of BRAID
    over this value is attributable to interval *shape/calibration*, not to the
    estimate being informative.
    """
    estimate = np.asarray(estimate, dtype=float)
    width = np.asarray(width, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if estimate.size == 0:
        return float("nan")
    low = np.clip(estimate - width / 2.0, clip[0], clip[1])
    high = np.clip(estimate + width / 2.0, clip[0], clip[1])
    return float(np.mean((truth >= low) & (truth <= high)))


def random_centered_coverage(
    width: np.ndarray,
    truth: np.ndarray,
    *,
    seed: int = 42,
    clip: tuple[float, float] = (0.0, 1.0),
) -> float:
    """Coverage of a width-matched interval centred at random, scored vs real truth.

    A stricter "random" baseline than centring on a uniform truth: it uses the
    actual observed truth values, isolating the contribution of *centring on the
    estimate* (compare with :func:`estimate_centered_coverage`).
    """
    width = np.asarray(width, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if width.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    centers = rng.uniform(clip[0], clip[1], size=width.size)
    low = np.clip(centers - width / 2.0, clip[0], clip[1])
    high = np.clip(centers + width / 2.0, clip[0], clip[1])
    return float(np.mean((truth >= low) & (truth <= high)))


@dataclass(frozen=True)
class SharpnessReport:
    n: int
    coverage: float
    mean_width: float
    interval_score: float
    estimate_centered_coverage: float
    random_centered_coverage: float

    @property
    def lift_over_estimate_centered(self) -> float:
        """The deconfounded lift: coverage gain over a same-width, same-centre
        interval. ~0 means the interval shape adds nothing beyond the estimate."""
        return self.coverage - self.estimate_centered_coverage

    @property
    def lift_over_random(self) -> float:
        return self.coverage - self.random_centered_coverage


def sharpness_report(
    low: np.ndarray,
    high: np.ndarray,
    estimate: np.ndarray,
    truth: np.ndarray,
    *,
    alpha: float = 0.05,
    seed: int = 42,
) -> SharpnessReport:
    """Full deconfounded calibration/sharpness report for a set of intervals."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    width = high - low
    return SharpnessReport(
        n=int(low.size),
        coverage=empirical_coverage(low, high, truth),
        mean_width=mean_width(low, high),
        interval_score=interval_score(low, high, truth, alpha),
        estimate_centered_coverage=estimate_centered_coverage(estimate, width, truth),
        random_centered_coverage=random_centered_coverage(width, truth, seed=seed),
    )
