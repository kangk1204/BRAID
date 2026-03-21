"""Statistical methods for PSI confidence estimation.

Uses a Beta-binomial model to compute credible intervals for PSI values:
    PSI ~ Beta(alpha_prior + inclusion, beta_prior + exclusion)

The 95% highest-density credible interval is computed via ``scipy.stats.beta``.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass

from scipy.stats import beta as beta_dist

from braid.splicing.psi import PSIResult

logger = logging.getLogger(__name__)


@dataclass
class BetaBinomialResult:
    """Result of Beta-binomial confidence interval estimation.

    Attributes:
        ci_low: Lower bound of the 95% credible interval.
        ci_high: Upper bound of the 95% credible interval.
        alpha_posterior: Posterior alpha parameter.
        beta_posterior: Posterior beta parameter.
        mean: Posterior mean.
    """

    ci_low: float
    ci_high: float
    alpha_posterior: float
    beta_posterior: float
    mean: float


def beta_binomial_ci(
    inclusion_count: float,
    total_count: float,
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    ci_level: float = 0.95,
) -> BetaBinomialResult:
    """Compute a Beta-binomial credible interval for a PSI estimate.

    Uses a conjugate Beta prior with the binomial likelihood of junction
    read counts to produce a posterior Beta distribution. The credible
    interval is the equal-tailed interval at the specified confidence level.

    Args:
        inclusion_count: Effective inclusion evidence count.
        total_count: Effective total evidence count (inclusion + exclusion).
        alpha_prior: Beta prior alpha parameter (default 1.0 = uniform).
        beta_prior: Beta prior beta parameter (default 1.0 = uniform).
        ci_level: Credible interval level (default 0.95).

    Returns:
        BetaBinomialResult with credible interval bounds and posterior
        parameters.
    """
    exclusion_count = total_count - inclusion_count
    alpha_post = alpha_prior + inclusion_count
    beta_post = beta_prior + exclusion_count

    tail = (1 - ci_level) / 2
    ci_low = float(beta_dist.ppf(tail, alpha_post, beta_post))
    ci_high = float(beta_dist.ppf(1 - tail, alpha_post, beta_post))
    mean = alpha_post / (alpha_post + beta_post)

    return BetaBinomialResult(
        ci_low=ci_low,
        ci_high=ci_high,
        alpha_posterior=alpha_post,
        beta_posterior=beta_post,
        mean=mean,
    )


def add_confidence_intervals(
    psi_results: list[PSIResult],
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
) -> None:
    """Add Beta-binomial credible intervals to PSI results in-place.

    Updates ``ci_low`` and ``ci_high`` on each PSIResult.

    Args:
        psi_results: List of PSI results to annotate.
        alpha_prior: Beta prior alpha parameter.
        beta_prior: Beta prior beta parameter.
    """
    for result in psi_results:
        if result.total_reads == 0 or math.isnan(result.psi):
            result.ci_low = 0.0
            result.ci_high = 1.0
            continue

        bb = beta_binomial_ci(
            inclusion_count=result.inclusion_count,
            total_count=result.total_reads,
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
        )
        result.ci_low = bb.ci_low
        result.ci_high = bb.ci_high


def psi_significance_filter(
    psi_results: list[PSIResult],
    min_reads: int = 10,
    max_ci_width: float = 1.0,
) -> list[PSIResult]:
    """Filter PSI results by minimum read depth and CI width.

    Args:
        psi_results: PSI results with CIs already computed.
        min_reads: Minimum total reads to retain event.
        max_ci_width: Maximum CI width to retain event.

    Returns:
        Filtered list of PSIResult objects passing the thresholds.
    """
    filtered = []
    for result in psi_results:
        if math.isnan(result.psi):
            continue
        if result.total_reads < min_reads:
            continue
        ci_width = result.ci_high - result.ci_low
        if ci_width > max_ci_width:
            continue
        filtered.append(result)

    logger.info(
        "PSI significance filter: %d / %d events pass (min_reads=%d, max_ci_width=%.2f)",
        len(filtered),
        len(psi_results),
        min_reads,
        max_ci_width,
    )
    return filtered
