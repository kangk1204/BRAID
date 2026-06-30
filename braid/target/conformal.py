"""Conformal prediction intervals for splicing PSI.

This module gives BRAID a *principled* alternative to the ad hoc, fit-to-target
interval inflation used by the overdispersed Beta posterior. Split conformal
prediction (and its Mondrian / support-stratified variant) produces prediction
intervals with a **distribution-free, finite-sample marginal coverage
guarantee**: given an exchangeable calibration set, a 1-alpha interval covers the
true PSI with probability at least 1-alpha, regardless of whether the underlying
posterior is well specified. This directly addresses the circularity of choosing
a width schedule to hit nominal coverage on the same data, and -- to our
knowledge -- conformal prediction has not previously been applied to RNA-seq
splicing PSI estimation.

Method (normalized split conformal):
    For each calibration event i with point estimate psi_hat_i, an uncertainty
    scale sigma_i (e.g. the posterior standard deviation), and observed truth
    psi_i, the nonconformity score is the standardized absolute residual
        r_i = |psi_i - psi_hat_i| / max(sigma_i, eps).
    The conformal quantile q is the ceil((n+1)(1-alpha))/n empirical quantile of
    {r_i}. A test event's interval is then
        [psi_hat - q * sigma, psi_hat + q * sigma]  clipped to [0, 1].

The Mondrian variant computes q_b separately within each support bin b, giving
approximate bin-conditional coverage (sharper where the data warrant it) while
preserving the finite-sample guarantee within each group of adequate size.

References: Vovk et al., *Algorithmic Learning in a Random World* (2005);
Lei et al., JASA (2018), "Distribution-Free Predictive Inference for Regression".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_EPS = 1e-9


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-valid conformal quantile of nonconformity scores.

    Returns the ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile, i.e. the
    rank that guarantees marginal coverage >= 1 - alpha for an exchangeable test
    point. If the required rank exceeds ``n`` (too few calibration points for the
    requested level) the interval is unbounded and ``+inf`` is returned.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = scores.size
    if n == 0:
        return float("inf")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        return float("inf")
    # rank-th smallest (1-indexed) == index rank-1 of the sorted scores
    return float(np.partition(scores, rank - 1)[rank - 1])


def nonconformity_scores(
    estimates: np.ndarray,
    truth: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Standardized absolute residuals ``|truth - estimate| / max(scale, eps)``."""
    estimates = np.asarray(estimates, dtype=float)
    truth = np.asarray(truth, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if not (estimates.shape == truth.shape == scale.shape):
        raise ValueError("estimates, truth, and scale must have the same shape")
    safe_scale = np.maximum(scale, _EPS)
    return np.abs(truth - estimates) / safe_scale


@dataclass(frozen=True)
class ConformalIntervals:
    """Result of conformal interval construction."""

    low: np.ndarray
    high: np.ndarray
    q: float
    alpha: float

    @property
    def width(self) -> np.ndarray:
        return self.high - self.low

    def covers(self, truth: np.ndarray) -> np.ndarray:
        truth = np.asarray(truth, dtype=float)
        return (truth >= self.low) & (truth <= self.high)

    def coverage(self, truth: np.ndarray) -> float:
        return float(np.mean(self.covers(truth)))


def split_conformal_intervals(
    cal_estimates: np.ndarray,
    cal_truth: np.ndarray,
    cal_scale: np.ndarray,
    test_estimates: np.ndarray,
    test_scale: np.ndarray,
    *,
    alpha: float = 0.05,
    clip: tuple[float, float] = (0.0, 1.0),
) -> ConformalIntervals:
    """Build normalized split-conformal PSI intervals.

    The calibration arrays provide observed (estimate, truth, scale) triples; the
    test arrays provide (estimate, scale) for events whose true PSI is unknown.
    Returns intervals with a finite-sample marginal coverage guarantee of
    >= 1 - alpha (under exchangeability of calibration and test events).
    """
    scores = nonconformity_scores(cal_estimates, cal_truth, cal_scale)
    q = conformal_quantile(scores, alpha)
    raw_test_estimates = np.asarray(test_estimates, dtype=float)
    raw_test_scale = np.asarray(test_scale, dtype=float)
    invalid_test = ~np.isfinite(raw_test_estimates) | ~np.isfinite(raw_test_scale)
    # ``np.maximum(nan, _EPS)`` remains NaN, so explicitly route undefined
    # deployment uncertainty to an infinite half-width. The clipped result is
    # the conservative full interval instead of a malformed ``(nan, nan)``.
    test_scale = np.where(
        np.isfinite(raw_test_scale), np.maximum(raw_test_scale, _EPS), np.inf,
    )
    test_estimates = np.where(np.isfinite(raw_test_estimates), raw_test_estimates, 0.0)
    half = q * test_scale
    low = np.clip(test_estimates - half, clip[0], clip[1])
    high = np.clip(test_estimates + half, clip[0], clip[1])
    low = np.where(invalid_test, clip[0], low)
    high = np.where(invalid_test, clip[1], high)
    return ConformalIntervals(low=low, high=high, q=q, alpha=alpha)


@dataclass
class MondrianConformalIntervals:
    """Per-bin (support-stratified) conformal intervals."""

    low: np.ndarray
    high: np.ndarray
    alpha: float
    q_by_bin: dict[str, float] = field(default_factory=dict)
    bin_of: np.ndarray | None = None

    @property
    def width(self) -> np.ndarray:
        return self.high - self.low

    def coverage(self, truth: np.ndarray) -> float:
        truth = np.asarray(truth, dtype=float)
        return float(np.mean((truth >= self.low) & (truth <= self.high)))


def assign_support_bins(
    support: np.ndarray,
    edges: tuple[float, ...] = (20, 50, 100, 250),
) -> np.ndarray:
    """Map total read support to ordered bin labels (matching the BRAID schedule).

    Default bins: ``<20, 20-49, 50-99, 100-249, 250+``. Non-finite or negative
    support is invalid and is routed to the widest (lowest-support, most
    conservative) bin, so it can never be assigned the tightest interval --
    ``np.digitize`` would otherwise send ``NaN`` to the top ``250+`` bin and
    produce an overconfident interval.
    """
    support = np.asarray(support, dtype=float)
    labels = ["<20", "20-49", "50-99", "100-249", "250+"]
    if len(edges) != len(labels) - 1:
        # np.digitize returns indices in [0, len(edges)]; the label scheme is fixed at
        # len(labels) slots, so a mismatched edge count indexes past ``labels``
        # (IndexError) on high-support events. Reject up front with a clear message
        # instead of crashing mid-run on one particular event's bin lookup.
        raise ValueError(
            f"assign_support_bins requires {len(labels) - 1} bin edges to match the "
            f"{len(labels)} fixed support-bin labels, got {len(edges)}: {tuple(edges)}."
        )
    safe = np.where(np.isfinite(support) & (support >= 0.0), support, -1.0)
    idx = np.digitize(safe, bins=list(edges), right=False)
    return np.array([labels[i] for i in idx], dtype=object)


def mondrian_conformal_intervals(
    cal_estimates: np.ndarray,
    cal_truth: np.ndarray,
    cal_scale: np.ndarray,
    cal_bins: np.ndarray,
    test_estimates: np.ndarray,
    test_scale: np.ndarray,
    test_bins: np.ndarray,
    *,
    alpha: float = 0.05,
    clip: tuple[float, float] = (0.0, 1.0),
    fallback_to_global: bool = True,
) -> MondrianConformalIntervals:
    """Support-stratified (Mondrian) split-conformal PSI intervals.

    A separate conformal quantile is computed within each support bin. Bins with
    too few calibration points (where the required rank exceeds ``n``) fall back
    to the pooled global quantile when ``fallback_to_global`` is set, preserving a
    finite (if looser) interval instead of an unbounded one.
    """
    cal_estimates = np.asarray(cal_estimates, dtype=float)
    cal_truth = np.asarray(cal_truth, dtype=float)
    cal_scale = np.asarray(cal_scale, dtype=float)
    cal_bins = np.asarray(cal_bins, dtype=object)
    raw_test_estimates = np.asarray(test_estimates, dtype=float)
    raw_test_scale = np.asarray(test_scale, dtype=float)
    invalid_test = ~np.isfinite(raw_test_estimates) | ~np.isfinite(raw_test_scale)
    # ``np.maximum(nan, _EPS)`` remains NaN, so explicitly route undefined
    # deployment uncertainty to an infinite half-width. The clipped result is
    # the conservative full interval instead of a malformed ``(nan, nan)``.
    test_scale = np.where(
        np.isfinite(raw_test_scale), np.maximum(raw_test_scale, _EPS), np.inf,
    )
    test_estimates = np.where(np.isfinite(raw_test_estimates), raw_test_estimates, 0.0)
    test_bins = np.asarray(test_bins, dtype=object)

    if not (cal_estimates.shape == cal_truth.shape == cal_scale.shape == cal_bins.shape):
        raise ValueError(
            "cal_estimates, cal_truth, cal_scale, and cal_bins must have the same shape"
        )
    if not (test_estimates.shape == test_scale.shape == test_bins.shape):
        raise ValueError(
            "test_estimates, test_scale, and test_bins must have the same shape"
        )

    global_scores = nonconformity_scores(cal_estimates, cal_truth, cal_scale)
    global_q = conformal_quantile(global_scores, alpha)

    q_by_bin: dict[str, float] = {}
    for b in np.unique(cal_bins):
        mask = cal_bins == b
        scores_b = nonconformity_scores(
            cal_estimates[mask], cal_truth[mask], cal_scale[mask]
        )
        q_b = conformal_quantile(scores_b, alpha)
        if not np.isfinite(q_b) and fallback_to_global:
            q_b = global_q
        q_by_bin[str(b)] = q_b

    low = np.empty_like(test_estimates)
    high = np.empty_like(test_estimates)
    for i, b in enumerate(test_bins):
        q_b = q_by_bin.get(str(b), global_q)
        if not np.isfinite(q_b) and fallback_to_global:
            q_b = global_q
        half = q_b * test_scale[i]
        low[i] = np.clip(test_estimates[i] - half, clip[0], clip[1])
        high[i] = np.clip(test_estimates[i] + half, clip[0], clip[1])
        if invalid_test[i]:
            low[i] = clip[0]
            high[i] = clip[1]

    return MondrianConformalIntervals(
        low=low, high=high, alpha=alpha, q_by_bin=q_by_bin, bin_of=test_bins
    )


# ---------------------------------------------------------------------------
# Production calibrator: a small, serializable, fitted conformal schedule.
# ---------------------------------------------------------------------------


def _q_to_json(q: float) -> float | None:
    """JSON cannot represent +inf; store an unbounded quantile as null."""
    return None if not np.isfinite(q) else float(q)


def _q_from_json(value: float | None) -> float:
    return float("inf") if value is None else float(value)


@dataclass(frozen=True)
class ConformalCalibrator:
    """A fitted, serializable split/Mondrian conformal calibrator.

    Holds only the per-support-bin conformal quantiles (and a pooled global
    fallback). Combined with a per-event uncertainty scale (the posterior
    standard deviation that ``bootstrap_psi`` already computes), it turns a point
    estimate into an interval with a distribution-free finite-sample coverage
    guarantee -- a property that, to our knowledge, upstream splicing callers'
    native intervals do not target.
    """

    alpha: float
    q_global: float
    q_by_bin: dict[str, float]
    bin_edges: tuple[float, ...] = (20, 50, 100, 250)
    scale_kind: str = "posterior_std"
    training_scope: str = "builtin_default"
    # Summary of the data the calibrator was fit on, used to flag distribution
    # shift at deployment (the conformal guarantee is conditional on exchangeability
    # of calibration and deployment events). Keys (all optional): ``n``,
    # ``support_median``, ``support_q25``, ``support_q75``.
    calibration_profile: dict[str, float] = field(default_factory=dict)
    # Optional event-type-stratified quantiles (Mondrian by rMATS event type:
    # SE/A3SS/A5SS/MXE/RI). When an ``event_type`` is supplied at lookup time and is
    # present here, its quantile takes precedence over the support-bin quantile -- AS
    # types have different sampling structure (e.g. MXE posterior SD ~2.6x SE) and,
    # where type-specific truth exists, different orthogonal-truth residual surfaces.
    # Empty by default -> behaviour identical to a support-only Mondrian calibrator.
    q_by_event_type: dict[str, float] = field(default_factory=dict)
    # Optional composite-group quantiles keyed by "<event_type>|<support_bin>" (e.g.
    # "A3SS|100-249"), enabling 2-D Mondrian (event-type x support). Sparse groups are
    # omitted at fit time and resolved by the q_for cascade below. Empty by default ->
    # behaviour identical to the event-type/support Mondrian calibrator. (artifact v2)
    q_by_group: dict[str, float] = field(default_factory=dict)

    def check_applicability(
        self,
        deployment_supports: np.ndarray,
        *,
        low_support_factor: float = 0.5,
    ) -> tuple[bool, str]:
        """Flag distribution shift between deployment data and the calibration set.

        The conformal coverage guarantee holds only when deployment events are
        exchangeable with the calibration events. We cannot measure deployment
        residuals (no truth at deploy time), but a materially **lower read-support
        regime** is a measurable, common shift: the per-bin quantiles were fit where
        sampling noise was negligible, so on much shallower data a fixed half-width
        can under-cover. Returns ``(ok, message)``; ``ok=False`` carries a
        human-readable warning recommending :meth:`robust_interval` or a refit.
        """
        prof = self.calibration_profile or {}
        cal_med = prof.get("support_median")
        sup = np.asarray(deployment_supports, dtype=float)
        sup = sup[np.isfinite(sup)]
        if not cal_med or sup.size == 0:
            return True, ""
        dep_med = float(np.median(sup))
        if dep_med < low_support_factor * float(cal_med):
            return False, (
                f"deployment median read support ({dep_med:.0f}) is well below the "
                f"calibration regime ({cal_med:.0f}); fixed-width conformal intervals "
                f"may under-cover. Use the depth-robust interval (default in "
                f"`braid differential`) or refit the calibrator on matched-depth data."
            )
        return True, ""

    def q_for(self, total_support: float, event_type: str | None = None) -> float:
        """Conformal quantile for an event via a coarsening cascade.

        Precedence: composite group ``"<event_type>|<support_bin>"`` (2-D Mondrian)
        -> event-type quantile -> support-bin quantile -> global. Each level is used
        only when present with a finite quantile, so sparse groups fall through to the
        next coarser level and ``event_type=None`` reproduces support-only Mondrian.

        This fixes the prior short-circuit in which an ``event_type`` hit returned
        immediately and never consulted the support bin, which made any 2-D
        (event-type x support) scheme impossible.
        """
        label = str(assign_support_bins(np.array([total_support]), self.bin_edges)[0])
        if event_type is not None:
            qg = self.q_by_group.get(f"{event_type}|{label}")
            if qg is not None and np.isfinite(qg):
                return float(qg)
            qt = self.q_by_event_type.get(event_type)
            if qt is not None and np.isfinite(qt):
                return float(qt)
        q = self.q_by_bin.get(label)
        if q is None or not np.isfinite(q):
            return self.q_global
        return q

    def interval(
        self,
        psi_hat: float,
        sigma: float,
        total_support: float,
        *,
        event_type: str | None = None,
        force_global: bool = False,
        clip: tuple[float, float] = (0.0, 1.0),
    ) -> tuple[float, float]:
        """Conformal interval ``[psi_hat - q*sigma, psi_hat + q*sigma]`` clipped to [0,1].

        An unbounded quantile (too few calibration points) yields the full
        ``clip`` range. Clipping preserves the coverage guarantee because the
        truth (PSI) always lies in ``[0, 1]``. ``event_type`` selects a type-specific
        quantile when one is available (see :meth:`q_for`); ``force_global`` bypasses
        the support/event-type/composite cascade and uses the pooled global quantile
        (the explicit "no support information" default).
        """
        q = self.q_global if force_global else self.q_for(total_support, event_type)
        s = float(sigma)
        p = float(psi_hat)
        if not np.isfinite(q) or not np.isfinite(s) or not np.isfinite(p):
            # Unbounded quantile (too few calibration points) or undefined spread:
            # fall back to the conservative full clip range rather than letting an
            # inf/NaN half-width propagate. (max(nan, 0.0) returns nan, so the
            # NaN-sigma case must be handled explicitly.)
            return float(clip[0]), float(clip[1])
        half = q * max(s, 0.0)
        low = min(max(p - half, clip[0]), clip[1])
        high = min(max(p + half, clip[0]), clip[1])
        return float(low), float(high)

    def robust_interval(
        self,
        psi_hat: float,
        sampling_std: float,
        total_support: float,
        *,
        z: float = 1.959963984540054,
        event_type: str | None = None,
        force_global: bool = False,
        clip: tuple[float, float] = (0.0, 1.0),
    ) -> tuple[float, float]:
        """Depth-robust interval for an ``absolute_dpsi`` calibrator.

        Combines the calibrated half-width ``q`` (which covers the orthogonal-truth
        residual floor at the depth the calibrator was fit) with the
        within-sample sampling spread ``z * sampling_std`` in quadrature::

            half = sqrt(q**2 + (z * sampling_std)**2)

        At fit-depth ``sampling_std`` is negligible so this reduces to ``+/- q``;
        on lower-depth deployment data the sampling term grows and keeps coverage at
        nominal **without a refit** (verified by binomial-thinning subsampling:
        full-depth 0.95 -> 0.05x depth 0.92, vs 0.89 for the plain ``+/- q``).
        Intended for the differential ΔPSI calibrator (``scale_kind ==
        "absolute_dpsi"``); ``q`` is an absolute half-width there, not a multiplier.
        ``event_type`` selects a type-specific quantile when available (see
        :meth:`q_for`); ``force_global`` bypasses the cascade for the pooled global q.
        """
        q = self.q_global if force_global else self.q_for(total_support, event_type)
        p = float(psi_hat)
        if (
            not np.isfinite(q)
            or not np.isfinite(float(sampling_std))
            or not np.isfinite(p)
        ):
            return float(clip[0]), float(clip[1])
        half = float(np.hypot(q, z * max(float(sampling_std), 0.0)))
        low = min(max(p - half, clip[0]), clip[1])
        high = min(max(p + half, clip[0]), clip[1])
        return float(low), float(high)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "conformal_psi_calibrator",
            "version": 2,
            "alpha": float(self.alpha),
            "q_global": _q_to_json(self.q_global),
            "q_by_bin": {k: _q_to_json(v) for k, v in self.q_by_bin.items()},
            "bin_edges": list(self.bin_edges),
            "scale_kind": self.scale_kind,
            "training_scope": self.training_scope,
            "calibration_profile": {k: float(v) for k, v in self.calibration_profile.items()},
            "q_by_event_type": {k: _q_to_json(v) for k, v in self.q_by_event_type.items()},
            "q_by_group": {k: _q_to_json(v) for k, v in self.q_by_group.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConformalCalibrator":
        q_global = _q_from_json(d.get("q_global"))
        q_by_bin = {k: _q_from_json(v) for k, v in dict(d.get("q_by_bin", {})).items()}
        q_by_event_type = {
            k: _q_from_json(v) for k, v in dict(d.get("q_by_event_type", {})).items()
        }
        q_by_group = {
            k: _q_from_json(v) for k, v in dict(d.get("q_by_group", {})).items()
        }
        # A conformal quantile is a non-negative half-width multiplier. A finite
        # negative q makes interval() emit half < 0 and silently invert the interval
        # (ci_low > ci_high); inf/nan stay valid because q_for()/interval() treat a
        # non-finite q as "unbounded" and fall back to the full clip range. Reject a
        # malformed custom calibrator here, at the JSON boundary, rather than shipping
        # an inverted-interval contract.
        named_q = [("q_global", q_global)]
        named_q += [(f"q_by_bin[{k}]", v) for k, v in q_by_bin.items()]
        named_q += [(f"q_by_event_type[{k}]", v) for k, v in q_by_event_type.items()]
        named_q += [(f"q_by_group[{k}]", v) for k, v in q_by_group.items()]
        for name, q in named_q:
            if q is not None and np.isfinite(q) and q < 0:
                raise ValueError(
                    f"conformal calibrator has a negative quantile {name}={q}; q values "
                    "must be >= 0 (a negative half-width inverts the interval)"
                )
        alpha = float(d["alpha"])
        if not 0.0 < alpha < 1.0:
            # alpha is the miscoverage level; outside (0, 1) it is meaningless and marks
            # a corrupted/hand-edited artifact. Reject it at the JSON boundary.
            raise ValueError(
                f"conformal calibrator alpha must be in (0, 1), got {alpha}"
            )
        bin_edges = tuple(d.get("bin_edges", (20, 50, 100, 250)))
        if len(bin_edges) != 4:
            # The support-bin label scheme is fixed at 5 labels (4 edges); a different
            # edge count crashes assign_support_bins on high-support events. Fail fast
            # here so the operator gets a --calibration-scoped message at load time.
            raise ValueError(
                "conformal calibrator bin_edges must have exactly 4 entries to match the "
                f"fixed 5 support-bin labels, got {len(bin_edges)}: {bin_edges}"
            )
        # A 4-entry bin_edges can still be non-numeric or non-monotonic. np.digitize
        # (in assign_support_bins) requires numeric, monotonic bins, and the fixed label
        # scheme assumes strictly *increasing* support thresholds; without this check a
        # malformed-but-right-length edge set passes load and instead crashes mid-run on
        # the first q_for lookup (TypeError for strings, "bins must be monotonically
        # increasing or decreasing" for out-of-order). Validate at the JSON boundary so
        # the operator gets a clean --calibration-scoped error.
        try:
            numeric_edges = [float(e) for e in bin_edges]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"conformal calibrator bin_edges must be numeric, got {bin_edges!r}"
            ) from exc
        if not all(np.isfinite(numeric_edges)):
            raise ValueError(
                f"conformal calibrator bin_edges must all be finite, got {bin_edges!r}"
            )
        if any(lo >= hi for lo, hi in zip(numeric_edges, numeric_edges[1:])):
            raise ValueError(
                f"conformal calibrator bin_edges must be strictly increasing, got {bin_edges!r}"
            )
        # Store the COERCED numeric edges, not the raw values: a numeric string like "20"
        # passes the float() checks above but, left as a str, crashes np.digitize (in
        # assign_support_bins) at q_for time. Preserve integer-ness so the all-integer
        # shipped/default edge set serializes byte-identically.
        bin_edges = tuple(int(v) if v.is_integer() else v for v in numeric_edges)
        return cls(
            alpha=alpha,
            q_global=q_global,
            q_by_bin=q_by_bin,
            bin_edges=bin_edges,
            scale_kind=str(d.get("scale_kind", "posterior_std")),
            training_scope=str(d.get("training_scope", "builtin_default")),
            calibration_profile={
                k: float(v) for k, v in dict(d.get("calibration_profile", {})).items()
            },
            q_by_event_type=q_by_event_type,
            q_by_group=q_by_group,
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    # A fitted calibrator JSON is a few KB (per-bin quantiles); 10 MB is orders of
    # magnitude of headroom. The cap bounds memory before the whole file is read into
    # memory and JSON-parsed, so a malformed/oversized ``--calibration`` path fails with
    # a clear error instead of exhausting memory (or hanging on a device/pipe).
    _MAX_CALIBRATION_BYTES = 10 * 1024 * 1024

    @classmethod
    def from_json(cls, path: str | Path) -> "ConformalCalibrator":
        p = Path(path)
        try:
            size = p.stat().st_size
        except OSError:
            size = 0  # non-regular path (pipe/device): let read_text surface the error
        if size > cls._MAX_CALIBRATION_BYTES:
            raise ValueError(
                f"calibrator JSON {p} is {size} bytes, exceeding the "
                f"{cls._MAX_CALIBRATION_BYTES}-byte limit; a fitted calibrator is a few "
                "KB. Refusing to load a file this large."
            )
        return cls.from_dict(json.loads(p.read_text()))


def fit_conformal_calibrator(
    estimates: np.ndarray,
    truth: np.ndarray,
    scale: np.ndarray,
    support: np.ndarray,
    *,
    alpha: float = 0.05,
    bin_edges: tuple[float, ...] = (20, 50, 100, 250),
    fallback_to_global: bool = True,
    training_scope: str = "fitted",
    event_types: np.ndarray | None = None,
    group_labels: np.ndarray | None = None,
    group_min_n: int = 20,
) -> ConformalCalibrator:
    """Fit a Mondrian conformal calibrator from observed (estimate, truth, scale, support).

    ``scale`` is the per-event uncertainty (e.g. the posterior std). Returns a
    calibrator carrying a global quantile plus one per support bin; bins with too
    few points fall back to the global quantile when ``fallback_to_global``. When
    ``event_types`` (one AS type per event) is given, an additional quantile per event
    type is fit (Mondrian by event type); a type with a non-finite quantile falls back
    to the global one. These take precedence over support bins at lookup time.
    """
    estimates = np.asarray(estimates, dtype=float)
    truth = np.asarray(truth, dtype=float)
    scale = np.asarray(scale, dtype=float)
    support = np.asarray(support, dtype=float)
    if not (estimates.shape == truth.shape == scale.shape == support.shape):
        raise ValueError("estimates, truth, scale, and support must have the same shape")
    if event_types is not None and np.asarray(event_types).shape != estimates.shape:
        raise ValueError("event_types must have the same shape as estimates")
    if group_labels is not None and np.asarray(group_labels).shape != estimates.shape:
        raise ValueError("group_labels must have the same shape as estimates")

    bins = assign_support_bins(support, bin_edges)
    q_global = conformal_quantile(nonconformity_scores(estimates, truth, scale), alpha)
    q_by_bin: dict[str, float] = {}
    for b in np.unique(bins):
        m = bins == b
        q_b = conformal_quantile(nonconformity_scores(estimates[m], truth[m], scale[m]), alpha)
        if not np.isfinite(q_b) and fallback_to_global:
            q_b = q_global
        q_by_bin[str(b)] = float(q_b)
    q_by_event_type: dict[str, float] = {}
    if event_types is not None:
        ets = np.asarray(event_types)
        for et in np.unique(ets):
            m = ets == et
            q_e = conformal_quantile(
                nonconformity_scores(estimates[m], truth[m], scale[m]), alpha)
            if not np.isfinite(q_e) and fallback_to_global:
                q_e = q_global
            q_by_event_type[str(et)] = float(q_e)
    q_by_group: dict[str, float] = {}
    if group_labels is not None:
        gl = np.asarray(group_labels)
        for g in np.unique(gl):
            m = gl == g
            if int(m.sum()) < group_min_n:
                continue  # sparse group -> resolved by the q_for cascade (no fit)
            q_g = conformal_quantile(
                nonconformity_scores(estimates[m], truth[m], scale[m]), alpha)
            if not np.isfinite(q_g) and fallback_to_global:
                q_g = q_global
            q_by_group[str(g)] = float(q_g)
    finite_support = support[np.isfinite(support)]
    profile: dict[str, float] = {"n": float(estimates.size)}
    if finite_support.size:
        profile.update(
            support_median=float(np.median(finite_support)),
            support_q25=float(np.percentile(finite_support, 25)),
            support_q75=float(np.percentile(finite_support, 75)),
        )
    return ConformalCalibrator(
        alpha=float(alpha),
        q_global=float(q_global),
        q_by_bin=q_by_bin,
        bin_edges=tuple(bin_edges),
        training_scope=training_scope,
        calibration_profile=profile,
        q_by_event_type=q_by_event_type,
        q_by_group=q_by_group,
    )


_DEFAULT_ARTIFACT = "default_psi_conformal.json"
_DIFFERENTIAL_ARTIFACT = "differential_dpsi_conformal.json"


@lru_cache(maxsize=1)
def load_default_conformal_calibrator() -> ConformalCalibrator:
    """Load the packaged default conformal calibrator shipped with BRAID.

    Resolves the artifact via ``importlib.resources`` so it works from an
    installed wheel. Raises ``FileNotFoundError`` if the package data is missing.
    """
    from importlib.resources import files

    resource = files("braid.target").joinpath("calibration_artifacts", _DEFAULT_ARTIFACT)
    with resource.open("r") as fh:
        return ConformalCalibrator.from_dict(json.load(fh))


@lru_cache(maxsize=1)
def load_differential_conformal_calibrator() -> ConformalCalibrator:
    """Load the packaged differential-ΔPSI conformal calibrator.

    This calibrator has ``scale_kind == "absolute_dpsi"``: its per-support-bin
    quantiles are absolute ΔPSI half-widths, fit on real RT-PCR residuals (TRA2 +
    circadian). The benchmark forms the interval as
    ``calibrator.interval(dpsi_mean, sigma=1.0, total_support, clip=(-1.0, 1.0))`` so
    the half-width is q itself (not q*std); the production ``braid differential`` path
    instead calls ``robust_interval(dpsi_mean, dpsi_std, total_support,
    event_type=...)``, which adds the sampling term in quadrature and honours any
    event-type/composite quantile through the ``q_for`` cascade. The shipped artifact
    carries only support-bin quantiles (its RT-PCR calibration set is cassette-exon
    only), so event_type currently cascades to the support bin; a refit calibrator with
    event-type quantiles would take effect without any code change. On real data this
    reaches nominal ΔPSI coverage where the Beta posterior and betAS under-cover.
    Raises ``FileNotFoundError`` if the package data is missing.
    """
    from importlib.resources import files

    resource = files("braid.target").joinpath(
        "calibration_artifacts", _DIFFERENTIAL_ARTIFACT)
    with resource.open("r") as fh:
        return ConformalCalibrator.from_dict(json.load(fh))


def require_scale_kind(
    calibrator: ConformalCalibrator, expected: str, context: str
) -> ConformalCalibrator:
    """Reject a custom calibrator whose ``scale_kind`` does not match the mode.

    The interval interpretation depends on ``scale_kind``: ``posterior_std`` scales the
    per-event posterior SD, while ``absolute_dpsi`` is an absolute ΔPSI half-width.
    Loading a calibrator fit for one mode into the other silently changes the interval
    units and breaks the coverage interpretation, so we fail fast with a clear error.
    """
    if calibrator.scale_kind != expected:
        raise ValueError(
            f"{context}: calibrator scale_kind={calibrator.scale_kind!r}, but this "
            f"mode requires {expected!r}; the interval units would be misinterpreted. "
            "Supply a calibrator fit for this mode."
        )
    return calibrator
