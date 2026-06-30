"""Caller-agnostic event record and the BRAID calibration step.

BRAID's calibrated interval needs only three things from an upstream splicing
caller: a per-event ΔPSI **point estimate**, a **read-support** value (to pick the
support-conditional conformal quantile), and an **event type** (SE/A3SS/...). Any
caller that emits a differential-splicing table can therefore be wrapped: the
adapters in this package normalise rMATS, MAJIQ, SUPPA2, and betAS output into the
:class:`CallerEvent` record below, and :func:`calibrate_events` adds the same
distribution-free calibrated 95% interval and confidence tier that
``braid differential`` produces for rMATS.

The calibration q is fit on real RT-PCR residuals (the packaged differential
calibrator). It is caller-agnostic *as a recipe*; for a finite-sample coverage
guarantee on a specific caller's point estimate you refit q on that caller's own
residuals (the within-study workflow), which ``--calibration PATH`` supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from braid.target.conformal import ConformalCalibrator

# A caller's read support is represented as ``total_support: float | None`` on
# CallerEvent. ``None`` -- not a numeric sentinel -- marks a caller that emits no
# countable support (e.g. SUPPA2 works from TPM/PSI, not junction counts). Using a
# magic value such as 1000.0 would collide with a real support of exactly 1000 and
# silently route a known-support event to the global quantile; ``None`` cannot
# collide. calibrate_event routes None-support events to the calibrator's pooled
# global quantile (the natural "no support information" default) rather than a
# support/event-type/composite bin, and surfaces it in the output (support_known).
KNOWN_EVENT_TYPES = ("SE", "A3SS", "A5SS", "MXE", "RI")


@dataclass(frozen=True)
class CallerEvent:
    """One differential-splicing event, normalised across callers.

    ``dpsi`` is the group1 − group2 point estimate (same orientation convention
    as rMATS ``IncLevelDifference``). ``total_support`` is the read evidence, or
    ``None`` for a caller that reports no countable support (SUPPA2, or MAJIQ
    without a reads column) -- such events are routed to the pooled global
    quantile, never a support bin. ``sampling_std`` is the within-RNA-seq standard
    deviation of ΔPSI when the caller exposes it (or it can be derived from junction
    counts), enabling the depth-robust interval; ``None`` falls back to the plain
    calibrated half-width.
    """

    event_id: str
    gene: str
    event_type: str
    dpsi: float
    total_support: float | None
    caller: str
    pvalue: float | None = None
    fdr: float | None = None
    sampling_std: float | None = None
    group1_psi: float | None = None
    group2_psi: float | None = None
    caller_low: float | None = None
    caller_high: float | None = None
    chrom: str = ""


def beta_dpsi_std(inc1: float, exc1: float, inc2: float, exc2: float) -> float:
    """Closed-form ΔPSI sampling SD from two independent Jeffreys-Beta posteriors.

    Each group's PSI ~ Beta(inc + 1/2, exc + 1/2); the variance of a Beta(a, b) is
    ``a b / ((a + b)^2 (a + b + 1))``. ΔPSI = PSI1 − PSI2 with independent groups,
    so ``Var(ΔPSI) = Var1 + Var2``. Used to give count-based callers a depth-aware
    sampling spread without drawing posterior samples.
    """

    def _var(inc: float, exc: float) -> float:
        a = max(float(inc), 0.0) + 0.5
        b = max(float(exc), 0.0) + 0.5
        s = a + b
        return a * b / (s * s * (s + 1.0))

    return float(np.sqrt(_var(inc1, exc1) + _var(inc2, exc2)))


def _caller_significant(ev: CallerEvent, threshold: float) -> bool | None:
    """Did the upstream caller flag this event? ``None`` if it reports neither."""
    if ev.fdr is not None and np.isfinite(ev.fdr):
        return bool(ev.fdr < threshold)
    if ev.pvalue is not None and np.isfinite(ev.pvalue):
        return bool(ev.pvalue < threshold)
    return None


def _event_type_key(event_type: str) -> str | None:
    """Pass a known rMATS type to the calibrator; unknown types cascade to support."""
    return event_type if event_type in KNOWN_EVENT_TYPES else None


def confidence_tier(reliable: bool, effect: bool, caller_sig: bool | None) -> str:
    """Confidence tier combining BRAID's calibrated interval with the caller's call.

    Shared by ``braid filter`` (via :func:`calibrate_event`) and ``braid
    differential`` as the common decision rule. The commands assign the same tier
    when they pass the same ``reliable`` / ``effect`` / ``caller_sig`` booleans;
    rMATS borderline cases can differ because ``filter`` reports caller-native
    estimates while ``differential`` uses its posterior-sampled center/interval.
    ``reliable`` = the calibrated interval excludes 0, ``effect`` = |ΔPSI| >= the
    effect cutoff, ``caller_sig`` = the upstream caller flagged it (rMATS FDR for
    differential; ``None`` only when the caller reports no significance at all).

    The informative case for users is ``caller-significant-only``: the upstream
    caller flagged the event but the calibrated interval (sized to orthogonal-truth
    discordance) still crosses zero, so it is a lower-priority validation target.
    """
    if caller_sig is None:  # caller gave no significance -> BRAID-only tiers
        if reliable and effect:
            return "high-confidence"
        if not reliable:
            return "not-reliable"
        # Reliable but sub-threshold effect: "supported" is defined as a reliable
        # *effect* (see below), so the identical (reliable, !effect) pair must NOT map
        # to "supported" here while the bool branch maps it to "not-significant". Mirror
        # the bool branch so the same booleans yield the same tier regardless of whether
        # the caller reports significance.
        return "not-significant"
    if reliable and effect and caller_sig:
        return "high-confidence"
    if reliable and effect:
        return "supported"  # BRAID-reliable effect the caller did not flag
    if caller_sig:
        return "caller-significant-only"  # flagged upstream but interval crosses 0
    return "not-significant"


def calibrate_event(
    ev: CallerEvent,
    calibrator: ConformalCalibrator,
    *,
    effect_cutoff: float = 0.1,
    sig_threshold: float = 0.05,
) -> dict[str, Any]:
    """Apply the BRAID calibrated interval + tier to one :class:`CallerEvent`."""
    et = _event_type_key(ev.event_type)
    clip = (-1.0, 1.0)
    # No read support (SUPPA2/MAJIQ) -> explicitly use the pooled global quantile, not
    # a support/event-type/composite bin that could be over-confident on a custom
    # calibrator. ``total_support is None`` is the contract; a numeric sentinel would
    # collide with a real support of that value and misroute a known-support event.
    # The bin arg is unused when force_global is set, so pass a harmless 0.0.
    force_global = ev.total_support is None
    support_for_bin = 0.0 if ev.total_support is None else ev.total_support
    if ev.sampling_std is not None and np.isfinite(ev.sampling_std):
        low, high = calibrator.robust_interval(
            ev.dpsi, ev.sampling_std, support_for_bin,
            event_type=et, force_global=force_global, clip=clip,
        )
    else:
        low, high = calibrator.interval(
            ev.dpsi, 1.0, support_for_bin,
            event_type=et, force_global=force_global, clip=clip,
        )
    reliable = bool(low > 0.0 or high < 0.0)
    effect = bool(abs(ev.dpsi) >= effect_cutoff)
    caller_sig = _caller_significant(ev, sig_threshold)
    return {
        "event_id": ev.event_id,
        "event_type": ev.event_type,
        "gene": ev.gene,
        "chrom": ev.chrom,
        "caller": ev.caller,
        "dpsi": float(ev.dpsi),
        "ci_low": float(low),
        "ci_high": float(high),
        "reliable": reliable,
        "effect": effect,
        "caller_significant": caller_sig,
        "tier": confidence_tier(reliable, effect, caller_sig),
        "pvalue": ev.pvalue,
        "fdr": ev.fdr,
        "total_support": (
            None if ev.total_support is None else float(ev.total_support)
        ),
        "support_known": ev.total_support is not None,
        "group1_psi": ev.group1_psi,
        "group2_psi": ev.group2_psi,
        "caller_low": ev.caller_low,
        "caller_high": ev.caller_high,
    }


def calibrate_events(
    events: list[CallerEvent],
    calibrator: ConformalCalibrator,
    *,
    effect_cutoff: float = 0.1,
    sig_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Calibrate every event; results carry the BRAID interval, reliable flag, tier.

    Sorted by descending calibrated distance of the interval from zero (the most
    confidently non-zero events first), so the top of the table is the validation
    short-list.
    """
    rows = [
        calibrate_event(
            e, calibrator, effect_cutoff=effect_cutoff, sig_threshold=sig_threshold
        )
        for e in events
    ]

    def _margin(r: dict) -> float:
        if r["ci_low"] > 0.0:
            return r["ci_low"]
        if r["ci_high"] < 0.0:
            return -r["ci_high"]
        return -1.0  # interval crosses zero -> below every reliable event

    rows.sort(key=_margin, reverse=True)
    return rows
