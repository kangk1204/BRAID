"""Caller-agnostic BRAID layer: wrap any splicing caller with calibrated intervals.

BRAID is a post-processing confidence layer, not a caller. The calibrated interval
needs only a ΔPSI point estimate, a read-support value, and an event type, so it
can be layered on rMATS, MAJIQ, SUPPA2, or betAS. This package normalises each
caller's native differential table into :class:`CallerEvent`, then
:func:`calibrate_events` adds the same distribution-free calibrated 95% interval
and confidence tier across all of them.
"""

from braid.adapters.base import (
    KNOWN_EVENT_TYPES,
    CallerEvent,
    beta_dpsi_std,
    calibrate_event,
    calibrate_events,
    confidence_tier,
)
from braid.adapters.parsers import (
    PARSERS,
    ParserConfig,
    ParseResult,
    ParseSummary,
    parse_betas,
    parse_majiq,
    parse_rmats,
    parse_suppa2,
)

__all__ = [
    "CallerEvent",
    "KNOWN_EVENT_TYPES",
    "beta_dpsi_std",
    "calibrate_event",
    "calibrate_events",
    "confidence_tier",
    "PARSERS",
    "ParserConfig",
    "ParseSummary",
    "ParseResult",
    "parse_rmats",
    "parse_majiq",
    "parse_suppa2",
    "parse_betas",
]
