"""BRAID v2 PSI inference for local alternative splicing events.

This module now follows a simple pipeline:
1. Extract filtered local evidence from BAM
2. Propose events from annotation and/or junction topology
3. Infer PSI with calibrated posterior intervals

The compatibility wrapper ``compute_psi_from_junctions`` is retained for
existing callers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from braid.target.extractor import load_gene_transcripts

logger = logging.getLogger(__name__)

DEFAULT_MIN_MAPQ = 1
CONFIDENT_CI_WIDTH_THRESHOLD = 0.2
CONFIDENT_CV_THRESHOLD = 0.5
DEFAULT_EVENT_SOURCE = "hybrid"
DEFAULT_UNCERTAINTY_MODEL = "overdispersed"
DEFAULT_MIN_EVENT_SUPPORT = 5
OVERDISPERSED_COUNT_SCALE = 0.01
SUPPORT_BIN_LABELS: tuple[tuple[int, int | None, str], ...] = (
    (0, 19, "<20"),
    (20, 49, "20-49"),
    (50, 99, "50-99"),
    (100, 249, "100-249"),
    (250, None, "250+"),
)
LEGACY_SUPPORT_SCALE_FACTORS = (
    (20, 0.85),
    (50, 1.0),
    (100, 1.2),
    (200, 1.5),
)
LEGACY_EVENT_TYPE_SCALE_FACTORS = {
    "A3SS": 1.0,
    "A5SS": 1.0,
    "SE": 0.9,
    "RI": 0.85,
    "MXE": 0.95,
}
NATIVE_JUNCTION_CHOICE_AUDIT_INTERVAL_FACTORS = {
    "<20": 1.0648875689615864,
    "20-49": 1.0648875689615864,
    "50-99": 1.1834802282272903,
    "100-249": 1.3553905316339745,
    "250+": 1.9906365530112617,
}
NATIVE_JUNCTION_CHOICE_INTERVAL_INFLATION_FACTORS = {
    "<20": 1.022158102170707,
    "20-49": 1.022158102170707,
    "50-99": 1.0255363419919137,
    "100-249": 1.2001565486593335,
    "250+": 1.2001565486593335,
}
NATIVE_JUNCTION_CHOICE_CONFIDENCE_WIDTH_THRESHOLDS = {
    "<20": CONFIDENT_CI_WIDTH_THRESHOLD,
    "20-49": CONFIDENT_CI_WIDTH_THRESHOLD,
    "50-99": CONFIDENT_CI_WIDTH_THRESHOLD,
    "100-249": CONFIDENT_CI_WIDTH_THRESHOLD,
    "250+": 0.60,
}


@dataclass(frozen=True)
class BRAIDConfig:
    """Configuration for BRAID v2 local PSI inference."""

    n_replicates: int = 500
    confidence_level: float = 0.95
    min_mapq: int = DEFAULT_MIN_MAPQ
    min_event_support: int = DEFAULT_MIN_EVENT_SUPPORT
    event_source: str = DEFAULT_EVENT_SOURCE
    annotation_gtf: str | None = None
    uncertainty_model: str = DEFAULT_UNCERTAINTY_MODEL
    seed: int | None = None
    schedule_mode: str = "native"
    calibration_schedule: dict[str, object] | None = None


@dataclass
class PSIResult:
    """PSI estimate with calibrated confidence interval."""

    event_id: str
    event_type: str
    gene: str
    chrom: str
    psi: float
    ci_low: float
    ci_high: float
    cv: float
    inclusion_count: int
    exclusion_count: int
    event_start: int | None = None
    event_end: int | None = None
    ci_width: float = 0.0
    is_confident: bool = False
    n_candidate_junctions: int = 0
    n_supported_junctions: int = 0
    proposal_source: str = ""
    evidence_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadSummary:
    """Minimal alignment summary needed for body evidence counting."""

    start: int
    end: int
    has_splice: bool


@dataclass(frozen=True)
class SpliceEvent:
    """Canonical event object produced before PSI inference."""

    event_id: str
    event_type: str
    gene: str
    chrom: str
    event_start: int | None
    event_end: int | None
    inclusion_junctions: tuple[tuple[int, int], ...] = ()
    exclusion_junctions: tuple[tuple[int, int], ...] = ()
    auxiliary_junctions: tuple[tuple[int, int], ...] = ()
    body_region: tuple[int, int] | None = None
    proposal_source: str = "denovo"
    metadata: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EventEvidence:
    """Observed support for one proposed event."""

    inclusion_count: int
    exclusion_count: int
    body_count: int = 0
    n_candidate_junctions: int = 0
    n_supported_junctions: int = 0
    evidence_breakdown: dict[str, int] = field(default_factory=dict)


def build_se_splice_event(
    *,
    event_id: str,
    gene: str,
    chrom: str,
    exon_start: int,
    exon_end: int,
    upstream_ee: int,
    downstream_es: int,
    upstream_es: int | None = None,
    downstream_ee: int | None = None,
    proposal_source: str = "external",
) -> SpliceEvent:
    """Build one canonical SE event from explicit exon/flank boundaries."""
    metadata = {
        "n_candidate_junctions": 3,
        "upstream_ee": upstream_ee,
        "downstream_es": downstream_es,
    }
    if upstream_es is not None:
        metadata["upstream_es"] = upstream_es
    if downstream_ee is not None:
        metadata["downstream_ee"] = downstream_ee
    return SpliceEvent(
        event_id=event_id,
        event_type="SE",
        gene=gene,
        chrom=chrom,
        event_start=exon_start,
        event_end=exon_end,
        inclusion_junctions=((upstream_ee, exon_start),),
        auxiliary_junctions=((exon_end, downstream_es),),
        exclusion_junctions=((upstream_ee, downstream_es),),
        body_region=(exon_start, exon_end),
        proposal_source=proposal_source,
        metadata=metadata,
    )


def merge_event_evidence(evidences: list[EventEvidence]) -> EventEvidence:
    """Merge multiple BAM-level evidence objects into one aggregated view."""
    breakdown: dict[str, int] = defaultdict(int)
    for evidence in evidences:
        for key, value in evidence.evidence_breakdown.items():
            breakdown[key] += value
    return EventEvidence(
        inclusion_count=sum(e.inclusion_count for e in evidences),
        exclusion_count=sum(e.exclusion_count for e in evidences),
        body_count=sum(e.body_count for e in evidences),
        n_candidate_junctions=max(
            (e.n_candidate_junctions for e in evidences),
            default=0,
        ),
        n_supported_junctions=max(
            (e.n_supported_junctions for e in evidences),
            default=0,
        ),
        evidence_breakdown=dict(breakdown),
    )


def extract_event_evidence_from_bam(
    bam_path: str,
    event: SpliceEvent,
    *,
    min_mapq: int = DEFAULT_MIN_MAPQ,
) -> EventEvidence:
    """Measure one event's evidence directly from a BAM region."""
    coords: list[int] = []
    for start, end in (
        event.inclusion_junctions
        + event.auxiliary_junctions
        + event.exclusion_junctions
    ):
        coords.extend((start, end))
    if event.body_region is not None:
        coords.extend(event.body_region)
    if not coords:
        return EventEvidence(0, 0)

    region_start = max(0, min(coords) - 1)
    region_end = max(coords) + 1
    junction_counts, reads = _extract_region_evidence(
        bam_path=bam_path,
        chrom=event.chrom,
        start=region_start,
        end=region_end,
        min_mapq=min_mapq,
    )
    return _measure_event_evidence(event, junction_counts, reads)


def _is_usable_alignment(read, min_mapq: int) -> bool:
    """Return True when a read is appropriate for PSI counting."""
    if read.is_unmapped or read.cigartuples is None:
        return False
    if read.is_secondary or read.is_supplementary:
        return False
    if read.is_duplicate or read.is_qcfail:
        return False
    if read.mapping_quality < min_mapq:
        return False
    return True


def _has_splice(read) -> bool:
    """Return True when the alignment contains an intron skip."""
    return any(op == 3 for op, _length in (read.cigartuples or ()))


def _is_confident_interval(
    ci_low: float,
    ci_high: float,
    cv: float,
    *,
    psi: float | None = None,
    event_type: str | None = None,
    inclusion_count: int = 0,
    exclusion_count: int = 0,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> tuple[float, bool]:
    """Return interval width and a conservative confidence flag."""
    ci_width = ci_high - ci_low
    width_threshold = confidence_width_threshold(
        inclusion_count,
        exclusion_count,
        event_type=event_type,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    cv_threshold = confidence_cv_threshold(
        inclusion_count,
        exclusion_count,
        event_type=event_type,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    strict_confident = (
        ci_width < width_threshold
        and np.isfinite(cv)
        and cv <= cv_threshold
    )
    effect_confident = False
    effect_threshold = confidence_effect_threshold(
        inclusion_count,
        exclusion_count,
        event_type=event_type,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    effect_snr_threshold = confidence_effect_snr_threshold(
        inclusion_count,
        exclusion_count,
        event_type=event_type,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    effect_cv_threshold = confidence_effect_cv_threshold(
        inclusion_count,
        exclusion_count,
        event_type=event_type,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    if (
        psi is not None
        and effect_threshold is not None
        and effect_snr_threshold is not None
    ):
        effect_strength = abs(float(psi) - 0.5)
        effect_snr = effect_strength / max(ci_width, 1e-9)
        effect_confident = (
            effect_strength >= effect_threshold
            and effect_snr >= effect_snr_threshold
            and (
                effect_cv_threshold is None
                or (
                    np.isfinite(cv)
                    and cv <= effect_cv_threshold
                )
            )
        )
    return ci_width, (strict_confident or effect_confident)


def classify_support_bin(total_support: int) -> str:
    """Assign one event support total to a named support bin."""
    for lower, upper, label in SUPPORT_BIN_LABELS:
        if total_support < lower:
            continue
        if upper is None or total_support <= upper:
            return label
    return SUPPORT_BIN_LABELS[-1][2]


def _default_native_calibration_schedule(
    *,
    base_scale: float | None = None,
) -> dict[str, object]:
    """Build the built-in native calibration schedule."""
    scale = OVERDISPERSED_COUNT_SCALE if base_scale is None else base_scale
    scale_by_bin: dict[str, float] = {}
    audit_by_bin: dict[str, float] = {}
    for _lower, _upper, label in SUPPORT_BIN_LABELS:
        audit_factor = NATIVE_JUNCTION_CHOICE_AUDIT_INTERVAL_FACTORS[label]
        audit_by_bin[label] = audit_factor
        scale_by_bin[label] = min(1.0, scale / (audit_factor ** 2))
    return {
        "mode": "builtin_native_schedule",
        "base_scale": scale,
        "training_scope": "builtin_default",
        "scale_by_bin": scale_by_bin,
        "audit_interval_factor_by_bin": audit_by_bin,
        "interval_inflation_by_bin": dict(NATIVE_JUNCTION_CHOICE_INTERVAL_INFLATION_FACTORS),
        "confidence_width_by_bin": dict(NATIVE_JUNCTION_CHOICE_CONFIDENCE_WIDTH_THRESHOLDS),
        "confidence_cv_by_bin": {
            label: CONFIDENT_CV_THRESHOLD
            for _lower, _upper, label in SUPPORT_BIN_LABELS
        },
        "confidence_effect_by_bin": {},
        "confidence_effect_snr_by_bin": {},
        "confidence_effect_cv_by_bin": {},
    }


def _resolve_native_calibration_schedule(
    calibration_schedule: dict[str, object] | None,
    *,
    base_scale: float | None = None,
) -> dict[str, object]:
    """Normalize one optional native calibration override."""
    schedule = _default_native_calibration_schedule(base_scale=base_scale)
    if not calibration_schedule:
        return schedule

    merged = dict(schedule)
    merged.update(calibration_schedule)
    merged_base_scale = float(merged.get("base_scale", schedule["base_scale"]))
    merged["base_scale"] = merged_base_scale

    default_scale_by_bin = dict(schedule["scale_by_bin"])
    override_scale_by_bin = {
        str(label): float(value)
        for label, value in (calibration_schedule.get("scale_by_bin") or {}).items()
    }
    scale_by_bin = {**default_scale_by_bin, **override_scale_by_bin}
    merged["scale_by_bin"] = scale_by_bin

    default_interval_by_bin = dict(schedule["interval_inflation_by_bin"])
    override_interval_by_bin = {
        str(label): float(value)
        for label, value in (calibration_schedule.get("interval_inflation_by_bin") or {}).items()
    }
    interval_by_bin = {**default_interval_by_bin, **override_interval_by_bin}
    merged["interval_inflation_by_bin"] = interval_by_bin

    default_conf_by_bin = dict(schedule["confidence_width_by_bin"])
    override_conf_by_bin = {
        str(label): float(value)
        for label, value in (calibration_schedule.get("confidence_width_by_bin") or {}).items()
    }
    conf_by_bin = {**default_conf_by_bin, **override_conf_by_bin}
    merged["confidence_width_by_bin"] = conf_by_bin

    default_cv_by_bin = dict(schedule["confidence_cv_by_bin"])
    override_cv_by_bin = {
        str(label): float(value)
        for label, value in (calibration_schedule.get("confidence_cv_by_bin") or {}).items()
    }
    cv_by_bin = {**default_cv_by_bin, **override_cv_by_bin}
    merged["confidence_cv_by_bin"] = cv_by_bin

    default_effect_by_bin = dict(schedule["confidence_effect_by_bin"])
    override_effect_by_bin = {
        str(label): float(value)
        for label, value in (calibration_schedule.get("confidence_effect_by_bin") or {}).items()
    }
    effect_by_bin = {**default_effect_by_bin, **override_effect_by_bin}
    merged["confidence_effect_by_bin"] = effect_by_bin

    default_effect_snr_by_bin = dict(schedule["confidence_effect_snr_by_bin"])
    override_effect_snr_by_bin = {
        str(label): float(value)
        for label, value in (calibration_schedule.get("confidence_effect_snr_by_bin") or {}).items()
    }
    effect_snr_by_bin = {**default_effect_snr_by_bin, **override_effect_snr_by_bin}
    merged["confidence_effect_snr_by_bin"] = effect_snr_by_bin

    default_effect_cv_by_bin = dict(schedule["confidence_effect_cv_by_bin"])
    override_effect_cv_by_bin = {
        str(label): float(value)
        for label, value in (calibration_schedule.get("confidence_effect_cv_by_bin") or {}).items()
    }
    effect_cv_by_bin = {**default_effect_cv_by_bin, **override_effect_cv_by_bin}
    merged["confidence_effect_cv_by_bin"] = effect_cv_by_bin

    default_audit_by_bin = dict(schedule["audit_interval_factor_by_bin"])
    override_audit_by_bin = {
        str(label): float(value)
        for label, value in (calibration_schedule.get("audit_interval_factor_by_bin") or {}).items()
    }
    audit_by_bin = {**default_audit_by_bin, **override_audit_by_bin}
    for label, value in scale_by_bin.items():
        if label not in override_audit_by_bin:
            audit_by_bin[label] = (
                (merged_base_scale / max(value, 1e-9)) ** 0.5
                if value > 0
                else float("inf")
            )
    merged["audit_interval_factor_by_bin"] = audit_by_bin
    return merged


def native_count_scale_schedule(
    *,
    event_type: str = "A3SS",
    base_scale: float | None = None,
) -> dict[str, dict[str, float | str]]:
    """Return the native support-aware scale schedule for one event type."""
    calibration = _resolve_native_calibration_schedule(None, base_scale=base_scale)
    scale = float(calibration["base_scale"])
    scale_by_bin = calibration["scale_by_bin"]
    audit_by_bin = calibration["audit_interval_factor_by_bin"]
    schedule: dict[str, dict[str, float | str]] = {}
    for _lower, _upper, label in SUPPORT_BIN_LABELS:
        schedule[label] = {
            "event_type": event_type,
            "base_scale": scale,
            "support_bin": label,
            "audit_interval_factor": float(audit_by_bin[label]),
            "effective_scale": float(scale_by_bin[label]),
            "mode": "native_support_schedule",
        }
    return schedule


def native_interval_inflation_schedule(
    *,
    event_type: str = "A3SS",
) -> dict[str, dict[str, float | str]]:
    """Return the native support-aware interval inflation schedule."""
    calibration = _resolve_native_calibration_schedule(None)
    interval_by_bin = calibration["interval_inflation_by_bin"]
    schedule: dict[str, dict[str, float | str]] = {}
    for _lower, _upper, label in SUPPORT_BIN_LABELS:
        schedule[label] = {
            "event_type": event_type,
            "support_bin": label,
            "interval_inflation_factor": float(interval_by_bin[label]),
            "mode": "native_interval_inflation",
        }
    return schedule


def native_confidence_width_schedule(
    *,
    event_type: str = "A3SS",
) -> dict[str, dict[str, float | str]]:
    """Return the native support-aware confidence-width schedule."""
    calibration = _resolve_native_calibration_schedule(None)
    conf_by_bin = calibration["confidence_width_by_bin"]
    schedule: dict[str, dict[str, float | str]] = {}
    for _lower, _upper, label in SUPPORT_BIN_LABELS:
        schedule[label] = {
            "event_type": event_type,
            "support_bin": label,
            "confidence_width_threshold": float(conf_by_bin[label]),
            "mode": "native_confidence_width",
        }
    return schedule


def native_confidence_cv_schedule(
    *,
    event_type: str = "A3SS",
) -> dict[str, dict[str, float | str]]:
    """Return the native support-aware confidence-CV schedule."""
    calibration = _resolve_native_calibration_schedule(None)
    cv_by_bin = calibration["confidence_cv_by_bin"]
    schedule: dict[str, dict[str, float | str]] = {}
    for _lower, _upper, label in SUPPORT_BIN_LABELS:
        schedule[label] = {
            "event_type": event_type,
            "support_bin": label,
            "confidence_cv_threshold": float(cv_by_bin[label]),
            "mode": "native_confidence_cv",
        }
    return schedule


def native_confidence_effect_schedule(
    *,
    event_type: str = "A3SS",
) -> dict[str, dict[str, float | str]]:
    """Return native effect-strength confidence overrides for enabled bins."""
    calibration = _resolve_native_calibration_schedule(None)
    effect_by_bin = calibration["confidence_effect_by_bin"]
    schedule: dict[str, dict[str, float | str]] = {}
    for _lower, _upper, label in SUPPORT_BIN_LABELS:
        if label not in effect_by_bin:
            continue
        schedule[label] = {
            "event_type": event_type,
            "support_bin": label,
            "confidence_effect_threshold": float(effect_by_bin[label]),
            "mode": "native_confidence_effect",
        }
    return schedule


def native_confidence_effect_snr_schedule(
    *,
    event_type: str = "A3SS",
) -> dict[str, dict[str, float | str]]:
    """Return native effect-SNR confidence overrides for enabled bins."""
    calibration = _resolve_native_calibration_schedule(None)
    effect_snr_by_bin = calibration["confidence_effect_snr_by_bin"]
    schedule: dict[str, dict[str, float | str]] = {}
    for _lower, _upper, label in SUPPORT_BIN_LABELS:
        if label not in effect_snr_by_bin:
            continue
        schedule[label] = {
            "event_type": event_type,
            "support_bin": label,
            "confidence_effect_snr_threshold": float(effect_snr_by_bin[label]),
            "mode": "native_confidence_effect_snr",
        }
    return schedule


def native_confidence_effect_cv_schedule(
    *,
    event_type: str = "A3SS",
) -> dict[str, dict[str, float | str]]:
    """Return optional native effect-gate CV overrides for enabled bins."""
    calibration = _resolve_native_calibration_schedule(None)
    effect_cv_by_bin = calibration["confidence_effect_cv_by_bin"]
    schedule: dict[str, dict[str, float | str]] = {}
    for _lower, _upper, label in SUPPORT_BIN_LABELS:
        if label not in effect_cv_by_bin:
            continue
        schedule[label] = {
            "event_type": event_type,
            "support_bin": label,
            "confidence_effect_cv_threshold": float(effect_cv_by_bin[label]),
            "mode": "native_confidence_effect_cv",
        }
    return schedule


def effective_count_scale_metadata(
    inclusion_count: int,
    exclusion_count: int,
    *,
    event_type: str | None = None,
    base_scale: float | None = None,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> dict[str, float | str | int]:
    """Return scale metadata for one posterior draw configuration."""
    scale = OVERDISPERSED_COUNT_SCALE if base_scale is None else base_scale
    total = max(inclusion_count + exclusion_count, 0)
    support_bin = classify_support_bin(total)

    if schedule_mode == "native" and event_type in {"A3SS", "A5SS"}:
        calibration = _resolve_native_calibration_schedule(
            calibration_schedule,
            base_scale=scale,
        )
        scale_by_bin = calibration["scale_by_bin"]
        audit_by_bin = calibration["audit_interval_factor_by_bin"]
        effective_scale = float(scale_by_bin[support_bin])
        audit_factor = float(audit_by_bin[support_bin])
        return {
            "schedule_mode": schedule_mode,
            "event_type": event_type or "",
            "total_support": total,
            "support_bin": support_bin,
            "base_scale": scale,
            "audit_interval_factor": audit_factor,
            "effective_scale": effective_scale,
            "training_scope": str(calibration.get("training_scope", "builtin_default")),
        }

    if schedule_mode == "fixed":
        return {
            "schedule_mode": schedule_mode,
            "event_type": event_type or "",
            "total_support": total,
            "support_bin": support_bin,
            "base_scale": scale,
            "effective_scale": min(1.0, scale),
        }

    event_factor = LEGACY_EVENT_TYPE_SCALE_FACTORS.get(event_type or "", 1.0)
    support_factor = 1.0
    for upper_bound, factor in LEGACY_SUPPORT_SCALE_FACTORS:
        if total < upper_bound:
            support_factor = factor
            break
    else:
        support_factor = 1.75
    effective_scale = min(1.0, scale * event_factor * support_factor)
    return {
        "schedule_mode": schedule_mode,
        "event_type": event_type or "",
        "total_support": total,
        "support_bin": support_bin,
        "base_scale": scale,
        "event_factor": event_factor,
        "support_factor": support_factor,
        "effective_scale": effective_scale,
    }


def effective_count_scale(
    inclusion_count: int,
    exclusion_count: int,
    *,
    event_type: str | None = None,
    base_scale: float | None = None,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> float:
    """Return an event-aware count scale for overdispersed posterior draws."""
    metadata = effective_count_scale_metadata(
        inclusion_count,
        exclusion_count,
        event_type=event_type,
        base_scale=base_scale,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    return float(metadata["effective_scale"])


def native_interval_inflation_factor(
    inclusion_count: int,
    exclusion_count: int,
    *,
    event_type: str | None = None,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> float:
    """Return the native support-aware interval inflation factor."""
    if schedule_mode != "native" or event_type not in {"A3SS", "A5SS"}:
        return 1.0
    calibration = _resolve_native_calibration_schedule(calibration_schedule)
    total = max(inclusion_count + exclusion_count, 0)
    support_bin = classify_support_bin(total)
    return float(calibration["interval_inflation_by_bin"][support_bin])


def confidence_width_threshold(
    inclusion_count: int,
    exclusion_count: int,
    *,
    event_type: str | None = None,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> float:
    """Return the confidence-width threshold for one event."""
    if schedule_mode != "native" or event_type not in {"A3SS", "A5SS"}:
        return CONFIDENT_CI_WIDTH_THRESHOLD
    calibration = _resolve_native_calibration_schedule(calibration_schedule)
    total = max(inclusion_count + exclusion_count, 0)
    support_bin = classify_support_bin(total)
    return float(calibration["confidence_width_by_bin"][support_bin])


def confidence_cv_threshold(
    inclusion_count: int,
    exclusion_count: int,
    *,
    event_type: str | None = None,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> float:
    """Return the confidence CV threshold for one event."""
    if schedule_mode != "native" or event_type not in {"A3SS", "A5SS"}:
        return CONFIDENT_CV_THRESHOLD
    calibration = _resolve_native_calibration_schedule(calibration_schedule)
    total = max(inclusion_count + exclusion_count, 0)
    support_bin = classify_support_bin(total)
    return float(calibration["confidence_cv_by_bin"][support_bin])


def confidence_effect_threshold(
    inclusion_count: int,
    exclusion_count: int,
    *,
    event_type: str | None = None,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> float | None:
    """Return one optional effect-strength gate for native confidence calls."""
    total = max(inclusion_count + exclusion_count, 0)
    if schedule_mode != "native" or event_type not in {"A3SS", "A5SS"}:
        return None
    calibration = _resolve_native_calibration_schedule(calibration_schedule)
    support_bin = classify_support_bin(total)
    value = calibration["confidence_effect_by_bin"].get(support_bin)
    if value is None:
        return None
    return float(value)


def confidence_effect_snr_threshold(
    inclusion_count: int,
    exclusion_count: int,
    *,
    event_type: str | None = None,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> float | None:
    """Return one optional effect-to-width signal threshold."""
    total = max(inclusion_count + exclusion_count, 0)
    if schedule_mode != "native" or event_type not in {"A3SS", "A5SS"}:
        return None
    calibration = _resolve_native_calibration_schedule(calibration_schedule)
    support_bin = classify_support_bin(total)
    value = calibration["confidence_effect_snr_by_bin"].get(support_bin)
    if value is None:
        return None
    return float(value)


def confidence_effect_cv_threshold(
    inclusion_count: int,
    exclusion_count: int,
    *,
    event_type: str | None = None,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> float | None:
    """Return one optional CV ceiling for effect-aware confidence overrides."""
    total = max(inclusion_count + exclusion_count, 0)
    if schedule_mode != "native" or event_type not in {"A3SS", "A5SS"}:
        return None
    calibration = _resolve_native_calibration_schedule(calibration_schedule)
    support_bin = classify_support_bin(total)
    value = calibration["confidence_effect_cv_by_bin"].get(support_bin)
    if value is None:
        return None
    return float(value)


def sample_psi_posterior(
    inclusion_count: int,
    exclusion_count: int,
    *,
    n_replicates: int = 500,
    seed: int | None = None,
    model: str = DEFAULT_UNCERTAINTY_MODEL,
    event_type: str | None = None,
    base_scale: float | None = None,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> np.ndarray:
    """Draw posterior PSI samples for one event."""
    total = inclusion_count + exclusion_count
    if total <= 0:
        return np.zeros(n_replicates, dtype=float)

    rng = np.random.default_rng(seed)
    if model == "poisson":
        inc_samples = rng.poisson(max(inclusion_count, 0.5), size=n_replicates)
        exc_samples = rng.poisson(max(exclusion_count, 0.5), size=n_replicates)
        totals = inc_samples + exc_samples
        psi_samples = np.zeros(n_replicates, dtype=float)
        valid = totals > 0
        psi_samples[valid] = inc_samples[valid] / totals[valid]
        return psi_samples

    count_scale = effective_count_scale(
        inclusion_count,
        exclusion_count,
        event_type=event_type,
        base_scale=base_scale,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    return rng.beta(
        inclusion_count * count_scale + 0.5,
        exclusion_count * count_scale + 0.5,
        size=n_replicates,
    )


def bootstrap_psi(
    inclusion_count: int,
    exclusion_count: int,
    n_replicates: int = 500,
    confidence_level: float = 0.95,
    seed: int | None = None,
    model: str = DEFAULT_UNCERTAINTY_MODEL,
    event_type: str | None = None,
    base_scale: float | None = None,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> tuple[float, float, float, float]:
    """Compute PSI with a posterior predictive confidence interval."""
    total = inclusion_count + exclusion_count
    if total == 0:
        return 0.0, 0.0, 0.0, float("nan")

    psi = inclusion_count / total
    psi_samples = sample_psi_posterior(
        inclusion_count,
        exclusion_count,
        n_replicates=n_replicates,
        seed=seed,
        model=model,
        event_type=event_type,
        base_scale=base_scale,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(psi_samples, 100 * alpha / 2))
    ci_high = float(np.percentile(psi_samples, 100 * (1 - alpha / 2)))
    inflation_factor = native_interval_inflation_factor(
        inclusion_count,
        exclusion_count,
        event_type=event_type,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )
    if inflation_factor > 1.0:
        low_span = max(0.0, psi - ci_low)
        high_span = max(0.0, ci_high - psi)
        ci_low = max(0.0, psi - inflation_factor * low_span)
        ci_high = min(1.0, psi + inflation_factor * high_span)
    mean_psi = float(np.mean(psi_samples))
    std_psi = float(np.std(psi_samples))
    cv = std_psi / mean_psi if mean_psi > 0 else float("nan")
    return psi, ci_low, ci_high, cv


def _extract_region_evidence(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
    min_mapq: int,
) -> tuple[dict[tuple[int, int], int], list[ReadSummary]]:
    """Extract local junction counts and simplified read summaries."""
    import pysam

    junction_counts: dict[tuple[int, int], int] = {}
    read_summaries: list[ReadSummary] = []

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        candidate_chroms = [chrom]
        if chrom.startswith("chr"):
            candidate_chroms.append(chrom[3:])
        else:
            candidate_chroms.append(f"chr{chrom}")

        selected_chrom = None
        references = set(getattr(bam, "references", ()) or ())
        for candidate in candidate_chroms:
            if references and candidate not in references:
                continue
            selected_chrom = candidate
            break
        if selected_chrom is None:
            selected_chrom = chrom

        try:
            read_iter = bam.fetch(selected_chrom, start, end)
        except ValueError:
            fallback_iter = None
            for candidate in candidate_chroms:
                if candidate == selected_chrom:
                    continue
                try:
                    fallback_iter = bam.fetch(candidate, start, end)
                    selected_chrom = candidate
                    break
                except ValueError:
                    continue
            if fallback_iter is None:
                return junction_counts, read_summaries
            read_iter = fallback_iter

        for read in read_iter:
            if not _is_usable_alignment(read, min_mapq):
                continue
            read_summaries.append(ReadSummary(
                start=read.reference_start,
                end=read.reference_end,
                has_splice=_has_splice(read),
            ))

            pos = read.reference_start
            for op, length in read.cigartuples:
                if op == 3:
                    jstart, jend = pos, pos + length
                    if jstart >= start and jend <= end:
                        key = (jstart, jend)
                        junction_counts[key] = junction_counts.get(key, 0) + 1
                if op in (0, 2, 3, 7, 8):
                    pos += length

    return junction_counts, read_summaries


def _propose_grouped_junction_events(
    junction_counts: dict[tuple[int, int], int],
    gene: str,
    chrom: str,
) -> list[SpliceEvent]:
    """Propose grouped A3SS/A5SS events from shared splice sites."""
    events: list[SpliceEvent] = []

    donors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    acceptors: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for jstart, jend in junction_counts:
        donors[jstart].append((jstart, jend))
        acceptors[jend].append((jstart, jend))

    for donor, juncs in donors.items():
        if len(juncs) < 2:
            continue
        for junc in sorted(juncs):
            others = tuple(sorted(oj for oj in juncs if oj != junc))
            events.append(SpliceEvent(
                event_id=f"A3SS:{junc[0]}-{junc[1]}",
                event_type="A3SS",
                gene=gene,
                chrom=chrom,
                event_start=junc[0],
                event_end=junc[1],
                inclusion_junctions=(junc,),
                exclusion_junctions=others,
                proposal_source="denovo",
                metadata={
                    "n_candidate_junctions": len(juncs),
                },
            ))

    for acceptor, juncs in acceptors.items():
        if len(juncs) < 2:
            continue
        for junc in sorted(juncs):
            others = tuple(sorted(oj for oj in juncs if oj != junc))
            events.append(SpliceEvent(
                event_id=f"A5SS:{junc[0]}-{junc[1]}",
                event_type="A5SS",
                gene=gene,
                chrom=chrom,
                event_start=junc[0],
                event_end=junc[1],
                inclusion_junctions=(junc,),
                exclusion_junctions=others,
                proposal_source="denovo",
                metadata={
                    "n_candidate_junctions": len(juncs),
                },
            ))

    return events


def _propose_annotation_se_events(
    gene: str,
    chrom: str,
    annotation_gtf: str,
) -> list[SpliceEvent]:
    """Propose skipped-exon events directly from annotation transcript models."""
    exon_events: dict[tuple[int, int], dict[str, set[tuple[int, int]]]] = {}
    transcripts = load_gene_transcripts(annotation_gtf, gene, chrom=chrom)

    for transcript in transcripts:
        exons = list(transcript.exons)
        if len(exons) < 3:
            continue
        for idx in range(1, len(exons) - 1):
            prev_exon = exons[idx - 1]
            exon = exons[idx]
            next_exon = exons[idx + 1]
            key = exon
            entry = exon_events.setdefault(key, {
                "left": set(),
                "right": set(),
                "skip": set(),
            })
            entry["left"].add((prev_exon[1], exon[0]))
            entry["right"].add((exon[1], next_exon[0]))
            entry["skip"].add((prev_exon[1], next_exon[0]))

    events: list[SpliceEvent] = []
    for (exon_start, exon_end), groups in sorted(exon_events.items()):
        events.append(SpliceEvent(
            event_id=f"SE:{exon_start}-{exon_end}",
            event_type="SE",
            gene=gene,
            chrom=chrom,
            event_start=exon_start,
            event_end=exon_end,
            inclusion_junctions=tuple(sorted(groups["left"])),
            auxiliary_junctions=tuple(sorted(groups["right"])),
            exclusion_junctions=tuple(sorted(groups["skip"])),
            body_region=(exon_start, exon_end),
            proposal_source="annotation",
            metadata={
                "n_candidate_junctions": (
                    len(groups["left"])
                    + len(groups["right"])
                    + len(groups["skip"])
                ),
            },
        ))
    return events


def _propose_de_novo_se_events(
    junction_counts: dict[tuple[int, int], int],
    gene: str,
    chrom: str,
) -> list[SpliceEvent]:
    """Fallback SE proposal from observed junction topology only."""
    event_map: dict[tuple[int, int], dict[str, set[tuple[int, int]]]] = {}
    all_juncs = sorted(junction_counts.keys())
    for d1, a1 in all_juncs:
        for d2, a2 in all_juncs:
            if d2 <= a1 or d2 - a1 > 50000:
                continue
            skip_key = (d1, a2)
            if skip_key not in junction_counts:
                continue
            key = (a1, d2)
            entry = event_map.setdefault(key, {
                "left": set(),
                "right": set(),
                "skip": set(),
            })
            entry["left"].add((d1, a1))
            entry["right"].add((d2, a2))
            entry["skip"].add(skip_key)

    events: list[SpliceEvent] = []
    for (event_start, event_end), groups in sorted(event_map.items()):
        events.append(SpliceEvent(
            event_id=f"SE:{event_start}-{event_end}",
            event_type="SE",
            gene=gene,
            chrom=chrom,
            event_start=event_start,
            event_end=event_end,
            inclusion_junctions=tuple(sorted(groups["left"])),
            auxiliary_junctions=tuple(sorted(groups["right"])),
            exclusion_junctions=tuple(sorted(groups["skip"])),
            body_region=(event_start, event_end),
            proposal_source="denovo",
            metadata={
                "n_candidate_junctions": (
                    len(groups["left"])
                    + len(groups["right"])
                    + len(groups["skip"])
                ),
            },
        ))
    return events


def _propose_ri_events(
    junction_counts: dict[tuple[int, int], int],
    gene: str,
    chrom: str,
) -> list[SpliceEvent]:
    """Propose RI events from observed splice junctions."""
    events: list[SpliceEvent] = []
    for jstart, jend in sorted(junction_counts):
        if jend - jstart > 50000 or jend - jstart < 50:
            continue
        events.append(SpliceEvent(
            event_id=f"RI:{jstart}-{jend}",
            event_type="RI",
            gene=gene,
            chrom=chrom,
            event_start=jstart,
            event_end=jend,
            exclusion_junctions=((jstart, jend),),
            body_region=(jstart, jend),
            proposal_source="denovo",
            metadata={"n_candidate_junctions": 1},
        ))
    return events


def _count_exon_body_reads(
    reads: list[ReadSummary],
    exon_start: int,
    exon_end: int,
) -> int:
    """Count non-spliced reads with substantial exon-body overlap."""
    exon_len = exon_end - exon_start
    if exon_len <= 0:
        return 0
    min_overlap = min(max(25, exon_len // 2), exon_len)
    count = 0
    for read in reads:
        if read.has_splice:
            continue
        overlap = min(read.end, exon_end) - max(read.start, exon_start)
        if overlap >= min_overlap:
            count += 1
    return count


def _count_intronic_body_reads(
    reads: list[ReadSummary],
    intron_start: int,
    intron_end: int,
) -> int:
    """Count unspliced reads that support intron retention."""
    ri_count = 0
    for read in reads:
        if read.has_splice:
            continue
        if read.start <= intron_start + 10 and read.end >= intron_end - 10:
            ri_count += 1
        elif (
            read.start >= intron_start
            and read.end <= intron_end
            and read.end - read.start > 50
        ):
            ri_count += 1
    return ri_count


def _measure_event_evidence(
    event: SpliceEvent,
    junction_counts: dict[tuple[int, int], int],
    reads: list[ReadSummary],
) -> EventEvidence:
    """Convert a proposed event into local inclusion/exclusion counts."""
    if event.event_type in {"A3SS", "A5SS"}:
        inclusion_count = sum(junction_counts.get(j, 0) for j in event.inclusion_junctions)
        exclusion_count = sum(junction_counts.get(j, 0) for j in event.exclusion_junctions)
        unique_junctions = set(event.inclusion_junctions + event.exclusion_junctions)
        return EventEvidence(
            inclusion_count=inclusion_count,
            exclusion_count=exclusion_count,
            n_candidate_junctions=max(
                event.metadata.get("n_candidate_junctions", 0),
                len(unique_junctions),
            ),
            n_supported_junctions=sum(1 for j in unique_junctions if junction_counts.get(j, 0) > 0),
            evidence_breakdown={
                "junction_inclusion": inclusion_count,
                "junction_exclusion": exclusion_count,
            },
        )

    if event.event_type == "SE":
        left_best = max((junction_counts.get(j, 0) for j in event.inclusion_junctions), default=0)
        right_best = max((junction_counts.get(j, 0) for j in event.auxiliary_junctions), default=0)
        skip_best = max((junction_counts.get(j, 0) for j in event.exclusion_junctions), default=0)
        body_count = 0
        if event.body_region is not None:
            body_count = _count_exon_body_reads(
                reads,
                event.body_region[0],
                event.body_region[1],
            )

        paired_inclusion = min(left_best, right_best) if left_best and right_best else 0
        rescued_inclusion = (
            min(body_count, max(left_best, right_best))
            if body_count and (left_best or right_best)
            else 0
        )
        inclusion_count = max(paired_inclusion, rescued_inclusion)
        unique_junctions = set(
            event.inclusion_junctions
            + event.auxiliary_junctions
            + event.exclusion_junctions
        )
        return EventEvidence(
            inclusion_count=inclusion_count,
            exclusion_count=skip_best,
            body_count=body_count,
            n_candidate_junctions=max(
                event.metadata.get("n_candidate_junctions", 0),
                len(unique_junctions),
            ),
            n_supported_junctions=sum(1 for j in unique_junctions if junction_counts.get(j, 0) > 0),
            evidence_breakdown={
                "left_junction": left_best,
                "right_junction": right_best,
                "skip_junction": skip_best,
                "body_count": body_count,
                "paired_inclusion": paired_inclusion,
                "rescued_inclusion": rescued_inclusion,
            },
        )

    if event.event_type == "RI":
        if event.body_region is None:
            return EventEvidence(0, 0)
        ri_count = _count_intronic_body_reads(
            reads,
            event.body_region[0],
            event.body_region[1],
        )
        jcount = sum(junction_counts.get(j, 0) for j in event.exclusion_junctions)
        unique_junctions = set(event.exclusion_junctions)
        return EventEvidence(
            inclusion_count=ri_count,
            exclusion_count=jcount,
            body_count=ri_count,
            n_candidate_junctions=max(
                event.metadata.get("n_candidate_junctions", 0),
                len(unique_junctions),
            ),
            n_supported_junctions=sum(1 for j in unique_junctions if junction_counts.get(j, 0) > 0),
            evidence_breakdown={
                "retained_body": ri_count,
                "splice_junction": jcount,
            },
        )

    return EventEvidence(0, 0)


def _infer_event(
    event: SpliceEvent,
    evidence: EventEvidence,
    config: BRAIDConfig,
    rng: np.random.Generator,
) -> PSIResult | None:
    """Infer PSI for one event from measured evidence."""
    total = evidence.inclusion_count + evidence.exclusion_count
    if total < config.min_event_support:
        return None

    psi, ci_low, ci_high, cv = bootstrap_psi(
        evidence.inclusion_count,
        evidence.exclusion_count,
        n_replicates=config.n_replicates,
        confidence_level=config.confidence_level,
        seed=int(rng.integers(0, 2**31)),
        model=config.uncertainty_model,
        event_type=event.event_type,
        schedule_mode=config.schedule_mode,
        calibration_schedule=config.calibration_schedule,
    )
    ci_width, is_confident = _is_confident_interval(
        ci_low,
        ci_high,
        cv,
        psi=psi,
        event_type=event.event_type,
        inclusion_count=evidence.inclusion_count,
        exclusion_count=evidence.exclusion_count,
        schedule_mode=config.schedule_mode,
        calibration_schedule=config.calibration_schedule,
    )
    return PSIResult(
        event_id=event.event_id,
        event_type=event.event_type,
        gene=event.gene,
        chrom=event.chrom,
        psi=psi,
        ci_low=ci_low,
        ci_high=ci_high,
        cv=cv,
        inclusion_count=evidence.inclusion_count,
        exclusion_count=evidence.exclusion_count,
        event_start=event.event_start,
        event_end=event.event_end,
        ci_width=ci_width,
        is_confident=is_confident,
        n_candidate_junctions=evidence.n_candidate_junctions,
        n_supported_junctions=evidence.n_supported_junctions,
        proposal_source=event.proposal_source,
        evidence_breakdown=evidence.evidence_breakdown,
    )


def compute_psi_from_junctions(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
    gene: str | None = None,
    n_replicates: int = 500,
    confidence_level: float = 0.95,
    min_mapq: int = DEFAULT_MIN_MAPQ,
    seed: int | None = None,
    annotation_gtf: str | None = None,
    event_source: str = DEFAULT_EVENT_SOURCE,
    uncertainty_model: str = DEFAULT_UNCERTAINTY_MODEL,
    min_event_support: int = DEFAULT_MIN_EVENT_SUPPORT,
    schedule_mode: str = "native",
    calibration_schedule: dict[str, object] | None = None,
) -> list[PSIResult]:
    """Compute PSI + CI for local AS events using BRAID v2 inference."""
    config = BRAIDConfig(
        n_replicates=n_replicates,
        confidence_level=confidence_level,
        min_mapq=min_mapq,
        min_event_support=min_event_support,
        event_source=event_source,
        annotation_gtf=annotation_gtf,
        uncertainty_model=uncertainty_model,
        seed=seed,
        schedule_mode=schedule_mode,
        calibration_schedule=calibration_schedule,
    )

    junction_counts, reads = _extract_region_evidence(
        bam_path=bam_path,
        chrom=chrom,
        start=start,
        end=end,
        min_mapq=config.min_mapq,
    )
    if not junction_counts:
        return []

    gene_name = gene or ""
    rng = np.random.default_rng(config.seed)
    proposed_events: list[SpliceEvent] = []
    proposed_events.extend(_propose_grouped_junction_events(junction_counts, gene_name, chrom))
    proposed_events.extend(_propose_ri_events(junction_counts, gene_name, chrom))

    annotation_se_events: list[SpliceEvent] = []
    if (
        config.event_source in {"hybrid", "annotation"}
        and config.annotation_gtf
        and gene_name
    ):
        annotation_se_events = _propose_annotation_se_events(
            gene=gene_name,
            chrom=chrom,
            annotation_gtf=config.annotation_gtf,
        )

    if annotation_se_events:
        proposed_events.extend(annotation_se_events)
    elif config.event_source in {"hybrid", "denovo"}:
        proposed_events.extend(_propose_de_novo_se_events(junction_counts, gene_name, chrom))

    results: list[PSIResult] = []
    seen_keys: set[tuple[str, int | None, int | None, str]] = set()
    for event in proposed_events:
        key = (event.event_type, event.event_start, event.event_end, event.event_id)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        evidence = _measure_event_evidence(event, junction_counts, reads)
        inferred = _infer_event(event, evidence, config, rng)
        if inferred is not None:
            results.append(inferred)

    results.sort(
        key=lambda result: (
            result.event_type,
            result.event_start if result.event_start is not None else -1,
            result.event_end if result.event_end is not None else -1,
            result.event_id,
        )
    )
    return results


def format_psi_report(results: list[PSIResult]) -> str:
    """Format PSI results as a text report."""
    lines: list[str] = []
    lines.append(
        f"{'Event':<30} {'PSI':>6} {'CI_low':>7} {'CI_high':>7} "
        f"{'CV':>6} {'Inc':>5} {'Exc':>5} {'Conf':>5}"
    )
    lines.append("-" * 80)
    for result in results:
        conf = "Y" if result.is_confident else "N"
        lines.append(
            f"{result.event_id:<30} {result.psi:>5.1%} {result.ci_low:>6.1%} "
            f"{result.ci_high:>6.1%} {result.cv:>6.2f} "
            f"{result.inclusion_count:>5} {result.exclusion_count:>5} {conf:>5}"
        )
    return "\n".join(lines)
