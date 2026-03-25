"""BRAID v2 combined feature extractor.

Merges junction (A1-A10), coverage (B1-B4, C1-C2), annotation (C3-C7),
and differential (B5-B6, D1-D6) features into a single 28-feature dict.
"""

from __future__ import annotations

from braid.target.rmats_bootstrap import RmatsEvent
from braid.v2.annotation import extract_annotation_features
from braid.v2.coverage import extract_coverage_features
from braid.v2.differential import extract_differential_features
from braid.v2.junction import extract_junction_features


def extract_all_features(
    bam_path: str,
    event: RmatsEvent,
    reference_path: str | None = None,
    gtf_path: str | None = None,
    all_events: list[RmatsEvent] | None = None,
) -> dict[str, float]:
    """Extract all 28 BRAID v2 features for one rMATS event.

    Parameters
    ----------
    bam_path:
        Path to a coordinate-sorted, indexed BAM file.
    event:
        An ``RmatsEvent`` with populated coordinate and replicate fields.
    reference_path:
        Path to indexed reference FASTA (for splice motif features).
        ``None`` produces NaN for C3/C4.
    gtf_path:
        Path to gene annotation GTF (for C5 overlapping gene check).
        ``None`` produces NaN for C5.
    all_events:
        Full list of events for overlap counting (C6).
        ``None`` produces NaN for C6.

    Returns
    -------
    dict with all 28 feature names as keys. Missing features are ``NaN``.
    """
    # A1-A10: Junction quality features (10)
    junction_feats = extract_junction_features(bam_path, event)

    # B1-B4, C1-C2: Coverage / read-quality features (6)
    coverage_feats = extract_coverage_features(bam_path, event)

    # C3-C7: Annotation / context features (5)
    annotation_feats = extract_annotation_features(
        event,
        reference_path=reference_path,
        gtf_path=gtf_path,
        all_events=all_events,
        bam_path=bam_path,
    )

    # B5-B6, D1-D6: Differential / replicate features (8)
    differential_feats = extract_differential_features(event)

    # Merge all (total: 10 + 6 + 5 + 8 = 29, but C7 uses bam_path via annotation)
    merged: dict[str, float] = {}
    merged.update(junction_feats)
    merged.update(coverage_feats)
    merged.update(annotation_feats)
    merged.update(differential_feats)

    return merged
