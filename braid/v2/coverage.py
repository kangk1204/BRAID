"""BRAID v2 coverage and read-quality feature extractor.

Extracts 6 BAM-derived features per rMATS splicing event:
  B1  dup_fraction_inc             Fraction of inclusion reads marked as PCR duplicates
  B2  dup_fraction_exc             Same for exclusion reads
  B3  unique_start_fraction_inc    Distinct alignment starts / total inc reads
  B4  unique_start_fraction_exc    Same for exclusion reads
  C1  exon_body_coverage_uniformity  CV of per-base coverage across the skipped exon
  C2  exon_body_mean_coverage      Mean per-base coverage of the skipped exon body
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pysam

if TYPE_CHECKING:
    from pysam import AlignedSegment

from braid.target.rmats_bootstrap import RmatsEvent
from braid.v2.junction import _resolve_chrom
from braid.v2.junction import _collect_junction_reads

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dup_fraction(reads: list[tuple[AlignedSegment, int, int]]) -> float:
    """Fraction of reads with the PCR duplicate flag (0x400)."""
    if not reads:
        return float("nan")
    n_dup = sum(1 for r, _a, _j in reads if r.flag & 0x400)
    return n_dup / len(reads)


def _unique_start_fraction(reads: list[tuple[AlignedSegment, int, int]]) -> float:
    """Distinct alignment start positions / total reads."""
    if not reads:
        return float("nan")
    starts = {r.reference_start for r, _a, _j in reads}
    return len(starts) / len(reads)


def _exon_body_coverage(
    bam: pysam.AlignmentFile,
    chrom: str,
    start: int,
    end: int,
) -> tuple[float, float]:
    """Return (cv, mean) of per-base coverage over [start, end).

    CV is coefficient of variation (std / mean). Returns (NaN, NaN) for
    zero-length exons or regions with no coverage.
    """
    length = end - start
    if length <= 0:
        return float("nan"), float("nan")

    cov = np.zeros(length, dtype=np.float64)
    try:
        for pileup_col in bam.pileup(
            chrom,
            max(0, start),
            end,
            truncate=True,
            stepper="nofilter",
        ):
            pos = pileup_col.reference_pos
            idx = pos - start
            if 0 <= idx < length:
                cov[idx] = pileup_col.nsegments
    except (ValueError, KeyError):
        return float("nan"), float("nan")

    mean_cov = float(np.mean(cov))
    if mean_cov == 0.0:
        return float("nan"), 0.0

    std_cov = float(np.std(cov))
    cv = std_cov / mean_cov
    return cv, mean_cov


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_coverage_features(
    bam_path: str,
    event: RmatsEvent,
    region_flank: int = 100,
) -> dict[str, float]:
    """Extract 6 coverage/read-quality features for one SE event.

    Parameters
    ----------
    bam_path:
        Path to a coordinate-sorted, indexed BAM file.
    event:
        An ``RmatsEvent`` with populated coordinate fields.
    region_flank:
        Extra bases around event junctions when fetching reads.

    Returns
    -------
    dict with keys B1..B4, C1..C2 (descriptive names). Missing features are ``NaN``.
    """
    nan = float("nan")
    empty: dict[str, float] = {
        "dup_fraction_inc": nan,
        "dup_fraction_exc": nan,
        "unique_start_fraction_inc": nan,
        "unique_start_fraction_exc": nan,
        "exon_body_coverage_uniformity": nan,
        "exon_body_mean_coverage": nan,
    }

    upstream_ee = event.upstream_ee
    downstream_es = event.downstream_es
    exon_start = event.exon_start
    exon_end = event.exon_end

    if upstream_ee is None or downstream_es is None:
        return empty

    # Inclusion junctions: upstream_ee -> exon_start, exon_end -> downstream_es
    inc_junctions: list[tuple[int, int]] = [
        (upstream_ee, exon_start),
        (exon_end, downstream_es),
    ]
    # Exclusion junction: upstream_ee -> downstream_es
    exc_junctions: list[tuple[int, int]] = [
        (upstream_ee, downstream_es),
    ]

    region_start = min(upstream_ee, exon_start) - region_flank
    region_end = max(exon_end, downstream_es) + region_flank

    bam = pysam.AlignmentFile(bam_path, "rb")
    try:
        resolved = _resolve_chrom(bam, event.chrom)
        if resolved is None:
            bam.close()
            return empty
        chrom = resolved
        inc_data = _collect_junction_reads(
            bam, chrom, inc_junctions, region_start, region_end,
        )
        exc_data = _collect_junction_reads(
            bam, chrom, exc_junctions, region_start, region_end,
        )

        # B1-B2: Duplicate fraction
        features: dict[str, float] = {}
        features["dup_fraction_inc"] = _dup_fraction(inc_data)
        features["dup_fraction_exc"] = _dup_fraction(exc_data)

        # B3-B4: Unique start fraction
        features["unique_start_fraction_inc"] = _unique_start_fraction(inc_data)
        features["unique_start_fraction_exc"] = _unique_start_fraction(exc_data)

        # C1-C2: Exon body coverage
        cv, mean_cov = _exon_body_coverage(bam, chrom, exon_start, exon_end)
        features["exon_body_coverage_uniformity"] = cv
        features["exon_body_mean_coverage"] = mean_cov
    finally:
        bam.close()

    return features
