"""BRAID v2 junction quality feature extractor.

Extracts 10 BAM-derived quality features per rMATS splicing event:
  A1  median_mapq_inc          Median MAPQ of inclusion junction reads
  A2  median_mapq_exc          Median MAPQ of exclusion junction reads
  A3  frac_mapq0_inc           Fraction of inclusion reads with MAPQ=0
  A4  frac_mapq0_exc           Fraction of exclusion reads with MAPQ=0
  A5  min_anchor_inc           Minimum anchor length across inclusion reads
  A6  min_anchor_exc           Minimum anchor length across exclusion reads
  A7  median_anchor_inc        Median anchor length for inclusion reads
  A8  frac_short_anchor_inc    Fraction of inc reads with anchor < 8bp
  A9  mismatch_rate_near_junc  Mismatches within 5bp of splice site / total bases
  A10 strand_consistency        Fraction of junction reads agreeing on XS tag
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pysam

if TYPE_CHECKING:
    from pysam import AlignedSegment

from braid.target.rmats_bootstrap import RmatsEvent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SHORT_ANCHOR_THRESHOLD = 8
_MISMATCH_WINDOW = 5  # bases on each side of the splice site


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _junction_spans(read: AlignedSegment) -> list[tuple[int, int, int, int]]:
    """Return (intron_start, intron_end, left_anchor, right_anchor) per N op.

    *intron_start* and *intron_end* are 0-based half-open genomic coordinates
    of the inferred intron.  Anchor lengths count aligned (M/=/X) bases
    immediately flanking the N operation.
    """
    if read.cigartuples is None:
        return []

    spans: list[tuple[int, int, int, int]] = []
    ref_pos = read.reference_start
    ops = read.cigartuples

    for idx, (op, length) in enumerate(ops):
        if op == 3:  # N = reference skip (intron)
            intron_start = ref_pos
            intron_end = ref_pos + length

            # Left anchor: walk backwards through consuming-ref ops
            left_anchor = 0
            for j in range(idx - 1, -1, -1):
                prev_op, prev_len = ops[j]
                if prev_op in (0, 7, 8):  # M, =, X
                    left_anchor += prev_len
                else:
                    break

            # Right anchor: walk forwards through consuming-ref ops
            right_anchor = 0
            for j in range(idx + 1, len(ops)):
                next_op, next_len = ops[j]
                if next_op in (0, 7, 8):  # M, =, X
                    right_anchor += next_len
                else:
                    break

            spans.append((intron_start, intron_end, left_anchor, right_anchor))

        # Advance reference position for ref-consuming ops
        if op in (0, 2, 3, 7, 8):  # M, D, N, =, X
            ref_pos += length

    return spans


def _matches_junction(
    read: AlignedSegment,
    donor: int,
    acceptor: int,
    tolerance: int = 0,
) -> tuple[bool, int]:
    """Check if *read* spans the junction donor->acceptor.

    Returns (matched, min_anchor) where min_anchor is the smaller of the
    left and right anchors for the matching N operation.  If unmatched,
    min_anchor is 0.
    """
    for intron_start, intron_end, left_anc, right_anc in _junction_spans(read):
        if (
            abs(intron_start - donor) <= tolerance
            and abs(intron_end - acceptor) <= tolerance
        ):
            return True, min(left_anc, right_anc)
    return False, 0


def _mismatches_near_splice(
    read: AlignedSegment,
    splice_sites: list[int],
    window: int = _MISMATCH_WINDOW,
) -> tuple[int, int]:
    """Count mismatches within *window* bp of any splice site.

    Returns (n_mismatches, n_total_bases) within the window regions.
    Uses the MD tag when available; falls back to NM tag heuristic.
    """
    if read.reference_start is None or read.reference_end is None:
        return 0, 0

    # Build set of reference positions near splice sites
    near_positions: set[int] = set()
    for site in splice_sites:
        for offset in range(-window, window):
            near_positions.add(site + offset)

    # Restrict to read's reference span
    rstart = read.reference_start
    rend = read.reference_end
    near_positions = {p for p in near_positions if rstart <= p < rend}

    if not near_positions:
        return 0, 0

    total_bases = len(near_positions)

    # Try aligned pairs for precise mismatch detection
    try:
        aligned_pairs = read.get_aligned_pairs(with_seq=True)
    except (ValueError, AttributeError):
        aligned_pairs = None

    if aligned_pairs is not None:
        mismatches = 0
        for qpos, rpos, ref_base in aligned_pairs:
            if rpos is None or rpos not in near_positions:
                continue
            if qpos is None:
                # Deletion at this position
                mismatches += 1
                continue
            if ref_base is not None and ref_base.islower():
                # pysam convention: lowercase = mismatch
                mismatches += 1
        return mismatches, total_bases

    # Fallback: use NM tag, distribute proportionally
    nm = read.get_tag("NM") if read.has_tag("NM") else 0
    if nm == 0:
        return 0, total_bases
    read_len = read.query_alignment_length or 1
    estimated = int(round(nm * total_bases / read_len))
    return estimated, total_bases


def _collect_junction_reads(
    bam: pysam.AlignmentFile,
    chrom: str,
    junctions: list[tuple[int, int]],
    region_start: int,
    region_end: int,
) -> list[tuple[AlignedSegment, int, int]]:
    """Fetch reads that span any of the listed junctions.

    Returns list of (read, min_anchor, junction_index).
    """
    results: list[tuple[AlignedSegment, int, int]] = []
    try:
        reads = bam.fetch(chrom, max(0, region_start), region_end)
    except (ValueError, KeyError):
        return results

    for read in reads:
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        if read.cigartuples is None:
            continue
        for jidx, (donor, acceptor) in enumerate(junctions):
            matched, min_anc = _matches_junction(read, donor, acceptor)
            if matched:
                results.append((read, min_anc, jidx))
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_junction_features(
    bam_path: str,
    event: RmatsEvent,
    region_flank: int = 100,
) -> dict[str, float]:
    """Extract 10 junction-quality features for one SE event.

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
    dict with keys A1..A10 (descriptive names). Missing features are ``NaN``.
    """
    nan = float("nan")
    empty: dict[str, float] = {
        "median_mapq_inc": nan,
        "median_mapq_exc": nan,
        "frac_mapq0_inc": nan,
        "frac_mapq0_exc": nan,
        "min_anchor_inc": nan,
        "min_anchor_exc": nan,
        "median_anchor_inc": nan,
        "frac_short_anchor_inc": nan,
        "mismatch_rate_near_junction": nan,
        "strand_consistency": nan,
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

    all_junctions = inc_junctions + exc_junctions
    splice_sites = sorted(
        {upstream_ee, exon_start, exon_end, downstream_es}
    )

    region_start = min(upstream_ee, exon_start) - region_flank
    region_end = max(exon_end, downstream_es) + region_flank

    bam = pysam.AlignmentFile(bam_path, "rb")
    try:
        # Collect inclusion reads
        inc_data = _collect_junction_reads(
            bam, event.chrom, inc_junctions, region_start, region_end
        )
        # Collect exclusion reads
        exc_data = _collect_junction_reads(
            bam, event.chrom, exc_junctions, region_start, region_end
        )
    finally:
        bam.close()

    # ------------------------------------------------------------------
    # A1, A3: MAPQ for inclusion reads
    # ------------------------------------------------------------------
    inc_mapqs = np.array(
        [r.mapping_quality for r, _anc, _ji in inc_data], dtype=np.float64
    )
    exc_mapqs = np.array(
        [r.mapping_quality for r, _anc, _ji in exc_data], dtype=np.float64
    )

    features: dict[str, float] = {}

    features["median_mapq_inc"] = (
        float(np.median(inc_mapqs)) if len(inc_mapqs) > 0 else nan
    )
    features["median_mapq_exc"] = (
        float(np.median(exc_mapqs)) if len(exc_mapqs) > 0 else nan
    )
    features["frac_mapq0_inc"] = (
        float(np.mean(inc_mapqs == 0)) if len(inc_mapqs) > 0 else nan
    )
    features["frac_mapq0_exc"] = (
        float(np.mean(exc_mapqs == 0)) if len(exc_mapqs) > 0 else nan
    )

    # ------------------------------------------------------------------
    # A5-A8: Anchor lengths
    # ------------------------------------------------------------------
    inc_anchors = np.array(
        [anc for _r, anc, _ji in inc_data], dtype=np.float64
    )
    exc_anchors = np.array(
        [anc for _r, anc, _ji in exc_data], dtype=np.float64
    )

    features["min_anchor_inc"] = (
        float(np.min(inc_anchors)) if len(inc_anchors) > 0 else nan
    )
    features["min_anchor_exc"] = (
        float(np.min(exc_anchors)) if len(exc_anchors) > 0 else nan
    )
    features["median_anchor_inc"] = (
        float(np.median(inc_anchors)) if len(inc_anchors) > 0 else nan
    )
    features["frac_short_anchor_inc"] = (
        float(np.mean(inc_anchors < _SHORT_ANCHOR_THRESHOLD))
        if len(inc_anchors) > 0
        else nan
    )

    # ------------------------------------------------------------------
    # A9: Mismatch rate near splice sites
    # ------------------------------------------------------------------
    all_reads = [r for r, _a, _j in inc_data] + [r for r, _a, _j in exc_data]
    total_mm = 0
    total_bases = 0
    for read in all_reads:
        mm, bases = _mismatches_near_splice(read, splice_sites)
        total_mm += mm
        total_bases += bases

    features["mismatch_rate_near_junction"] = (
        total_mm / total_bases if total_bases > 0 else nan
    )

    # ------------------------------------------------------------------
    # A10: Strand consistency (XS tag)
    # ------------------------------------------------------------------
    xs_tags: list[str] = []
    for read in all_reads:
        if read.has_tag("XS"):
            xs_tags.append(read.get_tag("XS"))

    if len(xs_tags) > 0:
        from collections import Counter

        counts = Counter(xs_tags)
        most_common_count = counts.most_common(1)[0][1]
        features["strand_consistency"] = most_common_count / len(xs_tags)
    else:
        features["strand_consistency"] = nan

    return features
