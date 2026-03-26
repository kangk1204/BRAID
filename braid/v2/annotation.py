"""BRAID v2 annotation and genomic context feature extractor.

Extracts 5 annotation features per rMATS splicing event:
  C3  splice_motif_inc              1.0 if canonical GT-AG or GC-AG for inclusion junctions
  C4  splice_motif_exc              Same for exclusion junction
  C5  overlapping_gene_flag         1.0 if an annotated gene on opposite strand overlaps
  C6  n_overlapping_events          Number of other events overlapping this region
  C7  flanking_exon_coverage_ratio  coverage(upstream exon) / coverage(downstream exon)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import numpy as np
import pysam

if TYPE_CHECKING:
    pass

from braid.target.rmats_bootstrap import RmatsEvent
from braid.v2.junction import _resolve_chrom

logger = logging.getLogger(__name__)

# Canonical splice motifs: donor-acceptor dinucleotide pairs
_CANONICAL_MOTIFS = frozenset({("GT", "AG"), ("GC", "AG")})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_splice_motif(
    fasta: pysam.FastaFile,
    chrom: str,
    strand: str,
    donor: int,
    acceptor: int,
) -> float:
    """Return 1.0 if the junction has a canonical splice motif, else 0.0.

    For '+' strand: donor dinucleotide at [donor, donor+2),
                    acceptor dinucleotide at [acceptor-2, acceptor).
    For '-' strand: reverse complement of acceptor/donor positions.
    """
    try:
        chrom_len = fasta.get_reference_length(chrom)
    except (KeyError, ValueError):
        return float("nan")

    # Ensure coordinates are within bounds
    if donor < 0 or acceptor < 2 or donor + 2 > chrom_len or acceptor > chrom_len:
        return float("nan")

    try:
        donor_seq = fasta.fetch(chrom, donor, donor + 2).upper()
        acceptor_seq = fasta.fetch(chrom, acceptor - 2, acceptor).upper()
    except (ValueError, KeyError):
        return float("nan")

    if strand == "+":
        motif = (donor_seq, acceptor_seq)
    else:
        # Reverse complement for minus strand
        comp = str.maketrans("ACGT", "TGCA")
        donor_rc = donor_seq.translate(comp)[::-1]
        acceptor_rc = acceptor_seq.translate(comp)[::-1]
        # On minus strand, the acceptor site is the "donor" in genomic coords
        motif = (acceptor_rc, donor_rc)

    return 1.0 if motif in _CANONICAL_MOTIFS else 0.0


def _parse_gtf_genes(
    gtf_path: str,
) -> list[tuple[str, int, int, str]]:
    """Parse gene records from a GTF file.

    Returns list of (chrom, start, end, strand) for gene features.
    """
    genes: list[tuple[str, int, int, str]] = []
    try:
        with open(gtf_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9:
                    continue
                if parts[2] != "gene":
                    continue
                chrom = parts[0]
                start = int(parts[3]) - 1  # GTF is 1-based
                end = int(parts[4])
                strand = parts[6]
                genes.append((chrom, start, end, strand))
    except (OSError, IOError) as exc:
        logger.warning("Failed to parse GTF %s: %s", gtf_path, exc)
    return genes


def _has_overlapping_opposite_strand_gene(
    event: RmatsEvent,
    genes: list[tuple[str, int, int, str]],
) -> float:
    """Return 1.0 if any gene on opposite strand overlaps the event region."""
    opposite = "-" if event.strand == "+" else "+"
    ev_start = event.exon_start
    ev_end = event.exon_end

    for g_chrom, g_start, g_end, g_strand in genes:
        if g_chrom != event.chrom:
            continue
        if g_strand != opposite:
            continue
        # Check overlap
        if g_start < ev_end and g_end > ev_start:
            return 1.0
    return 0.0


def _count_overlapping_events(
    event: RmatsEvent,
    all_events: list[RmatsEvent],
) -> int:
    """Count events in all_events that overlap this event's region."""
    count = 0
    ev_start = event.exon_start
    ev_end = event.exon_end

    for other in all_events:
        if other.event_id == event.event_id:
            continue
        if other.chrom != event.chrom:
            continue
        # Use the full event span for overlap
        other_start = other.exon_start
        other_end = other.exon_end
        if other_start < ev_end and other_end > ev_start:
            count += 1
    return count


def _flanking_exon_coverage_ratio(
    bam: pysam.AlignmentFile,
    chrom: str,
    upstream_es: int,
    upstream_ee: int,
    downstream_es: int,
    downstream_ee: int,
) -> float:
    """Compute coverage(upstream) / coverage(downstream) for flanking exons.

    Returns NaN if either exon has zero coverage or invalid coordinates.
    """
    up_len = upstream_ee - upstream_es
    dn_len = downstream_ee - downstream_es
    if up_len <= 0 or dn_len <= 0:
        return float("nan")

    def _mean_coverage(start: int, end: int) -> float:
        total = 0
        length = end - start
        try:
            for col in bam.pileup(
                chrom, max(0, start), end, truncate=True, stepper="nofilter",
            ):
                pos = col.reference_pos
                if start <= pos < end:
                    total += col.nsegments
        except (ValueError, KeyError):
            return 0.0
        return total / length if length > 0 else 0.0

    up_cov = _mean_coverage(upstream_es, upstream_ee)
    dn_cov = _mean_coverage(downstream_es, downstream_ee)

    if dn_cov == 0.0:
        return float("nan")
    return up_cov / dn_cov


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_annotation_features(
    event: RmatsEvent,
    reference_path: str | None = None,
    gtf_path: str | None = None,
    all_events: list[RmatsEvent] | None = None,
    bam_path: str | None = None,
) -> dict[str, float]:
    """Extract 5 annotation/context features for one SE event.

    Parameters
    ----------
    event:
        An ``RmatsEvent`` with populated coordinate fields.
    reference_path:
        Path to indexed FASTA (for splice motif check). ``None`` => NaN for C3/C4.
    gtf_path:
        Path to gene annotation GTF (for C5). ``None`` => NaN.
    all_events:
        List of all events for overlap counting (C6). ``None`` => NaN.
    bam_path:
        Path to BAM file for flanking exon coverage (C7). ``None`` => NaN.

    Returns
    -------
    dict with keys C3..C7 (descriptive names). Missing features are ``NaN``.
    """
    nan = float("nan")
    features: dict[str, float] = {
        "splice_motif_inc": nan,
        "splice_motif_exc": nan,
        "overlapping_gene_flag": nan,
        "n_overlapping_events": nan,
        "flanking_exon_coverage_ratio": nan,
    }

    upstream_ee = event.upstream_ee
    downstream_es = event.downstream_es
    exon_start = event.exon_start
    exon_end = event.exon_end

    # ------------------------------------------------------------------
    # C3-C4: Splice motif validation
    # ------------------------------------------------------------------
    if reference_path is not None and upstream_ee is not None and downstream_es is not None:
        try:
            fasta = pysam.FastaFile(reference_path)
            try:
                # Inclusion: upstream_ee -> exon_start and exon_end -> downstream_es
                motif_inc_1 = _check_splice_motif(
                    fasta, event.chrom, event.strand, upstream_ee, exon_start,
                )
                motif_inc_2 = _check_splice_motif(
                    fasta, event.chrom, event.strand, exon_end, downstream_es,
                )
                # Both inclusion junctions must be canonical
                if np.isnan(motif_inc_1) or np.isnan(motif_inc_2):
                    features["splice_motif_inc"] = nan
                else:
                    features["splice_motif_inc"] = min(motif_inc_1, motif_inc_2)

                # Exclusion: upstream_ee -> downstream_es
                features["splice_motif_exc"] = _check_splice_motif(
                    fasta, event.chrom, event.strand, upstream_ee, downstream_es,
                )
            finally:
                fasta.close()
        except (OSError, IOError, ValueError) as exc:
            logger.warning("Cannot open reference %s: %s", reference_path, exc)

    # ------------------------------------------------------------------
    # C5: Overlapping gene on opposite strand
    # ------------------------------------------------------------------
    if gtf_path is not None:
        genes = _parse_gtf_genes(gtf_path)
        features["overlapping_gene_flag"] = _has_overlapping_opposite_strand_gene(
            event, genes,
        )

    # ------------------------------------------------------------------
    # C6: Number of overlapping events
    # ------------------------------------------------------------------
    if all_events is not None:
        features["n_overlapping_events"] = float(
            _count_overlapping_events(event, all_events)
        )

    # ------------------------------------------------------------------
    # C7: Flanking exon coverage ratio
    # ------------------------------------------------------------------
    if (
        bam_path is not None
        and event.upstream_es is not None
        and upstream_ee is not None
        and downstream_es is not None
        and event.downstream_ee is not None
    ):
        bam = pysam.AlignmentFile(bam_path, "rb")
        try:
            features["flanking_exon_coverage_ratio"] = _flanking_exon_coverage_ratio(
                bam,
                event.chrom,
                event.upstream_es,
                upstream_ee,
                downstream_es,
                event.downstream_ee,
            )
        finally:
            bam.close()

    return features
