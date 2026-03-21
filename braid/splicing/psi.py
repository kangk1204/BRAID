"""PSI (Percent Spliced In) quantification for alternative splicing events.

Computes junction-based PSI values using the formula:
    PSI = (I / lI) / (I / lI + S / lS)

where I = inclusion read count, S = exclusion read count,
lI = number of inclusion junctions, lS = number of exclusion junctions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from braid.io.bam_reader import JunctionEvidence, ReadData
from braid.splicing.events import ASEvent, EventType
from braid.utils.cigar import CIGAR_N

logger = logging.getLogger(__name__)


@dataclass
class PSIResult:
    """Result of PSI calculation for a single event.

    Attributes:
        event_id: Matching ASEvent identifier.
        psi: Percent Spliced In value in [0, 1], or NaN if insufficient reads.
        inclusion_count: Total inclusion junction reads.
        exclusion_count: Total exclusion junction reads.
        inclusion_length: Number of inclusion junctions.
        exclusion_length: Number of exclusion junctions.
        total_reads: Sum of inclusion and exclusion reads.
        ci_low: Lower bound of 95% credible interval (set by statistics module).
        ci_high: Upper bound of 95% credible interval (set by statistics module).
    """

    event_id: str
    psi: float
    inclusion_count: int
    exclusion_count: int
    inclusion_length: int
    exclusion_length: int
    total_reads: int
    ci_low: float = 0.0
    ci_high: float = 1.0


def _lookup_junction_count(
    junction_evidence: JunctionEvidence,
    donor: int,
    acceptor: int,
    strand: str | None = None,
) -> int:
    """Look up the read count for a specific junction in the evidence.

    Args:
        junction_evidence: Junction evidence for the chromosome.
        donor: Junction donor position (0-based).
        acceptor: Junction acceptor position (0-based exclusive).
        strand: Optional event strand ('+' or '-') used to filter out
            opposite-strand junctions. Ambiguous strand junctions are retained.

    Returns:
        Read count supporting the junction, or 0 if not found.
    """
    if len(junction_evidence.starts) == 0:
        return 0

    mask = (junction_evidence.starts == donor) & (junction_evidence.ends == acceptor)
    if strand in {"+", "-"} and len(junction_evidence.strands) == len(junction_evidence.starts):
        strand_code = 0 if strand == "+" else 1
        # Allow explicitly matching strand and ambiguous strand assignments.
        strand_mask = (junction_evidence.strands == strand_code) | (
            junction_evidence.strands == -1
        )
        mask = mask & strand_mask

    indices = np.where(mask)[0]
    if len(indices) > 0:
        return int(np.sum(junction_evidence.counts[indices]))
    return 0


def _count_intronic_body_reads(
    read_data: ReadData,
    intron_start: int,
    intron_end: int,
    strand: str | None = None,
) -> int:
    """Count unspliced reads that support intron retention."""
    if read_data.n_reads == 0 or intron_end <= intron_start:
        return 0

    strand_code = None
    if strand in {"+", "-"}:
        strand_code = 0 if strand == "+" else 1

    count = 0
    for read_idx in range(read_data.n_reads):
        if strand_code is not None and int(read_data.strands[read_idx]) != strand_code:
            continue

        read_start = int(read_data.positions[read_idx])
        read_end = int(read_data.end_positions[read_idx])
        if read_end <= intron_start or read_start >= intron_end:
            continue

        cigar_start = int(read_data.cigar_offsets[read_idx])
        cigar_end = int(read_data.cigar_offsets[read_idx + 1])
        has_splice = False
        for cigar_idx in range(cigar_start, cigar_end):
            if int(read_data.cigar_ops[cigar_idx]) == CIGAR_N:
                has_splice = True
                break
        if has_splice:
            continue

        if read_start <= intron_start + 10 and read_end >= intron_end - 10:
            count += 1
        elif (
            read_start >= intron_start
            and read_end <= intron_end
            and read_end - read_start > 50
        ):
            count += 1

    return count


def calculate_psi(
    event: ASEvent,
    junction_evidence: JunctionEvidence,
    read_data: ReadData | None = None,
) -> PSIResult:
    """Calculate PSI for a single alternative splicing event.

    Uses junction-based PSI formula:
        PSI = (I / lI) / (I / lI + S / lS)

    For retained intron events with no inclusion junctions, PSI requires
    intronic body-read evidence because the retained isoform has no splice
    junction of its own.

    Args:
        event: The AS event to quantify.
        junction_evidence: Junction evidence for the event's chromosome.
        read_data: Optional per-read evidence for the chromosome. When
            provided, retained introns use intronic body reads as inclusion
            evidence instead of falling back to junction-only heuristics.

    Returns:
        PSIResult with computed PSI and read counts.
    """
    # Count inclusion junction reads
    inclusion_count = 0
    for junc in event.inclusion_junctions:
        inclusion_count += _lookup_junction_count(
            junction_evidence, junc[0], junc[1], strand=event.strand
        )

    # Count exclusion junction reads
    exclusion_count = 0
    for junc in event.exclusion_junctions:
        exclusion_count += _lookup_junction_count(
            junction_evidence, junc[0], junc[1], strand=event.strand
        )

    inclusion_length = max(len(event.inclusion_junctions), 1)
    exclusion_length = max(len(event.exclusion_junctions), 1)
    total_reads = inclusion_count + exclusion_count

    # RI requires body-read evidence for the retained form because the
    # inclusion isoform has no splice junction. When read-level evidence is
    # available, use unspliced intronic coverage as inclusion support.
    if event.event_type == EventType.RI and len(event.inclusion_junctions) == 0:
        intron_start = int(
            event.coordinates.get(
                "intron_start",
                event.exclusion_junctions[0][0] if event.exclusion_junctions else 0,
            )
        )
        intron_end = int(
            event.coordinates.get(
                "intron_end",
                event.exclusion_junctions[0][1] if event.exclusion_junctions else 0,
            )
        )

        if read_data is None:
            logger.debug(
                "RI event %s lacks read-level body evidence; PSI is unavailable "
                "from junction counts alone",
                event.event_id,
            )
            return PSIResult(
                event_id=event.event_id,
                psi=float("nan"),
                inclusion_count=inclusion_count,
                exclusion_count=exclusion_count,
                inclusion_length=inclusion_length,
                exclusion_length=exclusion_length,
                total_reads=total_reads,
            )

        inclusion_count = _count_intronic_body_reads(
            read_data,
            intron_start,
            intron_end,
            strand=event.strand,
        )
        total_reads = inclusion_count + exclusion_count
        if total_reads == 0:
            psi = float("nan")
        else:
            psi = inclusion_count / total_reads
        return PSIResult(
            event_id=event.event_id,
            psi=psi,
            inclusion_count=inclusion_count,
            exclusion_count=exclusion_count,
            inclusion_length=inclusion_length,
            exclusion_length=exclusion_length,
            total_reads=total_reads,
        )

    if total_reads == 0:
        return PSIResult(
            event_id=event.event_id,
            psi=float("nan"),
            inclusion_count=inclusion_count,
            exclusion_count=exclusion_count,
            inclusion_length=inclusion_length,
            exclusion_length=exclusion_length,
            total_reads=total_reads,
        )

    # Standard PSI formula
    inc_normalized = inclusion_count / inclusion_length
    exc_normalized = exclusion_count / exclusion_length
    denominator = inc_normalized + exc_normalized

    if denominator == 0:
        psi = float("nan")
    else:
        psi = inc_normalized / denominator

    return PSIResult(
        event_id=event.event_id,
        psi=psi,
        inclusion_count=inclusion_count,
        exclusion_count=exclusion_count,
        inclusion_length=inclusion_length,
        exclusion_length=exclusion_length,
        total_reads=total_reads,
    )


def calculate_all_psi(
    events: list[ASEvent],
    junction_evidence_by_chrom: dict[str, JunctionEvidence],
    read_data_by_chrom: dict[str, ReadData] | None = None,
) -> list[PSIResult]:
    """Calculate PSI for all events using per-chromosome junction evidence.

    Args:
        events: List of detected AS events.
        junction_evidence_by_chrom: Mapping from chromosome name to
            JunctionEvidence.
        read_data_by_chrom: Optional mapping from chromosome name to
            per-read evidence. Retained introns use this to derive
            intronic body support.

    Returns:
        List of PSIResult, one per event, in the same order.
    """
    results: list[PSIResult] = []
    for event in events:
        je = junction_evidence_by_chrom.get(event.chrom)
        if je is None:
            # No junction evidence for this chromosome
            results.append(
                PSIResult(
                    event_id=event.event_id,
                    psi=float("nan"),
                    inclusion_count=0,
                    exclusion_count=0,
                    inclusion_length=max(len(event.inclusion_junctions), 1),
                    exclusion_length=max(len(event.exclusion_junctions), 1),
                    total_reads=0,
                )
            )
        else:
            read_data = None if read_data_by_chrom is None else read_data_by_chrom.get(event.chrom)
            results.append(calculate_psi(event, je, read_data=read_data))

    logger.info(
        "Calculated PSI for %d events (%.1f%% with sufficient reads)",
        len(results),
        100 * sum(1 for r in results if r.total_reads > 0) / max(len(results), 1),
    )
    return results
