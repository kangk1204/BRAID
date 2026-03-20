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

from braid.io.bam_reader import JunctionEvidence
from braid.splicing.events import ASEvent, EventType

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


def calculate_psi(
    event: ASEvent,
    junction_evidence: JunctionEvidence,
) -> PSIResult:
    """Calculate PSI for a single alternative splicing event.

    Uses junction-based PSI formula:
        PSI = (I / lI) / (I / lI + S / lS)

    For retained intron events with no inclusion junctions, PSI is based
    on the absence of the exclusion junction reads relative to the
    total region coverage.

    Args:
        event: The AS event to quantify.
        junction_evidence: Junction evidence for the event's chromosome.

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

    # For RI events with no inclusion junctions, PSI = 1 - exclusion_fraction
    if event.event_type == EventType.RI and len(event.inclusion_junctions) == 0:
        # PSI ~ 1 when intron is retained (no exclusion reads)
        if exclusion_count == 0:
            psi = 1.0
        else:
            psi = 0.0  # Intron is spliced out
        return PSIResult(
            event_id=event.event_id,
            psi=psi,
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
) -> list[PSIResult]:
    """Calculate PSI for all events using per-chromosome junction evidence.

    Args:
        events: List of detected AS events.
        junction_evidence_by_chrom: Mapping from chromosome name to
            JunctionEvidence.

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
            results.append(calculate_psi(event, je))

    logger.info(
        "Calculated PSI for %d events (%.1f%% with sufficient reads)",
        len(results),
        100 * sum(1 for r in results if r.total_reads > 0) / max(len(results), 1),
    )
    return results
