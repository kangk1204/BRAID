"""Alternative splicing event detection from assembled transcripts.

Detects seven types of alternative splicing events by pairwise comparison
of transcripts within each gene locus:

- SE:   Skipped Exon
- A5SS: Alternative 5' Splice Site
- A3SS: Alternative 3' Splice Site
- MXE:  Mutually Exclusive Exons
- RI:   Retained Intron
- AFE:  Alternative First Exon
- ALE:  Alternative Last Exon
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum

from braid.io.gtf_writer import TranscriptRecord

logger = logging.getLogger(__name__)


class EventType(IntEnum):
    """Alternative splicing event types."""

    SE = 0
    A5SS = 1
    A3SS = 2
    MXE = 3
    RI = 4
    AFE = 5
    ALE = 6


EVENT_TYPE_NAMES: dict[EventType, str] = {
    EventType.SE: "SE",
    EventType.A5SS: "A5SS",
    EventType.A3SS: "A3SS",
    EventType.MXE: "MXE",
    EventType.RI: "RI",
    EventType.AFE: "AFE",
    EventType.ALE: "ALE",
}


@dataclass
class ASEvent:
    """A single alternative splicing event.

    Attributes:
        event_id: Unique coordinate-based identifier.
        event_type: One of the seven AS event types.
        gene_id: Parent gene identifier.
        chrom: Chromosome name.
        strand: Genomic strand ('+' or '-').
        coordinates: Event-type-specific coordinate dictionary.
        inclusion_transcripts: Transcript IDs supporting the inclusion isoform.
        exclusion_transcripts: Transcript IDs supporting the exclusion isoform.
        inclusion_junctions: Junction coordinates for inclusion reads.
        exclusion_junctions: Junction coordinates for exclusion reads.
    """

    event_id: str
    event_type: EventType
    gene_id: str
    chrom: str
    strand: str
    coordinates: dict[str, int | tuple[int, int]] = field(default_factory=dict)
    inclusion_transcripts: list[str] = field(default_factory=list)
    exclusion_transcripts: list[str] = field(default_factory=list)
    inclusion_junctions: list[tuple[int, int]] = field(default_factory=list)
    exclusion_junctions: list[tuple[int, int]] = field(default_factory=list)


def _get_introns(exons: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Extract intron intervals from sorted exon list."""
    introns = []
    for i in range(len(exons) - 1):
        introns.append((exons[i][1], exons[i + 1][0]))
    return introns


def _make_event_id(event_type: EventType, chrom: str, strand: str, *coords: int) -> str:
    """Create a unique coordinate-based event identifier."""
    coord_str = "_".join(str(c) for c in coords)
    return f"{EVENT_TYPE_NAMES[event_type]}:{chrom}:{strand}:{coord_str}"


def _detect_skipped_exons(
    transcripts: list[TranscriptRecord],
    gene_id: str,
) -> list[ASEvent]:
    """Detect skipped exon (SE) events.

    Transcript A includes an exon E between flanking exons U and D.
    Transcript B has a junction spanning directly from U to D, skipping E.
    """
    events: dict[str, ASEvent] = {}

    # Build junction sets for each transcript
    tx_junctions: dict[str, set[tuple[int, int]]] = {}
    for tx in transcripts:
        junctions = set()
        for i in range(len(tx.exons) - 1):
            junctions.add((tx.exons[i][1], tx.exons[i + 1][0]))
        tx_junctions[tx.transcript_id] = junctions

    for i, tx_a in enumerate(transcripts):
        if len(tx_a.exons) < 3:
            continue
        for j, tx_b in enumerate(transcripts):
            if i == j:
                continue
            junctions_b = tx_junctions[tx_b.transcript_id]

            # For each internal exon in tx_a, check if tx_b skips it
            for exon_idx in range(1, len(tx_a.exons) - 1):
                skipped_exon = tx_a.exons[exon_idx]
                upstream_exon = tx_a.exons[exon_idx - 1]
                downstream_exon = tx_a.exons[exon_idx + 1]

                # The skip junction goes from upstream end to downstream start
                skip_junction = (upstream_exon[1], downstream_exon[0])

                if skip_junction in junctions_b:
                    # Verify tx_b also has the flanking exons (at least overlapping)
                    has_upstream = any(
                        e[1] == upstream_exon[1] for e in tx_b.exons
                    )
                    has_downstream = any(
                        e[0] == downstream_exon[0] for e in tx_b.exons
                    )
                    if not (has_upstream and has_downstream):
                        continue

                    eid = _make_event_id(
                        EventType.SE,
                        tx_a.chrom,
                        tx_a.strand,
                        upstream_exon[1],
                        skipped_exon[0],
                        skipped_exon[1],
                        downstream_exon[0],
                    )

                    if eid not in events:
                        events[eid] = ASEvent(
                            event_id=eid,
                            event_type=EventType.SE,
                            gene_id=gene_id,
                            chrom=tx_a.chrom,
                            strand=tx_a.strand,
                            coordinates={
                                "upstream_exon_end": upstream_exon[1],
                                "skipped_exon_start": skipped_exon[0],
                                "skipped_exon_end": skipped_exon[1],
                                "downstream_exon_start": downstream_exon[0],
                            },
                            inclusion_junctions=[
                                (upstream_exon[1], skipped_exon[0]),
                                (skipped_exon[1], downstream_exon[0]),
                            ],
                            exclusion_junctions=[skip_junction],
                        )

                    event = events[eid]
                    if tx_a.transcript_id not in event.inclusion_transcripts:
                        event.inclusion_transcripts.append(tx_a.transcript_id)
                    if tx_b.transcript_id not in event.exclusion_transcripts:
                        event.exclusion_transcripts.append(tx_b.transcript_id)

    return list(events.values())


def _detect_alternative_5ss(
    transcripts: list[TranscriptRecord],
    gene_id: str,
) -> list[ASEvent]:
    """Detect alternative 5' splice site (A5SS) events.

    Two transcripts share the same acceptor (intron end) but differ in
    donor (intron start). On + strand, this means different exon 3' boundary.
    """
    events: dict[str, ASEvent] = {}

    for i, tx_a in enumerate(transcripts):
        introns_a = _get_introns(tx_a.exons)
        for j, tx_b in enumerate(transcripts):
            if j <= i:
                continue
            introns_b = _get_introns(tx_b.exons)

            for ia in introns_a:
                for ib in introns_b:
                    # Same acceptor (end), different donor (start)
                    if ia[1] == ib[1] and ia[0] != ib[0]:
                        event_type = (
                            EventType.A3SS if tx_a.strand == "-" else EventType.A5SS
                        )
                        short_donor = min(ia[0], ib[0])
                        long_donor = max(ia[0], ib[0])

                        eid = _make_event_id(
                            event_type,
                            tx_a.chrom,
                            tx_a.strand,
                            short_donor,
                            long_donor,
                            ia[1],
                        )

                        # On + strand: short_donor = longer exon = inclusion
                        # On - strand: short_donor = shorter exon = exclusion
                        # (transcription is reversed, so swap inc/exc)
                        if tx_a.strand == "-":
                            inc_junctions = [(long_donor, ia[1])]
                            exc_junctions = [(short_donor, ia[1])]
                        else:
                            inc_junctions = [(short_donor, ia[1])]
                            exc_junctions = [(long_donor, ia[1])]

                        if eid not in events:
                            events[eid] = ASEvent(
                                event_id=eid,
                                event_type=event_type,
                                gene_id=gene_id,
                                chrom=tx_a.chrom,
                                strand=tx_a.strand,
                                coordinates={
                                    "short_donor": short_donor,
                                    "long_donor": long_donor,
                                    "acceptor": ia[1],
                                },
                                inclusion_junctions=inc_junctions,
                                exclusion_junctions=exc_junctions,
                            )

                        event = events[eid]
                        # On + strand: inclusion = longer exon (short donor)
                        # On - strand: inclusion = longer exon (long donor)
                        if tx_a.strand == "-":
                            inc_tx = tx_a if ia[0] == long_donor else tx_b
                            exc_tx = tx_b if ia[0] == long_donor else tx_a
                        else:
                            inc_tx = tx_a if ia[0] == short_donor else tx_b
                            exc_tx = tx_b if ia[0] == short_donor else tx_a

                        if inc_tx.transcript_id not in event.inclusion_transcripts:
                            event.inclusion_transcripts.append(inc_tx.transcript_id)
                        if exc_tx.transcript_id not in event.exclusion_transcripts:
                            event.exclusion_transcripts.append(exc_tx.transcript_id)

    return list(events.values())


def _detect_alternative_3ss(
    transcripts: list[TranscriptRecord],
    gene_id: str,
) -> list[ASEvent]:
    """Detect alternative 3' splice site (A3SS) events.

    Two transcripts share the same donor (intron start) but differ in
    acceptor (intron end). On + strand, different exon 5' boundary.
    """
    events: dict[str, ASEvent] = {}

    for i, tx_a in enumerate(transcripts):
        introns_a = _get_introns(tx_a.exons)
        for j, tx_b in enumerate(transcripts):
            if j <= i:
                continue
            introns_b = _get_introns(tx_b.exons)

            for ia in introns_a:
                for ib in introns_b:
                    # Same donor (start), different acceptor (end)
                    if ia[0] == ib[0] and ia[1] != ib[1]:
                        event_type = (
                            EventType.A5SS if tx_a.strand == "-" else EventType.A3SS
                        )
                        short_acceptor = min(ia[1], ib[1])
                        long_acceptor = max(ia[1], ib[1])

                        eid = _make_event_id(
                            event_type,
                            tx_a.chrom,
                            tx_a.strand,
                            ia[0],
                            short_acceptor,
                            long_acceptor,
                        )

                        # On + strand: long_acceptor = longer exon = inclusion
                        # On - strand: long_acceptor = shorter exon = exclusion
                        # (transcription is reversed, so swap inc/exc)
                        if tx_a.strand == "-":
                            inc_junctions = [(ia[0], short_acceptor)]
                            exc_junctions = [(ia[0], long_acceptor)]
                        else:
                            inc_junctions = [(ia[0], long_acceptor)]
                            exc_junctions = [(ia[0], short_acceptor)]

                        if eid not in events:
                            events[eid] = ASEvent(
                                event_id=eid,
                                event_type=event_type,
                                gene_id=gene_id,
                                chrom=tx_a.chrom,
                                strand=tx_a.strand,
                                coordinates={
                                    "donor": ia[0],
                                    "short_acceptor": short_acceptor,
                                    "long_acceptor": long_acceptor,
                                },
                                inclusion_junctions=inc_junctions,
                                exclusion_junctions=exc_junctions,
                            )

                        event = events[eid]
                        # On + strand: inclusion = longer exon (long acceptor)
                        # On - strand: inclusion = longer exon (short acceptor)
                        if tx_a.strand == "-":
                            inc_tx = tx_a if ia[1] == short_acceptor else tx_b
                            exc_tx = tx_b if ia[1] == short_acceptor else tx_a
                        else:
                            inc_tx = tx_a if ia[1] == long_acceptor else tx_b
                            exc_tx = tx_b if ia[1] == long_acceptor else tx_a

                        if inc_tx.transcript_id not in event.inclusion_transcripts:
                            event.inclusion_transcripts.append(inc_tx.transcript_id)
                        if exc_tx.transcript_id not in event.exclusion_transcripts:
                            event.exclusion_transcripts.append(exc_tx.transcript_id)

    return list(events.values())


def _detect_mutually_exclusive_exons(
    transcripts: list[TranscriptRecord],
    gene_id: str,
) -> list[ASEvent]:
    """Detect mutually exclusive exon (MXE) events.

    Between same flanking exons U and D, transcript A uses exon X while
    transcript B uses non-overlapping exon Y.
    """
    events: dict[str, ASEvent] = {}

    for i, tx_a in enumerate(transcripts):
        if len(tx_a.exons) < 3:
            continue
        for j, tx_b in enumerate(transcripts):
            if j <= i or len(tx_b.exons) < 3:
                continue

            # Find shared upstream and downstream exon boundaries
            for ea_idx in range(1, len(tx_a.exons) - 1):
                exon_x = tx_a.exons[ea_idx]
                upstream_a = tx_a.exons[ea_idx - 1]
                downstream_a = tx_a.exons[ea_idx + 1]

                for eb_idx in range(1, len(tx_b.exons) - 1):
                    exon_y = tx_b.exons[eb_idx]
                    upstream_b = tx_b.exons[eb_idx - 1]
                    downstream_b = tx_b.exons[eb_idx + 1]

                    # Same flanking exon boundaries
                    if upstream_a[1] != upstream_b[1]:
                        continue
                    if downstream_a[0] != downstream_b[0]:
                        continue

                    # Exons must be non-overlapping and different
                    if exon_x == exon_y:
                        continue
                    if max(exon_x[0], exon_y[0]) < min(exon_x[1], exon_y[1]):
                        continue  # Overlapping

                    # Order consistently
                    if exon_x[0] > exon_y[0]:
                        continue  # Only report once (X before Y)

                    eid = _make_event_id(
                        EventType.MXE,
                        tx_a.chrom,
                        tx_a.strand,
                        upstream_a[1],
                        exon_x[0],
                        exon_x[1],
                        exon_y[0],
                        exon_y[1],
                        downstream_a[0],
                    )

                    if eid not in events:
                        events[eid] = ASEvent(
                            event_id=eid,
                            event_type=EventType.MXE,
                            gene_id=gene_id,
                            chrom=tx_a.chrom,
                            strand=tx_a.strand,
                            coordinates={
                                "upstream_exon_end": upstream_a[1],
                                "exon_x_start": exon_x[0],
                                "exon_x_end": exon_x[1],
                                "exon_y_start": exon_y[0],
                                "exon_y_end": exon_y[1],
                                "downstream_exon_start": downstream_a[0],
                            },
                            inclusion_junctions=[
                                (upstream_a[1], exon_x[0]),
                                (exon_x[1], downstream_a[0]),
                            ],
                            exclusion_junctions=[
                                (upstream_a[1], exon_y[0]),
                                (exon_y[1], downstream_a[0]),
                            ],
                        )

                    event = events[eid]
                    if tx_a.transcript_id not in event.inclusion_transcripts:
                        event.inclusion_transcripts.append(tx_a.transcript_id)
                    if tx_b.transcript_id not in event.exclusion_transcripts:
                        event.exclusion_transcripts.append(tx_b.transcript_id)

    return list(events.values())


def _detect_retained_introns(
    transcripts: list[TranscriptRecord],
    gene_id: str,
) -> list[ASEvent]:
    """Detect retained intron (RI) events.

    Transcript A has one exon spanning a region where transcript B has two
    exons with an intron between them.
    """
    events: dict[str, ASEvent] = {}

    for i, tx_a in enumerate(transcripts):
        for j, tx_b in enumerate(transcripts):
            if i == j:
                continue
            introns_b = _get_introns(tx_b.exons)

            for intron in introns_b:
                intron_start, intron_end = intron
                # Check if tx_a has an exon spanning this intron
                for exon in tx_a.exons:
                    if exon[0] <= intron_start and exon[1] >= intron_end:
                        eid = _make_event_id(
                            EventType.RI,
                            tx_a.chrom,
                            tx_a.strand,
                            intron_start,
                            intron_end,
                        )

                        if eid not in events:
                            events[eid] = ASEvent(
                                event_id=eid,
                                event_type=EventType.RI,
                                gene_id=gene_id,
                                chrom=tx_a.chrom,
                                strand=tx_a.strand,
                                coordinates={
                                    "intron_start": intron_start,
                                    "intron_end": intron_end,
                                },
                                inclusion_junctions=[],
                                exclusion_junctions=[(intron_start, intron_end)],
                            )

                        event = events[eid]
                        if tx_a.transcript_id not in event.inclusion_transcripts:
                            event.inclusion_transcripts.append(tx_a.transcript_id)
                        if tx_b.transcript_id not in event.exclusion_transcripts:
                            event.exclusion_transcripts.append(tx_b.transcript_id)

    return list(events.values())


def _detect_alternative_first_exons(
    transcripts: list[TranscriptRecord],
    gene_id: str,
) -> list[ASEvent]:
    """Detect alternative first exon (AFE) events.

    Different first exons converging to the same downstream exon via
    different junctions.
    """
    events: dict[str, ASEvent] = {}

    for i, tx_a in enumerate(transcripts):
        if len(tx_a.exons) < 2:
            continue
        for j, tx_b in enumerate(transcripts):
            if j <= i or len(tx_b.exons) < 2:
                continue

            first_a = tx_a.exons[0]
            first_b = tx_b.exons[0]
            second_a = tx_a.exons[1]
            second_b = tx_b.exons[1]

            # Different first exons, same second exon start (convergence)
            if first_a != first_b and second_a[0] == second_b[0]:
                # Order by first exon start
                if first_a[0] > first_b[0]:
                    continue

                eid = _make_event_id(
                    EventType.AFE,
                    tx_a.chrom,
                    tx_a.strand,
                    first_a[0],
                    first_a[1],
                    first_b[0],
                    first_b[1],
                    second_a[0],
                )

                if eid not in events:
                    events[eid] = ASEvent(
                        event_id=eid,
                        event_type=EventType.AFE,
                        gene_id=gene_id,
                        chrom=tx_a.chrom,
                        strand=tx_a.strand,
                        coordinates={
                            "first_exon_a_start": first_a[0],
                            "first_exon_a_end": first_a[1],
                            "first_exon_b_start": first_b[0],
                            "first_exon_b_end": first_b[1],
                            "downstream_exon_start": second_a[0],
                        },
                        inclusion_junctions=[(first_a[1], second_a[0])],
                        exclusion_junctions=[(first_b[1], second_b[0])],
                    )

                event = events[eid]
                if tx_a.transcript_id not in event.inclusion_transcripts:
                    event.inclusion_transcripts.append(tx_a.transcript_id)
                if tx_b.transcript_id not in event.exclusion_transcripts:
                    event.exclusion_transcripts.append(tx_b.transcript_id)

    return list(events.values())


def _detect_alternative_last_exons(
    transcripts: list[TranscriptRecord],
    gene_id: str,
) -> list[ASEvent]:
    """Detect alternative last exon (ALE) events.

    Same upstream exon diverging to different last exons.
    """
    events: dict[str, ASEvent] = {}

    for i, tx_a in enumerate(transcripts):
        if len(tx_a.exons) < 2:
            continue
        for j, tx_b in enumerate(transcripts):
            if j <= i or len(tx_b.exons) < 2:
                continue

            last_a = tx_a.exons[-1]
            last_b = tx_b.exons[-1]
            penult_a = tx_a.exons[-2]
            penult_b = tx_b.exons[-2]

            # Different last exons, same penultimate exon end (divergence)
            if last_a != last_b and penult_a[1] == penult_b[1]:
                # Order by last exon start
                if last_a[0] > last_b[0]:
                    continue

                eid = _make_event_id(
                    EventType.ALE,
                    tx_a.chrom,
                    tx_a.strand,
                    penult_a[1],
                    last_a[0],
                    last_a[1],
                    last_b[0],
                    last_b[1],
                )

                if eid not in events:
                    events[eid] = ASEvent(
                        event_id=eid,
                        event_type=EventType.ALE,
                        gene_id=gene_id,
                        chrom=tx_a.chrom,
                        strand=tx_a.strand,
                        coordinates={
                            "upstream_exon_end": penult_a[1],
                            "last_exon_a_start": last_a[0],
                            "last_exon_a_end": last_a[1],
                            "last_exon_b_start": last_b[0],
                            "last_exon_b_end": last_b[1],
                        },
                        inclusion_junctions=[(penult_a[1], last_a[0])],
                        exclusion_junctions=[(penult_a[1], last_b[0])],
                    )

                event = events[eid]
                if tx_a.transcript_id not in event.inclusion_transcripts:
                    event.inclusion_transcripts.append(tx_a.transcript_id)
                if tx_b.transcript_id not in event.exclusion_transcripts:
                    event.exclusion_transcripts.append(tx_b.transcript_id)

    return list(events.values())


def detect_events_for_gene(
    transcripts: list[TranscriptRecord],
    gene_id: str,
) -> list[ASEvent]:
    """Detect all AS events for a single gene's transcripts.

    Runs all seven detector functions and returns a deduplicated list.

    Args:
        transcripts: Transcripts belonging to the same gene.
        gene_id: Gene identifier.

    Returns:
        List of deduplicated AS events.
    """
    if len(transcripts) < 2:
        return []

    all_events: list[ASEvent] = []
    all_events.extend(_detect_skipped_exons(transcripts, gene_id))
    all_events.extend(_detect_alternative_5ss(transcripts, gene_id))
    all_events.extend(_detect_alternative_3ss(transcripts, gene_id))
    all_events.extend(_detect_mutually_exclusive_exons(transcripts, gene_id))
    all_events.extend(_detect_retained_introns(transcripts, gene_id))
    all_events.extend(_detect_alternative_first_exons(transcripts, gene_id))
    all_events.extend(_detect_alternative_last_exons(transcripts, gene_id))

    return all_events


def detect_all_events(transcripts: list[TranscriptRecord]) -> list[ASEvent]:
    """Detect all alternative splicing events across all genes.

    Groups transcripts by gene_id, then runs per-gene detection.

    Args:
        transcripts: All assembled transcripts.

    Returns:
        List of all detected AS events across all genes.
    """
    # Group by gene_id
    gene_groups: dict[str, list[TranscriptRecord]] = defaultdict(list)
    for tx in transcripts:
        gene_groups[tx.gene_id].append(tx)

    all_events: list[ASEvent] = []
    for gene_id, gene_txs in gene_groups.items():
        if len(gene_txs) < 2:
            continue
        events = detect_events_for_gene(gene_txs, gene_id)
        all_events.extend(events)

    logger.info(
        "Detected %d AS events across %d genes",
        len(all_events),
        len(gene_groups),
    )
    return all_events
