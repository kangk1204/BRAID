"""I/O for alternative splicing event results.

Supports two output formats:
- TSV: Full event table with PSI, CIs, and confidence scores.
- IOE: SUPPA2-compatible inclusion/exclusion format for interoperability.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

from braid.splicing.events import EVENT_TYPE_NAMES, ASEvent
from braid.splicing.psi import PSIResult

logger = logging.getLogger(__name__)

TSV_HEADER = [
    "event_id",
    "event_type",
    "gene_id",
    "chrom",
    "strand",
    "psi",
    "inclusion_count",
    "exclusion_count",
    "total_reads",
    "ci_low",
    "ci_high",
    "ci_width",
    "confidence_score",
    "inclusion_transcripts",
    "exclusion_transcripts",
    "inclusion_junctions",
    "exclusion_junctions",
    "coordinates",
]


def write_events_tsv(
    output_path: str | Path,
    events: list[ASEvent],
    psi_results: list[PSIResult],
    scores: list[float] | None = None,
) -> None:
    """Write events with PSI and scores to a TSV file.

    Args:
        output_path: Path to the output TSV file.
        events: List of detected AS events.
        psi_results: PSI results corresponding to events (same order).
        scores: Optional confidence scores (same order as events).
    """
    if scores is None:
        scores = [0.0] * len(events)
    elif len(scores) != len(events):
        raise ValueError(
            f"scores length ({len(scores)}) does not match events length ({len(events)})."
        )

    psi_lookup: dict[str, PSIResult] = {r.event_id: r for r in psi_results}

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(TSV_HEADER)

        for event, score in zip(events, scores):
            psi = psi_lookup.get(event.event_id)
            psi_val = psi.psi if psi is not None else float("nan")
            inc_count = psi.inclusion_count if psi else 0
            exc_count = psi.exclusion_count if psi else 0
            total = psi.total_reads if psi else 0
            ci_low = psi.ci_low if psi else 0.0
            ci_high = psi.ci_high if psi else 1.0
            ci_width = ci_high - ci_low

            psi_str = f"{psi_val:.4f}" if not math.isnan(psi_val) else "NA"

            inc_tx = ",".join(event.inclusion_transcripts)
            exc_tx = ",".join(event.exclusion_transcripts)
            inc_junc = ";".join(f"{j[0]}-{j[1]}" for j in event.inclusion_junctions)
            exc_junc = ";".join(f"{j[0]}-{j[1]}" for j in event.exclusion_junctions)
            coords = ";".join(f"{k}={v}" for k, v in event.coordinates.items())

            writer.writerow([
                event.event_id,
                EVENT_TYPE_NAMES[event.event_type],
                event.gene_id,
                event.chrom,
                event.strand,
                psi_str,
                inc_count,
                exc_count,
                total,
                f"{ci_low:.4f}",
                f"{ci_high:.4f}",
                f"{ci_width:.4f}",
                f"{score:.4f}",
                inc_tx,
                exc_tx,
                inc_junc,
                exc_junc,
                coords,
            ])

    logger.info("Wrote %d events to %s", len(events), output_path)


def read_events_tsv(input_path: str | Path) -> list[dict[str, str]]:
    """Read an events TSV file into a list of dictionaries.

    Args:
        input_path: Path to the TSV file.

    Returns:
        List of row dictionaries keyed by column header.
    """
    rows: list[dict[str, str]] = []
    with open(input_path, newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(dict(row))
    return rows


def write_ioe(
    output_path: str | Path,
    events: list[ASEvent],
) -> None:
    """Write events in SUPPA2-compatible IOE format.

    The IOE format has columns: seqname, gene_id, event_id,
    inclusion_transcripts, total_transcripts.

    Args:
        output_path: Path to the output IOE file.
        events: List of AS events to write.
    """
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow([
            "seqname",
            "gene_id",
            "event_id",
            "inclusion_transcripts",
            "total_transcripts",
        ])

        for event in events:
            all_tx = sorted(
                set(event.inclusion_transcripts + event.exclusion_transcripts)
            )
            writer.writerow([
                event.chrom,
                event.gene_id,
                event.event_id,
                ",".join(event.inclusion_transcripts),
                ",".join(all_tx),
            ])

    logger.info("Wrote %d events in IOE format to %s", len(events), output_path)
