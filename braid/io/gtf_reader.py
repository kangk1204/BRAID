"""GTF reader for loading guide transcripts from long-read or reference data.

Parses transcript structures from GTF files and organizes them by
chromosome and strand for efficient locus-level lookup during guided
assembly.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def read_guide_gtf(
    gtf_path: str,
) -> dict[tuple[str, str], list[list[tuple[int, int]]]]:
    """Read transcript exon structures from a GTF file.

    Parses exon features grouped by transcript_id, producing a dictionary
    keyed by ``(chrom, strand)`` with lists of exon-coordinate lists.
    Coordinates are converted from GTF 1-based inclusive to 0-based
    half-open format.

    Args:
        gtf_path: Path to the GTF file.

    Returns:
        Dictionary mapping ``(chrom, strand)`` to a list of transcripts,
        where each transcript is a sorted list of ``(start, end)`` exon
        intervals in 0-based half-open coordinates.
    """
    # transcript_id -> (chrom, strand, [(start, end), ...])
    tx_data: dict[str, tuple[str, str, list[tuple[int, int]]]] = {}

    n_lines = 0
    n_exons = 0

    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            n_lines += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != "exon":
                continue

            chrom = fields[0]
            strand = fields[6]
            # GTF is 1-based inclusive -> 0-based half-open
            start = int(fields[3]) - 1
            end = int(fields[4])

            # Parse transcript_id from attributes
            attrs = fields[8]
            tid = _parse_attribute(attrs, "transcript_id")
            if tid is None:
                continue

            n_exons += 1
            if tid not in tx_data:
                tx_data[tid] = (chrom, strand, [])
            tx_data[tid][2].append((start, end))

    # Organize by (chrom, strand)
    result: dict[tuple[str, str], list[list[tuple[int, int]]]] = defaultdict(list)
    for tid, (chrom, strand, exons) in tx_data.items():
        if not exons:
            continue
        # Sort exons by start position
        exons.sort(key=lambda e: e[0])
        result[(chrom, strand)].append(exons)

    n_tx = len(tx_data)
    n_keys = len(result)
    logger.info(
        "Guide GTF: %d lines, %d exons, %d transcripts across %d (chrom,strand) groups",
        n_lines, n_exons, n_tx, n_keys,
    )

    return dict(result)


def _parse_attribute(attrs: str, key: str) -> str | None:
    """Extract a single attribute value from a GTF attribute string.

    Args:
        attrs: The GTF attribute column (semicolon-separated key-value pairs).
        key: The attribute key to extract.

    Returns:
        The attribute value (unquoted) or ``None`` if not found.
    """
    for part in attrs.split(";"):
        part = part.strip()
        if not part:
            continue
        # Typical format: key "value"  or  key value
        tokens = part.split(None, 1)
        if len(tokens) == 2 and tokens[0] == key:
            return tokens[1].strip('"').strip("'")
    return None
