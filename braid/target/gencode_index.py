"""Pre-indexed GENCODE annotation for fast region queries.

Loads the entire GENCODE GTF once and builds chromosome-indexed
data structures for O(1) lookup of reference transcripts overlapping
any genomic region.
"""

from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class IndexedAnnotation:
    """Pre-indexed GENCODE annotation for fast region queries.

    Attributes:
        n_transcripts: Total number of transcripts loaded.
        n_chromosomes: Number of chromosomes with transcripts.
    """

    # chrom -> sorted list of (start, end, transcript_dict)
    _chrom_data: dict[str, list[tuple[int, int, dict]]] = field(
        default_factory=dict,
    )
    # chrom -> sorted start positions for binary search
    _chrom_starts: dict[str, list[int]] = field(default_factory=dict)
    n_transcripts: int = 0
    n_chromosomes: int = 0


def load_annotation_index(gtf_path: str) -> IndexedAnnotation:
    """Load and index a GTF annotation file.

    Reads the entire GTF once and builds sorted per-chromosome
    transcript lists for efficient region overlap queries.

    Args:
        gtf_path: Path to GTF annotation file.

    Returns:
        Indexed annotation ready for fast queries.
    """
    import time
    t0 = time.perf_counter()

    # Parse all transcripts
    tx_data: dict[str, dict] = {}  # tid -> {exons, ...}
    chrom_txs: dict[str, list] = defaultdict(list)

    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            chrom = fields[0]
            feat_type = fields[2]
            start = int(fields[3]) - 1
            end = int(fields[4])
            attrs = fields[8]

            tid = _parse_attr(attrs, "transcript_id")
            if not tid:
                continue

            if feat_type == "transcript":
                gname = _parse_attr(attrs, "gene_name")
                tx_data[tid] = {
                    "transcript_id": tid,
                    "gene_name": gname,
                    "chrom": chrom,
                    "start": start,
                    "end": end,
                    "exons": [],
                }
            elif feat_type == "exon":
                if tid in tx_data:
                    tx_data[tid]["exons"].append((start, end))

    # Sort exons and build chromosome index
    idx = IndexedAnnotation()
    for tid, tx in tx_data.items():
        tx["exons"].sort(key=lambda e: e[0])
        chrom = tx["chrom"]
        chrom_txs[chrom].append((tx["start"], tx["end"], tx))

    for chrom, txs in chrom_txs.items():
        txs.sort(key=lambda x: x[0])
        idx._chrom_data[chrom] = txs
        idx._chrom_starts[chrom] = [t[0] for t in txs]

    idx.n_transcripts = len(tx_data)
    idx.n_chromosomes = len(chrom_txs)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Loaded GENCODE index: %d transcripts, %d chroms in %.2fs",
        idx.n_transcripts, idx.n_chromosomes, elapsed,
    )

    return idx


def query_region(
    idx: IndexedAnnotation,
    chrom: str,
    start: int,
    end: int,
) -> list[dict]:
    """Query reference transcripts overlapping a region.

    Uses binary search for O(log n) lookup.

    Args:
        idx: Pre-indexed annotation.
        chrom: Chromosome name.
        start: Region start.
        end: Region end.

    Returns:
        List of transcript dicts overlapping the region.
    """
    if chrom not in idx._chrom_data:
        return []

    starts = idx._chrom_starts[chrom]
    data = idx._chrom_data[chrom]

    # Binary search for first transcript that could overlap
    lo = bisect.bisect_left(starts, start - 1_000_000)  # generous window
    lo = max(0, lo)

    results: list[dict] = []
    for i in range(lo, len(data)):
        tx_start, tx_end, tx = data[i]
        if tx_start > end:
            break
        if tx_end > start and tx_start < end:
            results.append(tx)

    return results


def _parse_attr(attrs: str, key: str) -> str | None:
    """Extract an attribute value from a GTF attribute string."""
    for part in attrs.split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(None, 1)
        if len(tokens) == 2 and tokens[0] == key:
            return tokens[1].strip('"')
    return None
