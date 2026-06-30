"""FASTQ file reader for de novo assembly.

Provides streaming FASTQ parsing with optional quality filtering and
read trimming.  Supports both plain-text and gzip-compressed files.
"""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FastqRead:
    """A single FASTQ read.

    Attributes:
        name: Read identifier (without leading '@').
        sequence: DNA sequence string (uppercase ACGTN).
        quality: Phred+33 quality string.
    """

    name: str
    sequence: str
    quality: str

    @property
    def length(self) -> int:
        """Return the length of the read sequence."""
        return len(self.sequence)


def read_fastq(
    path: str | Path,
    min_length: int = 0,
    min_avg_quality: float = 0.0,
    trim_quality: int = 0,
) -> list[FastqRead]:
    """Read all records from a FASTQ file.

    Supports both plain-text and gzip-compressed (.gz) FASTQ files.
    Optionally filters reads by length and average quality, and performs
    3'-end quality trimming.

    Args:
        path: Path to FASTQ file (plain or .gz).
        min_length: Minimum read length to retain (after trimming).
        min_avg_quality: Minimum average Phred quality to retain.
        trim_quality: If > 0, trim 3' bases below this quality threshold
            using a sliding window approach.

    Returns:
        List of FastqRead objects passing all filters.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTQ file not found: {path}")

    opener = gzip.open if path.suffix == ".gz" else open
    reads: list[FastqRead] = []
    total = 0
    filtered = 0

    with opener(path, "rt") as fh:
        while True:
            header = fh.readline().rstrip("\n")
            if not header:
                break
            seq = fh.readline().rstrip("\n")
            fh.readline()  # '+' line
            qual = fh.readline().rstrip("\n")

            if not header.startswith("@"):
                logger.warning("Malformed FASTQ record: %s", header[:50])
                continue

            total += 1
            name = header[1:].split()[0]

            # Quality trimming from 3' end
            if trim_quality > 0:
                seq, qual = _trim_3prime(seq, qual, trim_quality)

            # Length filter
            if len(seq) < min_length:
                filtered += 1
                continue

            # Average quality filter
            if min_avg_quality > 0.0:
                avg_q = _avg_quality(qual)
                if avg_q < min_avg_quality:
                    filtered += 1
                    continue

            reads.append(FastqRead(
                name=name,
                sequence=seq.upper(),
                quality=qual,
            ))

    logger.info(
        "Read %d FASTQ records from %s (%d filtered, %d retained)",
        total, path.name, filtered, len(reads),
    )
    return reads


def stream_fastq_sequences(
    path: str | Path,
    min_length: int = 0,
) -> list[str]:
    """Read only sequences from a FASTQ file (memory-efficient).

    Skips quality filtering for maximum throughput when only sequences
    are needed (e.g., for k-mer counting).

    Args:
        path: Path to FASTQ file (plain or .gz).
        min_length: Minimum read length to retain.

    Returns:
        List of uppercase DNA sequence strings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FASTQ file not found: {path}")

    opener = gzip.open if path.suffix == ".gz" else open
    sequences: list[str] = []

    with opener(path, "rt") as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().rstrip("\n")
            fh.readline()  # '+' line
            fh.readline()  # quality line

            if len(seq) >= min_length:
                sequences.append(seq.upper())

    logger.info(
        "Loaded %d sequences from %s",
        len(sequences), path.name if hasattr(path, 'name') else path,
    )
    return sequences


def _trim_3prime(seq: str, qual: str, threshold: int) -> tuple[str, str]:
    """Trim low-quality bases from the 3' end.

    Uses a simple right-to-left scan removing bases below the threshold.

    Args:
        seq: DNA sequence.
        qual: Phred+33 quality string.
        threshold: Minimum quality score to keep.

    Returns:
        Tuple of (trimmed_sequence, trimmed_quality).
    """
    end = len(seq)
    while end > 0 and (ord(qual[end - 1]) - 33) < threshold:
        end -= 1
    return seq[:end], qual[:end]


def _avg_quality(qual: str) -> float:
    """Compute average Phred quality score from a quality string.

    Args:
        qual: Phred+33 encoded quality string.

    Returns:
        Average quality score as a float.
    """
    if not qual:
        return 0.0
    return sum(ord(c) - 33 for c in qual) / len(qual)
