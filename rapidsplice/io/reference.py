"""Reference genome handling for transcript assembly.

Provides efficient access to reference sequences for splice site motif
validation and transcript sequence extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pysam

logger = logging.getLogger(__name__)

# Canonical splice site dinucleotide motifs (donor, acceptor)
CANONICAL_MOTIFS: set[tuple[str, str]] = {
    ("GT", "AG"),  # U2-type major
    ("GC", "AG"),  # U2-type minor
    ("AT", "AC"),  # U12-type
}


class ReferenceGenome:
    """Interface to a reference genome FASTA file.

    Uses pysam.FastaFile for indexed random access to reference sequences.
    Supports splice site motif validation and sequence extraction.
    """

    def __init__(self, fasta_path: str) -> None:
        """Open a reference genome FASTA file.

        Args:
            fasta_path: Path to a FASTA file (must have .fai index).

        Raises:
            FileNotFoundError: If the FASTA file does not exist.
            ValueError: If the FASTA index (.fai) is missing.
        """
        path = Path(fasta_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference FASTA not found: {fasta_path}")
        fai_path = Path(f"{fasta_path}.fai")
        if not fai_path.exists():
            logger.info("FASTA index not found, attempting to create: %s", fai_path)
            pysam.faidx(fasta_path)
        self._fasta = pysam.FastaFile(fasta_path)
        self._path = fasta_path
        logger.info(
            "Loaded reference: %s (%d sequences)", fasta_path, self._fasta.nreferences
        )

    @property
    def chromosomes(self) -> list[str]:
        """Return list of reference sequence names."""
        return list(self._fasta.references)

    @property
    def chromosome_lengths(self) -> dict[str, int]:
        """Return dictionary mapping chromosome names to lengths."""
        return dict(zip(self._fasta.references, self._fasta.lengths))

    def fetch_sequence(self, chrom: str, start: int, end: int) -> str:
        """Fetch a reference sequence for a genomic region.

        Args:
            chrom: Chromosome name.
            start: 0-based start position.
            end: 0-based exclusive end position.

        Returns:
            Upper-case DNA sequence string.
        """
        return self._fasta.fetch(chrom, start, end).upper()

    def get_splice_motif(
        self, chrom: str, intron_start: int, intron_end: int
    ) -> tuple[str, str]:
        """Extract the donor and acceptor dinucleotide motifs for an intron.

        Args:
            chrom: Chromosome name.
            intron_start: 0-based start of the intron.
            intron_end: 0-based exclusive end of the intron.

        Returns:
            Tuple of (donor_motif, acceptor_motif), each a 2-character string.
        """
        donor = self._fasta.fetch(chrom, intron_start, intron_start + 2).upper()
        acceptor = self._fasta.fetch(chrom, intron_end - 2, intron_end).upper()
        return donor, acceptor

    def is_canonical_junction(
        self, chrom: str, intron_start: int, intron_end: int, strand: str = "+"
    ) -> bool:
        """Check if a splice junction has canonical donor/acceptor motifs.

        Args:
            chrom: Chromosome name.
            intron_start: 0-based start of the intron.
            intron_end: 0-based exclusive end of the intron.
            strand: Transcript strand ('+' or '-').

        Returns:
            True if the junction has a canonical splice motif.
        """
        donor, acceptor = self.get_splice_motif(chrom, intron_start, intron_end)
        if strand == "-":
            # Reverse complement
            rc = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
            donor_rc = rc.get(donor[1], "N") + rc.get(donor[0], "N")
            acceptor_rc = rc.get(acceptor[1], "N") + rc.get(acceptor[0], "N")
            return (acceptor_rc, donor_rc) in CANONICAL_MOTIFS
        return (donor, acceptor) in CANONICAL_MOTIFS

    def infer_strand_from_motif(
        self, chrom: str, intron_start: int, intron_end: int
    ) -> str:
        """Infer transcript strand from splice site motifs.

        Args:
            chrom: Chromosome name.
            intron_start: 0-based start of the intron.
            intron_end: 0-based exclusive end of the intron.

        Returns:
            '+', '-', or '.' if strand cannot be determined.
        """
        donor, acceptor = self.get_splice_motif(chrom, intron_start, intron_end)
        if (donor, acceptor) in CANONICAL_MOTIFS:
            return "+"
        # Check reverse complement
        rc = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
        donor_rc = rc.get(donor[1], "N") + rc.get(donor[0], "N")
        acceptor_rc = rc.get(acceptor[1], "N") + rc.get(acceptor[0], "N")
        if (acceptor_rc, donor_rc) in CANONICAL_MOTIFS:
            return "-"
        return "."

    def validate_junctions(
        self,
        chrom: str,
        junction_starts: np.ndarray,
        junction_ends: np.ndarray,
    ) -> np.ndarray:
        """Validate multiple junctions for canonical splice motifs.

        Args:
            chrom: Chromosome name.
            junction_starts: Array of intron start positions.
            junction_ends: Array of intron end positions.

        Returns:
            Boolean array, True for junctions with canonical motifs on either strand.
        """
        n = len(junction_starts)
        valid = np.zeros(n, dtype=bool)
        for i in range(n):
            fwd = self.is_canonical_junction(chrom, int(junction_starts[i]),
                                             int(junction_ends[i]), "+")
            rev = self.is_canonical_junction(chrom, int(junction_starts[i]),
                                             int(junction_ends[i]), "-")
            valid[i] = fwd or rev
        return valid

    def close(self) -> None:
        """Close the FASTA file handle."""
        self._fasta.close()

    def __enter__(self) -> ReferenceGenome:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._fasta.close()
        except Exception:
            pass
