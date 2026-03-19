"""BAM file reader for GPU-accelerated RNA-seq transcript assembly.

Extracts aligned read data from BAM files into NumPy arrays suitable for
bulk transfer to GPU memory. Uses pysam for BAM I/O and produces flat,
columnar data structures that map directly to GPU buffers.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pysam

from braid.core.cigar import CIGAR_N

logger = logging.getLogger(__name__)

# Default flag filter: unmapped (4) | mate unmapped (8) | not primary (256) |
# fails QC (512) | duplicate (1024) = 1796 (plus mate unmapped excluded from
# the canonical 1804 because single-end reads would all fail).
_DEFAULT_FILTER_FLAGS: int = 1796


@dataclass(slots=True)
class ReadData:
    """Bulk-extracted read information stored as columnar NumPy arrays.

    All per-read arrays have length ``n_reads``.  CIGAR data is stored as two
    flat parallel arrays (``cigar_ops`` and ``cigar_lens``) with a per-read
    offset array (``cigar_offsets``, length ``n_reads + 1``) that indexes into
    the flat arrays so that read *i* owns entries
    ``cigar_offsets[i]:cigar_offsets[i+1]``.

    Attributes:
        chrom_ids: Chromosome / reference-sequence index per read (int32).
        positions: 0-based reference start position per read (int64).
        end_positions: Reference end position per read, exclusive (int64).
        strands: Strand encoding per read, 0 = forward, 1 = reverse (int8).
        mapping_qualities: Mapping quality per read (uint8).
        is_paired: Whether the read is part of a proper pair (bool).
        is_read1: Whether the read is the first in a pair (bool).
        mate_positions: Mate reference start, -1 if unpaired or missing (int64).
        mate_chrom_ids: Mate reference-sequence index, -1 if unpaired (int32).
        cigar_ops: Flat CIGAR operation codes for all reads (uint8).
        cigar_lens: Flat CIGAR operation lengths for all reads (int32).
        cigar_offsets: Per-read start index into the flat CIGAR arrays (int64).
        query_names: Read names kept as a Python list (CPU only).
        n_reads: Total number of reads extracted.
    """

    chrom_ids: np.ndarray
    positions: np.ndarray
    end_positions: np.ndarray
    strands: np.ndarray
    mapping_qualities: np.ndarray
    is_paired: np.ndarray
    is_read1: np.ndarray
    mate_positions: np.ndarray
    mate_chrom_ids: np.ndarray
    cigar_ops: np.ndarray
    cigar_lens: np.ndarray
    cigar_offsets: np.ndarray
    query_names: list[str]
    n_reads: int


@dataclass(slots=True)
class JunctionEvidence:
    """Splice-junction evidence aggregated from aligned reads.

    Each entry represents a unique donor-acceptor junction with an associated
    support count and strand consensus.

    Attributes:
        chrom: Chromosome name.
        starts: Junction donor (5') positions, 0-based (int64).
        ends: Junction acceptor (3') positions, 0-based exclusive (int64).
        counts: Number of reads supporting each junction (int32).
        strands: Strand of the supporting reads (int8, 0=fwd, 1=rev, -1=ambiguous).
    """

    chrom: str
    starts: np.ndarray
    ends: np.ndarray
    counts: np.ndarray
    strands: np.ndarray


@dataclass(slots=True)
class JunctionExtractionStats:
    """Stage counts from lightweight BAM junction extraction."""

    chrom: str
    raw_junctions: int = 0
    anchor_filtered_junctions: int = 0
    motif_filtered_junctions: int = 0
    output_junctions: int = 0


def _empty_read_data() -> ReadData:
    """Construct an empty ``ReadData`` with zero reads."""
    return ReadData(
        chrom_ids=np.empty(0, dtype=np.int32),
        positions=np.empty(0, dtype=np.int64),
        end_positions=np.empty(0, dtype=np.int64),
        strands=np.empty(0, dtype=np.int8),
        mapping_qualities=np.empty(0, dtype=np.uint8),
        is_paired=np.empty(0, dtype=np.bool_),
        is_read1=np.empty(0, dtype=np.bool_),
        mate_positions=np.empty(0, dtype=np.int64),
        mate_chrom_ids=np.empty(0, dtype=np.int32),
        cigar_ops=np.empty(0, dtype=np.uint8),
        cigar_lens=np.empty(0, dtype=np.int32),
        cigar_offsets=np.zeros(1, dtype=np.int64),
        query_names=[],
        n_reads=0,
    )


class BamReader:
    """High-throughput BAM reader that extracts reads into columnar NumPy arrays.

    Opens an indexed BAM file and provides methods for extracting reads from
    specific regions or the entire file.  Reads are filtered by mapping quality
    and SAM flags at extraction time.  The output ``ReadData`` structure is
    designed for efficient bulk transfer to GPU memory.

    Args:
        bam_path: Path to a coordinate-sorted, indexed BAM file.
        min_mapq: Minimum mapping quality to retain a read (inclusive).
        required_flags: SAM flags that must all be set (bitwise AND).
        filter_flags: SAM flags that must all be unset; reads with any of these
            bits set are discarded.  Default 1796 filters unmapped, not-primary,
            failed-QC, and duplicate reads.

    Raises:
        FileNotFoundError: If *bam_path* does not exist.
        FileNotFoundError: If the BAM index (``.bai``) is missing.
    """

    def __init__(
        self,
        bam_path: str,
        min_mapq: int = 0,
        required_flags: int = 0,
        filter_flags: int = _DEFAULT_FILTER_FLAGS,
    ) -> None:
        path = Path(bam_path)
        if not path.exists():
            raise FileNotFoundError(f"BAM file not found: {bam_path}")

        # pysam will raise its own error if the index is missing, but we give
        # a friendlier message up front.
        index_candidates = [
            path.with_suffix(".bam.bai"),
            path.with_name(path.name + ".bai"),
        ]
        if not any(p.exists() for p in index_candidates):
            raise FileNotFoundError(
                f"BAM index not found.  Expected one of: "
                f"{', '.join(str(p) for p in index_candidates)}"
            )

        self._bam_path: str = str(path)
        self._min_mapq: int = min_mapq
        self._required_flags: int = required_flags
        self._filter_flags: int = filter_flags

        # Open the file once to cache header metadata, then close so that each
        # fetch opens its own handle (safe for potential future multi-threading).
        with pysam.AlignmentFile(self._bam_path, "rb") as af:
            header = af.header
            self._chromosomes: list[str] = list(header.references)
            self._chromosome_lengths: dict[str, int] = dict(
                zip(header.references, header.lengths)
            )

        logger.info(
            "Opened BAM %s (%d references)", self._bam_path, len(self._chromosomes)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chromosomes(self) -> list[str]:
        """List of reference sequence names from the BAM header."""
        return list(self._chromosomes)

    @property
    def chromosome_lengths(self) -> dict[str, int]:
        """Mapping of reference sequence name to length."""
        return dict(self._chromosome_lengths)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_region(
        self,
        chrom: str,
        start: int | None = None,
        end: int | None = None,
    ) -> ReadData:
        """Extract all reads overlapping a genomic region.

        Args:
            chrom: Reference sequence name (must exist in the BAM header).
            start: 0-based start coordinate (inclusive).  ``None`` means the
                beginning of the chromosome.
            end: 0-based end coordinate (exclusive).  ``None`` means the end of
                the chromosome.

        Returns:
            A ``ReadData`` instance with all qualifying reads in the region.

        Raises:
            ValueError: If *chrom* is not present in the BAM header.
        """
        if chrom not in self._chromosome_lengths:
            raise ValueError(
                f"Chromosome {chrom!r} not found in BAM header.  "
                f"Available: {', '.join(self._chromosomes[:10])}"
                f"{'...' if len(self._chromosomes) > 10 else ''}"
            )
        with pysam.AlignmentFile(self._bam_path, "rb") as af:
            return self._extract_reads(af.fetch(chrom, start, end))

    def fetch_all(self) -> ReadData:
        """Extract all mapped reads from the BAM file.

        Returns:
            A ``ReadData`` instance containing every qualifying read across all
            reference sequences.
        """
        with pysam.AlignmentFile(self._bam_path, "rb") as af:
            return self._extract_reads(af.fetch(until_eof=True))

    def fetch_locus_reads(
        self,
        chrom: str,
        start: int,
        end: int,
    ) -> ReadData:
        """Extract reads for a single gene locus with tighter filtering.

        This is functionally equivalent to :meth:`fetch_region` but applies an
        additional post-filter: only reads whose *alignment start* falls within
        ``[start, end)`` are kept.  This avoids collecting reads that merely
        overlap the boundary, which is useful when processing loci
        independently.

        Args:
            chrom: Chromosome name.
            start: Locus start (0-based, inclusive).
            end: Locus end (0-based, exclusive).

        Returns:
            ``ReadData`` containing reads whose start position is within the
            specified interval.
        """
        if chrom not in self._chromosome_lengths:
            raise ValueError(
                f"Chromosome {chrom!r} not found in BAM header."
            )
        with pysam.AlignmentFile(self._bam_path, "rb") as af:
            return self._extract_reads(
                af.fetch(chrom, start, end),
                locus_start=start,
                locus_end=end,
            )

    def count_reads(self, chrom: str | None = None) -> int:
        """Return the number of mapped reads, optionally for a single chromosome.

        Uses the BAM index statistics for a fast count without iterating over
        reads.  Falls back to iteration if index stats are unavailable.

        Args:
            chrom: If given, count only reads on this reference.  Otherwise
                count reads on all references.

        Returns:
            Number of mapped reads passing the current flag/quality filters.
        """
        with pysam.AlignmentFile(self._bam_path, "rb") as af:
            try:
                idx_stats = af.get_index_statistics()
            except Exception:
                # Fallback: iterate and count.
                return self._count_by_iteration(af, chrom)

            if chrom is not None:
                for stat in idx_stats:
                    if stat.contig == chrom:
                        return stat.mapped
                raise ValueError(f"Chromosome {chrom!r} not found in BAM index.")

            return sum(stat.mapped for stat in idx_stats)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_by_iteration(
        self,
        af: pysam.AlignmentFile,
        chrom: str | None,
    ) -> int:
        """Count qualifying reads by iterating over the BAM file.

        Args:
            af: An open ``pysam.AlignmentFile``.
            chrom: Optional chromosome to restrict counting.

        Returns:
            Number of reads passing filters.
        """
        count = 0
        iterator = af.fetch(chrom) if chrom else af.fetch(until_eof=True)
        for read in iterator:
            if self._passes_filters(read):
                count += 1
        return count

    def _passes_filters(self, read: pysam.AlignedSegment) -> bool:
        """Return ``True`` if a read passes mapping quality and flag filters.

        Args:
            read: A pysam aligned segment.

        Returns:
            Whether the read satisfies all filter criteria.
        """
        if read.is_unmapped:
            return False
        if read.mapping_quality < self._min_mapq:
            return False
        flag = read.flag
        if self._required_flags and (flag & self._required_flags) != self._required_flags:
            return False
        if flag & self._filter_flags:
            return False
        return True

    def _extract_reads(
        self,
        iterator: pysam.IteratorRow,
        locus_start: int | None = None,
        locus_end: int | None = None,
    ) -> ReadData:
        """Core extraction loop: iterate pysam reads into columnar lists.

        We collect into Python lists (pysam iteration is already Python-speed)
        and convert to NumPy arrays at the end.  CIGAR data is gathered into
        two flat lists (ops, lens) with a per-read offset list.

        Args:
            iterator: A pysam fetch iterator.
            locus_start: If set, only keep reads whose alignment start >= this.
            locus_end: If set, only keep reads whose alignment start < this.

        Returns:
            Populated ``ReadData``.
        """
        # Per-read lists
        chrom_ids: list[int] = []
        positions: list[int] = []
        end_positions: list[int] = []
        strands: list[int] = []
        mapqs: list[int] = []
        paired: list[bool] = []
        read1: list[bool] = []
        mate_pos: list[int] = []
        mate_chroms: list[int] = []
        query_names: list[str] = []

        # Flat CIGAR lists
        all_cigar_ops: list[int] = []
        all_cigar_lens: list[int] = []
        cigar_offsets: list[int] = [0]

        n_reads = 0

        for read in iterator:
            if not self._passes_filters(read):
                continue

            ref_start: int = read.reference_start
            # Locus restriction: skip reads whose start is outside the window.
            if locus_start is not None and ref_start < locus_start:
                continue
            if locus_end is not None and ref_start >= locus_end:
                continue

            chrom_ids.append(read.reference_id)
            positions.append(ref_start)
            ref_end = read.reference_end
            end_positions.append(ref_end if ref_end is not None else ref_start)
            if read.has_tag("XS"):
                strands.append(1 if read.get_tag("XS") == "-" else 0)
            else:
                strands.append(1 if read.is_reverse else 0)
            mapqs.append(read.mapping_quality)
            paired.append(read.is_paired)
            read1.append(read.is_read1)

            # Mate information
            if read.is_paired and not read.mate_is_unmapped:
                mate_pos.append(read.next_reference_start)
                mate_chroms.append(read.next_reference_id)
            else:
                mate_pos.append(-1)
                mate_chroms.append(-1)

            query_names.append(read.query_name or "")

            # Flatten CIGAR
            cigar_tuples = read.cigartuples
            if cigar_tuples is not None:
                for op, length in cigar_tuples:
                    all_cigar_ops.append(op)
                    all_cigar_lens.append(length)
                cigar_offsets.append(len(all_cigar_ops))
            else:
                # Reads without CIGAR (should not pass filters, but be safe)
                cigar_offsets.append(len(all_cigar_ops))

            n_reads += 1

        if n_reads == 0:
            return _empty_read_data()

        # Convert to NumPy arrays
        return ReadData(
            chrom_ids=np.array(chrom_ids, dtype=np.int32),
            positions=np.array(positions, dtype=np.int64),
            end_positions=np.array(end_positions, dtype=np.int64),
            strands=np.array(strands, dtype=np.int8),
            mapping_qualities=np.array(mapqs, dtype=np.uint8),
            is_paired=np.array(paired, dtype=np.bool_),
            is_read1=np.array(read1, dtype=np.bool_),
            mate_positions=np.array(mate_pos, dtype=np.int64),
            mate_chrom_ids=np.array(mate_chroms, dtype=np.int32),
            cigar_ops=np.array(all_cigar_ops, dtype=np.uint8),
            cigar_lens=np.array(all_cigar_lens, dtype=np.int32),
            cigar_offsets=np.array(cigar_offsets, dtype=np.int64),
            query_names=query_names,
            n_reads=n_reads,
        )


def extract_junctions(read_data: ReadData, chrom: str) -> JunctionEvidence:
    """Extract unique splice junctions with support counts from ``ReadData``.

    Walks the flat CIGAR arrays looking for ``N`` (skip / intron) operations
    and reconstructs the reference coordinates of each junction.  Identical
    junctions from different reads are collapsed, and a majority-vote strand
    is assigned per unique junction.

    Args:
        read_data: Columnar read data as returned by :class:`BamReader`.
        chrom: Chromosome name to record in the returned evidence.

    Returns:
        ``JunctionEvidence`` containing deduplicated junctions, their support
        counts, and strand assignments.
    """
    if read_data.n_reads == 0:
        return JunctionEvidence(
            chrom=chrom,
            starts=np.empty(0, dtype=np.int64),
            ends=np.empty(0, dtype=np.int64),
            counts=np.empty(0, dtype=np.int32),
            strands=np.empty(0, dtype=np.int8),
        )

    # Collect (start, end) -> list of strand values
    junction_strands: dict[tuple[int, int], list[int]] = {}

    cigar_ops = read_data.cigar_ops
    cigar_lens = read_data.cigar_lens
    cigar_offsets = read_data.cigar_offsets
    ref_starts = read_data.positions
    read_strands = read_data.strands

    for i in range(read_data.n_reads):
        pos = int(ref_starts[i])
        strand = int(read_strands[i])
        offset_start = int(cigar_offsets[i])
        offset_end = int(cigar_offsets[i + 1])

        for j in range(offset_start, offset_end):
            op = cigar_ops[j]
            length = int(cigar_lens[j])
            if op == CIGAR_N:
                key = (pos, pos + length)
                if key not in junction_strands:
                    junction_strands[key] = []
                junction_strands[key].append(strand)
                pos += length
            elif op in (0, 2, 7, 8):  # M, D, EQ, X — reference-consuming
                pos += length
            # I, S, H, P do not advance reference position

    if not junction_strands:
        return JunctionEvidence(
            chrom=chrom,
            starts=np.empty(0, dtype=np.int64),
            ends=np.empty(0, dtype=np.int64),
            counts=np.empty(0, dtype=np.int32),
            strands=np.empty(0, dtype=np.int8),
        )

    # Build sorted output arrays
    sorted_keys = sorted(junction_strands.keys())
    n_junctions = len(sorted_keys)
    starts = np.empty(n_junctions, dtype=np.int64)
    ends = np.empty(n_junctions, dtype=np.int64)
    counts = np.empty(n_junctions, dtype=np.int32)
    out_strands = np.empty(n_junctions, dtype=np.int8)

    for idx, (jstart, jend) in enumerate(sorted_keys):
        strand_list = junction_strands[(jstart, jend)]
        starts[idx] = jstart
        ends[idx] = jend
        counts[idx] = len(strand_list)

        # Majority-vote strand; mark ambiguous as -1 if tied.
        strand_counts = Counter(strand_list)
        if len(strand_counts) == 1:
            out_strands[idx] = next(iter(strand_counts))
        else:
            most_common = strand_counts.most_common(2)
            if most_common[0][1] > most_common[1][1]:
                out_strands[idx] = most_common[0][0]
            else:
                out_strands[idx] = -1  # ambiguous

    return JunctionEvidence(
        chrom=chrom,
        starts=starts,
        ends=ends,
        counts=counts,
        strands=out_strands,
    )


def extract_junctions_from_bam(
    bam_path: str,
    chrom: str,
    min_mapq: int = 0,
    min_anchor_length: int = 8,
    reference: object | None = None,
    return_stats: bool = False,
    region_start: int | None = None,
    region_end: int | None = None,
    strandedness: str = "none",
) -> tuple[JunctionEvidence, int] | tuple[JunctionEvidence, int, JunctionExtractionStats]:
    """Extract splice junctions directly from a BAM file without loading reads.

    This is a lightweight alternative to :func:`extract_junctions` that reads
    only the CIGAR strings and positions from the BAM, avoiding the memory
    cost of loading all read data.  Useful for large genomes where per-
    chromosome read data may not fit comfortably in memory.

    Args:
        bam_path: Path to the coordinate-sorted, indexed BAM file.
        chrom: Chromosome name to extract junctions from.
        min_mapq: Minimum mapping quality threshold.
        min_anchor_length: Minimum aligned bases flanking each splice
            junction.  Junctions with either anchor shorter than this
            are rejected as likely alignment artefacts.
        reference: Optional :class:`ReferenceGenome` instance.  When
            provided, junctions are validated against canonical splice
            motifs and strand is inferred from the motif when XS tag
            is missing.
        return_stats: Whether to return extraction statistics.
        region_start: Optional start coordinate to restrict extraction.
        region_end: Optional end coordinate to restrict extraction.
        strandedness: Library strand protocol: ``"none"`` (default),
            ``"rf"`` (fr-firststrand/dUTP), or ``"fr"`` (fr-secondstrand).
            Used to infer junction strand when XS tag is absent.

    Returns:
        Tuple of (``JunctionEvidence`` with deduplicated junctions,
        number of spliced reads encountered).
    """
    junction_strands: dict[tuple[int, int], list[int]] = {}
    raw_junction_keys: set[tuple[int, int]] = set()
    n_spliced = 0

    with pysam.AlignmentFile(bam_path, "rb") as af:
        for read in af.fetch(chrom, region_start, region_end):
            if read.is_unmapped or read.mapping_quality < min_mapq:
                continue
            flag = read.flag
            if flag & _DEFAULT_FILTER_FLAGS:
                continue

            cigar_tuples = read.cigartuples
            if cigar_tuples is None:
                continue

            pos = read.reference_start
            if read.has_tag("XS"):
                strand = 1 if read.get_tag("XS") == "-" else 0
            elif strandedness == "rf":
                # fr-firststrand (dUTP): read1 maps antisense
                is_read1 = not (flag & 0x80)  # not read2
                strand = 0 if (read.is_reverse == is_read1) else 1
            elif strandedness == "fr":
                # fr-secondstrand: read1 maps sense
                is_read1 = not (flag & 0x80)
                strand = 1 if (read.is_reverse == is_read1) else 0
            else:
                strand = 1 if read.is_reverse else 0

            has_splice = False
            # Track anchor lengths flanking each N operation.
            # Walk CIGAR once, recording aligned-base runs between introns.
            aligned_runs: list[int] = []  # aligned bases in current run
            junctions_in_read: list[tuple[int, int]] = []
            current_aligned = 0

            for op, length in cigar_tuples:
                if op == CIGAR_N:
                    # Record anchor before this intron
                    aligned_runs.append(current_aligned)
                    current_aligned = 0
                    has_splice = True
                    junctions_in_read.append((pos, pos + length))
                    pos += length
                elif op in (0, 7, 8):  # M, EQ, X -- reference-consuming aligned
                    current_aligned += length
                    pos += length
                elif op == 2:  # D -- reference-consuming but not aligned
                    pos += length
                # I, S, H, P do not advance reference position

            if has_splice:
                # Final anchor after last intron
                aligned_runs.append(current_aligned)
                n_spliced += 1

                # Filter junctions by anchor length.
                # aligned_runs[i] is the anchor before junction i,
                # aligned_runs[i+1] is the anchor after junction i.
                for ji, (j_start, j_end) in enumerate(junctions_in_read):
                    raw_junction_keys.add((j_start, j_end))
                    left_anchor = aligned_runs[ji]
                    right_anchor = aligned_runs[ji + 1]
                    if left_anchor < min_anchor_length or right_anchor < min_anchor_length:
                        continue
                    key = (j_start, j_end)
                    if key not in junction_strands:
                        junction_strands[key] = []
                    junction_strands[key].append(strand)

    extraction_stats = JunctionExtractionStats(
        chrom=chrom,
        raw_junctions=len(raw_junction_keys),
        anchor_filtered_junctions=len(junction_strands),
        motif_filtered_junctions=len(junction_strands),
        output_junctions=len(junction_strands),
    )

    if not junction_strands:
        evidence = JunctionEvidence(
            chrom=chrom,
            starts=np.empty(0, dtype=np.int64),
            ends=np.empty(0, dtype=np.int64),
            counts=np.empty(0, dtype=np.int32),
            strands=np.empty(0, dtype=np.int8),
        )
        if return_stats:
            return evidence, n_spliced, extraction_stats
        return evidence, n_spliced

    sorted_keys = sorted(junction_strands.keys())
    n_junctions = len(sorted_keys)
    starts = np.empty(n_junctions, dtype=np.int64)
    ends = np.empty(n_junctions, dtype=np.int64)
    counts = np.empty(n_junctions, dtype=np.int32)
    out_strands = np.empty(n_junctions, dtype=np.int8)

    for idx, (jstart, jend) in enumerate(sorted_keys):
        strand_list = junction_strands[(jstart, jend)]
        starts[idx] = jstart
        ends[idx] = jend
        counts[idx] = len(strand_list)
        strand_counts = Counter(strand_list)
        if len(strand_counts) == 1:
            out_strands[idx] = next(iter(strand_counts))
        else:
            most_common = strand_counts.most_common(2)
            if most_common[0][1] > most_common[1][1]:
                out_strands[idx] = most_common[0][0]
            else:
                out_strands[idx] = -1

    # Apply motif validation when reference is provided.
    if reference is not None and n_junctions > 0:
        valid_mask = reference.validate_junctions(chrom, starts, ends)
        n_valid = int(np.sum(valid_mask))
        if n_valid < n_junctions:
            logger.info(
                "Anchor+motif filter: %d/%d junctions on %s",
                n_valid, n_junctions, chrom,
            )
            starts = starts[valid_mask]
            ends = ends[valid_mask]
            counts = counts[valid_mask]
            out_strands = out_strands[valid_mask]

        # Infer strand from motif for ALL junctions with ambiguous strand
        # (not just when some were filtered — fixes logic hole where
        # all-canonical junctions never got strand inference).
        _strand_str_to_int = {"+": 0, "-": 1, ".": -1}
        if hasattr(reference, "infer_strand_from_motif"):
            for idx in range(len(starts)):
                if out_strands[idx] == -1:
                    inferred = reference.infer_strand_from_motif(
                        chrom, int(starts[idx]), int(ends[idx]),
                    )
                    if inferred is not None and inferred != ".":
                        out_strands[idx] = _strand_str_to_int.get(
                            inferred, -1,
                        )
        extraction_stats.motif_filtered_junctions = int(len(starts))

    extraction_stats.output_junctions = int(len(starts))

    evidence = JunctionEvidence(
        chrom=chrom,
        starts=starts,
        ends=ends,
        counts=counts,
        strands=out_strands,
    )
    if return_stats:
        return evidence, n_spliced, extraction_stats
    return evidence, n_spliced
