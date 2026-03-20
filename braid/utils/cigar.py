"""CIGAR string parsing and manipulation utilities.

Provides efficient extraction of alignment features from CIGAR operations,
including junction detection, exon blocks, and coverage intervals.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit, prange

# CIGAR operation codes (BAM encoding)
CIGAR_M = 0   # alignment match (can be sequence match or mismatch)
CIGAR_I = 1   # insertion to the reference
CIGAR_D = 2   # deletion from the reference
CIGAR_N = 3   # skipped region from the reference (intron in RNA-seq)
CIGAR_S = 4   # soft clipping
CIGAR_H = 5   # hard clipping
CIGAR_P = 6   # padding
CIGAR_EQ = 7  # sequence match
CIGAR_X = 8   # sequence mismatch

# Operations that consume reference bases
_REF_CONSUMING = {CIGAR_M, CIGAR_D, CIGAR_N, CIGAR_EQ, CIGAR_X}

# Operations that contribute to alignment (match/mismatch on reference)
_ALIGN_OPS = {CIGAR_M, CIGAR_EQ, CIGAR_X}


class Junction(NamedTuple):
    """A splice junction defined by intron start and end positions."""

    start: int  # 0-based, first base of intron
    end: int    # 0-based, exclusive end of intron


class ExonBlock(NamedTuple):
    """A contiguous aligned block on the reference."""

    start: int  # 0-based inclusive
    end: int    # 0-based exclusive


def extract_junctions(cigar_tuples: list[tuple[int, int]], ref_start: int) -> list[Junction]:
    """Extract splice junctions from a CIGAR alignment.

    Args:
        cigar_tuples: List of (operation, length) pairs from pysam.
        ref_start: 0-based reference start position of the alignment.

    Returns:
        List of Junction objects (intron coordinates).
    """
    junctions: list[Junction] = []
    pos = ref_start
    for op, length in cigar_tuples:
        if op == CIGAR_N:
            junctions.append(Junction(start=pos, end=pos + length))
            pos += length
        elif op in _REF_CONSUMING:
            pos += length
    return junctions


def extract_exon_blocks(cigar_tuples: list[tuple[int, int]], ref_start: int) -> list[ExonBlock]:
    """Extract exon alignment blocks from a CIGAR alignment.

    Args:
        cigar_tuples: List of (operation, length) pairs.
        ref_start: 0-based reference start position.

    Returns:
        List of ExonBlock objects (contiguous aligned regions).
    """
    blocks: list[ExonBlock] = []
    pos = ref_start
    block_start: int | None = None

    for op, length in cigar_tuples:
        if op in _ALIGN_OPS:
            if block_start is None:
                block_start = pos
            pos += length
        elif op == CIGAR_D:
            if block_start is None:
                block_start = pos
            pos += length
        elif op == CIGAR_N:
            if block_start is not None:
                blocks.append(ExonBlock(start=block_start, end=pos))
                block_start = None
            pos += length
        # I, S, H, P do not consume reference
    if block_start is not None:
        blocks.append(ExonBlock(start=block_start, end=pos))
    return blocks


@njit(parallel=True, cache=True)
def batch_extract_junctions(
    cigar_ops: np.ndarray,
    cigar_lens: np.ndarray,
    cigar_offsets: np.ndarray,
    ref_starts: np.ndarray,
    junction_starts: np.ndarray,
    junction_ends: np.ndarray,
    junction_counts: np.ndarray,
) -> None:
    """Extract junctions from multiple reads in parallel using Numba.

    All arrays are pre-allocated. For each read i, junctions are written
    starting at junction_offsets[i].

    Args:
        cigar_ops: Flat array of CIGAR operations for all reads.
        cigar_lens: Flat array of CIGAR lengths for all reads.
        cigar_offsets: Start index of each read's CIGAR in the flat arrays (size n_reads+1).
        ref_starts: Reference start positions per read.
        junction_starts: Output array for junction start positions (pre-allocated).
        junction_ends: Output array for junction end positions (pre-allocated).
        junction_counts: Output array for number of junctions per read.
    """
    n_reads = len(ref_starts)
    for i in prange(n_reads):
        pos = ref_starts[i]
        junc_idx = cigar_offsets[i]  # Write position in output
        count = 0
        for j in range(cigar_offsets[i], cigar_offsets[i + 1]):
            op = cigar_ops[j]
            length = cigar_lens[j]
            if op == 3:  # CIGAR_N
                junction_starts[junc_idx + count] = pos
                junction_ends[junc_idx + count] = pos + length
                count += 1
                pos += length
            elif op in (0, 2, 7, 8):  # M, D, EQ, X
                pos += length
        junction_counts[i] = count


@njit(cache=True)
def cigar_reference_length(cigar_ops: np.ndarray, cigar_lens: np.ndarray) -> int:
    """Compute the reference length consumed by a CIGAR string.

    Args:
        cigar_ops: Array of CIGAR operation codes.
        cigar_lens: Array of CIGAR operation lengths.

    Returns:
        Total reference bases consumed.
    """
    total = 0
    for i in range(len(cigar_ops)):
        op = cigar_ops[i]
        if op in (0, 2, 3, 7, 8):  # M, D, N, EQ, X
            total += cigar_lens[i]
    return total
