"""GPU-accelerated k-mer extraction and counting.

Provides 2-bit DNA encoding, k-mer extraction from sequences, canonical
k-mer computation (lexicographically smaller of forward/reverse-complement),
and sort-based k-mer counting suitable for GPU execution.

The 2-bit encoding maps: A=0, C=1, G=2, T=3.  K-mers are packed into
64-bit integers, supporting k up to 31.

Performance-critical inner loops use Numba JIT compilation when available,
with automatic fallback to pure NumPy/Python.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op decorator when Numba is unavailable."""
        def decorator(func):  # type: ignore[no-untyped-def]
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_K = 31  # Maximum k-mer size (fits in 64-bit integer with 2-bit encoding)

# 2-bit encoding: A=0, C=1, G=2, T=3
_BASE_TO_BITS = np.zeros(256, dtype=np.uint8)
_BASE_TO_BITS[ord("A")] = 0
_BASE_TO_BITS[ord("C")] = 1
_BASE_TO_BITS[ord("G")] = 2
_BASE_TO_BITS[ord("T")] = 3
_BASE_TO_BITS[ord("a")] = 0
_BASE_TO_BITS[ord("c")] = 1
_BASE_TO_BITS[ord("g")] = 2
_BASE_TO_BITS[ord("t")] = 3

# Valid bases mask
_VALID_BASE = np.zeros(256, dtype=np.bool_)
for _b in b"ACGTacgt":
    _VALID_BASE[_b] = True

_BITS_TO_BASE = np.array([ord("A"), ord("C"), ord("G"), ord("T")],
                         dtype=np.uint8)

# Complement: A<->T (0<->3), C<->G (1<->2)
_COMPLEMENT_BITS = np.array([3, 2, 1, 0], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class KmerCountTable:
    """Result of k-mer counting.

    Attributes:
        kmers: Sorted array of unique canonical k-mer encodings.
        counts: Corresponding count for each k-mer.
        k: K-mer size used.
    """

    kmers: np.ndarray   # uint64
    counts: np.ndarray  # uint32
    k: int


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


@njit(cache=True)
def _encode_sequence_numba(
    bits: np.ndarray,
    valid: np.ndarray,
    n: int,
    k: int,
    mask: np.uint64,
    result: np.ndarray,
) -> int:
    """Numba-JIT inner loop for k-mer extraction.

    Args:
        bits: 2-bit encoded base values (uint64).
        valid: Boolean mask of valid bases.
        n: Sequence length.
        k: K-mer size.
        mask: Bitmask for k bases.
        result: Pre-allocated output array.

    Returns:
        Number of k-mers extracted.
    """
    count = 0
    current = np.uint64(0)
    valid_run = 0
    for i in range(n):
        if valid[i]:
            current = ((current << np.uint64(2)) | bits[i]) & mask
            valid_run += 1
        else:
            valid_run = 0
            current = np.uint64(0)
        if valid_run >= k:
            result[count] = current
            count += 1
    return count


@njit(cache=True)
def _reverse_complement_kmers_numba(
    kmers: np.ndarray,
    k: int,
    result: np.ndarray,
) -> None:
    """Numba-JIT reverse complement for a batch of k-mers.

    Args:
        kmers: Input k-mer encodings.
        k: K-mer size.
        result: Pre-allocated output array.
    """
    for i in range(len(kmers)):
        rc = np.uint64(0)
        val = kmers[i]
        for _ in range(k):
            base = val & np.uint64(3)
            complement = np.uint64(3) ^ base
            rc = (rc << np.uint64(2)) | complement
            val >>= np.uint64(2)
        result[i] = rc


def encode_sequence(seq: bytes | str, k: int) -> np.ndarray:
    """Extract all k-mer encodings from a DNA sequence.

    Encodes each k-mer as a 64-bit integer using 2-bit encoding.  K-mers
    containing non-ACGT characters are skipped.  Uses Numba JIT for the
    inner loop when available.

    Args:
        seq: DNA sequence as bytes or string.
        k: K-mer size (1 to 31).

    Returns:
        Array of uint64 k-mer encodings.  Length is at most ``len(seq) - k + 1``.
    """
    if k < 1 or k > MAX_K:
        raise ValueError(f"k must be 1..{MAX_K}, got {k}")

    if isinstance(seq, str):
        seq = seq.encode("ascii")

    n = len(seq)
    if n < k:
        return np.empty(0, dtype=np.uint64)

    seq_arr = np.frombuffer(seq, dtype=np.uint8)
    valid = _VALID_BASE[seq_arr]
    bits = _BASE_TO_BITS[seq_arr].astype(np.uint64)

    mask = np.uint64((1 << (2 * k)) - 1)
    max_kmers = n - k + 1
    result = np.empty(max_kmers, dtype=np.uint64)

    count = _encode_sequence_numba(bits, valid, n, k, mask, result)

    return result[:count]


def reverse_complement_kmer(kmer: np.uint64, k: int) -> np.uint64:
    """Compute the reverse complement of a single k-mer encoding.

    Args:
        kmer: 2-bit encoded k-mer.
        k: K-mer size.

    Returns:
        2-bit encoded reverse complement.
    """
    rc = np.uint64(0)
    val = np.uint64(kmer)
    for _ in range(k):
        base = val & np.uint64(3)
        complement = np.uint64(3) ^ base  # A<->T, C<->G
        rc = (rc << np.uint64(2)) | complement
        val >>= np.uint64(2)
    return rc


def reverse_complement_kmers(kmers: np.ndarray, k: int) -> np.ndarray:
    """Compute reverse complements for an array of k-mers.

    Uses Numba JIT for the inner loop when available.

    Args:
        kmers: Array of uint64 k-mer encodings.
        k: K-mer size.

    Returns:
        Array of uint64 reverse complement encodings.
    """
    if len(kmers) == 0:
        return np.empty(0, dtype=np.uint64)

    result = np.zeros(len(kmers), dtype=np.uint64)
    _reverse_complement_kmers_numba(kmers, k, result)
    return result


def canonicalize_kmers(kmers: np.ndarray, k: int) -> np.ndarray:
    """Convert k-mers to canonical form (min of forward and reverse complement).

    Args:
        kmers: Array of uint64 k-mer encodings.
        k: K-mer size.

    Returns:
        Array of canonical k-mer encodings.
    """
    rc = reverse_complement_kmers(kmers, k)
    return np.minimum(kmers, rc)


def decode_kmer(kmer: np.uint64, k: int) -> str:
    """Decode a 2-bit encoded k-mer back to a DNA string.

    Args:
        kmer: 2-bit encoded k-mer.
        k: K-mer size.

    Returns:
        DNA string of length k.
    """
    bases = []
    val = int(kmer)
    for _ in range(k):
        bases.append("ACGT"[val & 3])
        val >>= 2
    return "".join(reversed(bases))


def count_kmers(
    sequences: list[bytes | str],
    k: int,
    min_count: int = 2,
    canonical: bool = True,
) -> KmerCountTable:
    """Count k-mers across multiple sequences.

    Extracts all k-mers from the given sequences, optionally canonicalizes
    them, and counts occurrences using sort-and-reduce.  This algorithm is
    naturally parallelizable on GPU (radix sort + segmented reduce).

    Args:
        sequences: List of DNA sequences (bytes or str).
        k: K-mer size (1 to 31).
        min_count: Minimum count to retain a k-mer.
        canonical: If True, use canonical (strand-independent) k-mers.

    Returns:
        KmerCountTable with sorted unique k-mers and their counts.
    """
    if k < 1 or k > MAX_K:
        raise ValueError(f"k must be 1..{MAX_K}, got {k}")

    # Extract all k-mers from all sequences
    all_kmers: list[np.ndarray] = []
    for seq in sequences:
        kmers = encode_sequence(seq, k)
        if len(kmers) > 0:
            all_kmers.append(kmers)

    if not all_kmers:
        return KmerCountTable(
            kmers=np.empty(0, dtype=np.uint64),
            counts=np.empty(0, dtype=np.uint32),
            k=k,
        )

    merged = np.concatenate(all_kmers)

    if canonical:
        merged = canonicalize_kmers(merged, k)

    # Sort-and-count: sort k-mers, then run-length encode
    merged.sort()

    # Find boundaries between distinct k-mers
    if len(merged) == 0:
        return KmerCountTable(
            kmers=np.empty(0, dtype=np.uint64),
            counts=np.empty(0, dtype=np.uint32),
            k=k,
        )

    changes = np.concatenate([
        [True],
        merged[1:] != merged[:-1],
    ])
    unique_kmers = merged[changes]
    change_indices = np.where(changes)[0]
    counts = np.diff(np.concatenate([change_indices, [len(merged)]]))
    counts = counts.astype(np.uint32)

    # Filter by minimum count
    if min_count > 1:
        mask = counts >= min_count
        unique_kmers = unique_kmers[mask]
        counts = counts[mask]

    logger.info(
        "Counted %d unique k-mers (k=%d) from %d total, %d above min_count=%d",
        len(unique_kmers), k, len(merged), len(unique_kmers), min_count,
    )

    return KmerCountTable(kmers=unique_kmers, counts=counts, k=k)


def extract_prefix_suffix(kmer: np.uint64, k: int) -> tuple[np.uint64, np.uint64]:
    """Extract the (k-1)-mer prefix and suffix of a k-mer.

    In the de Bruijn graph, nodes are (k-1)-mers and edges are k-mers.
    The prefix is the first (k-1) bases, the suffix is the last (k-1) bases.

    Args:
        kmer: 2-bit encoded k-mer.
        k: K-mer size.

    Returns:
        Tuple of (prefix, suffix) as uint64 (k-1)-mer encodings.
    """
    mask = np.uint64((1 << (2 * (k - 1))) - 1)
    suffix = kmer & mask
    prefix = (kmer >> np.uint64(2)) & mask
    return prefix, suffix


def extract_prefixes_suffixes(
    kmers: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract prefix and suffix (k-1)-mers for an array of k-mers.

    Args:
        kmers: Array of uint64 k-mer encodings.
        k: K-mer size.

    Returns:
        Tuple of (prefixes, suffixes) arrays.
    """
    mask = np.uint64((1 << (2 * (k - 1))) - 1)
    suffixes = kmers & mask
    prefixes = (kmers >> np.uint64(2)) & mask
    return prefixes, suffixes
