"""Interval operations for genomic coordinate manipulation.

Provides efficient merging, intersection, and overlap detection for
genomic intervals used in splice graph construction.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numba import njit


class Interval(NamedTuple):
    """A genomic interval [start, end) with 0-based half-open coordinates."""

    start: int
    end: int
    data: int = 0  # Generic integer payload (e.g., read index)


@njit(cache=True)
def merge_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge overlapping intervals.

    Input intervals must be sorted by start position.

    Args:
        starts: Sorted array of interval start positions.
        ends: Array of interval end positions.

    Returns:
        Tuple of (merged_starts, merged_ends).
    """
    n = len(starts)
    if n == 0:
        return np.empty(0, dtype=starts.dtype), np.empty(0, dtype=ends.dtype)

    merged_s = np.empty(n, dtype=starts.dtype)
    merged_e = np.empty(n, dtype=ends.dtype)
    count = 0
    cur_s = starts[0]
    cur_e = ends[0]

    for i in range(1, n):
        if starts[i] <= cur_e:
            cur_e = max(cur_e, ends[i])
        else:
            merged_s[count] = cur_s
            merged_e[count] = cur_e
            count += 1
            cur_s = starts[i]
            cur_e = ends[i]

    merged_s[count] = cur_s
    merged_e[count] = cur_e
    count += 1
    return merged_s[:count], merged_e[:count]


@njit(cache=True)
def intersect_sorted_intervals(
    starts_a: np.ndarray, ends_a: np.ndarray,
    starts_b: np.ndarray, ends_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute intersection of two sorted interval sets.

    Args:
        starts_a, ends_a: First sorted interval set.
        starts_b, ends_b: Second sorted interval set.

    Returns:
        Tuple of (intersection_starts, intersection_ends).
    """
    na, nb = len(starts_a), len(starts_b)
    max_out = na + nb
    out_s = np.empty(max_out, dtype=starts_a.dtype)
    out_e = np.empty(max_out, dtype=ends_a.dtype)
    count = 0
    i, j = 0, 0

    while i < na and j < nb:
        s = max(starts_a[i], starts_b[j])
        e = min(ends_a[i], ends_b[j])
        if s < e:
            out_s[count] = s
            out_e[count] = e
            count += 1
        if ends_a[i] < ends_b[j]:
            i += 1
        else:
            j += 1

    return out_s[:count], out_e[:count]


@njit(cache=True)
def find_overlapping(
    query_start: int, query_end: int,
    target_starts: np.ndarray, target_ends: np.ndarray,
) -> np.ndarray:
    """Find indices of target intervals overlapping with a query interval.

    Targets must be sorted by start position. Uses binary search for efficiency.

    Args:
        query_start: Query interval start.
        query_end: Query interval end.
        target_starts: Sorted array of target start positions.
        target_ends: Array of target end positions.

    Returns:
        Array of indices into target arrays that overlap the query.
    """
    n = len(target_starts)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    # Binary search for first target that could overlap
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if target_ends[mid] <= query_start:
            lo = mid + 1
        else:
            hi = mid

    result = np.empty(n, dtype=np.int64)
    count = 0
    for i in range(lo, n):
        if target_starts[i] >= query_end:
            break
        if target_ends[i] > query_start:
            result[count] = i
            count += 1

    return result[:count]


@njit(cache=True)
def _compute_coverage_fallback(
    starts: np.ndarray, ends: np.ndarray,
    region_start: int, region_end: int,
) -> np.ndarray:
    """Pure Python fallback for coverage computation.

    Args:
        starts: Array of interval start positions.
        ends: Array of interval end positions.
        region_start: Start of the region to compute coverage for.
        region_end: End of the region.

    Returns:
        1D array of per-base coverage values for [region_start, region_end).
    """
    length = region_end - region_start
    cov = np.zeros(length, dtype=np.int32)
    for i in range(len(starts)):
        s = max(int(starts[i]) - region_start, 0)
        e = min(int(ends[i]) - region_start, length)
        if s < e:
            for j in range(s, e):
                cov[j] += 1
    return cov


# Try to import Numba kernel at module level.
try:
    from rapidsplice.cuda.kernels import parallel_coverage_scan as _numba_coverage_scan
    _HAS_NUMBA_COVERAGE = True
except (ImportError, Exception):
    _HAS_NUMBA_COVERAGE = False
    _numba_coverage_scan = None


def compute_coverage(
    starts: np.ndarray, ends: np.ndarray,
    region_start: int, region_end: int,
) -> np.ndarray:
    """Compute per-base coverage for a genomic region.

    Uses the Numba JIT ``parallel_coverage_scan`` kernel when available
    for significantly faster computation on large regions.

    Args:
        starts: Array of interval start positions.
        ends: Array of interval end positions.
        region_start: Start of the region to compute coverage for.
        region_end: End of the region.

    Returns:
        1D array of per-base coverage values for [region_start, region_end).
    """
    if _HAS_NUMBA_COVERAGE:
        try:
            return _numba_coverage_scan(
                np.asarray(starts, dtype=np.int64),
                np.asarray(ends, dtype=np.int64),
                region_start,
                region_end,
            ).astype(np.int32)
        except Exception:
            pass
    return _compute_coverage_fallback(starts, ends, region_start, region_end)
