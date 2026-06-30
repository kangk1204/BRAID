"""Tests for utility modules: cigar, interval, and stats.

Exercises CIGAR parsing, interval operations, coverage computation,
assembly statistics, and the Timer context manager.
"""

from __future__ import annotations

import numpy as np

from braid.utils.cigar import (
    CIGAR_D,
    CIGAR_I,
    CIGAR_M,
    CIGAR_N,
    CIGAR_S,
    ExonBlock,
    Junction,
    batch_extract_junctions,
    cigar_reference_length,
    extract_exon_blocks,
    extract_junctions,
)
from braid.utils.interval import (
    compute_coverage,
    find_overlapping,
    intersect_sorted_intervals,
    merge_intervals,
)
from braid.utils.stats import AssemblyStats, Timer

# ===================================================================
# CIGAR utility tests
# ===================================================================


class TestExtractJunctionsFromCigar:
    """Test Junction extraction from CIGAR tuples."""

    def test_single_junction(self) -> None:
        """CIGAR 100M 100N 100M starting at position 0 produces one junction."""
        cigar_tuples = [(CIGAR_M, 100), (CIGAR_N, 100), (CIGAR_M, 100)]
        junctions = extract_junctions(cigar_tuples, ref_start=0)
        assert len(junctions) == 1
        assert junctions[0] == Junction(start=100, end=200)

    def test_two_junctions(self) -> None:
        """CIGAR with two N operations produces two junctions."""
        cigar_tuples = [
            (CIGAR_M, 50),
            (CIGAR_N, 100),
            (CIGAR_M, 30),
            (CIGAR_N, 200),
            (CIGAR_M, 50),
        ]
        junctions = extract_junctions(cigar_tuples, ref_start=10)
        assert len(junctions) == 2
        assert junctions[0] == Junction(start=60, end=160)
        assert junctions[1] == Junction(start=190, end=390)

    def test_no_junction(self) -> None:
        """CIGAR with no N operations produces no junctions."""
        cigar_tuples = [(CIGAR_M, 150)]
        junctions = extract_junctions(cigar_tuples, ref_start=100)
        assert len(junctions) == 0

    def test_with_non_ref_ops(self) -> None:
        """Insertions and soft clips do not affect junction positions."""
        cigar_tuples = [
            (CIGAR_S, 5),
            (CIGAR_M, 50),
            (CIGAR_I, 10),
            (CIGAR_N, 200),
            (CIGAR_M, 50),
        ]
        junctions = extract_junctions(cigar_tuples, ref_start=0)
        assert len(junctions) == 1
        # S does not consume reference; I does not consume reference
        # pos after S=0, after M50 -> 50, after I10 -> still 50, then N200 -> junction (50, 250)
        assert junctions[0] == Junction(start=50, end=250)

    def test_with_deletion(self) -> None:
        """Deletions consume reference and shift junction positions."""
        cigar_tuples = [
            (CIGAR_M, 50),
            (CIGAR_D, 10),
            (CIGAR_M, 40),
            (CIGAR_N, 100),
            (CIGAR_M, 50),
        ]
        junctions = extract_junctions(cigar_tuples, ref_start=0)
        assert len(junctions) == 1
        # pos: 0 + M50=50, + D10=60, + M40=100, then N100 -> junction (100, 200)
        assert junctions[0] == Junction(start=100, end=200)


class TestExtractExonBlocks:
    """Test ExonBlock extraction from CIGAR tuples."""

    def test_single_block(self) -> None:
        """A simple match produces one exon block."""
        cigar_tuples = [(CIGAR_M, 100)]
        blocks = extract_exon_blocks(cigar_tuples, ref_start=50)
        assert len(blocks) == 1
        assert blocks[0] == ExonBlock(start=50, end=150)

    def test_split_by_intron(self) -> None:
        """An intron (N) splits the alignment into two exon blocks."""
        cigar_tuples = [(CIGAR_M, 100), (CIGAR_N, 50), (CIGAR_M, 80)]
        blocks = extract_exon_blocks(cigar_tuples, ref_start=0)
        assert len(blocks) == 2
        assert blocks[0] == ExonBlock(start=0, end=100)
        assert blocks[1] == ExonBlock(start=150, end=230)

    def test_deletion_does_not_split(self) -> None:
        """A deletion (D) within an exon block does not split it."""
        cigar_tuples = [(CIGAR_M, 50), (CIGAR_D, 5), (CIGAR_M, 50)]
        blocks = extract_exon_blocks(cigar_tuples, ref_start=0)
        assert len(blocks) == 1
        assert blocks[0] == ExonBlock(start=0, end=105)

    def test_empty_cigar(self) -> None:
        """An empty CIGAR produces no blocks."""
        blocks = extract_exon_blocks([], ref_start=0)
        assert len(blocks) == 0


class TestCigarReferenceLength:
    """Test reference length calculation from CIGAR operations."""

    def test_simple_match(self) -> None:
        """A single M consumes reference bases equal to its length."""
        ops = np.array([CIGAR_M], dtype=np.uint8)
        lens = np.array([150], dtype=np.int32)
        assert cigar_reference_length(ops, lens) == 150

    def test_match_with_intron(self) -> None:
        """M + N + M: all three consume reference."""
        ops = np.array([CIGAR_M, CIGAR_N, CIGAR_M], dtype=np.uint8)
        lens = np.array([100, 200, 100], dtype=np.int32)
        assert cigar_reference_length(ops, lens) == 400

    def test_insertion_not_counted(self) -> None:
        """Insertions (I) do not consume reference bases."""
        ops = np.array([CIGAR_M, CIGAR_I, CIGAR_M], dtype=np.uint8)
        lens = np.array([50, 10, 50], dtype=np.int32)
        assert cigar_reference_length(ops, lens) == 100

    def test_deletion_counted(self) -> None:
        """Deletions (D) consume reference bases."""
        ops = np.array([CIGAR_M, CIGAR_D, CIGAR_M], dtype=np.uint8)
        lens = np.array([50, 10, 50], dtype=np.int32)
        assert cigar_reference_length(ops, lens) == 110


class TestBatchExtractJunctions:
    """Test the numba batch junction extraction."""

    def test_batch_two_reads(self) -> None:
        """Extract junctions from two reads in parallel.

        Read 0: 100M 100N 50M (1 junction)
        Read 1: 200M (0 junctions)
        """
        cigar_ops = np.array([CIGAR_M, CIGAR_N, CIGAR_M, CIGAR_M], dtype=np.uint8)
        cigar_lens = np.array([100, 100, 50, 200], dtype=np.int32)
        cigar_offsets = np.array([0, 3, 4], dtype=np.int64)
        ref_starts = np.array([0, 500], dtype=np.int64)

        # Pre-allocate output arrays large enough (use cigar_offsets total length)
        max_junctions = len(cigar_ops)
        junction_starts = np.zeros(max_junctions, dtype=np.int64)
        junction_ends = np.zeros(max_junctions, dtype=np.int64)
        junction_counts = np.zeros(2, dtype=np.int64)

        batch_extract_junctions(
            cigar_ops, cigar_lens, cigar_offsets, ref_starts,
            junction_starts, junction_ends, junction_counts,
        )

        assert junction_counts[0] == 1
        assert junction_counts[1] == 0
        assert junction_starts[0] == 100
        assert junction_ends[0] == 200

    def test_batch_single_read_two_junctions(self) -> None:
        """Single read with two junctions."""
        cigar_ops = np.array(
            [CIGAR_M, CIGAR_N, CIGAR_M, CIGAR_N, CIGAR_M], dtype=np.uint8,
        )
        cigar_lens = np.array([50, 100, 30, 200, 50], dtype=np.int32)
        cigar_offsets = np.array([0, 5], dtype=np.int64)
        ref_starts = np.array([10], dtype=np.int64)

        max_junctions = len(cigar_ops)
        junction_starts = np.zeros(max_junctions, dtype=np.int64)
        junction_ends = np.zeros(max_junctions, dtype=np.int64)
        junction_counts = np.zeros(1, dtype=np.int64)

        batch_extract_junctions(
            cigar_ops, cigar_lens, cigar_offsets, ref_starts,
            junction_starts, junction_ends, junction_counts,
        )

        assert junction_counts[0] == 2
        # Junction 1: pos=10+50=60, junction (60, 160)
        assert junction_starts[0] == 60
        assert junction_ends[0] == 160
        # Junction 2: pos=160+30=190, junction (190, 390)
        assert junction_starts[1] == 190
        assert junction_ends[1] == 390


# ===================================================================
# Interval utility tests
# ===================================================================


class TestMergeIntervals:
    """Test interval merging with the numba-compiled merge_intervals."""

    def test_non_overlapping(self) -> None:
        """Non-overlapping intervals remain separate."""
        starts = np.array([10, 30, 50], dtype=np.int64)
        ends = np.array([20, 40, 60], dtype=np.int64)
        ms, me = merge_intervals(starts, ends)
        assert len(ms) == 3
        np.testing.assert_array_equal(ms, [10, 30, 50])
        np.testing.assert_array_equal(me, [20, 40, 60])

    def test_overlapping(self) -> None:
        """Overlapping intervals are merged into one."""
        starts = np.array([10, 15, 30], dtype=np.int64)
        ends = np.array([25, 35, 50], dtype=np.int64)
        ms, me = merge_intervals(starts, ends)
        assert len(ms) == 1
        np.testing.assert_array_equal(ms, [10])
        np.testing.assert_array_equal(me, [50])

    def test_abutting(self) -> None:
        """Abutting intervals (end == next start) are merged."""
        starts = np.array([10, 20], dtype=np.int64)
        ends = np.array([20, 30], dtype=np.int64)
        ms, me = merge_intervals(starts, ends)
        assert len(ms) == 1
        np.testing.assert_array_equal(ms, [10])
        np.testing.assert_array_equal(me, [30])

    def test_empty(self) -> None:
        """Empty input produces empty output."""
        starts = np.array([], dtype=np.int64)
        ends = np.array([], dtype=np.int64)
        ms, me = merge_intervals(starts, ends)
        assert len(ms) == 0


class TestIntersectIntervals:
    """Test interval intersection."""

    def test_overlapping_sets(self) -> None:
        """Two overlapping interval sets produce correct intersections."""
        starts_a = np.array([10, 30], dtype=np.int64)
        ends_a = np.array([25, 50], dtype=np.int64)
        starts_b = np.array([20, 40], dtype=np.int64)
        ends_b = np.array([35, 60], dtype=np.int64)

        is_, ie = intersect_sorted_intervals(starts_a, ends_a, starts_b, ends_b)
        # Intersection 1: max(10,20)=20, min(25,35)=25 -> [20, 25)
        # Intersection 2: max(30,20)=30, min(50,35)=35 -> [30, 35)
        # Intersection 3: max(30,40)=40, min(50,60)=50 -> [40, 50)
        assert len(is_) == 3
        np.testing.assert_array_equal(is_, [20, 30, 40])
        np.testing.assert_array_equal(ie, [25, 35, 50])

    def test_no_overlap(self) -> None:
        """Non-overlapping interval sets produce empty intersection."""
        starts_a = np.array([10], dtype=np.int64)
        ends_a = np.array([20], dtype=np.int64)
        starts_b = np.array([30], dtype=np.int64)
        ends_b = np.array([40], dtype=np.int64)

        is_, ie = intersect_sorted_intervals(starts_a, ends_a, starts_b, ends_b)
        assert len(is_) == 0


class TestFindOverlapping:
    """Test overlap queries."""

    def test_basic_overlap(self) -> None:
        """Query overlaps two out of three targets."""
        target_starts = np.array([10, 30, 50], dtype=np.int64)
        target_ends = np.array([25, 45, 65], dtype=np.int64)

        result = find_overlapping(20, 40, target_starts, target_ends)
        # [10,25) overlaps [20,40): yes (25>20 and 10<40)
        # [30,45) overlaps [20,40): yes
        # [50,65) overlaps [20,40): no (50>=40)
        assert len(result) == 2
        assert 0 in result
        assert 1 in result

    def test_no_overlap(self) -> None:
        """Query that overlaps nothing returns empty array."""
        target_starts = np.array([100, 200], dtype=np.int64)
        target_ends = np.array([150, 250], dtype=np.int64)
        result = find_overlapping(0, 50, target_starts, target_ends)
        assert len(result) == 0

    def test_empty_targets(self) -> None:
        """Empty target array returns empty result."""
        result = find_overlapping(
            10, 20,
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
        assert len(result) == 0


class TestComputeCoverage:
    """Test per-base coverage computation."""

    def test_single_interval(self) -> None:
        """A single interval covering part of the region."""
        starts = np.array([10], dtype=np.int64)
        ends = np.array([20], dtype=np.int64)
        cov = compute_coverage(starts, ends, region_start=0, region_end=30)
        assert cov.shape == (30,)
        # Bases 0-9: 0 coverage, bases 10-19: 1, bases 20-29: 0
        assert np.sum(cov[0:10]) == 0
        assert np.all(cov[10:20] == 1)
        assert np.sum(cov[20:30]) == 0

    def test_two_overlapping(self) -> None:
        """Two overlapping intervals produce coverage of 2 in their overlap."""
        starts = np.array([10, 15], dtype=np.int64)
        ends = np.array([25, 30], dtype=np.int64)
        cov = compute_coverage(starts, ends, region_start=10, region_end=30)
        assert cov.shape == (20,)
        # [10, 15): coverage 1; [15, 25): coverage 2; [25, 30): coverage 1
        assert np.all(cov[0:5] == 1)    # positions 10-14
        assert np.all(cov[5:15] == 2)   # positions 15-24
        assert np.all(cov[15:20] == 1)  # positions 25-29


# ===================================================================
# Stats utility tests
# ===================================================================


class TestAssemblyStats:
    """Test stats summary formatting."""

    def test_summary_format(self) -> None:
        """Summary string contains key fields."""
        stats = AssemblyStats(
            total_reads=100_000,
            mapped_reads=95_000,
            spliced_reads=50_000,
            total_junctions=1_000,
            unique_junctions=500,
            total_loci=20,
            assembled_transcripts=50,
            multi_exon_transcripts=40,
            single_exon_transcripts=10,
            filtered_transcripts=5,
            elapsed_seconds=12.5,
        )
        summary = stats.summary()
        assert "100,000" in summary
        assert "95,000" in summary
        assert "BRAID" in summary
        assert "12.5" in summary

    def test_per_stage_times(self) -> None:
        """Per-stage timing appears in summary when populated."""
        stats = AssemblyStats()
        stats.per_stage_times["read_extraction"] = 2.5
        stats.per_stage_times["graph_construction"] = 3.1
        summary = stats.summary()
        assert "read_extraction" in summary
        assert "graph_construction" in summary

    def test_defaults(self) -> None:
        """Default stats have zero values."""
        stats = AssemblyStats()
        assert stats.total_reads == 0
        assert stats.assembled_transcripts == 0
        assert stats.elapsed_seconds == 0.0


class TestTimer:
    """Test timer context manager."""

    def test_timer_records_elapsed(self) -> None:
        """Timer records a positive elapsed time after the block exits."""
        timer = Timer(name="test_block")
        with timer:
            # Do a minimal operation to ensure some time passes
            _ = sum(range(1000))
        assert timer.elapsed > 0.0

    def test_timer_updates_stats(self) -> None:
        """Timer updates AssemblyStats.per_stage_times when given a stats object."""
        stats = AssemblyStats()
        with Timer(name="my_stage", stats=stats):
            _ = sum(range(1000))
        assert "my_stage" in stats.per_stage_times
        assert stats.per_stage_times["my_stage"] > 0.0

    def test_timer_name(self) -> None:
        """Timer stores its name correctly."""
        timer = Timer(name="graph_build")
        assert timer.name == "graph_build"
