"""Tests for braid.v2.junction feature extractor."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pysam
import pytest

from braid.target.rmats_bootstrap import RmatsEvent
from braid.v2.junction import (
    _junction_spans,
    _matches_junction,
    extract_junction_features,
)

# ---------------------------------------------------------------------------
# Helpers to build synthetic BAM files
# ---------------------------------------------------------------------------

_HEADER = {
    "HD": {"VN": "1.6", "SO": "coordinate"},
    "SQ": [{"SN": "chr1", "LN": 100_000}],
}


def _make_event(
    chrom: str = "chr1",
    upstream_ee: int = 1000,
    exon_start: int = 2000,
    exon_end: int = 2200,
    downstream_es: int = 3000,
) -> RmatsEvent:
    """Create a minimal SE event for testing."""
    return RmatsEvent(
        event_id="SE_1",
        event_type="SE",
        chrom=chrom,
        strand="+",
        gene="TESTGENE",
        inc_count=10,
        exc_count=5,
        rmats_psi=0.67,
        rmats_fdr=0.01,
        rmats_dpsi=0.2,
        exon_start=exon_start,
        exon_end=exon_end,
        upstream_ee=upstream_ee,
        downstream_ee=downstream_es + 200,
        upstream_es=upstream_ee - 200,
        downstream_es=downstream_es,
    )


def _write_bam(path: str, reads: list[pysam.AlignedSegment]) -> None:
    """Write reads to a sorted, indexed BAM file."""
    with pysam.AlignmentFile(path, "wb", header=_HEADER) as bam:
        for r in reads:
            bam.write(r)
    pysam.sort("-o", path, path)
    pysam.index(path)


def _make_spliced_read(
    name: str,
    ref_start: int,
    left_match: int,
    intron_len: int,
    right_match: int,
    mapq: int = 60,
    xs_tag: str | None = "+",
) -> pysam.AlignedSegment:
    """Create a spliced read with one N operation: leftM + intronN + rightM."""
    seg = pysam.AlignedSegment()
    seg.query_name = name
    seg.reference_id = 0
    seg.reference_start = ref_start
    seg.mapping_quality = mapq
    seg.cigar = [(0, left_match), (3, intron_len), (0, right_match)]
    seg.query_sequence = "A" * (left_match + right_match)
    seg.query_qualities = pysam.qualitystring_to_array(
        "I" * (left_match + right_match)
    )
    seg.flag = 0
    if xs_tag is not None:
        seg.set_tag("XS", xs_tag)
    return seg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJunctionSpans:
    """Unit tests for the _junction_spans helper."""

    def test_single_intron(self) -> None:
        read = _make_spliced_read("r1", 950, 50, 1000, 40)
        spans = _junction_spans(read)
        assert len(spans) == 1
        intron_start, intron_end, left_anc, right_anc = spans[0]
        assert intron_start == 1000
        assert intron_end == 2000
        assert left_anc == 50
        assert right_anc == 40

    def test_no_intron(self) -> None:
        seg = pysam.AlignedSegment()
        seg.query_name = "r_nointron"
        seg.reference_id = 0
        seg.reference_start = 500
        seg.mapping_quality = 60
        seg.cigar = [(0, 100)]
        seg.query_sequence = "A" * 100
        seg.query_qualities = pysam.qualitystring_to_array("I" * 100)
        seg.flag = 0
        assert _junction_spans(seg) == []


class TestExtractJunctionFeatures:
    """Integration tests for the full feature extractor."""

    def test_basic_se_event(self, tmp_path: str) -> None:
        """Inclusion + exclusion reads produce sane feature values."""
        event = _make_event()
        bam_path = os.path.join(str(tmp_path), "test.bam")

        reads: list[pysam.AlignedSegment] = []

        # 5 inclusion reads spanning upstream_ee(1000) -> exon_start(2000)
        for i in range(5):
            reads.append(
                _make_spliced_read(
                    f"inc_left_{i}",
                    ref_start=1000 - 30,
                    left_match=30,
                    intron_len=1000,
                    right_match=25,
                    mapq=60,
                    xs_tag="+",
                )
            )

        # 5 inclusion reads spanning exon_end(2200) -> downstream_es(3000)
        for i in range(5):
            reads.append(
                _make_spliced_read(
                    f"inc_right_{i}",
                    ref_start=2200 - 20,
                    left_match=20,
                    intron_len=800,
                    right_match=35,
                    mapq=60,
                    xs_tag="+",
                )
            )

        # 3 exclusion reads spanning upstream_ee(1000) -> downstream_es(3000)
        for i in range(3):
            reads.append(
                _make_spliced_read(
                    f"exc_{i}",
                    ref_start=1000 - 40,
                    left_match=40,
                    intron_len=2000,
                    right_match=40,
                    mapq=50,
                    xs_tag="+",
                )
            )

        _write_bam(bam_path, reads)

        features = extract_junction_features(bam_path, event)

        assert features["median_mapq_inc"] == 60.0
        assert features["median_mapq_exc"] == 50.0
        assert features["frac_mapq0_inc"] == 0.0
        assert features["frac_mapq0_exc"] == 0.0
        # Min anchor: inc left has min(30,25)=25, inc right has min(20,35)=20
        assert features["min_anchor_inc"] == 20.0
        assert features["min_anchor_exc"] == 40.0
        # No short anchors (all >= 8)
        assert features["frac_short_anchor_inc"] == 0.0
        # All reads have XS=+, so strand consistency = 1.0
        assert features["strand_consistency"] == 1.0

    def test_no_reads_returns_nan(self, tmp_path: str) -> None:
        """When no reads overlap the event, all features are NaN."""
        event = _make_event(
            upstream_ee=50_000,
            exon_start=51_000,
            exon_end=51_200,
            downstream_es=52_000,
        )
        bam_path = os.path.join(str(tmp_path), "empty.bam")
        # Write BAM with no reads
        _write_bam(bam_path, [])

        features = extract_junction_features(bam_path, event)

        for key, val in features.items():
            assert np.isnan(val), f"{key} should be NaN but got {val}"

    def test_mapq0_and_short_anchors(self, tmp_path: str) -> None:
        """Reads with MAPQ=0 and short anchors are correctly detected."""
        event = _make_event()
        bam_path = os.path.join(str(tmp_path), "lowq.bam")

        reads: list[pysam.AlignedSegment] = []

        # 4 inclusion reads: 2 with MAPQ=0, 2 with MAPQ=60
        # All with short left anchor (5bp)
        for i in range(2):
            reads.append(
                _make_spliced_read(
                    f"inc_q0_{i}",
                    ref_start=1000 - 5,
                    left_match=5,
                    intron_len=1000,
                    right_match=30,
                    mapq=0,
                    xs_tag="+",
                )
            )
        for i in range(2):
            reads.append(
                _make_spliced_read(
                    f"inc_q60_{i}",
                    ref_start=1000 - 5,
                    left_match=5,
                    intron_len=1000,
                    right_match=30,
                    mapq=60,
                    xs_tag="-",
                )
            )

        _write_bam(bam_path, reads)

        features = extract_junction_features(bam_path, event)

        assert features["frac_mapq0_inc"] == 0.5
        # All inc reads have min(5, 30) = 5, which is < 8
        assert features["frac_short_anchor_inc"] == 1.0
        assert features["min_anchor_inc"] == 5.0
        # Strand: 2 "+", 2 "-" -> consistency = 0.5
        assert features["strand_consistency"] == 0.5

    def test_missing_coordinates_returns_nan(self) -> None:
        """Event without upstream_ee/downstream_es returns all NaN."""
        event = RmatsEvent(
            event_id="SE_bad",
            event_type="SE",
            chrom="chr1",
            strand="+",
            gene="BAD",
            inc_count=0,
            exc_count=0,
            rmats_psi=0.0,
            rmats_fdr=1.0,
            rmats_dpsi=0.0,
            upstream_ee=None,
            downstream_es=None,
        )
        # No BAM needed — should return NaN immediately
        features = extract_junction_features("/dev/null", event)
        for key, val in features.items():
            assert np.isnan(val), f"{key} should be NaN but got {val}"

    def test_no_xs_tag_returns_nan_strand(self, tmp_path: str) -> None:
        """Reads without XS tag produce NaN strand_consistency."""
        event = _make_event()
        bam_path = os.path.join(str(tmp_path), "noxs.bam")

        reads = [
            _make_spliced_read(
                "r_noxs",
                ref_start=1000 - 30,
                left_match=30,
                intron_len=1000,
                right_match=25,
                mapq=60,
                xs_tag=None,
            )
        ]
        _write_bam(bam_path, reads)

        features = extract_junction_features(bam_path, event)
        assert np.isnan(features["strand_consistency"])
