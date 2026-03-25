"""Tests for BRAID v2 coverage, annotation, differential, and combined feature extractors."""

from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pysam
import pytest

from braid.target.rmats_bootstrap import RmatsEvent
from braid.v2.annotation import extract_annotation_features
from braid.v2.coverage import extract_coverage_features
from braid.v2.differential import extract_differential_features
from braid.v2.features import extract_all_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HEADER = {
    "HD": {"VN": "1.6", "SO": "coordinate"},
    "SQ": [{"SN": "chr1", "LN": 100_000}],
}


def _make_event(
    chrom: str = "chr1",
    upstream_es: int = 800,
    upstream_ee: int = 1000,
    exon_start: int = 2000,
    exon_end: int = 2200,
    downstream_es: int = 3000,
    downstream_ee: int = 3200,
    sample_1_inc_replicates: tuple[int, ...] = (20, 25, 15),
    sample_1_exc_replicates: tuple[int, ...] = (10, 8, 12),
    sample_2_inc_replicates: tuple[int, ...] = (5, 7, 6),
    sample_2_exc_replicates: tuple[int, ...] = (15, 18, 14),
) -> RmatsEvent:
    """Create a fully-populated SE event for testing."""
    s1_inc = sum(sample_1_inc_replicates)
    s1_exc = sum(sample_1_exc_replicates)
    s2_inc = sum(sample_2_inc_replicates)
    s2_exc = sum(sample_2_exc_replicates)
    return RmatsEvent(
        event_id="SE_test_1",
        event_type="SE",
        chrom=chrom,
        strand="+",
        gene="TESTGENE",
        inc_count=s1_inc,
        exc_count=s1_exc,
        rmats_psi=0.67,
        rmats_fdr=0.005,
        rmats_dpsi=-0.3,
        exon_start=exon_start,
        exon_end=exon_end,
        upstream_es=upstream_es,
        upstream_ee=upstream_ee,
        downstream_es=downstream_es,
        downstream_ee=downstream_ee,
        sample_1_inc_count=s1_inc,
        sample_1_exc_count=s1_exc,
        sample_2_inc_count=s2_inc,
        sample_2_exc_count=s2_exc,
        sample_1_inc_replicates=sample_1_inc_replicates,
        sample_1_exc_replicates=sample_1_exc_replicates,
        sample_2_inc_replicates=sample_2_inc_replicates,
        sample_2_exc_replicates=sample_2_exc_replicates,
        sample_1_psi=0.67,
        sample_2_psi=0.28,
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
    is_duplicate: bool = False,
) -> pysam.AlignedSegment:
    """Create a spliced read with one N operation."""
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
    seg.flag = 0x400 if is_duplicate else 0
    if xs_tag is not None:
        seg.set_tag("XS", xs_tag)
    return seg


def _make_unspliced_read(
    name: str,
    ref_start: int,
    length: int,
    mapq: int = 60,
) -> pysam.AlignedSegment:
    """Create a simple aligned read with no splicing."""
    seg = pysam.AlignedSegment()
    seg.query_name = name
    seg.reference_id = 0
    seg.reference_start = ref_start
    seg.mapping_quality = mapq
    seg.cigar = [(0, length)]
    seg.query_sequence = "A" * length
    seg.query_qualities = pysam.qualitystring_to_array("I" * length)
    seg.flag = 0
    return seg


# ---------------------------------------------------------------------------
# Coverage feature tests
# ---------------------------------------------------------------------------


class TestCoverageFeatures:
    """Tests for braid.v2.coverage.extract_coverage_features."""

    def test_dup_fraction_and_unique_starts(self, tmp_path: str) -> None:
        """PCR duplicate reads and unique start positions are counted correctly."""
        event = _make_event()
        bam_path = os.path.join(str(tmp_path), "test.bam")

        reads: list[pysam.AlignedSegment] = []

        # 4 inclusion reads spanning upstream_ee(1000) -> exon_start(2000)
        # 2 are PCR duplicates, all from different start positions
        for i in range(4):
            reads.append(
                _make_spliced_read(
                    f"inc_{i}",
                    ref_start=1000 - 30 - i * 5,
                    left_match=30 + i * 5,
                    intron_len=1000,
                    right_match=25,
                    is_duplicate=(i < 2),
                )
            )

        # 2 exclusion reads spanning upstream_ee(1000) -> downstream_es(3000)
        # Same start position, 1 duplicate
        for i in range(2):
            reads.append(
                _make_spliced_read(
                    f"exc_{i}",
                    ref_start=960,
                    left_match=40,
                    intron_len=2000,
                    right_match=40,
                    is_duplicate=(i == 0),
                )
            )

        _write_bam(bam_path, reads)
        features = extract_coverage_features(bam_path, event)

        assert features["dup_fraction_inc"] == pytest.approx(0.5)
        assert features["dup_fraction_exc"] == pytest.approx(0.5)
        # 4 inc reads from 4 distinct starts
        assert features["unique_start_fraction_inc"] == pytest.approx(1.0)
        # 2 exc reads from same start
        assert features["unique_start_fraction_exc"] == pytest.approx(0.5)

    def test_no_reads_returns_nan(self, tmp_path: str) -> None:
        """When no junction reads exist, B1-B4 are NaN."""
        event = _make_event(
            upstream_ee=50_000,
            exon_start=51_000,
            exon_end=51_200,
            downstream_es=52_000,
        )
        bam_path = os.path.join(str(tmp_path), "empty.bam")
        _write_bam(bam_path, [])
        features = extract_coverage_features(bam_path, event)

        assert np.isnan(features["dup_fraction_inc"])
        assert np.isnan(features["dup_fraction_exc"])
        assert np.isnan(features["unique_start_fraction_inc"])
        assert np.isnan(features["unique_start_fraction_exc"])

    def test_missing_coordinates_returns_nan(self) -> None:
        """Event without flanking coords returns all NaN."""
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
        features = extract_coverage_features("/dev/null", event)
        for key, val in features.items():
            assert np.isnan(val), f"{key} should be NaN but got {val}"


# ---------------------------------------------------------------------------
# Differential feature tests
# ---------------------------------------------------------------------------


class TestDifferentialFeatures:
    """Tests for braid.v2.differential.extract_differential_features."""

    def test_replicate_statistics(self) -> None:
        """Per-replicate PSI variance and range are computed correctly."""
        event = _make_event(
            sample_1_inc_replicates=(20, 25, 15),
            sample_1_exc_replicates=(10, 8, 12),
        )
        features = extract_differential_features(event)

        # Per-replicate PSI: 20/30=0.667, 25/33=0.758, 15/27=0.556
        psi_vals = [20 / 30, 25 / 33, 15 / 27]
        expected_var = float(np.var(psi_vals, ddof=1))
        expected_range = max(psi_vals) - min(psi_vals)

        assert features["replicate_psi_variance"] == pytest.approx(expected_var, rel=1e-4)
        assert features["replicate_psi_range"] == pytest.approx(expected_range, rel=1e-4)
        assert features["dpsi_ctrl_replicates"] == pytest.approx(expected_range, rel=1e-4)

    def test_support_counts_and_asymmetry(self) -> None:
        """Total support and log2 asymmetry are correct."""
        event = _make_event()
        features = extract_differential_features(event)

        ctrl_total = sum(event.sample_1_inc_replicates) + sum(event.sample_1_exc_replicates)
        kd_total = sum(event.sample_2_inc_replicates) + sum(event.sample_2_exc_replicates)

        assert features["total_support_ctrl"] == float(ctrl_total)
        assert features["total_support_kd"] == float(kd_total)
        assert features["support_asymmetry"] == pytest.approx(
            abs(math.log2(ctrl_total / kd_total)), rel=1e-6,
        )

    def test_passthrough_values(self) -> None:
        """FDR and |dPSI| are passed through correctly."""
        event = _make_event()
        features = extract_differential_features(event)

        assert features["rmats_fdr"] == 0.005
        assert features["abs_dpsi"] == pytest.approx(0.3)

    def test_single_replicate(self) -> None:
        """Single replicate yields zero variance and range."""
        event = _make_event(
            sample_1_inc_replicates=(50,),
            sample_1_exc_replicates=(10,),
            sample_2_inc_replicates=(5,),
            sample_2_exc_replicates=(15,),
        )
        features = extract_differential_features(event)
        assert features["replicate_psi_variance"] == 0.0
        assert features["replicate_psi_range"] == 0.0

    def test_no_replicates_returns_nan(self) -> None:
        """Empty replicate tuples yield NaN."""
        event = _make_event(
            sample_1_inc_replicates=(),
            sample_1_exc_replicates=(),
            sample_2_inc_replicates=(),
            sample_2_exc_replicates=(),
        )
        features = extract_differential_features(event)
        assert np.isnan(features["replicate_psi_variance"])
        assert np.isnan(features["replicate_psi_range"])


# ---------------------------------------------------------------------------
# Annotation feature tests
# ---------------------------------------------------------------------------


class TestAnnotationFeatures:
    """Tests for braid.v2.annotation.extract_annotation_features."""

    def test_overlapping_events_count(self) -> None:
        """n_overlapping_events counts events that share the exon region."""
        event = _make_event()
        overlapping = _make_event()
        overlapping.event_id = "SE_test_2"
        overlapping.exon_start = 2100  # overlaps 2000-2200

        non_overlapping = _make_event()
        non_overlapping.event_id = "SE_test_3"
        non_overlapping.exon_start = 5000
        non_overlapping.exon_end = 5200

        all_events = [event, overlapping, non_overlapping]
        features = extract_annotation_features(event, all_events=all_events)

        assert features["n_overlapping_events"] == 1.0

    def test_no_reference_returns_nan_motifs(self) -> None:
        """Without reference FASTA, splice motif features are NaN."""
        event = _make_event()
        features = extract_annotation_features(event)
        assert np.isnan(features["splice_motif_inc"])
        assert np.isnan(features["splice_motif_exc"])

    def test_no_gtf_returns_nan_overlap(self) -> None:
        """Without GTF, overlapping gene flag is NaN."""
        event = _make_event()
        features = extract_annotation_features(event)
        assert np.isnan(features["overlapping_gene_flag"])


# ---------------------------------------------------------------------------
# Combined feature extraction tests
# ---------------------------------------------------------------------------


class TestExtractAllFeatures:
    """Tests for braid.v2.features.extract_all_features."""

    def test_all_28_features_present(self, tmp_path: str) -> None:
        """Combined extractor returns exactly 29 feature keys."""
        event = _make_event()
        bam_path = os.path.join(str(tmp_path), "test.bam")

        reads: list[pysam.AlignedSegment] = []
        # A few inclusion reads
        for i in range(3):
            reads.append(
                _make_spliced_read(
                    f"inc_{i}",
                    ref_start=970,
                    left_match=30,
                    intron_len=1000,
                    right_match=25,
                )
            )
        # A few exclusion reads
        for i in range(2):
            reads.append(
                _make_spliced_read(
                    f"exc_{i}",
                    ref_start=960,
                    left_match=40,
                    intron_len=2000,
                    right_match=40,
                )
            )
        _write_bam(bam_path, reads)

        features = extract_all_features(bam_path, event)

        # 10 junction + 6 coverage + 5 annotation + 8 differential = 29
        expected_keys = {
            # A1-A10 junction
            "median_mapq_inc", "median_mapq_exc",
            "frac_mapq0_inc", "frac_mapq0_exc",
            "min_anchor_inc", "min_anchor_exc",
            "median_anchor_inc", "frac_short_anchor_inc",
            "mismatch_rate_near_junction", "strand_consistency",
            # B1-B4, C1-C2 coverage
            "dup_fraction_inc", "dup_fraction_exc",
            "unique_start_fraction_inc", "unique_start_fraction_exc",
            "exon_body_coverage_uniformity", "exon_body_mean_coverage",
            # C3-C7 annotation
            "splice_motif_inc", "splice_motif_exc",
            "overlapping_gene_flag", "n_overlapping_events",
            "flanking_exon_coverage_ratio",
            # B5-B6, D1-D6 differential
            "replicate_psi_variance", "replicate_psi_range",
            "dpsi_ctrl_replicates",
            "total_support_ctrl", "total_support_kd",
            "support_asymmetry", "rmats_fdr", "abs_dpsi",
        }
        assert set(features.keys()) == expected_keys
        assert len(features) == 29

    def test_all_values_are_float(self, tmp_path: str) -> None:
        """Every feature value is a float (including NaN)."""
        event = _make_event()
        bam_path = os.path.join(str(tmp_path), "test.bam")
        _write_bam(bam_path, [])

        features = extract_all_features(bam_path, event)

        for key, val in features.items():
            assert isinstance(val, float), f"{key} is {type(val).__name__}, expected float"

    def test_differential_features_independent_of_bam(self, tmp_path: str) -> None:
        """Differential features don't depend on BAM content."""
        event = _make_event()
        bam_path = os.path.join(str(tmp_path), "test.bam")
        _write_bam(bam_path, [])

        all_features = extract_all_features(bam_path, event)
        diff_only = extract_differential_features(event)

        for key, val in diff_only.items():
            if np.isnan(val):
                assert np.isnan(all_features[key])
            else:
                assert all_features[key] == pytest.approx(val)
