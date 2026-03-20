"""Tests for PSI quantification and statistical confidence intervals."""

from __future__ import annotations

import math

import numpy as np

from rapidsplice.io.bam_reader import JunctionEvidence
from rapidsplice.io.gtf_writer import TranscriptRecord
from rapidsplice.splicing.events import ASEvent, EventType
from rapidsplice.splicing.psi import PSIResult, calculate_all_psi, calculate_psi
from rapidsplice.splicing.statistics import (
    add_confidence_intervals,
    beta_binomial_ci,
    psi_significance_filter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tx(
    tid: str,
    gene_id: str,
    exons: list[tuple[int, int]],
    strand: str = "+",
    chrom: str = "chr1",
) -> TranscriptRecord:
    """Create a TranscriptRecord from exon list."""
    return TranscriptRecord(
        transcript_id=tid,
        gene_id=gene_id,
        chrom=chrom,
        strand=strand,
        start=exons[0][0],
        end=exons[-1][1],
        exons=exons,
    )


def _make_junction_evidence(
    chrom: str,
    junctions: list[tuple[int, int, int]],
) -> JunctionEvidence:
    """Create JunctionEvidence from (start, end, count) tuples."""
    if not junctions:
        return JunctionEvidence(
            chrom=chrom,
            starts=np.empty(0, dtype=np.int64),
            ends=np.empty(0, dtype=np.int64),
            counts=np.empty(0, dtype=np.int32),
            strands=np.empty(0, dtype=np.int8),
        )
    starts = np.array([j[0] for j in junctions], dtype=np.int64)
    ends = np.array([j[1] for j in junctions], dtype=np.int64)
    counts = np.array([j[2] for j in junctions], dtype=np.int32)
    strands = np.zeros(len(junctions), dtype=np.int8)
    return JunctionEvidence(chrom=chrom, starts=starts, ends=ends, counts=counts, strands=strands)


# ---------------------------------------------------------------------------
# PSI calculation tests
# ---------------------------------------------------------------------------


class TestCalculatePSI:
    """Tests for PSI calculation."""

    def test_balanced_se(self) -> None:
        """PSI = 0.5 for evenly-split SE event."""
        event = ASEvent(
            event_id="SE:chr1:+:200_300_400_500",
            event_type=EventType.SE,
            gene_id="g1",
            chrom="chr1",
            strand="+",
            inclusion_junctions=[(200, 300), (400, 500)],
            exclusion_junctions=[(200, 500)],
        )
        je = _make_junction_evidence("chr1", [
            (200, 300, 10),  # inclusion junction 1
            (400, 500, 10),  # inclusion junction 2
            (200, 500, 10),  # exclusion junction
        ])
        result = calculate_psi(event, je)
        # I=20, lI=2, S=10, lS=1 → I/lI=10, S/lS=10 → PSI=0.5
        assert abs(result.psi - 0.5) < 0.01
        assert result.total_reads == 30

    def test_fully_included(self) -> None:
        """PSI = 1.0 when only inclusion reads exist."""
        event = ASEvent(
            event_id="SE:chr1:+:200_300_400_500",
            event_type=EventType.SE,
            gene_id="g1",
            chrom="chr1",
            strand="+",
            inclusion_junctions=[(200, 300), (400, 500)],
            exclusion_junctions=[(200, 500)],
        )
        je = _make_junction_evidence("chr1", [
            (200, 300, 20),
            (400, 500, 20),
        ])
        result = calculate_psi(event, je)
        assert result.psi == 1.0

    def test_fully_excluded(self) -> None:
        """PSI = 0.0 when only exclusion reads exist."""
        event = ASEvent(
            event_id="SE:chr1:+:200_300_400_500",
            event_type=EventType.SE,
            gene_id="g1",
            chrom="chr1",
            strand="+",
            inclusion_junctions=[(200, 300), (400, 500)],
            exclusion_junctions=[(200, 500)],
        )
        je = _make_junction_evidence("chr1", [
            (200, 500, 15),
        ])
        result = calculate_psi(event, je)
        assert result.psi == 0.0

    def test_no_reads_nan(self) -> None:
        """PSI is NaN when no reads support any junction."""
        event = ASEvent(
            event_id="SE:chr1:+:200_300_400_500",
            event_type=EventType.SE,
            gene_id="g1",
            chrom="chr1",
            strand="+",
            inclusion_junctions=[(200, 300)],
            exclusion_junctions=[(200, 500)],
        )
        je = _make_junction_evidence("chr1", [])
        result = calculate_psi(event, je)
        assert math.isnan(result.psi)
        assert result.total_reads == 0

    def test_retained_intron(self) -> None:
        """RI event PSI with no inclusion junctions."""
        event = ASEvent(
            event_id="RI:chr1:+:300_400",
            event_type=EventType.RI,
            gene_id="g1",
            chrom="chr1",
            strand="+",
            inclusion_junctions=[],
            exclusion_junctions=[(300, 400)],
        )
        # With exclusion reads → intron is spliced out → PSI=0
        je = _make_junction_evidence("chr1", [(300, 400, 10)])
        result = calculate_psi(event, je)
        assert result.psi == 0.0


class TestCalculateAllPSI:
    """Tests for batch PSI calculation."""

    def test_missing_chromosome(self) -> None:
        """Events on missing chromosomes get NaN PSI."""
        event = ASEvent(
            event_id="SE:chr2:+:100_200_300_400",
            event_type=EventType.SE,
            gene_id="g1",
            chrom="chr2",
            strand="+",
            inclusion_junctions=[(100, 200)],
            exclusion_junctions=[(100, 400)],
        )
        results = calculate_all_psi([event], {"chr1": _make_junction_evidence("chr1", [])})
        assert len(results) == 1
        assert math.isnan(results[0].psi)


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------


class TestBetaBinomialCI:
    """Tests for Beta-binomial credible intervals."""

    def test_uniform_prior(self) -> None:
        """With uniform prior and no data, CI spans [0, 1]."""
        result = beta_binomial_ci(0, 0)
        assert result.ci_low < 0.1
        assert result.ci_high > 0.9

    def test_high_inclusion(self) -> None:
        """High inclusion count yields high CI lower bound."""
        result = beta_binomial_ci(95, 100)
        assert result.ci_low > 0.85
        assert result.ci_high > 0.95

    def test_low_inclusion(self) -> None:
        """Low inclusion count yields low CI upper bound."""
        result = beta_binomial_ci(5, 100)
        assert result.ci_low < 0.1
        assert result.ci_high < 0.15

    def test_balanced(self) -> None:
        """Balanced counts center CI around 0.5."""
        result = beta_binomial_ci(50, 100)
        assert 0.35 < result.ci_low < 0.45
        assert 0.55 < result.ci_high < 0.65

    def test_ci_ordering(self) -> None:
        """CI low is always less than CI high."""
        for inc in [0, 10, 50, 90, 100]:
            result = beta_binomial_ci(inc, 100)
            assert result.ci_low <= result.ci_high


class TestAddConfidenceIntervals:
    """Tests for adding CIs to PSI results."""

    def test_adds_ci(self) -> None:
        """CIs are added to PSI results in-place."""
        results = [
            PSIResult("e1", 0.8, 80, 20, 2, 1, 100),
            PSIResult("e2", 0.5, 50, 50, 1, 1, 100),
        ]
        add_confidence_intervals(results)
        assert results[0].ci_low > 0.0
        assert results[0].ci_high < 1.0
        assert results[0].ci_low < results[0].ci_high

    def test_zero_reads_full_ci(self) -> None:
        """Zero-read events get [0, 1] CI."""
        results = [PSIResult("e1", float("nan"), 0, 0, 1, 1, 0)]
        add_confidence_intervals(results)
        assert results[0].ci_low == 0.0
        assert results[0].ci_high == 1.0


class TestPSISignificanceFilter:
    """Tests for PSI significance filtering."""

    def test_filters_low_reads(self) -> None:
        """Events with fewer than min_reads are filtered out."""
        results = [
            PSIResult("e1", 0.8, 80, 20, 2, 1, 100, 0.7, 0.9),
            PSIResult("e2", 0.5, 3, 3, 1, 1, 6, 0.2, 0.8),
        ]
        filtered = psi_significance_filter(results, min_reads=10)
        assert len(filtered) == 1
        assert filtered[0].event_id == "e1"

    def test_filters_wide_ci(self) -> None:
        """Events with CI wider than threshold are filtered."""
        results = [
            PSIResult("e1", 0.8, 80, 20, 2, 1, 100, 0.7, 0.9),
            PSIResult("e2", 0.5, 10, 10, 1, 1, 20, 0.1, 0.9),
        ]
        filtered = psi_significance_filter(results, min_reads=5, max_ci_width=0.5)
        assert len(filtered) == 1
        assert filtered[0].event_id == "e1"
