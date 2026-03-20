"""Tests for the TargetSplice targeted assembly module."""

from __future__ import annotations

import pytest

from rapidsplice.target.comparator import (
    _chains_match,
    _exons_to_intron_chain,
    classify_isoform,
)
from rapidsplice.target.extractor import (
    TargetRegion,
    parse_region_string,
)


class TestTargetRegion:
    """Tests for TargetRegion dataclass."""

    def test_length(self) -> None:
        r = TargetRegion(chrom="1", start=100, end=200)
        assert r.length == 100

    def test_with_flank(self) -> None:
        r = TargetRegion(chrom="1", start=100, end=200)
        flanked = r.with_flank(50)
        assert flanked.start == 50
        assert flanked.end == 250

    def test_with_flank_clamp_zero(self) -> None:
        r = TargetRegion(chrom="1", start=10, end=100)
        flanked = r.with_flank(20)
        assert flanked.start == 0

    def test_preserves_metadata(self) -> None:
        r = TargetRegion(
            chrom="17", start=100, end=200,
            strand="-", gene_name="TP53", gene_id="ENSG001",
        )
        flanked = r.with_flank(10)
        assert flanked.gene_name == "TP53"
        assert flanked.strand == "-"


class TestParseRegionString:
    """Tests for parse_region_string."""

    def test_standard_format(self) -> None:
        r = parse_region_string("chr17:7668402-7687538")
        assert r.chrom == "chr17"
        assert r.start == 7668401  # 1-based to 0-based
        assert r.end == 7687538

    def test_no_chr_prefix(self) -> None:
        r = parse_region_string("17:100-200")
        assert r.chrom == "17"
        assert r.start == 99
        assert r.end == 200

    def test_commas_in_coords(self) -> None:
        r = parse_region_string("chr1:1,000-2,000")
        assert r.start == 999
        assert r.end == 2000

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError):
            parse_region_string("invalid")

    def test_missing_end(self) -> None:
        with pytest.raises(ValueError):
            parse_region_string("chr1:100")


class TestExonsToIntronChain:
    """Tests for _exons_to_intron_chain helper."""

    def test_two_exons(self) -> None:
        exons = [(100, 200), (300, 400)]
        introns = _exons_to_intron_chain(exons)
        assert introns == [(200, 300)]

    def test_three_exons(self) -> None:
        exons = [(100, 200), (300, 400), (500, 600)]
        introns = _exons_to_intron_chain(exons)
        assert introns == [(200, 300), (400, 500)]

    def test_single_exon(self) -> None:
        exons = [(100, 200)]
        introns = _exons_to_intron_chain(exons)
        assert introns == []


class TestChainsMatch:
    """Tests for _chains_match helper."""

    def test_exact_match(self) -> None:
        a = [(100, 200), (300, 400)]
        b = [(100, 200), (300, 400)]
        assert _chains_match(a, b, 0) is True

    def test_within_tolerance(self) -> None:
        a = [(100, 200), (300, 400)]
        b = [(101, 199), (301, 399)]
        assert _chains_match(a, b, 2) is True

    def test_beyond_tolerance(self) -> None:
        a = [(100, 200)]
        b = [(110, 200)]
        assert _chains_match(a, b, 5) is False

    def test_different_lengths(self) -> None:
        a = [(100, 200)]
        b = [(100, 200), (300, 400)]
        assert _chains_match(a, b, 0) is False


class TestClassifyIsoform:
    """Tests for classify_isoform."""

    def _make_ref(
        self,
        exons: list[tuple[int, int]],
        tid: str = "TX1",
    ) -> dict:
        return {
            "exons": exons,
            "transcript_id": tid,
            "gene_name": "TEST",
        }

    def test_exact_match(self) -> None:
        exons = [(100, 200), (300, 400), (500, 600)]
        ref = [self._make_ref(exons, "REF.1")]
        cls = classify_isoform(exons, ref)
        assert cls.category == "exact_match"
        assert cls.matched_transcript_id == "REF.1"

    def test_novel_combination(self) -> None:
        # Query uses junctions from two different ref transcripts
        query = [(100, 200), (300, 400), (500, 600)]
        ref_a = self._make_ref([(100, 200), (300, 400)])
        ref_b = self._make_ref([(300, 400), (500, 600)])
        cls = classify_isoform(query, [ref_a, ref_b])
        assert cls.category == "novel_combination"

    def test_novel_junction(self) -> None:
        query = [(100, 200), (350, 400)]  # junction 200-350 is novel
        ref = [self._make_ref([(100, 200), (300, 400)])]  # junction 200-300
        cls = classify_isoform(query, ref)
        assert cls.category == "novel_junction"
        assert cls.n_novel_junctions == 1

    def test_single_exon(self) -> None:
        query = [(100, 200)]
        ref = [self._make_ref([(100, 200), (300, 400)])]
        cls = classify_isoform(query, ref)
        assert cls.category == "single_exon"

    def test_novel_exon(self) -> None:
        # Query has an exon not overlapping any reference exon
        query = [(100, 200), (800, 900)]  # 800-900 is novel
        ref = [self._make_ref([(100, 200), (300, 400)])]
        cls = classify_isoform(query, ref)
        assert cls.category == "novel_exon"

    def test_empty_reference(self) -> None:
        query = [(100, 200), (300, 400)]
        cls = classify_isoform(query, [])
        # No ref exons → all exons are novel → novel_exon
        assert cls.category == "novel_exon"
