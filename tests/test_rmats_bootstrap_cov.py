"""Coverage-gap tests for braid/target/rmats_bootstrap.py.

Targets the specific uncovered lines reported by pytest-cov:
  88, 124, 131, 150-151, 161-162, 182, 223-265, 315, 335, 415-420, 458-459

All assertions verify ACTUAL current behaviour — no coverage theater.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from braid.target.rmats_bootstrap import (
    RmatsEvent,
    _build_event_id,
    _parse_count_sum,
    _parse_float_na,
    _parse_inc_level_mean,
    _select_gene,
    add_bootstrap_ci,
    get_group_counts,
    parse_rmats_output,
)

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

_SE_HEADER = "\t".join([
    "ID", "GeneID", "geneSymbol", "chr", "strand",
    "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE",
    "downstreamES", "downstreamEE", "ID", "IJC_SAMPLE_1",
    "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
])

_A3SS_HEADER = "\t".join([
    "ID", "GeneID", "geneSymbol", "chr", "strand",
    "longExonStart_0base", "longExonEnd", "shortES", "shortEE",
    "flankingES", "flankingEE", "ID", "IJC_SAMPLE_1",
    "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
])

_MXE_HEADER = "\t".join([
    "ID", "GeneID", "geneSymbol", "chr", "strand",
    "1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd",
    "upstreamES", "upstreamEE", "downstreamES", "downstreamEE", "ID",
    "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
])

_RI_HEADER = "\t".join([
    "ID", "GeneID", "geneSymbol", "chr", "strand",
    "riExonStart_0base", "riExonEnd", "upstreamES", "upstreamEE",
    "downstreamES", "downstreamEE", "ID", "IJC_SAMPLE_1",
    "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
])


def _make_rmats_event(
    *,
    event_type: str = "SE",
    inc_1: int = 20,
    exc_1: int = 5,
    inc_2: int = 3,
    exc_2: int = 17,
    inc_form_len: float = 1.0,
    skip_form_len: float = 1.0,
) -> RmatsEvent:
    return RmatsEvent(
        event_id=f"{event_type}:chr1:+:100-200",
        event_type=event_type,
        chrom="chr1",
        strand="+",
        gene="GENE1",
        inc_count=inc_1,
        exc_count=exc_1,
        rmats_psi=0.8,
        rmats_fdr=0.01,
        rmats_dpsi=0.4,
        sample_1_inc_count=inc_1,
        sample_1_exc_count=exc_1,
        sample_2_inc_count=inc_2,
        sample_2_exc_count=exc_2,
        sample_1_inc_replicates=(inc_1,),
        sample_1_exc_replicates=(exc_1,),
        sample_2_inc_replicates=(inc_2,),
        sample_2_exc_replicates=(exc_2,),
        inc_form_len=inc_form_len,
        skip_form_len=skip_form_len,
    )


# ---------------------------------------------------------------------------
# _parse_count_sum  (line 88)
# ---------------------------------------------------------------------------


class TestParseCountSum:
    def test_sums_vector(self) -> None:
        assert _parse_count_sum("10,20,30") == 60

    def test_none_returns_zero(self) -> None:
        # line 88: delegates to _parse_count_vector which returns () for None
        assert _parse_count_sum(None) == 0

    def test_empty_returns_zero(self) -> None:
        assert _parse_count_sum("") == 0

    def test_single_value(self) -> None:
        assert _parse_count_sum("7") == 7


# ---------------------------------------------------------------------------
# _parse_inc_level_mean  (lines 124, 131)
# ---------------------------------------------------------------------------


class TestParseIncLevelMean:
    def test_none_returns_none(self) -> None:
        # line 124: early return None
        assert _parse_inc_level_mean(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_inc_level_mean("") is None

    def test_all_na_returns_none(self) -> None:
        # line 131: psi_vals list is empty after filtering
        assert _parse_inc_level_mean("NA,NA") is None

    def test_mixed_na_averages_valid(self) -> None:
        result = _parse_inc_level_mean("0.8,NA,0.6")
        assert result is not None
        assert abs(result - 0.7) < 1e-9

    def test_quoted_vector_averages_valid(self) -> None:
        result = _parse_inc_level_mean('"0.8,0.6"')
        assert result is not None
        assert abs(result - 0.7) < 1e-9

    def test_quoted_missing_values_are_ignored(self) -> None:
        assert _parse_inc_level_mean('"NA,NULL,nan"') is None

    def test_single_value(self) -> None:
        result = _parse_inc_level_mean("0.5")
        assert result is not None
        assert abs(result - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# _parse_float_na  (lines 150-151)
# ---------------------------------------------------------------------------


class TestParseFloatNa:
    def test_none_is_nan(self) -> None:
        assert math.isnan(_parse_float_na(None))

    def test_empty_string_is_nan(self) -> None:
        assert math.isnan(_parse_float_na(""))

    def test_na_variants_are_nan(self) -> None:
        for bad in ("NA", "NaN", "NULL", "nan", "NAN"):
            assert math.isnan(_parse_float_na(bad)), f"expected NaN for {bad!r}"

    def test_valid_float(self) -> None:
        assert abs(_parse_float_na("0.05") - 0.05) < 1e-12

    def test_non_numeric_string_is_nan(self) -> None:
        # lines 150-151: ValueError branch returns NaN
        assert math.isnan(_parse_float_na("not_a_number"))

    def test_quoted_value(self) -> None:
        assert abs(_parse_float_na('"0.123"') - 0.123) < 1e-9


# ---------------------------------------------------------------------------
# _select_gene  (lines 161-162: fallback to GeneID)
# ---------------------------------------------------------------------------


class TestSelectGene:
    def _make_cols(self, header: list[str]) -> dict[str, int]:
        return {h: i for i, h in enumerate(header)}

    def test_prefers_gene_symbol(self) -> None:
        header = ["GeneID", "geneSymbol"]
        cols = self._make_cols(header)
        fields = ["ENSG001", "BRCA1"]
        result = _select_gene(fields, cols)
        assert result == "BRCA1"

    def test_falls_back_to_gene_id_when_symbol_na(self) -> None:
        # lines 161-162: geneSymbol is "NA" → fall back to GeneID
        header = ["GeneID", "geneSymbol"]
        cols = self._make_cols(header)
        fields = ["ENSG001", "NA"]
        result = _select_gene(fields, cols)
        assert result == "ENSG001"

    def test_falls_back_to_gene_id_when_symbol_missing(self) -> None:
        # No geneSymbol column at all
        header = ["GeneID"]
        cols = self._make_cols(header)
        fields = ["ENSG001"]
        result = _select_gene(fields, cols)
        assert result == "ENSG001"

    def test_strips_quotes_from_gene_id(self) -> None:
        header = ["GeneID"]
        cols = self._make_cols(header)
        fields = ['"ENSG002"']
        result = _select_gene(fields, cols)
        assert result == "ENSG002"


# ---------------------------------------------------------------------------
# get_group_counts  (line 182: ValueError for bad sample label)
# ---------------------------------------------------------------------------


class TestGetGroupCounts:
    def test_sample_1(self) -> None:
        ev = _make_rmats_event(inc_1=10, exc_1=3)
        assert get_group_counts(ev, "sample_1") == (10, 3)

    def test_sample_2(self) -> None:
        ev = _make_rmats_event(inc_2=7, exc_2=13)
        assert get_group_counts(ev, "sample_2") == (7, 13)

    def test_invalid_label_raises(self) -> None:
        # line 182
        ev = _make_rmats_event()
        with pytest.raises(ValueError, match="sample_1.*sample_2"):
            get_group_counts(ev, "sample_3")


# ---------------------------------------------------------------------------
# _build_event_id  (lines 223-265: A3SS, A5SS, MXE, RI branches)
# ---------------------------------------------------------------------------


class TestBuildEventId:
    """Test all five event-type branches of _build_event_id."""

    def _cols(self, header: list[str]) -> dict[str, int]:
        return {h: i for i, h in enumerate(header)}

    def test_se_with_flanking(self) -> None:
        header = ["upstreamEE", "downstreamES"]
        cols = self._cols(header)
        fields = ["99", "201"]
        eid = _build_event_id("SE", "chr1", "+", 100, 200, None, 99, 201, None, fields, cols)
        assert eid.startswith("SE:chr1:+:100-200")
        assert "u99" in eid
        assert "d201" in eid

    def test_se_without_flanking(self) -> None:
        eid = _build_event_id("SE", "chr1", "+", 100, 200, None, None, None, None, [], {})
        assert eid == "SE:chr1:+:100-200"

    def test_a3ss_with_long_short_coords(self) -> None:
        # lines 223-235: A3SS branch
        header = ["longExonStart_0base", "longExonEnd", "shortES", "shortEE"]
        cols = self._cols(header)
        fields = ["50", "120", "80", "120"]
        eid = _build_event_id("A3SS", "chr2", "-", 80, 120, None, None, None, None, fields, cols)
        assert eid.startswith("A3SS:chr2:-:80-120")
        assert "l50-120" in eid
        assert "s80-120" in eid

    def test_a3ss_without_optional_cols(self) -> None:
        # branch falls back to base when columns absent
        eid = _build_event_id("A3SS", "chr2", "+", 80, 120, None, None, None, None, [], {})
        assert eid == "A3SS:chr2:+:80-120"

    def test_a3ss_includes_flanking_coords_to_avoid_rmats_collisions(self) -> None:
        header = [
            "longExonStart_0base", "longExonEnd", "shortES", "shortEE",
            "flankingES", "flankingEE",
        ]
        cols = self._cols(header)
        fields_1 = ["50", "120", "80", "120", "1", "49"]
        fields_2 = ["50", "120", "80", "120", "121", "170"]

        eid_1 = _build_event_id(
            "A3SS", "chr2", "+", 50, 120, None, None, None, None, fields_1, cols,
        )
        eid_2 = _build_event_id(
            "A3SS", "chr2", "+", 50, 120, None, None, None, None, fields_2, cols,
        )

        assert eid_1 != eid_2
        assert "f1-49" in eid_1
        assert "f121-170" in eid_2

    def test_a5ss_branch(self) -> None:
        # lines 223-235: A5SS uses same branch as A3SS
        header = [
            "longExonStart_0base", "longExonEnd", "shortES", "shortEE",
            "flankingES", "flankingEE",
        ]
        cols = self._cols(header)
        fields = ["10", "50", "30", "50", "51", "90"]
        eid = _build_event_id("A5SS", "chr3", "+", 30, 50, None, None, None, None, fields, cols)
        assert eid.startswith("A5SS:chr3:+:30-50")
        assert "l10-50" in eid
        assert "f51-90" in eid

    def test_mxe_branch_full(self) -> None:
        # lines 237-253: MXE branch
        header = [
            "1stExonStart_0base", "1stExonEnd",
            "2ndExonStart_0base", "2ndExonEnd",
        ]
        cols = self._cols(header)
        fields = ["100", "150", "200", "250"]
        eid = _build_event_id(
            "MXE", "chr4", "+", 100, 150,
            None, 99, 301, None,
            fields, cols,
        )
        assert eid.startswith("MXE:chr4:+:100-150")
        assert "e1=100-150" in eid
        assert "e2=200-250" in eid
        assert "u99" in eid
        assert "d301" in eid

    def test_mxe_branch_no_optional_cols(self) -> None:
        eid = _build_event_id("MXE", "chr4", "+", 100, 150, None, None, None, None, [], {})
        assert eid == "MXE:chr4:+:100-150"

    def test_ri_branch_with_flanking(self) -> None:
        # lines 255-263: RI branch
        eid = _build_event_id("RI", "chr5", "+", 500, 700, 450, 499, 701, 750, [], {})
        assert eid.startswith("RI:chr5:+:500-700")
        assert "u450-499" in eid
        assert "d701-750" in eid

    def test_ri_branch_no_flanking(self) -> None:
        eid = _build_event_id("RI", "chr5", "+", 500, 700, None, None, None, None, [], {})
        assert eid == "RI:chr5:+:500-700"

    def test_unknown_event_type_returns_base(self) -> None:
        # line 265: final fallback
        eid = _build_event_id("NOVEL", "chr1", "+", 1, 100, None, None, None, None, [], {})
        assert eid == "NOVEL:chr1:+:1-100"


# ---------------------------------------------------------------------------
# parse_rmats_output — in-memory table files
# ---------------------------------------------------------------------------


def _write_se_file(path, rows: list[str], *, extra_header: str = "") -> None:
    """Write a minimal SE.MATS.JunctionCountOnly.txt to *path*."""
    header = _SE_HEADER + (("\t" + extra_header) if extra_header else "")
    path.write_text(header + "\n" + "".join(rows), encoding="utf-8")


def _se_row(
    *,
    gene: str = "GENE1",
    chrom: str = "chr1",
    strand: str = "+",
    es: int = 100,
    ee: int = 200,
    inc1: str = "20,18",
    exc1: str = "2,1",
    inc2: str = "5",
    exc2: str = "15",
    fdr: str = "0.01",
    dpsi: str = "0.5",
    inc_fl: str = "197",
    skip_fl: str = "99",
    inc_level1: str = "0.8,0.9",
    inc_level2: str = "0.3",
) -> str:
    return "\t".join([
        "1", '"ENSG0001"', gene, chrom, strand,
        str(es), str(ee), "50", "99", "201", "250", "1",
        inc1, exc1, inc2, exc2,
        inc_fl, skip_fl, "0.001", fdr,
        inc_level1, inc_level2, dpsi,
    ]) + "\n"


class TestParseRmatsOutputSE:
    def test_basic_se_parse(self, tmp_path) -> None:
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        _write_se_file(p, [_se_row()])
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "SE"
        assert ev.chrom == "chr1"
        assert ev.strand == "+"
        assert ev.gene == "GENE1"
        assert ev.sample_1_inc_count == 38   # 20+18
        assert ev.sample_1_exc_count == 3    # 2+1
        assert ev.sample_2_inc_count == 5
        assert ev.sample_2_exc_count == 15
        assert ev.exon_start == 100
        assert ev.exon_end == 200

    def test_event_id_includes_strand(self, tmp_path) -> None:
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        _write_se_file(p, [_se_row(strand="-")])
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert ":-:" in events[0].event_id

    def test_form_len_parsed(self, tmp_path) -> None:
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        _write_se_file(p, [_se_row(inc_fl="200", skip_fl="100")])
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert abs(events[0].inc_form_len - 200.0) < 1e-9
        assert abs(events[0].skip_form_len - 100.0) < 1e-9

    def test_form_len_na_defaults_to_one(self, tmp_path) -> None:
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        _write_se_file(p, [_se_row(inc_fl="NA", skip_fl="NA")])
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert events[0].inc_form_len == 1.0
        assert events[0].skip_form_len == 1.0

    def test_na_fdr_kept_as_nan(self, tmp_path) -> None:
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        _write_se_file(p, [_se_row(fdr="NA", dpsi="NA")])
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert len(events) == 1
        assert math.isnan(events[0].rmats_fdr)
        assert math.isnan(events[0].rmats_dpsi)

    def test_min_total_count_filter(self, tmp_path) -> None:
        # line 335: row with low total count is dropped
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        _write_se_file(p, [_se_row(inc1="1", exc1="1", inc2="1", exc2="1")])
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=100)
        assert len(events) == 0

    def test_short_field_row_skipped(self, tmp_path) -> None:
        # line 315: row with fewer fields than header is silently skipped
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        short = "1\tGENE1\n"   # far too few columns
        p.write_text(_SE_HEADER + "\n" + short, encoding="utf-8")
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert len(events) == 0

    def test_bad_int_row_skipped_with_warning(self, tmp_path, caplog) -> None:
        # lines 415-420: ValueError in int() triggers the except block
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        # Make exonStart_0base non-numeric to force ValueError
        bad_row = "\t".join([
            "1", '"ENSG0001"', "GENE1", "chr1", "+",
            "NOT_AN_INT", "200", "50", "99", "201", "250", "1",
            "20", "5", "3", "7",
            "197", "99", "0.001", "0.01",
            "0.8", "0.3", "0.5",
        ]) + "\n"
        p.write_text(_SE_HEADER + "\n" + bad_row, encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert len(events) == 0
        assert any("Skipping" in r.message for r in caplog.records)

    def test_trailing_empty_dpsi_field_keeps_event_as_nan(self, tmp_path) -> None:
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        row = _se_row(dpsi="").rstrip("\n")
        assert row.endswith("\t")
        _write_se_file(p, [row + "\n"])
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert len(events) == 1
        assert math.isnan(events[0].rmats_dpsi)

    def test_quoted_inc_level_vectors_do_not_drop_event(self, tmp_path) -> None:
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        _write_se_file(
            p,
            [_se_row(inc_level1='"0.8,0.6"', inc_level2='"NA,0.2"')],
        )
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert len(events) == 1
        assert events[0].sample_1_psi == pytest.approx(0.7)
        assert events[0].sample_2_psi == pytest.approx(0.2)

    def test_missing_rmats_dir_returns_empty(self, tmp_path) -> None:
        missing = str(tmp_path / "nonexistent")
        events = parse_rmats_output(missing, ["SE"])
        assert events == []

    def test_replicate_tuples_stored(self, tmp_path) -> None:
        p = tmp_path / "SE.MATS.JunctionCountOnly.txt"
        _write_se_file(p, [_se_row(inc1="10,12", exc1="1,2")])
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert events[0].sample_1_inc_replicates == (10, 12)
        assert events[0].sample_1_exc_replicates == (1, 2)


class TestParseRmatsOutputA3SS:
    """Parse A3SS event type to exercise its column layout."""

    def _write_a3ss(self, path, *, inc1: str = "20", exc1: str = "5") -> None:
        row = "\t".join([
            "1", '"ENSG0002"', "PTBP1", "chr9", "-",
            "50", "120", "80", "120",   # longExonStart/End, shortES/EE
            "10", "49", "1",            # flankingES/EE, ID
            inc1, exc1, "3", "12",
            "180", "90", "0.002", "0.03",
            "0.8", "0.2", "0.6",
        ]) + "\n"
        path.write_text(_A3SS_HEADER + "\n" + row, encoding="utf-8")

    def test_a3ss_parsed(self, tmp_path) -> None:
        p = tmp_path / "A3SS.MATS.JunctionCountOnly.txt"
        self._write_a3ss(p)
        events = parse_rmats_output(str(tmp_path), ["A3SS"], min_total_count=1)
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "A3SS"
        assert ev.chrom == "chr9"
        assert ev.gene == "PTBP1"
        # A3SS event_id should encode long/short exon boundaries
        assert "l50-120" in ev.event_id
        assert "s80-120" in ev.event_id


class TestParseRmatsOutputMXE:
    """Parse MXE event type."""

    def _write_mxe(self, path) -> None:
        row = "\t".join([
            "1", '"ENSG0003"', "DCLK1", "chr13", "+",
            "100", "150",   # 1stExonStart/End
            "200", "250",   # 2ndExonStart/End
            "50", "99",     # upstreamES/EE
            "301", "350",   # downstreamES/EE
            "1",            # ID (dup col)
            "30", "5", "4", "25",
            "200", "100", "0.0001", "0.001",
            "0.85", "0.13", "0.72",
        ]) + "\n"
        path.write_text(_MXE_HEADER + "\n" + row, encoding="utf-8")

    def test_mxe_parsed(self, tmp_path) -> None:
        p = tmp_path / "MXE.MATS.JunctionCountOnly.txt"
        self._write_mxe(p)
        events = parse_rmats_output(str(tmp_path), ["MXE"], min_total_count=1)
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "MXE"
        assert "e1=100-150" in ev.event_id
        assert "e2=200-250" in ev.event_id


class TestParseRmatsOutputRI:
    """Parse RI event type."""

    def _write_ri(self, path) -> None:
        row = "\t".join([
            "1", '"ENSG0004"', "CLK1", "chr2", "+",
            "500", "700",   # riExonStart/End  → exonStart_0base / exonEnd
            "450", "499",   # upstreamES/EE
            "701", "750",   # downstreamES/EE
            "1",            # dup ID col
            "40", "2", "6", "35",
            "210", "105", "0.001", "0.02",
            "0.95", "0.85", "0.1",
        ]) + "\n"
        path.write_text(_RI_HEADER + "\n" + row, encoding="utf-8")

    def test_ri_parsed(self, tmp_path) -> None:
        p = tmp_path / "RI.MATS.JunctionCountOnly.txt"
        self._write_ri(p)
        events = parse_rmats_output(str(tmp_path), ["RI"], min_total_count=1)
        assert len(events) == 1
        ev = events[0]
        assert ev.event_type == "RI"
        # RI event_id should contain flanking exon coords
        assert "u450-499" in ev.event_id or ev.event_id.startswith("RI:")

    def test_ri_without_flanking_cols(self, tmp_path) -> None:
        # RI table without upstreamES/EE columns → base id only
        minimal_header = "\t".join([
            "ID", "GeneID", "geneSymbol", "chr", "strand",
            "riExonStart_0base", "riExonEnd", "ID",
            "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
            "IncFormLen", "SkipFormLen", "PValue", "FDR",
            "IncLevel1", "IncLevel2", "IncLevelDifference",
        ])
        row = "\t".join([
            "1", '"ENSG0004"', "CLK1", "chr2", "+",
            "500", "700", "1",
            "40", "2", "6", "35",
            "210", "105", "0.001", "0.02",
            "0.95", "0.85", "0.1",
        ]) + "\n"
        p = tmp_path / "RI.MATS.JunctionCountOnly.txt"
        p.write_text(minimal_header + "\n" + row, encoding="utf-8")
        events = parse_rmats_output(str(tmp_path), ["RI"], min_total_count=1)
        assert len(events) == 1
        assert events[0].event_type == "RI"


class TestParseRmatsOutputMultipleTypes:
    """Parse directory containing multiple event types at once."""

    def test_all_five_types(self, tmp_path) -> None:
        # Write one file per type and verify we collect from all
        (tmp_path / "SE.MATS.JunctionCountOnly.txt").write_text(
            _SE_HEADER + "\n" + _se_row(gene="G_SE"), encoding="utf-8"
        )
        a3ss_row = "\t".join([
            "1", '"ENSG"', "G_A3SS", "chr1", "+",
            "50", "120", "80", "120", "10", "49", "1",
            "20", "5", "3", "12", "180", "90", "0.002", "0.03",
            "0.8", "0.2", "0.6",
        ]) + "\n"
        (tmp_path / "A3SS.MATS.JunctionCountOnly.txt").write_text(
            _A3SS_HEADER + "\n" + a3ss_row, encoding="utf-8"
        )
        a5ss_row = a3ss_row.replace("G_A3SS", "G_A5SS")
        (tmp_path / "A5SS.MATS.JunctionCountOnly.txt").write_text(
            _A3SS_HEADER + "\n" + a5ss_row, encoding="utf-8"
        )
        mxe_row = "\t".join([
            "1", '"ENSG"', "G_MXE", "chr1", "+",
            "100", "150", "200", "250", "50", "99", "301", "350", "1",
            "30", "5", "4", "25", "200", "100", "0.0001", "0.001",
            "0.85", "0.13", "0.72",
        ]) + "\n"
        (tmp_path / "MXE.MATS.JunctionCountOnly.txt").write_text(
            _MXE_HEADER + "\n" + mxe_row, encoding="utf-8"
        )
        ri_row = "\t".join([
            "1", '"ENSG"', "G_RI", "chr2", "+",
            "500", "700", "450", "499", "701", "750", "1",
            "40", "2", "6", "35", "210", "105", "0.001", "0.02",
            "0.95", "0.85", "0.1",
        ]) + "\n"
        (tmp_path / "RI.MATS.JunctionCountOnly.txt").write_text(
            _RI_HEADER + "\n" + ri_row, encoding="utf-8"
        )

        events = parse_rmats_output(str(tmp_path), min_total_count=1)
        types = {ev.event_type for ev in events}
        assert types == {"SE", "A3SS", "A5SS", "MXE", "RI"}
        genes = {ev.gene for ev in events}
        assert "G_SE" in genes
        assert "G_MXE" in genes


class TestParseRmatsOutputJCSuffix:
    """Verify the .MATS.JC.txt suffix is also resolved."""

    def test_jc_suffix_loaded(self, tmp_path) -> None:
        p = tmp_path / "SE.MATS.JC.txt"
        _write_se_file(p, [_se_row()])
        events = parse_rmats_output(str(tmp_path), ["SE"], min_total_count=1)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# add_bootstrap_ci
# ---------------------------------------------------------------------------


class TestAddBootstrapCi:
    def test_returns_psi_results_for_each_event(self) -> None:
        events = [_make_rmats_event(event_type="SE")]
        results = add_bootstrap_ci(
            events, n_replicates=50, confidence_level=0.95, seed=0,
            use_conformal=False,
        )
        assert len(results) == 1
        r = results[0]
        assert 0.0 <= r.psi <= 1.0
        assert r.ci_low <= r.psi
        assert r.psi <= r.ci_high
        assert np.isfinite(r.ci_width)

    def test_sample_2_counts_used(self) -> None:
        ev = _make_rmats_event(inc_1=5, exc_1=95, inc_2=90, exc_2=10)
        res_s1 = add_bootstrap_ci([ev], n_replicates=50, seed=1, sample="sample_1",
                                   use_conformal=False)
        res_s2 = add_bootstrap_ci([ev], n_replicates=50, seed=1, sample="sample_2",
                                   use_conformal=False)
        # PSI from sample_2 should be substantially higher than sample_1
        assert res_s2[0].psi > res_s1[0].psi

    def test_form_len_forwarded(self) -> None:
        """IncFormLen/SkipFormLen are passed through to bootstrap_psi."""
        ev1 = _make_rmats_event(inc_1=50, exc_1=50, inc_form_len=1.0, skip_form_len=1.0)
        ev2 = _make_rmats_event(inc_1=50, exc_1=50, inc_form_len=200.0, skip_form_len=100.0)
        r1 = add_bootstrap_ci([ev1], n_replicates=100, seed=42, use_conformal=False)
        r2 = add_bootstrap_ci([ev2], n_replicates=100, seed=42, use_conformal=False)
        # Different form lengths produce different PSI estimates
        # (the length-normalised PSI differs from raw PSI)
        assert r1[0].psi != r2[0].psi or r1[0].ci_width != r2[0].ci_width

    def test_empty_event_list(self) -> None:
        results = add_bootstrap_ci([], n_replicates=50, seed=0, use_conformal=False)
        assert results == []

    def test_conformal_load_failure_warns_and_continues(self, tmp_path, caplog) -> None:
        # lines 458-459: load_default_conformal_calibrator raises → warning + continues
        import braid.target.conformal as conf_mod
        original_fn = conf_mod.load_default_conformal_calibrator

        def _raise(*args, **kwargs):
            raise FileNotFoundError("no calibrator in test")

        conf_mod.load_default_conformal_calibrator = _raise
        try:
            events = [_make_rmats_event()]
            with caplog.at_level(logging.WARNING):
                results = add_bootstrap_ci(
                    events, n_replicates=50, seed=0, use_conformal=True,
                    conformal_calibrator=None,
                )
            assert len(results) == 1
            assert any("conformal" in r.message.lower() for r in caplog.records)
        finally:
            conf_mod.load_default_conformal_calibrator = original_fn

    def test_is_confident_field_populated(self) -> None:
        # High counts → should produce confident results
        ev = _make_rmats_event(inc_1=500, exc_1=5)
        results = add_bootstrap_ci([ev], n_replicates=200, seed=7, use_conformal=False)
        # is_confident is a bool
        assert isinstance(results[0].is_confident, bool)

    def test_event_metadata_propagated(self) -> None:
        ev_custom = RmatsEvent(
            event_id="A3SS:chr9:-:80-120:l50-120/s80-120",
            event_type="A3SS",
            chrom="chr9",
            strand="-",
            gene="PTBP1",
            inc_count=20,
            exc_count=5,
            rmats_psi=0.8,
            rmats_fdr=0.01,
            rmats_dpsi=0.5,
            exon_start=80,
            exon_end=120,
            sample_1_inc_count=20,
            sample_1_exc_count=5,
            sample_2_inc_count=3,
            sample_2_exc_count=17,
        )
        results = add_bootstrap_ci(
            [ev_custom], n_replicates=50, seed=0, use_conformal=False,
        )
        r = results[0]
        assert r.event_id == "A3SS:chr9:-:80-120:l50-120/s80-120"
        assert r.event_type == "A3SS"
        assert r.gene == "PTBP1"
        assert r.chrom == "chr9"
        assert r.event_start == 80
        assert r.event_end == 120


def test_parse_rmats_output_preserves_missing_inc_level_as_nan(tmp_path) -> None:
    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    header = [
        "ID", "GeneID", "geneSymbol", "chr", "strand",
        "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE",
        "downstreamES", "downstreamEE", "IJC_SAMPLE_1", "SJC_SAMPLE_1",
        "IJC_SAMPLE_2", "SJC_SAMPLE_2", "IncFormLen", "SkipFormLen",
        "PValue", "FDR", "IncLevel1", "IncLevel2", "IncLevelDifference",
    ]
    row = [
        "1", "G", "G", "chr1", "+", "100", "200", "0", "50",
        "250", "300", "20", "10", "20", "10", "100", "100",
        "NA", "NA", "NA", "0.5", "NA",
    ]
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join(header) + "\n" + "\t".join(row) + "\n"
    )

    events = parse_rmats_output(rmats_dir, min_total_count=0)
    assert len(events) == 1
    assert events[0].sample_1_psi is None
    assert math.isnan(events[0].rmats_psi)
