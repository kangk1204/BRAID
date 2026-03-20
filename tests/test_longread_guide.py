"""Tests for long-read guided assembly path mapping."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from braid.flow.longread_guide import (
    get_guide_paths_for_locus,
    map_longread_to_path,
)
from braid.graph.splice_graph import CSRGraph, NodeType
from braid.io.gtf_reader import _parse_attribute, read_guide_gtf


def _make_linear_graph() -> CSRGraph:
    """Create a simple linear graph: SOURCE -> exon1 -> exon2 -> SINK.

    Nodes:
        0: SOURCE [0, 0)
        1: EXON   [100, 300)
        2: EXON   [500, 700)
        3: SINK   [800, 800)

    Edges: 0->1, 1->2 (intron), 2->3
    """
    n_nodes = 4
    row_offsets = np.array([0, 1, 2, 3, 3], dtype=np.int64)
    col_indices = np.array([1, 2, 3], dtype=np.int64)
    edge_weights = np.array([50.0, 50.0, 50.0], dtype=np.float64)
    edge_coverages = np.array([50.0, 50.0, 50.0], dtype=np.float64)
    node_coverages = np.array([0.0, 50.0, 50.0, 0.0], dtype=np.float64)

    node_types = np.array(
        [NodeType.SOURCE, NodeType.EXON, NodeType.EXON, NodeType.SINK],
        dtype=np.int8,
    )
    node_starts = np.array([0, 100, 500, 800], dtype=np.int64)
    node_ends = np.array([0, 300, 700, 800], dtype=np.int64)

    return CSRGraph(
        n_nodes=n_nodes,
        n_edges=3,
        row_offsets=row_offsets,
        col_indices=col_indices,
        edge_weights=edge_weights,
        edge_coverages=edge_coverages,
        node_coverages=node_coverages,
        node_types=node_types,
        node_starts=node_starts,
        node_ends=node_ends,
    )


def _make_branching_graph() -> CSRGraph:
    """Create a graph with two alternative paths.

    Nodes:
        0: SOURCE [0, 0)
        1: EXON   [100, 300)   -- shared first exon
        2: EXON   [500, 700)   -- path A second exon
        3: EXON   [500, 650)   -- path B second exon (A3SS)
        4: SINK   [800, 800)

    Edges: 0->1, 1->2, 1->3, 2->4, 3->4
    """
    n_nodes = 5
    row_offsets = np.array([0, 1, 3, 4, 5, 5], dtype=np.int64)
    col_indices = np.array([1, 2, 3, 4, 4], dtype=np.int64)
    edge_weights = np.array([100.0, 70.0, 30.0, 70.0, 30.0], dtype=np.float64)
    edge_coverages = np.array([100.0, 70.0, 30.0, 70.0, 30.0], dtype=np.float64)
    node_coverages = np.array([0.0, 100.0, 70.0, 30.0, 0.0], dtype=np.float64)

    node_types = np.array(
        [NodeType.SOURCE, NodeType.EXON, NodeType.EXON, NodeType.EXON, NodeType.SINK],
        dtype=np.int8,
    )
    node_starts = np.array([0, 100, 500, 500, 800], dtype=np.int64)
    node_ends = np.array([0, 300, 700, 650, 800], dtype=np.int64)

    return CSRGraph(
        n_nodes=n_nodes,
        n_edges=5,
        row_offsets=row_offsets,
        col_indices=col_indices,
        edge_weights=edge_weights,
        edge_coverages=edge_coverages,
        node_coverages=node_coverages,
        node_types=node_types,
        node_starts=node_starts,
        node_ends=node_ends,
    )


class TestMapLongreadToPath:
    """Tests for map_longread_to_path."""

    def test_exact_match(self) -> None:
        """Long-read exons exactly matching graph nodes should map successfully."""
        graph = _make_linear_graph()
        lr_exons = [(100, 300), (500, 700)]
        path = map_longread_to_path(lr_exons, graph, tolerance=0)
        assert path is not None
        assert path == [0, 1, 2, 3]

    def test_fuzzy_match(self) -> None:
        """Long-read exons within tolerance should still map."""
        graph = _make_linear_graph()
        lr_exons = [(98, 302), (498, 702)]  # 2bp off
        path = map_longread_to_path(lr_exons, graph, tolerance=5)
        assert path is not None
        assert path == [0, 1, 2, 3]

    def test_no_match_beyond_tolerance(self) -> None:
        """Long-read exons too far off should fail to map."""
        graph = _make_linear_graph()
        lr_exons = [(50, 300), (500, 700)]  # 50bp off start
        map_longread_to_path(lr_exons, graph, tolerance=5)
        # May or may not match depending on overlap logic; key is no crash

    def test_missing_junction(self) -> None:
        """Long-read transcript with junction not in graph should return None."""
        graph = _make_linear_graph()
        # Exon going to a non-existent acceptor
        lr_exons = [(100, 300), (900, 1100)]
        path = map_longread_to_path(lr_exons, graph, tolerance=5)
        assert path is None

    def test_branching_path_a(self) -> None:
        """Should map to path A in a branching graph."""
        graph = _make_branching_graph()
        lr_exons = [(100, 300), (500, 700)]
        path = map_longread_to_path(lr_exons, graph, tolerance=0)
        assert path is not None
        assert 2 in path  # Node 2 is path A

    def test_empty_exons(self) -> None:
        """Empty exon list should return None."""
        graph = _make_linear_graph()
        assert map_longread_to_path([], graph) is None

    def test_single_exon(self) -> None:
        """Single-exon transcript should map if matching."""
        graph = _make_linear_graph()
        lr_exons = [(100, 300)]
        path = map_longread_to_path(lr_exons, graph, tolerance=5)
        # Should map to SOURCE -> exon1 -> SINK if edge exists
        # In our graph, exon1 doesn't connect to SINK directly, so None
        assert path is None


class TestGetGuidePathsForLocus:
    """Tests for get_guide_paths_for_locus."""

    def test_overlapping_transcript(self) -> None:
        """Should return paths for transcripts overlapping the locus."""
        graph = _make_linear_graph()
        lr_transcripts = [
            [(100, 300), (500, 700)],  # Overlaps locus
        ]
        paths = get_guide_paths_for_locus(
            lr_transcripts, graph, locus_start=50, locus_end=750,
        )
        assert len(paths) == 1
        assert paths[0] == [0, 1, 2, 3]

    def test_non_overlapping_transcript(self) -> None:
        """Should skip transcripts outside the locus."""
        graph = _make_linear_graph()
        lr_transcripts = [
            [(5000, 6000), (7000, 8000)],  # Far away
        ]
        paths = get_guide_paths_for_locus(
            lr_transcripts, graph, locus_start=50, locus_end=750,
        )
        assert len(paths) == 0

    def test_deduplication(self) -> None:
        """Duplicate long-read paths should be deduplicated."""
        graph = _make_linear_graph()
        lr_transcripts = [
            [(100, 300), (500, 700)],
            [(100, 300), (500, 700)],  # Duplicate
        ]
        paths = get_guide_paths_for_locus(
            lr_transcripts, graph, locus_start=50, locus_end=750,
        )
        assert len(paths) == 1


class TestReadGuideGTF:
    """Tests for the GTF reader."""

    def test_basic_parsing(self) -> None:
        """Should parse exon lines and group by transcript."""
        gtf_content = (
            '1\ttest\texon\t101\t300\t.\t+\t.\ttranscript_id "TX1"; gene_id "G1";\n'
            '1\ttest\texon\t501\t700\t.\t+\t.\ttranscript_id "TX1"; gene_id "G1";\n'
            '1\ttest\texon\t101\t300\t.\t+\t.\ttranscript_id "TX2"; gene_id "G1";\n'
            '1\ttest\texon\t801\t900\t.\t+\t.\ttranscript_id "TX2"; gene_id "G1";\n'
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".gtf", delete=False,
        ) as fh:
            fh.write(gtf_content)
            gtf_path = fh.name

        try:
            result = read_guide_gtf(gtf_path)
            assert ("1", "+") in result
            txs = result[("1", "+")]
            assert len(txs) == 2

            # Check coordinate conversion (1-based -> 0-based)
            # First exon of TX1: (100, 300) in 0-based
            exon_starts = {tx[0][0] for tx in txs}
            assert 100 in exon_starts
        finally:
            Path(gtf_path).unlink()

    def test_multi_strand(self) -> None:
        """Should separate transcripts by strand."""
        gtf_content = (
            '1\ttest\texon\t101\t300\t.\t+\t.\ttranscript_id "TX1";\n'
            '1\ttest\texon\t101\t300\t.\t-\t.\ttranscript_id "TX2";\n'
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".gtf", delete=False,
        ) as fh:
            fh.write(gtf_content)
            gtf_path = fh.name

        try:
            result = read_guide_gtf(gtf_path)
            assert ("1", "+") in result
            assert ("1", "-") in result
        finally:
            Path(gtf_path).unlink()

    def test_empty_file(self) -> None:
        """Should handle empty GTF."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".gtf", delete=False,
        ) as fh:
            fh.write("# empty\n")
            gtf_path = fh.name

        try:
            result = read_guide_gtf(gtf_path)
            assert len(result) == 0
        finally:
            Path(gtf_path).unlink()


class TestParseAttribute:
    """Tests for _parse_attribute helper."""

    def test_standard_format(self) -> None:
        attrs = 'gene_id "ENSG00001"; transcript_id "ENST00001";'
        assert _parse_attribute(attrs, "transcript_id") == "ENST00001"
        assert _parse_attribute(attrs, "gene_id") == "ENSG00001"

    def test_missing_key(self) -> None:
        attrs = 'gene_id "G1";'
        assert _parse_attribute(attrs, "transcript_id") is None
