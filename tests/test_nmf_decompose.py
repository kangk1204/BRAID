"""Tests for NMF-based transcript decomposition.

Exercises the read-fragment matrix construction, NMF factorization,
isoform extraction, and the full decompose_nmf pipeline.
"""

from __future__ import annotations

import numpy as np

from rapidsplice.flow.nmf_decompose import (
    NMFDecomposeConfig,
    _build_fragments,
    _build_read_fragment_matrix,
    _estimate_isoform_weights,
    _extract_isoforms_from_H,
    _merge_adjacent_fragments,
    _nmf_multiplicative,
    _select_k,
    decompose_nmf,
)
from rapidsplice.graph.splice_graph import CSRGraph, NodeType, SpliceGraph

# ===================================================================
# Helpers
# ===================================================================


def _make_csr_graph(
    n_nodes: int,
    edges: list[tuple[int, int, float]],
    node_starts: list[int],
    node_ends: list[int],
    node_types: list[int],
) -> CSRGraph:
    """Build a CSRGraph from explicit edge list."""
    adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n_nodes)}
    for u, v, w in edges:
        adj[u].append((v, w))

    row_offsets = [0]
    col_indices = []
    edge_weights = []
    for u in range(n_nodes):
        neighbors = sorted(adj[u], key=lambda x: x[0])
        for v, w in neighbors:
            col_indices.append(v)
            edge_weights.append(w)
        row_offsets.append(len(col_indices))

    return CSRGraph(
        row_offsets=np.array(row_offsets, dtype=np.int32),
        col_indices=np.array(col_indices, dtype=np.int32),
        edge_weights=np.array(edge_weights, dtype=np.float32),
        edge_coverages=np.array(edge_weights, dtype=np.float32),
        node_coverages=np.zeros(n_nodes, dtype=np.float32),
        node_starts=np.array(node_starts, dtype=np.int64),
        node_ends=np.array(node_ends, dtype=np.int64),
        node_types=np.array(node_types, dtype=np.int8),
        n_nodes=n_nodes,
        n_edges=len(edges),
    )


def _make_splice_graph(
    chrom: str = "chr1",
    strand: str = "+",
    locus_start: int = 0,
    locus_end: int = 1000,
) -> SpliceGraph:
    """Create a minimal SpliceGraph for metadata."""
    return SpliceGraph(
        chrom=chrom, strand=strand,
        locus_start=locus_start, locus_end=locus_end,
    )


# ===================================================================
# Tests: _build_fragments
# ===================================================================


class TestBuildFragments:
    """Test fragment extraction from splice graph nodes."""

    def test_simple_two_exon(self) -> None:
        """Two exon nodes produce two fragments."""
        csr = _make_csr_graph(
            n_nodes=4,
            edges=[(0, 1, 10), (1, 2, 10), (2, 3, 10)],
            node_starts=[0, 100, 300, 500],
            node_ends=[0, 200, 400, 500],
            node_types=[NodeType.SOURCE, NodeType.EXON, NodeType.EXON, NodeType.SINK],
        )
        graph = _make_splice_graph()
        frags = _build_fragments(graph, csr)
        assert len(frags) == 2
        assert frags[0] == (100, 200)
        assert frags[1] == (300, 400)

    def test_no_exon_nodes(self) -> None:
        """Graph with only source/sink produces no fragments."""
        csr = _make_csr_graph(
            n_nodes=2,
            edges=[(0, 1, 10)],
            node_starts=[0, 0],
            node_ends=[0, 0],
            node_types=[NodeType.SOURCE, NodeType.SINK],
        )
        graph = _make_splice_graph()
        frags = _build_fragments(graph, csr)
        assert len(frags) == 0

    def test_duplicate_fragments_deduplicated(self) -> None:
        """Duplicate exon coordinates produce unique fragments."""
        csr = _make_csr_graph(
            n_nodes=5,
            edges=[(0, 1, 10), (0, 2, 10), (1, 3, 10), (2, 3, 10), (3, 4, 10)],
            node_starts=[0, 100, 100, 300, 500],
            node_ends=[0, 200, 200, 400, 500],
            node_types=[
                NodeType.SOURCE, NodeType.EXON, NodeType.EXON,
                NodeType.EXON, NodeType.SINK,
            ],
        )
        graph = _make_splice_graph()
        frags = _build_fragments(graph, csr)
        assert len(frags) == 2  # (100,200) and (300,400)


# ===================================================================
# Tests: _build_read_fragment_matrix
# ===================================================================


class TestBuildReadFragmentMatrix:
    """Test read-fragment coverage matrix construction."""

    def test_simple_overlap(self) -> None:
        """Reads overlapping fragments are marked correctly."""
        fragments = [(100, 200), (300, 400), (500, 600)]
        read_positions = np.array([90, 290, 490], dtype=np.int64)
        read_ends = np.array([210, 410, 610], dtype=np.int64)

        X = _build_read_fragment_matrix(
            read_positions, read_ends, None, None, fragments,
        )
        assert X.shape == (3, 3)
        assert X[0, 0] == 1.0  # Read 0 overlaps fragment 0
        assert X[0, 1] == 0.0  # Read 0 doesn't overlap fragment 1
        assert X[1, 1] == 1.0  # Read 1 overlaps fragment 1
        assert X[2, 2] == 1.0  # Read 2 overlaps fragment 2

    def test_spanning_read(self) -> None:
        """A read spanning multiple fragments marks all of them."""
        fragments = [(100, 200), (300, 400), (500, 600)]
        read_positions = np.array([50], dtype=np.int64)
        read_ends = np.array([650], dtype=np.int64)

        X = _build_read_fragment_matrix(
            read_positions, read_ends, None, None, fragments,
        )
        assert X.shape == (1, 3)
        np.testing.assert_array_equal(X[0], [1.0, 1.0, 1.0])

    def test_spliced_read_excludes_intron_fragments(self) -> None:
        """Spliced reads exclude fragments within the intron."""
        fragments = [(100, 200), (250, 350), (400, 500)]
        read_positions = np.array([90], dtype=np.int64)
        read_ends = np.array([510], dtype=np.int64)
        # Junction from 200 to 400 (skips fragment at 250-350)
        junction_starts = [[200]]
        junction_ends = [[400]]

        X = _build_read_fragment_matrix(
            read_positions, read_ends, junction_starts, junction_ends,
            fragments,
        )
        assert X.shape == (1, 3)
        assert X[0, 0] == 1.0  # Before intron
        assert X[0, 1] == 0.0  # Within intron — excluded
        assert X[0, 2] == 1.0  # After intron

    def test_empty_reads(self) -> None:
        """Empty read arrays produce a valid matrix."""
        fragments = [(100, 200)]
        X = _build_read_fragment_matrix(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            None, None, fragments,
        )
        assert X.shape[0] >= 1  # At least 1 row (padding)
        assert X.shape[1] >= 1


# ===================================================================
# Tests: _nmf_multiplicative
# ===================================================================


class TestNMFMultiplicative:
    """Test NMF factorization."""

    def test_rank_one_recovery(self) -> None:
        """NMF with K=1 recovers a rank-1 matrix approximately."""
        # Create a rank-1 matrix
        w = np.array([[3], [1], [2]], dtype=np.float64)
        h = np.array([[1, 2, 1]], dtype=np.float64)
        X = w @ h  # 3x3 rank-1 matrix

        W_est, H_est, err = _nmf_multiplicative(X.astype(np.float32), k=1, max_iter=200)
        reconstruction = W_est @ H_est
        # Relative error should be small
        rel_err = np.linalg.norm(X - reconstruction) / np.linalg.norm(X)
        assert rel_err < 0.1

    def test_shapes(self) -> None:
        """Output shapes match expectations."""
        m, n, k = 10, 5, 3
        X = np.random.rand(m, n).astype(np.float32)
        W, H, err = _nmf_multiplicative(X, k=k, max_iter=50)
        assert W.shape == (m, k)
        assert H.shape == (k, n)
        assert err >= 0

    def test_non_negative(self) -> None:
        """W and H are non-negative."""
        X = np.random.rand(8, 4).astype(np.float32)
        W, H, _ = _nmf_multiplicative(X, k=2, max_iter=50)
        assert np.all(W >= 0)
        assert np.all(H >= 0)


# ===================================================================
# Tests: _select_k
# ===================================================================


class TestSelectK:
    """Test BIC-based K selection."""

    def test_rank_one_selects_k1(self) -> None:
        """A pure rank-1 matrix should select K=1."""
        w = np.array([[3], [1], [2], [4], [1]], dtype=np.float64)
        h = np.array([[1, 2, 3, 1]], dtype=np.float64)
        X = (w @ h).astype(np.float32)
        k = _select_k(X, max_k=5, max_iter=50)
        assert k == 1

    def test_max_k_respected(self) -> None:
        """Selected K never exceeds max_k."""
        X = np.random.rand(10, 5).astype(np.float32)
        k = _select_k(X, max_k=3, max_iter=30)
        assert 1 <= k <= 3


# ===================================================================
# Tests: _extract_isoforms_from_H
# ===================================================================


class TestExtractIsoformsFromH:
    """Test isoform extraction from the H matrix."""

    def test_two_isoforms(self) -> None:
        """Two distinct H rows produce two distinct isoforms."""
        fragments = [(100, 200), (300, 400), (500, 600)]
        # Isoform 1: uses all 3 fragments
        # Isoform 2: uses fragments 0 and 2 (skips middle)
        H = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ], dtype=np.float64)

        isoforms = _extract_isoforms_from_H(H, fragments, threshold=0.3)
        assert len(isoforms) == 2

        # One of them should have 2 exons (skipped middle)
        lens = sorted([len(isoforms[0]), len(isoforms[1])])
        assert lens == [2, 3] or lens == [1, 2]

    def test_empty_row_skipped(self) -> None:
        """Zero-valued H rows produce no isoform."""
        fragments = [(100, 200)]
        H = np.array([[0.0], [1.0]], dtype=np.float64)
        isoforms = _extract_isoforms_from_H(H, fragments, threshold=0.3)
        assert len(isoforms) == 1

    def test_adjacent_merge(self) -> None:
        """Adjacent fragments are merged into single exons."""
        fragments = [(100, 200), (200, 300), (400, 500)]
        H = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        isoforms = _extract_isoforms_from_H(H, fragments, threshold=0.3)
        assert len(isoforms) == 1
        # First two fragments should merge: (100, 300), then (400, 500)
        assert len(isoforms[0]) == 2
        assert isoforms[0][0] == (100, 300)
        assert isoforms[0][1] == (400, 500)


# ===================================================================
# Tests: _estimate_isoform_weights
# ===================================================================


class TestEstimateIsoformWeights:
    """Test NNLS-based isoform weight estimation."""

    def test_single_isoform(self) -> None:
        """Single isoform gets all the weight."""
        fragments = [(100, 200), (300, 400)]
        isoforms = [[(100, 200), (300, 400)]]
        # 10 reads each covering both fragments
        X = np.ones((10, 2), dtype=np.float32)
        weights = _estimate_isoform_weights(X, isoforms, fragments)
        assert len(weights) == 1
        assert weights[0] > 0

    def test_two_isoforms_different_coverage(self) -> None:
        """Two isoforms get proportional weights."""
        fragments = [(100, 200), (300, 400), (500, 600)]
        isoforms = [
            [(100, 200), (300, 400), (500, 600)],  # uses all 3
            [(100, 200), (500, 600)],                # skips middle
        ]
        # 20 reads covering all 3, 10 reads covering only 0 and 2
        X = np.zeros((30, 3), dtype=np.float32)
        X[:20, :] = 1.0       # First 20 reads cover all fragments
        X[20:, 0] = 1.0       # Next 10 cover fragment 0
        X[20:, 2] = 1.0       # and fragment 2

        weights = _estimate_isoform_weights(X, isoforms, fragments)
        assert len(weights) == 2
        assert weights[0] > 0
        assert weights[1] > 0


# ===================================================================
# Tests: _merge_adjacent_fragments
# ===================================================================


class TestMergeAdjacentFragments:
    """Test fragment merging utility."""

    def test_adjacent(self) -> None:
        """Adjacent fragments are merged."""
        result = _merge_adjacent_fragments([(100, 200), (200, 300)])
        assert result == [(100, 300)]

    def test_gap(self) -> None:
        """Non-adjacent fragments remain separate."""
        result = _merge_adjacent_fragments([(100, 200), (400, 500)])
        assert result == [(100, 200), (400, 500)]

    def test_empty(self) -> None:
        """Empty input produces empty output."""
        assert _merge_adjacent_fragments([]) == []

    def test_overlapping(self) -> None:
        """Overlapping fragments are merged."""
        result = _merge_adjacent_fragments([(100, 250), (200, 300)])
        assert result == [(100, 300)]


# ===================================================================
# Tests: decompose_nmf (integration)
# ===================================================================


class TestDecomposeNMF:
    """Integration tests for the full NMF decomposition pipeline."""

    def _make_two_exon_graph(self) -> tuple[CSRGraph, SpliceGraph]:
        """Build a simple 2-exon splice graph."""
        csr = _make_csr_graph(
            n_nodes=4,
            edges=[(0, 1, 20), (1, 2, 20), (2, 3, 20)],
            node_starts=[0, 100, 300, 500],
            node_ends=[0, 200, 400, 500],
            node_types=[
                NodeType.SOURCE, NodeType.EXON,
                NodeType.EXON, NodeType.SINK,
            ],
        )
        graph = _make_splice_graph()
        return csr, graph

    def test_basic_decomposition(self) -> None:
        """NMF produces at least one transcript for a simple graph."""
        csr, graph = self._make_two_exon_graph()

        # Reads covering both exons
        read_pos = np.array([90, 95, 100, 110, 290, 300], dtype=np.int64)
        read_ends = np.array([210, 210, 210, 210, 410, 410], dtype=np.int64)

        transcripts = decompose_nmf(
            csr, graph,
            read_positions=read_pos,
            read_ends=read_ends,
            config=NMFDecomposeConfig(min_transcript_coverage=0.5),
        )
        assert len(transcripts) >= 1
        # All transcripts should have exon coordinates
        for tx in transcripts:
            assert len(tx.exon_coords) > 0

    def test_no_reads_uses_graph_weights(self) -> None:
        """No reads still produces transcripts from graph edge weights."""
        csr, graph = self._make_two_exon_graph()
        transcripts = decompose_nmf(
            csr, graph,
            read_positions=np.array([], dtype=np.int64),
            read_ends=np.array([], dtype=np.int64),
        )
        # Hybrid uses graph edge weights for NNLS, so transcripts are still found
        assert len(transcripts) >= 1

    def test_no_fragments_returns_empty(self) -> None:
        """Graph with no exon nodes returns empty."""
        csr = _make_csr_graph(
            n_nodes=2,
            edges=[(0, 1, 10)],
            node_starts=[0, 0],
            node_ends=[0, 0],
            node_types=[NodeType.SOURCE, NodeType.SINK],
        )
        graph = _make_splice_graph()
        transcripts = decompose_nmf(
            csr, graph,
            read_positions=np.array([10], dtype=np.int64),
            read_ends=np.array([20], dtype=np.int64),
        )
        assert len(transcripts) == 0

    def test_transcripts_sorted_by_weight(self) -> None:
        """Returned transcripts are sorted by descending weight."""
        csr, graph = self._make_two_exon_graph()
        read_pos = np.array([90] * 50, dtype=np.int64)
        read_ends = np.array([410] * 50, dtype=np.int64)

        transcripts = decompose_nmf(
            csr, graph,
            read_positions=read_pos,
            read_ends=read_ends,
            config=NMFDecomposeConfig(min_transcript_coverage=0.1),
        )
        if len(transcripts) > 1:
            for i in range(len(transcripts) - 1):
                assert transcripts[i].weight >= transcripts[i + 1].weight

    def test_with_junctions(self) -> None:
        """NMF handles spliced reads with junction information."""
        # 3 exon graph with alternative splicing
        csr = _make_csr_graph(
            n_nodes=5,
            edges=[
                (0, 1, 20), (1, 2, 10), (1, 3, 10),
                (2, 4, 10), (3, 4, 10),
            ],
            node_starts=[0, 100, 300, 500, 700],
            node_ends=[0, 200, 400, 600, 700],
            node_types=[
                NodeType.SOURCE, NodeType.EXON, NodeType.EXON,
                NodeType.EXON, NodeType.SINK,
            ],
        )
        graph = _make_splice_graph()

        # Some reads span exon 1->2, others span exon 1->3
        read_pos = np.array([100, 100, 100, 100, 100, 100], dtype=np.int64)
        read_ends = np.array([410, 410, 410, 610, 610, 610], dtype=np.int64)
        # Junction starts/ends for spliced reads
        junc_starts = [[200], [200], [200], [200], [200], [200]]
        junc_ends = [[300], [300], [300], [500], [500], [500]]

        transcripts = decompose_nmf(
            csr, graph,
            read_positions=read_pos,
            read_ends=read_ends,
            read_junction_starts=junc_starts,
            read_junction_ends=junc_ends,
            config=NMFDecomposeConfig(min_transcript_coverage=0.1),
        )
        assert len(transcripts) >= 1

    def test_config_defaults(self) -> None:
        """NMFDecomposeConfig has reasonable defaults."""
        cfg = NMFDecomposeConfig()
        assert cfg.max_isoforms == 20
        assert cfg.nmf_max_iter == 200
        assert cfg.fragment_threshold == 0.3
        assert cfg.min_transcript_coverage == 1.0
        assert cfg.min_relative_abundance == 0.02
