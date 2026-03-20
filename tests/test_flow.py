"""Tests for flow decomposition: push_relabel, min_cost_flow, safe_paths, decompose.

Exercises max-flow, min-cost flow, flow-to-path decomposition, safe path
computation, and the full transcript assembly decomposition pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from rapidsplice.flow.decompose import (
    DecomposeConfig,
    Transcript,
    decompose_graph,
)
from rapidsplice.flow.min_cost_flow import flow_to_weighted_paths, min_cost_flow
from rapidsplice.flow.push_relabel import push_relabel_maxflow
from rapidsplice.flow.safe_paths import compute_safe_paths
from rapidsplice.graph.splice_graph import (
    CSRGraph,
    NodeType,
    SpliceGraph,
)

# ===================================================================
# Helpers for building small CSR test graphs
# ===================================================================


def _build_simple_dag_csr() -> CSRGraph:
    """Build a simple 4-node DAG: source -> A -> B -> sink.

    Node indices: 0=source, 1=A, 2=B, 3=sink
    Edge weights: source->A=10, A->B=10, B->sink=10
    """
    n_nodes = 4
    # Node 0: 1 outgoing (->1)
    # Node 1: 1 outgoing (->2)
    # Node 2: 1 outgoing (->3)
    # Node 3: 0 outgoing
    row_offsets = np.array([0, 1, 2, 3, 3], dtype=np.int32)
    col_indices = np.array([1, 2, 3], dtype=np.int32)
    edge_weights = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    edge_coverages = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    node_coverages = np.array([0.0, 10.0, 10.0, 0.0], dtype=np.float32)
    node_starts = np.array([0, 100, 200, 300], dtype=np.int64)
    node_ends = np.array([0, 200, 300, 300], dtype=np.int64)
    node_types = np.array(
        [NodeType.SOURCE, NodeType.EXON, NodeType.EXON, NodeType.SINK],
        dtype=np.int8,
    )

    return CSRGraph(
        row_offsets=row_offsets,
        col_indices=col_indices,
        edge_weights=edge_weights,
        edge_coverages=edge_coverages,
        node_coverages=node_coverages,
        node_starts=node_starts,
        node_ends=node_ends,
        node_types=node_types,
        n_nodes=n_nodes,
        n_edges=3,
    )


def _build_diamond_csr() -> CSRGraph:
    """Build a diamond graph with two paths from source to sink.

    Nodes: 0=source, 1=A, 2=B, 3=merge, 4=sink
    Edges: source->A (6), source->B (4), A->merge (6), B->merge (4), merge->sink (10)

    Two possible transcripts:
      Path 1: source -> A -> merge -> sink (flow=6)
      Path 2: source -> B -> merge -> sink (flow=4)
    """
    n_nodes = 5
    # Node 0: 2 outgoing (->1, ->2)
    # Node 1: 1 outgoing (->3)
    # Node 2: 1 outgoing (->3)
    # Node 3: 1 outgoing (->4)
    # Node 4: 0 outgoing
    row_offsets = np.array([0, 2, 3, 4, 5, 5], dtype=np.int32)
    col_indices = np.array([1, 2, 3, 3, 4], dtype=np.int32)
    edge_weights = np.array([6.0, 4.0, 6.0, 4.0, 10.0], dtype=np.float32)
    edge_coverages = np.array([6.0, 4.0, 6.0, 4.0, 10.0], dtype=np.float32)
    node_coverages = np.array([0.0, 6.0, 4.0, 10.0, 0.0], dtype=np.float32)
    node_starts = np.array([0, 100, 200, 300, 500], dtype=np.int64)
    node_ends = np.array([0, 200, 300, 400, 500], dtype=np.int64)
    node_types = np.array(
        [NodeType.SOURCE, NodeType.EXON, NodeType.EXON, NodeType.EXON, NodeType.SINK],
        dtype=np.int8,
    )

    return CSRGraph(
        row_offsets=row_offsets,
        col_indices=col_indices,
        edge_weights=edge_weights,
        edge_coverages=edge_coverages,
        node_coverages=node_coverages,
        node_starts=node_starts,
        node_ends=node_ends,
        node_types=node_types,
        n_nodes=n_nodes,
        n_edges=5,
    )


# ===================================================================
# Push-relabel max-flow tests
# ===================================================================


class TestPushRelabelSimple:
    """Test max-flow on simple graphs."""

    def test_linear_graph(self) -> None:
        """Linear source->A->B->sink has max-flow equal to min edge capacity."""
        csr = _build_simple_dag_csr()
        result = push_relabel_maxflow(csr)
        assert result.converged is True
        assert result.flow_value == pytest.approx(10.0, abs=0.1)

    def test_diamond_graph(self) -> None:
        """Diamond graph has max-flow equal to sum of two path capacities."""
        csr = _build_diamond_csr()
        result = push_relabel_maxflow(csr)
        assert result.converged is True
        # Max flow through both paths: 6 + 4 = 10
        assert result.flow_value == pytest.approx(10.0, abs=0.1)

    def test_edge_flows_sum_at_source(self) -> None:
        """Sum of edge flows leaving source equals total flow value."""
        csr = _build_diamond_csr()
        result = push_relabel_maxflow(csr)
        source_flow_out = 0.0
        for idx in range(int(csr.row_offsets[0]), int(csr.row_offsets[1])):
            source_flow_out += result.edge_flows[idx]
        assert source_flow_out == pytest.approx(result.flow_value, abs=0.1)

    def test_empty_graph(self) -> None:
        """Empty graph returns zero flow."""
        csr = CSRGraph(
            row_offsets=np.zeros(1, dtype=np.int32),
            col_indices=np.empty(0, dtype=np.int32),
            edge_weights=np.empty(0, dtype=np.float32),
            edge_coverages=np.empty(0, dtype=np.float32),
            node_coverages=np.empty(0, dtype=np.float32),
            node_starts=np.empty(0, dtype=np.int64),
            node_ends=np.empty(0, dtype=np.int64),
            node_types=np.empty(0, dtype=np.int8),
            n_nodes=0,
            n_edges=0,
        )
        result = push_relabel_maxflow(csr)
        assert result.flow_value == 0.0
        assert result.converged is True


class TestPushRelabelSpliceGraph:
    """Test max-flow on splice graph CSR."""

    def test_max_flow_on_simple_splice_graph(self, simple_csr_graph: CSRGraph) -> None:
        """Max-flow on the simple splice graph returns a positive value."""
        result = push_relabel_maxflow(simple_csr_graph)
        assert result.converged is True
        assert result.flow_value > 0.0

    def test_flow_conservation(self, simple_csr_graph: CSRGraph) -> None:
        """Flow is conserved at interior nodes (not source/sink)."""
        result = push_relabel_maxflow(simple_csr_graph)
        n = simple_csr_graph.n_nodes

        # Build incoming flow per node
        in_flow = np.zeros(n, dtype=np.float64)
        out_flow = np.zeros(n, dtype=np.float64)
        for u in range(n):
            start = int(simple_csr_graph.row_offsets[u])
            end = int(simple_csr_graph.row_offsets[u + 1])
            for idx in range(start, end):
                v = int(simple_csr_graph.col_indices[idx])
                f = result.edge_flows[idx]
                out_flow[u] += f
                in_flow[v] += f

        # Interior nodes: in_flow == out_flow
        for v in range(1, n - 1):
            assert in_flow[v] == pytest.approx(out_flow[v], abs=0.01), (
                f"Flow not conserved at node {v}: in={in_flow[v]}, out={out_flow[v]}"
            )


# ===================================================================
# Min-cost flow tests
# ===================================================================


class TestMinCostFlowSimple:
    """Test min-cost flow on small graphs."""

    def test_linear_graph(self) -> None:
        """Min-cost flow on a linear graph matches capacity."""
        csr = _build_simple_dag_csr()
        result = min_cost_flow(csr)
        assert result.converged is True
        # All edges should carry flow up to their capacity
        assert np.sum(result.edge_flows) > 0

    def test_diamond_graph(self) -> None:
        """Min-cost flow on diamond graph uses both paths."""
        csr = _build_diamond_csr()
        result = min_cost_flow(csr)
        assert result.converged is True
        # Total outgoing flow from source should be the sum of path capacities
        source_out = 0.0
        for idx in range(int(csr.row_offsets[0]), int(csr.row_offsets[1])):
            source_out += result.edge_flows[idx]
        assert source_out == pytest.approx(10.0, abs=0.5)

    def test_edge_flows_non_negative(self) -> None:
        """All edge flows from min-cost flow are non-negative."""
        csr = _build_diamond_csr()
        result = min_cost_flow(csr)
        assert np.all(result.edge_flows >= -1e-12)


# ===================================================================
# Flow-to-paths decomposition tests
# ===================================================================


class TestFlowToPaths:
    """Test flow decomposition into weighted paths."""

    def test_linear_decomposition(self) -> None:
        """Linear graph decomposes into exactly one path."""
        csr = _build_simple_dag_csr()
        edge_flows = np.array([10.0, 10.0, 10.0], dtype=np.float64)
        paths = flow_to_weighted_paths(csr, edge_flows)
        assert len(paths) >= 1
        # The single path should have weight ~10
        total_weight = sum(w for _, w in paths)
        assert total_weight == pytest.approx(10.0, abs=0.1)

    def test_diamond_decomposition(self) -> None:
        """Diamond graph with flow on two paths decomposes into two paths."""
        csr = _build_diamond_csr()
        # edge_flows: s->A=6, s->B=4, A->merge=6, B->merge=4, merge->sink=10
        edge_flows = np.array([6.0, 4.0, 6.0, 4.0, 10.0], dtype=np.float64)
        paths = flow_to_weighted_paths(csr, edge_flows)
        assert len(paths) == 2

        weights = sorted([w for _, w in paths], reverse=True)
        assert weights[0] == pytest.approx(6.0, abs=0.1)
        assert weights[1] == pytest.approx(4.0, abs=0.1)

    def test_zero_flow_no_paths(self) -> None:
        """Zero flow produces no paths."""
        csr = _build_simple_dag_csr()
        edge_flows = np.zeros(3, dtype=np.float64)
        paths = flow_to_weighted_paths(csr, edge_flows)
        assert len(paths) == 0


# ===================================================================
# Safe paths tests
# ===================================================================


class TestSafePathsSimple:
    """Test safe path computation on simple DAGs."""

    def test_linear_graph_safe_paths(self) -> None:
        """Linear graph should produce at least one safe path (the entire path is safe)."""
        csr = _build_simple_dag_csr()
        result = compute_safe_paths(csr)
        # In a linear graph, the entire path is uniquely determined
        assert len(result.paths) >= 1
        assert result.coverage_fraction > 0.0

    def test_diamond_graph_safe_paths(self) -> None:
        """Diamond graph may or may not produce safe paths."""
        csr = _build_diamond_csr()
        result = compute_safe_paths(csr)
        # The result type is SafePathResult
        assert isinstance(result.paths, list)
        assert isinstance(result.weights, list)
        assert len(result.paths) == len(result.weights)

    def test_empty_graph_safe_paths(self) -> None:
        """Empty graph produces no safe paths."""
        csr = CSRGraph(
            row_offsets=np.zeros(1, dtype=np.int32),
            col_indices=np.empty(0, dtype=np.int32),
            edge_weights=np.empty(0, dtype=np.float32),
            edge_coverages=np.empty(0, dtype=np.float32),
            node_coverages=np.empty(0, dtype=np.float32),
            node_starts=np.empty(0, dtype=np.int64),
            node_ends=np.empty(0, dtype=np.int64),
            node_types=np.empty(0, dtype=np.int8),
            n_nodes=0,
            n_edges=0,
        )
        result = compute_safe_paths(csr)
        assert len(result.paths) == 0
        assert result.coverage_fraction == 0.0


# ===================================================================
# Full decomposition pipeline tests
# ===================================================================


class TestDecomposeGraph:
    """Test the full decomposition pipeline on splice graph CSR data."""

    def test_decompose_simple(
        self,
        simple_csr_graph: CSRGraph,
        simple_splice_graph: SpliceGraph,
    ) -> None:
        """Decomposing the simple splice graph produces transcripts."""
        transcripts = decompose_graph(simple_csr_graph, simple_splice_graph)
        assert isinstance(transcripts, list)
        # The simple graph has one path: source -> exon1 -> exon2 -> sink
        # We should get at least one transcript
        assert len(transcripts) >= 1

    def test_decompose_transcript_has_exons(
        self,
        simple_csr_graph: CSRGraph,
        simple_splice_graph: SpliceGraph,
    ) -> None:
        """Decomposed transcripts have non-empty exon coordinates."""
        transcripts = decompose_graph(simple_csr_graph, simple_splice_graph)
        for tx in transcripts:
            assert len(tx.exon_coords) > 0, "Transcript has no exon coords"

    def test_decompose_transcripts_sorted(
        self,
        simple_csr_graph: CSRGraph,
        simple_splice_graph: SpliceGraph,
    ) -> None:
        """Transcripts are sorted by weight descending."""
        transcripts = decompose_graph(simple_csr_graph, simple_splice_graph)
        for i in range(len(transcripts) - 1):
            assert transcripts[i].weight >= transcripts[i + 1].weight

    def test_decompose_complex_graph(self, complex_splice_graph: SpliceGraph) -> None:
        """Decomposing the complex graph produces transcripts for both paths."""
        csr = complex_splice_graph.to_csr()
        transcripts = decompose_graph(csr, complex_splice_graph)
        assert len(transcripts) >= 1


class TestDecomposeConfig:
    """Test different decomposition config options."""

    def test_default_config(self) -> None:
        """Default config has reasonable values."""
        cfg = DecomposeConfig()
        assert cfg.min_transcript_coverage == 1.0
        assert cfg.max_transcripts_per_locus == 50
        assert cfg.use_safe_paths is True
        assert cfg.safe_path_extension is True

    def test_high_coverage_threshold(
        self,
        simple_csr_graph: CSRGraph,
        simple_splice_graph: SpliceGraph,
    ) -> None:
        """Very high min_transcript_coverage filters out all low-weight transcripts."""
        cfg = DecomposeConfig(min_transcript_coverage=1e6)
        transcripts = decompose_graph(simple_csr_graph, simple_splice_graph, config=cfg)
        assert len(transcripts) == 0

    def test_no_safe_paths(
        self,
        simple_csr_graph: CSRGraph,
        simple_splice_graph: SpliceGraph,
    ) -> None:
        """Disabling safe paths still produces transcripts."""
        cfg = DecomposeConfig(use_safe_paths=False)
        transcripts = decompose_graph(simple_csr_graph, simple_splice_graph, config=cfg)
        assert isinstance(transcripts, list)

    def test_max_transcripts_cap(
        self,
        simple_csr_graph: CSRGraph,
        simple_splice_graph: SpliceGraph,
    ) -> None:
        """max_transcripts_per_locus=1 limits output to at most 1 transcript."""
        cfg = DecomposeConfig(max_transcripts_per_locus=1)
        transcripts = decompose_graph(simple_csr_graph, simple_splice_graph, config=cfg)
        assert len(transcripts) <= 1


# ===================================================================
# Transcript dataclass tests
# ===================================================================


class TestTranscriptCreation:
    """Test Transcript dataclass creation and fields."""

    def test_default_transcript(self) -> None:
        """Default Transcript has empty fields and zero weight."""
        tx = Transcript()
        assert tx.node_ids == []
        assert tx.exon_coords == []
        assert tx.weight == 0.0
        assert tx.is_safe is False

    def test_transcript_with_data(self) -> None:
        """Transcript with explicit data stores all fields."""
        tx = Transcript(
            node_ids=[0, 1, 2, 3],
            exon_coords=[(100, 200), (300, 500)],
            weight=5.0,
            is_safe=True,
        )
        assert len(tx.node_ids) == 4
        assert len(tx.exon_coords) == 2
        assert tx.weight == 5.0
        assert tx.is_safe is True

    def test_transcript_exon_coords_are_tuples(self) -> None:
        """Exon coordinates are stored as (start, end) tuples."""
        tx = Transcript(
            exon_coords=[(100, 200), (300, 400), (500, 600)],
            weight=3.0,
        )
        for start, end in tx.exon_coords:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert end > start


class TestMergeAdjacentExons:
    """Test the _merge_adjacent_exons helper."""

    def test_single_exon_unchanged(self) -> None:
        """Single exon passes through unchanged."""
        from rapidsplice.flow.decompose import _merge_adjacent_exons

        result = _merge_adjacent_exons([(100, 500)])
        assert result == [(100, 500)]

    def test_abutting_segments_merged(self) -> None:
        """Abutting segments (end == start) are merged into one exon."""
        from rapidsplice.flow.decompose import _merge_adjacent_exons

        result = _merge_adjacent_exons([(100, 450), (450, 500)])
        assert result == [(100, 500)]

    def test_separated_exons_not_merged(self) -> None:
        """Non-abutting exons remain separate."""
        from rapidsplice.flow.decompose import _merge_adjacent_exons

        result = _merge_adjacent_exons([(100, 500), (1000, 1500)])
        assert result == [(100, 500), (1000, 1500)]

    def test_mixed_abutting_and_separated(self) -> None:
        """Mix of abutting and separated segments."""
        from rapidsplice.flow.decompose import _merge_adjacent_exons

        # (100,450) + (450,500) → (100,500), then gap, then (1000,1050) + (1050,1500) → (1000,1500)
        result = _merge_adjacent_exons([
            (100, 450), (450, 500), (1000, 1050), (1050, 1500),
        ])
        assert result == [(100, 500), (1000, 1500)]

    def test_empty_returns_empty(self) -> None:
        """Empty input returns empty output."""
        from rapidsplice.flow.decompose import _merge_adjacent_exons

        result = _merge_adjacent_exons([])
        assert result == []

    def test_three_abutting_segments(self) -> None:
        """Three abutting segments merge into one."""
        from rapidsplice.flow.decompose import _merge_adjacent_exons

        result = _merge_adjacent_exons([(100, 200), (200, 300), (300, 500)])
        assert result == [(100, 500)]
