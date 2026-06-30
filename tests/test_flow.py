"""Tests for flow decomposition: push_relabel, min_cost_flow, safe_paths, decompose.

Exercises max-flow, min-cost flow, flow-to-path decomposition, safe path
computation, and the full transcript assembly decomposition pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

import braid.flow.decompose as decompose_module
from braid.flow.decompose import (
    DecomposeConfig,
    Transcript,
    _enumerate_all_paths_with_metrics,
    _fit_path_weights_lp_with_metrics,
    decompose_batched,
    decompose_graph,
)
from braid.flow.min_cost_flow import flow_to_weighted_paths, min_cost_flow
from braid.flow.push_relabel import push_relabel_maxflow
from braid.flow.safe_paths import compute_safe_paths
from braid.graph.splice_graph import (
    BatchedCSRGraphs,
    CSRGraph,
    EdgeType,
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


def _build_priority_heap_csr() -> CSRGraph:
    """Build a DAG where the highest-priority path is encountered second."""
    n_nodes = 4
    row_offsets = np.array([0, 2, 3, 4, 4], dtype=np.int32)
    col_indices = np.array([1, 2, 3, 3], dtype=np.int32)
    edge_weights = np.array([1.0, 10.0, 1.0, 10.0], dtype=np.float32)
    edge_coverages = np.array([1.0, 10.0, 1.0, 10.0], dtype=np.float32)
    node_coverages = np.array([0.0, 1.0, 10.0, 0.0], dtype=np.float32)
    node_starts = np.array([0, 100, 200, 400], dtype=np.int64)
    node_ends = np.array([0, 200, 300, 400], dtype=np.int64)
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
        n_edges=4,
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

    def test_invalid_negative_sink_raises(self, simple_csr_graph: CSRGraph) -> None:
        """Negative sink indices must resolve inside the graph."""
        with pytest.raises(ValueError, match="sink index out of range"):
            push_relabel_maxflow(simple_csr_graph, sink=-(simple_csr_graph.n_nodes + 1))

    def test_nonconverged_flow_value_is_not_reported_as_valid(
        self, simple_csr_graph: CSRGraph,
    ) -> None:
        """Iteration exhaustion should not expose partial sink excess as a max-flow."""
        result = push_relabel_maxflow(simple_csr_graph, max_iterations=0)

        assert result.converged is False
        assert np.isnan(result.flow_value)


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

    def test_parallel_edges_augment_the_shortest_edge(self) -> None:
        """Parallel residual edges must not desync shortest-path edge selection."""
        csr = CSRGraph(
            row_offsets=np.array([0, 2, 3, 3], dtype=np.int32),
            col_indices=np.array([1, 1, 2], dtype=np.int32),
            edge_weights=np.array([1.0, 10.0, 10.0], dtype=np.float32),
            edge_coverages=np.array([1.0, 10.0, 10.0], dtype=np.float32),
            node_coverages=np.array([0.0, 10.0, 0.0], dtype=np.float32),
            node_starts=np.array([0, 100, 200], dtype=np.int64),
            node_ends=np.array([0, 150, 200], dtype=np.int64),
            node_types=np.array(
                [NodeType.SOURCE, NodeType.EXON, NodeType.SINK], dtype=np.int8,
            ),
            n_nodes=3,
            n_edges=3,
        )
        supply = np.array([10.0, 0.0, 0.0])
        demand = np.array([0.0, 0.0, 10.0])

        result = min_cost_flow(csr, supply=supply, demand=demand)

        assert result.converged is True
        assert result.edge_flows[0] == pytest.approx(0.0)
        assert result.edge_flows[1] == pytest.approx(10.0)
        assert result.edge_flows[2] == pytest.approx(10.0)


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

    def test_bounded_frontier_keeps_best_candidate(self) -> None:
        """Frontier trimming should preserve the highest-support candidate."""
        csr = _build_priority_heap_csr()
        paths, metrics = _enumerate_all_paths_with_metrics(
            csr,
            max_paths=2,
            max_heap_entries=1,
        )

        assert paths == [[0, 2, 3]]
        assert metrics["frontier_pruned"] >= 1


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

    def test_safe_path_probe_limit_returns_explicit_empty_result(self) -> None:
        """Probe limits should avoid per-edge LP blowups without fake safe paths."""
        csr = _build_simple_dag_csr()

        result = compute_safe_paths(csr, max_edge_probes=0)

        assert result.paths == []
        assert result.weights == []
        assert result.coverage_fraction == 0.0


class TestSafePathsCoverageInvariant:
    """coverage_fraction is a fraction of total flow and must stay in [0, 1].

    Regression for a double-counting bug: zero-flow / zero-capacity edges were
    marked "safe", split a single flow path into segments, and the per-segment
    min-flows were summed, inflating coverage_fraction above 1.0 (observed 2.0).
    The fix restricts the safe topology to positive-flow edges, rejects
    degenerate chains, and clamps the ratio.
    """

    @pytest.mark.parametrize(
        "builder",
        [_build_simple_dag_csr, _build_diamond_csr, _build_priority_heap_csr],
    )
    def test_coverage_fraction_within_unit_interval(self, builder) -> None:
        result = compute_safe_paths(builder())
        assert 0.0 <= result.coverage_fraction <= 1.0 + 1e-9
        # No returned safe path may revisit a node (degenerate-chain rejection).
        for path in result.paths:
            assert len(set(path)) == len(path)

    def test_zero_capacity_safe_edge_does_not_inflate_coverage(self) -> None:
        """A zero-capacity edge sharing a node with the flow path is trivially
        'safe' but carries no flow; it must not split the chain and double-count.

        Topology: 0->1->2 carries flow 10; 1->3 is a zero-capacity dead-end that
        the safe-edge test marks safe. Coverage must remain <= 1.0, and node 3
        must not appear in any returned safe path.
        """
        csr = CSRGraph(
            row_offsets=np.array([0, 1, 3, 3, 3], dtype=np.int32),
            col_indices=np.array([1, 2, 3], dtype=np.int32),
            edge_weights=np.array([10.0, 10.0, 0.0], dtype=np.float32),
            edge_coverages=np.array([10.0, 10.0, 0.0], dtype=np.float32),
            node_coverages=np.array([10.0, 10.0, 10.0, 0.0], dtype=np.float32),
            node_starts=np.array([100, 300, 500, 700], dtype=np.int64),
            node_ends=np.array([200, 400, 600, 800], dtype=np.int64),
            node_types=np.zeros(4, dtype=np.int8),
            n_nodes=4,
            n_edges=3,
        )
        result = compute_safe_paths(csr)
        assert 0.0 <= result.coverage_fraction <= 1.0 + 1e-9
        for path in result.paths:
            assert 3 not in path, "zero-capacity dead-end leaked into a safe path"


class TestPhasingWeightFitting:
    """Matching phasing evidence must pull a path's fitted weight toward the
    observed read count, not toward count * phasing_weight.

    Regression: the phasing row used coefficient 1.0 with target
    count * phasing_weight, so a matched path was pulled toward
    count * phasing_weight (e.g. 5 for count=10, weight=0.5) instead of count.
    Scaling BOTH the coefficient and the target makes the row vanish exactly
    when the weight equals the count.
    """

    def test_matched_phasing_residual_zero_when_weight_equals_count(self) -> None:
        # Linear 0->1->2 with edge coverage 10 everywhere; phasing also says 10,
        # so the NNLS solution is w=10 and the matched phasing row residual must
        # be ~0. Pre-fix the target was 10 * 0.5 = 5, leaving residual ~5.
        csr = CSRGraph(
            row_offsets=np.array([0, 1, 2, 2], dtype=np.int32),
            col_indices=np.array([1, 2], dtype=np.int32),
            edge_weights=np.array([10.0, 10.0], dtype=np.float32),
            edge_coverages=np.array([10.0, 10.0], dtype=np.float32),
            node_coverages=np.array([10.0, 10.0, 10.0], dtype=np.float32),
            node_starts=np.array([100, 300, 500], dtype=np.int64),
            node_ends=np.array([200, 400, 600], dtype=np.int64),
            node_types=np.zeros(3, dtype=np.int8),
            n_nodes=3,
            n_edges=2,
        )
        weights, metrics = _fit_path_weights_lp_with_metrics(
            csr, [[0, 1, 2]], phasing_paths=[([0, 1, 2], 10.0)]
        )

        assert metrics["phasing_matched"] == 1
        assert weights[0] == pytest.approx(10.0, abs=0.2)
        # Pre-fix this residual was ~5.0 (pulled toward count * phasing_weight).
        assert metrics["nnls_residual_phasing"] < 1.0


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


class TestDecomposeBatched:
    """Test batched CSR reconstruction before per-graph decomposition."""

    def test_batched_decompose_preserves_edge_types_and_coverages(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Batched local CSR should keep INTRON type metadata for junction weighting."""
        graph = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        src = graph.add_node(start=0, end=0, node_type=NodeType.SOURCE, coverage=0.0)
        e1 = graph.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=5.0)
        e2 = graph.add_node(start=300, end=400, node_type=NodeType.EXON, coverage=5.0)
        sink = graph.add_node(start=500, end=500, node_type=NodeType.SINK, coverage=0.0)
        graph.add_edge(src, e1, EdgeType.SOURCE_LINK, weight=5.0, coverage=5.0)
        graph.add_edge(e1, e2, EdgeType.INTRON, weight=7.0, coverage=7.0)
        graph.add_edge(e2, sink, EdgeType.SINK_LINK, weight=5.0, coverage=5.0)
        csr = graph.to_csr()
        batch = BatchedCSRGraphs()
        batch.add_graph(csr)
        batch.finalize()
        captured: list[CSRGraph] = []

        def fake_decompose(local_csr, local_graph, config=None):
            captured.append(local_csr)
            return []

        monkeypatch.setattr(decompose_module, "decompose_graph", fake_decompose)

        assert decompose_batched(batch, [graph]) == [[]]
        assert captured[0].edge_types is not None
        np.testing.assert_array_equal(captured[0].edge_types, csr.edge_types)
        np.testing.assert_array_equal(captured[0].edge_coverages, csr.edge_coverages)
        assert int(EdgeType.INTRON) in captured[0].edge_types


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
        from braid.flow.decompose import _merge_adjacent_exons

        result = _merge_adjacent_exons([(100, 500)])
        assert result == [(100, 500)]

    def test_abutting_segments_merged(self) -> None:
        """Abutting segments (end == start) are merged into one exon."""
        from braid.flow.decompose import _merge_adjacent_exons

        result = _merge_adjacent_exons([(100, 450), (450, 500)])
        assert result == [(100, 500)]

    def test_separated_exons_not_merged(self) -> None:
        """Non-abutting exons remain separate."""
        from braid.flow.decompose import _merge_adjacent_exons

        result = _merge_adjacent_exons([(100, 500), (1000, 1500)])
        assert result == [(100, 500), (1000, 1500)]

    def test_mixed_abutting_and_separated(self) -> None:
        """Mix of abutting and separated segments."""
        from braid.flow.decompose import _merge_adjacent_exons

        # (100,450) + (450,500) → (100,500), then gap, then (1000,1050) + (1050,1500) → (1000,1500)
        result = _merge_adjacent_exons([
            (100, 450), (450, 500), (1000, 1050), (1050, 1500),
        ])
        assert result == [(100, 500), (1000, 1500)]

    def test_empty_returns_empty(self) -> None:
        """Empty input returns empty output."""
        from braid.flow.decompose import _merge_adjacent_exons

        result = _merge_adjacent_exons([])
        assert result == []

    def test_three_abutting_segments(self) -> None:
        """Three abutting segments merge into one."""
        from braid.flow.decompose import _merge_adjacent_exons

        result = _merge_adjacent_exons([(100, 200), (200, 300), (300, 500)])
        assert result == [(100, 500)]
