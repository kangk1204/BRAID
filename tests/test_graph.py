"""Tests for graph modules: splice_graph, builder, and BatchedCSRGraphs.

Exercises graph creation, mutation, topological sorting, CSR conversion,
batching, simplification, and the SpliceGraphBuilder.
"""

from __future__ import annotations

import numpy as np
import pytest

from braid.graph.builder import (
    GraphBuilderConfig,
    LocusDefinition,
    SpliceGraphBuilder,
)
from braid.pipeline import (
    PipelineConfig,
    _effective_builder_profile,
    _make_builder_config,
)
from braid.graph.splice_graph import (
    BatchedCSRGraphs,
    CSRGraph,
    EdgeType,
    NodeType,
    SpliceGraph,
)
from braid.io.bam_reader import JunctionEvidence, ReadData

# ===================================================================
# SpliceGraph tests
# ===================================================================


class TestSpliceGraphCreation:
    """Test basic graph creation and properties."""

    def test_properties(self, simple_splice_graph: SpliceGraph) -> None:
        """Simple splice graph has correct chromosome, strand, and locus."""
        assert simple_splice_graph.chrom == "chr1"
        assert simple_splice_graph.strand == "+"
        assert simple_splice_graph.locus_start == 100
        assert simple_splice_graph.locus_end == 500

    def test_node_count(self, simple_splice_graph: SpliceGraph) -> None:
        """Simple graph has 4 nodes: SOURCE, 2 EXON, SINK."""
        assert simple_splice_graph.n_nodes == 4

    def test_edge_count(self, simple_splice_graph: SpliceGraph) -> None:
        """Simple graph has 3 edges."""
        assert simple_splice_graph.n_edges == 3

    def test_source_and_sink(self, simple_splice_graph: SpliceGraph) -> None:
        """Source and sink IDs are retrievable."""
        source_id = simple_splice_graph.source_id
        sink_id = simple_splice_graph.sink_id
        source_node = simple_splice_graph.get_node(source_id)
        sink_node = simple_splice_graph.get_node(sink_id)
        assert source_node.node_type == NodeType.SOURCE
        assert sink_node.node_type == NodeType.SINK


class TestAddNodesAndEdges:
    """Test adding nodes and edges to a fresh graph."""

    def test_add_nodes(self) -> None:
        """Adding nodes increments the node count."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=1000)
        g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        n1 = g.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=5.0)
        g.add_node(start=1000, end=1000, node_type=NodeType.SINK)
        assert g.n_nodes == 3
        assert g.get_node(n1).coverage == 5.0
        assert g.get_node(n1).length == 100

    def test_add_edge(self) -> None:
        """Adding an edge connects two nodes."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        n0 = g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        n1 = g.add_node(start=100, end=200, node_type=NodeType.EXON)
        g.add_edge(n0, n1, EdgeType.SOURCE_LINK, weight=3.0)
        assert g.n_edges == 1
        assert n1 in g.get_successors(n0)
        assert n0 in g.get_predecessors(n1)

    def test_duplicate_source_raises(self) -> None:
        """Adding a second SOURCE node raises ValueError."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        with pytest.raises(ValueError, match="SOURCE"):
            g.add_node(start=0, end=0, node_type=NodeType.SOURCE)

    def test_duplicate_sink_raises(self) -> None:
        """Adding a second SINK node raises ValueError."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        g.add_node(start=500, end=500, node_type=NodeType.SINK)
        with pytest.raises(ValueError, match="SINK"):
            g.add_node(start=500, end=500, node_type=NodeType.SINK)

    def test_edge_to_nonexistent_raises(self) -> None:
        """Adding an edge to a non-existent node raises KeyError."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        n0 = g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        with pytest.raises(KeyError):
            g.add_edge(n0, 999, EdgeType.SOURCE_LINK)

    def test_get_edge(self) -> None:
        """get_edge returns the correct SpliceEdge object."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        n0 = g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        n1 = g.add_node(start=100, end=200, node_type=NodeType.EXON)
        g.add_edge(n0, n1, EdgeType.SOURCE_LINK, weight=5.0, coverage=5.0)
        edge = g.get_edge(n0, n1)
        assert edge is not None
        assert edge.weight == 5.0
        assert edge.edge_type == EdgeType.SOURCE_LINK

    def test_get_nonexistent_edge(self) -> None:
        """get_edge returns None for non-existent edge."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        n0 = g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        n1 = g.add_node(start=100, end=200, node_type=NodeType.EXON)
        assert g.get_edge(n0, n1) is None


class TestTopologicalOrder:
    """Test topological sort on splice graphs."""

    def test_simple_order(self, simple_splice_graph: SpliceGraph) -> None:
        """Topological order places source before exons before sink."""
        order = simple_splice_graph.topological_order()
        assert len(order) == 4
        source_pos = order.index(simple_splice_graph.source_id)
        sink_pos = order.index(simple_splice_graph.sink_id)
        assert source_pos < sink_pos

    def test_all_nodes_present(self, simple_splice_graph: SpliceGraph) -> None:
        """Topological order contains all node IDs."""
        order = simple_splice_graph.topological_order()
        assert len(order) == simple_splice_graph.n_nodes


class TestGraphValidation:
    """Test graph validation."""

    def test_valid_graph(self, simple_splice_graph: SpliceGraph) -> None:
        """Simple splice graph passes validation."""
        assert simple_splice_graph.validate() is True

    def test_missing_source_fails(self) -> None:
        """Graph without SOURCE fails validation."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        g.add_node(start=100, end=200, node_type=NodeType.EXON)
        g.add_node(start=500, end=500, node_type=NodeType.SINK)
        assert g.validate() is False

    def test_disconnected_node_fails(self) -> None:
        """Graph with an unreachable node fails validation."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        src = g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        e1 = g.add_node(start=100, end=200, node_type=NodeType.EXON)
        g.add_node(start=300, end=400, node_type=NodeType.EXON)  # disconnected
        sink = g.add_node(start=500, end=500, node_type=NodeType.SINK)
        g.add_edge(src, e1, EdgeType.SOURCE_LINK)
        g.add_edge(e1, sink, EdgeType.SINK_LINK)
        assert g.validate() is False


class TestToCSR:
    """Test CSR conversion."""

    def test_csr_node_count(self, simple_splice_graph: SpliceGraph) -> None:
        """CSR has the same number of nodes as the SpliceGraph."""
        csr = simple_splice_graph.to_csr()
        assert csr.n_nodes == simple_splice_graph.n_nodes

    def test_csr_edge_count(self, simple_splice_graph: SpliceGraph) -> None:
        """CSR has the same number of edges as the SpliceGraph."""
        csr = simple_splice_graph.to_csr()
        assert csr.n_edges == simple_splice_graph.n_edges

    def test_csr_row_offsets_shape(self, simple_splice_graph: SpliceGraph) -> None:
        """Row offsets has length n_nodes + 1."""
        csr = simple_splice_graph.to_csr()
        assert csr.row_offsets.shape == (csr.n_nodes + 1,)

    def test_csr_last_offset(self, simple_splice_graph: SpliceGraph) -> None:
        """Last element of row_offsets equals n_edges."""
        csr = simple_splice_graph.to_csr()
        assert csr.row_offsets[-1] == csr.n_edges

    def test_csr_dtypes(self, simple_splice_graph: SpliceGraph) -> None:
        """CSR arrays have correct dtypes."""
        csr = simple_splice_graph.to_csr()
        assert csr.row_offsets.dtype == np.int32
        assert csr.col_indices.dtype == np.int32
        assert csr.edge_weights.dtype == np.float32
        assert csr.node_starts.dtype == np.int64
        assert csr.node_types.dtype == np.int8


class TestCSRProperties:
    """Test CSR has correct structure."""

    def test_source_is_first(self, simple_csr_graph: CSRGraph) -> None:
        """Source node (topological order) is at index 0 with type SOURCE."""
        assert int(simple_csr_graph.node_types[0]) == int(NodeType.SOURCE)

    def test_sink_is_last(self, simple_csr_graph: CSRGraph) -> None:
        """Sink node is at the last index with type SINK."""
        assert int(simple_csr_graph.node_types[-1]) == int(NodeType.SINK)

    def test_edges_are_forward(self, simple_csr_graph: CSRGraph) -> None:
        """All edges go from lower to higher node indices (DAG property)."""
        for u in range(simple_csr_graph.n_nodes):
            start = int(simple_csr_graph.row_offsets[u])
            end = int(simple_csr_graph.row_offsets[u + 1])
            for idx in range(start, end):
                v = int(simple_csr_graph.col_indices[idx])
                assert v > u, f"Edge {u}->{v} violates DAG ordering"


class TestBatchedCSR:
    """Test BatchedCSRGraphs with multiple small graphs."""

    def test_batch_three_graphs(self) -> None:
        """Batch three simple graphs and verify global array sizes."""
        batch = BatchedCSRGraphs()
        total_nodes_expected = 0
        total_edges_expected = 0

        for i in range(3):
            g = SpliceGraph(
                chrom="chr1", strand="+",
                locus_start=i * 1000, locus_end=(i + 1) * 1000,
            )
            src = g.add_node(start=i * 1000, end=i * 1000, node_type=NodeType.SOURCE)
            exon = g.add_node(
                start=i * 1000 + 100, end=i * 1000 + 200,
                node_type=NodeType.EXON, coverage=5.0,
            )
            sink = g.add_node(
                start=(i + 1) * 1000, end=(i + 1) * 1000,
                node_type=NodeType.SINK,
            )
            g.add_edge(src, exon, EdgeType.SOURCE_LINK, weight=5.0)
            g.add_edge(exon, sink, EdgeType.SINK_LINK, weight=5.0)

            csr = g.to_csr()
            total_nodes_expected += csr.n_nodes
            total_edges_expected += csr.n_edges
            batch.add_graph(csr, {"chrom": "chr1", "index": i})

        batch.finalize()

        assert batch.n_graphs == 3
        assert batch.total_nodes == total_nodes_expected
        assert batch.total_edges == total_edges_expected
        assert batch.row_offsets.shape == (total_nodes_expected + 1,)
        assert batch.col_indices.shape == (total_edges_expected,)

    def test_batch_graph_range(self) -> None:
        """get_graph_range returns correct node/edge slices."""
        batch = BatchedCSRGraphs()
        g1 = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        src1 = g1.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        ex1 = g1.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=5.0)
        sk1 = g1.add_node(start=500, end=500, node_type=NodeType.SINK)
        g1.add_edge(src1, ex1, EdgeType.SOURCE_LINK, weight=5.0)
        g1.add_edge(ex1, sk1, EdgeType.SINK_LINK, weight=5.0)
        csr1 = g1.to_csr()
        batch.add_graph(csr1)

        g2 = SpliceGraph(chrom="chr1", strand="+", locus_start=1000, locus_end=2000)
        src2 = g2.add_node(start=1000, end=1000, node_type=NodeType.SOURCE)
        ex2 = g2.add_node(start=1100, end=1200, node_type=NodeType.EXON, coverage=3.0)
        sk2 = g2.add_node(start=2000, end=2000, node_type=NodeType.SINK)
        g2.add_edge(src2, ex2, EdgeType.SOURCE_LINK, weight=3.0)
        g2.add_edge(ex2, sk2, EdgeType.SINK_LINK, weight=3.0)
        csr2 = g2.to_csr()
        batch.add_graph(csr2)

        batch.finalize()

        ns0, ne0, es0, ee0 = batch.get_graph_range(0)
        ns1, ne1, es1, ee1 = batch.get_graph_range(1)

        assert ns0 == 0
        assert ne0 == csr1.n_nodes
        assert ns1 == csr1.n_nodes
        assert ne1 == csr1.n_nodes + csr2.n_nodes
        assert es0 == 0
        assert ee0 == csr1.n_edges
        assert es1 == csr1.n_edges
        assert ee1 == csr1.n_edges + csr2.n_edges

    def test_add_after_finalize_raises(self) -> None:
        """Adding a graph after finalize raises RuntimeError."""
        batch = BatchedCSRGraphs()
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        src = g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        sink = g.add_node(start=500, end=500, node_type=NodeType.SINK)
        g.add_edge(src, sink, EdgeType.SOURCE_LINK, weight=1.0)
        batch.add_graph(g.to_csr())
        batch.finalize()
        with pytest.raises(RuntimeError):
            batch.add_graph(g.to_csr())

    def test_get_meta(self) -> None:
        """Metadata stored with add_graph is retrievable after finalize."""
        batch = BatchedCSRGraphs()
        g = SpliceGraph(chrom="chr2", strand="-", locus_start=0, locus_end=500)
        src = g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        sink = g.add_node(start=500, end=500, node_type=NodeType.SINK)
        g.add_edge(src, sink, EdgeType.SOURCE_LINK, weight=1.0)
        batch.add_graph(g.to_csr(), {"chrom": "chr2", "strand": "-"})
        batch.finalize()
        meta = batch.get_meta(0)
        assert meta["chrom"] == "chr2"
        assert meta["strand"] == "-"


class TestGraphSimplify:
    """Test simplification of trivial chains."""

    def test_simplify_trivial_chain(self) -> None:
        """A -> B -> C where A has one out and B has one in/one out merges into A -> C."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=1000)
        src = g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        a = g.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=10.0)
        b = g.add_node(start=200, end=300, node_type=NodeType.EXON, coverage=10.0)
        c = g.add_node(start=300, end=400, node_type=NodeType.EXON, coverage=10.0)
        sink = g.add_node(start=1000, end=1000, node_type=NodeType.SINK)

        g.add_edge(src, a, EdgeType.SOURCE_LINK, weight=10.0)
        g.add_edge(a, b, EdgeType.CONTINUATION, weight=10.0, coverage=10.0)
        g.add_edge(b, c, EdgeType.CONTINUATION, weight=10.0, coverage=10.0)
        g.add_edge(c, sink, EdgeType.SINK_LINK, weight=10.0)

        original_nodes = g.n_nodes
        g.simplify()
        # b should be merged into a, reducing node count by 1
        assert g.n_nodes < original_nodes
        assert g.validate()

    def test_simplify_does_not_remove_branches(self) -> None:
        """Simplify preserves nodes with multiple successors or predecessors."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=1000)
        src = g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        a = g.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=10.0)
        b = g.add_node(start=200, end=300, node_type=NodeType.EXON, coverage=6.0)
        c = g.add_node(start=250, end=350, node_type=NodeType.EXON, coverage=4.0)
        sink = g.add_node(start=1000, end=1000, node_type=NodeType.SINK)

        g.add_edge(src, a, EdgeType.SOURCE_LINK, weight=10.0)
        g.add_edge(a, b, EdgeType.INTRON, weight=6.0)
        g.add_edge(a, c, EdgeType.INTRON, weight=4.0)
        g.add_edge(b, sink, EdgeType.SINK_LINK, weight=6.0)
        g.add_edge(c, sink, EdgeType.SINK_LINK, weight=4.0)

        original_nodes = g.n_nodes
        g.simplify()
        # No trivial chains exist (a has 2 successors), so nothing changes
        assert g.n_nodes == original_nodes


class TestGraphRemoveNode:
    """Test node removal."""

    def test_remove_exon_node(self) -> None:
        """Removing an exon node removes all its incident edges."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        src = g.add_node(start=0, end=0, node_type=NodeType.SOURCE)
        exon = g.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=5.0)
        sink = g.add_node(start=500, end=500, node_type=NodeType.SINK)
        g.add_edge(src, exon, EdgeType.SOURCE_LINK, weight=5.0)
        g.add_edge(exon, sink, EdgeType.SINK_LINK, weight=5.0)

        g.remove_node(exon)
        assert g.n_nodes == 2
        assert g.n_edges == 0
        assert g.get_successors(src) == []
        assert g.get_predecessors(sink) == []

    def test_remove_nonexistent_raises(self) -> None:
        """Removing a non-existent node raises KeyError."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        with pytest.raises(KeyError):
            g.remove_node(999)


class TestComplexGraph:
    """Test complex graph with alternative splicing paths."""

    def test_complex_node_count(self, complex_splice_graph: SpliceGraph) -> None:
        """Complex graph has 7 nodes: SOURCE + 5 EXON + SINK."""
        assert complex_splice_graph.n_nodes == 7

    def test_complex_edge_count(self, complex_splice_graph: SpliceGraph) -> None:
        """Complex graph has 7 edges."""
        assert complex_splice_graph.n_edges == 7

    def test_complex_validates(self, complex_splice_graph: SpliceGraph) -> None:
        """Complex graph passes validation."""
        assert complex_splice_graph.validate() is True

    def test_complex_topological_order(self, complex_splice_graph: SpliceGraph) -> None:
        """Topological order is consistent with edge directions."""
        order = complex_splice_graph.topological_order()
        pos = {nid: i for i, nid in enumerate(order)}
        source_id = complex_splice_graph.source_id
        sink_id = complex_splice_graph.sink_id
        assert pos[source_id] == 0 or pos[source_id] < pos[sink_id]

    def test_complex_csr_roundtrip(self, complex_splice_graph: SpliceGraph) -> None:
        """CSR conversion preserves node and edge counts."""
        csr = complex_splice_graph.to_csr()
        assert csr.n_nodes == complex_splice_graph.n_nodes
        assert csr.n_edges == complex_splice_graph.n_edges


# ===================================================================
# SpliceGraphBuilder tests
# ===================================================================


class TestBuilderIdentifyLoci:
    """Test locus identification from junction evidence."""

    def test_single_junction_single_locus(
        self, simple_junction_evidence: JunctionEvidence,
    ) -> None:
        """A single junction produces one locus."""
        builder = SpliceGraphBuilder(
            config=GraphBuilderConfig(min_junction_support=1),
        )
        loci = builder.identify_loci(simple_junction_evidence, chrom_length=10000)
        assert len(loci) == 1
        locus = loci[0]
        assert locus.chrom == "chr1"
        assert locus.start <= 200
        assert locus.end >= 300

    def test_two_distant_junctions_two_loci(self) -> None:
        """Two junctions far apart produce two separate loci."""
        evidence = JunctionEvidence(
            chrom="chr1",
            starts=np.array([100, 5000], dtype=np.int64),
            ends=np.array([200, 5100], dtype=np.int64),
            counts=np.array([10, 10], dtype=np.int32),
            strands=np.array([0, 0], dtype=np.int8),
        )
        builder = SpliceGraphBuilder(
            config=GraphBuilderConfig(min_junction_support=1),
        )
        loci = builder.identify_loci(evidence, chrom_length=100000)
        assert len(loci) == 2

    def test_overlapping_junctions_one_locus(self) -> None:
        """Overlapping junctions are grouped into one locus."""
        evidence = JunctionEvidence(
            chrom="chr1",
            starts=np.array([100, 150], dtype=np.int64),
            ends=np.array([200, 250], dtype=np.int64),
            counts=np.array([5, 5], dtype=np.int32),
            strands=np.array([0, 0], dtype=np.int8),
        )
        builder = SpliceGraphBuilder(
            config=GraphBuilderConfig(min_junction_support=1),
        )
        loci = builder.identify_loci(evidence, chrom_length=100000)
        assert len(loci) == 1

    def test_long_internal_exon_is_kept_in_one_locus(self) -> None:
        """Junction-only locus clustering should bridge typical internal exons."""
        evidence = JunctionEvidence(
            chrom="chr1",
            starts=np.array([1000, 1700], dtype=np.int64),
            ends=np.array([1200, 1900], dtype=np.int64),
            counts=np.array([5, 5], dtype=np.int32),
            strands=np.array([0, 0], dtype=np.int8),
        )
        builder = SpliceGraphBuilder(
            config=GraphBuilderConfig(min_junction_support=1, locus_flank=800),
        )

        loci = builder.identify_loci(evidence, chrom_length=100000)

        assert len(loci) == 1
        assert loci[0].start <= 200
        assert loci[0].end >= 2700

    def test_mixed_strand_locus_can_leave_ambiguous_unassigned(self) -> None:
        """Correctness mode should avoid forcing ambiguous junctions onto one strand."""
        evidence = JunctionEvidence(
            chrom="chr1",
            starts=np.array([100, 120, 140], dtype=np.int64),
            ends=np.array([200, 220, 240], dtype=np.int64),
            counts=np.array([10, 10, 10], dtype=np.int32),
            strands=np.array([0, 1, -1], dtype=np.int8),
        )
        builder = SpliceGraphBuilder(
            config=GraphBuilderConfig(
                min_junction_support=1,
                assign_ambiguous_junctions_to_dominant_strand=False,
            ),
        )
        loci = builder.identify_loci(evidence, chrom_length=100000)
        assert len(loci) == 2
        # Ambiguous junction is always assigned to the dominant strand
        # (fwd wins tie with >=), so fwd gets 2 junctions, rev gets 1.
        assert sorted(len(locus.junction_indices) for locus in loci) == [1, 2]


class TestBuilderBuildGraph:
    """Test graph building from reads and junctions."""

    def test_build_graph_produces_valid_graph(
        self,
        simple_read_data: ReadData,
        simple_junction_evidence: JunctionEvidence,
    ) -> None:
        """Building a graph from synthetic reads produces a valid graph."""
        builder = SpliceGraphBuilder(
            config=GraphBuilderConfig(
                min_junction_support=1,
                min_exon_coverage=0.0,
                min_exon_length=1,
                min_intron_length=1,
            ),
        )

        locus = LocusDefinition(
            chrom="chr1",
            start=90,
            end=510,
            strand="+",
            junction_indices=[0],
        )

        graph = builder.build_graph(locus, simple_read_data, simple_junction_evidence)
        assert graph is not None
        assert graph.n_nodes >= 3  # SOURCE + at least 1 EXON + SINK
        assert graph.n_edges >= 2
        assert graph.validate()

    def test_build_graph_returns_none_for_no_junctions(self) -> None:
        """Build returns None when junction evidence is empty after filtering."""
        builder = SpliceGraphBuilder(
            config=GraphBuilderConfig(min_junction_support=100),
        )
        evidence = JunctionEvidence(
            chrom="chr1",
            starts=np.array([200], dtype=np.int64),
            ends=np.array([300], dtype=np.int64),
            counts=np.array([1], dtype=np.int32),
            strands=np.array([0], dtype=np.int8),
        )
        # Create a minimal ReadData
        read_data = ReadData(
            chrom_ids=np.zeros(1, dtype=np.int32),
            positions=np.array([100], dtype=np.int64),
            end_positions=np.array([400], dtype=np.int64),
            strands=np.zeros(1, dtype=np.int8),
            mapping_qualities=np.full(1, 60, dtype=np.uint8),
            is_paired=np.zeros(1, dtype=np.bool_),
            is_read1=np.zeros(1, dtype=np.bool_),
            mate_positions=np.full(1, -1, dtype=np.int64),
            mate_chrom_ids=np.full(1, -1, dtype=np.int32),
            cigar_ops=np.array([0, 3, 0], dtype=np.uint8),
            cigar_lens=np.array([100, 100, 100], dtype=np.int32),
            cigar_offsets=np.array([0, 3], dtype=np.int64),
            query_names=["read_0"],
            n_reads=1,
        )

        locus = LocusDefinition(
            chrom="chr1", start=90, end=410, strand="+",
            junction_indices=[0],
        )

        graph = builder.build_graph(locus, read_data, evidence)
        assert graph is None


class TestBuilderProfiles:
    """Test pipeline-level builder profile translation."""

    def test_relaxed_pruning_alias_maps_to_aggressive_recall(self) -> None:
        """Legacy relaxed-pruning flag should keep its old semantics."""
        cfg = PipelineConfig(
            bam_path="synthetic.bam",
            relaxed_pruning_experiment=True,
        )
        assert _effective_builder_profile(cfg) == "aggressive_recall"

    def test_conservative_correctness_profile_disables_fabricated_edges(self) -> None:
        """Correctness profile should soften filters without fabricating terminal links."""
        cfg = PipelineConfig(
            bam_path="synthetic.bam",
            builder_profile="conservative_correctness",
            min_coverage=1.0,
        )
        builder_cfg = _make_builder_config(cfg)
        assert builder_cfg.locus_flank == 800
        assert builder_cfg.junction_merge_distance == 2
        assert builder_cfg.min_relative_junction_support == pytest.approx(0.01)
        assert builder_cfg.min_relative_exon_coverage == pytest.approx(0.005)
        assert builder_cfg.assign_ambiguous_junctions_to_dominant_strand is False
        assert builder_cfg.add_fallback_terminal_edges is False

    def test_aggressive_recall_profile_expands_locus_flank(self) -> None:
        """Recall-oriented profile should widen locus bundling before graph build."""
        cfg = PipelineConfig(
            bam_path="synthetic.bam",
            builder_profile="aggressive_recall",
            min_coverage=1.0,
        )

        builder_cfg = _make_builder_config(cfg)

        assert builder_cfg.locus_flank == 1200


# ===================================================================
# Segment graph exon boundary tests
# ===================================================================


class TestSegmentGraphExonBoundaries:
    """Test the segment graph approach for exon boundary determination."""

    def test_exon_skipping_produces_skip_edge(self) -> None:
        """Classic exon-skipping: E1-E2-E3 and E1-E3 paths."""
        from braid.graph.builder import _determine_exon_boundaries

        # Junctions: E1->E2 (500->1000), E2->E3 (1500->2000), E1->E3 skip (500->2000)
        j_starts = np.array([500, 1500, 500], dtype=np.int64)
        j_ends = np.array([1000, 2000, 2000], dtype=np.int64)
        exons = _determine_exon_boundaries(j_starts, j_ends, 100, 2500, 10)

        # Should produce: (100,500), (1000,1500), (2000,2500)
        starts = [e[0] for e in exons]
        ends = [e[1] for e in exons]
        assert 100 in starts
        assert 500 in ends
        assert 1000 in starts
        assert 1500 in ends
        assert 2000 in starts
        assert 2500 in ends

    def test_a3ss_produces_separate_segments(self) -> None:
        """A3SS: two acceptors at 1000 and 1050 must produce distinct segments."""
        from braid.graph.builder import _determine_exon_boundaries

        # Junctions: (500->1000) and (500->1050) and (1500->2000)
        j_starts = np.array([500, 500, 1500], dtype=np.int64)
        j_ends = np.array([1000, 1050, 2000], dtype=np.int64)
        exons = _determine_exon_boundaries(j_starts, j_ends, 100, 2500, 10)

        # Must have segments starting at BOTH 1000 AND 1050
        starts = {e[0] for e in exons}
        assert 1000 in starts, f"Missing acceptor 1000, got starts: {starts}"
        assert 1050 in starts, f"Missing acceptor 1050, got starts: {starts}"

    def test_a5ss_produces_separate_segments(self) -> None:
        """A5SS: two donors at 450 and 500 must produce distinct segments."""
        from braid.graph.builder import _determine_exon_boundaries

        # Junctions: (450->1000) and (500->1000) and (1500->2000)
        j_starts = np.array([450, 500, 1500], dtype=np.int64)
        j_ends = np.array([1000, 1000, 2000], dtype=np.int64)
        exons = _determine_exon_boundaries(j_starts, j_ends, 100, 2500, 10)

        # Must have segments ending at BOTH 450 AND 500
        ends = {e[1] for e in exons}
        assert 450 in ends, f"Missing donor 450, got ends: {ends}"
        assert 500 in ends, f"Missing donor 500, got ends: {ends}"

    def test_mxe_produces_correct_exons(self) -> None:
        """MXE: mutually exclusive exons produce all four exon nodes."""
        from braid.graph.builder import _determine_exon_boundaries

        # E1->E2a (500->1000), E2a->E3 (1500->2000),
        # E1->E2b (500->1600), E2b->E3 (1900->2000)
        j_starts = np.array([500, 1500, 500, 1900], dtype=np.int64)
        j_ends = np.array([1000, 2000, 1600, 2000], dtype=np.int64)
        exons = _determine_exon_boundaries(j_starts, j_ends, 100, 2500, 10)

        # Must have E2a (starting at 1000, ending at 1500) and
        # E2b (starting at 1600, ending at 1900)
        starts = {e[0] for e in exons}
        ends = {e[1] for e in exons}
        assert 1000 in starts, f"Missing E2a start, got: {starts}"
        assert 1500 in ends, f"Missing E2a end, got: {ends}"
        assert 1600 in starts, f"Missing E2b start, got: {starts}"
        assert 1900 in ends, f"Missing E2b end, got: {ends}"

    def test_no_junctions_returns_single_exon(self) -> None:
        """No junctions: entire locus is one exon."""
        from braid.graph.builder import _determine_exon_boundaries

        j_starts = np.array([], dtype=np.int64)
        j_ends = np.array([], dtype=np.int64)
        exons = _determine_exon_boundaries(j_starts, j_ends, 100, 500, 10)
        assert exons == [(100, 500)]

    def test_segment_graph_no_overlapping_exons(self) -> None:
        """Segment graph should never produce overlapping exon intervals."""
        from braid.graph.builder import _determine_exon_boundaries

        # Complex case with multiple alternative splice sites
        j_starts = np.array([500, 500, 500, 1500, 1500], dtype=np.int64)
        j_ends = np.array([1000, 1050, 1100, 2000, 2050], dtype=np.int64)
        exons = _determine_exon_boundaries(j_starts, j_ends, 100, 2500, 1)

        # Verify no overlaps
        for i in range(len(exons) - 1):
            assert exons[i][1] <= exons[i + 1][0], (
                f"Overlap: {exons[i]} and {exons[i + 1]}"
            )
