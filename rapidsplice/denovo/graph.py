"""De Bruijn graph for de novo RNA-seq assembly.

Constructs a compacted de Bruijn graph from counted k-mers.  Nodes are
(k-1)-mers (or unitig sequences after compaction) and edges represent
k-mer overlaps.  Coverage information is propagated from the k-mer
count table to enable coverage-guided transcript assembly.

The graph supports compaction (merging linear chains of nodes into
unitigs), which dramatically reduces graph size while preserving all
branching structure needed for transcript reconstruction.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from rapidsplice.denovo.kmer import (
    KmerCountTable,
    decode_kmer,
    extract_prefixes_suffixes,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DBGNode:
    """A node in the de Bruijn graph.

    Attributes:
        node_id: Unique integer identifier.
        kmer_encoding: The (k-1)-mer encoding for this node.
        sequence: DNA sequence string for display/output.
        coverage: Average coverage across the unitig (after compaction).
        in_edges: Set of node IDs with edges into this node.
        out_edges: Set of node IDs with edges out of this node.
        unitig_length: Length in bases of the unitig (k-1 for uncompacted).
    """

    node_id: int
    kmer_encoding: np.uint64
    sequence: str
    coverage: float = 0.0
    in_edges: set[int] = field(default_factory=set)
    out_edges: set[int] = field(default_factory=set)
    unitig_length: int = 0

    @property
    def in_degree(self) -> int:
        """Number of incoming edges."""
        return len(self.in_edges)

    @property
    def out_degree(self) -> int:
        """Number of outgoing edges."""
        return len(self.out_edges)

    @property
    def is_branching(self) -> bool:
        """Whether this node is a branching point."""
        return self.in_degree > 1 or self.out_degree > 1


@dataclass(slots=True)
class DBGEdge:
    """An edge in the de Bruijn graph.

    Attributes:
        source: Source node ID.
        target: Target node ID.
        kmer_encoding: The k-mer encoding that created this edge.
        coverage: Number of reads supporting this edge.
    """

    source: int
    target: int
    kmer_encoding: np.uint64
    coverage: float = 0.0


@dataclass
class DeBruijnGraph:
    """Compacted de Bruijn graph for de novo assembly.

    Attributes:
        nodes: Dictionary mapping node ID to DBGNode.
        edges: List of all edges.
        k: K-mer size used to build the graph.
        n_original_nodes: Number of nodes before compaction.
        n_original_edges: Number of edges before compaction.
    """

    nodes: dict[int, DBGNode] = field(default_factory=dict)
    edges: list[DBGEdge] = field(default_factory=list)
    k: int = 25
    n_original_nodes: int = 0
    n_original_edges: int = 0

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self.edges)

    def sources(self) -> list[int]:
        """Return node IDs with no incoming edges (source nodes)."""
        return [nid for nid, n in self.nodes.items() if n.in_degree == 0]

    def sinks(self) -> list[int]:
        """Return node IDs with no outgoing edges (sink nodes)."""
        return [nid for nid, n in self.nodes.items() if n.out_degree == 0]

    def get_edge_coverage(self, src: int, tgt: int) -> float:
        """Get coverage of edge from src to tgt.

        Args:
            src: Source node ID.
            tgt: Target node ID.

        Returns:
            Edge coverage, or 0.0 if edge not found.
        """
        for e in self.edges:
            if e.source == src and e.target == tgt:
                return e.coverage
        return 0.0


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_debruijn_graph(
    kmer_table: KmerCountTable,
) -> DeBruijnGraph:
    """Build a de Bruijn graph from a k-mer count table.

    Creates a node for each unique (k-1)-mer prefix and suffix in the
    k-mer table, and an edge for each k-mer connecting its prefix to
    its suffix.  Edge coverage is taken from the k-mer count.

    Args:
        kmer_table: K-mer count table with unique canonical k-mers and counts.

    Returns:
        An uncompacted DeBruijnGraph.
    """
    k = kmer_table.k
    kmers = kmer_table.kmers
    counts = kmer_table.counts

    if len(kmers) == 0:
        return DeBruijnGraph(k=k)

    # Extract prefixes and suffixes
    prefixes, suffixes = extract_prefixes_suffixes(kmers, k)

    # Assign unique IDs to (k-1)-mers
    all_km1mers = np.concatenate([prefixes, suffixes])
    unique_km1mers = np.unique(all_km1mers)
    km1mer_to_id: dict[int, int] = {
        int(km): i for i, km in enumerate(unique_km1mers)
    }

    # Create nodes
    nodes: dict[int, DBGNode] = {}
    for km, nid in km1mer_to_id.items():
        seq = decode_kmer(np.uint64(km), k - 1)
        nodes[nid] = DBGNode(
            node_id=nid,
            kmer_encoding=np.uint64(km),
            sequence=seq,
            unitig_length=k - 1,
        )

    # Create edges and compute node coverages
    edges: list[DBGEdge] = []
    node_cov_sum: dict[int, float] = defaultdict(float)
    node_cov_count: dict[int, int] = defaultdict(int)

    for i in range(len(kmers)):
        src_id = km1mer_to_id[int(prefixes[i])]
        tgt_id = km1mer_to_id[int(suffixes[i])]
        cov = float(counts[i])

        edges.append(DBGEdge(
            source=src_id,
            target=tgt_id,
            kmer_encoding=kmers[i],
            coverage=cov,
        ))

        nodes[src_id].out_edges.add(tgt_id)
        nodes[tgt_id].in_edges.add(src_id)

        # Accumulate coverage for averaging
        node_cov_sum[src_id] += cov
        node_cov_count[src_id] += 1
        node_cov_sum[tgt_id] += cov
        node_cov_count[tgt_id] += 1

    # Set average coverage per node
    for nid in nodes:
        if node_cov_count[nid] > 0:
            nodes[nid].coverage = node_cov_sum[nid] / node_cov_count[nid]

    graph = DeBruijnGraph(
        nodes=nodes,
        edges=edges,
        k=k,
        n_original_nodes=len(nodes),
        n_original_edges=len(edges),
    )

    logger.info(
        "Built de Bruijn graph: %d nodes, %d edges (k=%d)",
        graph.n_nodes, graph.n_edges, k,
    )
    return graph


# ---------------------------------------------------------------------------
# Graph compaction
# ---------------------------------------------------------------------------


def compact_graph(graph: DeBruijnGraph) -> DeBruijnGraph:
    """Compact the de Bruijn graph by merging linear chains into unitigs.

    A linear chain is a sequence of nodes where each has exactly one
    incoming and one outgoing edge.  These are merged into a single
    unitig node, reducing graph complexity while preserving branching
    structure.

    Args:
        graph: Uncompacted de Bruijn graph.

    Returns:
        Compacted DeBruijnGraph with unitig nodes.
    """
    if graph.n_nodes == 0:
        return graph

    k = graph.k
    visited: set[int] = set()
    chains: list[list[int]] = []

    # Find linear chains starting from branching/source nodes
    for nid in graph.nodes:
        if nid in visited:
            continue
        node = graph.nodes[nid]
        # Start a chain from branching points, sources, or unvisited nodes
        if node.in_degree != 1 or node.out_degree != 1:
            # Try to extend forward from each outgoing edge
            for out_nid in list(node.out_edges):
                chain = _extend_chain(graph, out_nid, visited)
                if chain:
                    chains.append(chain)

    # Also find isolated cycles (all nodes with in=1, out=1)
    for nid in graph.nodes:
        if nid not in visited:
            chain = _extend_chain(graph, nid, visited)
            if chain:
                chains.append(chain)

    if not chains:
        logger.info("No linear chains to compact")
        return graph

    # Build compacted graph
    new_nodes: dict[int, DBGNode] = {}
    new_edges: list[DBGEdge] = []
    next_id = 0

    # Map: old node ID -> new node ID (for chain members)
    old_to_new: dict[int, int] = {}

    # Create unitig nodes from chains
    for chain in chains:
        if len(chain) < 2:
            continue

        # Build unitig sequence: first node's sequence + last base of each
        # subsequent node
        seq = graph.nodes[chain[0]].sequence
        total_cov = graph.nodes[chain[0]].coverage
        for i in range(1, len(chain)):
            seq += graph.nodes[chain[i]].sequence[-1]
            total_cov += graph.nodes[chain[i]].coverage

        avg_cov = total_cov / len(chain)
        unitig_node = DBGNode(
            node_id=next_id,
            kmer_encoding=graph.nodes[chain[0]].kmer_encoding,
            sequence=seq,
            coverage=avg_cov,
            unitig_length=len(seq),
        )
        new_nodes[next_id] = unitig_node

        for old_id in chain:
            old_to_new[old_id] = next_id

        next_id += 1

    # Copy non-chain nodes
    for nid, node in graph.nodes.items():
        if nid not in old_to_new:
            new_node = DBGNode(
                node_id=next_id,
                kmer_encoding=node.kmer_encoding,
                sequence=node.sequence,
                coverage=node.coverage,
                unitig_length=node.unitig_length,
            )
            new_nodes[next_id] = new_node
            old_to_new[nid] = next_id
            next_id += 1

    # Rebuild edges using new IDs
    edge_set: set[tuple[int, int]] = set()
    for edge in graph.edges:
        new_src = old_to_new.get(edge.source, edge.source)
        new_tgt = old_to_new.get(edge.target, edge.target)
        if new_src == new_tgt:
            continue  # Skip self-loops from compaction
        if (new_src, new_tgt) not in edge_set:
            edge_set.add((new_src, new_tgt))
            new_edges.append(DBGEdge(
                source=new_src,
                target=new_tgt,
                kmer_encoding=edge.kmer_encoding,
                coverage=edge.coverage,
            ))
            new_nodes[new_src].out_edges.add(new_tgt)
            new_nodes[new_tgt].in_edges.add(new_src)

    compacted = DeBruijnGraph(
        nodes=new_nodes,
        edges=new_edges,
        k=k,
        n_original_nodes=graph.n_original_nodes,
        n_original_edges=graph.n_original_edges,
    )

    logger.info(
        "Compacted graph: %d -> %d nodes, %d -> %d edges (%d chains merged)",
        graph.n_nodes, compacted.n_nodes,
        graph.n_edges, compacted.n_edges,
        len(chains),
    )
    return compacted


def _extend_chain(
    graph: DeBruijnGraph,
    start: int,
    visited: set[int],
) -> list[int]:
    """Extend a linear chain starting from a node.

    Follows the chain as long as each node has exactly one incoming and
    one outgoing edge.

    Args:
        graph: The de Bruijn graph.
        start: Starting node ID.
        visited: Set of already-visited node IDs (updated in place).

    Returns:
        List of node IDs forming the chain, or empty if start is not
        part of a linear chain.
    """
    if start in visited:
        return []

    node = graph.nodes.get(start)
    if node is None:
        return []

    # Node must be continuable (in_degree=1, out_degree=1) to be part of chain
    if node.in_degree != 1 or node.out_degree != 1:
        return []

    chain = [start]
    visited.add(start)

    # Extend forward
    current = start
    while True:
        cur_node = graph.nodes[current]
        if cur_node.out_degree != 1:
            break
        next_id = next(iter(cur_node.out_edges))
        next_node = graph.nodes.get(next_id)
        if next_node is None or next_id in visited:
            break
        if next_node.in_degree != 1:
            break
        chain.append(next_id)
        visited.add(next_id)
        current = next_id

    return chain if len(chain) >= 2 else []
