"""Splice graph data structures for GPU-accelerated RNA-seq transcript assembly.

This module defines the core splice graph representation used throughout
RapidSplice. A splice graph is a directed acyclic graph (DAG) where nodes
represent exonic regions and edges represent splicing events (introns) or
continuations between adjacent exons. Every graph has a virtual SOURCE node
and SINK node to provide a single-entry, single-exit structure required by
flow decomposition algorithms.

The module provides three levels of representation:

1. **SpliceGraph** -- mutable adjacency-list representation used during
   graph construction and simplification (CPU-side).
2. **CSRGraph** -- immutable compressed sparse row (CSR) representation
   for a single locus, ready for GPU transfer.
3. **BatchedCSRGraphs** -- multiple CSR graphs packed into contiguous
   arrays so that thousands of loci can be processed in a single GPU
   kernel launch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NodeType(IntEnum):
    """Type of a node in the splice graph."""

    SOURCE = 0
    SINK = 1
    EXON = 2


class EdgeType(IntEnum):
    """Type of an edge in the splice graph."""

    INTRON = 0
    CONTINUATION = 1
    SOURCE_LINK = 2
    SINK_LINK = 3


# ---------------------------------------------------------------------------
# Node / Edge dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SpliceNode:
    """A single node in the splice graph.

    Attributes:
        node_id: Unique integer identifier within the graph.
        start: 0-based genomic start coordinate (inclusive).
        end: 0-based genomic end coordinate (exclusive).
        node_type: Whether this node is SOURCE, SINK, or EXON.
        coverage: Average read coverage across this node.
        length: Genomic span of the node (``end - start``).
    """

    node_id: int
    start: int
    end: int
    node_type: NodeType
    coverage: float
    length: int


@dataclass(slots=True)
class SpliceEdge:
    """A directed edge in the splice graph.

    Attributes:
        src: Source node identifier.
        dst: Destination node identifier.
        edge_type: The kind of connection this edge represents.
        weight: Number of reads supporting this junction.
        coverage: Average coverage at the junction boundary.
    """

    src: int
    dst: int
    edge_type: EdgeType
    weight: float
    coverage: float


# ---------------------------------------------------------------------------
# CSRGraph -- immutable GPU-ready representation for one locus
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CSRGraph:
    """Compressed sparse row (CSR) representation of a splice graph.

    All arrays are NumPy arrays with explicit dtypes chosen for efficient
    GPU transfer (32-bit where possible, 64-bit for genomic coordinates).

    Attributes:
        row_offsets: ``int32`` array of length ``n_nodes + 1``.  For node *i*
            the outgoing edges are stored in ``col_indices[row_offsets[i]:row_offsets[i+1]]``.
        col_indices: ``int32`` array of length ``n_edges`` -- destination
            node for each outgoing edge.
        edge_weights: ``float32`` array of length ``n_edges`` -- junction
            read support for each edge.
        edge_coverages: ``float32`` array of length ``n_edges`` -- average
            coverage at each junction.
        node_coverages: ``float32`` array of length ``n_nodes``.
        node_starts: ``int64`` array of length ``n_nodes`` -- 0-based
            genomic start per node.
        node_ends: ``int64`` array of length ``n_nodes`` -- 0-based
            exclusive genomic end per node.
        node_types: ``int8`` array of length ``n_nodes`` -- :class:`NodeType`
            ordinal per node.
        n_nodes: Total number of nodes.
        n_edges: Total number of edges.
    """

    row_offsets: np.ndarray
    col_indices: np.ndarray
    edge_weights: np.ndarray
    edge_coverages: np.ndarray
    node_coverages: np.ndarray
    node_starts: np.ndarray
    node_ends: np.ndarray
    node_types: np.ndarray
    n_nodes: int
    n_edges: int


# ---------------------------------------------------------------------------
# SpliceGraph -- mutable adjacency-list representation
# ---------------------------------------------------------------------------


class SpliceGraph:
    """Mutable splice graph for a single gene locus.

    Uses adjacency lists internally so that nodes and edges can be added,
    removed, and the graph simplified before conversion to the immutable
    CSR layout.  A virtual *SOURCE* and *SINK* node are **not** created
    automatically -- call :pymethod:`add_node` with the appropriate
    :class:`NodeType` to insert them.

    Parameters:
        chrom: Chromosome / contig name (e.g. ``"chr1"``).
        strand: Strand, either ``"+"`` or ``"-"``.
        locus_start: 0-based start of the genomic locus.
        locus_end: 0-based exclusive end of the genomic locus.
    """

    def __init__(
        self,
        chrom: str,
        strand: str,
        locus_start: int,
        locus_end: int,
    ) -> None:
        self._chrom: str = chrom
        self._strand: str = strand
        self._locus_start: int = locus_start
        self._locus_end: int = locus_end
        self.runtime_diagnostics: dict[str, int | float] = {}

        # Node storage -- maps node_id -> SpliceNode
        self._nodes: dict[int, SpliceNode] = {}
        self._next_node_id: int = 0

        # Adjacency lists  (node_id -> list[node_id])
        self._successors: dict[int, list[int]] = {}
        self._predecessors: dict[int, list[int]] = {}

        # Edge storage -- maps (src, dst) -> SpliceEdge
        self._edges: dict[tuple[int, int], SpliceEdge] = {}

        # Cached special node ids
        self._source_id: int | None = None
        self._sink_id: int | None = None

    # ---- properties -------------------------------------------------------

    @property
    def chrom(self) -> str:
        """Chromosome / contig name."""
        return self._chrom

    @property
    def strand(self) -> str:
        """Strand (``'+'`` or ``'-'``)."""
        return self._strand

    @property
    def locus_start(self) -> int:
        """0-based genomic start of the locus."""
        return self._locus_start

    @property
    def locus_end(self) -> int:
        """0-based exclusive genomic end of the locus."""
        return self._locus_end

    @property
    def n_nodes(self) -> int:
        """Number of nodes currently in the graph."""
        return len(self._nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges currently in the graph."""
        return len(self._edges)

    @property
    def source_id(self) -> int:
        """Node ID of the virtual SOURCE node.

        Raises:
            ValueError: If no SOURCE node has been added.
        """
        if self._source_id is None:
            raise ValueError("No SOURCE node has been added to the graph.")
        return self._source_id

    @property
    def sink_id(self) -> int:
        """Node ID of the virtual SINK node.

        Raises:
            ValueError: If no SINK node has been added.
        """
        if self._sink_id is None:
            raise ValueError("No SINK node has been added to the graph.")
        return self._sink_id

    # ---- mutation ----------------------------------------------------------

    def add_node(
        self,
        start: int,
        end: int,
        node_type: NodeType,
        coverage: float = 0.0,
    ) -> int:
        """Add a node to the splice graph.

        Parameters:
            start: 0-based genomic start coordinate (inclusive).
            end: 0-based genomic end coordinate (exclusive).
            node_type: :class:`NodeType` of the new node.
            coverage: Average read coverage across the node.

        Returns:
            The integer ID assigned to the new node.

        Raises:
            ValueError: If a second SOURCE or SINK is added.
        """
        if node_type == NodeType.SOURCE:
            if self._source_id is not None:
                raise ValueError("Graph already has a SOURCE node.")
        if node_type == NodeType.SINK:
            if self._sink_id is not None:
                raise ValueError("Graph already has a SINK node.")

        nid = self._next_node_id
        self._next_node_id += 1

        node = SpliceNode(
            node_id=nid,
            start=start,
            end=end,
            node_type=node_type,
            coverage=coverage,
            length=end - start,
        )
        self._nodes[nid] = node
        self._successors[nid] = []
        self._predecessors[nid] = []

        if node_type == NodeType.SOURCE:
            self._source_id = nid
        elif node_type == NodeType.SINK:
            self._sink_id = nid

        return nid

    def add_edge(
        self,
        src: int,
        dst: int,
        edge_type: EdgeType,
        weight: float = 0.0,
        coverage: float = 0.0,
    ) -> None:
        """Add a directed edge to the splice graph.

        If the edge ``(src, dst)`` already exists it is silently replaced.

        Parameters:
            src: Source node ID.
            dst: Destination node ID.
            edge_type: :class:`EdgeType` of the new edge.
            weight: Number of junction-supporting reads.
            coverage: Average coverage at the junction.

        Raises:
            KeyError: If *src* or *dst* does not exist in the graph.
        """
        if src not in self._nodes:
            raise KeyError(f"Source node {src} does not exist.")
        if dst not in self._nodes:
            raise KeyError(f"Destination node {dst} does not exist.")

        edge = SpliceEdge(
            src=src,
            dst=dst,
            edge_type=edge_type,
            weight=weight,
            coverage=coverage,
        )

        existing = (src, dst) in self._edges
        self._edges[(src, dst)] = edge

        if not existing:
            self._successors[src].append(dst)
            self._predecessors[dst].append(src)

    def remove_node(self, node_id: int) -> None:
        """Remove a node and all incident edges from the graph.

        Parameters:
            node_id: ID of the node to remove.

        Raises:
            KeyError: If *node_id* does not exist.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} does not exist.")

        # Remove outgoing edges
        for dst in list(self._successors[node_id]):
            self._edges.pop((node_id, dst), None)
            self._predecessors[dst].remove(node_id)

        # Remove incoming edges
        for src in list(self._predecessors[node_id]):
            self._edges.pop((src, node_id), None)
            self._successors[src].remove(node_id)

        # Update cached special ids
        node = self._nodes[node_id]
        if node.node_type == NodeType.SOURCE:
            self._source_id = None
        elif node.node_type == NodeType.SINK:
            self._sink_id = None

        del self._successors[node_id]
        del self._predecessors[node_id]
        del self._nodes[node_id]

    # ---- queries -----------------------------------------------------------

    def get_node(self, node_id: int) -> SpliceNode:
        """Return the :class:`SpliceNode` for the given ID.

        Raises:
            KeyError: If *node_id* does not exist.
        """
        if node_id not in self._nodes:
            raise KeyError(f"Node {node_id} does not exist.")
        return self._nodes[node_id]

    def get_successors(self, node_id: int) -> list[int]:
        """Return IDs of all successor (child) nodes.

        Raises:
            KeyError: If *node_id* does not exist.
        """
        if node_id not in self._successors:
            raise KeyError(f"Node {node_id} does not exist.")
        return list(self._successors[node_id])

    def get_predecessors(self, node_id: int) -> list[int]:
        """Return IDs of all predecessor (parent) nodes.

        Raises:
            KeyError: If *node_id* does not exist.
        """
        if node_id not in self._predecessors:
            raise KeyError(f"Node {node_id} does not exist.")
        return list(self._predecessors[node_id])

    def get_edge(self, src: int, dst: int) -> SpliceEdge | None:
        """Return the edge between *src* and *dst*, or ``None`` if absent."""
        return self._edges.get((src, dst))

    def get_edges_from(self, node_id: int) -> list[SpliceEdge]:
        """Return all outgoing edges from *node_id*.

        Raises:
            KeyError: If *node_id* does not exist.
        """
        if node_id not in self._successors:
            raise KeyError(f"Node {node_id} does not exist.")
        return [self._edges[(node_id, dst)] for dst in self._successors[node_id]]

    def get_edges_to(self, node_id: int) -> list[SpliceEdge]:
        """Return all incoming edges to *node_id*.

        Raises:
            KeyError: If *node_id* does not exist.
        """
        if node_id not in self._predecessors:
            raise KeyError(f"Node {node_id} does not exist.")
        return [self._edges[(src, node_id)] for src in self._predecessors[node_id]]

    # ---- algorithms --------------------------------------------------------

    def topological_order(self) -> list[int]:
        """Compute a topological ordering of the nodes using Kahn's algorithm.

        Returns:
            List of node IDs in topological order.

        Raises:
            RuntimeError: If the graph contains a cycle and therefore no
                topological order exists.
        """
        in_degree: dict[int, int] = {nid: len(preds) for nid, preds in self._predecessors.items()}
        queue: deque[int] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        order: list[int] = []

        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in self._successors[nid]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if len(order) != len(self._nodes):
            raise RuntimeError(
                f"Graph contains a cycle: topological sort visited {len(order)} "
                f"of {len(self._nodes)} nodes."
            )
        return order

    def simplify(self) -> None:
        """Remove trivial chains by merging consecutive nodes with single in/out edges.

        A node *v* is a trivial chain interior node when:
        - *v* has exactly one predecessor *u* and one successor *w*.
        - *u* has exactly one successor (which is *v*).
        - Both edges (u, v) and (v, w) are CONTINUATION edges.

        When this pattern is found, *v* is removed: the edge (u, v) is
        deleted, node *v* is deleted, and the edge (v, w) is replaced by a
        new CONTINUATION edge (u, w).  The coverage of the merged edge is
        the mean of the two original edges' coverages and the weight is
        summed.

        The predecessor node *u*'s ``end`` coordinate is **not** extended
        because node boundaries correspond to splice sites and must not be
        altered.  Only the edge routing changes.

        This method iterates until no further merges are possible.
        """
        changed = True
        while changed:
            changed = False
            # Snapshot current node ids; iteration order does not matter.
            for nid in list(self._nodes):
                if nid not in self._nodes:
                    continue  # already removed in this pass
                node = self._nodes[nid]

                # Skip special nodes
                if node.node_type != NodeType.EXON:
                    continue

                preds = self._predecessors.get(nid, [])
                succs = self._successors.get(nid, [])

                if len(preds) != 1 or len(succs) != 1:
                    continue

                pred_id = preds[0]
                succ_id = succs[0]

                # Predecessor must have exactly one successor (this node)
                if len(self._successors[pred_id]) != 1:
                    continue

                edge_in = self._edges.get((pred_id, nid))
                edge_out = self._edges.get((nid, succ_id))

                if edge_in is None or edge_out is None:
                    continue

                # Both must be CONTINUATION edges
                if edge_in.edge_type != EdgeType.CONTINUATION:
                    continue
                if edge_out.edge_type != EdgeType.CONTINUATION:
                    continue

                # Merge: remove node nid, create direct edge pred -> succ
                merged_weight = edge_in.weight + edge_out.weight
                merged_coverage = (edge_in.coverage + edge_out.coverage) / 2.0

                # Extend predecessor node to cover the removed node's span
                pred_node = self._nodes[pred_id]
                new_end = max(pred_node.end, node.end)
                self._nodes[pred_id] = SpliceNode(
                    node_id=pred_id,
                    start=pred_node.start,
                    end=new_end,
                    node_type=pred_node.node_type,
                    coverage=(pred_node.coverage + node.coverage) / 2.0,
                    length=new_end - pred_node.start,
                )

                self.remove_node(nid)
                self.add_edge(
                    pred_id,
                    succ_id,
                    EdgeType.CONTINUATION,
                    weight=merged_weight,
                    coverage=merged_coverage,
                )
                changed = True

    def validate(self) -> bool:
        """Check graph consistency.

        Validates all of the following:
        1. The graph is a DAG (topological order exists).
        2. Exactly one SOURCE and one SINK node exist.
        3. Every EXON node is reachable from SOURCE.
        4. Every EXON node can reach SINK.
        5. All edge endpoints reference existing nodes.
        6. Adjacency lists are consistent with the edge dictionary.

        Returns:
            ``True`` if all checks pass; ``False`` otherwise.
        """
        # Check SOURCE / SINK existence
        if self._source_id is None or self._sink_id is None:
            return False

        if self._source_id not in self._nodes or self._sink_id not in self._nodes:
            return False

        # Check DAG property
        try:
            self.topological_order()
        except RuntimeError:
            return False

        # Forward reachability from SOURCE
        forward_reachable: set[int] = set()
        queue: deque[int] = deque([self._source_id])
        while queue:
            nid = queue.popleft()
            if nid in forward_reachable:
                continue
            forward_reachable.add(nid)
            for succ in self._successors.get(nid, []):
                if succ not in forward_reachable:
                    queue.append(succ)

        # Backward reachability from SINK
        backward_reachable: set[int] = set()
        queue = deque([self._sink_id])
        while queue:
            nid = queue.popleft()
            if nid in backward_reachable:
                continue
            backward_reachable.add(nid)
            for pred in self._predecessors.get(nid, []):
                if pred not in backward_reachable:
                    queue.append(pred)

        # Every node must be reachable from SOURCE and able to reach SINK
        for nid in self._nodes:
            if nid not in forward_reachable:
                return False
            if nid not in backward_reachable:
                return False

        # Edge endpoint consistency
        for (src, dst), edge in self._edges.items():
            if src not in self._nodes or dst not in self._nodes:
                return False
            if edge.src != src or edge.dst != dst:
                return False
            if dst not in self._successors.get(src, []):
                return False
            if src not in self._predecessors.get(dst, []):
                return False

        # Adjacency list consistency: every entry in adj lists has a matching edge
        for nid, succs in self._successors.items():
            for s in succs:
                if (nid, s) not in self._edges:
                    return False
        for nid, preds in self._predecessors.items():
            for p in preds:
                if (p, nid) not in self._edges:
                    return False

        return True

    # ---- conversion --------------------------------------------------------

    def to_csr(self) -> CSRGraph:
        """Convert the splice graph to compressed sparse row format.

        Nodes are renumbered to a dense ``[0, n_nodes)`` range based on their
        topological order.  This guarantees that for every edge ``(u, v)`` in
        the CSR the inequality ``u < v`` holds, which is useful for GPU
        dynamic-programming kernels that sweep left-to-right.

        Returns:
            A :class:`CSRGraph` instance with all arrays allocated as NumPy
            arrays.

        Raises:
            RuntimeError: If the graph contains a cycle.
        """
        if len(self._nodes) == 0:
            return CSRGraph(
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

        topo = self.topological_order()
        n = len(topo)

        # Old ID -> new dense ID
        old_to_new: dict[int, int] = {old_id: new_id for new_id, old_id in enumerate(topo)}

        # Build per-node outgoing edge lists (in new id space), sorted by dst
        out_edges: list[list[tuple[int, float, float]]] = [[] for _ in range(n)]
        for (src, dst), edge in self._edges.items():
            new_src = old_to_new[src]
            new_dst = old_to_new[dst]
            out_edges[new_src].append((new_dst, edge.weight, edge.coverage))

        # Sort each adjacency list by destination for deterministic CSR
        for adj in out_edges:
            adj.sort(key=lambda t: t[0])

        # Allocate CSR arrays
        total_edges = len(self._edges)
        row_offsets = np.empty(n + 1, dtype=np.int32)
        col_indices = np.empty(total_edges, dtype=np.int32)
        edge_weights = np.empty(total_edges, dtype=np.float32)
        edge_coverages = np.empty(total_edges, dtype=np.float32)
        node_coverages = np.empty(n, dtype=np.float32)
        node_starts = np.empty(n, dtype=np.int64)
        node_ends = np.empty(n, dtype=np.int64)
        node_types = np.empty(n, dtype=np.int8)

        offset = 0
        for new_id in range(n):
            row_offsets[new_id] = offset
            old_id = topo[new_id]
            node = self._nodes[old_id]
            node_coverages[new_id] = node.coverage
            node_starts[new_id] = node.start
            node_ends[new_id] = node.end
            node_types[new_id] = int(node.node_type)

            for dst_new, w, c in out_edges[new_id]:
                col_indices[offset] = dst_new
                edge_weights[offset] = w
                edge_coverages[offset] = c
                offset += 1

        row_offsets[n] = offset

        return CSRGraph(
            row_offsets=row_offsets,
            col_indices=col_indices,
            edge_weights=edge_weights,
            edge_coverages=edge_coverages,
            node_coverages=node_coverages,
            node_starts=node_starts,
            node_ends=node_ends,
            node_types=node_types,
            n_nodes=n,
            n_edges=total_edges,
        )

    # ---- dunder methods ----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SpliceGraph(chrom={self._chrom!r}, strand={self._strand!r}, "
            f"locus={self._locus_start}-{self._locus_end}, "
            f"nodes={self.n_nodes}, edges={self.n_edges})"
        )


# ---------------------------------------------------------------------------
# BatchedCSRGraphs -- multiple splice graphs packed for batch GPU processing
# ---------------------------------------------------------------------------


class BatchedCSRGraphs:
    """Multiple CSR splice graphs packed into contiguous arrays.

    This enables processing thousands of independent gene-locus splice graphs
    in a single GPU kernel launch.  Each graph occupies a contiguous slice of
    the global node and edge arrays.  Per-graph node IDs in ``col_indices``
    are offset so that they index into the **global** arrays directly.

    Typical workflow::

        batch = BatchedCSRGraphs()
        for locus_graph in locus_graphs:
            csr = locus_graph.to_csr()
            batch.add_graph(csr, {"chrom": "chr1", "strand": "+"})
        batch.finalize()
        # batch.row_offsets, batch.col_indices, ... are now contiguous arrays

    After :meth:`finalize`, the batch is immutable.
    """

    def __init__(self) -> None:
        self._graphs: list[CSRGraph] = []
        self._meta: list[dict[str, Any]] = []
        self._finalized: bool = False

        # Populated by finalize()
        self._graph_offsets: np.ndarray | None = None
        self._edge_offsets: np.ndarray | None = None
        self._row_offsets: np.ndarray | None = None
        self._col_indices: np.ndarray | None = None
        self._edge_weights: np.ndarray | None = None
        self._edge_coverages: np.ndarray | None = None
        self._node_coverages: np.ndarray | None = None
        self._node_starts: np.ndarray | None = None
        self._node_ends: np.ndarray | None = None
        self._node_types: np.ndarray | None = None
        self._total_nodes: int = 0
        self._total_edges: int = 0

    # ---- building ----------------------------------------------------------

    def add_graph(self, graph: CSRGraph, graph_meta: dict[str, Any] | None = None) -> int:
        """Add a CSR graph to the batch.

        Parameters:
            graph: The :class:`CSRGraph` to append.
            graph_meta: Optional metadata dictionary (e.g. chromosome, strand).

        Returns:
            The 0-based index of the newly added graph within the batch.

        Raises:
            RuntimeError: If :meth:`finalize` has already been called.
        """
        if self._finalized:
            raise RuntimeError("Cannot add graphs after finalize() has been called.")

        idx = len(self._graphs)
        self._graphs.append(graph)
        self._meta.append(graph_meta if graph_meta is not None else {})
        return idx

    def finalize(self) -> None:
        """Pack all added graphs into contiguous arrays.

        After this call the individual graphs can no longer be added and the
        packed arrays are available via the batch properties.

        Raises:
            RuntimeError: If already finalized or if no graphs were added.
        """
        if self._finalized:
            raise RuntimeError("Batch has already been finalized.")
        if len(self._graphs) == 0:
            raise RuntimeError("No graphs have been added to the batch.")

        n_graphs = len(self._graphs)

        # Compute cumulative offsets
        node_counts = [g.n_nodes for g in self._graphs]
        edge_counts = [g.n_edges for g in self._graphs]

        # graph_offsets[i] = cumulative node count before graph i
        graph_offsets = np.empty(n_graphs + 1, dtype=np.int64)
        graph_offsets[0] = 0
        for i, nc in enumerate(node_counts):
            graph_offsets[i + 1] = graph_offsets[i] + nc

        edge_offsets = np.empty(n_graphs + 1, dtype=np.int64)
        edge_offsets[0] = 0
        for i, ec in enumerate(edge_counts):
            edge_offsets[i + 1] = edge_offsets[i] + ec

        total_nodes = int(graph_offsets[n_graphs])
        total_edges = int(edge_offsets[n_graphs])

        # Allocate global arrays
        # row_offsets has total_nodes + 1 entries (one extra sentinel per the
        # standard CSR convention applied to the *global* array).
        global_row_offsets = np.empty(total_nodes + 1, dtype=np.int32)
        global_col_indices = np.empty(total_edges, dtype=np.int32)
        global_edge_weights = np.empty(total_edges, dtype=np.float32)
        global_edge_coverages = np.empty(total_edges, dtype=np.float32)
        global_node_coverages = np.empty(total_nodes, dtype=np.float32)
        global_node_starts = np.empty(total_nodes, dtype=np.int64)
        global_node_ends = np.empty(total_nodes, dtype=np.int64)
        global_node_types = np.empty(total_nodes, dtype=np.int8)

        for i, g in enumerate(self._graphs):
            n_off = int(graph_offsets[i])
            e_off = int(edge_offsets[i])

            # Copy node-level arrays
            ns = n_off
            ne = n_off + g.n_nodes
            global_node_coverages[ns:ne] = g.node_coverages
            global_node_starts[ns:ne] = g.node_starts
            global_node_ends[ns:ne] = g.node_ends
            global_node_types[ns:ne] = g.node_types

            # Copy row_offsets, shifting by the global edge offset
            global_row_offsets[ns:ne] = g.row_offsets[:g.n_nodes] + e_off

            # Copy edge-level arrays, shifting col_indices by the node offset
            es = e_off
            ee = e_off + g.n_edges
            global_col_indices[es:ee] = g.col_indices + n_off
            global_edge_weights[es:ee] = g.edge_weights
            global_edge_coverages[es:ee] = g.edge_coverages

        # Sentinel value for the last row_offset entry
        global_row_offsets[total_nodes] = total_edges

        self._graph_offsets = graph_offsets
        self._edge_offsets = edge_offsets
        self._row_offsets = global_row_offsets
        self._col_indices = global_col_indices
        self._edge_weights = global_edge_weights
        self._edge_coverages = global_edge_coverages
        self._node_coverages = global_node_coverages
        self._node_starts = global_node_starts
        self._node_ends = global_node_ends
        self._node_types = global_node_types
        self._total_nodes = total_nodes
        self._total_edges = total_edges
        self._finalized = True

    # ---- properties (available after finalize) -----------------------------

    def _require_finalized(self) -> None:
        """Raise if finalize() has not been called."""
        if not self._finalized:
            raise RuntimeError("Batch has not been finalized. Call finalize() first.")

    @property
    def graph_offsets(self) -> np.ndarray:
        """Node offset per graph (int64, length ``n_graphs + 1``)."""
        self._require_finalized()
        assert self._graph_offsets is not None
        return self._graph_offsets

    @property
    def edge_offsets(self) -> np.ndarray:
        """Edge offset per graph (int64, length ``n_graphs + 1``)."""
        self._require_finalized()
        assert self._edge_offsets is not None
        return self._edge_offsets

    @property
    def row_offsets(self) -> np.ndarray:
        """Global CSR row pointers (int32, length ``total_nodes + 1``)."""
        self._require_finalized()
        assert self._row_offsets is not None
        return self._row_offsets

    @property
    def col_indices(self) -> np.ndarray:
        """Global CSR column indices (int32, length ``total_edges``)."""
        self._require_finalized()
        assert self._col_indices is not None
        return self._col_indices

    @property
    def edge_weights(self) -> np.ndarray:
        """All edge weights packed contiguously (float32, length ``total_edges``)."""
        self._require_finalized()
        assert self._edge_weights is not None
        return self._edge_weights

    @property
    def node_coverages(self) -> np.ndarray:
        """All node coverages packed contiguously (float32, length ``total_nodes``)."""
        self._require_finalized()
        assert self._node_coverages is not None
        return self._node_coverages

    @property
    def n_graphs(self) -> int:
        """Number of graphs in the batch."""
        self._require_finalized()
        return len(self._graphs)

    @property
    def total_nodes(self) -> int:
        """Total number of nodes across all batched graphs."""
        self._require_finalized()
        return self._total_nodes

    @property
    def total_edges(self) -> int:
        """Total number of edges across all batched graphs."""
        self._require_finalized()
        return self._total_edges

    # ---- queries -----------------------------------------------------------

    def get_graph_range(self, graph_idx: int) -> tuple[int, int, int, int]:
        """Return the node and edge index ranges for a given graph.

        Parameters:
            graph_idx: 0-based graph index within the batch.

        Returns:
            A tuple ``(node_start, node_end, edge_start, edge_end)`` where
            each pair defines a half-open range into the global arrays.

        Raises:
            RuntimeError: If the batch has not been finalized.
            IndexError: If *graph_idx* is out of range.
        """
        self._require_finalized()
        assert self._graph_offsets is not None
        assert self._edge_offsets is not None

        if graph_idx < 0 or graph_idx >= len(self._graphs):
            raise IndexError(
                f"graph_idx {graph_idx} out of range [0, {len(self._graphs)})."
            )

        node_start = int(self._graph_offsets[graph_idx])
        node_end = int(self._graph_offsets[graph_idx + 1])
        edge_start = int(self._edge_offsets[graph_idx])
        edge_end = int(self._edge_offsets[graph_idx + 1])

        return node_start, node_end, edge_start, edge_end

    def get_meta(self, graph_idx: int) -> dict[str, Any]:
        """Return the metadata dictionary for a given graph.

        Parameters:
            graph_idx: 0-based graph index within the batch.

        Returns:
            The metadata dict passed to :meth:`add_graph`.
        """
        self._require_finalized()
        if graph_idx < 0 or graph_idx >= len(self._graphs):
            raise IndexError(
                f"graph_idx {graph_idx} out of range [0, {len(self._graphs)})."
            )
        return self._meta[graph_idx]

    def __repr__(self) -> str:
        if self._finalized:
            return (
                f"BatchedCSRGraphs(n_graphs={len(self._graphs)}, "
                f"total_nodes={self._total_nodes}, "
                f"total_edges={self._total_edges}, finalized=True)"
            )
        return (
            f"BatchedCSRGraphs(pending_graphs={len(self._graphs)}, finalized=False)"
        )
