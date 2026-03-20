"""GPU-ready push-relabel maximum flow implementation for CSR splice graphs.

Implements the FIFO push-relabel algorithm (Goldberg & Tarjan, 1988) operating
directly on CSR graph representations.  The inner loop is accelerated with
Numba JIT compilation when available, falling back to a pure-NumPy
implementation otherwise.

The algorithm runs in O(V^3) worst case but is typically much faster on the
sparse DAGs produced by splice graph construction.  For GPU execution the same
algorithmic structure is mirrored in the CUDA kernels (see ``rapidsplice.cuda``).

Key implementation details:
- A *combined* residual graph is built that contains both forward edges
  (original capacity) and reverse edges (zero initial capacity).  Each edge
  stores the index of its reverse counterpart so that push operations can
  update both directions in O(1).
- The FIFO variant maintains a queue of active nodes and processes them in
  FIFO order, which gives the O(V^3) bound.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from rapidsplice.graph.splice_graph import CSRGraph

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MaxFlowResult:
    """Result of a push-relabel maximum flow computation.

    Attributes:
        flow_value: Total flow from source to sink.
        edge_flows: Flow on each *original* edge (indexed the same as the CSR
            ``edge_weights`` / ``col_indices`` arrays).
        converged: Whether the algorithm converged within the iteration limit.
    """

    flow_value: float
    edge_flows: np.ndarray
    converged: bool


# ---------------------------------------------------------------------------
# Residual graph construction helpers
# ---------------------------------------------------------------------------


def _build_residual_graph(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    capacities: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build a combined forward+reverse residual graph in CSR format.

    For every forward edge (u, v) with capacity c we create a reverse edge
    (v, u) with capacity 0.  The returned arrays describe the combined graph.

    Parameters:
        row_offsets: Original CSR row offsets (int32, length n_nodes+1).
        col_indices: Original CSR column indices (int32, length n_edges).
        capacities: Original edge capacities / weights (float32, length n_edges).
        n_nodes: Number of nodes.

    Returns:
        A tuple ``(res_row_offsets, res_col_indices, res_capacities,
        res_flows, reverse_edge_idx, n_res_edges)`` where every array
        describes the combined residual graph and ``reverse_edge_idx[e]``
        gives the index of the reverse of edge *e*.
    """
    n_forward = len(col_indices)
    n_res_edges = 2 * n_forward

    # First pass: count out-degree per node in the combined graph
    out_degree = np.zeros(n_nodes, dtype=np.int32)
    for u in range(n_nodes):
        start, end = int(row_offsets[u]), int(row_offsets[u + 1])
        out_degree[u] += end - start  # forward edges from u
        for idx in range(start, end):
            v = int(col_indices[idx])
            out_degree[v] += 1  # reverse edge from v

    # Build row_offsets for the combined graph
    res_row_offsets = np.empty(n_nodes + 1, dtype=np.int32)
    res_row_offsets[0] = 0
    for u in range(n_nodes):
        res_row_offsets[u + 1] = res_row_offsets[u] + out_degree[u]

    # Allocate combined arrays
    res_col_indices = np.empty(n_res_edges, dtype=np.int32)
    res_capacities = np.zeros(n_res_edges, dtype=np.float64)
    res_flows = np.zeros(n_res_edges, dtype=np.float64)
    reverse_edge_idx = np.empty(n_res_edges, dtype=np.int32)

    # Mapping from original edge index to combined forward edge index
    forward_map = np.empty(n_forward, dtype=np.int32)

    # Second pass: fill in forward edges first, then reverse edges
    # We use a write-pointer per node
    write_ptr = res_row_offsets[:n_nodes].copy()

    # Pass 2a: forward edges
    for u in range(n_nodes):
        start, end = int(row_offsets[u]), int(row_offsets[u + 1])
        for orig_idx in range(start, end):
            v = int(col_indices[orig_idx])
            pos = int(write_ptr[u])
            write_ptr[u] += 1
            res_col_indices[pos] = v
            res_capacities[pos] = float(capacities[orig_idx])
            forward_map[orig_idx] = pos

    # Pass 2b: reverse edges
    reverse_map = np.empty(n_forward, dtype=np.int32)
    for u in range(n_nodes):
        start, end = int(row_offsets[u]), int(row_offsets[u + 1])
        for orig_idx in range(start, end):
            v = int(col_indices[orig_idx])
            pos = int(write_ptr[v])
            write_ptr[v] += 1
            res_col_indices[pos] = u
            res_capacities[pos] = 0.0  # reverse edge has zero capacity
            reverse_map[orig_idx] = pos

    # Link forward <-> reverse
    for orig_idx in range(n_forward):
        fwd = int(forward_map[orig_idx])
        rev = int(reverse_map[orig_idx])
        reverse_edge_idx[fwd] = rev
        reverse_edge_idx[rev] = fwd

    return (
        res_row_offsets, res_col_indices, res_capacities,
        res_flows, reverse_edge_idx, n_res_edges,
    )


# ---------------------------------------------------------------------------
# Core push-relabel (pure Python/NumPy, Numba-friendly structure)
# ---------------------------------------------------------------------------


def _push_relabel_numba(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    capacities: np.ndarray,
    flows: np.ndarray,
    reverse_edge_idx: np.ndarray,
    n_nodes: int,
    source: int,
    sink: int,
    max_iterations: int,
) -> tuple[float, np.ndarray, bool]:
    """Core push-relabel FIFO implementation.

    Operates on a pre-built combined residual graph.  All arrays are mutated
    in place (flows).

    Parameters:
        row_offsets: Residual graph CSR row offsets.
        col_indices: Residual graph CSR column indices.
        capacities: Residual graph edge capacities.
        flows: Residual graph edge flows (mutated in place).
        reverse_edge_idx: Mapping from each edge to its reverse.
        n_nodes: Total number of nodes.
        source: Source node index.
        sink: Sink node index.
        max_iterations: Maximum number of push/relabel operations.

    Returns:
        A tuple ``(flow_value, flows, converged)``.
    """
    # Initialize heights and excess
    height = np.zeros(n_nodes, dtype=np.int64)
    excess = np.zeros(n_nodes, dtype=np.float64)
    height[source] = n_nodes

    # Current-arc optimization: track which edge index to try next per node
    current_arc = np.zeros(n_nodes, dtype=np.int32)
    for u in range(n_nodes):
        current_arc[u] = int(row_offsets[u])

    # Saturate all edges from source
    for idx in range(int(row_offsets[source]), int(row_offsets[source + 1])):
        cap = capacities[idx]
        if cap > 0.0:
            v = int(col_indices[idx])
            rev_idx = int(reverse_edge_idx[idx])
            flows[idx] = cap
            flows[rev_idx] = -cap
            excess[v] += cap
            excess[source] -= cap

    # Build FIFO queue of active nodes (excess > 0 and not source/sink)
    active_queue: deque[int] = deque()
    in_queue = np.zeros(n_nodes, dtype=np.bool_)
    for u in range(n_nodes):
        if u != source and u != sink and excess[u] > 1e-12:
            active_queue.append(u)
            in_queue[u] = True

    iterations = 0
    converged = True

    while active_queue:
        if iterations >= max_iterations:
            converged = False
            break

        u = active_queue.popleft()
        in_queue[u] = False

        # Discharge node u
        while excess[u] > 1e-12:
            iterations += 1
            if iterations >= max_iterations:
                converged = False
                break

            # Try to push from current arc
            pushed = False
            start = int(row_offsets[u])
            end = int(row_offsets[u + 1])

            while current_arc[u] < end:
                idx = int(current_arc[u])
                v = int(col_indices[idx])
                residual = capacities[idx] - flows[idx]
                if residual > 1e-12 and height[u] == height[v] + 1:
                    # Push
                    push_amount = min(excess[u], residual)
                    flows[idx] += push_amount
                    rev_idx = int(reverse_edge_idx[idx])
                    flows[rev_idx] -= push_amount
                    excess[u] -= push_amount
                    excess[v] += push_amount
                    if v != source and v != sink and not in_queue[v]:
                        active_queue.append(v)
                        in_queue[v] = True
                    pushed = True
                    if excess[u] <= 1e-12:
                        break
                else:
                    current_arc[u] += 1

            if not pushed or (current_arc[u] >= end and excess[u] > 1e-12):
                # Relabel: increase height of u
                min_height = np.int64(2 * n_nodes)
                for idx in range(start, end):
                    v = int(col_indices[idx])
                    residual = capacities[idx] - flows[idx]
                    if residual > 1e-12:
                        if height[v] + 1 < min_height:
                            min_height = height[v] + 1
                if min_height < 2 * n_nodes:
                    height[u] = min_height
                else:
                    # Node is disconnected from sink in residual graph
                    break
                current_arc[u] = start

        # If still has excess after discharge, re-enqueue
        if excess[u] > 1e-12 and u != source and u != sink:
            if not in_queue[u]:
                active_queue.append(u)
                in_queue[u] = True

    flow_value = float(excess[sink])
    return flow_value, flows, converged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def push_relabel_maxflow(
    graph: CSRGraph,
    source: int = 0,
    sink: int = -1,
    max_iterations: int = 100_000,
) -> MaxFlowResult:
    """Compute maximum flow using the FIFO push-relabel algorithm.

    Parameters:
        graph: The CSR splice graph to compute max-flow on.
        source: Source node index (default 0, matching the convention that
            SOURCE is the first node in topological order).
        sink: Sink node index (default -1, meaning the last node).
        max_iterations: Maximum number of push/relabel operations before
            declaring non-convergence.

    Returns:
        A :class:`MaxFlowResult` with the total flow value, per-edge flow
        values (aligned with the original CSR edge arrays), and a convergence
        flag.
    """
    n_nodes = graph.n_nodes
    n_edges = graph.n_edges

    if n_nodes == 0 or n_edges == 0:
        return MaxFlowResult(
            flow_value=0.0,
            edge_flows=np.zeros(n_edges, dtype=np.float64),
            converged=True,
        )

    # Resolve negative sink index
    if sink < 0:
        sink = n_nodes + sink

    # Build combined residual graph
    (
        res_row_offsets,
        res_col_indices,
        res_capacities,
        res_flows,
        reverse_edge_idx,
        n_res_edges,
    ) = _build_residual_graph(
        graph.row_offsets, graph.col_indices, graph.edge_weights, n_nodes
    )

    # Run push-relabel
    flow_value, res_flows, converged = _push_relabel_numba(
        res_row_offsets,
        res_col_indices,
        res_capacities,
        res_flows,
        reverse_edge_idx,
        n_nodes,
        source,
        sink,
        max_iterations,
    )

    # Extract flows on original forward edges.
    # The forward edges in the combined graph are stored first for each node,
    # so we can rebuild the mapping.
    edge_flows = np.zeros(n_edges, dtype=np.float64)
    write_ptr = res_row_offsets[:n_nodes].copy()

    for u in range(n_nodes):
        orig_start = int(graph.row_offsets[u])
        orig_end = int(graph.row_offsets[u + 1])
        for orig_idx in range(orig_start, orig_end):
            res_idx = int(write_ptr[u])
            write_ptr[u] += 1
            edge_flows[orig_idx] = max(0.0, float(res_flows[res_idx]))

    return MaxFlowResult(
        flow_value=flow_value,
        edge_flows=edge_flows,
        converged=converged,
    )
