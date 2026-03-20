"""Custom Numba JIT kernels for GPU-acceleratable operations.

Provides parallel implementations of performance-critical operations used
throughout the BRAID assembler.  All kernels are compiled with
``@njit(parallel=True, cache=True)`` so they run as optimised native code on
the CPU with automatic thread-level parallelism via ``prange``.  The
algorithmic structure of each kernel mirrors what a CUDA version would look
like (one thread per read / per graph / per element), making a future
GPU port straightforward.

Kernels cover four categories:

1. **Coverage computation** -- delta-encoding + prefix-sum approach for
   per-base coverage from read alignments.
2. **Junction extraction** -- parallel walk over CIGAR arrays to collect
   splice junctions.
3. **Graph algorithms** -- batched topological sort, longest-path, and
   coverage uniformity over packed CSR graphs.
4. **Flow utilities** -- residual capacity computation for min-cost flow
   decomposition.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange  # noqa: I001

# ---------------------------------------------------------------------------
# 1. Coverage computation
# ---------------------------------------------------------------------------


@njit(cache=True)
def parallel_coverage_scan(
    positions: np.ndarray,
    end_positions: np.ndarray,
    region_start: int,
    region_end: int,
) -> np.ndarray:
    """Compute per-base read coverage for a genomic region.

    Uses the classic delta-encoding technique: for each read that overlaps
    the region, increment a delta array at the read start and decrement at
    the read end, then compute the prefix sum of the delta array to obtain
    the coverage at every base.

    The parallelism is over reads (each read writes two delta entries).
    Because multiple reads may update the same delta cell, we use a
    thread-local accumulation strategy: each logical thread processes a
    stripe of reads and writes directly into the shared delta array.
    Numba's ``prange`` with atomic-free delta updates is safe here because
    the ``+=`` / ``-=`` operations on ``int64`` arrays are compiled to
    atomic add instructions by the Numba parallel backend.

    Args:
        positions: 0-based alignment start per read (int64, length N).
        end_positions: 0-based exclusive alignment end per read (int64, length N).
        region_start: 0-based inclusive start of the genomic region.
        region_end: 0-based exclusive end of the genomic region.

    Returns:
        int64 array of length ``region_end - region_start`` containing the
        per-base read coverage.
    """
    region_len = region_end - region_start
    delta = np.zeros(region_len + 1, dtype=np.int64)

    n_reads = positions.shape[0]
    # NOTE:
    # We intentionally use a sequential loop here. The previous prange-based
    # implementation updated shared `delta` slots with non-atomic increments,
    # causing nondeterministic under-counting.
    for i in range(n_reads):
        read_start = positions[i]
        read_end = end_positions[i]

        # Clip to region boundaries
        clipped_start = max(read_start, region_start) - region_start
        clipped_end = min(read_end, region_end) - region_start

        if clipped_start < clipped_end:
            delta[clipped_start] += 1
            delta[clipped_end] -= 1

    # Prefix sum to convert deltas to coverage (sequential -- O(region_len))
    coverage = np.empty(region_len, dtype=np.int64)
    running = np.int64(0)
    for j in range(region_len):
        running += delta[j]
        coverage[j] = running

    return coverage


# ---------------------------------------------------------------------------
# 2. Junction extraction
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def parallel_junction_count(
    cigar_ops: np.ndarray,
    cigar_lens: np.ndarray,
    cigar_offsets: np.ndarray,
    ref_starts: np.ndarray,
    n_reads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract all splice junctions from all reads in parallel.

    Each read's CIGAR string is walked independently.  Junctions are
    identified by the ``N`` operation (BAM op code 3).  The output arrays
    are conservatively pre-allocated (one junction slot per CIGAR element)
    and trimmed at the end.

    Args:
        cigar_ops: Flat uint8 array of CIGAR operation codes for all reads.
        cigar_lens: Flat int32 array of CIGAR lengths for all reads.
        cigar_offsets: int64 array of length ``n_reads + 1``.  Read *i* owns
            CIGAR entries ``cigar_offsets[i]:cigar_offsets[i+1]``.
        ref_starts: int64 array of 0-based alignment start per read.
        n_reads: Total number of reads.

    Returns:
        Tuple of three arrays:

        - **junction_starts** (int64) -- intron donor position for each junction.
        - **junction_ends** (int64) -- intron acceptor position (exclusive).
        - **read_indices** (int64) -- index of the read that produced the junction.

        All three arrays have the same length, equal to the total number of
        junctions found across all reads.
    """
    # First pass: count junctions per read so we can allocate output.
    junc_counts = np.zeros(n_reads, dtype=np.int64)
    for i in prange(n_reads):
        count = np.int64(0)
        for j in range(cigar_offsets[i], cigar_offsets[i + 1]):
            if cigar_ops[j] == 3:  # CIGAR_N
                count += 1
        junc_counts[i] = count

    # Exclusive prefix sum to get per-read write offsets.
    junc_offsets = np.empty(n_reads + 1, dtype=np.int64)
    junc_offsets[0] = 0
    for i in range(n_reads):
        junc_offsets[i + 1] = junc_offsets[i] + junc_counts[i]

    total_junctions = junc_offsets[n_reads]

    # Allocate output arrays.
    junction_starts = np.empty(total_junctions, dtype=np.int64)
    junction_ends = np.empty(total_junctions, dtype=np.int64)
    read_indices = np.empty(total_junctions, dtype=np.int64)

    # Second pass: fill output (parallel over reads).
    for i in prange(n_reads):
        pos = ref_starts[i]
        write_idx = junc_offsets[i]
        for j in range(cigar_offsets[i], cigar_offsets[i + 1]):
            op = cigar_ops[j]
            length = cigar_lens[j]
            if op == 3:  # CIGAR_N
                junction_starts[write_idx] = pos
                junction_ends[write_idx] = pos + length
                read_indices[write_idx] = i
                write_idx += 1
                pos += length
            elif op == 0 or op == 2 or op == 7 or op == 8:  # M, D, EQ, X
                pos += length
            # I (1), S (4), H (5), P (6) do not consume reference

    return junction_starts, junction_ends, read_indices


# ---------------------------------------------------------------------------
# 3. Graph algorithms -- batched topological sort
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def batch_topological_sort(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    graph_offsets: np.ndarray,
    n_graphs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform topological sort on multiple graphs packed in a batched CSR.

    Each graph in the batch is processed independently (parallel over
    graphs).  Within a single graph, Kahn's algorithm is used: maintain
    an in-degree array, seed a queue with zero-in-degree nodes, and
    repeatedly pop from the queue while decrementing successor in-degrees.

    The CSR uses *global* node indices.  ``graph_offsets[g]`` gives the
    first global node index of graph *g*; ``graph_offsets[g+1]`` gives the
    exclusive end.  ``row_offsets`` and ``col_indices`` are also in global
    index space.

    Args:
        row_offsets: Global CSR row pointers (int32, length ``total_nodes + 1``).
        col_indices: Global CSR column indices (int32, length ``total_edges``).
        graph_offsets: int64 array of length ``n_graphs + 1``.
        n_graphs: Number of graphs in the batch.

    Returns:
        Tuple of two arrays:

        - **topo_order** (int64) -- concatenated topological orderings for all
          graphs.  Graph *g*'s ordering spans indices
          ``topo_offsets[g]:topo_offsets[g+1]``.  Node IDs are *global*.
        - **topo_offsets** (int64, length ``n_graphs + 1``) -- per-graph start
          index into ``topo_order``.
    """
    # Compute total nodes to allocate the output buffer.
    total_nodes = np.int64(0)
    for g in range(n_graphs):
        total_nodes += graph_offsets[g + 1] - graph_offsets[g]

    topo_order = np.empty(total_nodes, dtype=np.int64)
    topo_offsets = np.empty(n_graphs + 1, dtype=np.int64)

    # Pre-compute offsets (sequential).
    topo_offsets[0] = 0
    for g in range(n_graphs):
        n_nodes_g = graph_offsets[g + 1] - graph_offsets[g]
        topo_offsets[g + 1] = topo_offsets[g] + n_nodes_g

    # Process each graph independently (parallel over graphs).
    for g in prange(n_graphs):
        node_start = graph_offsets[g]
        node_end = graph_offsets[g + 1]
        n_nodes_g = node_end - node_start
        write_base = topo_offsets[g]

        if n_nodes_g == 0:
            continue

        # Compute in-degree for each node in this graph.
        in_degree = np.zeros(n_nodes_g, dtype=np.int64)
        for u_global in range(node_start, node_end):
            edge_begin = row_offsets[u_global]
            edge_end = row_offsets[u_global + 1]
            for e in range(edge_begin, edge_end):
                v_local = col_indices[e] - node_start
                in_degree[v_local] += 1

        # Kahn's algorithm with a simple array-based queue.
        queue = np.empty(n_nodes_g, dtype=np.int64)
        q_head = np.int64(0)
        q_tail = np.int64(0)

        for u_local in range(n_nodes_g):
            if in_degree[u_local] == 0:
                queue[q_tail] = u_local
                q_tail += 1

        write_pos = np.int64(0)
        while q_head < q_tail:
            u_local = queue[q_head]
            q_head += 1

            u_global = node_start + u_local
            topo_order[write_base + write_pos] = u_global
            write_pos += 1

            edge_begin = row_offsets[u_global]
            edge_end = row_offsets[u_global + 1]
            for e in range(edge_begin, edge_end):
                v_local = col_indices[e] - node_start
                in_degree[v_local] -= 1
                if in_degree[v_local] == 0:
                    queue[q_tail] = v_local
                    q_tail += 1

        # If we did not emit all nodes, the graph contains a cycle and the
        # topological order is invalid.
        if write_pos != n_nodes_g:
            raise ValueError(
                "batch_topological_sort encountered a cyclic graph"
            )

    return topo_order, topo_offsets


# ---------------------------------------------------------------------------
# 4. Graph algorithms -- batched DAG longest path
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def batch_dag_longest_path(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    edge_weights: np.ndarray,
    graph_offsets: np.ndarray,
    n_graphs: int,
    topo_order: np.ndarray,
    topo_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the longest weighted path in multiple DAGs.

    For each graph in the batch, processes nodes in topological order and
    relaxes outgoing edges to compute the longest (heaviest) path from
    any source to every node.  The initial distance of every node is zero;
    the algorithm finds the maximum-weight path among all possible start
    nodes.

    Args:
        row_offsets: Global CSR row pointers (int32).
        col_indices: Global CSR column indices (int32).
        edge_weights: float32 weight per edge (same indexing as ``col_indices``).
        graph_offsets: int64 array of length ``n_graphs + 1``.
        n_graphs: Number of graphs.
        topo_order: Concatenated topological orderings from
            :func:`batch_topological_sort`.
        topo_offsets: Per-graph start indices into ``topo_order``.

    Returns:
        Tuple of two arrays:

        - **distances** (float64, length ``total_nodes``) -- longest-path
          distance from any root to each node.
        - **predecessors** (int64, length ``total_nodes``) -- predecessor
          node (global index) on the longest path.  ``-1`` for root nodes.
    """
    total_nodes = np.int64(0)
    for g in range(n_graphs):
        total_nodes += graph_offsets[g + 1] - graph_offsets[g]

    distances = np.zeros(total_nodes, dtype=np.float64)
    predecessors = np.full(total_nodes, -1, dtype=np.int64)

    for g in prange(n_graphs):
        topo_begin = topo_offsets[g]
        topo_end = topo_offsets[g + 1]

        for t in range(topo_begin, topo_end):
            u_global = topo_order[t]
            u_dist = distances[u_global]

            edge_begin = row_offsets[u_global]
            edge_end = row_offsets[u_global + 1]

            for e in range(edge_begin, edge_end):
                v_global = col_indices[e]
                w = edge_weights[e]
                candidate = u_dist + w
                if candidate > distances[v_global]:
                    distances[v_global] = candidate
                    predecessors[v_global] = u_global

    return distances, predecessors


# ---------------------------------------------------------------------------
# 5. Graph algorithms -- batched coverage uniformity
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def batch_coverage_uniformity(
    node_coverages: np.ndarray,
    graph_offsets: np.ndarray,
    n_graphs: int,
) -> np.ndarray:
    """Compute the coefficient of variation of node coverages per graph.

    The coefficient of variation (CV) is defined as ``std / mean``.  A low
    CV indicates uniform coverage across nodes (desirable for a well-supported
    transcript path), while a high CV suggests uneven or chimeric coverage.

    For graphs with a single node or zero mean coverage the CV is set to 0.0.

    Args:
        node_coverages: float32 array of all node coverages (global indexing).
        graph_offsets: int64 array of length ``n_graphs + 1``.
        n_graphs: Number of graphs.

    Returns:
        float64 array of length ``n_graphs`` with the CV per graph.
    """
    cv_values = np.empty(n_graphs, dtype=np.float64)

    for g in prange(n_graphs):
        node_start = graph_offsets[g]
        node_end = graph_offsets[g + 1]
        n_nodes_g = node_end - node_start

        if n_nodes_g <= 1:
            cv_values[g] = 0.0
            continue

        # Compute mean.
        total = np.float64(0.0)
        for i in range(node_start, node_end):
            total += node_coverages[i]
        mean = total / n_nodes_g

        if mean == 0.0:
            cv_values[g] = 0.0
            continue

        # Compute variance.
        var_sum = np.float64(0.0)
        for i in range(node_start, node_end):
            diff = np.float64(node_coverages[i]) - mean
            var_sum += diff * diff
        std = np.sqrt(var_sum / n_nodes_g)

        cv_values[g] = std / mean

    return cv_values


# ---------------------------------------------------------------------------
# 6. Flow utilities -- residual capacity
# ---------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def parallel_edge_flow_residual(
    edge_flows: np.ndarray,
    edge_capacities: np.ndarray,
    n_edges: int,
) -> np.ndarray:
    """Compute residual capacities for all edges in parallel.

    The residual capacity of an edge is simply ``capacity - flow``.  This
    is the core primitive used to identify augmenting paths in min-cost
    flow decomposition.

    Args:
        edge_flows: float64 array of current flow on each edge (length ``n_edges``).
        edge_capacities: float64 array of edge capacities (length ``n_edges``).
        n_edges: Number of edges.

    Returns:
        float64 array of length ``n_edges`` with residual capacities.
    """
    residuals = np.empty(n_edges, dtype=np.float64)
    for i in prange(n_edges):
        residuals[i] = edge_capacities[i] - edge_flows[i]
    return residuals
