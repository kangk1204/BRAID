"""Min-cost flow solver for splice graphs via successive shortest paths.

Implements the successive shortest paths (SSP) algorithm with Bellman-Ford
shortest path computation on the residual graph.  This is the standard
approach for min-cost flow on general networks (Ahuja, Magnanti & Orlin 1993,
Chapter 9).

For splice graph transcript assembly the edge costs are set to the negative
of edge weights so that maximizing weighted flow is equivalent to minimizing
cost.  The algorithm iteratively finds shortest augmenting paths from supply
nodes to demand nodes and pushes flow along them until all demand is met or
no augmenting path exists.

Additionally provides a greedy flow decomposition utility that converts an
edge-flow solution into a set of weighted source-to-sink paths, which is the
final step in transcript enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from braid.graph.splice_graph import CSRGraph

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MinCostFlowResult:
    """Result of a min-cost flow computation.

    Attributes:
        total_cost: Total cost of the optimal flow.
        edge_flows: Flow value on each edge (aligned with CSR edge arrays).
        converged: Whether the algorithm found a feasible flow satisfying all
            supply/demand constraints.
    """

    total_cost: float
    edge_flows: np.ndarray
    converged: bool


# ---------------------------------------------------------------------------
# Bellman-Ford shortest path
# ---------------------------------------------------------------------------


def _bellman_ford_shortest_path(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    costs: np.ndarray,
    residuals: np.ndarray,
    n_nodes: int,
    source: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Single-source shortest paths via Bellman-Ford on the residual graph.

    Only edges with positive residual capacity are considered.  Supports
    negative edge costs (which arise from the negative-weight formulation).

    Parameters:
        row_offsets: Residual graph CSR row offsets (int32, length n_nodes+1).
        col_indices: Residual graph CSR column indices (int32).
        costs: Residual graph edge costs (float64).
        residuals: Residual capacity per edge (float64); only edges with
            ``residuals[e] > 0`` are traversable.
        n_nodes: Number of nodes.
        source: Source node index.

    Returns:
        A tuple ``(distances, predecessors)`` where ``distances[v]`` is the
        shortest-path cost from *source* to *v* (or ``inf`` if unreachable)
        and ``predecessors[v]`` is the predecessor node on that path (or -1).
    """
    inf = np.float64(1e18)
    dist = np.full(n_nodes, inf, dtype=np.float64)
    pred_node = np.full(n_nodes, -1, dtype=np.int32)
    pred_edge = np.full(n_nodes, -1, dtype=np.int32)
    dist[source] = 0.0

    # Standard Bellman-Ford: n-1 relaxation rounds
    for _ in range(n_nodes - 1):
        updated = False
        for u in range(n_nodes):
            if dist[u] >= inf:
                continue
            start = int(row_offsets[u])
            end = int(row_offsets[u + 1])
            for idx in range(start, end):
                if residuals[idx] <= 1e-12:
                    continue
                v = int(col_indices[idx])
                new_dist = dist[u] + costs[idx]
                if new_dist < dist[v] - 1e-12:
                    dist[v] = new_dist
                    pred_node[v] = u
                    pred_edge[v] = idx
                    updated = True
        if not updated:
            break

    return dist, pred_node


# ---------------------------------------------------------------------------
# Residual graph construction
# ---------------------------------------------------------------------------


def _build_residual_graph_ssp(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    capacities: np.ndarray,
    edge_costs: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build combined forward+reverse residual graph for SSP algorithm.

    Parameters:
        row_offsets: Original CSR row offsets.
        col_indices: Original CSR column indices.
        capacities: Edge capacities.
        edge_costs: Edge costs (may be negative for max-weight flow).
        n_nodes: Number of nodes.

    Returns:
        ``(res_row_offsets, res_col_indices, res_capacities, res_costs,
        res_flows, reverse_edge_idx)`` for the combined graph.
    """
    n_forward = len(col_indices)
    n_res = 2 * n_forward

    # Count out-degree in combined graph
    out_degree = np.zeros(n_nodes, dtype=np.int32)
    for u in range(n_nodes):
        start, end = int(row_offsets[u]), int(row_offsets[u + 1])
        out_degree[u] += end - start
        for idx in range(start, end):
            v = int(col_indices[idx])
            out_degree[v] += 1

    res_row_offsets = np.empty(n_nodes + 1, dtype=np.int32)
    res_row_offsets[0] = 0
    for u in range(n_nodes):
        res_row_offsets[u + 1] = res_row_offsets[u] + out_degree[u]

    res_col_indices = np.empty(n_res, dtype=np.int32)
    res_capacities = np.zeros(n_res, dtype=np.float64)
    res_costs = np.zeros(n_res, dtype=np.float64)
    res_flows = np.zeros(n_res, dtype=np.float64)
    reverse_edge_idx = np.empty(n_res, dtype=np.int32)

    write_ptr = res_row_offsets[:n_nodes].copy()

    # Forward edges
    forward_map = np.empty(n_forward, dtype=np.int32)
    for u in range(n_nodes):
        start, end = int(row_offsets[u]), int(row_offsets[u + 1])
        for orig_idx in range(start, end):
            pos = int(write_ptr[u])
            write_ptr[u] += 1
            res_col_indices[pos] = int(col_indices[orig_idx])
            res_capacities[pos] = float(capacities[orig_idx])
            res_costs[pos] = float(edge_costs[orig_idx])
            forward_map[orig_idx] = pos

    # Reverse edges (capacity 0, negative cost)
    reverse_map = np.empty(n_forward, dtype=np.int32)
    for u in range(n_nodes):
        start, end = int(row_offsets[u]), int(row_offsets[u + 1])
        for orig_idx in range(start, end):
            v = int(col_indices[orig_idx])
            pos = int(write_ptr[v])
            write_ptr[v] += 1
            res_col_indices[pos] = u
            res_capacities[pos] = 0.0
            res_costs[pos] = -float(edge_costs[orig_idx])
            reverse_map[orig_idx] = pos

    # Link forward <-> reverse
    for orig_idx in range(n_forward):
        fwd = int(forward_map[orig_idx])
        rev = int(reverse_map[orig_idx])
        reverse_edge_idx[fwd] = rev
        reverse_edge_idx[rev] = fwd

    return res_row_offsets, res_col_indices, res_capacities, res_costs, res_flows, reverse_edge_idx


# ---------------------------------------------------------------------------
# Successive shortest paths algorithm
# ---------------------------------------------------------------------------


def min_cost_flow(
    graph: CSRGraph,
    supply: np.ndarray | None = None,
    demand: np.ndarray | None = None,
) -> MinCostFlowResult:
    """Compute a minimum-cost flow on the splice graph.

    Uses the successive shortest paths algorithm with Bellman-Ford.  Edge
    costs are set to ``-edge_weights`` so that maximizing the total weighted
    flow is equivalent to minimizing cost.

    Parameters:
        graph: The CSR splice graph.
        supply: Per-node supply values (positive = produces flow).  If
            ``None``, the source node (node 0) is given supply equal to
            the sum of its outgoing capacities.
        demand: Per-node demand values (positive = consumes flow).  If
            ``None``, the sink node (last node) absorbs all flow.

    Returns:
        A :class:`MinCostFlowResult` containing the total cost, edge flows,
        and convergence status.
    """
    n_nodes = graph.n_nodes
    n_edges = graph.n_edges

    if n_nodes == 0 or n_edges == 0:
        return MinCostFlowResult(
            total_cost=0.0,
            edge_flows=np.zeros(n_edges, dtype=np.float64),
            converged=True,
        )

    source = 0
    sink = n_nodes - 1

    # Determine supply and demand
    if supply is None or demand is None:
        # Source supplies the sum of outgoing edge capacities
        total_supply = float(
            np.sum(
                graph.edge_weights[
                    int(graph.row_offsets[source]) : int(graph.row_offsets[source + 1])
                ]
            )
        )
        supply_arr = np.zeros(n_nodes, dtype=np.float64)
        demand_arr = np.zeros(n_nodes, dtype=np.float64)
        supply_arr[source] = total_supply
        demand_arr[sink] = total_supply
    else:
        supply_arr = supply.astype(np.float64, copy=True)
        demand_arr = demand.astype(np.float64, copy=True)

    # Edge costs = negative weights (we minimize cost -> maximize weighted flow)
    edge_costs = -graph.edge_weights.astype(np.float64)

    # Build residual graph
    (
        res_row_offsets,
        res_col_indices,
        res_capacities,
        res_costs,
        res_flows,
        reverse_edge_idx,
    ) = _build_residual_graph_ssp(
        graph.row_offsets, graph.col_indices, graph.edge_weights, edge_costs, n_nodes
    )

    # Net supply = supply - demand
    net_supply = supply_arr - demand_arr
    excess = net_supply.copy()

    total_cost = 0.0
    max_augmentations = n_nodes * n_edges + 1  # safeguard
    converged = True

    for _ in range(max_augmentations):
        # Find a supply node with positive excess
        supply_node = -1
        for u in range(n_nodes):
            if excess[u] > 1e-12:
                supply_node = u
                break
        if supply_node == -1:
            break  # all supply consumed

        # Find shortest path from supply_node to any demand node
        residuals = res_capacities - res_flows
        dist, pred_node = _bellman_ford_shortest_path(
            res_row_offsets, res_col_indices, res_costs, residuals, n_nodes, supply_node
        )

        # Find best demand node (negative excess and reachable)
        best_demand = -1
        best_dist = np.float64(1e18)
        for u in range(n_nodes):
            if excess[u] < -1e-12 and dist[u] < best_dist:
                best_dist = dist[u]
                best_demand = u
        if best_demand == -1:
            converged = False
            break

        # Trace path and find bottleneck
        path_nodes: list[int] = []
        v = best_demand
        while v != supply_node:
            path_nodes.append(v)
            v = int(pred_node[v])
            if v == -1:
                break
        if v == -1:
            converged = False
            break
        path_nodes.append(supply_node)
        path_nodes.reverse()

        # Find the bottleneck flow along the path
        bottleneck = min(excess[supply_node], -excess[best_demand])
        current = supply_node
        for i in range(1, len(path_nodes)):
            nxt = path_nodes[i]
            # Find the edge current -> nxt in the residual graph
            for idx in range(int(res_row_offsets[current]), int(res_row_offsets[current + 1])):
                residual_cap = res_capacities[idx] - res_flows[idx]
                if int(res_col_indices[idx]) == nxt and residual_cap > 1e-12:
                    bottleneck = min(bottleneck, residual_cap)
                    break
            current = nxt

        if bottleneck <= 1e-12:
            converged = False
            break

        # Augment flow along the path
        current = supply_node
        for i in range(1, len(path_nodes)):
            nxt = path_nodes[i]
            for idx in range(int(res_row_offsets[current]), int(res_row_offsets[current + 1])):
                res_cap = res_capacities[idx] - res_flows[idx]
                if int(res_col_indices[idx]) == nxt and res_cap > 1e-12:
                    res_flows[idx] += bottleneck
                    rev = int(reverse_edge_idx[idx])
                    res_flows[rev] -= bottleneck
                    total_cost += bottleneck * res_costs[idx]
                    break
            current = nxt

        excess[supply_node] -= bottleneck
        excess[best_demand] += bottleneck

    # Extract original edge flows from the residual graph
    edge_flows = np.zeros(n_edges, dtype=np.float64)
    write_ptr = res_row_offsets[:n_nodes].copy()
    for u in range(n_nodes):
        orig_start = int(graph.row_offsets[u])
        orig_end = int(graph.row_offsets[u + 1])
        for orig_idx in range(orig_start, orig_end):
            res_idx = int(write_ptr[u])
            write_ptr[u] += 1
            edge_flows[orig_idx] = max(0.0, float(res_flows[res_idx]))

    return MinCostFlowResult(
        total_cost=total_cost,
        edge_flows=edge_flows,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Flow decomposition into weighted paths
# ---------------------------------------------------------------------------


def flow_to_weighted_paths(
    graph: CSRGraph,
    edge_flows: np.ndarray,
) -> list[tuple[list[int], float]]:
    """Decompose an edge flow into weighted source-to-sink paths.

    Uses a greedy algorithm: repeatedly find a path from source (node 0) to
    sink (last node) in the residual flow graph, extract the minimum flow
    along it, and subtract.

    Parameters:
        graph: The CSR splice graph.
        edge_flows: Flow on each edge (float64, length n_edges).

    Returns:
        A list of ``(path, weight)`` tuples where *path* is a list of node
        IDs from source to sink and *weight* is the flow value along that
        path.
    """
    n_nodes = graph.n_nodes
    if n_nodes == 0:
        return []

    source = 0
    sink = n_nodes - 1

    remaining = edge_flows.astype(np.float64, copy=True)
    paths: list[tuple[list[int], float]] = []

    max_decompositions = int(np.sum(edge_flows > 1e-12)) + 1

    for _ in range(max_decompositions):
        # DFS / greedy path from source to sink following positive-flow edges
        path = _find_flow_path(
            graph.row_offsets, graph.col_indices, remaining, n_nodes, source, sink
        )
        if path is None:
            break

        # Find bottleneck flow on this path
        min_flow = np.float64(1e18)
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            for idx in range(int(graph.row_offsets[u]), int(graph.row_offsets[u + 1])):
                if int(graph.col_indices[idx]) == v:
                    if remaining[idx] < min_flow:
                        min_flow = remaining[idx]
                    break

        if min_flow <= 1e-12:
            break

        # Subtract flow from path
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            for idx in range(int(graph.row_offsets[u]), int(graph.row_offsets[u + 1])):
                if int(graph.col_indices[idx]) == v:
                    remaining[idx] -= min_flow
                    break

        paths.append((path, float(min_flow)))

    return paths


def _find_flow_path(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    flows: np.ndarray,
    n_nodes: int,
    source: int,
    sink: int,
) -> list[int] | None:
    """Find a source-to-sink path along edges with positive flow using DFS.

    At each node, the outgoing edge with the largest remaining flow is chosen
    to produce paths that carry as much flow as possible (reducing the number
    of decomposition rounds).

    Parameters:
        row_offsets: CSR row offsets.
        col_indices: CSR column indices.
        flows: Remaining flow per edge.
        n_nodes: Number of nodes.
        source: Source node index.
        sink: Sink node index.

    Returns:
        A list of node IDs from source to sink, or ``None`` if no path with
        positive flow exists.
    """
    predecessor = np.full(n_nodes, -1, dtype=np.int32)
    visited = np.zeros(n_nodes, dtype=np.bool_)
    stack: list[int] = [source]
    visited[source] = True

    while stack:
        u = stack[-1]
        if u == sink:
            # Reconstruct path
            path: list[int] = []
            v = sink
            while v != -1:
                path.append(v)
                v = int(predecessor[v])
            path.reverse()
            return path

        # Find unvisited neighbor with largest positive flow
        best_v = -1
        best_flow = 0.0
        for idx in range(int(row_offsets[u]), int(row_offsets[u + 1])):
            v = int(col_indices[idx])
            if not visited[v] and flows[idx] > 1e-12:
                if flows[idx] > best_flow:
                    best_flow = flows[idx]
                    best_v = v

        if best_v >= 0:
            predecessor[best_v] = u
            visited[best_v] = True
            stack.append(best_v)
        else:
            stack.pop()

    return None
