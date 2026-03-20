"""Safe flow decomposition for splice graphs.

Implements the safe-path decomposition framework inspired by the theory of
*safe and complete* flow decompositions (WABI 2024, Cáceres et al.).  A
subpath is *safe* if it appears in **every** optimal flow decomposition of the
graph.  Safe paths provide high-confidence transcript fragments that do not
depend on the (typically non-unique) choice of decomposition.

The algorithm works as follows:

1. Solve a min-cost flow LP on the splice graph to obtain an optimal edge
   flow vector.
2. For each edge, determine whether its flow value is *uniquely determined*
   by solving two auxiliary LPs that attempt to increase and decrease the
   edge flow while maintaining optimality.  An edge whose flow cannot
   change in either direction is "safe".
3. Chain consecutive safe edges into maximal safe subpaths.

The LP relaxation is handled by :func:`scipy.optimize.linprog`.  Because
splice graphs are DAGs with moderate size (typically < 10 000 edges per
locus), the LP solves are fast even without GPU acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import OptimizeResult, linprog

from rapidsplice.graph.splice_graph import CSRGraph

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SafePathResult:
    """Result of safe flow decomposition.

    Attributes:
        paths: List of safe subpaths, each a sequence of node IDs.
        weights: Flow value associated with each safe subpath.
        coverage_fraction: Fraction of total graph flow that is explained
            by the safe subpaths (sum of safe-path weights / total flow).
    """

    paths: list[list[int]] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    coverage_fraction: float = 0.0


# ---------------------------------------------------------------------------
# LP-based optimal flow solver
# ---------------------------------------------------------------------------


def _solve_flow_lp(
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    edge_weights: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Solve the min-cost flow LP to find optimal edge flows.

    Formulation:
        minimize   -sum_e  w_e * f_e           (maximize weighted flow)
        subject to sum_{out(v)} f_e - sum_{in(v)} f_e = 0   for interior v
                   0 <= f_e <= w_e             for each edge e

    The source (node 0) and sink (node n_nodes-1) are unconstrained in
    conservation (they produce / absorb flow).

    Parameters:
        row_offsets: CSR row offsets (int32, length n_nodes+1).
        col_indices: CSR column indices (int32, length n_edges).
        edge_weights: Edge weights / capacities (float32, length n_edges).
        n_nodes: Number of nodes in the graph.

    Returns:
        Optimal edge flow vector (float64, length n_edges).
    """
    n_edges = len(col_indices)
    if n_edges == 0:
        return np.zeros(0, dtype=np.float64)

    source = 0
    sink = n_nodes - 1

    # Objective: minimize -w^T f  (equivalently maximize weighted flow)
    c = -edge_weights.astype(np.float64)

    # Flow conservation: for each interior node v (not source, not sink),
    #   sum of outgoing flows - sum of incoming flows = 0
    # This gives (n_nodes - 2) equality constraints.
    interior_nodes = [v for v in range(n_nodes) if v != source and v != sink]
    n_constraints = len(interior_nodes)

    if n_constraints == 0:
        # Trivial graph (e.g. source -> sink directly): max flow on each edge
        return edge_weights.astype(np.float64).clip(min=0.0)

    # Build the incidence sub-matrix for interior nodes
    # A_eq[i, e] = +1 if edge e leaves interior_node[i]
    # A_eq[i, e] = -1 if edge e enters interior_node[i]
    node_to_row = {v: i for i, v in enumerate(interior_nodes)}
    A_eq = np.zeros((n_constraints, n_edges), dtype=np.float64)
    b_eq = np.zeros(n_constraints, dtype=np.float64)

    for u in range(n_nodes):
        start = int(row_offsets[u])
        end = int(row_offsets[u + 1])
        for idx in range(start, end):
            v = int(col_indices[idx])
            if u in node_to_row:
                A_eq[node_to_row[u], idx] += 1.0  # outgoing from u
            if v in node_to_row:
                A_eq[node_to_row[v], idx] -= 1.0  # incoming to v

    # Bounds: 0 <= f_e <= w_e
    bounds = [(0.0, max(float(edge_weights[e]), 0.0)) for e in range(n_edges)]

    result: OptimizeResult = linprog(
        c,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"presolve": True, "disp": False},
    )

    if result.success:
        return np.asarray(result.x, dtype=np.float64)

    # Fallback: return zero flow if LP is infeasible
    return np.zeros(n_edges, dtype=np.float64)


# ---------------------------------------------------------------------------
# Safe edge identification
# ---------------------------------------------------------------------------


def _identify_safe_edges(
    flows: np.ndarray,
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    edge_weights: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Identify edges whose flow is uniquely determined in all optimal solutions.

    For each edge *e* with optimal flow *f_e > 0*, we check whether *f_e*
    can be decreased or increased while maintaining the same optimal
    objective value.  If neither direction allows change, the edge is safe.

    We solve two auxiliary LPs per edge:
    - minimize f_e   subject to the same constraints + optimal objective bound
    - maximize f_e   subject to the same constraints + optimal objective bound

    If both give the same value as *f_e*, the edge is safe.

    To avoid solving 2*n_edges LPs (which could be expensive), we use a
    batch approach: solve a single LP with the optimal objective as a
    constraint, then probe each edge by checking reduced costs and basis
    status from the HiGHS solver.

    For practical splice graphs (< 10k edges), we use the direct per-edge
    probing approach with early termination for zero-flow edges.

    Parameters:
        flows: Optimal edge flow values (float64, length n_edges).
        row_offsets: CSR row offsets.
        col_indices: CSR column indices.
        edge_weights: Edge capacities.
        n_nodes: Number of nodes.

    Returns:
        Boolean array (length n_edges) where ``True`` indicates a safe edge.
    """
    n_edges = len(col_indices)
    safe = np.zeros(n_edges, dtype=np.bool_)

    if n_edges == 0:
        return safe

    source = 0
    sink = n_nodes - 1

    # Optimal objective value
    opt_obj = float(-np.dot(edge_weights.astype(np.float64), flows))

    # Build conservation constraints (same as in _solve_flow_lp)
    interior_nodes = [v for v in range(n_nodes) if v != source and v != sink]
    n_conservation = len(interior_nodes)
    node_to_row = {v: i for i, v in enumerate(interior_nodes)}

    A_eq = np.zeros((n_conservation + 1, n_edges), dtype=np.float64)
    b_eq = np.zeros(n_conservation + 1, dtype=np.float64)

    for u in range(n_nodes):
        start = int(row_offsets[u])
        end = int(row_offsets[u + 1])
        for idx in range(start, end):
            v = int(col_indices[idx])
            if u in node_to_row:
                A_eq[node_to_row[u], idx] += 1.0
            if v in node_to_row:
                A_eq[node_to_row[v], idx] -= 1.0

    # Last constraint: objective must equal optimal value
    # -w^T f = opt_obj  =>  -w^T f = opt_obj
    A_eq[n_conservation, :] = -edge_weights.astype(np.float64)
    b_eq[n_conservation] = opt_obj

    bounds = [(0.0, max(float(edge_weights[e]), 0.0)) for e in range(n_edges)]

    for e in range(n_edges):
        if flows[e] < 1e-12:
            # Zero-flow edges: check if they could carry flow in some
            # optimal decomposition.  If the edge capacity is positive,
            # try to maximize flow on it.
            if edge_weights[e] < 1e-12:
                safe[e] = True  # zero-capacity edge is trivially safe
                continue

            # Try to maximize f_e
            c_max = np.zeros(n_edges, dtype=np.float64)
            c_max[e] = -1.0  # minimize -f_e = maximize f_e
            result_max: OptimizeResult = linprog(
                c_max,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options={"presolve": True, "disp": False},
            )
            if result_max.success and result_max.x[e] < 1e-9:
                safe[e] = True  # flow is always 0 in optimal solutions
            continue

        # Positive-flow edge: check if flow is uniquely determined
        # Try to minimize f_e
        c_min = np.zeros(n_edges, dtype=np.float64)
        c_min[e] = 1.0  # minimize f_e
        result_min: OptimizeResult = linprog(
            c_min,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "disp": False},
        )
        if not result_min.success:
            continue

        min_val = result_min.x[e]
        if abs(min_val - flows[e]) > 1e-6:
            continue  # flow can be decreased, not safe

        # Try to maximize f_e
        c_max = np.zeros(n_edges, dtype=np.float64)
        c_max[e] = -1.0
        result_max = linprog(
            c_max,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={"presolve": True, "disp": False},
        )
        if not result_max.success:
            continue

        max_val = result_max.x[e]
        if abs(max_val - flows[e]) > 1e-6:
            continue  # flow can be increased, not safe

        safe[e] = True

    return safe


# ---------------------------------------------------------------------------
# Chain safe edges into maximal subpaths
# ---------------------------------------------------------------------------


def _chain_safe_edges(
    safe_mask: np.ndarray,
    row_offsets: np.ndarray,
    col_indices: np.ndarray,
    n_nodes: int,
) -> list[list[int]]:
    """Chain consecutive safe edges into maximal safe subpaths.

    Starting from each safe edge that either begins at source or whose
    source node has no safe incoming edge, we greedily extend forward
    through consecutive safe edges to build maximal chains.

    Parameters:
        safe_mask: Boolean array marking safe edges (length n_edges).
        row_offsets: CSR row offsets.
        col_indices: CSR column indices.
        n_nodes: Number of nodes.

    Returns:
        List of safe subpaths, each a list of node IDs.
    """
    # Build reverse adjacency for safe edges: for each node, which safe
    # edges enter it
    safe_in_count = np.zeros(n_nodes, dtype=np.int32)
    safe_in_source = np.full(n_nodes, -1, dtype=np.int32)  # source of the unique safe in-edge

    for u in range(n_nodes):
        start = int(row_offsets[u])
        end = int(row_offsets[u + 1])
        for idx in range(start, end):
            if safe_mask[idx]:
                v = int(col_indices[idx])
                safe_in_count[v] += 1
                safe_in_source[v] = u

    # Build forward safe adjacency: for each node, count of safe out-edges
    safe_out_count = np.zeros(n_nodes, dtype=np.int32)
    safe_out_target = np.full(n_nodes, -1, dtype=np.int32)

    for u in range(n_nodes):
        start = int(row_offsets[u])
        end = int(row_offsets[u + 1])
        for idx in range(start, end):
            if safe_mask[idx]:
                v = int(col_indices[idx])
                safe_out_count[u] += 1
                safe_out_target[u] = v

    # A node is a chain start if it has a safe outgoing edge AND either:
    #   - has no safe incoming edge, or
    #   - has more than one safe incoming edge (branching point), or
    #   - the predecessor has more than one safe outgoing edge
    # We need maximal chains of nodes u0 -> u1 -> ... -> uk where each
    # consecutive pair (ui, u_{i+1}) is connected by a safe edge AND ui
    # has exactly one safe out-edge AND u_{i+1} has exactly one safe in-edge
    # (to keep the chain unambiguous).

    used_edges = np.zeros(len(col_indices), dtype=np.bool_)
    paths: list[list[int]] = []

    for start_node in range(n_nodes):
        # Check if start_node can begin a new chain
        if safe_out_count[start_node] == 0:
            continue

        # Check if this node should start a chain (no unique safe in-edge,
        # or the safe in-edge's source has multiple safe out-edges)
        is_chain_start = False
        if safe_in_count[start_node] == 0:
            is_chain_start = True
        elif safe_in_count[start_node] > 1:
            is_chain_start = True
        else:
            # Exactly one safe in-edge: check if predecessor has multiple safe outs
            pred = int(safe_in_source[start_node])
            if safe_out_count[pred] > 1:
                is_chain_start = True

        if not is_chain_start:
            continue

        # Extend forward through safe edges, handling potential multi-out
        # We start a separate chain for each safe out-edge from start_node
        start_edges: list[int] = []
        s = int(row_offsets[start_node])
        e = int(row_offsets[start_node + 1])
        for idx in range(s, e):
            if safe_mask[idx] and not used_edges[idx]:
                start_edges.append(idx)

        for edge_idx in start_edges:
            if used_edges[edge_idx]:
                continue
            chain = [start_node]
            current_edge = edge_idx
            while True:
                used_edges[current_edge] = True
                v = int(col_indices[current_edge])
                chain.append(v)

                # Try to continue: v must have exactly one safe out-edge
                if safe_out_count[v] != 1:
                    break
                # And the target of that edge must have exactly one safe in-edge
                nxt = int(safe_out_target[v])
                if safe_in_count[nxt] != 1:
                    break

                # Find the safe out-edge from v
                found = False
                for idx2 in range(int(row_offsets[v]), int(row_offsets[v + 1])):
                    if safe_mask[idx2] and not used_edges[idx2]:
                        current_edge = idx2
                        found = True
                        break
                if not found:
                    break

            if len(chain) >= 2:
                paths.append(chain)

    return paths


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_safe_paths(
    graph: CSRGraph,
    max_paths: int = 1000,
) -> SafePathResult:
    """Compute safe subpaths guaranteed to appear in every optimal flow decomposition.

    The algorithm:
    1. Solve an LP to find an optimal flow on the graph.
    2. Identify edges whose flow value is uniquely determined across all
       optimal solutions (safe edges).
    3. Chain consecutive safe edges into maximal safe subpaths.

    Parameters:
        graph: The CSR splice graph to decompose.
        max_paths: Maximum number of safe paths to return.

    Returns:
        A :class:`SafePathResult` containing the safe subpaths, their
        flow weights, and the fraction of total flow they explain.
    """
    n_nodes = graph.n_nodes
    n_edges = graph.n_edges

    if n_nodes == 0 or n_edges == 0:
        return SafePathResult(paths=[], weights=[], coverage_fraction=0.0)

    # Step 1: compute optimal flow via LP
    flows = _solve_flow_lp(
        graph.row_offsets, graph.col_indices, graph.edge_weights, n_nodes
    )

    total_flow = float(np.sum(flows[int(graph.row_offsets[0]) : int(graph.row_offsets[1])]))
    if total_flow < 1e-12:
        return SafePathResult(paths=[], weights=[], coverage_fraction=0.0)

    # Step 2: identify safe edges
    safe_mask = _identify_safe_edges(
        flows, graph.row_offsets, graph.col_indices, graph.edge_weights, n_nodes
    )

    # Step 3: chain safe edges into subpaths
    chains = _chain_safe_edges(safe_mask, graph.row_offsets, graph.col_indices, n_nodes)

    # Compute the flow weight for each safe path: the minimum flow on any
    # edge in the path (which should be the same for all edges in a properly
    # safe chain, but we take the minimum for robustness).
    paths: list[list[int]] = []
    weights: list[float] = []

    for chain in chains:
        if len(chain) < 2:
            continue
        min_flow = np.float64(1e18)
        for i in range(len(chain) - 1):
            u = chain[i]
            v = chain[i + 1]
            for idx in range(int(graph.row_offsets[u]), int(graph.row_offsets[u + 1])):
                if int(graph.col_indices[idx]) == v:
                    if flows[idx] < min_flow:
                        min_flow = flows[idx]
                    break
        if min_flow > 1e-12:
            paths.append(chain)
            weights.append(float(min_flow))
        if len(paths) >= max_paths:
            break

    safe_flow = sum(weights)
    coverage_fraction = safe_flow / total_flow if total_flow > 1e-12 else 0.0

    return SafePathResult(
        paths=paths,
        weights=weights,
        coverage_fraction=coverage_fraction,
    )
