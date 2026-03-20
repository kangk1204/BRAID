"""High-level flow decomposition engine for transcript assembly.

This module ties together safe-path computation, max-flow, and min-cost flow
into a complete transcript assembly pipeline.  Given a splice graph for a
single gene locus (in both mutable :class:`SpliceGraph` and immutable
:class:`CSRGraph` forms), it produces a ranked list of :class:`Transcript`
objects with genomic coordinates and abundance estimates.

Algorithm overview:

1. **Safe paths** (optional): Compute subpaths that are guaranteed to appear
   in every optimal flow decomposition.  These provide high-confidence
   transcript skeletons.
2. **Safe-path extension** (optional): Extend each safe subpath to a full
   source-to-sink path by greedily following the highest-weight edges in
   both directions.
3. **Residual decomposition**: Subtract the flow explained by safe/extended
   paths, then run min-cost flow decomposition on the residual graph to
   recover additional transcripts.
4. **Merging and filtering**: Merge compatible transcripts, discard
   low-abundance transcripts, and sort by weight.

The batched entry point :func:`decompose_batched` processes multiple loci
sequentially on the CPU; GPU-level batching is handled at the kernel layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from braid.flow.min_cost_flow import flow_to_weighted_paths, min_cost_flow
from braid.flow.safe_paths import compute_safe_paths
from braid.graph.splice_graph import (
    BatchedCSRGraphs,
    CSRGraph,
    NodeType,
    SpliceGraph,
)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Transcript:
    """A single assembled transcript.

    Attributes:
        node_ids: Sequence of node IDs in the splice graph traversed by
            this transcript (including source/sink if present in the path).
        exon_coords: Genomic coordinates ``(start, end)`` for each exon
            in the transcript.  Source and sink virtual nodes are excluded.
        weight: Estimated abundance of this transcript (flow value).
        is_safe: Whether this transcript is based entirely on safe subpaths
            (i.e., guaranteed to appear in every optimal flow decomposition).
    """

    node_ids: list[int] = field(default_factory=list)
    exon_coords: list[tuple[int, int]] = field(default_factory=list)
    weight: float = 0.0
    is_safe: bool = False


@dataclass(slots=True)
class DecomposeConfig:
    """Configuration for flow decomposition.

    Attributes:
        min_transcript_coverage: Minimum flow weight for a transcript to be
            retained.  Transcripts with weight below this threshold are
            discarded.
        min_relative_abundance: Minimum abundance as a fraction of the
            highest-weight transcript at the same locus.  Transcripts with
            relative abundance below this threshold are removed to suppress
            chimeric path artifacts.
        min_relative_isoform_weight: Minimum weight as a fraction of the
            dominant isoform's weight at the same locus.  Paths whose NNLS
            weight is at least ``max_weight * min_relative_isoform_weight``
            are retained even if they fall below ``min_transcript_coverage``.
            This preserves low-expression isoforms at multi-isoform loci.
        max_transcripts_per_locus: Maximum number of transcripts to report
            per gene locus.
        use_safe_paths: Whether to compute safe paths as a first pass before
            general flow decomposition.
        safe_path_extension: Whether to extend safe subpaths to full
            source-to-sink transcripts using greedy edge following.
    """

    min_transcript_coverage: float = 1.0
    min_relative_abundance: float = 0.02
    min_relative_isoform_weight: float = 0.01
    max_transcripts_per_locus: int = 50
    max_paths: int = 2000
    use_safe_paths: bool = True
    safe_path_extension: bool = True
    junction_weight: float = 2.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _merge_adjacent_exons(
    exon_coords: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge adjacent exon segments that share a boundary.

    The segment graph splits exons at alternative splice sites, producing
    sub-exon nodes like ``(1000, 1050)`` and ``(1050, 1500)`` for a single
    biological exon ``(1000, 1500)``.  After path enumeration, these
    abutting segments must be merged back into contiguous exon intervals.

    Args:
        exon_coords: Raw exon coordinates from path nodes, in genomic order.

    Returns:
        Merged exon coordinates where abutting segments are combined.
    """
    if len(exon_coords) <= 1:
        return exon_coords

    merged: list[tuple[int, int]] = [exon_coords[0]]
    for start, end in exon_coords[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            # Abutting or overlapping -- merge.
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _extend_safe_path(
    safe_path: list[int],
    graph_csr: CSRGraph,
    edge_flows: np.ndarray,
) -> list[int]:
    """Extend a safe subpath to a full source-to-sink path.

    Uses the highest-weight edges (from the optimal flow) to extend the
    safe subpath both backward toward the source and forward toward the
    sink.

    Parameters:
        safe_path: Node ID sequence of the safe subpath to extend.
        graph_csr: The CSR graph.
        edge_flows: Optimal edge flow values.

    Returns:
        A complete path from source (node 0) to sink (last node) that
        contains the safe subpath as a contiguous subsequence.
    """
    n_nodes = graph_csr.n_nodes
    source = 0
    sink = n_nodes - 1

    # Build reverse adjacency: for each node, list of (predecessor, edge_idx)
    in_edges: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for u in range(n_nodes):
        start = int(graph_csr.row_offsets[u])
        end = int(graph_csr.row_offsets[u + 1])
        for idx in range(start, end):
            v = int(graph_csr.col_indices[idx])
            in_edges[v].append((u, idx))

    # Extend backward from safe_path[0] to source
    backward: list[int] = []
    current = safe_path[0]
    visited_back: set[int] = set(safe_path)
    while current != source:
        best_pred = -1
        best_flow = -1.0
        for pred, idx in in_edges[current]:
            if pred not in visited_back and edge_flows[idx] > best_flow:
                best_flow = edge_flows[idx]
                best_pred = pred
        if best_pred == -1:
            # Fallback: pick any predecessor with positive capacity
            for pred, idx in in_edges[current]:
                if pred not in visited_back and graph_csr.edge_weights[idx] > 0:
                    best_pred = pred
                    break
        if best_pred == -1:
            break
        backward.append(best_pred)
        visited_back.add(best_pred)
        current = best_pred
    backward.reverse()

    # Extend forward from safe_path[-1] to sink
    forward: list[int] = []
    current = safe_path[-1]
    visited_fwd: set[int] = set(safe_path) | set(backward)
    while current != sink:
        best_succ = -1
        best_flow = -1.0
        start = int(graph_csr.row_offsets[current])
        end = int(graph_csr.row_offsets[current + 1])
        for idx in range(start, end):
            v = int(graph_csr.col_indices[idx])
            if v not in visited_fwd and edge_flows[idx] > best_flow:
                best_flow = edge_flows[idx]
                best_succ = v
        if best_succ == -1:
            for idx in range(start, end):
                v = int(graph_csr.col_indices[idx])
                if v not in visited_fwd and graph_csr.edge_weights[idx] > 0:
                    best_succ = v
                    break
        if best_succ == -1:
            break
        forward.append(best_succ)
        visited_fwd.add(best_succ)
        current = best_succ

    return backward + safe_path + forward


def _paths_to_transcripts(
    paths: list[tuple[list[int], float]],
    node_starts: np.ndarray,
    node_ends: np.ndarray,
    node_types: np.ndarray,
    is_safe: bool = False,
) -> list[Transcript]:
    """Convert node-ID paths to Transcript objects with genomic coordinates.

    Source and sink virtual nodes are excluded from the exon coordinate
    list (they have no genomic extent).

    Parameters:
        paths: List of ``(node_id_sequence, weight)`` tuples.
        node_starts: Genomic start per node (int64).
        node_ends: Genomic end per node (int64).
        node_types: Node type ordinals (int8).
        is_safe: Whether these transcripts are derived from safe paths.

    Returns:
        List of :class:`Transcript` objects.
    """
    transcripts: list[Transcript] = []

    for path, weight in paths:
        exon_coords: list[tuple[int, int]] = []
        for nid in path:
            ntype = int(node_types[nid])
            if ntype == int(NodeType.EXON):
                exon_coords.append((int(node_starts[nid]), int(node_ends[nid])))

        # Merge adjacent sub-exon segments from the segment graph.
        exon_coords = _merge_adjacent_exons(exon_coords)

        if len(exon_coords) == 0:
            continue

        transcripts.append(
            Transcript(
                node_ids=list(path),
                exon_coords=exon_coords,
                weight=weight,
                is_safe=is_safe,
            )
        )

    return transcripts


def _subtract_path_flows(
    edge_flows: np.ndarray,
    paths: list[tuple[list[int], float]],
    graph_csr: CSRGraph,
) -> np.ndarray:
    """Subtract flow consumed by a set of paths from the edge flow vector.

    Parameters:
        edge_flows: Current edge flow values (float64, length n_edges).
        paths: List of ``(node_id_sequence, weight)`` tuples to subtract.
        graph_csr: The CSR graph.

    Returns:
        Updated edge flow vector with subtracted flows (clipped at 0).
    """
    residual = edge_flows.copy()

    for path, weight in paths:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            for idx in range(int(graph_csr.row_offsets[u]), int(graph_csr.row_offsets[u + 1])):
                if int(graph_csr.col_indices[idx]) == v:
                    residual[idx] = max(0.0, residual[idx] - weight)
                    break

    return residual


def _merge_compatible_transcripts(
    transcripts: list[Transcript],
) -> list[Transcript]:
    """Merge transcripts that traverse exactly the same set of exons.

    Two transcripts are compatible if they have identical exon coordinate
    lists.  Merging sums their weights and preserves the safe flag only if
    both are safe.

    Parameters:
        transcripts: Input transcript list.

    Returns:
        Deduplicated transcript list with merged weights.
    """
    groups: dict[tuple[tuple[int, int], ...], Transcript] = {}

    for tx in transcripts:
        key = tuple(tx.exon_coords)
        if key in groups:
            existing = groups[key]
            groups[key] = Transcript(
                node_ids=existing.node_ids,
                exon_coords=existing.exon_coords,
                weight=existing.weight + tx.weight,
                is_safe=existing.is_safe and tx.is_safe,
            )
        else:
            groups[key] = Transcript(
                node_ids=list(tx.node_ids),
                exon_coords=list(tx.exon_coords),
                weight=tx.weight,
                is_safe=tx.is_safe,
            )

    return list(groups.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _enumerate_all_paths(
    graph_csr: CSRGraph,
    max_paths: int = 2000,
) -> list[list[int]]:
    """Enumerate source-to-sink paths in a DAG."""
    paths, _ = _enumerate_all_paths_with_metrics(graph_csr, max_paths=max_paths)
    return paths


def _enumerate_all_paths_with_metrics(
    graph_csr: CSRGraph,
    max_paths: int = 2000,
) -> tuple[list[list[int]], dict[str, float | int]]:
    """Enumerate source-to-sink paths in a DAG, prioritized by edge coverage.

    Uses a priority queue (max-heap by minimum edge weight along the path)
    so that the highest-support paths are enumerated first when the path
    limit is reached before exhaustive enumeration completes.

    Parameters:
        graph_csr: CSR graph (must be a DAG with source=0, sink=n-1).
        max_paths: Maximum number of paths to enumerate.

    Returns:
        List of node-ID paths from source to sink.
    """
    import heapq

    n_nodes = graph_csr.n_nodes
    source = 0
    sink = n_nodes - 1
    paths: list[list[int]] = []

    # Safety limits to prevent unbounded resource usage.
    max_heap_entries = 50000
    max_iterations = 500000

    # Max-heap: negate min_weight for heapq (min-heap).
    # Each entry: (-min_edge_weight, path)
    # Start with +inf bottleneck so the first traversed edge defines path score.
    initial: tuple[float, list[int]] = (float("-inf"), [source])
    heap: list[tuple[float, list[int]]] = [initial]

    max_heap_size = len(heap)
    iterations = 0
    while heap and len(paths) < max_paths:
        iterations += 1
        if iterations > max_iterations:
            break
        neg_min_w, path = heapq.heappop(heap)
        node = path[-1]
        if node == sink:
            paths.append(path)
            continue
        # Cycle protection: build visited set for the current path so we
        # never revisit a node (the graph should be a DAG, but a cycle
        # caused by a graph-construction bug would otherwise loop forever).
        visited = set(path)
        start = int(graph_csr.row_offsets[node])
        end = int(graph_csr.row_offsets[node + 1])
        for idx in range(start, end):
            v = int(graph_csr.col_indices[idx])
            if v in visited:
                continue
            w = float(graph_csr.edge_weights[idx])
            if w > 0:
                # Enforce heap size limit to prevent memory explosion.
                if len(heap) >= max_heap_entries:
                    continue
                new_min = min(-neg_min_w, w)
                heapq.heappush(heap, (-new_min, path + [v]))
                max_heap_size = max(max_heap_size, len(heap))

    hit_max_paths = len(paths) >= max_paths and len(heap) > 0
    return paths, {
        "all_paths_total": len(paths),
        "max_paths_hit": int(hit_max_paths),
        "max_heap_size": max_heap_size,
    }


def _fit_path_weights_lp(
    graph_csr: CSRGraph,
    paths: list[list[int]],
    phasing_paths: list[tuple[list[int], float]] | None = None,
    phasing_weight: float = 0.5,
    junction_weight: float = 2.0,
) -> np.ndarray:
    """Assign weights to enumerated paths by fitting to edge coverage."""
    weights, _ = _fit_path_weights_lp_with_metrics(
        graph_csr,
        paths,
        phasing_paths=phasing_paths,
        phasing_weight=phasing_weight,
        junction_weight=junction_weight,
    )
    return weights


def _fit_path_weights_lp_with_metrics(
    graph_csr: CSRGraph,
    paths: list[list[int]],
    phasing_paths: list[tuple[list[int], float]] | None = None,
    phasing_weight: float = 0.5,
    junction_weight: float = 2.0,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Assign weights to enumerated paths by fitting to edge coverage.

    Solves a weighted non-negative least squares (NNLS) problem:
    minimize ``||W * (A @ w - b)||^2`` subject to ``w >= 0``, where W is
    a diagonal weight matrix that gives intron (junction) edges higher
    importance than continuation edges.  Junction counts are direct
    isoform evidence, while exon coverage is shared across isoforms.

    When phasing evidence is provided, additional rows are added to the NNLS
    matrix to encourage paths consistent with paired-end read evidence.

    Parameters:
        graph_csr: CSR graph.
        paths: List of node-ID paths.
        phasing_paths: Optional list of (node_id_sequence, read_count)
            phasing constraints from paired-end reads.
        phasing_weight: Weight multiplier for phasing rows (0-1).  Higher
            values give phasing evidence more influence.
        junction_weight: Weight multiplier for intron (junction) edges
            relative to continuation edges (default 2.0).

    Returns:
        Array of path weights (non-negative).
    """
    from scipy.optimize import nnls

    n_edges = graph_csr.n_edges
    n_paths = len(paths)

    if n_paths == 0 or n_edges == 0:
        return np.zeros(n_paths, dtype=np.float64), {
            "path_basis_size": n_paths,
            "nnls_residual_total": 0.0,
            "nnls_residual_edge": 0.0,
            "nnls_residual_phasing": 0.0,
            "phasing_constraints": len(phasing_paths or []),
            "phasing_matched": 0,
            "phasing_match_rate": 0.0,
            "a_condition_number": 0.0,
        }

    # Build edge-to-index mapping
    edge_map: dict[tuple[int, int], int] = {}
    for u in range(graph_csr.n_nodes):
        start = int(graph_csr.row_offsets[u])
        end = int(graph_csr.row_offsets[u + 1])
        for idx in range(start, end):
            v = int(graph_csr.col_indices[idx])
            edge_map[(u, v)] = idx

    # Build per-edge weight vector W for junction-weighted NNLS.
    # Intron edges (typically have edge_type info via edge_coverages matching
    # edge_weights for junction edges) get higher weight.
    edge_type_weights = np.ones(n_edges, dtype=np.float64)
    if hasattr(graph_csr, 'edge_coverages') and graph_csr.edge_coverages is not None:
        for eidx in range(n_edges):
            # Intron edges have coverage == weight (set from junction count).
            # Continuation edges have coverage == min(adj exon coverages).
            # Heuristic: if coverage equals weight, it's likely a junction edge.
            ew = float(graph_csr.edge_weights[eidx])
            ec = float(graph_csr.edge_coverages[eidx])
            if ew > 0 and abs(ew - ec) < 0.01:
                edge_type_weights[eidx] = junction_weight

    # Build path-edge incidence matrix A (n_edges x n_paths)
    A = np.zeros((n_edges, n_paths), dtype=np.float64)
    for pi, path in enumerate(paths):
        for i in range(len(path) - 1):
            edge_idx = edge_map.get((path[i], path[i + 1]))
            if edge_idx is not None:
                A[edge_idx, pi] = 1.0

    # Target: edge weights (coverage)
    b = graph_csr.edge_weights.astype(np.float64)

    # Apply edge-type weighting: W * A and W * b
    W = edge_type_weights
    A_weighted = A * W[:, np.newaxis]
    b_weighted = b * W
    A_edge = A_weighted
    b_edge = b_weighted

    # Add phasing constraints as extra rows
    matched_phasing_rows = 0
    A_phasing = np.empty((0, n_paths), dtype=np.float64)
    b_phasing = np.empty(0, dtype=np.float64)
    if phasing_paths and phasing_weight > 0:
        n_phasing = len(phasing_paths)
        A_phasing = np.zeros((n_phasing, n_paths), dtype=np.float64)
        b_phasing = np.zeros(n_phasing, dtype=np.float64)

        for ri, (phase_nodes, phase_count) in enumerate(phasing_paths):
            row_has_match = False
            for pi, path in enumerate(paths):
                # Check if this path contains the phasing subsequence
                if _contains_subsequence(path, phase_nodes):
                    A_phasing[ri, pi] = 1.0
                    row_has_match = True
            if row_has_match:
                matched_phasing_rows += 1
            b_phasing[ri] = phase_count * phasing_weight

        # Stack: [edge constraints; phasing constraints]
        A_weighted = np.vstack([A_weighted, A_phasing])
        b_weighted = np.concatenate([b_weighted, b_phasing])

    # Solve NNLS: minimize ||W * (A @ w - b)||^2 subject to w >= 0
    weights, _residual = nnls(A_weighted, b_weighted)
    if A_weighted.size > 0 and max(A_weighted.shape) <= 512:
        condition_number = float(np.linalg.cond(A_weighted))
    else:
        condition_number = float("nan")

    residual_total = A_weighted @ weights - b_weighted
    residual_edge = A_edge @ weights - b_edge
    residual_phasing = A_phasing @ weights - b_phasing
    metrics: dict[str, float | int] = {
        "path_basis_size": n_paths,
        "nnls_residual_total": float(np.linalg.norm(residual_total)),
        "nnls_residual_edge": float(np.linalg.norm(residual_edge)),
        "nnls_residual_phasing": float(np.linalg.norm(residual_phasing)),
        "phasing_constraints": len(phasing_paths or []),
        "phasing_matched": matched_phasing_rows,
        "phasing_match_rate": (
            float(matched_phasing_rows / len(phasing_paths))
            if phasing_paths else 0.0
        ),
        "a_condition_number": condition_number,
    }
    return weights, metrics


def _contains_subsequence(path: list[int], subseq: list[int]) -> bool:
    """Check if path contains subseq as a (not necessarily contiguous) subsequence.

    Parameters:
        path: The full path (list of node IDs).
        subseq: The subsequence to find.

    Returns:
        True if all elements of subseq appear in path in order.
    """
    if not subseq:
        return True
    si = 0
    for node in path:
        if node == subseq[si]:
            si += 1
            if si == len(subseq):
                return True
    return False


def _inject_phasing_seed_paths(
    graph_csr: CSRGraph,
    existing_paths: list[list[int]],
    phasing_paths: list[tuple[list[int], float]],
) -> list[list[int]]:
    """Add phasing-supported seed paths to the enumerated path list.

    For each phasing constraint (a node subsequence from paired-end reads),
    this function attempts to extend it to a full source-to-sink path by
    greedily following the highest-weight edges.  If the resulting path is
    not already present in ``existing_paths``, it is appended.

    This ensures that read-pair-supported isoforms are always included in
    the NNLS basis, even when the priority-queue path enumeration did not
    discover them naturally.

    Parameters:
        graph_csr: CSR graph (DAG with source=0, sink=n-1).
        existing_paths: Already-enumerated source-to-sink paths.
        phasing_paths: List of ``(node_id_sequence, read_count)`` from
            paired-end phasing evidence.

    Returns:
        Extended path list with any newly discovered seed paths appended.
    """
    n_nodes = graph_csr.n_nodes
    source = 0
    sink = n_nodes - 1

    # Build set of existing path tuples for deduplication.
    existing_set: set[tuple[int, ...]] = {tuple(p) for p in existing_paths}

    # Sort phasing paths by read count (descending) so highest-evidence
    # paths are injected first.
    sorted_phasing = sorted(phasing_paths, key=lambda x: -x[1])

    # Build reverse adjacency for backward extension.
    in_edges: list[list[tuple[int, int]]] = [[] for _ in range(n_nodes)]
    for u in range(n_nodes):
        start = int(graph_csr.row_offsets[u])
        end = int(graph_csr.row_offsets[u + 1])
        for idx in range(start, end):
            v = int(graph_csr.col_indices[idx])
            in_edges[v].append((u, idx))

    # Cap seed injections to avoid blowing up the basis.
    max_seeds = 200
    injected = 0

    for phase_nodes, _count in sorted_phasing:
        if injected >= max_seeds:
            break
        if not phase_nodes:
            continue

        # Validate that all phasing nodes exist in the graph.
        if any(n < 0 or n >= n_nodes for n in phase_nodes):
            continue

        # Extend backward from first phasing node to source.
        backward: list[int] = []
        current = phase_nodes[0]
        visited: set[int] = set(phase_nodes)
        while current != source:
            best_pred = -1
            best_weight = -1.0
            for pred, idx in in_edges[current]:
                if pred not in visited:
                    w = float(graph_csr.edge_weights[idx])
                    if w > best_weight:
                        best_weight = w
                        best_pred = pred
            if best_pred == -1:
                break
            backward.append(best_pred)
            visited.add(best_pred)
            current = best_pred
        backward.reverse()

        # Extend forward from last phasing node to sink.
        forward: list[int] = []
        current = phase_nodes[-1]
        visited_fwd: set[int] = visited | set(backward)
        while current != sink:
            best_succ = -1
            best_weight = -1.0
            start = int(graph_csr.row_offsets[current])
            end = int(graph_csr.row_offsets[current + 1])
            for idx in range(start, end):
                v = int(graph_csr.col_indices[idx])
                if v not in visited_fwd:
                    w = float(graph_csr.edge_weights[idx])
                    if w > best_weight:
                        best_weight = w
                        best_succ = v
            if best_succ == -1:
                break
            forward.append(best_succ)
            visited_fwd.add(best_succ)
            current = best_succ

        full_path = backward + list(phase_nodes) + forward

        # Verify the path is a valid source-to-sink path.
        if not full_path or full_path[0] != source or full_path[-1] != sink:
            continue

        # Verify all consecutive edges exist.
        valid = True
        for i in range(len(full_path) - 1):
            u = full_path[i]
            found = False
            s = int(graph_csr.row_offsets[u])
            e = int(graph_csr.row_offsets[u + 1])
            for idx in range(s, e):
                if int(graph_csr.col_indices[idx]) == full_path[i + 1]:
                    found = True
                    break
            if not found:
                valid = False
                break
        if not valid:
            continue

        path_key = tuple(full_path)
        if path_key not in existing_set:
            existing_paths.append(full_path)
            existing_set.add(path_key)
            injected += 1

    return existing_paths


def decompose_graph(
    graph_csr: CSRGraph,
    graph: SpliceGraph,
    config: DecomposeConfig | None = None,
    phasing_paths: list[tuple[list[int], float]] | None = None,
) -> list[Transcript]:
    """Decompose a splice graph into assembled transcripts."""
    transcripts, _ = decompose_graph_with_metrics(
        graph_csr,
        graph,
        config=config,
        phasing_paths=phasing_paths,
    )
    return transcripts


def decompose_graph_with_metrics(
    graph_csr: CSRGraph,
    graph: SpliceGraph,
    config: DecomposeConfig | None = None,
    phasing_paths: list[tuple[list[int], float]] | None = None,
    guide_paths: list[list[int]] | None = None,
) -> tuple[list[Transcript], dict[str, float | int]]:
    """Decompose a splice graph into assembled transcripts.

    This is the main entry point for transcript assembly on a single locus.
    The algorithm uses a two-phase approach:

    1. **Path enumeration**: Enumerate all distinct source-to-sink paths in
       the splice DAG.
    2. **NNLS weight fitting**: Assign path weights by solving a non-negative
       least squares problem to best explain observed edge coverages.
       When phasing evidence is available, it is incorporated as additional
       constraints favoring paths consistent with paired-end reads.
    3. (Optional) **Safe path boost**: Safe subpaths get priority retention.
    4. **Filter and rank**: Remove low-abundance transcripts and sort.

    This approach avoids the greedy bias of flow decomposition and naturally
    recovers all alternative splicing isoforms present in the data.

    Parameters:
        graph_csr: Immutable CSR representation of the splice graph.
        graph: Mutable :class:`SpliceGraph` (used for metadata access).
        config: Decomposition configuration.  Uses defaults if ``None``.
        phasing_paths: Optional list of ``(node_id_sequence, read_count)``
            from paired-end phasing.  Used to boost paths consistent with
            paired-end evidence.
        guide_paths: Optional list of source-to-sink node paths from
            long-read transcripts.  Injected as seed paths in the NNLS
            basis to bias toward long-read-supported isoforms.

    Returns:
        Sorted list of :class:`Transcript` objects, highest abundance first.
    """
    if config is None:
        config = DecomposeConfig()

    n_nodes = graph_csr.n_nodes
    n_edges = graph_csr.n_edges

    if n_nodes < 2 or n_edges == 0:
        return [], {
            "all_paths_total": 0,
            "max_paths_hit": 0,
            "max_heap_size": 0,
            "path_basis_size": 0,
            "nnls_residual_total": 0.0,
            "nnls_residual_edge": 0.0,
            "nnls_residual_phasing": 0.0,
            "phasing_constraints": len(phasing_paths or []),
            "phasing_matched": 0,
            "phasing_match_rate": 0.0,
            "a_condition_number": 0.0,
            "fallback_min_cost_flow": 0,
            "accepted_paths": 0,
            "merged_transcripts": 0,
        }

    # Phase 1: Enumerate all distinct source-to-sink paths
    all_paths, path_metrics = _enumerate_all_paths_with_metrics(
        graph_csr,
        max_paths=config.max_paths,
    )

    # Phase 1b: Inject phasing-supported seed paths that the priority queue
    # may not have discovered.  This ensures read-pair-supported isoforms are
    # always considered in the NNLS fitting.
    if phasing_paths and all_paths:
        all_paths = _inject_phasing_seed_paths(
            graph_csr, all_paths, phasing_paths,
        )
        path_metrics["phasing_seed_paths_injected"] = (
            len(all_paths) - path_metrics["all_paths_total"]
        )
        path_metrics["all_paths_total"] = len(all_paths)

    # Phase 1c: Inject long-read guide paths as guaranteed seed paths.
    guide_path_indices: set[int] = set()
    if guide_paths and all_paths:
        existing_set = {tuple(p) for p in all_paths}
        for gp in guide_paths:
            gp_key = tuple(gp)
            if gp_key not in existing_set:
                guide_path_indices.add(len(all_paths))
                all_paths.append(gp)
                existing_set.add(gp_key)
            else:
                # Mark existing path as guide-supported
                for idx, ep in enumerate(all_paths):
                    if tuple(ep) == gp_key:
                        guide_path_indices.add(idx)
                        break
        path_metrics["guide_paths_injected"] = len(guide_path_indices)
        path_metrics["all_paths_total"] = len(all_paths)
    elif guide_paths and not all_paths:
        all_paths = list(guide_paths)
        guide_path_indices = set(range(len(all_paths)))
        path_metrics["guide_paths_injected"] = len(guide_path_indices)
        path_metrics["all_paths_total"] = len(all_paths)

    if not all_paths:
        # Fallback to greedy flow decomposition
        mcf_result = min_cost_flow(graph_csr)
        flow_paths = flow_to_weighted_paths(graph_csr, mcf_result.edge_flows)
        return _paths_to_transcripts(
            flow_paths,
            graph_csr.node_starts,
            graph_csr.node_ends,
            graph_csr.node_types,
            is_safe=False,
        ), {
            **path_metrics,
            "path_basis_size": 0,
            "nnls_residual_total": 0.0,
            "nnls_residual_edge": 0.0,
            "nnls_residual_phasing": 0.0,
            "phasing_constraints": len(phasing_paths or []),
            "phasing_matched": 0,
            "phasing_match_rate": 0.0,
            "a_condition_number": 0.0,
            "fallback_min_cost_flow": 1,
            "accepted_paths": len(flow_paths),
            "merged_transcripts": len(flow_paths),
        }

    # Phase 2: Fit path weights using junction-weighted NNLS (with optional phasing)
    weights, fit_metrics = _fit_path_weights_lp_with_metrics(
        graph_csr, all_paths, phasing_paths=phasing_paths,
        junction_weight=config.junction_weight,
    )

    # Phase 3: Identify safe paths for tagging
    safe_node_set: set[int] = set()
    if config.use_safe_paths:
        safe_result = compute_safe_paths(graph_csr)
        for sp in safe_result.paths:
            safe_node_set.update(sp)

    # Phase 4: Build weighted path list
    # Use both absolute coverage threshold and relative isoform weight to
    # preserve low-expression isoforms at multi-isoform loci.  The relative
    # threshold relaxes the absolute minimum only when the dominant isoform
    # itself passes the absolute threshold (so an explicitly high threshold
    # is always respected).
    max_weight = float(np.max(weights)) if len(weights) > 0 else 0.0
    use_relative = max_weight >= config.min_transcript_coverage
    relative_threshold = max(
        max_weight * config.min_relative_isoform_weight,
        0.1,  # absolute floor to avoid retaining noise
    )
    weighted_paths: list[tuple[list[int], float]] = []
    for path, w in zip(all_paths, weights):
        passes_absolute = w >= config.min_transcript_coverage
        passes_relative = use_relative and w >= relative_threshold
        if passes_absolute or passes_relative:
            weighted_paths.append((path, float(w)))

    # Convert to transcripts
    all_transcripts: list[Transcript] = []
    for path, w in weighted_paths:
        exon_coords: list[tuple[int, int]] = []
        is_safe = all(nid in safe_node_set for nid in path)
        for nid in path:
            ntype = int(graph_csr.node_types[nid])
            if ntype == int(NodeType.EXON):
                exon_coords.append(
                    (int(graph_csr.node_starts[nid]), int(graph_csr.node_ends[nid]))
                )
        # Merge adjacent sub-exon segments from the segment graph.
        exon_coords = _merge_adjacent_exons(exon_coords)
        if exon_coords:
            all_transcripts.append(
                Transcript(
                    node_ids=list(path),
                    exon_coords=exon_coords,
                    weight=w,
                    is_safe=is_safe,
                )
            )

    # Merge compatible transcripts (same exon structure)
    all_transcripts = _merge_compatible_transcripts(all_transcripts)

    # Sort by weight descending
    all_transcripts.sort(key=lambda tx: tx.weight, reverse=True)

    # Filter by relative abundance to suppress chimeric artifacts
    if all_transcripts and config.min_relative_abundance > 0:
        max_weight = all_transcripts[0].weight
        threshold = max_weight * config.min_relative_abundance
        all_transcripts = [
            tx for tx in all_transcripts if tx.weight >= threshold
        ]

    # Cap at max transcripts
    if len(all_transcripts) > config.max_transcripts_per_locus:
        all_transcripts = all_transcripts[: config.max_transcripts_per_locus]

    return all_transcripts, {
        **path_metrics,
        **fit_metrics,
        "fallback_min_cost_flow": 0,
        "accepted_paths": len(weighted_paths),
        "merged_transcripts": len(all_transcripts),
    }


def decompose_batched(
    batched: BatchedCSRGraphs,
    graphs: list[SpliceGraph],
    config: DecomposeConfig | None = None,
) -> list[list[Transcript]]:
    """Decompose multiple splice graphs into transcripts.

    Processes each graph sequentially on the CPU.  GPU-level batching is
    handled at the CUDA kernel layer and is transparent to this function.

    Parameters:
        batched: A :class:`BatchedCSRGraphs` containing packed CSR data
            for all loci.
        graphs: List of :class:`SpliceGraph` objects corresponding to
            each graph in the batch (same order as they were added).
        config: Decomposition configuration.  Shared across all graphs.

    Returns:
        A list of transcript lists, one per graph in the batch.
    """
    if config is None:
        config = DecomposeConfig()

    results: list[list[Transcript]] = []

    for graph_idx in range(batched.n_graphs):
        node_start, node_end, edge_start, edge_end = batched.get_graph_range(graph_idx)
        n_nodes_local = node_end - node_start
        n_edges_local = edge_end - edge_start

        if n_nodes_local < 2 or n_edges_local == 0:
            results.append([])
            continue

        # Extract a local CSRGraph from the batched arrays.
        # Row offsets need to be shifted so they index into the local edge
        # arrays starting from 0.
        local_row_offsets = (
            batched.row_offsets[node_start : node_end + 1] - edge_start
        ).astype(np.int32)

        # Column indices need to be shifted so they reference local node IDs
        local_col_indices = (
            batched.col_indices[edge_start:edge_end] - node_start
        ).astype(np.int32)

        local_csr = CSRGraph(
            row_offsets=local_row_offsets,
            col_indices=local_col_indices,
            edge_weights=batched.edge_weights[edge_start:edge_end].copy(),
            edge_coverages=np.zeros(n_edges_local, dtype=np.float32),
            node_coverages=batched.node_coverages[node_start:node_end].copy(),
            node_starts=np.zeros(n_nodes_local, dtype=np.int64),
            node_ends=np.zeros(n_nodes_local, dtype=np.int64),
            node_types=np.zeros(n_nodes_local, dtype=np.int8),
            n_nodes=n_nodes_local,
            n_edges=n_edges_local,
        )

        # Fill in node-level arrays from the batch's internal storage
        if hasattr(batched, '_node_starts') and batched._node_starts is not None:
            local_csr.node_starts = batched._node_starts[node_start:node_end].copy()
        if hasattr(batched, '_node_ends') and batched._node_ends is not None:
            local_csr.node_ends = batched._node_ends[node_start:node_end].copy()
        if hasattr(batched, '_node_types') and batched._node_types is not None:
            local_csr.node_types = batched._node_types[node_start:node_end].copy()
        if hasattr(batched, '_edge_coverages') and batched._edge_coverages is not None:
            local_csr.edge_coverages = batched._edge_coverages[edge_start:edge_end].copy()

        local_graph = graphs[graph_idx]
        transcripts = decompose_graph(local_csr, local_graph, config)
        results.append(transcripts)

    return results
