"""Graph simplification for de novo assembly.

Implements standard de Bruijn graph cleaning operations for RNA-seq
assembly:

- **Tip removal**: Remove short dead-end paths likely caused by
  sequencing errors.
- **Bubble collapsing**: Merge alternative paths between the same
  start/end nodes when one path has much lower coverage (likely an
  error variant).
- **Low-coverage edge removal**: Remove edges below a minimum coverage
  threshold.

These operations reduce graph complexity while preserving true
biological variation (splice isoforms, allelic variants).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from braid.denovo.graph import DeBruijnGraph

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SimplifyConfig:
    """Configuration for graph simplification.

    Attributes:
        min_tip_length: Maximum unitig length (in bases) for a dead-end
            to be considered a tip and removed.  Typically 2*k.
        min_coverage: Minimum edge coverage to retain.  Edges below this
            are removed during low-coverage filtering.
        bubble_coverage_ratio: Maximum coverage ratio between alternative
            bubble paths for the weaker one to be collapsed.  E.g., 0.1
            means the weaker path must have < 10% of the stronger path's
            coverage.
        max_bubble_length: Maximum path length (in nodes) to consider
            for bubble detection.
        max_iterations: Maximum rounds of iterative simplification.
    """

    min_tip_length: int = 50
    min_coverage: float = 2.0
    bubble_coverage_ratio: float = 0.1
    max_bubble_length: int = 5
    max_iterations: int = 3


@dataclass(slots=True)
class SimplifyStats:
    """Statistics from graph simplification.

    Attributes:
        tips_removed: Number of tip nodes removed.
        bubbles_collapsed: Number of bubbles collapsed.
        low_cov_edges_removed: Number of low-coverage edges removed.
        isolated_nodes_removed: Number of isolated nodes removed.
        iterations: Number of simplification rounds performed.
    """

    tips_removed: int = 0
    bubbles_collapsed: int = 0
    low_cov_edges_removed: int = 0
    isolated_nodes_removed: int = 0
    iterations: int = 0


def simplify_graph(
    graph: DeBruijnGraph,
    cfg: SimplifyConfig | None = None,
) -> SimplifyStats:
    """Apply iterative graph simplification in place.

    Repeatedly applies tip removal, bubble collapsing, and low-coverage
    filtering until the graph stabilizes or max iterations is reached.

    Args:
        graph: De Bruijn graph to simplify (modified in place).
        cfg: Simplification parameters.

    Returns:
        Statistics about what was removed.
    """
    if cfg is None:
        cfg = SimplifyConfig()

    stats = SimplifyStats()

    for iteration in range(cfg.max_iterations):
        changed = False

        # 1. Remove tips (short dead-ends)
        n_tips = _remove_tips(graph, cfg.min_tip_length)
        stats.tips_removed += n_tips
        if n_tips > 0:
            changed = True

        # 2. Remove low-coverage edges
        n_low_cov = _remove_low_coverage_edges(graph, cfg.min_coverage)
        stats.low_cov_edges_removed += n_low_cov
        if n_low_cov > 0:
            changed = True

        # 3. Collapse bubbles
        n_bubbles = _collapse_bubbles(
            graph, cfg.bubble_coverage_ratio, cfg.max_bubble_length,
        )
        stats.bubbles_collapsed += n_bubbles
        if n_bubbles > 0:
            changed = True

        # 4. Remove isolated nodes
        n_isolated = _remove_isolated_nodes(graph)
        stats.isolated_nodes_removed += n_isolated
        if n_isolated > 0:
            changed = True

        stats.iterations = iteration + 1

        logger.debug(
            "Simplify iteration %d: %d tips, %d low-cov edges, "
            "%d bubbles, %d isolated",
            iteration + 1, n_tips, n_low_cov, n_bubbles, n_isolated,
        )

        if not changed:
            break

    logger.info(
        "Graph simplified in %d iterations: %d tips, %d low-cov edges, "
        "%d bubbles, %d isolated nodes removed. Final: %d nodes, %d edges",
        stats.iterations, stats.tips_removed, stats.low_cov_edges_removed,
        stats.bubbles_collapsed, stats.isolated_nodes_removed,
        graph.n_nodes, graph.n_edges,
    )
    return stats


def _remove_tips(graph: DeBruijnGraph, min_tip_length: int) -> int:
    """Remove short dead-end paths (tips) from the graph.

    A tip is a node that has either no incoming or no outgoing edges,
    whose total unitig length is below the threshold, AND whose
    coverage is significantly lower than its neighbor.  This prevents
    removing legitimate transcript endpoints.

    Tips typically arise from sequencing errors at read ends.

    Args:
        graph: De Bruijn graph (modified in place).
        min_tip_length: Maximum unitig length for tip removal.

    Returns:
        Number of tip nodes removed.
    """
    tips_to_remove: list[int] = []

    for nid, node in graph.nodes.items():
        if node.unitig_length >= min_tip_length:
            continue

        # A tip is a dead-end with short sequence
        is_source_tip = node.in_degree == 0 and node.out_degree == 1
        is_sink_tip = node.out_degree == 0 and node.in_degree == 1

        if not (is_source_tip or is_sink_tip):
            continue

        # Only remove if coverage is much lower than the connected node
        # (error-derived tips have low coverage relative to the main path)
        neighbor_ids = node.out_edges | node.in_edges
        if not neighbor_ids:
            continue

        max_neighbor_cov = max(
            graph.nodes[nn].coverage
            for nn in neighbor_ids
            if nn in graph.nodes
        )
        if max_neighbor_cov > 0 and node.coverage < max_neighbor_cov * 0.15:
            tips_to_remove.append(nid)

    for nid in tips_to_remove:
        _remove_node(graph, nid)

    return len(tips_to_remove)


def _remove_low_coverage_edges(
    graph: DeBruijnGraph,
    min_coverage: float,
) -> int:
    """Remove edges with coverage below the threshold.

    Args:
        graph: De Bruijn graph (modified in place).
        min_coverage: Minimum edge coverage to retain.

    Returns:
        Number of edges removed.
    """
    edges_to_keep: list = []
    removed = 0

    for edge in graph.edges:
        if edge.coverage < min_coverage:
            # Remove edge references from nodes
            src = graph.nodes.get(edge.source)
            tgt = graph.nodes.get(edge.target)
            if src is not None:
                src.out_edges.discard(edge.target)
            if tgt is not None:
                tgt.in_edges.discard(edge.source)
            removed += 1
        else:
            edges_to_keep.append(edge)

    graph.edges = edges_to_keep
    return removed


def _collapse_bubbles(
    graph: DeBruijnGraph,
    coverage_ratio: float,
    max_length: int,
) -> int:
    """Collapse simple bubbles in the graph.

    A bubble is two alternative paths between the same pair of start/end
    nodes.  When one path has much lower coverage, it is likely a
    sequencing error and is removed.

    This implementation detects simple bubbles where a node has exactly
    two outgoing edges that converge to the same node within
    max_length steps.

    Args:
        graph: De Bruijn graph (modified in place).
        coverage_ratio: Maximum ratio of weak/strong path coverage for
            collapsing.
        max_length: Maximum number of nodes in a bubble path.

    Returns:
        Number of bubbles collapsed.
    """
    collapsed = 0

    # Find nodes with exactly 2 outgoing edges (potential bubble starts)
    bubble_starts = [
        nid for nid, node in graph.nodes.items()
        if node.out_degree == 2
    ]

    for start_nid in bubble_starts:
        start_node = graph.nodes.get(start_nid)
        if start_node is None or start_node.out_degree != 2:
            continue

        out_list = list(start_node.out_edges)
        path_a = _trace_linear_path(graph, out_list[0], max_length)
        path_b = _trace_linear_path(graph, out_list[1], max_length)

        if not path_a or not path_b:
            continue

        # Check if both paths converge to the same end node
        end_a = path_a[-1]
        end_b = path_b[-1]

        # Get successors of each path end
        end_a_node = graph.nodes.get(end_a)
        end_b_node = graph.nodes.get(end_b)
        if end_a_node is None or end_b_node is None:
            continue

        # Find common successor
        common = end_a_node.out_edges & end_b_node.out_edges
        if not common:
            continue

        # Compute path coverages
        cov_a = _path_coverage(graph, path_a)
        cov_b = _path_coverage(graph, path_b)

        if cov_a == 0 and cov_b == 0:
            continue

        # Determine weaker path
        if cov_a <= cov_b:
            weak_path, strong_cov = path_a, cov_b
            weak_cov = cov_a
        else:
            weak_path, strong_cov = path_b, cov_a
            weak_cov = cov_b

        if strong_cov > 0 and weak_cov / strong_cov < coverage_ratio:
            # Remove weaker path
            for nid in weak_path:
                _remove_node(graph, nid)
            collapsed += 1

    return collapsed


def _trace_linear_path(
    graph: DeBruijnGraph,
    start: int,
    max_length: int,
) -> list[int]:
    """Trace a linear path from a starting node.

    Follows the chain of nodes with in_degree=1, out_degree=1 until
    reaching a branching point or dead-end.

    Args:
        graph: De Bruijn graph.
        start: Starting node ID.
        max_length: Maximum path length in nodes.

    Returns:
        List of node IDs in the path, or empty if start doesn't exist.
    """
    path = []
    current = start

    for _ in range(max_length):
        node = graph.nodes.get(current)
        if node is None:
            break
        path.append(current)
        if node.out_degree != 1:
            break
        current = next(iter(node.out_edges))
        if graph.nodes.get(current) is None:
            break
        if graph.nodes[current].in_degree != 1:
            break

    return path


def _path_coverage(graph: DeBruijnGraph, path: list[int]) -> float:
    """Compute average coverage along a path.

    Args:
        graph: De Bruijn graph.
        path: List of node IDs.

    Returns:
        Average node coverage along the path.
    """
    if not path:
        return 0.0
    total = sum(
        graph.nodes[nid].coverage
        for nid in path
        if nid in graph.nodes
    )
    return total / len(path)


def _remove_node(graph: DeBruijnGraph, nid: int) -> None:
    """Remove a node and its connected edges from the graph.

    Args:
        graph: De Bruijn graph (modified in place).
        nid: Node ID to remove.
    """
    node = graph.nodes.get(nid)
    if node is None:
        return

    # Remove from neighbors' edge sets
    for in_nid in list(node.in_edges):
        neighbor = graph.nodes.get(in_nid)
        if neighbor is not None:
            neighbor.out_edges.discard(nid)

    for out_nid in list(node.out_edges):
        neighbor = graph.nodes.get(out_nid)
        if neighbor is not None:
            neighbor.in_edges.discard(nid)

    # Remove edges involving this node
    graph.edges = [
        e for e in graph.edges
        if e.source != nid and e.target != nid
    ]

    # Remove node
    del graph.nodes[nid]


def _remove_isolated_nodes(graph: DeBruijnGraph) -> int:
    """Remove nodes with no edges (isolated after other operations).

    Args:
        graph: De Bruijn graph (modified in place).

    Returns:
        Number of isolated nodes removed.
    """
    isolated = [
        nid for nid, node in graph.nodes.items()
        if node.in_degree == 0 and node.out_degree == 0
    ]
    for nid in isolated:
        del graph.nodes[nid]
    return len(isolated)
