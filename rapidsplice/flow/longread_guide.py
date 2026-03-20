"""Long-read guided path mapping for splice graph decomposition.

Maps long-read transcript exon structures (from PacBio Iso-Seq or ONT)
to node paths through a CSR splice graph.  Mapped paths are used as
guide (seed) paths in NNLS decomposition, biasing the solver toward
isoforms with independent long-read support.

The mapping handles the segment graph architecture where biological exons
are split at alternative splice sites into multiple graph nodes.
"""

from __future__ import annotations

import bisect
import logging

import numpy as np

from rapidsplice.graph.splice_graph import CSRGraph, NodeType

logger = logging.getLogger(__name__)


def map_longread_to_path(
    lr_exons: list[tuple[int, int]],
    graph_csr: CSRGraph,
    tolerance: int = 5,
) -> list[int] | None:
    """Map a long-read transcript to a node path through the splice graph.

    For each long-read exon, finds all EXON-type graph nodes whose
    coordinate intervals are contained within (or overlap within
    *tolerance* bp of) the long-read exon.  Between consecutive
    long-read exons, verifies that an intron edge exists connecting the
    last node of exon N to the first node of exon N+1.

    Args:
        lr_exons: List of (start, end) exon intervals in 0-based
            half-open coordinates, sorted by start.
        graph_csr: CSR splice graph.
        tolerance: Maximum coordinate mismatch in bp for fuzzy matching
            of exon boundaries.

    Returns:
        A complete source-to-sink node path, or ``None`` if the mapping
        fails (missing junctions, no matching nodes, etc.).
    """
    if not lr_exons or graph_csr.n_nodes < 3:
        return None

    # Build sorted arrays of exon node coordinates for binary search
    node_starts = graph_csr.node_starts
    node_ends = graph_csr.node_ends
    node_types = graph_csr.node_types

    # Identify exon nodes and sort by start coordinate
    exon_nodes: list[int] = []
    for nid in range(graph_csr.n_nodes):
        if node_types[nid] == NodeType.EXON:
            exon_nodes.append(nid)

    if not exon_nodes:
        return None

    exon_nodes.sort(key=lambda n: (int(node_starts[n]), int(node_ends[n])))
    exon_starts = [int(node_starts[n]) for n in exon_nodes]

    # Build adjacency set for edge existence checks
    adj: dict[int, set[int]] = {}
    for u in range(graph_csr.n_nodes):
        start = int(graph_csr.row_offsets[u])
        end = int(graph_csr.row_offsets[u + 1])
        if start < end:
            adj[u] = {int(graph_csr.col_indices[i]) for i in range(start, end)}

    # Find source and sink nodes
    source_node = None
    sink_node = None
    for nid in range(graph_csr.n_nodes):
        if node_types[nid] == NodeType.SOURCE:
            source_node = nid
        elif node_types[nid] == NodeType.SINK:
            sink_node = nid
    if source_node is None or sink_node is None:
        return None

    # Map each long-read exon to graph nodes
    path_segments: list[list[int]] = []

    for lr_start, lr_end in lr_exons:
        # Binary search for candidate nodes
        lo = bisect.bisect_left(exon_starts, lr_start - tolerance)
        hi = bisect.bisect_right(exon_starts, lr_end + tolerance)

        # Find nodes contained within or overlapping the long-read exon
        segment_nodes: list[int] = []
        for idx in range(max(0, lo - 1), min(len(exon_nodes), hi + 1)):
            nid = exon_nodes[idx]
            ns = int(node_starts[nid])
            ne = int(node_ends[nid])

            # Check overlap with tolerance
            if ns >= lr_start - tolerance and ne <= lr_end + tolerance:
                # Node entirely within long-read exon (with tolerance)
                segment_nodes.append(nid)
            elif ns < lr_end + tolerance and ne > lr_start - tolerance:
                # Partial overlap — accept if substantial
                overlap = min(ne, lr_end) - max(ns, lr_start)
                node_len = ne - ns
                if node_len > 0 and overlap / node_len > 0.5:
                    segment_nodes.append(nid)

        if not segment_nodes:
            return None

        # Sort by start coordinate
        segment_nodes.sort(key=lambda n: int(node_starts[n]))

        # If multiple nodes match, find the best connected chain.
        # This handles alternative splice site nodes (same start, different
        # ends) by picking the node whose end best matches the long-read exon.
        if len(segment_nodes) > 1:
            segment_nodes = _find_best_connected_chain(
                segment_nodes, adj, node_starts, node_ends,
                lr_start, lr_end,
            )
            if not segment_nodes:
                return None

        path_segments.append(segment_nodes)

    # Verify intron edges between consecutive exon segments
    for i in range(len(path_segments) - 1):
        last_node = path_segments[i][-1]
        first_node = path_segments[i + 1][0]
        if first_node not in adj.get(last_node, set()):
            return None

    # Assemble full path
    path: list[int] = [source_node]

    # Connect source to first exon node
    first_exon_node = path_segments[0][0]
    if first_exon_node not in adj.get(source_node, set()):
        # Try to find source connection via intermediate node
        return None

    for segment in path_segments:
        path.extend(segment)

    # Connect last exon node to sink
    last_exon_node = path_segments[-1][-1]
    if sink_node not in adj.get(last_exon_node, set()):
        return None

    path.append(sink_node)

    return path


def _find_best_connected_chain(
    candidates: list[int],
    adj: dict[int, set[int]],
    node_starts: np.ndarray,
    node_ends: np.ndarray,
    lr_start: int,
    lr_end: int,
) -> list[int]:
    """Find the best connected chain of nodes matching a long-read exon.

    When multiple graph nodes overlap a single long-read exon (e.g., due to
    alternative splice sites creating multiple nodes at the same position),
    this function selects the chain of connected nodes that best covers the
    long-read exon interval.

    Args:
        candidates: Candidate node IDs sorted by start coordinate.
        adj: Adjacency dict (node -> set of successors).
        node_starts: Node start coordinate array.
        node_ends: Node end coordinate array.
        lr_start: Long-read exon start.
        lr_end: Long-read exon end.

    Returns:
        The best connected chain, or empty list if no valid chain found.
    """
    if len(candidates) <= 1:
        return candidates

    # Group candidates by start position
    groups: dict[int, list[int]] = {}
    for nid in candidates:
        ns = int(node_starts[nid])
        groups.setdefault(ns, []).append(nid)

    sorted_starts = sorted(groups.keys())

    # For each start position, pick the node whose end best matches
    # what's needed for connectivity to the next group
    def _build_chain(pos_idx: int, current_chain: list[int]) -> list[int] | None:
        if pos_idx >= len(sorted_starts):
            return current_chain

        ns = sorted_starts[pos_idx]
        # Sort candidates by boundary match quality (prefer exact match)
        sorted_cands = sorted(
            groups[ns],
            key=lambda n: abs(int(node_ends[n]) - lr_end) + abs(int(node_starts[n]) - lr_start),
        )
        for nid in sorted_cands:
            # Check connectivity from previous node
            if current_chain:
                prev = current_chain[-1]
                if nid not in adj.get(prev, set()):
                    continue

            result = _build_chain(pos_idx + 1, current_chain + [nid])
            if result is not None:
                return result

        # Try skipping this position (node not needed)
        return _build_chain(pos_idx + 1, current_chain)

    result = _build_chain(0, [])
    if result and len(result) > 0:
        return result

    # Fallback: pick the single best-matching node
    best_nid = min(
        candidates,
        key=lambda n: abs(int(node_ends[n]) - lr_end) + abs(int(node_starts[n]) - lr_start),
    )
    return [best_nid]


def get_guide_paths_for_locus(
    lr_transcripts: list[list[tuple[int, int]]],
    graph_csr: CSRGraph,
    locus_start: int,
    locus_end: int,
    tolerance: int = 5,
) -> list[list[int]]:
    """Map long-read transcripts overlapping a locus to graph paths.

    Filters the long-read transcript set to those overlapping the locus
    interval, then maps each to a node path.  Only successfully mapped
    paths are returned.

    Args:
        lr_transcripts: List of long-read transcripts, each a sorted
            list of (start, end) exon intervals.
        graph_csr: CSR splice graph for this locus.
        locus_start: Locus start coordinate.
        locus_end: Locus end coordinate.
        tolerance: Coordinate matching tolerance in bp.

    Returns:
        List of successfully mapped source-to-sink node paths.
    """
    guide_paths: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    for lr_exons in lr_transcripts:
        if not lr_exons:
            continue

        # Check overlap with locus
        tx_start = lr_exons[0][0]
        tx_end = lr_exons[-1][1]
        if tx_end < locus_start or tx_start > locus_end:
            continue

        path = map_longread_to_path(lr_exons, graph_csr, tolerance)
        if path is not None:
            path_key = tuple(path)
            if path_key not in seen:
                seen.add(path_key)
                guide_paths.append(path)

    if guide_paths:
        logger.debug(
            "Mapped %d/%d long-read transcripts to guide paths for locus [%d, %d)",
            len(guide_paths),
            sum(1 for lr in lr_transcripts
                if lr and lr[-1][1] >= locus_start and lr[0][0] <= locus_end),
            locus_start,
            locus_end,
        )

    return guide_paths
