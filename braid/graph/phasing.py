"""Paired-end read phasing path extraction.

Extracts phasing paths from paired-end and multi-junction reads,
which constrain flow decomposition to produce biologically consistent
transcript isoforms (following Scallop's phasing approach).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from braid.graph.splice_graph import NodeType, SpliceGraph
from braid.io.bam_reader import ReadData

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PhasingPath:
    """A phasing constraint extracted from a read or read pair.

    Represents a sequence of exon nodes that must co-occur in at least
    one assembled transcript.
    """

    node_ids: list[int]
    weight: float = 1.0
    read_count: int = 1


@dataclass(slots=True)
class PhasingEvidence:
    """Collection of phasing paths for a gene locus."""

    paths: list[PhasingPath]
    locus_chrom: str
    locus_start: int
    locus_end: int

    @property
    def n_paths(self) -> int:
        """Return the number of distinct phasing paths."""
        return len(self.paths)

    @property
    def total_weight(self) -> float:
        """Return total weight of all phasing paths."""
        return sum(p.weight for p in self.paths)


def extract_phasing_paths(
    graph: SpliceGraph,
    read_data: ReadData,
    chrom_id: int,
    min_path_length: int = 2,
) -> PhasingEvidence:
    """Extract phasing paths from reads mapped to a splice graph.

    A phasing path is a sequence of graph nodes that a single read
    (or read pair) spans. Multi-junction reads that cross >=2 junctions
    provide the strongest phasing signal.

    Args:
        graph: The splice graph for this locus.
        read_data: Bulk-extracted read data.
        chrom_id: Numeric chromosome ID.
        min_path_length: Minimum number of exon nodes in a phasing path.

    Returns:
        PhasingEvidence with deduplicated phasing paths.
    """
    if read_data.n_reads == 0:
        return PhasingEvidence(
            paths=[], locus_chrom=graph.chrom,
            locus_start=graph.locus_start, locus_end=graph.locus_end,
        )

    # Build position-to-node lookup
    node_lookup = _build_node_lookup(graph)

    # Extract per-read phasing paths
    raw_paths: dict[tuple[int, ...], int] = {}

    chrom_mask = read_data.chrom_ids == chrom_id
    indices = np.where(chrom_mask)[0]

    # Group paired reads by name for pair-aware phasing
    pair_groups: dict[str, list[int]] = {}
    for idx in indices:
        name = read_data.query_names[idx]
        if name not in pair_groups:
            pair_groups[name] = []
        pair_groups[name].append(idx)

    for read_indices in pair_groups.values():
        # Collect all exon blocks from all reads in this pair
        all_blocks: list[tuple[int, int]] = []
        for idx in read_indices:
            blocks = _read_exon_blocks(read_data, idx)
            all_blocks.extend(blocks)

        if not all_blocks:
            continue

        # Sort blocks by position
        all_blocks.sort()

        # Map blocks to graph nodes
        path_nodes: list[int] = []
        for block_start, block_end in all_blocks:
            node_id = _find_node_for_block(node_lookup, block_start, block_end)
            if node_id is not None and (not path_nodes or path_nodes[-1] != node_id):
                path_nodes.append(node_id)

        if len(path_nodes) >= min_path_length:
            key = tuple(path_nodes)
            raw_paths[key] = raw_paths.get(key, 0) + 1

    # Convert to PhasingPath objects
    paths = [
        PhasingPath(node_ids=list(node_ids), weight=float(count), read_count=count)
        for node_ids, count in raw_paths.items()
    ]

    # Sort by weight descending
    paths.sort(key=lambda p: -p.weight)

    logger.debug(
        "Extracted %d phasing paths from %d read groups for %s:%d-%d",
        len(paths), len(pair_groups), graph.chrom, graph.locus_start, graph.locus_end,
    )

    return PhasingEvidence(
        paths=paths, locus_chrom=graph.chrom,
        locus_start=graph.locus_start, locus_end=graph.locus_end,
    )


def apply_phasing_constraints(
    graph: SpliceGraph,
    phasing: PhasingEvidence,
    min_support: int = 2,
) -> list[PhasingPath]:
    """Filter phasing paths by support and validate against graph topology.

    Args:
        graph: The splice graph.
        phasing: Phasing evidence to filter.
        min_support: Minimum read support for a phasing path.

    Returns:
        List of validated phasing paths.
    """
    valid_paths: list[PhasingPath] = []

    for path in phasing.paths:
        if path.read_count < min_support:
            continue

        # Validate: each consecutive pair of nodes must be connected
        is_valid = True
        for i in range(len(path.node_ids) - 1):
            u = path.node_ids[i]
            v = path.node_ids[i + 1]
            # Check direct edge or reachability through intermediate nodes
            if not _nodes_connected(graph, u, v):
                is_valid = False
                break

        if is_valid:
            valid_paths.append(path)

    logger.debug(
        "Validated %d/%d phasing paths (min_support=%d)",
        len(valid_paths), len(phasing.paths), min_support,
    )
    return valid_paths


def _build_node_lookup(graph: SpliceGraph) -> list[tuple[int, int, int]]:
    """Build a sorted list of (start, end, node_id) for exon nodes."""
    lookup: list[tuple[int, int, int]] = []
    for nid in range(graph.n_nodes):
        node = graph.get_node(nid)
        if node.node_type == NodeType.EXON:
            lookup.append((node.start, node.end, nid))
    lookup.sort()
    return lookup


def _find_node_for_block(
    node_lookup: list[tuple[int, int, int]],
    block_start: int,
    block_end: int,
) -> int | None:
    """Find the graph node that best matches a read alignment block.

    Uses overlap-based matching: the node with the largest overlap
    with the block wins, requiring at least 50% reciprocal overlap.
    """
    best_node: int | None = None
    best_overlap = 0

    for node_start, node_end, node_id in node_lookup:
        if node_start >= block_end:
            break
        if node_end <= block_start:
            continue

        overlap_start = max(block_start, node_start)
        overlap_end = min(block_end, node_end)
        overlap = overlap_end - overlap_start

        if overlap <= 0:
            continue

        block_len = block_end - block_start
        node_len = node_end - node_start
        block_frac = overlap / max(block_len, 1)
        node_frac = overlap / max(node_len, 1)

        # Require at least 50% overlap of the block
        if block_frac >= 0.5 and overlap > best_overlap:
            best_overlap = overlap
            best_node = node_id
        elif node_frac >= 0.8 and overlap > best_overlap:
            best_overlap = overlap
            best_node = node_id

    return best_node


def _read_exon_blocks(
    read_data: ReadData, read_idx: int
) -> list[tuple[int, int]]:
    """Extract exon alignment blocks from a single read's CIGAR."""
    blocks: list[tuple[int, int]] = []
    start = int(read_data.cigar_offsets[read_idx])
    end = int(read_data.cigar_offsets[read_idx + 1])
    pos = int(read_data.positions[read_idx])
    block_start: int | None = None

    for j in range(start, end):
        op = read_data.cigar_ops[j]
        length = int(read_data.cigar_lens[j])
        if op in (0, 7, 8):  # M, EQ, X
            if block_start is None:
                block_start = pos
            pos += length
        elif op == 2:  # D
            if block_start is None:
                block_start = pos
            pos += length
        elif op == 3:  # N (intron)
            if block_start is not None:
                blocks.append((block_start, pos))
                block_start = None
            pos += length
        # I, S, H, P: do not consume reference

    if block_start is not None:
        blocks.append((block_start, pos))
    return blocks


def _nodes_connected(graph: SpliceGraph, u: int, v: int) -> bool:
    """Check if node v is reachable from node u within a short distance."""
    # Direct edge
    if v in graph.get_successors(u):
        return True
    # Two-hop (through one intermediate node)
    for mid in graph.get_successors(u):
        if v in graph.get_successors(mid):
            return True
    return False
