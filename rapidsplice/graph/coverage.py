"""Coverage computation for splice graph nodes and edges.

Computes per-base and per-node coverage from aligned reads,
using either GPU-accelerated or CPU-based parallel scanning.
"""

from __future__ import annotations

import logging

import numpy as np

from rapidsplice.cuda.kernels import parallel_coverage_scan
from rapidsplice.graph.splice_graph import NodeType, SpliceGraph
from rapidsplice.io.bam_reader import ReadData

logger = logging.getLogger(__name__)


def compute_node_coverages(
    graph: SpliceGraph,
    read_data: ReadData,
    chrom_id: int,
) -> None:
    """Compute and assign per-node average coverage from read data.

    Modifies the graph in-place, setting coverage on each EXON node.

    Args:
        graph: The splice graph to update.
        read_data: Bulk-extracted read data for the relevant region.
        chrom_id: Numeric chromosome ID to filter reads.
    """
    if read_data.n_reads == 0:
        return

    # Filter to reads on this chromosome
    chrom_mask = read_data.chrom_ids == chrom_id
    positions = read_data.positions[chrom_mask]
    end_positions = read_data.end_positions[chrom_mask]

    if len(positions) == 0:
        return

    for node_id in range(graph.n_nodes):
        node = graph.get_node(node_id)
        if node.node_type != NodeType.EXON:
            continue

        cov = _region_mean_coverage(positions, end_positions, node.start, node.end)
        graph._nodes[node_id] = type(node)(
            node_id=node.node_id,
            start=node.start,
            end=node.end,
            node_type=node.node_type,
            coverage=cov,
            length=node.length,
        )


def compute_edge_coverages(
    graph: SpliceGraph,
    read_data: ReadData,
    chrom_id: int,
) -> None:
    """Compute and assign edge coverages from junction-spanning reads.

    For intron edges, coverage is the number of reads spanning the junction.
    For other edges, coverage is the average of the two endpoint node coverages.

    Args:
        graph: The splice graph to update.
        read_data: Bulk-extracted read data.
        chrom_id: Numeric chromosome ID.
    """
    if read_data.n_reads == 0:
        return

    # Extract junctions from CIGAR
    junctions = _extract_read_junctions(read_data, chrom_id)

    for key, edge in list(graph._edges.items()):
        src_node = graph.get_node(edge.src)
        dst_node = graph.get_node(edge.dst)

        if edge.edge_type.value == 0:  # INTRON
            # Count reads with this exact junction
            count = 0
            for js, je in junctions:
                if js == src_node.end and je == dst_node.start:
                    count += 1
            graph._edges[key] = type(edge)(
                src=edge.src, dst=edge.dst, edge_type=edge.edge_type,
                weight=edge.weight, coverage=float(count),
            )
        else:
            # Average of endpoint coverages
            avg = (src_node.coverage + dst_node.coverage) / 2.0
            graph._edges[key] = type(edge)(
                src=edge.src, dst=edge.dst, edge_type=edge.edge_type,
                weight=edge.weight, coverage=avg,
            )


def compute_region_coverage(
    read_data: ReadData,
    chrom_id: int,
    region_start: int,
    region_end: int,
) -> np.ndarray:
    """Compute per-base coverage for a genomic region.

    Uses the optimized parallel scan kernel.

    Args:
        read_data: Bulk-extracted read data.
        chrom_id: Chromosome ID to filter reads.
        region_start: 0-based region start.
        region_end: 0-based exclusive region end.

    Returns:
        1D array of per-base coverage values.
    """
    if read_data.n_reads == 0:
        return np.zeros(region_end - region_start, dtype=np.int32)

    chrom_mask = read_data.chrom_ids == chrom_id
    positions = read_data.positions[chrom_mask]
    end_positions = read_data.end_positions[chrom_mask]

    if len(positions) == 0:
        return np.zeros(region_end - region_start, dtype=np.int32)

    return parallel_coverage_scan(positions, end_positions, region_start, region_end)


def _region_mean_coverage(
    positions: np.ndarray,
    end_positions: np.ndarray,
    region_start: int,
    region_end: int,
) -> float:
    """Compute mean coverage over a region using delta-encoding."""
    length = region_end - region_start
    if length <= 0:
        return 0.0

    cov = parallel_coverage_scan(positions, end_positions, region_start, region_end)
    return float(np.mean(cov))


def _extract_read_junctions(
    read_data: ReadData,
    chrom_id: int,
) -> list[tuple[int, int]]:
    """Extract all splice junctions from reads on a chromosome.

    Args:
        read_data: Bulk-extracted read data.
        chrom_id: Chromosome ID to filter.

    Returns:
        List of (intron_start, intron_end) tuples.
    """
    junctions: list[tuple[int, int]] = []
    chrom_mask = read_data.chrom_ids == chrom_id
    indices = np.where(chrom_mask)[0]

    for idx in indices:
        start = read_data.cigar_offsets[idx]
        end = read_data.cigar_offsets[idx + 1]
        pos = int(read_data.positions[idx])
        for j in range(start, end):
            op = read_data.cigar_ops[j]
            length = int(read_data.cigar_lens[j])
            if op == 3:  # CIGAR_N
                junctions.append((pos, pos + length))
                pos += length
            elif op in (0, 2, 7, 8):  # M, D, EQ, X
                pos += length
    return junctions
