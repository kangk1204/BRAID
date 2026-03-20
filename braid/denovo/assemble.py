"""Transcript path extraction from de Bruijn graph.

Implements coverage-guided path enumeration through the compacted,
simplified de Bruijn graph to reconstruct full-length transcripts.
Uses a greedy approach inspired by Trinity's Butterfly module:

1. Start from high-coverage source nodes (or all sources if few).
2. At each branching point, follow the highest-coverage outgoing edge.
3. Record the path and subtract coverage (to allow finding lower-
   expressed isoforms on subsequent passes).
4. Repeat until all paths above the minimum coverage are exhausted.

For RNA-seq data, coverage-guided traversal naturally recovers
dominant isoforms first, with subsequent passes finding alternative
splice forms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from braid.denovo.graph import DeBruijnGraph

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeNovoTranscript:
    """A transcript assembled from the de Bruijn graph.

    Attributes:
        transcript_id: Unique identifier.
        sequence: Full DNA sequence of the assembled transcript.
        coverage: Average coverage across the path.
        path_node_ids: Node IDs traversed in the graph.
        length: Length of the transcript in bases.
    """

    transcript_id: str
    sequence: str
    coverage: float
    path_node_ids: list[int] = field(default_factory=list)
    length: int = 0

    def __post_init__(self) -> None:
        """Set length from sequence if not provided."""
        if self.length == 0 and self.sequence:
            self.length = len(self.sequence)


@dataclass(slots=True)
class AssemblyConfig:
    """Configuration for transcript path extraction.

    Attributes:
        min_transcript_length: Minimum transcript length in bases.
        min_coverage: Minimum average path coverage to report.
        max_paths: Maximum number of paths to extract.
        coverage_fraction: Fraction of edge coverage to subtract after
            using an edge in a path (for multi-isoform recovery).
        max_path_length: Maximum path length in nodes to prevent
            infinite traversal in cyclic graphs.
    """

    min_transcript_length: int = 200
    min_coverage: float = 2.0
    max_paths: int = 10000
    coverage_fraction: float = 0.8
    max_path_length: int = 50000


def extract_transcripts(
    graph: DeBruijnGraph,
    cfg: AssemblyConfig | None = None,
) -> list[DeNovoTranscript]:
    """Extract transcript sequences from the de Bruijn graph.

    Uses iterative coverage-guided path traversal to recover multiple
    isoforms.  At each branching point, the highest-coverage outgoing
    edge is followed.  After each path is found, edge coverages are
    reduced to allow discovery of lower-expressed isoforms.

    Args:
        graph: Simplified and compacted de Bruijn graph.
        cfg: Assembly configuration parameters.

    Returns:
        List of assembled transcripts sorted by coverage (descending).
    """
    if cfg is None:
        cfg = AssemblyConfig()

    if graph.n_nodes == 0:
        return []

    # Build edge coverage lookup for modification
    edge_cov: dict[tuple[int, int], float] = {}
    for edge in graph.edges:
        edge_cov[(edge.source, edge.target)] = edge.coverage

    transcripts: list[DeNovoTranscript] = []
    path_id = 0

    # Get source nodes sorted by coverage (highest first)
    source_nodes = _get_start_nodes(graph)

    for start_nid in source_nodes:
        if path_id >= cfg.max_paths:
            break

        # Extract paths starting from this source
        while path_id < cfg.max_paths:
            path = _find_best_path(
                graph, start_nid, edge_cov, cfg.max_path_length,
            )
            if not path or len(path) < 2:
                break

            # Compute path coverage
            path_cov = _compute_path_coverage(path, edge_cov)
            if path_cov < cfg.min_coverage:
                break

            # Build transcript sequence from path nodes
            sequence = _path_to_sequence(graph, path)
            if len(sequence) < cfg.min_transcript_length:
                # Subtract coverage even for short paths to avoid loops
                _subtract_coverage(path, edge_cov, cfg.coverage_fraction)
                break

            transcript = DeNovoTranscript(
                transcript_id=f"DENOVO_{path_id:06d}",
                sequence=sequence,
                coverage=path_cov,
                path_node_ids=list(path),
            )
            transcripts.append(transcript)
            path_id += 1

            # Subtract coverage along the used path
            _subtract_coverage(path, edge_cov, cfg.coverage_fraction)

    # Sort by coverage descending
    transcripts.sort(key=lambda t: t.coverage, reverse=True)

    logger.info(
        "Extracted %d transcripts from de Bruijn graph "
        "(min_len=%d, min_cov=%.1f)",
        len(transcripts), cfg.min_transcript_length, cfg.min_coverage,
    )
    return transcripts


def _get_start_nodes(graph: DeBruijnGraph) -> list[int]:
    """Get starting nodes for path traversal, sorted by coverage.

    Prefers source nodes (in_degree=0) but falls back to all nodes
    if no sources exist (cyclic graph).

    Args:
        graph: De Bruijn graph.

    Returns:
        List of node IDs sorted by coverage (highest first).
    """
    sources = graph.sources()
    if sources:
        candidates = sources
    else:
        # Fall back to highest-coverage nodes for cyclic graphs
        candidates = list(graph.nodes.keys())

    # Sort by coverage descending
    candidates.sort(
        key=lambda nid: graph.nodes[nid].coverage,
        reverse=True,
    )
    return candidates


def _find_best_path(
    graph: DeBruijnGraph,
    start: int,
    edge_cov: dict[tuple[int, int], float],
    max_length: int,
) -> list[int]:
    """Find the highest-coverage path from a starting node.

    At each branching point, follows the outgoing edge with the highest
    remaining coverage.

    Args:
        graph: De Bruijn graph.
        start: Starting node ID.
        edge_cov: Current edge coverage dictionary.
        max_length: Maximum path length in nodes.

    Returns:
        List of node IDs forming the path.
    """
    if start not in graph.nodes:
        return []

    path = [start]
    visited: set[int] = {start}
    current = start

    for _ in range(max_length):
        node = graph.nodes.get(current)
        if node is None or node.out_degree == 0:
            break

        # Choose highest-coverage outgoing edge
        best_next = -1
        best_cov = -1.0

        for out_nid in node.out_edges:
            if out_nid in visited:
                continue
            cov = edge_cov.get((current, out_nid), 0.0)
            if cov > best_cov:
                best_cov = cov
                best_next = out_nid

        if best_next < 0 or best_cov <= 0:
            break

        path.append(best_next)
        visited.add(best_next)
        current = best_next

    return path


def _compute_path_coverage(
    path: list[int],
    edge_cov: dict[tuple[int, int], float],
) -> float:
    """Compute the minimum edge coverage along a path.

    Uses minimum rather than average because the bottleneck coverage
    determines the transcript abundance.

    Args:
        path: List of node IDs.
        edge_cov: Edge coverage dictionary.

    Returns:
        Minimum edge coverage along the path.
    """
    if len(path) < 2:
        return 0.0

    min_cov = float("inf")
    for i in range(len(path) - 1):
        cov = edge_cov.get((path[i], path[i + 1]), 0.0)
        min_cov = min(min_cov, cov)

    return min_cov if min_cov != float("inf") else 0.0


def _path_to_sequence(
    graph: DeBruijnGraph,
    path: list[int],
) -> str:
    """Convert a path of node IDs to a DNA sequence.

    For a compacted graph, takes the full sequence of the first node and
    appends the non-overlapping suffix of each subsequent node.  For
    uncompacted (k-1)-mer nodes, appends the last character.

    Args:
        graph: De Bruijn graph.
        path: List of node IDs.

    Returns:
        DNA sequence string.
    """
    if not path:
        return ""

    first_node = graph.nodes[path[0]]
    k = graph.k

    # Start with first node's full sequence
    seq_parts = [first_node.sequence]

    for i in range(1, len(path)):
        node = graph.nodes[path[i]]
        if node.unitig_length > k - 1:
            # Compacted unitig: overlap is (k-1) bases
            overlap = k - 1
            seq_parts.append(node.sequence[overlap:])
        else:
            # Uncompacted: just add the last base
            seq_parts.append(node.sequence[-1])

    return "".join(seq_parts)


def _subtract_coverage(
    path: list[int],
    edge_cov: dict[tuple[int, int], float],
    fraction: float,
) -> None:
    """Subtract coverage from edges along a used path.

    This enables multi-pass discovery of lower-expressed isoforms
    that share edges with already-discovered transcripts.

    Args:
        path: List of node IDs.
        edge_cov: Edge coverage dictionary (modified in place).
        fraction: Fraction of minimum path coverage to subtract.
    """
    if len(path) < 2:
        return

    min_cov = _compute_path_coverage(path, edge_cov)
    subtract = min_cov * fraction

    for i in range(len(path) - 1):
        key = (path[i], path[i + 1])
        if key in edge_cov:
            edge_cov[key] = max(0.0, edge_cov[key] - subtract)


def write_fasta(
    transcripts: list[DeNovoTranscript],
    output_path: str,
) -> None:
    """Write assembled transcripts to a FASTA file.

    Args:
        transcripts: List of assembled transcripts.
        output_path: Path to the output FASTA file.
    """
    with open(output_path, "w") as fh:
        for tx in transcripts:
            fh.write(f">{tx.transcript_id} len={tx.length} cov={tx.coverage:.1f}\n")
            # Write sequence in 80-character lines
            seq = tx.sequence
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i + 80] + "\n")

    logger.info("Wrote %d transcripts to %s", len(transcripts), output_path)
