"""De novo RNA-seq assembly pipeline.

Orchestrates the full de novo assembly workflow from FASTQ reads to
assembled transcript sequences:

1. Read FASTQ input
2. Extract and count k-mers
3. Build de Bruijn graph
4. Compact and simplify graph
5. Extract transcript paths
6. Write output FASTA

All stages log timing and statistics.  The pipeline is designed for
GPU acceleration but falls back to CPU NumPy for all operations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from braid.denovo.assemble import (
    AssemblyConfig,
    DeNovoTranscript,
    extract_transcripts,
    write_fasta,
)
from braid.denovo.fastq import stream_fastq_sequences
from braid.denovo.graph import (
    build_debruijn_graph,
    compact_graph,
)
from braid.denovo.kmer import count_kmers
from braid.denovo.simplify import SimplifyConfig, simplify_graph

logger = logging.getLogger(__name__)


@dataclass
class DeNovoConfig:
    """Configuration for the de novo assembly pipeline.

    Attributes:
        fastq_paths: List of input FASTQ file paths.
        output_path: Output FASTA file path.
        k: K-mer size for de Bruijn graph construction.
        min_kmer_count: Minimum k-mer count to retain (error filtering).
        canonical_kmers: Use canonical (strand-independent) k-mers.
        min_read_length: Minimum read length to include.
        min_tip_length: Maximum unitig length for tip removal.
        min_edge_coverage: Minimum edge coverage to retain.
        bubble_coverage_ratio: Coverage ratio threshold for bubble
            collapsing.
        min_transcript_length: Minimum assembled transcript length.
        min_transcript_coverage: Minimum transcript path coverage.
        max_transcripts: Maximum number of transcripts to report.
        simplify_iterations: Maximum simplification iterations.
    """

    fastq_paths: list[str]
    output_path: str = "denovo_transcripts.fa"
    k: int = 25
    min_kmer_count: int = 3
    canonical_kmers: bool = True
    min_read_length: int = 50
    min_tip_length: int = 50
    min_edge_coverage: float = 2.0
    bubble_coverage_ratio: float = 0.1
    min_transcript_length: int = 200
    min_transcript_coverage: float = 2.0
    max_transcripts: int = 100000
    simplify_iterations: int = 3


@dataclass(slots=True)
class DeNovoStats:
    """Statistics from the de novo assembly pipeline.

    Attributes:
        n_reads: Total number of reads processed.
        n_total_kmers: Total k-mers extracted.
        n_unique_kmers: Unique k-mers above min count.
        n_graph_nodes: Nodes in the initial graph.
        n_graph_edges: Edges in the initial graph.
        n_compacted_nodes: Nodes after compaction.
        n_compacted_edges: Edges after compaction.
        n_final_nodes: Nodes after simplification.
        n_final_edges: Edges after simplification.
        n_transcripts: Number of assembled transcripts.
        total_transcript_bases: Total bases in assembled transcripts.
        n50: N50 length of assembled transcripts.
        elapsed_seconds: Total pipeline wall time.
    """

    n_reads: int = 0
    n_total_kmers: int = 0
    n_unique_kmers: int = 0
    n_graph_nodes: int = 0
    n_graph_edges: int = 0
    n_compacted_nodes: int = 0
    n_compacted_edges: int = 0
    n_final_nodes: int = 0
    n_final_edges: int = 0
    n_transcripts: int = 0
    total_transcript_bases: int = 0
    n50: int = 0
    elapsed_seconds: float = 0.0


def run_denovo_assembly(
    config: DeNovoConfig,
) -> tuple[list[DeNovoTranscript], DeNovoStats]:
    """Run the full de novo assembly pipeline.

    Args:
        config: Pipeline configuration.

    Returns:
        Tuple of (transcripts, stats).
    """
    t0 = time.perf_counter()
    stats = DeNovoStats()

    # --- Stage 1: Read FASTQ input ---
    logger.info("Stage 1: Reading FASTQ input...")
    t1 = time.perf_counter()

    all_sequences: list[str] = []
    for fq_path in config.fastq_paths:
        seqs = stream_fastq_sequences(fq_path, min_length=config.min_read_length)
        all_sequences.extend(seqs)

    stats.n_reads = len(all_sequences)
    logger.info(
        "Read %d sequences in %.1fs",
        stats.n_reads, time.perf_counter() - t1,
    )

    if not all_sequences:
        logger.warning("No sequences loaded, returning empty assembly")
        stats.elapsed_seconds = time.perf_counter() - t0
        return [], stats

    # --- Stage 2: K-mer counting ---
    logger.info("Stage 2: Counting k-mers (k=%d)...", config.k)
    t2 = time.perf_counter()

    # De Bruijn graph construction requires non-canonical k-mers to
    # preserve the prefix/suffix adjacency relationship.  Canonical
    # k-mers break this because RC(kmer) has different prefix/suffix.
    kmer_table = count_kmers(
        all_sequences,
        k=config.k,
        min_count=config.min_kmer_count,
        canonical=False,
    )

    stats.n_unique_kmers = len(kmer_table.kmers)
    stats.n_total_kmers = int(kmer_table.counts.sum()) if len(kmer_table.counts) > 0 else 0
    logger.info(
        "Counted %d unique k-mers in %.1fs",
        stats.n_unique_kmers, time.perf_counter() - t2,
    )

    if stats.n_unique_kmers == 0:
        logger.warning("No k-mers above threshold, returning empty assembly")
        stats.elapsed_seconds = time.perf_counter() - t0
        return [], stats

    # --- Stage 3: Build de Bruijn graph ---
    logger.info("Stage 3: Building de Bruijn graph...")
    t3 = time.perf_counter()

    graph = build_debruijn_graph(kmer_table)
    stats.n_graph_nodes = graph.n_nodes
    stats.n_graph_edges = graph.n_edges
    logger.info(
        "Built graph with %d nodes, %d edges in %.1fs",
        stats.n_graph_nodes, stats.n_graph_edges,
        time.perf_counter() - t3,
    )

    # --- Stage 4: Compact graph ---
    logger.info("Stage 4: Compacting graph...")
    t4 = time.perf_counter()

    graph = compact_graph(graph)
    stats.n_compacted_nodes = graph.n_nodes
    stats.n_compacted_edges = graph.n_edges
    logger.info(
        "Compacted to %d nodes, %d edges in %.1fs",
        stats.n_compacted_nodes, stats.n_compacted_edges,
        time.perf_counter() - t4,
    )

    # --- Stage 5: Simplify graph ---
    logger.info("Stage 5: Simplifying graph...")
    t5 = time.perf_counter()

    # Tip length threshold must scale with k to avoid removing real nodes
    effective_tip_length = max(config.min_tip_length, 2 * config.k)
    simplify_cfg = SimplifyConfig(
        min_tip_length=effective_tip_length,
        min_coverage=config.min_edge_coverage,
        bubble_coverage_ratio=config.bubble_coverage_ratio,
        max_iterations=config.simplify_iterations,
    )
    simplify_graph(graph, simplify_cfg)

    stats.n_final_nodes = graph.n_nodes
    stats.n_final_edges = graph.n_edges
    logger.info(
        "Simplified to %d nodes, %d edges in %.1fs",
        stats.n_final_nodes, stats.n_final_edges,
        time.perf_counter() - t5,
    )

    # --- Stage 6: Extract transcripts ---
    logger.info("Stage 6: Extracting transcripts...")
    t6 = time.perf_counter()

    assembly_cfg = AssemblyConfig(
        min_transcript_length=config.min_transcript_length,
        min_coverage=config.min_transcript_coverage,
        max_paths=config.max_transcripts,
    )
    transcripts = extract_transcripts(graph, assembly_cfg)

    stats.n_transcripts = len(transcripts)
    stats.total_transcript_bases = sum(t.length for t in transcripts)
    stats.n50 = _compute_n50([t.length for t in transcripts])
    logger.info(
        "Extracted %d transcripts (N50=%d) in %.1fs",
        stats.n_transcripts, stats.n50,
        time.perf_counter() - t6,
    )

    # --- Stage 7: Write output ---
    if transcripts:
        logger.info("Stage 7: Writing output to %s...", config.output_path)
        write_fasta(transcripts, config.output_path)

    stats.elapsed_seconds = time.perf_counter() - t0
    _log_summary(stats)

    return transcripts, stats


def _compute_n50(lengths: list[int]) -> int:
    """Compute the N50 statistic for a set of lengths.

    N50 is the length at which 50% of the total assembly is in
    sequences of at least this length.

    Args:
        lengths: List of sequence lengths.

    Returns:
        N50 length, or 0 if no sequences.
    """
    if not lengths:
        return 0
    sorted_lengths = sorted(lengths, reverse=True)
    total = sum(sorted_lengths)
    running = 0
    for length in sorted_lengths:
        running += length
        if running >= total / 2:
            return length
    return 0


def _log_summary(stats: DeNovoStats) -> None:
    """Log a summary of assembly statistics.

    Args:
        stats: Pipeline statistics.
    """
    logger.info(
        "\n=== De Novo Assembly Summary ===\n"
        "  Reads:            %d\n"
        "  Unique k-mers:    %d\n"
        "  Graph:            %d nodes, %d edges\n"
        "  After compaction: %d nodes, %d edges\n"
        "  After simplify:   %d nodes, %d edges\n"
        "  Transcripts:      %d\n"
        "  Total bases:      %d\n"
        "  N50:              %d\n"
        "  Time:             %.1fs\n"
        "================================",
        stats.n_reads, stats.n_unique_kmers,
        stats.n_graph_nodes, stats.n_graph_edges,
        stats.n_compacted_nodes, stats.n_compacted_edges,
        stats.n_final_nodes, stats.n_final_edges,
        stats.n_transcripts, stats.total_transcript_bases,
        stats.n50, stats.elapsed_seconds,
    )
