"""Main assembly pipeline orchestrating all BRAID modules."""

from __future__ import annotations

import logging
import math
import multiprocessing
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

from braid.cuda.backend import get_backend, set_backend
from braid.diagnostics import (
    ChromosomeDiagnostics,
    DiagnosticsCollector,
    LocusDiagnostics,
)
from braid.flow.bootstrap import BootstrapConfig, bootstrap_confidence
from braid.flow.decompose import DecomposeConfig, Transcript
from braid.flow.decomposer import run_decomposer_pair
from braid.flow.longread_guide import get_guide_paths_for_locus
from braid.flow.nmf_decompose import NMFDecomposeConfig, decompose_nmf
from braid.graph.builder import GraphBuilderConfig, SpliceGraphBuilder
from braid.graph.phasing import apply_phasing_constraints, extract_phasing_paths
from braid.graph.splice_graph import SpliceGraph
from braid.io.bam_reader import (
    BamReader,
    ReadData,
    extract_junctions_from_bam,
)
from braid.io.gtf_writer import Gff3Writer, GtfWriter, TranscriptRecord, compute_expression
from braid.io.reference import ReferenceGenome
from braid.scoring.features import extract_features, features_to_array
from braid.scoring.filter import FilterConfig, TranscriptFilter
from braid.scoring.model import TranscriptScorer
from braid.utils.cigar import CIGAR_D, CIGAR_EQ, CIGAR_M, CIGAR_N, CIGAR_X
from braid.utils.stats import AssemblyStats, Timer

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the BRAID assembly pipeline.

    Controls every user-facing parameter of the assembler, from input
    paths and output format to algorithmic thresholds and hardware
    backend selection.

    Attributes:
        bam_path: Path to the coordinate-sorted, indexed BAM file.
        reference_path: Optional path to a reference genome FASTA (with
            ``.fai`` index).  When provided, splice-site motif validation
            is enabled.
        output_path: Filesystem path for the assembled transcript output.
        output_format: Output annotation format, either ``"gtf"`` or
            ``"gff3"``.
        backend: Compute backend selection.  ``"auto"`` probes for a GPU
            and falls back to CPU.  ``"cpu"`` and ``"gpu"`` force the
            respective backend.
        threads: Number of worker threads for CPU-parallel stages.
        min_mapq: Minimum read mapping quality to retain during BAM
            extraction (inclusive).
        min_junction_support: Minimum number of reads required to support
            a splice junction before it is included in graph construction.
        min_coverage: Minimum average per-base read coverage for an exon
            or transcript to be retained.
        min_transcript_score: Minimum ML (or heuristic) quality score for
            a transcript to survive filtering.  Range ``[0, 1]``.
        max_intron_length: Maximum intron (junction span) length in base
            pairs.  Junctions exceeding this are discarded as likely
            alignment artefacts.
        use_safe_paths: Whether to compute safe subpaths as a first pass
            during flow decomposition.
        use_ml_scoring: Whether to use the ML-based transcript scorer.
            When ``False``, a lightweight heuristic formula is used
            instead.
        model_path: Optional path to a pre-trained ``joblib``-serialized
            scoring model.  Ignored when ``use_ml_scoring`` is ``False``.
        chromosomes: Optional list of chromosome names to restrict
            assembly to.  ``None`` processes all chromosomes present in
            the BAM header.
        verbose: Enable verbose (DEBUG-level) logging output.
    """

    bam_path: str
    reference_path: str | None = None
    output_path: str = "braid_output.gtf"
    output_format: str = "gtf"
    backend: str = "auto"
    threads: int = 1
    min_mapq: int = 0
    min_junction_support: int = 2
    min_phasing_support: int = 1
    min_coverage: float = 1.0
    min_transcript_score: float = 0.3
    max_intron_length: int = 500_000
    use_safe_paths: bool = False
    use_ml_scoring: bool = True
    model_path: str | None = None
    chromosomes: list[str] | None = None
    verbose: bool = False
    detect_splicing_events: bool = False
    use_nmf_decomposition: bool = False
    use_neural_decomposition: bool = False
    neural_decomposer_path: str | None = None
    use_junction_scoring: bool = False
    junction_scorer_path: str | None = None
    use_gnn_scoring: bool = False
    gnn_scorer_path: str | None = None
    use_transformer_classifier: bool = False
    transformer_classifier_path: str | None = None
    use_neural_psi: bool = False
    neural_psi_path: str | None = None
    use_boundary_detection: bool = False
    boundary_detector_path: str | None = None
    optimize_filter_thresholds: bool = False
    diagnostics_dir: str | None = None
    decomposer: str = "legacy"
    shadow_decomposer: str | None = None
    builder_profile: str = "default"
    adaptive_junction_filter: bool = True
    enable_motif_validation: bool = True
    min_anchor_length: int = 8
    max_paths: int = 2000
    min_relative_abundance: float = 0.01
    max_terminal_exon_length: int = 5000
    relaxed_pruning_experiment: bool = False
    disable_single_exon: bool = False
    strandedness: str = "none"
    enable_bootstrap: bool = False
    bootstrap_replicates: int = 100
    guide_gtf_path: str | None = None
    guide_tolerance: int = 5


_REF_CONSUMING = {CIGAR_M, CIGAR_D, CIGAR_N, CIGAR_EQ, CIGAR_X}


def _cap_terminal_exons(
    records: list[TranscriptRecord],
    max_length: int,
) -> int:
    """Trim oversized terminal exons on multi-exon transcripts.

    For each transcript with two or more exons, if the first or last exon
    exceeds *max_length* base pairs it is trimmed back to exactly
    *max_length*.  Single-exon transcripts are left untouched (they have
    no intron anchors to validate boundaries).  The transcript-level
    ``start`` and ``end`` fields are updated to match the (possibly new)
    exon boundaries.

    Args:
        records: Mutable list of assembled transcript records.
        max_length: Maximum allowed length (bp) for the first and last
            exon of a multi-exon transcript.

    Returns:
        The number of exons that were trimmed.
    """
    trimmed = 0
    for rec in records:
        exons = rec.exons
        if len(exons) < 2:
            continue

        # First exon
        first_start, first_end = exons[0]
        if first_end - first_start > max_length:
            exons[0] = (first_end - max_length, first_end)
            trimmed += 1

        # Last exon
        last_start, last_end = exons[-1]
        if last_end - last_start > max_length:
            exons[-1] = (last_start, last_start + max_length)
            trimmed += 1

        # Sync transcript boundaries
        rec.start = exons[0][0]
        rec.end = exons[-1][1]

    return trimmed


def _effective_builder_profile(cfg: PipelineConfig) -> str:
    """Resolve the active builder profile, honoring the legacy alias."""
    if cfg.builder_profile != "default":
        return cfg.builder_profile
    if cfg.relaxed_pruning_experiment:
        return "aggressive_recall"
    return "default"


def _make_builder_config(cfg: PipelineConfig) -> GraphBuilderConfig:
    """Translate the user-facing pipeline config into builder parameters."""
    profile = _effective_builder_profile(cfg)
    locus_flank = 800
    min_exon_coverage = cfg.min_coverage
    junction_merge_distance = 10
    min_relative_junction_support = 0.03
    min_relative_exon_coverage = 0.01
    assign_ambiguous = True
    add_fallback_terminal_edges = True

    if profile == "conservative_correctness":
        locus_flank = 800
        min_exon_coverage = min(cfg.min_coverage, 0.5)
        junction_merge_distance = 2
        min_relative_junction_support = 0.01
        min_relative_exon_coverage = 0.005
        assign_ambiguous = False
        add_fallback_terminal_edges = False
    elif profile == "aggressive_recall":
        locus_flank = 1200
        min_exon_coverage = min(cfg.min_coverage, 0.1)
        junction_merge_distance = 0
        min_relative_junction_support = 0.0
        min_relative_exon_coverage = 0.0
        assign_ambiguous = False
        add_fallback_terminal_edges = False

    return GraphBuilderConfig(
        locus_flank=locus_flank,
        min_junction_support=cfg.min_junction_support,
        min_exon_coverage=min_exon_coverage,
        max_intron_length=cfg.max_intron_length,
        junction_merge_distance=junction_merge_distance,
        min_relative_junction_support=min_relative_junction_support,
        min_relative_exon_coverage=min_relative_exon_coverage,
        assign_ambiguous_junctions_to_dominant_strand=assign_ambiguous,
        add_fallback_terminal_edges=add_fallback_terminal_edges,
    )


def _extract_per_read_junctions(
    read_data: ReadData,
) -> tuple[list[list[int]], list[list[int]]]:
    """Extract per-read junction start/end positions from CIGAR data.

    Walks through each read's CIGAR operations and records intron
    (N-operation) boundaries.

    Args:
        read_data: Bulk-extracted read data with CIGAR arrays.

    Returns:
        (junction_starts, junction_ends) where each is a list of lists,
        one per read, containing the genomic start and end of each junction
        (intron) spanned by that read.
    """
    n_reads = read_data.n_reads
    junction_starts: list[list[int]] = []
    junction_ends: list[list[int]] = []

    for i in range(n_reads):
        cig_start = int(read_data.cigar_offsets[i])
        cig_end = int(read_data.cigar_offsets[i + 1])
        pos = int(read_data.positions[i])
        js: list[int] = []
        je: list[int] = []

        for ci in range(cig_start, cig_end):
            op = int(read_data.cigar_ops[ci])
            length = int(read_data.cigar_lens[ci])
            if op == CIGAR_N:
                js.append(pos)
                je.append(pos + length)
            if op in _REF_CONSUMING:
                pos += length

        junction_starts.append(js)
        junction_ends.append(je)

    return junction_starts, junction_ends


def _process_chromosome_worker(
    cfg: PipelineConfig,
    chrom: str,
) -> list[TranscriptRecord]:
    """Standalone worker for processing a single chromosome in a subprocess.

    Creates its own BAM reader, graph builder, scorer, and filter so it can
    run in a ``ProcessPoolExecutor`` without sharing state.  This avoids
    the I/O contention that cripples threaded BAM access on large files.

    Args:
        cfg: Pipeline configuration (fully serializable dataclass).
        chrom: Chromosome name to process.

    Returns:
        List of assembled ``TranscriptRecord`` objects for this chromosome.
    """
    # Re-initialise logging in the worker process.
    log = logging.getLogger("braid")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG if cfg.verbose else logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(handler.level)

    # Create worker-local resources.
    bam_reader = BamReader(bam_path=cfg.bam_path, min_mapq=cfg.min_mapq)
    reference = (
        ReferenceGenome(cfg.reference_path)
        if cfg.reference_path is not None
        else None
    )
    graph_builder = SpliceGraphBuilder(config=_make_builder_config(cfg))
    model_path = cfg.model_path if cfg.use_ml_scoring else None
    scorer = TranscriptScorer(model_path=model_path)
    transcript_filter = TranscriptFilter(
        config=FilterConfig(
            min_score=cfg.min_transcript_score,
            min_coverage=cfg.min_coverage,
            min_junction_support=cfg.min_junction_support,
        ),
    )

    # Load guide GTF if configured
    guide_transcripts = None
    if cfg.guide_gtf_path is not None:
        from braid.io.gtf_reader import read_guide_gtf
        guide_transcripts = read_guide_gtf(cfg.guide_gtf_path)

    chrom_length = bam_reader.chromosome_lengths.get(chrom, 0)
    logger.info("Worker: processing chromosome %s (length %d)", chrom, chrom_length)

    # --- Pass 1: Junction extraction ---
    junctions, n_spliced, extraction_stats = extract_junctions_from_bam(
        cfg.bam_path, chrom, min_mapq=cfg.min_mapq,
        min_anchor_length=cfg.min_anchor_length,
        reference=reference if cfg.enable_motif_validation else None,
        return_stats=True,
        strandedness=cfg.strandedness,
    )
    n_junctions = extraction_stats.output_junctions
    logger.info("Worker: extracted %d junctions on %s", n_junctions, chrom)

    # --- Depth-adaptive junction filtering ---
    if (
        cfg.adaptive_junction_filter
        and _effective_builder_profile(cfg) == "default"
        and n_spliced > 0
        and n_junctions > 0
    ):
        adaptive_min = max(
            cfg.min_junction_support,
            min(5, int(math.sqrt(n_spliced / 100_000))),
        )
        base_min = graph_builder.config.min_junction_support
        if adaptive_min > base_min:
            logger.info(
                "Worker: depth-adaptive filter %d on %s (%d spliced reads)",
                adaptive_min, chrom, n_spliced,
            )
            keep = junctions.counts >= adaptive_min
            junctions = type(junctions)(
                chrom=junctions.chrom,
                starts=junctions.starts[keep],
                ends=junctions.ends[keep],
                counts=junctions.counts[keep],
                strands=junctions.strands[keep],
            )
            n_junctions = len(junctions.starts)

    # --- Identify loci ---
    loci = (
        graph_builder.identify_loci(junctions, chrom_length)
        if n_junctions > 0
        else []
    )
    logger.info("Worker: %d loci on %s", len(loci), chrom)

    # --- Per-locus assembly ---
    all_records: list[TranscriptRecord] = []
    gene_counter = 0
    has_paired_reads: bool | None = None

    for locus in loci:
        read_data = bam_reader.fetch_region(chrom, locus.start, locus.end)
        if read_data.n_reads == 0:
            continue

        if has_paired_reads is None:
            if hasattr(read_data, 'is_paired') and read_data.is_paired is not None:
                has_paired_reads = bool(np.any(read_data.is_paired))
            else:
                sample_size = min(200, read_data.n_reads)
                sample_names = read_data.query_names[:sample_size]
                has_paired_reads = len(set(sample_names)) < len(sample_names)

        graph = graph_builder.build_graph(locus, read_data, junctions)
        if graph is None:
            continue

        # Phasing
        phasing_for_decomp: list[tuple[list[int], float]] | None = None
        if has_paired_reads:
            chrom_id = int(read_data.chrom_ids[0]) if read_data.n_reads > 0 else -1
            phasing = extract_phasing_paths(graph, read_data, chrom_id)
            if phasing.n_paths > 0:
                valid_phasing = apply_phasing_constraints(
                    graph, phasing, min_support=cfg.min_phasing_support,
                )
                phasing_for_decomp = [
                    (p.node_ids, p.weight) for p in valid_phasing
                ] or None

        # Decompose
        graph_csr = graph.to_csr()
        topo = graph.topological_order()
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(topo)}
        if phasing_for_decomp:
            remapped = []
            for node_ids, weight in phasing_for_decomp:
                try:
                    remapped_ids = [old_to_new[nid] for nid in node_ids]
                except KeyError:
                    continue
                remapped.append((remapped_ids, weight))
            phasing_for_decomp = remapped or None

        # Map long-read guide paths for this locus
        locus_guide_paths: list[list[int]] | None = None
        if guide_transcripts is not None:
            strand = graph.strand if graph.strand in ("+", "-") else "+"
            lr_key = (chrom, strand)
            lr_txs = guide_transcripts.get(lr_key, [])
            if lr_txs:
                locus_guide_paths = get_guide_paths_for_locus(
                    lr_txs, graph_csr, locus.start, locus.end,
                    tolerance=cfg.guide_tolerance,
                ) or None

        decompose_config = DecomposeConfig(
            min_transcript_coverage=cfg.min_coverage,
            min_relative_abundance=cfg.min_relative_abundance,
            use_safe_paths=cfg.use_safe_paths,
            max_paths=cfg.max_paths,
        )
        primary_run, _ = run_decomposer_pair(
            graph_csr, graph,
            config=decompose_config,
            phasing_paths=phasing_for_decomp,
            guide_paths=locus_guide_paths,
            mode=cfg.decomposer,
            shadow_mode=None,  # skip shadow in parallel workers
        )
        transcripts = primary_run.transcripts

        if not transcripts:
            continue

        # Intron chain merging
        if len(transcripts) > 1:
            transcripts = transcript_filter.merge_identical_intron_chains(
                transcripts,
            )

        # Score and filter
        scores_list: list[float] = []
        features_list = []
        for tx in transcripts:
            feat = extract_features(tx, graph, transcripts, junctions)
            features_list.append(feat)
            feat_array = features_to_array(feat)
            score = scorer.score(feat_array)
            scores_list.append(score)

        scores_array = np.array(scores_list, dtype=np.float64)
        surviving_indices, _ = (
            transcript_filter.filter_transcripts_with_diagnostics(
                transcripts, scores_array, features_list,
            )
        )
        surviving_transcripts = [transcripts[i] for i in surviving_indices]
        surviving_scores = [scores_list[i] for i in surviving_indices]

        # Bootstrap confidence intervals
        bootstrap_results: list[dict[str, float]] | None = None
        if cfg.enable_bootstrap and surviving_transcripts and len(surviving_transcripts) > 0:
            paths = [tx.node_ids for tx in surviving_transcripts]
            try:
                boot_cfg = BootstrapConfig(
                    n_replicates=cfg.bootstrap_replicates,
                    seed=42,
                )
                boot_result = bootstrap_confidence(graph_csr, paths, boot_cfg)
                bootstrap_results = [
                    {
                        "ci_low": tc.weight_ci_low,
                        "ci_high": tc.weight_ci_high,
                        "presence": tc.presence_rate,
                        "cv": tc.cv,
                    }
                    for tc in boot_result.transcripts
                ]
            except Exception as exc:
                logger.debug("Bootstrap failed for locus: %s", exc)

        if surviving_transcripts:
            gene_counter += 1
            gene_prefix = f"RSPG{chrom}.{gene_counter}"
            strand = graph.strand if graph.strand in ("+", "-") else "+"
            for tx_idx, (tx, score) in enumerate(
                zip(surviving_transcripts, surviving_scores), start=1,
            ):
                exons = tx.exon_coords
                if not exons:
                    continue
                rec = TranscriptRecord(
                    transcript_id=f"{gene_prefix}.{tx_idx}",
                    gene_id=gene_prefix,
                    chrom=graph.chrom,
                    strand=strand,
                    start=exons[0][0],
                    end=exons[-1][1],
                    exons=list(exons),
                    score=score,
                    coverage=tx.weight,
                )
                if bootstrap_results and (tx_idx - 1) < len(bootstrap_results):
                    br = bootstrap_results[tx_idx - 1]
                    rec.bootstrap_ci_low = br["ci_low"]
                    rec.bootstrap_ci_high = br["ci_high"]
                    rec.bootstrap_presence = br["presence"]
                    rec.bootstrap_cv = br["cv"]
                all_records.append(rec)

    # --- Single-exon gene detection ---
    if not cfg.disable_single_exon:
        se_records = _detect_single_exon_genes_standalone(
            cfg, chrom, chrom_length, loci, gene_counter,
        )
        all_records.extend(se_records)

    if reference is not None:
        reference.close()

    logger.info(
        "Worker: %s done — %d transcripts (%d single-exon)",
        chrom, len(all_records), len(se_records),
    )
    return all_records


def _detect_single_exon_genes_standalone(
    cfg: PipelineConfig,
    chrom: str,
    chrom_length: int,
    existing_loci: list,
    gene_counter_start: int,
) -> list[TranscriptRecord]:
    """Standalone single-exon gene detection for use in worker processes.

    Mirrors ``AssemblyPipeline._detect_single_exon_genes`` but operates
    without access to the pipeline instance.

    Args:
        cfg: Pipeline configuration.
        chrom: Chromosome name.
        chrom_length: Chromosome length in bp.
        existing_loci: Loci from junction-based assembly.
        gene_counter_start: Gene counter offset.

    Returns:
        List of single-exon ``TranscriptRecord`` objects.
    """
    import pysam as _pysam

    min_se_coverage = max(50.0, cfg.min_coverage * 20)
    min_se_length = 500
    min_se_coverage_uniformity = 1.0  # stricter CV for uniformity

    # Build gap regions.
    locus_intervals = sorted((loc.start, loc.end) for loc in existing_loci)
    merged: list[tuple[int, int]] = []
    for s, e in locus_intervals:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    gaps: list[tuple[int, int]] = []
    prev_end = 0
    for s, e in merged:
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < chrom_length:
        gaps.append((prev_end, chrom_length))

    locus_stranded = sorted(
        (loc.start, loc.end, loc.strand) for loc in existing_loci
    )

    def _overlaps(isl_start: int, isl_end: int, isl_strand: str) -> bool:
        for ls, le, lst in locus_stranded:
            if ls >= isl_end:
                break
            if le <= isl_start:
                continue
            if isl_strand == ".":
                return True
            if lst != "." and lst != isl_strand:
                return True
        return False

    records: list[TranscriptRecord] = []
    gene_counter = gene_counter_start

    try:
        with _pysam.AlignmentFile(cfg.bam_path, "rb") as af:
            window_size = 500
            chunk_bp = 1_000_000

            for gap_start, gap_end in gaps:
                try:
                    n_reads = af.count(chrom, gap_start, gap_end)
                except (ValueError, KeyError):
                    continue
                if n_reads < min_se_coverage:
                    continue

                in_island = False
                island_start = 0
                island_cov_sum = 0.0
                island_cov_sq_sum = 0.0
                island_cov_count = 0

                for chunk_start in range(gap_start, gap_end, chunk_bp):
                    chunk_end = min(chunk_start + chunk_bp, gap_end)
                    try:
                        acgt = af.count_coverage(
                            chrom, chunk_start, chunk_end,
                            quality_threshold=0,
                        )
                        total_cov = np.zeros(
                            chunk_end - chunk_start, dtype=np.int32,
                        )
                        for arr in acgt:
                            total_cov += np.array(arr, dtype=np.int32)
                    except (ValueError, KeyError):
                        continue

                    for offset in range(0, len(total_cov), window_size):
                        bin_end = min(offset + window_size, len(total_cov))
                        bin_slice = total_cov[offset:bin_end]
                        bin_cov = float(np.mean(bin_slice))
                        abs_start = chunk_start + offset

                        if bin_cov >= min_se_coverage:
                            if not in_island:
                                island_start = abs_start
                                island_cov_sum = 0.0
                                island_cov_sq_sum = 0.0
                                island_cov_count = 0
                                in_island = True
                            n_bp = bin_end - offset
                            island_cov_sum += float(np.sum(bin_slice))
                            island_cov_sq_sum += float(
                                np.sum(bin_slice.astype(np.float64) ** 2),
                            )
                            island_cov_count += n_bp
                        else:
                            if in_island:
                                island_end = abs_start
                                island_length = island_end - island_start
                                if (
                                    island_length >= min_se_length
                                    and island_cov_count > 0
                                ):
                                    avg_cov = island_cov_sum / island_cov_count
                                    variance = (
                                        island_cov_sq_sum / island_cov_count
                                        - avg_cov ** 2
                                    )
                                    std_cov = math.sqrt(max(0.0, variance))
                                    cv = std_cov / avg_cov if avg_cov > 0 else 0.0
                                    if cv <= min_se_coverage_uniformity:
                                        strand = _infer_strand_static(
                                            af, chrom, island_start, island_end,
                                        )
                                        if not _overlaps(
                                            island_start, island_end, strand,
                                        ):
                                            gene_counter += 1
                                            gid = f"RSPG{chrom}.{gene_counter}"
                                            records.append(TranscriptRecord(
                                                transcript_id=f"{gid}.1",
                                                gene_id=gid,
                                                chrom=chrom,
                                                strand=strand,
                                                start=island_start,
                                                end=island_end,
                                                exons=[(island_start, island_end)],
                                                score=0.6,
                                                coverage=avg_cov,
                                            ))
                                in_island = False

                # Flush last island.
                if in_island:
                    island_end = gap_end
                    island_length = island_end - island_start
                    if island_length >= min_se_length and island_cov_count > 0:
                        avg_cov = island_cov_sum / island_cov_count
                        variance = (
                            island_cov_sq_sum / island_cov_count
                            - avg_cov ** 2
                        )
                        std_cov = math.sqrt(max(0.0, variance))
                        cv = std_cov / avg_cov if avg_cov > 0 else 0.0
                        if cv <= min_se_coverage_uniformity:
                            strand = _infer_strand_static(
                                af, chrom, island_start, island_end,
                            )
                            if not _overlaps(island_start, island_end, strand):
                                gene_counter += 1
                                gid = f"RSPG{chrom}.{gene_counter}"
                                records.append(TranscriptRecord(
                                    transcript_id=f"{gid}.1",
                                    gene_id=gid,
                                    chrom=chrom,
                                    strand=strand,
                                    start=island_start,
                                    end=island_end,
                                    exons=[(island_start, island_end)],
                                    score=0.6,
                                    coverage=avg_cov,
                                ))
    except Exception as exc:
        logger.warning(
            "Single-exon detection failed on %s: %s", chrom, exc,
        )

    return records


def _infer_strand_static(
    af: object,
    chrom: str,
    start: int,
    end: int,
) -> str:
    """Infer strand from read orientations (standalone version).

    Args:
        af: Open pysam AlignmentFile.
        chrom: Chromosome name.
        start: Region start.
        end: Region end.

    Returns:
        ``'+'``, ``'-'``, or ``'.'``.
    """
    fwd = 0
    rev = 0
    count = 0
    for read in af.fetch(chrom, start, end):
        if read.is_unmapped or read.is_secondary or read.is_supplementary:
            continue
        count += 1
        if count > 200:
            break
        try:
            xs = read.get_tag("XS")
            if xs == "+":
                fwd += 1
            elif xs == "-":
                rev += 1
            continue
        except KeyError:
            pass
        if read.is_reverse:
            rev += 1
        else:
            fwd += 1
    if fwd > rev * 2:
        return "+"
    if rev > fwd * 2:
        return "-"
    return "+"  # default to forward strand when ambiguous


class AssemblyPipeline:
    """Orchestrates the full BRAID transcript assembly workflow.

    The pipeline proceeds through the following stages for each chromosome:

    1. Fetch aligned reads from the BAM file.
    2. Extract splice junctions from CIGAR strings.
    3. Build splice graphs (one per gene locus).
    4. Compute per-node and per-edge read coverage.
    5. Extract paired-end phasing constraints.
    6. Decompose graphs into candidate transcripts via flow algorithms.
    7. Score transcripts using an ML model or heuristic fallback.
    8. Filter low-confidence and redundant transcripts.

    After all chromosomes are processed, FPKM/TPM expression values are
    computed globally and the output annotation file is written.

    Args:
        config: Pipeline configuration controlling all parameters.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config: PipelineConfig = config
        self._stats: AssemblyStats = AssemblyStats()
        self._stats_lock = threading.Lock()
        self._bam_reader: BamReader | None = None
        self._reference: ReferenceGenome | None = None
        self._scorer: TranscriptScorer | None = None
        self._graph_builder: SpliceGraphBuilder | None = None
        self._transcript_filter: TranscriptFilter | None = None
        self._neural_decomposer: object | None = None
        self._junction_scorer: object | None = None
        self._boundary_detector: object | None = None
        self._batch_processor: object | None = None
        self._diagnostics: DiagnosticsCollector | None = None
        self._diagnostics_finalized = False
        if config.diagnostics_dir:
            self._diagnostics = DiagnosticsCollector(config.diagnostics_dir)
        self._setup_logging()

    def run(self) -> str:
        """Run the full assembly pipeline and return the output file path.

        Executes all stages end-to-end: backend initialisation, BAM reading,
        per-chromosome graph construction and decomposition, scoring,
        filtering, expression quantification, and annotation output.

        Returns:
            The filesystem path to the written output file (GTF or GFF3).

        Raises:
            FileNotFoundError: If the BAM file or reference FASTA cannot be
                found.
            RuntimeError: If the pipeline encounters an unrecoverable error
                during assembly.
        """
        pipeline_start = time.perf_counter()
        cfg = self._config

        try:
            # ----- Step 1: Initialise backend --------------------------------
            with Timer("Backend initialisation", self._stats):
                set_backend(cfg.backend, threads=cfg.threads)
                logger.info("Active backend: %s", get_backend())

                # Initialise BatchProcessor for GPU-accelerated coverage.
                if get_backend() == "gpu":
                    from braid.cuda.batch import BatchProcessor
                    self._batch_processor = BatchProcessor(use_gpu=True)
                    logger.info("GPU BatchProcessor enabled")

            # ----- Step 2: Open BAM reader -----------------------------------
            with Timer("BAM reader initialisation", self._stats):
                self._bam_reader = BamReader(
                    bam_path=cfg.bam_path,
                    min_mapq=cfg.min_mapq,
                )
                self._stats.total_reads = self._bam_reader.count_reads()
                self._stats.mapped_reads = self._stats.total_reads
                logger.info(
                    "BAM contains %d mapped reads across %d references",
                    self._stats.mapped_reads,
                    len(self._bam_reader.chromosomes),
                )

            # ----- Step 3: Open reference genome (optional) ------------------
            if cfg.reference_path is not None:
                with Timer("Reference genome loading", self._stats):
                    self._reference = ReferenceGenome(cfg.reference_path)
                    logger.info("Reference genome loaded: %s", cfg.reference_path)

            # ----- Initialise sub-components ---------------------------------
            builder_profile = _effective_builder_profile(cfg)
            if builder_profile != "default":
                logger.info("Builder profile enabled: %s", builder_profile)
            self._graph_builder = SpliceGraphBuilder(
                config=_make_builder_config(cfg),
            )

            model_path = cfg.model_path if cfg.use_ml_scoring else None
            self._scorer = TranscriptScorer(model_path=model_path)
            logger.info("Transcript scoring mode: %s", self._scorer.mode)

            self._transcript_filter = TranscriptFilter(
                config=FilterConfig(
                    min_score=cfg.min_transcript_score,
                    min_coverage=cfg.min_coverage,
                    min_junction_support=cfg.min_junction_support,
                ),
            )

            # ----- ML module initialization (optional) -----------------------
            if cfg.use_neural_decomposition:
                from braid.flow.neural_decompose import NeuralDecomposer
                self._neural_decomposer = NeuralDecomposer(
                    model_path=cfg.neural_decomposer_path,
                )
                logger.info("Neural decomposer enabled")

            if cfg.use_junction_scoring:
                from braid.scoring.junction_scorer import JunctionScorer
                self._junction_scorer = JunctionScorer(
                    model_path=cfg.junction_scorer_path,
                )
                logger.info("Junction scorer enabled")

            if cfg.use_boundary_detection:
                from braid.graph.boundary_detector import BoundaryDetector
                self._boundary_detector = BoundaryDetector(
                    model_path=cfg.boundary_detector_path,
                )
                logger.info("Boundary detector enabled")

            # ----- Step 3c: Load guide GTF (optional) ------------------------
            if cfg.guide_gtf_path is not None:
                from braid.io.gtf_reader import read_guide_gtf
                self._guide_transcripts = read_guide_gtf(cfg.guide_gtf_path)
                logger.info(
                    "Guide GTF loaded: %d (chrom,strand) groups",
                    len(self._guide_transcripts),
                )
            else:
                self._guide_transcripts = None

            # ----- Step 4: Determine chromosomes to process ------------------
            bam_chroms = self._bam_reader.chromosomes
            if cfg.chromosomes is not None:
                target_chroms: list[str] = []
                for chrom in cfg.chromosomes:
                    if chrom in bam_chroms:
                        target_chroms.append(chrom)
                    else:
                        logger.warning(
                            "Requested chromosome %r not found in BAM header; skipping.",
                            chrom,
                        )
                if not target_chroms:
                    logger.warning("No requested chromosomes found in BAM. Exiting.")
                    self._write_empty_output()
                    return cfg.output_path
            else:
                target_chroms = list(bam_chroms)

            logger.info("Processing %d chromosome(s)", len(target_chroms))

            # ----- Step 5: Per-chromosome assembly ---------------------------
            all_records: list[TranscriptRecord] = []

            with Timer("Per-chromosome assembly", self._stats):
                if cfg.threads > 1 and len(target_chroms) > 1:
                    all_records = self._process_chromosomes_parallel(
                        target_chroms, cfg.threads,
                    )
                else:
                    for chrom in target_chroms:
                        chrom_records = self._process_chromosome(chrom)
                        all_records.extend(chrom_records)

            self._stats.assembled_transcripts = len(all_records)
            self._stats.multi_exon_transcripts = sum(
                1 for r in all_records if len(r.exons) > 1
            )
            self._stats.single_exon_transcripts = sum(
                1 for r in all_records if len(r.exons) <= 1
            )

            # ----- Step 5b: Cap oversized terminal exons --------------------
            if cfg.max_terminal_exon_length > 0:
                n_trimmed = _cap_terminal_exons(
                    all_records, cfg.max_terminal_exon_length,
                )
                if n_trimmed > 0:
                    logger.info(
                        "Capped %d oversized terminal exons (max %d bp)",
                        n_trimmed,
                        cfg.max_terminal_exon_length,
                    )

            # ----- Step 6: Compute expression values (FPKM / TPM) -----------
            with Timer("Expression quantification", self._stats):
                if all_records and self._stats.mapped_reads > 0:
                    compute_expression(
                        all_records,
                        total_mapped_reads=self._stats.mapped_reads,
                    )

            # ----- Step 7: Write output --------------------------------------
            with Timer("Output writing", self._stats):
                if cfg.output_format == "gff3":
                    writer: GtfWriter | Gff3Writer = Gff3Writer(cfg.output_path)
                else:
                    writer = GtfWriter(cfg.output_path)
                writer.write_transcripts(all_records)
                logger.info(
                    "Wrote %d transcripts to %s", len(all_records), cfg.output_path,
                )

            # ----- Step 8: Optional AS event detection ------------------------
            if cfg.detect_splicing_events and all_records:
                self._run_splicing_analysis(all_records, target_chroms)

            # ----- Step 9: Summary statistics --------------------------------
            self._stats.elapsed_seconds = time.perf_counter() - pipeline_start
            logger.info("\n%s", self._stats.summary())
            return cfg.output_path
        finally:
            self._stats.elapsed_seconds = time.perf_counter() - pipeline_start
            self._finalize_diagnostics()
            if self._reference is not None:
                self._reference.close()

    def _process_chromosome(self, chrom: str) -> list[TranscriptRecord]:
        """Process a single chromosome through the assembly pipeline.

        Uses a two-pass approach for memory efficiency:
        1. Lightweight BAM scan for junctions only (no full read loading).
        2. Per-locus BAM fetch for graph building and decomposition.

        Args:
            chrom: Name of the chromosome / reference sequence to process.

        Returns:
            List of ``TranscriptRecord`` objects assembled from this
            chromosome.
        """
        assert self._bam_reader is not None
        assert self._graph_builder is not None
        assert self._scorer is not None
        assert self._transcript_filter is not None

        cfg = self._config
        chrom_length = self._bam_reader.chromosome_lengths.get(chrom, 0)

        logger.info("Processing chromosome %s (length %d)", chrom, chrom_length)
        chrom_stage_timings: dict[str, float] = {}

        # --- Pass 1: Lightweight junction extraction from BAM ---
        # Pass reference for integrated motif validation during extraction.
        stage_start = time.perf_counter()
        junctions, n_spliced, extraction_stats = extract_junctions_from_bam(
            cfg.bam_path, chrom, min_mapq=cfg.min_mapq,
            min_anchor_length=cfg.min_anchor_length,
            reference=self._reference if cfg.enable_motif_validation else None,
            return_stats=True,
            strandedness=cfg.strandedness,
        )
        self._record_stage_timing(
            chrom_stage_timings, "junction_extraction", stage_start,
        )
        raw_junctions = extraction_stats.raw_junctions
        anchor_filtered_junctions = extraction_stats.anchor_filtered_junctions
        motif_filtered_junctions = extraction_stats.motif_filtered_junctions
        n_junctions = extraction_stats.output_junctions
        with self._stats_lock:
            self._stats.spliced_reads += n_spliced
            self._stats.total_junctions += n_junctions
            self._stats.unique_junctions += n_junctions
        logger.info("Extracted %d unique junctions on %s", n_junctions, chrom)

        # --- Depth-adaptive junction filtering ---
        # Apply a global filter proportional to sequencing depth,
        # capped at 10. For deep data (e.g. 50M spliced reads),
        # adaptive_min = min(10, sqrt(50M/100K)) = 10.
        stage_start = time.perf_counter()
        if (
            cfg.adaptive_junction_filter
            and
            _effective_builder_profile(cfg) == "default"
            and n_spliced > 0
            and n_junctions > 0
        ):
            adaptive_min = max(
                cfg.min_junction_support,
                min(5, int(math.sqrt(n_spliced / 100_000))),
            )
            base_min = (
                self._graph_builder.config.min_junction_support
                if self._graph_builder is not None
                else cfg.min_junction_support
            )
            if adaptive_min > base_min:
                logger.info(
                    "Depth-adaptive junction filter: %d (from %d spliced reads on %s)",
                    adaptive_min, n_spliced, chrom,
                )
                keep = junctions.counts >= adaptive_min
                junctions = type(junctions)(
                    chrom=junctions.chrom,
                    starts=junctions.starts[keep],
                    ends=junctions.ends[keep],
                    counts=junctions.counts[keep],
                    strands=junctions.strands[keep],
                )
                n_junctions = len(junctions.starts)
                logger.info(
                    "After adaptive filter: %d junctions on %s",
                    n_junctions, chrom,
                )
        self._record_stage_timing(
            chrom_stage_timings, "adaptive_junction_filter", stage_start,
        )

        # --- Identify loci from junctions ---
        stage_start = time.perf_counter()
        if n_junctions > 0:
            loci = self._graph_builder.identify_loci(
                junctions, chrom_length,
            )
        else:
            loci = []
        self._record_stage_timing(
            chrom_stage_timings, "identify_loci", stage_start,
        )
        n_loci = len(loci)
        with self._stats_lock:
            self._stats.total_loci += n_loci
        logger.info("Identified %d loci on %s", n_loci, chrom)

        # --- Pass 2: Per-locus read fetch, graph building, decomposition ---
        all_chrom_records: list[TranscriptRecord] = []
        gene_counter = 0
        has_paired_reads: bool | None = None  # detect lazily

        for locus in loci:
            locus_diag = LocusDiagnostics(
                chrom=chrom,
                start=locus.start,
                end=locus.end,
                strand=locus.strand,
                raw_junctions=len(locus.junction_indices),
                decomposition_method="nmf" if cfg.use_nmf_decomposition else cfg.decomposer,
                scorer_mode=self._scorer.mode,
            )

            # Fetch reads only for this locus
            stage_start = time.perf_counter()
            read_data = self._bam_reader.fetch_region(
                chrom, locus.start, locus.end,
            )
            self._record_stage_timing(
                locus_diag.stage_timings, "read_fetch", stage_start,
            )
            locus_diag.n_reads = read_data.n_reads
            if read_data.n_reads == 0:
                locus_diag.skipped_reason = "no_reads"
                self._record_locus_diagnostics(locus_diag)
                continue

            # Lazy paired-end detection (once per chromosome)
            if has_paired_reads is None:
                if hasattr(read_data, 'is_paired') and read_data.is_paired is not None:
                    has_paired_reads = bool(np.any(read_data.is_paired))
                else:
                    sample_size = min(200, read_data.n_reads)
                    sample_names = read_data.query_names[:sample_size]
                    has_paired_reads = len(set(sample_names)) < len(sample_names)

            # Build graph for this locus
            stage_start = time.perf_counter()
            graph = self._graph_builder.build_graph(locus, read_data, junctions)
            self._record_stage_timing(
                locus_diag.stage_timings, "graph_build", stage_start,
            )
            if graph is None:
                locus_diag.skipped_reason = "graph_builder_returned_none"
                self._record_locus_diagnostics(locus_diag)
                continue
            locus_diag.graph_built = True
            locus_diag.graph_nodes = graph.n_nodes
            locus_diag.graph_edges = graph.n_edges
            graph_diag = getattr(graph, "runtime_diagnostics", {})
            locus_diag.fallback_source_edges_added = int(
                graph_diag.get("fallback_source_edges_added", 0),
            )
            locus_diag.fallback_sink_edges_added = int(
                graph_diag.get("fallback_sink_edges_added", 0),
            )

            # Phasing (only for paired-end data)
            phasing_for_decomp: list[tuple[list[int], float]] | None = None
            raw_phasing_paths = 0
            stage_start = time.perf_counter()
            if has_paired_reads:
                chrom_id = int(read_data.chrom_ids[0]) if read_data.n_reads > 0 else -1
                phasing = extract_phasing_paths(graph, read_data, chrom_id)
                raw_phasing_paths = phasing.n_paths
                if phasing.n_paths > 0:
                    valid_phasing = apply_phasing_constraints(
                        graph,
                        phasing,
                        min_support=cfg.min_phasing_support,
                    )
                    phasing_for_decomp = [
                        (p.node_ids, p.weight) for p in valid_phasing
                    ]
                    if not phasing_for_decomp:
                        phasing_for_decomp = None
            self._record_stage_timing(
                locus_diag.stage_timings, "phasing", stage_start,
            )

            # Decompose graph into transcripts
            stage_start = time.perf_counter()
            graph_csr = graph.to_csr()
            phasing_for_decomp, _remapped_phasing_drops = self._remap_phasing_paths_to_csr(
                graph, phasing_for_decomp,
            )
            locus_diag.phasing_paths = len(phasing_for_decomp or [])
            locus_diag.phasing_paths_dropped = max(
                0, raw_phasing_paths - locus_diag.phasing_paths,
            )

            if cfg.use_nmf_decomposition:
                # NMF-based decomposition: use read-fragment matrix
                read_junc_starts, read_junc_ends = _extract_per_read_junctions(
                    read_data,
                )
                nmf_config = NMFDecomposeConfig(
                    min_transcript_coverage=cfg.min_coverage,
                )
                transcripts = decompose_nmf(
                    graph_csr, graph,
                    read_positions=read_data.positions,
                    read_ends=read_data.end_positions,
                    read_junction_starts=read_junc_starts,
                    read_junction_ends=read_junc_ends,
                    config=nmf_config,
                )
            else:
                # Map long-read guide paths for this locus
                locus_guide_paths: list[list[int]] | None = None
                if self._guide_transcripts is not None:
                    strand = graph.strand if graph.strand in ("+", "-") else "+"
                    lr_key = (chrom, strand)
                    lr_txs = self._guide_transcripts.get(lr_key, [])
                    if lr_txs:
                        locus_guide_paths = get_guide_paths_for_locus(
                            lr_txs, graph_csr,
                            locus.start, locus.end,
                            tolerance=cfg.guide_tolerance,
                        ) or None

                decompose_config = DecomposeConfig(
                    min_transcript_coverage=cfg.min_coverage,
                    use_safe_paths=cfg.use_safe_paths,
                    max_paths=cfg.max_paths,
                )
                primary_run, shadow_run = run_decomposer_pair(
                    graph_csr,
                    graph,
                    config=decompose_config,
                    phasing_paths=phasing_for_decomp,
                    guide_paths=locus_guide_paths,
                    mode=cfg.decomposer,
                    shadow_mode=cfg.shadow_decomposer,
                )
                transcripts = primary_run.transcripts
                locus_diag.decomposition_method = primary_run.metadata.label
                locus_diag.decomposition_metrics = dict(primary_run.metadata.metrics)
                if shadow_run is not None:
                    locus_diag.shadow_decomposition_method = shadow_run.metadata.label
                    locus_diag.shadow_decomposition_metrics = dict(
                        shadow_run.metadata.metrics,
                    )
            self._record_stage_timing(
                locus_diag.stage_timings, "decomposition", stage_start,
            )

            locus_diag.candidates_before_merge = len(transcripts)
            if not transcripts:
                locus_diag.skipped_reason = "no_candidate_transcripts"
                self._record_locus_diagnostics(locus_diag)
                continue

            # Post-decomposition intron chain merging: consolidate
            # transcripts with identical splicing patterns but different
            # terminal exon boundaries.
            stage_start = time.perf_counter()
            if len(transcripts) > 1:
                transcripts = self._transcript_filter.merge_identical_intron_chains(
                    transcripts,
                )
            self._record_stage_timing(
                locus_diag.stage_timings, "merge_intron_chains", stage_start,
            )
            locus_diag.candidates_after_merge = len(transcripts)

            if not cfg.use_nmf_decomposition and shadow_run is not None:
                stage_start = time.perf_counter()
                shadow_transcripts = list(shadow_run.transcripts)
                locus_diag.shadow_candidates_before_merge = len(shadow_transcripts)
                if len(shadow_transcripts) > 1:
                    shadow_transcripts = self._transcript_filter.merge_identical_intron_chains(
                        shadow_transcripts,
                    )
                locus_diag.shadow_candidates_after_merge = len(shadow_transcripts)
                if shadow_transcripts:
                    shadow_surviving, _, shadow_filter_diagnostics = (
                        self._score_and_filter_transcripts(
                            shadow_transcripts,
                            graph,
                            junctions,
                        )
                    )
                    locus_diag.shadow_surviving_transcripts = len(shadow_surviving)
                    locus_diag.shadow_filter_diagnostics = asdict(
                        shadow_filter_diagnostics,
                    )
                self._record_stage_timing(
                    locus_diag.stage_timings, "shadow_score_filter", stage_start,
                )

            stage_start = time.perf_counter()
            surviving_transcripts, surviving_scores, filter_diagnostics = (
                self._score_and_filter_transcripts(transcripts, graph, junctions)
            )
            self._record_stage_timing(
                locus_diag.stage_timings, "score_filter", stage_start,
            )
            locus_diag.filter_diagnostics = asdict(filter_diagnostics)
            locus_diag.surviving_transcripts = len(surviving_transcripts)
            if not surviving_transcripts:
                locus_diag.skipped_reason = "filtered_out"

            n_filtered_out = len(transcripts) - len(surviving_transcripts)
            with self._stats_lock:
                self._stats.filtered_transcripts += n_filtered_out

            if surviving_transcripts:
                # Bootstrap confidence intervals
                bootstrap_data: list[dict[str, float]] | None = None
                if cfg.enable_bootstrap and len(surviving_transcripts) > 0:
                    paths = [tx.node_ids for tx in surviving_transcripts]
                    try:
                        boot_cfg = BootstrapConfig(
                            n_replicates=cfg.bootstrap_replicates,
                            seed=42,
                        )
                        boot_result = bootstrap_confidence(graph_csr, paths, boot_cfg)
                        bootstrap_data = [
                            {
                                "ci_low": tc.weight_ci_low,
                                "ci_high": tc.weight_ci_high,
                                "presence": tc.presence_rate,
                                "cv": tc.cv,
                            }
                            for tc in boot_result.transcripts
                        ]
                    except Exception as exc:
                        logger.debug("Bootstrap failed for locus: %s", exc)

                gene_counter += 1
                gene_prefix = f"RSPG{chrom}.{gene_counter}"
                stage_start = time.perf_counter()
                records = self._transcripts_to_records(
                    surviving_transcripts,
                    surviving_scores,
                    graph,
                    gene_prefix,
                    bootstrap_data=bootstrap_data,
                )
                self._record_stage_timing(
                    locus_diag.stage_timings, "record_conversion", stage_start,
                )
                all_chrom_records.extend(records)

            self._record_locus_diagnostics(locus_diag)

        # --- Single-exon gene detection ---
        # Scan for high-coverage regions without junctions to recover
        # single-exon genes that the junction-centric pipeline misses.
        stage_start = time.perf_counter()
        if not self._config.disable_single_exon:
            se_records = self._detect_single_exon_genes(
                chrom, chrom_length, loci, gene_counter,
            )
        else:
            se_records = []
        self._record_stage_timing(
            chrom_stage_timings, "single_exon_detection", stage_start,
        )
        gene_counter += len(se_records)
        all_chrom_records.extend(se_records)

        if self._diagnostics is not None:
            self._diagnostics.record_chromosome(
                ChromosomeDiagnostics(
                    chrom=chrom,
                    spliced_reads=n_spliced,
                    raw_junctions=raw_junctions,
                    anchor_filtered_junctions=anchor_filtered_junctions,
                    motif_filtered_junctions=motif_filtered_junctions,
                    filtered_junctions=n_junctions,
                    n_loci=n_loci,
                    stage_timings=chrom_stage_timings,
                )
            )

        logger.info(
            "Chromosome %s: %d transcripts from %d loci (%d single-exon)",
            chrom, len(all_chrom_records), n_loci, len(se_records),
        )
        return all_chrom_records

    def _process_chromosomes_parallel(
        self,
        chroms: list[str],
        n_workers: int,
    ) -> list[TranscriptRecord]:
        """Process chromosomes in parallel using a process pool.

        Uses ``ProcessPoolExecutor`` so each worker operates in its own
        process with an independent BAM file handle, avoiding the I/O
        contention that cripples ``ThreadPoolExecutor`` on large BAM files.

        Args:
            chroms: List of chromosome names to process.
            n_workers: Number of parallel worker processes.

        Returns:
            Aggregated list of ``TranscriptRecord`` from all chromosomes.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        all_records: list[TranscriptRecord] = []

        # Use the fork-safe 'spawn' start method to avoid issues with
        # pysam file handles across forked processes.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx,
        ) as pool:
            failed_chroms: list[str] = []
            futures = {
                pool.submit(
                    _process_chromosome_worker, self._config, chrom,
                ): chrom
                for chrom in chroms
            }
            for future in as_completed(futures):
                chrom = futures[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                except Exception:
                    logger.exception(
                        "Error processing chromosome %s", chrom,
                    )
                    failed_chroms.append(chrom)

        if failed_chroms:
            failed = ", ".join(sorted(set(failed_chroms)))
            logger.warning(
                "Failed to process %d chromosome(s): %s",
                len(failed_chroms), failed,
            )
            # Continue with successful chromosomes instead of aborting
            if not all_records:
                raise RuntimeError(
                    f"All chromosomes failed: {failed}"
                )

        return all_records

    def _record_locus_diagnostics(self, record: LocusDiagnostics) -> None:
        """Write a locus-level diagnostic record if instrumentation is enabled."""
        if self._diagnostics is not None:
            self._diagnostics.record_locus(record)

    @staticmethod
    def _record_stage_timing(
        stage_timings: dict[str, float],
        stage_name: str,
        stage_start: float,
    ) -> None:
        """Store a rounded elapsed time for a named stage."""
        stage_timings[stage_name] = round(time.perf_counter() - stage_start, 6)

    def _finalize_diagnostics(self) -> None:
        """Flush diagnostics exactly once per pipeline run."""
        if self._diagnostics is None or self._diagnostics_finalized:
            return
        self._diagnostics.finalize(self._stats)
        self._diagnostics_finalized = True

    def _score_and_filter_transcripts(
        self,
        transcripts: list[Transcript],
        graph: SpliceGraph,
        junctions: object,
    ) -> tuple[list[Transcript], list[float], object]:
        """Score transcripts and return surviving transcripts plus scores."""
        assert self._scorer is not None
        assert self._transcript_filter is not None

        scores_list: list[float] = []
        features_list = []
        for tx in transcripts:
            feat = extract_features(tx, graph, transcripts, junctions)
            features_list.append(feat)
            feat_array = features_to_array(feat)
            score = self._scorer.score(feat_array)
            scores_list.append(score)

        scores_array = np.array(scores_list, dtype=np.float64)
        surviving_indices, filter_diagnostics = (
            self._transcript_filter.filter_transcripts_with_diagnostics(
                transcripts, scores_array, features_list,
            )
        )
        surviving_transcripts = [transcripts[i] for i in surviving_indices]
        surviving_scores = [scores_list[i] for i in surviving_indices]
        return surviving_transcripts, surviving_scores, filter_diagnostics

    @staticmethod
    def _remap_phasing_paths_to_csr(
        graph: SpliceGraph,
        phasing_paths: list[tuple[list[int], float]] | None,
    ) -> tuple[list[tuple[list[int], float]] | None, int]:
        """Convert graph-node phasing paths to CSR node ids."""
        if not phasing_paths:
            return None, 0

        topo = graph.topological_order()
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(topo)}
        remapped: list[tuple[list[int], float]] = []
        dropped = 0
        for node_ids, weight in phasing_paths:
            try:
                remapped_ids = [old_to_new[node_id] for node_id in node_ids]
            except KeyError:
                dropped += 1
                continue
            remapped.append((remapped_ids, weight))
        return remapped or None, dropped

    def _transcripts_to_records(
        self,
        transcripts: list[Transcript],
        scores: list[float],
        graph: SpliceGraph,
        gene_prefix: str,
        bootstrap_data: list[dict[str, float]] | None = None,
    ) -> list[TranscriptRecord]:
        """Convert internal Transcript objects to TranscriptRecord for output.

        Constructs ``TranscriptRecord`` instances suitable for GTF/GFF3
        writing from the flow-decomposition ``Transcript`` objects, their
        quality scores, and the parent splice graph metadata.

        Args:
            transcripts: List of assembled ``Transcript`` objects from flow
                decomposition.
            scores: Per-transcript quality scores, same length and order as
                *transcripts*.
            graph: The splice graph from which the transcripts were derived
                (provides chromosome, strand, and locus coordinates).
            gene_prefix: A string prefix used to generate unique gene and
                transcript identifiers (e.g. ``"RSPGchr1.5"``).
            bootstrap_data: Optional per-transcript bootstrap CI statistics.

        Returns:
            List of ``TranscriptRecord`` objects ready for annotation output.
        """
        records: list[TranscriptRecord] = []
        strand = graph.strand if graph.strand in ("+", "-") else "+"

        for tx_idx, (tx, score) in enumerate(zip(transcripts, scores), start=1):
            exons = tx.exon_coords
            if not exons:
                continue

            transcript_id = f"{gene_prefix}.{tx_idx}"
            gene_id = gene_prefix

            tx_start = exons[0][0]
            tx_end = exons[-1][1]

            # Compute average coverage from weight (abundance estimate)
            coverage = tx.weight

            record = TranscriptRecord(
                transcript_id=transcript_id,
                gene_id=gene_id,
                chrom=graph.chrom,
                strand=strand,
                start=tx_start,
                end=tx_end,
                exons=list(exons),
                score=score,
                coverage=coverage,
            )
            if bootstrap_data and (tx_idx - 1) < len(bootstrap_data):
                br = bootstrap_data[tx_idx - 1]
                record.bootstrap_ci_low = br["ci_low"]
                record.bootstrap_ci_high = br["ci_high"]
                record.bootstrap_presence = br["presence"]
                record.bootstrap_cv = br["cv"]
            records.append(record)

        return records

    def _detect_single_exon_genes(
        self,
        chrom: str,
        chrom_length: int,
        existing_loci: list,
        gene_counter_start: int,
    ) -> list[TranscriptRecord]:
        """Detect single-exon genes from high-coverage regions without junctions.

        Only scans gaps between existing junction-based loci to avoid redundant
        BAM traversal.  Regions above the minimum coverage and length thresholds
        are emitted as single-exon transcripts.

        Args:
            chrom: Chromosome name.
            chrom_length: Length of the chromosome.
            existing_loci: List of ``LocusDefinition`` objects from junction-based
                assembly (used to exclude already-assembled regions).
            gene_counter_start: Gene counter offset for unique IDs.

        Returns:
            List of single-exon ``TranscriptRecord`` objects.
        """
        assert self._bam_reader is not None

        cfg = self._config
        min_se_coverage = max(50.0, cfg.min_coverage * 20)
        min_se_length = 500
        min_se_coverage_uniformity = 1.0  # max coefficient of variation

        # Build sorted, merged intervals of existing loci.
        locus_intervals: list[tuple[int, int]] = []
        for loc in existing_loci:
            locus_intervals.append((loc.start, loc.end))
        locus_intervals.sort()

        # Merge overlapping loci to compute gap regions efficiently.
        merged: list[tuple[int, int]] = []
        for s, e in locus_intervals:
            if merged and s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        # Compute gap regions (intervals NOT covered by any locus).
        gaps: list[tuple[int, int]] = []
        prev_end = 0
        for s, e in merged:
            if s > prev_end:
                gaps.append((prev_end, s))
            prev_end = max(prev_end, e)
        if prev_end < chrom_length:
            gaps.append((prev_end, chrom_length))

        # Build sorted list of (start, end, strand) for antisense filtering.
        locus_stranded: list[tuple[int, int, str]] = sorted(
            (loc.start, loc.end, loc.strand) for loc in existing_loci
        )

        def _overlaps_existing_locus(
            isl_start: int, isl_end: int, isl_strand: str,
        ) -> bool:
            """Return True if the single-exon island should be skipped.

            Skip if: (a) strand is ambiguous ('.') and any locus overlaps, or
            (b) a locus on the opposite strand overlaps.
            """
            for ls, le, lst in locus_stranded:
                if ls >= isl_end:
                    break  # sorted — no further overlap possible
                if le <= isl_start:
                    continue
                # Overlap exists.
                if isl_strand == ".":
                    return True  # ambiguous → likely coverage bleed
                if lst != "." and lst != isl_strand:
                    return True  # opposite strand → antisense noise
            return False

        import pysam

        records: list[TranscriptRecord] = []
        gene_counter = gene_counter_start

        try:
            with pysam.AlignmentFile(cfg.bam_path, "rb") as af:
                window_size = 500
                chunk_bp = 1_000_000  # 1 Mb chunks

                for gap_start, gap_end in gaps:
                    # Quick check: skip gaps with no reads.
                    try:
                        n_reads = af.count(chrom, gap_start, gap_end)
                    except (ValueError, KeyError):
                        continue
                    if n_reads < min_se_coverage:
                        continue

                    in_island = False
                    island_start = 0
                    island_cov_sum = 0.0
                    island_cov_sq_sum = 0.0
                    island_cov_count = 0

                    for chunk_start in range(gap_start, gap_end, chunk_bp):
                        chunk_end = min(chunk_start + chunk_bp, gap_end)
                        try:
                            acgt = af.count_coverage(
                                chrom, chunk_start, chunk_end,
                                quality_threshold=0,
                            )
                            total_cov = np.zeros(
                                chunk_end - chunk_start, dtype=np.int32,
                            )
                            for arr in acgt:
                                total_cov += np.array(arr, dtype=np.int32)
                        except (ValueError, KeyError):
                            continue

                        for offset in range(0, len(total_cov), window_size):
                            bin_end = min(offset + window_size, len(total_cov))
                            bin_slice = total_cov[offset:bin_end]
                            bin_cov = float(np.mean(bin_slice))
                            abs_start = chunk_start + offset

                            if bin_cov >= min_se_coverage:
                                if not in_island:
                                    island_start = abs_start
                                    island_cov_sum = 0.0
                                    island_cov_sq_sum = 0.0
                                    island_cov_count = 0
                                    in_island = True
                                n_bp = bin_end - offset
                                island_cov_sum += float(np.sum(bin_slice))
                                island_cov_sq_sum += float(
                                    np.sum(bin_slice.astype(np.float64) ** 2),
                                )
                                island_cov_count += n_bp
                            else:
                                if in_island:
                                    island_end = abs_start
                                    island_length = island_end - island_start
                                    if (
                                        island_length >= min_se_length
                                        and island_cov_count > 0
                                    ):
                                        avg_cov = (
                                            island_cov_sum / island_cov_count
                                        )
                                        # Coverage uniformity check (CV).
                                        variance = (
                                            island_cov_sq_sum / island_cov_count
                                            - avg_cov ** 2
                                        )
                                        std_cov = math.sqrt(max(0.0, variance))
                                        cv = (
                                            std_cov / avg_cov
                                            if avg_cov > 0
                                            else 0.0
                                        )
                                        if cv > min_se_coverage_uniformity:
                                            in_island = False
                                            continue
                                        strand = self._infer_strand_from_reads(
                                            af, chrom, island_start, island_end,
                                        )
                                        # Antisense / ambiguous-strand filter.
                                        if _overlaps_existing_locus(
                                            island_start, island_end, strand,
                                        ):
                                            in_island = False
                                            continue
                                        gene_counter += 1
                                        gid = f"RSPG{chrom}.{gene_counter}"
                                        tid = f"{gid}.1"
                                        records.append(TranscriptRecord(
                                            transcript_id=tid,
                                            gene_id=gid,
                                            chrom=chrom,
                                            strand=strand,
                                            start=island_start,
                                            end=island_end,
                                            exons=[(island_start, island_end)],
                                            score=0.6,
                                            coverage=avg_cov,
                                        ))
                                    in_island = False

                    # Flush last island in this gap.
                    if in_island:
                        island_end = gap_end
                        island_length = island_end - island_start
                        if (
                            island_length >= min_se_length
                            and island_cov_count > 0
                        ):
                            avg_cov = island_cov_sum / island_cov_count
                            # Coverage uniformity check (CV).
                            variance = (
                                island_cov_sq_sum / island_cov_count
                                - avg_cov ** 2
                            )
                            std_cov = math.sqrt(max(0.0, variance))
                            cv = std_cov / avg_cov if avg_cov > 0 else 0.0
                            if cv <= min_se_coverage_uniformity:
                                strand = self._infer_strand_from_reads(
                                    af, chrom, island_start, island_end,
                                )
                                # Antisense / ambiguous-strand filter.
                                if not _overlaps_existing_locus(
                                    island_start, island_end, strand,
                                ):
                                    gene_counter += 1
                                    gid = f"RSPG{chrom}.{gene_counter}"
                                    tid = f"{gid}.1"
                                    records.append(TranscriptRecord(
                                        transcript_id=tid,
                                        gene_id=gid,
                                        chrom=chrom,
                                        strand=strand,
                                        start=island_start,
                                        end=island_end,
                                        exons=[(island_start, island_end)],
                                        score=0.6,
                                        coverage=avg_cov,
                                    ))

        except Exception as exc:
            logger.warning(
                "Single-exon detection failed on %s: %s", chrom, exc,
            )

        if records:
            logger.info(
                "Detected %d single-exon genes on %s", len(records), chrom,
            )
        return records

    @staticmethod
    def _infer_strand_from_reads(
        af: object,
        chrom: str,
        start: int,
        end: int,
    ) -> str:
        """Infer strand of a single-exon region from read orientations.

        Uses XS tags when available, falls back to first-read-strand for
        strand-specific libraries.

        Args:
            af: Open pysam AlignmentFile.
            chrom: Chromosome name.
            start: Region start.
            end: Region end.

        Returns:
            ``'+'``, ``'-'``, or ``'.'``.
        """
        fwd = 0
        rev = 0
        count = 0
        for read in af.fetch(chrom, start, end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            count += 1
            if count > 200:
                break
            # Prefer XS tag
            try:
                xs = read.get_tag("XS")
                if xs == "+":
                    fwd += 1
                elif xs == "-":
                    rev += 1
                continue
            except KeyError:
                pass
            # Fallback: use alignment direction
            if read.is_reverse:
                rev += 1
            else:
                fwd += 1
        if fwd > rev * 2:
            return "+"
        if rev > fwd * 2:
            return "-"
        return "+"  # default to forward strand when ambiguous

    def _run_splicing_analysis(
        self,
        records: list[TranscriptRecord],
        chroms: list[str],
    ) -> None:
        """Run post-assembly alternative splicing event detection.

        Detects AS events, calculates PSI with junction evidence from the
        BAM file, and writes an events TSV alongside the GTF output.

        Args:
            records: Assembled transcript records.
            chroms: Chromosomes that were processed.
        """
        from braid.splicing.classifier import EventClassifier
        from braid.io.bam_reader import BamReader
        from braid.splicing.events import EventType, detect_all_events
        from braid.splicing.io import write_events_tsv
        from braid.splicing.psi import calculate_all_psi
        from braid.splicing.statistics import add_confidence_intervals

        cfg = self._config

        with Timer("Splicing event detection", self._stats):
            events = detect_all_events(records)
            logger.info("Detected %d AS events", len(events))

            if not events:
                return

            ri_chroms = {
                event.chrom
                for event in events
                if event.event_type == EventType.RI
            }

            # Extract junction evidence per chromosome
            je_by_chrom = {}
            for chrom in chroms:
                je_by_chrom[chrom], _ = extract_junctions_from_bam(
                    cfg.bam_path, chrom, min_mapq=cfg.min_mapq,
                    min_anchor_length=cfg.min_anchor_length,
                    reference=self._reference if cfg.enable_motif_validation else None,
                )

            read_data_by_chrom = None
            if ri_chroms:
                bam_reader = BamReader(
                    cfg.bam_path,
                    min_mapq=cfg.min_mapq,
                )
                read_data_by_chrom = {
                    chrom: bam_reader.fetch_region(chrom)
                    for chrom in sorted(ri_chroms)
                }

            psi_results = calculate_all_psi(
                events,
                je_by_chrom,
                read_data_by_chrom=read_data_by_chrom,
            )
            add_confidence_intervals(psi_results)

            # Neural PSI refinement (optional)
            if cfg.use_neural_psi:
                from braid.splicing.neural_psi import (
                    NeuralPSIEstimator,
                )
                neural_psi = NeuralPSIEstimator(
                    model_path=cfg.neural_psi_path
                )
                for psi_r in psi_results:
                    if psi_r.total_reads > 0:
                        result = neural_psi.estimate(
                            psi_r.inclusion_count,
                            psi_r.exclusion_count,
                        )
                        psi_r.ci_low = result.ci_low
                        psi_r.ci_high = result.ci_high

            # Score events
            if cfg.use_transformer_classifier:
                from braid.splicing.classifier import (
                    TransformerEventClassifier,
                )
                classifier_obj = TransformerEventClassifier(
                    model_path=cfg.transformer_classifier_path,
                )
                scores = classifier_obj.score_batch(events, psi_results)
            else:
                classifier_obj = EventClassifier()
                scores = classifier_obj.score_batch(events, psi_results)

            # Write events TSV alongside GTF
            events_path = (
                cfg.output_path.rsplit(".", 1)[0] + "_events.tsv"
            )
            write_events_tsv(events_path, events, psi_results, scores)
            logger.info("Wrote AS events to %s", events_path)

    def _write_empty_output(self) -> None:
        """Write an empty output file when no transcripts are assembled."""
        cfg = self._config
        if cfg.output_format == "gff3":
            writer: GtfWriter | Gff3Writer = Gff3Writer(cfg.output_path)
        else:
            writer = GtfWriter(cfg.output_path)
        writer.write_transcripts([])
        logger.info("Wrote empty output to %s", cfg.output_path)

    def _setup_logging(self) -> None:
        """Configure logging based on the verbose flag in the pipeline config.

        Sets the root logger level to ``DEBUG`` when verbose mode is enabled,
        or ``INFO`` otherwise.  A stream handler with a timestamped format is
        attached if the root logger does not already have handlers.
        """
        level = logging.DEBUG if self._config.verbose else logging.INFO
        root_logger = logging.getLogger("braid")
        root_logger.setLevel(level)

        if not root_logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)
