"""Target gene transcript assembler with bootstrap confidence intervals.

Performs high-resolution transcript assembly for a single gene or genomic
region, concentrating all computational resources on one locus to achieve
maximum isoform resolution with per-transcript statistical confidence.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from rapidsplice.flow.bootstrap import (
    BootstrapConfig,
    BootstrapResult,
    bootstrap_confidence,
)
from rapidsplice.flow.decompose import DecomposeConfig
from rapidsplice.flow.decomposer import run_decomposer_pair
from rapidsplice.graph.builder import GraphBuilderConfig, SpliceGraphBuilder
from rapidsplice.io.bam_reader import BamReader, extract_junctions_from_bam
from rapidsplice.io.reference import ReferenceGenome
from rapidsplice.scoring.features import extract_features, features_to_array
from rapidsplice.scoring.filter import FilterConfig, TranscriptFilter
from rapidsplice.scoring.model import TranscriptScorer
from rapidsplice.target.extractor import ExtractionStats, TargetRegion

logger = logging.getLogger(__name__)


@dataclass
class TargetConfig:
    """Configuration for targeted transcript assembly.

    Attributes:
        bam_path: Path to coordinate-sorted, indexed BAM file.
        reference_path: Path to reference genome FASTA.
        region: Target genomic region.
        flank: Flanking bp to include around the target region.
        min_mapq: Minimum mapping quality.
        min_junction_support: Minimum reads supporting a junction.
        min_coverage: Minimum transcript coverage.
        max_paths: Maximum paths to enumerate (higher = more thorough).
        bootstrap_replicates: Number of bootstrap resampling iterations.
        bootstrap_confidence: Confidence level for intervals.
        min_presence_rate: Minimum bootstrap presence to report isoform.
        min_anchor_length: Minimum splice junction anchor length.
        use_ml_scoring: Whether to use ML-based transcript scoring.
    """

    bam_path: str
    reference_path: str | None = None
    region: TargetRegion | None = None
    flank: int = 1000
    min_mapq: int = 0
    min_junction_support: int = 2
    min_coverage: float = 1.0
    max_paths: int = 5000  # Much higher for single gene
    bootstrap_replicates: int = 200
    bootstrap_confidence: float = 0.95
    min_presence_rate: float = 0.3
    min_anchor_length: int = 8
    use_ml_scoring: bool = True
    min_transcript_score: float = 0.2  # Lower threshold for targeted
    strandedness: str = "none"  # "none", "rf", or "fr"
    annotation_gtf: str | None = None  # GTF for isoform classification


@dataclass
class IsoformResult:
    """A single assembled isoform with confidence statistics.

    Attributes:
        transcript_id: Unique identifier.
        exons: List of (start, end) exon coordinates.
        strand: Genomic strand.
        weight: NNLS abundance weight.
        score: Quality score.
        n_junctions: Number of splice junctions.
        ci_low: Bootstrap confidence interval lower bound.
        ci_high: Bootstrap confidence interval upper bound.
        presence_rate: Fraction of bootstrap replicates where isoform appears.
        cv: Coefficient of variation of abundance across replicates.
        is_novel: Whether this isoform has novel junctions vs reference.
    """

    transcript_id: str
    exons: list[tuple[int, int]]
    strand: str
    weight: float
    score: float
    n_junctions: int = 0
    ci_low: float = 0.0
    ci_high: float = 0.0
    presence_rate: float = 0.0
    cv: float = 0.0
    is_novel: bool = False
    classification: str = ""
    matched_ref_id: str | None = None


@dataclass
class TargetAssemblyResult:
    """Complete result of targeted transcript assembly.

    Attributes:
        region: The target region assembled.
        isoforms: List of assembled isoforms with confidence.
        extraction_stats: Read extraction statistics.
        n_paths_enumerated: Total paths considered.
        assembly_time_seconds: Wall clock time for assembly.
        bootstrap_time_seconds: Wall clock time for bootstrap.
    """

    region: TargetRegion
    isoforms: list[IsoformResult] = field(default_factory=list)
    extraction_stats: ExtractionStats | None = None
    n_paths_enumerated: int = 0
    assembly_time_seconds: float = 0.0
    bootstrap_time_seconds: float = 0.0

    @property
    def n_isoforms(self) -> int:
        """Number of assembled isoforms."""
        return len(self.isoforms)

    @property
    def n_confident(self) -> int:
        """Number of isoforms with presence rate >= 0.5."""
        return sum(1 for iso in self.isoforms if iso.presence_rate >= 0.5)


def assemble_target(config: TargetConfig) -> TargetAssemblyResult:
    """Run targeted transcript assembly on a single gene/region.

    This is the main entry point for the TargetSplice pipeline.
    It performs read extraction, splice graph construction, NNLS
    decomposition with exhaustive path enumeration, and bootstrap
    confidence interval estimation.

    Args:
        config: Assembly configuration including BAM path, region,
            and algorithm parameters.

    Returns:
        Complete assembly result with per-isoform confidence intervals.

    Raises:
        ValueError: If no target region is specified.
    """
    if config.region is None:
        raise ValueError("No target region specified")

    region = config.region
    flanked = region.with_flank(config.flank)
    result = TargetAssemblyResult(region=region)

    t0 = time.perf_counter()

    # --- Step 1: Extract junctions ---
    reference = None
    if config.reference_path:
        reference = ReferenceGenome(config.reference_path)

    junctions, n_spliced, extraction_stats = extract_junctions_from_bam(
        config.bam_path,
        flanked.chrom,
        min_mapq=config.min_mapq,
        min_anchor_length=config.min_anchor_length,
        reference=reference,
        return_stats=True,
        region_start=flanked.start,
        region_end=flanked.end,
        strandedness=getattr(config, "strandedness", "none"),
    )

    # Junctions are already region-bounded from extraction
    target_junctions = junctions
    n_target_junctions = len(target_junctions.starts)

    result.extraction_stats = ExtractionStats(
        total_reads=0,  # Not tracked in junction-only extraction
        spliced_reads=n_spliced,
        unique_junctions=n_target_junctions,
    )

    logger.info(
        "Target %s: %d junctions in region %s:%d-%d",
        region.gene_name or "region",
        n_target_junctions,
        flanked.chrom,
        flanked.start,
        flanked.end,
    )

    if n_target_junctions == 0:
        logger.warning("No junctions found in target region")
        result.assembly_time_seconds = time.perf_counter() - t0
        if reference:
            reference.close()
        return result

    # --- Step 2: Build splice graph ---
    builder_config = GraphBuilderConfig(
        min_junction_support=config.min_junction_support,
        min_relative_junction_support=0.01,  # Very permissive for single gene
        terminal_coverage_dropoff=0.2,
        max_terminal_extension=config.flank,
        locus_flank=config.flank,
        min_relative_exon_coverage=0.005,
    )
    graph_builder = SpliceGraphBuilder(config=builder_config)

    bam_reader = BamReader(bam_path=config.bam_path, min_mapq=config.min_mapq)

    # Create a pseudo-locus covering the target region
    from rapidsplice.graph.builder import LocusDefinition
    locus = LocusDefinition(
        chrom=flanked.chrom,
        start=flanked.start,
        end=flanked.end,
        strand=region.strand if region.strand in ("+", "-") else "+",
        junction_indices=list(range(n_target_junctions)),
    )

    read_data = bam_reader.fetch_region(flanked.chrom, flanked.start, flanked.end)
    if read_data.n_reads == 0:
        logger.warning("No reads in target region")
        result.assembly_time_seconds = time.perf_counter() - t0
        if reference:
            reference.close()
        return result

    graph = graph_builder.build_graph(locus, read_data, target_junctions)
    if graph is None:
        logger.warning("Could not build splice graph for target region")
        result.assembly_time_seconds = time.perf_counter() - t0
        if reference:
            reference.close()
        return result

    # --- Step 3: Decompose with exhaustive enumeration ---
    graph_csr = graph.to_csr()

    decompose_config = DecomposeConfig(
        min_transcript_coverage=config.min_coverage,
        min_relative_abundance=0.005,  # Very permissive for target
        use_safe_paths=False,
        max_paths=config.max_paths,
    )

    primary_run, _ = run_decomposer_pair(
        graph_csr,
        graph,
        config=decompose_config,
        mode="legacy",
    )
    transcripts = primary_run.transcripts
    result.n_paths_enumerated = primary_run.metadata.metrics.get(
        "all_paths_total", 0,
    )

    if not transcripts:
        logger.warning("No transcripts assembled for target region")
        result.assembly_time_seconds = time.perf_counter() - t0
        if reference:
            reference.close()
        return result

    # Intron chain merging
    transcript_filter = TranscriptFilter(
        config=FilterConfig(
            min_score=config.min_transcript_score,
            min_coverage=config.min_coverage,
            min_junction_support=config.min_junction_support,
        ),
    )
    if len(transcripts) > 1:
        transcripts = transcript_filter.merge_identical_intron_chains(transcripts)

    # Score
    scorer = TranscriptScorer(model_path=None)
    scores: list[float] = []
    features_list = []
    for tx in transcripts:
        feat = extract_features(tx, graph, transcripts, target_junctions)
        features_list.append(feat)
        feat_array = features_to_array(feat)
        score = scorer.score(feat_array)
        scores.append(score)

    # Soft filter — keep more for targeted analysis
    scores_array = np.array(scores, dtype=np.float64)
    surviving_indices, _ = transcript_filter.filter_transcripts_with_diagnostics(
        transcripts, scores_array, features_list,
    )
    surviving = [transcripts[i] for i in surviving_indices]
    surviving_scores = [scores[i] for i in surviving_indices]

    assembly_time = time.perf_counter() - t0
    result.assembly_time_seconds = assembly_time

    logger.info(
        "Target assembly: %d transcripts (%d paths, %.2fs)",
        len(surviving),
        result.n_paths_enumerated,
        assembly_time,
    )

    # --- Step 4: Bootstrap confidence intervals ---
    t_boot = time.perf_counter()
    boot_result: BootstrapResult | None = None

    if surviving and config.bootstrap_replicates > 0:
        paths = [tx.node_ids for tx in surviving]
        try:
            boot_cfg = BootstrapConfig(
                n_replicates=config.bootstrap_replicates,
                confidence_level=config.bootstrap_confidence,
                min_presence_rate=config.min_presence_rate,
                seed=None,  # True randomness for independent replicates
            )
            boot_result = bootstrap_confidence(graph_csr, paths, boot_cfg)
        except Exception as exc:
            logger.warning("Bootstrap failed: %s", exc)

    result.bootstrap_time_seconds = time.perf_counter() - t_boot

    # --- Step 5: Build isoform results ---
    gene_name = region.gene_name or "TARGET"
    strand = graph.strand if graph.strand in ("+", "-") else "+"

    for idx, (tx, score) in enumerate(
        zip(surviving, surviving_scores), start=1,
    ):
        exons = tx.exon_coords
        if not exons:
            continue

        n_junctions = max(0, len(exons) - 1)

        iso = IsoformResult(
            transcript_id=f"{gene_name}.{idx}",
            exons=list(exons),
            strand=strand,
            weight=tx.weight,
            score=score,
            n_junctions=n_junctions,
        )

        if boot_result and (idx - 1) < len(boot_result.transcripts):
            bc = boot_result.transcripts[idx - 1]
            iso.ci_low = bc.weight_ci_low
            iso.ci_high = bc.weight_ci_high
            iso.presence_rate = bc.presence_rate
            iso.cv = bc.cv if not np.isnan(bc.cv) else 0.0

        result.isoforms.append(iso)

    # Sort by weight (abundance) descending
    result.isoforms.sort(key=lambda x: x.weight, reverse=True)

    # --- Step 6: Reference comparison ---
    if config.annotation_gtf and result.isoforms:
        from rapidsplice.target.comparator import classify_all_isoforms
        classifications = classify_all_isoforms(
            [iso.exons for iso in result.isoforms],
            config.annotation_gtf,
            region.chrom,
            region.start,
            region.end,
        )
        for iso, cls in zip(result.isoforms, classifications):
            iso.classification = cls.category
            iso.matched_ref_id = cls.matched_transcript_id
            iso.is_novel = cls.category not in ("exact_match", "single_exon")

    if reference:
        reference.close()

    logger.info(
        "Target assembly complete: %d isoforms (%d confident), "
        "bootstrap %.2fs, total %.2fs",
        result.n_isoforms,
        result.n_confident,
        result.bootstrap_time_seconds,
        result.assembly_time_seconds + result.bootstrap_time_seconds,
    )

    return result


def format_target_report(result: TargetAssemblyResult) -> str:
    """Format a human-readable text report of target assembly results.

    Args:
        result: Assembly result to format.

    Returns:
        Multi-line string report.
    """
    lines: list[str] = []
    region = result.region
    lines.append(f"{'='*70}")
    lines.append("  TargetSplice Assembly Report")
    lines.append(f"{'='*70}")
    lines.append(f"  Gene:     {region.gene_name or 'N/A'}")
    lines.append(f"  Region:   {region.chrom}:{region.start+1}-{region.end}")
    lines.append(f"  Strand:   {region.strand}")
    lines.append(f"  Length:   {region.length:,} bp")
    lines.append("")

    if result.extraction_stats:
        es = result.extraction_stats
        lines.append(f"  Reads:     {es.total_reads:,}")
        lines.append(f"  Spliced:   {es.spliced_reads:,}")
        lines.append(f"  Junctions: {es.unique_junctions}")
        lines.append(f"  Coverage:  {es.mean_coverage:.1f}x")
    lines.append("")

    lines.append(f"  Paths enumerated:  {result.n_paths_enumerated}")
    lines.append(f"  Isoforms found:    {result.n_isoforms}")
    lines.append(f"  High-confidence:   {result.n_confident}")
    lines.append(f"  Assembly time:     {result.assembly_time_seconds:.2f}s")
    lines.append(f"  Bootstrap time:    {result.bootstrap_time_seconds:.2f}s")
    lines.append("")

    if result.isoforms:
        has_class = any(iso.classification for iso in result.isoforms)
        hdr = (f"  {'ID':<18} {'Exons':>5} {'Weight':>7} "
               f"{'CI_low':>7} {'CI_high':>7} {'Pres':>6} {'CV':>5}")
        sep = (f"  {'-'*18} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*5}")
        if has_class:
            hdr += f"  {'Classification':<20}"
            sep += f"  {'-'*20}"
        lines.append(hdr)
        lines.append(sep)
        for iso in result.isoforms:
            row = (
                f"  {iso.transcript_id:<18} {len(iso.exons):>5} "
                f"{iso.weight:>7.1f} {iso.ci_low:>7.1f} {iso.ci_high:>7.1f} "
                f"{iso.presence_rate:>5.0%} {iso.cv:>5.2f}"
            )
            if has_class:
                cls_label = iso.classification or "N/A"
                if iso.matched_ref_id:
                    cls_label += f" ({iso.matched_ref_id})"
                row += f"  {cls_label:<20}"
            lines.append(row)
    lines.append(f"{'='*70}")

    return "\n".join(lines)
