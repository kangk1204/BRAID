"""FASTQ-to-target assembly pipeline.

Extracts a target genomic region (gene ± flanking genes) as a mini
reference FASTA, aligns FASTQ reads directly to it using mappy/minimap2,
then runs targeted assembly with bootstrap confidence intervals.

This bypasses genome-wide alignment entirely, giving:
- Sub-minute total runtime (alignment + assembly)
- Zero off-target noise
- Aligner-independent results
- Fusion read detection from chimeric alignments
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import pysam

logger = logging.getLogger(__name__)


@dataclass
class FastqTargetConfig:
    """Configuration for FASTQ-based targeted assembly.

    Attributes:
        fastq_paths: One or two FASTQ file paths (single-end or paired-end).
        reference_path: Path to reference genome FASTA.
        annotation_gtf: Path to GTF annotation for gene lookup.
        gene: Target gene name.
        flank_genes: Number of flanking genes on each side to include.
        flank_bp: Additional bp flanking beyond gene boundaries.
        min_mapq: Minimum mapping quality.
        bootstrap_replicates: Number of bootstrap resampling iterations.
        threads: Number of alignment threads.
    """

    fastq_paths: list[str]
    reference_path: str
    annotation_gtf: str
    gene: str
    flank_genes: int = 5
    flank_bp: int = 10000
    min_mapq: int = 0
    bootstrap_replicates: int = 500
    threads: int = 4


@dataclass
class FastqTargetResult:
    """Result of FASTQ-based targeted assembly."""

    gene: str
    region_chrom: str
    region_start: int
    region_end: int
    region_length: int
    mini_ref_length: int
    n_reads_aligned: int
    n_reads_spliced: int
    alignment_time: float
    assembly_time: float
    bootstrap_time: float
    isoforms: list = field(default_factory=list)
    flanking_genes: list[str] = field(default_factory=list)


def extract_target_fasta(
    reference_path: str,
    chrom: str,
    start: int,
    end: int,
    output_path: str,
) -> int:
    """Extract a genomic region as a FASTA file.

    Args:
        reference_path: Path to reference genome FASTA (indexed).
        chrom: Chromosome name.
        start: 0-based start coordinate.
        end: 0-based exclusive end coordinate.
        output_path: Path to write the extracted FASTA.

    Returns:
        Length of extracted sequence.
    """
    with pysam.FastaFile(reference_path) as fasta:
        seq = fasta.fetch(chrom, start, end)

    with open(output_path, "w", encoding="utf-8") as fh:
        # Use original coordinates in the header for coordinate mapping
        fh.write(f">{chrom}:{start}-{end}\n")
        # Write in 80-char lines
        for i in range(0, len(seq), 80):
            fh.write(seq[i : i + 80] + "\n")

    return len(seq)


def _clamp_region_to_reference(
    reference_path: str,
    chrom: str,
    start: int,
    end: int,
) -> tuple[int, int]:
    """Clamp a half-open genomic interval to the reference contig length."""
    with pysam.FastaFile(reference_path) as fasta:
        contig_len = fasta.get_reference_length(chrom)

    clamped_start = max(0, min(start, contig_len))
    clamped_end = max(clamped_start, min(end, contig_len))
    return clamped_start, clamped_end


def find_flanking_region(
    gtf_path: str,
    gene_name: str,
    flank_genes: int = 5,
    flank_bp: int = 10000,
) -> tuple[str, int, int, list[str]]:
    """Find the target gene plus flanking genes region.

    Looks up the target gene in the GTF, then extends the region to
    include *flank_genes* genes on each side, plus *flank_bp* additional
    base pairs.

    Args:
        gtf_path: Path to GTF annotation.
        gene_name: Target gene name.
        flank_genes: Number of genes to include on each side.
        flank_bp: Additional bp flanking.

    Returns:
        Tuple of (chrom, start, end, list_of_gene_names_in_region).

    Raises:
        ValueError: If gene not found.
    """
    query = gene_name.upper()

    # Collect all genes on the same chromosome
    genes: list[tuple[int, int, str, str]] = []  # start, end, name, chrom
    target_chrom = None
    target_idx = -1

    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue

            attrs = fields[8]
            gname = _parse_attr(attrs, "gene_name")
            if not gname:
                continue

            chrom = fields[0]
            start = int(fields[3]) - 1
            end = int(fields[4])

            if gname.upper() == query:
                target_chrom = chrom
                # Don't break — need to collect all genes on this chrom

            genes.append((start, end, gname, chrom))

    if target_chrom is None:
        raise ValueError(f"Gene {gene_name!r} not found in {gtf_path}")

    # Filter to same chromosome, sort by position
    chrom_genes = [
        (s, e, n) for s, e, n, c in genes if c == target_chrom
    ]
    chrom_genes.sort(key=lambda x: x[0])

    # Find target index
    for i, (s, e, n) in enumerate(chrom_genes):
        if n.upper() == query:
            target_idx = i
            break

    if target_idx < 0:
        raise ValueError(f"Gene {gene_name!r} not found on chromosome")

    # Extend to include flanking genes
    lo = max(0, target_idx - flank_genes)
    hi = min(len(chrom_genes), target_idx + flank_genes + 1)

    region_start = max(0, chrom_genes[lo][0] - flank_bp)
    region_end = chrom_genes[hi - 1][1] + flank_bp

    gene_names = [n for _, _, n in chrom_genes[lo:hi]]

    logger.info(
        "Target region: %s:%d-%d (%d bp, %d genes: %s)",
        target_chrom,
        region_start,
        region_end,
        region_end - region_start,
        len(gene_names),
        ", ".join(gene_names[:5]) + ("..." if len(gene_names) > 5 else ""),
    )

    return target_chrom, region_start, region_end, gene_names


def _decode_stderr(raw: bytes | str | None, limit: int = 500) -> str:
    """Decode captured subprocess stderr for an error message (bounded)."""
    if not raw:
        return "(no stderr captured)"
    if isinstance(raw, bytes):
        raw = raw.decode(errors="replace")
    return raw.strip()[:limit]


def _pipe_aligner_to_sorter(
    aligner_cmd: list[str],
    sorter_cmd: list[str],
) -> tuple[int, bytes, int, bytes]:
    """Run ``aligner_cmd | sorter_cmd`` without risking a stderr-pipe deadlock.

    The aligner's stderr is drained to a temporary file rather than an OS pipe.
    A verbose aligner can emit far more than the ~64 KB pipe buffer holds; if
    that stderr were an unread ``PIPE`` the aligner would block writing it, never
    close its stdout, and the downstream sorter's ``communicate()`` would wait
    forever for stdout EOF — hanging the whole pipeline. A temporary file has no
    such backpressure limit, so both stages always run to completion.

    Returns ``(aligner_returncode, aligner_stderr, sorter_returncode,
    sorter_stderr)``.
    """
    with tempfile.TemporaryFile() as aligner_err_f:
        aligner = subprocess.Popen(
            aligner_cmd, stdout=subprocess.PIPE, stderr=aligner_err_f,
        )
        try:
            sorter = subprocess.Popen(
                sorter_cmd,
                stdin=aligner.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except BaseException:
            # The aligner is already running. If the sorter never launched (e.g.
            # samtools missing from PATH) nothing will read the aligner's stdout,
            # so reap it now rather than leak a process blocked on a full pipe.
            if aligner.stdout is not None:
                aligner.stdout.close()
            aligner.terminate()
            aligner.wait()
            raise
        # Let the aligner receive SIGPIPE if the sorter dies early.
        aligner.stdout.close()
        _, sorter_err = sorter.communicate()
        aligner_ret = aligner.wait()
        aligner_err_f.seek(0)
        aligner_err = aligner_err_f.read()
    return aligner_ret, aligner_err, sorter.returncode, sorter_err


def align_fastq_to_target(
    fastq_paths: list[str],
    target_fasta: str,
    output_bam: str,
    threads: int = 4,
) -> tuple[int, int, float]:
    """Align FASTQ reads to a target FASTA using HISAT2.

    Builds a HISAT2 splice-aware index for the mini reference, aligns
    reads with ``--dta`` mode, and produces a sorted, indexed BAM.
    HISAT2 is preferred over minimap2 for short reads (<200bp) because
    it provides XS tags and splice-aware alignment natively.

    Args:
        fastq_paths: FASTQ file paths (1 for SE, 2 for PE).
        target_fasta: Path to target region FASTA.
        output_bam: Path for output BAM.
        threads: Alignment threads.

    Returns:
        Tuple of (n_aligned, n_spliced, elapsed_seconds).
    """
    if len(fastq_paths) == 0:
        raise ValueError("At least one FASTQ path is required")
    if len(fastq_paths) > 2:
        raise ValueError("Expected one single-end FASTQ or two paired-end FASTQs")

    t0 = time.perf_counter()

    # Build HISAT2 index for target FASTA
    idx_prefix = target_fasta + ".hisat2"
    subprocess.run(
        ["hisat2-build", "-q", target_fasta, idx_prefix],
        check=True, capture_output=True,
    )

    # Build alignment command
    hisat2_cmd = [
        "hisat2",
        "-x", idx_prefix,
        "-p", str(threads),
        "--dta",
        "--no-unal",
    ]

    if len(fastq_paths) >= 2:
        hisat2_cmd.extend(["-1", fastq_paths[0], "-2", fastq_paths[1]])
    else:
        hisat2_cmd.extend(["-U", fastq_paths[0]])

    # Pipe HISAT2 SAM output → samtools sort → BAM. The helper drains hisat2's
    # stderr to a temp file so a verbose aligner cannot deadlock the pipeline on
    # a full stderr pipe (see _pipe_aligner_to_sorter).
    hisat2_ret, hisat2_err, sort_ret, sort_err = _pipe_aligner_to_sorter(
        hisat2_cmd,
        ["samtools", "sort", "-@", str(threads), "-o", output_bam, "-"],
    )

    # Distinguish an external-tool failure from a successful run that simply
    # produced zero alignments. A non-zero return code from either stage means
    # the BAM is invalid and must NOT be reported as "0 reads aligned".
    if hisat2_ret != 0:
        raise RuntimeError(
            f"hisat2 alignment failed (exit {hisat2_ret}): "
            f"{_decode_stderr(hisat2_err)}"
        )
    if sort_ret != 0:
        raise RuntimeError(
            f"samtools sort failed (exit {sort_ret}): "
            f"{_decode_stderr(sort_err)}"
        )

    if not os.path.exists(output_bam) or os.path.getsize(output_bam) == 0:
        logger.warning("HISAT2 ran successfully but produced no alignments")
        return 0, 0, time.perf_counter() - t0

    # Index
    subprocess.run(
        ["samtools", "index", output_bam],
        check=True, capture_output=True,
    )

    # Count aligned and spliced reads
    n_aligned = 0
    n_spliced = 0
    with pysam.AlignmentFile(output_bam, "rb") as bam:
        for read in bam:
            if not read.is_unmapped:
                n_aligned += 1
                if read.cigartuples and any(
                    op == 3 for op, _ in read.cigartuples
                ):
                    n_spliced += 1

    elapsed = time.perf_counter() - t0

    logger.info(
        "Aligned %d reads (%d spliced) in %.2fs",
        n_aligned, n_spliced, elapsed,
    )

    return n_aligned, n_spliced, elapsed


def run_fastq_target_pipeline(
    config: FastqTargetConfig,
) -> FastqTargetResult:
    """Run the complete FASTQ-to-target assembly pipeline.

    Steps:
    1. Look up target gene + flanking genes
    2. Extract region as mini FASTA
    3. Align FASTQ reads to mini reference
    4. Run targeted assembly with bootstrap CI

    Args:
        config: Pipeline configuration.

    Returns:
        Complete assembly result.
    """
    from braid.target.assembler import TargetConfig, assemble_target
    from braid.target.extractor import TargetRegion

    # Step 1: Find target region
    chrom, reg_start, reg_end, gene_names = find_flanking_region(
        config.annotation_gtf,
        config.gene,
        flank_genes=config.flank_genes,
        flank_bp=config.flank_bp,
    )
    reg_start, reg_end = _clamp_region_to_reference(
        config.reference_path, chrom, reg_start, reg_end,
    )

    result = FastqTargetResult(
        gene=config.gene,
        region_chrom=chrom,
        region_start=reg_start,
        region_end=reg_end,
        region_length=reg_end - reg_start,
        mini_ref_length=0,
        n_reads_aligned=0,
        n_reads_spliced=0,
        alignment_time=0.0,
        assembly_time=0.0,
        bootstrap_time=0.0,
        flanking_genes=gene_names,
    )

    with tempfile.TemporaryDirectory(prefix="targetsplice_") as tmpdir:
        # Step 2: Extract mini reference
        mini_fasta = os.path.join(tmpdir, "target.fa")
        seq_len = extract_target_fasta(
            config.reference_path, chrom, reg_start, reg_end, mini_fasta,
        )
        result.mini_ref_length = seq_len
        logger.info("Mini reference: %d bp", seq_len)

        # Step 3: Align FASTQ to mini reference
        mini_bam = os.path.join(tmpdir, "aligned.bam")
        n_aligned, n_spliced, align_time = align_fastq_to_target(
            config.fastq_paths,
            mini_fasta,
            mini_bam,
            threads=config.threads,
        )
        result.n_reads_aligned = n_aligned
        result.n_reads_spliced = n_spliced
        result.alignment_time = align_time

        if n_aligned == 0:
            logger.warning("No reads aligned to target region")
            return result

        # Step 4: Run targeted assembly on the mini BAM
        # Coordinates in the mini BAM are relative to the extracted region
        # The FASTA header contains the original coordinates
        target_region = TargetRegion(
            chrom=f"{chrom}:{reg_start}-{reg_end}",
            start=0,
            end=seq_len,
            strand=".",
            gene_name=config.gene,
        )

        # Find target gene position within the extracted region
        from braid.target.extractor import lookup_gene
        gene_region = lookup_gene(config.annotation_gtf, config.gene)
        if gene_region:
            # Convert to local coordinates
            local_start = max(0, gene_region.start - reg_start)
            local_end = min(seq_len, gene_region.end - reg_start)
            target_region = TargetRegion(
                chrom=f"{chrom}:{reg_start}-{reg_end}",
                start=local_start,
                end=local_end,
                strand=gene_region.strand,
                gene_name=config.gene,
            )

        assembly_config = TargetConfig(
            bam_path=mini_bam,
            reference_path=mini_fasta,
            region=target_region,
            flank=config.flank_bp,
            min_mapq=config.min_mapq,
            bootstrap_replicates=config.bootstrap_replicates,
            max_paths=5000,
            annotation_gtf=None,  # Skip comparison for now (coords differ)
        )

        assembly_result = assemble_target(assembly_config)
        result.assembly_time = assembly_result.assembly_time_seconds
        result.bootstrap_time = assembly_result.bootstrap_time_seconds

        # Convert local coordinates back to genomic
        for iso in assembly_result.isoforms:
            genomic_exons = [
                (s + reg_start, e + reg_start) for s, e in iso.exons
            ]
            iso.exons = genomic_exons

        result.isoforms = assembly_result.isoforms

    return result


def format_fastq_target_report(result: FastqTargetResult) -> str:
    """Format a text report for FASTQ pipeline results."""
    lines: list[str] = []
    lines.append(f"{'='*70}")
    lines.append("  TargetSplice FASTQ Pipeline Report")
    lines.append(f"{'='*70}")
    lines.append(f"  Gene:           {result.gene}")
    lines.append(
        f"  Region:         {result.region_chrom}:"
        f"{result.region_start+1}-{result.region_end}"
    )
    lines.append(f"  Region length:  {result.region_length:,} bp")
    lines.append(f"  Mini-ref:       {result.mini_ref_length:,} bp")
    lines.append(
        f"  Flanking genes: "
        f"{', '.join(result.flanking_genes[:8])}"
        f"{'...' if len(result.flanking_genes) > 8 else ''}"
    )
    lines.append("")
    lines.append(f"  Reads aligned:  {result.n_reads_aligned:,}")
    lines.append(f"  Spliced reads:  {result.n_reads_spliced:,}")
    lines.append(f"  Align time:     {result.alignment_time:.2f}s")
    lines.append(f"  Assembly time:  {result.assembly_time:.2f}s")
    lines.append(f"  Bootstrap time: {result.bootstrap_time:.2f}s")
    total = result.alignment_time + result.assembly_time + result.bootstrap_time
    lines.append(f"  Total time:     {total:.2f}s")
    lines.append("")

    lines.append(f"  Isoforms found: {len(result.isoforms)}")
    n_confident = sum(1 for i in result.isoforms if i.presence_rate >= 0.5)
    lines.append(f"  High-confidence: {n_confident}")
    lines.append("")

    if result.isoforms:
        lines.append(
            f"  {'ID':<18} {'Exons':>5} {'Weight':>7} "
            f"{'CI_low':>7} {'CI_high':>7} {'Pres':>6} {'CV':>5}"
        )
        lines.append(
            f"  {'-'*18} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*5}"
        )
        for iso in result.isoforms:
            lines.append(
                f"  {iso.transcript_id:<18} {len(iso.exons):>5} "
                f"{iso.weight:>7.1f} {iso.ci_low:>7.1f} "
                f"{iso.ci_high:>7.1f} "
                f"{iso.presence_rate:>5.0%} {iso.cv:>5.2f}"
            )

    lines.append(f"{'='*70}")
    return "\n".join(lines)


def _parse_attr(attrs: str, key: str) -> str | None:
    """Extract an attribute value from a GTF attribute string."""
    for part in attrs.split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(None, 1)
        if len(tokens) == 2 and tokens[0] == key:
            return tokens[1].strip('"').strip("'")
    return None
