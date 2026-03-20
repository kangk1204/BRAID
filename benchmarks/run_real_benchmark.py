"""Real-data benchmark for RapidSplice on ENCODE RNA-seq samples.

Downloads, aligns, and evaluates transcript assembly on real ENCODE10
RNA-seq data using HISAT2 alignment and GFFcompare evaluation against
GENCODE v38 annotations.

Usage:
    python benchmarks/run_real_benchmark.py --sample SRR387661
    python benchmarks/run_real_benchmark.py --sample SRR387661 --threads 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent / "real_benchmark"
ANNOTATION_GTF_CHR = BASE_DIR / "annotation" / "gencode.v38.annotation.gtf"
ANNOTATION_GTF_NOCHR = BASE_DIR / "annotation" / "gencode.v38.nochr.gtf"
HISAT2_INDEX = BASE_DIR / "reference" / "grch38" / "genome"
REFERENCE_FASTA = BASE_DIR / "reference" / "grch38" / "genome.fa"


@dataclass
class RealBenchmarkConfig:
    """Configuration for real-data benchmark."""

    sample_id: str = "SRR387661"
    threads: int = 4
    output_dir: str = str(BASE_DIR / "results")
    skip_download: bool = False
    skip_align: bool = False
    chr_filter: str | None = None
    braid_only: bool = False
    braid_decomposer: str = "legacy"
    braid_builder_profile: str = "default"
    braid_min_junction_support: int = 3
    braid_min_coverage: float = 1.0
    braid_min_score: float = 0.1
    braid_max_paths: int = 500
    braid_enable_motif_validation: bool = True
    braid_diagnostics_dir: str | None = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def run_command(
    cmd: list[str],
    description: str,
    timeout: int = 7200,
) -> tuple[float, float]:
    """Run a command and return (elapsed_seconds, peak_memory_mb)."""
    logger.info("Running %s: %s", description, " ".join(cmd))
    start = time.perf_counter()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - start

        if result.returncode != 0:
            logger.error(
                "%s failed (exit %d):\nstdout: %s\nstderr: %s",
                description, result.returncode,
                result.stdout[-500:] if result.stdout else "",
                result.stderr[-500:] if result.stderr else "",
            )
            raise RuntimeError(f"{description} failed with exit code {result.returncode}")

        return elapsed, 0.0
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        logger.error("%s timed out after %ds", description, timeout)
        raise RuntimeError(f"{description} timed out after {timeout}s")


def _build_braid_command(
    bam_path: str,
    output_gtf: str,
    config: RealBenchmarkConfig,
    *,
    chromosomes: list[str] | None = None,
) -> list[str]:
    """Build the RapidSplice command for the real-data benchmark."""
    cmd = [
        sys.executable,
        "-m",
        "braid.cli",
        "assemble",
        bam_path,
        "-o",
        output_gtf,
        "-t",
        str(config.threads),
        "-c",
        str(config.braid_min_coverage),
        "-s",
        str(config.braid_min_score),
        "-j",
        str(config.braid_min_junction_support),
        "--decomposer",
        config.braid_decomposer,
        "--builder-profile",
        config.braid_builder_profile,
        "--max-paths",
        str(config.braid_max_paths),
    ]
    if config.braid_enable_motif_validation:
        cmd.extend(["-r", str(REFERENCE_FASTA)])
    else:
        cmd.append("--no-motif-validation")
    if chromosomes:
        cmd.extend(["--chromosomes", ",".join(chromosomes)])
    if config.braid_diagnostics_dir:
        cmd.extend(["--diagnostics-dir", config.braid_diagnostics_dir])
    return cmd


def _parse_chr_filter(raw: str | None) -> list[str] | None:
    """Parse a comma-separated chromosome filter."""
    if raw is None:
        return None
    chroms = [item.strip() for item in raw.split(",") if item.strip()]
    return chroms or None


def _filter_annotation_gtf(
    reference_gtf: Path,
    output_gtf: Path,
    chromosomes: list[str],
) -> Path:
    """Write a chromosome-filtered annotation GTF for proxy benchmarking."""
    chrom_set = set(chromosomes)
    with open(reference_gtf, encoding="utf-8") as src, open(
        output_gtf, "w", encoding="utf-8",
    ) as dst:
        for line in src:
            if line.startswith("#"):
                dst.write(line)
                continue
            parts = line.split("\t", 1)
            if parts and parts[0] in chrom_set:
                dst.write(line)
    return output_gtf


def _detect_contig_style_from_idxstats_output(idxstats_output: str) -> str:
    """Infer contig namespace style from samtools idxstats output."""
    for line in idxstats_output.splitlines():
        fields = line.split("\t")
        if len(fields) < 4:
            continue
        contig = fields[0].strip()
        if not contig or contig == "*":
            continue
        return "chr" if contig.startswith("chr") else "nochr"
    raise RuntimeError("Unable to determine contig naming style from samtools idxstats output.")


def _detect_contig_style_from_fai(fai_path: Path) -> str:
    """Infer contig namespace style from a FASTA index."""
    with open(fai_path, encoding="utf-8") as fh:
        for line in fh:
            fields = line.split("\t")
            if not fields:
                continue
            contig = fields[0].strip()
            if not contig:
                continue
            return "chr" if contig.startswith("chr") else "nochr"
    raise RuntimeError(f"Unable to determine contig naming style from FASTA index: {fai_path}")


def _annotation_for_contig_style(contig_style: str) -> Path:
    """Select the matching GENCODE annotation for the BAM/reference naming style."""
    if contig_style == "chr":
        return ANNOTATION_GTF_CHR
    if contig_style == "nochr":
        return ANNOTATION_GTF_NOCHR
    raise ValueError(f"Unsupported contig style: {contig_style}")


def align_with_hisat2(
    fastq_1: str,
    fastq_2: str,
    output_bam: str,
    index: str,
    threads: int,
) -> float:
    """Align paired-end reads with HISAT2 and sort with samtools.

    Returns elapsed time in seconds.
    """
    sam_path = output_bam.replace(".bam", ".sam")

    # HISAT2 alignment
    hisat2_cmd = [
        "hisat2", "--dta",
        "-x", index,
        "-1", fastq_1,
        "-2", fastq_2,
        "-p", str(threads),
        "--no-mixed", "--no-discordant",
        "-S", sam_path,
    ]

    elapsed, _ = run_command(hisat2_cmd, "HISAT2 alignment", timeout=7200)

    # Sort and convert to BAM
    sort_cmd = [
        "samtools", "sort",
        "-@", str(threads),
        "-o", output_bam,
        sam_path,
    ]
    run_command(sort_cmd, "samtools sort", timeout=3600)

    # Index BAM
    index_cmd = ["samtools", "index", "-@", str(threads), output_bam]
    run_command(index_cmd, "samtools index", timeout=600)

    # Clean up SAM
    if os.path.exists(sam_path):
        os.remove(sam_path)

    return elapsed


def run_gffcompare(
    predicted_gtf: str,
    reference_gtf: str,
    output_prefix: str,
) -> dict[str, float]:
    """Run gffcompare and parse sensitivity/precision metrics.

    Returns dict with transcript_sensitivity, transcript_precision, etc.
    """
    cmd = [
        "gffcompare",
        "-r", reference_gtf,
        "-o", output_prefix,
        predicted_gtf,
    ]
    run_command(cmd, "gffcompare", timeout=600)

    # Parse the .stats file
    stats_file = output_prefix + ".stats"
    metrics: dict[str, float] = {}

    if not os.path.exists(stats_file):
        logger.error("gffcompare stats file not found: %s", stats_file)
        return metrics

    with open(stats_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse lines like:
            #           Sensitivity | Precision
            #         Base level:   88.0     |    72.3
            #         Exon level:   75.2     |    68.1
            #       Intron level:   82.1     |    80.5
            # Intron chain level:   45.3     |    42.1
            #  Transcript level:    35.2     |    31.8
            #       Locus level:    42.1     |    38.7
            if "|" in line and "level" in line.lower():
                parts = line.split("|")
                left = parts[0].strip()
                right = parts[1].strip() if len(parts) > 1 else ""

                # Extract level name and sensitivity
                level_parts = left.split(":")
                if len(level_parts) == 2:
                    level_name = level_parts[0].strip().lower().replace(" ", "_")
                    try:
                        sensitivity = float(level_parts[1].strip())
                    except ValueError:
                        continue
                    try:
                        precision = float(right)
                    except ValueError:
                        precision = 0.0

                    metrics[f"{level_name}_sensitivity"] = sensitivity / 100.0
                    metrics[f"{level_name}_precision"] = precision / 100.0

            # Also parse matching statistics
            # Matching intron chains: 12345
            if line.startswith("Matching intron chains"):
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        metrics["matching_intron_chains"] = int(parts[1].strip())
                    except ValueError:
                        pass

            # Total multi-exon transcripts
            if "multi-exon transcripts" in line.lower():
                parts = line.split(":")
                if len(parts) == 2:
                    try:
                        val = int(parts[1].strip().split()[0])
                        if "reference" in line.lower():
                            metrics["ref_multi_exon"] = val
                        elif "query" in line.lower():
                            metrics["query_multi_exon"] = val
                    except (ValueError, IndexError):
                        pass

    return metrics


def count_transcripts(gtf_path: str) -> int:
    """Count the number of transcript features in a GTF file."""
    count = 0
    with open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 3 and parts[2] == "transcript":
                count += 1
    return count


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_benchmark(config: RealBenchmarkConfig) -> dict:
    """Run the full real-data benchmark pipeline."""
    sample = config.sample_id
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fastq_dir = BASE_DIR / "fastq"
    bam_dir = BASE_DIR / "bam"
    bam_dir.mkdir(parents=True, exist_ok=True)

    fastq_1 = str(fastq_dir / f"{sample}_1.fastq.gz")
    fastq_2 = str(fastq_dir / f"{sample}_2.fastq.gz")
    bam_path = str(bam_dir / f"{sample}.hisat2.sorted.bam")

    results: dict = {
        "config": asdict(config),
        "sample": sample,
        "tools": {},
    }
    proxy_chromosomes = _parse_chr_filter(config.chr_filter) if config.braid_only else None

    # --- Step 1: Check inputs ---
    if not os.path.exists(str(ANNOTATION_GTF_CHR)):
        logger.error("GENCODE annotation not found: %s", ANNOTATION_GTF_CHR)
        return results
    if not os.path.exists(str(ANNOTATION_GTF_NOCHR)):
        logger.error("GENCODE annotation not found: %s", ANNOTATION_GTF_NOCHR)
        return results

    hisat2_check = str(HISAT2_INDEX) + ".1.ht2"
    if not os.path.exists(hisat2_check):
        logger.error("HISAT2 index not found: %s", hisat2_check)
        return results
    if config.braid_enable_motif_validation and not os.path.exists(str(REFERENCE_FASTA)):
        logger.error("Reference FASTA not found for motif validation: %s", REFERENCE_FASTA)
        return results
    reference_fai = Path(f"{REFERENCE_FASTA}.fai")
    if config.braid_enable_motif_validation and not reference_fai.exists():
        logger.error("Reference FASTA index not found for motif validation: %s", reference_fai)
        return results

    # --- Step 2: Align ---
    if not config.skip_align:
        if not os.path.exists(fastq_1) or not os.path.exists(fastq_2):
            logger.error("FASTQ files not found: %s, %s", fastq_1, fastq_2)
            return results

        logger.info("Aligning %s with HISAT2...", sample)
        align_time = align_with_hisat2(
            fastq_1, fastq_2, bam_path,
            str(HISAT2_INDEX), config.threads,
        )
        results["alignment_time"] = align_time
        logger.info("Alignment completed in %.1fs", align_time)
    else:
        logger.info("Skipping alignment (--skip-align)")

    if not os.path.exists(bam_path):
        logger.error("BAM file not found: %s", bam_path)
        return results

    # Check BAM has reads
    check = subprocess.run(
        ["samtools", "idxstats", bam_path],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "samtools idxstats failed for "
            f"{bam_path}: {check.stderr.strip() or 'no stderr output'}"
        )
    idxstats_rows: list[tuple[str, int]] = []
    for line in check.stdout.strip().split("\n"):
        if not line or line.startswith("*"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        idxstats_rows.append((fields[0], int(fields[2])))
    available_chromosomes = {chrom for chrom, _ in idxstats_rows}
    bam_contig_style = _detect_contig_style_from_idxstats_output(check.stdout)
    annotation_gtf = _annotation_for_contig_style(bam_contig_style)
    logger.info(
        "Detected BAM contig style '%s'; using annotation %s",
        bam_contig_style, annotation_gtf,
    )
    if config.braid_enable_motif_validation:
        reference_contig_style = _detect_contig_style_from_fai(reference_fai)
        if reference_contig_style != bam_contig_style:
            raise RuntimeError(
                "BAM/reference contig naming mismatch: "
                f"BAM={bam_contig_style}, reference={reference_contig_style}"
            )
        logger.info(
            "Reference FASTA contig style '%s' matches BAM naming.",
            reference_contig_style,
        )
    if config.chr_filter and not config.braid_only:
        logger.warning(
            "--chr filter is ignored in real benchmark to keep tool "
            "comparisons fair across assemblers."
        )
    if proxy_chromosomes:
        missing = sorted(set(proxy_chromosomes) - available_chromosomes)
        if missing:
            raise RuntimeError(
                "Requested proxy chromosome(s) not present in BAM: "
                + ", ".join(missing)
            )
        proxy_annotation = out_dir / f"{sample}.proxy.annotation.gtf"
        annotation_gtf = _filter_annotation_gtf(
            annotation_gtf, proxy_annotation, proxy_chromosomes,
        )
        logger.info(
            "Proxy mode enabled for RapidSplice-only run on chromosomes: %s",
            ",".join(proxy_chromosomes),
        )
    total_reads = sum(
        read_count
        for chrom, read_count in idxstats_rows
        if proxy_chromosomes is None or chrom in set(proxy_chromosomes)
    )
    results["total_aligned_reads"] = total_reads
    results["annotation_gtf"] = str(annotation_gtf)
    results["bam_contig_style"] = bam_contig_style
    if proxy_chromosomes:
        results["proxy_chromosomes"] = proxy_chromosomes
    logger.info("BAM has %d aligned reads", total_reads)

    # --- Step 3: Run RapidSplice ---
    rs_gtf = str(out_dir / f"{sample}_braid.gtf")
    if os.path.exists(rs_gtf):
        os.remove(rs_gtf)
    rs_cmd = _build_braid_command(
        bam_path,
        rs_gtf,
        config,
        chromosomes=proxy_chromosomes,
    )

    rs_start = time.perf_counter()
    run_command(rs_cmd, "RapidSplice", timeout=3600)
    rs_elapsed = time.perf_counter() - rs_start

    if os.path.exists(rs_gtf):
        rs_n_tx = count_transcripts(rs_gtf)
        rs_prefix = str(out_dir / f"{sample}_braid_gffcmp")
        rs_metrics = run_gffcompare(rs_gtf, str(annotation_gtf), rs_prefix)
        results["tools"]["RapidSplice"] = {
            "gtf_path": rs_gtf,
            "runtime_seconds": rs_elapsed,
            "n_transcripts": rs_n_tx,
            "metrics": rs_metrics,
            "diagnostics_dir": config.braid_diagnostics_dir,
        }
        logger.info(
            "RapidSplice: %d transcripts in %.1fs", rs_n_tx, rs_elapsed,
        )

    # --- Step 4: Run StringTie ---
    if not config.braid_only and shutil.which("stringtie"):
        st_gtf = str(out_dir / f"{sample}_stringtie.gtf")
        if os.path.exists(st_gtf):
            os.remove(st_gtf)
        st_cmd = [
            "stringtie", bam_path,
            "-o", st_gtf,
            "-p", str(config.threads),
        ]

        st_start = time.perf_counter()
        run_command(st_cmd, "StringTie", timeout=3600)
        st_elapsed = time.perf_counter() - st_start

        if os.path.exists(st_gtf):
            st_n_tx = count_transcripts(st_gtf)
            st_prefix = str(out_dir / f"{sample}_stringtie_gffcmp")
            st_metrics = run_gffcompare(st_gtf, str(annotation_gtf), st_prefix)
            results["tools"]["StringTie"] = {
                "gtf_path": st_gtf,
                "runtime_seconds": st_elapsed,
                "n_transcripts": st_n_tx,
                "metrics": st_metrics,
            }
            logger.info(
                "StringTie: %d transcripts in %.1fs", st_n_tx, st_elapsed,
            )

    # --- Step 5: Run Scallop2 ---
    scallop_bin = shutil.which("scallop2") or shutil.which("scallop")
    if not config.braid_only and scallop_bin:
        sc_gtf = str(out_dir / f"{sample}_scallop.gtf")
        if os.path.exists(sc_gtf):
            os.remove(sc_gtf)
        sc_cmd = [scallop_bin, "-i", bam_path, "-o", sc_gtf]

        sc_start = time.perf_counter()
        run_command(sc_cmd, "Scallop2", timeout=3600)
        sc_elapsed = time.perf_counter() - sc_start

        if os.path.exists(sc_gtf):
            sc_n_tx = count_transcripts(sc_gtf)
            sc_prefix = str(out_dir / f"{sample}_scallop_gffcmp")
            sc_metrics = run_gffcompare(sc_gtf, str(annotation_gtf), sc_prefix)
            results["tools"]["Scallop2"] = {
                "gtf_path": sc_gtf,
                "runtime_seconds": sc_elapsed,
                "n_transcripts": sc_n_tx,
                "metrics": sc_metrics,
            }
            logger.info(
                "Scallop2: %d transcripts in %.1fs", sc_n_tx, sc_elapsed,
            )

    # --- Save results ---
    results_path = str(out_dir / f"{sample}_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # --- Print summary ---
    print("\n" + "=" * 80)
    print(f"REAL-DATA BENCHMARK: {sample}")
    print("=" * 80)
    print(f"Aligned reads: {total_reads:,}")
    print(f"Reference: GENCODE v38 ({annotation_gtf})")
    if proxy_chromosomes:
        print(f"Proxy chromosomes: {','.join(proxy_chromosomes)}")
    print("-" * 80)
    print(f"{'Tool':<15} {'TxSn':>8} {'TxPr':>8} {'ExSn':>8} {'ExPr':>8}"
          f" {'InSn':>8} {'InPr':>8} {'Time(s)':>10} {'#Tx':>8}")
    print("-" * 80)

    for tool_name, tool_data in results["tools"].items():
        m = tool_data.get("metrics", {})
        tx_sn = m.get("transcript_level_sensitivity", 0) * 100
        tx_pr = m.get("transcript_level_precision", 0) * 100
        ex_sn = m.get("exon_level_sensitivity", 0) * 100
        ex_pr = m.get("exon_level_precision", 0) * 100
        in_sn = m.get("intron_level_sensitivity", 0) * 100
        in_pr = m.get("intron_level_precision", 0) * 100
        runtime = tool_data.get("runtime_seconds", 0)
        n_tx = tool_data.get("n_transcripts", 0)
        print(
            f"{tool_name:<15} {tx_sn:>7.1f}% {tx_pr:>7.1f}% {ex_sn:>7.1f}%"
            f" {ex_pr:>7.1f}% {in_sn:>7.1f}% {in_pr:>7.1f}%"
            f" {runtime:>10.1f} {n_tx:>8}",
        )

    print("=" * 80)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for real-data benchmark."""
    parser = argparse.ArgumentParser(
        description="Run real-data RNA-seq transcript assembly benchmark.",
    )
    parser.add_argument(
        "--sample", default="SRR387661",
        help="SRA accession number (default: SRR387661, K562)",
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of threads (default: 4)",
    )
    parser.add_argument(
        "--output-dir", default=str(BASE_DIR / "results"),
        help="Output directory",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip FASTQ download",
    )
    parser.add_argument(
        "--skip-align", action="store_true",
        help="Skip HISAT2 alignment (use existing BAM)",
    )
    parser.add_argument(
        "--chr", dest="chr_filter", default=None,
        help="Restrict to specific chromosomes (comma-separated)",
    )
    parser.add_argument(
        "--braid-only", action="store_true",
        help="Run only RapidSplice on the real-data benchmark.",
    )
    parser.add_argument(
        "--decomposer", choices=["legacy", "iterative_v2"], default="legacy",
        help="RapidSplice decomposer to use (default: legacy).",
    )
    parser.add_argument(
        "--builder-profile",
        choices=["default", "conservative_correctness", "aggressive_recall"],
        default="default",
        help="RapidSplice builder profile (default: default).",
    )
    parser.add_argument(
        "--min-junction-support", type=int, default=3,
        help="RapidSplice minimum junction support (default: 3).",
    )
    parser.add_argument(
        "--min-coverage", type=float, default=1.0,
        help="RapidSplice minimum coverage (default: 1.0).",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.1,
        help="RapidSplice minimum transcript score (default: 0.1).",
    )
    parser.add_argument(
        "--max-paths", type=int, default=500,
        help="RapidSplice max_paths for legacy decomposition (default: 500).",
    )
    parser.add_argument(
        "--no-motif-validation", action="store_true",
        help="Disable RapidSplice motif validation. Leave off for real-data motif-on validation.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default=None,
        help="Optional directory for RapidSplice diagnostics JSONL/summary output.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = RealBenchmarkConfig(
        sample_id=args.sample,
        threads=args.threads,
        output_dir=args.output_dir,
        skip_download=args.skip_download,
        skip_align=args.skip_align,
        chr_filter=args.chr_filter,
        braid_only=args.braid_only,
        braid_decomposer=args.decomposer,
        braid_builder_profile=args.builder_profile,
        braid_min_junction_support=args.min_junction_support,
        braid_min_coverage=args.min_coverage,
        braid_min_score=args.min_score,
        braid_max_paths=args.max_paths,
        braid_enable_motif_validation=not args.no_motif_validation,
        braid_diagnostics_dir=args.diagnostics_dir,
    )

    run_benchmark(config)


if __name__ == "__main__":
    main()
