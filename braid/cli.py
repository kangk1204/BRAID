"""Command-line interface for BRAID."""

from __future__ import annotations

import argparse
import logging
import sys

from braid import __version__
from braid.output_safety import csv_safe
from braid.pipeline import AssemblyPipeline, PipelineConfig

logger = logging.getLogger(__name__)

_PUBLIC_COMMANDS = ("psi", "differential", "filter", "example", "doctor")
_HIDDEN_LEGACY_COMMANDS = (
    "run",
    "assemble",
    "analyze",
    "denovo",
    "dashboard",
    "target",
    "fastq-target",
)


def _hide_subparser_from_help(
    subparsers: argparse._SubParsersAction,
    command: str,
) -> None:
    """Keep a legacy subcommand accepted without advertising it in top help."""
    # argparse has no public hidden-but-accepted subcommand API. The parser
    # choice remains registered; only the top-level help row is removed.
    help_actions = getattr(subparsers, "_choices_actions", [])
    setattr(
        subparsers,
        "_choices_actions",
        [action for action in help_actions if action.dest != command],
    )


def _add_assembly_args(parser: argparse.ArgumentParser) -> None:
    """Add assembly-specific arguments to a parser.

    Args:
        parser: The argparse parser or subparser.
    """
    parser.add_argument(
        "bam",
        nargs="+",
        help="Input BAM file(s) (coordinate-sorted, indexed). "
             "Multiple BAMs are assembled independently; a summary "
             "table is written when more than one BAM is provided.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path. For single BAM: GTF file (default: braid_output.gtf). "
             "For multiple BAMs: output directory (default: braid_output/).",
    )
    parser.add_argument(
        "-f", "--format",
        default="gtf",
        choices=["gtf", "gff3"],
        dest="output_format",
        help="Output format: gtf or gff3 (default: gtf).",
    )
    parser.add_argument(
        "-r", "--reference",
        default=None,
        help="Reference genome FASTA file (with .fai index).",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Compute backend: auto, cpu, or gpu (default: auto).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Force GPU backend (shorthand for --backend gpu).",
    )
    parser.add_argument(
        "-t", "--threads",
        type=int,
        default=1,
        help="Number of threads for parallel processing (default: 1).",
    )
    parser.add_argument(
        "-q", "--min-mapq",
        type=int,
        default=0,
        help="Minimum mapping quality (default: 0).",
    )
    parser.add_argument(
        "--fast-read-count",
        action="store_true",
        default=False,
        help="Use the BAM index mapped total for the startup read count, skipping "
             "the full filtered scan (faster on very large BAMs, but a fast upper "
             "bound that over-counts secondary/supplementary reads, not exact).",
    )
    parser.add_argument(
        "-j", "--min-junction-support",
        type=int,
        default=2,
        help="Minimum junction support reads (default: 2).",
    )
    parser.add_argument(
        "--min-phasing-support",
        type=int,
        default=1,
        help="Minimum validated phasing-path support reads (default: 1).",
    )
    parser.add_argument(
        "-c", "--min-coverage",
        type=float,
        default=1.0,
        help="Minimum transcript coverage (default: 1.0).",
    )
    parser.add_argument(
        "-s", "--min-score",
        type=float,
        default=0.3,
        help="Minimum transcript score (default: 0.3).",
    )
    parser.add_argument(
        "--max-intron-length",
        type=int,
        default=500_000,
        help="Maximum intron length in base pairs (default: 500000).",
    )
    parser.add_argument(
        "--min-anchor-length",
        type=int,
        default=8,
        help="Minimum aligned anchor length flanking each junction (default: 8).",
    )
    parser.add_argument(
        "--max-paths",
        type=int,
        default=2000,
        help="Maximum source-to-sink paths considered by decomposition (default: 2000, "
             "matching PipelineConfig/DecomposeConfig).",
    )
    parser.add_argument(
        "--max-terminal-exon",
        type=int,
        default=5000,
        help=(
            "Maximum terminal exon length in bp for multi-exon "
            "transcripts (default: 5000). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--no-adaptive-junction-filter",
        action="store_true",
        default=False,
        help="Disable depth-adaptive junction pre-filtering.",
    )
    parser.add_argument(
        "--no-motif-validation",
        action="store_true",
        default=False,
        help="Disable reference-based splice-motif validation during junction extraction.",
    )
    parser.add_argument(
        "--no-safe-paths",
        action="store_true",
        default=False,
        help="Disable safe path decomposition.",
    )
    parser.add_argument(
        "--no-ml-scoring",
        action="store_true",
        default=False,
        help="Disable ML scoring, use heuristic instead.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to a trusted pre-trained scoring model (joblib file). Loading "
             "deserializes/executes code from the file, so it is refused unless you "
             "opt in with BRAID_ALLOW_UNTRUSTED_MODEL=1; use only models you trust.",
    )
    parser.add_argument(
        "--chromosomes",
        default=None,
        help="Comma-separated list of chromosomes to process.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        default=None,
        help="Optional directory for chromosome/locus diagnostics JSONL output.",
    )
    parser.add_argument(
        "--engine",
        default="braid_v2",
        choices=["legacy", "iterative_v2", "sota", "braid_v2"],
        help="Primary assembly engine to use (default: braid_v2).",
    )
    parser.add_argument(
        "--decomposer",
        dest="engine",
        choices=["legacy", "iterative_v2", "sota", "braid_v2"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--shadow-engine",
        default=None,
        choices=["legacy", "iterative_v2", "sota", "braid_v2"],
        help="Optional shadow engine for diagnostics-only comparison.",
    )
    parser.add_argument(
        "--shadow-decomposer",
        dest="shadow_engine",
        choices=["legacy", "iterative_v2", "sota", "braid_v2"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=256,
        help="Maximum number of braid_v2 candidate paths retained per locus (default: 256).",
    )
    parser.add_argument(
        "--candidate-beam-width",
        type=int,
        default=32,
        help="Per-node beam width for braid_v2 candidate search (default: 32).",
    )
    parser.add_argument(
        "--complexity-penalty",
        type=float,
        default=0.05,
        help="Relative pruning threshold for braid_v2 sparse fitting (default: 0.05).",
    )
    parser.add_argument(
        "--relaxed-pruning",
        action="store_true",
        default=False,
        help="Relax early junction/exon pruning for correctness experiments.",
    )
    parser.add_argument(
        "--builder-profile",
        default="default",
        choices=["default", "conservative_correctness", "aggressive_recall"],
        help=(
            "Builder pruning profile. 'conservative_correctness' softens early "
            "filters while preserving structure; 'aggressive_recall' disables "
            "most early pruning. --relaxed-pruning is kept as a legacy alias "
            "for aggressive_recall."
        ),
    )
    parser.add_argument(
        "--no-single-exon",
        action="store_true",
        default=False,
        help="Disable single-exon gene detection (reduces false positives).",
    )
    parser.add_argument(
        "--strandedness",
        choices=["none", "rf", "fr"],
        default="none",
        help="Library strand protocol: none (default), rf (dUTP), fr.",
    )
    parser.add_argument(
        "--nmf",
        action="store_true",
        default=False,
        help="Use NMF-based read-fragment matrix decomposition.",
    )
    parser.add_argument(
        "--guide-gtf",
        default=None,
        help="Long-read transcript GTF for guided assembly (e.g., PacBio Iso-Seq).",
    )
    parser.add_argument(
        "--guide-tolerance",
        type=int,
        default=5,
        help="Exon boundary tolerance in bp for guide path matching (default: 5).",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Enable bootstrap confidence intervals for transcript abundances.",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=100,
        help="Number of bootstrap resampling replicates (default: 100).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output (DEBUG-level logging).",
    )


def _create_assemble_parser() -> argparse.ArgumentParser:
    """Create the standalone assemble parser for backward compatibility.

    Returns:
        A fully configured parser for assembly-only usage.
    """
    parser = argparse.ArgumentParser(
        prog="braid",
        description=(
            "BRAID: RNA-seq splicing confidence and assembly toolkit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_assembly_args(parser)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with subcommands for the BRAID CLI.

    Returns:
        A fully configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog="braid",
        description=(
            "BRAID: calibrated confidence intervals and confidence tiers for "
            "rMATS outputs plus caller-agnostic differential tables."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  braid differential --rmats-dir rMATS_output/ -o differential.tsv\n"
            "  braid psi --rmats-dir rMATS_output/ -o psi.tsv\n"
            "  braid filter --caller majiq deltapsi.tsv -o braid_filter\n"
            "  braid example\n"
            "  braid doctor\n"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="{" + ",".join(_PUBLIC_COMMANDS) + "}",
    )

    # --- run subcommand (unified pipeline) ---
    from braid.commands.run import add_run_subparser
    add_run_subparser(subparsers)

    # --- psi subcommand (NEW) ---
    from braid.commands.psi import add_psi_subparser
    add_psi_subparser(subparsers)

    # --- differential subcommand (NEW) ---
    from braid.commands.differential import add_differential_subparser
    add_differential_subparser(subparsers)

    # --- filter subcommand: caller-agnostic calibration layer ---
    from braid.commands.filter_cmd import add_filter_subparser
    add_filter_subparser(subparsers)

    # --- example subcommand (zero-setup demo) ---
    from braid.commands.example import add_example_subparser
    add_example_subparser(subparsers)

    # --- assemble subcommand ---
    assemble_parser = subparsers.add_parser(
        "assemble",
        help=argparse.SUPPRESS,
    )
    _add_assembly_args(assemble_parser)

    # --- analyze subcommand ---
    analyze_parser = subparsers.add_parser(
        "analyze",
        help=argparse.SUPPRESS,
    )
    analyze_parser.add_argument(
        "gtf",
        help="Input GTF file with assembled transcripts.",
    )
    analyze_parser.add_argument(
        "bam",
        help="Input BAM file for junction read counts.",
    )
    analyze_parser.add_argument(
        "-o", "--output",
        default="braid_events.tsv",
        help="Output events TSV file (default: braid_events.tsv).",
    )
    analyze_parser.add_argument(
        "--ioe",
        default=None,
        help="Optional output IOE file (SUPPA2-compatible format).",
    )
    analyze_parser.add_argument(
        "--min-reads",
        type=int,
        default=10,
        help="Minimum junction reads for PSI significance (default: 10).",
    )
    analyze_parser.add_argument(
        "-q", "--min-mapq",
        type=int,
        default=0,
        help="Minimum mapping quality for junction extraction (default: 0).",
    )
    analyze_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output.",
    )

    # --- denovo subcommand ---
    denovo_parser = subparsers.add_parser(
        "denovo",
        help=argparse.SUPPRESS,
    )
    denovo_parser.add_argument(
        "fastq",
        nargs="+",
        help="Input FASTQ file(s) (plain or .gz).",
    )
    denovo_parser.add_argument(
        "-o", "--output",
        default="denovo_transcripts.fa",
        help="Output FASTA file (default: denovo_transcripts.fa).",
    )
    denovo_parser.add_argument(
        "-k", "--kmer-size",
        type=int,
        default=25,
        help="K-mer size for de Bruijn graph (default: 25, max: 31).",
    )
    denovo_parser.add_argument(
        "--min-kmer-count",
        type=int,
        default=3,
        help="Minimum k-mer count to retain (default: 3).",
    )
    denovo_parser.add_argument(
        "--min-length",
        type=int,
        default=200,
        help="Minimum transcript length in bases (default: 200).",
    )
    denovo_parser.add_argument(
        "--min-coverage",
        type=float,
        default=2.0,
        help="Minimum path coverage for transcript output (default: 2.0).",
    )
    denovo_parser.add_argument(
        "--no-canonical",
        action="store_true",
        default=False,
        help="Disable canonical k-mers (strand-specific mode).",
    )
    denovo_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output.",
    )

    # --- dashboard subcommand ---
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help=argparse.SUPPRESS,
    )
    dashboard_parser.add_argument(
        "events_tsv",
        help="Events TSV from `braid analyze` (gene_id/psi/event_type schema). "
             "`braid psi`/`braid differential` outputs use a different schema "
             "and are not dashboard inputs.",
    )
    dashboard_parser.add_argument(
        "gtf",
        help="GTF file with assembled transcripts.",
    )
    dashboard_parser.add_argument(
        "--bam",
        default=None,
        help="Optional BAM file for Sashimi plot coverage.",
    )
    dashboard_parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port (default: 8501).",
    )
    dashboard_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output.",
    )

    # --- doctor subcommand ---
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check whether BRAID and its optional tools are installed.",
    )
    doctor_parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Require optional benchmark tools to be present as well.",
    )
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print a machine-readable JSON install report.",
    )
    doctor_parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Require the optional GPU stack (cupy) to be installed cleanly.",
    )

    # --- target subcommand ---
    target_parser = subparsers.add_parser(
        "target",
        help=argparse.SUPPRESS,
    )
    target_parser.add_argument(
        "bam",
        help="Input BAM file (coordinate-sorted, indexed).",
    )
    target_parser.add_argument(
        "--gene",
        default=None,
        help="Gene name to assemble (e.g. TP53, BRCA1). Requires --gtf.",
    )
    target_parser.add_argument(
        "--region",
        default=None,
        help="Genomic region (e.g. chr17:7668402-7687538). Alternative to --gene.",
    )
    target_parser.add_argument(
        "--gtf",
        default=None,
        help="GTF annotation for gene lookup (required with --gene).",
    )
    target_parser.add_argument(
        "-r", "--reference",
        default=None,
        help="Reference genome FASTA for splice motif validation.",
    )
    target_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file (default: stdout).",
    )
    target_parser.add_argument(
        "--format",
        choices=["text", "gtf", "json"],
        default="text",
        help="Output format (default: text).",
    )
    target_parser.add_argument(
        "--flank",
        type=int,
        default=1000,
        help="Flanking bp around target region (default: 1000).",
    )
    target_parser.add_argument(
        "--max-paths",
        type=int,
        default=5000,
        help="Maximum paths to enumerate (default: 5000).",
    )
    target_parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=200,
        help="Bootstrap replicates for confidence intervals (default: 200).",
    )
    target_parser.add_argument(
        "--min-presence",
        type=float,
        default=0.3,
        help="Minimum bootstrap presence rate to report (default: 0.3).",
    )
    target_parser.add_argument(
        "--strandedness",
        choices=["none", "rf", "fr"],
        default="none",
        help="Library strand protocol: none (default), rf (dUTP/fr-firststrand), fr.",
    )
    target_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output.",
    )

    # --- fastq-target subcommand ---
    fq_parser = subparsers.add_parser(
        "fastq-target",
        help=argparse.SUPPRESS,
    )
    fq_parser.add_argument(
        "fastq",
        nargs="+",
        help="FASTQ file(s) (1 for SE, 2 for PE).",
    )
    fq_parser.add_argument(
        "--gene",
        required=True,
        help="Target gene name (e.g. TP53).",
    )
    fq_parser.add_argument(
        "--gtf",
        required=True,
        help="GTF annotation file for gene lookup.",
    )
    fq_parser.add_argument(
        "-r", "--reference",
        required=True,
        help="Reference genome FASTA.",
    )
    fq_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file (default: stdout).",
    )
    fq_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text).",
    )
    fq_parser.add_argument(
        "--flank-genes",
        type=int,
        default=5,
        help="Number of flanking genes on each side (default: 5).",
    )
    fq_parser.add_argument(
        "--flank-bp",
        type=int,
        default=10000,
        help="Additional flanking bp (default: 10000).",
    )
    fq_parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=500,
        help="Bootstrap replicates (default: 500).",
    )
    fq_parser.add_argument(
        "-t", "--threads",
        type=int,
        default=4,
        help="Alignment threads (default: 4).",
    )
    fq_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output.",
    )

    for command in _HIDDEN_LEGACY_COMMANDS:
        _hide_subparser_from_help(subparsers, command)

    return parser


def _parse_chromosomes(raw: str | None) -> list[str] | None:
    """Parse a comma-separated chromosome list string.

    Args:
        raw: Comma-separated chromosome names, or None.

    Returns:
        List of chromosome name strings, or None.
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    result = [c.strip() for c in stripped.split(",") if c.strip()]
    return result or None


def _build_pipeline_config(
    bam_path: str,
    args: argparse.Namespace,
    output_path: str,
) -> PipelineConfig:
    """Build a PipelineConfig for one BAM file."""
    chromosomes = _parse_chromosomes(getattr(args, "chromosomes", None))
    backend = getattr(args, "backend", "auto")
    if getattr(args, "gpu", False):
        backend = "gpu"
    decomposer = getattr(
        args, "engine", getattr(args, "decomposer", "braid_v2"),
    )
    shadow_decomposer = getattr(
        args, "shadow_engine", getattr(args, "shadow_decomposer", None),
    )
    if getattr(args, "nmf", False):
        if decomposer in {"sota", "braid_v2"}:
            decomposer = "legacy"
        elif decomposer != "legacy":
            raise SystemExit(
                "--nmf cannot be combined with --engine other than legacy, sota, or braid_v2."
            )
        if shadow_decomposer is not None:
            raise SystemExit(
                "--nmf cannot be combined with --shadow-engine."
            )
    return PipelineConfig(
        bam_path=bam_path,
        reference_path=getattr(args, "reference", None),
        output_path=output_path,
        output_format=getattr(args, "output_format", "gtf"),
        backend=backend,
        threads=getattr(args, "threads", 1),
        min_mapq=getattr(args, "min_mapq", 0),
        fast_read_count=getattr(args, "fast_read_count", False),
        min_junction_support=getattr(args, "min_junction_support", 2),
        min_phasing_support=getattr(args, "min_phasing_support", 1),
        min_coverage=getattr(args, "min_coverage", 1.0),
        min_transcript_score=getattr(args, "min_score", 0.3),
        max_intron_length=getattr(args, "max_intron_length", 500_000),
        min_anchor_length=getattr(args, "min_anchor_length", 8),
        max_paths=getattr(args, "max_paths", 2000),
        use_safe_paths=not getattr(args, "no_safe_paths", False),
        use_ml_scoring=not getattr(args, "no_ml_scoring", False),
        model_path=getattr(args, "model", None),
        chromosomes=chromosomes,
        verbose=getattr(args, "verbose", False),
        use_nmf_decomposition=getattr(args, "nmf", False),
        diagnostics_dir=getattr(args, "diagnostics_dir", None),
        decomposer=decomposer,
        shadow_decomposer=shadow_decomposer,
        candidate_budget=getattr(args, "candidate_budget", 256),
        candidate_beam_width=getattr(args, "candidate_beam_width", 32),
        complexity_penalty=getattr(args, "complexity_penalty", 0.05),
        max_terminal_exon_length=getattr(args, "max_terminal_exon", 5000),
        adaptive_junction_filter=not getattr(
            args, "no_adaptive_junction_filter", False,
        ),
        enable_motif_validation=not getattr(args, "no_motif_validation", False),
        relaxed_pruning_experiment=getattr(args, "relaxed_pruning", False),
        builder_profile=getattr(args, "builder_profile", "default"),
        disable_single_exon=getattr(args, "no_single_exon", False),
        strandedness=getattr(args, "strandedness", "none"),
        enable_bootstrap=getattr(args, "bootstrap", False),
        bootstrap_replicates=getattr(args, "bootstrap_replicates", 100),
        guide_gtf_path=getattr(args, "guide_gtf", None),
        guide_tolerance=getattr(args, "guide_tolerance", 5),
    )


def _parse_gtf_bootstrap_attributes(gtf_path: str) -> list[dict]:
    """Parse isoform-level attributes from a BRAID GTF with bootstrap info."""
    import os
    records: list[dict] = []
    if not os.path.exists(gtf_path):
        return records
    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            attrs = fields[8]
            rec: dict = {}
            for part in attrs.split(";"):
                part = part.strip()
                if not part:
                    continue
                key_val = part.split('"')
                if len(key_val) >= 2:
                    key = key_val[0].strip()
                    val = key_val[1].strip()
                    rec[key] = val
            records.append(rec)
    return records


def _write_summary_table(
    sample_results: dict[str, list[dict]],
    output_path: str,
) -> None:
    """Write a cross-sample summary TSV from per-sample GTF attributes."""
    import numpy as np

    # Collect all transcript IDs
    all_tids: dict[str, dict] = {}
    sample_names = list(sample_results.keys())

    for sample, records in sample_results.items():
        for rec in records:
            tid = rec.get("transcript_id", "")
            gid = rec.get("gene_id", "")
            if not tid:
                continue
            if tid not in all_tids:
                all_tids[tid] = {"transcript_id": tid, "gene_id": gid}
            all_tids[tid][f"{sample}_TPM"] = rec.get("TPM", "0")
            all_tids[tid][f"{sample}_cov"] = rec.get("cov", "0")
            all_tids[tid][f"{sample}_cv"] = rec.get("bootstrap_cv", "NA")
            all_tids[tid][f"{sample}_ci_low"] = rec.get("bootstrap_ci_low", "NA")
            all_tids[tid][f"{sample}_ci_high"] = rec.get("bootstrap_ci_high", "NA")
            all_tids[tid][f"{sample}_presence"] = rec.get("bootstrap_presence", "NA")

    # Compute cross-sample stats
    rows = []
    for tid, info in sorted(all_tids.items()):
        cvs = []
        for s in sample_names:
            cv_val = info.get(f"{s}_cv", "NA")
            if cv_val != "NA":
                try:
                    cvs.append(float(cv_val))
                except ValueError:
                    pass
        info["mean_cv"] = f"{np.mean(cvs):.4f}" if cvs else "NA"
        info["n_samples_detected"] = str(sum(
            1 for s in sample_names
            if info.get(f"{s}_TPM", "0") not in ("0", "0.0", "0.00", "NA")
        ))
        info["confident"] = "yes" if (cvs and np.mean(cvs) <= 0.2) else "no"
        rows.append(info)

    # Write TSV
    base_cols = ["transcript_id", "gene_id"]
    sample_cols = []
    for s in sample_names:
        sample_cols.extend([
            f"{s}_TPM", f"{s}_cv", f"{s}_ci_low", f"{s}_ci_high", f"{s}_presence",
        ])
    summary_cols = ["mean_cv", "n_samples_detected", "confident"]
    all_cols = base_cols + sample_cols + summary_cols

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\t".join(csv_safe(c) for c in all_cols) + "\n")
        for row in rows:
            f.write("\t".join(
                csv_safe(str(row.get(c, "NA"))) for c in all_cols
            ) + "\n")


def _run_assemble(args: argparse.Namespace) -> None:
    """Run the assembly pipeline from parsed arguments.

    Supports multiple BAM files: each is assembled independently,
    and a cross-sample summary table is generated.
    """
    import os

    bam_list = args.bam if isinstance(args.bam, list) else [args.bam]
    n_bams = len(bam_list)
    output = args.output or ("braid_output.gtf" if n_bams == 1 else "braid_output")

    # Output contract (per --help and the README): a single BAM writes a GTF
    # FILE at args.output; multiple BAMs write a DIRECTORY of per-sample GTFs
    # plus a cross-sample summary. Writing a directory for the single-BAM case
    # contradicts the documented file-path contract.
    if n_bams == 1:
        gtf_out = output
        parent = os.path.dirname(gtf_out)
        if parent:
            os.makedirs(parent, exist_ok=True)
        print(f"Assembling {os.path.basename(bam_list[0])} ({bam_list[0]})...")
        config = _build_pipeline_config(bam_list[0], args, gtf_out)
        pipeline = AssemblyPipeline(config)
        pipeline.run()
        print(f"  → {gtf_out}")
        return

    outdir = output
    os.makedirs(outdir, exist_ok=True)

    sample_results: dict[str, list[dict]] = {}
    gtf_paths: list[str] = []

    for bam_path in bam_list:
        sample_name = os.path.splitext(os.path.basename(bam_path))[0]
        gtf_out = os.path.join(outdir, f"{sample_name}.gtf")
        gtf_paths.append(gtf_out)

        print(f"Assembling {sample_name} ({bam_path})...")
        config = _build_pipeline_config(bam_path, args, gtf_out)
        pipeline = AssemblyPipeline(config)
        pipeline.run()
        print(f"  → {gtf_out}")

        # Parse bootstrap attributes from GTF
        records = _parse_gtf_bootstrap_attributes(gtf_out)
        sample_results[sample_name] = records

    # Write summary table
    summary_path = os.path.join(outdir, "summary.tsv")
    _write_summary_table(sample_results, summary_path)

    n_total = sum(len(r) for r in sample_results.values())
    print(f"\nAll {n_bams} samples assembled.")
    print(f"Per-sample GTFs: {outdir}/*.gtf")
    print(f"Summary table:   {summary_path} ({n_total} isoform-sample entries)")


def _run_analyze(args: argparse.Namespace) -> None:
    """Run alternative splicing event detection from parsed arguments.

    Args:
        args: Parsed CLI arguments with gtf, bam, output, etc.
    """
    from braid.io.bam_reader import BamReader, extract_junctions_from_bam
    from braid.splicing.classifier import EventClassifier
    from braid.splicing.events import EventType, detect_all_events
    from braid.splicing.io import write_events_tsv, write_ioe
    from braid.splicing.psi import calculate_all_psi
    from braid.splicing.statistics import (
        add_confidence_intervals,
        psi_significance_filter,
    )

    _setup_logging(getattr(args, "verbose", False))

    # Read transcripts from GTF
    transcripts = _read_gtf_transcripts(args.gtf)
    if not transcripts:
        print("No transcripts found in GTF file.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(transcripts)} transcripts from {args.gtf}")

    # Detect AS events
    events = detect_all_events(transcripts)
    print(f"Detected {len(events)} alternative splicing events")

    if not events:
        write_events_tsv(args.output, [], [])
        print(f"No events found. Empty output written to: {args.output}")
        return

    ri_chroms = {
        event.chrom
        for event in events
        if event.event_type == EventType.RI
    }

    # Extract junction evidence from BAM for each chromosome
    chroms = sorted({e.chrom for e in events})
    junction_evidence_by_chrom = {}
    for chrom in chroms:
        junction_evidence_by_chrom[chrom], _ = extract_junctions_from_bam(
            args.bam, chrom, min_mapq=getattr(args, "min_mapq", 0),
        )

    read_data_by_chrom = None
    if ri_chroms:
        bam_reader = BamReader(
            args.bam,
            min_mapq=getattr(args, "min_mapq", 0),
        )
        read_data_by_chrom = {
            chrom: bam_reader.fetch_region(chrom)
            for chrom in sorted(ri_chroms)
        }

    # Calculate PSI
    psi_results = calculate_all_psi(
        events,
        junction_evidence_by_chrom,
        read_data_by_chrom=read_data_by_chrom,
    )
    add_confidence_intervals(psi_results)

    # Apply minimum-read significance filter.
    min_reads = int(getattr(args, "min_reads", 10))
    psi_results = psi_significance_filter(psi_results, min_reads=min_reads)
    keep_ids = {r.event_id for r in psi_results}
    events = [e for e in events if e.event_id in keep_ids]

    if not events:
        write_events_tsv(args.output, [], [])
        print(
            "All events were filtered out by significance thresholds "
            f"(min_reads={min_reads}). Empty output written to: {args.output}"
        )
        return

    # Score events
    classifier = EventClassifier()
    scores = classifier.score_batch(events, psi_results)

    # Write output
    write_events_tsv(args.output, events, psi_results, scores)
    print(f"Events written to: {args.output}")

    if args.ioe:
        write_ioe(args.ioe, events)
        print(f"IOE written to: {args.ioe}")


def _run_dashboard(args: argparse.Namespace) -> None:
    """Launch the interactive Streamlit dashboard.

    Args:
        args: Parsed CLI arguments with events_tsv, gtf, etc.
    """
    import importlib.resources
    import importlib.util
    import subprocess

    # Preflight: the dashboard is an optional extra. Fail fast with a clear
    # install hint rather than letting Streamlit's own "No module named
    # streamlit" message scroll past while the command still exits 0.
    if importlib.util.find_spec("streamlit") is None:
        print(
            "braid dashboard requires the optional dashboard dependencies "
            "(Streamlit, pandas, plotly). Install them with:\n"
            "    pip install -e \".[dashboard]\"",
            file=sys.stderr,
        )
        raise SystemExit(1)

    app_path = str(
        importlib.resources.files("braid.dashboard") / "app.py"
    )

    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", str(args.port),
        "--",
        "--events-tsv", args.events_tsv,
        "--gtf", args.gtf,
    ]
    if args.bam:
        cmd.extend(["--bam", args.bam])

    print(f"Launching dashboard on port {args.port}...")
    # Propagate a non-zero Streamlit exit so launch failures are not reported
    # as success to CI smoke tests / workflow managers.
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _run_denovo(args: argparse.Namespace) -> None:
    """Run de novo assembly from parsed arguments.

    Args:
        args: Parsed CLI arguments with fastq, output, kmer_size, etc.
    """
    from braid.denovo.pipeline import DeNovoConfig, run_denovo_assembly

    _setup_logging(getattr(args, "verbose", False))

    config = DeNovoConfig(
        fastq_paths=args.fastq,
        output_path=args.output,
        k=getattr(args, "kmer_size", 25),
        min_kmer_count=getattr(args, "min_kmer_count", 3),
        canonical_kmers=not getattr(args, "no_canonical", False),
        min_transcript_length=getattr(args, "min_length", 200),
        min_transcript_coverage=getattr(args, "min_coverage", 2.0),
    )

    transcripts, stats = run_denovo_assembly(config)
    print(
        f"De novo assembly complete: {stats.n_transcripts} transcripts, "
        f"N50={stats.n50}, time={stats.elapsed_seconds:.1f}s"
    )
    if transcripts:
        print(f"Output written to: {config.output_path}")


def _read_gtf_transcripts(gtf_path: str) -> list:
    """Parse transcript records from a GTF file.

    Args:
        gtf_path: Path to the GTF file.

    Returns:
        List of TranscriptRecord objects.
    """
    from braid.io.gtf_writer import TranscriptRecord

    transcripts_dict: dict[str, dict] = {}

    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            if feature_type not in ("transcript", "exon"):
                continue

            chrom = parts[0]
            start = int(parts[3]) - 1  # Convert to 0-based
            end = int(parts[4])       # Already exclusive in 0-based
            strand = parts[6]

            # Parse attributes
            attrs = {}
            for attr in parts[8].rstrip(";").split(";"):
                attr = attr.strip()
                if not attr:
                    continue
                if " " in attr:
                    key, val = attr.split(" ", 1)
                    attrs[key] = val.strip('"')
                elif "=" in attr:
                    key, val = attr.split("=", 1)
                    attrs[key] = val.strip('"')

            tid = attrs.get("transcript_id", "")
            gid = attrs.get("gene_id", "")

            if feature_type == "transcript":
                transcripts_dict[tid] = {
                    "transcript_id": tid,
                    "gene_id": gid,
                    "chrom": chrom,
                    "strand": strand,
                    "start": start,
                    "end": end,
                    "exons": [],
                }
            elif feature_type == "exon":
                if tid not in transcripts_dict:
                    transcripts_dict[tid] = {
                        "transcript_id": tid,
                        "gene_id": gid,
                        "chrom": chrom,
                        "strand": strand,
                        "start": start,
                        "end": end,
                        "exons": [],
                    }
                transcripts_dict[tid]["exons"].append((start, end))
                # Update transcript bounds
                if start < transcripts_dict[tid]["start"]:
                    transcripts_dict[tid]["start"] = start
                if end > transcripts_dict[tid]["end"]:
                    transcripts_dict[tid]["end"] = end

    results = []
    for info in transcripts_dict.values():
        exons = sorted(info["exons"])
        if not exons:
            continue
        try:
            results.append(TranscriptRecord(
                transcript_id=info["transcript_id"],
                gene_id=info["gene_id"],
                chrom=info["chrom"],
                strand=info["strand"],
                start=info["start"],
                end=info["end"],
                exons=exons,
            ))
        except ValueError:
            continue

    return results


def _setup_logging(verbose: bool) -> None:
    """Configure logging for CLI commands.

    Args:
        verbose: If True, set DEBUG level; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
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


def _run_target(args: argparse.Namespace) -> None:
    """Run targeted single-gene transcript assembly.

    Args:
        args: Parsed CLI arguments.
    """
    from braid.target.assembler import (
        TargetConfig,
        assemble_target,
        format_target_report,
    )
    from braid.target.extractor import lookup_gene, parse_region_string

    _setup_logging(getattr(args, "verbose", False))

    # Resolve target region
    gene = getattr(args, "gene", None)
    region_str = getattr(args, "region", None)
    gtf_path = getattr(args, "gtf", None)

    if gene and region_str:
        raise SystemExit(
            "--gene and --region are mutually exclusive; "
            "specify only one target selector: --gene or --region."
        )

    if gene and not gtf_path:
        raise SystemExit("--gene requires --gtf for gene coordinate lookup.")

    if gene:
        region = lookup_gene(gtf_path, gene)
        if region is None:
            raise SystemExit(f"Gene {gene!r} not found in {gtf_path}")
    elif region_str:
        region = parse_region_string(region_str)
    else:
        raise SystemExit("Either --gene or --region must be specified.")

    config = TargetConfig(
        bam_path=args.bam,
        reference_path=getattr(args, "reference", None),
        region=region,
        flank=getattr(args, "flank", 1000),
        max_paths=getattr(args, "max_paths", 5000),
        bootstrap_replicates=getattr(args, "bootstrap_replicates", 200),
        min_presence_rate=getattr(args, "min_presence", 0.3),
        strandedness=getattr(args, "strandedness", "none"),
        annotation_gtf=getattr(args, "gtf", None),
    )

    result = assemble_target(config)

    output_format = getattr(args, "format", "text")
    output_path = getattr(args, "output", None)

    if output_format == "text":
        report = format_target_report(result)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(report + "\n")
            print(f"Report written to: {output_path}")
        else:
            print(report)
    elif output_format == "gtf":
        import math

        from braid.io.gtf_writer import GtfWriter, TranscriptRecord

        def _finite_or_none(value: float | None) -> float | None:
            if value is None:
                return None
            value = float(value)
            return value if math.isfinite(value) else None

        records = []
        for iso in result.isoforms:
            rec = TranscriptRecord(
                transcript_id=iso.transcript_id,
                gene_id=region.gene_name or "TARGET",
                chrom=region.chrom,
                strand=iso.strand,
                start=iso.exons[0][0],
                end=iso.exons[-1][1],
                exons=iso.exons,
                score=iso.score,
                coverage=iso.weight,
                bootstrap_ci_low=iso.ci_low if iso.ci_low is not None else None,
                bootstrap_ci_high=iso.ci_high if iso.ci_high is not None else None,
                bootstrap_presence=(
                    iso.presence_rate if iso.presence_rate is not None else None
                ),
                bootstrap_cv=iso.cv if iso.cv is not None else None,
            )
            records.append(rec)
        out = output_path or "target_assembly.gtf"
        writer = GtfWriter(out, source="TargetSplice")
        writer.write_transcripts(records)
        print(f"GTF written to: {out}")
    elif output_format == "json":
        import json
        data = {
            "region": {
                "chrom": region.chrom,
                "start": region.start,
                "end": region.end,
                "strand": region.strand,
                "gene": region.gene_name,
            },
            "n_isoforms": result.n_isoforms,
            "n_confident": result.n_confident,
            "assembly_time": result.assembly_time_seconds,
            "bootstrap_time": result.bootstrap_time_seconds,
            "isoforms": [
                {
                    "id": iso.transcript_id,
                    "exons": iso.exons,
                    "weight": iso.weight,
                    "score": iso.score,
                    "ci_low": iso.ci_low,
                    "ci_high": iso.ci_high,
                    "presence_rate": iso.presence_rate,
                    "cv": iso.cv,
                    "n_junctions": iso.n_junctions,
                }
                for iso in result.isoforms
            ],
        }
        text = json.dumps(data, indent=2)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"JSON written to: {output_path}")
        else:
            print(text)


def _run_fastq_target(args: argparse.Namespace) -> None:
    """Run FASTQ-to-target assembly pipeline.

    Args:
        args: Parsed CLI arguments.
    """
    from braid.target.fastq_pipeline import (
        FastqTargetConfig,
        format_fastq_target_report,
        run_fastq_target_pipeline,
    )

    _setup_logging(getattr(args, "verbose", False))

    config = FastqTargetConfig(
        fastq_paths=args.fastq,
        reference_path=args.reference,
        annotation_gtf=args.gtf,
        gene=args.gene,
        flank_genes=getattr(args, "flank_genes", 5),
        flank_bp=getattr(args, "flank_bp", 10000),
        bootstrap_replicates=getattr(args, "bootstrap_replicates", 500),
        threads=getattr(args, "threads", 4),
    )

    result = run_fastq_target_pipeline(config)

    output_format = getattr(args, "format", "text")
    output_path = getattr(args, "output", None)

    if output_format == "text":
        report = format_fastq_target_report(result)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as fh:
                fh.write(report + "\n")
            print(f"Report written to: {output_path}")
        else:
            print(report)
    elif output_format == "json":
        import json
        data = {
            "gene": result.gene,
            "region": f"{result.region_chrom}:{result.region_start}-{result.region_end}",
            "region_length": result.region_length,
            "reads_aligned": result.n_reads_aligned,
            "reads_spliced": result.n_reads_spliced,
            "alignment_time": result.alignment_time,
            "assembly_time": result.assembly_time,
            "bootstrap_time": result.bootstrap_time,
            "flanking_genes": result.flanking_genes,
            "isoforms": [
                {
                    "id": iso.transcript_id,
                    "exons": iso.exons,
                    "weight": iso.weight,
                    "ci_low": iso.ci_low,
                    "ci_high": iso.ci_high,
                    "presence_rate": iso.presence_rate,
                    "cv": iso.cv,
                }
                for iso in result.isoforms
            ],
        }
        text = json.dumps(data, indent=2)
        if output_path:
            with open(output_path, "w") as fh:
                fh.write(text + "\n")
        else:
            print(text)


def _run_doctor(args: argparse.Namespace) -> None:
    """Run the installation diagnostics command.

    Args:
        args: Parsed CLI arguments.
    """
    from braid.doctor import report_install_check

    exit_code = report_install_check(
        strict=getattr(args, "strict", False),
        gpu=getattr(args, "gpu", False),
        json_output=getattr(args, "json", False),
    )
    raise SystemExit(exit_code)


def main() -> None:
    """Parse command-line arguments and dispatch to the appropriate subcommand.

    This is the main entry point invoked by the ``braid`` console script.
    Supported public subcommands: psi, differential, filter, example, and doctor.
    Legacy subcommands remain accepted for backward compatibility but are not
    advertised in the top-level help.

    When the first positional argument is not a recognised subcommand and
    looks like a file path (contains ``/`` or ``.``), the legacy assemble
    parser is used for backward compatibility.  Otherwise an error is
    printed directing the user to ``braid -h``.
    """
    # Handle backward compatibility: detect if first arg is a subcommand
    # or a BAM file path
    known_commands = {
        "assemble", "analyze", "denovo", "dashboard", "doctor", "target",
        "fastq-target", "psi", "differential", "diff", "filter", "run", "example",
        "-h", "--help", "--version",
    }
    if len(sys.argv) > 1 and sys.argv[1] not in known_commands:
        first_arg = sys.argv[1]
        # Only fall through to legacy assemble when the first arg is an
        # alignment file.  Dotty typos like ``run.py`` should remain unknown
        # commands instead of being reported as missing BAM inputs.
        if not first_arg.lower().endswith((".bam", ".cram", ".sam")):
            print(
                f"Unknown command: {first_arg}. "
                "Run 'braid -h' for help.",
                file=sys.stderr,
            )
            sys.exit(1)
        # Looks like a BAM/CRAM file path — use legacy parser
        legacy_parser = _create_assemble_parser()
        args = legacy_parser.parse_args()
        try:
            _run_assemble(args)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as exc:
            print(f"Runtime error: {exc}", file=sys.stderr)
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user.", file=sys.stderr)
            sys.exit(130)
        except Exception as exc:
            logger.exception("Unexpected error")
            print(f"Unexpected error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == "assemble":
            _run_assemble(args)
        elif args.command == "analyze":
            _run_analyze(args)
        elif args.command == "denovo":
            _run_denovo(args)
        elif args.command == "dashboard":
            _run_dashboard(args)
        elif args.command == "doctor":
            _run_doctor(args)
        elif args.command == "target":
            _run_target(args)
        elif args.command == "fastq-target":
            _run_fastq_target(args)
        elif args.command in ("psi", "differential", "diff", "filter", "run", "example"):
            args.func(args)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as exc:
        print(f"Runtime error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        # Input/validation errors (e.g. effect-cutoff < 0, malformed rMATS row) are
        # user-facing: fail short and clear, not as an "Unexpected error" traceback.
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        logger.exception("Unexpected error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
