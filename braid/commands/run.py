"""BRAID unified ``run`` subcommand with auto-detected mode.

Usage examples:
    # Case 1: BAM only -> transcript assembly + CI
    braid run sample1.bam sample2.bam -o results/

    # Case 2: BAM + StringTie GTF -> isoform CI
    braid run *.bam --stringtie merged.gtf -o results/

    # Case 3: BAM + rMATS -> event PSI CI
    braid run *.bam --rmats rMATS_output/ -o results/

    # Case 4: Two groups + rMATS -> differential + tiers
    braid run --ctrl c1.bam c2.bam --treat kd.bam --rmats rMATS_output/ -o results/
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

Mode = Literal["differential", "psi", "score", "assemble"]



def _collect_bams(args: argparse.Namespace, mode: Mode) -> list[str]:
    """Collect all BAM files from arguments, validating presence.

    Args:
        args: Parsed CLI arguments.
        mode: The detected run mode.

    Returns:
        List of BAM file paths.

    Raises:
        SystemExit: If no BAM files are available when required.
    """
    if mode == "differential":
        return list(args.ctrl) + list(args.treat)

    bams = list(getattr(args, "bam", None) or [])
    if not bams:
        raise SystemExit(
            f"Error: at least one BAM file is required for {mode} mode."
        )
    return bams


def _run_differential_mode(args: argparse.Namespace, output_dir: str) -> None:
    """Delegate to the existing differential subcommand logic.

    Args:
        args: Parsed CLI arguments.
        output_dir: Output directory.
    """
    from braid.commands.differential import run_differential

    # Build a namespace compatible with the differential subcommand
    diff_args = argparse.Namespace(
        ctrl=args.ctrl,
        treat=args.treat,
        rmats_dir=args.rmats,
        output=os.path.join(output_dir, "differential.tsv"),
        replicates=args.replicates,
        confidence=args.confidence,
        effect_cutoff=0.1,
        min_support=20,
        seed=args.seed,
        verbose=getattr(args, "verbose", False),
    )
    run_differential(diff_args)


def _run_psi_mode(
    args: argparse.Namespace,
    bams: list[str],
    output_dir: str,
) -> None:
    """Delegate to the existing psi subcommand logic.

    Args:
        args: Parsed CLI arguments.
        bams: List of BAM file paths.
        output_dir: Output directory.
    """
    from braid.commands.psi import run_psi

    psi_args = argparse.Namespace(
        bam=bams,
        rmats_dir=args.rmats,
        gtf=getattr(args, "gtf", None),
        gene=None,
        region=None,
        output=os.path.join(output_dir, "psi.tsv"),
        replicates=args.replicates,
        confidence=args.confidence,
        min_support=10,
        seed=args.seed,
        verbose=getattr(args, "verbose", False),
    )
    run_psi(psi_args)


def _run_score_mode(
    args: argparse.Namespace,
    bams: list[str],
    output_dir: str,
) -> None:
    """Run StringTie isoform scoring with bootstrap CI.

    Args:
        args: Parsed CLI arguments.
        bams: List of BAM file paths.
        output_dir: Output directory.
    """
    from braid.target.stringtie_bootstrap import (
        STBootstrapConfig,
        run_stringtie_bootstrap,
    )

    for bam_path in bams:
        sample_name = os.path.splitext(os.path.basename(bam_path))[0]
        print(f"Scoring isoforms for {sample_name} ({bam_path})...")

        config = STBootstrapConfig(
            stringtie_gtf=args.stringtie,
            bam_path=bam_path,
            n_replicates=args.replicates,
            confidence_level=args.confidence,
            seed=args.seed,
        )
        results = run_stringtie_bootstrap(config)

        out_path = os.path.join(output_dir, f"{sample_name}_isoform_ci.tsv")
        _write_score_results(results, out_path)
        print(f"  -> {out_path} ({sum(r.n_isoforms for r in results)} isoforms)")


def _write_score_results(
    results: list,
    output_path: str,
) -> None:
    """Write StringTie bootstrap scoring results to TSV.

    Args:
        results: List of GeneBootstrapResult objects.
        output_path: Output TSV path.
    """
    header = [
        "gene_id", "transcript_id", "chrom", "strand", "n_exons",
        "stringtie_cov", "stringtie_tpm",
        "nnls_weight", "ci_low", "ci_high",
        "presence_rate", "cv", "is_stable",
    ]
    with open(output_path, "w") as fh:
        fh.write("\t".join(header) + "\n")
        for gene_result in results:
            for iso in gene_result.isoforms:
                row = [
                    gene_result.gene_id,
                    iso.transcript_id,
                    iso.chrom,
                    iso.strand,
                    str(iso.n_exons),
                    f"{iso.stringtie_cov:.2f}",
                    f"{iso.stringtie_tpm:.2f}",
                    f"{iso.nnls_weight:.4f}",
                    f"{iso.ci_low:.4f}",
                    f"{iso.ci_high:.4f}",
                    f"{iso.presence_rate:.4f}",
                    f"{iso.cv:.4f}",
                    "yes" if iso.is_stable else "no",
                ]
                fh.write("\t".join(row) + "\n")


def _run_assemble_mode(
    args: argparse.Namespace,
    bams: list[str],
    output_dir: str,
) -> None:
    """Run de novo assembly pipeline on BAM files.

    Delegates to the existing ``_run_assemble`` logic in ``braid.cli``.

    Args:
        args: Parsed CLI arguments.
        bams: List of BAM file paths.
        output_dir: Output directory.
    """
    from braid.cli import _run_assemble

    # Build a namespace compatible with the assemble subcommand
    assemble_args = argparse.Namespace(
        bam=bams,
        output=output_dir,
        output_format="gtf",
        reference=None,
        backend="auto",
        gpu=False,
        threads=1,
        min_mapq=0,
        min_junction_support=2,
        min_phasing_support=1,
        min_coverage=1.0,
        min_score=0.3,
        max_intron_length=500_000,
        min_anchor_length=8,
        max_paths=500,
        max_terminal_exon=5000,
        no_adaptive_junction_filter=False,
        no_motif_validation=False,
        no_safe_paths=False,
        no_ml_scoring=False,
        model=None,
        chromosomes=None,
        diagnostics_dir=None,
        decomposer="legacy",
        shadow_decomposer=None,
        relaxed_pruning=False,
        builder_profile="default",
        no_single_exon=False,
        strandedness="none",
        nmf=False,
        guide_gtf=None,
        guide_tolerance=5,
        bootstrap=True,
        bootstrap_replicates=args.replicates,
        verbose=getattr(args, "verbose", False),
    )
    _run_assemble(assemble_args)


def _detect_applicable_modes(args: argparse.Namespace) -> list[Mode]:
    """Detect ALL applicable run modes from CLI arguments.

    Multiple modes can be applicable when multiple inputs are given.
    For example, ``--ctrl --treat --rmats --stringtie`` triggers
    score, psi (skipped when groups present), and differential.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Ordered list of modes to execute sequentially.
    """
    has_ctrl = bool(getattr(args, "ctrl", None))
    has_treat = bool(getattr(args, "treat", None))
    has_rmats = bool(getattr(args, "rmats", None))
    has_stringtie = bool(getattr(args, "stringtie", None))

    # Validate group arguments
    if (has_ctrl or has_treat) and not (has_ctrl and has_treat):
        raise SystemExit(
            "Error: --ctrl and --treat must both be provided for "
            "differential mode."
        )
    if (has_ctrl and has_treat) and not has_rmats:
        raise SystemExit(
            "Error: --rmats is required for differential mode."
        )

    modes: list[Mode] = []

    # 1. StringTie given → score mode (isoform CI)
    if has_stringtie:
        modes.append("score")

    # 2. rMATS given without groups → psi mode (event CI)
    if has_rmats and not (has_ctrl and has_treat):
        modes.append("psi")

    # 3. ctrl/treat + rMATS → differential mode (ΔPSI + tiers)
    if has_ctrl and has_treat and has_rmats:
        modes.append("differential")

    # 4. Nothing else → assemble
    if not modes:
        modes.append("assemble")

    return modes


def run_main(args: argparse.Namespace) -> None:
    """Entry point for the unified ``run`` subcommand.

    Detects ALL applicable modes from arguments and runs them
    sequentially.  For example, providing ``--stringtie``,
    ``--rmats``, ``--ctrl``, and ``--treat`` will run score,
    then differential.

    Args:
        args: Parsed CLI arguments from ``add_run_subparser``.
    """
    verbose = getattr(args, "verbose", False)
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    modes = _detect_applicable_modes(args)
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"BRAID run: {len(modes)} mode(s) detected: {', '.join(modes)}")
    print(f"  Output dir: {output_dir}")
    has_ctrl = bool(getattr(args, "ctrl", None))
    has_treat = bool(getattr(args, "treat", None))
    if has_ctrl and has_treat:
        print(f"  Control:    {len(args.ctrl)} BAMs")
        print(f"  Treatment:  {len(args.treat)} BAMs")
    if getattr(args, "rmats", None):
        print(f"  rMATS dir:  {args.rmats}")
    if getattr(args, "stringtie", None):
        print(f"  StringTie:  {args.stringtie}")
    print()

    executed: list[str] = []

    for mode in modes:
        print(f"--- Running {mode} mode ---")
        bams = _collect_bams(args, mode)
        print(f"  BAM files: {len(bams)}")

        if mode == "differential":
            _run_differential_mode(args, output_dir)
        elif mode == "psi":
            _run_psi_mode(args, bams, output_dir)
        elif mode == "score":
            _run_score_mode(args, bams, output_dir)
        elif mode == "assemble":
            _run_assemble_mode(args, bams, output_dir)

        executed.append(mode)
        print()

    # Final summary
    print("=" * 50)
    print(f"BRAID run complete. Results in: {output_dir}/")
    print(f"Modes executed ({len(executed)}): {', '.join(executed)}")
    print("=" * 50)


def add_run_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``run`` subcommand with the main CLI parser.

    Args:
        subparsers: The subparsers action from the main argument parser.
    """
    parser = subparsers.add_parser(
        "run",
        help="Unified pipeline: auto-detects mode from inputs.",
        description=(
            "Unified BRAID pipeline that auto-detects the analysis mode:\n\n"
            "  - BAM only            -> assemble (de novo assembly + CI)\n"
            "  - BAM + --stringtie   -> score (isoform CI)\n"
            "  - BAM + --rmats       -> psi (event PSI CI)\n"
            "  - --ctrl/--treat + --rmats -> differential (DPSI + tiers)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  braid run sample1.bam sample2.bam -o results/\n"
            "  braid run *.bam --stringtie merged.gtf -o results/\n"
            "  braid run *.bam --rmats rMATS_output/ -o results/\n"
            "  braid run --ctrl c1.bam c2.bam --treat kd.bam "
            "--rmats rMATS_output/ -o results/\n"
        ),
    )

    # Positional: BAM files (optional when --ctrl/--treat used)
    parser.add_argument(
        "bam",
        nargs="*",
        help="Input BAM file(s). Optional when --ctrl/--treat are provided.",
    )

    # Group arguments for differential mode
    parser.add_argument(
        "--ctrl",
        nargs="+",
        default=None,
        help="Control group BAM file(s) (for differential mode).",
    )
    parser.add_argument(
        "--treat",
        nargs="+",
        default=None,
        help="Treatment group BAM file(s) (for differential mode).",
    )

    # Input annotations
    parser.add_argument(
        "--stringtie",
        default=None,
        help="StringTie GTF for isoform scoring mode.",
    )
    parser.add_argument(
        "--rmats",
        default=None,
        help="rMATS output directory for PSI or differential mode.",
    )
    parser.add_argument(
        "--gtf",
        default=None,
        help="Reference annotation GTF (GENCODE) for event proposal.",
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        default="braid_results",
        help="Output directory (default: braid_results/).",
    )

    # Bootstrap parameters
    parser.add_argument(
        "--replicates",
        type=int,
        default=500,
        help="Bootstrap replicates (default: 500).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    # Misc
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output (DEBUG-level logging).",
    )

    parser.set_defaults(func=run_main)
