"""BRAID command-line interface.

Usage:
    braid score --stringtie output.gtf [--bam sample.bam] [--reference genome.fa]
    braid psi --bam sample.bam --gene TP53 --gtf gencode.gtf
    braid target sample.bam --gene TP53 --gtf gencode.gtf --reference genome.fa
    braid dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

__version__ = "1.0.0"

logger = logging.getLogger("braid")


def _setup_logging(verbose: bool) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)
    root = logging.getLogger("braid")
    root.addHandler(handler)
    root.setLevel(level)


def create_parser() -> argparse.ArgumentParser:
    """Create the BRAID argument parser."""
    parser = argparse.ArgumentParser(
        prog="braid",
        description="BRAID: Bootstrap Resampling for Assembly Isoform Dependability",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # --- score subcommand ---
    score_p = sub.add_parser("score", help="Add bootstrap CI to StringTie isoforms.")
    score_p.add_argument("--stringtie", required=True, help="StringTie GTF output.")
    score_p.add_argument("--bam", default=None, help="BAM file (optional, for BAM-based mode).")
    score_p.add_argument("--reference", "-r", default=None, help="Reference FASTA.")
    score_p.add_argument("--replicates", "-R", type=int, default=200, help="Bootstrap replicates.")
    score_p.add_argument("--output", "-o", default=None, help="Output file.")
    score_p.add_argument("--format", choices=["text", "json"], default="text")
    score_p.add_argument("-v", "--verbose", action="store_true")

    # --- psi subcommand ---
    psi_p = sub.add_parser("psi", help="Single-sample PSI with bootstrap CI.")
    psi_p.add_argument("--bam", required=True, help="BAM file.")
    psi_p.add_argument("--gene", default=None, help="Gene name.")
    psi_p.add_argument("--region", default=None, help="Region (chr:start-end).")
    psi_p.add_argument("--gtf", default=None, help="GTF for gene lookup.")
    psi_p.add_argument("--replicates", "-R", type=int, default=500)
    psi_p.add_argument("--output", "-o", default=None)
    psi_p.add_argument("-v", "--verbose", action="store_true")

    # --- target subcommand ---
    target_p = sub.add_parser("target", help="Targeted single-gene assembly + CI.")
    target_p.add_argument("bam", help="BAM file.")
    target_p.add_argument("--gene", default=None, help="Gene name.")
    target_p.add_argument("--region", default=None, help="Region.")
    target_p.add_argument("--gtf", default=None, help="GTF annotation.")
    target_p.add_argument("--reference", "-r", default=None, help="Reference FASTA.")
    target_p.add_argument("--replicates", "-R", type=int, default=200)
    target_p.add_argument("--output", "-o", default=None)
    target_p.add_argument("--format", choices=["text", "json", "gtf"], default="text")
    target_p.add_argument("-v", "--verbose", action="store_true")

    # --- dashboard subcommand ---
    dash_p = sub.add_parser("dashboard", help="Launch interactive dashboard.")
    dash_p.add_argument("--port", type=int, default=8501)
    dash_p.add_argument("--pkl", default=None, help="Pre-computed results .pkl file.")

    return parser


def _run_score(args: argparse.Namespace) -> None:
    """Run bootstrap scoring on StringTie output."""
    from braid.core.stringtie_bootstrap import (
        STBootstrapConfig,
        format_bootstrap_report,
        run_stringtie_bootstrap,
    )

    config = STBootstrapConfig(
        stringtie_gtf=args.stringtie,
        bam_path=args.bam or "",
        reference_path=args.reference,
        n_replicates=args.replicates,
    )
    results = run_stringtie_bootstrap(config)
    report = format_bootstrap_report(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")
        print(f"Report written to: {args.output}")
    else:
        print(report)


def _run_psi(args: argparse.Namespace) -> None:
    """Run PSI bootstrap."""
    from braid.core.psi_bootstrap import (
        compute_psi_from_junctions,
        format_psi_report,
    )
    from braid.target.extractor import lookup_gene, parse_region_string

    if args.gene and args.gtf:
        region = lookup_gene(args.gtf, args.gene)
        if not region:
            print(f"Gene {args.gene!r} not found.", file=sys.stderr)
            sys.exit(1)
        chrom, start, end = region.chrom, region.start, region.end
    elif args.region:
        r = parse_region_string(args.region)
        chrom, start, end = r.chrom, r.start, r.end
    else:
        print("Either --gene or --region required.", file=sys.stderr)
        sys.exit(1)

    results = compute_psi_from_junctions(
        args.bam, chrom, start, end,
        n_replicates=args.replicates,
    )
    report = format_psi_report(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report + "\n")
    else:
        print(report)


def _run_dashboard(args: argparse.Namespace) -> None:
    """Launch Streamlit dashboard."""
    import subprocess
    import os

    app_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")
    cmd = ["streamlit", "run", app_path, "--server.port", str(args.port)]
    if args.pkl:
        cmd.extend(["--", "--pkl", args.pkl])
    subprocess.run(cmd)


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    verbose = getattr(args, "verbose", False)
    _setup_logging(verbose)

    try:
        if args.command == "score":
            _run_score(args)
        elif args.command == "psi":
            _run_psi(args)
        elif args.command == "target":
            # Reuse existing target assembler
            print("Target mode: use rapidsplice target for now.")
        elif args.command == "dashboard":
            _run_dashboard(args)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        logger.exception("Error")
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
