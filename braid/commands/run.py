"""BRAID legacy ``run`` subcommand with auto-detected mode.

Usage examples::

    # Event PSI CI from rMATS
    braid run *.bam --rmats rMATS_output/ -o results/

    # Differential + tiers
    braid run --ctrl c1.bam c2.bam --treat kd.bam \\
        --rmats rMATS_output/ -o results/

PSI and differential modes compute posteriors from rMATS junction
count tables. BAM paths are recorded for provenance but counts are
not re-extracted from the BAM files.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

Mode = Literal["differential", "psi"]



def _collect_bams(args: argparse.Namespace, mode: Mode) -> list[str]:
    """Collect all BAM files from arguments, validating presence.

    Gathers BAMs from positional args first; if none, falls back to
    --ctrl + --treat BAMs for differential runs.

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
    # Fall back to --ctrl + --treat BAMs for non-differential modes
    if not bams:
        ctrl = list(getattr(args, "ctrl", None) or [])
        treat = list(getattr(args, "treat", None) or [])
        bams = ctrl + treat
    if not bams:
        # psi mode in `run` is only entered when --rmats is present, and psi reads
        # its counts from the rMATS tables (BAMs are optional provenance), so an
        # rMATS-only run must not hard-fail — consistent with `braid psi --rmats-dir`.
        if mode == "psi" and getattr(args, "rmats", None):
            return []
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
        effect_cutoff=getattr(args, "effect_cutoff", 0.1),
        fdr=getattr(args, "fdr", 0.05),
        min_support=(
            args.min_support
            if getattr(args, "min_support", None) is not None
            else 20
        ),
        seed=args.seed,
        verbose=getattr(args, "verbose", False),
        # Forward the calibration controls so `run --no-conformal`/`--calibration`
        # are honoured in differential mode (not silently ignored).
        use_conformal=getattr(args, "use_conformal", True),
        calibration=getattr(args, "calibration", None),
        differential_model=getattr(
            args,
            "differential_model",
            "auto" if getattr(args, "replicate_aware", True) else "sum",
        ),
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
        output=os.path.join(output_dir, "psi.tsv"),
        replicates=args.replicates,
        confidence=args.confidence,
        min_support=(
            args.min_support
            if getattr(args, "min_support", None) is not None
            else 10
        ),
        seed=args.seed,
        verbose=getattr(args, "verbose", False),
        use_conformal=getattr(args, "use_conformal", True),
        calibration=getattr(args, "calibration", None),
        allow_replicate_fallback=getattr(args, "allow_replicate_fallback", False),
    )
    run_psi(psi_args)


def _detect_applicable_modes(args: argparse.Namespace) -> list[Mode]:
    """Detect ALL applicable run modes from CLI arguments.

    Multiple modes can be applicable when multiple rMATS inputs are given.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Ordered list of modes to execute sequentially.
    """
    has_ctrl = bool(getattr(args, "ctrl", None))
    has_treat = bool(getattr(args, "treat", None))
    has_rmats = bool(getattr(args, "rmats", None))

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

    # 1. rMATS given without groups → psi mode (event CI)
    if has_rmats and not (has_ctrl and has_treat):
        modes.append("psi")

    # 2. ctrl/treat + rMATS → differential mode (ΔPSI + tiers)
    if has_ctrl and has_treat and has_rmats:
        modes.append("differential")

    if not modes:
        raise SystemExit("Error: --rmats is required for braid run.")

    return modes


def run_main(args: argparse.Namespace) -> None:
    """Entry point for the unified ``run`` subcommand.

    Detects ALL applicable rMATS modes from arguments and runs them
    sequentially.

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
            "  - --rmats [+ BAM]          -> psi (event PSI CI; counts from rMATS,\n"
            "                                BAMs optional provenance)\n"
            "  - --ctrl/--treat + --rmats -> differential (DPSI + tiers)\n\n"
            "BAMs from --ctrl/--treat are\n"
            "used for all modes when positional BAMs are not provided.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
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
        "--rmats",
        default=None,
        help="rMATS output directory for PSI or differential mode.",
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
        help="Bootstrap/posterior draws, not biological replicate count (default: 500).",
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
    parser.add_argument(
        "--min-support",
        type=int,
        default=None,
        help="Minimum read support for PSI/differential event filtering.",
    )
    parser.add_argument(
        "--effect-cutoff",
        type=float,
        default=0.1,
        help="Absolute dPSI cutoff for differential tiering (default: 0.1).",
    )
    parser.add_argument(
        "--fdr",
        type=float,
        default=0.05,
        help="rMATS FDR threshold for differential tiering (default: 0.05).",
    )
    parser.add_argument(
        "--no-conformal",
        dest="use_conformal",
        action="store_false",
        help="Disable conformal calibration (applies to both PSI and differential modes).",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Path to a conformal calibration JSON for PSI mode.",
    )
    parser.add_argument(
        "--allow-replicate-fallback",
        action="store_true",
        default=False,
        help="Allow rMATS/BAM replicate-count mismatch in PSI mode.",
    )
    parser.add_argument(
        "--differential-model",
        choices=["auto", "rep", "sum"],
        default="auto",
        help="Differential ΔPSI posterior model: auto, rep, or sum (default: auto).",
    )
    parser.add_argument(
        "--no-replicate-aware",
        dest="differential_model",
        action="store_const",
        const="sum",
        help="Compatibility alias for --differential-model sum.",
    )
    parser.set_defaults(use_conformal=True)

    # Misc
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose output (DEBUG-level logging).",
    )

    parser.set_defaults(func=run_main)
