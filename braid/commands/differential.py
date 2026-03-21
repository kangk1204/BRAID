"""BRAID differential subcommand: two-group ΔPSI confidence intervals.

Usage:
    braid differential \
        --ctrl ctrl_rep1.bam ctrl_rep2.bam \
        --treat kd.bam \
        --rmats-dir rMATS_output/ \
        -o tiers.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


def add_differential_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``differential`` subcommand."""
    parser = subparsers.add_parser(
        "differential",
        help="Two-group differential splicing with calibrated confidence tiers.",
        description=(
            "Post-process rMATS output with BRAID confidence tiers. "
            "Computes per-event ΔPSI posterior CI and classifies events "
            "as supported or high-confidence."
        ),
    )
    parser.add_argument(
        "--ctrl", nargs="+", required=True,
        help="Control BAM file(s) (used for logging; counts are read from rMATS tables).",
    )
    parser.add_argument(
        "--treat", nargs="+", required=True,
        help="Treatment BAM file(s) (used for logging; counts are read from rMATS tables).",
    )
    parser.add_argument(
        "--rmats-dir", required=True,
        help="rMATS output directory.",
    )
    parser.add_argument(
        "-o", "--output", default="braid_differential.tsv",
        help="Output TSV (default: braid_differential.tsv).",
    )
    parser.add_argument(
        "--replicates", type=int, default=500,
        help="Posterior samples (default: 500).",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--effect-cutoff", type=float, default=0.1,
        help="|ΔPSI| effect-size cutoff (default: 0.1).",
    )
    parser.add_argument(
        "--min-support", type=int, default=20,
        help="Minimum total support per group (default: 20).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output.",
    )
    parser.set_defaults(func=run_differential)


def run_differential(args: argparse.Namespace) -> None:
    """Execute the differential subcommand."""
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    from braid.target.rmats_bootstrap import (
        get_group_counts,
        parse_rmats_output,
    )
    from braid.target.psi_bootstrap import sample_psi_posterior

    logger.info(
        "BRAID differential: %d ctrl BAMs, %d treat BAMs",
        len(args.ctrl), len(args.treat),
    )
    logger.info(
        "Note: PSI computed from rMATS junction count tables, not directly from BAM."
    )

    events = parse_rmats_output(
        args.rmats_dir,
        min_total_count=args.min_support,
    )
    logger.info("Parsed %d rMATS events", len(events))

    alpha = 1.0 - args.confidence
    rng = np.random.default_rng(args.seed)
    results = []

    for ev in events:
        ctrl_inc, ctrl_exc = get_group_counts(ev, sample="sample_1")
        treat_inc, treat_exc = get_group_counts(ev, sample="sample_2")

        ctrl_total = ctrl_inc + ctrl_exc
        treat_total = treat_inc + treat_exc

        if ctrl_total < args.min_support or treat_total < args.min_support:
            continue

        # Posterior samples for each group
        seed_val = int(rng.integers(0, 2**31))
        ctrl_samples = sample_psi_posterior(
            ctrl_inc, ctrl_exc,
            n_replicates=args.replicates,
            seed=seed_val,
            event_type=ev.event_type,
        )
        treat_samples = sample_psi_posterior(
            treat_inc, treat_exc,
            n_replicates=args.replicates,
            seed=seed_val + 1,
            event_type=ev.event_type,
        )

        # ΔPSI posterior
        dpsi_samples = treat_samples - ctrl_samples
        dpsi_mean = float(np.mean(dpsi_samples))
        dpsi_ci_low = float(np.percentile(dpsi_samples, 100 * alpha / 2))
        dpsi_ci_high = float(np.percentile(dpsi_samples, 100 * (1 - alpha / 2)))
        excludes_zero = dpsi_ci_low > 0 or dpsi_ci_high < 0
        prob_large = float(np.mean(np.abs(dpsi_samples) >= args.effect_cutoff))

        # Per-group PSI
        ctrl_psi = ctrl_inc / ctrl_total if ctrl_total > 0 else 0
        treat_psi = treat_inc / treat_total if treat_total > 0 else 0

        # Tiers
        rmats_sig = ev.rmats_fdr < 0.05 if np.isfinite(ev.rmats_fdr) else False
        supported = (
            rmats_sig
            and abs(dpsi_mean) >= args.effect_cutoff
            and prob_large >= 0.5
            and ctrl_total + treat_total >= args.min_support
        )
        high_confidence = supported and excludes_zero

        tier = "high-confidence" if high_confidence else (
            "supported" if supported else (
                "significant" if rmats_sig else "not-significant"
            )
        )

        results.append({
            "event_id": ev.event_id,
            "event_type": ev.event_type,
            "gene": ev.gene,
            "chrom": ev.chrom,
            "ctrl_psi": ctrl_psi,
            "treat_psi": treat_psi,
            "dpsi_mean": dpsi_mean,
            "dpsi_ci_low": dpsi_ci_low,
            "dpsi_ci_high": dpsi_ci_high,
            "dpsi_ci_excludes_zero": excludes_zero,
            "prob_large_effect": prob_large,
            "rmats_fdr": ev.rmats_fdr,
            "rmats_dpsi": ev.rmats_dpsi,
            "ctrl_support": ctrl_total,
            "treat_support": treat_total,
            "tier": tier,
        })

    # Write output (always creates the file, even if empty)
    _write_differential_tsv(results, args.output)

    if not results:
        logger.warning(
            "No events passed filters (min_support=%d). "
            "Output file %s contains only the header.",
            args.min_support,
            args.output,
        )

    # Summary
    tiers: dict[str, int] = {}
    for r in results:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1

    print(f"BRAID differential: {len(results)} events → {args.output}")
    for tier_name in ["high-confidence", "supported", "significant", "not-significant"]:
        if tier_name in tiers:
            print(f"  {tier_name}: {tiers[tier_name]}")


def _write_differential_tsv(results: list[dict], output: str) -> None:
    """Write differential results to TSV.

    Always writes the header so that the output file exists even when
    *results* is empty.

    Args:
        results: List of result dictionaries.
        output: Path for the output TSV file.
    """
    cols = [
        "event_id", "event_type", "gene", "chrom",
        "ctrl_psi", "treat_psi", "dpsi_mean",
        "dpsi_ci_low", "dpsi_ci_high", "dpsi_ci_excludes_zero",
        "prob_large_effect", "rmats_fdr", "rmats_dpsi",
        "ctrl_support", "treat_support", "tier",
    ]
    with open(output, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in results:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    vals.append(f"{v:.6f}")
                elif isinstance(v, bool):
                    vals.append("yes" if v else "no")
                else:
                    vals.append(str(v))
            f.write("\t".join(vals) + "\n")
