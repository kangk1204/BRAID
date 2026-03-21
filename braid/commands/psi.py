"""BRAID psi subcommand: per-event PSI confidence intervals.

Modes:
  Single sample:    braid psi --bam sample.bam --rmats-dir rMATS_output/
  Multi-replicate:  braid psi --bam rep1.bam rep2.bam --rmats-dir rMATS_output/
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

logger = logging.getLogger(__name__)


def add_psi_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``psi`` subcommand."""
    parser = subparsers.add_parser(
        "psi",
        help="Compute per-event PSI with calibrated confidence intervals.",
        description=(
            "Add calibrated PSI confidence intervals to rMATS events. "
            "Accepts one or more BAM files (replicates of the same condition)."
        ),
    )
    parser.add_argument(
        "--bam", nargs="+",
        help=(
            "BAM file(s). Multiple files are treated as biological replicates "
            "of the same condition. In rMATS mode, BAM paths are used for "
            "logging only; PSI counts are read from rMATS junction count tables."
        ),
    )
    parser.add_argument(
        "--rmats-dir",
        help="rMATS output directory (reads SE/A3SS/A5SS/MXE/RI tables).",
    )
    parser.add_argument(
        "--gtf",
        help="GTF annotation for de novo event proposal (used when --rmats-dir is not given).",
    )
    parser.add_argument(
        "--gene",
        help="Restrict analysis to a single gene (requires --gtf).",
    )
    parser.add_argument(
        "--region",
        help="Restrict analysis to a genomic region (e.g. 17:7668402-7687538).",
    )
    parser.add_argument(
        "-o", "--output", default="braid_psi.tsv",
        help="Output TSV file (default: braid_psi.tsv).",
    )
    parser.add_argument(
        "--replicates", type=int, default=500,
        help="Bootstrap replicates (default: 500).",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--min-support", type=int, default=10,
        help="Minimum total junction count (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output.",
    )
    parser.set_defaults(func=run_psi)


def run_psi(args: argparse.Namespace) -> None:
    """Execute the psi subcommand."""
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    bams = args.bam
    n_bams = len(bams)
    mode = "multi-replicate" if n_bams > 1 else "single-sample"
    logger.info("BRAID psi: %s mode (%d BAM files)", mode, n_bams)

    results = []

    if args.rmats_dir:
        from braid.target.rmats_bootstrap import (
            add_bootstrap_ci,
            parse_rmats_output,
        )
        events = parse_rmats_output(
            args.rmats_dir, min_total_count=args.min_support,
        )
        logger.info("Parsed %d rMATS events", len(events))

        if n_bams == 1:
            results = add_bootstrap_ci(
                events,
                n_replicates=args.replicates,
                confidence_level=args.confidence,
                seed=args.seed,
                sample="sample_1",
            )
        else:
            # Multi-replicate: use per-replicate counts from rMATS table
            # (IJC_SAMPLE_1 stores comma-separated per-replicate counts).
            # Each replicate gets its own posterior, then we combine
            # biological + sampling variance.
            logger.info(
                "%d BAMs provided — using per-replicate counts from rMATS "
                "for biological variance estimation.",
                n_bams,
            )
            all_rep_results = []
            for i in range(n_bams):
                rep_results = _bootstrap_per_replicate(
                    events, i,
                    n_replicates=args.replicates,
                    confidence_level=args.confidence,
                    seed=args.seed + i,
                )
                all_rep_results.append(rep_results)

            # Combine across replicates
            results = _combine_replicate_results(
                all_rep_results, args.confidence,
            )

    elif args.gtf or args.gene or args.region:
        from braid.target.extractor import lookup_gene
        from braid.target.psi_bootstrap import compute_psi_from_junctions

        for bam in bams:
            if args.gene and args.gtf:
                region = lookup_gene(args.gtf, args.gene)
                if region is None:
                    logger.error("Gene %s not found in %s", args.gene, args.gtf)
                    sys.exit(1)
                chrom, start, end = region.chrom, region.start, region.end
            elif args.region:
                parts = args.region.replace(":", "-").split("-")
                chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            else:
                logger.error("Specify --gene + --gtf or --region")
                sys.exit(1)

            bam_results = compute_psi_from_junctions(
                bam, chrom, start, end,
                gene=args.gene,
                n_replicates=args.replicates,
                confidence_level=args.confidence,
                seed=args.seed,
                annotation_gtf=args.gtf,
            )
            results.extend(bam_results)
    else:
        logger.error("Specify --rmats-dir or --gtf/--gene/--region")
        sys.exit(1)

    # Write output
    _write_psi_tsv(results, args.output, mode)
    print(f"BRAID psi: {len(results)} events → {args.output} ({mode})")


def _bootstrap_per_replicate(
    events: list,
    replicate_index: int,
    n_replicates: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> list:
    """Bootstrap PSI using per-replicate counts from rMATS.

    rMATS stores per-replicate junction counts as comma-separated
    values in IJC_SAMPLE_1 / SJC_SAMPLE_1. This function extracts
    the counts for one specific replicate and runs the posterior.
    Falls back to group-sum if replicate-level counts are unavailable.
    """
    from braid.target.psi_bootstrap import (
        CONFIDENT_CI_WIDTH_THRESHOLD,
        CONFIDENT_CV_THRESHOLD,
        PSIResult,
        bootstrap_psi,
    )

    rng = np.random.default_rng(seed)
    results = []

    for ev in events:
        # Try per-replicate counts
        inc_reps = getattr(ev, "sample_1_inc_replicates", ())
        exc_reps = getattr(ev, "sample_1_exc_replicates", ())

        if inc_reps and replicate_index < len(inc_reps):
            inc = inc_reps[replicate_index]
            exc = exc_reps[replicate_index] if replicate_index < len(exc_reps) else 0
        else:
            # Fallback to group sum
            inc = ev.sample_1_inc_count
            exc = ev.sample_1_exc_count

        psi, ci_low, ci_high, cv = bootstrap_psi(
            inc, exc,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            seed=int(rng.integers(0, 2**31)),
        )
        ci_width = ci_high - ci_low
        is_confident = (
            ci_width < CONFIDENT_CI_WIDTH_THRESHOLD
            and np.isfinite(cv)
            and cv <= CONFIDENT_CV_THRESHOLD
        )
        results.append(PSIResult(
            event_id=ev.event_id,
            event_type=ev.event_type,
            gene=ev.gene,
            chrom=ev.chrom,
            psi=psi,
            ci_low=ci_low,
            ci_high=ci_high,
            cv=cv,
            inclusion_count=inc,
            exclusion_count=exc,
            event_start=ev.exon_start,
            event_end=ev.exon_end,
            ci_width=ci_width,
            is_confident=is_confident,
        ))

    return results


def _combine_replicate_results(
    all_rep_results: list[list],
    confidence_level: float,
) -> list:
    """Combine per-replicate PSI results."""
    from scipy.stats import norm

    if not all_rep_results or not all_rep_results[0]:
        return []

    z = float(norm.ppf(1 - (1 - confidence_level) / 2))
    combined = []
    base = all_rep_results[0]

    for i, base_r in enumerate(base):
        rep_psis = []
        rep_vars = []
        for rep in all_rep_results:
            if i < len(rep):
                rep_psis.append(rep[i].psi)
                width = rep[i].ci_high - rep[i].ci_low
                rep_vars.append((width / (2 * z)) ** 2)

        if len(rep_psis) < 2:
            combined.append(base_r)
            continue

        bio_var = float(np.var(rep_psis, ddof=1))
        samp_var = float(np.mean(rep_vars))
        total_std = float(np.sqrt(bio_var + samp_var))
        mean_psi = float(np.mean(rep_psis))

        ci_low = max(0.0, mean_psi - z * total_std)
        ci_high = min(1.0, mean_psi + z * total_std)
        cv = total_std / mean_psi if mean_psi > 0 else float("nan")

        # Sum counts across replicates for output
        total_inc = sum(rep[i].inclusion_count for rep in all_rep_results if i < len(rep))
        total_exc = sum(rep[i].exclusion_count for rep in all_rep_results if i < len(rep))

        from dataclasses import replace
        combined.append(replace(
            base_r,
            psi=mean_psi,
            ci_low=ci_low,
            ci_high=ci_high,
            cv=cv,
            ci_width=ci_high - ci_low,
            inclusion_count=total_inc,
            exclusion_count=total_exc,
            is_confident=(ci_high - ci_low) < 0.2 and np.isfinite(cv) and cv <= 0.5,
        ))

    return combined


def _write_psi_tsv(results: list, output: str, mode: str) -> None:
    """Write PSI results to TSV."""
    with open(output, "w") as f:
        f.write(
            "event_id\tevent_type\tgene\tchrom\tPSI\tCI_low\tCI_high\t"
            "CI_width\tCV\tinc_count\texc_count\tconfident\tmode\n"
        )
        for r in results:
            f.write(
                f"{r.event_id}\t{r.event_type}\t{r.gene}\t{r.chrom}\t"
                f"{r.psi:.4f}\t{r.ci_low:.4f}\t{r.ci_high:.4f}\t"
                f"{getattr(r, 'ci_width', r.ci_high - r.ci_low):.4f}\t"
                f"{r.cv:.4f}\t{r.inclusion_count}\t{r.exclusion_count}\t"
                f"{'yes' if r.is_confident else 'no'}\t{mode}\n"
            )
