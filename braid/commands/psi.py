"""BRAID psi subcommand: per-event PSI confidence intervals.

Modes:
  Group summary:    braid psi --rmats-dir rMATS_output/
  Multi-replicate:  braid psi --bam rep1.bam rep2.bam --rmats-dir rMATS_output/
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from braid.output_safety import csv_safe

logger = logging.getLogger(__name__)


def add_psi_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``psi`` subcommand."""
    parser = subparsers.add_parser(
        "psi",
        help="Compute per-event PSI with calibrated confidence intervals.",
        description=(
            "Add calibrated PSI confidence intervals to rMATS events. "
            "Optional BAM paths are logging and replicate-count intent only; "
            "PSI counts are read from rMATS tables."
        ),
    )
    parser.add_argument(
        "--bam", nargs="+",
        help=(
            "Optional BAM file(s). Multiple files are treated as biological "
            "replicate intent for the same condition. Paths are used for "
            "logging only; PSI counts are read from rMATS junction count tables."
        ),
    )
    parser.add_argument(
        "--rmats-dir",
        help="rMATS output directory (reads SE/A3SS/A5SS/MXE/RI tables).",
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
        help="Confidence level (default: 0.95). In single-sample conformal mode, "
             "interval alpha comes from the calibrator JSON; multi-replicate mode "
             "uses this value for across-replicate combination.",
    )
    parser.add_argument(
        "--min-support", type=int, default=10,
        help="Minimum junction count of the REPORTED sample (default: 10). In "
             "single-sample mode this is applied to the --sample condition's own "
             "support, not the two-group total.",
    )
    parser.add_argument(
        "--sample", choices=("sample_1", "sample_2"), default="sample_1",
        help="Which rMATS condition's PSI to report in single-sample mode "
             "(default: sample_1 = rMATS --b1). Multi-replicate mode reports "
             "sample_1 only.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail on the first malformed rMATS row instead of skipping it "
             "(data-integrity mode).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output.",
    )
    parser.add_argument(
        "--no-conformal", dest="use_conformal", action="store_false",
        help="Disable conformal calibration and use the legacy overdispersed-Beta "
             "inflation intervals instead (default: conformal, using the "
             "calibrator's fitted alpha).",
    )
    parser.add_argument(
        "--calibration",
        help="Path to a fitted conformal calibrator JSON, overriding the shipped "
             "default. Ignored with --no-conformal.",
    )
    parser.add_argument(
        "--allow-replicate-fallback", action="store_true",
        help="In multi-replicate rMATS mode, proceed when the BAM count does not "
             "match the rMATS table's per-replicate vector length. The analysis "
             "always uses ALL of the table's per-replicate counts; the BAM count "
             "is only an intent signal (default: hard error on any mismatch to "
             "avoid duplicating group-sum totals or silently dropping "
             "replicates).",
    )
    parser.set_defaults(func=run_psi)


def _resolve_conformal_calibrator(args: argparse.Namespace):
    """Load the conformal calibrator for this run (custom path, else shipped default).

    Returns ``None`` when ``--no-conformal`` is set or no calibrator is available.
    """
    if not getattr(args, "use_conformal", True):
        if getattr(args, "calibration", None):
            logger.warning(
                "--calibration %s is ignored because --no-conformal disables the "
                "conformal layer; drop one of the two conflicting options.",
                args.calibration,
            )
        return None
    path = getattr(args, "calibration", None)
    from braid.target.conformal import (
        ConformalCalibrator,
        load_default_conformal_calibrator,
        require_scale_kind,
    )
    if path:
        # An explicit --calibration is the user's assertion that THIS file is the
        # calibrator. A missing / unreadable / malformed file is therefore a hard
        # error, not a silent fall-back to the legacy overdispersed-Beta intervals
        # (which would run to exit 0 and mislabel the output as conformal-
        # calibrated). Mirrors `braid differential`'s _resolve_differential_calibrator.
        try:
            cal = ConformalCalibrator.from_json(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"--calibration file not found: {path}") from exc
        except OSError as exc:
            # A directory, a permission error, etc. (FileNotFoundError handled
            # above); surface a --calibration-scoped message instead of a raw
            # IsADirectoryError/PermissionError.
            raise OSError(
                f"--calibration path could not be read: {path} ({exc})"
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            # json.JSONDecodeError subclasses ValueError; KeyError/TypeError surface
            # from a valid-JSON-but-wrong-schema artifact (e.g. missing 'alpha').
            raise ValueError(
                f"--calibration {path} is not a valid ConformalCalibrator JSON: {exc}"
            ) from exc
        return require_scale_kind(cal, "posterior_std", f"--calibration {path}")
    # No custom path: the shipped default may legitimately be absent in some
    # installs, so tolerate its absence and fall back to legacy intervals.
    try:
        return load_default_conformal_calibrator()
    except (FileNotFoundError, OSError) as exc:
        logger.warning(
            "Default conformal calibrator unavailable (%s); using legacy intervals",
            exc,
        )
        return None


def run_psi(args: argparse.Namespace) -> None:
    """Execute the psi subcommand."""
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # A negative --min-support makes both support gates vacuously true: the
    # parse_rmats_output two-group filter (count >= min_total_count) and the
    # single-sample own-support filter (sum(...) >= min_support) would admit
    # evidence-free rows (e.g. inc=exc=0, PSI=0). Reject it here, matching
    # `braid differential` (differential.py).
    if args.min_support < 0:
        raise ValueError(f"min_support must be >= 0, got {args.min_support}")
    # Validate the posterior knobs UPFRONT, matching `braid differential`. Otherwise
    # these are only caught per-event deep inside bootstrap_psi, so an invalid value
    # is accepted silently whenever zero events pass the support filter (the run then
    # writes a header-only file instead of failing fast).
    if not 0.0 < args.confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {args.confidence}")
    if args.replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {args.replicates}")

    # --bam is optional in rMATS mode (counts come from the rMATS table); guard
    # against the argparse default of None so a missing --bam is a clear error
    # rather than `TypeError: object of type 'NoneType' has no len()`.
    bams = args.bam or []
    n_bams = len(bams)
    mode = "multi-replicate" if n_bams > 1 else "single-sample"
    logger.info("BRAID psi: %s mode (%d BAM files)", mode, n_bams)
    conformal_calibrator = _resolve_conformal_calibrator(args)

    results = []

    if args.rmats_dir:
        from braid.commands.rmats_input import require_rmats_tables
        from braid.target.rmats_bootstrap import (
            add_bootstrap_ci,
            parse_rmats_output,
        )
        require_rmats_tables(args.rmats_dir, logger)
        events = parse_rmats_output(
            args.rmats_dir, min_total_count=args.min_support,
            strict=getattr(args, "strict", False),
        )
        logger.info("Parsed %d rMATS events", len(events))

        if n_bams <= 1:
            # 0 or 1 BAM: summarize the rMATS group counts directly. (rMATS
            # mode does not need a BAM; any BAM passed is logging-only here.)
            from braid.target.rmats_bootstrap import get_group_counts
            sample = getattr(args, "sample", "sample_1")
            # --min-support must hold on the REPORTED sample's own support: an
            # event with no reads in `sample` but many in the other condition
            # would otherwise pass parse_rmats_output's two-group-total filter and
            # emit an evidence-free PSI row (e.g. inc=exc=0, PSI=0).
            events = [
                e for e in events
                if sum(get_group_counts(e, sample=sample)) >= args.min_support
            ]
            results = add_bootstrap_ci(
                events,
                n_replicates=args.replicates,
                confidence_level=args.confidence,
                seed=args.seed,
                sample=sample,
                use_conformal=getattr(args, "use_conformal", True),
                conformal_calibrator=conformal_calibrator,
            )
        else:
            # Multi-replicate: use per-replicate counts from rMATS table
            # (IJC_SAMPLE_1 stores comma-separated per-replicate counts).
            # Each replicate gets its own posterior, then we combine
            # biological + sampling variance.
            if getattr(args, "sample", "sample_1") != "sample_1":
                # The per-replicate biological-variance path models sample_1's
                # per-replicate vectors only. Refuse --sample sample_2 here rather
                # than silently reporting sample_1.
                logger.error(
                    "braid psi --sample %s is not supported in multi-replicate "
                    "mode (multiple --bam): the per-replicate path reports "
                    "sample_1 only. Omit --bam to report %s in single-sample "
                    "mode, or pass --sample sample_1.",
                    args.sample, args.sample,
                )
                sys.exit(1)
            logger.info(
                "%d BAMs provided — using per-replicate counts from rMATS "
                "for biological variance estimation.",
                n_bams,
            )
            n_eff_reps = _resolve_effective_replicate_count(
                events, n_bams,
                allow_fallback=getattr(args, "allow_replicate_fallback", False),
            )
            all_rep_results = []
            for i in range(n_eff_reps):
                rep_results = _bootstrap_per_replicate(
                    events, i,
                    n_replicates=args.replicates,
                    confidence_level=args.confidence,
                    seed=args.seed + i,
                    conformal_calibrator=conformal_calibrator,
                )
                all_rep_results.append(rep_results)

            # Combine across replicates
            results = _combine_replicate_results(
                all_rep_results, args.confidence,
            )

    else:
        logger.error("--rmats-dir is required for braid psi.")
        sys.exit(1)

    # Write output (record which condition the PSI column reports)
    reported_sample = (
        getattr(args, "sample", "sample_1") if n_bams <= 1 else "sample_1"
    )
    _write_psi_tsv(results, args.output, mode, sample=reported_sample)
    print(f"BRAID psi: {len(results)} events → {args.output} ({mode})")


def _resolve_effective_replicate_count(
    events: list,
    n_bams: int,
    allow_fallback: bool,
) -> int:
    """Decide how many per-replicate posteriors to draw for multi-replicate PSI.

    The rMATS table's comma-separated per-replicate IJC/SJC vector is the ground
    truth for how many biological replicates exist; in rMATS mode the BAM paths
    are logging-only, so the BAM *count* is just an intent signal. We therefore
    always analyse ALL of the table's per-replicate counts and treat a BAM/table
    count mismatch as a hard error by default.

    Both mismatch directions are unsafe:

    * More BAMs than table replicates would, under the old code, force the surplus
      replicates onto the group-sum fallback, double-counting reads and
      understating biological variance.
    * Fewer BAMs than table replicates would, under the old code, bootstrap only
      the first ``n_bams`` table replicates and silently *undercount* the totals
      (e.g. a 3-replicate 60/30 table reported as 30/15).

    ``allow_fallback`` downgrades the mismatch to a warning and proceeds using all
    of the table's per-replicate counts (never a truncated prefix), so the
    group-sum total is preserved either way.

    A replicate only "exists" when BOTH the inclusion and exclusion vectors carry
    it, so the per-event count is ``min(len(IJC), len(SJC))`` — matching the
    completeness rule in ``_bootstrap_per_replicate``. Using the inclusion length
    alone would let a row with a shorter exclusion vector draw a surplus index
    that then falls back to the group-sum total, double-counting reads in the
    combine step. A malformed row with unequal vector lengths therefore lowers
    ``n_table_reps`` and surfaces as the BAM/table mismatch error above.
    """
    rep_lengths = [
        min(
            len(getattr(ev, "sample_1_inc_replicates", ()) or ()),
            len(getattr(ev, "sample_1_exc_replicates", ()) or ()),
        )
        for ev in events
    ]
    n_table_reps = min(rep_lengths) if rep_lengths else 0

    if n_table_reps < 1:
        logger.error(
            "Multi-replicate PSI requested (%d BAMs) but the rMATS table carries "
            "no per-replicate IJC/SJC counts (need comma-separated IJC_SAMPLE_1 / "
            "SJC_SAMPLE_1). Provide a single BAM for group-level analysis, or an "
            "rMATS table with per-replicate counts.",
            n_bams,
        )
        sys.exit(1)

    if n_bams != n_table_reps:
        if not allow_fallback:
            logger.error(
                "Replicate-count mismatch: %d BAM(s) but %d per-replicate count(s) "
                "in the rMATS table. Aligning a different number of BAMs than the "
                "table's replicate vector either duplicates the group-sum totals "
                "(too many BAMs) or silently drops valid table replicates and "
                "undercounts reads/variance (too few BAMs). Re-run with %d BAM(s), "
                "regenerate the rMATS table with matching replicates, or pass "
                "--allow-replicate-fallback to analyse all %d table replicate(s).",
                n_bams, n_table_reps, n_table_reps, n_table_reps,
            )
            sys.exit(1)
        logger.warning(
            "Replicate-count mismatch: %d BAM(s) vs %d per-replicate count(s) in "
            "the rMATS table; --allow-replicate-fallback set, analysing all %d "
            "table replicate(s) (BAM count ignored).",
            n_bams, n_table_reps, n_table_reps,
        )

    return n_table_reps


def _bootstrap_per_replicate(
    events: list,
    replicate_index: int,
    n_replicates: int = 500,
    confidence_level: float = 0.95,
    seed: int = 42,
    conformal_calibrator=None,
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
    n_fallback = 0

    for ev in events:
        # Try per-replicate counts. A replicate is only "complete" when BOTH the
        # inclusion and exclusion vectors carry it; rMATS stores parallel
        # IJC_SAMPLE_1 / SJC_SAMPLE_1 vectors, so a shorter exclusion vector means
        # a malformed/partially parsed row. We must NOT read that missing
        # exclusion as an observed zero (which would push PSI toward 1 with false
        # confidence) — such replicates fall back to the group-sum total instead.
        inc_reps = getattr(ev, "sample_1_inc_replicates", ()) or ()
        exc_reps = getattr(ev, "sample_1_exc_replicates", ()) or ()
        n_complete_reps = min(len(inc_reps), len(exc_reps))

        if replicate_index < n_complete_reps:
            inc = inc_reps[replicate_index]
            exc = exc_reps[replicate_index]
        else:
            # Fallback to group sum. This collapses biological-replicate
            # variance into the group total, so we surface it (via the warning
            # below) rather than silently substituting. A missing per-replicate
            # exclusion count lands here too, so it is never treated as a real 0.
            inc = ev.sample_1_inc_count
            exc = ev.sample_1_exc_count
            n_fallback += 1

        psi, ci_low, ci_high, cv = bootstrap_psi(
            inc, exc,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            seed=int(rng.integers(0, 2**31)),
            event_type=ev.event_type,
            conformal_calibrator=conformal_calibrator,
            inc_form_len=ev.inc_form_len,
            skip_form_len=ev.skip_form_len,
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

    if n_fallback:
        logger.warning(
            "Replicate %d: %d/%d events lacked per-replicate rMATS counts and "
            "fell back to group-sum totals; biological-replicate variance is "
            "understated for those events. Check that the rMATS table stores "
            "comma-separated per-replicate IJC/SJC and that the BAM count "
            "matches the replicate vector length.",
            replicate_index, n_fallback, len(events),
        )

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


def _write_psi_tsv(
    results: list, output: str, mode: str, sample: str = "sample_1"
) -> None:
    """Write PSI results to TSV.

    ``sample`` records which rMATS condition the PSI column reports (sample_1 =
    rMATS --b1, sample_2 = --b2), so a single-sample sample_2 run is self-describing
    from the file alone.
    """
    with open(output, "w", encoding="utf-8") as f:
        f.write(
            "event_id\tevent_type\tgene\tchrom\tsample\tPSI\tCI_low\tCI_high\t"
            "CI_width\tCV\tinc_count\texc_count\tconfident\tmode\n"
        )
        for r in results:
            f.write(
                f"{csv_safe(str(r.event_id))}\t{csv_safe(str(r.event_type))}\t"
                f"{csv_safe(str(r.gene))}\t{csv_safe(str(r.chrom))}\t{sample}\t"
                f"{r.psi:.4f}\t{r.ci_low:.4f}\t{r.ci_high:.4f}\t"
                f"{getattr(r, 'ci_width', r.ci_high - r.ci_low):.4f}\t"
                f"{r.cv:.4f}\t{r.inclusion_count}\t{r.exclusion_count}\t"
                f"{'yes' if r.is_confident else 'no'}\t{mode}\n"
            )
