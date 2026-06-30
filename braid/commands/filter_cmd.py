"""BRAID ``filter`` subcommand: calibrate any caller's differential output.

Wraps rMATS, MAJIQ, SUPPA2, or betAS: reads the caller's native differential
table, adds BRAID's distribution-free calibrated 95% ΔPSI interval, a "reliable"
flag (interval excludes 0), and a confidence tier, and writes a TSV, an Excel
workbook, and a publication-ready figure.

Usage:
    braid filter --caller rmats   rMATS_output/        -o braid_filter
    braid filter --caller majiq   deltapsi.tsv         -o braid_filter
    braid filter --caller suppa2  diff.dpsi            -o braid_filter
    braid filter --caller betas   betas_diff.tsv       -o braid_filter
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import replace

import numpy as np

logger = logging.getLogger(__name__)


def add_filter_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``filter`` subcommand."""
    parser = subparsers.add_parser(
        "filter",
        help="Cross-caller report: unify rMATS/MAJIQ/SUPPA2/betAS into one schema.",
        description=(
            "Cross-caller reporting layer: add BRAID calibrated ΔPSI intervals, a "
            "reliable flag, and confidence tiers on top of an existing caller's "
            "differential output, in one unified schema, with TSV + Excel + figure. "
            "Uses each caller's native ΔPSI point estimate (for rMATS, the raw "
            "IncLevelDifference). For the deepest rMATS-only analysis (posterior ΔPSI "
            "+ rMATS-specific columns), use `braid differential` instead."
        ),
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Caller output: an rMATS directory, or a MAJIQ/SUPPA2/betAS table file. "
             "Omit when using --example.",
    )
    parser.add_argument(
        "--caller", required=True, choices=("rmats", "majiq", "suppa2", "betas"),
        help="Upstream splicing caller that produced the input.",
    )
    parser.add_argument(
        "--example", action="store_true",
        help="Run on the bundled tiny example for --caller (no input path needed); "
             "works after a plain pip install.",
    )
    parser.add_argument(
        "-o", "--output", default="braid_filter", metavar="PREFIX",
        help="Output path prefix (writes PREFIX.tsv, PREFIX.xlsx, and "
             "PREFIX.png/.pdf/.svg; default braid_filter).",
    )
    parser.add_argument(
        "--effect-cutoff", type=float, default=0.1,
        help="|ΔPSI| effect-size cutoff for tiers (default: 0.1).",
    )
    parser.add_argument(
        "--sig-threshold", type=float, default=0.05,
        help="Upstream FDR/p-value significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--min-support", type=int, default=20,
        help="Minimum rMATS total junction support to keep an event (default: 20).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="rMATS caller only: fail on the first malformed row instead of "
             "skipping it (data-integrity mode).",
    )
    parser.add_argument(
        "--flip-sign", action="store_true",
        help="Reverse the parsed ΔPSI direction before BRAID calibration. Also mirrors "
             "reported caller intervals and swaps group PSI fields when present.",
    )
    parser.add_argument(
        "--contrast", default=None,
        help="SUPPA2 caller only: select a contrast prefix such as 'Ctrl-KD', using "
             "'Ctrl-KD_dPSI' with its matching p-value column.",
    )
    parser.add_argument(
        "--dpsi-column", default=None,
        help="Exact ΔPSI column header to use instead of caller-specific fuzzy matching.",
    )
    parser.add_argument(
        "--pvalue-column", default=None,
        help="Exact p-value/significance column header to use. For MAJIQ this is the "
             "P(|dPSI|) probability column, converted to pvalue = 1 - P.",
    )
    parser.add_argument(
        "--fdr-column", default=None,
        help="Exact FDR/q-value column header to use where available.",
    )
    parser.add_argument(
        "--support-column", default=None,
        help="Exact read-support column header to use where available.",
    )
    parser.add_argument(
        "--event-id-column", default=None,
        help="Exact event-id column header to use where available.",
    )
    parser.add_argument(
        "--gene-column", default=None,
        help="Exact gene column header to use where available.",
    )
    parser.add_argument(
        "--calibration", default=None, metavar="PATH",
        help="ΔPSI ConformalCalibrator JSON (default: packaged calibrator fit on "
             "real RT-PCR residuals). Supply your own within-study refit here for a "
             "per-caller finite-sample coverage guarantee.",
    )
    parser.add_argument(
        "--top-n", type=int, default=25,
        help="Number of top events drawn in figure panel A (default: 25).",
    )
    parser.add_argument(
        "--no-figure", dest="make_figure", action="store_false",
        help="Skip the figure (still writes TSV + Excel).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output.",
    )
    parser.set_defaults(func=run_filter, make_figure=True)


def _resolve_calibrator(path: str | None):
    from braid.target.conformal import (
        ConformalCalibrator,
        load_differential_conformal_calibrator,
        require_scale_kind,
    )
    if path:
        return require_scale_kind(
            ConformalCalibrator.from_json(path), "absolute_dpsi", f"--calibration {path}"
        )
    return load_differential_conformal_calibrator()


def _flip_event_sign(ev):
    caller_low = None if ev.caller_high is None else -float(ev.caller_high)
    caller_high = None if ev.caller_low is None else -float(ev.caller_low)
    return replace(
        ev,
        dpsi=-float(ev.dpsi),
        group1_psi=ev.group2_psi,
        group2_psi=ev.group1_psi,
        caller_low=caller_low,
        caller_high=caller_high,
    )


def _support_route(events) -> str:
    if not events:
        return "none"
    known = [e.total_support is not None for e in events]
    if all(known):
        return "known -> support/event-type calibrator"
    if not any(known):
        return "unknown -> global calibrator"
    return "mixed -> known events use support bins; unknown events use global calibrator"


def _print_parse_summary(summary, events, *, sign: str) -> None:
    def val(x) -> str:
        return str(x) if x not in (None, "") else "none"

    print("Parse summary:")
    print(f"  caller={summary.caller}")
    if summary.contrast:
        print(f"  contrast={summary.contrast}")
    print(f"  dpsi_column={val(summary.dpsi_column)}")
    print(f"  pvalue_column={val(summary.pvalue_column)}")
    print(f"  fdr_column={val(summary.fdr_column)}")
    print(f"  support_column={val(summary.support_column)}")
    print(f"  event_id_column={val(summary.event_id_column)}")
    print(f"  gene_column={val(summary.gene_column)}")
    print(f"  parsed={summary.parsed}")
    print(f"  dropped_missing_dpsi={summary.dropped_missing_dpsi}")
    print(f"  dropped_out_of_range={summary.dropped_out_of_range}")
    print(f"  sign={sign}")
    print(f"  support={_support_route(events)}")


def run_filter(args: argparse.Namespace) -> None:
    """Execute the filter subcommand."""
    verbose = getattr(args, "verbose", False)
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    if getattr(args, "example", False):
        if args.input is not None:
            raise SystemExit("Pass either an input path or --example, not both.")
        from pathlib import Path

        import braid

        base = Path(braid.__file__).resolve().parent / "examples" / "filter"
        args.input = str({
            "rmats": base / "rmats_output",
            "majiq": base / "majiq_deltapsi.tsv",
            "suppa2": base / "suppa2_diffSplice.dpsi",
            "betas": base / "betas_differential.tsv",
        }[args.caller])
    elif args.input is None:
        raise SystemExit("filter requires an input path (or --example).")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"input not found: {args.input}")
    if not 0.0 <= args.sig_threshold <= 1.0:
        raise ValueError(f"sig-threshold must be in [0, 1], got {args.sig_threshold}")
    if args.effect_cutoff < 0.0:
        raise ValueError(f"effect-cutoff must be >= 0, got {args.effect_cutoff}")
    # top_n slices the figure short-list as rows[:top_n]; a negative value silently
    # drops events via Python slice semantics (rows[:-1] == all but the last) and 0
    # yields an empty figure. Reject anything < 1 here rather than emit a misleading
    # figure.
    if args.top_n < 1:
        raise ValueError(f"top-n must be >= 1, got {args.top_n}")

    from braid.adapters import PARSERS, ParserConfig, ParseResult, calibrate_events
    from braid.adapters.report import make_figure, tier_counts, write_excel, write_tsv

    try:
        calibrator = _resolve_calibrator(args.calibration)
    except (FileNotFoundError, OSError) as exc:
        raise SystemExit(
            "No ΔPSI conformal calibrator available. The packaged artifact is "
            "missing; supply one with --calibration PATH."
        ) from exc

    parser_fn = PARSERS[args.caller]
    parser_config = ParserConfig(
        contrast=getattr(args, "contrast", None),
        dpsi_column=getattr(args, "dpsi_column", None),
        pvalue_column=getattr(args, "pvalue_column", None),
        fdr_column=getattr(args, "fdr_column", None),
        support_column=getattr(args, "support_column", None),
        event_id_column=getattr(args, "event_id_column", None),
        gene_column=getattr(args, "gene_column", None),
    )
    if args.caller == "rmats":
        parsed = parser_fn(
            args.input, min_support=args.min_support,
            strict=getattr(args, "strict", False),
            config=parser_config,
            return_summary=True,
        )
    else:
        parsed = parser_fn(args.input, config=parser_config, return_summary=True)
    if not isinstance(parsed, ParseResult):
        raise TypeError(f"{args.caller} parser did not return a ParseResult")
    events = parsed.events
    flip_sign = getattr(args, "flip_sign", False)
    if flip_sign:
        events = [_flip_event_sign(ev) for ev in events]
    if verbose:
        _print_parse_summary(
            parsed.summary, events,
            sign="flipped" if flip_sign else "as_input",
        )
    logger.info("Parsed %d events from %s output", len(events), args.caller)
    if not events:
        logger.warning("No events parsed from %s; check the input format.", args.caller)

    rows = calibrate_events(
        events, calibrator,
        effect_cutoff=args.effect_cutoff, sig_threshold=args.sig_threshold,
    )

    # Distribution-shift warning (only meaningful where support is real).
    known = np.array(
        [r["total_support"] for r in rows if r["support_known"]], dtype=float
    )
    if known.size:
        ok, msg = calibrator.check_applicability(known)
        if not ok:
            logger.warning("Calibration applicability: %s", msg)
    elif rows:
        logger.info(
            "%s reports no read support; events use the pooled global quantile "
            "(support and event-type bins are bypassed). Support-conditional "
            "sharpening needs counts (rMATS/betAS).",
            args.caller,
        )

    out = args.output
    tsv_path = f"{out}.tsv"
    write_tsv(rows, tsv_path)
    wrote_xlsx = write_excel(rows, f"{out}.xlsx", caller=args.caller)
    wrote_fig = (
        make_figure(rows, out, caller=args.caller, top_n=args.top_n)
        if args.make_figure else False
    )

    counts = tier_counts(rows)
    n_reliable = sum(1 for r in rows if r["reliable"])
    print(f"BRAID filter ({args.caller}): {len(rows)} events -> {tsv_path}")
    if wrote_xlsx:
        print(f"  Excel : {out}.xlsx")
    if wrote_fig:
        print(f"  Figure: {out}.png / .pdf / .svg")
    print(f"  reliable (interval excludes 0): {n_reliable}")
    for tier in (
        "high-confidence", "supported", "caller-significant-only",
        "not-significant", "not-reliable",
    ):
        if tier in counts:
            print(f"    {tier}: {counts[tier]}")
