"""``braid example`` — run BRAID end-to-end on tiny synthetic data, zero setup.

A new user can verify BRAID works and see its headline output (calibrated ΔPSI
confidence intervals + confidence tiers) in a few seconds, without needing their
own BAMs or rMATS output. It synthesizes a minimal rMATS-style table, runs the
``differential`` pipeline on it, and prints an interpreted summary.
"""
from __future__ import annotations

import argparse
import os
import tempfile

_HEADER = [
    "ID", "GeneID", "geneSymbol", "chr", "strand", "exonStart_0base", "exonEnd",
    "upstreamES", "upstreamEE", "downstreamES", "downstreamEE",
    "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
]


def _row(eid, gene, es, inc1, sjc1, inc2, sjc2, fdr, dpsi):
    psi1 = inc1 / (inc1 + sjc1)
    psi2 = inc2 / (inc2 + sjc2)
    return [
        str(eid), gene, gene, "chr1", "+", str(es), str(es + 100),
        str(es - 200), str(es - 100), str(es + 200), str(es + 300),
        str(inc1), str(sjc1), str(inc2), str(sjc2),
        "100", "100", "0.0", str(fdr),
        f"{psi1:.3f}", f"{psi2:.3f}", str(dpsi),
    ]


def _write_demo_rmats(rmats_dir: str) -> int:
    """Write a tiny synthetic SE.MATS.JC.txt with illustrative events."""
    os.makedirs(rmats_dir, exist_ok=True)
    rows = [
        # gene, exonStart, IJC1, SJC1, IJC2, SJC2, FDR, rMATS dPSI
        _row(1, "STRONG_UP", 1000, 270, 30, 90, 210, 0.0008, 0.60),    # ctrl 0.9 -> treat 0.3
        _row(2, "STRONG_DOWN", 5000, 40, 160, 180, 20, 0.0011, -0.60),  # ctrl 0.2 -> treat 0.9
        _row(3, "NO_CHANGE", 9000, 150, 150, 150, 150, 0.82, 0.00),     # 0.5 -> 0.5
        _row(4, "LOW_COVERAGE", 13000, 14, 6, 7, 13, 0.04, 0.35),       # real-ish but shallow
    ]
    lines = ["\t".join(_HEADER)] + ["\t".join(r) for r in rows]
    with open(os.path.join(rmats_dir, "SE.MATS.JC.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return len(rows)


def add_example_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``example`` subcommand."""
    parser = subparsers.add_parser(
        "example",
        help="Run BRAID on tiny synthetic data (no inputs needed) to see calibrated output.",
        description=(
            "Zero-setup demo: synthesizes a tiny rMATS-style table, runs the BRAID "
            "differential pipeline, and prints calibrated ΔPSI intervals + confidence "
            "tiers. Use this right after install to verify everything works."
        ),
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Directory to write the demo inputs/outputs (default: a temp directory).",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Keep the generated demo files (default: temp dir is left in place and printed).",
    )
    parser.set_defaults(func=run_example)


def run_example(args: argparse.Namespace) -> None:
    """Execute the demo."""
    from braid.commands.differential import run_differential

    out_dir = args.output or tempfile.mkdtemp(prefix="braid_example_")
    # os.makedirs(..., exist_ok=True) still raises FileExistsError when the path exists
    # as a FILE (exist_ok only forgives an existing directory). For a zero-setup demo
    # command, turn that into a clear ValueError instead of an "Unexpected error"
    # traceback when a new user points -o at a file.
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        raise ValueError(
            f"--output {out_dir} exists and is not a directory; pass a directory path "
            "(or omit -o to use a temp directory)."
        )
    os.makedirs(out_dir, exist_ok=True)
    rmats_dir = os.path.join(out_dir, "rmats")
    out_tsv = os.path.join(out_dir, "braid_differential.tsv")
    n = _write_demo_rmats(rmats_dir)

    print("BRAID example — calibrated differential splicing on synthetic data")
    print(f"  (no BAMs needed; wrote {n} synthetic rMATS SE events to {rmats_dir})\n")

    diff_args = argparse.Namespace(
        rmats_dir=rmats_dir, output=out_tsv, replicates=2000, confidence=0.95,
        effect_cutoff=0.1, min_support=20, seed=42, verbose=False,
        ctrl=None, treat=None, use_conformal=True, calibration=None,
    )
    run_differential(diff_args)

    _print_interpreted(out_tsv)
    print(f"\nFull table: {out_tsv}")
    print("Try it on your own data:")
    print("  braid differential --rmats-dir <rMATS_output_dir> -o my_results.tsv")
    print("  braid psi --bam reps*.bam --rmats-dir <rMATS_output_dir> -o my_psi.tsv")


def _print_interpreted(out_tsv: str) -> None:
    """Pretty-print the demo output with a plain-language reading."""
    import csv

    with open(out_tsv) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        print("  (no events passed filters)")
        return
    print(f"\n  {'gene':<14}{'ΔPSI':>7}{'  95% calibrated interval':<26}{'tier':<16}")
    print(f"  {'-'*60}")
    for r in rows:
        lo, hi = float(r["ci_low"]), float(r["ci_high"])
        print(f"  {r['gene']:<14}{float(r['dpsi']):>7.2f}  "
              f"[{lo:+.2f}, {hi:+.2f}]            {r['tier']:<16}")
    print(f"  {'-'*60}")
    print("  Reading it: the interval is a DISTRIBUTION-FREE calibrated 95% range for the")
    print("  true ΔPSI (fit on real RT-PCR residuals). 'high-confidence' = the interval")
    print("  excludes zero; a wide interval honestly reflects how much you should trust the")
    print("  call. This calibrated coverage is what BRAID adds on top of rMATS.")
