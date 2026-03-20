#!/usr/bin/env python3
"""Grid-search BRAID overdispersed count scale on the PacBio benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, ".")

from benchmarks.rtpcr_benchmark import benchmark_pacbio_junction_validation


def _parse_scales(value: str) -> list[float]:
    """Parse a comma-separated scale list."""
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _pick_best(
    rows: list[dict],
    *,
    target_coverage_low: float,
    target_coverage_high: float,
    min_confident_count: int,
) -> dict | None:
    """Select one scale using coverage-first benchmark gating."""
    if not rows:
        return None

    target_mid = (target_coverage_low + target_coverage_high) / 2
    usable = [
        row for row in rows
        if row.get("confident_count", 0) >= min_confident_count
    ]
    in_band = [
        row for row in usable
        if target_coverage_low <= row["ci_coverage"] <= target_coverage_high
    ]
    if in_band:
        return max(
            in_band,
            key=lambda row: (
                row.get("confident_count", 0),
                row.get("confident_accuracy") or 0.0,
                -abs(row["ci_coverage"] - target_mid),
                -row["r_squared"],
            ),
        )

    # If nothing reaches the target band, prefer the highest coverage among
    # scales that still leave a usable confident subset.
    if usable:
        return max(
            usable,
            key=lambda row: (
                row["ci_coverage"],
                row.get("confident_accuracy") or 0.0,
                row.get("confident_count", 0),
                -row["mae"],
                row["r_squared"],
            ),
        )

    # Final fallback when every scale collapses the confident subset.
    return max(
        rows,
        key=lambda row: (
            row["ci_coverage"],
            row.get("confident_count", 0),
            row.get("confident_accuracy") or 0.0,
            -row["mae"],
            row["r_squared"],
        ),
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scales",
        default="0.01,0.02,0.03,0.05,0.08,0.1,0.2,0.3,0.5,0.7,1.0",
        help="Comma-separated count scales to evaluate.",
    )
    parser.add_argument(
        "--target-coverage-low",
        type=float,
        default=0.80,
        help="Lower bound of the desired CI coverage band.",
    )
    parser.add_argument(
        "--target-coverage-high",
        type=float,
        default=0.90,
        help="Upper bound of the desired CI coverage band.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/overdispersed_scale_grid.json",
        help="Path to the JSON output file.",
    )
    parser.add_argument(
        "--min-confident-count",
        type=int,
        default=25,
        help="Minimum confident events required for a scale to be considered usable.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the grid search and persist the selected scale."""
    args = parse_args()
    scales = _parse_scales(args.scales)
    rows: list[dict] = []

    print("=" * 60)
    print("  BRAID Overdispersed Scale Grid Search")
    print("=" * 60)

    for scale in scales:
        print(f"\n[scale={scale:.3f}]")
        result = benchmark_pacbio_junction_validation(count_scale=scale)
        if not result:
            continue
        rows.append(result)

    best = _pick_best(
        rows,
        target_coverage_low=args.target_coverage_low,
        target_coverage_high=args.target_coverage_high,
        min_confident_count=args.min_confident_count,
    )

    payload = {
        "metadata": {
            "scales": scales,
            "target_coverage_low": args.target_coverage_low,
            "target_coverage_high": args.target_coverage_high,
            "min_confident_count": args.min_confident_count,
        },
        "results": rows,
        "selected_scale": best,
    }

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    if best:
        print("\nSelected scale:")
        print(
            f"  scale={best['count_scale']:.3f} "
            f"coverage={best['ci_coverage']:.1%} "
            f"confident={best.get('confident_count', 0)} "
            f"conf_acc={(best.get('confident_accuracy') or 0.0):.1%}"
        )
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
