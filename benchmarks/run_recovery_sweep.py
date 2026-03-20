"""Grid sweep for no-motif synthetic recovery experiments.

This wrapper reuses ``run_decomposer_comparison.py`` for each point in a small
grid over builder profile and minimum junction support, then consolidates the
per-run comparison JSON files into a sweep summary TSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("benchmark_results") / "recovery_sweep"
DEFAULT_VARIANTS = [
    "legacy_no_motif_validation",
    "iterative_v2_no_motif_validation",
]


def _comparison_script_path() -> Path:
    return Path(__file__).resolve().with_name("run_decomposer_comparison.py")


def _cell_output_dir(base_dir: Path, builder_profile: str, min_junction_support: int) -> Path:
    return base_dir / f"profile_{builder_profile}__j{min_junction_support}"


def _run_cell(
    *,
    output_dir: Path,
    builder_profile: str,
    min_junction_support: int,
    variants: list[str],
    shadow_decomposer: str | None,
    threads: int,
    backend: str,
    min_coverage: float,
    min_score: float,
    min_phasing_support: int,
) -> Path:
    cmd = [
        sys.executable,
        str(_comparison_script_path()),
        "--output-dir",
        str(output_dir),
        "--builder-profile",
        builder_profile,
        "--min-junction-support",
        str(min_junction_support),
        "--threads",
        str(threads),
        "--backend",
        backend,
        "--min-coverage",
        str(min_coverage),
        "--min-score",
        str(min_score),
        "--min-phasing-support",
        str(min_phasing_support),
        "--variants",
        *variants,
    ]
    if shadow_decomposer is not None:
        cmd.extend(["--shadow-decomposer", shadow_decomposer])
    logger.info("Running sweep cell: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return output_dir / "comparison.json"


def _flatten_rows(
    comparison: dict[str, Any],
    *,
    requested_builder_profile: str,
    min_junction_support: int,
    comparison_json: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    motif_compat = comparison.get("dataset", {}).get("motif_compatibility", {})
    for variant_row in comparison.get("variants", []):
        metrics = variant_row.get("metrics", {})
        diag_summary = (variant_row.get("diagnostics") or {}).get("summary", {})
        rows.append(
            {
                "requested_builder_profile": requested_builder_profile,
                "builder_profile": variant_row.get("builder_profile", requested_builder_profile),
                "min_junction_support": min_junction_support,
                "comparison_json": str(comparison_json.resolve()),
                "variant": variant_row.get("variant"),
                "decomposer": variant_row.get("decomposer"),
                "motif_validation": variant_row.get("motif_validation"),
                "status": variant_row.get("status"),
                "transcript_sensitivity": metrics.get("transcript_sensitivity"),
                "transcript_precision": metrics.get("transcript_precision"),
                "intron_sensitivity": metrics.get("intron_sensitivity"),
                "intron_precision": metrics.get("intron_precision"),
                "runtime_seconds": variant_row.get("runtime_seconds"),
                "peak_memory_mb": variant_row.get("peak_memory_mb"),
                "surviving_transcripts": diag_summary.get("total_surviving_transcripts"),
                "graphs_built": diag_summary.get("graphs_built"),
                "canonical_truth_fraction": motif_compat.get("canonical_truth_fraction"),
            }
        )
    return rows


def _write_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    json_path = output_dir / "sweep_summary.json"
    tsv_path = output_dir / "sweep_summary.tsv"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=2, sort_keys=True)

    fieldnames = [
        "requested_builder_profile",
        "builder_profile",
        "min_junction_support",
        "variant",
        "decomposer",
        "motif_validation",
        "status",
        "transcript_sensitivity",
        "transcript_precision",
        "intron_sensitivity",
        "intron_precision",
        "runtime_seconds",
        "peak_memory_mb",
        "surviving_transcripts",
        "graphs_built",
        "canonical_truth_fraction",
        "comparison_json",
    ]
    with open(tsv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    builder_profiles = list(dict.fromkeys(args.builder_profiles))
    min_junction_support_values = list(dict.fromkeys(args.min_junction_support_values))
    for builder_profile in builder_profiles:
        for min_junction_support in min_junction_support_values:
            cell_dir = _cell_output_dir(output_dir, builder_profile, min_junction_support)
            comparison_json = _run_cell(
                output_dir=cell_dir,
                builder_profile=builder_profile,
                min_junction_support=min_junction_support,
                variants=args.variants,
                shadow_decomposer=args.shadow_decomposer,
                threads=args.threads,
                backend=args.backend,
                min_coverage=args.min_coverage,
                min_score=args.min_score,
                min_phasing_support=args.min_phasing_support,
            )
            comparison = json.loads(comparison_json.read_text(encoding="utf-8"))
            rows.extend(
                _flatten_rows(
                    comparison,
                    requested_builder_profile=builder_profile,
                    min_junction_support=min_junction_support,
                    comparison_json=comparison_json,
                )
            )

    _write_outputs(output_dir, rows)
    return {"rows": rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run no-motif synthetic recovery sweeps"
            " over builder profile and junction support."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for sweep outputs (default: benchmark_results/recovery_sweep).",
    )
    parser.add_argument(
        "--builder-profiles",
        nargs="+",
        choices=["default", "conservative_correctness", "aggressive_recall"],
        default=["default", "conservative_correctness", "aggressive_recall"],
        help="Builder profiles to sweep.",
    )
    parser.add_argument(
        "--min-junction-support-values",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Minimum junction support values to sweep.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help="Variant names forwarded to run_decomposer_comparison.py.",
    )
    parser.add_argument(
        "--shadow-decomposer",
        choices=["legacy", "iterative_v2"],
        default=None,
        help="Optional shadow decomposer forwarded to each comparison run.",
    )
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--min-phasing-support", type=int, default=1)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
    )
    run_sweep(args)


if __name__ == "__main__":
    main()
