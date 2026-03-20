"""Run a matrix of BRAID real-data benchmarks with optional proxy mode."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VariantSpec:
    """One BRAID real-data benchmark variant."""

    name: str
    decomposer: str
    builder_profile: str


def _default_output_dir(mode: str) -> Path:
    """Return a timestamped default output directory for proxy/nightly runs."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("real_benchmark") / "results" / mode / stamp


def _variant_specs(
    variant_names: list[str],
    builder_profile: str,
) -> list[VariantSpec]:
    variants: list[VariantSpec] = []
    for name in variant_names:
        if name == "legacy":
            variants.append(VariantSpec(
                name=name, decomposer="legacy",
                builder_profile=builder_profile,
            ))
        elif name == "iterative_v2":
            variants.append(
                VariantSpec(name=name, decomposer="iterative_v2", builder_profile=builder_profile)
            )
        else:
            raise ValueError(f"Unsupported variant: {name}")
    return variants


def _build_benchmark_command(
    variant: VariantSpec,
    *,
    mode: str,
    proxy_chromosomes: str | None,
    sample: str,
    threads: int,
    output_dir: Path,
    diagnostics_dir: Path,
    skip_align: bool,
    min_junction_support: int,
    min_coverage: float,
    min_score: float,
    max_paths: int,
    motif_validation: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "benchmarks/run_real_benchmark.py",
        "--sample",
        sample,
        "--braid-only",
        "--threads",
        str(threads),
        "--decomposer",
        variant.decomposer,
        "--builder-profile",
        variant.builder_profile,
        "--min-junction-support",
        str(min_junction_support),
        "--min-coverage",
        str(min_coverage),
        "--min-score",
        str(min_score),
        "--max-paths",
        str(max_paths),
        "--output-dir",
        str(output_dir),
        "--diagnostics-dir",
        str(diagnostics_dir),
    ]
    if skip_align:
        cmd.append("--skip-align")
    if not motif_validation:
        cmd.append("--no-motif-validation")
    if mode == "proxy":
        if not proxy_chromosomes:
            raise ValueError("proxy_chromosomes must be set in proxy mode")
        cmd.extend(["--chr", proxy_chromosomes])
    return cmd


def _iter_process_tree(root_pid: int) -> list[psutil.Process]:
    """Return the current process tree rooted at *root_pid*."""
    try:
        root = psutil.Process(root_pid)
    except psutil.Error:
        return []
    processes = [root]
    try:
        processes.extend(root.children(recursive=True))
    except psutil.Error:
        pass
    return processes


def _prime_cpu_counters(processes: list[psutil.Process], seen: set[int]) -> None:
    """Prime psutil CPU counters for new processes."""
    for proc in processes:
        if proc.pid in seen:
            continue
        try:
            proc.cpu_percent(None)
            seen.add(proc.pid)
        except psutil.Error:
            continue


def _sample_process_tree(
    root_pid: int,
    seen: set[int],
    sample_interval: float,
) -> list[dict[str, Any]]:
    """Sample CPU and memory for a process tree after a sleep interval."""
    _prime_cpu_counters(_iter_process_tree(root_pid), seen)
    time.sleep(sample_interval)
    rows: list[dict[str, Any]] = []
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    for proc in _iter_process_tree(root_pid):
        try:
            mem = proc.memory_info()
            rows.append(
                {
                    "timestamp": timestamp,
                    "pid": proc.pid,
                    "ppid": proc.ppid(),
                    "cpu_percent": round(proc.cpu_percent(None), 3),
                    "rss_mb": round(mem.rss / (1024 * 1024), 3),
                    "vms_mb": round(mem.vms / (1024 * 1024), 3),
                    "cmd": " ".join(proc.cmdline()),
                }
            )
        except psutil.Error:
            continue
    return rows


def _write_cpu_samples(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write sampled process metrics to TSV."""
    fieldnames = ["timestamp", "pid", "ppid", "cpu_percent", "rss_mb", "vms_mb", "cmd"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _load_results_json(run_dir: Path, sample: str) -> dict[str, Any]:
    """Load the per-run results JSON produced by run_real_benchmark.py."""
    results_path = run_dir / f"{sample}_results.json"
    if not results_path.exists():
        return {}
    with open(results_path, encoding="utf-8") as fh:
        return json.load(fh)


def _summarize_variant_result(
    variant: VariantSpec,
    *,
    mode: str,
    proxy_chromosomes: str | None,
    run_dir: Path,
    sample: str,
    elapsed_seconds: float,
    cpu_rows: list[dict[str, Any]],
    returncode: int,
    timed_out: bool,
) -> dict[str, Any]:
    """Build one summary row from a completed variant run."""
    results = _load_results_json(run_dir, sample)
    braid = results.get("tools", {}).get("BRAID", {})
    metrics = braid.get("metrics", {})
    peak_tree_rss_mb = max((row["rss_mb"] for row in cpu_rows), default=0.0)
    row = {
        "variant": variant.name,
        "decomposer": variant.decomposer,
        "builder_profile": variant.builder_profile,
        "mode": mode,
        "status": "timed_out" if timed_out else ("ok" if returncode == 0 else "failed"),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "peak_tree_rss_mb": round(peak_tree_rss_mb, 3),
        "transcript_sensitivity": metrics.get("transcript_level_sensitivity", ""),
        "transcript_precision": metrics.get("transcript_level_precision", ""),
        "exon_sensitivity": metrics.get("exon_level_sensitivity", ""),
        "exon_precision": metrics.get("exon_level_precision", ""),
        "intron_sensitivity": metrics.get("intron_level_sensitivity", ""),
        "intron_precision": metrics.get("intron_level_precision", ""),
        "annotation_gtf": "",
        "proxy_chromosomes": proxy_chromosomes if mode == "proxy" and proxy_chromosomes else "",
        "results_json": str((run_dir / f"{sample}_results.json").resolve()),
        "cpu_samples_tsv": str((run_dir / "cpu_samples.tsv").resolve()),
        "diagnostics_dir": str((run_dir / "diagnostics").resolve()),
    }
    if results:
        row["annotation_gtf"] = results.get("annotation_gtf", "")
        if "proxy_chromosomes" in results:
            row["proxy_chromosomes"] = ",".join(results["proxy_chromosomes"])
    return row


def _run_variant(
    variant: VariantSpec,
    args: argparse.Namespace,
    *,
    mode: str,
    proxy_chromosomes: str | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one real-data benchmark variant and collect logs plus CPU samples."""
    run_dir = output_dir / variant.name
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = run_dir / "diagnostics"
    stdout_log = run_dir / "stdout.log"
    stderr_log = run_dir / "stderr.log"
    cpu_samples_path = run_dir / "cpu_samples.tsv"
    command = _build_benchmark_command(
        variant,
        mode=mode,
        proxy_chromosomes=proxy_chromosomes,
        sample=args.sample,
        threads=args.threads,
        output_dir=run_dir,
        diagnostics_dir=diagnostics_dir,
        skip_align=args.skip_align,
        min_junction_support=args.min_junction_support,
        min_coverage=args.min_coverage,
        min_score=args.min_score,
        max_paths=args.max_paths,
        motif_validation=not args.no_motif_validation,
    )
    logger.info("Running %s variant: %s", variant.name, " ".join(command))

    start = time.perf_counter()
    with open(stdout_log, "w", encoding="utf-8") as out_fh, open(
        stderr_log, "w", encoding="utf-8",
    ) as err_fh:
        process = subprocess.Popen(
            command,
            stdout=out_fh,
            stderr=err_fh,
            cwd=Path.cwd(),
            text=True,
        )
        cpu_rows: list[dict[str, Any]] = []
        seen_pids: set[int] = set()
        timed_out = False
        while process.poll() is None:
            cpu_rows.extend(
                _sample_process_tree(process.pid, seen_pids, args.sample_interval),
            )
            if (time.perf_counter() - start) > args.timeout_per_run:
                timed_out = True
                for child in _iter_process_tree(process.pid):
                    try:
                        child.kill()
                    except psutil.Error:
                        continue
                process.kill()
                break
        if timed_out:
            process.wait()
        elapsed = time.perf_counter() - start

    _write_cpu_samples(cpu_samples_path, cpu_rows)
    row = _summarize_variant_result(
        variant,
        mode=mode,
        proxy_chromosomes=proxy_chromosomes,
        run_dir=run_dir,
        sample=args.sample,
        elapsed_seconds=elapsed,
        cpu_rows=cpu_rows,
        returncode=process.returncode or 0,
        timed_out=timed_out,
    )
    row["command"] = command
    row["stdout_log"] = str(stdout_log.resolve())
    row["stderr_log"] = str(stderr_log.resolve())
    return row


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Persist matrix results as JSON and TSV."""
    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"variants": rows}, fh, indent=2, sort_keys=True)

    fieldnames = [
        "variant",
        "decomposer",
        "builder_profile",
        "mode",
        "status",
        "elapsed_seconds",
        "peak_tree_rss_mb",
        "transcript_sensitivity",
        "transcript_precision",
        "exon_sensitivity",
        "exon_precision",
        "intron_sensitivity",
        "intron_precision",
        "annotation_gtf",
        "proxy_chromosomes",
        "results_json",
        "cpu_samples_tsv",
        "diagnostics_dir",
        "stdout_log",
        "stderr_log",
    ]
    with open(output_dir / "summary.tsv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser(default_mode: str | None = None) -> argparse.ArgumentParser:
    """Build the CLI parser for real-data variant matrix runs."""
    parser = argparse.ArgumentParser(
        description="Run a real-data BRAID variant matrix in full or proxy mode.",
    )
    parser.add_argument("--sample", default="SRR387661")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to real_benchmark/results/<mode>/<timestamp>.",
    )
    parser.add_argument("--variants", nargs="+", default=["legacy", "iterative_v2"])
    parser.add_argument(
        "--builder-profile",
        choices=["default", "conservative_correctness", "aggressive_recall"],
        default="conservative_correctness",
    )
    parser.add_argument(
        "--skip-align",
        action="store_true",
        dest="skip_align",
        help="Reuse existing BAMs instead of re-running alignment (default).",
    )
    parser.add_argument(
        "--run-align",
        action="store_false",
        dest="skip_align",
        help="Force alignment instead of reusing existing BAMs.",
    )
    parser.set_defaults(skip_align=True)
    parser.add_argument("--min-junction-support", type=int, default=3)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--min-score", type=float, default=0.1)
    parser.add_argument("--max-paths", type=int, default=500)
    parser.add_argument("--no-motif-validation", action="store_true")
    parser.add_argument("--sample-interval", type=float, default=5.0)
    parser.add_argument("--timeout-per-run", type=float, default=6 * 3600)
    parser.add_argument("--mode", choices=["nightly", "proxy"], default=default_mode or "nightly")
    parser.add_argument(
        "--proxy-chromosomes",
        default="21,22",
        help="Comma-separated chromosomes for proxy mode (default: 21,22).",
    )
    return parser


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    """Run the requested variant matrix and write summary outputs."""
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else _default_output_dir(args.mode).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = args.mode
    proxy_chromosomes = args.proxy_chromosomes if mode == "proxy" else None
    rows = []
    for variant in _variant_specs(args.variants, args.builder_profile):
        rows.append(
            _run_variant(
                variant,
                args,
                mode=mode,
                proxy_chromosomes=proxy_chromosomes,
                output_dir=output_dir,
            )
        )
    _write_summary(output_dir, rows)
    return {"variants": rows}


def main(default_mode: str | None = None) -> None:
    """CLI entry point."""
    parser = build_parser(default_mode=default_mode)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    run_matrix(args)


if __name__ == "__main__":
    main()
