"""Installation diagnostics for BRAID."""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import json
import platform
import shutil
import sys
from typing import Any

CORE_PACKAGE_SPECS: list[dict[str, Any]] = [
    {
        "label": "braid",
        "module": "rapidsplice",
        "distribution": "braid",
        "distribution_fallbacks": ["rapidsplice"],
        "note": "legacy module path: rapidsplice",
    },
    {"label": "pysam", "module": "pysam", "distribution": "pysam"},
    {"label": "numpy", "module": "numpy", "distribution": "numpy"},
    {"label": "scipy", "module": "scipy", "distribution": "scipy"},
    {"label": "scikit-learn", "module": "sklearn", "distribution": "scikit-learn"},
    {"label": "intervaltree", "module": "intervaltree", "distribution": "intervaltree"},
    {"label": "numba", "module": "numba", "distribution": "numba"},
    {"label": "networkx", "module": "networkx", "distribution": "networkx"},
]

BENCHMARK_PACKAGE_SPECS: list[dict[str, Any]] = [
    {
        "label": "pyliftover",
        "module": "pyliftover",
        "distribution": "pyliftover",
        "note": "QKI target liftover",
    },
]

OPTIONAL_PACKAGE_SPECS: list[dict[str, Any]] = [
    {
        "label": "cupy",
        "module": "cupy",
        "distribution": "cupy-cuda12x",
        "note": "GPU-only",
    },
    {
        "label": "streamlit",
        "module": "streamlit",
        "distribution": "streamlit",
        "note": "dashboard",
    },
    {
        "label": "pandas",
        "module": "pandas",
        "distribution": "pandas",
        "note": "dashboard / benchmark reports",
    },
    {
        "label": "plotly",
        "module": "plotly",
        "distribution": "plotly",
        "note": "dashboard / figures",
    },
]

OPTIONAL_COMMAND_SPECS: list[dict[str, Any]] = [
    {"label": "samtools", "command": "samtools", "note": "BAM sorting/indexing and benchmarks"},
    {"label": "hisat2", "command": "hisat2", "note": "alignment and benchmark prep"},
    {"label": "stringtie", "command": "stringtie", "note": "benchmark baseline / bootstrap"},
    {"label": "rmats.py", "command": "rmats.py", "note": "QKI benchmark / splicing comparisons"},
    {"label": "gffcompare", "command": "gffcompare", "note": "benchmark evaluation"},
]


def _check_python_package(spec: dict[str, Any]) -> dict[str, Any]:
    """Check whether one Python package is importable."""
    module_name = spec["module"]
    dist_name = spec["distribution"]
    fallback_dists = list(spec.get("distribution_fallbacks", []))
    try:
        module = importlib.import_module(module_name)
        version = "unknown"
        for candidate in [dist_name, *fallback_dists]:
            try:
                version = importlib_metadata.version(candidate)
                break
            except importlib_metadata.PackageNotFoundError:
                continue
        return {
            "label": spec["label"],
            "module": module_name,
            "distribution": dist_name,
            "installed": True,
            "version": version,
            "path": getattr(module, "__file__", None),
            "note": spec.get("note"),
        }
    except Exception as exc:  # pragma: no cover - defensive and user-facing.
        return {
            "label": spec["label"],
            "module": module_name,
            "distribution": dist_name,
            "installed": False,
            "version": None,
            "path": None,
            "note": spec.get("note"),
            "error": str(exc),
        }


def _check_command(spec: dict[str, Any]) -> dict[str, Any]:
    """Check whether one external command is on PATH."""
    path = shutil.which(spec["command"])
    return {
        "label": spec["label"],
        "command": spec["command"],
        "installed": path is not None,
        "path": path,
        "note": spec.get("note"),
    }


def build_install_report(*, strict: bool = False, gpu: bool = False) -> dict[str, Any]:
    """Build a machine-readable install report.

    Args:
        strict: If True, treat benchmark package and command checks as required
            for the final status. This is useful when validating the BRAID
            benchmark/paper stack.
        gpu: If True, require GPU-only dependencies as well.

    Returns:
        A nested dictionary with core package checks, optional package checks,
        optional command checks, and summary flags.
    """
    core_packages = [_check_python_package(spec) for spec in CORE_PACKAGE_SPECS]
    benchmark_packages = [_check_python_package(spec) for spec in BENCHMARK_PACKAGE_SPECS]
    optional_packages = [_check_python_package(spec) for spec in OPTIONAL_PACKAGE_SPECS]
    optional_commands = [_check_command(spec) for spec in OPTIONAL_COMMAND_SPECS]

    core_ok = all(item["installed"] for item in core_packages)
    benchmark_ok = all(item["installed"] for item in benchmark_packages)
    optional_ok = all(item["installed"] for item in optional_packages)
    commands_ok = all(item["installed"] for item in optional_commands)
    gpu_ok = all(item["installed"] for item in optional_packages if item["label"] == "cupy")

    status_ok = core_ok
    if strict:
        status_ok = status_ok and benchmark_ok and commands_ok
    if gpu:
        status_ok = status_ok and gpu_ok

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
        },
        "core_packages": core_packages,
        "benchmark_packages": benchmark_packages,
        "optional_packages": optional_packages,
        "optional_commands": optional_commands,
        "summary": {
            "strict": strict,
            "gpu": gpu,
            "core_ready": core_ok,
            "benchmark_packages_ready": benchmark_ok,
            "optional_packages_ready": optional_ok,
            "optional_commands_ready": commands_ok,
            "gpu_ready": gpu_ok,
            "status": "ok" if status_ok else "needs_attention",
            "missing_core_packages": [
                item["label"] for item in core_packages if not item["installed"]
            ],
            "missing_benchmark_packages": [
                item["label"] for item in benchmark_packages if not item["installed"]
            ],
            "missing_optional_packages": [
                item["label"] for item in optional_packages if not item["installed"]
            ],
            "missing_gpu_packages": [
                item["label"]
                for item in optional_packages
                if item["label"] == "cupy" and not item["installed"]
            ],
            "missing_optional_commands": [
                item["label"] for item in optional_commands if not item["installed"]
            ],
        },
    }


def format_install_report(report: dict[str, Any]) -> str:
    """Format an install report for human-readable terminal output."""
    lines: list[str] = []
    lines.append("BRAID install check")
    lines.append(
        "Platform: "
        f"{report['platform']['system']} {report['platform']['release']} "
        f"({report['platform']['machine']})"
    )
    lines.append(f"Python: {report['python']['version']}")
    lines.append("")

    def add_section(title: str, rows: list[dict[str, Any]]) -> None:
        lines.append(title)
        for row in rows:
            status = "OK" if row["installed"] else "MISSING"
            if not row["installed"] and row.get("error"):
                extra = row["error"]
            else:
                extra = row.get("version") or row.get("path") or row.get("note") or ""
            if row["installed"] and row.get("note"):
                extra = f"{row['version']} ({row['note']})" if row.get("version") else row["note"]
            suffix = f" - {extra}" if extra else ""
            lines.append(f"  [{status}] {row['label']}{suffix}")
        lines.append("")

    add_section("Core Python packages", report["core_packages"])
    add_section("Benchmark Python packages", report["benchmark_packages"])
    add_section("Optional Python packages", report["optional_packages"])
    add_section("Optional command-line tools", report["optional_commands"])

    summary = report["summary"]
    if summary["status"] == "ok":
        lines.append("Status: OK")
    else:
        lines.append("Status: needs attention")

    if summary["missing_core_packages"]:
        lines.append(
            "Missing core packages: " + ", ".join(summary["missing_core_packages"])
        )
    if summary["missing_benchmark_packages"]:
        lines.append(
            "Missing benchmark Python packages: "
            + ", ".join(summary["missing_benchmark_packages"])
        )
    if summary["missing_optional_packages"]:
        lines.append(
            "Missing optional Python packages: "
            + ", ".join(summary["missing_optional_packages"])
        )
    if summary["missing_optional_commands"]:
        lines.append(
            "Missing optional benchmark tools: "
            + ", ".join(summary["missing_optional_commands"])
        )
    if summary["gpu"] and summary["missing_gpu_packages"]:
        lines.append(
            "Missing GPU Python packages: "
            + ", ".join(summary["missing_gpu_packages"])
        )

    if summary["strict"]:
        lines.append(
            "Strict mode checks the BRAID benchmark stack: core packages, "
            "benchmark Python packages, and benchmark command-line tools."
        )

    if summary["gpu"]:
        lines.append(
            "GPU mode checks the optional CUDA stack and requires cupy to import cleanly."
        )

    lines.append(
        "Tip: missing optional tools are fine for core BRAID Python workflows. "
        "Install them only if you plan to run the benchmark scripts, FASTQ workflows, "
        "or the interactive dashboard."
    )
    return "\n".join(lines)


def report_install_check(
    *,
    strict: bool = False,
    gpu: bool = False,
    json_output: bool = False,
) -> int:
    """Print a BRAID install report and return an exit code."""
    report = build_install_report(strict=strict, gpu=gpu)
    if json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_install_report(report))
    return 0 if report["summary"]["status"] == "ok" else 1
