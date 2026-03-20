"""Tests for real-data BRAID variant matrix runner."""

from __future__ import annotations

import sys
from argparse import Namespace

from benchmarks.run_real_variant_matrix import (
    VariantSpec,
    _build_benchmark_command,
    _summarize_variant_result,
    build_parser,
)


def test_build_benchmark_command_for_proxy_mode_includes_chr_and_diagnostics(tmp_path) -> None:
    """Proxy matrix runs should forward chr filter and diagnostics path."""
    cmd = _build_benchmark_command(
        VariantSpec(name="legacy", decomposer="legacy", builder_profile="conservative_correctness"),
        mode="proxy",
        proxy_chromosomes="21,22",
        sample="SRR387661",
        threads=8,
        output_dir=tmp_path / "variant",
        diagnostics_dir=tmp_path / "variant" / "diagnostics",
        skip_align=True,
        min_junction_support=3,
        min_coverage=1.0,
        min_score=0.1,
        max_paths=500,
        motif_validation=True,
    )

    assert cmd[:2] == [sys.executable, "benchmarks/run_real_benchmark.py"]
    assert "--braid-only" in cmd
    assert "--chr" in cmd
    assert cmd[cmd.index("--chr") + 1] == "21,22"
    assert "--diagnostics-dir" in cmd


def test_build_benchmark_command_for_nightly_mode_omits_chr(tmp_path) -> None:
    """Full nightly runs should not inject proxy chromosomes."""
    cmd = _build_benchmark_command(
        VariantSpec(name="iterative_v2", decomposer="iterative_v2", builder_profile="conservative_correctness"),
        mode="nightly",
        proxy_chromosomes=None,
        sample="SRR387661",
        threads=8,
        output_dir=tmp_path / "variant",
        diagnostics_dir=tmp_path / "variant" / "diagnostics",
        skip_align=True,
        min_junction_support=3,
        min_coverage=1.0,
        min_score=0.1,
        max_paths=500,
        motif_validation=False,
    )

    assert "--chr" not in cmd
    assert "--no-motif-validation" in cmd


def test_summarize_variant_result_keeps_proxy_chromosomes_without_results_json(tmp_path) -> None:
    """Proxy summary rows should keep the requested chromosome subset on timeout/failure."""
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    variant = VariantSpec(
        name="legacy",
        decomposer="legacy",
        builder_profile="conservative_correctness",
    )

    row = _summarize_variant_result(
        variant,
        mode="proxy",
        proxy_chromosomes="21,22",
        run_dir=run_dir,
        sample="SRR387661",
        elapsed_seconds=10.0,
        cpu_rows=[],
        returncode=1,
        timed_out=True,
    )

    assert row["status"] == "timed_out"
    assert row["proxy_chromosomes"] == "21,22"
    assert row["annotation_gtf"] == ""


def test_build_parser_defaults_to_skip_align_but_allows_run_align_override() -> None:
    """The matrix runner should default to BAM reuse while allowing explicit alignment."""
    parser = build_parser(default_mode="proxy")

    args = parser.parse_args([])
    assert args.output_dir is None
    assert isinstance(args, Namespace)
    assert args.skip_align is True

    args = parser.parse_args(["--output-dir", "/tmp/out"])
    assert isinstance(args, Namespace)
    assert args.skip_align is True

    args = parser.parse_args(["--output-dir", "/tmp/out", "--run-align"])
    assert args.skip_align is False
