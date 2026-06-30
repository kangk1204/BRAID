"""Tests for real-data benchmark command construction."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmarks.run_benchmark import resolve_assembled_annotation_path
from benchmarks.run_real_benchmark import (
    ANNOTATION_GTF_CHR,
    ANNOTATION_GTF_NOCHR,
    REFERENCE_FASTA,
    RealBenchmarkConfig,
    _annotation_for_contig_style,
    _build_braid_command,
    _detect_contig_style_from_idxstats_output,
    _filter_annotation_gtf,
    count_transcripts,
)


def test_build_braid_command_passes_reference_when_motif_validation_enabled() -> None:
    """Motif-on real benchmark runs must pass the real reference FASTA."""
    cmd = _build_braid_command(
        "sample.bam",
        "out.gtf",
        RealBenchmarkConfig(
            threads=2,
            braid_engine="iterative_v2",
            braid_builder_profile="conservative_correctness",
            braid_max_paths=1024,
            braid_enable_motif_validation=True,
        ),
    )

    assert cmd[:4] == [sys.executable, "-m", "braid.cli", "assemble"]
    assert "-r" in cmd
    assert cmd[cmd.index("-r") + 1] == str(REFERENCE_FASTA)
    assert "--engine" in cmd
    assert cmd[cmd.index("--engine") + 1] == "iterative_v2"
    assert "--no-motif-validation" not in cmd


def test_build_braid_command_disables_motif_validation_explicitly() -> None:
    """Motif-off runs should not pass a reference FASTA and must add the flag."""
    cmd = _build_braid_command(
        "sample.bam",
        "out.gtf",
        RealBenchmarkConfig(braid_enable_motif_validation=False),
    )

    assert "--no-motif-validation" in cmd
    assert "-r" not in cmd


def test_build_braid_command_forwards_chromosomes_and_diagnostics() -> None:
    """Proxy runs should forward chromosome filters and diagnostics output."""
    cmd = _build_braid_command(
        "sample.bam",
        "out.gtf",
        RealBenchmarkConfig(braid_diagnostics_dir="diag"),
        chromosomes=["21", "22"],
    )

    assert "--chromosomes" in cmd
    assert cmd[cmd.index("--chromosomes") + 1] == "21,22"
    assert "--diagnostics-dir" in cmd
    assert cmd[cmd.index("--diagnostics-dir") + 1] == "diag"


def test_detect_contig_style_and_annotation_selection() -> None:
    """Real benchmark should match chr/nochr annotation to BAM namespace."""
    idxstats_chr = "chr1\t100\t10\t0\n*\t0\t0\t0\n"
    idxstats_nochr = "1\t100\t10\t0\n"

    assert _detect_contig_style_from_idxstats_output(idxstats_chr) == "chr"
    assert _detect_contig_style_from_idxstats_output(idxstats_nochr) == "nochr"
    assert _annotation_for_contig_style("chr") == ANNOTATION_GTF_CHR
    assert _annotation_for_contig_style("nochr") == ANNOTATION_GTF_NOCHR


def test_filter_annotation_gtf_keeps_only_requested_chromosomes(tmp_path) -> None:
    """Proxy evaluation should use an annotation filtered to selected chromosomes."""
    src = tmp_path / "input.gtf"
    src.write_text(
        "# header\n"
        "21\tsrc\texon\t1\t10\t.\t+\t.\tgene_id \"g1\"; transcript_id \"t1\";\n"
        "22\tsrc\texon\t1\t10\t.\t+\t.\tgene_id \"g2\"; transcript_id \"t2\";\n"
        "1\tsrc\texon\t1\t10\t.\t+\t.\tgene_id \"g3\"; transcript_id \"t3\";\n",
        encoding="utf-8",
    )
    out = tmp_path / "filtered.gtf"

    _filter_annotation_gtf(src, out, ["21", "22"])

    filtered_lines = out.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("21\tsrc\texon") for line in filtered_lines)
    assert any(line.startswith("22\tsrc\texon") for line in filtered_lines)
    assert not any(line.startswith("1\tsrc\texon") for line in filtered_lines)


def test_resolve_assembled_annotation_path_handles_directory_outputs(tmp_path) -> None:
    """Benchmark runners should accept CLI outputs materialized as directories."""
    out_dir = tmp_path / "assembled.gtf"
    out_dir.mkdir()
    (out_dir / "summary.tsv").write_text("sample\tpath\n", encoding="utf-8")
    sample_gtf = out_dir / "sample.gtf"
    sample_gtf.write_text(
        "chr1\tsrc\ttranscript\t1\t10\t.\t+\t.\tgene_id \"g1\"; transcript_id \"t1\";\n",
        encoding="utf-8",
    )

    assert resolve_assembled_annotation_path(out_dir) == sample_gtf
    assert count_transcripts(str(out_dir)) == 1


def test_head_to_head_method_seed_offset_is_stable() -> None:
    from benchmarks.headtohead.head_to_head_coverage import _stable_method_seed_offset

    assert _stable_method_seed_offset("MAJIQ") == _stable_method_seed_offset("MAJIQ")
    assert isinstance(_stable_method_seed_offset("BRAID-conformal"), int)


def test_comprehensive_benchmark_betas_alignment_fails_fast() -> None:
    from benchmarks.headtohead.comprehensive_benchmark import _real_betas_interval_for_target

    assert _real_betas_interval_for_target({0: (-0.2, 0.1)}, {0: -0.1}, 0, -0.1) == (-0.2, 0.1)
    with pytest.raises(ValueError, match="row-order alignment"):
        _real_betas_interval_for_target({0: (-0.2, 0.1)}, {0: -0.1}, 1, -0.1)
    with pytest.raises(ValueError, match="row-order alignment"):
        _real_betas_interval_for_target({0: (-0.2, 0.1)}, {0: -0.1}, 0, 0.5)


def test_head_to_head_betas_loader_rejects_missing_rows(tmp_path) -> None:
    """Missing real-betAS rows must not become perfect-coverage [-1, 1] intervals."""
    from benchmarks.headtohead.head_to_head_coverage import _load_betas_intervals

    intervals = tmp_path / "betas.tsv"
    intervals.write_text(
        "key\tdpsi_mean\tlow_0.50\thigh_0.50\tlow_0.80\thigh_0.80\t"
        "low_0.90\thigh_0.90\tlow_0.95\thigh_0.95\tlow_0.99\thigh_0.99\n"
        "ev0\t0.0\t-0.1\t0.1\t-0.2\t0.2\t-0.3\t0.3\t-0.4\t0.4\t-0.5\t0.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing betAS interval rows"):
        _load_betas_intervals(str(intervals), n=2)


def test_headline_claim_surfaces_do_not_reintroduce_stale_numbers() -> None:
    """Tracked reviewer-facing claim surfaces should not drift from current JSON."""
    claim_paths = [
        Path("BENCHMARK_HANDOFF.md"),
        Path("benchmarks/headtohead/README.md"),
        Path("braid/commands/differential.py"),
        Path("braid/target/conformal.py"),
        *sorted(Path("benchmarks/headtohead").glob("*.py")),
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in claim_paths)

    for stale in [
        # superseded pre-structure-aware-matching headline numbers
        "0.964",
        "0.727",
        "0.619",
        "1.518",
        "1.754",
        "0.959",
        # superseded draft artifacts
        "0.944 coverage over 143 events",
        "p≤3e-14",
        "112-event",
        "irreducible platform",
        "platform-discordance floor",
        "every populated bin",
        "does NOT under-cover",
    ]:
        assert stale not in text

    assert "betAS (real R package) | 0.734 | 1.414" in text
    assert "vs betAS 33-0 (p=2.33e-10)" in text
    assert "50-99 under-covers for all three policies" in text
