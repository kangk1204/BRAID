"""Tests for real-data benchmark command construction."""

from __future__ import annotations

import sys

from benchmarks.run_real_benchmark import (
    ANNOTATION_GTF_CHR,
    ANNOTATION_GTF_NOCHR,
    REFERENCE_FASTA,
    RealBenchmarkConfig,
    _annotation_for_contig_style,
    _build_braid_command,
    _detect_contig_style_from_idxstats_output,
    _filter_annotation_gtf,
)


def test_build_braid_command_passes_reference_when_motif_validation_enabled() -> None:
    """Motif-on real benchmark runs must pass the real reference FASTA."""
    cmd = _build_braid_command(
        "sample.bam",
        "out.gtf",
        RealBenchmarkConfig(
            threads=2,
            braid_decomposer="iterative_v2",
            braid_builder_profile="conservative_correctness",
            braid_max_paths=1024,
            braid_enable_motif_validation=True,
        ),
    )

    assert cmd[:4] == [sys.executable, "-m", "braid.cli", "assemble"]
    assert "-r" in cmd
    assert cmd[cmd.index("-r") + 1] == str(REFERENCE_FASTA)
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
