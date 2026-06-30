"""Tests for the `braid assemble` output-path contract.

`--help` and the README promise: a single BAM writes a GTF **file**, multiple
BAMs write a **directory** of per-sample GTFs plus a summary. The implementation
used to `os.makedirs(args.output)` unconditionally, so `braid assemble -o
transcripts.gtf` on one BAM produced a *directory* named `transcripts.gtf/`,
contradicting the documented contract. These tests pin the contract.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from braid.cli import _run_assemble, create_parser


def test_assemble_parser_defers_default_output_to_runner():
    args = create_parser().parse_args(["assemble", "sample.bam"])
    assert args.output is None


class _StubPipeline:
    """Writes a minimal GTF to the config's output path, like the real run()."""

    def __init__(self, config: object) -> None:
        self.config = config

    def run(self) -> str:
        path = self.config.output_path
        with open(path, "w") as f:
            f.write(
                'chr1\tbraid\ttranscript\t1\t100\t.\t+\t.\t'
                'gene_id "G"; transcript_id "G.1";\n'
            )
        return path


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "braid.cli._build_pipeline_config",
        lambda bam, args, out: SimpleNamespace(output_path=out),
    )
    monkeypatch.setattr("braid.cli.AssemblyPipeline", _StubPipeline)


def test_single_bam_output_is_a_file(tmp_path, monkeypatch):
    _patch(monkeypatch)
    out = tmp_path / "transcripts.gtf"
    _run_assemble(Namespace(
        bam=[str(tmp_path / "s.bam")], output=str(out), output_format="gtf",
    ))
    assert out.is_file()
    assert not out.is_dir()
    # No directory-mode artifacts for a single BAM.
    assert not (tmp_path / "transcripts.gtf" / "summary.tsv").exists()


def test_multi_bam_output_is_a_directory(tmp_path, monkeypatch):
    _patch(monkeypatch)
    outdir = tmp_path / "results"
    _run_assemble(Namespace(
        bam=[str(tmp_path / "a.bam"), str(tmp_path / "b.bam")],
        output=str(outdir), output_format="gtf",
    ))
    assert outdir.is_dir()
    assert (outdir / "a.gtf").is_file()
    assert (outdir / "b.gtf").is_file()
    assert (outdir / "summary.tsv").is_file()


def test_single_bam_default_output_is_documented_gtf(monkeypatch, tmp_path):
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _run_assemble(Namespace(
        bam=[str(tmp_path / "s.bam")], output=None, output_format="gtf",
    ))
    assert (tmp_path / "braid_output.gtf").is_file()


def test_multi_bam_default_output_is_documented_directory(monkeypatch, tmp_path):
    _patch(monkeypatch)
    monkeypatch.chdir(tmp_path)
    _run_assemble(Namespace(
        bam=[str(tmp_path / "a.bam"), str(tmp_path / "b.bam")],
        output=None, output_format="gtf",
    ))
    assert (tmp_path / "braid_output").is_dir()
    assert (tmp_path / "braid_output" / "summary.tsv").is_file()
