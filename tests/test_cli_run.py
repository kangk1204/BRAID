"""CLI regression tests for BRAID subcommands.

Each test invokes ``python -m braid ...`` via subprocess and checks
exit codes and key strings in output.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_braid(*args: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m braid <args>`` and capture output.

    Args:
        args: CLI arguments after ``braid``.

    Returns:
        Completed process with captured stdout and stderr.
    """
    return subprocess.run(
        [sys.executable, "-m", "braid", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_top_level_help_exposes_only_supported_public_commands() -> None:
    """``braid -h`` advertises the supported rMATS post-processing surface."""
    result = _run_braid("-h")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr

    for command in ("psi", "differential", "example", "doctor"):
        assert command in combined

    hidden_terms = (
        "The Four Modes",
        "braid run",
        "assemble",
        "analyze",
        "denovo",
        "dashboard",
        "target",
        "fastq-target",
        "StringTie",
        "GPU",
    )
    for term in hidden_terms:
        assert term not in combined


def test_legacy_run_help_still_works_when_not_advertised() -> None:
    """Hidden legacy commands remain callable for compatibility."""
    result = _run_braid("run", "-h")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "auto-detects" in combined.lower()


def test_run_help() -> None:
    """``braid run -h`` exits 0 and mentions rMATS auto-detection."""
    result = _run_braid("run", "-h")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "auto-detects" in combined.lower(), (
        f"Expected 'auto-detects' in help output, got:\n{combined}"
    )
    assert "--fdr" in combined


def test_psi_help() -> None:
    """``braid psi -h`` exits 0 and exposes only rMATS-backed inputs."""
    result = _run_braid("psi", "-h")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "--bam" in combined, (
        f"Expected '--bam' in help output, got:\n{combined}"
    )
    assert "--rmats-dir" in combined
    assert "--gtf" not in combined
    assert "--gene" not in combined
    assert "--region" not in combined


def test_differential_help() -> None:
    """``braid differential -h`` exits 0 and shows the required --rmats-dir option."""
    result = _run_braid("differential", "-h")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "--rmats-dir" in combined, (
        f"Expected '--rmats-dir' in help output, got:\n{combined}"
    )
    # The decorative --ctrl/--treat options were removed; they must not reappear.
    assert "--ctrl" not in combined and "--treat" not in combined


def test_diff_is_alias_for_differential() -> None:
    """``braid diff`` is accepted as a short alias for ``braid differential``."""
    from braid.cli import create_parser
    from braid.commands.differential import run_differential

    args = create_parser().parse_args(["diff", "--rmats-dir", "X"])
    assert args.command in {"diff", "differential"}
    assert args.func is run_differential

    result = _run_braid("diff", "-h")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "--rmats-dir" in (result.stdout + result.stderr)
    top = _run_braid("-h")
    assert "differential (diff)" in (top.stdout + top.stderr)


def test_diff_alias_runs_identically_to_differential(tmp_path) -> None:
    """``braid diff`` produces byte-identical output to ``braid differential``."""
    from pathlib import Path

    import braid

    rmats_dir = Path(braid.__file__).parent / "examples" / "filter" / "rmats_output"
    diff_out = tmp_path / "diff.tsv"
    full_out = tmp_path / "differential.tsv"

    r1 = _run_braid("diff", "--rmats-dir", str(rmats_dir), "-o", str(diff_out))
    r2 = _run_braid("differential", "--rmats-dir", str(rmats_dir), "-o", str(full_out))
    assert r1.returncode == 0, f"stderr: {r1.stderr}"
    assert r2.returncode == 0, f"stderr: {r2.stderr}"
    assert diff_out.read_text(encoding="utf-8") == full_out.read_text(encoding="utf-8")


def test_run_no_args() -> None:
    """``braid run`` without arguments should print help or error, not crash."""
    result = _run_braid("run")
    # Accept either exit 0 (help printed) or exit 2 (argparse error) —
    # the key requirement is that it does NOT crash with a traceback.
    assert result.returncode in (0, 1, 2), (
        f"Unexpected exit code {result.returncode}; "
        f"stderr: {result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined, (
        f"Command crashed with traceback:\n{combined}"
    )


def test_unknown_command() -> None:
    """``braid nonexistent`` should give a clear error, not silently assemble."""
    result = _run_braid("nonexistent")
    assert result.returncode != 0, (
        "Expected non-zero exit for unknown command"
    )
    combined = result.stdout + result.stderr
    assert "Unknown command" in combined, (
        f"Expected 'Unknown command' error, got:\n{combined}"
    )


def test_version() -> None:
    """``braid --version`` prints the version string."""
    result = _run_braid("--version")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "0.1.0" in combined, (
        f"Expected '0.1.0' in version output, got:\n{combined}"
    )


def test_run_help_is_rmats_only() -> None:
    """``braid run -h`` exposes only rMATS-backed run modes."""
    result = _run_braid("run", "-h")
    assert result.returncode == 0, f"stderr: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "--rmats" in combined, (
        f"Expected '--rmats' in run help output, got:\n{combined}"
    )
    assert "BAM only" not in combined
    assert "assemble" not in combined
    assert "--bootstrap" not in combined
    assert "--stringtie" not in combined
    assert "StringTie" not in combined
    assert "score mode" not in combined


def test_differential_empty_output() -> None:
    """``braid differential`` with empty rMATS input produces exit 0 and output file."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal empty SE.MATS.JC.txt with just the header
        rmats_file = os.path.join(tmpdir, "SE.MATS.JC.txt")
        header = (
            "ID\tGeneID\tgeneSymbol\tchr\tstrand\t"
            "exonStart_0base\texonEnd\tupstreamES\tupstreamEE\t"
            "downstreamES\tdownstreamEE\t"
            "ID.1\tIJC_SAMPLE_1\tSJC_SAMPLE_1\t"
            "IJC_SAMPLE_2\tSJC_SAMPLE_2\t"
            "IncFormLen\tSkipFormLen\t"
            "PValue\tFDR\tIncLevel1\tIncLevel2\tIncLevelDifference\n"
        )
        with open(rmats_file, "w") as f:
            f.write(header)

        out_file = os.path.join(tmpdir, "out.tsv")
        result = _run_braid(
            "differential",
            "--rmats-dir", tmpdir,
            "-o", out_file,
        )
        assert result.returncode == 0, (
            f"Expected exit 0, got {result.returncode}; "
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert os.path.exists(out_file), (
            f"Output file {out_file} was not created"
        )


def test_run_requires_rmats_for_bam_only_inputs() -> None:
    """``braid run`` no longer falls back to hidden de novo assembly."""
    from argparse import Namespace

    import braid.commands.run as run_mod

    args = Namespace(ctrl=None, treat=None, rmats=None)
    with pytest.raises(SystemExit, match="--rmats is required"):
        run_mod._detect_applicable_modes(args)


def test_run_psi_mode_forwards_contract_flags(monkeypatch, tmp_path) -> None:
    from argparse import Namespace

    import braid.commands.run as run_mod

    captured = {}

    def fake_run_psi(args):
        captured.update(vars(args))

    monkeypatch.setattr("braid.commands.psi.run_psi", fake_run_psi)
    outdir = tmp_path / "run_out"
    run_mod._run_psi_mode(
        Namespace(
            rmats="/x/rmats",
            gtf="ann.gtf",
            replicates=5,
            confidence=0.9,
            min_support=17,
            seed=3,
            verbose=True,
            use_conformal=False,
            calibration="cal.json",
            allow_replicate_fallback=True,
        ),
        ["/x/a.bam", "/x/b.bam"],
        str(outdir),
    )

    assert captured["min_support"] == 17
    assert captured["use_conformal"] is False
    assert captured["calibration"] == "cal.json"
    assert captured["allow_replicate_fallback"] is True


def test_run_differential_mode_forwards_fdr_contract(monkeypatch, tmp_path) -> None:
    from argparse import Namespace

    import braid.commands.run as run_mod

    captured = {}

    def fake_run_differential(args):
        captured.update(vars(args))

    monkeypatch.setattr(
        "braid.commands.differential.run_differential",
        fake_run_differential,
    )
    outdir = tmp_path / "run_out"
    run_mod._run_differential_mode(
        Namespace(
            ctrl=["c.bam"],
            treat=["t.bam"],
            rmats="/x/rmats",
            replicates=5,
            confidence=0.9,
            effect_cutoff=0.2,
            fdr=0.1,
            min_support=17,
            seed=3,
            verbose=True,
        ),
        str(outdir),
    )

    assert captured["effect_cutoff"] == 0.2
    assert captured["fdr"] == 0.1
    assert captured["min_support"] == 17


def test_run_rmats_only_psi_succeeds(tmp_path) -> None:
    """Regression: ``braid run --rmats <dir>`` with no BAM runs psi from the rMATS
    tables (BAMs are optional provenance), consistent with ``braid psi --rmats-dir``;
    it must not hard-fail with "at least one BAM file is required"."""
    import os

    rmats_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "braid", "examples", "filter", "rmats_output",
    )
    out = tmp_path / "run_out"
    result = _run_braid("run", "--rmats", rmats_dir, "-o", str(out))
    assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
    assert (out / "psi.tsv").exists()
