#!/usr/bin/env python3
"""Regenerate the PacBio long-read PSI-calibration validation from public data.

The committed ``benchmarks/results/rtpcr_benchmark.json`` holds only bin-level
summaries; the 204 per-event records needed to (a) deconfound the sharpness
result with a fair estimate-centred comparator and (b) run a betAS head-to-head
live in an external tree that is no longer on disk. This driver rebuilds them
end-to-end from the exact public accessions, so the make-or-break experiments
become reproducible from a clean machine.

Pipeline (one command, when the tools are installed):
  1. Download short-read K562 RNA-seq (SRA SRR387661) and the PacBio long-read
     transcript models (ENCODE ENCSR589FUJ; GTF ENCFF652QLH), plus GENCODE v38
     and a GRCh38 reference (unless --skip-download / paths provided).
  2. Align short reads (HISAT2 or STAR) -> coordinate-sorted, indexed BAM.
  3. Run BRAID's PacBio validation (benchmarks/rtpcr_benchmark.py), which derives
     short-read inclusion/exclusion counts and long-read junction PSI per event
     and fits the leave-one-gene-out calibration.
  4. Emit a MINIMAL, commit-safe per-event fixture (event id, support, inc/exc,
     point estimate, long-read PSI, CI) -- never the raw 275 GB inputs.
  5. Run benchmarks/deconfounded_sharpness_analysis.py on that fixture.

This script intentionally shells out to external bioinformatics tools; it checks
for them and prints actionable instructions rather than failing opaquely. Use
--dry-run to print the plan without executing.

Accessions (see paper Data Availability):
  short-read K562     : SRA SRR387661
  PacBio long-read    : ENCODE ENCSR589FUJ  (GTF ENCFF652QLH)
  annotation          : GENCODE v38
  reference           : GRCh38 primary assembly
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SRR = "SRR387661"
ENCODE_LR_GTF = "ENCFF652QLH"  # PacBio long-read transcript models (K562)
REQUIRED_TOOLS = {
    "fasterq-dump": "sra-tools (conda install -c bioconda sra-tools)",
    "samtools": "samtools (conda install -c bioconda samtools)",
}
ALIGNERS = {
    "hisat2": "hisat2 + hisat2-build (conda install -c bioconda hisat2)",
    "STAR": "STAR (conda install -c bioconda star)",
}


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


def _run(cmd: list[str], *, dry_run: bool, cwd: Path | None = None) -> None:
    printable = " ".join(cmd)
    print(f"  $ {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def check_environment(aligner: str) -> list[str]:
    """Return a list of missing-tool messages (empty if the environment is ready)."""
    missing = []
    for tool, how in REQUIRED_TOOLS.items():
        if not _have(tool):
            missing.append(f"{tool!r} not found -> install {how}")
    if not _have(aligner):
        missing.append(f"{aligner!r} not found -> install {ALIGNERS[aligner]}")
    return missing


def step_download(work: Path, *, dry_run: bool) -> Path:
    """Fetch FASTQ for SRR387661 and the PacBio GTF. Returns the fastq dir."""
    raw = work / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"[1/5] Download {SRR} (FASTQ) and {ENCODE_LR_GTF} (PacBio GTF)")
    _run(["prefetch", SRR, "-O", str(raw)], dry_run=dry_run)
    _run(["fasterq-dump", "--split-files", "-O", str(raw), str(raw / SRR)], dry_run=dry_run)
    lr_gtf = raw / f"{ENCODE_LR_GTF}.gtf.gz"
    _run(
        ["curl", "-L", "-o", str(lr_gtf),
         f"https://www.encodeproject.org/files/{ENCODE_LR_GTF}/@@download/{ENCODE_LR_GTF}.gtf.gz"],
        dry_run=dry_run,
    )
    return raw


def step_align(
    work: Path, raw: Path, reference: Path, aligner: str, threads: int, *, dry_run: bool
) -> Path:
    """Align short reads to a coordinate-sorted, indexed BAM. Returns the BAM path."""
    print(f"[2/5] Align {SRR} with {aligner} ({threads} threads)")
    bam_dir = work / "bam"
    bam_dir.mkdir(parents=True, exist_ok=True)
    bam = bam_dir / f"{SRR}.sorted.bam"
    r1, r2 = raw / f"{SRR}_1.fastq", raw / f"{SRR}_2.fastq"
    if aligner == "hisat2":
        idx = reference.with_suffix("")  # expects prebuilt hisat2 index prefix
        sam = bam_dir / f"{SRR}.sam"
        _run(["hisat2", "-p", str(threads), "-x", str(idx),
              "-1", str(r1), "-2", str(r2), "-S", str(sam)], dry_run=dry_run)
        _run(["samtools", "sort", "-@", str(threads), "-o", str(bam), str(sam)], dry_run=dry_run)
    else:  # STAR
        _run(["STAR", "--runThreadN", str(threads), "--genomeDir", str(reference),
              "--readFilesIn", str(r1), str(r2),
              "--outSAMtype", "BAM", "SortedByCoordinate",
              "--outFileNamePrefix", str(bam_dir / f"{SRR}.")], dry_run=dry_run)
        # STAR names its output {prefix}Aligned.sortedByCoord.out.bam; rename it to the
        # expected coordinate-sorted BAM path.
        star_bam = bam_dir / f"{SRR}.Aligned.sortedByCoord.out.bam"
        if not dry_run and star_bam.exists():
            star_bam.replace(bam)
    _run(["samtools", "index", str(bam)], dry_run=dry_run)
    return bam


def step_validate(work: Path, bam: Path, lr_gtf: Path, gtf: Path, *, dry_run: bool) -> Path:
    """Run BRAID's PacBio validation; returns the rtpcr_benchmark.json path."""
    print("[3/5] Run BRAID PacBio validation (per-event PSI + LOGO calibration)")
    out = work / "results" / "rtpcr_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # The long-read GTF must be uncompressed for the validation parser.
    if str(lr_gtf).endswith(".gz"):
        plain = lr_gtf.with_suffix("")  # drop the .gz
        if not dry_run and not plain.exists():
            _run(["gunzip", "-k", str(lr_gtf)], dry_run=dry_run)
        lr_gtf = plain
    # rtpcr_benchmark.py reads its inputs from these env vars; BAM, annotation, and
    # long-read GTF must share one chromosome-naming convention (all "chr" here).
    os.environ["BRAID_PACBIO_BAM"] = str(bam)
    os.environ["BRAID_PACBIO_GTF"] = str(gtf)
    os.environ["BRAID_PACBIO_LR_GTF"] = str(lr_gtf)
    _run([sys.executable, "benchmarks/rtpcr_benchmark.py", "--output", str(out)], dry_run=dry_run)
    return out


def step_emit_fixture(rtpcr_json: Path, fixture_out: Path, *, dry_run: bool) -> None:
    """Extract a minimal, commit-safe per-event fixture from the full result."""
    print(f"[4/5] Emit minimal per-event fixture -> {fixture_out}")
    if dry_run or not rtpcr_json.exists():
        print("  (skipped: dry-run or rtpcr_benchmark.json absent)")
        return
    data = json.loads(rtpcr_json.read_text())
    events = data.get("pacbio_psi", {}).get("per_event") or data.get("events") or []
    keep = ("event_id", "event_type", "gene", "support", "inclusion", "exclusion",
            "psi_hat", "lr_psi", "ci_low", "ci_high")
    minimal = [{k: ev[k] for k in keep if k in ev} for ev in events]
    fixture_out.parent.mkdir(parents=True, exist_ok=True)
    fixture_out.write_text(json.dumps({"per_event": minimal}, indent=2))
    print(f"  wrote {len(minimal)} per-event records")


def step_analyze(fixture_out: Path, *, dry_run: bool) -> None:
    print("[5/5] Deconfounded sharpness + conformal analysis")
    _run([sys.executable, "benchmarks/deconfounded_sharpness_analysis.py",
          "--fixture", str(fixture_out)], dry_run=dry_run)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--workdir", type=Path, default=Path("benchmark_results/pacbio_regen"))
    p.add_argument("--reference", type=Path, help="HISAT2 index prefix or STAR genomeDir")
    p.add_argument("--annotation", type=Path, help="GENCODE v38 GTF")
    p.add_argument("--aligner", choices=sorted(ALIGNERS), default="hisat2")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--skip-download", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    p.add_argument("--fixture-out", type=Path,
                   default=Path("benchmarks/results/pacbio_per_event_fixture.json"))
    args = p.parse_args(argv)

    missing = check_environment(args.aligner)
    if missing and not args.dry_run:
        print("Environment not ready:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("\nRe-run with --dry-run to print the plan, or install the tools above.",
              file=sys.stderr)
        return 2

    work = args.workdir
    work.mkdir(parents=True, exist_ok=True)
    raw = (work / "raw") if args.skip_download else step_download(work, dry_run=args.dry_run)
    if args.reference is None:
        print("ERROR: --reference (HISAT2 index prefix or STAR genomeDir) is required.",
              file=sys.stderr)
        return 2
    bam = step_align(work, raw, args.reference, args.aligner, args.threads, dry_run=args.dry_run)
    lr_gtf = raw / f"{ENCODE_LR_GTF}.gtf.gz"
    rtpcr_json = step_validate(work, bam, lr_gtf, args.annotation, dry_run=args.dry_run)
    step_emit_fixture(rtpcr_json, args.fixture_out, dry_run=args.dry_run)
    step_analyze(args.fixture_out, dry_run=args.dry_run)
    print("\nDone. The per-event fixture enables the deconfounded sharpness and "
          "betAS head-to-head experiments (paper Limitations).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
