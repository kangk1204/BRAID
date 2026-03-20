#!/usr/bin/env python3
"""Multi-dataset benchmark for RapidSplice vs StringTie.

Aligns FASTQ files with HISAT2, runs both assemblers, evaluates with
GFFcompare against GENCODE v38, and generates a combined benchmark report.

Usage:
    python benchmarks/run_multi_dataset_benchmark.py --dataset GM12878
    python benchmarks/run_multi_dataset_benchmark.py --dataset IMR90
    python benchmarks/run_multi_dataset_benchmark.py --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetConfig:
    """Configuration for a single RNA-seq dataset."""

    name: str
    sra_id: str
    fastq_dir: str
    bam_path: str
    read_length: int
    description: str
    chr_prefix: bool = False  # Whether BAM uses "chr" prefix


BASE_DIR = Path(__file__).resolve().parent.parent / "real_benchmark"
REF_DIR = BASE_DIR / "reference" / "grch38"
GENOME_FA = REF_DIR / "genome.fa"
HISAT2_INDEX = REF_DIR / "genome"
GENCODE_NOCHR = BASE_DIR / "annotation" / "gencode.v38.nochr.gtf"
GENCODE_CHR = BASE_DIR / "annotation" / "gencode.v38.annotation.gtf"
RESULTS_DIR = BASE_DIR / "results"

DATASETS = {
    "K562": DatasetConfig(
        name="K562",
        sra_id="SRR387661",
        fastq_dir=str(BASE_DIR / "fastq" / "K562"),
        bam_path=str(BASE_DIR / "bam" / "SRR387661.bam"),
        read_length=76,
        description="K562 CML cell line (ENCODE, 124.8M reads)",
        chr_prefix=False,
    ),
    "GM12878": DatasetConfig(
        name="GM12878",
        sra_id="ENCFF550SET",
        fastq_dir=str(BASE_DIR / "fastq" / "GM12878"),
        bam_path=str(BASE_DIR / "bam" / "GM12878_ENCFF550SET.bam"),
        read_length=101,
        description="GM12878 lymphoblastoid (ENCODE Tier 1, STAR/GRCh38)",
        chr_prefix=True,
    ),
    "IMR90": DatasetConfig(
        name="IMR90",
        sra_id="ENCFF560TMJ",
        fastq_dir=str(BASE_DIR / "fastq" / "IMR90"),
        bam_path=str(BASE_DIR / "bam" / "IMR90_ENCFF560TMJ.bam"),
        read_length=101,
        description="IMR90 lung fibroblast (ENCODE10, STAR/GRCh38)",
        chr_prefix=True,
    ),
}


def run_cmd(cmd: list[str], desc: str, timeout: int = 86400) -> bool:
    """Run a shell command with logging."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd[:5])}...")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, timeout=timeout, capture_output=False,
        )
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s (exit code {result.returncode})")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def align_with_hisat2(ds: DatasetConfig, threads: int = 8) -> bool:
    """Align FASTQ files with HISAT2."""
    bam_path = Path(ds.bam_path)
    if bam_path.exists():
        print(f"  BAM already exists: {bam_path}")
        return True

    fastq_dir = Path(ds.fastq_dir)
    r1 = fastq_dir / f"{ds.sra_id}_1.fastq"
    r2 = fastq_dir / f"{ds.sra_id}_2.fastq"

    # Check for gzipped versions
    if not r1.exists() and (fastq_dir / f"{ds.sra_id}_1.fastq.gz").exists():
        r1 = fastq_dir / f"{ds.sra_id}_1.fastq.gz"
        r2 = fastq_dir / f"{ds.sra_id}_2.fastq.gz"

    if not r1.exists():
        print(f"  ERROR: FASTQ not found: {r1}")
        return False

    bam_path.parent.mkdir(parents=True, exist_ok=True)
    sam_path = bam_path.with_suffix(".sam")
    sorted_bam = bam_path

    # HISAT2 alignment
    hisat2_cmd = [
        "hisat2",
        "-p", str(threads),
        "--dta",
        "--no-mixed",
        "--no-discordant",
        "-x", str(HISAT2_INDEX),
        "-1", str(r1),
        "-2", str(r2),
    ]

    print(f"  Aligning {ds.name} with HISAT2...")
    with open(sam_path, "w") as sam_fh:
        proc = subprocess.Popen(hisat2_cmd, stdout=sam_fh, stderr=subprocess.PIPE)
        _, stderr = proc.communicate()
        hisat2_log = bam_path.parent / f"hisat2_{ds.name}.log"
        hisat2_log.write_text(stderr.decode())
        print(f"  HISAT2 log written to {hisat2_log}")
        if proc.returncode != 0:
            print(f"  HISAT2 failed with exit code {proc.returncode}")
            return False

    # Sort and index
    if not run_cmd(
        ["samtools", "sort", "-@", str(threads), "-o", str(sorted_bam), str(sam_path)],
        f"Sorting BAM for {ds.name}",
    ):
        return False

    run_cmd(
        ["samtools", "index", "-@", str(threads), str(sorted_bam)],
        f"Indexing BAM for {ds.name}",
    )

    # Cleanup SAM
    sam_path.unlink(missing_ok=True)
    return True


def run_rapidsplice(ds: DatasetConfig, threads: int = 8) -> str | None:
    """Run RapidSplice on a dataset."""
    output_gtf = RESULTS_DIR / f"rapidsplice_{ds.name}.gtf"
    if output_gtf.exists():
        print(f"  RapidSplice output exists: {output_gtf}")
        return str(output_gtf)

    cmd = [
        sys.executable, "-m", "rapidsplice", "assemble",
        "--bam", ds.bam_path,
        "--reference", str(GENOME_FA),
        "-o", str(output_gtf),
        "-t", str(threads),
        "--bootstrap",
        "--bootstrap-replicates", "100",
    ]

    t0 = time.time()
    if run_cmd(cmd, f"RapidSplice assembly on {ds.name}"):
        elapsed = time.time() - t0
        print(f"  RapidSplice {ds.name}: {elapsed:.1f}s")
        return str(output_gtf)
    return None


def run_stringtie(ds: DatasetConfig, threads: int = 8) -> str | None:
    """Run StringTie on a dataset."""
    output_gtf = RESULTS_DIR / f"stringtie_{ds.name}.gtf"
    if output_gtf.exists():
        print(f"  StringTie output exists: {output_gtf}")
        return str(output_gtf)

    cmd = [
        "stringtie",
        ds.bam_path,
        "-p", str(threads),
        "-o", str(output_gtf),
    ]

    t0 = time.time()
    if run_cmd(cmd, f"StringTie assembly on {ds.name}"):
        elapsed = time.time() - t0
        print(f"  StringTie {ds.name}: {elapsed:.1f}s")
        return str(output_gtf)
    return None


def run_gffcompare(gtf_path: str, label: str, chr_prefix: bool = False) -> dict | None:
    """Run GFFcompare and parse results."""
    gencode = str(GENCODE_CHR if chr_prefix else GENCODE_NOCHR)
    prefix = str(RESULTS_DIR / label)
    cmd = [
        "gffcompare",
        "-r", gencode,
        "-o", prefix,
        gtf_path,
    ]
    if not run_cmd(cmd, f"GFFcompare for {label}"):
        return None

    # Parse stats file
    stats_file = f"{prefix}.stats"
    if not os.path.exists(stats_file):
        print(f"  Stats file not found: {stats_file}")
        return None

    results: dict = {"label": label}
    with open(stats_file) as fh:
        for line in fh:
            line = line.strip()
            if "Intron level" in line and ":" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    vals = parts[1].strip().split()
                    if len(vals) >= 2:
                        results["intron_sn"] = float(vals[0])
                        results["intron_pr"] = float(vals[1])
            elif "Transcript level" in line and ":" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    vals = parts[1].strip().split()
                    if len(vals) >= 2:
                        results["tx_sn"] = float(vals[0])
                        results["tx_pr"] = float(vals[1])
            elif "matching transcripts" in line.lower():
                # Extract number before "matching"
                for word in line.split():
                    try:
                        results["exact_matches"] = int(word)
                        break
                    except ValueError:
                        continue

    # Count transcripts
    tx_count = 0
    try:
        with open(gtf_path) as fh:
            for line in fh:
                if not line.startswith("#"):
                    fields = line.split("\t")
                    if len(fields) >= 3 and fields[2] == "transcript":
                        tx_count += 1
    except Exception:
        pass
    results["n_transcripts"] = tx_count

    return results


def generate_report(all_results: dict[str, list[dict]]) -> None:
    """Generate a combined benchmark report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("matplotlib not available, skipping PDF report")
        return

    report_path = RESULTS_DIR / "multi_dataset_report.pdf"
    with PdfPages(str(report_path)) as pdf:
        # Page 1: Summary table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        ax.set_title("RapidSplice Multi-Dataset Benchmark", fontsize=16, fontweight="bold", pad=20)

        headers = ["Dataset", "Tool", "Transcripts", "IntronPr", "TxSn", "TxPr", "Exact Match"]
        table_data = []

        for dataset_name, results in all_results.items():
            for r in results:
                table_data.append([
                    dataset_name,
                    r.get("tool", "?"),
                    f"{r.get('n_transcripts', 0):,}",
                    f"{r.get('intron_pr', 0):.1f}%",
                    f"{r.get('tx_sn', 0):.1f}%",
                    f"{r.get('tx_pr', 0):.1f}%",
                    f"{r.get('exact_matches', 0):,}",
                ])

        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=headers,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            # Color header
            for j in range(len(headers)):
                table[0, j].set_facecolor("#2c3e50")
                table[0, j].set_text_props(color="white", fontweight="bold")

            # Color rows by tool
            for i, row in enumerate(table_data, start=1):
                color = "#e8f5e9" if "RapidSplice" in row[1] else "#e3f2fd"
                for j in range(len(headers)):
                    table[i, j].set_facecolor(color)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Intron Precision comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        datasets = list(all_results.keys())
        if datasets:
            # IntronPr bar chart
            ax = axes[0]
            x = range(len(datasets))
            width = 0.35

            rs_intron_pr = []
            st_intron_pr = []
            for ds in datasets:
                rs_val = next((r["intron_pr"] for r in all_results[ds] if "RapidSplice" in r.get("tool", "")), 0)
                st_val = next((r["intron_pr"] for r in all_results[ds] if "StringTie" in r.get("tool", "")), 0)
                rs_intron_pr.append(rs_val)
                st_intron_pr.append(st_val)

            ax.bar([i - width/2 for i in x], rs_intron_pr, width, label="RapidSplice", color="#2ecc71")
            ax.bar([i + width/2 for i in x], st_intron_pr, width, label="StringTie", color="#3498db")
            ax.set_ylabel("Intron Precision (%)")
            ax.set_title("Intron Precision")
            ax.set_xticks(list(x))
            ax.set_xticklabels(datasets)
            ax.legend()
            ax.set_ylim(70, 100)

            # Exact matches bar chart
            ax = axes[1]
            rs_exact = []
            st_exact = []
            for ds in datasets:
                rs_val = next((r["exact_matches"] for r in all_results[ds] if "RapidSplice" in r.get("tool", "")), 0)
                st_val = next((r["exact_matches"] for r in all_results[ds] if "StringTie" in r.get("tool", "")), 0)
                rs_exact.append(rs_val)
                st_exact.append(st_val)

            ax.bar([i - width/2 for i in x], rs_exact, width, label="RapidSplice", color="#2ecc71")
            ax.bar([i + width/2 for i in x], st_exact, width, label="StringTie", color="#3498db")
            ax.set_ylabel("Exact Transcript Matches")
            ax.set_title("Exact Matches")
            ax.set_xticks(list(x))
            ax.set_xticklabels(datasets)
            ax.legend()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nReport saved to: {report_path}")


def process_dataset(ds: DatasetConfig, threads: int = 8) -> list[dict] | None:
    """Process a single dataset through alignment, assembly, and evaluation."""
    print(f"\n{'#'*60}")
    print(f"  Processing: {ds.name} — {ds.description}")
    print(f"{'#'*60}")

    # Step 1: Check BAM exists, align if needed
    if not Path(ds.bam_path).exists():
        print(f"  BAM not found, aligning from FASTQ...")
        if not align_with_hisat2(ds, threads):
            print(f"  FAILED: Could not align {ds.name}")
            return None

    if not Path(ds.bam_path).exists():
        print(f"  BAM still not found after alignment attempt: {ds.bam_path}")
        return None

    # Step 2: Run assemblers
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rs_gtf = run_rapidsplice(ds, threads)
    st_gtf = run_stringtie(ds, threads)

    # Step 3: Evaluate with GFFcompare
    results = []
    if rs_gtf:
        r = run_gffcompare(rs_gtf, f"rs_{ds.name}", chr_prefix=ds.chr_prefix)
        if r:
            r["tool"] = "RapidSplice"
            results.append(r)

    if st_gtf:
        r = run_gffcompare(st_gtf, f"st_{ds.name}", chr_prefix=ds.chr_prefix)
        if r:
            r["tool"] = "StringTie"
            results.append(r)

    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-dataset RapidSplice benchmark")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        help="Dataset to benchmark",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available datasets",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of threads (default: 8)",
    )
    parser.add_argument(
        "--align-only",
        action="store_true",
        help="Only perform alignment, skip assembly",
    )
    args = parser.parse_args()

    if args.all:
        target_datasets = list(DATASETS.keys())
    elif args.dataset:
        target_datasets = [args.dataset]
    else:
        parser.print_help()
        return

    all_results: dict[str, list[dict]] = {}
    for ds_name in target_datasets:
        ds = DATASETS[ds_name]

        if args.align_only:
            if not Path(ds.bam_path).exists():
                align_with_hisat2(ds, args.threads)
            continue

        results = process_dataset(ds, args.threads)
        if results:
            all_results[ds_name] = results

    if not args.align_only and all_results:
        # Print summary
        print(f"\n{'='*60}")
        print("  BENCHMARK SUMMARY")
        print(f"{'='*60}")
        for ds_name, results in all_results.items():
            print(f"\n  {ds_name}:")
            for r in results:
                print(
                    f"    {r['tool']:15s}  IntronPr={r.get('intron_pr', 0):5.1f}%  "
                    f"TxSn={r.get('tx_sn', 0):4.1f}%  TxPr={r.get('tx_pr', 0):5.1f}%  "
                    f"Exact={r.get('exact_matches', 0):,}"
                )

        generate_report(all_results)


if __name__ == "__main__":
    main()
