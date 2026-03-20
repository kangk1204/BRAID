#!/usr/bin/env python3
"""Compare TargetSplice vs StringTie on the same gene regions."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time

import pysam

sys.path.insert(0, ".")

from rapidsplice.target.assembler import TargetConfig, assemble_target
from rapidsplice.target.comparator import classify_all_isoforms
from rapidsplice.target.extractor import lookup_gene

CANCER_GENES = [
    "TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "BRAF", "PIK3CA", "PTEN",
    "RB1", "MYC", "APC", "VHL", "CDH1", "ERBB2", "ALK", "ABL1", "BCR",
    "NRAS", "HRAS", "RAF1", "MTOR", "AKT1", "MDM2", "CDKN2A", "WT1",
    "NPM1", "FLT3", "KIT", "JAK2", "STAT3", "NOTCH1", "CTNNB1", "SMAD4",
    "ARID1A", "IDH1", "IDH2", "DNMT3A", "TET2", "EZH2", "SUZ12",
    "RUNX1", "GATA2", "CEBPA", "PML", "RARA", "BCL2", "MCL1", "BAX",
    "CASP3", "CASP8",
]

BAM = "real_benchmark/bam/SRR387661.bam"
GTF = "real_benchmark/annotation/gencode.v38.nochr.gtf"
REF = "real_benchmark/reference/grch38/genome.fa"


def run_stringtie_region(
    bam: str, chrom: str, start: int, end: int,
) -> list[list[tuple[int, int]]]:
    """Run StringTie on a specific region and return exon lists."""
    with tempfile.NamedTemporaryFile(suffix=".bam", delete=False) as tmp:
        tmp_bam = tmp.name

    # Extract region
    subprocess.run(
        ["samtools", "view", "-b", bam, f"{chrom}:{start}-{end}",
         "-o", tmp_bam],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["samtools", "index", tmp_bam],
        check=True, capture_output=True,
    )

    with tempfile.NamedTemporaryFile(suffix=".gtf", delete=False) as tmp:
        tmp_gtf = tmp.name

    subprocess.run(
        ["stringtie", tmp_bam, "-o", tmp_gtf, "-p", "4"],
        capture_output=True,
    )

    # Parse StringTie output
    tx_exons: dict[str, list[tuple[int, int]]] = {}
    with open(tmp_gtf) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "exon":
                continue
            attrs = fields[8]
            tid = None
            for part in attrs.split(";"):
                part = part.strip()
                if part.startswith("transcript_id"):
                    tid = part.split('"')[1]
            if tid:
                s = int(fields[3]) - 1
                e = int(fields[4])
                tx_exons.setdefault(tid, []).append((s, e))

    import os
    os.unlink(tmp_bam)
    os.unlink(tmp_bam + ".bai")
    os.unlink(tmp_gtf)

    result = []
    for tid, exons in tx_exons.items():
        exons.sort()
        if len(exons) >= 2:  # multi-exon only
            result.append(exons)
    return result


def main() -> None:
    """Run head-to-head comparison."""
    ts_results = []
    st_results = []

    for gene in CANCER_GENES:
        region = lookup_gene(GTF, gene)
        if region is None:
            continue

        # TargetSplice
        config = TargetConfig(
            bam_path=BAM, reference_path=REF, region=region,
            flank=1000, max_paths=5000, bootstrap_replicates=200,
            annotation_gtf=GTF,
        )
        try:
            ts_result = assemble_target(config)
        except Exception:
            continue

        ts_exact = sum(
            1 for i in ts_result.isoforms if i.classification == "exact_match"
        )
        ts_total = len([i for i in ts_result.isoforms if len(i.exons) >= 2])

        # StringTie on same region
        st_exons_list = run_stringtie_region(
            BAM, region.chrom, region.start, region.end,
        )
        if st_exons_list:
            st_cls = classify_all_isoforms(
                st_exons_list, GTF,
                region.chrom, region.start, region.end,
            )
            st_exact = sum(1 for c in st_cls if c.category == "exact_match")
            st_total = len(st_exons_list)
        else:
            st_exact = 0
            st_total = 0

        ts_results.append({
            "gene": gene, "total": ts_total, "exact": ts_exact,
        })
        st_results.append({
            "gene": gene, "total": st_total, "exact": st_exact,
        })

        print(
            f"  {gene:<10} TS: {ts_exact}/{ts_total:<3}  "
            f"ST: {st_exact}/{st_total:<3}"
        )

    # Summary
    ts_total_all = sum(r["total"] for r in ts_results)
    ts_exact_all = sum(r["exact"] for r in ts_results)
    st_total_all = sum(r["total"] for r in st_results)
    st_exact_all = sum(r["exact"] for r in st_results)

    print(f"\n{'='*50}")
    print(f"  COMPARISON SUMMARY ({len(ts_results)} genes)")
    print(f"{'='*50}")
    print(f"  TargetSplice: {ts_exact_all}/{ts_total_all} exact "
          f"({ts_exact_all/max(ts_total_all,1):.1%})")
    print(f"  StringTie:    {st_exact_all}/{st_total_all} exact "
          f"({st_exact_all/max(st_total_all,1):.1%})")

    ts_wins = sum(
        1 for t, s in zip(ts_results, st_results) if t["exact"] > s["exact"]
    )
    st_wins = sum(
        1 for t, s in zip(ts_results, st_results) if s["exact"] > t["exact"]
    )
    ties = len(ts_results) - ts_wins - st_wins
    print(f"  TS wins: {ts_wins}, ST wins: {st_wins}, Ties: {ties}")

    # Save
    with open("benchmarks/results/target_vs_stringtie.json", "w") as f:
        json.dump({
            "targetsplice": ts_results,
            "stringtie": st_results,
        }, f, indent=2)


if __name__ == "__main__":
    main()
