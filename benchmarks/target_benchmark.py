#!/usr/bin/env python3
"""Benchmark TargetSplice on cancer-relevant genes.

Runs TargetSplice on 50+ genes in K562, classifies assembled isoforms
against GENCODE, and generates a summary report.
"""

from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, ".")

from rapidsplice.target.assembler import TargetConfig, assemble_target
from rapidsplice.target.extractor import lookup_gene

# 50 cancer-relevant genes with diverse splicing complexity
CANCER_GENES = [
    "TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "BRAF", "PIK3CA", "PTEN",
    "RB1", "MYC", "APC", "VHL", "CDH1", "ERBB2", "ALK", "ABL1", "BCR",
    "NRAS", "HRAS", "RAF1", "MTOR", "AKT1", "MDM2", "CDKN2A", "WT1",
    "NPM1", "FLT3", "KIT", "JAK2", "STAT3", "NOTCH1", "CTNNB1", "SMAD4",
    "ARID1A", "IDH1", "IDH2", "DNMT3A", "TET2", "EZH2", "SUZ12",
    "RUNX1", "GATA2", "CEBPA", "PML", "RARA", "BCL2", "MCL1", "BAX",
    "CASP3", "CASP8",
]

BAM_PATH = "real_benchmark/bam/SRR387661.bam"
GTF_PATH = "real_benchmark/annotation/gencode.v38.nochr.gtf"
REF_PATH = "real_benchmark/reference/grch38/genome.fa"


def main() -> None:
    """Run benchmark on all cancer genes."""
    results = []
    total_t0 = time.time()

    for gene in CANCER_GENES:
        region = lookup_gene(GTF_PATH, gene)
        if region is None:
            print(f"  SKIP {gene}: not found in annotation")
            continue

        config = TargetConfig(
            bam_path=BAM_PATH,
            reference_path=REF_PATH,
            region=region,
            flank=1000,
            max_paths=5000,
            bootstrap_replicates=200,
            min_presence_rate=0.3,
            annotation_gtf=GTF_PATH,
        )

        try:
            result = assemble_target(config)
        except Exception as e:
            print(f"  FAIL {gene}: {e}")
            continue

        n_exact = sum(
            1 for iso in result.isoforms if iso.classification == "exact_match"
        )
        n_novel_combo = sum(
            1 for iso in result.isoforms
            if iso.classification == "novel_combination"
        )
        n_novel_junc = sum(
            1 for iso in result.isoforms
            if iso.classification == "novel_junction"
        )
        n_novel_exon = sum(
            1 for iso in result.isoforms
            if iso.classification == "novel_exon"
        )

        gene_result = {
            "gene": gene,
            "chrom": region.chrom,
            "region_length": region.length,
            "n_junctions": result.extraction_stats.unique_junctions
            if result.extraction_stats else 0,
            "n_isoforms": result.n_isoforms,
            "n_confident": result.n_confident,
            "n_exact_match": n_exact,
            "n_novel_combo": n_novel_combo,
            "n_novel_junction": n_novel_junc,
            "n_novel_exon": n_novel_exon,
            "assembly_time": result.assembly_time_seconds,
            "bootstrap_time": result.bootstrap_time_seconds,
            "paths_enumerated": result.n_paths_enumerated,
        }
        results.append(gene_result)

        print(
            f"  {gene:<10} {result.n_isoforms:>2} isoforms "
            f"({n_exact} exact, {n_novel_combo} combo, "
            f"{n_novel_junc} novel_j, {n_novel_exon} novel_e) "
            f"{result.assembly_time_seconds:.2f}s"
        )

    total_time = time.time() - total_t0

    # Summary
    print(f"\n{'='*60}")
    print(f"  BENCHMARK SUMMARY ({len(results)} genes)")
    print(f"{'='*60}")

    total_isoforms = sum(r["n_isoforms"] for r in results)
    total_confident = sum(r["n_confident"] for r in results)
    total_exact = sum(r["n_exact_match"] for r in results)
    total_novel_combo = sum(r["n_novel_combo"] for r in results)
    total_novel_junc = sum(r["n_novel_junction"] for r in results)
    total_novel_exon = sum(r["n_novel_exon"] for r in results)
    total_assembly = sum(r["assembly_time"] for r in results)

    print(f"  Total isoforms:      {total_isoforms}")
    print(f"  High-confidence:     {total_confident}")
    print(f"  Exact matches:       {total_exact}")
    print(f"  Novel combinations:  {total_novel_combo}")
    print(f"  Novel junctions:     {total_novel_junc}")
    print(f"  Novel exons:         {total_novel_exon}")
    print(f"  Total assembly time: {total_assembly:.1f}s")
    print(f"  Total wall time:     {total_time:.1f}s")
    print(f"  Mean per gene:       {total_assembly/len(results):.2f}s")

    if total_isoforms > 0:
        print(f"\n  Precision (exact/total): "
              f"{total_exact/total_isoforms:.1%}")
        print(f"  Confident rate:          "
              f"{total_confident/total_isoforms:.1%}")

    # Save JSON
    output = {
        "genes": results,
        "summary": {
            "n_genes": len(results),
            "total_isoforms": total_isoforms,
            "total_confident": total_confident,
            "total_exact": total_exact,
            "total_novel_combo": total_novel_combo,
            "total_novel_junc": total_novel_junc,
            "total_novel_exon": total_novel_exon,
            "total_assembly_time": total_assembly,
            "total_wall_time": total_time,
        },
    }
    with open("benchmarks/results/target_benchmark.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to benchmarks/results/target_benchmark.json")


if __name__ == "__main__":
    main()
