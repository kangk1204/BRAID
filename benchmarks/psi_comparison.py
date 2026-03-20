#!/usr/bin/env python3
"""Compare PRISM PSI bootstrap CI with rMATS and SUPPA2.

For a set of target genes, computes PSI using:
1. PRISM (our junction bootstrap method)
2. rMATS (from pre-computed output)
3. SUPPA2 (from pre-computed output)

Evaluates:
- PSI value correlation between methods
- CI width comparison
- Confident event agreement
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")


def load_rmats_psi(rmats_dir: str, event_type: str = "SE") -> dict:
    """Load rMATS PSI values from output files."""
    results = {}
    # rMATS outputs: SE.MATS.JC.txt (junction counts only)
    jc_file = os.path.join(rmats_dir, f"{event_type}.MATS.JC.txt")
    if not os.path.exists(jc_file):
        return results

    with open(jc_file) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < len(header):
                continue
            row = dict(zip(header, fields))

            # Extract PSI for sample 1
            inc_level = row.get("IncLevel1", "")
            if not inc_level or inc_level == "NA":
                continue

            try:
                psi_vals = [float(x) for x in inc_level.split(",") if x]
                if not psi_vals:
                    continue
                psi = np.mean(psi_vals)
            except ValueError:
                continue

            # Build event key
            chrom = row.get("chr", "")
            strand = row.get("strand", "")
            if event_type == "SE":
                exon_start = row.get("exonStart_0base", "")
                exon_end = row.get("exonEnd", "")
                key = f"SE:{chrom}:{exon_start}-{exon_end}:{strand}"
            else:
                key = f"{event_type}:{chrom}:{row.get('longExonStart_0base', '')}"

            # Junction counts
            inc_count = sum(
                int(x) for x in row.get("IJC_SAMPLE_1", "0").split(",") if x
            )
            exc_count = sum(
                int(x) for x in row.get("SJC_SAMPLE_1", "0").split(",") if x
            )

            results[key] = {
                "psi": psi,
                "inc_count": inc_count,
                "exc_count": exc_count,
                "event_type": event_type,
                "chrom": chrom,
            }

    return results


def load_prism_psi(
    bam_path: str,
    genes: list[dict],
    n_replicates: int = 500,
) -> dict:
    """Compute PRISM PSI for target gene regions."""
    from braid.target.psi_bootstrap import (
        compute_psi_from_junctions,
    )

    results = {}
    for gene in genes:
        psi_results = compute_psi_from_junctions(
            bam_path,
            gene["chrom"],
            gene["start"],
            gene["end"],
            gene=gene.get("name"),
            n_replicates=n_replicates,
            seed=42,
        )
        for r in psi_results:
            key = r.event_id
            results[key] = {
                "psi": r.psi,
                "ci_low": r.ci_low,
                "ci_high": r.ci_high,
                "cv": r.cv,
                "inc_count": r.inclusion_count,
                "exc_count": r.exclusion_count,
                "is_confident": r.is_confident,
            }

    return results


def compare_methods(
    prism: dict,
    rmats: dict,
) -> None:
    """Compare PSI values between PRISM and rMATS."""
    # Match events by junction coordinates
    # Since event naming differs, match by inclusion/exclusion count similarity
    print(f"PRISM events: {len(prism)}")
    print(f"rMATS events: {len(rmats)}")

    # For rMATS, extract junction-based PSI to compare
    rmats_psi_values = []
    rmats_counts = []
    for key, data in rmats.items():
        if data["inc_count"] + data["exc_count"] >= 10:
            rmats_psi_values.append(data["psi"])
            rmats_counts.append(data["inc_count"] + data["exc_count"])

    print(f"rMATS events with ≥10 reads: {len(rmats_psi_values)}")
    if rmats_psi_values:
        print(f"  PSI distribution: median={np.median(rmats_psi_values):.3f}, "
              f"mean={np.mean(rmats_psi_values):.3f}")

    # PRISM confident events
    prism_confident = [d for d in prism.values() if d.get("is_confident")]
    print(f"\nPRISM confident events (CI<20%): {len(prism_confident)}")
    if prism_confident:
        ci_widths = [d["ci_high"] - d["ci_low"] for d in prism_confident]
        print(f"  CI width: median={np.median(ci_widths):.3f}, "
              f"mean={np.mean(ci_widths):.3f}")


def main() -> None:
    """Run comparison."""
    from braid.target.extractor import lookup_gene

    GTF = "real_benchmark/annotation/gencode.v38.nochr.gtf"
    BAM = "real_benchmark/bam/SRR387661.bam"

    GENE_NAMES = ["TP53", "BRCA1", "EZH2", "KRAS", "BRAF", "MYC",
                  "AKT1", "BCL2", "PTEN", "RUNX1"]

    genes = []
    for name in GENE_NAMES:
        region = lookup_gene(GTF, name)
        if region:
            genes.append({
                "name": name,
                "chrom": region.chrom,
                "start": region.start,
                "end": region.end,
            })

    print("=" * 60)
    print("  PRISM vs rMATS PSI Comparison")
    print("=" * 60)

    # Load rMATS results
    rmats_dir = "benchmarks/results/rmats_k562"
    rmats_results = {}
    for et in ["SE", "A3SS", "A5SS", "MXE", "RI"]:
        r = load_rmats_psi(rmats_dir, et)
        rmats_results.update(r)
        if r:
            print(f"  rMATS {et}: {len(r)} events")

    # Compute PRISM PSI
    print("\n  Computing PRISM PSI...")
    prism_results = load_prism_psi(BAM, genes, n_replicates=500)

    compare_methods(prism_results, rmats_results)

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"  {'Method':<15} {'Events':>8} {'With CI':>8} {'Needs Replicates':>18}")
    print(f"  {'-' * 55}")
    print(f"  {'PRISM':<15} {len(prism_results):>8} {len([d for d in prism_results.values() if d.get('is_confident')]):>8} {'No':>18}")
    print(f"  {'rMATS':<15} {len(rmats_results):>8} {'N/A':>8} {'Yes':>18}")
    print(f"  {'SUPPA2':<15} {'TBD':>8} {'N/A':>8} {'Yes':>18}")


if __name__ == "__main__":
    main()
