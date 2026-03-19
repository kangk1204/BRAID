#!/usr/bin/env python3
"""Comprehensive bootstrap CI benchmark across multiple datasets and CV thresholds.

Runs StringTie + Bootstrap CI on:
1. Polyester simulation (perfect ground truth)
2. K562 real data (GENCODE v38)
3. GM12878 real data (GENCODE v38)
4. ENCODE10 datasets (if available)

For each dataset, evaluates precision-recall at CV thresholds from 0.01 to 1.0.
Generates JSON results and summary tables.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, ".")


def run_polyester_benchmark() -> dict:
    """Run benchmark on Polyester-simulated data with perfect ground truth."""
    print("\n" + "=" * 60)
    print("  BENCHMARK 1: Polyester Simulation")
    print("=" * 60)

    sim_dir = "benchmarks/simulation"
    sim_bam = os.path.join(sim_dir, "simulated.sorted.bam")
    sim_gtf_truth = os.path.join(sim_dir, "reference.gtf")

    if not os.path.exists(sim_bam):
        print("  Generating Polyester simulation...")
        _generate_polyester_simulation(sim_dir)

    if not os.path.exists(sim_bam):
        print("  SKIP: Simulation data not available")
        return {}

    # Run StringTie
    st_gtf = os.path.join(sim_dir, "stringtie.gtf")
    if not os.path.exists(st_gtf):
        subprocess.run(
            ["stringtie", sim_bam, "-o", st_gtf, "-p", "4"],
            capture_output=True,
        )

    # Run bootstrap + evaluation
    return _run_bootstrap_evaluation(
        st_gtf, sim_bam, None, sim_gtf_truth, "Polyester",
    )


def run_real_data_benchmark(
    name: str,
    st_gtf: str,
    bam: str,
    ref: str | None,
    gencode: str,
) -> dict:
    """Run benchmark on real data against GENCODE."""
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK: {name}")
    print(f"{'=' * 60}")

    if not os.path.exists(st_gtf):
        print(f"  SKIP: {st_gtf} not found")
        return {}

    return _run_bootstrap_evaluation(st_gtf, bam, ref, gencode, name)


def _run_bootstrap_evaluation(
    st_gtf: str,
    bam: str,
    ref: str | None,
    annotation_gtf: str,
    dataset_name: str,
) -> dict:
    """Core evaluation: bootstrap + precision-recall at multiple CV thresholds."""
    from braid.target.stringtie_bootstrap import (
        _parse_stringtie_gtf,
    )
    from braid.target.gencode_index import load_annotation_index, query_region
    from braid.target.comparator import classify_isoform

    # Parse StringTie
    genes = _parse_stringtie_gtf(st_gtf)
    print(f"  StringTie: {len(genes)} genes")

    # Load annotation index
    idx = load_annotation_index(annotation_gtf)

    # BAM-free bootstrap
    rng = np.random.default_rng(42)
    R = 200

    t0 = time.time()
    all_data: list[dict] = []

    for gene_id, gene_data in genes.items():
        txs = [t for t in gene_data["transcripts"] if len(t["exons"]) >= 2]
        if not txs:
            continue

        # Build junction matrix
        all_juncs: set[tuple[int, int]] = set()
        for tx in txs:
            for i in range(len(tx["exons"]) - 1):
                all_juncs.add((tx["exons"][i][1], tx["exons"][i + 1][0]))
        if not all_juncs:
            continue

        junc_list = sorted(all_juncs)
        n_j = len(junc_list)
        n_p = len(txs)

        A = np.zeros((n_j, n_p), dtype=np.float64)
        covs = np.array(
            [tx.get("cov", 1.0) for tx in txs], dtype=np.float64,
        )
        for pi, tx in enumerate(txs):
            for i in range(len(tx["exons"]) - 1):
                junc = (tx["exons"][i][1], tx["exons"][i + 1][0])
                ji = junc_list.index(junc)
                A[ji, pi] = 1.0

        b = A @ covs
        b = np.maximum(b, 0.1)

        try:
            weights, _ = nnls(A, b)
        except Exception:
            continue

        # Bootstrap
        wm = np.zeros((R, n_p), dtype=np.float64)
        for r in range(R):
            br = rng.poisson(b).astype(np.float64)
            br = np.maximum(br, 0.1)
            try:
                wm[r], _ = nnls(A, br)
            except Exception:
                pass

        # GENCODE comparison
        gs = min(e[0] for tx in txs for e in tx["exons"])
        ge = max(e[1] for tx in txs for e in tx["exons"])
        ref_txs = query_region(idx, gene_data["chrom"], gs, ge)

        for pi, tx in enumerate(txs):
            col = wm[:, pi]
            mean_w = float(np.mean(col))
            cv = float(np.std(col) / mean_w) if mean_w > 0 else float("nan")
            presence = float(np.sum(col > 0) / R)

            cls = classify_isoform(tx["exons"], ref_txs)
            all_data.append({
                "cv": cv,
                "presence": presence,
                "weight": float(weights[pi]),
                "exact": cls.category == "exact_match",
                "category": cls.category,
            })

    elapsed = time.time() - t0
    n_total = len(all_data)
    n_exact = sum(1 for d in all_data if d["exact"])

    print(f"  {n_total} isoforms, {n_exact} exact ({n_exact/max(n_total,1):.1%})")
    print(f"  Time: {elapsed:.1f}s")

    # Evaluate at multiple CV thresholds
    cv_thresholds = [
        0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15,
        0.2, 0.3, 0.5, 0.7, 1.0, float("inf"),
    ]

    pr_curve: list[dict] = []
    print(f"\n  {'CV≤':<8} {'N':>7} {'Exact':>7} {'Prec':>8} {'Recall':>8} {'F1':>7}")
    print(f"  {'-' * 48}")

    for cv_max in cv_thresholds:
        f = [
            d for d in all_data
            if (d["cv"] <= cv_max if not np.isnan(d["cv"]) else cv_max == float("inf"))
        ]
        ne = sum(1 for d in f if d["exact"])
        n = len(f)
        prec = ne / n if n > 0 else 0
        recall = ne / max(n_exact, 1)
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0

        label = f"{cv_max:.2f}" if cv_max < float("inf") else "All"
        print(f"  {label:<8} {n:>7} {ne:>7} {prec:>7.1%} {recall:>7.1%} {f1:>6.3f}")

        pr_curve.append({
            "cv_threshold": cv_max if cv_max < float("inf") else 999,
            "n_isoforms": n,
            "n_exact": ne,
            "precision": prec,
            "recall": recall,
            "f1": f1,
        })

    return {
        "dataset": dataset_name,
        "n_genes": len(genes),
        "n_isoforms": n_total,
        "n_exact": n_exact,
        "baseline_precision": n_exact / max(n_total, 1),
        "time_seconds": elapsed,
        "pr_curve": pr_curve,
    }


def _generate_polyester_simulation(sim_dir: str) -> None:
    """Generate Polyester simulation if R/Bioconductor available."""
    os.makedirs(sim_dir, exist_ok=True)
    # Check if simulation already exists from earlier benchmark
    existing = "benchmarks/simulated_reads"
    if os.path.exists(existing):
        print(f"  Found existing simulation at {existing}")
        return
    print("  Polyester simulation requires R. Skipping.")


def main() -> None:
    """Run all benchmarks."""
    results = []

    GENCODE = "real_benchmark/annotation/gencode.v38.nochr.gtf"
    REF = "real_benchmark/reference/grch38/genome.fa"

    # Benchmark 1: K562
    r = run_real_data_benchmark(
        "K562",
        "real_benchmark/results/stringtie.gtf",
        "real_benchmark/bam/SRR387661.bam",
        REF, GENCODE,
    )
    if r:
        results.append(r)

    # Benchmark 2: GM12878
    r = run_real_data_benchmark(
        "GM12878",
        "real_benchmark/results/stringtie_GM12878_rf.gtf",
        "real_benchmark/bam/GM12878_ENCFF550SET.nochr.bam",
        REF, GENCODE,
    )
    if r:
        results.append(r)

    # Benchmark 3: IMR90
    r = run_real_data_benchmark(
        "IMR90",
        "real_benchmark/results/stringtie_IMR90_rf.gtf",
        "real_benchmark/bam/IMR90_ENCFF560TMJ.nochr.bam",
        REF, GENCODE,
    )
    if r:
        results.append(r)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  COMPREHENSIVE BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Dataset':<12} {'Isoforms':>9} {'Exact':>7} {'Base Prec':>10} "
          f"{'CV≤0.05':>9} {'CV≤0.1':>9} {'Best F1':>8}")
    print(f"  {'-' * 67}")

    for r in results:
        curve = r["pr_curve"]
        cv005 = next((p for p in curve if abs(p["cv_threshold"] - 0.05) < 0.001), None)
        cv01 = next((p for p in curve if abs(p["cv_threshold"] - 0.1) < 0.001), None)
        best_f1 = max(p["f1"] for p in curve)
        best_f1_cv = next(p["cv_threshold"] for p in curve if p["f1"] == best_f1)

        print(
            f"  {r['dataset']:<12} {r['n_isoforms']:>9,} {r['n_exact']:>7,} "
            f"{r['baseline_precision']:>9.1%} "
            f"{cv005['precision'] if cv005 else 0:>8.1%} "
            f"{cv01['precision'] if cv01 else 0:>8.1%} "
            f"{best_f1:>7.3f} (CV≤{best_f1_cv:.2f})"
        )

    # Save all results
    output_path = "benchmarks/results/comprehensive_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
