#!/bin/bash
set -euo pipefail
cd /home/keunsoo/projects/23_rna-seq_assembler

echo "=== BRAID Multi-Replicate Benchmark ==="
echo "Waiting for all downloads and alignments..."

# Wait for all background jobs
while pgrep -f "fasterq.*SRR1173|hisat2.*SRR1173|wget.*ENCFF729" > /dev/null 2>&1; do
    echo "  $(date +%H:%M:%S) Still running..."
    sleep 60
done
echo "All jobs complete"

QKI=real_benchmark/rtpcr_benchmark/qki
REF=real_benchmark/reference/grch38/genome
GTF=real_benchmark/annotation/gencode.v38.nochr.gtf
RUN_GM12878=${RUN_GM12878:-0}

# Align QKI control rep2 if FASTQ available
if [ -f "$QKI/SRR1173997.fastq" ] && [ ! -f "$QKI/ctrl_rep2.bam" ]; then
    echo "Aligning QKI ctrl rep2..."
    hisat2 -x $REF -U $QKI/SRR1173997.fastq --dta -p 8 --no-unal 2>$QKI/hisat2_ctrl2.log | \
        samtools sort -@ 4 -o $QKI/ctrl_rep2.bam -
    samtools index $QKI/ctrl_rep2.bam
fi

# Align QKI KD rep2 if needed
if [ -f "$QKI/SRR1173999.fastq" ] && [ ! -f "$QKI/kd_rep2.bam" ]; then
    echo "Aligning QKI KD rep2..."
    hisat2 -x $REF -U $QKI/SRR1173999.fastq --dta -p 8 --no-unal 2>$QKI/hisat2_kd2.log | \
        samtools sort -@ 4 -o $QKI/kd_rep2.bam -
    samtools index $QKI/kd_rep2.bam
fi

# Index QKI ctrl rep1
samtools index $QKI/ctrl_rep1.bam 2>/dev/null || true

if [ "$RUN_GM12878" = "1" ]; then
    # Prepare GM12878 Rep1 without mutating the source BAM
    GM_REP1=real_benchmark/bam/GM12878_Rep1_ENCFF729OOW.bam
    GM_REP1_NOCHR=real_benchmark/bam/GM12878_Rep1.nochr.bam
    if [ -f "$GM_REP1_NOCHR" ]; then
        echo "GM12878 Rep1 ready"
    elif [ -f "$GM_REP1" ]; then
        TMP_HEADER=/tmp/gm_rep1_header.$$.sam
        samtools view -H "$GM_REP1" | sed 's/SN:chr/SN:/g' > "$TMP_HEADER"
        samtools reheader "$TMP_HEADER" "$GM_REP1" > "$GM_REP1_NOCHR"
        rm -f "$TMP_HEADER"
        samtools index -@ 8 "$GM_REP1_NOCHR"
        echo "GM12878 Rep1 ready"
    fi
fi

echo ""
echo "=== Running Multi-Replicate Benchmarks ==="

python3 << 'PYEOF'
import os
import sys, json, time, numpy as np
sys.path.insert(0, ".")

results = {
    "metadata": {
        "qki_dir": "real_benchmark/rtpcr_benchmark/qki",
        "gtf": "real_benchmark/annotation/gencode.v38.nochr.gtf",
        "seed": 42,
    },
    "qki_multi_replicate": {},
    "gm12878_multi_replicate": {},
}

# ============================================
# Benchmark 2a: QKI Biological Replicates
# ============================================
print("=" * 60)
print("  Benchmark 2a: QKI Multi-Replicate")
print("=" * 60)

import os
QKI = "real_benchmark/rtpcr_benchmark/qki"
ctrl1 = f"{QKI}/ctrl_rep1.bam"
ctrl2 = f"{QKI}/ctrl_rep2.bam"

if os.path.exists(ctrl1) and os.path.exists(ctrl2):
    from rapidsplice.target.multi_replicate_bootstrap import multi_replicate_psi
    from rapidsplice.target.psi_bootstrap import compute_psi_from_junctions
    from rapidsplice.target.extractor import lookup_gene

    GTF = "real_benchmark/annotation/gencode.v38.nochr.gtf"

    GENES = ["EZH2", "AKT1", "STAT3", "RUNX1", "BCL2L1"]

    for gene in GENES:
        region = lookup_gene(GTF, gene)
        if not region: continue

        # Single-sample
        single = compute_psi_from_junctions(
            ctrl1,
            region.chrom,
            region.start,
            region.end,
            gene=gene,
            n_replicates=200,
            confidence_level=0.95,
            seed=42,
        )

        # Multi-replicate
        multi = multi_replicate_psi(
            [ctrl1, ctrl2], region.chrom, region.start, region.end,
            n_bootstrap=200, seed=42,
        )

        if single and multi:
            s_widths = [r.ci_high - r.ci_low for r in single if r.is_confident]
            m_widths = [r.ci_high - r.ci_low for r in multi if r.is_confident]
            qki_result = {
                "single_event_count": len(single),
                "multi_event_count": len(multi),
                "single_confident_count": len(s_widths),
                "multi_confident_count": len(m_widths),
                "single_median_ci_width": float(np.median(s_widths)) if s_widths else None,
                "multi_median_ci_width": float(np.median(m_widths)) if m_widths else None,
            }
            results["qki_multi_replicate"][gene] = qki_result
            single_msg = (
                f"Single-sample: {len(single)} events, "
                f"CI width median={np.median(s_widths):.3f}"
                if s_widths else
                f"Single-sample: {len(single)} events, CI width median=N/A"
            )
            multi_msg = (
                f"Multi-rep:     {len(multi)} events, "
                f"CI width median={np.median(m_widths):.3f}"
                if m_widths else
                f"Multi-rep:     {len(multi)} events, CI width median=N/A"
            )

            print(f"\n  {gene}:")
            print(f"    {single_msg}")
            print(f"    {multi_msg}")
        else:
            results["qki_multi_replicate"][gene] = {
                "single_event_count": len(single) if single else 0,
                "multi_event_count": len(multi) if multi else 0,
                "single_confident_count": 0,
                "multi_confident_count": 0,
                "single_median_ci_width": None,
                "multi_median_ci_width": None,
            }
else:
    print("  QKI replicate BAMs not ready yet")
    results["qki_multi_replicate"]["status"] = "replicate_bams_not_ready"

# ============================================
# Benchmark 2b: GM12878 Biological Replicates
# ============================================
print("\n" + "=" * 60)
print("  Benchmark 2b: GM12878 Multi-Replicate Isoform CI")
print("=" * 60)

gm_rep1 = "real_benchmark/bam/GM12878_Rep1.nochr.bam"
gm_rep2 = "real_benchmark/bam/GM12878_ENCFF550SET.nochr.bam"
run_gm12878 = os.environ.get("RUN_GM12878") == "1"

if not run_gm12878:
    print("  Skipped: out of BRAID-only scope")
    results["gm12878_multi_replicate"] = {"status": "skipped_braid_only_scope"}
elif os.path.exists(gm_rep1) and os.path.exists(gm_rep2):
    from rapidsplice.target.multi_replicate_bootstrap import multi_replicate_isoform_bootstrap
    from rapidsplice.target.stringtie_bootstrap import STBootstrapConfig, run_stringtie_bootstrap

    ST_GTF = "real_benchmark/results/stringtie_GM12878_rf.gtf"
    REF = "real_benchmark/reference/grch38/genome.fa"

    # Single-sample (Rep2 only - already have results)
    print("  Single-sample (Rep2): loading previous results...")

    # Multi-replicate (Rep1 + Rep2)
    print("  Multi-replicate (Rep1+Rep2): computing...")
    t0 = time.time()
    multi_results = multi_replicate_isoform_bootstrap(
        ST_GTF, [gm_rep1, gm_rep2], REF, n_bootstrap=100, seed=42,
    )
    elapsed = time.time() - t0

    print(f"  {len(multi_results)} isoforms, {elapsed:.1f}s")
    if multi_results:
        cvs = [r["total_cv"] for r in multi_results if not np.isnan(r.get("total_cv", float("nan")))]
        bio_stds = [r["bio_std"] for r in multi_results]
        sampling_stds = [
            r["sampling_std"]
            for r in multi_results
            if not np.isnan(r.get("sampling_std", float("nan")))
        ]
        gm_result = {
            "n_isoforms": len(multi_results),
            "elapsed_seconds": float(elapsed),
            "median_total_cv": float(np.median(cvs)) if cvs else None,
            "median_bio_std": float(np.median(bio_stds)) if bio_stds else None,
            "median_sampling_std": float(np.median(sampling_stds)) if sampling_stds else None,
        }
        results["gm12878_multi_replicate"] = gm_result
        print(f"  Total CV median: {np.median(cvs):.3f}")
        print(f"  Bio std median:  {np.median(bio_stds):.3f}")
        if sampling_stds:
            print(f"  Sampling std median: {np.median(sampling_stds):.3f}")
        else:
            print("  Sampling std median: N/A")
    else:
        results["gm12878_multi_replicate"] = {
            "n_isoforms": 0,
            "elapsed_seconds": float(elapsed),
            "median_total_cv": None,
            "median_bio_std": None,
            "median_sampling_std": None,
        }
else:
    print("  GM12878 replicate BAMs not ready yet")
    results["gm12878_multi_replicate"] = {"status": "replicate_bams_not_ready"}

print("\n" + "=" * 60)
print("  Benchmark Complete")
print("=" * 60)

output = "benchmarks/results/multirep_benchmark.json"
os.makedirs(os.path.dirname(output), exist_ok=True)
with open(output, "w", encoding="utf-8") as handle:
    json.dump(results, handle, indent=2)
print(f"Saved to {output}")
PYEOF
