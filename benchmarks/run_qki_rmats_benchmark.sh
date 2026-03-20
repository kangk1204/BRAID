#!/bin/bash
set -euo pipefail
cd /home/keunsoo/projects/23_rna-seq_assembler

QKI_DIR=real_benchmark/rtpcr_benchmark/qki
GTF=real_benchmark/annotation/gencode.v38.nochr.gtf
RMATS_DIR=benchmarks/results/qki_rmats
RMATS_TMP=$RMATS_DIR/tmp
OUTPUT_JSON=$QKI_DIR/qki_rmats_benchmark_results.json
THREADS=${RMATS_THREADS:-8}

mkdir -p "$RMATS_DIR" "$RMATS_TMP"

python3 benchmarks/liftover_qki_targets.py \
    --qki-dir "$QKI_DIR"

CTRL1=$QKI_DIR/ctrl_rep1.bam
CTRL2=$QKI_DIR/ctrl_rep2.bam
KD1=$QKI_DIR/qki_kd.bam
B1=$RMATS_DIR/b1.txt
B2=$RMATS_DIR/b2.txt

printf "%s,%s\n" "$CTRL1" "$CTRL2" > "$B1"
printf "%s\n" "$KD1" > "$B2"

if [ "${FORCE_RMATS:-0}" = "1" ] || { [ ! -f "$RMATS_DIR/SE.MATS.JC.txt" ] && [ ! -f "$RMATS_DIR/SE.MATS.JunctionCountOnly.txt" ]; }; then
    rmats.py \
        --b1 "$B1" \
        --b2 "$B2" \
        --gtf "$GTF" \
        --od "$RMATS_DIR" \
        --tmp "$RMATS_TMP" \
        -t single \
        --readLength 50 \
        --variable-read-length \
        --libType fr-unstranded \
        --nthread "$THREADS" \
        --task both
fi

python3 benchmarks/qki_rmats_benchmark.py \
    --rmats-dir "$RMATS_DIR" \
    --qki-dir "$QKI_DIR" \
    --output "$OUTPUT_JSON" \
    --tolerance 50 \
    --min-overlap-fraction 0.5 \
    --min-total-count 1 \
    --n-replicates 200 \
    --confidence-level 0.95 \
    --fdr-threshold 0.05 \
    --seed 42

echo ""
echo "Saved benchmark JSON to $OUTPUT_JSON"
