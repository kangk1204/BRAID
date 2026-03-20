#!/bin/bash
set -euo pipefail

cd /home/keunsoo/projects/23_rna-seq_assembler

QKI_DIR=real_benchmark/rtpcr_benchmark/qki
HISAT2_IDX=real_benchmark/reference/grch38/genome
GTF=real_benchmark/annotation/gencode.v38.nochr.gtf
FASTQ=$QKI_DIR/SRR1173996.fastq
CONTROL_BAM=$QKI_DIR/ctrl_rep1.bam
KD_FASTQ=$QKI_DIR/SRR1173998.fastq
KD_BAM=$QKI_DIR/qki_kd.bam
OUTPUT_JSON=$QKI_DIR/braid_benchmark_results.json

echo "=== QKI BRAID Benchmark ==="

python3 benchmarks/liftover_qki_targets.py \
    --qki-dir "$QKI_DIR"

if [ ! -f "$CONTROL_BAM" ]; then
    if [ ! -f "$FASTQ" ]; then
        echo "Missing both $CONTROL_BAM and $FASTQ"
        exit 1
    fi
    echo "Aligning control replicate..."
    hisat2 -x "$HISAT2_IDX" -U "$FASTQ" --dta -p 8 --no-unal 2>"$QKI_DIR/hisat2_ctrl1.log" | \
        samtools sort -@ 4 -o "$CONTROL_BAM" -
    samtools index "$CONTROL_BAM"
fi

if [ ! -f "$KD_BAM" ]; then
    if [ ! -f "$KD_FASTQ" ]; then
        echo "Missing both $KD_BAM and $KD_FASTQ"
        exit 1
    fi
    echo "Aligning KD replicate..."
    hisat2 -x "$HISAT2_IDX" -U "$KD_FASTQ" --dta -p 8 --no-unal 2>"$QKI_DIR/hisat2_kd.log" | \
        samtools sort -@ 4 -o "$KD_BAM" -
    samtools index "$KD_BAM"
fi

python3 benchmarks/qki_benchmark.py \
    --sample-bam "ctrl_rep1=$CONTROL_BAM" \
    --sample-bam "qki_kd=$KD_BAM" \
    --qki-dir "$QKI_DIR" \
    --gtf "$GTF" \
    --output "$OUTPUT_JSON"

echo ""
echo "Saved benchmark JSON to $OUTPUT_JSON"
