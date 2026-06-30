#!/usr/bin/env bash
# Real BAM-level downsample -> rMATS rerun for the depth-titration VALIDATION.
#
# Validates the binomial-thinning surrogate against ground-truth re-runs on the
# cheapest dataset (GSE59335). For each (fraction, seed) it downsamples the b1/b2
# BAMs with samtools, re-runs rMATS in its own conda env, and writes the resulting
# SE.MATS.JC.txt to a scratch dir for the coverage comparison step.
#
# Usage:  run_rmats_depth.sh FRAC SEED
#   FRAC : 1.0 (no downsample, param check) or e.g. 0.25, 0.50
#   SEED : integer subsample seed
#
# Original rMATS params recovered from the in-repo run:
#   -t paired, --readLength 101, hg19 GENCODE v19 GTF, novelSS OFF (default),
#   b1=SRR1513329/330/331  b2=SRR1513332/333/334.
set -euo pipefail

REPO=/home/keunsoo/Projects/26_BRAID
cd "$REPO"

FRAC="${1:?usage: run_rmats_depth.sh FRAC SEED}"
SEED="${2:?usage: run_rmats_depth.sh FRAC SEED}"

RMATS_PYBIN=/home/keunsoo/mambaforge/envs/rmats/bin/python
RMATS_PY=/home/keunsoo/mambaforge/envs/rmats/bin/rmats.py
GTF="data/public_benchmarks/reference/hg19/gencode.v19.annotation.gtf"
RL=101
NT=16
BAMDIR="data/public_benchmarks/GSE59335/bam"
B1=(SRR1513329 SRR1513330 SRR1513331)
B2=(SRR1513332 SRR1513333 SRR1513334)

TAG="GSE59335_f${FRAC}_s${SEED}"
SCRATCH="/tmp/rmats_depth/${TAG}"
OUT="benchmarks/results/depth_rerun/${TAG}"
mkdir -p "$SCRATCH/bam" "$SCRATCH/od" "$SCRATCH/tmp" "$OUT"

# samtools -s takes SEED.FRACTION as a single float (e.g. seed 2, 25% -> 2.25).
SARG=$(awk "BEGIN{printf \"%g\", ${SEED} + ${FRAC}}")

prep_bam() {  # echoes the path to use for a given SRR
  local srr="$1" src="$BAMDIR/$1.sorted.bam"
  if [ "$FRAC" = "1.0" ]; then echo "$src"; return; fi
  local dst="$SCRATCH/bam/$srr.sub.bam"
  samtools view -@ "$NT" -b -s "$SARG" "$src" -o "$dst"
  samtools index -@ "$NT" "$dst"
  echo "$dst"
}

b1_paths=""; for s in "${B1[@]}"; do b1_paths+="$(prep_bam "$s"),"; done
b2_paths=""; for s in "${B2[@]}"; do b2_paths+="$(prep_bam "$s"),"; done
printf '%s\n' "${b1_paths%,}" > "$SCRATCH/b1.txt"
printf '%s\n' "${b2_paths%,}" > "$SCRATCH/b2.txt"

echo "[$(date +%H:%M:%S)] rMATS $TAG (sarg=$SARG) ..."
"$RMATS_PYBIN" "$RMATS_PY" \
  --b1 "$SCRATCH/b1.txt" --b2 "$SCRATCH/b2.txt" \
  --gtf "$GTF" -t paired --readLength "$RL" --variable-read-length \
  --nthread "$NT" --od "$SCRATCH/od" --tmp "$SCRATCH/tmp" \
  --task both --cstat 0.0001 > "$OUT/rmats.log" 2>&1

cp "$SCRATCH/od/SE.MATS.JC.txt" "$OUT/SE.MATS.JC.txt"
echo "[$(date +%H:%M:%S)] done $TAG -> $OUT/SE.MATS.JC.txt"
wc -l "$OUT/SE.MATS.JC.txt"
# free the bulky scratch (keep only the SE table copy)
rm -rf "$SCRATCH/tmp" "$SCRATCH/bam"
