#!/usr/bin/env bash
# Phase-3 non-SE coverage validation: build a GRCh38 STAR index, then run BRAID's
# PacBio long-read validation driver (K562 SRR387661 short-read vs ENCFF652QLH long-read
# transcript models). The underlying rtpcr_benchmark.py scores junction-centric A3SS/A5SS
# events -> orthogonal (long-read) coverage truth for NON-SE event types.
set -euo pipefail
# Run from the repo root (one level up from benchmarks/).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WD=benchmark_results/pacbio_regen
REF=$WD/ref
mkdir -p "$REF"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

GENCODE=https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38

# 1. GRCh38 primary assembly + GENCODE v38 annotation
if [ ! -f "$REF/genome.fa" ]; then
  log "download GRCh38 primary assembly (~845 MB)"
  curl -sL --retry 3 -o "$REF/genome.fa.gz" "$GENCODE/GRCh38.primary_assembly.genome.fa.gz"
  gunzip -f "$REF/genome.fa.gz"
fi
if [ ! -f "$REF/gencode.v38.gtf" ]; then
  log "download GENCODE v38 GTF"
  curl -sL --retry 3 -o "$REF/gencode.v38.gtf.gz" "$GENCODE/gencode.v38.annotation.gtf.gz"
  gunzip -f "$REF/gencode.v38.gtf.gz"
fi

# 2. STAR genome index (114 GB RAM available)
if [ ! -f "$REF/star/SAindex" ]; then
  log "STAR genomeGenerate (GRCh38)"
  mkdir -p "$REF/star"
  STAR --runMode genomeGenerate --genomeDir "$REF/star" \
       --genomeFastaFiles "$REF/genome.fa" --sjdbGTFfile "$REF/gencode.v38.gtf" \
       --sjdbOverhang 100 --runThreadN 16
fi

# 3. Driver: download SRR387661 + ENCFF652QLH, align (STAR), score A3SS/A5SS vs long-read
log "run PacBio long-read validation driver (downloads SRA + long-read GTF, aligns, scores)"
python benchmarks/regenerate_pacbio_validation.py \
    --reference "$REF/star" --annotation "$REF/gencode.v38.gtf" \
    --aligner STAR --threads 16 \
    --fixture-out "$WD/pacbio_nonSE_fixture.json"
log "DONE -> $WD/pacbio_nonSE_fixture.json"
