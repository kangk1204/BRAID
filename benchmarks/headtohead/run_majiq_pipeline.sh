#!/usr/bin/env bash
# MAJIQ v3 (rna_majiq 3.0.23) deltapsi pipeline on GSE59335 TRA2 knockdown.
# Groups fixed from the rMATS b1/b2 design: control=SRR1513329/330/331,
# knockdown=SRR1513332/333/334. dPSI is computed KD - control (psi1=KD, psi2=CTRL)
# to match the RT-PCR truth convention used for every other tool (head_to_head swap=True).
# MAJIQ academic license: reviewers must obtain their own .lic from
# https://majiq.biociphers.org/ (click-through academic agreement) -- it is never bundled.
# Configurable via env vars:
#   MAJIQ_BIN      path to the majiq binary           (default: `majiq` on PATH)
#   MAJIQ_LICENSE  path to your academic .lic file    (default: $HOME/majiq_license_academic_official.lic)
#   GFF3           hg19 GENCODE annotation (GFF3)      (default: data/.../gencode.v19.annotation.gff3)
set -euo pipefail

MJ="${MAJIQ_BIN:-majiq}"
LIC="${MAJIQ_LICENSE:-$HOME/majiq_license_academic_official.lic}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GFF3="${GFF3:-$ROOT/data/public_benchmarks/reference/hg19/gencode.v19.annotation.gff3}"
WD="$ROOT/data/public_benchmarks/GSE59335/majiq"
BUILD="$WD/build_out"
NTHREADS=8

cd "$ROOT"
SG="$BUILD/splicegraph.zarr"   # new-majiq v3 writes splicegraph.zarr (not .sg)
if [ -d "$SG" ]; then
    echo "[$(date +%T)] === splicegraph.zarr exists, skipping build ==="
else
    echo "[$(date +%T)] === MAJIQ build ==="
    "$MJ" build "$GFF3" "$WD/experiments.tsv" "$BUILD" \
        --license "$LIC" --nthreads "$NTHREADS" --overwrite
fi

echo "[$(date +%T)] === build outputs ==="
ls -la "$BUILD"

# Discover the SJ files build produced, per group (match by SRR id).
sj_for() { for id in "$@"; do find "$BUILD" -name "*SRR${id}*.sj" ; done | sort -u | tr '\n' ' '; }
CTRL_SJ=$(sj_for 1513329 1513330 1513331)
KD_SJ=$(sj_for 1513332 1513333 1513334)
echo "CTRL_SJ=$CTRL_SJ"
echo "KD_SJ=$KD_SJ"

echo "[$(date +%T)] === psi-coverage (control) ==="
# shellcheck disable=SC2086
"$MJ" psi-coverage "$SG" "$WD/control.psicov" $CTRL_SJ \
    --license "$LIC" --nthreads "$NTHREADS" --overwrite

echo "[$(date +%T)] === psi-coverage (knockdown) ==="
# shellcheck disable=SC2086
"$MJ" psi-coverage "$SG" "$WD/knockdown.psicov" $KD_SJ \
    --license "$LIC" --nthreads "$NTHREADS" --overwrite

echo "[$(date +%T)] === deltapsi (KD - control) ==="
"$MJ" deltapsi -psi1 "$WD/knockdown.psicov" -psi2 "$WD/control.psicov" -n KD CTRL \
    --splicegraph "$SG" \
    --output-tsv "$WD/deltapsi.tsv" \
    --output-voila "$WD/deltapsi.voila" \
    --license "$LIC" --nthreads "$NTHREADS" --overwrite

echo "[$(date +%T)] === deltapsi TSV head ==="
head -5 "$WD/deltapsi.tsv"
echo "[$(date +%T)] === DONE ==="
