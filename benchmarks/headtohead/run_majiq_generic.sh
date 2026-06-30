#!/usr/bin/env bash
# Generic MAJIQ v3 deltapsi runner: build -> psi-coverage(g1,g2) -> deltapsi (g1 - g2).
# Args via env:
#   GFF3, WD (majiq work dir w/ experiments.tsv), N1 N2 (group names = deltapsi psi1/psi2),
#   G1_IDS G2_IDS (space-separated SRR ids for group1=psi1 and group2=psi2).
# dPSI is psi1 - psi2 == N1 - N2.
# MAJIQ academic license: supply your own .lic from https://majiq.biociphers.org/ (never bundled).
#   MAJIQ_BIN (default `majiq` on PATH), MAJIQ_LICENSE (default $HOME/majiq_license_academic_official.lic)
set -euo pipefail

MJ="${MAJIQ_BIN:-majiq}"
LIC="${MAJIQ_LICENSE:-$HOME/majiq_license_academic_official.lic}"
NTHREADS=8
BUILD="$WD/build_out"
SG="$BUILD/splicegraph.zarr"

# Run from the repo root (two levels up from benchmarks/headtohead/) so the relative
# data/ paths in WD/GFF3 resolve regardless of where the script is invoked.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -d "$SG" ]; then
    echo "[$(date +%T)] splicegraph.zarr exists -> skip build"
else
    echo "[$(date +%T)] === build ==="
    "$MJ" build "$GFF3" "$WD/experiments.tsv" "$BUILD" \
        --license "$LIC" --nthreads "$NTHREADS" --overwrite
fi
ls -la "$BUILD" | head -3

sj_for() { for id in $1; do find "$BUILD" -name "*SRR${id}*.sj"; done | sort -u | tr '\n' ' '; }
G1_SJ=$(sj_for "$G1_IDS"); G2_SJ=$(sj_for "$G2_IDS")
echo "G1($N1)_SJ=$G1_SJ"; echo "G2($N2)_SJ=$G2_SJ"

echo "[$(date +%T)] === psi-coverage $N1 ==="
# shellcheck disable=SC2086
"$MJ" psi-coverage "$SG" "$WD/${N1}.psicov" $G1_SJ --license "$LIC" --nthreads "$NTHREADS" --overwrite
echo "[$(date +%T)] === psi-coverage $N2 ==="
# shellcheck disable=SC2086
"$MJ" psi-coverage "$SG" "$WD/${N2}.psicov" $G2_SJ --license "$LIC" --nthreads "$NTHREADS" --overwrite

echo "[$(date +%T)] === deltapsi ($N1 - $N2) ==="
"$MJ" deltapsi -psi1 "$WD/${N1}.psicov" -psi2 "$WD/${N2}.psicov" -n "$N1" "$N2" \
    --splicegraph "$SG" --output-tsv "$WD/deltapsi.tsv" --output-voila "$WD/deltapsi.voila" \
    --license "$LIC" --nthreads "$NTHREADS" --overwrite
echo "[$(date +%T)] === deltapsi TSV header ==="; grep -v '^#' "$WD/deltapsi.tsv" | head -1
echo "[$(date +%T)] === DONE ==="
