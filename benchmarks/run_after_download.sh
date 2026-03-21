#!/bin/bash
# Wait for BAM downloads to complete, then run benchmarks
set -e

BRAID_DATA_DIR="${BRAID_DATA_DIR:-real_benchmark}"

BAM_DIR="$BRAID_DATA_DIR/bam"
GM_BAM="$BAM_DIR/GM12878_ENCFF550SET.bam"
IMR_BAM="$BAM_DIR/IMR90_ENCFF560TMJ.bam"

echo "Waiting for BAM downloads to complete..."

# Wait for GM12878 (19.3 GB)
while true; do
    size=$(stat -c%s "$GM_BAM" 2>/dev/null || echo 0)
    if [ "$size" -gt 19000000000 ]; then
        echo "GM12878 BAM download complete ($size bytes)"
        break
    fi
    # Check if wget still running
    if ! pgrep -f "ENCFF550SET" > /dev/null 2>&1; then
        echo "GM12878 download finished (wget exited, size=$size)"
        break
    fi
    echo "  GM12878: $(numfmt --to=iec $size) / 19.3G"
    sleep 30
done

# Wait for IMR90 (16.6 GB)
while true; do
    size=$(stat -c%s "$IMR_BAM" 2>/dev/null || echo 0)
    if [ "$size" -gt 16000000000 ]; then
        echo "IMR90 BAM download complete ($size bytes)"
        break
    fi
    if ! pgrep -f "ENCFF560TMJ" > /dev/null 2>&1; then
        echo "IMR90 download finished (wget exited, size=$size)"
        break
    fi
    echo "  IMR90: $(numfmt --to=iec $size) / 16.6G"
    sleep 30
done

echo ""
echo "Both downloads complete. Indexing BAMs..."

samtools index -@ 8 "$GM_BAM" &
samtools index -@ 8 "$IMR_BAM" &
wait
echo "BAMs indexed."

echo ""
echo "Running GM12878 benchmark..."

# StringTie GM12878
echo "=== StringTie GM12878 ==="
RESULTS="$BRAID_DATA_DIR/results"
mkdir -p "$RESULTS"
stringtie "$GM_BAM" -p 8 -o "$RESULTS/stringtie_GM12878.gtf"
echo "StringTie GM12878 done"

# BRAID GM12878 (with bootstrap)
echo "=== BRAID GM12878 ==="
python -m braid assemble \
    --bam "$GM_BAM" \
    --reference "$BRAID_DATA_DIR/reference/grch38/genome.fa" \
    -o "$RESULTS/braid_GM12878.gtf" \
    -t 8 --bootstrap
echo "BRAID GM12878 done"

# GFFcompare GM12878
GENCODE_CHR="$BRAID_DATA_DIR/annotation/gencode.v38.annotation.gtf"

echo "=== GFFcompare GM12878 ==="
gffcompare -r "$GENCODE_CHR" -o "$RESULTS/rs_GM12878" "$RESULTS/braid_GM12878.gtf"
gffcompare -r "$GENCODE_CHR" -o "$RESULTS/st_GM12878" "$RESULTS/stringtie_GM12878.gtf"

echo ""
echo "Running IMR90 benchmark..."

# StringTie IMR90
echo "=== StringTie IMR90 ==="
stringtie "$IMR_BAM" -p 8 -o "$RESULTS/stringtie_IMR90.gtf"
echo "StringTie IMR90 done"

# BRAID IMR90 (with bootstrap)
echo "=== BRAID IMR90 ==="
python -m braid assemble \
    --bam "$IMR_BAM" \
    --reference "$BRAID_DATA_DIR/reference/grch38/genome.fa" \
    -o "$RESULTS/braid_IMR90.gtf" \
    -t 8 --bootstrap
echo "BRAID IMR90 done"

# GFFcompare IMR90
echo "=== GFFcompare IMR90 ==="
gffcompare -r "$GENCODE_CHR" -o "$RESULTS/rs_IMR90" "$RESULTS/braid_IMR90.gtf"
gffcompare -r "$GENCODE_CHR" -o "$RESULTS/st_IMR90" "$RESULTS/stringtie_IMR90.gtf"

echo ""
echo "============================="
echo "  ALL BENCHMARKS COMPLETE"
echo "============================="
echo ""

# Print results
for ds in GM12878 IMR90; do
    echo "=== $ds ==="
    echo "--- BRAID ---"
    grep -E "Intron level|Transcript level" "$RESULTS/rs_${ds}.stats" 2>/dev/null || echo "  (no stats)"
    echo "--- StringTie ---"
    grep -E "Intron level|Transcript level" "$RESULTS/st_${ds}.stats" 2>/dev/null || echo "  (no stats)"
    echo ""
done
