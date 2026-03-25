## BRAID Benchmark Handoff

This document records the current benchmark state for `BRAID` so the work can be resumed on another computer without reconstructing context.

### Repository snapshot

- Repo: `/home/keunsoo/projects/24_BRAID`
- Checked commit during this handoff: `16491f6`
- Test status: `451 passed, 4 warnings`

### What is already available

Current result files present in `benchmarks/results/`:

- `rtpcr_benchmark.json`
- `qki_rmats_benchmark_results.json`
- `multirep_benchmark.json`
- `comprehensive_benchmark.json`
- `pacbio_sharpness.json`
- `qki_roc_analysis.json`
- `tra2_braid_results.json`
- `tardbp_braid_results.json`
- `additional_benchmarks_status.md`

### Environment setup on a new machine

Recommended benchmark environment:

```bash
cd /path/to/BRAID
conda env create -f environment-benchmark.yml
conda activate braid-benchmark
braid doctor --strict
```

If `stringtie` / `rmats.py` are missing:

```bash
mamba install -c conda-forge -c bioconda samtools hisat2 gffcompare stringtie rmats
```

### Data layout expected by benchmark scripts

Most benchmark scripts assume:

- `BRAID_DATA_DIR=real_benchmark`

Important subpaths used by scripts:

- `real_benchmark/annotation/gencode.v38.nochr.gtf`
- `real_benchmark/reference/grch38/genome.fa`
- `real_benchmark/rtpcr_benchmark/qki/`
- `real_benchmark/bam/`
- `real_benchmark/results/`

If the new machine stores benchmark data elsewhere:

```bash
export BRAID_DATA_DIR=/absolute/path/to/real_benchmark
```

### Benchmark 1: PacBio CI calibration

Script:

```bash
python benchmarks/rtpcr_benchmark.py
```

Code entry:

- `benchmarks/rtpcr_benchmark.py`

Output:

- `benchmarks/results/rtpcr_benchmark.json`

Current result snapshot:

- `204` A3SS/A5SS events
- correlation `r = 0.8311`
- `R^2 = 0.6907`
- `MAE = 0.1398`
- calibrated CI coverage `94.61%`
- uncalibrated fixed-scale coverage `72.55%`
- calibrated confident events `10`
- calibrated confident accuracy `100%`
- uncalibrated confident events `33`
- uncalibrated confident accuracy `90.9%`

Current support-bin summary from `rtpcr_benchmark.json`:

| Support bin | Events | Coverage | Median CI width | Confident |
|---|---:|---:|---:|---:|
| `<20` | 5 | 1.0000 | 1.0000 | 0 |
| `20-49` | 16 | 0.9375 | 1.0000 | 0 |
| `50-99` | 39 | 0.9487 | 0.9802 | 0 |
| `100-249` | 42 | 0.9286 | 0.9503 | 2 |
| `250+` | 102 | 0.9510 | 0.5132 | 8 |

Sharpness check:

- file: `benchmarks/results/pacbio_sharpness.json`
- overall random-width baseline coverage: `59.14%`
- BRAID overall coverage: `94.61%`
- largest useful lift is in `250+` support bin: `95.1%` vs random `44.7%`

### Benchmark 2: QKI rMATS + BRAID hybrid

Runner:

```bash
bash benchmarks/run_qki_rmats_benchmark.sh
```

What it does:

1. lifts QKI target tables with `benchmarks/liftover_qki_targets.py`
2. creates `b1.txt` and `b2.txt` from QKI BAMs
3. runs `rmats.py` if `SE.MATS.JC.txt` is missing
4. runs `benchmarks/qki_rmats_benchmark.py`

Main inputs expected:

- `real_benchmark/rtpcr_benchmark/qki/ctrl_rep1.bam`
- `real_benchmark/rtpcr_benchmark/qki/ctrl_rep2.bam`
- `real_benchmark/rtpcr_benchmark/qki/qki_kd.bam`
- `real_benchmark/annotation/gencode.v38.nochr.gtf`

Output:

- `benchmarks/results/qki_rmats_benchmark_results.json`

Current result snapshot:

- validated targets: `80`
- RT-PCR failed targets: `25`
- matched null controls: `80`
- validated matched: `71/80`
- validated significant: `44/80`
- validated supported: `14/80`
- validated high-confidence: `10/80`
- failed matched: `22/25`
- failed significant: `17/25`
- failed supported: `6/25`
- failed high-confidence: `4/25`
- matched null significant: `0/80`
- matched null supported: `0/80`
- matched null high-confidence: `0/80`
- high-support matched null supported: `0/32`
- high-support matched null high-confidence: `0/32`

Related analysis file:

- `benchmarks/results/qki_roc_analysis.json`

Useful interpretation from current results:

- plain `rMATS FDR < 0.05` gives `44 TP / 17 FP` on the validated-vs-failed view
- BRAID `supported` gives `14 TP / 6 FP`
- BRAID `high-confidence` gives `10 TP / 4 FP`
- matched null controls stay at `0` for supported and high-confidence

### Benchmark 3: TRA2 KD

Current artifact only:

- `benchmarks/results/tra2_braid_results.json`

Current numbers:

- total events: `42,443`
- rMATS significant: `1,566`
- rMATS non-significant: `40,877`
- BRAID supported on significant: `517`
- BRAID high-confidence on significant: `27`
- BRAID supported on non-significant: `0`
- BRAID high-confidence on non-significant: `0`

Notes:

- this artifact is already present
- I did not find a checked-in standalone runner in the repo for regenerating it directly from scratch
- benchmark background is summarized in `benchmarks/additional_datasets.md`

### Benchmark 4: TARDBP KD

Current artifact only:

- `benchmarks/results/tardbp_braid_results.json`

Current numbers:

- total events: `84,838`
- rMATS significant: `2,382`
- rMATS non-significant: `82,456`
- BRAID supported on significant: `1,120`
- BRAID high-confidence on significant: `23`
- BRAID supported on non-significant: `0`
- BRAID high-confidence on non-significant: `0`

Notes:

- this artifact is already present
- as with TRA2, the repo currently contains the result JSON but not a clearly named regeneration script next to it

### Benchmark 5: Multi-replicate benchmark

Runner:

```bash
bash benchmarks/run_multirep_benchmark.sh
```

Optional GM12878 enable:

```bash
RUN_GM12878=1 bash benchmarks/run_multirep_benchmark.sh
```

What it does:

- QKI multi-replicate PSI for `EZH2`, `AKT1`, `STAT3`, `RUNX1`, `BCL2L1`
- optional GM12878 isoform multi-replicate bootstrap

Output:

- `benchmarks/results/multirep_benchmark.json`

Current result snapshot:

QKI sub-benchmark:

- `EZH2`: single confident `0`, multi confident `1`, multi median CI width `0.1951`
- `AKT1`: single confident `1`, multi confident `7`, single median `0.1240`, multi median `0.0805`
- `STAT3`: no confident events
- `RUNX1`: no confident events
- `BCL2L1`: no confident events

GM12878 sub-benchmark:

- isoforms: `32,771`
- elapsed: `1376.76 s`
- median total CV: `0.3588`
- median biological std: `7.7294`
- median sampling std: `4.0797`
- bio/sampling ratio: about `1.9x`

Important script behavior:

- `RUN_GM12878=0` skips GM12878
- QKI alignment of missing replicate BAMs only happens if the corresponding FASTQ is present
- if raw FASTQs are removed, the script can still run from existing BAMs

### Benchmark 6: 3-cell-line isoform confidence benchmark

Runner:

```bash
python benchmarks/bootstrap_benchmark.py
```

Output:

- `benchmarks/results/comprehensive_benchmark.json`

Datasets used by the script:

- K562
- GM12878
- IMR90

Current result snapshot:

| Dataset | Isoforms | Baseline precision | Precision at CV <= 0.20 |
|---|---:|---:|---:|
| K562 | 43,522 | 39.30% | 61.10% |
| GM12878 | 47,818 | 42.61% | 72.92% |
| IMR90 | 37,538 | 45.46% | 59.11% |

Notes:

- the script evaluates many CV thresholds, not just `0.20`
- results are annotation-based and BAM-free in the bootstrap phase

### Additional benchmark status

File:

- `benchmarks/results/additional_benchmarks_status.md`

Current status:

- usable/completed: QKI, PacBio, GM12878 multi-replicate, TRA2, TARDBP, 3-cell-line isoform benchmark
- unusable so far: PC3E/GS689, MAJIQ mouse

### Recommended order to resume work on another computer

1. clone the repo and create `braid-benchmark` environment
2. restore or mount `real_benchmark/` data and set `BRAID_DATA_DIR` if needed
3. run `braid doctor --strict`
4. verify tests:

```bash
python -m pytest -q
```

5. verify current committed benchmark outputs still exist:

```bash
ls benchmarks/results
```

6. rerun in this order if you want to reproduce headline results:

```bash
python benchmarks/rtpcr_benchmark.py
bash benchmarks/run_qki_rmats_benchmark.sh
RUN_GM12878=1 bash benchmarks/run_multirep_benchmark.sh
python benchmarks/bootstrap_benchmark.py
```

7. compare regenerated outputs against:

- `benchmarks/results/rtpcr_benchmark.json`
- `benchmarks/results/qki_rmats_benchmark_results.json`
- `benchmarks/results/multirep_benchmark.json`
- `benchmarks/results/comprehensive_benchmark.json`
- `benchmarks/results/tra2_braid_results.json`
- `benchmarks/results/tardbp_braid_results.json`

### Caveats to remember

- Some benchmark scripts assume large local data under `real_benchmark/`.
- QKI and GM12878 reruns may depend on BAMs already existing if raw FASTQs are gone.
- TRA2 and TARDBP currently have checked-in result JSONs, but not an equally obvious one-command regeneration script next to them.
- `paper/` and some generated figures are ignored in git, so manuscript reproduction may require local artifacts outside the repo snapshot.

### Quick command summary

```bash
# tests
python -m pytest -q

# PacBio calibration
python benchmarks/rtpcr_benchmark.py

# QKI hybrid benchmark
bash benchmarks/run_qki_rmats_benchmark.sh

# multi-replicate benchmark
RUN_GM12878=1 bash benchmarks/run_multirep_benchmark.sh

# 3-cell-line isoform confidence benchmark
python benchmarks/bootstrap_benchmark.py
```
