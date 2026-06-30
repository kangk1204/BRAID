# BRAID Benchmark Handoff

This document records the current benchmark state for `BRAID` so the work can be
resumed on another computer without reconstructing context. It supersedes the
older v2/QKI-era handoff; the current headline is the **conformal head-to-head
coverage benchmark**.

## Repository snapshot

- Repo: `/home/keunsoo/Projects/26_BRAID`
- Working branch: `phase0-repro-floor` (canonical; `main` is reset to this state)
- Test status: `python -m pytest tests/ -q` -> **961 passed**
- Lint: `ruff check braid/` -> clean

## Current thesis (what the paper claims)

BRAID is a caller-agnostic confidence layer that post-processes rMATS count
tables and adds **conformally calibrated Delta PSI intervals**. The benchmark
question is interval *coverage of orthogonal RT-PCR Delta PSI* at nominal 95%,
not point accuracy or detection. The manuscript is
`paper/braid_calibrated_dpsi_manuscript.md` (+ compiled
`paper/braid_calibrated_dpsi.pdf`); `paper/` is gitignored / local-only.

## Datasets (real, external RT-PCR truth)

All under `data/public_benchmarks/` (the old `real_benchmark/` tree is gone):

- **TRA2 / GSE59335** (human MDA-MB-231 TRA2A/B KD). RT-PCR targets from the TRA2
  study, reused via the SUPPA2 supplement. `rmats/SE.MATS.JC.txt`, `majiq/deltapsi.tsv`,
  `targets/validated_events.tsv` + `failed_events.tsv`.
- **Circadian / GSE54651** (mouse cerebellum vs liver, 3v3). RT-PCR truth via the
  MAJIQ/SUPPA2 supplements (Zhang 2014). `meta/gse54651_circadian_positive_events.tsv`.
- **SRS354082 / PC3E vs GS689.Li** (human prostate EMT, 3v3; Shen 2014 rMATS study).
  6 local BAMs; `meta/rmats_pc3e_gs689_positive_events.tsv` (34 RT-PCR-positive SE).

## Headline results (all reproduced 2026-06-16)

Pooled common four-method set (n=139, 135 genes), coverage @95% / interval score:

| Method | Coverage | Interval score |
|---|---:|---:|
| MAJIQ v3 (Gaussian from posterior mean/SD) | 0.518 | 2.040 |
| betAS (real R package) | 0.734 | 1.414 |
| rMATS IncLevel t-CI | 0.633 | 1.625 |
| **BRAID conformal** | **0.971** | **0.720** |

- Paired exact McNemar (BRAID covers & competitor misses : reverse):
  vs MAJIQ 63-0 (p=2.17e-19), vs betAS 33-0 (p=2.33e-10), vs rMATS 48-1 (p=1.78e-13).
- Full rMATS-matched set, MAJIQ not required (n=196): BRAID 0.949, betAS 0.765, rMATS 0.679.
- SRS354082 full 34-event: rMATS-perRep 0.412, betAS 0.912, BRAID-conformal-abs 0.971.
- Recalibration ablation (same conformal applied to each method's point estimate):
  MAJIQ 0.950, betAS 0.957, rMATS 0.971, BRAID 0.971 -> nominal coverage is from the
  calibration construction; the informative axis is width.
- No-refit cross-dataset transfer: TRA2-fit -> circadian 1.000; circadian-fit -> TRA2 0.911.
- Variance decomposition (TRA2): ~94% of RNA-seq-to-RT-PCR error variance remains
  outside the read-sampling component; this residual includes platform/assay-definition
  discordance and any systematic offset, which is why sampling-only intervals under-cover.

## Comparator tools

- **rMATS**: rMATS-turbo, `mamba` env `rmats`. Outputs in `data/public_benchmarks/<ds>/rmats/`.
- **betAS**: real R package, `mamba` env `betas`, driven by
  `benchmarks/headtohead/run_betas.R`; per-event intervals exported to
  `benchmarks/headtohead/<ds>_betas_intervals.tsv` (canonical for SRS = `srs_betas_*`).
- **MAJIQ**: `rna_majiq` v3.0.23, `deltapsi.tsv`; comparator interval = mean +/- 1.96*SD
  (Gaussian reconstruction; native asymmetric interval not exported).

## Reproduce the headline (no large data needed for the report tables)

```bash
cd /home/keunsoo/Projects/26_BRAID
python -m pytest tests/ -q                                  # 961 passed

# canonical 4-method x 3-dataset table -> benchmarks/results/headtohead_coverage.json
python benchmarks/headtohead/comprehensive_benchmark.py

# recalibration ablation -> benchmarks/results/recalibration_ablation.json
python benchmarks/headtohead/recalibration_ablation.py

# single-dataset SRS354082 full head-to-head
python benchmarks/headtohead/head_to_head_coverage.py \
  --rmats-se data/public_benchmarks/SRS354082/rmats/SE.MATS.JC.txt \
  --validated data/public_benchmarks/meta/rmats_pc3e_gs689_positive_events.tsv \
  --betas-tsv benchmarks/headtohead/srs_betas_intervals.tsv --name SRS354082

# cross-dataset no-refit transfer
python benchmarks/headtohead/cross_dataset_transfer.py \
  --tra2-se data/public_benchmarks/GSE59335/rmats/SE.MATS.JC.txt \
  --tra2-validated data/public_benchmarks/GSE59335/targets/validated_events.tsv \
  --tra2-failed data/public_benchmarks/GSE59335/targets/failed_events.tsv \
  --tra2-betas benchmarks/headtohead/tra2_betas_intervals.tsv \
  --circ-se data/public_benchmarks/GSE54651/rmats/SE.MATS.JC.txt \
  --circ-tsv data/public_benchmarks/meta/gse54651_circadian_positive_events.tsv \
  --circ-betas benchmarks/headtohead/circ_betas_intervals.tsv

# figures (paper/figures + benchmarks/headtohead/figures)
python benchmarks/headtohead/make_figures.py
```

Narrative + tables are also kept in `benchmarks/headtohead/RESULTS.md` and `README.md`.

## Status of older (pre-reframe) benchmarks

These artifacts still exist in `benchmarks/results/` but are NOT the current
headline, and several were removed from the manuscript:

- **QKI (GSE55215)**: kept only as a detection/recall cross-check. The original
  "QKI rMATS FPR" headline did **not** reproduce on GRCh38 (FPR 68% -> 2%) and was
  **cut** from the manuscript. `qki_rmats_benchmark_results.json`, `qki_roc_analysis.json`.
- **TARDBP**: **removed** from the manuscript (the ENCODE accession did not resolve).
  `tardbp_braid_results.json` retained for provenance only.
- **PacBio non-SE**: realigned to a reproducible **252-event** A3SS/A5SS set
  (cross-fit coverage 0.929, uncalibrated 0.714); the older 204-event numbers are
  superseded. `rtpcr_benchmark.json`, `pacbio_sharpness.json`.
- **3-cell-line isoform (legacy)**: `comprehensive_benchmark.json` (depends on data
  no longer in the repo).

## Caveats

- `paper/` and some figures are gitignored (manuscript is local-only).
- Large BAM/rMATS/MAJIQ outputs under `data/public_benchmarks/` are local; regenerate
  from the public accessions listed in `benchmarks/headtohead/README.md` for a release.
- Do not sum rMATS replicate counts; per-replicate preservation is required.
- The conformal calibrator is JSON-only (no pickle).
