# BRAID head-to-head benchmark (reviewer reproduction guide)

This directory holds the code and small derived inputs for BRAID's central empirical
claim: a **distribution-free conformal ΔPSI interval reaches nominal coverage where real
generative tools (MAJIQ, betAS) and rMATS under-cover** the orthogonal RT-PCR truth.

Every method below is a **real tool run on real data** — no stand-ins:

| method | what it is |
|---|---|
| **MAJIQ** | MAJIQ v3 (`rna_majiq` 3.0.23) Bayesian ΔPSI posterior credible interval |
| **betAS** | betAS Beta-posterior interval (the R package, via `run_betas.R`) |
| **rMATS** | rMATS per-replicate IncLevel Student-t CI (Welch df) |
| **BRAID-conformal** | BRAID's k-fold cross-fit absolute-residual Mondrian split-conformal interval |

## Headline result (`comprehensive_benchmark.py`, pooled over 3 datasets, common 4-method set n=139)

| method | cov@95 | Wilson 95% CI | width | interval score ↓ |
|---|---|---|---|---|
| MAJIQ (real binary) | 0.518 | [0.436, 0.599] | 0.173 | 2.040 |
| betAS (real tool) | 0.734 | [0.655, 0.800] | 0.405 | 1.414 |
| rMATS IncLevel t-CI | 0.633 | [0.550, 0.709] | 0.259 | 1.625 |
| **BRAID-conformal** | **0.971** | **[0.928, 0.989]** | 0.612 | **0.720** |

On the larger rMATS-matched pool that does not require a MAJIQ match (n=196): BRAID 0.949 [0.909, 0.972], betAS 0.765, rMATS 0.679.

Only BRAID's interval includes nominal 0.95, and it also wins the (proper) interval score —
it is the best calibrated *and* sharpest-for-its-coverage interval, not merely the widest.

**Paired test (the statistically correct comparison).** Coverage is a paired design — the
same events scored by every method — so the proper test is **McNemar's exact**, not
non-overlapping marginal CIs. On the pooled set BRAID-conformal dominates every
competitor: at most one event is covered by a competitor but missed by BRAID
(`c ≤ 1`), and McNemar's exact test is overwhelmingly significant in every comparison:

| comparison | b (BRAID covers, other misses) | c (other covers, BRAID misses) | McNemar exact p |
|---|---|---|---|
| BRAID vs MAJIQ | 63 | **0** | 2.2×10⁻¹⁹ |
| BRAID vs betAS | 33 | **0** | 2.3×10⁻¹⁰ |
| BRAID vs rMATS | 48 | **1** | 1.8×10⁻¹³ |

The 139 events span 135 genes (design effect 1.03 → no meaningful pseudoreplication; gene-
clustered bootstrap CI matches the naive Wilson CI). The `rMATS IncLevel t-CI` uses a
Student-t (Welch df) quantile — the correct small-sample choice for n=3 replicates; a normal
approximation would under-cover artefactually and is *not* used. rMATS itself emits a
likelihood-ratio test, not an interval, so this is a fair naive-baseline interval from its
per-replicate IncLevels. Reproduce all of the above with `python comprehensive_benchmark.py`
and audit the statistics with `python stat_review.py`.

## Quick reproduction with prepared data

From the repository root, create the benchmark environment and expose the prepared
reviewer data bundle at the paths expected by the scripts:

```bash
conda env create -f environment-benchmark.yml
conda activate braid-benchmark
python -m pip install -e ".[benchmark,report]"

export BRAID_BENCH=/path/to/braid_bench
mkdir -p data/public_benchmarks
ln -sfnT "${BRAID_BENCH}/data/public_benchmarks/GSE59335" data/public_benchmarks/GSE59335
ln -sfnT "${BRAID_BENCH}/data/public_benchmarks/GSE54651" data/public_benchmarks/GSE54651
ln -sfnT "${BRAID_BENCH}/data/public_benchmarks/SRS354082" data/public_benchmarks/SRS354082
ln -sfnT "${BRAID_BENCH}/data/public_benchmarks/meta" data/public_benchmarks/meta

cd benchmarks/headtohead
python comprehensive_benchmark.py
python stat_review.py
python majiq_coverage.py
python pooled_coverage.py
python variance_decomposition.py
```

The central check is the `comprehensive_benchmark.py` pooled common set:
`n=139`; MAJIQ `0.518`, betAS `0.734`, rMATS `0.633`, and BRAID-conformal
`0.971` coverage. The larger rMATS-matched set is `n=196`, with BRAID-conformal
coverage `0.949`.

## Datasets

| id | organism / build | groups (SRA) | truth | usable for |
|---|---|---|---|---|
| **GSE59335** (TRA2 KD) | human / hg19 (SRP044265) | control `SRR1513329/330/331` vs knockdown `SRR1513332/333/334` | cassette-exon RT-PCR ΔPSI (83 pos + 44 neg) | coverage |
| **GSE54651** (circadian) | mouse / mm10 (SRP036186) | cerebellum `SRR1158546/548/550` vs liver `SRR1158578/580/582` | junction RT-PCR ΔPSI (50) | coverage |
| **SRS354082** (PC3E/GS689) | human / hg19 | epithelial PC3E `SRR536348/350/352` vs mesenchymal GS689.Li `SRR536342/344/346` | cassette-exon RT-PCR ΔPSI (34, from the rMATS study) | coverage |
| **GSE55215** (QKI KD) | human / hg19 (SRP038702) | sh-Ctrl `SRR1173996/997` vs sh-QKI `SRR1173998/999` | qualitative exon list | detection only |

ΔPSI orientation matches the RT-PCR tables: TRA2 = knockdown − control, circadian = liver −
cerebellum (applied identically to every method).

## What is in this directory vs what you must regenerate

**Provided here** (so you can re-run the coverage scripts without re-running betAS):
`*_betas_intervals.tsv` (real betAS 50–95% intervals), `*_betas_truth.tsv`, `*_betas_counts.tsv`,
`*_headtohead*.json`, and all analysis code.
For SRS354082, the canonical real-betAS export is `srs_betas_*`; `pc3e_betas_*` is a
legacy draft export retained only for provenance.

**Not in the repo (too large / raw):** the `data/public_benchmarks/<dataset>/` tree — BAMs,
rMATS output, MAJIQ output, and the curated RT-PCR truth tables. Regenerate with the steps
below (accessions above; the curated truth tables are derived from each study's published
RT-PCR supplement and are available from the authors on request).

## Reproduce from scratch

```bash
# 0. Environments (see ../../environment*.yml). rMATS + samtools/hisat2 via bioconda;
#    MAJIQ v3 in its own env; betAS as an R package.

# 1. Download FASTQs for the accessions above (SRA / ENA) and align (HISAT2 --dta),
#    sort + index -> data/public_benchmarks/<DS>/bam/SRRxxxx.sorted.bam(.bai)

# 2. rMATS per dataset (b1/b2 = the two groups; BRAID maps --b1 -> control/sample_1,
#    --b2 -> treatment/sample_2, ΔPSI = b1 - b2 = rMATS IncLevelDifference)
#    -> data/public_benchmarks/<DS>/rmats/

# 3. MAJIQ v3 (needs your OWN academic .lic from https://majiq.biociphers.org/):
MAJIQ_LICENSE=~/your_majiq.lic bash run_majiq_pipeline.sh          # TRA2 (hg19)
WD=data/public_benchmarks/GSE54651/majiq \
  GFF3=data/public_benchmarks/reference/mm10/gencode.vM25.annotation.gff3 \
  N1=LIVER N2=CEREB G1_IDS="1158578 1158580 1158582" G2_IDS="1158546 1158548 1158550" \
  MAJIQ_LICENSE=~/your_majiq.lic bash run_majiq_generic.sh          # circadian (mm10)

# 4. betAS intervals (provided, or regenerate):
python export_for_betas.py --rmats-se .../SE.MATS.JC.txt --validated ... --failed ... \
  --swap-groups --out-counts tra2_betas_counts.tsv --out-truth tra2_betas_truth.tsv
Rscript run_betas.R tra2_betas_counts.tsv tra2_betas_intervals.tsv

# 5. The benchmark tables:
python comprehensive_benchmark.py     # 4 methods x 3 datasets, per-dataset + pooled
python majiq_coverage.py              # MAJIQ vs real-betAS vs conformal (TRA2 detail)
python pooled_coverage.py             # BRAID vs betAS vs rMATS pooled coverage + CIs
python variance_decomposition.py      # platform-vs-sampling variance (why generative under-covers)
```

## Script index

| script | produces |
|---|---|
| `comprehensive_benchmark.py` | headline 4-method × 3-dataset table (coverage, width, interval score, Wilson CIs) |
| `majiq_coverage.py` | MAJIQ (real) vs betAS (real) vs BRAID-conformal on identical TRA2 events |
| `pooled_coverage.py` | pooled BRAID/betAS/rMATS coverage with Wilson CIs |
| `cross_dataset_transfer.py` | does a calibrator fit on one dataset transfer to another (no refit) |
| `bayesian_interval_coverage.py` | license-free MAJIQ-class stand-ins (beta-binomial, heterogen) — sanity |
| `variance_decomposition.py` | sampling vs orthogonal-truth residual variance decomposition |
| `head_to_head_coverage.py` | shared event parsing, matching, and interval estimators (library) |
| `run_majiq_pipeline.sh` / `run_majiq_generic.sh` | MAJIQ v3 build→psi-coverage→deltapsi |
| `export_for_betas.py` / `run_betas.R` | per-replicate counts → real betAS intervals |

## MAJIQ license

MAJIQ requires an academic license file obtained by registering at
<https://majiq.biociphers.org/> (a click-through academic-use agreement). It is **never**
bundled here; supply your own via `MAJIQ_LICENSE=/path/to/your.lic`.
