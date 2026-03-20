# BRAID Project Guide

> **For any LLM or developer picking up this project.**
> Last updated: 2026-03-21. Commit: `293ddab`.

---

## What is BRAID?

**BRAID** (Bootstrap Reliability Assessment for Isoform Detection) is a post-processing tool that adds **calibrated confidence intervals** to RNA-seq alternative splicing analysis.

It does **not** replace existing tools. It sits on top of **rMATS** (event detection) and **StringTie** (transcript assembly) and answers: **"Among the events already detected, which ones should I trust?"**

**GitHub**: https://github.com/kangk1204/BRAID

---

## Repository Structure

```
BRAID/
├── braid/                  # Main Python package (79 files)
│   ├── __init__.py         # version = "0.1.0"
│   ├── cli.py              # CLI entry point (1200 lines)
│   ├── pipeline.py         # Full assembly pipeline
│   ├── target/             # ★ Core BRAID confidence modules
│   │   ├── psi_bootstrap.py        # PSI inference (overdispersed Beta posterior)
│   │   ├── rmats_bootstrap.py      # rMATS integration
│   │   ├── multi_replicate_bootstrap.py  # Multi-rep variance decomposition
│   │   ├── stringtie_bootstrap.py  # StringTie isoform CI
│   │   ├── extractor.py           # Gene/transcript extraction from GTF
│   │   ├── assembler.py           # Target gene assembly
│   │   ├── comparator.py          # Result comparison
│   │   └── gencode_index.py       # GENCODE annotation index
│   ├── graph/              # Splice graph construction
│   ├── flow/               # Flow decomposition (NNLS, NMF, neural)
│   ├── scoring/            # ML transcript scoring
│   ├── splicing/           # AS event detection
│   ├── io/                 # BAM/GTF I/O
│   ├── cuda/               # GPU acceleration (optional)
│   ├── denovo/             # De novo assembly
│   ├── dashboard/          # Streamlit dashboard
│   └── utils/              # Utilities
├── tests/                  # 443 tests (all in-memory, no external data)
├── benchmarks/             # Benchmark scripts
├── paper/                  # Manuscript
│   ├── braid.tex           # LaTeX source
│   ├── braid.pdf           # Compiled PDF (180KB)
│   ├── generate_figures.py # Figure generation script
│   └── figures/            # 5 figures × 3 formats (PDF/PNG/JPG)
├── demo/                   # Quick-start demo
│   ├── run_demo.py         # Generates interactive HTML report
│   └── results/            # Demo output
├── pyproject.toml          # Package config
├── README.md               # User documentation
└── LICENSE                 # MIT
```

---

## Core Algorithm

### Overdispersed Beta Posterior

Standard Poisson/binomial bootstraps produce overconfident intervals (72.5% CI coverage). BRAID uses:

```
Ψ | n_inc, n_exc ~ Beta(α · n_inc + 0.5, α · n_exc + 0.5)
```

where `α < 1` is a count-scaling parameter. Smaller `α` → wider intervals.

### Support-Adaptive Calibration

`α` is learned per support bin via **leave-one-gene-out cross-validation** on PacBio long-read ground truth:

| Support bin | α (median) | CI coverage | Median CI width |
|-------------|-----------|-------------|-----------------|
| <20         | 0.0096    | 100.0%      | 1.000           |
| 20-49       | 0.0096    | 93.8%       | 1.000           |
| 50-99       | 0.0094    | 94.9%       | 0.980           |
| 100-249     | 0.0087    | 92.9%       | 0.950           |
| 250+        | 0.0058    | 95.1%       | 0.513           |

### Confidence Tiers (QKI benchmark)

- **Supported**: rMATS FDR ≤ 0.05, |ΔPSI| ≥ 0.1, posterior probability above threshold
- **High-confidence**: CI for ΔPSI excludes zero
- **Near-strict**: probability criterion met, CI criterion not

---

## Key Benchmark Results

### 1. PacBio Cross-Validation (204 events, 15 genes)

| Metric | BRAID v2 | Legacy Poisson |
|--------|---------|----------------|
| CI coverage | **94.6%** | 72.5% |
| Correlation (r) | 0.831 | 0.831 |
| R² | 0.691 | 0.691 |
| Confident accuracy | **100%** (10/10) | 90.9% (33) |

### 2. QKI RT-PCR Benchmark (80 validated + 80 null control)

| Tier | Validated | Null Control | FPR |
|------|-----------|-------------|-----|
| rMATS significant | 44 (55%) | 0 | 0% |
| + BRAID supported | 14 (18%) | 0 | 0% |
| + BRAID high-conf | 10 (13%) | 0 | 0% |

**Zero false positives across ALL tiers.**

### 3. GM12878 Multi-Replicate (32,771 isoforms)

| Component | Median σ |
|-----------|---------|
| Biological (between-rep) | 7.73 |
| Sampling (within-rep) | 4.08 |
| Ratio | 1.9× |

Biological variance dominates → multi-replicate mode essential.

---

## Key Files for Understanding the Code

| File | Purpose | Lines |
|------|---------|-------|
| `braid/target/psi_bootstrap.py` | **Core**: PSI inference, BRAIDConfig, event proposal, evidence counting | ~680 |
| `braid/target/rmats_bootstrap.py` | rMATS output parser + BRAID CI | ~235 |
| `braid/target/multi_replicate_bootstrap.py` | Multi-rep variance decomposition | ~440 |
| `braid/target/stringtie_bootstrap.py` | StringTie isoform bootstrap | ~300 |
| `braid/cli.py` | CLI (assemble, analyze, target, denovo, dashboard) | 1200 |
| `benchmarks/qki_rmats_benchmark.py` | QKI RT-PCR benchmark driver | ~1800 |
| `benchmarks/rtpcr_benchmark.py` | PacBio cross-validation benchmark | ~1250 |

---

## How to Install and Test

```bash
# Clone
git clone https://github.com/kangk1204/BRAID.git
cd BRAID

# Install
pip install -e ".[dev]"

# Test (443 tests, ~5 seconds, no external data)
python -m pytest tests/ -v

# Demo (generates interactive HTML)
python demo/run_demo.py
# → Open demo/results/demo_report.html in browser

# CLI
braid --version   # 0.1.0
braid --help      # 7 subcommands
```

---

## How to Reproduce Benchmark Results

### Prerequisites
```bash
conda install -c bioconda samtools hisat2 stringtie rmats gffcompare
pip install -e ".[benchmark]"
```

### Data locations (on dev machine)
```
/home/keunsoo/projects/23_rna-seq_assembler/real_benchmark/
├── bam/
│   ├── SRR387661.bam            # K562 RNA-seq (15GB)
│   ├── GM12878_ENCFF550SET.nochr.bam  # GM12878 Rep2 (19GB)
│   ├── GM12878_Rep1.nochr.bam   # GM12878 Rep1 (23GB)
│   └── IMR90_ENCFF560TMJ.nochr.bam   # IMR90 (17GB)
├── annotation/
│   └── gencode.v38.nochr.gtf    # GENCODE v38
├── reference/grch38/
│   └── genome.fa                # GRCh38 FASTA
└── rtpcr_benchmark/qki/
    ├── ctrl_rep1.bam            # QKI control rep1
    ├── ctrl_rep2.bam            # QKI control rep2
    ├── qki_kd.bam               # QKI knockdown
    └── rmats_output/            # rMATS results
```

### Run benchmarks
```bash
# PacBio validation
python benchmarks/rtpcr_benchmark.py

# QKI RT-PCR
python benchmarks/qki_rmats_benchmark.py

# Multi-replicate
bash benchmarks/run_multirep_benchmark.sh
```

---

## Manuscript

- **LaTeX source**: `paper/braid.tex`
- **Compiled PDF**: `paper/braid.pdf` (180KB, 5 figures)
- **Compile**: `cd paper && tectonic braid.tex`
- **Target journal**: Bioinformatics (Oxford)

### Figures

| # | File | Content |
|---|------|---------|
| 1 | `fig1_workflow` | BRAID pipeline: Input → Processing → Output |
| 2 | `fig2_pacbio_validation` | PacBio scatter (r=0.831) + CI coverage by support + CI width |
| 3 | `fig3_qki_benchmark` | QKI tier bars (44→14→10, null=0) + filtering funnel |
| 4 | `fig4_calibration_multirep` | Legacy vs BRAID calibration + GM12878 variance decomposition |
| 5 | `fig5_positioning` | Before/after BRAID: event prioritization |

Each figure available in 3 formats: `paper/figures/figN_name.{pdf,png,jpg}` (300 DPI).

Regenerate: `python paper/generate_figures.py`

---

## Architecture Decisions

1. **Complement, not replace**: BRAID adds CI to rMATS/StringTie output, not a new caller.
2. **Conservative tiering**: Zero FPR on null controls by design; low recall is acceptable.
3. **Overdispersed posterior**: Beta(α·n + 0.5, α·n + 0.5) with α << 1 accounts for RNA-seq overdispersion.
4. **Leave-one-gene-out**: Prevents overfitting calibration to specific genes.
5. **Multi-replicate**: Bio variance ≈ 2× sampling variance; single-sample CI is conservative fallback.
6. **Annotation-guided SE**: Use GENCODE exon models to propose SE events; de novo fallback when annotation unavailable.

---

## Known Limitations

- Confident events are few (10/80 in QKI) — conservative by design.
- Calibration trained on A3SS/A5SS (15 genes). SE/MXE/RI calibration assumed, not directly validated.
- De novo SE detection limited (1/80 without rMATS). Use rMATS integration.
- Overdispersed posterior is heuristic; fully generative model could improve.

---

## Environment

- Python 3.10+, NumPy, SciPy, pysam, scikit-learn, numba
- Optional: PyTorch (ML features), CuPy (GPU), Streamlit (dashboard)
- Dev: WSL2, AMD Ryzen 9 7940HS, 16GB RAM
- 443 tests, all passing, ~5 seconds

---

## Contact

Keunsoo Kang — kangk1204@gmail.com
