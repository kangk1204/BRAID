# BRAID — Bootstrap Reliability Assessment for Isoform Detection

**A confidence quantification layer for RNA-seq alternative splicing analysis.**

BRAID adds calibrated confidence intervals to existing AS analysis tools (StringTie, rMATS), enabling researchers to distinguish high-confidence events from uncertain ones — without replacing their current workflow.

## Key Results

### PacBio Cross-Validation (204 AS events, 15 genes)

| Metric | Value |
|--------|-------|
| CI Coverage | **94.6%** (target: 95%) |
| Correlation (r) | **0.831** |
| R² | 0.691 |
| Confident Accuracy | **100%** (10/10) |
| Calibration | Leave-one-gene-out CV, support-adaptive |

### QKI RT-PCR Benchmark (80 validated + 80 null control SE events)

| Tier | Validated (80) | Null Control (80) | FPR |
|------|---------------|-------------------|-----|
| rMATS significant | **44** (55%) | 0 | 0% |
| + BRAID supported | **14** (18%) | 0 | 0% |
| + BRAID high-confidence | **10** (13%) | 0 | 0% |

**Zero false positives across all tiers.** BRAID adds confidence stratification on top of rMATS detection.

### GM12878 Multi-Replicate (32,771 isoforms, Rep1 + Rep2)

| Variance Source | Median |
|----------------|--------|
| Biological σ | **7.73** |
| Sampling σ | 4.08 |
| Total CV | 0.359 |

Biological variance dominates sampling variance by ~2x — **multi-replicate mode is essential for calibrated CI**.

## Installation

```bash
pip install braid-rna
# or
pip install -e ".[dev]"
```

Dependencies: numpy, scipy, pysam. Optional: torch (ML features).

## Usage

### 1. Single-Sample Confidence (StringTie)

```bash
braid score --stringtie output.gtf --bam sample.bam --annotation gencode.gtf
```

### 2. Multi-Replicate Mode (Recommended)

```bash
braid score --stringtie output.gtf --bam rep1.bam rep2.bam rep3.bam
```

### 3. rMATS Integration

```bash
# Run rMATS first
rmats.py --b1 ctrl.txt --b2 kd.txt --gtf annotation.gtf -t paired ...

# Add BRAID confidence tiers
braid rmats --rmats-dir rMATS_output/ --replicates 500
```

### 4. Target Gene Analysis

```python
from braid.core.psi_bootstrap import compute_psi_from_junctions

results = compute_psi_from_junctions(
    bam_path="sample.bam",
    chrom="7", start=55000000, end=55200000,
    gene="EZH2",
    annotation_gtf="gencode.v38.gtf",
)

for r in results:
    print(f"{r.event_id}: PSI={r.psi:.1%} CI=[{r.ci_low:.1%},{r.ci_high:.1%}] "
          f"confident={r.is_confident}")
```

## Architecture

```
Input:
  StringTie GTF + BAMs  ──→  Isoform confidence (CV, CI, presence)
  rMATS output + BAMs   ──→  AS event PSI CI + confidence tiers
  BAMs only             ──→  De novo event detection + CI

Modes:
  Single-sample    →  Sampling uncertainty (conservative)
  Multi-replicate  →  Biological + Sampling (calibrated)

Output:
  Per-isoform:   weight, CV, CI, presence rate
  Per-AS-event:  PSI, CI, support tier, confidence flag
```

## Design Principles

1. **Complements, doesn't replace** existing tools (StringTie, rMATS)
2. **Uncertainty-aware ranking** — CI, CV, support-aware confidence
3. **Conservative tiering** — zero false positives on null controls
4. **Works with or without replicates** — better with them

## Citation

If you use BRAID, please cite:
> Kang, K. (2026). BRAID: Bootstrap Reliability Assessment for Isoform Detection. *Bioinformatics* (submitted).

## License

MIT
