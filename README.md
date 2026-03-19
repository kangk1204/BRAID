# BRAID

**Bootstrap Resampling for Assembly Isoform Dependability**

A post-assembly tool that adds per-isoform bootstrap confidence intervals to RNA-seq transcript assembly output. Works with StringTie, Scallop, or any assembler producing GTF output.

## Key Results

| Dataset | Baseline Precision | + BRAID CV≤0.05 | Improvement |
|---------|-------------------|-----------------|-------------|
| K562 (43,522 isoforms) | 39.3% | **81.3%** | +42pp |
| GM12878 (47,818 isoforms) | 42.6% | **95.2%** | +53pp |
| IMR90 (37,538 isoforms) | 45.5% | **80.1%** | +35pp |

**PacBio long-read independent validation:** CV≤0.05 → 82.1% validated (isoforms), CV≤0.1 → 98.2% validated (AS events)

## How It Works

```
StringTie GTF → Parse isoform structures
              → Extract junction read counts (from BAM or coverage)
              → Poisson bootstrap resampling (R=200)
              → Re-solve NNLS per replicate
              → Per-isoform CV, CI, presence rate
```

Bootstrap CV predicts whether an assembled isoform is correct:
- **CV ≤ 0.05**: 81-95% precision (high confidence)
- **CV ≤ 0.1**: 70-88% precision
- **CV ≤ 0.2**: 59-73% precision (best F1)

## Installation

```bash
pip install -e .
```

Requirements: Python 3.10+, numpy, scipy, pysam

## Usage

### 1. Score StringTie isoforms with bootstrap CI

```bash
# BAM-based (uses actual junction read counts)
braid score --stringtie output.gtf --bam sample.bam --reference genome.fa

# BAM-free (uses StringTie coverage, 6x faster)
braid score --stringtie output.gtf
```

### 2. Single-sample PSI with bootstrap CI

```bash
braid psi --bam sample.bam --gene TP53 --gtf gencode.gtf
```

### 3. Targeted single-gene assembly + CI

```bash
braid target sample.bam --gene TP53 --gtf gencode.gtf --reference genome.fa
```

### 4. Interactive dashboard

```bash
braid dashboard
# Opens browser at http://localhost:8501
```

## Output Example

```
TP53 Alternative Splicing PSI + Bootstrap CI:

Event                    PSI    CI_low  CI_high    CV   Inc   Exc  Confident
A3SS:7670715-7673534   86.5%   74.3%   96.9%   0.07    32     5     Y
A3SS:7670715-7673206    8.1%    0.0%   17.3%   0.53     3    34     Y
A3SS:7670715-7673222    5.4%    0.0%   14.3%   0.70     2    35     Y
```

## Comparison with Existing Tools

| Feature | BRAID | Kallisto/Salmon | rMATS | SUPPA2 |
|---------|-------|----------------|-------|--------|
| Bootstrap CI for assembly | **Yes** | No (quant only) | No | No |
| Single-sample PSI CI | **Yes** | N/A | No | No |
| Needs replicates | **No** | N/A | Yes | Yes |
| Novel isoforms | **Yes** | No (ref only) | No | No |
| Assembler-agnostic | **Yes** | N/A | N/A | N/A |

## Citation

If you use BRAID, please cite:

> Kim, K. (2026). BRAID: Quantifying uncertainty in transcript isoform detection
> and alternative splicing from RNA-seq via junction bootstrap.

## License

MIT License

## RT-PCR Validation

Validated against 80 RT-PCR confirmed exon skipping events from the QKI paper (Zong et al., PLOS Genetics 2014):

| BRAID Prediction | RT-PCR Validated | RT-PCR Failed | PPV |
|-----------------|-----------------|--------------|-----|
| Confident | 20 | 6 | **76.9%** |
| Uncertain | 26 | 8 | — |

Three independent validations converge: GENCODE (81%), PacBio (82%), **RT-PCR (77%)**.

## ENCODE RBP Knockdown Validation

TARDBP knockdown in K562 (ENCODE, 87,665 AS events):

| ENCODE Classification | BRAID CV median | BRAID Confident |
|----------------------|----------------|-----------------|
| Significant (FDR<0.05) | **0.172** | **50%** |
| Non-significant | 0.288 | 36% |

BRAID assigns lower CV (higher confidence) to experimentally validated differential splicing events.
