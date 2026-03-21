# BRAID

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**BRAID adds calibrated confidence intervals to StringTie and rMATS results.**

Given aligned RNA-seq BAMs, BRAID wraps existing transcript assemblers and
splicing tools and augments every reported isoform or splicing event with
bootstrap-based confidence intervals, uncertainty tiers, and reproducibility
scores.  The goal is not to replace StringTie or rMATS but to tell you *how
much you should trust* their output.

---

## The Four Modes of `braid run`

BRAID auto-detects what you want based on the inputs you provide:

```bash
# Mode 1 — Assemble: BAM only -> de novo assembly + CI
braid run sample1.bam sample2.bam -o results/

# Mode 2 — Score: BAM + StringTie GTF -> isoform-level CI
braid run *.bam --stringtie merged.gtf -o results/

# Mode 3 — PSI: BAM + rMATS output -> event-level PSI CI
braid run *.bam --rmats rMATS_output/ -o results/

# Mode 4 — Differential: two groups + rMATS -> differential splicing + tiers
braid run --ctrl c1.bam c2.bam --treat kd.bam --rmats rMATS_output/ -o results/
```

| Mode | Inputs | What BRAID adds |
|---|---|---|
| **assemble** | BAM files | Transcript assembly with per-isoform bootstrap CI |
| **score** | BAM + StringTie GTF | Confidence intervals on StringTie isoform abundances |
| **psi** | BAM + rMATS dir | Calibrated PSI posterior intervals per splicing event |
| **differential** | ctrl BAMs + treat BAMs + rMATS | Tiered differential splicing calls with CI |

---

## Complete Pipeline Example

A typical workflow from raw data to confidence-annotated results:

```bash
# 1. Align reads (e.g. HISAT2)
hisat2 -x genome -1 R1.fq.gz -2 R2.fq.gz --dta | samtools sort -o aligned.bam
samtools index aligned.bam

# 2. Assemble transcripts with StringTie
stringtie aligned.bam -G gencode.v38.gtf -o stringtie.gtf -p 8

# 3. Detect differential splicing with rMATS (two-group comparison)
rmats.py --b1 ctrl.txt --b2 treat.txt --gtf gencode.v38.gtf \
    --od rmats_out/ -t paired --readLength 150

# 4. Add BRAID confidence intervals
braid run --ctrl ctrl.bam --treat treat.bam \
    --rmats rmats_out/ --stringtie stringtie.gtf -o braid_results/
```

BRAID reads the existing StringTie and rMATS output, resamples junction
counts via Poisson bootstrap, and writes augmented results with CI bounds,
uncertainty tiers, and reproducibility flags.

---

## What BRAID Provides

- **Per-isoform confidence intervals** on StringTie transcript abundances.
- **Calibrated PSI posteriors** for every rMATS splicing event.
- **Tiered differential calls** (high / medium / low confidence) with
  bootstrap-derived FDR calibration.
- **Multi-replicate variance decomposition** separating biological from
  sampling uncertainty.
- **De novo assembly path** when no upstream tool output is available.
- **Optional GPU acceleration** for heavier workloads; CPU fallback by default.

---

## Project Layout

- `benchmarks/`: benchmark scripts and result summaries
- `paper/`: manuscript source
- `tests/`: regression tests

---

## Choose Your Install

Use the smallest option that matches your goal:

- `Core CPU install`: run BRAID bootstrap and scoring workflows.
- `BRAID benchmark / paper install`: reproduce QKI, PacBio, and paper figures.
- `GPU install`: optional Ubuntu path for CUDA 12 systems.

All commands below assume you are already in the repository root.

## Installation

### 1. Core CPU Install

```bash
conda env create -f environment.yml
conda activate braid
braid doctor
```

This is the recommended path for most users on macOS or Ubuntu.

### 2. BRAID Benchmark / Paper Install

If you want to run the BRAID benchmarks, regenerate the paper numbers, or use
the QKI/PacBio workflows, use the benchmark environment:

```bash
conda env create -f environment-benchmark.yml
conda activate braid-benchmark
braid doctor --strict
```

`braid doctor --strict` checks:

- core Python packages
- BRAID benchmark Python packages such as `pyliftover`
- external tools such as `samtools`, `hisat2`, `gffcompare`, `stringtie`, and
  `rmats.py`

### 3. Optional GPU Install

Use this only on Ubuntu systems with an NVIDIA GPU and CUDA 12.

```bash
conda env create -f environment-gpu.yml
conda activate braid-gpu
braid doctor --gpu
```

`braid doctor --gpu` verifies the optional CUDA Python stack. If you also
need the BRAID benchmark tools, combine both checks:

```bash
braid doctor --strict --gpu
```

### 4. Manual pip Install

```bash
python -m pip install -e .
braid doctor
```

If you need the BRAID Python extras without the full conda benchmark
environment:

```bash
python -m pip install -e ".[benchmark]"
```

## StringTie and rMATS Installation

If you already created `environment-benchmark.yml`, you can skip this section:
`stringtie` and `rmats` are already included there.

If you are starting from an existing conda environment, install the BRAID
benchmark tools like this:

```bash
conda activate braid
mamba install -c conda-forge -c bioconda samtools hisat2 gffcompare stringtie rmats
```

If you do not have `mamba`, the same command works with `conda install`:

```bash
conda activate braid
conda install -c conda-forge -c bioconda samtools hisat2 gffcompare stringtie rmats
```

You can also install just one tool at a time:

```bash
mamba install -c conda-forge -c bioconda stringtie
mamba install -c conda-forge -c bioconda rmats
```

Quick checks:

```bash
stringtie --version
rmats.py --help
braid doctor --strict
```

Bioconda currently publishes both `stringtie` and `rmats`, so the commands above
work on modern conda setups without a manual source build. The official package
pages are:

- StringTie: <https://bioconda.github.io/recipes/stringtie/README.html>
- rMATS: <https://bioconda.github.io/recipes/rmats/README.html>

If `rmats.py` still does not appear on `PATH`, open a new shell and activate the
conda environment again. BRAID itself does not need `stringtie` or `rmats` for
core confidence-interval workflows; they are only required for benchmark and
paper reproduction.


### Optional: Development Install

```bash
python -m pip install -e ".[dev]"
```

---

## Quick Start

### Minimal usage

```bash
braid run sample.bam -o results/
```

Auto-detects mode (assemble) and writes transcripts with bootstrap CI.

### Add confidence to StringTie output

```bash
braid run sample.bam --stringtie transcripts.gtf -o results/
```

### Add confidence to rMATS output

```bash
braid run sample.bam --rmats rmats_output/ -o results/
```

### Differential splicing with tiers

```bash
braid run --ctrl c1.bam c2.bam --treat kd1.bam kd2.bam --rmats rmats_output/ -o results/
```

### De novo assembly (legacy mode)

```bash
braid assemble aligned.bam -o transcripts.gtf -r genome.fa
```

### GPU-accelerated run

```bash
braid assemble aligned.bam --backend gpu -t 8 -o transcripts.gtf
```

### Verify the install

```bash
braid doctor
braid doctor --strict
braid doctor --gpu
```

### Run the demo (no BAM needed)

```bash
python demo/run_demo.py
```

This generates 200 synthetic AS events, runs BRAID bootstrap, and produces an
**interactive HTML report** at `demo/results/demo_report.html`. Open it in any
browser to explore:

- PSI scatter plot (BRAID vs ground truth, colored by event type)
- CI width vs read support
- CI coverage by support bin
- Forest plot with confidence intervals

### Run the full test suite

```bash
python -m pytest tests/ -v
```

443 tests, all in-memory (no external data needed).

---

## CLI Reference

```
usage: braid [-h] {run,psi,differential,assemble,analyze,denovo,dashboard,doctor,target,fq} ...
```

### `braid run` (recommended entry point)

```
braid run [BAMs...] [--stringtie GTF] [--rmats DIR]
          [--ctrl BAMs...] [--treat BAMs...]
          [-o OUTPUT_DIR] [-t THREADS]
```

Auto-detects mode from provided inputs. See the four modes above.

### `braid assemble` (direct assembly)

| Argument | Default | Description |
|---|---|---|
| `bam` | *(required)* | Input BAM file (coordinate-sorted, indexed) |
| `-o`, `--output` | `braid_output.gtf` | Output file path |
| `-f`, `--format` | `gtf` | Output format: `gtf` or `gff3` |
| `-r`, `--reference` | `None` | Reference genome FASTA (with `.fai` index) |
| `--backend` | `auto` | Compute backend: `auto`, `cpu`, or `gpu` |
| `-t`, `--threads` | `1` | Number of threads |
| `-q`, `--min-mapq` | `0` | Minimum mapping quality |
| `-j`, `--min-junction-support` | `2` | Minimum junction support reads |
| `-c`, `--min-coverage` | `1.0` | Minimum transcript coverage |
| `-s`, `--min-score` | `0.3` | Minimum transcript score (0-1) |
| `--max-intron-length` | `500000` | Maximum intron length (bp) |
| `--no-safe-paths` | `False` | Disable safe path decomposition |
| `--no-ml-scoring` | `False` | Disable ML scoring, use heuristic |
| `--model` | `None` | Path to pre-trained scoring model (`.joblib`) |
| `--chromosomes` | `None` | Comma-separated chromosome list |
| `-v`, `--verbose` | `False` | Verbose (DEBUG) logging |
| `--version` | | Show version and exit |

---

## Benchmarks

Benchmark scripts live in `benchmarks/`. The paper figures and evaluation
artifacts are generated from there. All benchmark scripts accept a
`BRAID_DATA_DIR` environment variable (default: `real_benchmark`) to locate
BAM files and reference data.

---

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Linting

```bash
ruff check .
```

### Install in Development Mode

```bash
pip install -e ".[dev]"
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
