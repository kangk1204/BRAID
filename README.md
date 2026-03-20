# BRAID

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**Calibrated confidence intervals for RNA-seq splicing analysis.**

This repository contains two layers:

- `BRAID`: the public tool name, paper, benchmarks, and confidence workflow.
- `rapidsplice`: the legacy internal Python module path kept for backward
  compatibility.

BRAID runs on macOS and Ubuntu. BRAID benchmarking works best in a conda
environment because it depends on external bioinformatics tools such as
`samtools`, `stringtie`, and `rmats.py`.

---

## What It Does

- **Transcript assembly** from coordinate-sorted BAM files.
- **Targeted splicing analysis** with PSI and confidence intervals.
- **Optional GPU acceleration** for heavier workloads.
- **CPU fallback by default**, so a laptop install is enough for basic use.
- **Benchmark and paper helpers** for reproducible evaluation.

---

## Project Layout

- `rapidsplice/`: legacy internal module path used by the `braid` CLI
- `benchmarks/`: benchmark scripts and result summaries
- `paper/`: manuscript source
- `tests/`: regression tests

---

## Choose Your Install

Use the smallest option that matches your goal:

- `Core CPU install`: run the assembler and core Python workflows.
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
basic assembly; they are only required for benchmark and paper reproduction.

The public CLI is `braid`. The legacy alias `rapidsplice` still works if you
already have old scripts.

### Optional: Development Install

```bash
python -m pip install -e ".[dev]"
```

---

## Quick Start

### Minimal usage

```bash
braid assemble aligned.bam
```

This reads a coordinate-sorted, indexed BAM file and writes assembled transcripts
to `braid_output.gtf`.

### With reference genome and custom output

```bash
braid assemble aligned.bam -o transcripts.gtf -r genome.fa
```

Providing a reference FASTA (with `.fai` index) enables splice-site motif
validation for improved precision.

### GPU-accelerated run

```bash
braid assemble aligned.bam --backend gpu -t 8 -o transcripts.gtf
```

### Process specific chromosomes

```bash
braid assemble aligned.bam --chromosomes chr1,chr2,chr3 -v
```

### Verify the install

```bash
braid doctor
braid doctor --strict
braid doctor --gpu
```

---

## CLI Reference

```
usage: braid [-h] [-o OUTPUT] [-f {gtf,gff3}] [-r REFERENCE]
                   [--backend {auto,cpu,gpu}] [-t THREADS] [-q MIN_MAPQ]
                   [-j MIN_JUNCTION_SUPPORT] [-c MIN_COVERAGE] [-s MIN_SCORE]
                   [--max-intron-length MAX_INTRON_LENGTH] [--no-safe-paths]
                   [--no-ml-scoring] [--model MODEL] [--chromosomes CHROMOSOMES]
                   [-v] [--version]
                   bam
```

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

## How It Works

BRAID's assembly path scans BAM files, builds splice graphs for each locus, decomposes
compatible paths into transcripts, and scores the results with a compact
feature-based filter. If you only want to run the assembler, you can stop at the
Install and Quick Start sections above.

---

## Benchmarks

Benchmark scripts live in `benchmarks/`. The paper figures and evaluation
artifacts are generated from there.

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
