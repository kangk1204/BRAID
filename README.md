# BRAID

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Live CI status badge (ruff + pytest are defined in .github/workflows/ci.yml).
     Add once the public repository URL is finalized and the repo is public, so the
     badge image actually resolves for unauthenticated viewers:
[![CI](https://github.com/kangk1204/BRAID/actions/workflows/ci.yml/badge.svg)](https://github.com/kangk1204/BRAID/actions/workflows/ci.yml) -->

**BRAID adds calibrated PSI / ΔPSI intervals and confidence tiers to rMATS
outputs, and can wrap MAJIQ, SUPPA2, and betAS differential tables through
`braid filter`.**

BRAID is a post-processing uncertainty layer. Its primary PSI and differential
modes read existing rMATS event/count tables, preserve the rMATS event
definitions and group convention, and add calibrated intervals, uncertainty
tiers, and reproducibility-oriented summary fields. The caller-agnostic
`braid filter` path normalizes rMATS, MAJIQ, SUPPA2, and betAS differential
outputs into the same calibrated ΔPSI reporting contract. The supported public
analysis surface is intentionally narrow:

- `braid differential` (short alias: `braid diff`): two-group ΔPSI intervals and confidence tiers.
- `braid psi`: single-condition per-event PSI intervals (multi-replicate mode reports rMATS sample_1 only; use single-sample mode for sample_2).
- `braid filter`: calibrate any caller (rMATS/MAJIQ/SUPPA2/betAS) into TSV + Excel + figure.
- `braid example`: no-data smoke demo.
- `braid doctor`: install and optional-tool diagnostics.

Experimental or legacy commands may remain in the package for development and
regression testing, but they are not the documented user-facing analysis path.

---

## Which command should I use?

| Your question | Command |
|---|---|
| What changed between two groups (from rMATS)? | `braid differential` |
| What is the PSI in one condition? | `braid psi` |
| I already have MAJIQ / SUPPA2 / betAS output | `braid filter` |
| I want an Excel + figure report (any caller) | `braid filter` |

`braid differential` is the **main rMATS workflow**: it bootstraps a posterior
ΔPSI from the junction counts, using rMATS per-replicate count vectors when
multiple complete samples are present, and writes the most detailed
rMATS-specific table.
`braid filter` is a **cross-caller reporting layer**: it puts rMATS, MAJIQ,
SUPPA2, and betAS on one common schema with an Excel workbook and a figure.

> **`braid differential` vs `braid filter --caller rmats`.** Both add a calibrated
> interval and confidence tier to an rMATS two-group result, share the canonical
> columns `dpsi`, `ci_low`, `ci_high`, `reliable`, `caller_significant`, `tier`, and
> share the one `confidence_tier` definition, so clear events get the **same tier**
> (e.g. a caller-significant event whose interval still crosses 0 is
> `caller-significant-only` in either command). They are **not identical, by
> design**, in the point estimate: `differential` reports a
> **posterior-sampled** ΔPSI (`dpsi`) plus rMATS-specific detail
> (`prob_large_effect`, `ctrl_psi`/`treat_psi`, per-group support), whereas
> `filter` reports each caller's **native** point estimate — for rMATS the raw
> `IncLevelDifference`, which `differential` keeps separately as `rmats_dpsi` — in
> the unified cross-caller schema with Excel + figure. Because the two paths center
> on slightly different estimates and interval widths, a **borderline** event can
> occasionally land in a different tier (the gap is the low-support Jeffreys-Beta
> shrinkage; high-support/clear events agree). For rMATS, treat `differential` as
> the **canonical** command; use `filter` to put several callers in one report, or
> to report rMATS's native estimate in the cross-caller schema.

---

## Prerequisites

BRAID is a Python 3.10+ command-line tool. Before installing, make sure you have:

| Tool | Why it is needed | Check it |
|------|------------------|----------|
| **Python 3.10+** | runs BRAID | `python --version` |
| **pip** | installs BRAID | `python -m pip --version` |
| **git** | `pip` uses it to download BRAID from GitHub | `git --version` |

If `python --version` reports nothing or a version below 3.10, install Python
first. For scientific use the simplest route is **Miniforge/Miniconda**
(https://github.com/conda-forge/miniforge), which ships Python 3.10+ and `pip`
together; after installing, run `conda create -n braid python=3.11` then
`conda activate braid`. Alternatives: the official installer at
https://www.python.org/downloads/, or your system package manager
(`sudo apt install python3 python3-pip` on Debian/Ubuntu, `brew install python`
on macOS). `pip` ships with Python; if `python -m pip --version` fails, run
`python -m ensurepip --upgrade`.

If `git --version` fails, install git the same way — `conda install git`,
`sudo apt install git` (Debian/Ubuntu), `brew install git` (macOS), or the
installer at https://git-scm.com/downloads — then run the install below.

## Quickstart

Install BRAID into an **isolated environment**. This is required on recent
Debian/Ubuntu, where installing into the system Python fails with
`error: externally-managed-environment` (PEP 668), and the conda route also pins
a Python version that has prebuilt wheels for every dependency:

```bash
# recommended: conda / Miniforge (python 3.11 so all dependency wheels exist)
conda create -n braid python=3.11 -y && conda activate braid
pip install "git+https://github.com/kangk1204/BRAID.git"
braid example
```

No conda? Use a virtual environment instead:

```bash
python3 -m venv ~/braid-venv && source ~/braid-venv/bin/activate
pip install "git+https://github.com/kangk1204/BRAID.git"
braid example
```

`pip` uses git to clone the repository for you, so there is nothing to download
by hand. For a command-line-only install,
`pipx install "git+https://github.com/kangk1204/BRAID.git"` also works and puts
`braid` on your PATH. If pip refuses to install or a dependency tries to build
from source, see [Troubleshooting installation](#troubleshooting-installation).

> **Coming back later?** With the conda or `venv` routes above, the environment is
> *created* once but must be *activated* in every new terminal session before the
> `braid` command is on your PATH. Each time you open a new terminal, activate it
> first, then run your `braid` commands:
>
> ```bash
> conda activate braid                 # if you used conda
> # or
> source ~/braid-venv/bin/activate     # if you used a venv
> braid --version                      # confirms the environment is active
> ```
>
> A `braid: command not found` error almost always means the environment is not
> activated. (The `pipx` route instead puts `braid` on your PATH permanently, so it
> needs no activation.)

`braid example` synthesizes a tiny rMATS-like dataset and runs the calibrated
differential path:

```text
  gene             ΔPSI  95% calibrated interval tier
  ------------------------------------------------------------
  STRONG_UP        0.60  [+0.25, +0.95]            high-confidence
  STRONG_DOWN     -0.70  [-1.00, -0.35]            high-confidence
  NO_CHANGE        0.00  [-0.35, +0.35]            not-significant
  LOW_COVERAGE     0.33  [-0.11, +0.78]            caller-significant-only
```

Run BRAID on **your own** rMATS output — replace `/path/to/rmats_output/` with the
directory rMATS actually wrote (the folder containing `SE.MATS.JC.txt`, etc.):

```bash
braid differential --rmats-dir /path/to/rmats_output/ -o differential.tsv
braid psi --rmats-dir /path/to/rmats_output/ -o psi.tsv
braid doctor
```

No rMATS output yet? Try BRAID on bundled data with no files of your own —
`braid example` (the synthetic differential run shown above), or
`braid filter --caller rmats --example`.

`--rmats-dir` must exist and contain at least one supported rMATS event table
such as `SE.MATS.JC.txt`, `SE.MATS.JunctionCountOnly.txt`, or
`SE.MATS.JCEC.txt`. A missing or empty directory is reported as an input error
(`rMATS directory not found: ...`) rather than as a successful empty analysis, so
the `rMATS_output/` placeholder in older snippets will fail until you point it at
a real path.

BRAID reads the rMATS tables **as-is — no reformatting**. From each event table it
uses the native columns `IJC_SAMPLE_1/2`, `SJC_SAMPLE_1/2` (inclusion/skipping
counts, replicates comma-separated), `IncFormLen`/`SkipFormLen`, `FDR`, and
`IncLevelDifference`; the other files in the folder (`fromGTF.*`, `*.raw.input.*`,
`summary.txt`, `b1.txt`/`b2.txt`) are ignored. Leave the per-replicate
comma-separated counts intact so the replicate-aware model can use them, and run
rMATS with control as `--b1` (= `sample_1`) so the `ctrl_psi`/ΔPSI signs are labelled
as intended.

### ΔPSI direction (`--swap-groups`)

By default `braid differential` reports **ΔPSI = sample_1 − sample_2 = control −
treatment**, exactly matching rMATS `IncLevelDifference` (so `dpsi` and `rmats_dpsi`
carry the same sign). The mapping is fixed entirely by how rMATS was run
(`--b1`/`--b2`), because the counts come from the rMATS tables.

If your ground-truth/RT-PCR convention runs the other way — or you simply want
treatment − control — pass `--swap-groups`:

```bash
# default: dpsi = sample_1 - sample_2 (= rMATS IncLevelDifference)
braid differential --rmats-dir rMATS_output/ -o differential.tsv

# reversed: dpsi = sample_2 - sample_1 (treatment - control)
braid differential --rmats-dir rMATS_output/ -o differential.tsv --swap-groups
```

Under `--swap-groups`, control becomes rMATS `sample_2`, the `ctrl_psi`/`treat_psi` and
`ctrl_support`/`treat_support` columns swap accordingly, and the reported `rmats_dpsi`
is **negated** so it stays sign-consistent with the swapped `dpsi`. Magnitude-based
fields are unchanged: `reliable` (interval excludes 0), the confidence `tier`, and the
FDR-based `caller_significant` flag are all direction-agnostic. This is the same
orientation the head-to-head benchmark applies via `event_counts(swap_groups=...)` for
datasets whose RT-PCR truth runs opposite to the rMATS `--b1`/`--b2` order (e.g.
TRA2/GSE59335 and circadian/GSE54651, where `--b1` is control but the RT-PCR table
reports knockdown − control).

---

## What BRAID Adds

- **Calibrated ΔPSI intervals** for two-group rMATS comparisons and supported
  caller differential tables.
- **Confidence tiers**, shared by `braid differential` and `braid filter` through
  one `confidence_tier` decision rule. Clear rMATS events agree; borderline
  events can differ because the two commands may center intervals on different
  estimates:
  `high-confidence`, `supported`, `caller-significant-only`, `not-significant`,
  or (when a caller reports no significance at all) `not-reliable`.
- **Calibrated PSI intervals** for single-condition rMATS event summaries.
- **Distribution-shift warnings** when the support regime is outside the shipped
  calibrator's effective range.
- **Strict group-convention messaging** so output labels match the rMATS
  `sample_1` / `sample_2` contract.

BRAID does not replace rMATS, MAJIQ, SUPPA2, or betAS event calling and should
not be read as a superior point-estimator claim. Its central claim is calibrated
uncertainty on top of existing event/count or differential-splicing tables.

---

## Calibration

BRAID ships two default calibrators:

- `braid differential` uses `differential_dpsi_conformal.json`, fit on real
  RT-PCR residuals (`n=162`), for calibrated ΔPSI intervals.
- `braid psi` uses `default_psi_conformal.json`, fit on a synthetic
  beta-binomial reference, for single-condition PSI intervals.
- `braid filter` also uses the ΔPSI calibrator by default. For a finite-sample
  guarantee on a specific non-rMATS caller, refit a calibrator on matched
  caller-vs-truth residuals and pass it with `--calibration`.

The shipped calibrators are 95% calibrators (`alpha=0.05`), and in every
conformal mode the interval alpha comes from the calibrator JSON, not from a CLI
flag — to change it, pass a custom calibrator fitted at the desired alpha with
`--calibration`. The per-command flags differ:

- `braid differential` and single-sample `braid psi` accept `--confidence`, but
  it resizes **only** the legacy `--no-conformal` percentile fallback; it does
  not refit or resize the default conformal intervals.
- `braid filter` always uses the conformal calibrator and has **no**
  `--confidence` or `--no-conformal` flag. To get intervals at a different alpha
  (or for a specific non-rMATS caller), pass a custom `--calibration`.
- Multi-replicate `braid psi` uses `--confidence` for the across-replicate
  combination step.

Use a custom calibrator or disable conformal calibration explicitly:

```bash
braid differential --rmats-dir rMATS_output/ --calibration my_dpsi_calibrator.json
braid psi --rmats-dir rMATS_output/ --calibration my_psi_calibrator.json
braid differential --rmats-dir rMATS_output/ --no-conformal
```

---

## Differential Mode

```bash
braid differential --rmats-dir rMATS_output/ -o differential.tsv
```

PSI counts come from the rMATS tables, not from any BAM files. BRAID maps:

- control = rMATS `sample_1` / rMATS `--b1`
- treatment = rMATS `sample_2` / rMATS `--b2`
- ΔPSI = `sample_1 - sample_2`, matching rMATS `IncLevelDifference`

Run rMATS with control samples as `--b1` if you want BRAID's `ctrl_psi` and
`treat_psi` output labels to match that biological interpretation.

When rMATS reports two or more complete, **informative** replicate count pairs in
both groups (replicates with no reads, `inc + exc == 0`, are dropped),
`braid differential --differential-model auto` uses those per-replicate vectors:
each posterior draw resamples biological replicates within each group, draws
replicate-level PSI values, averages them within group (equal weight per
replicate), and reports ΔPSI as `sample_1 - sample_2`. This propagates
between-replicate (biological) variance, which the pooled estimator ignores. Rows
with fewer than two informative replicates or with incomplete vectors fall back to
the legacy pooled-count posterior. Use `--differential-model rep` to require the
replicate model (it fails fast on any event lacking two informative pairs), or
`--differential-model sum` to force the pooled group-sum posterior for every row.
`--no-replicate-aware` remains a compatibility alias for `--differential-model
sum`. `--replicates` below controls the number of posterior draws, not the
biological replicate count.

Each output row records the estimator actually used (`differential_model` =
`rep` or `sum`) and the informative replicate count per group
(`ctrl_replicates` / `treat_replicates`). The reported `ctrl_psi` / `treat_psi` are
the per-group posterior means of the *same* draws that produce `dpsi`, so
`dpsi == ctrl_psi - treat_psi` holds for either estimator (the equal-weight
replicate mean and the pooled mean each stay self-consistent with their own ΔPSI).

Common options:

```bash
braid differential --rmats-dir rMATS_output/ \
  --confidence 0.95 \
  --effect-cutoff 0.10 \
  --fdr 0.05 \
  --min-support 20 \
  --differential-model auto \
  --replicates 500 \
  --seed 42 \
  -o differential.tsv
```

---

## PSI Mode

```bash
braid psi --rmats-dir rMATS_output/ -o psi.tsv
```

In rMATS mode, BAM paths are not used to re-extract counts. If supplied, they
serve as replicate-count intent/provenance for the per-replicate rMATS vectors:

```bash
braid psi --bam rep1.bam rep2.bam --rmats-dir rMATS_output/ -o psi.tsv
```

By default, BRAID fails on any mismatch between the number of supplied BAM paths
and the rMATS per-replicate count vector length. Use
`--allow-replicate-fallback` only when you intentionally want the degraded
group-sum fallback.

Common options:

```bash
braid psi --rmats-dir rMATS_output/ \
  --confidence 0.95 \
  --min-support 10 \
  --replicates 500 \
  --seed 42 \
  -o psi.tsv
```

---

## Caller-Agnostic Filter (`braid filter`)

BRAID is a confidence **layer**, not a caller — so you can run it on top of
**rMATS, MAJIQ, SUPPA2, or betAS**. It reads that tool's differential output, adds
a calibrated 95% ΔPSI interval, a `reliable` flag (interval excludes 0), and a
confidence `tier`, then writes a TSV, an Excel workbook, and a ready-to-use figure.

```bash
pip install -e ".[report]"   # enables the Excel + figure output (openpyxl, matplotlib)
```

### Input — one line per caller (runnable on the bundled samples)

A tiny ready-to-run sample for every caller is **bundled inside the installed
package**, so `--example` works straight after installing BRAID
(`pip install -e .`) without locating any data files yourself. The same files
live under `braid/examples/filter/` in the repo.

| Caller | What you give BRAID | Try it now (post-install) |
|---|---|---|
| rMATS  | the rMATS output **directory** (`*.MATS.JC.txt`) | `braid filter --caller rmats --example -o demo` |
| MAJIQ  | voila `deltapsi.tsv` | `braid filter --caller majiq --example -o demo` |
| SUPPA2 | `diffSplice` `.dpsi` table | `braid filter --caller suppa2 --example -o demo` |
| betAS  | betAS differential TSV | `braid filter --caller betas --example -o demo` |

On your own data, drop `--example` and pass the caller's output path, e.g.
`braid filter --caller majiq path/to/deltapsi.tsv -o out`. BRAID consumes each
tool's **native output as-is** (no reformatting): rMATS count tables, MAJIQ
`E(dPSI)`/probability, SUPPA2 ΔPSI/p-value, or betAS ΔPSI — per-caller support
handling is detailed below.

For caller exports whose contrast direction or headers are ambiguous, make the
parser contract explicit:

```bash
# reverse the caller's ΔPSI direction before BRAID calibration
braid filter --caller suppa2 diff.dpsi -o out --flip-sign

# choose one SUPPA2 multi-contrast pair: Ctrl-KD_dPSI + Ctrl-KD_p-val
braid filter --caller suppa2 diff.dpsi -o out --contrast Ctrl-KD

# override nonstandard headers exactly; -v prints the detected columns and drops
braid filter --caller betas betas.tsv -o out -v \
  --dpsi-column delta_custom --fdr-column q_custom \
  --event-id-column event_custom --gene-column gene_custom
```

Support and uncertainty handling is caller-specific:

- rMATS carries junction counts, so BRAID uses support bins and a closed-form
  ΔPSI sampling standard deviation for depth-robust intervals.
- MAJIQ carries ΔPSI and probability; when the table also contains
  `num_reads`/`reads` or `std`/`sd` columns, BRAID uses them for support binning
  or depth-robust intervals. The bundled MAJIQ sample includes `num_reads`.
- SUPPA2 reports ΔPSI/p-value but no countable read support, so BRAID uses the
  pooled global quantile and marks `support_known=no`.
- betAS reports ΔPSI and can include a native Beta interval and/or support
  column. A native lower/upper interval gives a sampling standard deviation;
  a `support`/`reads` column enables support binning. The bundled betAS sample
  has a native interval but no support column, so it uses the pooled global
  quantile.

The MAJIQ command above may log a calibration-applicability warning for the tiny
toy sample, then prints:

```
BRAID filter (majiq): 3 events -> demo.tsv
  Excel : demo.xlsx
  Figure: demo.png / .pdf / .svg
  reliable (interval excludes 0): 1
    high-confidence: 1
    caller-significant-only: 1
    not-significant: 1
```

and `demo.tsv` ranks the three events:

| tier | gene | dpsi | ci_low | ci_high |
|---|---:|---:|---:|---:|
| high-confidence | BIGSWITCH | 0.55 | 0.21 | 0.89 |
| caller-significant-only | MODEST | 0.18 | -0.10 | 0.46 |
| not-significant | NULLEVENT | -0.03 | -0.31 | 0.25 |

MODEST is the teaching case: MAJIQ called it significant, but its calibrated interval
still crosses 0, so BRAID drops it to *caller-significant-only* — validate it only
after BIGSWITCH. All four sample files contain the same three illustrative events.

### Output — `out.tsv`, `out.xlsx`, `out.png/.pdf/.svg`

Every event gets `dpsi` (the point estimate), `ci_low`/`ci_high` (the calibrated
95% interval), `reliable` (does the interval exclude 0?), `support_known`, and a
`tier`. The table is sorted by calibrated distance from zero, so reliable
non-zero events appear before intervals that cross zero. The figure shows those
top events with their intervals (panel A) and the tier counts (panel B).

### How to read the result — the tiers

| Tier | Meaning | What to do |
|---|---|---|
| **high-confidence** | caller flagged it **and** the calibrated interval excludes 0 with a real effect | validate first |
| **supported** | interval excludes 0 (real effect), but the caller did not flag it | good secondary candidate |
| **caller-significant-only** | the caller called it significant, but the calibrated interval still **crosses 0** | be cautious — low priority |
| **not-significant / not-reliable** | neither flags it | skip |

The tier that earns BRAID its keep is **caller-significant-only**: events your tool
reported as significant that BRAID's orthogonal-truth-calibrated interval says are
*not yet reliable* — exactly where calibrated confidence changes the decision.

> The shipped ΔPSI calibrator is fit on real RT-PCR residuals from the bundled
> calibration scope. For a coverage **guarantee on your own data or a specific
> non-rMATS caller**, validate a few dozen representative events by RT-PCR
> (include some negatives) and refit, then pass
> `--calibration my_calibrator.json`.

---

## Typical Upstream Workflow

For the rMATS PSI and differential modes, BRAID starts after rMATS has produced
event tables:

```bash
# Example upstream rMATS convention:
# control = --b1, treatment = --b2
rmats.py --b1 ctrl.txt --b2 treat.txt --gtf annotation.gtf \
  --od rMATS_output/ -t paired --readLength 150

braid differential --rmats-dir rMATS_output/ -o differential.tsv
braid psi --rmats-dir rMATS_output/ -o psi.tsv
```

BRAID reads the rMATS junction count and IncLevel columns, applies the calibrated
uncertainty layer, and writes TSV outputs. For caller-agnostic filtering, start
from the caller's native differential table (`rMATS`, MAJIQ `deltapsi.tsv`,
SUPPA2 `diffSplice .dpsi`, or betAS TSV). The documented analysis path does not
require raw FASTQ input or transcript assembly.

> **Advanced (BAM assembly).** BRAID also ships a `braid assemble` path that
> builds transcripts directly from a BAM. It is kept for backward compatibility
> and hidden from the default `braid --help`, so most users never need it. On
> very large BAMs it accepts `--fast-read-count`, which takes the BAM index
> `mapped` total for the startup read count instead of scanning the file once to
> apply the flag/quality filters exactly. It is a fast upper bound (it
> over-counts secondary/supplementary reads and ignores MAPQ), so the default
> stays exact and BRAID logs a warning when the approximation is used.

---

## Installation

### Supported platforms

| Platform | Status | Notes |
|----------|--------|-------|
| **Linux** (Ubuntu 20.04+) | Supported | Primary development/CI platform. |
| **macOS** (Intel and Apple Silicon M1/M2/M3) | Supported | `pysam`, `numba`, NumPy/SciPy ship native arm64 wheels; use the same pip command. |
| **Windows 11** | Via WSL2 | `pysam` (the BAM reader) has no native-Windows build. Install [WSL2](https://learn.microsoft.com/windows/wsl/install) with Ubuntu, then follow the Linux instructions inside WSL. |

If a `pip` wheel is unavailable for your system and a dependency tries to build
from source, install the scientific stack through conda first
(`conda install -c conda-forge -c bioconda pysam numba numpy scipy scikit-learn`)
and then run the BRAID install.

### Core Install

Most users only need the one-line install from [Quickstart](#quickstart):

```bash
pip install "git+https://github.com/kangk1204/BRAID.git"
```

To work on the BRAID source instead (editable install), first create and activate
an environment as in [Quickstart](#quickstart), then clone and install from the
checkout — `-e .` means "install the project in the current directory, in editable
mode":

```bash
conda create -n braid python=3.11 -y && conda activate braid   # or a venv
git clone https://github.com/kangk1204/BRAID.git
cd BRAID
python -m pip install -e .
braid doctor
```

### Troubleshooting installation

- **`error: externally-managed-environment` (PEP 668).** Your system Python is
  managed by the OS (common on recent Debian/Ubuntu). Do not use
  `--break-system-packages`; install into a conda env or a `venv` as shown in
  [Quickstart](#quickstart), or use `pipx`.
- **A dependency tries to build from source / no matching wheel.** Usually a very
  new Python (e.g. 3.13/3.14) without prebuilt `numba`, `llvmlite`, or `pysam`
  wheels yet — create the environment with a supported Python
  (`conda create -n braid python=3.11`). As a fallback, install the scientific
  stack from conda first
  (`conda install -c conda-forge -c bioconda pysam numba numpy scipy scikit-learn`).
- **`ensurepip is not available` when creating a venv.** Install the venv module:
  `sudo apt install python3-venv python3-full`.
- **`braid: command not found` after install.** The environment is not active —
  re-run `conda activate braid` (or `source <venv>/bin/activate`), or call it by
  its full path (`<env>/bin/braid`).

### Benchmark / Paper Environment

Use the benchmark environment if you need to reproduce repository benchmarks or
paper artifacts:

```bash
conda env create -f environment-benchmark.yml
conda activate braid-benchmark
braid doctor --strict
```

`braid doctor --strict` checks core packages plus optional benchmark packages
and external tools. Missing optional tools do not block the core rMATS
post-processing workflow.

### Paper benchmark reproduction

The package install examples above are enough to run BRAID and the bundled toy
examples, but the manuscript benchmarks need the repository checkout plus the
prepared public-benchmark data tree. The large `data/public_benchmarks/` inputs
(BAMs, rMATS outputs, MAJIQ outputs, and curated truth tables) are not stored in
GitHub. With the prepared reviewer data bundle available locally, set
`BRAID_BENCH` to that bundle and link the expected paths:

```bash
git clone https://github.com/kangk1204/BRAID.git
cd BRAID

conda env create -f environment-benchmark.yml
conda activate braid-benchmark
python -m pip install -e ".[benchmark,report]"

export BRAID_BENCH=/path/to/braid_bench
mkdir -p data/public_benchmarks benchmarks/application_dm1
ln -sfnT "${BRAID_BENCH}/data/public_benchmarks/GSE59335" data/public_benchmarks/GSE59335
ln -sfnT "${BRAID_BENCH}/data/public_benchmarks/GSE54651" data/public_benchmarks/GSE54651
ln -sfnT "${BRAID_BENCH}/data/public_benchmarks/SRS354082" data/public_benchmarks/SRS354082
ln -sfnT "${BRAID_BENCH}/data/public_benchmarks/meta" data/public_benchmarks/meta
ln -sfnT "${BRAID_BENCH}/GSE64357_esrp" data/public_benchmarks/GSE64357_esrp
ln -sfnT "${BRAID_BENCH}/application_dm1/rmats" benchmarks/application_dm1/rmats
```

Then run the paper-facing analyses:

```bash
cd benchmarks/headtohead
python comprehensive_benchmark.py
python stat_review.py
python majiq_coverage.py
python pooled_coverage.py
python variance_decomposition.py
cd ../..

python benchmarks/application_dm1/run_dm1_application.py
python benchmarks/application_esrp/run_esrp_application.py
```

Expected headline checks are:

- head-to-head common set: `n=139`; coverage MAJIQ `0.518`, betAS `0.734`,
  rMATS `0.633`, BRAID-conformal `0.971`; BRAID interval score `0.720`.
- full rMATS-matched set: `n=196`; BRAID-conformal coverage `0.949`.
- DM1 application: `144,975` events after support filtering, `967`
  rMATS-significant large-effect events, `68` BRAID high-confidence events.
- ESRP application: `27,169` events after support filtering and `9/12`
  testable anchor genes recovered as BRAID high-confidence.

The DM1 script defaults to `--differential-model sum` because that is the
manuscript reproduction model. The ESRP script defaults to
`--differential-model auto`; both scripts record the model in their summary JSON.
For full raw-data regeneration and MAJIQ/betAS details, see
`benchmarks/headtohead/README.md`.

### rMATS

BRAID consumes rMATS output. If you need to generate those tables, install rMATS
through Bioconda or your local workflow manager:

```bash
conda install -c conda-forge -c bioconda rmats
rmats.py --help
```

---

## Development

Install test dependencies:

```bash
python -m pip install -e ".[dev,dashboard]"
```

The dashboard extra is included because in-repo dashboard tests import
`pandas`/`plotly` helpers during collection. This does not make the dashboard a
documented analysis mode.

Include optional ML tests when needed:

```bash
python -m pip install -e ".[dev,dashboard,ml]"
```

Run checks:

```bash
python -m pytest tests/ -v
ruff check braid/ tests/
```

CI gates the package and tests. `benchmarks/` and `demo/` are reproduction and
exploration code, not the enforced lint scope.

---

## Project Layout

- `braid/commands/`: CLI entry points for the supported commands and internal
  development commands.
- `braid/target/`: rMATS parsing, PSI/ΔPSI bootstrap, and conformal calibration.
- `benchmarks/`: benchmark scripts and paper-result helpers.
- `tests/`: regression tests.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
