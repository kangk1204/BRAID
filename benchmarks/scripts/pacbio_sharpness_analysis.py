#!/usr/bin/env python3
"""PacBio CI coverage sharpness analysis.

Computes conditional coverage by CI width, random interval baselines,
and generates sharpness comparison figures.
"""

import json
import numpy as np
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────
INPUT = Path("/home/keunsoo/projects/24_BRAID/benchmarks/results/rtpcr_benchmark.json")
OUTPUT_JSON = Path("/home/keunsoo/projects/24_BRAID/benchmarks/results/pacbio_sharpness.json")
FIG_BASE = Path("/home/keunsoo/projects/24_BRAID/paper/figures/fig_pacbio_sharpness")

with open(INPUT) as f:
    data = json.load(f)

bins_data = data["pacbio_psi"]["support_bin_summary"]

# Ordered bin names
bin_order = ["<20", "20-49", "50-99", "100-249", "250+"]

# ── 1. CONDITIONAL COVERAGE BY CI WIDTH ────────────────────────────────────
# We only have bin-level medians, not per-event widths.
# Estimate: events with CI width < threshold are predominantly in the 250+ bin
# (width 0.513) and possibly 100-249 (width 0.950).
# For width < X: count events whose median_ci_width < X.

print("=" * 70)
print("1. CONDITIONAL COVERAGE BY CI WIDTH (bin-level estimation)")
print("=" * 70)

thresholds = [0.3, 0.5, 0.8]
for thresh in thresholds:
    # Bins whose median width is below the threshold
    n_events = 0
    covered_events = 0.0
    qualifying_bins = []
    for bn in bin_order:
        bd = bins_data[bn]
        if bd["median_ci_width"] < thresh:
            n_events += bd["n_events"]
            covered_events += bd["n_events"] * bd["ci_coverage"]
            qualifying_bins.append(bn)

    if n_events > 0:
        cond_cov = covered_events / n_events
        print(f"  CI width < {thresh:.1f}: {n_events} events, coverage = {cond_cov:.3f} "
              f"(bins: {qualifying_bins})")
    else:
        print(f"  CI width < {thresh:.1f}: 0 events (no bins have median width < {thresh})")

# Additional: the 250+ bin has median 0.513, so it straddles the 0.5 threshold.
# Roughly half its events may be below 0.5. Report this nuance.
print()
print("  NOTE: 250+ bin has median width 0.513. ~half its events have width < 0.5")
print("  No per-event data available; these are bin-median-based estimates.")
print()

# ── 2. RANDOM INTERVAL BASELINE ───────────────────────────────────────────
# For each bin: generate random intervals of the same median width,
# centered on uniform-random points in [0,1], clipped to [0,1].
# Coverage = P(true PSI in random interval).
# With uniform PSI ~ U(0,1): E[coverage] = width (when width <= 1).
# But we approximate using Monte Carlo with uniform PSI.

print("=" * 70)
print("2. RANDOM INTERVAL BASELINE (Monte Carlo, 100K samples)")
print("=" * 70)

np.random.seed(42)
N_MC = 100_000

random_coverages = {}
for bn in bin_order:
    w = bins_data[bn]["median_ci_width"]
    w = min(w, 1.0)  # clip

    # Generate random intervals: center ~ U(0,1), width = w, clipped to [0,1]
    centers = np.random.uniform(0, 1, N_MC)
    lo = np.clip(centers - w / 2, 0, 1)
    hi = np.clip(centers + w / 2, 0, 1)

    # True PSI ~ U(0,1)
    true_psi = np.random.uniform(0, 1, N_MC)

    # Coverage: fraction where true_psi falls in [lo, hi]
    covered = np.mean((true_psi >= lo) & (true_psi <= hi))
    random_coverages[bn] = covered

    # Analytical approximation for uniform: E[cov] = w - w^2/4 (for w<=1)
    # (accounting for clipping at boundaries)
    analytical = w - w**2 / 4 if w <= 1 else 1.0

    print(f"  Bin {bn:>8s}: width={w:.4f}, random_cov={covered:.4f} "
          f"(analytical~{analytical:.4f}), BRAID_cov={bins_data[bn]['ci_coverage']:.4f}")

print()

# ── 3. SHARPNESS TABLE ────────────────────────────────────────────────────
print("=" * 70)
print("3. SHARPNESS TABLE")
print("=" * 70)

header = f"{'Bin':>8s} | {'N':>4s} | {'BRAID Cov':>10s} | {'Random Cov':>10s} | {'BRAID Width':>11s} | {'Confident':>9s}"
print(header)
print("-" * len(header))

sharpness_rows = []
for bn in bin_order:
    bd = bins_data[bn]
    n = bd["n_events"]
    braid_cov = bd["ci_coverage"]
    rand_cov = random_coverages[bn]
    width = bd["median_ci_width"]
    conf = bd["confident_count"]

    row = {
        "bin": bn,
        "n_events": n,
        "braid_coverage": round(braid_cov, 4),
        "random_coverage": round(rand_cov, 4),
        "coverage_lift": round(braid_cov - rand_cov, 4),
        "braid_median_ci_width": round(width, 4),
        "confident_count": conf,
        "confident_accuracy": bd["confident_accuracy"],
    }
    sharpness_rows.append(row)

    print(f"{bn:>8s} | {n:>4d} | {braid_cov:>10.4f} | {rand_cov:>10.4f} | {width:>11.4f} | {conf:>9d}")

print()

# Overall
total_n = sum(bins_data[bn]["n_events"] for bn in bin_order)
total_braid_cov = sum(bins_data[bn]["n_events"] * bins_data[bn]["ci_coverage"] for bn in bin_order) / total_n
total_rand_cov = sum(bins_data[bn]["n_events"] * random_coverages[bn] for bn in bin_order) / total_n
total_width = sum(bins_data[bn]["n_events"] * bins_data[bn]["median_ci_width"] for bn in bin_order) / total_n
total_conf = sum(bins_data[bn]["confident_count"] for bn in bin_order)

print(f"{'Overall':>8s} | {total_n:>4d} | {total_braid_cov:>10.4f} | {total_rand_cov:>10.4f} | {total_width:>11.4f} | {total_conf:>9d}")
print()

# Coverage lift analysis
print("COVERAGE LIFT (BRAID - Random):")
for row in sharpness_rows:
    lift = row["coverage_lift"]
    print(f"  {row['bin']:>8s}: {lift:+.4f} {'(informative)' if lift > 0.05 else '(trivial - wide CI)'}")

print()
print("KEY INSIGHT:")
print("  - Low-support bins (<20, 20-49, 50-99, 100-249): CIs are nearly [0,1].")
print("    Random intervals of same width achieve ~same coverage. NOT informative.")
print(f"  - 250+ bin: BRAID coverage {bins_data['250+']['ci_coverage']:.3f} vs "
      f"random {random_coverages['250+']:.3f} = "
      f"{bins_data['250+']['ci_coverage'] - random_coverages['250+']:+.3f} lift")
print(f"    Width 0.513 => intervals are genuinely narrower and STILL achieve 95% coverage.")
print(f"    This is the only bin where CIs are meaningfully sharp.")
print()

# ── 4. SAVE RESULTS ───────────────────────────────────────────────────────
results = {
    "analysis": "PacBio CI Sharpness Analysis",
    "source": str(INPUT),
    "n_events_total": total_n,
    "conditional_coverage_by_width": {
        f"width_lt_{t}": {
            "n_events": sum(bins_data[bn]["n_events"] for bn in bin_order if bins_data[bn]["median_ci_width"] < t),
            "coverage": round(
                sum(bins_data[bn]["n_events"] * bins_data[bn]["ci_coverage"]
                    for bn in bin_order if bins_data[bn]["median_ci_width"] < t)
                / max(1, sum(bins_data[bn]["n_events"] for bn in bin_order if bins_data[bn]["median_ci_width"] < t)),
                4
            ),
        }
        for t in thresholds
    },
    "sharpness_table": sharpness_rows,
    "overall": {
        "braid_coverage": round(total_braid_cov, 4),
        "random_coverage": round(total_rand_cov, 4),
        "coverage_lift": round(total_braid_cov - total_rand_cov, 4),
        "weighted_mean_width": round(total_width, 4),
    },
    "interpretation": {
        "bins_with_informative_CIs": ["250+"],
        "bins_with_trivial_CIs": ["<20", "20-49", "50-99", "100-249"],
        "explanation": (
            "Only the 250+ support bin (102/204 events, 50%) has CIs narrow enough "
            "to be informative (median width 0.513). Lower-support bins have CIs "
            "spanning ~95-100% of [0,1], matching random interval coverage."
        ),
    },
}

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_JSON}")
print()

# ── 5. FIGURE ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18/2.54, 7/2.54), dpi=300)

x = np.arange(len(bin_order))
bar_w = 0.35

# Panel (a): BRAID vs Random coverage
braid_covs = [bins_data[bn]["ci_coverage"] for bn in bin_order]
rand_covs = [random_coverages[bn] for bn in bin_order]

bars1 = ax1.bar(x - bar_w/2, braid_covs, bar_w, label="BRAID", color="#2171B5", edgecolor="white", linewidth=0.5)
bars2 = ax1.bar(x + bar_w/2, rand_covs, bar_w, label="Random baseline", color="#BDBDBD", edgecolor="white", linewidth=0.5)
ax1.axhline(0.95, color="#E31A1C", linestyle="--", linewidth=0.8, label="95% target")
ax1.set_xticks(x)
ax1.set_xticklabels(bin_order, rotation=0)
ax1.set_xlabel("Junction support bin")
ax1.set_ylabel("CI coverage")
ax1.set_ylim(0.5, 1.05)
ax1.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="#CCCCCC")
ax1.set_title("(a) Coverage: BRAID vs random baseline")

# Add event counts on top
for i, bn in enumerate(bin_order):
    n = bins_data[bn]["n_events"]
    ax1.text(i, max(braid_covs[i], rand_covs[i]) + 0.015, f"n={n}",
             ha="center", va="bottom", fontsize=6, color="#555555")

# Panel (b): CI width by support bin
widths = [bins_data[bn]["median_ci_width"] for bn in bin_order]
colors = ["#BDBDBD" if w > 0.8 else "#FDB863" if w > 0.5 else "#2171B5" for w in widths]

ax2.bar(x, widths, 0.6, color=colors, edgecolor="white", linewidth=0.5)
ax2.axhline(0.95, color="#E31A1C", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.text(len(bin_order) - 0.5, 0.96, "trivial (width~1.0)", fontsize=6, color="#E31A1C",
         ha="right", va="bottom")
ax2.set_xticks(x)
ax2.set_xticklabels(bin_order, rotation=0)
ax2.set_xlabel("Junction support bin")
ax2.set_ylabel("Median CI width")
ax2.set_ylim(0, 1.15)
ax2.set_title("(b) CI width by support bin")

# Annotate the 250+ bar
ax2.annotate(f"{widths[-1]:.3f}", xy=(x[-1], widths[-1]), xytext=(x[-1] - 0.5, widths[-1] + 0.08),
             fontsize=7, color="#2171B5", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="#2171B5", lw=0.8))

plt.tight_layout(pad=0.5)

# Save in three formats
for ext in ["pdf", "png", "jpg"]:
    out_path = f"{FIG_BASE}.{ext}"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {out_path}")

plt.close()

# ── FINAL SUMMARY ──────────────────────────────────────────────────────────
print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Total events: {total_n}")
print(f"  Overall BRAID coverage: {total_braid_cov:.4f}")
print(f"  Overall random baseline: {total_rand_cov:.4f}")
print(f"  Overall lift: {total_braid_cov - total_rand_cov:+.4f}")
print()
print(f"  250+ bin (n=102): BRAID={bins_data['250+']['ci_coverage']:.3f}, "
      f"Random={random_coverages['250+']:.3f}, "
      f"Width={bins_data['250+']['median_ci_width']:.3f}")
print(f"  --> Only bin with genuinely informative CIs (width < 1.0)")
print(f"  --> Coverage lift = {bins_data['250+']['ci_coverage'] - random_coverages['250+']:+.3f}")
print()
print(f"  Low-support bins (n=102): Median widths 0.95-1.00 => trivially cover everything")
print(f"  Random intervals of same width achieve similar coverage")
print()
print("DONE.")
