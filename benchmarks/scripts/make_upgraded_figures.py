"""Upgraded renderers (design pilot) for Fig 3 and Fig 4.

Reads the COMMITTED result JSONs (no recompute, no fabricated values) and renders
the shared design system from figstyle. CI ranges are drawn faithfully:
  * Fig 3A: asymmetric Wilson 95% CI on the + BRAID coverage bars.
  * Fig 4A: held-out coverage band (coverage_lo .. coverage_hi) on the learning curve.
"""
# ruff: noqa: I001
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "benchmarks"))
import figstyle as F  # noqa: E402

F.apply()
RES = os.path.join(_REPO, "benchmarks/results")
OUT = os.path.join(_REPO, "outputs/figures/upgraded_preview")
os.makedirs(OUT, exist_ok=True)


def _save(fig, stem):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"),
                    dpi=320 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {OUT}/{stem}.png/.svg/.pdf")


def fig3():
    ab = json.load(open(os.path.join(RES, "recalibration_ablation.json")))["methods"]
    sup = json.load(open(os.path.join(RES, "suppa2_recalibration.json")))["by_method"]["SUPPA2"]
    callers = ["MAJIQ", "betAS", "rMATS", "SUPPA2"]
    src = {
        "MAJIQ": ab["MAJIQ (real binary)"], "betAS": ab["betAS (real tool)"],
        "rMATS": ab["rMATS IncLevel t-CI"],
        "SUPPA2": {"raw_coverage": None, "raw_width": None,
                   "recal_coverage": sup["recalibrated_coverage"],
                   "recal_wilson": sup["wilson"], "recal_width": sup["mean_width"]},
    }
    fig = plt.figure(figsize=(11.4, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.26)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # Panel A: native (muted gray) vs + BRAID (hero blue), Wilson CI on + BRAID.
    F.nominal_guide(axA, 0.95, axis="y")
    xs = range(len(callers))
    w = 0.38
    for i, c in enumerate(callers):
        d = src[c]
        if d["raw_coverage"] is not None:
            axA.bar(i - w / 2, d["raw_coverage"], w, color=F.NATIVE,
                    edgecolor="white", lw=0.7, zorder=3)
        else:
            axA.text(i - w / 2, 0.27, "no native\ninterval", ha="center", va="center",
                     fontsize=6.6, color="#8A8D90", style="italic")
        cov, lo, hi = d["recal_coverage"], d["recal_wilson"][0], d["recal_wilson"][1]
        axA.bar(i + w / 2, cov, w, color=F.HERO, edgecolor="white", lw=0.7, zorder=3,
                yerr=[[cov - lo], [hi - cov]], capsize=3,
                error_kw=dict(ecolor=F.INK, elinewidth=1.1, zorder=5))
    axA.set_xticks(list(xs))
    axA.set_xticklabels(callers, fontsize=9)
    axA.set_ylim(0, 1.05)
    axA.set_ylabel("Coverage of RT-PCR ΔPSI", fontsize=9)
    axA.set_title("Every caller under-covers; + BRAID reaches nominal", fontsize=9.5,
                  loc="left", fontweight="semibold")
    axA.text(-0.4, 0.952, "nominal 0.95", color=F.NOMINAL, fontsize=7.2, va="bottom")
    axA.legend(handles=[
        Line2D([0], [0], marker="s", ls="", ms=8, color=F.NATIVE, label="native interval"),
        Line2D([0], [0], marker="s", ls="", ms=8, color=F.HERO, label="+ BRAID (95% CI)"),
    ], loc="lower left", frameon=False, fontsize=7.6, ncol=1, handletextpad=0.3,
        labelspacing=0.3)

    # Panel B: calibration lift -- native (open) -> + BRAID (filled), per caller.
    for c in callers:
        d = src[c]
        col = F.MUTED[c]
        x1, y1 = d["recal_width"], d["recal_coverage"]
        if d["raw_width"] is not None:
            x0, y0 = d["raw_width"], d["raw_coverage"]
            axB.annotate("", xy=(x1, y1), xytext=(x0, y0),
                         arrowprops=dict(arrowstyle="-|>", color=col, lw=1.8,
                                         shrinkA=5, shrinkB=7, alpha=0.9))
            axB.scatter(x0, y0, s=70, facecolor="white", edgecolor=col, lw=1.6, zorder=5)
            # label at the (well-separated) native end, below-left of the open marker
            axB.annotate(c, (x0, y0), xytext=(x0 - 0.006, y0 - 0.028), fontsize=8,
                         ha="right", va="top", color=F.INK)
        axB.scatter(x1, y1, s=120, color=col, edgecolor="white", lw=1.1, zorder=6)
    # SUPPA2 has no native end -> label its filled (+ BRAID) point on the right
    s2 = src["SUPPA2"]
    axB.annotate("SUPPA2", (s2["recal_width"], s2["recal_coverage"]),
                 xytext=(s2["recal_width"] + 0.012, s2["recal_coverage"]),
                 fontsize=8, ha="left", va="center", color=F.INK)
    F.nominal_guide(axB, 0.95, axis="y")
    axB.set_xlim(0.10, 0.78)
    axB.set_ylim(0.44, 1.02)
    axB.set_xlabel("Mean interval width", fontsize=9)
    axB.set_ylabel("Coverage of RT-PCR ΔPSI", fontsize=9)
    axB.set_title("The calibration lift  (open = native, filled = + BRAID)",
                  fontsize=9.5, loc="left", fontweight="semibold")

    fig.suptitle(
        "BRAID recalibration is caller-agnostic: it lifts every caller to nominal coverage",
        fontsize=11.5, fontweight="bold", y=1.03, x=0.01, ha="left")
    for ax, ltr in zip((axA, axB), "AB"):
        F.panel_letter(ax, ltr)
    _save(fig, "fig3_tool_vs_braid_upgraded")
    plt.close(fig)


def fig4():
    d = json.load(open(os.path.join(RES, "within_study_calibration.json")))
    tra = d["datasets"]["tra2"]["by_N"]
    sc = d["strategy_comparison"]["by_strategy"]
    fig = plt.figure(figsize=(11.4, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.28)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # Panel A: learning curve, held-out coverage vs N, with coverage band.
    Ns = [r["N"] for r in tra]
    rnd = [r["random"]["coverage"] for r in tra]
    rlo = [r["random"]["coverage_lo"] for r in tra]
    rhi = [r["random"]["coverage_hi"] for r in tra]
    top = [r["tophit"]["coverage"] for r in tra]
    F.nominal_guide(axA, 0.95, axis="y")
    axA.fill_between(Ns, rlo, rhi, color=F.HERO, alpha=0.13, zorder=2)
    axA.plot(Ns, rnd, "-o", color=F.HERO, lw=2.2, ms=6, zorder=5,
             label="representative (random)")
    axA.plot(Ns, top, "--s", color=F.MUTED["MAJIQ"], lw=1.8, ms=5, zorder=4,
             label="top-hit only")
    axA.set_xlabel("RT-PCR-labelled events (N)", fontsize=9)
    axA.set_ylabel("Held-out coverage", fontsize=9)
    axA.set_ylim(0.80, 1.01)
    axA.set_title("~20 representative labels reach nominal coverage", fontsize=9.5,
                  loc="left", fontweight="semibold")
    axA.text(Ns[0], 0.955, "nominal 0.95", color=F.NOMINAL, fontsize=7.2,
             ha="left", va="bottom")
    axA.legend(loc="lower right", frameon=False, fontsize=7.6)

    # Panel B: acquisition strategy at N=30 -- reliable recovered, precision+coverage.
    keys = ["top_hit", "random", "with_negatives_measured", "lowsignal_assumed0"]
    labels = ["Top-hit\nonly", "Random", "Repr. +\nneg.", "Low-signal\n(circular)"]
    fr = [sc[k]["frac_reliable"] * 100 for k in keys]
    prec = [sc[k]["reliable_precision"] for k in keys]
    cov = [sc[k]["coverage"] for k in keys]
    # hero = the recommended strategy (representative + negatives); circular muted+hatched
    cols = [F.NATIVE, F.MUTED["betAS"], F.HERO, F.NATIVE]
    for i, (v, col) in enumerate(zip(fr, cols)):
        hatch = "///" if keys[i] == "lowsignal_assumed0" else None
        axB.bar(i, v, 0.62, color=col, edgecolor="white", lw=0.8, zorder=3, hatch=hatch)
        ann = "recovers none" if v == 0 else f"prec {prec[i]:.2f}\ncov {cov[i]:.2f}"
        axB.text(i, v + 0.4, ann, ha="center", va="bottom", fontsize=6.8,
                 fontweight="bold" if keys[i] == "with_negatives_measured" else "normal",
                 color=F.HERO if keys[i] == "with_negatives_measured" else F.INK)
    axB.set_xticks(range(len(keys)))
    axB.set_xticklabels(labels, fontsize=7.8)
    axB.set_ylim(0, max(fr) + 4)
    axB.set_ylabel("Reliable events recovered (%)", fontsize=9)
    axB.set_title("Representative + negatives recovers the most (top-hit: none)",
                  fontsize=9.3, loc="left", fontweight="semibold")

    fig.suptitle(
        "Within-study calibration from a few dozen RT-PCR validations "
        "(TRA2, n=112; leakage-free)",
        fontsize=11, fontweight="bold", y=1.03, x=0.01, ha="left")
    for ax, ltr in zip((axA, axB), "AB"):
        F.panel_letter(ax, ltr)
    _save(fig, "fig4_within_study_upgraded")
    plt.close(fig)


def fig_s2():
    d = json.load(open(os.path.join(RES, "adaptive_conformal_eval.json")))
    fig = plt.figure(figsize=(13.0, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], wspace=0.34)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # A: per-event scale vs |residual| Spearman rho (all weak -> constant is right)
    hs = d["hetero_spearman"]
    feats = ["posterior_std", "rMATS_SE", "log_support", "psi_extremity"]
    flab = ["posterior SD", "rMATS SE", "log support", "PSI extremity"]
    vals = [hs[f] for f in feats]
    yb = range(len(feats))
    axA.axvline(0, color="#cccccc", lw=0.8)
    for thr in (-0.23, 0.23):
        axA.axvline(thr, color=F.NOMINAL, ls=(0, (3, 3)), lw=1.0)
    axA.barh(list(yb), vals, color=F.NATIVE, edgecolor="white", lw=0.7, zorder=3)
    axA.set_yticks(list(yb))
    axA.set_yticklabels(flab, fontsize=8.2)
    axA.set_xlim(-0.4, 0.4)
    axA.set_xlabel("Spearman ρ vs |RNA-seq − RT-PCR| residual", fontsize=8.5)
    axA.set_title("No read-derived scale predicts the residual", fontsize=9.3,
                  loc="left", fontweight="semibold")
    axA.text(0.97, 0.96, f"|ρ| ≤ 0.23\nRF difficulty R² = {d['rf_difficulty_cv_r2']:.2f}",
             transform=axA.transAxes, ha="right", va="top", fontsize=7.0, color=F.INK)

    # B: interval score by scaling rule, sorted; constant band is the hero (best)
    smap = {
        "ones (baseline, constant)": "constant band", "posterior_std": "posterior-SD",
        "rMATS_SE": "rMATS-SE", "max(pstd, rMATS_SE)": "max(pSD,SE)",
        "learned-linear (cross-fit)": "learned-linear", "learned-RF (cross-fit)": "learned-RF",
    }
    sc = d["scales"]
    items = sorted(((smap[k], sc[k]["iscore"]) for k in smap), key=lambda t: t[1], reverse=True)
    for i, (lab, v) in enumerate(items):
        hero = lab == "constant band"
        axB.barh(i, v, 0.66, color=F.HERO if hero else F.NATIVE, edgecolor="white",
                 lw=0.8, zorder=3)
        axB.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=7.8,
                 fontweight="bold" if hero else "normal", color=F.HERO if hero else F.INK)
    axB.set_yticks(range(len(items)))
    axB.set_yticklabels([t[0] for t in items], fontsize=8)
    for t, (lab, _) in zip(axB.get_yticklabels(), items):
        if lab == "constant band":
            t.set_fontweight("bold")
            t.set_color(F.HERO)
    axB.set_xlim(0, 1.65)
    axB.set_xlabel("Interval score  (↓ lower is better)", fontsize=8.5)
    axB.set_title("A single constant width is interval-score optimal", fontsize=9.3,
                  loc="left", fontweight="semibold")
    axB.spines["left"].set_visible(False)
    axB.tick_params(axis="y", length=0)

    # C: width vs coverage; constant band is sharp AND at nominal
    F.nominal_guide(axC, 0.95, axis="y")
    for k, lab in smap.items():
        hero = lab == "constant band"
        s = sc[k]
        axC.scatter(s["width"], s["coverage"], s=180 if hero else 70,
                    marker="*" if hero else "o", color=F.HERO if hero else F.NATIVE,
                    edgecolor="white", lw=1.0, zorder=6 if hero else 4)
    axC.annotate("constant\nband", (sc["ones (baseline, constant)"]["width"],
                 sc["ones (baseline, constant)"]["coverage"]),
                 xytext=(sc["ones (baseline, constant)"]["width"] - 0.03,
                         sc["ones (baseline, constant)"]["coverage"] - 0.012),
                 fontsize=7.6, ha="right", va="top", color=F.HERO, fontweight="bold")
    axC.set_xlim(0.5, 1.3)
    axC.set_ylim(0.93, 0.98)
    axC.set_xlabel("Mean interval width", fontsize=8.5)
    axC.set_ylabel("Coverage", fontsize=8.5)
    axC.set_title("Adaptive scalings only widen or lose coverage", fontsize=9.3,
                  loc="left", fontweight="semibold")

    fig.suptitle("Per-event adaptive widths do not beat a constant conformal band "
                 "(n = 196 rMATS-matched)",
                 fontsize=10.8, fontweight="bold", y=1.04, x=0.01, ha="left")
    for ax, ltr in zip((axA, axB, axC), "ABC"):
        F.panel_letter(ax, ltr)
    _save(fig, "supp_fig2_adaptive_scaling_upgraded")
    plt.close(fig)


def fig_s3():
    d = json.load(open(os.path.join(RES, "sgnex_conditional_eval.json")))
    types = ["SE", "A3SS", "A5SS"]
    const, comp = d["constant_support"]["by_type"], d["adaptive_composite"]["by_type"]
    fig = plt.figure(figsize=(13.0, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 0.95], wspace=0.34)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    x = range(len(types))
    w = 0.38

    # A: coverage by event type, constant vs composite, Wilson CI
    F.nominal_guide(axA, 0.95, axis="y")
    for j, (src, col, lab, off) in enumerate(
            [(const, F.NATIVE, "constant band", -w / 2),
             (comp, F.HERO, "composite Mondrian", w / 2)]):
        cov = [src[t]["coverage"] for t in types]
        lo = [src[t]["coverage"] - src[t]["wilson"][0] for t in types]
        hi = [src[t]["wilson"][1] - src[t]["coverage"] for t in types]
        axA.bar([i + off for i in x], cov, w, color=col, edgecolor="white", lw=0.7,
                zorder=3, yerr=[lo, hi], capsize=2.5,
                error_kw=dict(ecolor=F.INK, elinewidth=1.0), label=lab)
    axA.set_xticks(list(x))
    axA.set_xticklabels(types, fontsize=8.5)
    axA.set_ylim(0.90, 0.97)
    axA.set_ylabel("Held-out coverage", fontsize=8.5)
    axA.set_title("Composite restores A3SS/A5SS to nominal", fontsize=9.3,
                  loc="left", fontweight="semibold")
    axA.legend(loc="lower left", frameon=False, fontsize=7.4)

    # B: mean width by event type (composite widens only the under-covered bins)
    for j, (src, col, off) in enumerate([(const, F.NATIVE, -w / 2), (comp, F.HERO, w / 2)]):
        axB.bar([i + off for i in x], [src[t]["mean_width"] for t in types], w,
                color=col, edgecolor="white", lw=0.7, zorder=3)
    axB.set_xticks(list(x))
    axB.set_xticklabels(types, fontsize=8.5)
    axB.set_ylabel("Mean interval width", fontsize=8.5)
    axB.set_title("Width added only where needed", fontsize=9.3, loc="left",
                  fontweight="semibold")

    # C: pooled interval score, constant / event-type-only / composite
    isc = [("constant", d["constant_support"]["pooled_interval_score"], F.NATIVE),
           ("event-type\nonly", d["adaptive_event_type"]["pooled_interval_score"], F.NATIVE),
           ("composite", d["adaptive_composite"]["pooled_interval_score"], F.HERO)]
    for i, (lab, v, col) in enumerate(isc):
        axC.bar(i, v, 0.6, color=col, edgecolor="white", lw=0.8, zorder=3)
        axC.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=7.8,
                 fontweight="bold" if col == F.HERO else "normal")
    axC.set_xticks(range(3))
    axC.set_xticklabels([t[0] for t in isc], fontsize=8)
    axC.set_ylim(0, 0.58)
    axC.set_ylabel("Pooled interval score (↓)", fontsize=8.5)
    axC.set_title("Composite matches constant sharpness", fontsize=9.3, loc="left",
                  fontweight="semibold")

    fig.suptitle("A composite event-type×support Mondrian calibrator restores "
                 "conditional coverage at no sharpness cost (SG-NEx, n = 46,160)",
                 fontsize=10.4, fontweight="bold", y=1.04, x=0.01, ha="left")
    for ax, ltr in zip((axA, axB, axC), "ABC"):
        F.panel_letter(ax, ltr)
    _save(fig, "supp_fig3_sgnex_conditional_upgraded")
    plt.close(fig)


TIER = {"high-confidence": F.HERO, "supported": "#4A9B6E",
        "significant": "#C7A6BE", "not-significant": "#D7D9DB"}


def fig_s1():
    p = os.path.join(_REPO, "outputs/figures/manuscript/S1/s1_source_data.xlsx")
    rates = pd.read_excel(p, sheet_name="panel_A_rates")
    mcc = pd.read_excel(p, sheet_name="panel_B_mcc")
    conf = pd.read_excel(p, sheet_name="panel_CD_confusion")
    order = ["rmats_fdr", "posterior_effect", "braid_supported",
             "calibrated_ci_excludes_zero", "jeffreys_ci_excludes_zero"]
    rlab = {"rmats_fdr": "rMATS\nFDR<0.05", "posterior_effect": "Posterior\neffect",
            "braid_supported": "BRAID\nsupported", "calibrated_ci_excludes_zero": "Calib.\nCI≠0",
            "jeffreys_ci_excludes_zero": "Jeffreys\nCI≠0"}

    def col(rid):
        return F.HERO if rid == "braid_supported" else F.NATIVE

    fig = plt.figure(figsize=(12.4, 4.3))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.9, 1.0], wspace=0.34)
    axA, axB, axCD = (fig.add_subplot(gs[0, i]) for i in range(3))

    # A: sensitivity & FPR per rule, with Wilson CI (sens up, FPR down as light bars)
    xs = range(len(order))
    w = 0.38
    for rid, xi in zip(order, xs):
        sens = rates[(rates.rule_id == rid) & (rates.metric == "Sensitivity")].iloc[0]
        fpr = rates[(rates.rule_id == rid) & (rates.metric == "False-positive rate")].iloc[0]
        axA.bar(xi - w / 2, sens.estimate, w, color=col(rid), edgecolor="white", lw=0.6,
                yerr=[[sens.estimate - sens.wilson_low], [sens.wilson_high - sens.estimate]],
                capsize=2, error_kw=dict(ecolor=F.INK, elinewidth=0.9), zorder=3)
        axA.bar(xi + w / 2, fpr.estimate, w, color=col(rid), edgecolor="white", lw=0.6,
                alpha=0.4, hatch="////",
                yerr=[[fpr.estimate - fpr.wilson_low], [fpr.wilson_high - fpr.estimate]],
                capsize=2, error_kw=dict(ecolor=F.INK, elinewidth=0.9), zorder=3)
    axA.set_xticks(list(xs))
    axA.set_xticklabels([rlab[r] for r in order], fontsize=7.2)
    axA.set_ylim(0, 1.0)
    axA.set_ylabel("Rate (95% Wilson CI)", fontsize=8.5)
    axA.set_title("Sensitivity (solid) vs false-positive rate (hatched)", fontsize=9,
                  loc="left", fontweight="semibold")

    # B: MCC per rule, BRAID-supported emphasised
    mo = mcc.set_index("rule_id").loc[order]
    for xi, rid in zip(xs, order):
        v = mo.loc[rid, "mcc"]
        axB.bar(xi, v, 0.66, color=col(rid), edgecolor="white", lw=0.7, zorder=3)
        axB.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=7.4,
                 fontweight="bold" if rid == "braid_supported" else "normal")
    axB.set_xticks(list(xs))
    axB.set_xticklabels([rlab[r] for r in order], fontsize=7.2)
    axB.set_ylim(0, 0.7)
    axB.set_ylabel("Matthews correlation (MCC)", fontsize=8.5)
    axB.set_title("BRAID-supported beats rMATS FDR", fontsize=9, loc="left",
                  fontweight="semibold")

    # CD: confusion matrices for rMATS vs BRAID-supported, stacked
    for k, (rid, title) in enumerate([("rmats_fdr", "rMATS FDR<0.05"),
                                      ("braid_supported", "BRAID supported")]):
        sub = conf[conf.rule_id == rid]
        m = [[int(sub[(sub.truth == t) & (sub.call == c)]["count"].iloc[0])
              for c in ["Called positive", "Not called"]]
             for t in ["RT-PCR positive", "RT-PCR negative"]]
        ax = axCD.inset_axes([0.0, 0.55 - 0.55 * k, 1.0, 0.40])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=60, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, m[i][j], ha="center", va="center", fontsize=9,
                        color="white" if m[i][j] > 35 else F.INK, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["call+", "call−"], fontsize=6.5)
        ax.set_yticklabels(["RT+", "RT−"], fontsize=6.5)
        ax.set_title(title, fontsize=7.8,
                     fontweight="bold" if rid == "braid_supported" else "normal",
                     color=F.HERO if rid == "braid_supported" else F.INK)
    axCD.axis("off")

    fig.suptitle("Auxiliary TRA2 detection: BRAID-supported cuts false positives 8→2 "
                 "at higher specificity and MCC (76 pos, 36 neg)",
                 fontsize=10.2, fontweight="bold", y=1.04, x=0.01, ha="left")
    for ax, ltr in zip((axA, axB, axCD), "ABC"):
        F.panel_letter(ax, ltr)
    _save(fig, "supp_fig1_detection_upgraded")
    plt.close(fig)


def _dm1():
    base = os.path.join(_REPO, "benchmarks/application_dm1/results")
    diff = pd.read_csv(os.path.join(base, "dm1_braid_differential.tsv"), sep="\t",
                       usecols=["gene", "dpsi", "rmats_fdr", "tier", "prob_large_effect"])
    anchor = pd.read_csv(os.path.join(base, "dm1_anchor_gene_summary.tsv"), sep="\t")
    return diff, anchor


def fig5():
    summ = json.load(open(os.path.join(_REPO,
                     "benchmarks/application_dm1/results/dm1_application_summary.json")))
    diff, anchor = _dm1()
    fig = plt.figure(figsize=(12.6, 8.2))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                          height_ratios=[1.0, 1.05])
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # A: filtering funnel
    tc = summ["tier_counts"]
    steps = [("events ≥ support", summ["events_after_braid_min_support"], F.NATIVE),
             ("rMATS big & FDR<0.05",
              summ["rmats_big_events_fdr_lt_0_05_abs_dpsi_ge_0_1"], "#C7A6BE"),
             ("BRAID supported", tc["supported"], TIER["supported"]),
             ("BRAID high-confidence", tc["high-confidence"], F.HERO)]
    for i, (lab, v, c) in enumerate(steps):
        axA.barh(len(steps) - 1 - i, v, color=c, edgecolor="white", lw=0.8, zorder=3, log=True)
        axA.text(v * 1.4, len(steps) - 1 - i, f"{v:,}", va="center", fontsize=8,
                 fontweight="bold" if c == F.HERO else "normal")
    axA.set_yticks(range(len(steps)))
    axA.set_yticklabels([s[0] for s in steps][::-1], fontsize=8)
    axA.set_xscale("log")
    axA.set_xlim(20, 5e6)
    axA.set_xlabel("events (log scale)", fontsize=8.5)
    axA.set_title("Calibrated filtering of 144,975 DM1 events", fontsize=9.3, loc="left",
                  fontweight="semibold")

    # B: dPSI vs -log10 FDR surface, by tier, anchors marked
    df = diff.copy()
    df["nlfdr"] = -np.log10(df["rmats_fdr"].clip(lower=1e-300))
    df["nlfdr"] = df["nlfdr"].clip(upper=50)
    rng = np.random.default_rng(7)
    ns = df[df.tier == "not-significant"]
    ns = ns.iloc[rng.choice(len(ns), size=min(8000, len(ns)), replace=False)]
    axB.scatter(ns.dpsi, ns.nlfdr, s=3, c=TIER["not-significant"], alpha=0.5, zorder=1,
                rasterized=True, edgecolors="none")
    for tier in ["significant", "supported", "high-confidence"]:
        s = df[df.tier == tier]
        axB.scatter(s.dpsi, s.nlfdr, s=8 if tier != "high-confidence" else 16,
                    c=TIER[tier], alpha=0.8, zorder=3, edgecolors="none",
                    rasterized=True, label=f"{tier} ({len(s):,})")
    axB.scatter(anchor.best_dpsi,
                (-np.log10(anchor.best_rmats_fdr.clip(lower=1e-300))).clip(upper=50),
                s=46, facecolor="none", edgecolor="black", lw=1.2, zorder=6, label="DM1 anchors")
    axB.axvline(0, color="#bbbbbb", lw=0.7)
    axB.set_xlim(-1.02, 1.02)
    axB.set_ylim(-2, 56)
    axB.set_xlabel("Disease − control ΔPSI", fontsize=8.5)
    axB.set_ylabel("−log10 rMATS FDR (capped 50)", fontsize=8.5)
    axB.set_title("Tier surface with curated anchors", fontsize=9.3, loc="left",
                  fontweight="semibold")
    axB.legend(loc="lower center", frameon=False, fontsize=6.4, ncol=2,
               handletextpad=0.2, columnspacing=1.0)

    # C: recovered high-confidence anchor events, calibrated 95% interval forest
    hc = anchor[anchor.best_tier == "high-confidence"].copy()
    hc = hc.sort_values("best_dpsi")
    yy = range(len(hc))
    for y, (_, r) in zip(yy, hc.iterrows()):
        axC.plot([r.best_ci_low, r.best_ci_high], [y, y], "-", color=F.HERO, lw=2.0, zorder=3)
        axC.plot([r.best_dpsi], [y], "o", color=F.HERO, ms=5, zorder=4)
    axC.axvline(0, color="#888", ls="--", lw=1)
    axC.set_yticks(list(yy))
    axC.set_yticklabels(hc.gene, fontsize=7.6)
    axC.set_xlim(-1.05, 1.05)
    axC.set_xlabel("Disease − control ΔPSI (calibrated 95%)", fontsize=8.5)
    axC.set_title("Recovered high-confidence anchor events", fontsize=9.3, loc="left",
                  fontweight="semibold")

    # D: anchor recovery summary
    cats = [("≥1 big rMATS event", summ["anchor_genes_with_big_rmats_event"]),
            ("≥1 BRAID supported", summ["anchor_genes_with_braid_supported_event"]),
            ("≥1 BRAID high-conf", summ["anchor_genes_with_braid_high_confidence_event"]),
            ("primary: high-conf", summ["primary_anchor_genes_with_braid_high_confidence_event"])]
    tot = [summ["anchor_genes_total"]] * 3 + [summ["primary_anchor_genes_total"]]
    for i, ((lab, v), t) in enumerate(zip(cats, tot)):
        axD.barh(len(cats) - 1 - i, t, color="#E7E9EB", edgecolor="white", zorder=2)
        axD.barh(len(cats) - 1 - i, v, color=F.HERO, edgecolor="white", zorder=3)
        axD.text(t + 0.2, len(cats) - 1 - i, f"{v}/{t}", va="center", fontsize=8)
    axD.set_yticks(range(len(cats)))
    axD.set_yticklabels([c[0] for c in cats][::-1], fontsize=7.2)
    axD.set_xlim(0, 17)
    axD.set_xlabel("anchor genes", fontsize=8.5)
    axD.set_title("Known DM1 anchors retained on the calibrated scale", fontsize=9.3,
                  loc="left", fontweight="semibold")

    fig.suptitle("BRAID prioritization on a public DM1 rMATS reanalysis (GSE201255)",
                 fontsize=11.5, fontweight="bold", y=0.99, x=0.01, ha="left")
    for ax, ltr in zip((axA, axB, axC, axD), "ABCD"):
        F.panel_letter(ax, ltr, y=1.08)
    _save(fig, "fig5_dm1_application_upgraded")
    plt.close(fig)


def fig6():
    base = os.path.join(_REPO, "benchmarks/application_dm1/results")
    genes = pd.read_csv(os.path.join(base, "dm1_top_braid_candidate_genes.tsv"), sep="\t")
    fig = plt.figure(figsize=(12.4, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.32)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    sc = "max_braid_score" if "max_braid_score" in genes.columns else genes.columns[-1]
    g = genes.sort_values(sc, ascending=True).tail(10)
    axA.barh(range(len(g)), g[sc], color=F.HERO, edgecolor="white", zorder=3)
    axA.set_yticks(range(len(g)))
    axA.set_yticklabels(g["gene"] if "gene" in g.columns else g.iloc[:, 0], fontsize=8)
    axA.set_xlabel("max BRAID score", fontsize=8.5)
    axA.set_title("Top non-anchor candidate genes (rMATS-sig + high-confidence)",
                  fontsize=9, loc="left", fontweight="semibold")

    dcol = "max_abs_dpsi" if "max_abs_dpsi" in genes.columns else None
    if dcol:
        axB.scatter(genes[dcol], genes[sc], s=46, color=F.HERO, edgecolor="white",
                    alpha=0.85, zorder=3)
        for _, r in g.tail(6).iterrows():
            axB.annotate(r["gene"], (r[dcol], r[sc]), fontsize=7, xytext=(3, 2),
                         textcoords="offset points")
        axB.set_xlabel("max |ΔPSI|", fontsize=8.5)
        axB.set_ylabel("max BRAID score", fontsize=8.5)
    axB.set_title("Candidates on the calibrated margin scale", fontsize=9, loc="left",
                  fontweight="semibold")

    fig.suptitle("BRAID score-based candidate prioritization in the DM1 reanalysis",
                 fontsize=11, fontweight="bold", y=1.03, x=0.01, ha="left")
    for ax, ltr in zip((axA, axB), "AB"):
        F.panel_letter(ax, ltr)
    _save(fig, "fig6_dm1_candidates_upgraded")
    plt.close(fig)


if __name__ == "__main__":
    fig3()
    fig4()
    fig_s2()
    fig_s3()
    fig_s1()
    fig5()
    fig6()
