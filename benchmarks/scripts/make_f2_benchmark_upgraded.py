"""Upgraded renderer for manuscript Figure 2 (design pilot).

Reads the COMMITTED source-data workbook (outputs/figures/manuscript/F2/
f2_source_data.xlsx) -- the same verified data the current figure uses -- and
renders an upgraded composition. No recomputation, no fabricated values.

Design system (vs the previous "rainbow + generic-title" look):
  * Hero/muted colour: BRAID is the only saturated colour (deep blue), the three
    comparators are desaturated so the eye goes to the result, not the palette.
  * Takeaway titles + an honest figure-level message, not bare panel labels.
  * Direct labelling (frameless legend / on-point labels), nominal-region shading.
  * Panel B sorted by value as horizontal bars with a "lower is better" cue.
  * Panel C annotates the only-BRAID-reaches-nominal message and the honest
    coverage/width trade-off.
"""
# ruff: noqa: I001
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "benchmarks"))
import figstyle  # noqa: E402

figstyle.apply()

SRC = os.path.join(_REPO, "outputs/figures/manuscript/F2/f2_source_data.xlsx")
OUT = os.path.join(_REPO, "outputs/figures/manuscript/F2/fig2_benchmark_upgraded")

# One accent (BRAID) + three muted comparators (desaturated Okabe-Ito).
HERO = "#08519C"          # BRAID: deep, saturated blue
MUTED = {
    "MAJIQ": "#E2B07A",   # muted ochre
    "betAS": "#86BBA8",   # muted teal
    "rMATS": "#C7A6BE",   # muted mauve
}
INK = "#222222"
GUIDE = "#9AA0A6"


def _load():
    a = pd.read_excel(SRC, sheet_name="panel_A_coverage")
    bc = pd.read_excel(SRC, sheet_name="panel_BC_pooled")
    return a, bc


def _color(label: str) -> str:
    return HERO if label == "BRAID" else MUTED[label]


def main() -> None:
    a, bc = _load()
    fig = plt.figure(figsize=(13.2, 4.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.0, 1.15], wspace=0.36)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    groups = ["Pooled", "TRA2", "Circadian", "SRS354082"]   # headline first (top)
    order = ["BRAID", "betAS", "rMATS", "MAJIQ"]             # BRAID top of each cluster
    nm = {"BRAID-conformal": "BRAID", "MAJIQ": "MAJIQ", "betAS": "betAS", "rMATS": "rMATS"}
    a = a.assign(label=a["method"].map(nm))

    # ---- Panel A: forest, hero/muted, nominal shading, group bands ---------
    gap, step = 1.30, 0.236
    y_top = (len(groups) - 1) * gap + 1.0
    axA.axvspan(0.95, 1.02, color="#2E7D32", alpha=0.06, zorder=0)   # nominal-or-better
    axA.axvline(0.95, color="#2E7D32", ls=(0, (4, 3)), lw=1.1, zorder=1)
    yticks, ylabels = [], []
    for gi, g in enumerate(groups):
        gc = y_top - gi * gap
        if gi % 2 == 0:
            axA.axhspan(gc - gap / 2, gc + gap / 2, color="#000000", alpha=0.035, zorder=0)
        sub = a[a["dataset"] == g].set_index("label")
        for mi, lab in enumerate(order):
            r = sub.loc[lab]
            y = gc + (1.5 - mi) * step
            hero = lab == "BRAID"
            axA.errorbar(
                r["coverage"], y,
                xerr=[[r["coverage"] - r["wilson_low"]], [r["wilson_high"] - r["coverage"]]],
                fmt="o", ms=8.5 if hero else 5.2,
                color=_color(lab), ecolor=_color(lab),
                elinewidth=2.4 if hero else 1.4, capsize=2.6 if hero else 1.8,
                mec="white", mew=1.1 if hero else 0.6,
                zorder=6 if hero else 4,
            )
        n = int(sub.iloc[0]["n"])
        yticks.append(gc)
        ylabels.append(f"{g}\n(n={n})")
    axA.set_yticks(yticks)
    axA.set_yticklabels(ylabels, fontsize=8)
    # bold the headline (Pooled) tick label
    axA.get_yticklabels()[0].set_fontweight("bold")
    axA.set_ylim(y_top - (len(groups) - 1) * gap - gap / 2 - 0.1, y_top + gap / 2 + 0.5)
    axA.set_xlim(0.30, 1.02)
    axA.set_xlabel("Coverage of RT-PCR ΔPSI", fontsize=9)
    axA.set_title("Only BRAID's interval reaches nominal coverage", fontsize=9.5, loc="left",
                  fontweight="semibold")
    axA.text(0.952, y_top + gap / 2 + 0.34, "nominal 0.95", color="#2E7D32",
             fontsize=7.2, ha="left", va="center", fontweight="medium")
    handles = [
        Line2D([0], [0], marker="o", ls="", ms=8 if k == "BRAID" else 5.5,
               color=_color(k), mec="white", mew=0.8, label=k)
        for k in order
    ]
    axA.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.3,
               handletextpad=0.3, labelspacing=0.28, ncol=1,
               bbox_to_anchor=(0.0, 0.66))

    # ---- Panel B: interval score, sorted, horizontal, BRAID emphasised ------
    pooled = bc.assign(label=bc["method"].map(nm)).sort_values("interval_score",
                                                               ascending=False)
    yb = range(len(pooled))
    for y, (_, r) in zip(yb, pooled.iterrows()):
        hero = r["label"] == "BRAID"
        axB.barh(y, r["interval_score"], height=0.66,
                 color=_color(r["label"]), edgecolor="white", lw=0.8,
                 zorder=3, alpha=1.0 if hero else 0.92)
        axB.text(r["interval_score"] + 0.04, y, f"{r['interval_score']:.2f}",
                 va="center", ha="left", fontsize=8,
                 fontweight="bold" if hero else "normal",
                 color=HERO if hero else INK)
    axB.set_yticks(list(yb))
    axB.set_yticklabels(pooled["label"], fontsize=8.5)
    for t, lab in zip(axB.get_yticklabels(), pooled["label"]):
        if lab == "BRAID":
            t.set_fontweight("bold")
            t.set_color(HERO)
    axB.set_xlim(0, 2.45)
    axB.set_xlabel("Interval score  (↓ lower is better)", fontsize=9)
    axB.set_title("BRAID is also the sharpest-calibrated", fontsize=9.5, loc="left",
                  fontweight="semibold")
    axB.spines["left"].set_visible(False)
    axB.tick_params(axis="y", length=0)

    # ---- Panel C: coverage vs width, only-BRAID-nominal + honest trade-off --
    axC.axhspan(0.95, 1.0, color="#2E7D32", alpha=0.06, zorder=0)
    axC.axhline(0.95, color="#2E7D32", ls=(0, (4, 3)), lw=1.1, zorder=1)
    # per-point label placement that clears the (large) markers
    LBL = {  # method: (dx, dy, ha, va)
        "BRAID": (-0.022, 0.0, "right", "center"),
        "betAS": (0.0, 0.045, "center", "bottom"),
        "rMATS": (0.0, 0.045, "center", "bottom"),
        "MAJIQ": (0.0, -0.05, "center", "top"),
    }
    for _, r in bc.assign(label=bc["method"].map(nm)).iterrows():
        hero = r["label"] == "BRAID"
        axC.scatter(r["mean_width"], r["coverage"], s=200 if hero else 95,
                    color=_color(r["label"]), edgecolor="white",
                    linewidth=1.3 if hero else 0.8, zorder=6 if hero else 4)
        dx, dy, ha, va = LBL[r["label"]]
        axC.annotate(r["label"], (r["mean_width"], r["coverage"]),
                     xytext=(r["mean_width"] + dx, r["coverage"] + dy),
                     fontsize=8, ha=ha, va=va,
                     fontweight="bold" if hero else "normal",
                     color=HERO if hero else INK)
    axC.set_xlim(0.10, 0.74)
    axC.set_ylim(0.45, 1.02)
    axC.set_xlabel("Mean interval width", fontsize=9)
    axC.set_ylabel("Coverage of RT-PCR ΔPSI", fontsize=9)
    axC.set_title("Calibrated coverage costs width — honestly", fontsize=9.5, loc="left",
                  fontweight="semibold")
    axC.text(0.115, 0.955, "nominal 0.95", color="#2E7D32", fontsize=7.2,
             ha="left", va="bottom", fontweight="medium")

    fig.suptitle(
        "BRAID is the only interval that covers RT-PCR ΔPSI at the nominal 95% rate, "
        "at the lowest interval score",
        fontsize=11.5, fontweight="bold", y=1.04, x=0.01, ha="left",
    )
    for ax, letter in zip((axA, axB, axC), "ABC"):
        ax.text(-0.02, 1.10, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="top", ha="right")

    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=320 if ext == "png" else None, bbox_inches="tight")
    print(f"wrote {OUT}.png/.svg/.pdf  (source: {os.path.relpath(SRC, _REPO)})")


if __name__ == "__main__":
    main()
