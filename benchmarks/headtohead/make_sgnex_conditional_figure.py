"""SG-NEx conditional-coverage figure (S3 Fig): Mondrian fixes event-type coverage.

Reads benchmarks/results/sgnex_conditional_eval.json (SG-NEx HepG2 vs K562 long-read
ΔPSI surface, n=46,160) and renders three panels:

  (A) Per-event-type coverage with Wilson 95% CIs for the constant support band vs
      the composite (event-type x support) Mondrian; the constant band under-covers
      A3SS/A5SS, the composite hugs 0.95.
  (B) Per-event-type mean width: the composite widens the under-covered types where
      needed (not a uniform inflation).
  (C) Pooled interval score for the three bands; the composite matches the constant
      band's sharpness while event-type-only grouping is worse.

Output: paper/figures/supp_fig3_sgnex_conditional.{png,pdf,svg} (+ outputs/ mirror).
Usage:  python benchmarks/headtohead/make_sgnex_conditional_figure.py
"""

# ruff: noqa: I001
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

import sys  # noqa: E402
sys.path.insert(0, os.path.join(_REPO, "benchmarks"))
import figstyle  # noqa: E402

figstyle.apply()

SRC = os.path.join(_REPO, "benchmarks/results/sgnex_conditional_eval.json")
OUT_DIR = os.path.join(_REPO, "outputs/figures/manuscript/sgnex_conditional")
PAPER_DIR = os.path.join(_REPO, "paper/figures")
STEM = "supp_fig3_sgnex_conditional"
TYPES = ["SE", "A3SS", "A5SS"]
CONST_C = "#9aa7b1"
COMP_C = "#1b6ca8"
EVT_C = "#d1894e"
NOMINAL = 0.95


def _save(fig: plt.Figure) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    for d in (OUT_DIR, PAPER_DIR):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(os.path.join(d, f"{STEM}.{ext}"),
                        dpi=300 if ext == "png" else None, bbox_inches="tight")


def _panel_coverage(ax: plt.Axes, const: dict, comp: dict) -> None:
    x = range(len(TYPES))
    for off, tab, color, fill, lab in (
        (-0.13, const, CONST_C, "none", "constant band"),
        (0.13, comp, COMP_C, COMP_C, "composite Mondrian"),
    ):
        bt = tab["by_type"]
        ys = [bt[t]["coverage"] for t in TYPES]
        lo = [bt[t]["coverage"] - bt[t]["wilson"][0] for t in TYPES]
        hi = [bt[t]["wilson"][1] - bt[t]["coverage"] for t in TYPES]
        ax.errorbar([i + off for i in x], ys, yerr=[lo, hi], fmt="o", color=color,
                    mfc=fill, mec=color, capsize=4, lw=1.4, ms=8, label=lab)
    ax.axhline(NOMINAL, color="#b03a2e", ls="--", lw=1)
    ax.text(len(TYPES) - 0.5, NOMINAL, " nominal 0.95", color="#b03a2e", va="bottom",
            ha="right", fontsize=8)
    ax.set_xticks(list(x), TYPES)
    ax.set_xlim(-0.5, len(TYPES) - 0.5)
    ax.set_ylabel("coverage @ 95%  (held-out)")
    ax.set_title("(A) Conditional coverage by event type")
    ax.legend(fontsize=8, loc="lower left", frameon=False)


def _panel_width(ax: plt.Axes, const: dict, comp: dict) -> None:
    x = range(len(TYPES))
    w = 0.36
    cw = [const["by_type"][t]["mean_width"] for t in TYPES]
    aw = [comp["by_type"][t]["mean_width"] for t in TYPES]
    ax.bar([i - w / 2 for i in x], cw, w, color=CONST_C, label="constant band")
    ax.bar([i + w / 2 for i in x], aw, w, color=COMP_C, label="composite Mondrian")
    ax.set_xticks(list(x), TYPES)
    ax.set_ylim(0, max(max(cw), max(aw)) * 1.2)
    ax.set_ylabel("mean interval width")
    ax.set_title("(B) Composite widens under-covered types")
    ax.legend(fontsize=8, loc="upper left", frameon=False)


def _panel_iscore(ax: plt.Axes, d: dict) -> None:
    names = ["constant_support", "adaptive_event_type", "adaptive_composite"]
    labels = ["constant", "event-type", "composite"]
    vals = [d[n]["pooled_interval_score"] for n in names]
    ax.bar([0, 1, 2], vals, color=[CONST_C, EVT_C, COMP_C], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(vals[0], color=CONST_C, ls=":", lw=1)
    ax.set_xticks([0, 1, 2], labels)
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_ylabel("pooled interval score  (lower is better)")
    ax.set_title("(C) Composite keeps sharpness")


def main() -> None:
    d = json.load(open(SRC, encoding="utf-8"))
    const, comp = d["constant_support"], d["adaptive_composite"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    _panel_coverage(axes[0], const, comp)
    _panel_width(axes[1], const, comp)
    _panel_iscore(axes[2], d)
    fig.suptitle(
        "Event-type/support composite Mondrian restores conditional coverage at no sharpness cost "
        f"(SG-NEx long-read ΔPSI, n={d['n']:,})", fontsize=11, y=1.04)
    fig.tight_layout()
    _save(fig)
    plt.close(fig)
    print(f"Wrote {STEM}.(png/pdf/svg) to paper/figures/ and outputs/")


if __name__ == "__main__":
    main()
