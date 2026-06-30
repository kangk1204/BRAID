"""Figure for the within-study partial-validation use case (active-learning style).

BRAID's within-study workflow is active learning for splicing confidence: you label
a few events by RT-PCR, those (estimate, truth) pairs calibrate the interval for
every remaining event, and the acquisition choice -- which events to label -- drives
how much you gain.

Reads benchmarks/results/within_study_calibration.json and renders two panels:
  (A) Learning curve: held-out coverage vs the number of RT-PCR-labelled events, for
      a representative (random) acquisition vs a top-hit-only acquisition. The
      representative curve reaches nominal 0.95 from ~20 labels; top-hit-only is
      mis-calibrated -- the classic active-learning failure of a biased query set.
  (B) Acquisition strategy at a fixed budget (N=30): fraction of the remaining
      events recovered as "reliable", per strategy, with the reliable-flag precision
      annotated. Representative + biologically-confident negatives recovers the most;
      top-hit recovers none; low-signal-as-negative is circular and not recommended.

Output: paper/figures/within_study_calibration.{png,pdf,svg} (+ outputs/ mirror).
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
SRC = os.path.join(_REPO, "benchmarks/results/within_study_calibration.json")
OUT_DIR = os.path.join(_REPO, "outputs/figures/manuscript/within_study")
PAPER_DIR = os.path.join(_REPO, "paper/figures")
STEM = "within_study_calibration"
RAND_C = "#1b6ca8"
TOP_C = "#d1495b"

# Acquisition strategies: short label, bar colour, "recommended?" (hatch if not).
_STRATEGY = [
    ("top_hit", "top-hit only", "#d1495b", False),
    ("random", "representative", "#1b6ca8", True),
    # Measuring the negatives' dPSI and assuming it is exactly 0 give identical
    # calibration on TRA2 (measured negatives are ~0), so they are one bar.
    ("with_negatives_measured", "+ negatives\n(measured = assumed 0)", "#2a9d8f", True),
    ("lowsignal_assumed0", "low-signal as neg\n(circular)", "#9aa0a6", False),
]


def _save(fig: plt.Figure) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    for d in (OUT_DIR, PAPER_DIR):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(os.path.join(d, f"{STEM}.{ext}"),
                        dpi=300 if ext == "png" else None, bbox_inches="tight")


def _panel_learning(ax: plt.Axes, rows: list) -> None:
    ns = [r["N"] for r in rows]
    for mode, c, lab in (("random", RAND_C, "representative acquisition"),
                         ("tophit", TOP_C, "top-hit only")):
        cov = [r[mode]["coverage"] for r in rows]
        lo = [max(0.0, r[mode]["coverage"] - r[mode]["coverage_lo"]) for r in rows]
        hi = [max(0.0, r[mode]["coverage_hi"] - r[mode]["coverage"]) for r in rows]
        ax.errorbar(ns, cov, yerr=[lo, hi], fmt="o-", color=c, capsize=3, lw=1.6,
                    ms=6, label=lab)
    ax.axhline(0.95, color="#444", ls="--", lw=1)
    ax.text(ns[-1], 0.95, " nominal 0.95", fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("RT-PCR-labelled events used for calibration")
    ax.set_ylabel("coverage of the remaining events")
    ax.set_title("(A) Label a few, calibrate the rest")
    ax.legend(fontsize=8, loc="lower right", frameon=False)


def _panel_strategy(ax: plt.Axes, by_strategy: dict, N: int) -> None:
    present = [s for s in _STRATEGY if s[0] in by_strategy]
    ys = list(range(len(present)))[::-1]
    for (key, lab, color, ok), y in zip(present, ys):
        m = by_strategy[key]
        frac = m["frac_reliable"] * 100.0
        ax.barh(y, frac, color=color, alpha=1.0 if ok else 0.55,
                hatch=None if ok else "//", edgecolor="white")
        prec = m.get("reliable_precision")
        tag = f"{frac:.1f}%"
        if prec == prec and frac > 0:  # not NaN
            tag += f"  (prec {prec:.2f}, cov {m['coverage']:.2f})"
        ax.text(frac + 0.3, y, tag, va="center", fontsize=7)
    ax.set_yticks(ys)
    ax.set_yticklabels([s[1] for s in present], fontsize=7)
    ax.set_xlabel("remaining events recovered as reliable (%)")
    ax.set_title(f"(B) Which events to label (acquisition, N={N})")
    ax.set_xlim(0, max(18, ax.get_xlim()[1] + 6))


def main() -> None:
    top = json.load(open(SRC, encoding="utf-8"))
    d = top["datasets"]["tra2"] if "datasets" in top else top   # primary panel = TRA2
    rows = d["by_N"]
    sc = top.get("strategy_comparison", {})
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    _panel_learning(axes[0], rows)
    if sc:
        _panel_strategy(axes[1], sc["by_strategy"], sc.get("N", 30))
    fig.suptitle(
        "Active-learning calibration of splicing confidence from a small "
        f"within-study RT-PCR set ({d['dataset'].upper()}, n={d['n']})",
        fontsize=11, y=1.04)
    fig.tight_layout()
    _save(fig)
    plt.close(fig)
    print(f"Wrote {STEM}.(png/pdf/svg) to paper/figures/ and outputs/")


if __name__ == "__main__":
    main()
