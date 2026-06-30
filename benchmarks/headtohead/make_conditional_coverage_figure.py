"""Killer figure for Stage 1: event-type Mondrian restores conditional coverage.

Reads benchmarks/results/adaptive_conditional_eval.json (PacBio long-read PSI
surface, n=252, A3SS/A5SS) and renders three panels:

  (A) Per-event-type coverage with Wilson 95% CIs: the constant band scatters
      (A3SS over-covers, A5SS under-covers, CIs miss 0.95); the event-type band
      hugs 0.95 within CI.
  (B) Per-event-type mean width: the adaptive band NARROWS the over-covered bin
      and WIDENS the under-covered bin -- width redistribution, not uniform
      inflation.
  (C) Pooled interval score (lower is better): adaptive is sharper overall.

Output: paper/figures/conditional_coverage.{png,pdf,svg} (+ outputs/ mirror).
Usage:  python benchmarks/headtohead/make_conditional_coverage_figure.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
SRC = os.path.join(_REPO, "benchmarks/results/adaptive_conditional_eval.json")
OUT_DIR = os.path.join(_REPO, "outputs/figures/manuscript/conditional_coverage")
PAPER_DIR = os.path.join(_REPO, "paper/figures")
STEM = "conditional_coverage"
TYPES = ["A3SS", "A5SS"]
CONST_C = "#9aa7b1"
ADAPT_C = "#1b6ca8"
NOMINAL = 0.95


def _save(fig: plt.Figure) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    for d in (OUT_DIR, PAPER_DIR):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(os.path.join(d, f"{STEM}.{ext}"),
                        dpi=300 if ext == "png" else None, bbox_inches="tight")


def _panel_coverage(ax: plt.Axes, const: dict, adapt: dict) -> None:
    x = range(len(TYPES))
    for off, tab, color, mk, lab, fill in (
        (-0.12, const, CONST_C, "o", "constant band", "none"),
        (0.12, adapt, ADAPT_C, "o", "event-type band", ADAPT_C),
    ):
        ys = [tab[t]["coverage"] for t in TYPES]
        lo = [tab[t]["coverage"] - tab[t]["wilson"][0] for t in TYPES]
        hi = [tab[t]["wilson"][1] - tab[t]["coverage"] for t in TYPES]
        ax.errorbar([i + off for i in x], ys, yerr=[lo, hi], fmt=mk, color=color,
                    mfc=fill, mec=color, capsize=4, lw=1.4, ms=9, label=lab)
    ax.axhline(NOMINAL, color="#b03a2e", ls="--", lw=1)
    ax.text(len(TYPES) - 0.5, NOMINAL, " nominal 0.95", color="#b03a2e", va="bottom", ha="right",
            fontsize=8)
    ax.set_xticks(list(x), TYPES)
    ax.set_xlim(-0.5, len(TYPES) - 0.5)
    ax.set_ylabel("coverage @ 95%  (held-out)")
    ax.set_title("(A) Conditional coverage by event type")
    ax.legend(fontsize=8, loc="lower center", frameon=False)


def _panel_width(ax: plt.Axes, const: dict, adapt: dict) -> None:
    x = range(len(TYPES))
    w = 0.36
    cw = [const[t]["mean_width"] for t in TYPES]
    aw = [adapt[t]["mean_width"] for t in TYPES]
    ax.bar([i - w / 2 for i in x], cw, w, color=CONST_C, label="constant band")
    ax.bar([i + w / 2 for i in x], aw, w, color=ADAPT_C, label="event-type band")
    for i, (c, a) in enumerate(zip(cw, aw)):
        pct = (a - c) / c * 100.0
        ax.annotate(f"{pct:+.0f}%", (i + w / 2, a), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8,
                    color="#1a7f37" if pct < 0 else "#b03a2e")
    ax.set_xticks(list(x), TYPES)
    ax.set_ylim(0, max(max(cw), max(aw)) * 1.22)
    ax.set_ylabel("mean interval width")
    ax.set_title("(B) Width is redistributed, not inflated")
    ax.legend(fontsize=8, loc="upper left", frameon=False)


def _panel_iscore(ax: plt.Axes, c_is: float, a_is: float) -> None:
    ax.bar([0, 1], [c_is, a_is], color=[CONST_C, ADAPT_C], width=0.6)
    for i, v in enumerate([c_is, a_is]):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1], ["constant", "event-type"])
    ax.set_ylabel("pooled interval score  (lower is better)")
    ax.set_ylim(0, max(c_is, a_is) * 1.18)
    ax.set_title("(C) Sharper overall")


def main() -> None:
    d = json.load(open(SRC, encoding="utf-8"))
    const = d["constant_band"]["by_event_type"]
    adapt = d["adaptive_band"]["by_event_type"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    _panel_coverage(axes[0], const, adapt)
    _panel_width(axes[1], const, adapt)
    _panel_iscore(axes[2], d["constant_band"]["pooled_interval_score"],
                  d["adaptive_band"]["pooled_interval_score"])
    fig.suptitle(
        "Event-type Mondrian conformal restores conditional coverage a single cutoff cannot "
        f"(PacBio long-read PSI, n={d['n']})", fontsize=11, y=1.04)
    fig.tight_layout()
    _save(fig)
    plt.close(fig)
    print(f"Wrote {STEM}.(png/pdf/svg) to paper/figures/ and outputs/")


if __name__ == "__main__":
    main()
