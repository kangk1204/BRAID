"""Figure: each caller alone vs each caller + BRAID recalibration.

Visualises the operative-layer result (manuscript Table 3 + the SUPPA2 ablation):
every caller's native interval under-covers the orthogonal RT-PCR truth, and the
same conformal recalibration brings each of them to nominal 0.95.

(A) Per-caller coverage, native vs +BRAID, with the nominal 0.95 line and Wilson
    95% CIs on the recalibrated bars. SUPPA2 has no native event-level interval, so
    only its +BRAID bar is shown.
(B) Coverage-vs-width "lift" plot: an arrow per caller from its native
    (narrow, under-covering) interval up to the +BRAID interval at nominal coverage.

Source data (committed): benchmarks/results/recalibration_ablation.json,
benchmarks/results/suppa2_recalibration.json.
Output: paper/figures/tool_vs_braid.{png,pdf,svg} (+ outputs/ mirror).
"""

# ruff: noqa: I001
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

import sys  # noqa: E402
sys.path.insert(0, os.path.join(_REPO, "benchmarks"))
import figstyle  # noqa: E402

figstyle.apply()

ABLATION = os.path.join(_REPO, "benchmarks/results/recalibration_ablation.json")
SUPPA2 = os.path.join(_REPO, "benchmarks/results/suppa2_recalibration.json")
OUT_DIR = os.path.join(_REPO, "outputs/figures/manuscript/tool_vs_braid")
PAPER_DIR = os.path.join(_REPO, "paper/figures")
STEM = "tool_vs_braid"

NATIVE_C = "#9aa0a6"
BRAID_C = "#1b6ca8"
CALLER_C = {
    "MAJIQ": "#d1495b", "betAS": "#edae49", "rMATS": "#00798c", "SUPPA2": "#8d5a99",
}
# Map the ablation JSON method keys to short caller labels.
_LABEL = {
    "MAJIQ (real binary)": "MAJIQ",
    "betAS (real tool)": "betAS",
    "rMATS IncLevel t-CI": "rMATS",
}


def _save(fig: plt.Figure) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    for d in (OUT_DIR, PAPER_DIR):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(os.path.join(d, f"{STEM}.{ext}"),
                        dpi=300 if ext == "png" else None, bbox_inches="tight")


def _load() -> list[dict]:
    ab = json.load(open(ABLATION, encoding="utf-8"))["methods"]
    rows: list[dict] = []
    for key, lab in _LABEL.items():
        m = ab[key]
        rows.append({
            "caller": lab,
            "native_cov": m["raw_coverage"], "native_w": m["raw_width"],
            "braid_cov": m["recal_coverage"], "braid_w": m["recal_width"],
            "wilson": m.get("recal_wilson"),
        })
    sup = json.load(open(SUPPA2, encoding="utf-8"))["by_method"]["SUPPA2"]
    rows.append({
        "caller": "SUPPA2", "native_cov": None, "native_w": None,
        "braid_cov": sup["recalibrated_coverage"], "braid_w": sup["mean_width"],
        "wilson": sup.get("wilson"),
    })
    return rows


def _panel_coverage(ax: plt.Axes, rows: list[dict]) -> None:
    labels = [r["caller"] for r in rows]
    x = range(len(rows))
    w = 0.38
    for i, r in enumerate(rows):
        if r["native_cov"] is not None:
            ax.bar(i - w / 2, r["native_cov"], w, color=NATIVE_C,
                   label="native" if i == 0 else None)
        else:  # SUPPA2: no native interval
            ax.text(i - w / 2, 0.02, "no native\ninterval", ha="center", va="bottom",
                    fontsize=6, color="#666", rotation=0)
        yerr = None
        if r["wilson"]:
            yerr = [[r["braid_cov"] - r["wilson"][0]], [r["wilson"][1] - r["braid_cov"]]]
        ax.bar(i + w / 2, r["braid_cov"], w, color=BRAID_C, yerr=yerr, capsize=3,
               label="+ BRAID" if i == 0 else None)
    ax.axhline(0.95, color="#444", ls="--", lw=1)
    ax.text(len(rows) - 0.5, 0.952, "nominal 0.95", fontsize=7, va="bottom", ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("coverage of RT-PCR ΔPSI")
    ax.set_title("(A) Each caller under-covers; + BRAID reaches nominal")
    ax.legend(fontsize=8, loc="lower left", frameon=False)


def _panel_lift(ax: plt.Axes, rows: list[dict]) -> None:
    for r in rows:
        c = CALLER_C[r["caller"]]
        if r["native_cov"] is not None:
            ax.annotate(
                "", xy=(r["braid_w"], r["braid_cov"]),
                xytext=(r["native_w"], r["native_cov"]),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=2, alpha=0.85),
            )
            ax.plot([r["native_w"]], [r["native_cov"]], "o", color=c, ms=6,
                    mfc="white", mec=c)
        ax.plot([r["braid_w"]], [r["braid_cov"]], "o", color=c, ms=8)
        ax.text(r["braid_w"], r["braid_cov"] + 0.012, r["caller"], fontsize=7,
                ha="center", color=c)
    ax.axhline(0.95, color="#444", ls="--", lw=1)
    ax.set_xlabel("mean interval width")
    ax.set_ylabel("coverage of RT-PCR ΔPSI")
    ax.set_ylim(0.40, 1.02)
    ax.set_title("(B) The calibration lift (open = native, filled = + BRAID)")
    handles = [Patch(color=CALLER_C[c], label=c) for c in CALLER_C]
    ax.legend(handles=handles, fontsize=7, loc="lower right", frameon=False, ncol=2)


def main() -> None:
    rows = _load()
    plt.rcParams.update(
        {"font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans"}
    )
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))
    _panel_coverage(axA, rows)
    _panel_lift(axB, rows)
    fig.suptitle(
        "BRAID recalibration is caller-agnostic: native intervals under-cover, "
        "+ BRAID reaches nominal coverage", fontsize=11, y=1.02)
    fig.tight_layout()
    _save(fig)
    plt.close(fig)
    print(f"Wrote {STEM}.(png/pdf/svg) to paper/figures/ and outputs/")


if __name__ == "__main__":
    main()
