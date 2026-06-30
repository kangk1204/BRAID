#!/usr/bin/env python3
# ruff: noqa: I001, E402
"""Figure: why BRAID's calibrated layer beats aggressively filtering a caller.

Two panels, both from real committed data (detection_filter_sweep.json):

  A  Calibration is not a filtering knob. Each caller's own padj<0.05 AND
     |dPSI|>=0.1 confident calls still under-cover RT-PCR truth with the caller's
     native interval (muted dot); BRAID's conformal interval on the SAME events
     is near-nominal (BRAID dot). The dumbbell is the coverage that filtering
     leaves on the table.

  B  Detection is at parity, so the calibration comes for free. Each caller's
     full (significance x |dPSI|) grid is a faint cloud; its precision-recall
     frontier and best-MCC operating point are drawn; BRAID's detection tier
     (star) sits on the same frontier -- adopting BRAID does not cost detection.

Source: benchmarks/headtohead/detection_filter_sweep.json
Outputs: outputs/figures/manuscript/filtering/fig_filtering_vs_calibration.{png,svg,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
import figstyle as FS

ROOT = _HERE.parents[1]
SRC = _HERE / "detection_filter_sweep.json"
OUT_DIR = ROOT / "outputs/figures/manuscript/filtering"
PAPER_DIR = ROOT / "paper/figures"
STEM = "fig_filtering_vs_calibration"
CALLERS = ["rMATS", "betAS", "MAJIQ"]


def _pareto_front(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Upper-left precision-recall frontier: points not dominated in (recall, precision)."""
    front = []
    for rec, prec in sorted(pts, key=lambda p: (-p[0], -p[1])):
        if all(not (r >= rec and p >= prec and (r, p) != (rec, prec)) for r, p in front):
            if not front or prec > front[-1][1]:
                front.append((rec, prec))
    return sorted(front)


def main() -> None:
    FS.apply()
    data = json.loads(SRC.read_text())
    a = data["analysis_a_filtering_does_not_fix_coverage"]
    grid = data["analysis_b_full_grid"]
    best = {k: v["best_by_mcc"] for k, v in data["analysis_b_detection_frontier"].items()}
    braid_full = data["braid_operating_points"]["full_panel"]["BRAID_supported"]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(FS.WIDTH_DOUBLE, 3.6),
        gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.34})
    fig.subplots_adjust(bottom=0.30, top=0.88)

    # ---------- Panel A: confident-call coverage, native vs BRAID (no RT-PCR) ---
    # Headline BRAID marker is the TRANSFER variant, which never sees TRA2 RT-PCR
    # (calibrated on a different dataset). Cross-fit (uses RT-PCR) is a faint open
    # diamond = the upper reference, so the figure does not over-credit BRAID.
    FS.nominal_guide(axA, 0.95, axis="x")
    ys = np.arange(len(CALLERS))[::-1]
    for y, name in zip(ys, CALLERS):
        v = a[name]
        nat = v["native_coverage"]
        xfer = v["braid_transfer_coverage_same_events"]
        xfit = v["braid_crossfit_coverage_same_events"]
        axA.plot([nat, xfer], [y, y], color=FS.INK, lw=1.0, zorder=2, alpha=0.6)
        axA.scatter([nat], [y], s=70, color=FS.hero_color(name), zorder=3,
                    edgecolor="white", linewidth=0.7)
        axA.scatter([xfit], [y], s=58, facecolor="none", edgecolor=FS.HERO,
                    linewidth=1.0, marker="D", zorder=3, alpha=0.7)
        axA.scatter([xfer], [y], s=85, color=FS.HERO, zorder=4, marker="D",
                    edgecolor="white", linewidth=0.7)
        axA.annotate(f"{nat:.2f}", (nat, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=6, color=FS.INK)
        axA.annotate(f"{xfer:.2f}", (xfer, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=6, color=FS.HERO)
        axA.text(0.02, y - 0.34, f"n={v['n_confident']} confident calls",
                 fontsize=5.6, color="#666", va="center")
    axA.set_yticks(ys)
    axA.set_yticklabels(CALLERS)
    axA.set_ylim(-0.7, len(CALLERS) - 0.3)
    axA.set_xlim(0.0, 1.02)
    axA.set_xlabel("Coverage of RT-PCR truth on the caller's own\n"
                   "padj<0.05 AND |ΔPSI|≥0.1 confident calls")
    axA.set_title("Strict filtering does not calibrate the interval",
                  fontsize=7.6, pad=6)
    # legend proxies, placed BELOW the axis so they never overlap the data rows
    axA.scatter([], [], s=70, color=FS.NATIVE, edgecolor="white",
                label="caller native")
    axA.scatter([], [], s=80, color=FS.HERO, marker="D", edgecolor="white",
                label="BRAID transfer (no RT-PCR here)")
    axA.scatter([], [], s=58, facecolor="none", edgecolor=FS.HERO, marker="D",
                label="BRAID cross-fit (uses RT-PCR, ref.)")
    axA.legend(loc="upper center", fontsize=5.4, frameon=False, ncol=1,
               handletextpad=0.4, bbox_to_anchor=(0.5, -0.30))
    FS.panel_letter(axA, "A", x=-0.18)

    # ---------- Panel B: detection precision-recall frontier -------------------
    # Best-MCC is folded into the legend label (not annotated inline) so the three
    # callers' near-coincident best points do not overprint each other.
    for name in CALLERS:
        col = FS.MUTED[name]
        cells = [(c["recall"], c["precision"]) for c in grid[name]
                 if c["recall"] is not None and c["precision"] is not None
                 and np.isfinite(c["recall"]) and np.isfinite(c["precision"])]
        rs = [r for r, _ in cells]
        ps = [p for _, p in cells]
        axB.scatter(rs, ps, s=10, color=col, alpha=0.28, zorder=2,
                    edgecolor="none")
        front = _pareto_front(cells)
        if front:
            axB.plot([r for r, _ in front], [p for _, p in front],
                     color=col, lw=1.3, zorder=3,
                     label=f"{name}  (best MCC {best[name]['mcc']:.2f})")
        bm = best[name]
        axB.scatter([bm["recall"]], [bm["precision"]], s=42, color=col, zorder=4,
                    edgecolor=FS.INK, linewidth=0.7)
    axB.scatter([braid_full["recall"]], [braid_full["precision"]], s=165,
                marker="*", color=FS.HERO, zorder=6, edgecolor="white",
                linewidth=0.8, label=f"BRAID tier  (MCC {braid_full['mcc']:.2f})")
    axB.set_xlabel("Recall (RT-PCR positives recovered)")
    axB.set_ylabel("Precision")
    axB.set_xlim(0.0, 1.0)
    axB.set_ylim(0.55, 1.02)
    axB.set_title("Detection: parity, so calibration comes free",
                  fontsize=7.6, pad=6)
    axB.legend(loc="lower left", fontsize=5.6, frameon=False, ncol=1,
               handletextpad=0.4)
    FS.panel_letter(axB, "B", x=-0.16)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FS.save_all(fig, str(OUT_DIR / STEM))
    if PAPER_DIR.exists():
        FS.save_all(fig, str(PAPER_DIR / STEM))
    plt.close(fig)


if __name__ == "__main__":
    main()
