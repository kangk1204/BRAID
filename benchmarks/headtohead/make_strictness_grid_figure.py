"""Strictness-grid figure: you cannot filter an rMATS operating point to coverage.

Sweeps the conventional rMATS confident-call grid -- significance (FDR < 0.10,
0.05, 0.01) crossed with an effect-size cutoff (|dPSI| >= 0.10, 0.20, 0.30) -- on
the one RT-PCR benchmark with both positives (76) and negatives (36), GSE59335
TRA2 (n = 112). For each cell it records the detection quality (Matthews
correlation against the RT-PCR calls) and two coverages of the RT-PCR dPSI: the
caller's own native interval, and BRAID's shipped conformal interval on exactly
the same selected events.

Message in one figure: stricter cutoffs trade away recall (MCC falls) and never
recalibrate the interval -- native coverage stays well below nominal at every
operating point -- whereas BRAID is at nominal on the same events. Filtering
selects events; it does not set the width needed to include the RNA-seq->RT-PCR
residual.

Real data only: rMATS table + RT-PCR targets are loaded through the exact head-
to-head loaders; nothing is synthesized. Source data is written to
benchmarks/results/strictness_grid.json and the figure to paper/figures/.
"""
from __future__ import annotations

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, os.path.join(_REPO, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figstyle  # noqa: E402
import head_to_head_coverage as H  # noqa: E402

from braid.target.conformal import load_differential_conformal_calibrator  # noqa: E402

figstyle.apply()

RMATS_SE = os.path.join(_REPO, "data/public_benchmarks/GSE59335/rmats/SE.MATS.JC.txt")
VALIDATED = os.path.join(_REPO, "data/public_benchmarks/GSE59335/targets/validated_events.tsv")
FAILED = os.path.join(_REPO, "data/public_benchmarks/GSE59335/targets/failed_events.tsv")
SRC_JSON = os.path.join(_REPO, "benchmarks/results/strictness_grid.json")
OUT_DIR = os.path.join(_REPO, "outputs/figures/manuscript/strictness_grid")
PAPER_DIR = os.path.join(_REPO, "paper/figures")
STEM = "strictness_grid"

FDRS = [0.10, 0.05, 0.01]
DPSIS = [0.10, 0.20, 0.30]
SUPPORT_FLOOR = 1
NOMINAL = 0.95
# BRAID detection operating point on this panel (S1 Fig): rMATS FDR<0.05 AND
# posterior P(|dPSI|>=0.1)>=0.5; reported for reference, not recomputed here.
BRAID_EFFECT_SUPPORTED_MCC = 0.564

C_RMATS = "#D55E00"   # Okabe-Ito orange
C_BRAID = "#0072B2"   # Okabe-Ito blue
C_NOMINAL = "#222222"


def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / den if den > 0 else float("nan")


def _build_rows() -> list[dict]:
    events = H.parse_se_table(RMATS_SE)
    targets = H.load_targets(VALIDATED, FAILED)
    rng = np.random.default_rng(7)
    cal = load_differential_conformal_calibrator()
    rows: list[dict] = []
    for t in targets:
        ev = H.match_event(t, events)
        if ev is None:
            continue
        ec = H.event_counts(ev, normalize=True, swap_groups=True)
        if (ec.a_c + ec.b_c) < SUPPORT_FLOOR or (ec.a_t + ec.b_t) < SUPPORT_FLOOR:
            continue
        bj_mean, _bj_std, _, _ = H.beta_interval(ec, 0.05, 0.5, rng)
        _rm, _rs, r_lo, r_hi = H.rmats_interval(ec, 0.05)
        b_lo, b_hi = cal.interval(float(bj_mean), 1.0, float(ec.total_support), clip=(-1.0, 1.0))
        rows.append({
            "truth": float(t.dpsi_rtpcr),
            "is_positive": bool(t.is_positive),
            "fdr": float(ec.rmats_fdr),
            "dpsi": float(ec.rmats_dpsi),
            "rmats_lo": float(r_lo), "rmats_hi": float(r_hi),
            "braid_lo": float(b_lo), "braid_hi": float(b_hi),
        })
    return rows


def _grid(rows: list[dict]) -> dict:
    npos = sum(r["is_positive"] for r in rows)
    nneg = len(rows) - npos
    cells = []
    for fdr in FDRS:
        for dc in DPSIS:
            called = [r for r in rows if r["fdr"] < fdr and abs(r["dpsi"]) >= dc]
            tp = sum(r["is_positive"] for r in called)
            fp = len(called) - tp
            fn = npos - tp
            tn = nneg - fp
            rcov = float(np.mean([r["rmats_lo"] <= r["truth"] <= r["rmats_hi"] for r in called])) \
                if called else float("nan")
            bcov = float(np.mean([r["braid_lo"] <= r["truth"] <= r["braid_hi"] for r in called])) \
                if called else float("nan")
            cells.append({
                "fdr": fdr, "dpsi_cut": dc, "n_call": len(called),
                "tp": tp, "fp": fp, "recall": tp / npos if npos else float("nan"),
                "mcc": _mcc(tp, fp, fn, tn),
                "rmats_coverage": rcov, "braid_coverage": bcov,
            })
    braid_full = float(np.mean([r["braid_lo"] <= r["truth"] <= r["braid_hi"] for r in rows]))
    return {
        "dataset": "GSE59335 TRA2 (RT-PCR positives + negatives)",
        "n_events": len(rows), "n_positive": npos, "n_negative": nneg,
        "nominal": NOMINAL, "fdr_levels": FDRS, "dpsi_cuts": DPSIS,
        "cells": cells,
        "braid_effect_supported_mcc": BRAID_EFFECT_SUPPORTED_MCC,
        "braid_full_panel_coverage": braid_full,
        "source": "benchmarks/headtohead/make_strictness_grid_figure.py",
    }


def _series(cells, fdr, key):
    by = {c["dpsi_cut"]: c[key] for c in cells if c["fdr"] == fdr}
    return [by[d] for d in DPSIS]


def _band(cells, key):
    """min/max across the three FDR levels at each dPSI cut (FDR-insensitivity band)."""
    lo, hi = [], []
    for d in DPSIS:
        vals = [c[key] for c in cells if c["dpsi_cut"] == d]
        lo.append(min(vals))
        hi.append(max(vals))
    return lo, hi


def _plot(grid: dict) -> plt.Figure:
    cells = grid["cells"]
    x = DPSIS
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.3))

    # --- Panel A: coverage gap ------------------------------------------------
    r_mid = _series(cells, 0.05, "rmats_coverage")
    b_mid = _series(cells, 0.05, "braid_coverage")
    r_lo, r_hi = _band(cells, "rmats_coverage")
    b_lo, b_hi = _band(cells, "braid_coverage")
    # the calibration gap a stricter cutoff cannot close
    axA.fill_between(x, r_mid, b_mid, color="#BBBBBB", alpha=0.30, zorder=1,
                     label="calibration gap")
    axA.fill_between(x, r_lo, r_hi, color=C_RMATS, alpha=0.18, zorder=2)
    axA.fill_between(x, b_lo, b_hi, color=C_BRAID, alpha=0.18, zorder=2)
    axA.axhline(NOMINAL, color=C_NOMINAL, ls="--", lw=0.9, zorder=3)
    axA.text(x[-1], NOMINAL + 0.006, "nominal 0.95", ha="right", va="bottom",
             fontsize=6, color=C_NOMINAL)
    axA.plot(x, b_mid, "-o", color=C_BRAID, lw=1.8, ms=5, zorder=5, label="BRAID (same events)")
    axA.plot(x, r_mid, "-o", color=C_RMATS, lw=1.8, ms=5, zorder=5, label="rMATS native interval")
    for xi, yi in zip(x, r_mid):
        axA.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, -11),
                     ha="center", fontsize=5.6, color=C_RMATS)
    for xi, yi in zip(x, b_mid):
        axA.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=5.6, color=C_BRAID)
    axA.set_xticks(x)
    axA.set_xlabel("rMATS effect-size cutoff  |ΔPSI| ≥", fontsize=7.5)
    axA.set_ylabel("Coverage of RT-PCR ΔPSI", fontsize=7.5)
    axA.set_ylim(0.45, 1.02)
    axA.set_title("A. No cutoff reaches nominal coverage", fontsize=8.5, fontweight="bold")
    axA.legend(loc="center left", fontsize=5.8, frameon=False)

    # --- Panel B: detection (MCC) --------------------------------------------
    m_mid = _series(cells, 0.05, "mcc")
    m_lo, m_hi = _band(cells, "mcc")
    axB.fill_between(x, m_lo, m_hi, color=C_RMATS, alpha=0.18, zorder=2)
    axB.plot(x, m_mid, "-o", color=C_RMATS, lw=1.8, ms=5, zorder=5,
             label="rMATS (FDR 0.10–0.01)")
    axB.axhline(grid["braid_effect_supported_mcc"], color=C_BRAID, ls="-", lw=1.6, zorder=4)
    axB.plot([x[0]], [grid["braid_effect_supported_mcc"]], marker="*", ms=11,
             color=C_BRAID, zorder=6, label="BRAID effect-supported")
    axB.text(x[-1], grid["braid_effect_supported_mcc"] + 0.012,
             f"BRAID {grid['braid_effect_supported_mcc']:.2f}", ha="right", va="bottom",
             fontsize=6, color=C_BRAID)
    for xi, yi in zip(x, m_mid):
        axB.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points", xytext=(0, -11),
                     ha="center", fontsize=5.6, color=C_RMATS)
    axB.set_xticks(x)
    axB.set_xlabel("rMATS effect-size cutoff  |ΔPSI| ≥", fontsize=7.5)
    axB.set_ylabel("Detection MCC vs RT-PCR", fontsize=7.5)
    axB.set_ylim(0.2, 0.7)
    axB.set_title("B. Stricter cutoffs only lose detection", fontsize=8.5, fontweight="bold")
    axB.legend(loc="upper right", fontsize=5.8, frameon=False)

    fig.suptitle(
        f"Filtering an rMATS operating point cannot reach calibrated coverage "
        f"(TRA2, n = {grid['n_events']}: {grid['n_positive']}+ / {grid['n_negative']}−)",
        fontsize=8.8, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    rows = _build_rows()
    grid = _grid(rows)
    os.makedirs(os.path.dirname(SRC_JSON), exist_ok=True)
    with open(SRC_JSON, "w", encoding="utf-8") as fh:
        json.dump(grid, fh, indent=1)
    fig = _plot(grid)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    for d in (OUT_DIR, PAPER_DIR):
        for ext in ("png", "pdf", "svg"):
            fig.savefig(os.path.join(d, f"{STEM}.{ext}"))
    plt.close(fig)
    print(f"n={grid['n_events']} pos={grid['n_positive']} neg={grid['n_negative']}")
    print(f"BRAID full-panel coverage = {grid['braid_full_panel_coverage']:.3f}")
    print(f"Wrote {STEM} (png/pdf/svg) to paper/figures + outputs, and the source JSON")


if __name__ == "__main__":
    main()
