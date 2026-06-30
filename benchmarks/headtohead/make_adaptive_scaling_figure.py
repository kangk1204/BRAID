"""Render the adaptive-scaling negative-result figure (S2 Fig).

Source: ``benchmarks/results/adaptive_conformal_eval.json`` (n = 196 rMATS-matched
events). The figure documents why BRAID uses a single calibrated constant band
rather than per-event adaptive interval widths.

Three panels:
  (A) Heteroscedasticity check: Spearman rho between each candidate per-event
      scale and the absolute RNA-seq-to-RT-PCR residual (all |rho| <= 0.27), with
      the learned-model cross-validated R^2 annotated.
  (B) Interval score per scaling (lower is better): the constant band is optimal.
  (C) Width-vs-coverage frontier: the constant band is sharp while staying at or
      above nominal coverage; adaptive scalings either inflate width or trade away
      coverage.

Usage::

    python benchmarks/headtohead/make_adaptive_scaling_figure.py
"""

# ruff: noqa: I001
from __future__ import annotations

import csv
import json
import math
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

SRC = os.path.join(_REPO, "benchmarks/results/adaptive_conformal_eval.json")
OUT_DIR = os.path.join(_REPO, "outputs/figures/manuscript/S2")
PAPER_DIR = os.path.join(_REPO, "paper/figures")
STEM = "supp_fig2_adaptive_scaling"

# Display labels and plot order for the candidate per-event scales (Panel A).
HETERO_LABELS = {
    "posterior_std": "posterior SD",
    "rMATS_SE": "rMATS SE",
    "log_support": "log support",
    "psi_extremity": "PSI extremity",
}

# Display labels for the evaluated scalings (Panels B/C); first entry is constant.
SCALE_LABELS = {
    "ones (baseline, constant)": "constant band",
    "posterior_std": "posterior SD",
    "rMATS_SE": "rMATS SE",
    "max(pstd, rMATS_SE)": "max(SD, SE)",
    "learned-linear (cross-fit)": "learned linear",
    "learned-RF (cross-fit)": "learned RF",
}
CONSTANT_KEY = "ones (baseline, constant)"
HIGHLIGHT = "#1b6ca8"
NEUTRAL = "#9aa7b1"


def _save(fig: plt.Figure) -> list[str]:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)
    written: list[str] = []
    for ext in ("png", "pdf", "svg"):
        for base in (os.path.join(OUT_DIR, STEM), os.path.join(PAPER_DIR, STEM)):
            path = f"{base}.{ext}"
            fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
            written.append(os.path.relpath(path, _REPO))
    return written


def _write_source_data(d: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "s2_source_data.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "key", "metric", "value"])
        for k, v in d["hetero_spearman"].items():
            w.writerow(["A", k, "spearman_rho_abs_residual", f"{v:.6f}"])
        w.writerow(["A", "learned_model", "rf_difficulty_cv_r2", f"{d['rf_difficulty_cv_r2']:.6f}"])
        for k, s in d["scales"].items():
            for metric in ("coverage", "width", "iscore"):
                w.writerow(["B/C", k, metric, f"{s[metric]:.6f}"])
    return os.path.relpath(path, _REPO)


def _panel_a(ax: plt.Axes, d: dict) -> None:
    items = [(HETERO_LABELS[k], d["hetero_spearman"][k]) for k in HETERO_LABELS]
    labels = [lab for lab, _ in items]
    vals = [v for _, v in items]
    y = range(len(items))
    # Data-derived band: the smallest 0.01 bound that contains every |rho|, so the
    # shaded region cannot understate the largest correlation (was a hardcoded 0.23
    # while psi_extremity reaches |rho| = 0.26).
    bound = math.ceil(max(abs(v) for v in vals) * 100) / 100
    ax.axvspan(-bound, bound, color="#dde6ec", zorder=0, label=f"|rho| <= {bound:.2f}")
    ax.barh(list(y), vals, color=NEUTRAL, edgecolor="#5b6b75", height=0.6, zorder=2)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(-0.5, 0.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"Spearman $\rho$ ( scale , |residual| )")
    ax.set_title("(A) No exploitable heteroscedasticity")
    ax.text(
        0.97, 0.06,
        f"learned model cross-val $R^2$ = {d['rf_difficulty_cv_r2']:.2f}\n"
        "(no better than a constant)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#b03a2e",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#b03a2e", alpha=0.9),
    )
    ax.legend(fontsize=7, loc="upper left", frameon=False)


def _ordered_scales(d: dict) -> list[str]:
    keys = list(d["scales"])
    # constant first, then remaining by ascending interval score
    rest = sorted((k for k in keys if k != CONSTANT_KEY), key=lambda k: d["scales"][k]["iscore"])
    return [CONSTANT_KEY] + rest


def _panel_b(ax: plt.Axes, d: dict) -> None:
    keys = _ordered_scales(d)
    labels = [SCALE_LABELS[k] for k in keys]
    vals = [d["scales"][k]["iscore"] for k in keys]
    colors = [HIGHLIGHT if k == CONSTANT_KEY else NEUTRAL for k in keys]
    x = range(len(keys))
    ax.bar(list(x), vals, color=colors, edgecolor="#444", width=0.7)
    base = d["scales"][CONSTANT_KEY]["iscore"]
    ax.axhline(base, color=HIGHLIGHT, lw=1, ls=":")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x), labels, rotation=28, ha="right")
    ax.set_ylabel("interval score  (lower is better)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("(B) Constant band has the best interval score")


def _panel_c(ax: plt.Axes, d: dict) -> None:
    # Per-point label offsets (points) to avoid collisions where markers cluster.
    label_off = {
        "ones (baseline, constant)": (8, 6),
        "posterior_std": (6, 6),
        "rMATS_SE": (6, -12),
        "max(pstd, rMATS_SE)": (6, 6),
        "learned-linear (cross-fit)": (-4, 10),
        "learned-RF (cross-fit)": (6, -14),
    }
    for k in _ordered_scales(d):
        s = d["scales"][k]
        is_const = k == CONSTANT_KEY
        ax.scatter(
            s["width"], s["coverage"],
            s=160 if is_const else 70,
            color=HIGHLIGHT if is_const else NEUTRAL,
            edgecolor="#222", zorder=3, marker="*" if is_const else "o",
        )
        ax.annotate(
            SCALE_LABELS[k], (s["width"], s["coverage"]),
            textcoords="offset points", xytext=label_off.get(k, (6, 5)),
            fontsize=7, ha="left",
        )
    ax.axhline(0.95, color="#b03a2e", lw=1, ls="--")
    ax.text(ax.get_xlim()[1], 0.95, "nominal 0.95", fontsize=7, color="#b03a2e",
            ha="right", va="bottom")
    ax.set_xlabel("mean interval width")
    ax.set_ylabel("coverage @ 95%")
    ax.set_title("(C) Width-coverage frontier")


def main() -> None:
    with open(SRC, encoding="utf-8") as fh:
        d = json.load(fh)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    _panel_a(axes[0], d)
    _panel_b(axes[1], d)
    _panel_c(axes[2], d)
    fig.suptitle(
        f"Per-event adaptive widths do not beat a constant conformal band "
        f"(n = {d['n']} rMATS-matched events)",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    written = _save(fig)
    plt.close(fig)
    src_csv = _write_source_data(d)
    print(f"Wrote {len(written)} figure files + {src_csv}")
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
