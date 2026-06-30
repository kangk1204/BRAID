"""Render the depth-titration figure from ``benchmarks/results/depth_titration.json``.

Three panels:
  (A) pooled ΔPSI coverage@95 vs sequencing-depth fraction, per method (H1).
  (B) pooled interval width@95 vs depth, per method (H2).
  (C) sampling-variance fraction vs depth, per dataset (H3).

Usage::

    python benchmarks/headtohead/make_depth_titration_figure.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

METHOD_STYLE = {
    "BRAID-conformal": ("#1b6ca8", "o", "-"),
    "BRAID-conformal-abs": ("#3aa0d1", "s", "-"),
    "rMATS-perRep": ("#d1495b", "^", "--"),
    "betAS": ("#edae49", "v", "--"),
    "BRAID-Jeffreys": ("#8d96a3", "D", ":"),
}


def main() -> None:
    src = os.path.join(_REPO, "benchmarks/results/depth_titration.json")
    with open(src, encoding="utf-8") as f:
        d = json.load(f)

    fracs = sorted(float(x) for x in d["config"]["fractions"])
    fkeys = [f"{x:.2f}" for x in fracs]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # Panel A: pooled coverage vs depth
    axa = axes[0]
    for m, (c, mk, ls) in METHOD_STYLE.items():
        ys = [d["pooled"].get(fk, {}).get(m, {}).get("coverage95") for fk in fkeys]
        axa.plot(fracs, ys, color=c, marker=mk, linestyle=ls, label=m, lw=2, ms=6)
    axa.axhline(0.95, color="k", lw=1, ls=":", alpha=0.7)
    axa.text(fracs[0], 0.955, "nominal 0.95", fontsize=8, va="bottom")
    axa.set_xlabel("library fraction (sequencing depth)")
    axa.set_ylabel("ΔPSI coverage @ 95%  (pooled)")
    axa.set_title("(A) Coverage vs depth — H1")
    axa.set_ylim(0.0, 1.02)
    axa.legend(fontsize=8, loc="lower left")

    # Panel B: pooled width vs depth
    axb = axes[1]
    for m, (c, mk, ls) in METHOD_STYLE.items():
        ys = [d["pooled"].get(fk, {}).get(m, {}).get("width95") for fk in fkeys]
        axb.plot(fracs, ys, color=c, marker=mk, linestyle=ls, label=m, lw=2, ms=6)
    axb.set_xlabel("library fraction (sequencing depth)")
    axb.set_ylabel("interval width @ 95%  (pooled)")
    axb.set_title("(B) Width vs depth — H2")
    axb.legend(fontsize=8, loc="upper right")

    # Panel C: sampling-variance fraction vs depth, per dataset
    axc = axes[2]
    palette = ["#1b6ca8", "#d1495b", "#edae49"]
    for (name, ds), col in zip(d["per_dataset"].items(), palette):
        ys = [ds.get(fk, {}).get("diagnostics", {}).get("sampling_var_fraction")
              for fk in fkeys]
        axc.plot(fracs, ys, color=col, marker="o", lw=2, ms=6, label=name)
    axc.set_xlabel("library fraction (sequencing depth)")
    axc.set_ylabel("sampling-variance fraction\n(posterior var / point-RMSE²)")
    axc.set_title("(C) Sampling fraction vs depth — H3")
    axc.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "BRAID depth titration — calibrated coverage is depth-robust; "
        "sampling-only intervals under-cover more as depth rises",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = os.path.join(_HERE, "figures", "fig_depth_titration.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    pdf = out[:-4] + ".pdf"
    fig.savefig(pdf)  # vector copy for the manuscript
    print(f"wrote {out}\nwrote {pdf}")


if __name__ == "__main__":
    main()
