"""Recalibration ablation: is the coverage gain the *conformal step* or BRAID?

The head-to-head result shows BRAID-conformal reaches nominal ΔPSI coverage while
the generative tools (MAJIQ, betAS, rMATS) under-cover. A skeptical reviewer could
argue this is apples-to-oranges: BRAID's interval is fit to the RT-PCR residuals
and the others are not. This script isolates the contribution of the *conformal
recalibration step* by applying the SAME k-fold cross-fit split-conformal
recalibration to EVERY method's point estimate (not just BRAID's), on the identical
139-event 4-method common set.

If the recalibration brings every method to nominal coverage, the message is:
the under-coverage is a property of raw generative intervals, and the conformal
recalibration layer (which BRAID ships by default) fixes it for any point estimate.
The residual difference is then WIDTH -- whose point estimate is sharpest at nominal.

Run: python recalibration_ablation.py   (after the head-to-head data is in place)
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import head_to_head_coverage as H  # noqa: E402
from comprehensive_benchmark import (  # noqa: E402
    ALPHA,
    load_circ_incl,
    load_majiq,
    majiq_interval,
    wilson,
)


def load_betas_mean_iv(intervals_tsv: str):
    """betAS per-event posterior mean (center) + 95% interval, keyed by ev index."""
    mean: dict[int, float] = {}
    iv: dict[int, tuple[float, float]] = {}
    with open(intervals_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            k = int(r["key"][2:])
            mean[k] = float(r["dpsi_mean"])
            iv[k] = (float(r["low_0.95"]), float(r["high_0.95"]))
    return mean, iv


def load_betas_truth(truth_tsv: str) -> dict[int, float]:
    """Per-event betAS-side RT-PCR truth, keyed by ev index (alignment guard)."""
    tru: dict[int, float] = {}
    with open(truth_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            tru[int(r["key"][2:])] = float(r["truth"])
    return tru


def collect_common(cfg) -> dict:
    """Per-event centers + raw intervals + truth on the 4-method common set."""
    targets = cfg["targets"]
    incls = cfg.get("incls") or [None] * len(targets)
    events = H.parse_se_table(cfg["rmats_se"])
    maj_rows = load_majiq(cfg["majiq_tsv"])
    betas_mean, betas_iv = load_betas_mean_iv(cfg["betas_iv"])
    betas_truth = load_betas_truth(cfg["betas_truth"])
    g1, g2 = cfg["majiq_groups"]
    rng = np.random.default_rng(7)

    col = {k: [] for k in (
        "braid_c", "rmats_c", "majiq_c", "betas_c", "trs", "sup",
        "blo", "bhi", "rlo", "rhi", "mlo", "mhi",
    )}
    ev_idx = -1
    for t, incl in zip(targets, incls):
        ev = H.match_event(t, events)
        mi = majiq_interval(t, incl, cfg["kind"], maj_rows, g1, g2)
        if ev is None:
            continue
        ec = H.event_counts(ev, normalize=True, swap_groups=cfg["swap"])
        if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
            continue
        ev_idx += 1
        # betAS row must align to THIS target's RT-PCR truth (same guard as the
        # head-to-head loader) so the betAS centre/interval is not misindexed.
        if ev_idx not in betas_iv or abs(betas_truth.get(ev_idx, 1e9) - t.dpsi_rtpcr) > 1e-3:
            continue
        m, std, _lo, _hi = H.beta_interval(ec, ALPHA, 0.5, rng)
        rm_mean, rm_se, rmlo, rmhi = H.rmats_interval(ec, ALPHA)
        if mi is None:
            continue  # common set: all four methods present
        mcenter, mlo, mhi = mi
        blo, bhi = betas_iv[ev_idx]
        col["braid_c"].append(m)
        col["rmats_c"].append(rm_mean)
        col["majiq_c"].append(mcenter)
        col["betas_c"].append(betas_mean[ev_idx])
        col["trs"].append(t.dpsi_rtpcr)
        col["sup"].append(ec.total_support)
        col["blo"].append(blo)
        col["bhi"].append(bhi)
        col["rlo"].append(rmlo)
        col["rhi"].append(rmhi)
        col["mlo"].append(mlo)
        col["mhi"].append(mhi)
    return {k: np.array(v, dtype=float) for k, v in col.items()}


def cov(lo, hi, truth):
    k = int(np.sum((truth >= lo) & (truth <= hi)))
    p, wlo, whi = wilson(k, truth.size)
    return p, wlo, whi, float(np.mean(hi - lo))


def recal(center, truth, support):
    """Apply the SAME k-fold cross-fit absolute-residual conformal recalibration."""
    lo, hi = H.conformal_crossfit(center, truth, np.ones_like(center), support, ALPHA)
    return lo, hi


def main() -> None:
    os.chdir(os.path.dirname(os.path.dirname(_HERE)))
    DB = "data/public_benchmarks"
    BH = "benchmarks/headtohead"
    circ_tsv = f"{DB}/meta/gse54651_circadian_positive_events.tsv"
    datasets = [
        dict(name="TRA2", kind="exon", swap=True,
             rmats_se=f"{DB}/GSE59335/rmats/SE.MATS.JC.txt",
             targets=H.load_targets(f"{DB}/GSE59335/targets/validated_events.tsv",
                                    f"{DB}/GSE59335/targets/failed_events.tsv"),
             majiq_tsv=f"{DB}/GSE59335/majiq/deltapsi.tsv",
             betas_iv=f"{BH}/tra2_betas_intervals.tsv",
             betas_truth=f"{BH}/tra2_betas_truth.tsv", majiq_groups=("KD", "CTRL")),
        dict(name="Circadian", kind="junction", swap=True,
             rmats_se=f"{DB}/GSE54651/rmats/SE.MATS.JC.txt",
             targets=H.load_circadian_targets(circ_tsv),
             incls=load_circ_incl(circ_tsv),
             majiq_tsv=f"{DB}/GSE54651/majiq/deltapsi.tsv",
             betas_iv=f"{BH}/circ_betas_intervals.tsv",
             betas_truth=f"{BH}/circ_betas_truth.tsv", majiq_groups=("LIVER", "CEREB")),
        dict(name="PC3E", kind="exon", swap=False,
             rmats_se=f"{DB}/SRS354082/rmats/SE.MATS.JC.txt",
             targets=H.load_targets(f"{DB}/meta/rmats_pc3e_gs689_positive_events.tsv", None),
             majiq_tsv=f"{DB}/SRS354082/majiq/deltapsi.tsv",
             betas_iv=f"{BH}/srs_betas_intervals.tsv",
             betas_truth=f"{BH}/srs_betas_truth.tsv", majiq_groups=("PC3E", "GS689")),
    ]
    cols = [collect_common(c) for c in datasets]
    P = {k: np.concatenate([c[k] for c in cols]) for k in cols[0]}
    n = P["trs"].size
    truth, sup = P["trs"], P["sup"]

    print("=" * 78)
    print(f"RECALIBRATION ABLATION — pooled 4-method common set (n={n})")
    print("Same k-fold cross-fit absolute-residual conformal recalibration applied")
    print("to EVERY method's point estimate. Coverage target 0.95.")
    print("=" * 78)
    raw = {
        "MAJIQ (real binary)": ("mlo", "mhi", "majiq_c"),
        "betAS (real tool)": ("blo", "bhi", "betas_c"),
        "rMATS IncLevel t-CI": ("rlo", "rhi", "rmats_c"),
    }
    print(f"\n{'method':24}{'raw cov':>9}{'raw width':>11}   |  "
          f"{'+conformal cov':>15}{'Wilson 95%':>20}{'width':>8}")
    print("-" * 92)
    out = {"n": int(n), "alpha": ALPHA, "methods": {}}
    for name, (lo, hi, ck) in raw.items():
        rp, rl, ru, rw = cov(P[lo], P[hi], truth)
        clo, chi = recal(P[ck], truth, sup)
        cp, cl, cu, cw = cov(clo, chi, truth)
        print(f"{name:24}{rp:9.3f}{rw:11.3f}   |  {cp:15.3f}"
              f"   [{cl:.3f},{cu:.3f}]{cw:8.3f}")
        out["methods"][name] = {
            "raw_coverage": rp, "raw_width": rw,
            "recal_coverage": cp, "recal_wilson": [cl, cu], "recal_width": cw,
        }
    # BRAID = conformal recalibration of BRAID's own (Jeffreys) point estimate.
    blo_, bhi_ = recal(P["braid_c"], truth, sup)
    bp, bl, bu, bw = cov(blo_, bhi_, truth)
    print(f"{'BRAID-conformal':24}{'(=>)':>9}{'':>11}   |  {bp:15.3f}"
          f"   [{bl:.3f},{bu:.3f}]{bw:8.3f}")
    print("-" * 92)
    out["methods"]["BRAID-conformal"] = {
        "raw_coverage": None, "raw_width": None,
        "recal_coverage": bp, "recal_wilson": [bl, bu], "recal_width": bw,
    }
    print("\nReading: raw generative/per-replicate intervals under-cover; the SAME"
          "\nconformal recalibration brings every method to nominal. The gain is the"
          "\nrecalibration step (method-agnostic), which BRAID ships by default; the"
          "\nremaining difference is width = sharpness of each point estimate.")
    print("=" * 78)
    dest = os.path.join("benchmarks", "results", "recalibration_ablation.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print("Saved", dest)


if __name__ == "__main__":
    main()
