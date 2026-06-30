#!/usr/bin/env python3
"""Cross-dataset conformal transfer: fit on one dataset, deploy on another (no refit).

This is the decisive generalization test. A skeptic can dismiss within-dataset
cross-fit coverage as "calibrated to its own residuals". So here we FIT the
absolute-residual Mondrian conformal quantile on dataset A only, then DEPLOY it
unchanged on dataset B, and measure whether nominal coverage still holds. betAS is
parametric (no fit), so its coverage is the natural out-of-the-box baseline -- a
fair head-to-head of "deploy a calibrator trained elsewhere" vs "trust the Beta".

Reports, per direction, at nominal 95%: empirical coverage and mean width for
transferred-conformal vs the real-betAS baseline on the held-out dataset.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import head_to_head_coverage as H  # noqa: E402

from braid.target.conformal import (  # noqa: E402
    assign_support_bins,
    conformal_quantile,
)


def build_arrays(rmats_se, targets, *, swap_groups, seed=7):
    """Return (points, truths, supports) ΔPSI arrays for a dataset."""
    events = H.parse_se_table(rmats_se)
    rng = np.random.default_rng(seed)
    pts, trs, sup = [], [], []
    for t in targets:
        ev = H.match_event(t, events)
        if ev is None:
            continue
        ec = H.event_counts(ev, normalize=True, swap_groups=swap_groups)
        if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
            continue
        m, _s, _lo, _hi = H.beta_interval(ec, 0.05, 0.5, rng)
        pts.append(m)
        trs.append(t.dpsi_rtpcr)
        sup.append(ec.total_support)
    return np.array(pts), np.array(trs), np.array(sup)


def fit_abs_conformal(points, truths, supports, alpha, edges=(20, 50, 100, 250)):
    """Fit absolute-residual Mondrian conformal quantiles q_global + q_by_bin."""
    resid = np.abs(truths - points)
    bins = assign_support_bins(supports, edges)
    q_global = conformal_quantile(resid, alpha)
    q_by_bin = {}
    for b in np.unique(bins):
        m = bins == b
        qb = conformal_quantile(resid[m], alpha)
        q_by_bin[str(b)] = qb if np.isfinite(qb) else q_global
    return q_global, q_by_bin


def apply_abs_conformal(points, supports, q_global, q_by_bin, edges=(20, 50, 100, 250)):
    bins = assign_support_bins(supports, edges)
    half = np.array([q_by_bin.get(str(b), q_global) for b in bins])
    half = np.where(np.isfinite(half), half, q_global)
    low = np.clip(points - half, -1.0, 1.0)
    high = np.clip(points + half, -1.0, 1.0)
    return low, high


def cov_width(low, high, truth):
    cov = float(np.mean((truth >= low) & (truth <= high)))
    width = float(np.mean(high - low))
    return cov, width


def betas_baseline(betas_tsv, truths):
    """cov@95 / width@95 of the real-betAS intervals on a dataset (row-order keyed)."""
    n = truths.size
    bi = H._load_betas_intervals(betas_tsv, n)
    lo, hi, _ = bi["0.95"]
    return cov_width(lo, hi, truths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tra2-se", required=True)
    ap.add_argument("--tra2-validated", required=True)
    ap.add_argument("--tra2-failed", required=True)
    ap.add_argument("--tra2-betas", required=True)
    ap.add_argument("--circ-se", required=True)
    ap.add_argument("--circ-tsv", required=True)
    ap.add_argument("--circ-betas", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    tra2_t = H.load_targets(args.tra2_validated, args.tra2_failed)
    circ_t = H.load_circadian_targets(args.circ_tsv)

    pt_a, tr_a, su_a = build_arrays(args.tra2_se, tra2_t, swap_groups=True)
    pt_b, tr_b, su_b = build_arrays(args.circ_se, circ_t, swap_groups=True)

    lvl = 1 - args.alpha
    print(f"\n{'='*78}")
    print(f"CROSS-DATASET CONFORMAL TRANSFER  (nominal {lvl:.0%})")
    print(f"  TRA2 n={pt_a.size}  |  Circadian n={pt_b.size}")
    print(f"{'-'*78}")
    print(f"{'fit -> deploy':<26}{'method':<22}{'coverage':>10}{'width':>9}")
    print(f"{'-'*78}")

    # TRA2 -> Circadian
    qg, qb = fit_abs_conformal(pt_a, tr_a, su_a, args.alpha)
    lo, hi = apply_abs_conformal(pt_b, su_b, qg, qb)
    c, w = cov_width(lo, hi, tr_b)
    print(f"{'TRA2 -> Circadian':<26}{'conformal-abs (transfer)':<22}{c:>10.3f}{w:>9.3f}")
    cb, wb = betas_baseline(args.circ_betas, tr_b)
    print(f"{'(on Circadian)':<26}{'betAS(real, no-fit)':<22}{cb:>10.3f}{wb:>9.3f}")

    # Circadian -> TRA2
    qg2, qb2 = fit_abs_conformal(pt_b, tr_b, su_b, args.alpha)
    lo2, hi2 = apply_abs_conformal(pt_a, su_a, qg2, qb2)
    c2, w2 = cov_width(lo2, hi2, tr_a)
    print(f"{'Circadian -> TRA2':<26}{'conformal-abs (transfer)':<22}{c2:>10.3f}{w2:>9.3f}")
    cb2, wb2 = betas_baseline(args.tra2_betas, tr_a)
    print(f"{'(on TRA2)':<26}{'betAS(real, no-fit)':<22}{cb2:>10.3f}{wb2:>9.3f}")
    print(f"{'='*78}")
    print("global conformal-abs half-widths:  "
          f"TRA2-fit q={qg:.3f}   Circadian-fit q={qg2:.3f}")


if __name__ == "__main__":
    main()
