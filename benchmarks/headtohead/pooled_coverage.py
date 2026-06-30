#!/usr/bin/env python3
"""Pooled coverage with Wilson CIs — the rigor a reviewer demands on small-n results.

Each dataset alone gives a coverage estimate with a WIDE binomial CI. Pooling
per-event coverage across datasets and reporting Wilson 95% CIs turns
"0.90-1.00 across small sets" into a better-powered supplemental statement.

Two pools:
  (1) HELD-OUT calibration coverage: TRA2 + circadian, BRAID conformal-abs via
      honest k-fold cross-fit (no in-sample leakage), vs real betAS and rMATS-perRep.
  (2) SUPPLEMENTAL TRANSFER coverage: shipped calibrator (q_global, NO refit) applied
      to author-reported rMATS ΔPSI on datasets it never saw. This is not the
      canonical SRS354082 real-betAS head-to-head; use comprehensive_benchmark.py for
      reviewer-facing coverage claims.
"""
from __future__ import annotations

import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import head_to_head_coverage as H  # noqa: E402

from braid.target.conformal import load_differential_conformal_calibrator  # noqa: E402


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def tra2_circ_events():
    """Per-event (truth, point, std, support, betAS lo/hi, rMATS lo/hi) for TRA2+circ."""
    specs = [
        ("data/public_benchmarks/GSE59335/rmats/SE.MATS.JC.txt",
         H.load_targets("data/public_benchmarks/GSE59335/targets/validated_events.tsv",
                        "data/public_benchmarks/GSE59335/targets/failed_events.tsv"),
         "tra2_betas_intervals.tsv"),
        ("data/public_benchmarks/GSE54651/rmats/SE.MATS.JC.txt",
         H.load_circadian_targets("data/public_benchmarks/meta/gse54651_circadian_positive_events.tsv"),
         "circ_betas_intervals.tsv"),
    ]
    rng = np.random.default_rng(7)
    pts, trs, std, sup, blo, bhi, rlo, rhi = [], [], [], [], [], [], [], []
    for se, targets, betas in specs:
        events = H.parse_se_table(se)
        rows = []
        for t in targets:
            ev = H.match_event(t, events)
            if ev is None:
                continue
            ec = H.event_counts(ev, normalize=True, swap_groups=True)
            if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
                continue
            m, s, _, _ = H.beta_interval(ec, 0.05, 0.5, rng)
            rm, _se, rl, rh = H.rmats_interval(ec, 0.05)
            rows.append((t.dpsi_rtpcr, m, s, ec.total_support, rl, rh))
        n = len(rows)
        bi = H._load_betas_intervals(os.path.join(_HERE, betas), n)
        blo_a, bhi_a, _ = bi["0.95"]
        for i, (tr, m, s, su, rl, rh) in enumerate(rows):
            trs.append(tr)
            pts.append(m)
            std.append(max(s, 1e-6))
            sup.append(su)
            blo.append(blo_a[i])
            bhi.append(bhi_a[i])
            rlo.append(rl)
            rhi.append(rh)
    return (np.array(trs), np.array(pts), np.array(std), np.array(sup),
            np.array(blo), np.array(bhi), np.array(rlo), np.array(rhi))


def pool_held_out():
    trs, pts, std, sup, blo, bhi, rlo, rhi = tra2_circ_events()
    lo, hi = H.conformal_crossfit(pts, trs, np.ones_like(std), sup, 0.05, seed=7)
    cov_conf = (trs >= lo) & (trs <= hi)
    cov_betas = (trs >= blo) & (trs <= bhi)
    cov_rmats = (trs >= rlo) & (trs <= rhi)
    n = trs.size
    print(f"\n{'='*74}\nPOOL 1 — held-out coverage, TRA2+circadian (n={n}), nominal 95%\n{'-'*74}")
    print(f"{'method':<26}{'covered':>9}{'coverage':>10}{'  Wilson 95% CI':<20}")
    for name, cov in [("BRAID conformal-abs (CV)", cov_conf),
                      ("betAS (real)", cov_betas),
                      ("rMATS-perRep", cov_rmats)]:
        k = int(cov.sum())
        p, c0, c1 = wilson(k, n)
        print(f"{name:<26}{k:>9}{p:>10.3f}   [{c0:.3f}, {c1:.3f}]")
    print(f"{'='*74}")


def transfer_event_cover(q):
    """Per-event coverage booleans for the shipped-calibrator transfer pool."""
    import pandas as pd
    covered = []
    src = []
    # SRS354082 (per-replicate PSI table)
    with open("data/public_benchmarks/meta/rmats_pc3e_gs689_positive_events.tsv") as f:
        import csv
        for r in csv.DictReader(f, delimiter="\t"):
            p1 = np.array([float(x) for x in r["pc3e_rnaseq_psi"].split(",")])
            p2 = np.array([float(x) for x in r["gs689_rnaseq_psi"].split(",")])
            dpsi = float(p1.mean() - p2.mean())
            truth = float(r["delta_psi_rtpcr"])
            covered.append(abs(truth - dpsi) <= q)
            src.append("SRS354082")
    # SUPPA supp prediction sheets: use the rMATS ΔPSI column
    xl = "data/public_benchmarks/meta/suppa2_tables.xlsx"
    for sheet, name in [("Table S5", "TRA2-S5"), ("Table S7", "circ-Zhang"),
                        ("Table S9", "Jurkat")]:
        df = pd.read_excel(xl, sheet_name=sheet, header=2)
        cols = {c.lower(): c for c in map(str, df.columns)}
        sc, dp, rt = cols.get("source"), cols.get("deltapsi"), cols.get("rt-pcr")
        if not (sc and dp and rt):
            continue
        sub = df[df[sc].astype(str).str.upper().str.startswith("RMATS")]
        d = pd.to_numeric(sub[dp], errors="coerce")
        t = pd.to_numeric(sub[rt], errors="coerce")
        for dd, tt in zip(d, t):
            if np.isfinite(dd) and np.isfinite(tt):
                covered.append(abs(tt - dd) <= q)
                src.append(name)
    return np.array(covered), np.array(src)


def pool_transfer():
    q = load_differential_conformal_calibrator().q_global
    cov, src = transfer_event_cover(q)
    n = cov.size
    print(f"\n{'='*74}\nPOOL 2 — TRANSFER coverage, shipped q={q:.3f} on rMATS ΔPSI, "
          f"NO refit (n={n})\n{'-'*74}")
    print(f"{'dataset':<16}{'n':>5}{'covered':>9}{'coverage':>10}{'  Wilson 95% CI':<20}")
    for name in ["SRS354082", "TRA2-S5", "circ-Zhang", "Jurkat"]:
        mask = src == name
        k = int(cov[mask].sum())
        nn = int(mask.sum())
        p, c0, c1 = wilson(k, nn)
        print(f"{name:<16}{nn:>5}{k:>9}{p:>10.3f}   [{c0:.3f}, {c1:.3f}]")
    k = int(cov.sum())
    p, c0, c1 = wilson(k, n)
    print(f"{'-'*74}")
    print(f"{'POOLED':<16}{n:>5}{k:>9}{p:>10.3f}   [{c0:.3f}, {c1:.3f}]")
    print(f"{'='*74}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(_HERE)))
    pool_held_out()
    pool_transfer()
