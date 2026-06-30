#!/usr/bin/env python3
"""Why is BRAID's detection sensitivity low, and what is the right fix?

Measures sensitivity AND false-positive rate (TRA2: 76 RT-PCR positives + 44
RT-PCR negatives) for several candidate BRAID detection rules, to separate two
questions that the conformal *coverage* interval conflates:

  * COVERAGE: "where would the RT-PCR ΔPSI land?" -> wide (absorbs platform
    discordance). Correct for the reported interval; WRONG for detection.
  * DETECTION: "is there a real ΔPSI between the two RNA-seq conditions?" -> a
    within-platform question; platform discordance cancels, so it needs the
    within-RNA-seq uncertainty, not the RT-PCR-coverage interval.

Rules compared (each at its natural operating point, plus an FPR-matched point):
  - conformal_excl0  : conformal 95% interval excludes 0 (current high-conf tier)
  - jeffreys_excl0   : Jeffreys posterior 95% CI excludes 0 (sampling only)
  - prob_large       : P(|ΔPSI| > cutoff) >= 0.5  (Jeffreys posterior)
  - replicate_z      : |ΔPSI| / between-replicate SE > 1.96
  - rmats_fdr        : rMATS FDR < 0.05  (replicate-aware test BRAID wraps)
  - conformal_pdetect: conformal p-value calibrated on the negatives (FPR<=alpha
    guaranteed by exchangeability) using the replicate z as the score.

The last is the proposed fix: a detection rule with a distribution-free FPR
guarantee (the detection analogue of the coverage guarantee), recovering
sensitivity while controlling false positives.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import head_to_head_coverage as H  # noqa: E402

from braid.target.conformal import load_differential_conformal_calibrator  # noqa: E402

RMATS_SE = "data/public_benchmarks/GSE59335/rmats/SE.MATS.JC.txt"
VALID = "data/public_benchmarks/GSE59335/targets/validated_events.tsv"
FAILED = "data/public_benchmarks/GSE59335/targets/failed_events.tsv"
CUTOFF = 0.1
Z = 1.959963984540054


def build(seed=7):
    events = H.parse_se_table(RMATS_SE)
    targets = H.load_targets(VALID, FAILED)
    cal = load_differential_conformal_calibrator()
    rng = np.random.default_rng(seed)
    rows = []
    for t in targets:
        ev = H.match_event(t, events)
        if ev is None:
            continue
        ec = H.event_counts(ev, normalize=True, swap_groups=True)
        if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
            continue
        s = H._dpsi_samples(ec, 0.5, rng, n=8000)
        dpsi_mean = float(s.mean())
        lo_j = float(np.percentile(s, 2.5))
        hi_j = float(np.percentile(s, 97.5))
        prob_large = float(np.mean(np.abs(s) >= CUTOFF))
        clo, chi = cal.interval(dpsi_mean, 1.0, ec.total_support, clip=(-1.0, 1.0))
        # replicate-aware z from per-replicate IncLevels
        i1 = np.asarray(ec.inclevel_ctrl, float)
        i2 = np.asarray(ec.inclevel_treat, float)
        if i1.size > 1 and i2.size > 1:
            se = np.sqrt(i1.var(ddof=1) / i1.size + i2.var(ddof=1) / i2.size)
            se = max(se, 1e-3)
            zsc = abs(i1.mean() - i2.mean()) / se
        else:
            zsc = 0.0
        rows.append({
            "label": 1 if t.is_positive else 0,
            "dpsi": dpsi_mean,
            "jeff_excl0": (lo_j > 0) or (hi_j < 0),
            "conf_excl0": (clo > 0) or (chi < 0),
            "prob_large": prob_large,
            "rmats_fdr": ec.rmats_fdr,
            "zsc": zsc,
        })
    return rows


def sens_fpr(rows, predicate):
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    sens = np.mean([predicate(r) for r in pos]) if pos else float("nan")
    fpr = np.mean([predicate(r) for r in neg]) if neg else float("nan")
    return float(sens), float(fpr)


def conformal_pdetect(rows, alpha=0.05):
    """Conformal detection: calibrate the score threshold on negatives so the
    false-positive rate is <= alpha (finite-sample, by exchangeability). Score =
    replicate-aware z. p = (1 + #{neg_z >= test_z}) / (n_neg + 1); call if p<=alpha."""
    neg_z = np.array([r["zsc"] for r in rows if r["label"] == 0])
    n = neg_z.size

    def predicate(r):
        p = (1 + int(np.sum(neg_z >= r["zsc"]))) / (n + 1)
        return p <= alpha
    return predicate


def main():
    os.chdir(os.path.dirname(os.path.dirname(_HERE)))  # repo root
    rows = build()
    npos = sum(r["label"] == 1 for r in rows)
    nneg = sum(r["label"] == 0 for r in rows)
    print(f"\n{'='*72}")
    print(f"BRAID detection rules on TRA2  ({npos} positives, {nneg} negatives)")
    print(f"{'-'*72}")
    print(f"{'rule':<22}{'sensitivity':>13}{'FPR':>9}{'note':>26}")
    print(f"{'-'*72}")
    rules = [
        ("conformal_excl0", lambda r: r["conf_excl0"], "current high-conf (too strict)"),
        ("jeffreys_excl0", lambda r: r["jeff_excl0"], "sampling-only (over-calls)"),
        ("prob_large>=0.5", lambda r: r["prob_large"] >= 0.5, "Jeffreys posterior"),
        ("replicate_z>1.96", lambda r: r["zsc"] > Z, "between-replicate test"),
        ("rmats_fdr<0.05", lambda r: np.isfinite(r["rmats_fdr"]) and r["rmats_fdr"] < 0.05,
         "rMATS (replicate-aware)"),
        ("supported(fdr&plarge)",
         lambda r: np.isfinite(r["rmats_fdr"]) and r["rmats_fdr"] < 0.05 and r["prob_large"] >= 0.5,
         "BRAID 'supported' tier"),
    ]
    for name, pred, note in rules:
        s, f = sens_fpr(rows, pred)
        print(f"{name:<22}{s:>13.3f}{f:>9.3f}{note:>26}")
    # proposed conformal detection at FPR<=0.05 and 0.10
    for a in (0.05, 0.10):
        s, f = sens_fpr(rows, conformal_pdetect(rows, a))
        print(f"{'conf_pdetect@'+str(a):<22}{s:>13.3f}{f:>9.3f}{'PROPOSED (FPR guarantee)':>26}")
    print(f"{'-'*72}")
    print("real tools (SUPPA2/rMATS/MAJIQ) sensitivity on positives: 0.64-0.67 "
          "(FPR not in Table S5)")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
