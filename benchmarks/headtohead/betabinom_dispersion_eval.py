"""Phase 1 (lever #3): is the ad-hoc count-scale lambda a formal dispersion, or a
coverage heuristic for the orthogonal-truth residual floor?

BRAID's overdispersed posterior is Beta(inc*lambda + 1/2, exc*lambda + 1/2) with a
heuristic lambda = 0.01 (effective counts shrunk 100x). The manuscript flags lambda
as "ad hoc rather than derived from a formal overdispersion model". This script
estimates the formal Beta-Binomial dispersion rho from biological replicates
(method of moments on per-replicate PSI), converts it to the count-scale lambda_BB
it implies, and compares it to the heuristic 0.01.

Hypothesis (continuing the width-irreducibility finding): replicate-based dispersion
captures only WITHIN-RNA-seq overdispersion, so lambda_BB will be far larger (a
narrower posterior) than the coverage-tuned 0.01 -- i.e. the heuristic lambda is
aggressive precisely because it must cover the orthogonal-truth residual floor,
which a within-RNA-seq dispersion model cannot see. That both removes the
"ad hoc" criticism (by explaining it) and motivates the conformal default.

Run: python betabinom_dispersion_eval.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import head_to_head_coverage as H  # noqa: E402
from comprehensive_benchmark import load_circ_incl  # noqa: E402

OVERDISPERSED_LAMBDA = 0.01  # braid.target.psi_bootstrap.OVERDISPERSED_COUNT_SCALE


def replicate_psis(ijc: tuple, sjc: tuple) -> np.ndarray:
    """Per-replicate PSI = IJC/(IJC+SJC) for replicates with non-zero coverage."""
    ijc = np.asarray(ijc, float)
    sjc = np.asarray(sjc, float)
    tot = ijc + sjc
    ok = tot > 0
    return ijc[ok] / tot[ok], tot[ok]


def main() -> None:
    os.chdir(os.path.dirname(os.path.dirname(_HERE)))
    DB = "data/public_benchmarks"
    circ_tsv = f"{DB}/meta/gse54651_circadian_positive_events.tsv"
    datasets = [
        dict(name="TRA2", rmats_se=f"{DB}/GSE59335/rmats/SE.MATS.JC.txt",
             targets=H.load_targets(f"{DB}/GSE59335/targets/validated_events.tsv",
                                    f"{DB}/GSE59335/targets/failed_events.tsv")),
        dict(name="Circ", rmats_se=f"{DB}/GSE54651/rmats/SE.MATS.JC.txt",
             targets=H.load_circadian_targets(circ_tsv), incls=load_circ_incl(circ_tsv)),
        dict(name="PC3E", rmats_se=f"{DB}/SRS354082/rmats/SE.MATS.JC.txt",
             targets=H.load_targets(f"{DB}/meta/rmats_pc3e_gs689_positive_events.tsv", None)),
    ]
    # Collect per-event, per-group overdispersion: observed cross-replicate PSI
    # variance vs the Binomial expectation PSI(1-PSI)/n_mean.
    overdisp_ratios = []
    rho_mom = []
    for cfg in datasets:
        events = H.parse_se_table(cfg["rmats_se"])
        targets = cfg["targets"]
        incls = cfg.get("incls") or [None] * len(targets)
        for t, _ in zip(targets, incls):
            ev = H.match_event(t, events)
            if ev is None:
                continue
            for ijc, sjc in ((ev.ijc1, ev.sjc1), (ev.ijc2, ev.sjc2)):
                psis, tots = replicate_psis(ijc, sjc)
                if psis.size < 2:
                    continue
                p = float(psis.mean())
                if p <= 0 or p >= 1:
                    continue
                n_mean = float(tots.mean())
                var_obs = float(psis.var(ddof=1))
                var_bin = p * (1 - p) / n_mean
                if var_bin <= 0:
                    continue
                ratio = var_obs / var_bin
                overdisp_ratios.append(ratio)
                # method-of-moments rho: var_obs = p(1-p)/n * (1 + (n-1)rho)
                rho = (ratio - 1.0) / max(n_mean - 1.0, 1.0)
                rho_mom.append(max(rho, 0.0))
    overdisp_ratios = np.array(overdisp_ratios)
    rho_mom = np.array(rho_mom)
    med_ratio = float(np.median(overdisp_ratios))
    med_rho = float(np.median(rho_mom))

    # A Beta-Binomial that inflates the variance by `med_ratio` over Binomial is
    # equivalent (for posterior width) to shrinking the effective counts by that
    # same factor: lambda_BB ~ 1 / med_ratio. This is the formal-dispersion analogue
    # of BRAID's count-scale.
    lambda_bb = 1.0 / med_ratio if med_ratio > 0 else float("nan")

    print("=" * 78)
    print(f"FORMAL BETA-BINOMIAL DISPERSION vs HEURISTIC lambda  (n={overdisp_ratios.size} "
          "event-groups, >=2 replicates)")
    print("=" * 78)
    print(f"  Median overdispersion ratio (Var_obs / Var_Binomial) : {med_ratio:.2f}x")
    print(f"  Median Beta-Binomial intra-class rho (MoM)           : {med_rho:.4f}")
    aggr = lambda_bb / OVERDISPERSED_LAMBDA
    print(f"  Implied formal count-scale  lambda_BB = 1/ratio       : {lambda_bb:.3f}")
    print(f"  BRAID heuristic count-scale lambda                   : {OVERDISPERSED_LAMBDA:.3f}")
    print(f"  Ratio (heuristic is this much MORE aggressive)       : {aggr:.0f}x")
    print("-" * 78)
    print("Reading: the formal within-RNA-seq Beta-Binomial dispersion implies a")
    print(f"posterior ~{aggr:.0f}x NARROWER than the coverage-tuned heuristic.")
    print("The heuristic lambda is aggressive because it must cover the orthogonal-truth")
    print("residual floor (Results: ~94% of dPSI error variance outside read sampling),")
    print("which a within-RNA-seq dispersion model cannot capture -- the very reason the")
    print("conformal recalibration on RT-PCR residuals is the production default.")
    print("=" * 78)

    import json
    out = {
        "n_event_groups": int(overdisp_ratios.size),
        "median_overdispersion_ratio": med_ratio,
        "median_betabinom_rho": med_rho,
        "lambda_BB_formal": lambda_bb,
        "lambda_heuristic": OVERDISPERSED_LAMBDA,
        "heuristic_more_aggressive_x": lambda_bb / OVERDISPERSED_LAMBDA,
    }
    dest = "benchmarks/results/betabinom_dispersion_eval.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print("Saved", dest)


if __name__ == "__main__":
    main()
