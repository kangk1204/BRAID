"""Confidence-tier validation on the TRA2 RT-PCR panel (Supplementary Table 5 / S6 Fig).

Does the BRAID confidence tier stratify orthogonal RT-PCR agreement? On the only
head-to-head dataset with a matched negative panel -- TRA2 / GSE59335, 76 RT-PCR
positive + 36 RT-PCR negative cassette-exon targets -- assign each event the
production ``confidence_tier`` and report, per tier: n, empirical RT-PCR coverage
(the calibrated interval contains the RT-PCR Delta PSI), the RT-PCR positive rate,
and the mean absolute effect.

Reuses ``detection_filter_sweep.build`` (the canonical matched-row reconstruction) so
the events, intervals, and labels are identical to S1 Fig. Three interval definitions
are reported so the stratification is shown to be robust to the calibration choice:
  * cross-fit  -- leakage-free k-fold conformal-abs (uses TRA2 truth; the panel the
                  headline TRA2 coverage is computed on),
  * transfer   -- absolute-residual quantile fit on circadian RT-PCR ONLY, deployed
                  on TRA2 unchanged (never sees TRA2 truth; the honest no-RT-PCR view),
  * default    -- the shipped ``braid differential`` calibrator (out-of-the-box).

Caveats (kept in the manuscript): the per-tier coverage is empirical, not a transferred
conformal guarantee (each tier is a post-hoc selected subset); the ``supported`` tier is
tiny on TRA2 because rMATS flags almost every large effect; this is a TRA2-only check.

Run:  cd benchmarks/headtohead && python per_tier_validation.py
"""
from __future__ import annotations

import json
import os

import detection_filter_sweep as D
import numpy as np

from braid.adapters.base import confidence_tier

EFFECT_CUTOFF = 0.10
FDR_THRESHOLD = 0.05
TIER_ORDER = ("high-confidence", "supported", "caller-significant-only", "not-significant")
INTERVALS = {
    "cross_fit": ("braid_lo", "braid_hi"),
    "transfer": ("braid_transfer_lo", "braid_transfer_hi"),
    "default": ("braid_default_lo", "braid_default_hi"),
}
OUT_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "per_tier_validation.json",
)


def _tier_for(row: dict, lo_key: str, hi_key: str) -> tuple[str, bool]:
    """Production tier + whether the interval covers the RT-PCR truth."""
    lo, hi = row[lo_key], row[hi_key]
    reliable = bool(lo > 0.0 or hi < 0.0)
    effect = bool(abs(row["bj_mean"]) >= EFFECT_CUTOFF)
    caller_sig = bool(row["rmats_fdr"] < FDR_THRESHOLD)
    tier = confidence_tier(reliable, effect, caller_sig)
    covered = bool(lo <= row["truth"] <= hi)
    return tier, covered


def _stratify(rows: list[dict], lo_key: str, hi_key: str) -> dict:
    per_tier: dict[str, dict] = {}
    # assigned: list of (tier, covered, row) for every event under this interval.
    assigned = [(*_tier_for(r, lo_key, hi_key), r) for r in rows]
    for tier in TIER_ORDER:
        members = [(cov, r) for (t, cov, r) in assigned if t == tier]
        if not members:
            per_tier[tier] = {"n": 0}
            continue
        covs = [cov for cov, _r in members]
        g = [r for _cov, r in members]
        per_tier[tier] = {
            "n": len(g),
            "rtpcr_coverage": float(np.mean(covs)),
            "rtpcr_positive_rate": float(np.mean([r["is_positive"] for r in g])),
            "mean_abs_truth": float(np.mean([abs(r["truth"]) for r in g])),
            "mean_abs_dpsi": float(np.mean([abs(r["bj_mean"]) for r in g])),
        }
    covs_all = [cov for (_t, cov, _r) in assigned]
    per_tier["ALL"] = {
        "n": len(rows),
        "rtpcr_coverage": float(np.mean(covs_all)),
        "rtpcr_positive_rate": float(np.mean([r["is_positive"] for r in rows])),
    }
    # Monotonicity: positive rate must rank high-confidence above not-significant.
    hc = per_tier["high-confidence"].get("rtpcr_positive_rate")
    ns = per_tier["not-significant"].get("rtpcr_positive_rate")
    per_tier["_positive_rate_monotone_hc_gt_ns"] = bool(
        hc is not None and ns is not None and hc > ns)
    return per_tier


def main() -> None:
    data = D.build()
    rows = data["rows"]
    out = {
        "dataset": "TRA2 (GSE59335, human)",
        "n": int(data["n"]),
        "n_positive": int(data["n_pos"]),
        "n_negative": int(data["n_neg"]),
        "effect_cutoff": EFFECT_CUTOFF,
        "fdr_threshold": FDR_THRESHOLD,
        "tier_order": list(TIER_ORDER),
        "interval_variants": {
            name: _stratify(rows, lo, hi) for name, (lo, hi) in INTERVALS.items()
        },
        "source": "benchmarks/headtohead/per_tier_validation.py "
                  "(reuses detection_filter_sweep.build; TRA2 76 pos + 36 neg)",
        "caveats": (
            "Per-tier coverage is empirical, not a transferred conformal guarantee "
            "(each tier is a post-hoc selected subset). The 'supported' tier is tiny on "
            "TRA2 because rMATS flags almost every large effect. TRA2-only."
        ),
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=1)

    cf = out["interval_variants"]["cross_fit"]
    print(f"per-tier validation -> {OUT_JSON}")
    print(f"  TRA2 n={out['n']} pos={out['n_positive']} neg={out['n_negative']}")
    hdr = f"  {'tier':26}{'n':>4}{'cov':>8}{'pos_rate':>10}"
    print(hdr)
    for tier in TIER_ORDER:
        s = cf[tier]
        if s["n"] == 0:
            print(f"  {tier:26}{0:>4}")
            continue
        print(f"  {tier:26}{s['n']:>4}{s['rtpcr_coverage']:>8.3f}"
              f"{s['rtpcr_positive_rate']:>10.2f}")
    mono = all(out["interval_variants"][v]["_positive_rate_monotone_hc_gt_ns"]
               for v in INTERVALS)
    print(f"  positive-rate monotone (high-confidence > not-significant) in all "
          f"3 interval variants: {mono}")


if __name__ == "__main__":
    main()
