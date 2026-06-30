#!/usr/bin/env python3
"""Does the differential calibrator's per-support-bin q schedule under-cover, and is
the non-monotonic q across bins a problem?

The shipped differential conformal calibrator carries per-support-bin quantiles whose
values are not monotone in depth (e.g. 100-249 -> 0.282 but 250+ -> 0.344). That is a
finite-sample tail-estimation effect: the RT-PCR calibration set (n=162) is dominated
by high-support cassette exons (250+ holds ~81%), so the sparse mid bins under-sample
the orthogonal-truth residual tail while the well-sampled 250+ bin recovers it. This
script asks the only question that matters operationally -- does the per-bin schedule
actually under-cover at deployment? -- by repeated K-fold cross-fitting three policies
on the real RT-PCR residual surface and reporting held-out per-bin coverage:

  A  per-bin (current)  : a quantile per support bin (rank>n falls back to global)
  B  min_bin_n=50       : a per-bin quantile only when the bin has >= 50 events
  C  global-only        : a single pooled quantile (no support stratification)

Conclusion (see the printed table / JSON): the current per-bin schedule meets nominal
coverage overall and in the evaluable bins (100-249 and 250+). The very sparse bins
(20-49 n=2, 50-99 n=6) are descriptive only; 50-99 under-covers for all three policies
and should not be used as a standalone calibration claim. The non-monotonic q is
therefore best treated as a finite-sample sampling artifact, not evidence that support
stratification improves coverage. Consistent with the homoscedastic, depth-independent
RNA-seq-to-RT-PCR discordance, the pooled global schedule (C) is marginally sharper at
similar overall coverage, so a near-constant width remains the honest default.
Read-only; writes benchmarks/results/.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import head_to_head_coverage as H  # noqa: E402
from cross_dataset_transfer import build_arrays  # noqa: E402

from braid.target.conformal import assign_support_bins, conformal_quantile  # noqa: E402

_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
DB = os.path.join(_REPO, "data", "public_benchmarks")
RESULTS = os.path.join(_REPO, "benchmarks", "results", "bin_policy_eval.json")
ALPHA = 0.05
EDGES = (20, 50, 100, 250)
K = 9
N_SEEDS = 20
BIN_ORDER = ["<20", "20-49", "50-99", "100-249", "250+"]


def _load() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tra2 = H.load_targets(
        os.path.join(DB, "GSE59335", "targets", "validated_events.tsv"),
        os.path.join(DB, "GSE59335", "targets", "failed_events.tsv"))
    circ = H.load_circadian_targets(
        os.path.join(DB, "meta", "gse54651_circadian_positive_events.tsv"))
    pa, ta, sa = build_arrays(
        os.path.join(DB, "GSE59335", "rmats", "SE.MATS.JC.txt"), tra2, swap_groups=True)
    pb, tb, sb = build_arrays(
        os.path.join(DB, "GSE54651", "rmats", "SE.MATS.JC.txt"), circ, swap_groups=True)
    return (np.concatenate([pa, pb]), np.concatenate([ta, tb]), np.concatenate([sa, sb]))


def _fit(points, truths, supports, *, min_bin_n: int, global_only: bool = False):
    resid = np.abs(truths - points)
    b = assign_support_bins(supports, EDGES)
    q_global = conformal_quantile(resid, ALPHA)
    q_by_bin: dict[str, float] = {}
    if not global_only:
        for u in np.unique(b):
            m = b == u
            q = conformal_quantile(resid[m], ALPHA)
            q_by_bin[str(u)] = q if (np.isfinite(q) and int(m.sum()) >= min_bin_n) else q_global
    return q_global, q_by_bin


def _half(supports, q_global, q_by_bin) -> np.ndarray:
    b = assign_support_bins(supports, EDGES)
    return np.array([q_by_bin.get(str(x), q_global) for x in b])


def main() -> None:
    points, truths, supports = _load()
    bins_all = assign_support_bins(supports, EDGES)
    counts = {b: int((bins_all == b).sum()) for b in BIN_ORDER if (bins_all == b).any()}

    policies = {
        "A_per_bin_current": dict(min_bin_n=1),
        "B_min_bin_n_50": dict(min_bin_n=50),
        "C_global_only": dict(min_bin_n=1, global_only=True),
    }
    acc = {name: {"cov": [], "width": [], "cov_by_bin": {}} for name in policies}
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(points.size)
        folds = np.array_split(idx, K)
        for name, kw in policies.items():
            for te in folds:
                tr = np.setdiff1d(idx, te)
                qg, qbb = _fit(points[tr], truths[tr], supports[tr], **kw)
                h = _half(supports[te], qg, qbb)
                lo = np.clip(points[te] - h, -1, 1)
                hi = np.clip(points[te] + h, -1, 1)
                cov = (truths[te] >= lo) & (truths[te] <= hi)
                acc[name]["cov"].extend(cov.tolist())
                acc[name]["width"].extend((hi - lo).tolist())
                tb_ = assign_support_bins(supports[te], EDGES)
                for bb in np.unique(tb_):
                    mm = tb_ == bb
                    acc[name]["cov_by_bin"].setdefault(str(bb), []).extend(cov[mm].tolist())

    report = {
        "n": int(points.size), "alpha": ALPHA, "k_folds": K, "n_seeds": N_SEEDS,
        "support_bin_counts": counts, "target_coverage": 1 - ALPHA, "policies": {},
    }
    for name, a in acc.items():
        report["policies"][name] = {
            "overall_coverage": float(np.mean(a["cov"])),
            "mean_width": float(np.mean(a["width"])),
            "coverage_by_bin": {b: float(np.mean(a["cov_by_bin"][b]))
                                for b in BIN_ORDER if b in a["cov_by_bin"]},
        }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"n={report['n']} | support-bin counts: {counts}")
    print(f"repeated {K}-fold cross-fit ({N_SEEDS} seeds), target coverage {1 - ALPHA:.2f}")
    for name, p in report["policies"].items():
        perbin = " ".join(f"{b}:{p['coverage_by_bin'][b]:.3f}"
                          for b in BIN_ORDER if b in p["coverage_by_bin"])
        print(f"  {name:20s} overall={p['overall_coverage']:.3f} "
              f"width={p['mean_width']:.3f} | per-bin {perbin}")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
