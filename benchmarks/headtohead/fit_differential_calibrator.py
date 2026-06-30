#!/usr/bin/env python3
"""Fit BRAID's production differential-ΔPSI conformal calibrator from real RT-PCR data.

This produces the shipped artifact ``braid/target/calibration_artifacts/
differential_dpsi_conformal.json`` -- an absolute-residual, Mondrian (support-
stratified) split-conformal calibrator fit on the pooled real RT-PCR residuals of
TRA2 (GSE59335) + circadian (GSE54651). It is the empirically-validated calibration
layer shown (see RESULTS.md) to reach nominal ΔPSI coverage on real data where
betAS / rMATS / the Jeffreys posterior systematically under-cover.

Stored as a ``ConformalCalibrator`` with ``scale_kind="absolute_dpsi"``: the half-width
is the conformal quantile q itself (constant per support bin), not q*posterior_std --
the std-normalized schedule is near-vacuous on this orthogonal-truth residual surface
(RESULTS.md, review caveat 1). The production ``braid differential`` path applies q via
``robust_interval(dpsi_mean, dpsi_std, total_support, event_type=...)`` (depth-robust +
the event-type/composite q_for cascade); the pure ``interval(sigma=1.0)`` form is the
benchmark variant. This generator fits only the global/support-bin quantiles: the RT-PCR
calibration set is cassette-exon (SE) only, so there is no non-SE truth to fit an
event-type quantile from -- event-type conditioning is demonstrated on the SG-NEx
long-read surface (sgnex_dpsi_validation.py), not on this RT-PCR differential artifact.
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
from cross_dataset_transfer import build_arrays, fit_abs_conformal  # noqa: E402

from braid.target.conformal import ConformalCalibrator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tra2-se", required=True)
    ap.add_argument("--tra2-validated", required=True)
    ap.add_argument("--tra2-failed", required=True)
    ap.add_argument("--circ-se", required=True)
    ap.add_argument("--circ-tsv", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tra2_t = H.load_targets(args.tra2_validated, args.tra2_failed)
    circ_t = H.load_circadian_targets(args.circ_tsv)
    pa, ta, sa = build_arrays(args.tra2_se, tra2_t, swap_groups=True)
    pb, tb, sb = build_arrays(args.circ_se, circ_t, swap_groups=True)

    points = np.concatenate([pa, pb])
    truths = np.concatenate([ta, tb])
    supports = np.concatenate([sa, sb])

    q_global, q_by_bin = fit_abs_conformal(points, truths, supports, args.alpha)
    profile = {
        "n": float(points.size),
        "support_median": float(np.median(supports)),
        "support_q25": float(np.percentile(supports, 25)),
        "support_q75": float(np.percentile(supports, 75)),
    }
    cal = ConformalCalibrator(
        alpha=float(args.alpha),
        q_global=float(q_global),
        q_by_bin={k: float(v) for k, v in q_by_bin.items()},
        bin_edges=(20, 50, 100, 250),
        scale_kind="absolute_dpsi",
        training_scope=f"real_rtpcr_tra2_circadian_n{points.size}",
        calibration_profile=profile,
    )
    cal.to_json(args.out)

    # Report in-sample coverage at 95% as a sanity check (deployment generalization
    # is established separately by cross_dataset_transfer.py).
    bins = H.assign_support_bins(supports)
    half = np.array([cal.q_for(s) for s in supports])
    low = np.clip(points - half, -1, 1)
    high = np.clip(points + half, -1, 1)
    cov = float(np.mean((truths >= low) & (truths <= high)))
    print(f"fit on n={points.size} real RT-PCR events (TRA2 + circadian)")
    print(f"  q_global={q_global:.3f}  q_by_bin={ {k: round(v,3) for k,v in q_by_bin.items()} }")
    print(f"  in-sample coverage@{1-args.alpha:.0%} = {cov:.3f}")
    print(f"  bins present: { {b: int(np.sum(bins==b)) for b in np.unique(bins)} }")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
