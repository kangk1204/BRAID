"""Within-study partial-validation use case: calibrate the rest from a few RT-PCR.

The most natural deployment of split conformal: in a single study you RT-PCR a
small subset of splicing events, use those (RNA-seq estimate, RT-PCR truth) pairs
as the calibration set, and BRAID then assigns a calibrated 95% interval -- and a
distribution-free "reliable" flag (interval excludes 0) -- to every remaining
un-validated event.

Two pre-registered questions, each measured on held-out (un-validated) events:
  Q1 (data efficiency): how many representative RT-PCR validations are needed for
     the remaining events to reach nominal coverage?
  Q2 (guardrail): does validating only the top-hit (largest |dPSI|) events -- what
     labs usually do -- break the calibration vs a representative random sample?

Leakage-free by construction: the held-out events never enter the calibration q.
Output: benchmarks/results/within_study_calibration.json
Usage:  python benchmarks/headtohead/within_study_calibration.py
"""

from __future__ import annotations

# ruff: noqa: I001
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from conditional_coverage_diagnostic import load_dpsi_dataset  # noqa: E402
from braid.target.conformal import conformal_quantile  # noqa: E402

ALPHA = 0.05
N_TRIALS = 300
N_GRID = (20, 30, 40, 50, 70, 90)
DATASET = "tra2"  # the within-study panel with the most continuous RT-PCR dPSI truth


def _one_trial(pts, tru, n, N, mode, rng):
    if mode == "random":                       # representative validation subset
        cal = rng.choice(n, N, replace=False)
    else:                                       # "top-hit": validate the largest effects
        cal = np.argsort(-np.abs(pts))[:N]
    test = np.setdiff1d(np.arange(n), cal)
    q = conformal_quantile(np.abs(tru[cal] - pts[cal]), ALPHA)
    if not np.isfinite(q):                      # too few calibration points -> full range
        q = 1.0
    lo = np.clip(pts[test] - q, -1.0, 1.0)
    hi = np.clip(pts[test] + q, -1.0, 1.0)
    cov = float(np.mean((tru[test] >= lo) & (tru[test] <= hi)))
    width = float(np.mean(hi - lo))
    excl = (lo > 0) | (hi < 0)                  # calibrated "reliable" flag
    n_rel = int(excl.sum())
    prec = (float(np.mean((np.sign(pts[test][excl]) == np.sign(tru[test][excl]))
                          & (np.abs(tru[test][excl]) > 0.05))) if n_rel else float("nan"))
    return cov, width, n_rel / len(test), prec, n_rel


def run(dataset: str = DATASET) -> dict:
    d = load_dpsi_dataset(dataset)
    pts, tru = d["points"], d["truths"]
    n = pts.size
    grid = [N for N in N_GRID if N <= n - 10]   # always leave >=10 held-out test events
    out = {"dataset": dataset, "n": int(n), "alpha": ALPHA, "n_trials": N_TRIALS,
           "orient_corr": float(d["orient_corr"]), "n_grid": grid, "by_N": []}
    for N in grid:
        row = {"N": N}
        for mode in ("random", "tophit"):
            rng = np.random.default_rng(0)
            res = np.array([_one_trial(pts, tru, n, N, mode, rng) for _ in range(N_TRIALS)])
            cov = res[:, 0]
            row[mode] = {
                "coverage": float(cov.mean()),
                "coverage_lo": float(np.percentile(cov, 2.5)),
                "coverage_hi": float(np.percentile(cov, 97.5)),
                "mean_width": float(res[:, 1].mean()),
                "frac_reliable": float(res[:, 2].mean()),
                "reliable_precision": (float(np.nanmean(res[:, 3]))
                                       if np.isfinite(res[:, 3]).any() else float("nan")),
                "n_reliable_mean": float(res[:, 4].mean()),
            }
        out["by_N"].append(row)
    return out


def _is_positive(dataset: str, keys: list) -> np.ndarray:
    from conditional_coverage_diagnostic import _read_tsv  # noqa: E402
    tbl = {r["key"]: r for r in _read_tsv(os.path.join(_HERE, f"{dataset}_betas_truth.tsv"))}
    return np.array([int(tbl[k]["is_positive"]) for k in keys])


def strategy_comparison(dataset: str = "tra2", N: int = 30, n_trials: int = 300) -> dict:
    """How the RT-PCR acquisition strategy (which events to validate) changes the
    calibration. Needs a panel with RT-PCR negatives (truth ~ 0); evaluated on
    held-out events with their REAL truth."""
    d = load_dpsi_dataset(dataset)
    pts, tru = d["points"], d["truths"]
    ispos = _is_positive(dataset, d["keys"])
    n = pts.size
    nneg, npos = N // 3, N - N // 3

    def evalc(cal, cal_truth):
        test = np.setdiff1d(np.arange(n), cal)
        q = conformal_quantile(np.abs(cal_truth - pts[cal]), ALPHA)
        q = 1.0 if not np.isfinite(q) else q
        lo, hi = np.clip(pts[test] - q, -1, 1), np.clip(pts[test] + q, -1, 1)
        cov = float(np.mean((tru[test] >= lo) & (tru[test] <= hi)))
        excl = (lo > 0) | (hi < 0)
        nr = int(excl.sum())
        prec = (float(np.mean((np.sign(pts[test][excl]) == np.sign(tru[test][excl]))
                              & (np.abs(tru[test][excl]) > 0.05))) if nr else float("nan"))
        return cov, q, nr / len(test), prec

    def one(strategy, rng):
        if strategy == "top_hit":
            cal = np.argsort(-np.abs(pts))[:N]
            return evalc(cal, tru[cal])
        if strategy == "random":
            cal = rng.choice(n, N, replace=False)
            return evalc(cal, tru[cal])
        if strategy in ("with_negatives_measured", "with_negatives_assumed0"):
            neg = rng.choice(np.where(ispos == 0)[0], nneg, replace=False)
            pos = rng.choice(np.where(ispos == 1)[0], npos, replace=False)
            cal = np.concatenate([neg, pos])
            ct = tru[cal].copy()
            if strategy == "with_negatives_assumed0":
                ct[:nneg] = 0.0           # assume known negatives = 0 (no RT-PCR run)
            return evalc(cal, ct)
        if strategy == "lowsignal_assumed0":  # circular: low RNA-seq signal assumed negative
            pneg = np.argsort(np.abs(pts))[:nneg]
            pos = rng.choice(np.setdiff1d(np.where(ispos == 1)[0], pneg), npos, replace=False)
            cal = np.concatenate([pneg, pos])
            ct = tru[cal].copy()
            ct[:nneg] = 0.0
            return evalc(cal, ct)
        raise ValueError(strategy)

    strategies = ["top_hit", "random", "with_negatives_measured",
                  "with_negatives_assumed0", "lowsignal_assumed0"]
    out = {"dataset": dataset, "N": N, "n_pos": int((ispos == 1).sum()),
           "n_neg": int((ispos == 0).sum()), "by_strategy": {}}
    for s in strategies:
        rng = np.random.default_rng(0)
        r = np.array([one(s, rng) for _ in range(n_trials)])
        m = np.nanmean(r, 0)
        out["by_strategy"][s] = {"coverage": float(m[0]), "q": float(m[1]),
                                 "frac_reliable": float(m[2]), "reliable_precision": float(m[3])}
    return out


def _json_safe(o):
    """Recursively convert NaN floats to None so the output is strict JSON."""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    if isinstance(o, float) and o != o:  # NaN
        return None
    return o


def main() -> None:
    out = {"datasets": {name: run(name) for name in ("tra2", "circ")},
           "strategy_comparison": strategy_comparison("tra2")}
    path = os.path.join(_REPO, "benchmarks/results/within_study_calibration.json")
    json.dump(_json_safe(out), open(path, "w"), indent=2, allow_nan=False)
    for name, res in out["datasets"].items():
        print(f"\n=== {name}  n={res['n']}  (orient corr {res['orient_corr']:.3f}) ===")
        print(f"{'N':>4}  {'mode':>7}  {'cov(rest)':>9}  {'width':>6}  {'%rel':>6}  {'prec':>6}")
        for row in res["by_N"]:
            for mode in ("random", "tophit"):
                m = row[mode]
                pr = m["frac_reliable"] * 100
                print(f"{row['N']:>4}  {mode:>7}  {m['coverage']:>9.3f}  {m['mean_width']:>6.3f}  "
                      f"{pr:>6.1f}  {m['reliable_precision']:>6.2f}")
    sc = out["strategy_comparison"]
    print(f"\n=== acquisition strategy (tra2, N={sc['N']}) ===")
    print(f"{'strategy':<26} {'cov':>6} {'q':>6} {'%rel':>6} {'prec':>6}")
    for s, m in sc["by_strategy"].items():
        print(f"{s:<26} {m['coverage']:>6.3f} {m['q']:>6.3f} {m['frac_reliable'] * 100:>5.1f}  "
              f"{m['reliable_precision']:>6.2f}")
    print(f"\nwrote {os.path.relpath(path, _REPO)}")


if __name__ == "__main__":
    main()
