"""SUPPA2 recalibration ablation: the caller-agnostic result extends to a 4th tool.

SUPPA2 (Trincado et al. 2018) emits a point Delta PSI + significance but no native
event-level 95% interval, so it is absent from the head-to-head interval benchmark.
This script shows that, like every other caller, SUPPA2's point estimate reaches
nominal coverage of RT-PCR truth once the same conformal residual recalibration is
applied -- reproducing the recalibration ablation (Table 3) on a fourth tool.

Data: the authors' canonical SUPPA2 / rMATS / MAJIQ TRA2 predictions paired with
RT-PCR truth, from Table S5 of the SUPPA2 supplement (Genome Biology 2018,
13059_2018_1417). Not redistributed here; download the supplement Excel from
  https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-018-1417-1/MediaObjects/13059_2018_1417_MOESM2_ESM.xlsx
to data/public_benchmarks/suppa2/TableS5_13059_2018_1417.xlsx (gitignored).

Recalibration: leakage-free k-fold cross-fit conformal on the absolute RNA-seq-to-
RT-PCR residual of each method's point estimate (global; the Table-S5 events carry
no read-support field), evaluated on the same events.
Output: benchmarks/results/suppa2_recalibration.json
Usage:  python benchmarks/headtohead/suppa2_recalibration.py [--s5-xlsx PATH]
"""

from __future__ import annotations

# ruff: noqa: I001
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multitool_compare import load_s5  # noqa: E402
from head_to_head_coverage import conformal_crossfit, interval_score  # noqa: E402
from comprehensive_benchmark import wilson  # noqa: E402

ALPHA = 0.05
DEFAULT_XLSX = os.path.join(_REPO, "data/public_benchmarks/suppa2/TableS5_13059_2018_1417.xlsx")


def recalibrate(dpsi: np.ndarray, rtpcr: np.ndarray) -> dict:
    ok = np.isfinite(dpsi) & np.isfinite(rtpcr)
    dpsi, rtpcr = dpsi[ok], rtpcr[ok]
    n = dpsi.size
    corr = float(np.corrcoef(dpsi, rtpcr)[0, 1]) if n > 2 else float("nan")
    if corr < 0:                              # orient to truth (per-method global convention)
        dpsi = -dpsi
    lo, hi = conformal_crossfit(dpsi, rtpcr, np.ones(n), np.full(n, 1000.0), ALPHA)
    cov = float(np.mean((rtpcr >= lo) & (rtpcr <= hi)))
    _, wl, wh = wilson(int(round(cov * n)), n)
    isc = float(np.mean([interval_score(y, a, b, ALPHA) for y, a, b in zip(rtpcr, lo, hi)]))
    return {"n": int(n), "orient_corr": corr, "recalibrated_coverage": cov,
            "wilson": [wl, wh], "interval_score": isc, "mean_width": float(np.mean(hi - lo))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--s5-xlsx", default=DEFAULT_XLSX)
    a = p.parse_args(argv)
    if not os.path.exists(a.s5_xlsx):
        raise SystemExit(f"SUPPA2 Table S5 not found at {a.s5_xlsx} (see module docstring).")
    data = load_s5(a.s5_xlsx)
    res = {"source": "Trincado 2018 Genome Biology, Table S5 (TRA2)", "alpha": ALPHA,
           "note": "SUPPA2 emits no native interval; only the recalibrated point is shown.",
           "by_method": {}}
    for method in ("SUPPA2", "rMATS", "MAJIQ"):
        recs = data.get(method, [])
        if recs:
            res["by_method"][method] = recalibrate(
                np.array([r["dpsi"] for r in recs], float),
                np.array([r["rtpcr"] for r in recs], float))
    out = os.path.join(_REPO, "benchmarks/results/suppa2_recalibration.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"{'method':>7}  {'n':>3}  {'recal cov':>9}  {'Wilson95':>15}  {'iscore':>7}")
    for m, r in res["by_method"].items():
        print(f"{m:>7}  {r['n']:>3}  {r['recalibrated_coverage']:>9.3f}  "
              f"[{r['wilson'][0]:.3f},{r['wilson'][1]:.3f}]  {r['interval_score']:>7.3f}")
    print(f"\nwrote {os.path.relpath(out, _REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
