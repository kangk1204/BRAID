#!/usr/bin/env python3
"""Test additional public RT-PCR-validated datasets from the SUPPA2 supplement.

The SUPPA2 supplement (Trincado 2018, 13059_2018_1417) carries several RT-PCR-
validated datasets with each method's RNA-seq ΔPSI prediction aligned to the
RT-PCR truth — all local, no download/alignment needed:
  - Table S5  : TRA2 knockdown (Best 2015 RNA-seq)            [re-confirm]
  - Table S7  : mouse circadian cerebellum/liver (Zhang 2015) [cross-check]
  - Table S9  : Jurkat T-cell stim vs unstim (Cole 2015)      [NEW dataset]

For each dataset we report, per method (real SUPPA2 / rMATS / MAJIQ), the point
accuracy vs RT-PCR (Pearson, RMSE, direction), and we test whether BRAID's SHIPPED
conformal calibrator generalizes: applying the absolute half-width q_global to the
rMATS ΔPSI (the caller BRAID wraps), with NO refit, does the interval cover the
RT-PCR ΔPSI at ~nominal 95%? (Coverage transfer, like the SRS354082 test.)
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from braid.target.conformal import load_differential_conformal_calibrator  # noqa: E402

XLSX = "data/public_benchmarks/meta/suppa2_tables.xlsx"
DATASETS = [
    ("Table S5", "TRA2 knockdown (Best 2015)"),
    ("Table S7", "Circadian cerebellum/liver (Zhang 2015)"),
    ("Table S9", "Jurkat T-cell stim/unstim (Cole 2015)"),
]


def load_sheet(xlsx, sheet):
    """Parse a SUPPA2 'predictions by method' sheet -> {method: [(dpsi, rtpcr)]}."""
    import pandas as pd
    df = pd.read_excel(xlsx, sheet_name=sheet, header=2)
    cols = {c.lower(): c for c in map(str, df.columns)}
    src = cols.get("source")
    dp = cols.get("deltapsi")
    rt = cols.get("rt-pcr") or cols.get("rt_pcr")
    if not (src and dp and rt):
        return {}
    out = {}
    for _, r in df.iterrows():
        s = str(r[src]).strip().upper()
        try:
            dpsi = float(r[dp])
            rtpcr = float(r[rt])
        except (ValueError, TypeError):
            continue
        name = ("MAJIQ" if s.startswith("MAJIQ") else "SUPPA2" if s.startswith("SUPPA")
                else "rMATS" if s.startswith("RMATS") else None)
        if name:
            out.setdefault(name, []).append((dpsi, rtpcr))
    return out


def point_metrics(pairs):
    d = np.array([p[0] for p in pairs])
    t = np.array([p[1] for p in pairs])
    clear = np.abs(t) > 0.05
    dir_acc = (float(np.mean(np.sign(d[clear]) == np.sign(t[clear])))
               if clear.any() else float("nan"))
    pear = float(np.corrcoef(d, t)[0, 1]) if d.size > 2 else float("nan")
    rmse = float(np.sqrt(np.mean((d - t) ** 2)))
    return len(pairs), dir_acc, pear, rmse


def main():
    os.chdir(os.path.dirname(os.path.dirname(_HERE)))
    cal = load_differential_conformal_calibrator()
    q = cal.q_global
    print(f"\nBRAID shipped conformal half-width q_global = {q:.3f}"
          f" (fit on TRA2+circadian, NO refit here)")

    for sheet, label in DATASETS:
        data = load_sheet(XLSX, sheet)
        if not data:
            print(f"\n[{sheet}] {label}: no parseable predictions")
            continue
        print(f"\n{'='*78}\n{label}   [{sheet}]\n{'-'*78}")
        print(f"{'method':<12}{'n':>5}{'dir-acc':>9}{'Pearson':>9}{'RMSE':>7}"
              f"{'BRAID-conf cov@95':>20}")
        for m in ["SUPPA2", "rMATS", "MAJIQ"]:
            if m not in data:
                continue
            n, da, pe, rm = point_metrics(data[m])
            d = np.array([p[0] for p in data[m]])
            t = np.array([p[1] for p in data[m]])
            # BRAID conformal transfer: interval = method ΔPSI ± q_global, cover RT-PCR?
            cov = float(np.mean((t >= np.clip(d - q, -1, 1)) & (t <= np.clip(d + q, -1, 1))))
            tag = "  <- BRAID wraps" if m == "rMATS" else ""
            print(f"{m:<12}{n:>5}{da:>9.3f}{pe:>9.3f}{rm:>7.3f}{cov:>20.3f}{tag}")
        print(f"{'-'*78}")
        print("BRAID-conf cov@95 = fraction of RT-PCR ΔPSI covered by [method ΔPSI ± q]; "
              "target 0.95.")
    print(f"\n{'='*78}")
    print("Generalization: if BRAID-conf cov@95 ~0.95 across all three independent datasets,")
    print("the shipped calibrator transfers (no per-dataset refit).")


if __name__ == "__main__":
    main()
