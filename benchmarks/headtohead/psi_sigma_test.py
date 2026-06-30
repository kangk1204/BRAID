#!/usr/bin/env python3
"""Test BRAID's calibration on the PSI-Sigma 130-RT-PCR validation set.

The PSI-Sigma supplement (Paik et al. 2019, Bioinformatics btz438) reports 130
RT-PCR-validated ΔPSI values plus each tool's RNA-seq ΔPSI prediction. That file
is paywalled (Oxford serves it via signed JS URLs; it is not Open Access on PMC,
GitHub, figshare, or EuropePMC), so it cannot be fetched headlessly.

==> MANUAL STEP (one time): download the supplementary file in a browser from
    https://academic.oup.com/bioinformatics/article/35/23/5048/5499131
    ("Supplementary data" tab) and save it as:
        data/public_benchmarks/meta/psi_sigma_supp.xlsx
    (or pass the path as the first CLI argument).

Then run:  python benchmarks/headtohead/psi_sigma_test.py [path]

This harness AUTO-DETECTS, across all sheets, a truth column (RT-PCR/RT-qPCR ΔPSI)
and per-method ΔPSI prediction columns, prints what it found (auditable), and for
each method reports point accuracy (Pearson/RMSE/direction) plus BRAID's shipped
conformal coverage: does [method ΔPSI ± q_global] cover the RT-PCR ΔPSI at ~0.95,
with NO refit? (the same transfer test used on TRA2/circadian/Jurkat).
"""
from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from braid.target.conformal import load_differential_conformal_calibrator  # noqa: E402

# The "PSI-Sigma 130 RT-PCR" set is actually REUSED from Bhate et al. 2015
# (Nat Commun 6:8768, PMC4635967, CC BY): Supplementary Data 1 =
# "RT-PCR validation of alternative splicing in mouse liver development"
# (~151 ESRP2-regulated events, file ncomms9768-s2.xlsx). Download it from
# https://pmc.ncbi.nlm.nih.gov/articles/PMC4635967/ (Supplementary Data 1) and
# drop it under data/public_benchmarks/meta/ -- this harness will auto-find it.
_DEFAULT_CANDIDATES = [
    "data/public_benchmarks/meta/bhate2015_rtpcr_supp.xlsx",
    "data/public_benchmarks/meta/ncomms9768-s2.xlsx",
    "data/public_benchmarks/meta/psi_sigma_supp.xlsx",
]
_TRUTH_HINTS = ("rt-pcr", "rt_pcr", "rtpcr", "rt-qpcr", "qpcr", "rt pcr")
_PRED_HINTS = ("psi-sigma", "psisigma", "rmats", "majiq", "suppa", "dexseq",
               "leafcutter", "whippet", "delta", "dpsi", "psi-sig")


def _find_file(argv):
    if len(argv) > 1 and os.path.exists(argv[1]):
        return argv[1]
    for c in _DEFAULT_CANDIDATES:
        if os.path.exists(c):
            return c
    meta = "data/public_benchmarks/meta"
    hits = sorted(set(
        glob.glob(f"{meta}/*sigma*") + glob.glob(f"{meta}/*btz438*")
        + glob.glob(f"{meta}/*bhate*") + glob.glob(f"{meta}/*ncomms9768*")
        + glob.glob(f"{meta}/*rtpcr*") + glob.glob(f"{meta}/*rt-pcr*")
    ))
    hits = [h for h in hits if h.lower().endswith((".xlsx", ".xls", ".csv"))]
    return hits[0] if hits else None


def _numeric(series):
    import pandas as pd
    return pd.to_numeric(series, errors="coerce")


def _looks_like_dpsi(vals):
    v = vals.dropna()
    if len(v) < 8:
        return False
    return float(v.between(-1.05, 1.05).mean()) > 0.8  # ΔPSI lives in [-1, 1]


def analyze_sheet(xl, sheet, q):
    import pandas as pd
    best = None
    for hdr in (0, 1, 2, 3):
        try:
            df = pd.read_excel(xl, sheet_name=sheet, header=hdr)
        except Exception:
            continue
        cols = {str(c): c for c in df.columns}
        truth_cols = [c for c in cols if any(h in c.lower() for h in _TRUTH_HINTS)]
        pred_cols = [c for c in cols
                     if any(h in c.lower() for h in _PRED_HINTS)
                     and c not in truth_cols and _looks_like_dpsi(_numeric(df[cols[c]]))]
        truth_cols = [c for c in truth_cols if _looks_like_dpsi(_numeric(df[cols[c]]))]
        if truth_cols and pred_cols:
            best = (hdr, df, cols, truth_cols, pred_cols)
            break
    if best is None:
        return None
    hdr, df, cols, truth_cols, pred_cols = best
    tcol = truth_cols[0]
    t_all = _numeric(df[cols[tcol]])
    print(f"\n[{sheet}] header row {hdr}: truth='{tcol}'  predictions={pred_cols}")
    print(f"  {'method-column':<28}{'n':>5}{'dir':>7}{'Pearson':>9}{'RMSE':>7}"
          f"{'BRAID-conf cov@95':>20}")
    rows = []
    for pc in pred_cols:
        d_all = _numeric(df[cols[pc]])
        m = t_all.notna() & d_all.notna()
        d, t = d_all[m].values, t_all[m].values
        if d.size < 8:
            continue
        clear = np.abs(t) > 0.05
        dirr = (float(np.mean(np.sign(d[clear]) == np.sign(t[clear])))
                if clear.any() else float("nan"))
        pear = float(np.corrcoef(d, t)[0, 1])
        rmse = float(np.sqrt(np.mean((d - t) ** 2)))
        cov = float(np.mean((t >= np.clip(d - q, -1, 1)) & (t <= np.clip(d + q, -1, 1))))
        print(f"  {pc:<28}{d.size:>5}{dirr:>7.2f}{pear:>9.3f}{rmse:>7.3f}{cov:>20.3f}")
        rows.append((pc, d.size, pear, rmse, cov))
    return rows


def main():
    path = _find_file(sys.argv)
    if path is None:
        print(__doc__)
        print("\n>>> File not found. Download it (manual step above) and re-run. <<<")
        sys.exit(2)
    import pandas as pd
    cal = load_differential_conformal_calibrator()
    q = cal.q_global
    print(f"PSI-Sigma RT-PCR validation test  (file: {path})")
    print(f"BRAID shipped conformal half-width q_global = {q:.3f} (no refit)")
    xl = pd.ExcelFile(path)
    print(f"sheets: {xl.sheet_names}")
    any_found = False
    for s in xl.sheet_names:
        res = analyze_sheet(xl, s, q)
        any_found = any_found or bool(res)
    if not any_found:
        print("\nNo (RT-PCR truth + ΔPSI prediction) table auto-detected. Sheet/column")
        print("dump follows so the harness can be pointed at the right columns:")
        for s in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=s, header=None, nrows=4)
            print(f"  [{s}] first rows:\n{df.to_string()[:500]}")
    else:
        print("\nBRAID-conf cov@95 ~0.95 => the shipped calibrator transfers to PSI-Sigma's")
        print("RT-PCR set with no refit (same result family as TRA2/circadian/Jurkat).")


if __name__ == "__main__":
    if os.path.isdir("data"):
        pass
    else:
        os.chdir(os.path.dirname(os.path.dirname(_HERE)))
    main()
