"""Compare real-rerun rMATS coverage against the binomial-thinning surrogate.

For each ``(fraction, rerun-SE-table)`` pair, run the SAME head-to-head coverage
computation used by the thinning titration (GSE59335 config: swap=True, the TRA2
validated/failed RT-PCR targets) and tabulate coverage@95 / width@95 per method
next to the thinning curve from ``depth_titration.json``.

The ``f=1.0`` row doubles as the parameter check: a faithful rerun should
reproduce the in-repo SE-table coverage (which the thinning f=1.0 column already
matches against the canonical artifact).

Usage::

    python benchmarks/headtohead/compare_rerun_vs_thinning.py \
        --inrepo data/public_benchmarks/GSE59335/rmats/SE.MATS.JC.txt \
        1.0 benchmarks/results/depth_rerun/GSE59335_f1.0_s0/SE.MATS.JC.txt \
        0.50 benchmarks/results/depth_rerun/GSE59335_f0.50_s0/SE.MATS.JC.txt \
        0.25 benchmarks/results/depth_rerun/GSE59335_f0.25_s0/SE.MATS.JC.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import head_to_head_coverage as H  # noqa: E402

DS_NAME = "TRA2 (GSE59335, human)"
METHODS = ("BRAID-conformal", "BRAID-conformal-abs", "rMATS-perRep",
           "betAS", "BRAID-Jeffreys")


def coverage_for(se_path: str) -> dict:
    targets = H.load_targets(
        "data/public_benchmarks/GSE59335/targets/validated_events.tsv",
        "data/public_benchmarks/GSE59335/targets/failed_events.tsv",
    )
    res = H.run_dataset(se_path, None, None, DS_NAME, seed=7,
                        betas_tsv=None, swap_groups=True, targets=targets)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inrepo", default=None, help="in-repo SE table for the f=1 param check")
    ap.add_argument("pairs", nargs="+",
                    help="alternating FRAC SE_PATH FRAC SE_PATH ...")
    args = ap.parse_args()
    os.chdir(_REPO)

    if len(args.pairs) % 2 != 0:
        ap.error("pairs must be alternating FRAC SE_PATH")
    pairs = [(args.pairs[i], args.pairs[i + 1]) for i in range(0, len(args.pairs), 2)]

    thin = {}
    tj = "benchmarks/results/depth_titration.json"
    if os.path.exists(tj):
        with open(tj, encoding="utf-8") as f:
            thin = json.load(f).get("per_dataset", {}).get(DS_NAME, {})

    if args.inrepo:
        r = coverage_for(args.inrepo)
        print(f"\n[param check] in-repo SE table: n_matched={r['n_matched']}")
        for m in METHODS:
            mr = r["methods"].get(m, {})
            print(f"    {m:<20} cov95={mr.get('coverage95'):.3f}  width95={mr.get('width95'):.3f}")

    print(f"\n{'frac':>5} {'method':<20}{'rerun_cov':>10}{'thin_cov':>10}{'Δcov':>8}"
          f"{'rerun_w':>9}{'thin_w':>9}  n_re/n_thin")
    print("-" * 86)
    for frac, se in pairs:
        if not os.path.exists(se):
            print(f"{frac:>5}  MISSING {se}")
            continue
        rr = coverage_for(se)
        fk = f"{float(frac):.2f}"
        tnode = thin.get(fk, {})
        tmeth = tnode.get("methods", {})
        n_thin = tnode.get("n_matched_mean")
        for m in METHODS:
            rc = rr["methods"].get(m, {}).get("coverage95")
            rw = rr["methods"].get(m, {}).get("width95")
            tc = tmeth.get(m, {}).get("coverage95_mean")
            tw = tmeth.get(m, {}).get("width95_mean")
            dcov = (rc - tc) if (rc is not None and tc is not None) else float("nan")
            tc_s = f"{tc:>10.3f}" if tc is not None else f"{'NA':>10}"
            tw_s = f"{tw:>9.3f}" if tw is not None else f"{'NA':>9}"
            nt = int(n_thin) if n_thin else "?"
            tail = f"  {rr['n_matched']}/{nt}" if m == METHODS[0] else ""
            print(f"{frac:>5} {m:<20}{rc:>10.3f}{tc_s}{dcov:>8.3f}{rw:>9.3f}{tw_s}{tail}")
        print("-" * 86)


if __name__ == "__main__":
    main()
