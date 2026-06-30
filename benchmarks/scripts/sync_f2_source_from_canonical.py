"""Regenerate the Figure-2 source workbook from the CANONICAL head-to-head JSON.

Eliminates the long-standing divergence where make_f2_benchmark.py recomputed the
head-to-head independently of benchmarks/results/headtohead_coverage.json (the source
the Methods names as canonical). After comprehensive_benchmark.py is re-run, this
bridge rewrites outputs/figures/manuscript/F2/f2_source_data.xlsx (panel_A_coverage +
panel_BC_pooled) directly from that JSON, so Figure 2 and Table 1 / Supplementary
Table 1 all derive from one artifact.
"""
from __future__ import annotations

import json
import os

import pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON = os.path.join(_REPO, "benchmarks/results/headtohead_coverage.json")
OUT = os.path.join(_REPO, "outputs/figures/manuscript/F2/f2_source_data.xlsx")

NAME = {"TRA2 (GSE59335, human)": "TRA2", "Circadian (GSE54651, mouse)": "Circadian",
        "PC3E/GS689 (SRA, human)": "SRS354082"}
METHODS = [("BRAID-conformal", "BRAID"), ("MAJIQ", "MAJIQ"),
           ("betAS", "betAS"), ("rMATS", "rMATS")]


def _row(panel, dataset, method, label, n, m):
    cov = m["coverage"]
    return {
        "panel": panel, "dataset": dataset, "method": method, "method_label": label,
        "n": n, "covered": int(round(cov * n)), "coverage": cov,
        "wilson_low": m["wilson95"][0], "wilson_high": m["wilson95"][1],
        "mean_width": m["width"], "interval_score": m["iscore"],
    }


def main() -> None:
    d = json.load(open(JSON, encoding="utf-8"))
    pooled = d["pooled_common"]
    a_rows = []
    for jname, short in NAME.items():
        ds = d["datasets"][jname]
        for method, label in METHODS:
            a_rows.append(_row("A", short, method, label, ds["n_common"],
                               ds["methods"][method]))
    for method, label in METHODS:
        a_rows.append(_row("A", "Pooled", method, label, pooled["n"],
                           pooled["methods"][method]))

    bc_rows = []
    for method, label in METHODS:
        m = pooled["methods"][method]
        r = _row("A", "Pooled", method, label, pooled["n"], m)
        r.update(panel_b_value=m["iscore"], panel_c_x=m["width"], panel_c_y=m["coverage"])
        bc_rows.append(r)

    with pd.ExcelWriter(OUT, engine="openpyxl") as w:
        pd.DataFrame(a_rows).to_excel(w, sheet_name="panel_A_coverage", index=False)
        pd.DataFrame(bc_rows).to_excel(w, sheet_name="panel_BC_pooled", index=False)
        pd.DataFrame([{"key": "source", "value": "headtohead_coverage.json (canonical)"},
                      {"key": "pooled_n", "value": pooled["n"]}]).to_excel(
            w, sheet_name="metadata", index=False)
    print(f"wrote {os.path.relpath(OUT, _REPO)} from canonical JSON "
          f"(pooled n={pooled['n']}, BRAID={pooled['methods']['BRAID-conformal']['coverage']:.3f})")


if __name__ == "__main__":
    main()
