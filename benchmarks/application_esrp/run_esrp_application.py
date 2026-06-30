#!/usr/bin/env python3
# ruff: noqa: I001
"""BRAID recovery analysis for the Esrp1/2 double-knockout splicing program.

Companion to the DM1 disease application (``run_dm1_application.py``). Where DM1
asks whether BRAID recovers known *disease* mis-splicing anchors, this analysis
asks the cross-species, regulator-loss version of the same question: in an
Esrp1/Esrp2 double-knockout (DKO) vs wild-type (WT) mouse comparison, does BRAID
confidently recover the canonical epithelial cassette exons that Esrp1/2 are
established to regulate?

Design
------
* Data: GSE64357 (Bebee et al., 2015, eLife) — Esrp1/2 DKO vs WT mouse, aligned
  with HISAT2 to mm10 and quantified with rMATS (``pipeline.sh``). rMATS is run
  with DKO as ``--b1`` (sample_1) and WT as ``--b2`` (sample_2), so BRAID's
  ``dpsi = ctrl_psi - treat_psi = PSI(DKO) - PSI(WT)``.
* Anchors: a literature-curated set of canonical Esrp1/2-regulated cassette
  exons (mouse symbols). These are a *published gene set*, not fabricated data;
  every PSI, interval, and tier reported here is computed by BRAID on the real
  rMATS table. Citations are in ``ANCHOR_GENES`` below.
* Recovery metric: mirroring DM1, we report magnitude (|dPSI|) and whether the
  BRAID-calibrated 95% interval excludes zero (a confident switch) per anchor.
  We deliberately do *not* assert a per-gene direction expectation — Esrp can
  promote either inclusion or exclusion depending on binding position — so the
  honest, defensible claim is "BRAID confidently recovers a large-magnitude
  switch at these established targets," not a signed prediction per exon.

This is regulator-target recovery evidence, not orthogonal RT-PCR validation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks"))
import figstyle  # noqa: E402

figstyle.apply()

APP_DIR = ROOT / "benchmarks" / "application_esrp"
RESULTS_DIR = APP_DIR / "results"
RMATS_DIR = ROOT / "data" / "public_benchmarks" / "GSE64357_esrp" / "rmats"
FIGURE_DIR = ROOT / "paper" / "figures"
DIFFERENTIAL_MODELS = ("sum", "auto", "rep")
MANUSCRIPT_DIFFERENTIAL_MODEL = "auto"

# Canonical Esrp1/2-regulated cassette-exon targets (mouse symbols).
# Refs: Warzecha 2009 Mol Cell (FGFR2); Warzecha 2010 EMBO J (epithelial ESRP
# program); Dittmar 2012 MCB (genome-wide ESRP network); Bebee 2015 eLife
# (the mouse Esrp1/2 DKO study these reads come from, GSE64357).
ANCHOR_GENES: dict[str, str] = {
    "Enah": "Mena INV exon — classic ESRP epithelial target (Warzecha 2010)",
    "Cd44": "variable-exon switch — ESRP/EMT target (Warzecha 2010)",
    "Ctnnd1": "p120-catenin isoform switch — ESRP target (Warzecha 2010)",
    "Slk": "ESRP-regulated cassette exon (Dittmar 2012; Bebee 2015)",
    "Arhgef11": "ESRP-regulated cassette exon (Dittmar 2012; Bebee 2015)",
    "Fnip1": "ESRP-regulated cassette exon (Dittmar 2012)",
    "Itga6": "integrin alpha-6 A/B exon — ESRP target (Warzecha 2010)",
    "Scrib": "ESRP-regulated cassette exon (Dittmar 2012)",
    "Flnb": "filamin-B exon — ESRP/EMT target (Li 2018; Dittmar 2012)",
    "Epb41l5": "ESRP-regulated cassette exon (Bebee 2015)",
    "Magi1": "ESRP-regulated cassette exon (Dittmar 2012)",
    "Numb": "ESRP-regulated cassette exon (Warzecha 2010)",
    "Map3k7": "TAK1 ESRP-regulated cassette exon (Dittmar 2012)",
    "Tcf7l2": "ESRP-regulated cassette exon (Dittmar 2012)",
    "Exoc7": "exocyst ESRP-regulated cassette exon (Dittmar 2012)",
}

# Most repeatedly validated, directly RT-PCR-confirmed ESRP targets.
PRIMARY_ANCHORS = {
    "Enah",
    "Cd44",
    "Ctnnd1",
    "Slk",
    "Arhgef11",
    "Fnip1",
    "Itga6",
    "Scrib",
    "Flnb",
    "Epb41l5",
}


@dataclass(frozen=True)
class Paths:
    braid_tsv: Path
    summary_json: Path
    anchor_tsv: Path
    fig_pdf: Path
    fig_png: Path
    fig_svg: Path


def paths() -> Paths:
    return Paths(
        braid_tsv=RESULTS_DIR / "esrp_braid_differential.tsv",
        summary_json=RESULTS_DIR / "esrp_application_summary.json",
        anchor_tsv=RESULTS_DIR / "esrp_anchor_gene_summary.tsv",
        fig_pdf=FIGURE_DIR / "fig_esrp_recovery.pdf",
        fig_png=FIGURE_DIR / "fig_esrp_recovery.png",
        fig_svg=FIGURE_DIR / "fig_esrp_recovery.svg",
    )


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def run_braid(p: Paths, replicates: int, differential_model: str, force: bool) -> None:
    se_table = RMATS_DIR / "SE.MATS.JC.txt"
    if not se_table.exists() or se_table.stat().st_size == 0:
        raise FileNotFoundError(
            f"rMATS SE table not found or empty: {se_table}. Run pipeline.sh first."
        )
    if p.braid_tsv.exists() and p.braid_tsv.stat().st_size > 0 and not force:
        return
    cmd = [
        sys.executable,
        "-m",
        "braid",
        "differential",
        "--rmats-dir",
        str(RMATS_DIR),
        "--output",
        str(p.braid_tsv),
        "--replicates",
        str(replicates),
        "--differential-model",
        differential_model,
        "--effect-cutoff",
        "0.1",
        "--min-support",
        "20",
        "--seed",
        "20260620",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["abs_dpsi"] = out["dpsi"].abs()
    out["abs_rmats_dpsi"] = out["rmats_dpsi"].abs()
    out["total_support"] = out["ctrl_support"] + out["treat_support"]
    out["ci_width"] = out["ci_high"] - out["ci_low"]
    out["is_big_rmats_event"] = (out["rmats_fdr"] < 0.05) & (out["abs_rmats_dpsi"] >= 0.1)
    out["is_anchor_gene"] = out["gene"].isin(ANCHOR_GENES)
    out["is_primary_anchor"] = out["gene"].isin(PRIMARY_ANCHORS)
    return out


def summarize_anchors(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene, role in ANCHOR_GENES.items():
        sub = df[df["gene"] == gene].copy()
        if sub.empty:
            rows.append({
                "gene": gene, "role": role, "events": 0, "big_rmats_events": 0,
                "braid_supported": 0, "braid_high_confidence": 0,
                "best_event_id": "", "best_tier": "", "best_dpsi": float("nan"),
                "best_ci_low": float("nan"), "best_ci_high": float("nan"),
                "best_ci_excludes_zero": False, "best_rmats_fdr": float("nan"),
                "best_total_support": 0,
            })
            continue
        supported = sub["tier"].isin(["supported", "high-confidence"])
        high_conf = sub["tier"] == "high-confidence"
        # Rank: high-confidence first, then supported, then by |dPSI| and support.
        rank_bucket = np.select(
            [high_conf, supported, sub["is_big_rmats_event"]],
            [0, 1, 2],
            default=3,
        )
        sub = sub.assign(rank_bucket=rank_bucket).sort_values(
            ["rank_bucket", "abs_dpsi", "total_support"],
            ascending=[True, False, False],
        )
        best = sub.iloc[0]
        rows.append({
            "gene": gene, "role": role, "events": int(len(sub)),
            "big_rmats_events": int(sub["is_big_rmats_event"].sum()),
            "braid_supported": int(supported.sum()),
            "braid_high_confidence": int(high_conf.sum()),
            "best_event_id": best["event_id"], "best_tier": best["tier"],
            "best_dpsi": float(best["dpsi"]),
            "best_ci_low": float(best["ci_low"]),
            "best_ci_high": float(best["ci_high"]),
            "best_ci_excludes_zero": bool(best["reliable"] == "yes")
            if isinstance(best["reliable"], str)
            else bool(best["reliable"]),
            "best_rmats_fdr": float(best["rmats_fdr"]),
            "best_total_support": int(best["total_support"]),
        })
    return pd.DataFrame(rows).sort_values(
        ["braid_high_confidence", "braid_supported", "big_rmats_events"],
        ascending=[False, False, False],
    )


def save_summary(
    p: Paths,
    df: pd.DataFrame,
    anchor: pd.DataFrame,
    differential_model: str,
) -> dict[str, object]:
    anchor.to_csv(p.anchor_tsv, sep="\t", index=False)
    tier_counts = {str(k): int(v) for k, v in df["tier"].value_counts().to_dict().items()}
    present = anchor[anchor["events"] > 0]
    summary = {
        "dataset": "GSE64357",
        "source": {
            "geo": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64357",
            "study": "Bebee et al. 2015 eLife — Esrp1/Esrp2 double-knockout mouse",
            "samples": {
                "DKO_b1": ["SRR1725976", "SRR1725977"],
                "WT_b2": ["SRR1725983", "SRR1725984"],
            },
            "alignment": "HISAT2 -> mm10 (gencode vM2)",
            "quantification": "rMATS turbo v4.3.0, paired, SE.MATS.JC",
        },
        "sample_convention": (
            "rMATS sample_1 = Esrp1/2 DKO (--b1), sample_2 = WT (--b2); "
            "BRAID dpsi = PSI(DKO) - PSI(WT)."
        ),
        "braid_differential_model": differential_model,
        "events_after_braid_min_support": int(len(df)),
        "rmats_big_events_fdr_lt_0_05_abs_dpsi_ge_0_1": int(df["is_big_rmats_event"].sum()),
        "tier_counts": tier_counts,
        "anchor_genes_total": len(ANCHOR_GENES),
        "anchor_genes_with_se_event": int((anchor["events"] > 0).sum()),
        "anchor_genes_with_big_rmats_event": int((anchor["big_rmats_events"] > 0).sum()),
        "anchor_genes_with_braid_supported_event": int((anchor["braid_supported"] > 0).sum()),
        "anchor_genes_with_braid_high_confidence_event": int(
            (anchor["braid_high_confidence"] > 0).sum()
        ),
        "anchor_genes_with_calibrated_ci_excluding_zero": int(
            (present["best_ci_excludes_zero"]).sum()
        ),
        "primary_anchor_genes_total": len(PRIMARY_ANCHORS),
        "primary_anchor_genes_with_testable_event": int(
            anchor[anchor["gene"].isin(PRIMARY_ANCHORS) & (anchor["events"] > 0)].shape[0]
        ),
        "primary_anchor_genes_with_braid_supported_event": int(
            anchor[anchor["gene"].isin(PRIMARY_ANCHORS) & (anchor["braid_supported"] > 0)].shape[0]
        ),
        "primary_anchor_genes_with_braid_high_confidence_event": int(
            anchor[
                anchor["gene"].isin(PRIMARY_ANCHORS) & (anchor["braid_high_confidence"] > 0)
            ].shape[0]
        ),
        "recovery_metric": (
            "Gene-level recovery: for each anchor gene the best event is taken across all "
            "rMATS event types (by tier, then |dPSI|, then support), and the gene is "
            "recovered when that event is BRAID high-confidence. The high-confidence tier "
            "already requires the calibrated 95% interval to exclude zero (and rMATS "
            "BH-FDR < 0.05), so 'high-confidence' and 'interval excludes zero' are the same "
            "criterion here, not independent evidence. Denominators are the genes with a "
            "testable event (anchor_genes_with_se_event / primary_anchor_genes_with_"
            "testable_event), since genes with no event cannot be recovered. The reported "
            "per-gene dPSI/interval is the max-selected best event (an upper bound on that "
            "gene's effect, not an unbiased estimate). Direction is reported, not asserted "
            "per gene."
        ),
        "limitations": [
            "Regulator-target recovery evidence, not orthogonal RT-PCR validation.",
            "Two DKO vs two WT replicates; anchors are a literature-curated gene set.",
            "MXE targets (e.g. Fgfr2 IIIb/IIIc) are not in the SE table by construction.",
            "Recovery is gene-level (best event of any type), not exon-level on the curated "
            "cassette exon; all recovered anchors here happen to have an SE best event.",
            "Calibrated intervals use the packaged conformal calibrator fit on human RT-PCR "
            "residuals, transferred to mouse; the nominal 95% level is a transfer "
            "assumption, not validated on this dataset.",
        ],
        "outputs": {
            "braid_tsv": str(p.braid_tsv.relative_to(ROOT)),
            "anchor_tsv": str(p.anchor_tsv.relative_to(ROOT)),
            "figure_pdf": str(p.fig_pdf.relative_to(ROOT)),
            "figure_png": str(p.fig_png.relative_to(ROOT)),
            "figure_svg": str(p.fig_svg.relative_to(ROOT)),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    p.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def make_figure(p: Paths, anchor: pd.DataFrame) -> None:
    # Forest plot of recovered anchors: best dPSI(DKO-WT) with calibrated 95% CI.
    recov = anchor[(anchor["events"] > 0) & (anchor["braid_supported"] > 0)].copy()
    recov = recov.sort_values("best_dpsi")
    if recov.empty:
        recov = anchor[anchor["events"] > 0].copy().sort_values("best_dpsi")
    fig, ax = plt.subplots(figsize=(3.5, max(2.2, 0.32 * len(recov) + 0.8)))
    if not recov.empty:
        y = np.arange(len(recov))
        x = recov["best_dpsi"].to_numpy(float)
        xerr = np.vstack([
            x - recov["best_ci_low"].to_numpy(float),
            recov["best_ci_high"].to_numpy(float) - x,
        ])
        hc = recov["braid_high_confidence"].to_numpy() > 0
        colors = np.where(hc, "#08519C", "#6BAED6")
        for yi, xi, lo, hi, ci in zip(y, x, xerr[0], xerr[1], colors):
            ax.errorbar(xi, yi, xerr=[[lo], [hi]], fmt="o", color=ci, ecolor=ci,
                        elinewidth=1.1, capsize=2, markersize=3.5)
        ax.axvline(0, color="#636363", linewidth=0.6)
        ax.set_yticks(y, recov["gene"])
        ax.tick_params(axis="both", labelsize=6.5)
        ax.set_xlabel("ΔPSI (Esrp DKO − WT), calibrated 95% CI", fontsize=7)
        ax.set_title("BRAID recovery of Esrp1/2 cassette-exon targets",
                     fontsize=8, pad=6)
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([0], [0], marker="o", color="#08519C", lw=0,
                       label="high-confidence (CI excludes 0)"),
                Line2D([0], [0], marker="o", color="#6BAED6", lw=0, label="supported"),
            ],
            fontsize=5.6, frameon=False, loc="lower right", handletextpad=0.3,
        )
        ax.margins(y=0.04)
    fig.tight_layout()
    fig.savefig(p.fig_pdf)
    fig.savefig(p.fig_png)
    fig.savefig(p.fig_svg)
    plt.close(fig)


def run(force: bool, replicates: int, differential_model: str) -> dict[str, object]:
    ensure_dirs()
    p = paths()
    run_braid(p, replicates=replicates, differential_model=differential_model, force=force)
    df = add_columns(pd.read_csv(p.braid_tsv, sep="\t"))
    anchor = summarize_anchors(df)
    summary = save_summary(p, df, anchor, differential_model=differential_model)
    make_figure(p, anchor)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate derived files.")
    parser.add_argument("--replicates", type=int, default=500, help="BRAID posterior draws.")
    parser.add_argument(
        "--differential-model",
        choices=DIFFERENTIAL_MODELS,
        default=MANUSCRIPT_DIFFERENTIAL_MODEL,
        help="BRAID differential estimator used for this application reproduction.",
    )
    args = parser.parse_args()
    summary = run(
        force=args.force,
        replicates=args.replicates,
        differential_model=args.differential_model,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
