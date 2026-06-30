#!/usr/bin/env python3
# ruff: noqa: I001
"""Build the BRAID DM1 application analysis for manuscript Figures 3 and 4.

The analysis is intentionally rMATS-first: it starts from the public
GSE201255 skipped-exon rMATS table, runs ``braid differential`` without
realigning reads, and then summarizes whether BRAID-prioritized events recover
known myotonic dystrophy type 1 (DM1) mis-splicing anchors.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "benchmarks"))
import figstyle  # noqa: E402

figstyle.apply()

APP_DIR = ROOT / "benchmarks" / "application_dm1"
RAW_DIR = APP_DIR / "raw"
RMATS_DIR = APP_DIR / "rmats"
RESULTS_DIR = APP_DIR / "results"
FIGURE_DIR = ROOT / "paper" / "figures"
FIGURE_PACKAGE_DIR = ROOT / "outputs" / "figures" / "manuscript"

SE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE201nnn/GSE201255/suppl/"
    "GSE201255_caseVScontrol_SE.MATS.JC.txt.gz"
)
SAMPLE_ORDER_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE201nnn/GSE201255/suppl/"
    "GSE201255_sample_order.txt.gz"
)
DIFFERENTIAL_MODELS = ("sum", "auto", "rep")
MANUSCRIPT_DIFFERENTIAL_MODEL = "sum"
FALLBACK_SAMPLE_COUNTS = {
    "case_samples": 58,
    "control_samples": 28,
    "adult_dm1_cases": 22,
    "congenital_dm1_cases": 36,
    "adult_controls": 7,
    "pediatric_controls": 21,
}

ANCHOR_GENES: dict[str, str] = {
    "CLCN1": "symptom-linked DM1 anchor",
    "INSR": "symptom-linked DM1 anchor",
    "BIN1": "symptom-linked DM1 anchor",
    "MBNL1": "RNA-toxicity/splicing anchor",
    "MBNL2": "RNA-toxicity/splicing anchor",
    "ATP2A1": "muscle-function DM1 anchor",
    "LDB3": "muscle-function DM1 anchor",
    "RYR1": "muscle-function DM1 anchor",
    "NFIX": "DM1 targeted-splicing-panel anchor",
    "CACNA1S": "excitation-contraction candidate",
    "TNNT3": "sarcomere candidate",
    "NEB": "sarcomere candidate",
    "TTN": "sarcomere candidate",
    "OBSCN": "sarcomere candidate",
    "DMD": "muscle-disease candidate",
}

PRIMARY_ANCHORS = {
    "CLCN1",
    "INSR",
    "BIN1",
    "MBNL1",
    "MBNL2",
    "ATP2A1",
    "LDB3",
    "RYR1",
    "NFIX",
}

MUSCLE_GENE_SETS: dict[str, set[str]] = {
    "DM1 anchor": PRIMARY_ANCHORS,
    "Sarcomere/structural": {"NEB", "TTN", "OBSCN", "TNNT3", "MYBPC1", "ACTN2", "MYH7"},
    "Excitation-contraction": {"RYR1", "ATP2A1", "CACNA1S", "BIN1", "CLCN1"},
    "RNA/splicing": {"MBNL1", "MBNL2", "RBFOX1", "RBFOX2", "CELF1", "CELF2", "CLK4"},
}


@dataclass(frozen=True)
class Paths:
    raw_se_gz: Path
    sample_order_gz: Path
    rmats_se: Path
    braid_tsv: Path
    summary_json: Path
    anchor_tsv: Path
    top_candidates_tsv: Path
    top_candidate_genes_tsv: Path
    fig3_pdf: Path
    fig3_png: Path
    fig3_svg: Path
    fig4_pdf: Path
    fig4_png: Path
    fig4_svg: Path


def paths() -> Paths:
    return Paths(
        raw_se_gz=RAW_DIR / "GSE201255_caseVScontrol_SE.MATS.JC.txt.gz",
        sample_order_gz=RAW_DIR / "GSE201255_sample_order.txt.gz",
        rmats_se=RMATS_DIR / "SE.MATS.JC.txt",
        braid_tsv=RESULTS_DIR / "dm1_braid_differential.tsv",
        summary_json=RESULTS_DIR / "dm1_application_summary.json",
        anchor_tsv=RESULTS_DIR / "dm1_anchor_gene_summary.tsv",
        top_candidates_tsv=RESULTS_DIR / "dm1_top_braid_candidates.tsv",
        top_candidate_genes_tsv=RESULTS_DIR / "dm1_top_braid_candidate_genes.tsv",
        fig3_pdf=FIGURE_DIR / "fig3_dm1_application.pdf",
        fig3_png=FIGURE_DIR / "fig3_dm1_application.png",
        fig3_svg=FIGURE_DIR / "fig3_dm1_application.svg",
        fig4_pdf=FIGURE_DIR / "fig4_dm1_candidate_prioritization.pdf",
        fig4_png=FIGURE_DIR / "fig4_dm1_candidate_prioritization.png",
        fig4_svg=FIGURE_DIR / "fig4_dm1_candidate_prioritization.svg",
    )


def ensure_dirs() -> None:
    for d in (RAW_DIR, RMATS_DIR, RESULTS_DIR, FIGURE_DIR, FIGURE_PACKAGE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(dest)


def gunzip_to(src: Path, dest: Path, force: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        return
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with gzip.open(src, "rb") as fin, tmp.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    tmp.replace(dest)


def read_sample_order(sample_order_gz: Path) -> dict[str, int]:
    with gzip.open(sample_order_gz, "rt", encoding="utf-8") as handle:
        rows = [line.strip().replace('"', "").split("\t") for line in handle if line.strip()]
    header = rows[0]
    case_idx = header.index("Case")
    control_idx = header.index("Control")
    cases = [row[case_idx] for row in rows[1:] if len(row) > case_idx and row[case_idx]]
    controls = [
        row[control_idx]
        for row in rows[1:]
        if len(row) > control_idx and row[control_idx]
    ]
    return {
        "case_samples": len(cases),
        "control_samples": len(controls),
        "adult_dm1_cases": sum(sample.startswith("DM1-") for sample in cases),
        "congenital_dm1_cases": sum(sample.startswith("CDM-") for sample in cases),
        "adult_controls": sum(sample.startswith("AdCo-") for sample in controls),
        "pediatric_controls": sum(sample.startswith("PeCo-") for sample in controls),
    }


def run_braid(
    paths_: Paths,
    replicates: int,
    differential_model: str,
    force: bool = False,
) -> None:
    if paths_.braid_tsv.exists() and paths_.braid_tsv.stat().st_size > 0 and not force:
        return
    cmd = [
        sys.executable,
        "-m",
        "braid",
        "differential",
        "--rmats-dir",
        str(RMATS_DIR),
        "--output",
        str(paths_.braid_tsv),
        "--replicates",
        str(replicates),
        "--differential-model",
        differential_model,
        "--effect-cutoff",
        "0.1",
        "--min-support",
        "20",
        "--seed",
        "20260619",
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def _signed_zero_margin(low: float, high: float) -> float:
    if low > 0:
        return low
    if high < 0:
        return -high
    return 0.0


def add_application_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["disease_psi"] = out["ctrl_psi"]
    out["control_psi"] = out["treat_psi"]
    out["disease_minus_control_dpsi"] = out["dpsi"]
    out["ci_width"] = out["ci_high"] - out["ci_low"]
    out["abs_dpsi"] = out["dpsi"].abs()
    out["abs_rmats_dpsi"] = out["rmats_dpsi"].abs()
    out["total_support"] = out["ctrl_support"] + out["treat_support"]
    out["signed_zero_margin"] = [
        _signed_zero_margin(float(lo), float(hi))
        for lo, hi in zip(out["ci_low"], out["ci_high"])
    ]
    out["braid_score"] = out["signed_zero_margin"].abs() * out["prob_large_effect"]
    out["is_anchor_gene"] = out["gene"].isin(ANCHOR_GENES)
    out["is_primary_anchor"] = out["gene"].isin(PRIMARY_ANCHORS)
    out["is_big_rmats_event"] = (out["rmats_fdr"] < 0.05) & (out["abs_rmats_dpsi"] >= 0.1)
    return out


def _first_or_nan(values: Iterable[float]) -> float:
    values = list(values)
    return float(values[0]) if values else float("nan")


def summarize_anchors(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene, role in ANCHOR_GENES.items():
        sub = df[df["gene"] == gene].copy()
        if sub.empty:
            rows.append({
                "gene": gene,
                "role": role,
                "events": 0,
                "big_rmats_events": 0,
                "braid_supported": 0,
                "braid_high_confidence": 0,
                "braid_supported_rmats_significant": 0,
                "braid_high_confidence_rmats_significant": 0,
                "best_event_id": "",
                "best_tier": "",
                "best_is_big_rmats_event": False,
                "best_is_rmats_significant": False,
                "best_dpsi": float("nan"),
                "best_ci_low": float("nan"),
                "best_ci_high": float("nan"),
                "best_braid_score": 0.0,
                "best_rmats_fdr": float("nan"),
                "best_total_support": 0,
            })
            continue
        supported = sub["tier"].isin(["supported", "high-confidence"])
        rmats_sig = sub["rmats_fdr"] < 0.05
        rank_bucket = np.select(
            [
                (sub["tier"] == "high-confidence") & rmats_sig,
                supported & rmats_sig,
                sub["is_big_rmats_event"],
                sub["tier"] == "high-confidence",
                supported,
            ],
            [0, 1, 2, 3, 4],
            default=5,
        )
        sub = (
            sub.assign(anchor_rank_bucket=rank_bucket)
            .sort_values(
                [
                    "anchor_rank_bucket",
                    "braid_score",
                    "abs_rmats_dpsi",
                    "total_support",
                ],
                ascending=[True, False, False, False],
            )
        )
        best = sub.iloc[0]
        rows.append({
            "gene": gene,
            "role": role,
            "events": int(len(sub)),
            "big_rmats_events": int(sub["is_big_rmats_event"].sum()),
            "braid_supported": int(supported.sum()),
            "braid_high_confidence": int((sub["tier"] == "high-confidence").sum()),
            "braid_supported_rmats_significant": int((supported & rmats_sig).sum()),
            "braid_high_confidence_rmats_significant": int(
                ((sub["tier"] == "high-confidence") & rmats_sig).sum()
            ),
            "best_event_id": best["event_id"],
            "best_tier": best["tier"],
            "best_is_big_rmats_event": bool(best["is_big_rmats_event"]),
            "best_is_rmats_significant": bool(best["rmats_fdr"] < 0.05),
            "best_dpsi": float(best["disease_minus_control_dpsi"]),
            "best_ci_low": float(best["ci_low"]),
            "best_ci_high": float(best["ci_high"]),
            "best_braid_score": float(best["braid_score"]),
            "best_rmats_fdr": float(best["rmats_fdr"]),
            "best_total_support": int(best["total_support"]),
        })
    return pd.DataFrame(rows).sort_values(
        ["braid_high_confidence", "best_braid_score", "big_rmats_events"],
        ascending=[False, False, False],
    )


def top_candidates(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    mask = (
        (df["tier"] == "high-confidence")
        & (~df["is_anchor_gene"])
        & (df["rmats_fdr"] < 0.05)
    )
    cols = [
        "event_id",
        "event_type",
        "gene",
        "chrom",
        "disease_minus_control_dpsi",
        "ci_low",
        "ci_high",
        "prob_large_effect",
        "braid_score",
        "rmats_fdr",
        "rmats_dpsi",
        "total_support",
    ]
    return df.loc[mask, cols].sort_values(
        ["braid_score", "prob_large_effect", "total_support"],
        ascending=[False, False, False],
    ).head(n)


def top_candidate_genes(candidates: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "gene",
                "representative_event_id",
                "representative_event_type",
                "max_braid_score",
                "max_abs_dpsi",
                "best_dpsi",
                "best_ci_low",
                "best_ci_high",
                "min_rmats_fdr",
                "max_total_support",
                "events",
            ]
        )
    ranked = candidates.sort_values(
        ["braid_score", "prob_large_effect", "total_support"],
        ascending=[False, False, False],
    )
    reps = ranked.drop_duplicates("gene", keep="first").set_index("gene")
    grouped = ranked.groupby("gene", as_index=False).agg(
        max_braid_score=("braid_score", "max"),
        max_abs_dpsi=("disease_minus_control_dpsi", lambda x: float(np.abs(x).max())),
        min_rmats_fdr=("rmats_fdr", "min"),
        max_total_support=("total_support", "max"),
        events=("event_id", "count"),
    )
    rows: list[dict[str, object]] = []
    for row in grouped.itertuples(index=False):
        rep = reps.loc[row.gene]
        rows.append({
            "gene": row.gene,
            "representative_event_id": rep["event_id"],
            "representative_event_type": rep["event_type"],
            "max_braid_score": float(row.max_braid_score),
            "max_abs_dpsi": float(row.max_abs_dpsi),
            "best_dpsi": float(rep["disease_minus_control_dpsi"]),
            "best_ci_low": float(rep["ci_low"]),
            "best_ci_high": float(rep["ci_high"]),
            "min_rmats_fdr": float(row.min_rmats_fdr),
            "max_total_support": int(row.max_total_support),
            "events": int(row.events),
        })
    return (
        pd.DataFrame(rows)
        .sort_values(["max_braid_score", "max_total_support"], ascending=[False, False])
        .head(n)
    )


def category_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    hc_genes = set(df.loc[df["tier"] == "high-confidence", "gene"])
    for label, genes in MUSCLE_GENE_SETS.items():
        present = sorted(hc_genes & genes)
        rows.append({
            "category": label,
            "high_confidence_genes": len(present),
            "genes": ", ".join(present),
        })
    return pd.DataFrame(rows)


def save_summary(
    paths_: Paths,
    df: pd.DataFrame,
    sample_counts: dict[str, int],
    differential_model: str,
) -> dict[str, object]:
    anchor = summarize_anchors(df)
    candidates = top_candidates(df)
    candidate_genes = top_candidate_genes(candidates)
    categories = category_counts(df)
    anchor.to_csv(paths_.anchor_tsv, sep="\t", index=False)
    candidates.to_csv(paths_.top_candidates_tsv, sep="\t", index=False)
    candidate_genes.to_csv(paths_.top_candidate_genes_tsv, sep="\t", index=False)
    categories.to_csv(RESULTS_DIR / "dm1_high_confidence_categories.tsv", sep="\t", index=False)

    tier_counts = {str(k): int(v) for k, v in df["tier"].value_counts().to_dict().items()}
    summary = {
        "dataset": "GSE201255",
        "source": {
            "rmats_se_url": SE_URL,
            "sample_order_url": SAMPLE_ORDER_URL,
            "geo": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE201255",
        },
        "sample_convention": (
            "GSE201255 supplementary rMATS sample_1 is disease case and sample_2 is control; "
            "BRAID dpsi is therefore disease minus control for this application."
        ),
        "sample_counts": sample_counts,
        "braid_differential_model": differential_model,
        "events_after_braid_min_support": int(len(df)),
        "rmats_big_events_fdr_lt_0_05_abs_dpsi_ge_0_1": int(df["is_big_rmats_event"].sum()),
        "tier_counts": tier_counts,
        "anchor_genes_total": len(ANCHOR_GENES),
        "anchor_genes_with_big_rmats_event": int((anchor["big_rmats_events"] > 0).sum()),
        "anchor_genes_with_braid_supported_event": int((anchor["braid_supported"] > 0).sum()),
        "anchor_genes_with_braid_high_confidence_event": int(
            (anchor["braid_high_confidence"] > 0).sum()
        ),
        "anchor_genes_with_braid_high_confidence_rmats_significant_event": int(
            (anchor["braid_high_confidence_rmats_significant"] > 0).sum()
        ),
        "primary_anchor_genes_total": len(PRIMARY_ANCHORS),
        "primary_anchor_genes_with_braid_high_confidence_event": int(
            anchor[
                anchor["gene"].isin(PRIMARY_ANCHORS)
                & (anchor["braid_high_confidence"] > 0)
            ].shape[0]
        ),
        "primary_anchor_genes_with_braid_high_confidence_rmats_significant_event": int(
            anchor[
                anchor["gene"].isin(PRIMARY_ANCHORS)
                & (anchor["braid_high_confidence_rmats_significant"] > 0)
            ].shape[0]
        ),
        "top_non_anchor_candidates_written": int(len(candidates)),
        "top_non_anchor_candidate_genes_written": int(len(candidate_genes)),
        "braid_score_definition": (
            "abs(calibrated distance from zero) multiplied by prob_large_effect; "
            "zero when the calibrated interval crosses zero."
        ),
        "outputs": {
            "braid_tsv": str(paths_.braid_tsv.relative_to(ROOT)),
            "anchor_tsv": str(paths_.anchor_tsv.relative_to(ROOT)),
            "top_candidates_tsv": str(paths_.top_candidates_tsv.relative_to(ROOT)),
            "top_candidate_genes_tsv": str(paths_.top_candidate_genes_tsv.relative_to(ROOT)),
            "figure_3_pdf": str(paths_.fig3_pdf.relative_to(ROOT)),
            "figure_3_png": str(paths_.fig3_png.relative_to(ROOT)),
            "figure_3_svg": str(paths_.fig3_svg.relative_to(ROOT)),
            "figure_4_pdf": str(paths_.fig4_pdf.relative_to(ROOT)),
            "figure_4_png": str(paths_.fig4_png.relative_to(ROOT)),
            "figure_4_svg": str(paths_.fig4_svg.relative_to(ROOT)),
        },
    }
    paths_.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _setup_plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=10)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def fig3_source_data(df: pd.DataFrame, anchor: pd.DataFrame) -> dict[str, pd.DataFrame]:
    big_rmats = df["is_big_rmats_event"]
    supported = df["tier"].isin(["supported", "high-confidence"])
    high_confidence = df["tier"] == "high-confidence"
    funnel = pd.DataFrame(
        {
            "panel": ["A"] * 4,
            "step": [
                "BRAID processed",
                "rMATS FDR<0.05 |dPSI|>=0.1",
                "BRAID supported among rMATS large-effect",
                "BRAID high-confidence among rMATS large-effect",
            ],
            "count": [
                int(len(df)),
                int(big_rmats.sum()),
                int((big_rmats & supported).sum()),
                int((big_rmats & high_confidence).sum()),
            ],
        }
    )

    rng = np.random.default_rng(20260619)
    plot_df = df.copy()
    if len(plot_df) > 50000:
        plot_df = plot_df.iloc[rng.choice(len(plot_df), size=50000, replace=False)].copy()
    fdr_cap = 50
    plot_df["neg_log10_rmats_fdr_capped"] = np.minimum(
        -np.log10(plot_df["rmats_fdr"].clip(lower=10 ** -fdr_cap)),
        fdr_cap,
    )
    plot_df["is_panel_b_primary_anchor_highlight"] = (
        plot_df["is_primary_anchor"]
        & (plot_df["tier"] == "high-confidence")
        & (plot_df["rmats_fdr"] < 0.05)
    )
    highlighted = df[
        df["is_primary_anchor"]
        & (df["tier"] == "high-confidence")
        & (df["rmats_fdr"] < 0.05)
    ].copy()
    highlighted["neg_log10_rmats_fdr_capped"] = np.minimum(
        -np.log10(highlighted["rmats_fdr"].clip(lower=10 ** -fdr_cap)),
        fdr_cap,
    )

    anchor_plot = anchor[anchor["braid_high_confidence_rmats_significant"] > 0].head(10).copy()
    anchor_plot = anchor_plot.sort_values("best_dpsi")
    score_plot = (
        anchor[anchor["best_braid_score"] > 0]
        .sort_values("best_braid_score", ascending=False)
        .head(12)
        .copy()
    )
    scatter_cols = [
        "event_id",
        "gene",
        "tier",
        "disease_minus_control_dpsi",
        "rmats_fdr",
        "neg_log10_rmats_fdr_capped",
        "is_anchor_gene",
        "is_primary_anchor",
        "is_panel_b_primary_anchor_highlight",
    ]
    highlighted_cols = [
        "event_id",
        "gene",
        "tier",
        "disease_minus_control_dpsi",
        "rmats_fdr",
        "neg_log10_rmats_fdr_capped",
    ]
    return {
        "panel_A_funnel": funnel,
        "panel_B_scatter": plot_df[scatter_cols].copy(),
        "panel_B_highlighted": highlighted[highlighted_cols].copy(),
        "panel_C_anchor_intervals": anchor_plot.copy(),
        "panel_D_anchor_scores": score_plot.copy(),
    }


def fig4_source_data(df: pd.DataFrame, candidate_genes: pd.DataFrame) -> dict[str, pd.DataFrame]:
    top = candidate_genes.head(15).copy().sort_values("max_braid_score")
    high = df[df["tier"] == "high-confidence"].copy()
    gene_summary = (
        high.groupby("gene", as_index=False)
        .agg(
            max_abs_dpsi=("abs_dpsi", "max"),
            max_braid_score=("braid_score", "max"),
            events=("event_id", "count"),
            total_support=("total_support", "max"),
        )
        .sort_values("max_braid_score", ascending=False)
        .head(60)
    )
    hc_genes = set(high["gene"])
    categories = pd.DataFrame(
        [
            {
                "category": label,
                "high_confidence_genes": len(hc_genes & genes),
                "genes": ", ".join(sorted(hc_genes & genes)),
            }
            for label, genes in MUSCLE_GENE_SETS.items()
        ]
    )
    return {
        "panel_A_top_candidates": top,
        "panel_B_gene_summary": gene_summary,
        "panel_C_categories": categories,
    }


def write_docx_legend(path: Path, paragraphs: list[str]) -> None:
    doc = Document()
    for paragraph in paragraphs:
        doc.add_paragraph(paragraph)
    doc.save(path)


def write_figure_package(
    figure_id: str,
    script: Path,
    inputs: list[str],
    figure_outputs: dict[str, Path],
    source_sheets: dict[str, pd.DataFrame],
    legend_paragraphs: list[str],
    panel_claims: dict[str, str],
    transformations: list[str],
    known_limitations: list[str],
) -> dict[str, str]:
    package_dir = FIGURE_PACKAGE_DIR / figure_id
    package_dir.mkdir(parents=True, exist_ok=True)
    source_xlsx = package_dir / f"{figure_id.lower()}_source_data.xlsx"
    legend_docx = package_dir / f"{figure_id.lower()}_legend.docx"
    manifest_json = package_dir / f"{figure_id.lower()}_manifest.json"
    validation_txt = package_dir / f"{figure_id.lower()}_validation.txt"

    with pd.ExcelWriter(source_xlsx, engine="openpyxl") as writer:
        for sheet_name, data in source_sheets.items():
            data.to_excel(writer, index=False, sheet_name=sheet_name[:31])
        pd.DataFrame(
            [
                {"key": "figure_id", "value": figure_id},
                {"key": "script", "value": _rel(script)},
                {"key": "generated_at", "value": datetime.now(timezone.utc).isoformat()},
            ]
        ).to_excel(writer, index=False, sheet_name="metadata")

    write_docx_legend(legend_docx, legend_paragraphs)

    package_outputs: dict[str, str] = {}
    for label, src in figure_outputs.items():
        dest = package_dir / f"{figure_id}_{label}{src.suffix}"
        shutil.copyfile(src, dest)
        package_outputs[f"{label}{src.suffix}"] = _rel(dest)

    all_files = [source_xlsx, legend_docx, *[ROOT / rel for rel in package_outputs.values()]]
    manifest = {
        "figure_id": figure_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": _rel(script),
        "inputs": inputs,
        "outputs": package_outputs,
        "source_data": _rel(source_xlsx),
        "legend_docx": _rel(legend_docx),
        "panel_claims": panel_claims,
        "transformations": transformations,
        "validation": {
            "source_sheets": {name: int(len(data)) for name, data in source_sheets.items()},
            "file_hashes": {_rel(path): _sha256(path) for path in all_files},
        },
        "known_limitations": known_limitations,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"{figure_id} validation",
        *[f"{name}_rows={len(data)}" for name, data in source_sheets.items()],
        *[
            f"{label}={ROOT.joinpath(rel).stat().st_size} bytes"
            for label, rel in package_outputs.items()
        ],
        f"source_data={source_xlsx.stat().st_size} bytes",
        f"legend_docx={legend_docx.stat().st_size} bytes",
    ]
    validation_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "source_data": _rel(source_xlsx),
        "legend_docx": _rel(legend_docx),
        "manifest": _rel(manifest_json),
        "validation": _rel(validation_txt),
    }


def make_fig3(paths_: Paths, df: pd.DataFrame, anchor: pd.DataFrame) -> None:
    _setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    funnel_labels = [
        "BRAID\nprocessed",
        "rMATS FDR<0.05\n|dPSI|>=0.1",
        "BRAID\nsupported",
        "BRAID\nhigh-conf.",
    ]
    big_rmats = df["is_big_rmats_event"]
    supported = df["tier"].isin(["supported", "high-confidence"])
    high_confidence = df["tier"] == "high-confidence"
    funnel_counts = [
        len(df),
        int(big_rmats.sum()),
        int((big_rmats & supported).sum()),
        int((big_rmats & high_confidence).sum()),
    ]
    colors = ["#bdbdbd", "#9ecae1", "#6baed6", "#2171b5"]
    ax_a.bar(range(len(funnel_labels)), funnel_counts, color=colors, edgecolor="white")
    ax_a.set_yscale("log")
    ax_a.set_ylim(10, max(funnel_counts) * 1.8)
    ax_a.set_xticks(range(len(funnel_labels)), funnel_labels, rotation=0)
    ax_a.set_ylabel("SE events (log scale)")
    ax_a.set_title("DM1 rMATS table filtered by BRAID")
    for i, value in enumerate(funnel_counts):
        ax_a.text(i, value, f"{value:,}", ha="center", va="bottom", fontsize=6)
    _panel_label(ax_a, "A")

    rng = np.random.default_rng(20260619)
    plot_df = df.copy()
    if len(plot_df) > 50000:
        plot_df = plot_df.iloc[rng.choice(len(plot_df), size=50000, replace=False)].copy()
    fdr_cap = 50
    plot_df["neg_log10_rmats_fdr_capped"] = np.minimum(
        -np.log10(plot_df["rmats_fdr"].clip(lower=10 ** -fdr_cap)),
        fdr_cap,
    )
    tier_color = {
        "not-significant": "#d9d9d9",
        "significant": "#9ecae1",
        "supported": "#fdae6b",
        "high-confidence": "#de2d26",
    }
    ax_b.scatter(
        plot_df["disease_minus_control_dpsi"],
        plot_df["neg_log10_rmats_fdr_capped"],
        s=3,
        c=plot_df["tier"].map(tier_color).fillna("#bdbdbd"),
        alpha=0.35,
        linewidths=0,
    )
    highlighted = df[
        df["is_primary_anchor"]
        & (df["tier"] == "high-confidence")
        & (df["rmats_fdr"] < 0.05)
    ].copy()
    highlighted["neg_log10_rmats_fdr_capped"] = np.minimum(
        -np.log10(highlighted["rmats_fdr"].clip(lower=10 ** -fdr_cap)),
        fdr_cap,
    )
    ax_b.scatter(
        highlighted["disease_minus_control_dpsi"],
        highlighted["neg_log10_rmats_fdr_capped"],
        s=16,
        facecolors="none",
        edgecolors="black",
        linewidths=0.6,
        label="known DM1 anchors",
    )
    ax_b.axvline(0, color="#636363", linewidth=0.5)
    ax_b.axvline(0.1, color="#969696", linewidth=0.4, linestyle="--")
    ax_b.axvline(-0.1, color="#969696", linewidth=0.4, linestyle="--")
    ax_b.set_xlabel("Disease-control dPSI")
    ax_b.set_ylabel("-log10 rMATS FDR (capped at 50)")
    ax_b.set_title("BRAID tiers on disease splicing events")
    ax_b.legend(frameon=False, loc="upper left")
    _panel_label(ax_b, "B")

    anchor_plot = anchor[anchor["braid_high_confidence_rmats_significant"] > 0].head(10).copy()
    anchor_plot = anchor_plot.sort_values("best_dpsi")
    y = np.arange(len(anchor_plot))
    x = anchor_plot["best_dpsi"].to_numpy(float)
    xerr = np.vstack([
        x - anchor_plot["best_ci_low"].to_numpy(float),
        anchor_plot["best_ci_high"].to_numpy(float) - x,
    ])
    ax_c.errorbar(x, y, xerr=xerr, fmt="o", color="#08519c", ecolor="#9ecae1",
                  elinewidth=1, capsize=2, markersize=3)
    ax_c.axvline(0, color="#636363", linewidth=0.6)
    ax_c.set_yticks(y, anchor_plot["gene"])
    ax_c.set_xlabel("Disease-control dPSI with calibrated 95% CI")
    ax_c.set_title("Recovered DM1 anchor events")
    _panel_label(ax_c, "C")

    score_plot = (
        anchor[anchor["best_braid_score"] > 0]
        .sort_values("best_braid_score", ascending=False)
        .head(12)
    )
    ax_d.barh(
        np.arange(len(score_plot)),
        score_plot["best_braid_score"],
        color=np.where(score_plot["gene"].isin(PRIMARY_ANCHORS), "#31a354", "#756bb1"),
    )
    ax_d.set_yticks(np.arange(len(score_plot)), score_plot["gene"])
    ax_d.invert_yaxis()
    ax_d.set_xlabel("BRAID score")
    ax_d.set_title("Anchor prioritization by calibrated margin")
    _panel_label(ax_d, "D")

    fig.tight_layout()
    fig.savefig(paths_.fig3_pdf)
    fig.savefig(paths_.fig3_png)
    fig.savefig(paths_.fig3_svg)
    plt.close(fig)


def make_fig4(
    paths_: Paths,
    df: pd.DataFrame,
    candidate_genes: pd.DataFrame,
) -> None:
    _setup_plot_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.7),
        gridspec_kw={"width_ratios": [1.2, 1, 1]},
        constrained_layout=True,
    )
    ax_a, ax_b, ax_c = axes

    top = candidate_genes.head(15).copy().sort_values("max_braid_score")
    ax_a.barh(np.arange(len(top)), top["max_braid_score"], color="#de2d26")
    ax_a.set_yticks(np.arange(len(top)), top["gene"])
    ax_a.set_xlabel("BRAID score")
    ax_a.set_title("A. Top non-anchor candidate genes", pad=8)

    high = df[df["tier"] == "high-confidence"].copy()
    gene_summary = (
        high.groupby("gene", as_index=False)
        .agg(
            max_abs_dpsi=("abs_dpsi", "max"),
            max_braid_score=("braid_score", "max"),
            events=("event_id", "count"),
            total_support=("total_support", "max"),
        )
        .sort_values("max_braid_score", ascending=False)
        .head(60)
    )
    ax_b.scatter(
        gene_summary["max_abs_dpsi"],
        gene_summary["max_braid_score"],
        s=np.clip(np.sqrt(gene_summary["total_support"]), 5, 45),
        c="#3182bd",
        alpha=0.65,
        linewidths=0,
    )
    for gene in ("NEB", "TTN", "TNNT3", "OBSCN", "CLK4", "MYBPC1"):
        row = gene_summary[gene_summary["gene"] == gene]
        if not row.empty:
            x_value = float(row["max_abs_dpsi"].iloc[0])
            y_value = float(row["max_braid_score"].iloc[0])
            right_aligned = x_value > 0.8
            ax_b.annotate(
                gene,
                xy=(x_value, y_value),
                xytext=(-2 if right_aligned else 2, 0),
                textcoords="offset points",
                ha="right" if right_aligned else "left",
                va="center",
                fontsize=5,
            )
    ax_b.set_xlabel("Max |dPSI|")
    ax_b.set_ylabel("Max BRAID score")
    ax_b.margins(x=0.08, y=0.08)
    ax_b.set_title("B. High-confidence genes", pad=8)

    rows = []
    hc_genes = set(high["gene"])
    for label, genes in MUSCLE_GENE_SETS.items():
        rows.append((label, len(hc_genes & genes)))
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    ax_c.bar(range(len(values)), values, color=["#31a354", "#756bb1", "#fd8d3c", "#6baed6"])
    ax_c.set_xticks(range(len(values)), labels, rotation=35, ha="right")
    ax_c.set_ylabel("High-confidence genes")
    ax_c.set_title("C. Curated biological classes", pad=8)
    for i, value in enumerate(values):
        ax_c.text(i, value, str(value), ha="center", va="bottom", fontsize=6)

    fig.savefig(paths_.fig4_pdf)
    fig.savefig(paths_.fig4_png)
    fig.savefig(paths_.fig4_svg)
    plt.close(fig)


def write_dm1_figure_packages(
    paths_: Paths,
    df: pd.DataFrame,
    anchor: pd.DataFrame,
    candidate_genes: pd.DataFrame,
) -> dict[str, dict[str, str]]:
    script = Path(__file__).resolve()
    fig3_package = write_figure_package(
        figure_id="F5",
        script=script,
        inputs=[
            "benchmarks/application_dm1/results/dm1_braid_differential.tsv",
            "benchmarks/application_dm1/results/dm1_anchor_gene_summary.tsv",
            "benchmarks/application_dm1/results/dm1_application_summary.json",
        ],
        figure_outputs={
            "dm1_application_pdf": paths_.fig3_pdf,
            "dm1_application_png": paths_.fig3_png,
            "dm1_application_svg": paths_.fig3_svg,
        },
        source_sheets=fig3_source_data(df, anchor),
        legend_paragraphs=[
            "Figure 3. DM1 public rMATS application.",
            (
                "A, GSE201255 skipped-exon events after BRAID processing, rMATS "
                "large-effect significance, BRAID support, and BRAID high-confidence "
                "filtering. B, BRAID confidence tiers over disease-control Delta PSI "
                "and rMATS FDR; black circles mark curated primary DM1 anchor events. "
                "C, Representative recovered anchor events with calibrated 95% "
                "intervals. D, Anchor prioritization by BRAID score."
            ),
        ],
        panel_claims={
            "A": "BRAID high-confidence filtering retains 68 rMATS large-effect SE events.",
            "B": "Primary DM1 anchor events are recovered within the high-confidence tier.",
            "C": (
                "Recovered anchor intervals exclude zero for the displayed "
                "high-confidence events."
            ),
            "D": "Known disease anchors can be ranked by calibrated margin on the BRAID scale.",
        },
        transformations=[
            "Downloaded the public GSE201255 case-versus-control rMATS skipped-exon table.",
            (
                "Ran braid differential with 500 posterior draws, effect cutoff "
                "0.1, and min support 20."
            ),
            "Ranked anchor events by rMATS-significant high-confidence status before BRAID score.",
        ],
        known_limitations=[
            (
                "This public rMATS application is disease-prioritization evidence, "
                "not orthogonal validation."
            ),
            (
                "Panel B plots a deterministic 50,000-event sample plus all "
                "highlighted primary anchors."
            ),
        ],
    )
    fig4_package = write_figure_package(
        figure_id="F6",
        script=script,
        inputs=[
            "benchmarks/application_dm1/results/dm1_top_braid_candidates.tsv",
            "benchmarks/application_dm1/results/dm1_top_braid_candidate_genes.tsv",
            "benchmarks/application_dm1/results/dm1_high_confidence_categories.tsv",
            "benchmarks/application_dm1/results/dm1_application_summary.json",
        ],
        figure_outputs={
            "dm1_candidates_pdf": paths_.fig4_pdf,
            "dm1_candidates_png": paths_.fig4_png,
            "dm1_candidates_svg": paths_.fig4_svg,
        },
        source_sheets=fig4_source_data(df, candidate_genes),
        legend_paragraphs=[
            (
                "Figure 4. BRAID score-based candidate prioritization in the DM1 "
                "public rMATS analysis."
            ),
            (
                "A, Top non-anchor candidate genes after requiring rMATS FDR < 0.05 "
                "and BRAID high-confidence status. B, Gene-level high-confidence "
                "events positioned by maximum absolute Delta PSI and maximum BRAID "
                "score. C, Recovery of curated DM1, sarcomere/structural, "
                "excitation-contraction, and RNA/splicing gene classes."
            ),
        ],
        panel_claims={
            "A": "Non-anchor high-confidence events aggregate to ranked candidate genes.",
            "B": "Candidate genes and known anchors occupy a shared BRAID score scale.",
            (
                "C"
            ): (
                "Curated DM1 and muscle-related classes are represented among "
                "high-confidence genes."
            ),
        },
        transformations=[
            "Excluded curated anchors from panel A candidate ranking.",
            "Aggregated high-confidence non-anchor events by gene using maximum BRAID score.",
            "Counted curated biological classes among high-confidence genes.",
        ],
        known_limitations=[
            "Non-anchor genes are prioritized hypotheses and are not independently validated here.",
            "Curated classes are descriptive overlays, not enrichment tests.",
        ],
    )
    return {"F5": fig3_package, "F6": fig4_package}


def load_sample_counts(sample_order_gz: Path, force: bool = False) -> dict[str, int]:
    try:
        download(SAMPLE_ORDER_URL, sample_order_gz, force=force)
        return read_sample_order(sample_order_gz)
    except Exception as exc:  # pragma: no cover - network fallback for reviewer runs
        print(
            f"Warning: could not read GSE201255 sample-order file ({exc}); "
            "using published sample counts.",
            file=sys.stderr,
        )
        return dict(FALLBACK_SAMPLE_COUNTS)


def ensure_rmats_table(paths_: Paths, force: bool = False) -> None:
    if paths_.rmats_se.exists() and paths_.rmats_se.stat().st_size > 0 and not force:
        return
    download(SE_URL, paths_.raw_se_gz, force=force)
    gunzip_to(paths_.raw_se_gz, paths_.rmats_se, force=force)


def run(force: bool, replicates: int, differential_model: str) -> dict[str, object]:
    ensure_dirs()
    p = paths()
    ensure_rmats_table(p, force=force)
    sample_counts = load_sample_counts(p.sample_order_gz, force=force)
    run_braid(
        p,
        replicates=replicates,
        differential_model=differential_model,
        force=force,
    )
    df = add_application_columns(pd.read_csv(p.braid_tsv, sep="\t"))
    summary = save_summary(p, df, sample_counts, differential_model=differential_model)
    anchor = pd.read_csv(p.anchor_tsv, sep="\t")
    candidate_genes = pd.read_csv(p.top_candidate_genes_tsv, sep="\t")
    make_fig3(p, df, anchor)
    make_fig4(p, df, candidate_genes)
    summary["figure_packages"] = write_dm1_figure_packages(p, df, anchor, candidate_genes)
    p.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate downloaded and derived files.",
    )
    parser.add_argument("--replicates", type=int, default=500, help="BRAID posterior draws.")
    parser.add_argument(
        "--differential-model",
        choices=DIFFERENTIAL_MODELS,
        default=MANUSCRIPT_DIFFERENTIAL_MODEL,
        help=(
            "BRAID differential estimator. The manuscript DM1 result uses 'sum' "
            "for exact reproduction; 'auto' is useful as a replicate-aware sensitivity run."
        ),
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
