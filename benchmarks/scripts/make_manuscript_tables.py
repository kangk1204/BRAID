#!/usr/bin/env python3
"""Build manuscript tables and supplementary source-data package.

The script collects the already-regenerated benchmark source data into one
submission-facing package:

  * Table 1: common four-method RT-PCR coverage benchmark
  * Table 2: SRS354082 full rMATS-matched benchmark
  * Supplementary tables for dataset-stratified coverage, conformal transfer,
    TRA2 detection/MCC, and the DM1 application

It intentionally reuses existing benchmark code and figure source workbooks
instead of reimplementing statistics in a second location.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "tables" / "manuscript"
PAPER_SUPP_DIR = ROOT / "paper" / "supplementary_data"

F2_SOURCE = ROOT / "outputs" / "figures" / "manuscript" / "F2" / "f2_source_data.xlsx"
S1_SOURCE = ROOT / "outputs" / "figures" / "manuscript" / "S1" / "s1_source_data.xlsx"
HEADTOHEAD_JSON = ROOT / "benchmarks" / "results" / "headtohead_coverage.json"
SUM_VS_REP_JSON = ROOT / "benchmarks" / "results" / "sum_vs_rep_eval.json"
PER_TIER_JSON = ROOT / "benchmarks" / "results" / "per_tier_validation.json"
DM1_SUMMARY_JSON = (
    ROOT / "benchmarks" / "application_dm1" / "results" / "dm1_application_summary.json"
)
DM1_ANCHORS = (
    ROOT / "benchmarks" / "application_dm1" / "results" / "dm1_anchor_gene_summary.tsv"
)
DM1_CANDIDATES = (
    ROOT / "benchmarks" / "application_dm1" / "results" / "dm1_top_braid_candidate_genes.tsv"
)
DM1_CATEGORIES = (
    ROOT / "benchmarks" / "application_dm1" / "results" / "dm1_high_confidence_categories.tsv"
)

TRA2_SE = ROOT / "data" / "public_benchmarks" / "GSE59335" / "rmats" / "SE.MATS.JC.txt"
TRA2_VALIDATED = (
    ROOT / "data" / "public_benchmarks" / "GSE59335" / "targets" / "validated_events.tsv"
)
TRA2_FAILED = (
    ROOT / "data" / "public_benchmarks" / "GSE59335" / "targets" / "failed_events.tsv"
)
TRA2_BETAS = ROOT / "benchmarks" / "headtohead" / "tra2_betas_intervals.tsv"
CIRC_SE = ROOT / "data" / "public_benchmarks" / "GSE54651" / "rmats" / "SE.MATS.JC.txt"
CIRC_TSV = ROOT / "data" / "public_benchmarks" / "meta" / "gse54651_circadian_positive_events.tsv"
CIRC_BETAS = ROOT / "benchmarks" / "headtohead" / "circ_betas_intervals.tsv"
SRS_SE = ROOT / "data" / "public_benchmarks" / "SRS354082" / "rmats" / "SE.MATS.JC.txt"
SRS_TRUTH = (
    ROOT / "data" / "public_benchmarks" / "meta" / "rmats_pc3e_gs689_positive_events.tsv"
)
SRS_BETAS = ROOT / "benchmarks" / "headtohead" / "srs_betas_intervals.tsv"

HEADTOHEAD_DIR = ROOT / "benchmarks" / "headtohead"
if str(HEADTOHEAD_DIR) not in sys.path:
    sys.path.insert(0, str(HEADTOHEAD_DIR))

import cross_dataset_transfer as CDT  # noqa: E402
import head_to_head_coverage as H  # noqa: E402

METHOD_LABELS = {
    "MAJIQ": "MAJIQ v3 Gaussian",
    "betAS": "betAS",
    "betAS(real)": "betAS",
    "rMATS": "rMATS IncLevel t-CI",
    "rMATS-perRep": "rMATS-perRep",
    "BRAID-conformal": "BRAID conformal",
    "BRAID-conformal-abs": "BRAID conformal-abs",
}
METHOD_ORDER = ["MAJIQ", "betAS", "rMATS", "BRAID-conformal"]
DATASET_ORDER = ["TRA2", "Circadian", "SRS354082"]
# sum_vs_rep_eval.json dataset keys -> manuscript dataset labels (display order).
DIFF_MODEL_DATASETS = [
    ("TRA2 (GSE59335)", "TRA2"),
    ("Circadian (GSE54651)", "Circadian"),
    ("SRS354082 (PC3E/GS689)", "SRS354082"),
]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        coerced = float(value)
        return None if math.isnan(coerced) else coerced
    return value


def _fmt3(value: float) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.3f}"


def _ci(low: float, high: float) -> str:
    return f"{_fmt3(low)}-{_fmt3(high)}"


def _ordered(df: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    rank = {value: i for i, value in enumerate(order)}
    out = df.copy()
    out["_rank"] = out[column].map(rank).fillna(len(rank))
    out = out.sort_values("_rank", kind="mergesort").drop(columns=["_rank"])
    return out.reset_index(drop=True)


def _sort_by_orders(df: pd.DataFrame, specs: list[tuple[str, list[str]]]) -> pd.DataFrame:
    out = df.copy()
    rank_cols = []
    for idx, (column, order) in enumerate(specs):
        rank_col = f"_rank_{idx}"
        rank = {value: i for i, value in enumerate(order)}
        out[rank_col] = out[column].map(rank).fillna(len(rank))
        rank_cols.append(rank_col)
    out = out.sort_values(rank_cols, kind="mergesort").drop(columns=rank_cols)
    return out.reset_index(drop=True)


def build_table_1() -> pd.DataFrame:
    pooled = pd.read_excel(F2_SOURCE, sheet_name="panel_BC_pooled")
    pooled = _ordered(pooled, "method", METHOD_ORDER)
    rows = []
    for _, row in pooled.iterrows():
        rows.append(
            {
                "method": METHOD_LABELS.get(row["method"], row["method"]),
                "coverage_at_95": float(row["coverage"]),
                "wilson_95_ci": _ci(row["wilson_low"], row["wilson_high"]),
                "mean_width": float(row["mean_width"]),
                "interval_score": float(row["interval_score"]),
                "n": int(row["n"]),
                "covered": int(row["covered"]),
                "source": _rel(F2_SOURCE),
                "source_sheet": "panel_BC_pooled",
            }
        )
    return pd.DataFrame(rows)


def build_supp_table_1() -> pd.DataFrame:
    stratified = pd.read_excel(F2_SOURCE, sheet_name="panel_A_coverage")
    stratified = stratified[stratified["dataset"].isin(DATASET_ORDER)]
    stratified = _sort_by_orders(
        stratified, [("dataset", DATASET_ORDER), ("method", METHOD_ORDER)]
    )
    rows = []
    for _, row in stratified.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "n": int(row["n"]),
                "method": METHOD_LABELS.get(row["method"], row["method"]),
                "coverage_at_95": float(row["coverage"]),
                "wilson_95_ci": _ci(row["wilson_low"], row["wilson_high"]),
                "mean_width": float(row["mean_width"]),
                "interval_score": float(row["interval_score"]),
                "source": _rel(F2_SOURCE),
                "source_sheet": "panel_A_coverage",
            }
        )
    return pd.DataFrame(rows)


def build_table_2() -> tuple[pd.DataFrame, dict[str, Any]]:
    res = H.run_dataset(
        str(SRS_SE),
        str(SRS_TRUTH),
        None,
        "SRS354082",
        seed=7,
        betas_tsv=str(SRS_BETAS),
        swap_groups=False,
    )
    method_order = ["rMATS-perRep", "betAS(real)", "BRAID-conformal-abs"]
    rows = []
    for method in method_order:
        stats = res["methods"][method]
        rows.append(
            {
                "method": METHOD_LABELS.get(method, method),
                "coverage_at_95": float(stats["coverage95"]),
                "mean_width": float(stats["width95"]),
                "interval_score": float(stats["interval_score95"]),
                "direction_accuracy": float(stats["direction_acc"]),
                "n_matched": int(res["n_matched"]),
                "n_positive": int(res["n_positive"]),
                "n_negative": int(res["n_negative"]),
                "source": "benchmarks/headtohead/head_to_head_coverage.py",
            }
        )
    return pd.DataFrame(rows), res


def build_supp_table_2() -> pd.DataFrame:
    alpha = 0.05
    tra2_targets = H.load_targets(str(TRA2_VALIDATED), str(TRA2_FAILED))
    circ_targets = H.load_circadian_targets(str(CIRC_TSV))
    pt_tra2, tr_tra2, sup_tra2 = CDT.build_arrays(
        str(TRA2_SE), tra2_targets, swap_groups=True
    )
    pt_circ, tr_circ, sup_circ = CDT.build_arrays(
        str(CIRC_SE), circ_targets, swap_groups=True
    )

    rows = []
    qg_tra2, qb_tra2 = CDT.fit_abs_conformal(pt_tra2, tr_tra2, sup_tra2, alpha)
    lo_circ, hi_circ = CDT.apply_abs_conformal(pt_circ, sup_circ, qg_tra2, qb_tra2)
    cov, width = CDT.cov_width(lo_circ, hi_circ, tr_circ)
    rows.append(
        {
            "fit_dataset": "TRA2",
            "deployment_dataset": "Circadian",
            "method": "BRAID conformal transfer",
            "coverage_at_95": cov,
            "mean_width": width,
            "fit_n": int(pt_tra2.size),
            "deployment_n": int(pt_circ.size),
            "fit_global_half_width": float(qg_tra2),
            "source": "benchmarks/headtohead/cross_dataset_transfer.py",
        }
    )
    cov, width = CDT.betas_baseline(str(CIRC_BETAS), tr_circ)
    rows.append(
        {
            "fit_dataset": "None",
            "deployment_dataset": "Circadian",
            "method": "betAS on Circadian",
            "coverage_at_95": cov,
            "mean_width": width,
            "fit_n": 0,
            "deployment_n": int(pt_circ.size),
            "fit_global_half_width": np.nan,
            "source": _rel(CIRC_BETAS),
        }
    )

    qg_circ, qb_circ = CDT.fit_abs_conformal(pt_circ, tr_circ, sup_circ, alpha)
    lo_tra2, hi_tra2 = CDT.apply_abs_conformal(pt_tra2, sup_tra2, qg_circ, qb_circ)
    cov, width = CDT.cov_width(lo_tra2, hi_tra2, tr_tra2)
    rows.append(
        {
            "fit_dataset": "Circadian",
            "deployment_dataset": "TRA2",
            "method": "BRAID conformal transfer",
            "coverage_at_95": cov,
            "mean_width": width,
            "fit_n": int(pt_circ.size),
            "deployment_n": int(pt_tra2.size),
            "fit_global_half_width": float(qg_circ),
            "source": "benchmarks/headtohead/cross_dataset_transfer.py",
        }
    )
    cov, width = CDT.betas_baseline(str(TRA2_BETAS), tr_tra2)
    rows.append(
        {
            "fit_dataset": "None",
            "deployment_dataset": "TRA2",
            "method": "betAS on TRA2",
            "coverage_at_95": cov,
            "mean_width": width,
            "fit_n": 0,
            "deployment_n": int(pt_tra2.size),
            "fit_global_half_width": np.nan,
            "source": _rel(TRA2_BETAS),
        }
    )
    return pd.DataFrame(rows)


def build_supp_table_diff_model() -> pd.DataFrame:
    """Differential-model robustness: pooled `sum` vs replicate-aware `rep`.

    Reads the committed sum_vs_rep_eval.json (both models through the same cross-fit
    conformal on the full rMATS-matched head-to-head set) and emits the paper-headline
    absolute-residual conformal coverage/width/interval-score per dataset and pooled.
    """
    data = json.loads(SUM_VS_REP_JSON.read_text())
    blocks = [(label, data["per_dataset"][key]) for key, label in DIFF_MODEL_DATASETS]
    blocks.append(("Pooled", data["pooled"]))
    rows = []
    for label, block in blocks:
        for model in ("sum", "rep"):
            stat = block[model]["conformal_abs"]
            rows.append(
                {
                    "dataset": label,
                    "n": int(block["n"]),
                    "model": model,
                    "coverage_at_95": float(stat["coverage95"]),
                    "mean_width": float(stat["mean_width"]),
                    "interval_score": float(stat["interval_score"]),
                    "source": _rel(SUM_VS_REP_JSON),
                }
            )
    return pd.DataFrame(rows)


def build_supp_table_per_tier() -> pd.DataFrame:
    """Confidence-tier validation on the TRA2 RT-PCR panel (76 pos + 36 neg).

    Reads the committed per_tier_validation.json and emits, per tier, the held-out
    cross-fit RT-PCR coverage, the RT-PCR positive rate, and the mean absolute effect,
    plus the positive rate under the transfer (no-TRA2) and shipped-default intervals so
    the monotone stratification is shown robust to the calibration choice.
    """
    data = json.loads(PER_TIER_JSON.read_text())
    cf = data["interval_variants"]["cross_fit"]
    tr = data["interval_variants"]["transfer"]
    df = data["interval_variants"]["default"]
    rows = []
    for tier in data["tier_order"]:
        s = cf[tier]
        if s.get("n", 0) == 0:
            continue
        rows.append(
            {
                "tier": tier,
                "n": int(s["n"]),
                "rtpcr_coverage_crossfit": float(s["rtpcr_coverage"]),
                "rtpcr_positive_rate": float(s["rtpcr_positive_rate"]),
                "mean_abs_dpsi": float(s["mean_abs_dpsi"]),
                "positive_rate_transfer": (
                    float(tr[tier]["rtpcr_positive_rate"]) if tr[tier].get("n") else None
                ),
                "positive_rate_default": (
                    float(df[tier]["rtpcr_positive_rate"]) if df[tier].get("n") else None
                ),
                "source": _rel(PER_TIER_JSON),
            }
        )
    return pd.DataFrame(rows)


def build_supp_table_3() -> pd.DataFrame:
    mcc = pd.read_excel(S1_SOURCE, sheet_name="panel_B_mcc")
    cols = [
        "rule_id",
        "rule_label",
        "n_positive",
        "n_negative",
        "tp",
        "fn",
        "fp",
        "tn",
        "sensitivity",
        "false_positive_rate",
        "specificity",
        "mcc",
    ]
    out = mcc.loc[:, cols].copy()
    out["production_braid_tier"] = out["rule_id"].eq("braid_supported")
    out["source"] = _rel(S1_SOURCE)
    out["source_sheet"] = "panel_B_mcc"
    return out


def build_dm1_summary() -> pd.DataFrame:
    summary = json.loads(DM1_SUMMARY_JSON.read_text())
    rows = []
    for key in [
        "dataset",
        "events_after_braid_min_support",
        "rmats_big_events_fdr_lt_0_05_abs_dpsi_ge_0_1",
        "anchor_genes_total",
        "anchor_genes_with_big_rmats_event",
        "anchor_genes_with_braid_supported_event",
        "anchor_genes_with_braid_high_confidence_event",
        "anchor_genes_with_braid_high_confidence_rmats_significant_event",
        "primary_anchor_genes_total",
        "primary_anchor_genes_with_braid_high_confidence_event",
        "primary_anchor_genes_with_braid_high_confidence_rmats_significant_event",
    ]:
        rows.append({"metric": key, "value": summary[key], "source": _rel(DM1_SUMMARY_JSON)})
    for key, value in summary["sample_counts"].items():
        rows.append(
            {
                "metric": f"sample_counts.{key}",
                "value": value,
                "source": _rel(DM1_SUMMARY_JSON),
            }
        )
    for key, value in summary["tier_counts"].items():
        rows.append(
            {
                "metric": f"tier_counts.{key}",
                "value": value,
                "source": _rel(DM1_SUMMARY_JSON),
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    display = df.loc[:, columns].copy()
    return display.to_markdown(index=False, floatfmt=".3f")


def write_markdown(tables: dict[str, pd.DataFrame], path: Path) -> None:
    parts = [
        "# BRAID Manuscript Tables",
        "",
        "Generated by `benchmarks/scripts/make_manuscript_tables.py`.",
        "",
        "## Table 1. Head-to-head coverage on the common four-method RT-PCR set",
        _markdown_table(
            tables["table_1_common_coverage"],
            ["method", "coverage_at_95", "wilson_95_ci", "mean_width", "interval_score"],
        ),
        "",
        "## Supplementary Table 1. Dataset-stratified common-set coverage",
        _markdown_table(
            tables["supp_table_1_dataset_coverage"],
            ["dataset", "n", "method", "coverage_at_95", "mean_width", "interval_score"],
        ),
        "",
        "## Table 2. SRS354082 full rMATS-matched coverage benchmark",
        _markdown_table(
            tables["table_2_srs354082_full"],
            [
                "method",
                "coverage_at_95",
                "mean_width",
                "interval_score",
                "direction_accuracy",
            ],
        ),
        "",
        "## Supplementary Table 2. Cross-dataset conformal transfer without refitting",
        _markdown_table(
            tables["supp_table_2_cross_dataset_transfer"],
            [
                "fit_dataset",
                "deployment_dataset",
                "method",
                "coverage_at_95",
                "mean_width",
            ],
        ),
        "",
        "## Supplementary Table 3. Differential-model robustness "
        "(pooled sum vs replicate-aware rep)",
        _markdown_table(
            tables["supp_table_diff_model"],
            ["dataset", "n", "model", "coverage_at_95", "mean_width", "interval_score"],
        ),
        "",
        "## Supplementary Table 4. TRA2 positive/negative detection operating points "
        "(S1 Fig source data)",
        _markdown_table(
            tables["supp_table_3_tra2_detection_mcc"],
            [
                "rule_label",
                "tp",
                "fn",
                "fp",
                "tn",
                "sensitivity",
                "false_positive_rate",
                "mcc",
            ],
        ),
        "",
        "## Supplementary Table 5. Confidence-tier validation on the TRA2 RT-PCR panel "
        "(76 positive + 36 negative; S6 Fig source data)",
        _markdown_table(
            tables["supp_table_per_tier_validation"],
            [
                "tier",
                "n",
                "rtpcr_coverage_crossfit",
                "rtpcr_positive_rate",
                "mean_abs_dpsi",
                "positive_rate_transfer",
                "positive_rate_default",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def write_workbook(tables: dict[str, pd.DataFrame], path: Path) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)


def build_manifest(
    tables: dict[str, pd.DataFrame],
    outputs: list[Path],
    srs_result: dict[str, Any],
) -> dict[str, Any]:
    inputs = [
        F2_SOURCE,
        S1_SOURCE,
        HEADTOHEAD_JSON,
        SUM_VS_REP_JSON,
        PER_TIER_JSON,
        DM1_SUMMARY_JSON,
        DM1_ANCHORS,
        DM1_CANDIDATES,
        DM1_CATEGORIES,
        TRA2_SE,
        TRA2_VALIDATED,
        TRA2_FAILED,
        TRA2_BETAS,
        CIRC_SE,
        CIRC_TSV,
        CIRC_BETAS,
        SRS_SE,
        SRS_TRUTH,
        SRS_BETAS,
    ]
    return {
        "package_id": "BRAID-manuscript-tables",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "benchmarks/scripts/make_manuscript_tables.py",
        "inputs": [
            {"path": _rel(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in inputs
            if path.exists()
        ],
        "outputs": [
            {"path": _rel(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in outputs
            if path.exists()
        ],
        "sheets": {
            sheet: {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
            for sheet, df in tables.items()
        },
        "central_claims": {
            "common_set_braid_coverage": float(
                tables["table_1_common_coverage"]
                .query("method == 'BRAID conformal'")
                .iloc[0]["coverage_at_95"]
            ),
            "srs354082_braid_abs_coverage": float(
                tables["table_2_srs354082_full"]
                .query("method == 'BRAID conformal-abs'")
                .iloc[0]["coverage_at_95"]
            ),
        },
        "auxiliary_metrics": {
            "tra2_rmats_mcc": float(
                tables["supp_table_3_tra2_detection_mcc"]
                .query("rule_id == 'rmats_fdr'")
                .iloc[0]["mcc"]
            ),
            "tra2_braid_supported_mcc": float(
                tables["supp_table_3_tra2_detection_mcc"]
                .query("rule_id == 'braid_supported'")
                .iloc[0]["mcc"]
            ),
        },
        "srs354082_headtohead_result": srs_result,
        "known_limitations": [
            "MCC is a secondary metric reported only for TRA2/GSE59335 because "
            "the other RT-PCR panels are positive-event coverage panels without "
            "an RT-PCR-negative arm.",
            "The table package is a manuscript-local derived artifact; source scripts "
            "and public accessions remain the reproducibility authority.",
        ],
    }


def write_validation(tables: dict[str, pd.DataFrame], outputs: list[Path], path: Path) -> None:
    t1_braid = (
        tables["table_1_common_coverage"]
        .query("method == 'BRAID conformal'")
        .iloc[0]["coverage_at_95"]
    )
    t2_braid = (
        tables["table_2_srs354082_full"]
        .query("method == 'BRAID conformal-abs'")
        .iloc[0]["coverage_at_95"]
    )
    rmats_mcc = (
        tables["supp_table_3_tra2_detection_mcc"]
        .query("rule_id == 'rmats_fdr'")
        .iloc[0]["mcc"]
    )
    braid_mcc = (
        tables["supp_table_3_tra2_detection_mcc"]
        .query("rule_id == 'braid_supported'")
        .iloc[0]["mcc"]
    )
    dm1_high = (
        tables["supp_table_4_dm1_application_summary"]
        .query("metric == 'tier_counts.high-confidence'")
        .iloc[0]["value"]
    )
    diff_sum = (
        tables["supp_table_diff_model"]
        .query("dataset == 'Pooled' and model == 'sum'")
        .iloc[0]["coverage_at_95"]
    )
    diff_rep = (
        tables["supp_table_diff_model"]
        .query("dataset == 'Pooled' and model == 'rep'")
        .iloc[0]["coverage_at_95"]
    )
    pt = tables["supp_table_per_tier_validation"].set_index("tier")["rtpcr_positive_rate"]
    pt_hc = float(pt.get("high-confidence", float("nan")))
    pt_ns = float(pt.get("not-significant", float("nan")))
    checks = [
        ("Table 1 BRAID coverage == 0.971223", abs(float(t1_braid) - 0.971223) < 1e-5),
        (
            "Table 2 BRAID-conformal-abs coverage == 0.970588",
            abs(float(t2_braid) - 0.970588) < 1e-5,
        ),
        (
            "TRA2 secondary MCC: BRAID effect-supported > rMATS",
            float(braid_mcc) > float(rmats_mcc),
        ),
        ("DM1 high-confidence tier count == 68", int(dm1_high) == 68),
        (
            "Differential-model sum/rep pooled coverage agree within 0.05",
            abs(float(diff_sum) - float(diff_rep)) < 0.05,
        ),
        (
            "Per-tier: high-confidence RT-PCR positive rate > not-significant",
            pt_hc > pt_ns,
        ),
        (
            "All declared outputs exist and are non-empty",
            all(p.exists() and p.stat().st_size > 0 for p in outputs),
        ),
    ]
    lines = ["Manuscript table package validation"]
    for name, passed in checks:
        lines.append(f"{'PASS' if passed else 'FAIL'}\t{name}")
    for sheet, df in tables.items():
        lines.append(f"SHEET\t{sheet}\trows={df.shape[0]}\tcols={df.shape[1]}")
    failed = [name for name, passed in checks if not passed]
    if failed:
        lines.append("FAILED_CHECKS\t" + "; ".join(failed))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failed:
        raise RuntimeError("Manuscript table package validation failed: " + "; ".join(failed))


def copy_to_paper_package(paths: list[Path]) -> list[Path]:
    PAPER_SUPP_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in paths:
        dest = PAPER_SUPP_DIR / path.name
        shutil.copy2(path, dest)
        copied.append(dest)
    return copied


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_2, srs_result = build_table_2()
    tables = {
        "table_1_common_coverage": build_table_1(),
        "supp_table_1_dataset_coverage": build_supp_table_1(),
        "table_2_srs354082_full": table_2,
        "supp_table_2_cross_dataset_transfer": build_supp_table_2(),
        "supp_table_diff_model": build_supp_table_diff_model(),
        "supp_table_3_tra2_detection_mcc": build_supp_table_3(),
        "supp_table_per_tier_validation": build_supp_table_per_tier(),
        "supp_table_4_dm1_application_summary": build_dm1_summary(),
        "supp_table_5_dm1_anchor_genes": pd.read_csv(DM1_ANCHORS, sep="\t"),
        "supp_table_6_dm1_candidate_genes": pd.read_csv(DM1_CANDIDATES, sep="\t"),
        "supp_table_7_dm1_high_conf_categories": pd.read_csv(DM1_CATEGORIES, sep="\t"),
    }

    xlsx = OUT_DIR / "manuscript_tables.xlsx"
    markdown = OUT_DIR / "manuscript_tables.md"
    srs_json = OUT_DIR / "srs354082_full_headtohead.json"
    manifest = OUT_DIR / "manuscript_tables_manifest.json"
    validation = OUT_DIR / "manuscript_tables_validation.txt"

    write_workbook(tables, xlsx)
    write_markdown(tables, markdown)
    srs_json.write_text(json.dumps(_json_safe(srs_result), indent=2) + "\n", encoding="utf-8")
    core_outputs = [xlsx, markdown, srs_json]
    write_validation(tables, core_outputs, validation)
    manifest_data = build_manifest(tables, [*core_outputs, validation], srs_result)
    manifest.write_text(json.dumps(_json_safe(manifest_data), indent=2) + "\n", encoding="utf-8")

    copied = copy_to_paper_package([xlsx, markdown, manifest, validation])
    print(f"Wrote {_rel(xlsx)}")
    print(f"Wrote {_rel(markdown)}")
    print(f"Wrote {_rel(manifest)}")
    print(f"Wrote {_rel(validation)}")
    print("Mirrored to paper/supplementary_data:")
    for path in copied:
        print(f"  {_rel(path)}")


if __name__ == "__main__":
    main()
