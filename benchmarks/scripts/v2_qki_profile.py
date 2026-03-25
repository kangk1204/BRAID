#!/usr/bin/env python3
"""BRAID v2 QKI Feature Profiling: GO/NO-GO checkpoint.

Loads QKI benchmark results, identifies 4 known false positives and 10 known
true positives (high_confidence validated events), extracts v2 features for
each, and compares feature distributions between TP and FP.

The goal is to determine whether the v2 feature set can discriminate between
true and false positives -- the critical decision point for model training.
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from braid.target.rmats_bootstrap import RmatsEvent
from braid.v2.scorer import heuristic_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BENCHMARK_JSON = PROJECT_ROOT / "benchmarks" / "results" / "qki_rmats_benchmark_results.json"
RMATS_SE_PATH = Path("/home/keunsoo/projects/23_rna-seq_assembler/real_benchmark/rtpcr_benchmark/qki/rmats_output/SE.MATS.JCEC.txt")
BAM_PATHS = {
    "ctrl_rep1": Path("/home/keunsoo/projects/23_rna-seq_assembler/real_benchmark/rtpcr_benchmark/qki/ctrl_rep1.bam"),
    "ctrl_rep2": Path("/home/keunsoo/projects/23_rna-seq_assembler/real_benchmark/rtpcr_benchmark/qki/ctrl_rep2.bam"),
    "kd": Path("/home/keunsoo/projects/23_rna-seq_assembler/real_benchmark/rtpcr_benchmark/qki/qki_kd.bam"),
}
OUTPUT_JSON = PROJECT_ROOT / "benchmarks" / "results" / "v2_feature_profiles.json"

# Known false positives (in both validated and failed cohorts)
FP_GENES = {"CLSTN1", "ESYT2", "FN1", "SMARCA1"}


# ---------------------------------------------------------------------------
# rMATS SE row parser
# ---------------------------------------------------------------------------


def _parse_rmats_se_rows(path: Path) -> dict[str, dict]:
    """Parse SE.MATS.JCEC.txt into a dict keyed by 'chr:exonStart-exonEnd'."""
    rows: dict[str, dict] = {}
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        col_idx = {name: i for i, name in enumerate(header)}
        for line in f:
            fields = line.rstrip("\n").split("\t")
            chrom = fields[col_idx["chr"]].replace('"', '').replace("chr", "")
            exon_start = int(fields[col_idx["exonStart_0base"]].replace('"', ''))
            exon_end = int(fields[col_idx["exonEnd"]].replace('"', ''))
            key = f"{chrom}:{exon_start}-{exon_end}"

            strand = fields[col_idx["strand"]].replace('"', '')
            gene = fields[col_idx["geneSymbol"]].replace('"', '')
            upstream_es = int(fields[col_idx["upstreamES"]].replace('"', ''))
            upstream_ee = int(fields[col_idx["upstreamEE"]].replace('"', ''))
            downstream_es = int(fields[col_idx["downstreamES"]].replace('"', ''))
            downstream_ee = int(fields[col_idx["downstreamEE"]].replace('"', ''))

            ijc1 = fields[col_idx["IJC_SAMPLE_1"]].replace('"', '')
            sjc1 = fields[col_idx["SJC_SAMPLE_1"]].replace('"', '')
            ijc2 = fields[col_idx["IJC_SAMPLE_2"]].replace('"', '')
            sjc2 = fields[col_idx["SJC_SAMPLE_2"]].replace('"', '')
            fdr_str = fields[col_idx["FDR"]].replace('"', '')
            dpsi_str = fields[col_idx["IncLevelDifference"]].replace('"', '')
            inc1_str = fields[col_idx["IncLevel1"]].replace('"', '')

            # Parse counts
            def _sum_counts(s: str) -> int:
                return sum(int(x) for x in s.split(",") if x and x != "NA")

            def _count_vector(s: str) -> tuple[int, ...]:
                return tuple(int(x) for x in s.split(",") if x and x != "NA")

            rows[key] = {
                "chrom": chrom,
                "strand": strand,
                "gene": gene,
                "exon_start": exon_start,
                "exon_end": exon_end,
                "upstream_es": upstream_es,
                "upstream_ee": upstream_ee,
                "downstream_es": downstream_es,
                "downstream_ee": downstream_ee,
                "sample_1_inc_count": _sum_counts(ijc1),
                "sample_1_exc_count": _sum_counts(sjc1),
                "sample_2_inc_count": _sum_counts(ijc2),
                "sample_2_exc_count": _sum_counts(sjc2),
                "sample_1_inc_replicates": _count_vector(ijc1),
                "sample_1_exc_replicates": _count_vector(sjc1),
                "sample_2_inc_replicates": _count_vector(ijc2),
                "sample_2_exc_replicates": _count_vector(sjc2),
                "rmats_fdr": float(fdr_str) if fdr_str and fdr_str != "NA" else float("nan"),
                "rmats_dpsi": float(dpsi_str) if dpsi_str and dpsi_str != "NA" else float("nan"),
                "inc_level_1": inc1_str,
            }
    return rows


def _build_rmats_event(row: dict, event_id: str) -> RmatsEvent:
    """Construct an RmatsEvent from a parsed rMATS row."""
    inc_count = row["sample_1_inc_count"] + row["sample_2_inc_count"]
    exc_count = row["sample_1_exc_count"] + row["sample_2_exc_count"]

    # Compute PSI from sample_1 counts
    s1_total = row["sample_1_inc_count"] + row["sample_1_exc_count"]
    rmats_psi = row["sample_1_inc_count"] / s1_total if s1_total > 0 else 0.0

    return RmatsEvent(
        event_id=event_id,
        event_type="SE",
        chrom=row["chrom"],
        strand=row["strand"],
        gene=row["gene"],
        inc_count=inc_count,
        exc_count=exc_count,
        rmats_psi=rmats_psi,
        rmats_fdr=row["rmats_fdr"],
        rmats_dpsi=row["rmats_dpsi"],
        exon_start=row["exon_start"],
        exon_end=row["exon_end"],
        sample_1_inc_count=row["sample_1_inc_count"],
        sample_1_exc_count=row["sample_1_exc_count"],
        sample_2_inc_count=row["sample_2_inc_count"],
        sample_2_exc_count=row["sample_2_exc_count"],
        sample_1_inc_replicates=row["sample_1_inc_replicates"],
        sample_1_exc_replicates=row["sample_1_exc_replicates"],
        sample_2_inc_replicates=row["sample_2_inc_replicates"],
        sample_2_exc_replicates=row["sample_2_exc_replicates"],
        upstream_es=row["upstream_es"],
        upstream_ee=row["upstream_ee"],
        downstream_es=row["downstream_es"],
        downstream_ee=row["downstream_ee"],
    )


# ---------------------------------------------------------------------------
# Differential-only feature extraction (no BAM needed)
# ---------------------------------------------------------------------------


def _extract_differential_features_from_json(event_json: dict) -> dict[str, float]:
    """Extract differential features from benchmark JSON event data."""
    nan = float("nan")

    fdr = event_json.get("rmats_fdr", nan)
    dpsi = event_json.get("rmats_dpsi", nan)
    abs_dpsi = abs(dpsi) if not (isinstance(dpsi, float) and math.isnan(dpsi)) else nan

    ctrl_inc = event_json.get("ctrl_inc", 0.0)
    ctrl_exc = event_json.get("ctrl_exc", 0.0)
    kd_inc = event_json.get("kd_inc", 0.0)
    kd_exc = event_json.get("kd_exc", 0.0)
    ctrl_total = ctrl_inc + ctrl_exc
    kd_total = kd_inc + kd_exc

    if ctrl_total > 0 and kd_total > 0:
        support_asymmetry = abs(math.log2(ctrl_total / kd_total))
    else:
        support_asymmetry = nan

    # Replicate variance from recount channels
    ctrl_channels = event_json.get("recount_ctrl_replicate_channels", [])
    rep_psi_vals = []
    for ch in ctrl_channels:
        inc = ch.get("effective_inclusion", 0.0)
        exc = ch.get("effective_exclusion", 0.0)
        total = inc + exc
        if total > 0:
            rep_psi_vals.append(inc / total)

    if len(rep_psi_vals) >= 2:
        rep_var = float(np.var(rep_psi_vals, ddof=1))
        rep_range = max(rep_psi_vals) - min(rep_psi_vals)
    elif len(rep_psi_vals) == 1:
        rep_var = 0.0
        rep_range = 0.0
    else:
        rep_var = nan
        rep_range = nan

    return {
        "replicate_psi_variance": rep_var,
        "replicate_psi_range": rep_range,
        "dpsi_ctrl_replicates": rep_range,
        "total_support_ctrl": ctrl_total,
        "total_support_kd": kd_total,
        "support_asymmetry": support_asymmetry,
        "rmats_fdr": fdr if fdr is not None else nan,
        "abs_dpsi": abs_dpsi,
    }


# ---------------------------------------------------------------------------
# Main profiling
# ---------------------------------------------------------------------------


def _safe_mean(values: list[float]) -> float:
    """Mean of non-NaN values, or NaN if empty."""
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(clean)) if clean else float("nan")


def _safe_std(values: list[float]) -> float:
    """Std of non-NaN values, or NaN if empty."""
    clean = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return float(np.std(clean, ddof=1)) if len(clean) >= 2 else float("nan")


def main() -> None:
    print("=" * 72)
    print("BRAID v2 QKI Feature Profiling — GO/NO-GO Checkpoint")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load benchmark results
    # ------------------------------------------------------------------
    print(f"\nLoading benchmark results from {BENCHMARK_JSON}")
    with open(BENCHMARK_JSON) as f:
        data = json.load(f)

    # Identify events
    validated = data["validated"]
    failed = data["failed"]

    # True positives: 10 high_confidence validated events
    tp_events = [e for e in validated if e.get("high_confidence") and e.get("matched")]
    # False positives: 4 known FP genes from failed cohort
    fp_events = [e for e in failed if e["gene"] in FP_GENES and e.get("matched")]

    print(f"  True positives (high_confidence validated): {len(tp_events)}")
    print(f"  False positives ({', '.join(sorted(FP_GENES))}): {len(fp_events)}")

    # ------------------------------------------------------------------
    # Load rMATS SE data for coordinate/count info
    # ------------------------------------------------------------------
    rmats_rows: dict[str, dict] = {}
    bam_available = False
    if RMATS_SE_PATH.exists():
        print(f"\nParsing rMATS SE output from {RMATS_SE_PATH}")
        rmats_rows = _parse_rmats_se_rows(RMATS_SE_PATH)
        print(f"  Parsed {len(rmats_rows)} SE events")

    # Check BAM availability
    bam_available = all(p.exists() for p in BAM_PATHS.values())
    print(f"\n  BAM files available: {bam_available}")
    if bam_available:
        for label, p in BAM_PATHS.items():
            print(f"    {label}: {p}")

    # ------------------------------------------------------------------
    # Extract features for each event
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("Extracting features...")
    print("-" * 72)

    tp_features: list[dict] = []
    fp_features: list[dict] = []

    all_target_events = [("TP", e) for e in tp_events] + [("FP", e) for e in fp_events]

    for label, event_json in all_target_events:
        gene = event_json["gene"]
        chrom = event_json["chrom"]
        exon_start = event_json["event_start"]
        exon_end = event_json["event_end"]
        lookup_key = f"{chrom}:{exon_start}-{exon_end}"

        # Extract differential features from benchmark JSON
        diff_features = _extract_differential_features_from_json(event_json)

        # Additional features from benchmark JSON
        extra_features = {
            "braid_dpsi": event_json.get("braid_dpsi", float("nan")),
            "braid_dpsi_ci_width": event_json.get("braid_dpsi_ci_width", float("nan")),
            "braid_dpsi_prob_abs_ge_cutoff": event_json.get("braid_dpsi_prob_abs_ge_cutoff", float("nan")),
            "ctrl_cv": event_json.get("ctrl_cv", float("nan")),
            "ctrl_psi": event_json.get("ctrl_psi", float("nan")),
            "kd_psi": event_json.get("kd_psi", float("nan")),
            "total_support": float(event_json.get("total_support", 0)),
        }

        # Try to get BAM-derived features
        bam_features: dict[str, float] = {}
        rmats_row = rmats_rows.get(lookup_key)

        if bam_available and rmats_row is not None:
            try:
                rmats_event = _build_rmats_event(rmats_row, event_json.get("event_id", lookup_key))

                # Use the first ctrl BAM for junction/coverage features
                bam_path = str(BAM_PATHS["ctrl_rep1"])

                # Need to prepend "chr" to chrom for BAM (BAM uses chr-prefixed names)
                # Check if BAM uses chr prefix
                import pysam
                bam_test = pysam.AlignmentFile(bam_path, "rb")
                bam_chroms = set(bam_test.references)
                bam_test.close()

                # Adjust chromosome name
                if f"chr{chrom}" in bam_chroms and chrom not in bam_chroms:
                    rmats_event.chrom = f"chr{chrom}"

                from braid.v2.junction import extract_junction_features
                from braid.v2.coverage import extract_coverage_features

                junction_feats = extract_junction_features(bam_path, rmats_event)
                coverage_feats = extract_coverage_features(bam_path, rmats_event)

                bam_features.update(junction_feats)
                bam_features.update(coverage_feats)

                print(f"  [{label}] {gene} — BAM features extracted ({sum(1 for v in bam_features.values() if not math.isnan(v))} non-NaN)")
            except Exception as exc:
                print(f"  [{label}] {gene} — BAM feature extraction failed: {exc}")
        else:
            reason = "no rMATS row" if rmats_row is None else "no BAMs"
            print(f"  [{label}] {gene} — differential features only ({reason})")

        # Merge all features
        features: dict[str, float] = {}
        features.update(diff_features)
        features.update(bam_features)
        features.update(extra_features)
        features["gene"] = gene  # type: ignore[assignment]
        features["label"] = label  # type: ignore[assignment]

        # Compute heuristic score
        features["heuristic_score"] = heuristic_score(features)

        if label == "TP":
            tp_features.append(features)
        else:
            fp_features.append(features)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("FEATURE COMPARISON: True Positives vs False Positives")
    print("=" * 72)

    # Collect all numeric feature keys (excluding gene/label)
    all_feature_keys = set()
    for f_dict in tp_features + fp_features:
        for k, v in f_dict.items():
            if k not in ("gene", "label") and isinstance(v, (int, float)):
                all_feature_keys.add(k)
    feature_keys = sorted(all_feature_keys)

    # Print header
    print(f"\n{'Feature':<35} {'TP mean':>10} {'FP mean':>10} {'Delta':>10} {'Discrimin':>10}")
    print("-" * 77)

    discriminating_features: list[dict] = []

    for key in feature_keys:
        tp_vals = [f.get(key, float("nan")) for f in tp_features if isinstance(f.get(key), (int, float))]
        fp_vals = [f.get(key, float("nan")) for f in fp_features if isinstance(f.get(key), (int, float))]

        tp_mean = _safe_mean(tp_vals)
        fp_mean = _safe_mean(fp_vals)

        if math.isnan(tp_mean) and math.isnan(fp_mean):
            continue

        if math.isnan(tp_mean) or math.isnan(fp_mean):
            delta = float("nan")
            discriminating = "?"
        else:
            delta = tp_mean - fp_mean
            # Cohen's d-like: |delta| / pooled_std
            tp_std = _safe_std(tp_vals)
            fp_std = _safe_std(fp_vals)
            if not math.isnan(tp_std) and not math.isnan(fp_std):
                pooled_std = math.sqrt((tp_std**2 + fp_std**2) / 2) if (tp_std + fp_std) > 0 else 1e-10
                effect_size = abs(delta) / pooled_std if pooled_std > 0 else 0
                if effect_size >= 0.8:
                    discriminating = "***"
                elif effect_size >= 0.5:
                    discriminating = "**"
                elif effect_size >= 0.2:
                    discriminating = "*"
                else:
                    discriminating = ""
            else:
                discriminating = "?"

        def _fmt(v: float) -> str:
            if math.isnan(v):
                return "NaN"
            if abs(v) < 0.001 and v != 0:
                return f"{v:.2e}"
            return f"{v:.4f}"

        print(f"  {key:<33} {_fmt(tp_mean):>10} {_fmt(fp_mean):>10} {_fmt(delta):>10} {discriminating:>10}")

        if discriminating in ("**", "***"):
            discriminating_features.append({
                "feature": key,
                "tp_mean": tp_mean,
                "fp_mean": fp_mean,
                "delta": delta,
                "effect_size_label": discriminating,
            })

    # ------------------------------------------------------------------
    # Per-event detail
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("PER-EVENT HEURISTIC SCORES")
    print("=" * 72)

    print(f"\n{'Label':<5} {'Gene':<12} {'Score':>7} {'FDR':>12} {'|dPSI|':>8} {'Support':>8} {'RepVar':>8}")
    print("-" * 62)

    for features in tp_features + fp_features:
        gene = features.get("gene", "?")
        label = features.get("label", "?")
        score = features.get("heuristic_score", float("nan"))
        fdr = features.get("rmats_fdr", float("nan"))
        abs_d = features.get("abs_dpsi", float("nan"))
        support = features.get("total_support", float("nan"))
        rep_var = features.get("replicate_psi_variance", float("nan"))

        def _fmt2(v: float, width: int = 8) -> str:
            if isinstance(v, float) and math.isnan(v):
                return "NaN".rjust(width)
            if abs(v) < 0.001 and v != 0:
                return f"{v:.2e}".rjust(width)
            return f"{v:.4f}".rjust(width)

        print(f"  {label:<5} {gene:<12} {_fmt2(score, 7)} {_fmt2(fdr, 12)} {_fmt2(abs_d, 8)} {_fmt2(support, 8)} {_fmt2(rep_var, 8)}")

    # ------------------------------------------------------------------
    # GO/NO-GO verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("GO/NO-GO VERDICT")
    print("=" * 72)

    tp_scores = [f.get("heuristic_score", 0) for f in tp_features]
    fp_scores = [f.get("heuristic_score", 0) for f in fp_features]

    tp_mean_score = _safe_mean(tp_scores)
    fp_mean_score = _safe_mean(fp_scores)

    n_discriminating = len(discriminating_features)

    print(f"\n  Mean heuristic score — TP: {tp_mean_score:.4f}, FP: {fp_mean_score:.4f}")
    print(f"  Score gap (TP - FP): {tp_mean_score - fp_mean_score:.4f}")
    print(f"  Features with medium/large effect size: {n_discriminating}")

    if discriminating_features:
        print("\n  Top discriminating features:")
        for df in discriminating_features:
            print(f"    {df['effect_size_label']} {df['feature']}: TP={df['tp_mean']:.4f}, FP={df['fp_mean']:.4f}, delta={df['delta']:.4f}")

    # ------------------------------------------------------------------
    # Critical check: overlap between TP and FP events
    # ------------------------------------------------------------------
    tp_coords = {(f.get("gene"), f.get("label")): f for f in tp_features}
    fp_coords_set = {f.get("gene") for f in fp_features}
    tp_only_genes = [f for f in tp_features if f.get("gene") not in fp_coords_set]
    shared_genes = [f for f in tp_features if f.get("gene") in fp_coords_set]

    print(f"\n  OVERLAP WARNING:")
    print(f"    TP events also in FP (same rMATS event, dual RT-PCR targets): {len(shared_genes)}")
    print(f"    TP-only events (unique to validated): {len(tp_only_genes)}")
    if shared_genes:
        print(f"    Shared genes: {', '.join(f.get('gene', '?') for f in shared_genes)}")
    if tp_only_genes:
        print(f"    TP-only genes: {', '.join(f.get('gene', '?') for f in tp_only_genes)}")

    # Corrected comparison: TP-only vs shared (FP)
    if tp_only_genes:
        tp_only_scores = [f.get("heuristic_score", 0) for f in tp_only_genes]
        shared_scores = [f.get("heuristic_score", 0) for f in shared_genes]
        tp_only_mean = _safe_mean(tp_only_scores)
        shared_mean = _safe_mean(shared_scores)
        print(f"\n  CORRECTED ANALYSIS (TP-only vs shared/ambiguous):")
        print(f"    TP-only mean score: {tp_only_mean:.4f}")
        print(f"    Shared (ambiguous) mean score: {shared_mean:.4f}")
        print(f"    Gap: {tp_only_mean - shared_mean:.4f}")

        # Feature comparison for TP-only vs shared
        print(f"\n  {'Feature':<35} {'TP-only':>10} {'Shared':>10} {'Delta':>10}")
        print("  " + "-" * 67)
        corrected_discriminators = 0
        for key in feature_keys:
            tpo_vals = [f.get(key, float("nan")) for f in tp_only_genes if isinstance(f.get(key), (int, float))]
            sh_vals = [f.get(key, float("nan")) for f in shared_genes if isinstance(f.get(key), (int, float))]
            tpo_mean = _safe_mean(tpo_vals)
            sh_mean = _safe_mean(sh_vals)
            if math.isnan(tpo_mean) and math.isnan(sh_mean):
                continue
            if math.isnan(tpo_mean) or math.isnan(sh_mean):
                delta = float("nan")
            else:
                delta = tpo_mean - sh_mean
            tpo_std = _safe_std(tpo_vals)
            sh_std = _safe_std(sh_vals)
            marker = ""
            if not math.isnan(delta) and not math.isnan(tpo_std) and not math.isnan(sh_std):
                pooled = math.sqrt((tpo_std**2 + sh_std**2) / 2) if (tpo_std + sh_std) > 0 else 1e-10
                es = abs(delta) / pooled if pooled > 0 else 0
                if es >= 0.8:
                    marker = "***"
                    corrected_discriminators += 1
                elif es >= 0.5:
                    marker = "**"
                    corrected_discriminators += 1

            def _f(v: float) -> str:
                if math.isnan(v):
                    return "NaN"
                if abs(v) < 0.001 and v != 0:
                    return f"{v:.2e}"
                return f"{v:.4f}"
            if marker:
                print(f"  {key:<33} {_f(tpo_mean):>10} {_f(sh_mean):>10} {_f(delta):>10}  {marker}")

    # Verdict
    if tp_mean_score - fp_mean_score > 0.05 or n_discriminating >= 2:
        verdict = "GO"
        reason = "Features show meaningful discrimination between TP and FP"
    elif tp_mean_score - fp_mean_score > 0.02 or n_discriminating >= 1:
        verdict = "CONDITIONAL GO"
        reason = "Marginal discrimination; BAM features may improve separation"
    else:
        verdict = "NO-GO"
        reason = "Features do not sufficiently discriminate TP from FP"

    # Adjust verdict if all FPs are identical to TPs
    if len(shared_genes) == len(fp_features):
        verdict = "CONDITIONAL GO"
        reason = (
            f"All {len(fp_features)} FP events are identical rMATS loci to TP events "
            f"(dual RT-PCR targets). Feature-based discrimination is NOT possible for "
            f"these cases -- the FP label reflects biological ambiguity, not sequencing "
            f"artifact. {n_discriminating} features discriminate TP-only ({len(tp_only_genes)} events) "
            f"from the shared pool."
        )

    print(f"\n  >>> VERDICT: {verdict} <<<")
    print(f"      Reason: {reason}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "metadata": {
            "benchmark_json": str(BENCHMARK_JSON),
            "rmats_se_path": str(RMATS_SE_PATH) if RMATS_SE_PATH.exists() else None,
            "bam_available": bam_available,
            "n_tp_events": len(tp_features),
            "n_fp_events": len(fp_features),
            "fp_genes": sorted(FP_GENES),
        },
        "verdict": {
            "decision": verdict,
            "reason": reason,
            "tp_mean_score": tp_mean_score,
            "fp_mean_score": fp_mean_score,
            "score_gap": tp_mean_score - fp_mean_score,
            "n_discriminating_features": n_discriminating,
        },
        "discriminating_features": discriminating_features,
        "tp_profiles": [
            {k: (v if not (isinstance(v, float) and math.isnan(v)) else None) for k, v in f.items()}
            for f in tp_features
        ],
        "fp_profiles": [
            {k: (v if not (isinstance(v, float) and math.isnan(v)) else None) for k, v in f.items()}
            for f in fp_features
        ],
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_JSON}")
    print("=" * 72)


if __name__ == "__main__":
    main()
