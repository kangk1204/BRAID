#!/usr/bin/env python3
"""Benchmark GM12878 single-vs-multi replicate calibration proxies."""

from __future__ import annotations

import json
import os
import sys
import time
from statistics import median

sys.path.insert(0, ".")

from rapidsplice.target.multi_replicate_bootstrap import (
    combine_isoform_bootstrap_results,
)
from rapidsplice.target.stringtie_bootstrap import (
    STBootstrapConfig,
    run_stringtie_bootstrap,
)


ST_GTF = "real_benchmark/results/stringtie_GM12878_rf.gtf"
ANNOTATED_GTF = "real_benchmark/results/st_GM12878_rf.annotated.gtf"
REF = "real_benchmark/reference/grch38/genome.fa"
REP1 = "real_benchmark/bam/GM12878_Rep1.nochr.bam"
REP2 = "real_benchmark/bam/GM12878_ENCFF550SET.nochr.bam"
OUTPUT = "benchmarks/results/gm12878_replicate_calibration.json"


def _parse_attr(attrs: str, key: str) -> str | None:
    """Parse one GTF attribute."""
    for part in attrs.split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(None, 1)
        if len(tokens) == 2 and tokens[0] == key:
            return tokens[1].strip('"')
    return None


def _load_annotation_support(path: str) -> dict[str, dict]:
    """Load transcript-level annotation support from annotated GTF."""
    support: dict[str, dict] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "transcript":
                continue
            attrs = fields[8]
            tid = _parse_attr(attrs, "transcript_id")
            if not tid:
                continue
            class_code = _parse_attr(attrs, "class_code")
            cmp_ref = _parse_attr(attrs, "cmp_ref")
            support[tid] = {
                "class_code": class_code,
                "cmp_ref": cmp_ref,
                "exact_match": class_code == "=",
                "annotated_supported": bool(cmp_ref),
            }
    return support


def _flatten_single(results: list[object]) -> dict[str, dict]:
    """Flatten StringTie bootstrap output to transcript-level rows."""
    flat: dict[str, dict] = {}
    for gene_result in results:
        for iso in gene_result.isoforms:
            ci_width = iso.ci_high - iso.ci_low
            is_confident = (
                ci_width < 0.2
                and iso.cv == iso.cv
                and iso.cv <= 0.5
            )
            flat[iso.transcript_id] = {
                "gene_id": iso.gene_id,
                "mean_weight": iso.nnls_weight,
                "ci_low": iso.ci_low,
                "ci_high": iso.ci_high,
                "ci_width": ci_width,
                "cv": iso.cv,
                "presence_rate": iso.presence_rate,
                "is_confident": is_confident,
                "is_stable": iso.is_stable,
            }
    return flat


def _flatten_multi(results: list[dict]) -> dict[str, dict]:
    """Index multi-replicate isoform rows by transcript ID."""
    flat: dict[str, dict] = {}
    for row in results:
        ci_width = row["ci_high"] - row["ci_low"]
        is_confident = (
            ci_width < 0.2
            and row["total_cv"] == row["total_cv"]
            and row["total_cv"] <= 0.5
        )
        flat[row["transcript_id"]] = dict(row, ci_width=ci_width, is_confident=is_confident)
    return flat


def _add_gene_fraction_metrics(rows: dict[str, dict]) -> None:
    """Add gene-normalized fraction metrics in-place."""
    gene_totals: dict[str, float] = {}
    for row in rows.values():
        gene_id = row["gene_id"]
        gene_totals[gene_id] = gene_totals.get(gene_id, 0.0) + row["mean_weight"]

    for row in rows.values():
        gene_total = gene_totals.get(row["gene_id"], 0.0)
        if gene_total > 0:
            row["mean_fraction"] = row["mean_weight"] / gene_total
            row["ci_low_fraction"] = row["ci_low"] / gene_total
            row["ci_high_fraction"] = row["ci_high"] / gene_total
            row["ci_width_fraction"] = row["ci_high_fraction"] - row["ci_low_fraction"]
            row["fraction_confident"] = row["ci_width_fraction"] < 0.2
        else:
            row["mean_fraction"] = 0.0
            row["ci_low_fraction"] = 0.0
            row["ci_high_fraction"] = 0.0
            row["ci_width_fraction"] = 0.0
            row["fraction_confident"] = False


def _cross_coverage(
    rep_a: dict[str, dict],
    rep_b: dict[str, dict],
    *,
    value_key: str = "mean_weight",
    ci_low_key: str = "ci_low",
    ci_high_key: str = "ci_high",
) -> tuple[int, int]:
    """Count how often one replicate's point estimate falls inside the other's CI."""
    hits = 0
    total = 0
    for tid in sorted(set(rep_a) & set(rep_b)):
        total += 1
        value = rep_a[tid][value_key]
        if rep_b[tid][ci_low_key] <= value <= rep_b[tid][ci_high_key]:
            hits += 1
    return hits, total


def _annotation_summary(
    rows: dict[str, dict],
    annotation_support: dict[str, dict],
    *,
    confidence_key: str = "is_confident",
    ci_width_key: str = "ci_width",
) -> dict:
    """Summarize annotation support for all vs confident isoforms."""
    shared = [tid for tid in rows if tid in annotation_support]
    confident = [tid for tid in shared if rows[tid].get(confidence_key)]

    def _rate(tids: list[str], key: str) -> float | None:
        if not tids:
            return None
        return sum(1 for tid in tids if annotation_support[tid][key]) / len(tids)

    ci_widths = [rows[tid][ci_width_key] for tid in shared]
    confident_widths = [rows[tid][ci_width_key] for tid in confident]
    return {
        "n_annotated_rows": len(shared),
        "n_confident": len(confident),
        "exact_match_rate_all": _rate(shared, "exact_match"),
        "exact_match_rate_confident": _rate(confident, "exact_match"),
        "annotated_supported_rate_all": _rate(shared, "annotated_supported"),
        "annotated_supported_rate_confident": _rate(confident, "annotated_supported"),
        "median_ci_width_all": (
            median(ci_widths) if ci_widths else None
        ),
        "median_ci_width_confident": (
            median(confident_widths)
            if confident_widths
            else None
        ),
    }


def main() -> None:
    """Run the GM12878 replicate calibration benchmark."""
    t0 = time.time()
    annotation_support = _load_annotation_support(ANNOTATED_GTF)

    rep1_results = run_stringtie_bootstrap(STBootstrapConfig(
        stringtie_gtf=ST_GTF,
        bam_path=REP1,
        reference_path=REF,
        n_replicates=100,
        seed=42,
    ))
    rep2_results = run_stringtie_bootstrap(STBootstrapConfig(
        stringtie_gtf=ST_GTF,
        bam_path=REP2,
        reference_path=REF,
        n_replicates=100,
        seed=43,
    ))
    multi_results = combine_isoform_bootstrap_results([rep1_results, rep2_results])

    rep1 = _flatten_single(rep1_results)
    rep2 = _flatten_single(rep2_results)
    multi = _flatten_multi(multi_results)
    _add_gene_fraction_metrics(rep1)
    _add_gene_fraction_metrics(rep2)
    _add_gene_fraction_metrics(multi)

    rep12_hits, rep12_total = _cross_coverage(rep1, rep2)
    rep21_hits, rep21_total = _cross_coverage(rep2, rep1)
    multi_r1_hits, multi_r1_total = _cross_coverage(rep1, multi)
    multi_r2_hits, multi_r2_total = _cross_coverage(rep2, multi)
    frac_rep12_hits, frac_rep12_total = _cross_coverage(
        rep1,
        rep2,
        value_key="mean_fraction",
        ci_low_key="ci_low_fraction",
        ci_high_key="ci_high_fraction",
    )
    frac_rep21_hits, frac_rep21_total = _cross_coverage(
        rep2,
        rep1,
        value_key="mean_fraction",
        ci_low_key="ci_low_fraction",
        ci_high_key="ci_high_fraction",
    )
    frac_multi_r1_hits, frac_multi_r1_total = _cross_coverage(
        rep1,
        multi,
        value_key="mean_fraction",
        ci_low_key="ci_low_fraction",
        ci_high_key="ci_high_fraction",
    )
    frac_multi_r2_hits, frac_multi_r2_total = _cross_coverage(
        rep2,
        multi,
        value_key="mean_fraction",
        ci_low_key="ci_low_fraction",
        ci_high_key="ci_high_fraction",
    )

    payload = {
        "metadata": {
            "stringtie_gtf": ST_GTF,
            "annotated_gtf": ANNOTATED_GTF,
            "replicate_bams": [REP1, REP2],
            "bootstrap_replicates_per_sample": 100,
            "elapsed_seconds": time.time() - t0,
        },
        "single_vs_single_cross_coverage": {
            "rep1_mean_in_rep2_ci": rep12_hits / rep12_total if rep12_total else None,
            "rep2_mean_in_rep1_ci": rep21_hits / rep21_total if rep21_total else None,
            "pairwise_mean": (
                (rep12_hits + rep21_hits) / (rep12_total + rep21_total)
                if (rep12_total + rep21_total)
                else None
            ),
            "n_pairs": rep12_total,
        },
        "multi_vs_single_cross_coverage": {
            "rep1_mean_in_multi_ci": (
                multi_r1_hits / multi_r1_total if multi_r1_total else None
            ),
            "rep2_mean_in_multi_ci": (
                multi_r2_hits / multi_r2_total if multi_r2_total else None
            ),
            "pairwise_mean": (
                (multi_r1_hits + multi_r2_hits) / (multi_r1_total + multi_r2_total)
                if (multi_r1_total + multi_r2_total)
                else None
            ),
            "n_pairs": multi_r1_total,
        },
        "single_vs_single_fraction_cross_coverage": {
            "rep1_mean_in_rep2_ci": (
                frac_rep12_hits / frac_rep12_total if frac_rep12_total else None
            ),
            "rep2_mean_in_rep1_ci": (
                frac_rep21_hits / frac_rep21_total if frac_rep21_total else None
            ),
            "pairwise_mean": (
                (frac_rep12_hits + frac_rep21_hits)
                / (frac_rep12_total + frac_rep21_total)
                if (frac_rep12_total + frac_rep21_total)
                else None
            ),
            "n_pairs": frac_rep12_total,
        },
        "multi_vs_single_fraction_cross_coverage": {
            "rep1_mean_in_multi_ci": (
                frac_multi_r1_hits / frac_multi_r1_total
                if frac_multi_r1_total
                else None
            ),
            "rep2_mean_in_multi_ci": (
                frac_multi_r2_hits / frac_multi_r2_total
                if frac_multi_r2_total
                else None
            ),
            "pairwise_mean": (
                (frac_multi_r1_hits + frac_multi_r2_hits)
                / (frac_multi_r1_total + frac_multi_r2_total)
                if (frac_multi_r1_total + frac_multi_r2_total)
                else None
            ),
            "n_pairs": frac_multi_r1_total,
        },
        "annotation_support": {
            "single_rep2": _annotation_summary(rep2, annotation_support),
            "multi_replicate": _annotation_summary(multi, annotation_support),
            "single_rep2_fraction": _annotation_summary(
                rep2,
                annotation_support,
                confidence_key="fraction_confident",
                ci_width_key="ci_width_fraction",
            ),
            "multi_replicate_fraction": _annotation_summary(
                multi,
                annotation_support,
                confidence_key="fraction_confident",
                ci_width_key="ci_width_fraction",
            ),
        },
        "counts": {
            "rep1_isoforms": len(rep1),
            "rep2_isoforms": len(rep2),
            "multi_isoforms": len(multi),
            "annotated_transcripts": len(annotation_support),
            "shared_single_isoforms": len(set(rep1) & set(rep2)),
            "shared_multi_vs_rep2": len(set(multi) & set(rep2)),
        },
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("Saved GM12878 replicate calibration to", OUTPUT)
    print(
        "Single pairwise coverage:",
        payload["single_vs_single_cross_coverage"]["pairwise_mean"],
    )
    print(
        "Multi pairwise coverage:",
        payload["multi_vs_single_cross_coverage"]["pairwise_mean"],
    )
    print(
        "Single fraction coverage:",
        payload["single_vs_single_fraction_cross_coverage"]["pairwise_mean"],
    )
    print(
        "Multi fraction coverage:",
        payload["multi_vs_single_fraction_cross_coverage"]["pairwise_mean"],
    )


if __name__ == "__main__":
    main()
