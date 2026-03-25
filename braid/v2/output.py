"""BRAID v2 output: write scored splicing events to TSV.

Writes all features, score, tier, and flags in a single TSV file
suitable for downstream analysis and filtering.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


# Ordered output columns: identity, then features, then scoring
_IDENTITY_COLUMNS: list[str] = [
    "event_id",
    "gene",
    "chrom",
    "strand",
    "exon_start",
    "exon_end",
]

_FEATURE_COLUMNS: list[str] = [
    # A: Junction quality
    "median_mapq_inc",
    "median_mapq_exc",
    "frac_mapq0_inc",
    "frac_mapq0_exc",
    "min_anchor_inc",
    "min_anchor_exc",
    "median_anchor_inc",
    "frac_short_anchor_inc",
    "mismatch_rate_near_junction",
    "strand_consistency",
    # B: Coverage / read quality
    "dup_fraction_inc",
    "dup_fraction_exc",
    "unique_start_fraction_inc",
    "unique_start_fraction_exc",
    "exon_body_coverage_uniformity",
    "exon_body_mean_coverage",
    # C: Annotation / context
    "splice_motif_inc",
    "splice_motif_exc",
    "overlapping_gene_flag",
    "n_overlapping_events",
    "flanking_exon_coverage_ratio",
    # D: Differential / replicate
    "replicate_psi_variance",
    "replicate_psi_range",
    "dpsi_ctrl_replicates",
    "total_support_ctrl",
    "total_support_kd",
    "support_asymmetry",
    "rmats_fdr",
    "abs_dpsi",
]

_SCORING_COLUMNS: list[str] = [
    "braid_v2_score",
    "braid_v2_tier",
    "braid_v2_scoring_method",
    "braid_v2_flags",
]


def _format_value(val: Any) -> str:
    """Format a value for TSV output."""
    if val is None:
        return "NA"
    if isinstance(val, float):
        if math.isnan(val):
            return "NA"
        if math.isinf(val):
            return "Inf" if val > 0 else "-Inf"
        return f"{val:.6g}"
    if isinstance(val, list):
        return ";".join(str(v) for v in val) if val else ""
    return str(val)


def write_scored_events(
    scored_events: list[dict[str, Any]],
    output_path: str | Path,
    extra_columns: list[str] | None = None,
) -> Path:
    """Write scored events to a TSV file.

    Parameters
    ----------
    scored_events:
        List of scored event dicts (output of ``scorer.score_events``).
    output_path:
        Destination TSV file path.
    extra_columns:
        Additional column names to include (appended after scoring columns).

    Returns
    -------
    Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build column order
    columns = list(_IDENTITY_COLUMNS) + list(_FEATURE_COLUMNS) + list(_SCORING_COLUMNS)
    if extra_columns:
        for col in extra_columns:
            if col not in columns:
                columns.append(col)

    # Also include any keys present in events but not in our column list
    all_keys: set[str] = set()
    for event in scored_events:
        all_keys.update(event.keys())
    extra_discovered = sorted(all_keys - set(columns))
    columns.extend(extra_discovered)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)

        for event in scored_events:
            row = [_format_value(event.get(col)) for col in columns]
            writer.writerow(row)

    return output_path
