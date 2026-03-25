"""BRAID v2 scorer: heuristic and trained model scoring for splicing events.

Provides penalty-based heuristic scoring and L1 logistic regression for
classifying splicing events into confidence tiers.

Tiers:
    score >= 0.8  -> high_confidence
    score >= 0.5  -> supported
    score <  0.5  -> uncertain
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Feature names used by the scorer (must match v2 feature extractor output)
# ---------------------------------------------------------------------------

_ALL_FEATURE_NAMES: list[str] = [
    # A: Junction quality (10)
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
    # B: Coverage / read quality (6)
    "dup_fraction_inc",
    "dup_fraction_exc",
    "unique_start_fraction_inc",
    "unique_start_fraction_exc",
    "exon_body_coverage_uniformity",
    "exon_body_mean_coverage",
    # C: Annotation / context (5)
    "splice_motif_inc",
    "splice_motif_exc",
    "overlapping_gene_flag",
    "n_overlapping_events",
    "flanking_exon_coverage_ratio",
    # D: Differential / replicate (8)
    "replicate_psi_variance",
    "replicate_psi_range",
    "dpsi_ctrl_replicates",
    "total_support_ctrl",
    "total_support_kd",
    "support_asymmetry",
    "rmats_fdr",
    "abs_dpsi",
]


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------


def _assign_tier(score: float) -> str:
    """Map a 0-1 confidence score to a tier label."""
    if score >= 0.8:
        return "high_confidence"
    if score >= 0.5:
        return "supported"
    return "uncertain"


# ---------------------------------------------------------------------------
# Heuristic scorer
# ---------------------------------------------------------------------------


def heuristic_score(features: dict[str, float]) -> float:
    """Rule-based confidence score without a trained model.

    Returns a float in [0, 1] where higher means more confident.

    Scoring logic (penalty-based):
        - Base = 1.0 - rmats_fdr  (or 0.5 if FDR unavailable)
        - Each violated quality check subtracts a fixed penalty
        - Result is clamped to [0, 1]
    """

    def _get(key: str, default: float = float("nan")) -> float:
        val = features.get(key, default)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)

    # Base score from FDR
    fdr = _get("rmats_fdr", float("nan"))
    if math.isnan(fdr):
        base = 0.5
    else:
        base = 1.0 - fdr

    penalty = 0.0

    # Penalty: high fraction of MAPQ=0 inclusion reads
    if _get("frac_mapq0_inc", 0.0) > 0.3:
        penalty += 0.2

    # Penalty: short minimum anchor for inclusion reads
    if _get("min_anchor_inc", 999.0) < 8:
        penalty += 0.15

    # Penalty: poor strand consistency
    if _get("strand_consistency", 1.0) < 0.8:
        penalty += 0.15

    # Penalty: non-canonical splice motif for inclusion junctions
    if _get("splice_motif_inc", 1.0) == 0:
        penalty += 0.1

    # Penalty: low diversity of read start positions
    if _get("unique_start_fraction_inc", 1.0) < 0.3:
        penalty += 0.15

    # Penalty: high replicate PSI variance
    if _get("replicate_psi_variance", 0.0) > 0.04:
        penalty += 0.1

    # Penalty: overlapping gene on opposite strand
    if _get("overlapping_gene_flag", 0.0) == 1:
        penalty += 0.15

    # Penalty: large support asymmetry between conditions
    if _get("support_asymmetry", 0.0) > 3.0:
        penalty += 0.1

    score = max(0.0, min(1.0, base - penalty))
    return score


# ---------------------------------------------------------------------------
# Trained model
# ---------------------------------------------------------------------------


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    """Train an L1-regularized logistic regression classifier.

    Parameters
    ----------
    X:
        Feature matrix of shape (n_samples, n_features).
        NaN values are imputed with column medians.
    y:
        Binary labels (1 = true positive, 0 = false positive).
    feature_names:
        Names corresponding to columns of X.

    Returns
    -------
    dict with keys:
        - model: fitted sklearn LogisticRegression
        - feature_names: list[str]
        - impute_medians: np.ndarray  (per-column medians used for imputation)
        - coefficients: dict[str, float]
        - intercept: float
        - train_accuracy: float
        - n_samples: int
        - n_features: int
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X_work = X.copy().astype(np.float64)

    # Impute NaN with column medians
    impute_medians = np.nanmedian(X_work, axis=0)
    for col_idx in range(X_work.shape[1]):
        mask = np.isnan(X_work[:, col_idx])
        if mask.any():
            X_work[mask, col_idx] = impute_medians[col_idx]

    # Replace any remaining NaN (all-NaN columns) with 0
    remaining_nan = np.isnan(X_work)
    if remaining_nan.any():
        X_work[remaining_nan] = 0.0
        impute_medians[np.isnan(impute_medians)] = 0.0

    model = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=1.0,
        max_iter=5000,
        random_state=42,
    )
    model.fit(X_work, y)

    y_pred = model.predict(X_work)
    acc = accuracy_score(y, y_pred)

    coefs = {
        name: float(model.coef_[0, i])
        for i, name in enumerate(feature_names)
    }

    return {
        "model": model,
        "feature_names": list(feature_names),
        "impute_medians": impute_medians,
        "coefficients": coefs,
        "intercept": float(model.intercept_[0]),
        "train_accuracy": acc,
        "n_samples": int(X_work.shape[0]),
        "n_features": int(X_work.shape[1]),
    }


# ---------------------------------------------------------------------------
# Event scoring
# ---------------------------------------------------------------------------


def _score_with_model(
    features: dict[str, float],
    model_info: dict[str, Any],
) -> float:
    """Score a single event using a trained model.

    Returns predicted probability of being a true positive (0-1).
    """
    feature_names = model_info["feature_names"]
    impute_medians = model_info["impute_medians"]
    model = model_info["model"]

    x = np.array(
        [features.get(name, float("nan")) for name in feature_names],
        dtype=np.float64,
    ).reshape(1, -1)

    # Impute NaN
    for col_idx in range(x.shape[1]):
        if np.isnan(x[0, col_idx]):
            x[0, col_idx] = impute_medians[col_idx]

    prob = model.predict_proba(x)[0, 1]
    return float(prob)


def score_events(
    features_list: list[dict[str, float]],
    model: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Score a list of events with a trained model or heuristic fallback.

    Parameters
    ----------
    features_list:
        List of feature dicts (one per event), as returned by
        ``extract_all_features``.
    model:
        Optional trained model dict from ``train_model``.
        If ``None``, uses ``heuristic_score``.

    Returns
    -------
    List of dicts, each containing:
        - score: float (0-1)
        - tier: str (high_confidence / supported / uncertain)
        - scoring_method: str (heuristic / model)
        - flags: list[str] (quality warnings)
        - All original features are preserved.
    """
    results: list[dict[str, Any]] = []

    for features in features_list:
        if model is not None:
            score = _score_with_model(features, model)
            method = "model"
        else:
            score = heuristic_score(features)
            method = "heuristic"

        tier = _assign_tier(score)

        # Generate quality warning flags
        flags: list[str] = []
        _add_flags(features, flags)

        result: dict[str, Any] = dict(features)
        result["braid_v2_score"] = round(score, 4)
        result["braid_v2_tier"] = tier
        result["braid_v2_scoring_method"] = method
        result["braid_v2_flags"] = flags

        results.append(result)

    return results


def _add_flags(features: dict[str, float], flags: list[str]) -> None:
    """Populate quality warning flags based on feature values."""

    def _get(key: str, default: float = float("nan")) -> float:
        val = features.get(key, default)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return default
        return float(val)

    if _get("frac_mapq0_inc", 0.0) > 0.3:
        flags.append("high_mapq0_rate")
    if _get("min_anchor_inc", 999.0) < 8:
        flags.append("short_anchor")
    if _get("strand_consistency", 1.0) < 0.8:
        flags.append("strand_inconsistent")
    if _get("splice_motif_inc", 1.0) == 0:
        flags.append("non_canonical_motif")
    if _get("unique_start_fraction_inc", 1.0) < 0.3:
        flags.append("low_start_diversity")
    if _get("replicate_psi_variance", 0.0) > 0.04:
        flags.append("high_replicate_variance")
    if _get("overlapping_gene_flag", 0.0) == 1:
        flags.append("overlapping_gene")
    if _get("support_asymmetry", 0.0) > 3.0:
        flags.append("extreme_support_asymmetry")
