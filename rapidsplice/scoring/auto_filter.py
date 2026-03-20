"""Bayesian optimization of transcript filter thresholds using Optuna.

Automatically tunes FilterConfig parameters (min_score, min_coverage,
min_junction_support, etc.) to maximize assembly accuracy on a validation
set. Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler for
efficient hyperparameter search.

Falls back to a grid search when Optuna is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rapidsplice.scoring.filter import FilterConfig

if TYPE_CHECKING:
    from rapidsplice.flow.decompose import Transcript
    from rapidsplice.scoring.features import TranscriptFeatures

logger = logging.getLogger(__name__)

_OPTUNA_AVAILABLE = False
try:
    import optuna

    _OPTUNA_AVAILABLE = True
except ImportError:
    pass


@dataclass
class OptimizationResult:
    """Result of filter threshold optimization.

    Attributes:
        best_config: Optimal FilterConfig found.
        best_score: Best objective value achieved.
        n_trials: Number of trials evaluated.
        history: List of (trial_number, score) tuples.
    """

    best_config: FilterConfig
    best_score: float
    n_trials: int
    history: list[tuple[int, float]]


def _config_from_params(params: dict[str, float | int | bool]) -> FilterConfig:
    """Create a FilterConfig from a parameter dictionary.

    Args:
        params: Parameter dictionary with filter threshold values.

    Returns:
        FilterConfig instance.
    """
    return FilterConfig(
        min_score=float(params.get("min_score", 0.3)),
        min_coverage=float(params.get("min_coverage", 1.0)),
        min_junction_support=int(params.get("min_junction_support", 2)),
        min_exon_length=int(params.get("min_exon_length", 50)),
        max_transcripts_per_locus=int(
            params.get("max_transcripts_per_locus", 30)
        ),
        remove_redundant=bool(params.get("remove_redundant", True)),
        merge_similar=bool(params.get("merge_similar", True)),
    )


def evaluate_filter_config(
    config: FilterConfig,
    transcripts: list[Transcript],
    scores: np.ndarray,
    features_list: list[TranscriptFeatures],
    reference_exon_chains: set[tuple[tuple[int, int], ...]] | None = None,
) -> float:
    """Evaluate a FilterConfig by computing F1 score against reference.

    When reference exon chains are provided, computes precision and recall
    of the filtered transcript set against the reference. Without reference,
    uses a proxy score based on coverage consistency and transcript count.

    Args:
        config: Filter configuration to evaluate.
        transcripts: All candidate transcripts.
        scores: Per-transcript quality scores.
        features_list: Per-transcript feature vectors.
        reference_exon_chains: Optional set of reference intron chains
            for supervised evaluation.

    Returns:
        F1 score in [0, 1] (supervised) or proxy score (unsupervised).
    """
    from rapidsplice.scoring.filter import TranscriptFilter

    filt = TranscriptFilter(config)
    surviving = filt.filter_transcripts(transcripts, scores, features_list)

    if len(surviving) == 0:
        return 0.0

    if reference_exon_chains is not None:
        # Supervised: compute F1 against reference
        predicted_chains: set[tuple[tuple[int, int], ...]] = set()
        for idx in surviving:
            chain = tuple(transcripts[idx].exon_coords)
            predicted_chains.add(chain)

        tp = len(predicted_chains & reference_exon_chains)
        fp = len(predicted_chains - reference_exon_chains)
        fn = len(reference_exon_chains - predicted_chains)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    # Unsupervised proxy: balance coverage quality and transcript count
    surviving_scores = scores[surviving]
    mean_score = float(np.mean(surviving_scores))

    # Penalize too few or too many transcripts
    n = len(surviving)
    count_penalty = min(n / 5.0, 1.0) * min(50.0 / max(n, 1), 1.0)

    # Coverage variance penalty
    coverages = np.array(
        [features_list[i].mean_coverage for i in surviving],
        dtype=np.float64,
    )
    if len(coverages) > 1 and np.mean(coverages) > 0:
        cv = float(np.std(coverages) / np.mean(coverages))
        cv_score = max(0.0, 1.0 - cv / 3.0)
    else:
        cv_score = 0.5

    return 0.4 * mean_score + 0.3 * count_penalty + 0.3 * cv_score


def optimize_filter_optuna(
    transcripts: list[Transcript],
    scores: np.ndarray,
    features_list: list[TranscriptFeatures],
    reference_exon_chains: set[tuple[tuple[int, int], ...]] | None = None,
    n_trials: int = 50,
    timeout: float | None = 60.0,
) -> OptimizationResult:
    """Optimize filter thresholds using Optuna TPE sampler.

    Args:
        transcripts: Candidate transcripts.
        scores: Quality scores array.
        features_list: Feature vectors.
        reference_exon_chains: Optional reference for supervised eval.
        n_trials: Maximum number of optimization trials.
        timeout: Maximum time in seconds.

    Returns:
        OptimizationResult with best configuration.
    """
    if not _OPTUNA_AVAILABLE:
        logger.warning(
            "Optuna not available; falling back to grid search."
        )
        return optimize_filter_grid(
            transcripts, scores, features_list, reference_exon_chains,
        )

    history: list[tuple[int, float]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "min_score": trial.suggest_float("min_score", 0.05, 0.8),
            "min_coverage": trial.suggest_float(
                "min_coverage", 0.1, 10.0, log=True
            ),
            "min_junction_support": trial.suggest_int(
                "min_junction_support", 1, 10
            ),
            "min_exon_length": trial.suggest_int(
                "min_exon_length", 20, 200, step=10
            ),
            "max_transcripts_per_locus": trial.suggest_int(
                "max_transcripts_per_locus", 5, 50, step=5
            ),
        }
        cfg = _config_from_params(params)
        score = evaluate_filter_config(
            cfg, transcripts, scores, features_list, reference_exon_chains,
        )
        history.append((trial.number, score))
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_params
    best_config = _config_from_params(best_params)
    best_score = study.best_value

    logger.info(
        "Optuna optimization complete: %d trials, best F1=%.4f",
        len(study.trials),
        best_score,
    )

    return OptimizationResult(
        best_config=best_config,
        best_score=best_score,
        n_trials=len(study.trials),
        history=history,
    )


def optimize_filter_grid(
    transcripts: list[Transcript],
    scores: np.ndarray,
    features_list: list[TranscriptFeatures],
    reference_exon_chains: set[tuple[tuple[int, int], ...]] | None = None,
) -> OptimizationResult:
    """Optimize filter thresholds using a coarse grid search.

    Fallback when Optuna is unavailable.

    Args:
        transcripts: Candidate transcripts.
        scores: Quality scores array.
        features_list: Feature vectors.
        reference_exon_chains: Optional reference for supervised eval.

    Returns:
        OptimizationResult with best configuration found.
    """
    best_config = FilterConfig()
    best_score = -1.0
    history: list[tuple[int, float]] = []
    trial_num = 0

    score_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    coverage_values = [0.5, 1.0, 2.0, 5.0]
    junction_values = [1, 2, 3, 5]

    for ms in score_values:
        for mc in coverage_values:
            for mj in junction_values:
                cfg = FilterConfig(
                    min_score=ms,
                    min_coverage=mc,
                    min_junction_support=mj,
                )
                score = evaluate_filter_config(
                    cfg, transcripts, scores, features_list,
                    reference_exon_chains,
                )
                history.append((trial_num, score))
                if score > best_score:
                    best_score = score
                    best_config = cfg
                trial_num += 1

    logger.info(
        "Grid search complete: %d trials, best score=%.4f",
        trial_num,
        best_score,
    )

    return OptimizationResult(
        best_config=best_config,
        best_score=best_score,
        n_trials=trial_num,
        history=history,
    )


class AutoFilterOptimizer:
    """Wrapper for automatic filter threshold optimization.

    Provides a simple interface to optimize FilterConfig parameters using
    either Optuna (preferred) or grid search (fallback).

    Args:
        n_trials: Maximum number of optimization trials.
        timeout: Maximum time in seconds for optimization.
    """

    def __init__(
        self,
        n_trials: int = 50,
        timeout: float | None = 60.0,
    ) -> None:
        self._n_trials = n_trials
        self._timeout = timeout
        self._result: OptimizationResult | None = None

    @property
    def is_optimized(self) -> bool:
        """Whether optimization has been run."""
        return self._result is not None

    @property
    def best_config(self) -> FilterConfig:
        """Best FilterConfig found, or defaults if not optimized."""
        if self._result is not None:
            return self._result.best_config
        return FilterConfig()

    @property
    def result(self) -> OptimizationResult | None:
        """Full optimization result."""
        return self._result

    def optimize(
        self,
        transcripts: list[Transcript],
        scores: np.ndarray,
        features_list: list[TranscriptFeatures],
        reference_exon_chains: set[tuple[tuple[int, int], ...]] | None = None,
    ) -> OptimizationResult:
        """Run filter threshold optimization.

        Args:
            transcripts: Candidate transcripts.
            scores: Quality scores.
            features_list: Feature vectors.
            reference_exon_chains: Optional reference for supervised eval.

        Returns:
            OptimizationResult with best configuration.
        """
        if _OPTUNA_AVAILABLE:
            self._result = optimize_filter_optuna(
                transcripts,
                scores,
                features_list,
                reference_exon_chains,
                n_trials=self._n_trials,
                timeout=self._timeout,
            )
        else:
            self._result = optimize_filter_grid(
                transcripts,
                scores,
                features_list,
                reference_exon_chains,
            )
        return self._result
