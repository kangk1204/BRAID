"""Random forest-based transcript scorer.

Provides the :class:`TranscriptScorer` class that assigns a quality score
(0--1 probability of being a true positive) to each candidate transcript
based on its feature vector. The scorer operates in two modes:

1. **ML mode** -- A pre-trained ``sklearn.ensemble.RandomForestClassifier``
   loaded from disk predicts the probability of class 1 (true transcript).
2. **Heuristic mode** -- When no trained model is available, a hand-crafted
   weighted combination of key features provides a reasonable fallback score.

The ML approach follows the Aletsch / Beaver transcript scoring methodology
(Shao & Kingsford, 2019; Gatter & Stadler, 2023), where a random forest is
trained on features extracted from transcripts whose exon-intron chains match
a reference annotation (positive) versus those that do not (negative).

Model serialization uses ``joblib`` for efficient storage of the scikit-learn
estimator together with training metadata (feature names, training timestamp,
hyperparameters).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from rapidsplice.scoring.features import feature_names

logger = logging.getLogger(__name__)


class TranscriptScorer:
    """Score candidate transcripts using a random forest or heuristic fallback.

    When a *model_path* is provided at construction the scorer loads a
    pre-trained model and uses it for all subsequent calls to :meth:`score`
    and :meth:`score_batch`.  Otherwise a lightweight heuristic formula is
    used (see :meth:`_heuristic_score`).

    Args:
        model_path: Optional filesystem path to a ``joblib``-serialized
            model file. If ``None`` or the path does not exist, the scorer
            falls back to heuristic mode.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model: RandomForestClassifier | None = None
        self._feature_names: list[str] = feature_names()
        self._trained: bool = False
        self._training_metadata: dict[str, object] = {}

        if model_path is not None and Path(model_path).exists():
            self.load(model_path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Whether a trained ML model is loaded and ready for scoring."""
        return self._trained and self._model is not None

    @property
    def n_features(self) -> int:
        """Expected number of input features."""
        return len(self._feature_names)

    @property
    def mode(self) -> str:
        """Human-readable scoring mode for diagnostics and logs."""
        if self.is_trained:
            return "trained_model"
        return "heuristic_fallback"

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, features: np.ndarray) -> float:
        """Score a single transcript given its feature vector.

        Args:
            features: 1-D array of shape ``(n_features,)``.

        Returns:
            A probability in [0, 1] indicating the likelihood that the
            transcript is a true positive.

        Raises:
            ValueError: If *features* has the wrong shape.
        """
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 1:
            raise ValueError(
                f"Expected 1-D feature array, got shape {features.shape}"
            )
        if len(features) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {len(features)}"
            )

        if self.is_trained:
            assert self._model is not None
            proba = self._model.predict_proba(features.reshape(1, -1))
            return float(proba[0, 1])
        return self._heuristic_score(features)

    def score_batch(self, features_matrix: np.ndarray) -> np.ndarray:
        """Score multiple transcripts at once.

        Args:
            features_matrix: 2-D array of shape ``(n_transcripts, n_features)``.

        Returns:
            1-D array of shape ``(n_transcripts,)`` with probabilities in
            [0, 1].

        Raises:
            ValueError: If *features_matrix* has the wrong shape.
        """
        features_matrix = np.asarray(features_matrix, dtype=np.float64)
        if features_matrix.ndim != 2:
            raise ValueError(
                f"Expected 2-D feature matrix, got shape {features_matrix.shape}"
            )
        if features_matrix.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features per sample, "
                f"got {features_matrix.shape[1]}"
            )

        if self.is_trained:
            assert self._model is not None
            proba = self._model.predict_proba(features_matrix)
            return proba[:, 1].astype(np.float64)

        # Vectorized heuristic fallback
        scores = np.array(
            [self._heuristic_score(row) for row in features_matrix],
            dtype=np.float64,
        )
        return scores

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 200,
        max_depth: int = 15,
    ) -> None:
        """Train the random forest classifier on labeled transcript data.

        The training data consists of feature vectors (rows of *X*) with
        binary labels (0 = false positive, 1 = true positive). The
        classifier is configured with class-weight balancing and
        out-of-bag scoring for monitoring generalization.

        Args:
            X: Training feature matrix of shape ``(n_samples, n_features)``.
            y: Binary label vector of shape ``(n_samples,)`` with values
                in {0, 1}.
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each decision tree.

        Raises:
            ValueError: If *X* or *y* have incompatible shapes or invalid
                values.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples: "
                f"X has {X.shape[0]}, y has {y.shape[0]}"
            )
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X must have {self.n_features} features, got {X.shape[1]}"
            )

        unique_labels = set(np.unique(y).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                f"y must contain only 0 and 1, found {unique_labels}"
            )

        logger.info(
            "Training TranscriptScorer: %d samples, %d features, "
            "%d estimators, max_depth=%d",
            X.shape[0], X.shape[1], n_estimators, max_depth,
        )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            oob_score=True,
            n_jobs=-1,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
        )
        clf.fit(X, y)

        self._model = clf
        self._trained = True
        self._training_metadata = {
            "n_samples": int(X.shape[0]),
            "n_positive": int(np.sum(y == 1)),
            "n_negative": int(np.sum(y == 0)),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "oob_score": float(clf.oob_score_),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "feature_names": list(self._feature_names),
        }

        logger.info(
            "Training complete. OOB accuracy: %.4f", clf.oob_score_,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the trained model and metadata to disk using joblib.

        Args:
            path: Filesystem path for the output file. Parent directories
                are created if they do not exist.

        Raises:
            RuntimeError: If no model has been trained or loaded.
        """
        if not self.is_trained:
            raise RuntimeError("No trained model to save.")

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self._model,
            "metadata": self._training_metadata,
            "feature_names": self._feature_names,
            "version": "1.0",
        }
        joblib.dump(payload, str(output_path), compress=3)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load a trained model from disk.

        Args:
            path: Filesystem path to a ``joblib``-serialized model file.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the file does not contain a valid model payload.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(str(filepath))

        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid model file format: expected dict, got {type(payload).__name__}"
            )
        if "model" not in payload:
            raise ValueError("Model file is missing the 'model' key.")

        model = payload["model"]
        if not isinstance(model, RandomForestClassifier):
            raise ValueError(
                f"Expected RandomForestClassifier, got {type(model).__name__}"
            )

        self._model = model
        self._trained = True
        self._training_metadata = payload.get("metadata", {})

        # Validate feature name compatibility
        stored_names = payload.get("feature_names", [])
        if stored_names and stored_names != self._feature_names:
            logger.warning(
                "Loaded model was trained with different feature names. "
                "Expected %d features, model has %d. Scoring may be incorrect.",
                len(self._feature_names),
                len(stored_names),
            )

        logger.info(
            "Model loaded from %s (metadata: %s)",
            path,
            {k: v for k, v in self._training_metadata.items()
             if k != "feature_names"},
        )

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importances(self) -> dict[str, float]:
        """Return a mapping of feature name to Gini importance.

        Importances are extracted from the trained random forest and
        normalized to sum to 1.0.

        Returns:
            Dictionary mapping each feature name to its importance score.

        Raises:
            RuntimeError: If no model has been trained or loaded.
        """
        if not self.is_trained:
            raise RuntimeError(
                "No trained model available. Train or load a model first."
            )
        assert self._model is not None
        importances = self._model.feature_importances_
        return dict(zip(self._feature_names, importances.tolist()))

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _heuristic_score(self, features: np.ndarray) -> float:
        """Compute a heuristic transcript quality score without a trained model.

        The score is a weighted sum of five normalized sub-scores derived from
        key features. This provides a reasonable baseline when no training data
        is available. The weights reflect the relative importance of each
        feature group as observed in published transcript assembly benchmarks.

        Sub-scores and weights:
            * **Coverage uniformity** (weight 0.30): Based on the
              ``coverage_uniformity`` feature. Higher uniformity indicates
              more consistent read support and is strongly predictive of
              true transcripts.
            * **Junction support ratio** (weight 0.25): Based on the
              ``junction_coverage_ratio`` feature. Transcripts whose junctions
              have support proportional to exon coverage are more reliable.
            * **Structural plausibility** (weight 0.20): Composite of
              ``exon_fraction`` and normalized exon count. Penalizes
              transcripts with abnormal exon density or extreme exon counts.
            * **Relative abundance** (weight 0.15): Based on the
              ``relative_coverage`` and ``log_coverage`` features. Higher
              relative abundance within the locus increases confidence.
            * **Safe path basis** (weight 0.10): Based on ``is_safe_path``
              and ``safe_path_fraction``. Transcripts derived from safe
              (unambiguous) paths are more reliable.

        Args:
            features: 1-D feature array of shape ``(n_features,)``.

        Returns:
            A score in [0, 1].
        """
        # Build a name -> index map for quick lookup
        names = self._feature_names
        idx = {name: i for i, name in enumerate(names)}

        # -- 1. Coverage uniformity (weight 0.30) --
        coverage_uniformity = float(features[idx["coverage_uniformity"]])
        # Clamp to [0, 1]
        uniformity_score = max(0.0, min(1.0, coverage_uniformity))

        # -- 2. Junction support ratio (weight 0.25) --
        n_junctions = float(features[idx["n_junctions"]])
        if n_junctions > 0:
            junction_ratio = float(features[idx["junction_coverage_ratio"]])
            # Ideal ratio is around 0.8-1.2; penalize deviation
            # Use a Gaussian-like transform centred at 1.0
            junction_score = np.exp(-0.5 * ((junction_ratio - 1.0) / 0.5) ** 2)
            junction_score = float(junction_score)

            # Penalize weak junctions
            has_weak = float(features[idx["has_weak_junction"]])
            if has_weak > 0.5:
                junction_score *= 0.7
        else:
            # Single-exon transcript: moderate junction score
            junction_score = 0.5

        # -- 3. Structural plausibility (weight 0.20) --
        exon_fraction = float(features[idx["exon_fraction"]])
        n_exons = float(features[idx["n_exons"]])

        # Exon fraction: most real transcripts have 0.01 to 1.0
        structure_score = min(1.0, exon_fraction)

        # Penalize extremely large exon counts (> 50 exons is unusual)
        if n_exons > 50:
            structure_score *= 50.0 / n_exons
        # Penalize very short transcripts
        total_length = float(features[idx["total_length"]])
        if total_length < 200:
            structure_score *= total_length / 200.0

        structure_score = max(0.0, min(1.0, structure_score))

        # -- 4. Relative abundance (weight 0.15) --
        relative_cov = float(features[idx["relative_coverage"]])
        log_cov = float(features[idx["log_coverage"]])

        # Combine relative coverage with absolute log-coverage
        # Transcripts with very low absolute coverage are penalized
        abundance_score = min(1.0, relative_cov * 2.0)  # scale up
        # Log-coverage sigmoid: ~0 at log_cov=0, ~1 at log_cov=5
        log_factor = 1.0 / (1.0 + np.exp(-1.0 * (log_cov - 2.5)))
        abundance_score = 0.5 * abundance_score + 0.5 * float(log_factor)
        abundance_score = max(0.0, min(1.0, abundance_score))

        # -- 5. Safe path basis (weight 0.10) --
        is_safe = float(features[idx["is_safe_path"]])
        safe_frac = float(features[idx["safe_path_fraction"]])
        safe_score = 0.6 * is_safe + 0.4 * safe_frac
        safe_score = max(0.0, min(1.0, safe_score))

        # -- Weighted combination --
        final = (
            0.30 * uniformity_score
            + 0.25 * junction_score
            + 0.20 * structure_score
            + 0.15 * abundance_score
            + 0.10 * safe_score
        )
        return max(0.0, min(1.0, final))
