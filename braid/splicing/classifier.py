"""ML-based confidence scoring for alternative splicing events.

Provides a GradientBoostingClassifier with automatic heuristic fallback
when insufficient training data is available. Optionally uses a
Transformer-based classifier for higher accuracy when PyTorch is available.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from braid.splicing.events import ASEvent
from braid.splicing.psi import PSIResult

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    pass

NUM_EVENT_TYPES = 7


@dataclass
class EventFeatures:
    """Feature vector for an alternative splicing event.

    Attributes:
        inclusion_count: Total inclusion junction reads.
        exclusion_count: Total exclusion junction reads.
        total_reads: Sum of inclusion and exclusion reads.
        psi: Percent Spliced In value.
        ci_low: Lower CI bound.
        ci_high: Upper CI bound.
        ci_width: Width of credible interval.
        log_total_reads: log(1 + total_reads).
        inclusion_ratio: inclusion_count / max(total_reads, 1).
        junction_balance: Minimum inclusion/exclusion ratio.
        n_inclusion_junctions: Number of inclusion junctions.
        n_exclusion_junctions: Number of exclusion junctions.
        n_inclusion_transcripts: Number of supporting inclusion transcripts.
        n_exclusion_transcripts: Number of supporting exclusion transcripts.
        event_type_onehot: One-hot encoding of event type (7 values).
    """

    inclusion_count: int = 0
    exclusion_count: int = 0
    total_reads: int = 0
    psi: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 1.0
    ci_width: float = 1.0
    log_total_reads: float = 0.0
    inclusion_ratio: float = 0.0
    junction_balance: float = 0.0
    n_inclusion_junctions: int = 0
    n_exclusion_junctions: int = 0
    n_inclusion_transcripts: int = 0
    n_exclusion_transcripts: int = 0
    event_type_onehot: list[float] = field(default_factory=lambda: [0.0] * NUM_EVENT_TYPES)


def extract_event_features(
    event: ASEvent,
    psi_result: PSIResult,
) -> EventFeatures:
    """Extract feature vector from an AS event and its PSI result.

    Args:
        event: The alternative splicing event.
        psi_result: The PSI quantification result.

    Returns:
        EventFeatures instance.
    """
    total = psi_result.total_reads
    psi_val = psi_result.psi if not math.isnan(psi_result.psi) else 0.5
    ci_width = psi_result.ci_high - psi_result.ci_low

    # Junction balance: how evenly reads are split
    if total > 0:
        inc_frac = psi_result.inclusion_count / total
        exc_frac = psi_result.exclusion_count / total
        junction_balance = min(inc_frac, exc_frac) / max(inc_frac, exc_frac, 1e-10)
    else:
        junction_balance = 0.0

    onehot = [0.0] * NUM_EVENT_TYPES
    onehot[int(event.event_type)] = 1.0

    return EventFeatures(
        inclusion_count=psi_result.inclusion_count,
        exclusion_count=psi_result.exclusion_count,
        total_reads=total,
        psi=psi_val,
        ci_low=psi_result.ci_low,
        ci_high=psi_result.ci_high,
        ci_width=ci_width,
        log_total_reads=math.log1p(total),
        inclusion_ratio=psi_result.inclusion_count / max(total, 1),
        junction_balance=junction_balance,
        n_inclusion_junctions=len(event.inclusion_junctions),
        n_exclusion_junctions=len(event.exclusion_junctions),
        n_inclusion_transcripts=len(event.inclusion_transcripts),
        n_exclusion_transcripts=len(event.exclusion_transcripts),
        event_type_onehot=onehot,
    )


def features_to_array(features: EventFeatures) -> np.ndarray:
    """Convert an EventFeatures to a 1D numpy array.

    Args:
        features: Event feature vector.

    Returns:
        Array of shape (21,) with all numeric features.
    """
    base = [
        features.inclusion_count,
        features.exclusion_count,
        features.total_reads,
        features.psi,
        features.ci_low,
        features.ci_high,
        features.ci_width,
        features.log_total_reads,
        features.inclusion_ratio,
        features.junction_balance,
        features.n_inclusion_junctions,
        features.n_exclusion_junctions,
        features.n_inclusion_transcripts,
        features.n_exclusion_transcripts,
    ]
    return np.array(base + features.event_type_onehot, dtype=np.float64)


FEATURE_NAMES: list[str] = [
    "inclusion_count",
    "exclusion_count",
    "total_reads",
    "psi",
    "ci_low",
    "ci_high",
    "ci_width",
    "log_total_reads",
    "inclusion_ratio",
    "junction_balance",
    "n_inclusion_junctions",
    "n_exclusion_junctions",
    "n_inclusion_transcripts",
    "n_exclusion_transcripts",
    "event_type_SE",
    "event_type_A5SS",
    "event_type_A3SS",
    "event_type_MXE",
    "event_type_RI",
    "event_type_AFE",
    "event_type_ALE",
]


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def heuristic_score(features: EventFeatures) -> float:
    """Compute a heuristic confidence score for an AS event.

    Formula:
        score = sigmoid(log(total_reads) * 0.3
                        + (1 - ci_width) * 0.4
                        + junction_balance * 0.3)

    Args:
        features: Event features.

    Returns:
        Confidence score in [0, 1].
    """
    raw = (
        features.log_total_reads * 0.3
        + (1.0 - features.ci_width) * 0.4
        + features.junction_balance * 0.3
    )
    return _sigmoid(raw)


class EventClassifier:
    """ML-based alternative splicing event classifier.

    Uses a GradientBoostingClassifier when trained, with automatic
    fallback to a heuristic scoring function.

    Args:
        n_estimators: Number of boosting stages.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
    ) -> None:
        self._model: GradientBoostingClassifier | None = None
        self._is_trained: bool = False
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate

    @property
    def is_trained(self) -> bool:
        """Whether the classifier has been trained on data."""
        return self._is_trained

    def train(
        self,
        feature_arrays: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Train the classifier on labeled event features.

        Args:
            feature_arrays: Feature matrix of shape (n_events, 21).
            labels: Binary labels of shape (n_events,), 1 = confident event.
        """
        if len(feature_arrays) < 10:
            logger.warning(
                "Insufficient training data (%d events); using heuristic fallback.",
                len(feature_arrays),
            )
            return

        self._model = GradientBoostingClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            random_state=42,
        )
        self._model.fit(feature_arrays, labels)
        self._is_trained = True
        logger.info(
            "Trained EventClassifier on %d events (%.1f%% positive)",
            len(labels),
            100 * labels.mean(),
        )

    def score(self, features: EventFeatures) -> float:
        """Score a single event.

        Uses the trained model if available, otherwise falls back to the
        heuristic scoring function.

        Args:
            features: Event features.

        Returns:
            Confidence score in [0, 1].
        """
        if self._is_trained and self._model is not None:
            arr = features_to_array(features).reshape(1, -1)
            return float(self._model.predict_proba(arr)[0, 1])
        return heuristic_score(features)

    def score_batch(
        self,
        events: list[ASEvent],
        psi_results: list[PSIResult],
    ) -> list[float]:
        """Score a batch of events.

        Args:
            events: List of AS events.
            psi_results: Corresponding PSI results.

        Returns:
            List of confidence scores.
        """
        scores: list[float] = []
        for event, psi in zip(events, psi_results):
            feat = extract_event_features(event, psi)
            scores.append(self.score(feat))
        return scores


# ---------------------------------------------------------------------------
# Feature dimension constant
# ---------------------------------------------------------------------------
N_EVENT_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Transformer-based classifier
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:

    class TransformerEventNet(nn.Module):
        """Transformer-based AS event classifier.

        Uses a small Transformer encoder to capture cross-feature interactions,
        followed by an MLP classifier head. Each feature is treated as a token,
        projected to a hidden dimension, then processed by self-attention.

        Architecture:
            Linear(1->d_model) per feature -> TransformerEncoder(2 layers, 4 heads)
            -> mean pool -> MLP(d_model->32->1) -> Sigmoid.
        """

        def __init__(
            self,
            n_features: int = N_EVENT_FEATURES,
            d_model: int = 32,
            nhead: int = 4,
            n_layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.n_features = n_features
            self.d_model = d_model

            # Project each scalar feature to d_model dimensions
            self.input_proj = nn.Linear(1, d_model)
            self.pos_embed = nn.Parameter(
                torch.randn(1, n_features, d_model) * 0.02
            )

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )

            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Feature tensor (batch, n_features).

            Returns:
                Confidence scores (batch, 1).
            """
            # x: (batch, n_features) -> (batch, n_features, 1)
            x = x.unsqueeze(-1)
            # Project to d_model
            h = self.input_proj(x)  # (batch, n_features, d_model)
            h = h + self.pos_embed
            # Transformer encoding
            h = self.encoder(h)  # (batch, n_features, d_model)
            # Mean pool across features
            h = h.mean(dim=1)  # (batch, d_model)
            return self.classifier(h)


class TransformerEventClassifier:
    """Transformer-based AS event classifier with GBM fallback.

    When PyTorch is available and the model is trained, uses a Transformer
    encoder to score events. Otherwise falls back to GBM or heuristic.

    Args:
        model_path: Optional path to saved Transformer weights.
        gbm_fallback: Whether to use GBM as intermediate fallback.
    """

    def __init__(
        self,
        model_path: str | None = None,
        gbm_fallback: bool = True,
    ) -> None:
        self._transformer: object | None = None
        self._is_trained = False
        self._gbm_classifier = EventClassifier() if gbm_fallback else None

        if model_path is not None and _TORCH_AVAILABLE:
            try:
                self._transformer = TransformerEventNet()
                state = torch.load(
                    model_path, map_location="cpu", weights_only=True
                )
                self._transformer.load_state_dict(state)
                self._transformer.eval()
                self._is_trained = True
                logger.info(
                    "Loaded Transformer event classifier from %s", model_path
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load Transformer classifier: %s", exc
                )
                self._transformer = None

    @property
    def is_trained(self) -> bool:
        """Whether a trained Transformer model is loaded."""
        return self._is_trained

    def score(self, features: EventFeatures) -> float:
        """Score a single event.

        Args:
            features: Event features.

        Returns:
            Confidence score in [0, 1].
        """
        if self._is_trained and _TORCH_AVAILABLE and self._transformer is not None:
            arr = features_to_array(features)
            with torch.no_grad():
                x = torch.from_numpy(arr).float().unsqueeze(0)
                score = self._transformer(x).item()
            return float(score)

        # Fallback to GBM or heuristic
        if self._gbm_classifier is not None:
            return self._gbm_classifier.score(features)
        return heuristic_score(features)

    def score_batch(
        self,
        events: list[ASEvent],
        psi_results: list[PSIResult],
    ) -> list[float]:
        """Score a batch of events.

        Args:
            events: List of AS events.
            psi_results: Corresponding PSI results.

        Returns:
            List of confidence scores.
        """
        if self._is_trained and _TORCH_AVAILABLE and self._transformer is not None:
            arrays = []
            for event, psi in zip(events, psi_results):
                feat = extract_event_features(event, psi)
                arrays.append(features_to_array(feat))
            batch = np.stack(arrays, axis=0)
            with torch.no_grad():
                x = torch.from_numpy(batch).float()
                scores = self._transformer(x).squeeze(-1).numpy()
            return scores.tolist()

        return [
            self.score(extract_event_features(e, p))
            for e, p in zip(events, psi_results)
        ]

    def train_model(
        self,
        feature_arrays: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """Train the Transformer classifier.

        Args:
            feature_arrays: Feature matrix (n_events, N_EVENT_FEATURES).
            labels: Binary labels (n_events,).
            n_epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Final training loss.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available; cannot train Transformer.")
            return float("nan")

        self._transformer = TransformerEventNet()
        optimizer = torch.optim.Adam(self._transformer.parameters(), lr=lr)
        criterion = nn.BCELoss()

        x = torch.from_numpy(feature_arrays).float()
        y = torch.from_numpy(labels).float().unsqueeze(-1)

        self._transformer.train()
        final_loss = 0.0
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            pred = self._transformer(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self._transformer.eval()
        self._is_trained = True
        logger.info(
            "Trained Transformer classifier: final loss=%.4f", final_loss
        )
        return final_loss

    def save(self, path: str) -> None:
        """Save trained Transformer weights.

        Args:
            path: Output file path.
        """
        if self._is_trained and _TORCH_AVAILABLE and self._transformer is not None:
            torch.save(self._transformer.state_dict(), path)
            logger.info("Saved Transformer classifier to %s", path)
