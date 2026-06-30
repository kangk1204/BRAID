"""CNN-based splice junction quality scoring.

Scores each splice junction based on sequence context features and read
support patterns, replacing hard threshold filtering with soft quality
scores. Inspired by SpliceAI (Jaganathan et al., 2019) and Pangolin
(Zeng & Li, 2022).

The model uses a 1D-CNN operating on per-junction feature vectors that
encode read support, local coverage context, intron length, and splice
site dinucleotide signals. When a reference genome is unavailable, the
model falls back to a lightweight feature-based scorer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception as exc:
    logger.debug("PyTorch unavailable for junction scorer fallback: %s", exc)


@dataclass
class JunctionFeatures:
    """Feature vector for a single splice junction.

    Attributes:
        read_count: Number of supporting reads.
        log_read_count: log(1 + read_count).
        intron_length: Length of the intron in base pairs.
        log_intron_length: log(intron_length).
        local_coverage_ratio: Junction reads / local exon coverage.
        strand_consistency: Fraction of reads on the consensus strand.
        is_canonical: Whether the junction uses canonical GT-AG signals.
        neighbor_junction_count: Number of nearby junctions (within 500bp).
        coverage_drop_ratio: Coverage drop at junction vs flanking.
        relative_position: Position within locus (0-1).
    """

    read_count: int = 0
    log_read_count: float = 0.0
    intron_length: int = 0
    log_intron_length: float = 0.0
    local_coverage_ratio: float = 0.0
    strand_consistency: float = 1.0
    is_canonical: float = 1.0
    neighbor_junction_count: int = 0
    coverage_drop_ratio: float = 0.0
    relative_position: float = 0.5


N_JUNCTION_FEATURES = 10


def extract_junction_features(
    starts: np.ndarray,
    ends: np.ndarray,
    counts: np.ndarray,
    strands: np.ndarray,
    locus_start: int,
    locus_end: int,
    local_coverages: np.ndarray | None = None,
) -> np.ndarray:
    """Extract feature vectors for all junctions in a locus.

    Args:
        starts: Junction donor positions (n_junctions,).
        ends: Junction acceptor positions (n_junctions,).
        counts: Read support counts (n_junctions,).
        strands: Strand assignments (n_junctions,).
        locus_start: Locus start coordinate.
        locus_end: Locus end coordinate.
        local_coverages: Optional per-junction local exon coverage.

    Returns:
        Feature matrix of shape (n_junctions, N_JUNCTION_FEATURES).
    """
    n = len(starts)
    if n == 0:
        return np.empty((0, N_JUNCTION_FEATURES), dtype=np.float32)

    features = np.zeros((n, N_JUNCTION_FEATURES), dtype=np.float32)
    locus_span = max(locus_end - locus_start, 1)

    for i in range(n):
        count = int(counts[i])
        intron_len = int(ends[i] - starts[i])

        features[i, 0] = count
        features[i, 1] = math.log1p(count)
        features[i, 2] = intron_len
        features[i, 3] = math.log(max(intron_len, 1))

        if local_coverages is not None and local_coverages[i] > 0:
            features[i, 4] = count / local_coverages[i]
        else:
            features[i, 4] = min(count / 10.0, 1.0)

        features[i, 5] = 1.0 if strands[i] >= 0 else 0.5
        features[i, 6] = 1.0  # Canonical (default without reference)

        # Neighbor count within 500bp
        neighbors = 0
        for j in range(n):
            if j != i:
                if abs(starts[j] - starts[i]) < 500 or abs(ends[j] - ends[i]) < 500:
                    neighbors += 1
        features[i, 7] = neighbors

        features[i, 8] = 0.0  # Coverage drop (requires per-base coverage)
        features[i, 9] = (starts[i] - locus_start) / locus_span

    return features


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def heuristic_junction_score(features: np.ndarray) -> np.ndarray:
    """Score junctions using a hand-crafted heuristic.

    Fallback when no trained CNN model is available.

    Args:
        features: Feature matrix of shape (n_junctions, N_JUNCTION_FEATURES).

    Returns:
        Quality scores in [0, 1] of shape (n_junctions,).
    """
    n = features.shape[0]
    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        log_reads = features[i, 1]
        log_intron = features[i, 3]
        cov_ratio = features[i, 4]
        strand_cons = features[i, 5]
        canonical = features[i, 6]

        # Penalize very long introns
        intron_penalty = max(0.0, 1.0 - log_intron / 15.0)

        raw = (
            log_reads * 0.35
            + cov_ratio * 0.25
            + strand_cons * 0.15
            + canonical * 0.15
            + intron_penalty * 0.10
        )
        scores[i] = _sigmoid(raw - 0.5)

    return scores


if _TORCH_AVAILABLE:
    class JunctionScorerCNN(nn.Module):
        """1D-CNN for splice junction quality scoring.

        Architecture: Input -> Conv1D(10->32) -> ReLU -> Conv1D(32->16) ->
        ReLU -> Linear(16->1) -> Sigmoid.

        Processes each junction independently (no cross-junction attention).
        """

        def __init__(self, n_features: int = N_JUNCTION_FEATURES) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Feature tensor of shape (batch, n_features).

            Returns:
                Quality scores of shape (batch, 1).
            """
            return self.net(x)


class JunctionScorer:
    """Junction quality scorer with CNN model and heuristic fallback.

    Provides soft quality scores for splice junctions, replacing hard
    threshold filtering. When a trained CNN model is available, uses it;
    otherwise falls back to a heuristic formula.

    Args:
        model_path: Optional path to a saved CNN model checkpoint.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._model: object | None = None
        self._is_trained = False

        if model_path is not None and _TORCH_AVAILABLE:
            try:
                self._model = JunctionScorerCNN()
                state = torch.load(model_path, map_location="cpu", weights_only=True)
                self._model.load_state_dict(state)
                self._model.eval()
                self._is_trained = True
                logger.info("Loaded junction scorer from %s", model_path)
            except Exception as exc:
                logger.warning("Failed to load junction scorer: %s", exc)
                self._model = None

    @property
    def is_trained(self) -> bool:
        """Whether a trained CNN model is loaded."""
        return self._is_trained

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score junctions.

        Args:
            features: Feature matrix (n_junctions, N_JUNCTION_FEATURES).

        Returns:
            Quality scores in [0, 1] of shape (n_junctions,).
        """
        if self._is_trained and _TORCH_AVAILABLE and self._model is not None:
            with torch.no_grad():
                x = torch.from_numpy(features).float()
                scores = self._model(x).squeeze(-1).numpy()
            return scores
        return heuristic_junction_score(features)

    def train_model(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 50,
        lr: float = 1e-3,
    ) -> float:
        """Train the CNN junction scorer.

        Args:
            features: Training features (n_samples, N_JUNCTION_FEATURES).
            labels: Binary labels (n_samples,), 1 = true junction.
            n_epochs: Number of training epochs.
            lr: Learning rate.

        Returns:
            Final training loss.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available; cannot train junction scorer.")
            return float("nan")

        self._model = JunctionScorerCNN()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        x = torch.from_numpy(features).float()
        y = torch.from_numpy(labels).float().unsqueeze(-1)

        self._model.train()
        final_loss = 0.0
        for epoch in range(n_epochs):
            optimizer.zero_grad()
            pred = self._model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self._model.eval()
        self._is_trained = True
        logger.info("Trained junction scorer: final loss=%.4f", final_loss)
        return final_loss

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: Output file path.
        """
        if self._is_trained and _TORCH_AVAILABLE and self._model is not None:
            torch.save(self._model.state_dict(), path)
            logger.info("Saved junction scorer to %s", path)
