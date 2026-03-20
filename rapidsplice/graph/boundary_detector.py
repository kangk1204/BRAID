"""1D-CNN exon boundary detector.

Uses a dilated 1D-CNN to predict exon-intron boundaries from per-base
coverage signals. Replaces the simple coverage-drop heuristic with a
learned boundary predictor that considers multi-scale coverage patterns.

The model operates on fixed-length windows (default 200bp) centred on
candidate boundary positions and outputs a probability of an exon-intron
boundary at the centre.

Falls back to a coverage-gradient heuristic when PyTorch is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except ImportError:
    pass

# Default window size for boundary detection
WINDOW_SIZE = 200
# Number of input channels (coverage, gradient, log_coverage)
N_CHANNELS = 3


@dataclass
class BoundaryPrediction:
    """Predicted exon boundary.

    Attributes:
        position: Genomic position of the boundary.
        probability: Probability of being a true boundary in [0, 1].
        boundary_type: 'start' (intron->exon) or 'end' (exon->intron).
    """

    position: int
    probability: float
    boundary_type: str


def extract_boundary_features(
    coverage: np.ndarray,
    position: int,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Extract a multi-channel feature window for a candidate boundary.

    Channels:
        0: Normalized coverage signal.
        1: Coverage gradient (first derivative).
        2: Log-transformed coverage.

    Args:
        coverage: Per-base coverage array for the region.
        position: Centre position in the coverage array.
        window_size: Window size (must be even).

    Returns:
        Feature array of shape (N_CHANNELS, window_size).
    """
    half = window_size // 2
    n = len(coverage)

    # Extract window with padding
    start = position - half
    end = position + half

    window = np.zeros(window_size, dtype=np.float32)
    src_start = max(0, start)
    src_end = min(n, end)
    dst_start = max(0, -start)
    dst_end = dst_start + (src_end - src_start)

    if src_end > src_start:
        window[dst_start:dst_end] = coverage[src_start:src_end]

    # Build multi-channel features
    features = np.zeros((N_CHANNELS, window_size), dtype=np.float32)

    # Channel 0: Normalized coverage
    max_cov = max(window.max(), 1.0)
    features[0] = window / max_cov

    # Channel 1: Coverage gradient
    gradient = np.gradient(window)
    grad_max = max(abs(gradient).max(), 1.0)
    features[1] = gradient / grad_max

    # Channel 2: Log-transformed coverage
    features[2] = np.log1p(window) / max(np.log1p(max_cov), 1.0)

    return features


def heuristic_boundary_score(
    coverage: np.ndarray,
    position: int,
    window_size: int = WINDOW_SIZE,
) -> float:
    """Score a candidate boundary using coverage gradient heuristic.

    Looks for sharp coverage transitions (drops or rises) near the
    candidate position.

    Args:
        coverage: Per-base coverage array.
        position: Candidate boundary position.
        window_size: Analysis window size.

    Returns:
        Boundary probability in [0, 1].
    """
    half = window_size // 2
    n = len(coverage)

    left_start = max(0, position - half)
    left_end = position
    right_start = position
    right_end = min(n, position + half)

    if left_end <= left_start or right_end <= right_start:
        return 0.0

    left_mean = float(np.mean(coverage[left_start:left_end]))
    right_mean = float(np.mean(coverage[right_start:right_end]))

    max_cov = max(left_mean, right_mean, 1.0)
    diff = abs(left_mean - right_mean) / max_cov

    # Sharp gradient near center
    grad_window = min(20, half)
    near_left = position - grad_window
    near_right = position + grad_window
    if near_left >= 0 and near_right < n:
        near_left_mean = float(np.mean(coverage[near_left:position]))
        near_right_mean = float(np.mean(coverage[position:near_right]))
        near_diff = abs(near_left_mean - near_right_mean) / max(
            max(near_left_mean, near_right_mean), 1.0
        )
    else:
        near_diff = diff

    score = 0.5 * diff + 0.5 * near_diff
    return float(min(score, 1.0))


if _TORCH_AVAILABLE:

    class BoundaryCNN(nn.Module):
        """Dilated 1D-CNN for exon boundary detection.

        Architecture:
            Conv1D(3->16, k=7, d=1) -> BN -> ReLU ->
            Conv1D(16->32, k=5, d=2) -> BN -> ReLU ->
            Conv1D(32->32, k=5, d=4) -> BN -> ReLU ->
            AdaptiveAvgPool1D(1) -> Linear(32->1) -> Sigmoid

        Dilated convolutions capture multi-scale coverage patterns.
        """

        def __init__(
            self,
            n_channels: int = N_CHANNELS,
        ) -> None:
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv1d(n_channels, 16, kernel_size=7, padding=3, dilation=1),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=5, padding=4, dilation=2),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=5, padding=8, dilation=4),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Sequential(
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Input tensor (batch, n_channels, window_size).

            Returns:
                Boundary probabilities (batch, 1).
            """
            h = self.conv_layers(x)
            h = self.pool(h).squeeze(-1)
            return self.classifier(h)


class BoundaryDetector:
    """Exon boundary detector with CNN model and heuristic fallback.

    Predicts whether candidate genomic positions represent true
    exon-intron boundaries based on local coverage patterns.

    Args:
        model_path: Optional path to saved CNN weights.
        window_size: Analysis window size in base pairs.
    """

    def __init__(
        self,
        model_path: str | None = None,
        window_size: int = WINDOW_SIZE,
    ) -> None:
        self._model: object | None = None
        self._is_trained = False
        self._window_size = window_size

        if model_path is not None and _TORCH_AVAILABLE:
            try:
                self._model = BoundaryCNN()
                state = torch.load(
                    model_path, map_location="cpu", weights_only=True
                )
                self._model.load_state_dict(state)
                self._model.eval()
                self._is_trained = True
                logger.info(
                    "Loaded boundary detector from %s", model_path
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load boundary detector: %s", exc
                )
                self._model = None

    @property
    def is_trained(self) -> bool:
        """Whether a trained CNN model is loaded."""
        return self._is_trained

    def predict(
        self,
        coverage: np.ndarray,
        candidate_positions: list[int],
    ) -> list[BoundaryPrediction]:
        """Predict boundary probabilities for candidate positions.

        Args:
            coverage: Per-base coverage array.
            candidate_positions: List of genomic positions to evaluate.

        Returns:
            List of BoundaryPrediction objects.
        """
        if not candidate_positions:
            return []

        if self._is_trained and _TORCH_AVAILABLE and self._model is not None:
            # Batch predict with CNN
            features_list = []
            for pos in candidate_positions:
                feat = extract_boundary_features(
                    coverage, pos, self._window_size
                )
                features_list.append(feat)

            batch = np.stack(features_list, axis=0)
            with torch.no_grad():
                x = torch.from_numpy(batch).float()
                probs = self._model(x).squeeze(-1).numpy()

            results = []
            for i, pos in enumerate(candidate_positions):
                btype = self._infer_boundary_type(coverage, pos)
                results.append(
                    BoundaryPrediction(
                        position=pos,
                        probability=float(probs[i]),
                        boundary_type=btype,
                    )
                )
            return results

        # Heuristic fallback
        results = []
        for pos in candidate_positions:
            prob = heuristic_boundary_score(
                coverage, pos, self._window_size
            )
            btype = self._infer_boundary_type(coverage, pos)
            results.append(
                BoundaryPrediction(
                    position=pos,
                    probability=prob,
                    boundary_type=btype,
                )
            )
        return results

    def refine_boundaries(
        self,
        coverage: np.ndarray,
        initial_start: int,
        initial_end: int,
        search_range: int = 50,
    ) -> tuple[int, int]:
        """Refine exon boundaries by searching for the best boundary positions.

        Evaluates positions within search_range of the initial boundaries
        and returns the positions with highest boundary probability.

        Args:
            coverage: Per-base coverage array.
            initial_start: Initial exon start position.
            initial_end: Initial exon end position.
            search_range: Number of base pairs to search on each side.

        Returns:
            Tuple of (refined_start, refined_end).
        """
        n = len(coverage)

        # Search for best start boundary
        start_candidates = list(range(
            max(0, initial_start - search_range),
            min(n, initial_start + search_range + 1),
        ))
        if start_candidates:
            start_preds = self.predict(coverage, start_candidates)
            best_start_pred = max(start_preds, key=lambda p: p.probability)
            refined_start = best_start_pred.position
        else:
            refined_start = initial_start

        # Search for best end boundary
        end_candidates = list(range(
            max(0, initial_end - search_range),
            min(n, initial_end + search_range + 1),
        ))
        if end_candidates:
            end_preds = self.predict(coverage, end_candidates)
            best_end_pred = max(end_preds, key=lambda p: p.probability)
            refined_end = best_end_pred.position
        else:
            refined_end = initial_end

        # Ensure valid range
        if refined_end <= refined_start:
            return initial_start, initial_end

        return refined_start, refined_end

    @staticmethod
    def _infer_boundary_type(
        coverage: np.ndarray,
        position: int,
    ) -> str:
        """Infer whether a boundary is a start or end based on coverage.

        Args:
            coverage: Per-base coverage array.
            position: Boundary position.

        Returns:
            'start' if coverage rises, 'end' if coverage drops.
        """
        n = len(coverage)
        window = 10
        left_start = max(0, position - window)
        right_end = min(n, position + window)

        if left_start >= position or position >= right_end:
            return "start"

        left_mean = float(np.mean(coverage[left_start:position]))
        right_mean = float(np.mean(coverage[position:right_end]))

        return "start" if right_mean > left_mean else "end"

    def train_model(
        self,
        coverage_windows: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """Train the boundary detection CNN.

        Args:
            coverage_windows: Training features (n_samples, N_CHANNELS, window_size).
            labels: Binary labels (n_samples,), 1 = true boundary.
            n_epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Final training loss.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available; cannot train CNN.")
            return float("nan")

        self._model = BoundaryCNN()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        x = torch.from_numpy(coverage_windows).float()
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
        logger.info(
            "Trained boundary detector: final loss=%.4f", final_loss
        )
        return final_loss

    def save(self, path: str) -> None:
        """Save trained model weights.

        Args:
            path: Output file path.
        """
        if self._is_trained and _TORCH_AVAILABLE and self._model is not None:
            torch.save(self._model.state_dict(), path)
            logger.info("Saved boundary detector to %s", path)
