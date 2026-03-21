"""Neural-guided flow decomposition for transcript assembly.

Augments the NNLS path weight fitting with a learned path plausibility
prior, improving decomposition accuracy at complex loci with many
overlapping isoforms.

Architecture:
    1. Path Scoring Network: MLP that scores candidate paths for
       biological plausibility.
    2. Regularized NNLS: Original coverage fitting + learned plausibility
       regularization term.

Falls back to standard NNLS when PyTorch is unavailable.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.optimize import nnls

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception as exc:
    logger.debug("PyTorch unavailable for neural decomposer fallback: %s", exc)

# Path feature dimension
PATH_FEATURE_DIM = 12


def extract_path_features(
    path: list[int],
    node_starts: np.ndarray,
    node_ends: np.ndarray,
    node_coverages: np.ndarray,
    node_types: np.ndarray,
    edge_weights: dict[tuple[int, int], float],
) -> np.ndarray:
    """Extract a feature vector for a single source-to-sink path.

    Args:
        path: Node IDs along the path.
        node_starts: Node start coordinates.
        node_ends: Node end coordinates.
        node_coverages: Per-node coverage values.
        node_types: Node type integers.
        edge_weights: Mapping (src, dst) -> edge weight.

    Returns:
        Feature vector of shape (PATH_FEATURE_DIM,).
    """
    feats = np.zeros(PATH_FEATURE_DIM, dtype=np.float32)

    # Filter to exon nodes (type 1 typically)
    exon_nodes = [n for n in path if int(node_types[n]) == 1]
    n_exons = len(exon_nodes)

    feats[0] = n_exons
    feats[1] = math.log1p(n_exons)

    if exon_nodes:
        lengths = [int(node_ends[n]) - int(node_starts[n]) for n in exon_nodes]
        coverages = [float(node_coverages[n]) for n in exon_nodes]

        feats[2] = np.mean(lengths) / 1000.0
        feats[3] = np.std(lengths) / 1000.0 if len(lengths) > 1 else 0.0
        feats[4] = np.mean(coverages)
        feats[5] = np.min(coverages) / max(np.max(coverages), 1.0)
        feats[6] = np.std(coverages) / max(np.mean(coverages), 1.0)

        # Intron lengths
        introns = []
        for k in range(len(exon_nodes) - 1):
            intron_len = int(node_starts[exon_nodes[k + 1]]) - int(
                node_ends[exon_nodes[k]]
            )
            introns.append(intron_len)
        if introns:
            feats[7] = np.mean(introns) / 10000.0
            feats[8] = math.log1p(max(introns))

    # Edge weight consistency along path
    edge_ws = []
    for k in range(len(path) - 1):
        key = (path[k], path[k + 1])
        if key in edge_weights:
            edge_ws.append(edge_weights[key])
    if edge_ws:
        feats[9] = np.mean(edge_ws)
        feats[10] = np.min(edge_ws) / max(np.max(edge_ws), 1.0)

    # Total span
    if exon_nodes:
        feats[11] = (
            int(node_ends[exon_nodes[-1]]) - int(node_starts[exon_nodes[0]])
        ) / 100000.0

    return feats


def heuristic_path_plausibility(features: np.ndarray) -> np.ndarray:
    """Score paths for biological plausibility using a heuristic.

    Args:
        features: Path feature matrix (n_paths, PATH_FEATURE_DIM).

    Returns:
        Plausibility scores in [0, 1] of shape (n_paths,).
    """
    n = features.shape[0]
    scores = np.zeros(n, dtype=np.float64)

    for i in range(n):
        n_exons = features[i, 0]
        cov_uniformity = features[i, 5]
        cov_cv = features[i, 6]
        edge_consistency = features[i, 10]

        # Prefer multi-exon transcripts with uniform coverage
        exon_score = min(n_exons / 5.0, 1.0) * 0.2
        uniformity_score = cov_uniformity * 0.3
        cv_penalty = max(0, 1.0 - cov_cv) * 0.2
        edge_score = edge_consistency * 0.3

        scores[i] = min(exon_score + uniformity_score + cv_penalty + edge_score, 1.0)

    return scores


if _TORCH_AVAILABLE:
    class PathScorerMLP(nn.Module):
        """MLP for scoring path biological plausibility.

        Architecture: Input(12) -> Linear(32) -> ReLU -> Linear(16)
        -> ReLU -> Linear(1) -> Sigmoid.
        """

        def __init__(self, input_dim: int = PATH_FEATURE_DIM) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Path features (batch, input_dim).

            Returns:
                Plausibility scores (batch, 1).
            """
            return self.net(x)


def fit_path_weights_neural(
    edge_path_matrix: np.ndarray,
    observed_coverages: np.ndarray,
    path_features: np.ndarray,
    plausibility_model: object | None = None,
    regularization_weight: float = 0.1,
) -> np.ndarray:
    """Fit path weights using NNLS with neural plausibility regularization.

    Solves:
        min ||A @ w - b||^2 + lambda * ||w * (1 - plausibility)||^2

    where plausibility scores come from the path scorer network.

    Args:
        edge_path_matrix: Binary incidence matrix (n_edges, n_paths).
        observed_coverages: Target edge coverage vector (n_edges,).
        path_features: Path features (n_paths, PATH_FEATURE_DIM).
        plausibility_model: Optional trained PathScorerMLP.
        regularization_weight: Weight for the plausibility penalty.

    Returns:
        Path weight vector (n_paths,).
    """
    n_edges, n_paths = edge_path_matrix.shape

    # Get plausibility scores
    if plausibility_model is not None and _TORCH_AVAILABLE:
        with torch.no_grad():
            x = torch.from_numpy(path_features).float()
            plausibility = plausibility_model(x).squeeze(-1).numpy()
    else:
        plausibility = heuristic_path_plausibility(path_features)

    # Build augmented system: [A; sqrt(lambda) * diag(1 - plausibility)]
    penalty_diag = np.diag(
        np.sqrt(regularization_weight) * (1.0 - plausibility)
    )
    A_aug = np.vstack([edge_path_matrix, penalty_diag])
    b_aug = np.concatenate([observed_coverages, np.zeros(n_paths)])

    # Solve augmented NNLS
    weights, _ = nnls(A_aug, b_aug)

    return weights


class NeuralDecomposer:
    """Neural-guided flow decomposition wrapper.

    Provides the plausibility-regularized NNLS solver as a drop-in
    replacement for standard NNLS in the decomposition pipeline.

    Args:
        model_path: Optional path to trained PathScorerMLP weights.
        regularization_weight: Lambda for plausibility penalty.
    """

    def __init__(
        self,
        model_path: str | None = None,
        regularization_weight: float = 0.1,
    ) -> None:
        self._model: object | None = None
        self._is_trained = False
        self._reg_weight = regularization_weight

        if model_path is not None and _TORCH_AVAILABLE:
            try:
                self._model = PathScorerMLP()
                state = torch.load(model_path, map_location="cpu", weights_only=True)
                self._model.load_state_dict(state)
                self._model.eval()
                self._is_trained = True
                logger.info("Loaded neural decomposer from %s", model_path)
            except Exception as exc:
                logger.warning("Failed to load neural decomposer: %s", exc)
                self._model = None

    @property
    def is_trained(self) -> bool:
        """Whether a trained path scorer is loaded."""
        return self._is_trained

    def fit_weights(
        self,
        edge_path_matrix: np.ndarray,
        observed_coverages: np.ndarray,
        path_features: np.ndarray,
    ) -> np.ndarray:
        """Fit path weights with neural regularization.

        Args:
            edge_path_matrix: (n_edges, n_paths) incidence matrix.
            observed_coverages: (n_edges,) coverage targets.
            path_features: (n_paths, PATH_FEATURE_DIM) features.

        Returns:
            Path weight vector (n_paths,).
        """
        return fit_path_weights_neural(
            edge_path_matrix,
            observed_coverages,
            path_features,
            plausibility_model=self._model,
            regularization_weight=self._reg_weight,
        )

    def train_model(
        self,
        path_features: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """Train the path plausibility scorer.

        Args:
            path_features: Training features (n_samples, PATH_FEATURE_DIM).
            labels: Binary labels (n_samples,), 1 = true path.
            n_epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Final training loss.
        """
        if not _TORCH_AVAILABLE:
            logger.warning("PyTorch not available; cannot train.")
            return float("nan")

        self._model = PathScorerMLP()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        x = torch.from_numpy(path_features).float()
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
        logger.info("Trained path scorer: final loss=%.4f", final_loss)
        return final_loss

    def save(self, path: str) -> None:
        """Save trained model weights.

        Args:
            path: Output file path.
        """
        if self._is_trained and _TORCH_AVAILABLE and self._model is not None:
            torch.save(self._model.state_dict(), path)
