"""Graph Neural Network transcript scorer using GATv2.

Replaces the RandomForest scorer with a GATv2-based model that operates
directly on the splice graph structure, learning relational patterns
between exons and junctions for transcript quality prediction.

When PyTorch Geometric is not available, falls back to the existing
RandomForest or heuristic scorer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
_PYG_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
    try:
        from torch_geometric.data import Data
        from torch_geometric.nn import GATv2Conv
        _PYG_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

# Node feature dimension: coverage, length, type_onehot(4), position
NODE_FEATURE_DIM = 8
# Edge feature dimension: weight, coverage, type_onehot(4)
EDGE_FEATURE_DIM = 6
# Hidden dimension for GNN layers
HIDDEN_DIM = 64
# Number of attention heads
NUM_HEADS = 4


@dataclass
class GraphData:
    """Lightweight graph data container for non-PyG environments.

    Attributes:
        node_features: Node feature matrix (n_nodes, NODE_FEATURE_DIM).
        edge_index: Edge index array (2, n_edges).
        edge_features: Edge feature matrix (n_edges, EDGE_FEATURE_DIM).
        path_node_mask: Binary mask of nodes in the transcript path (n_nodes,).
    """

    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    path_node_mask: np.ndarray


def splice_graph_to_features(
    csr_graph: object,
    splice_graph: object,
    path_nodes: list[int] | None = None,
) -> GraphData:
    """Convert a splice graph to feature arrays for GNN input.

    Args:
        csr_graph: CSRGraph object with arrays.
        splice_graph: SpliceGraph object with metadata.
        path_nodes: Optional list of node IDs in the candidate transcript.

    Returns:
        GraphData with node/edge features and path mask.
    """
    n_nodes = csr_graph.n_nodes

    # Node features
    node_feats = np.zeros((n_nodes, NODE_FEATURE_DIM), dtype=np.float32)
    for i in range(n_nodes):
        node_feats[i, 0] = csr_graph.node_coverages[i] / max(
            csr_graph.node_coverages.max(), 1.0
        )
        node_feats[i, 1] = (
            csr_graph.node_ends[i] - csr_graph.node_starts[i]
        ) / 10000.0
        nt = int(csr_graph.node_types[i])
        if nt < 4:
            node_feats[i, 2 + nt] = 1.0
        # Relative position within locus
        locus_span = max(
            int(csr_graph.node_ends[-1]) - int(csr_graph.node_starts[0]), 1
        )
        node_feats[i, 6] = (
            csr_graph.node_starts[i] - csr_graph.node_starts[0]
        ) / locus_span
        node_feats[i, 7] = (
            csr_graph.node_ends[i] - csr_graph.node_starts[0]
        ) / locus_span

    # Edge index and features
    src_list: list[int] = []
    dst_list: list[int] = []
    edge_feats_list: list[list[float]] = []

    for i in range(n_nodes):
        start_idx = int(csr_graph.row_offsets[i])
        end_idx = int(csr_graph.row_offsets[i + 1])
        for e in range(start_idx, end_idx):
            j = int(csr_graph.col_indices[e])
            src_list.append(i)
            dst_list.append(j)
            ef = [0.0] * EDGE_FEATURE_DIM
            ef[0] = float(csr_graph.edge_weights[e]) / max(
                float(csr_graph.edge_weights.max()), 1.0
            )
            ef[1] = float(csr_graph.edge_coverages[e]) / max(
                float(csr_graph.edge_coverages.max()), 1.0
            )
            et = int(csr_graph.edge_types[e])
            if et < 4:
                ef[2 + et] = 1.0
            edge_feats_list.append(ef)

    if src_list:
        edge_index = np.array([src_list, dst_list], dtype=np.int64)
        edge_features = np.array(edge_feats_list, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_features = np.empty((0, EDGE_FEATURE_DIM), dtype=np.float32)

    # Path mask
    path_mask = np.zeros(n_nodes, dtype=np.float32)
    if path_nodes:
        for nid in path_nodes:
            if 0 <= nid < n_nodes:
                path_mask[nid] = 1.0

    return GraphData(
        node_features=node_feats,
        edge_index=edge_index,
        edge_features=edge_features,
        path_node_mask=path_mask,
    )


if _PYG_AVAILABLE:
    class GNNTranscriptScorer(nn.Module):
        """GATv2-based transcript scorer.

        Architecture:
            - 2 GATv2 layers with multi-head attention
            - Path-conditioned readout: mask graph to transcript nodes
            - 2-layer MLP classifier

        The model scores how likely a candidate transcript (a path through
        the splice graph) is to be a real biological transcript.
        """

        def __init__(
            self,
            node_dim: int = NODE_FEATURE_DIM,
            edge_dim: int = EDGE_FEATURE_DIM,
            hidden_dim: int = HIDDEN_DIM,
            num_heads: int = NUM_HEADS,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()

            self.conv1 = GATv2Conv(
                node_dim, hidden_dim, heads=num_heads,
                edge_dim=edge_dim, dropout=dropout, concat=False,
            )
            self.conv2 = GATv2Conv(
                hidden_dim, hidden_dim, heads=num_heads,
                edge_dim=edge_dim, dropout=dropout, concat=False,
            )
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)

            # Path-conditioned classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + node_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
            path_mask: torch.Tensor,
            batch: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Forward pass.

            Args:
                x: Node features (n_nodes, node_dim).
                edge_index: Edge index (2, n_edges).
                edge_attr: Edge features (n_edges, edge_dim).
                path_mask: Binary mask for transcript nodes (n_nodes,).
                batch: Batch assignment vector for batched graphs.

            Returns:
                Transcript quality score (batch_size, 1).
            """
            x_orig = x

            # GNN layers
            h = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
            h = F.relu(self.bn2(self.conv2(h, edge_index, edge_attr)))

            # Path-conditioned pooling: mean of masked node embeddings
            mask = path_mask.unsqueeze(-1)
            masked_h = h * mask

            if batch is not None:
                # Batched: use scatter mean
                from torch_geometric.utils import scatter
                denom = scatter(mask.squeeze(-1), batch, reduce="sum").clamp(min=1)
                pooled = scatter(masked_h, batch.unsqueeze(-1).expand_as(masked_h),
                                 dim=0, reduce="sum")
                pooled = pooled / denom.unsqueeze(-1)

                # Also pool original features for skip connection
                masked_orig = x_orig * mask
                pooled_orig = scatter(
                    masked_orig, batch.unsqueeze(-1).expand_as(masked_orig),
                    dim=0, reduce="sum",
                )
                pooled_orig = pooled_orig / denom.unsqueeze(-1)
            else:
                # Single graph
                n_masked = mask.sum().clamp(min=1)
                pooled = masked_h.sum(dim=0, keepdim=True) / n_masked
                pooled_orig = (x_orig * mask).sum(dim=0, keepdim=True) / n_masked

            # Classify
            combined = torch.cat([pooled, pooled_orig], dim=-1)
            return self.classifier(combined)


class GNNScorer:
    """Wrapper for GNN-based transcript scoring with fallback.

    Uses GATv2 model when PyTorch Geometric is available and a trained
    model exists. Otherwise falls back to the provided fallback scorer.

    Args:
        model_path: Optional path to saved GNN weights.
        fallback_fn: Callable that scores a feature array -> float.
    """

    def __init__(
        self,
        model_path: str | None = None,
        fallback_fn: object | None = None,
    ) -> None:
        self._model: object | None = None
        self._is_trained = False
        self._fallback_fn = fallback_fn

        if model_path is not None and _PYG_AVAILABLE:
            try:
                self._model = GNNTranscriptScorer()
                state = torch.load(model_path, map_location="cpu", weights_only=True)
                self._model.load_state_dict(state)
                self._model.eval()
                self._is_trained = True
                logger.info("Loaded GNN scorer from %s", model_path)
            except Exception as exc:
                logger.warning("Failed to load GNN scorer: %s", exc)
                self._model = None

    @property
    def is_trained(self) -> bool:
        """Whether a trained GNN model is loaded."""
        return self._is_trained

    def score(
        self,
        graph_data: GraphData,
    ) -> float:
        """Score a single transcript in its splice graph context.

        Args:
            graph_data: Graph data with path mask for the transcript.

        Returns:
            Quality score in [0, 1].
        """
        if self._is_trained and _PYG_AVAILABLE and self._model is not None:
            with torch.no_grad():
                x = torch.from_numpy(graph_data.node_features)
                ei = torch.from_numpy(graph_data.edge_index)
                ea = torch.from_numpy(graph_data.edge_features)
                pm = torch.from_numpy(graph_data.path_node_mask)
                score = self._model(x, ei, ea, pm)
                return float(score.item())

        # Fallback: use path mask to compute simple features
        mask = graph_data.path_node_mask > 0
        if mask.sum() == 0:
            return 0.0
        path_coverage = graph_data.node_features[mask, 0].mean()
        return float(np.clip(path_coverage, 0.0, 1.0))

    def train_model(
        self,
        graph_data_list: list[GraphData],
        labels: np.ndarray,
        n_epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """Train the GNN scorer on labeled transcript data.

        Args:
            graph_data_list: List of GraphData objects.
            labels: Binary labels (n_samples,).
            n_epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Final training loss.
        """
        if not _PYG_AVAILABLE:
            logger.warning("PyTorch Geometric not available; cannot train GNN.")
            return float("nan")

        # Convert to PyG Data objects
        data_list = []
        for gd, label in zip(graph_data_list, labels):
            data = Data(
                x=torch.from_numpy(gd.node_features),
                edge_index=torch.from_numpy(gd.edge_index),
                edge_attr=torch.from_numpy(gd.edge_features),
                path_mask=torch.from_numpy(gd.path_node_mask),
                y=torch.tensor([float(label)]),
            )
            data_list.append(data)

        self._model = GNNTranscriptScorer()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self._model.train()
        final_loss = 0.0
        for epoch in range(n_epochs):
            total_loss = 0.0
            for data in data_list:
                optimizer.zero_grad()
                pred = self._model(
                    data.x, data.edge_index, data.edge_attr, data.path_mask,
                )
                loss = criterion(pred, data.y.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            final_loss = total_loss / len(data_list)

        self._model.eval()
        self._is_trained = True
        logger.info("Trained GNN scorer: final loss=%.4f", final_loss)
        return final_loss

    def save(self, path: str) -> None:
        """Save trained model weights.

        Args:
            path: Output file path.
        """
        if self._is_trained and _PYG_AVAILABLE and self._model is not None:
            torch.save(self._model.state_dict(), path)
            logger.info("Saved GNN scorer to %s", path)
