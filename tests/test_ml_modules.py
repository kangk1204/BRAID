"""Tests for ML/DL enhancement modules.

Tests junction scoring, GNN scorer, neural decomposition, auto filter,
Transformer classifier, neural PSI estimation, and boundary detection.
"""

from __future__ import annotations

import math

import numpy as np
import pytest


def _has_working_torch() -> bool:
    """Return True only when PyTorch imports cleanly."""
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


_REQUIRES_WORKING_TORCH = pytest.mark.skipif(
    not _has_working_torch(),
    reason="requires a working PyTorch install",
)

# ---------------------------------------------------------------------------
# Junction Scorer Tests
# ---------------------------------------------------------------------------


class TestJunctionScorer:
    """Tests for the CNN-based junction quality scorer."""

    def test_extract_junction_features_basic(self) -> None:
        from braid.scoring.junction_scorer import (
            N_JUNCTION_FEATURES,
            extract_junction_features,
        )

        starts = np.array([100, 500, 800], dtype=np.int64)
        ends = np.array([200, 600, 900], dtype=np.int64)
        counts = np.array([10, 5, 20], dtype=np.int64)
        strands = np.array([0, 0, 1], dtype=np.int64)

        features = extract_junction_features(
            starts, ends, counts, strands, 50, 1000,
        )

        assert features.shape == (3, N_JUNCTION_FEATURES)
        assert features[0, 0] == 10  # read_count
        assert features[0, 1] == pytest.approx(math.log1p(10), abs=1e-5)
        assert features[2, 0] == 20

    def test_extract_junction_features_empty(self) -> None:
        from braid.scoring.junction_scorer import (
            N_JUNCTION_FEATURES,
            extract_junction_features,
        )

        features = extract_junction_features(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            0, 1000,
        )
        assert features.shape == (0, N_JUNCTION_FEATURES)

    def test_heuristic_junction_score(self) -> None:
        from braid.scoring.junction_scorer import (
            extract_junction_features,
            heuristic_junction_score,
        )

        starts = np.array([100, 500], dtype=np.int64)
        ends = np.array([200, 600], dtype=np.int64)
        counts = np.array([50, 2], dtype=np.int64)
        strands = np.array([0, 0], dtype=np.int64)

        features = extract_junction_features(
            starts, ends, counts, strands, 0, 1000,
        )
        scores = heuristic_junction_score(features)

        assert scores.shape == (2,)
        assert 0.0 <= scores[0] <= 1.0
        assert 0.0 <= scores[1] <= 1.0
        # High-count junction should score higher
        assert scores[0] > scores[1]

    def test_junction_scorer_wrapper(self) -> None:
        from braid.scoring.junction_scorer import (
            JunctionScorer,
            extract_junction_features,
        )

        scorer = JunctionScorer()
        assert not scorer.is_trained

        starts = np.array([100], dtype=np.int64)
        ends = np.array([200], dtype=np.int64)
        counts = np.array([10], dtype=np.int64)
        strands = np.array([0], dtype=np.int64)

        features = extract_junction_features(
            starts, ends, counts, strands, 0, 1000,
        )
        scores = scorer.score(features)
        assert scores.shape == (1,)

    @_REQUIRES_WORKING_TORCH
    def test_junction_scorer_train(self) -> None:
        from braid.scoring.junction_scorer import JunctionScorer

        scorer = JunctionScorer()
        features = np.random.randn(20, 10).astype(np.float32)
        labels = np.random.randint(0, 2, size=20).astype(np.float32)

        loss = scorer.train_model(features, labels, n_epochs=5, lr=1e-3)
        assert scorer.is_trained
        assert not math.isnan(loss)


# ---------------------------------------------------------------------------
# GNN Model Tests
# ---------------------------------------------------------------------------


class TestGNNModel:
    """Tests for the GATv2-based transcript scorer."""

    def test_graph_data_creation(self) -> None:
        from braid.scoring.gnn_model import GraphData

        gd = GraphData(
            node_features=np.zeros((5, 8), dtype=np.float32),
            edge_index=np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
            edge_features=np.zeros((3, 6), dtype=np.float32),
            path_node_mask=np.array([1, 1, 1, 0, 0], dtype=np.float32),
        )
        assert gd.node_features.shape == (5, 8)
        assert gd.edge_index.shape == (2, 3)

    def test_gnn_scorer_fallback(self) -> None:
        from braid.scoring.gnn_model import GNNScorer, GraphData

        scorer = GNNScorer()
        assert not scorer.is_trained

        gd = GraphData(
            node_features=np.random.rand(4, 8).astype(np.float32),
            edge_index=np.array([[0, 1], [1, 2]], dtype=np.int64),
            edge_features=np.random.rand(2, 6).astype(np.float32),
            path_node_mask=np.array([1, 1, 0, 0], dtype=np.float32),
        )

        score = scorer.score(gd)
        assert 0.0 <= score <= 1.0

    def test_gnn_scorer_empty_mask(self) -> None:
        from braid.scoring.gnn_model import GNNScorer, GraphData

        scorer = GNNScorer()
        gd = GraphData(
            node_features=np.random.rand(3, 8).astype(np.float32),
            edge_index=np.array([[0], [1]], dtype=np.int64),
            edge_features=np.random.rand(1, 6).astype(np.float32),
            path_node_mask=np.zeros(3, dtype=np.float32),
        )

        score = scorer.score(gd)
        assert score == 0.0


# ---------------------------------------------------------------------------
# Neural Decomposition Tests
# ---------------------------------------------------------------------------


class TestNeuralDecompose:
    """Tests for neural-guided flow decomposition."""

    def test_extract_path_features(self) -> None:
        from braid.flow.neural_decompose import (
            PATH_FEATURE_DIM,
            extract_path_features,
        )

        path = [0, 1, 2, 3]
        node_starts = np.array([100, 200, 500, 600], dtype=np.int64)
        node_ends = np.array([200, 400, 550, 700], dtype=np.int64)
        node_coverages = np.array([10.0, 15.0, 12.0, 8.0], dtype=np.float64)
        node_types = np.array([0, 1, 1, 0], dtype=np.int32)
        edge_weights = {(0, 1): 10.0, (1, 2): 12.0, (2, 3): 8.0}

        feats = extract_path_features(
            path, node_starts, node_ends, node_coverages,
            node_types, edge_weights,
        )
        assert feats.shape == (PATH_FEATURE_DIM,)
        assert feats[0] == 2  # 2 exon nodes (type 1)

    def test_heuristic_path_plausibility(self) -> None:
        from braid.flow.neural_decompose import (
            PATH_FEATURE_DIM,
            heuristic_path_plausibility,
        )

        features = np.random.rand(5, PATH_FEATURE_DIM).astype(np.float32)
        scores = heuristic_path_plausibility(features)
        assert scores.shape == (5,)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_fit_path_weights_neural(self) -> None:
        from braid.flow.neural_decompose import fit_path_weights_neural

        # 3 edges, 2 paths
        A = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float64)
        b = np.array([10.0, 15.0, 5.0], dtype=np.float64)
        features = np.random.rand(2, 12).astype(np.float32)

        weights = fit_path_weights_neural(A, b, features)
        assert weights.shape == (2,)
        assert all(w >= 0 for w in weights)

    def test_neural_decomposer_wrapper(self) -> None:
        from braid.flow.neural_decompose import NeuralDecomposer

        decomposer = NeuralDecomposer()
        assert not decomposer.is_trained

        A = np.array([[1, 1], [1, 0]], dtype=np.float64)
        b = np.array([10.0, 6.0], dtype=np.float64)
        features = np.random.rand(2, 12).astype(np.float32)

        weights = decomposer.fit_weights(A, b, features)
        assert weights.shape == (2,)

    @_REQUIRES_WORKING_TORCH
    def test_neural_decomposer_train(self) -> None:
        from braid.flow.neural_decompose import NeuralDecomposer

        decomposer = NeuralDecomposer()
        features = np.random.rand(30, 12).astype(np.float32)
        labels = np.random.randint(0, 2, size=30).astype(np.float32)

        loss = decomposer.train_model(features, labels, n_epochs=5, lr=1e-3)
        assert decomposer.is_trained
        assert not math.isnan(loss)


# ---------------------------------------------------------------------------
# Auto Filter Optimizer Tests
# ---------------------------------------------------------------------------


class TestAutoFilter:
    """Tests for Optuna-based filter optimization."""

    @pytest.fixture()
    def mock_transcripts(self) -> tuple:
        """Create minimal mock transcripts for filter testing."""
        from braid.flow.decompose import Transcript
        from braid.scoring.features import TranscriptFeatures

        transcripts = [
            Transcript(
                node_ids=[0, 1, 2],
                exon_coords=[(100, 200), (300, 400)],
                weight=10.0,
                is_safe=True,
            ),
            Transcript(
                node_ids=[0, 3, 2],
                exon_coords=[(100, 200), (500, 600)],
                weight=5.0,
                is_safe=False,
            ),
            Transcript(
                node_ids=[0, 1, 3, 2],
                exon_coords=[(100, 200), (300, 400), (500, 600)],
                weight=8.0,
                is_safe=True,
            ),
        ]

        scores = np.array([0.8, 0.3, 0.6], dtype=np.float64)

        features_list = []
        for tx in transcripts:
            total_len = sum(e[1] - e[0] for e in tx.exon_coords)
            f = TranscriptFeatures(
                n_exons=len(tx.exon_coords),
                n_junctions=max(0, len(tx.exon_coords) - 1),
                total_length=total_len,
                mean_coverage=tx.weight,
                min_junction_support=3,
            )
            features_list.append(f)

        return transcripts, scores, features_list

    def test_evaluate_filter_config_unsupervised(self, mock_transcripts: tuple) -> None:
        from braid.scoring.auto_filter import evaluate_filter_config
        from braid.scoring.filter import FilterConfig

        transcripts, scores, features_list = mock_transcripts
        cfg = FilterConfig(min_score=0.2, min_coverage=1.0)

        score = evaluate_filter_config(
            cfg, transcripts, scores, features_list,
        )
        assert 0.0 <= score <= 1.0

    def test_evaluate_filter_config_supervised(self, mock_transcripts: tuple) -> None:
        from braid.scoring.auto_filter import evaluate_filter_config
        from braid.scoring.filter import FilterConfig

        transcripts, scores, features_list = mock_transcripts

        # Reference contains one matching exon chain
        ref_chains = {
            ((100, 200), (300, 400)),
            ((100, 200), (300, 400), (500, 600)),
        }

        cfg = FilterConfig(min_score=0.2)
        score = evaluate_filter_config(
            cfg, transcripts, scores, features_list, ref_chains,
        )
        assert 0.0 <= score <= 1.0

    def test_grid_search(self, mock_transcripts: tuple) -> None:
        from braid.scoring.auto_filter import optimize_filter_grid

        transcripts, scores, features_list = mock_transcripts

        result = optimize_filter_grid(
            transcripts, scores, features_list,
        )
        assert result.n_trials > 0
        assert result.best_score >= 0.0
        assert result.best_config is not None

    def test_optuna_optimization(self, mock_transcripts: tuple) -> None:
        from braid.scoring.auto_filter import (
            optimize_filter_optuna,
        )

        transcripts, scores, features_list = mock_transcripts

        result = optimize_filter_optuna(
            transcripts, scores, features_list,
            n_trials=5, timeout=10.0,
        )
        assert result.n_trials > 0
        assert result.best_score >= 0.0
        assert len(result.history) > 0

    def test_auto_filter_optimizer_wrapper(self, mock_transcripts: tuple) -> None:
        from braid.scoring.auto_filter import AutoFilterOptimizer

        transcripts, scores, features_list = mock_transcripts

        optimizer = AutoFilterOptimizer(n_trials=3, timeout=5.0)
        assert not optimizer.is_optimized

        result = optimizer.optimize(transcripts, scores, features_list)
        assert optimizer.is_optimized
        assert optimizer.best_config is not None
        assert result.best_score >= 0.0


# ---------------------------------------------------------------------------
# Transformer Classifier Tests
# ---------------------------------------------------------------------------


class TestTransformerClassifier:
    """Tests for the Transformer-based AS event classifier."""

    def test_transformer_event_classifier_heuristic_fallback(self) -> None:
        from braid.splicing.classifier import (
            EventFeatures,
            TransformerEventClassifier,
        )

        clf = TransformerEventClassifier()
        assert not clf.is_trained

        features = EventFeatures(
            inclusion_count=50,
            exclusion_count=10,
            total_reads=60,
            psi=0.83,
            log_total_reads=math.log1p(60),
            junction_balance=0.2,
        )
        score = clf.score(features)
        assert 0.0 <= score <= 1.0

    @_REQUIRES_WORKING_TORCH
    def test_transformer_train_and_score(self) -> None:
        from braid.splicing.classifier import TransformerEventClassifier

        clf = TransformerEventClassifier(gbm_fallback=False)

        features = np.random.rand(20, 21).astype(np.float64)
        labels = np.random.randint(0, 2, size=20).astype(np.float64)

        loss = clf.train_model(features, labels, n_epochs=5, lr=1e-3)
        assert clf.is_trained
        assert not math.isnan(loss)

    def test_transformer_score_batch(self) -> None:
        from braid.splicing.classifier import (
            TransformerEventClassifier,
        )
        from braid.splicing.events import ASEvent, EventType
        from braid.splicing.psi import PSIResult

        clf = TransformerEventClassifier(gbm_fallback=False)

        # Train first
        features = np.random.rand(20, 21).astype(np.float64)
        labels = np.random.randint(0, 2, size=20).astype(np.float64)
        clf.train_model(features, labels, n_epochs=5, lr=1e-3)

        events = [
            ASEvent(
                event_id="test_1",
                event_type=EventType.SE,
                gene_id="gene1",
                chrom="chr1",
                strand="+",
                coordinates={"upstream_exon": (100, 200)},
                inclusion_transcripts=["tx1"],
                exclusion_transcripts=["tx2"],
                inclusion_junctions=[(200, 500)],
                exclusion_junctions=[(200, 700)],
            ),
        ]
        psi_results = [
            PSIResult(
                event_id="test_1",
                psi=0.7,
                inclusion_count=30,
                exclusion_count=10,
                inclusion_length=1,
                exclusion_length=1,
                total_reads=40,
            ),
        ]

        scores = clf.score_batch(events, psi_results)
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0


# ---------------------------------------------------------------------------
# Neural PSI Tests
# ---------------------------------------------------------------------------


class TestNeuralPSI:
    """Tests for neural PSI estimation."""

    def test_extract_psi_features(self) -> None:
        from braid.splicing.neural_psi import (
            PSI_FEATURE_DIM,
            extract_psi_features,
        )

        feats = extract_psi_features(
            inclusion_count=50,
            exclusion_count=10,
            n_inclusion_junctions=2,
            n_exclusion_junctions=1,
            event_type=0,
        )
        assert feats.shape == (PSI_FEATURE_DIM,)
        assert feats[0] == pytest.approx(math.log1p(50), abs=1e-5)

    def test_beta_binomial_fallback(self) -> None:
        from braid.splicing.neural_psi import beta_binomial_fallback

        result = beta_binomial_fallback(50, 10)
        assert 0.0 <= result.psi_mean <= 1.0
        assert result.ci_low < result.ci_high
        assert result.ci_low >= 0.0
        assert result.ci_high <= 1.0

    def test_beta_binomial_fallback_equal_counts(self) -> None:
        from braid.splicing.neural_psi import beta_binomial_fallback

        result = beta_binomial_fallback(20, 20)
        assert abs(result.psi_mean - 0.5) < 0.05

    def test_neural_psi_estimator_fallback(self) -> None:
        from braid.splicing.neural_psi import NeuralPSIEstimator

        estimator = NeuralPSIEstimator()
        assert not estimator.is_trained

        result = estimator.estimate(30, 10)
        assert 0.0 <= result.psi_mean <= 1.0
        assert result.ci_low < result.ci_high

    @_REQUIRES_WORKING_TORCH
    def test_neural_psi_train(self) -> None:
        from braid.splicing.neural_psi import NeuralPSIEstimator

        estimator = NeuralPSIEstimator()
        features = np.random.rand(30, 8).astype(np.float32)
        targets = np.random.rand(30).astype(np.float32)

        loss = estimator.train_model(features, targets, n_epochs=5, lr=1e-3)
        assert estimator.is_trained
        assert not math.isnan(loss)

    @_REQUIRES_WORKING_TORCH
    def test_neural_psi_trained_estimate(self) -> None:
        from braid.splicing.neural_psi import NeuralPSIEstimator

        estimator = NeuralPSIEstimator()
        features = np.random.rand(30, 8).astype(np.float32)
        targets = np.random.rand(30).astype(np.float32)
        estimator.train_model(features, targets, n_epochs=10, lr=1e-3)

        result = estimator.estimate(50, 10, event_type=0)
        assert 0.0 <= result.psi_mean <= 1.0
        assert len(result.mixture_weights) == 3
        assert len(result.mixture_means) == 3


# ---------------------------------------------------------------------------
# Boundary Detector Tests
# ---------------------------------------------------------------------------


class TestBoundaryDetector:
    """Tests for the 1D-CNN exon boundary detector."""

    def test_extract_boundary_features(self) -> None:
        from braid.graph.boundary_detector import (
            N_CHANNELS,
            WINDOW_SIZE,
            extract_boundary_features,
        )

        coverage = np.concatenate([
            np.ones(100) * 50,
            np.ones(100) * 5,
        ]).astype(np.float32)

        features = extract_boundary_features(coverage, 100, WINDOW_SIZE)
        assert features.shape == (N_CHANNELS, WINDOW_SIZE)

    def test_extract_boundary_features_edge(self) -> None:
        from braid.graph.boundary_detector import (
            N_CHANNELS,
            extract_boundary_features,
        )

        coverage = np.ones(50, dtype=np.float32) * 10
        features = extract_boundary_features(coverage, 5, 200)
        assert features.shape == (N_CHANNELS, 200)

    def test_heuristic_boundary_score(self) -> None:
        from braid.graph.boundary_detector import heuristic_boundary_score

        # Create coverage with a sharp drop
        coverage = np.concatenate([
            np.ones(100) * 100,
            np.ones(100) * 2,
        ]).astype(np.float32)

        # At boundary
        score_at_boundary = heuristic_boundary_score(coverage, 100)
        # Away from boundary
        score_away = heuristic_boundary_score(coverage, 50)

        assert score_at_boundary > score_away

    def test_boundary_detector_fallback(self) -> None:
        from braid.graph.boundary_detector import BoundaryDetector

        detector = BoundaryDetector()
        assert not detector.is_trained

        coverage = np.concatenate([
            np.ones(100) * 50,
            np.ones(100) * 5,
        ]).astype(np.float32)

        preds = detector.predict(coverage, [50, 100, 150])
        assert len(preds) == 3
        assert all(0.0 <= p.probability <= 1.0 for p in preds)
        assert all(p.boundary_type in ("start", "end") for p in preds)

    def test_boundary_detector_empty(self) -> None:
        from braid.graph.boundary_detector import BoundaryDetector

        detector = BoundaryDetector()
        preds = detector.predict(np.zeros(100), [])
        assert len(preds) == 0

    def test_refine_boundaries(self) -> None:
        from braid.graph.boundary_detector import BoundaryDetector

        detector = BoundaryDetector()

        # Coverage with clear boundary at position 100
        coverage = np.concatenate([
            np.ones(100) * 50,
            np.ones(100) * 2,
        ]).astype(np.float32)

        refined_start, refined_end = detector.refine_boundaries(
            coverage, 95, 105, search_range=20,
        )
        assert refined_start >= 0
        assert refined_end > refined_start

    @_REQUIRES_WORKING_TORCH
    def test_boundary_detector_train(self) -> None:
        from braid.graph.boundary_detector import BoundaryDetector

        detector = BoundaryDetector()
        windows = np.random.rand(20, 3, 200).astype(np.float32)
        labels = np.random.randint(0, 2, size=20).astype(np.float32)

        loss = detector.train_model(windows, labels, n_epochs=5, lr=1e-3)
        assert detector.is_trained
        assert not math.isnan(loss)

    def test_boundary_detector_trained_predict(self) -> None:
        from braid.graph.boundary_detector import BoundaryDetector

        detector = BoundaryDetector()
        windows = np.random.rand(20, 3, 200).astype(np.float32)
        labels = np.random.randint(0, 2, size=20).astype(np.float32)
        detector.train_model(windows, labels, n_epochs=5, lr=1e-3)

        coverage = np.random.rand(200).astype(np.float32) * 50
        preds = detector.predict(coverage, [50, 100, 150])
        assert len(preds) == 3
        assert all(0.0 <= p.probability <= 1.0 for p in preds)
