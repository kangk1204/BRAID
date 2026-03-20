"""Tests for scoring modules: features, model, and filter.

Exercises feature extraction, feature-to-array conversion, heuristic scoring,
model training/prediction, and the transcript filtering pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from rapidsplice.flow.decompose import Transcript
from rapidsplice.graph.splice_graph import EdgeType, NodeType, SpliceGraph
from rapidsplice.io.bam_reader import JunctionEvidence
from rapidsplice.scoring.features import (
    TranscriptFeatures,
    extract_features,
    feature_names,
    features_to_array,
)
from rapidsplice.scoring.filter import (
    FilterConfig,
    TranscriptFilter,
)
from rapidsplice.scoring.model import TranscriptScorer

# ===================================================================
# Helper: build a mini locus for feature extraction
# ===================================================================


def _make_mini_locus() -> tuple[
    Transcript,
    SpliceGraph,
    list[Transcript],
    JunctionEvidence,
]:
    """Build a minimal locus with one 2-exon transcript for feature tests.

    Returns:
        (transcript, graph, locus_transcripts, junction_evidence)
    """
    graph = SpliceGraph(chrom="chr1", strand="+", locus_start=100, locus_end=500)
    source_id = graph.add_node(start=100, end=100, node_type=NodeType.SOURCE, coverage=0.0)
    exon1_id = graph.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=10.0)
    exon2_id = graph.add_node(start=300, end=500, node_type=NodeType.EXON, coverage=8.0)
    sink_id = graph.add_node(start=500, end=500, node_type=NodeType.SINK, coverage=0.0)

    graph.add_edge(source_id, exon1_id, EdgeType.SOURCE_LINK, weight=10.0, coverage=10.0)
    graph.add_edge(exon1_id, exon2_id, EdgeType.INTRON, weight=5.0, coverage=5.0)
    graph.add_edge(exon2_id, sink_id, EdgeType.SINK_LINK, weight=8.0, coverage=8.0)

    tx = Transcript(
        node_ids=[source_id, exon1_id, exon2_id, sink_id],
        exon_coords=[(100, 200), (300, 500)],
        weight=5.0,
        is_safe=True,
    )

    junction_evidence = JunctionEvidence(
        chrom="chr1",
        starts=np.array([200], dtype=np.int64),
        ends=np.array([300], dtype=np.int64),
        counts=np.array([5], dtype=np.int32),
        strands=np.array([0], dtype=np.int8),
    )

    return tx, graph, [tx], junction_evidence


# ===================================================================
# Feature extraction tests
# ===================================================================


class TestFeatureExtraction:
    """Test feature extraction produces correct number of features."""

    def test_extract_features_returns_dataclass(self) -> None:
        """extract_features returns a TranscriptFeatures instance."""
        tx, graph, locus_txs, junc_ev = _make_mini_locus()
        feat = extract_features(tx, graph, locus_txs, junc_ev)
        assert isinstance(feat, TranscriptFeatures)

    def test_all_fields_populated(self) -> None:
        """All feature fields are populated with finite float values."""
        tx, graph, locus_txs, junc_ev = _make_mini_locus()
        feat = extract_features(tx, graph, locus_txs, junc_ev)
        arr = features_to_array(feat)
        assert np.all(np.isfinite(arr)), "Some features are not finite"

    def test_coverage_features_positive(self) -> None:
        """Mean and max coverage features are positive for a covered transcript."""
        tx, graph, locus_txs, junc_ev = _make_mini_locus()
        feat = extract_features(tx, graph, locus_txs, junc_ev)
        assert feat.mean_coverage > 0.0
        assert feat.max_coverage > 0.0

    def test_structure_features(self) -> None:
        """Structure features reflect exon count and length."""
        tx, graph, locus_txs, junc_ev = _make_mini_locus()
        feat = extract_features(tx, graph, locus_txs, junc_ev)
        assert feat.n_exons == 2.0
        # Total length = (200-100) + (500-300) = 100 + 200 = 300
        assert feat.total_length == 300.0
        assert feat.n_junctions == 1.0

    def test_graph_context_features(self) -> None:
        """Graph context features reflect the graph topology."""
        tx, graph, locus_txs, junc_ev = _make_mini_locus()
        feat = extract_features(tx, graph, locus_txs, junc_ev)
        assert feat.graph_n_nodes == 4.0
        assert feat.graph_n_edges == 3.0
        assert feat.is_safe_path == 1.0

    def test_single_exon_transcript(self) -> None:
        """A single-exon transcript has 0 junctions and is_single_exon=1."""
        graph = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
        src = graph.add_node(start=0, end=0, node_type=NodeType.SOURCE, coverage=0.0)
        exon = graph.add_node(start=100, end=400, node_type=NodeType.EXON, coverage=15.0)
        sink = graph.add_node(start=500, end=500, node_type=NodeType.SINK, coverage=0.0)
        graph.add_edge(src, exon, EdgeType.SOURCE_LINK, weight=15.0, coverage=15.0)
        graph.add_edge(exon, sink, EdgeType.SINK_LINK, weight=15.0, coverage=15.0)

        tx = Transcript(
            node_ids=[src, exon, sink],
            exon_coords=[(100, 400)],
            weight=15.0,
            is_safe=False,
        )
        junc_ev = JunctionEvidence(
            chrom="chr1",
            starts=np.array([], dtype=np.int64),
            ends=np.array([], dtype=np.int64),
            counts=np.array([], dtype=np.int32),
            strands=np.array([], dtype=np.int8),
        )
        feat = extract_features(tx, graph, [tx], junc_ev)
        assert feat.is_single_exon == 1.0
        assert feat.n_junctions == 0.0


class TestFeaturesToArray:
    """Test conversion of TranscriptFeatures to numpy array."""

    def test_array_length(self) -> None:
        """Array length matches the number of feature names."""
        feat = TranscriptFeatures()
        arr = features_to_array(feat)
        assert len(arr) == len(feature_names())

    def test_array_dtype(self) -> None:
        """Array has float64 dtype."""
        feat = TranscriptFeatures(mean_coverage=10.0, n_exons=3.0)
        arr = features_to_array(feat)
        assert arr.dtype == np.float64

    def test_roundtrip_values(self) -> None:
        """Specific field values appear at the correct position in the array."""
        feat = TranscriptFeatures(mean_coverage=42.0, n_exons=5.0)
        arr = features_to_array(feat)
        names = feature_names()
        mean_cov_idx = names.index("mean_coverage")
        n_exons_idx = names.index("n_exons")
        assert arr[mean_cov_idx] == 42.0
        assert arr[n_exons_idx] == 5.0


class TestFeatureNames:
    """Test feature names list length and uniqueness."""

    def test_feature_count(self) -> None:
        """There are exactly 50 features (15+10+15+10)."""
        names = feature_names()
        assert len(names) == 50

    def test_unique_names(self) -> None:
        """All feature names are unique."""
        names = feature_names()
        assert len(set(names)) == len(names)

    def test_expected_names_present(self) -> None:
        """Key feature names are present in the list."""
        names = feature_names()
        assert "mean_coverage" in names
        assert "n_exons" in names
        assert "n_junctions" in names
        assert "is_safe_path" in names
        assert "graph_complexity" in names


# ===================================================================
# Heuristic scorer tests
# ===================================================================


class TestHeuristicScorer:
    """Test heuristic scoring produces values in [0, 1]."""

    def test_mode_reports_heuristic_fallback(self) -> None:
        """An untrained scorer reports heuristic fallback mode."""
        scorer = TranscriptScorer()
        assert scorer.mode == "heuristic_fallback"

    def test_heuristic_score_range(self) -> None:
        """Heuristic score is in [0, 1]."""
        scorer = TranscriptScorer()
        assert not scorer.is_trained

        # Create a feature vector with reasonable values
        feat = TranscriptFeatures(
            mean_coverage=50.0,
            coverage_uniformity=0.9,
            n_exons=3.0,
            total_length=1000.0,
            exon_fraction=0.3,
            relative_coverage=0.5,
            log_coverage=6.0,
            junction_coverage_ratio=0.8,
            n_junctions=2.0,
            has_weak_junction=0.0,
            is_safe_path=1.0,
            safe_path_fraction=1.0,
        )
        arr = features_to_array(feat)
        score = scorer.score(arr)
        assert 0.0 <= score <= 1.0

    def test_heuristic_zero_coverage(self) -> None:
        """Transcript with zero coverage gets a low score."""
        scorer = TranscriptScorer()
        feat = TranscriptFeatures()  # all zeros
        arr = features_to_array(feat)
        score = scorer.score(arr)
        assert 0.0 <= score <= 1.0
        # Score should be low since everything is zero
        assert score < 0.5

    def test_heuristic_batch_scoring(self) -> None:
        """Batch scoring produces one score per sample."""
        scorer = TranscriptScorer()
        n_features = scorer.n_features
        X = np.random.default_rng(42).uniform(0, 10, size=(5, n_features))
        scores = scorer.score_batch(X)
        assert scores.shape == (5,)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


# ===================================================================
# Train and predict tests
# ===================================================================


class TestScorerTrainAndPredict:
    """Train on synthetic data and predict."""

    def test_train_and_predict(self) -> None:
        """Train a scorer on synthetic data and verify predictions are in [0,1]."""
        scorer = TranscriptScorer()
        n_features = scorer.n_features
        rng = np.random.default_rng(42)

        # Generate synthetic training data
        n_samples = 100
        X = rng.uniform(0, 10, size=(n_samples, n_features))
        # Labels: class 1 for "good" features (high coverage uniformity)
        names = feature_names()
        cov_uniform_idx = names.index("coverage_uniformity")
        y = (X[:, cov_uniform_idx] > 5.0).astype(np.int64)

        # Ensure both classes are present
        y[0] = 0
        y[1] = 1

        scorer.train(X, y, n_estimators=10, max_depth=5)
        assert scorer.is_trained
        assert scorer.mode == "trained_model"

        # Predict on new data
        X_test = rng.uniform(0, 10, size=(10, n_features))
        scores = scorer.score_batch(X_test)
        assert scores.shape == (10,)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_single_score_after_training(self) -> None:
        """score() works for a single sample after training."""
        scorer = TranscriptScorer()
        n_features = scorer.n_features
        rng = np.random.default_rng(123)

        n_samples = 50
        X = rng.uniform(0, 10, size=(n_samples, n_features))
        y = rng.integers(0, 2, size=n_samples).astype(np.int64)
        y[0] = 0
        y[1] = 1

        scorer.train(X, y, n_estimators=10, max_depth=5)

        single_features = rng.uniform(0, 10, size=(n_features,))
        score = scorer.score(single_features)
        assert 0.0 <= score <= 1.0

    def test_feature_importances(self) -> None:
        """feature_importances() returns a dict with all feature names after training."""
        scorer = TranscriptScorer()
        n_features = scorer.n_features
        rng = np.random.default_rng(77)

        X = rng.uniform(0, 10, size=(50, n_features))
        y = rng.integers(0, 2, size=50).astype(np.int64)
        y[0] = 0
        y[1] = 1

        scorer.train(X, y, n_estimators=10, max_depth=5)
        importances = scorer.feature_importances()
        assert len(importances) == n_features
        assert all(v >= 0.0 for v in importances.values())


# ===================================================================
# Filter tests
# ===================================================================


def _make_scored_transcripts(
    n: int = 5,
) -> tuple[list[Transcript], np.ndarray, list[TranscriptFeatures]]:
    """Create n scored transcripts with varying quality for filter testing.

    Returns:
        (transcripts, scores, features_list)
    """
    transcripts: list[Transcript] = []
    features_list: list[TranscriptFeatures] = []
    scores_list: list[float] = []

    for i in range(n):
        weight = float(10 - i)  # decreasing weight
        tx = Transcript(
            node_ids=[0, 1, 2],
            exon_coords=[(100 + i * 10, 200 + i * 10), (300 + i * 10, 500 + i * 10)],
            weight=weight,
            is_safe=(i < 2),
        )
        transcripts.append(tx)

        feat = TranscriptFeatures(
            mean_coverage=weight * 2,
            n_exons=2.0,
            total_length=float((200 + i * 10 - 100 - i * 10) + (500 + i * 10 - 300 - i * 10)),
            n_junctions=1.0,
            min_junction_support=weight,
        )
        features_list.append(feat)
        scores_list.append(0.8 - i * 0.15)  # decreasing scores: 0.8, 0.65, 0.5, 0.35, 0.2

    scores = np.array(scores_list, dtype=np.float64)
    return transcripts, scores, features_list


class TestFilterByScore:
    """Test score-based filtering."""

    def test_filter_removes_low_scores(self) -> None:
        """Transcripts with scores below threshold are removed."""
        transcripts, scores, features = _make_scored_transcripts()
        filt = TranscriptFilter(config=FilterConfig(
            min_score=0.5,
            min_coverage=0.0,
            min_junction_support=0,
            min_exon_length=0,
            remove_redundant=False,
        ))
        surviving = filt.filter_transcripts(transcripts, scores, features)
        # scores: 0.8, 0.65, 0.5, 0.35, 0.2 -> indices 0, 1, 2 survive (>= 0.5)
        assert 0 in surviving
        assert 1 in surviving
        assert 2 in surviving
        assert 3 not in surviving
        assert 4 not in surviving


class TestFilterByCoverage:
    """Test coverage-based filtering."""

    def test_filter_removes_low_coverage(self) -> None:
        """Transcripts with coverage below threshold are removed."""
        transcripts, scores, features = _make_scored_transcripts()
        filt = TranscriptFilter(config=FilterConfig(
            min_score=0.0,
            min_coverage=15.0,
            min_junction_support=0,
            min_exon_length=0,
            remove_redundant=False,
        ))
        surviving = filt.filter_transcripts(transcripts, scores, features)
        # Coverages: 20, 18, 16, 14, 12 -> only 0, 1, 2 have coverage >= 15
        for idx in surviving:
            assert features[idx].mean_coverage >= 15.0

    def test_filter_diagnostics_capture_stage_counts(self) -> None:
        """Stage diagnostics reflect candidate counts after each filter."""
        transcripts, scores, features = _make_scored_transcripts()
        filt = TranscriptFilter(config=FilterConfig(
            min_score=0.5,
            min_coverage=17.0,
            min_junction_support=10,
            min_exon_length=250,
            remove_redundant=False,
            max_transcripts_per_locus=10,
        ))

        surviving, diagnostics = filt.filter_transcripts_with_diagnostics(
            transcripts, scores, features,
        )

        assert surviving == [0]
        assert diagnostics.initial == 5
        assert diagnostics.after_score == 3
        assert diagnostics.after_coverage == 2
        assert diagnostics.after_junction_support == 1
        assert diagnostics.after_length == 1
        assert diagnostics.after_redundancy == 1
        assert diagnostics.after_cap == 1


class TestRemoveRedundant:
    """Test redundancy removal."""

    def test_duplicate_intron_chain_removed(self) -> None:
        """A transcript with identical intron chain is removed (lower-scored)."""
        # Transcript A and B share the same intron chain (200, 300)
        tx_a = Transcript(
            exon_coords=[(100, 200), (300, 500)],
            weight=10.0,
        )
        tx_b = Transcript(
            exon_coords=[(100, 200), (300, 500)],
            weight=5.0,
        )
        filt = TranscriptFilter(config=FilterConfig(remove_redundant=True))
        surviving = filt.remove_redundant_transcripts([tx_a, tx_b])
        # tx_b is a duplicate of tx_a with lower weight, so it's removed
        assert 0 in surviving
        assert 1 not in surviving

    def test_different_intron_chain_kept(self) -> None:
        """Transcripts with different intron chains (e.g. exon-skipping) are kept."""
        # Transcript A: 3 exons, intron chain (200, 300), (500, 600)
        # Transcript B: 2 exons, intron chain (200, 600) -- exon-skipping isoform
        tx_a = Transcript(
            exon_coords=[(100, 200), (300, 500), (600, 800)],
            weight=10.0,
        )
        tx_b = Transcript(
            exon_coords=[(100, 200), (600, 800)],
            weight=5.0,
        )
        filt = TranscriptFilter(config=FilterConfig(remove_redundant=True))
        surviving = filt.remove_redundant_transcripts([tx_a, tx_b])
        # Both should be kept — different intron chains = different isoforms
        assert len(surviving) == 2

    def test_non_subset_kept(self) -> None:
        """Non-subset transcripts are both kept."""
        tx_a = Transcript(
            exon_coords=[(100, 200), (300, 500)],
            weight=10.0,
        )
        tx_b = Transcript(
            exon_coords=[(100, 200), (600, 700)],
            weight=5.0,
        )
        filt = TranscriptFilter(config=FilterConfig(remove_redundant=True))
        surviving = filt.remove_redundant_transcripts([tx_a, tx_b])
        assert len(surviving) == 2


class TestMergeIdenticalChains:
    """Test merging identical intron chains."""

    def test_merge_same_chain(self) -> None:
        """Two transcripts with the same intron chain are merged."""
        tx_a = Transcript(
            exon_coords=[(100, 200), (300, 500)],
            weight=6.0,
            is_safe=True,
        )
        tx_b = Transcript(
            exon_coords=[(90, 200), (300, 520)],
            weight=4.0,
            is_safe=False,
        )
        filt = TranscriptFilter(config=FilterConfig(merge_similar=True))
        merged = filt.merge_identical_intron_chains([tx_a, tx_b])
        # Both have the same intron chain: (200, 300)
        assert len(merged) == 1
        assert merged[0].weight == pytest.approx(10.0)
        # First exon start should be min(100, 90) = 90
        assert merged[0].exon_coords[0][0] == 90
        # Last exon end should be max(500, 520) = 520
        assert merged[0].exon_coords[-1][1] == 520

    def test_different_chains_not_merged(self) -> None:
        """Transcripts with different intron chains are not merged."""
        tx_a = Transcript(
            exon_coords=[(100, 200), (300, 500)],
            weight=6.0,
        )
        tx_b = Transcript(
            exon_coords=[(100, 250), (400, 500)],
            weight=4.0,
        )
        filt = TranscriptFilter(config=FilterConfig(merge_similar=True))
        merged = filt.merge_identical_intron_chains([tx_a, tx_b])
        # Intron chains differ: (200, 300) vs (250, 400)
        assert len(merged) == 2

    def test_single_exon_not_merged(self) -> None:
        """Single-exon transcripts are not merged even if they overlap."""
        tx_a = Transcript(exon_coords=[(100, 500)], weight=5.0)
        tx_b = Transcript(exon_coords=[(100, 500)], weight=3.0)
        filt = TranscriptFilter(config=FilterConfig(merge_similar=True))
        merged = filt.merge_identical_intron_chains([tx_a, tx_b])
        # Single-exon transcripts have no intron chain (empty tuple), kept as-is
        assert len(merged) == 2


class TestFilterConfigDefaults:
    """Test default config values."""

    def test_default_values(self) -> None:
        """Default FilterConfig has expected values."""
        cfg = FilterConfig()
        assert cfg.min_score == 0.3
        assert cfg.min_coverage == 1.0
        assert cfg.min_junction_support == 2
        assert cfg.min_exon_length == 50
        assert cfg.max_transcripts_per_locus == 30
        assert cfg.remove_redundant is True
        assert cfg.merge_similar is True

    def test_custom_values(self) -> None:
        """Custom FilterConfig stores provided values."""
        cfg = FilterConfig(
            min_score=0.5,
            min_coverage=3.0,
            min_junction_support=5,
            min_exon_length=100,
            max_transcripts_per_locus=10,
            remove_redundant=False,
            merge_similar=False,
        )
        assert cfg.min_score == 0.5
        assert cfg.min_coverage == 3.0
        assert cfg.min_junction_support == 5
        assert cfg.max_transcripts_per_locus == 10
        assert cfg.remove_redundant is False
        assert cfg.merge_similar is False
