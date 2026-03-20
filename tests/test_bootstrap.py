"""Tests for bootstrap confidence interval estimation."""

from __future__ import annotations

import numpy as np
import pytest

from rapidsplice.flow.bootstrap import (
    BootstrapConfig,
    BootstrapResult,
    TranscriptConfidence,
    _resample_edge_weights,
    bootstrap_confidence,
    format_confidence_gtf_attributes,
)
from rapidsplice.graph.splice_graph import CSRGraph


def _make_simple_graph() -> tuple[CSRGraph, list[list[int]]]:
    """Create a simple linear graph with 2 alternative paths.

    Graph:  SOURCE -> exon1 -> exon2 -> SINK   (path A)
            SOURCE -> exon1 -> exon3 -> SINK   (path B)

    4 nodes: 0=SOURCE, 1=exon1, 2=exon2, 3=exon3, 4=SINK
    """
    n_nodes = 5
    # Edges: 0->1, 1->2, 1->3, 2->4, 3->4
    row_offsets = np.array([0, 1, 3, 4, 5, 5], dtype=np.int64)
    col_indices = np.array([1, 2, 3, 4, 4], dtype=np.int64)
    edge_weights = np.array([100.0, 70.0, 30.0, 70.0, 30.0], dtype=np.float64)
    edge_coverages = np.array([100.0, 70.0, 30.0, 70.0, 30.0], dtype=np.float64)

    from rapidsplice.graph.splice_graph import NodeType
    node_types = np.array([
        NodeType.SOURCE, NodeType.EXON, NodeType.EXON,
        NodeType.EXON, NodeType.SINK,
    ], dtype=np.int8)
    node_starts = np.array([0, 100, 300, 500, 700], dtype=np.int64)
    node_ends = np.array([0, 200, 400, 600, 700], dtype=np.int64)

    node_coverages = np.array([0.0, 100.0, 70.0, 30.0, 0.0], dtype=np.float64)

    graph = CSRGraph(
        n_nodes=n_nodes,
        n_edges=5,
        row_offsets=row_offsets,
        col_indices=col_indices,
        edge_weights=edge_weights,
        edge_coverages=edge_coverages,
        node_coverages=node_coverages,
        node_types=node_types,
        node_starts=node_starts,
        node_ends=node_ends,
    )

    paths = [
        [0, 1, 2, 4],  # path A (weight ~70)
        [0, 1, 3, 4],  # path B (weight ~30)
    ]
    return graph, paths


class TestBootstrapConfidence:
    """Tests for the bootstrap_confidence function."""

    def test_basic_two_paths(self) -> None:
        """Bootstrap should recover approximate weights for two clear paths."""
        graph, paths = _make_simple_graph()
        config = BootstrapConfig(n_replicates=50, seed=42)
        result = bootstrap_confidence(graph, paths, config)

        assert isinstance(result, BootstrapResult)
        assert len(result.transcripts) == 2
        assert result.n_replicates == 50

        # Path A (weight ~70) should have higher mean than path B (~30)
        tc_a, tc_b = result.transcripts
        assert tc_a.weight_mean > tc_b.weight_mean

        # Both should have high presence rates (strong signal)
        assert tc_a.presence_rate > 0.8
        assert tc_b.presence_rate > 0.5

    def test_confidence_interval_contains_true_value(self) -> None:
        """95% CI should contain the true weight most of the time."""
        graph, paths = _make_simple_graph()
        config = BootstrapConfig(
            n_replicates=200, confidence_level=0.95, seed=123
        )
        result = bootstrap_confidence(graph, paths, config)

        tc_a = result.transcripts[0]
        # True weight is ~70; CI should contain it
        assert tc_a.weight_ci_low < 70.0 < tc_a.weight_ci_high

    def test_weight_matrix_shape(self) -> None:
        """Weight matrix should have shape (n_replicates, n_paths)."""
        graph, paths = _make_simple_graph()
        config = BootstrapConfig(n_replicates=30, seed=42)
        result = bootstrap_confidence(graph, paths, config)

        assert result.weight_matrix.shape == (30, 2)

    def test_empty_paths(self) -> None:
        """Should handle empty path list gracefully."""
        graph, _ = _make_simple_graph()
        result = bootstrap_confidence(graph, [], BootstrapConfig(n_replicates=10))

        assert len(result.transcripts) == 0
        assert result.n_stable == 0

    def test_single_path(self) -> None:
        """Single path should have very high presence rate."""
        graph, paths = _make_simple_graph()
        config = BootstrapConfig(n_replicates=50, seed=42)
        result = bootstrap_confidence(graph, paths[:1], config)

        assert len(result.transcripts) == 1
        assert result.transcripts[0].presence_rate > 0.9

    def test_deterministic_with_seed(self) -> None:
        """Results should be reproducible with same seed."""
        graph, paths = _make_simple_graph()
        config1 = BootstrapConfig(n_replicates=50, seed=42)
        config2 = BootstrapConfig(n_replicates=50, seed=42)

        r1 = bootstrap_confidence(graph, paths, config1)
        r2 = bootstrap_confidence(graph, paths, config2)

        np.testing.assert_array_equal(r1.weight_matrix, r2.weight_matrix)

    def test_n_stable_count(self) -> None:
        """n_stable should count transcripts above min_presence_rate."""
        graph, paths = _make_simple_graph()
        config = BootstrapConfig(
            n_replicates=50, min_presence_rate=0.9, seed=42
        )
        result = bootstrap_confidence(graph, paths, config)

        manual_count = sum(
            1 for tc in result.transcripts
            if tc.presence_rate >= 0.9
        )
        assert result.n_stable == manual_count

    def test_multinomial_mode(self) -> None:
        """Multinomial resampling should also work."""
        graph, paths = _make_simple_graph()
        config = BootstrapConfig(
            n_replicates=30, resample_mode="multinomial", seed=42
        )
        result = bootstrap_confidence(graph, paths, config)
        assert len(result.transcripts) == 2
        assert result.transcripts[0].weight_mean > 0

    def test_cv_finite_for_present_transcripts(self) -> None:
        """CV should be finite for transcripts with positive mean weight."""
        graph, paths = _make_simple_graph()
        config = BootstrapConfig(n_replicates=50, seed=42)
        result = bootstrap_confidence(graph, paths, config)

        for tc in result.transcripts:
            if tc.weight_mean > 0:
                assert np.isfinite(tc.cv)


class TestResampleEdgeWeights:
    """Tests for the _resample_edge_weights helper."""

    def test_poisson_preserves_mean(self) -> None:
        """Poisson resampling should approximately preserve mean counts."""
        rng = np.random.default_rng(42)
        original = np.array([100.0, 50.0, 200.0])
        is_junction = np.array([True, True, True])

        means = []
        for _ in range(1000):
            resampled = _resample_edge_weights(original, is_junction, rng, "poisson")
            means.append(resampled)
        mean_arr = np.mean(means, axis=0)

        # Mean should be close to original (within 5%)
        np.testing.assert_allclose(mean_arr, original, rtol=0.05)

    def test_resampled_nonnegative(self) -> None:
        """Resampled weights should always be non-negative."""
        rng = np.random.default_rng(42)
        original = np.array([10.0, 5.0, 1.0, 50.0])
        is_junction = np.array([True, True, True, False])

        for _ in range(100):
            resampled = _resample_edge_weights(original, is_junction, rng, "poisson")
            assert np.all(resampled > 0)

    def test_invalid_mode_raises(self) -> None:
        """Invalid resample mode should raise ValueError."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Unknown resample_mode"):
            _resample_edge_weights(
                np.array([1.0]), np.array([True]), rng, "invalid"
            )


class TestFormatGTFAttributes:
    """Tests for GTF attribute formatting."""

    def test_format_output(self) -> None:
        """Should produce valid GTF attribute string."""
        tc = TranscriptConfidence(
            path_index=0,
            weight_mean=50.0,
            weight_median=48.0,
            weight_ci_low=30.5,
            weight_ci_high=72.1,
            presence_rate=0.95,
            cv=0.23,
            weights=np.array([50.0]),
        )
        result = format_confidence_gtf_attributes(tc)
        assert 'bootstrap_ci_low "30.50"' in result
        assert 'bootstrap_ci_high "72.10"' in result
        assert 'bootstrap_presence "0.950"' in result
        assert 'bootstrap_cv "0.230"' in result
