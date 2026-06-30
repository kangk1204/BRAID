"""Tests for the decomposer interface abstraction layer."""

from __future__ import annotations

import numpy as np
import pytest

from braid.flow.decompose import DecomposeConfig
from braid.flow.decomposer import (
    BraidV2Decomposer,
    IterativeV2Decomposer,
    LegacyPathNNLSDecomposer,
    SotaHybridDecomposer,
    _enumerate_braid_v2_candidates,
    resolve_decomposer,
    run_decomposer_pair,
)
from braid.graph.splice_graph import CSRGraph, NodeType, SpliceGraph


def _build_non_topological_csr() -> CSRGraph:
    """Build a DAG whose node IDs are not in topological order."""
    row_offsets = np.array([0, 1, 2, 3, 3], dtype=np.int32)
    col_indices = np.array([2, 3, 1], dtype=np.int32)
    edge_weights = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    edge_coverages = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    node_coverages = np.array([0.0, 5.0, 5.0, 0.0], dtype=np.float32)
    node_starts = np.array([0, 300, 100, 400], dtype=np.int64)
    node_ends = np.array([0, 400, 200, 400], dtype=np.int64)
    node_types = np.array(
        [NodeType.SOURCE, NodeType.EXON, NodeType.EXON, NodeType.SINK],
        dtype=np.int8,
    )

    return CSRGraph(
        row_offsets=row_offsets,
        col_indices=col_indices,
        edge_weights=edge_weights,
        edge_coverages=edge_coverages,
        node_coverages=node_coverages,
        node_starts=node_starts,
        node_ends=node_ends,
        node_types=node_types,
        n_nodes=4,
        n_edges=3,
    )


def _build_priority_csr() -> CSRGraph:
    """Build a two-path DAG where the best path appears second in adjacency order."""
    row_offsets = np.array([0, 2, 3, 4, 4], dtype=np.int32)
    col_indices = np.array([1, 2, 3, 3], dtype=np.int32)
    edge_weights = np.array([1.0, 10.0, 1.0, 10.0], dtype=np.float32)
    edge_coverages = np.array([1.0, 10.0, 1.0, 10.0], dtype=np.float32)
    node_coverages = np.array([0.0, 1.0, 10.0, 0.0], dtype=np.float32)
    node_starts = np.array([0, 100, 200, 400], dtype=np.int64)
    node_ends = np.array([0, 200, 300, 400], dtype=np.int64)
    node_types = np.array(
        [NodeType.SOURCE, NodeType.EXON, NodeType.EXON, NodeType.SINK],
        dtype=np.int8,
    )

    return CSRGraph(
        row_offsets=row_offsets,
        col_indices=col_indices,
        edge_weights=edge_weights,
        edge_coverages=edge_coverages,
        node_coverages=node_coverages,
        node_starts=node_starts,
        node_ends=node_ends,
        node_types=node_types,
        n_nodes=4,
        n_edges=4,
    )


def test_resolve_decomposer_modes() -> None:
    """Both supported mode identifiers resolve to concrete implementations."""
    assert isinstance(resolve_decomposer("legacy"), LegacyPathNNLSDecomposer)
    assert isinstance(resolve_decomposer("iterative_v2"), IterativeV2Decomposer)
    assert isinstance(resolve_decomposer("braid_v2"), BraidV2Decomposer)
    assert isinstance(resolve_decomposer("sota"), SotaHybridDecomposer)


def test_resolve_decomposer_rejects_unknown_mode() -> None:
    """Unknown decomposer modes fail fast."""
    with pytest.raises(ValueError, match="Unsupported decomposer mode"):
        resolve_decomposer("mystery_v3")


def test_iterative_v2_is_separately_addressable(
    simple_csr_graph: CSRGraph,
    simple_splice_graph: SpliceGraph,
) -> None:
    """iterative_v2 returns separate metadata and its own metrics payload."""
    primary, shadow = run_decomposer_pair(
        simple_csr_graph,
        simple_splice_graph,
        config=DecomposeConfig(),
        mode="iterative_v2",
        shadow_mode="legacy",
    )

    assert shadow is not None
    assert primary.metadata.requested_mode == "iterative_v2"
    assert primary.metadata.delegated_mode is None
    assert primary.metadata.implementation_mode == "iterative_v2_residual"
    assert primary.metadata.label == "iterative_v2"
    assert primary.metadata.metrics["accepted_paths"] >= 1
    assert shadow.metadata.requested_mode == "legacy"
    assert shadow.metadata.is_shadow is True
    assert len(primary.transcripts) == len(shadow.transcripts)
    assert [tx.exon_coords for tx in primary.transcripts] == [
        tx.exon_coords for tx in shadow.transcripts
    ]


def test_iterative_v2_metrics_report_phasing_usage(
    simple_csr_graph: CSRGraph,
    simple_splice_graph: SpliceGraph,
) -> None:
    """iterative_v2 reports phasing match metrics when given a seed path."""
    primary, _shadow = run_decomposer_pair(
        simple_csr_graph,
        simple_splice_graph,
        config=DecomposeConfig(),
        phasing_paths=[([1, 2], 3.0)],
        mode="iterative_v2",
        shadow_mode=None,
    )

    assert primary.metadata.metrics["phasing_constraints"] == 1
    assert primary.metadata.metrics["phasing_matched"] == 1
    assert primary.metadata.metrics["phasing_match_rate"] == pytest.approx(1.0)


def test_iterative_v2_handles_non_topological_csr(
    simple_splice_graph: SpliceGraph,
) -> None:
    """iterative_v2 no longer assumes CSR node IDs are already topological."""
    csr = _build_non_topological_csr()
    decomposer = IterativeV2Decomposer()

    result = decomposer.decompose(csr, simple_splice_graph)

    assert result.metadata.metrics["accepted_paths"] == 1
    assert len(result.transcripts) == 1
    assert result.transcripts[0].node_ids == [0, 2, 1, 3]
    assert result.transcripts[0].exon_coords == [(100, 200), (300, 400)]


def test_iterative_v2_honors_guide_paths(
    simple_splice_graph: SpliceGraph,
) -> None:
    """iterative_v2 keeps long-read guide paths instead of ignoring them."""
    csr = _build_priority_csr()
    decomposer = IterativeV2Decomposer()

    result = decomposer.decompose(
        csr,
        simple_splice_graph,
        config=DecomposeConfig(max_transcripts_per_locus=1),
        guide_paths=[[0, 2, 3]],
    )

    assert result.metadata.metrics["guide_paths_constraints"] == 1
    assert result.metadata.metrics["guide_paths_matched"] == 1
    assert len(result.transcripts) == 1
    assert result.transcripts[0].node_ids == [0, 2, 3]


def test_braid_v2_keeps_simple_locus_assembly_working(
    simple_csr_graph: CSRGraph,
    simple_splice_graph: SpliceGraph,
) -> None:
    """braid_v2 should still assemble simple loci and expose sparse-fit metrics."""
    decomposer = resolve_decomposer("braid_v2")
    result = decomposer.decompose(
        simple_csr_graph,
        simple_splice_graph,
        config=DecomposeConfig(),
    )

    assert result.metadata.requested_mode == "braid_v2"
    assert result.metadata.delegated_mode is None
    assert result.metadata.implementation_mode == "braid_v2_sparse_global"
    assert result.metadata.metrics["candidate_count_before_prune"] >= 1
    assert "fit_loss_total" in result.metadata.metrics
    assert len(result.transcripts) >= 1


def test_braid_v2_honors_guided_locus(
    simple_splice_graph: SpliceGraph,
) -> None:
    """braid_v2 should keep guide-supported paths in the retained basis."""
    csr = _build_priority_csr()
    decomposer = resolve_decomposer("braid_v2")

    result = decomposer.decompose(
        csr,
        simple_splice_graph,
        config=DecomposeConfig(max_transcripts_per_locus=1),
        guide_paths=[[0, 2, 3]],
    )

    assert result.metadata.requested_mode == "braid_v2"
    assert result.metadata.metrics["guide_constraints"] >= 1
    assert len(result.transcripts) == 1
    assert result.transcripts[0].node_ids == [0, 2, 3]


def test_braid_v2_beam_keeps_late_better_candidate() -> None:
    """Per-node beam replacement should admit a late candidate above the bucket floor."""
    csr = CSRGraph(
        row_offsets=np.array([0, 5, 6, 7, 8, 9, 10, 11, 11], dtype=np.int32),
        col_indices=np.array([1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 7], dtype=np.int32),
        edge_weights=np.array(
            [1.0, 2.0, 3.0, 4.0, 100.0, 1.0, 2.0, 3.0, 4.0, 100.0, 100.0],
            dtype=np.float32,
        ),
        edge_coverages=np.array(
            [1.0, 2.0, 3.0, 4.0, 100.0, 1.0, 2.0, 3.0, 4.0, 100.0, 100.0],
            dtype=np.float32,
        ),
        node_coverages=np.array([0.0, 1.0, 2.0, 3.0, 4.0, 100.0, 10.0, 0.0]),
        node_starts=np.array([0, 100, 200, 300, 400, 500, 600, 700], dtype=np.int64),
        node_ends=np.array([0, 150, 250, 350, 450, 550, 650, 700], dtype=np.int64),
        node_types=np.array(
            [
                NodeType.SOURCE,
                NodeType.EXON,
                NodeType.EXON,
                NodeType.EXON,
                NodeType.EXON,
                NodeType.EXON,
                NodeType.EXON,
                NodeType.SINK,
            ],
            dtype=np.int8,
        ),
        n_nodes=8,
        n_edges=11,
    )

    candidates, metrics, _, _ = _enumerate_braid_v2_candidates(
        csr,
        DecomposeConfig(candidate_beam_width=4, candidate_budget=10),
        phasing_paths=None,
        guide_paths=None,
    )

    assert [0, 5, 6, 7] in candidates
    assert metrics["candidate_frontier_pruned"] >= 1


def test_sota_is_backward_compatible_alias_for_braid_v2(
    simple_csr_graph: CSRGraph,
    simple_splice_graph: SpliceGraph,
) -> None:
    """sota remains a compatibility alias but delegates to braid_v2."""
    decomposer = resolve_decomposer("sota")
    result = decomposer.decompose(simple_csr_graph, simple_splice_graph)

    assert result.metadata.requested_mode == "sota"
    assert result.metadata.delegated_mode == "braid_v2"
    assert result.metadata.implementation_mode.startswith("sota_router->braid_v2")


def test_shadow_same_as_primary_is_skipped(
    simple_csr_graph: CSRGraph,
    simple_splice_graph: SpliceGraph,
) -> None:
    """Requesting the same shadow mode does not rerun the decomposer."""
    primary, shadow = run_decomposer_pair(
        simple_csr_graph,
        simple_splice_graph,
        config=DecomposeConfig(),
        mode="legacy",
        shadow_mode="legacy",
    )

    assert primary.metadata.requested_mode == "legacy"
    assert shadow is None


def test_legacy_reports_enumeration_metrics(
    simple_csr_graph: CSRGraph,
    simple_splice_graph: SpliceGraph,
) -> None:
    """legacy mode exposes enumeration and NNLS metrics for diagnostics."""
    primary, shadow = run_decomposer_pair(
        simple_csr_graph,
        simple_splice_graph,
        config=DecomposeConfig(),
        mode="legacy",
        shadow_mode=None,
    )

    assert shadow is None
    assert primary.metadata.metrics["all_paths_total"] >= 1
    assert "max_paths_hit" in primary.metadata.metrics
    assert "nnls_residual_total" in primary.metadata.metrics
