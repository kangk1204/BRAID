"""Tests for the decomposer interface abstraction layer."""

from __future__ import annotations

import pytest

from rapidsplice.flow.decompose import DecomposeConfig
from rapidsplice.flow.decomposer import (
    IterativeV2Decomposer,
    LegacyPathNNLSDecomposer,
    resolve_decomposer,
    run_decomposer_pair,
)
from rapidsplice.graph.splice_graph import CSRGraph, SpliceGraph


def test_resolve_decomposer_modes() -> None:
    """Both supported mode identifiers resolve to concrete implementations."""
    assert isinstance(resolve_decomposer("legacy"), LegacyPathNNLSDecomposer)
    assert isinstance(resolve_decomposer("iterative_v2"), IterativeV2Decomposer)


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
