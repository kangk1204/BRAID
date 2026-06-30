"""Tests for graph-flow PSI (braid/flow/graph_psi.py) — self-contained."""
from __future__ import annotations

import pytest

from braid.flow.graph_psi import (
    exon_inclusion_psi,
    is_multi_context_exon,
)
from braid.graph.splice_graph import EdgeType, NodeType, SpliceGraph


def _multi_context_locus(t1, t2, t3, t4):
    """E2 included via E3 (t1) and E4 (t2), skipped via E3 (t3) and E4 (t4)."""
    g = SpliceGraph(chrom="chr1", strand="+", locus_start=100, locus_end=900)
    s = g.add_node(start=100, end=100, node_type=NodeType.SOURCE, coverage=0)
    e1 = g.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=t1 + t2 + t3 + t4)
    e2 = g.add_node(start=300, end=400, node_type=NodeType.EXON, coverage=t1 + t2)
    e3 = g.add_node(start=500, end=600, node_type=NodeType.EXON, coverage=t1 + t3)
    e4 = g.add_node(start=700, end=800, node_type=NodeType.EXON, coverage=t2 + t4)
    snk = g.add_node(start=900, end=900, node_type=NodeType.SINK, coverage=0)
    g.add_edge(s, e1, EdgeType.SOURCE_LINK, weight=t1 + t2 + t3 + t4, coverage=t1 + t2 + t3 + t4)
    g.add_edge(e1, e2, EdgeType.INTRON, weight=t1 + t2, coverage=t1 + t2)
    g.add_edge(e1, e3, EdgeType.INTRON, weight=t3, coverage=t3)
    g.add_edge(e1, e4, EdgeType.INTRON, weight=t4, coverage=t4)
    g.add_edge(e2, e3, EdgeType.INTRON, weight=t1, coverage=t1)
    g.add_edge(e2, e4, EdgeType.INTRON, weight=t2, coverage=t2)
    g.add_edge(e3, snk, EdgeType.SINK_LINK, weight=t1 + t3, coverage=t1 + t3)
    g.add_edge(e4, snk, EdgeType.SINK_LINK, weight=t2 + t4, coverage=t2 + t4)
    return g.to_csr(), e2


def _simple_se_locus(inc, skip):
    g = SpliceGraph(chrom="chr1", strand="+", locus_start=100, locus_end=600)
    s = g.add_node(start=100, end=100, node_type=NodeType.SOURCE, coverage=0)
    e1 = g.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=inc + skip)
    e2 = g.add_node(start=250, end=300, node_type=NodeType.EXON, coverage=inc)
    e3 = g.add_node(start=400, end=500, node_type=NodeType.EXON, coverage=inc + skip)
    snk = g.add_node(start=600, end=600, node_type=NodeType.SINK, coverage=0)
    g.add_edge(s, e1, EdgeType.SOURCE_LINK, weight=inc + skip, coverage=inc + skip)
    g.add_edge(e1, e2, EdgeType.INTRON, weight=inc, coverage=inc)
    g.add_edge(e2, e3, EdgeType.INTRON, weight=inc, coverage=inc)
    g.add_edge(e1, e3, EdgeType.INTRON, weight=skip, coverage=skip)
    g.add_edge(e3, snk, EdgeType.SINK_LINK, weight=inc + skip, coverage=inc + skip)
    return g.to_csr(), e2


def test_flow_psi_recovers_truth_on_multi_context_exon() -> None:
    csr, e2 = _multi_context_locus(10.0, 50.0, 10.0, 10.0)  # true PSI = 60/80 = 0.75
    assert exon_inclusion_psi(csr, e2) == pytest.approx(0.75, abs=0.03)


def test_multi_context_detection() -> None:
    csr, e2 = _multi_context_locus(10.0, 50.0, 10.0, 10.0)
    assert is_multi_context_exon(csr, e2) is True


def test_simple_se_equals_junction_ratio_and_not_flagged() -> None:
    csr, e2 = _simple_se_locus(70.0, 30.0)  # junction PSI = 0.70
    assert exon_inclusion_psi(csr, e2) == pytest.approx(0.70, abs=0.03)
    assert is_multi_context_exon(csr, e2) is False  # simple SE: junction ratio is exact


def test_no_flow_returns_nan() -> None:
    import math
    csr, e2 = _simple_se_locus(0.0, 0.0)
    assert math.isnan(exon_inclusion_psi(csr, e2))
