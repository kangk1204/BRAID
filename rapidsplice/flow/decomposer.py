"""Decomposer interface layer for transcript assembly.

This module provides a thin abstraction in front of concrete graph
decomposition implementations. It exists to support shadow execution and
mode-specific diagnostics while keeping the legacy path-NNLS code path intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from rapidsplice.flow.decompose import (
    DecomposeConfig,
    Transcript,
    decompose_graph_with_metrics,
)
from rapidsplice.graph.splice_graph import CSRGraph, NodeType, SpliceGraph

DecomposerMode = Literal["legacy", "iterative_v2"]


@dataclass(slots=True)
class DecomposerMetadata:
    """Execution metadata for one decomposer run."""

    requested_mode: DecomposerMode
    implementation_mode: str
    delegated_mode: str | None = None
    is_shadow: bool = False
    metrics: dict[str, float | int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Human-readable mode label for diagnostics."""
        if self.delegated_mode is not None:
            return f"{self.requested_mode}->{self.delegated_mode}"
        return self.requested_mode


@dataclass(slots=True)
class DecomposerRun:
    """Result bundle for one decomposer execution."""

    transcripts: list[Transcript]
    metadata: DecomposerMetadata


class GraphDecomposer(Protocol):
    """Common interface for graph decomposers."""

    mode: DecomposerMode

    def decompose(
        self,
        graph_csr: CSRGraph,
        graph: SpliceGraph,
        config: DecomposeConfig | None = None,
        phasing_paths: list[tuple[list[int], float]] | None = None,
        guide_paths: list[list[int]] | None = None,
        *,
        is_shadow: bool = False,
    ) -> DecomposerRun:
        """Decompose a graph into transcript candidates."""


class LegacyPathNNLSDecomposer:
    """Adapter over the existing path enumeration + NNLS implementation."""

    mode: DecomposerMode = "legacy"

    def decompose(
        self,
        graph_csr: CSRGraph,
        graph: SpliceGraph,
        config: DecomposeConfig | None = None,
        phasing_paths: list[tuple[list[int], float]] | None = None,
        guide_paths: list[list[int]] | None = None,
        *,
        is_shadow: bool = False,
    ) -> DecomposerRun:
        transcripts, metrics = decompose_graph_with_metrics(
            graph_csr,
            graph,
            config=config,
            phasing_paths=phasing_paths,
            guide_paths=guide_paths,
        )
        metadata = DecomposerMetadata(
            requested_mode=self.mode,
            implementation_mode="legacy_path_nnls",
            delegated_mode=None,
            is_shadow=is_shadow,
            metrics=metrics,
        )
        return DecomposerRun(transcripts=transcripts, metadata=metadata)


class IterativeV2Decomposer:
    """Residual-based greedy transcript extractor.

    This implementation avoids all-path enumeration. It first consumes
    validated phasing constraints as hard seeds, then repeatedly extracts the
    widest residual source-to-sink path and subtracts its bottleneck support.
    """

    mode: DecomposerMode = "iterative_v2"

    def decompose(
        self,
        graph_csr: CSRGraph,
        graph: SpliceGraph,
        config: DecomposeConfig | None = None,
        phasing_paths: list[tuple[list[int], float]] | None = None,
        guide_paths: list[list[int]] | None = None,
        *,
        is_shadow: bool = False,
    ) -> DecomposerRun:
        if config is None:
            config = DecomposeConfig()

        transcripts, metrics = _decompose_iterative_residual(
            graph_csr,
            config=config,
            phasing_paths=phasing_paths,
        )
        metadata = DecomposerMetadata(
            requested_mode=self.mode,
            implementation_mode="iterative_v2_residual",
            delegated_mode=None,
            is_shadow=is_shadow,
            metrics=metrics,
        )
        return DecomposerRun(transcripts=transcripts, metadata=metadata)


def _merge_adjacent_exons(
    exon_coords: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge abutting exon segments into transcript exon intervals."""
    if len(exon_coords) <= 1:
        return exon_coords

    merged = [exon_coords[0]]
    for start, end in exon_coords[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _build_edge_lookup(graph_csr: CSRGraph) -> dict[tuple[int, int], int]:
    """Map a CSR edge pair to its flat edge index."""
    edge_lookup: dict[tuple[int, int], int] = {}
    for u in range(graph_csr.n_nodes):
        start = int(graph_csr.row_offsets[u])
        end = int(graph_csr.row_offsets[u + 1])
        for idx in range(start, end):
            v = int(graph_csr.col_indices[idx])
            edge_lookup[(u, v)] = idx
    return edge_lookup


def _path_edge_indices(
    path: list[int],
    edge_lookup: dict[tuple[int, int], int],
) -> list[int]:
    """Return flat edge indices for a path, or an empty list if invalid."""
    indices: list[int] = []
    for i in range(len(path) - 1):
        edge_idx = edge_lookup.get((path[i], path[i + 1]))
        if edge_idx is None:
            return []
        indices.append(edge_idx)
    return indices


def _sum_path_support(
    path: list[int],
    edge_indices: list[int],
    graph_csr: CSRGraph,
    residual_edges: np.ndarray,
) -> float:
    """Score a path using residual edge support plus exon-node coverage."""
    score = float(np.sum(residual_edges[edge_indices])) if edge_indices else 0.0
    for node_id in path:
        if int(graph_csr.node_types[node_id]) == int(NodeType.EXON):
            score += 0.1 * float(graph_csr.node_coverages[node_id])
    return score


def _widest_path_between(
    graph_csr: CSRGraph,
    residual_edges: np.ndarray,
    source: int,
    target: int,
) -> tuple[list[int], float, float] | None:
    """Find the best residual path between two nodes in the DAG."""
    if source == target:
        return [source], float("inf"), 0.0
    if source < 0 or target >= graph_csr.n_nodes or source > target:
        return None

    best_bottleneck = np.zeros(graph_csr.n_nodes, dtype=np.float64)
    best_score = np.full(graph_csr.n_nodes, -np.inf, dtype=np.float64)
    predecessor = np.full(graph_csr.n_nodes, -1, dtype=np.int32)
    best_bottleneck[source] = np.inf
    best_score[source] = 0.0

    for u in range(source, target + 1):
        if best_bottleneck[u] <= 0.0:
            continue
        start = int(graph_csr.row_offsets[u])
        end = int(graph_csr.row_offsets[u + 1])
        for edge_idx in range(start, end):
            v = int(graph_csr.col_indices[edge_idx])
            if v > target:
                continue
            residual = float(residual_edges[edge_idx])
            if residual <= 0.0:
                continue
            cand_bottleneck = min(best_bottleneck[u], residual)
            cand_score = best_score[u] + residual
            if int(graph_csr.node_types[v]) == int(NodeType.EXON):
                cand_score += 0.1 * float(graph_csr.node_coverages[v])

            if (
                cand_bottleneck > best_bottleneck[v] + 1e-9
                or (
                    abs(cand_bottleneck - best_bottleneck[v]) <= 1e-9
                    and cand_score > best_score[v] + 1e-9
                )
            ):
                best_bottleneck[v] = cand_bottleneck
                best_score[v] = cand_score
                predecessor[v] = u

    if best_bottleneck[target] <= 0.0:
        return None

    path: list[int] = []
    cur = target
    while cur != -1:
        path.append(int(cur))
        if cur == source:
            break
        cur = int(predecessor[cur])

    if not path or path[-1] != source:
        return None

    path.reverse()
    return path, float(best_bottleneck[target]), float(best_score[target])


def _build_phasing_seed_path(
    graph_csr: CSRGraph,
    residual_edges: np.ndarray,
    edge_lookup: dict[tuple[int, int], int],
    required_subpath: list[int],
) -> tuple[list[int], float, float] | None:
    """Extend a validated phasing path to a full source-to-sink path."""
    if not required_subpath:
        return None

    source = 0
    sink = graph_csr.n_nodes - 1
    start_node = required_subpath[0]
    end_node = required_subpath[-1]

    internal_edges = _path_edge_indices(required_subpath, edge_lookup)
    if len(required_subpath) > 1 and not internal_edges:
        return None
    if any(float(residual_edges[idx]) <= 0.0 for idx in internal_edges):
        return None

    prefix = _widest_path_between(graph_csr, residual_edges, source, start_node)
    suffix = _widest_path_between(graph_csr, residual_edges, end_node, sink)
    if prefix is None or suffix is None:
        return None

    prefix_path, prefix_bottleneck, prefix_score = prefix
    suffix_path, suffix_bottleneck, suffix_score = suffix
    combined = prefix_path[:-1] + required_subpath + suffix_path[1:]
    edge_indices = _path_edge_indices(combined, edge_lookup)
    if not edge_indices:
        return None

    bottleneck_terms = [prefix_bottleneck, suffix_bottleneck]
    if internal_edges:
        bottleneck_terms.append(
            min(float(residual_edges[idx]) for idx in internal_edges),
        )
    finite_terms = [term for term in bottleneck_terms if term != float("inf")]
    bottleneck = min(finite_terms) if finite_terms else float("inf")
    score = prefix_score + suffix_score + _sum_path_support(
        required_subpath,
        internal_edges,
        graph_csr,
        residual_edges,
    )
    return combined, float(bottleneck), float(score)


def _transcript_from_path(
    graph_csr: CSRGraph,
    path: list[int],
    abundance: float,
) -> Transcript | None:
    """Convert a source-to-sink node path into a transcript."""
    exon_coords: list[tuple[int, int]] = []
    for node_id in path:
        if int(graph_csr.node_types[node_id]) != int(NodeType.EXON):
            continue
        exon_coords.append(
            (int(graph_csr.node_starts[node_id]), int(graph_csr.node_ends[node_id])),
        )
    exon_coords = _merge_adjacent_exons(exon_coords)
    if not exon_coords:
        return None
    return Transcript(
        node_ids=list(path),
        exon_coords=exon_coords,
        weight=float(abundance),
        is_safe=False,
    )


def _merge_transcripts(transcripts: list[Transcript]) -> list[Transcript]:
    """Merge repeated extractions of the same transcript structure."""
    merged: dict[tuple[tuple[int, int], ...], Transcript] = {}
    for tx in transcripts:
        key = tuple(tx.exon_coords)
        if key in merged:
            prev = merged[key]
            merged[key] = Transcript(
                node_ids=prev.node_ids,
                exon_coords=prev.exon_coords,
                weight=prev.weight + tx.weight,
                is_safe=prev.is_safe and tx.is_safe,
            )
        else:
            merged[key] = Transcript(
                node_ids=list(tx.node_ids),
                exon_coords=list(tx.exon_coords),
                weight=tx.weight,
                is_safe=tx.is_safe,
            )
    return list(merged.values())


def _subtract_path_residual(
    residual_edges: np.ndarray,
    edge_indices: list[int],
    amount: float,
) -> None:
    """Subtract transcript abundance from residual edge support."""
    for idx in edge_indices:
        residual_edges[idx] = max(0.0, float(residual_edges[idx]) - amount)


def _decompose_iterative_residual(
    graph_csr: CSRGraph,
    config: DecomposeConfig,
    phasing_paths: list[tuple[list[int], float]] | None = None,
) -> tuple[list[Transcript], dict[str, float | int]]:
    """Extract transcripts by iteratively peeling highest-support paths."""
    if graph_csr.n_nodes < 2 or graph_csr.n_edges == 0:
        return [], {
            "accepted_paths": 0,
            "iterations": 0,
            "phasing_constraints": 0,
            "phasing_matched": 0,
            "phasing_match_rate": 0.0,
            "residual_edge_fraction": 0.0,
        }

    edge_lookup = _build_edge_lookup(graph_csr)
    residual_edges = graph_csr.edge_weights.astype(np.float64).copy()
    total_edge_support = float(np.sum(residual_edges))
    accepted: list[Transcript] = []
    iterations = 0
    phasing_total = len(phasing_paths or [])
    phasing_matched = 0
    max_rounds = max(8, config.max_transcripts_per_locus * 4)

    sorted_phasing = sorted(
        phasing_paths or [],
        key=lambda item: (-float(item[1]), len(item[0])),
    )
    for phase_nodes, _phase_weight in sorted_phasing:
        if len(accepted) >= config.max_transcripts_per_locus:
            break
        candidate = _build_phasing_seed_path(
            graph_csr,
            residual_edges,
            edge_lookup,
            phase_nodes,
        )
        if candidate is None:
            continue
        path, bottleneck, _score = candidate
        edge_indices = _path_edge_indices(path, edge_lookup)
        if not edge_indices or bottleneck < config.min_transcript_coverage:
            continue
        transcript = _transcript_from_path(graph_csr, path, bottleneck)
        if transcript is None:
            continue
        accepted.append(transcript)
        phasing_matched += 1
        iterations += 1
        _subtract_path_residual(residual_edges, edge_indices, bottleneck)

    source = 0
    sink = graph_csr.n_nodes - 1
    while len(accepted) < config.max_transcripts_per_locus and iterations < max_rounds:
        candidate = _widest_path_between(graph_csr, residual_edges, source, sink)
        if candidate is None:
            break
        path, bottleneck, _score = candidate
        edge_indices = _path_edge_indices(path, edge_lookup)
        if not edge_indices or bottleneck < config.min_transcript_coverage:
            break
        transcript = _transcript_from_path(graph_csr, path, bottleneck)
        if transcript is None:
            break
        accepted.append(transcript)
        iterations += 1
        _subtract_path_residual(residual_edges, edge_indices, bottleneck)

    transcripts = _merge_transcripts(accepted)
    transcripts.sort(key=lambda tx: tx.weight, reverse=True)

    if transcripts and config.min_relative_abundance > 0:
        max_weight = transcripts[0].weight
        threshold = max_weight * config.min_relative_abundance
        transcripts = [tx for tx in transcripts if tx.weight >= threshold]

    if len(transcripts) > config.max_transcripts_per_locus:
        transcripts = transcripts[: config.max_transcripts_per_locus]

    residual_fraction = 0.0
    if total_edge_support > 0:
        residual_fraction = float(np.sum(residual_edges) / total_edge_support)

    metrics: dict[str, float | int] = {
        "accepted_paths": len(accepted),
        "merged_transcripts": len(transcripts),
        "iterations": iterations,
        "phasing_constraints": phasing_total,
        "phasing_matched": phasing_matched,
        "phasing_match_rate": (
            float(phasing_matched / phasing_total) if phasing_total > 0 else 0.0
        ),
        "residual_edge_fraction": residual_fraction,
    }
    return transcripts, metrics


def resolve_decomposer(mode: str) -> GraphDecomposer:
    """Resolve a user-facing mode identifier to a decomposer instance."""
    if mode == "legacy":
        return LegacyPathNNLSDecomposer()
    if mode == "iterative_v2":
        return IterativeV2Decomposer()
    raise ValueError(f"Unsupported decomposer mode: {mode}")


def run_decomposer_pair(
    graph_csr: CSRGraph,
    graph: SpliceGraph,
    *,
    config: DecomposeConfig | None = None,
    phasing_paths: list[tuple[list[int], float]] | None = None,
    guide_paths: list[list[int]] | None = None,
    mode: DecomposerMode = "legacy",
    shadow_mode: DecomposerMode | None = None,
) -> tuple[DecomposerRun, DecomposerRun | None]:
    """Run the requested decomposer and optional shadow decomposer."""
    primary = resolve_decomposer(mode).decompose(
        graph_csr,
        graph,
        config=config,
        phasing_paths=phasing_paths,
        guide_paths=guide_paths,
        is_shadow=False,
    )

    if shadow_mode is None or shadow_mode == mode:
        return primary, None

    shadow = resolve_decomposer(shadow_mode).decompose(
        graph_csr,
        graph,
        config=config,
        phasing_paths=phasing_paths,
        guide_paths=guide_paths,
        is_shadow=True,
    )
    return primary, shadow


__all__ = [
    "DecomposerMetadata",
    "DecomposerMode",
    "DecomposerRun",
    "GraphDecomposer",
    "IterativeV2Decomposer",
    "LegacyPathNNLSDecomposer",
    "resolve_decomposer",
    "run_decomposer_pair",
]
