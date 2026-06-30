"""Decomposer interface layer for transcript assembly.

This module provides a thin abstraction in front of concrete graph
decomposition implementations. It exists to support shadow execution and
mode-specific diagnostics while keeping the legacy path-NNLS code path intact.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from braid.flow.decompose import (
    DecomposeConfig,
    Transcript,
    decompose_graph_with_metrics,
    solve_nnls_regularized,
)
from braid.graph.splice_graph import CSRGraph, NodeType, SpliceGraph

DecomposerMode = Literal["legacy", "iterative_v2", "sota", "braid_v2"]


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
    validated phasing constraints and long-read guide paths as hard seeds,
    then repeatedly extracts the widest residual source-to-sink path and
    subtracts its bottleneck support.
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
            guide_paths=guide_paths,
        )
        metadata = DecomposerMetadata(
            requested_mode=self.mode,
            implementation_mode="iterative_v2_residual",
            delegated_mode=None,
            is_shadow=is_shadow,
            metrics=metrics,
        )
        return DecomposerRun(transcripts=transcripts, metadata=metadata)


class BraidV2Decomposer:
    """Beam-search candidate generation plus sparse global fitting.

    This engine replaces exhaustive source-to-sink enumeration with a bounded
    candidate search. Candidate weights are then fit globally and pruned with
    a complexity-aware sparse re-fit so the final basis stays explainable.
    """

    mode: DecomposerMode = "braid_v2"

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

        transcripts, metrics = _decompose_braid_v2(
            graph_csr,
            config=config,
            phasing_paths=phasing_paths,
            guide_paths=guide_paths,
        )
        metadata = DecomposerMetadata(
            requested_mode=self.mode,
            implementation_mode="braid_v2_sparse_global",
            delegated_mode=None,
            is_shadow=is_shadow,
            metrics=metrics,
        )
        return DecomposerRun(transcripts=transcripts, metadata=metadata)


class SotaHybridDecomposer:
    """Backward-compatible alias for the current flagship engine."""

    mode: DecomposerMode = "sota"

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

        backend = BraidV2Decomposer()
        run = backend.decompose(
            graph_csr,
            graph,
            config=config,
            phasing_paths=phasing_paths,
            guide_paths=guide_paths,
            is_shadow=is_shadow,
        )
        metadata = DecomposerMetadata(
            requested_mode=self.mode,
            implementation_mode=f"sota_router->{run.metadata.implementation_mode}",
            delegated_mode="braid_v2",
            is_shadow=is_shadow,
            metrics=run.metadata.metrics,
        )
        return DecomposerRun(transcripts=run.transcripts, metadata=metadata)


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


def _csr_topological_order(graph_csr: CSRGraph) -> list[int]:
    """Compute a topological order directly from a CSR DAG."""
    n_nodes = graph_csr.n_nodes
    if n_nodes == 0:
        return []

    in_degree = np.zeros(n_nodes, dtype=np.int32)
    for u in range(n_nodes):
        start = int(graph_csr.row_offsets[u])
        end = int(graph_csr.row_offsets[u + 1])
        for idx in range(start, end):
            v = int(graph_csr.col_indices[idx])
            in_degree[v] += 1

    queue: deque[int] = deque(int(nid) for nid in np.nonzero(in_degree == 0)[0])
    order: list[int] = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        start = int(graph_csr.row_offsets[nid])
        end = int(graph_csr.row_offsets[nid + 1])
        for idx in range(start, end):
            v = int(graph_csr.col_indices[idx])
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    if len(order) != n_nodes:
        raise RuntimeError(
            "CSR graph is not acyclic; widest-path decomposition requires a DAG.",
        )
    return order


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
    if source < 0 or target < 0 or source >= graph_csr.n_nodes or target >= graph_csr.n_nodes:
        return None
    if source == target:
        return [source], float("inf"), 0.0

    topo_order = _csr_topological_order(graph_csr)
    topo_index = np.empty(graph_csr.n_nodes, dtype=np.int32)
    for pos, node_id in enumerate(topo_order):
        topo_index[node_id] = pos

    source_pos = int(topo_index[source])
    target_pos = int(topo_index[target])
    if source_pos > target_pos:
        return None

    best_bottleneck = np.zeros(graph_csr.n_nodes, dtype=np.float64)
    best_score = np.full(graph_csr.n_nodes, -np.inf, dtype=np.float64)
    predecessor = np.full(graph_csr.n_nodes, -1, dtype=np.int32)
    best_bottleneck[source] = np.inf
    best_score[source] = 0.0

    for u in topo_order[source_pos : target_pos + 1]:
        if best_bottleneck[u] <= 0.0:
            continue
        start = int(graph_csr.row_offsets[u])
        end = int(graph_csr.row_offsets[u + 1])
        for edge_idx in range(start, end):
            v = int(graph_csr.col_indices[edge_idx])
            if topo_index[v] > target_pos:
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
    guide_paths: list[list[int]] | None = None,
) -> tuple[list[Transcript], dict[str, float | int]]:
    """Extract transcripts by iteratively peeling highest-support paths."""
    if graph_csr.n_nodes < 2 or graph_csr.n_edges == 0:
        return [], {
            "accepted_paths": 0,
            "iterations": 0,
            "phasing_constraints": 0,
            "phasing_matched": 0,
            "phasing_match_rate": 0.0,
            "guide_paths_constraints": 0,
            "guide_paths_matched": 0,
            "guide_paths_match_rate": 0.0,
            "residual_edge_fraction": 0.0,
        }

    edge_lookup = _build_edge_lookup(graph_csr)
    residual_edges = graph_csr.edge_weights.astype(np.float64).copy()
    total_edge_support = float(np.sum(residual_edges))
    accepted: list[Transcript] = []
    accepted_keys: set[tuple[int, ...]] = set()
    iterations = 0
    phasing_total = len(phasing_paths or [])
    phasing_matched = 0
    guide_total = len(guide_paths or [])
    guide_matched = 0
    source = 0
    sink = graph_csr.n_nodes - 1
    max_rounds = max(8, config.max_transcripts_per_locus * 4)

    def _accept_seed_path(path: list[int], abundance: float) -> bool:
        key = tuple(path)
        if key in accepted_keys:
            return False

        edge_indices = _path_edge_indices(path, edge_lookup)
        if not edge_indices or abundance < config.min_transcript_coverage:
            return False

        transcript = _transcript_from_path(graph_csr, path, abundance)
        if transcript is None:
            return False

        accepted.append(transcript)
        accepted_keys.add(key)
        _subtract_path_residual(residual_edges, edge_indices, abundance)
        return True

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
        if not _accept_seed_path(path, bottleneck):
            continue
        phasing_matched += 1
        iterations += 1

    sorted_guides = list(guide_paths or [])
    for guide_nodes in sorted_guides:
        if len(accepted) >= config.max_transcripts_per_locus:
            break
        if not guide_nodes or guide_nodes[0] != source or guide_nodes[-1] != sink:
            continue

        edge_indices = _path_edge_indices(guide_nodes, edge_lookup)
        if not edge_indices:
            continue
        if any(float(residual_edges[idx]) <= 0.0 for idx in edge_indices):
            continue

        guide_support = min(float(residual_edges[idx]) for idx in edge_indices)
        if _accept_seed_path(guide_nodes, guide_support):
            guide_matched += 1
            iterations += 1

    while len(accepted) < config.max_transcripts_per_locus and iterations < max_rounds:
        candidate = _widest_path_between(graph_csr, residual_edges, source, sink)
        if candidate is None:
            break
        path, bottleneck, _score = candidate
        if tuple(path) in accepted_keys:
            break
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
        "guide_paths_constraints": guide_total,
        "guide_paths_matched": guide_matched,
        "guide_paths_match_rate": (
            float(guide_matched / guide_total) if guide_total > 0 else 0.0
        ),
        "residual_edge_fraction": residual_fraction,
    }
    return transcripts, metrics


def _contains_ordered_subsequence(path: list[int], subseq: list[int]) -> bool:
    """Return True when *subseq* appears in *path* preserving order."""
    if not subseq:
        return True
    idx = 0
    for node in path:
        if node == subseq[idx]:
            idx += 1
            if idx == len(subseq):
                return True
    return False


def _guide_prefix_hits(path: list[int], guide_paths: list[list[int]] | None) -> int:
    """Count guide paths that agree with the current path prefix."""
    if not guide_paths:
        return 0
    hits = 0
    for guide in guide_paths:
        if len(path) > len(guide):
            continue
        if all(a == b for a, b in zip(path, guide)):
            hits += 1
    return hits


def _path_support_estimate(
    graph_csr: CSRGraph,
    path: list[int],
    edge_lookup: dict[tuple[int, int], int],
) -> float:
    """Estimate candidate abundance from the minimum edge support on the path."""
    edge_indices = _path_edge_indices(path, edge_lookup)
    if not edge_indices:
        return 0.0
    return float(min(float(graph_csr.edge_weights[idx]) for idx in edge_indices))


def _enumerate_braid_v2_candidates(
    graph_csr: CSRGraph,
    config: DecomposeConfig,
    phasing_paths: list[tuple[list[int], float]] | None,
    guide_paths: list[list[int]] | None,
) -> tuple[list[list[int]], dict[str, float | int], set[int], set[int]]:
    """Enumerate a bounded candidate set using beam search plus hard seeds."""
    import heapq

    if graph_csr.n_nodes < 2 or graph_csr.n_edges == 0:
        return [], {
            "candidate_budget": int(config.candidate_budget),
            "candidate_budget_hit": 0,
            "candidate_frontier_pruned": 0,
            "candidate_states_expanded": 0,
            "candidate_seed_guides": 0,
            "candidate_seed_phasing": 0,
        }, set(), set()

    source = 0
    sink = graph_csr.n_nodes - 1
    edge_lookup = _build_edge_lookup(graph_csr)
    residual_edges = graph_csr.edge_weights.astype(np.float64)
    candidates: list[list[int]] = []
    candidate_keys: set[tuple[int, ...]] = set()
    guide_candidate_indices: set[int] = set()
    phasing_candidate_indices: set[int] = set()

    def _add_candidate(path: list[int]) -> int | None:
        key = tuple(path)
        if not path or path[0] != source or path[-1] != sink:
            return None
        if key in candidate_keys:
            return None
        if not _path_edge_indices(path, edge_lookup):
            return None
        candidate_keys.add(key)
        candidates.append(list(path))
        return len(candidates) - 1

    for phase_nodes, _phase_weight in sorted(
        phasing_paths or [],
        key=lambda item: (-float(item[1]), len(item[0])),
    ):
        candidate = _build_phasing_seed_path(
            graph_csr,
            residual_edges,
            edge_lookup,
            phase_nodes,
        )
        if candidate is None:
            continue
        idx = _add_candidate(candidate[0])
        if idx is not None:
            phasing_candidate_indices.add(idx)
        if len(candidates) >= config.candidate_budget:
            break

    for guide_path in guide_paths or []:
        idx = _add_candidate(guide_path)
        if idx is not None:
            guide_candidate_indices.add(idx)
        if len(candidates) >= config.candidate_budget:
            break

    beam_width = max(4, int(config.candidate_beam_width))
    frontier: list[tuple[float, float, list[int]]] = [(0.0, 0.0, [source])]
    node_score_buckets: list[list[float]] = [[] for _ in range(graph_csr.n_nodes)]
    node_score_buckets[source].append(0.0)
    frontier_pruned = 0
    states_expanded = 0

    while frontier and len(candidates) < config.candidate_budget:
        neg_score, _neg_bottleneck, path = heapq.heappop(frontier)
        score = -neg_score
        node = path[-1]
        if node == sink:
            _add_candidate(path)
            continue

        states_expanded += 1
        start = int(graph_csr.row_offsets[node])
        end = int(graph_csr.row_offsets[node + 1])
        visited = set(path)
        for idx in range(start, end):
            nxt = int(graph_csr.col_indices[idx])
            if nxt in visited:
                continue
            edge_weight = float(graph_csr.edge_weights[idx])
            if edge_weight <= 0.0:
                continue

            next_path = path + [nxt]
            next_score = score + float(np.log1p(edge_weight))
            if int(graph_csr.node_types[nxt]) == int(NodeType.EXON):
                next_score += 0.05 * float(graph_csr.node_coverages[nxt])
            guide_hits = _guide_prefix_hits(next_path, guide_paths)
            if guide_hits:
                next_score += 0.35 * guide_hits

            bucket = node_score_buckets[nxt]
            if len(bucket) < beam_width:
                bucket.append(next_score)
            else:
                min_idx = min(range(len(bucket)), key=bucket.__getitem__)
                if next_score <= bucket[min_idx] + 1e-9:
                    frontier_pruned += 1
                    continue
                bucket[min_idx] = next_score

            next_bottleneck = min(
                edge_weight,
                -_neg_bottleneck if _neg_bottleneck < 0 else edge_weight,
            )
            heapq.heappush(frontier, (-next_score, -next_bottleneck, next_path))

    metrics: dict[str, float | int] = {
        "candidate_budget": int(config.candidate_budget),
        "candidate_budget_hit": int(len(candidates) >= config.candidate_budget),
        "candidate_frontier_pruned": frontier_pruned,
        "candidate_states_expanded": states_expanded,
        "candidate_seed_guides": len(guide_candidate_indices),
        "candidate_seed_phasing": len(phasing_candidate_indices),
    }
    return candidates, metrics, guide_candidate_indices, phasing_candidate_indices


def _solve_candidate_weights(
    graph_csr: CSRGraph,
    paths: list[list[int]],
    *,
    phasing_paths: list[tuple[list[int], float]] | None,
    guide_candidate_indices: set[int],
    protected_candidate_indices: set[int],
    config: DecomposeConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit candidate weights with junction-weighted NNLS and soft priors."""
    n_edges = graph_csr.n_edges
    n_paths = len(paths)
    if n_paths == 0 or n_edges == 0:
        return np.zeros(n_paths, dtype=np.float64), {
            "fit_loss_total": 0.0,
            "fit_loss_edge": 0.0,
            "fit_loss_phasing": 0.0,
            "fit_loss_guide": 0.0,
            "candidate_condition_number": 0.0,
            "phasing_constraints": len(phasing_paths or []),
            "phasing_matched": 0,
            "guide_constraints": len(guide_candidate_indices),
            "guide_matched": 0,
        }

    edge_lookup = _build_edge_lookup(graph_csr)
    A = np.zeros((n_edges, n_paths), dtype=np.float64)
    path_support = np.zeros(n_paths, dtype=np.float64)
    for pi, path in enumerate(paths):
        edge_indices = _path_edge_indices(path, edge_lookup)
        for edge_idx in edge_indices:
            A[edge_idx, pi] = 1.0
        path_support[pi] = _path_support_estimate(graph_csr, path, edge_lookup)

    edge_type_weights = np.ones(n_edges, dtype=np.float64)
    if hasattr(graph_csr, "edge_coverages") and graph_csr.edge_coverages is not None:
        for eidx in range(n_edges):
            ew = float(graph_csr.edge_weights[eidx])
            ec = float(graph_csr.edge_coverages[eidx])
            if ew > 0 and abs(ew - ec) < 0.01:
                edge_type_weights[eidx] = config.junction_weight

    b = graph_csr.edge_weights.astype(np.float64)
    A_edge = A * edge_type_weights[:, np.newaxis]
    b_edge = b * edge_type_weights
    rows = [A_edge]
    targets = [b_edge]

    phasing_matched = 0
    A_phasing = np.empty((0, n_paths), dtype=np.float64)
    b_phasing = np.empty(0, dtype=np.float64)
    if phasing_paths:
        A_phasing = np.zeros((len(phasing_paths), n_paths), dtype=np.float64)
        b_phasing = np.zeros(len(phasing_paths), dtype=np.float64)
        for ri, (phase_nodes, phase_count) in enumerate(phasing_paths):
            matched = False
            for pi, path in enumerate(paths):
                if _contains_ordered_subsequence(path, phase_nodes):
                    A_phasing[ri, pi] = 0.75
                    protected_candidate_indices.add(pi)
                    matched = True
            if matched:
                phasing_matched += 1
            b_phasing[ri] = float(phase_count) * 0.75
        rows.append(A_phasing)
        targets.append(b_phasing)

    guide_matched = 0
    A_guide = np.empty((0, n_paths), dtype=np.float64)
    b_guide = np.empty(0, dtype=np.float64)
    if guide_candidate_indices:
        guide_rows = sorted(guide_candidate_indices)
        A_guide = np.zeros((len(guide_rows), n_paths), dtype=np.float64)
        b_guide = np.zeros(len(guide_rows), dtype=np.float64)
        for ri, pi in enumerate(guide_rows):
            A_guide[ri, pi] = 0.35
            b_guide[ri] = max(path_support[pi], config.min_transcript_coverage) * 0.35
            guide_matched += 1
        rows.append(A_guide)
        targets.append(b_guide)

    A_weighted = np.vstack(rows)
    b_weighted = np.concatenate(targets)
    weights, condition_number = solve_nnls_regularized(A_weighted, b_weighted)

    fit_total = A_weighted @ weights - b_weighted
    fit_edge = A_edge @ weights - b_edge
    fit_phasing = A_phasing @ weights - b_phasing
    fit_guide = A_guide @ weights - b_guide
    metrics: dict[str, float | int] = {
        "fit_loss_total": float(np.linalg.norm(fit_total)),
        "fit_loss_edge": float(np.linalg.norm(fit_edge)),
        "fit_loss_phasing": float(np.linalg.norm(fit_phasing)),
        "fit_loss_guide": float(np.linalg.norm(fit_guide)),
        "candidate_condition_number": condition_number,
        "phasing_constraints": len(phasing_paths or []),
        "phasing_matched": phasing_matched,
        "guide_constraints": len(guide_candidate_indices),
        "guide_matched": guide_matched,
    }
    return weights, metrics


def _decompose_braid_v2(
    graph_csr: CSRGraph,
    config: DecomposeConfig,
    phasing_paths: list[tuple[list[int], float]] | None = None,
    guide_paths: list[list[int]] | None = None,
) -> tuple[list[Transcript], dict[str, float | int]]:
    """Run braid_v2 candidate search, sparse fitting, and transcript selection."""
    candidates, search_metrics, guide_candidate_indices, phasing_candidate_indices = (
        _enumerate_braid_v2_candidates(
            graph_csr,
            config,
            phasing_paths,
            guide_paths,
        )
    )
    if not candidates:
        return [], {
            **search_metrics,
            "candidate_count_before_prune": 0,
            "candidate_count_after_prune": 0,
            "candidate_pruned": 0,
            "accepted_paths": 0,
            "merged_transcripts": 0,
        }

    active_indices = np.arange(len(candidates), dtype=np.int32)
    protected_candidate_indices = set(guide_candidate_indices) | set(phasing_candidate_indices)
    fit_metrics: dict[str, float | int] = {}
    weights = np.zeros(len(candidates), dtype=np.float64)
    max_refits = 4
    for _ in range(max_refits):
        active_paths = [candidates[int(idx)] for idx in active_indices]
        local_guides = {
            pos for pos, idx in enumerate(active_indices)
            if int(idx) in guide_candidate_indices
        }
        local_protected = {
            pos for pos, idx in enumerate(active_indices)
            if int(idx) in protected_candidate_indices
        }
        weights_local, fit_metrics = _solve_candidate_weights(
            graph_csr,
            active_paths,
            phasing_paths=phasing_paths,
            guide_candidate_indices=local_guides,
            protected_candidate_indices=local_protected,
            config=config,
        )
        weights = np.zeros(len(candidates), dtype=np.float64)
        weights[active_indices] = weights_local

        if len(active_indices) <= 4 or len(weights_local) == 0:
            break
        max_weight = float(np.max(weights_local))
        prune_threshold = max(
            config.min_transcript_coverage * 0.5,
            max_weight * config.complexity_penalty,
        )
        keep_local = [
            pos
            for pos, idx in enumerate(active_indices)
            if (
                weights_local[pos] >= prune_threshold
                or int(idx) in protected_candidate_indices
            )
        ]
        if not keep_local or len(keep_local) == len(active_indices):
            break
        active_indices = active_indices[np.array(keep_local, dtype=np.int32)]

    weighted_candidates: list[tuple[list[int], float]] = []
    max_weight = float(np.max(weights)) if len(weights) else 0.0
    relative_threshold = max(
        max_weight * config.min_relative_isoform_weight,
        0.1,
    ) if max_weight > 0 else config.min_transcript_coverage
    use_relative = max_weight >= config.min_transcript_coverage
    for path, weight in zip(candidates, weights):
        if weight <= 0:
            continue
        passes_absolute = weight >= config.min_transcript_coverage
        passes_relative = use_relative and weight >= relative_threshold
        if passes_absolute or passes_relative:
            weighted_candidates.append((path, float(weight)))

    transcripts: list[Transcript] = []
    for path, weight in weighted_candidates:
        transcript = _transcript_from_path(graph_csr, path, weight)
        if transcript is not None:
            transcripts.append(transcript)
    transcripts = _merge_transcripts(transcripts)
    transcripts.sort(key=lambda tx: tx.weight, reverse=True)
    if transcripts and config.min_relative_abundance > 0:
        abundance_floor = transcripts[0].weight * config.min_relative_abundance
        transcripts = [tx for tx in transcripts if tx.weight >= abundance_floor]
    if len(transcripts) > config.max_transcripts_per_locus:
        transcripts = transcripts[: config.max_transcripts_per_locus]

    metrics: dict[str, float | int] = {
        **search_metrics,
        **fit_metrics,
        "candidate_count_before_prune": len(candidates),
        "candidate_count_after_prune": len(active_indices),
        "candidate_pruned": len(candidates) - len(active_indices),
        "accepted_paths": len(weighted_candidates),
        "merged_transcripts": len(transcripts),
    }
    return transcripts, metrics


def resolve_decomposer(mode: str) -> GraphDecomposer:
    """Resolve a user-facing mode identifier to a decomposer instance."""
    if mode == "legacy":
        return LegacyPathNNLSDecomposer()
    if mode == "iterative_v2":
        return IterativeV2Decomposer()
    if mode == "braid_v2":
        return BraidV2Decomposer()
    if mode == "sota":
        return SotaHybridDecomposer()
    raise ValueError(f"Unsupported decomposer mode: {mode}")


def run_decomposer_pair(
    graph_csr: CSRGraph,
    graph: SpliceGraph,
    *,
    config: DecomposeConfig | None = None,
    phasing_paths: list[tuple[list[int], float]] | None = None,
    guide_paths: list[list[int]] | None = None,
    mode: DecomposerMode = "braid_v2",
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
    "BraidV2Decomposer",
    "IterativeV2Decomposer",
    "LegacyPathNNLSDecomposer",
    "SotaHybridDecomposer",
    "resolve_decomposer",
    "run_decomposer_pair",
]
