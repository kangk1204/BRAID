"""Graph-Guided NMF Hybrid transcript decomposition.

Combines NMF-based model selection with graph-topology-aware path
decomposition. Pure NMF fails because individual reads cover only 1-2
fragments, so NMF cannot link exons across introns without splice-spanning
reads (~2.6% of all reads). This hybrid solves that:

Algorithm:

1. **Fragment definition**: Extract exon-fragment intervals from splice
   graph nodes (same as pure NMF).
2. **Read-fragment matrix**: Build binary matrix X (reads x fragments).
3. **NMF for K estimation**: Factor X ≈ W x H on a subsampled read set,
   use BIC to determine the optimal number of isoforms K.
4. **Graph path enumeration**: Enumerate all source-to-sink paths in the
   splice graph DAG (respects graph topology).
5. **Path-NNLS with K constraint**: Fit path weights via NNLS on edge
   coverage, then retain only top-K paths by weight.

This approach:
- Uses NMF's statistical model selection to determine K (avoids over/under
  fitting the number of isoforms)
- Uses graph topology for path structure (no fragment-linking problem)
- Uses NNLS for weight estimation (same as path-NNLS decompose.py)
- Falls back to pure path-NNLS when NMF K-estimation is not needed

Reference: Lee & Seung (1999) "Learning the parts of objects by NMF"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import nnls

from rapidsplice.flow.decompose import (
    Transcript,
    _enumerate_all_paths,
    _fit_path_weights_lp,
    _merge_compatible_transcripts,
)
from rapidsplice.graph.splice_graph import CSRGraph, NodeType, SpliceGraph

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class NMFDecomposeConfig:
    """Configuration for Graph-Guided NMF hybrid decomposition.

    Attributes:
        max_isoforms: Maximum K for NMF model selection.
        min_transcript_coverage: Minimum weight to keep a transcript.
        min_relative_abundance: Minimum fraction of max weight.
        nmf_max_iter: Maximum NMF iterations.
        nmf_tol: Convergence tolerance for NMF.
        fragment_threshold: Threshold for binarising H matrix rows.
        fallback_to_nnls: Use path-NNLS for simple loci (<=3 exons).
        max_paths: Maximum number of graph paths to enumerate.
    """

    max_isoforms: int = 20
    min_transcript_coverage: float = 1.0
    min_relative_abundance: float = 0.02
    nmf_max_iter: int = 200
    nmf_tol: float = 1e-4
    fragment_threshold: float = 0.3
    fallback_to_nnls: bool = True
    max_paths: int = 500


def _build_fragments(
    graph: SpliceGraph,
    graph_csr: CSRGraph,
) -> list[tuple[int, int]]:
    """Build exon-fragment intervals from splice graph nodes.

    Fragments are the exon nodes of the splice graph, which are already
    maximal non-overlapping intervals between consecutive splice sites.

    Args:
        graph: The splice graph.
        graph_csr: CSR representation.

    Returns:
        Sorted list of (start, end) genomic intervals for each fragment.
    """
    fragments: list[tuple[int, int]] = []
    for nid in range(graph_csr.n_nodes):
        if int(graph_csr.node_types[nid]) == int(NodeType.EXON):
            s = int(graph_csr.node_starts[nid])
            e = int(graph_csr.node_ends[nid])
            if e > s:
                fragments.append((s, e))
    return sorted(set(fragments))


def _build_read_fragment_matrix(
    read_positions: np.ndarray,
    read_ends: np.ndarray,
    read_junction_starts: list[list[int]] | None,
    read_junction_ends: list[list[int]] | None,
    fragments: list[tuple[int, int]],
) -> np.ndarray:
    """Build a read-fragment coverage matrix.

    For each read, determine which fragments it covers. A read covers a
    fragment if it overlaps the fragment by at least 1bp.  For spliced
    reads, only fragments on the same side of each junction are counted
    (the read "skips" fragments in the intron).

    Args:
        read_positions: Start positions of reads (n_reads,).
        read_ends: End positions of reads (n_reads,).
        read_junction_starts: Per-read list of junction start positions.
        read_junction_ends: Per-read list of junction end positions.
        fragments: Sorted list of (start, end) fragment intervals.

    Returns:
        Binary matrix X of shape (n_reads, n_fragments).
    """
    n_reads = len(read_positions)
    n_frags = len(fragments)

    if n_reads == 0 or n_frags == 0:
        return np.zeros((max(n_reads, 1), max(n_frags, 1)), dtype=np.float32)

    frag_starts = np.array([f[0] for f in fragments], dtype=np.int64)
    frag_ends = np.array([f[1] for f in fragments], dtype=np.int64)
    rpos = read_positions.astype(np.int64)
    rend = read_ends.astype(np.int64)

    # Vectorised overlap in chunks to limit memory
    chunk_size = 10000
    X = np.zeros((n_reads, n_frags), dtype=np.float32)
    for c in range(0, n_reads, chunk_size):
        ce = min(c + chunk_size, n_reads)
        X[c:ce] = (
            (rpos[c:ce, None] < frag_ends[None, :])
            & (rend[c:ce, None] > frag_starts[None, :])
        ).astype(np.float32)

    # For spliced reads: zero out fragments within introns
    if read_junction_starts is not None:
        for i in range(min(n_reads, len(read_junction_starts))):
            for js, je in zip(read_junction_starts[i], read_junction_ends[i]):
                intron_mask = (frag_starts >= js) & (frag_ends <= je)
                X[i, intron_mask] = 0.0

    return X


def _nmf_multiplicative(
    X: np.ndarray,
    k: int,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Non-negative Matrix Factorization using multiplicative update rules.

    Factorises X ≈ W @ H where X (m x n), W (m x k), H (k x n), all
    non-negative.  Uses the Lee & Seung (1999) multiplicative update rules
    with Frobenius norm objective.

    Args:
        X: Input matrix (m x n), non-negative.
        k: Number of components.
        max_iter: Maximum iterations.
        tol: Convergence tolerance (relative change in Frobenius norm).

    Returns:
        (W, H, reconstruction_error) tuple.
    """
    m, n = X.shape
    eps = 1e-10

    rng = np.random.RandomState(42)
    W = rng.rand(m, k).astype(np.float64) + eps
    H = rng.rand(k, n).astype(np.float64) + eps

    prev_err = float("inf")

    for iteration in range(max_iter):
        WtX = W.T @ X
        WtWH = W.T @ W @ H + eps
        H *= WtX / WtWH

        XHt = X @ H.T
        WHHt = W @ H @ H.T + eps
        W *= XHt / WHHt

        if iteration % 10 == 0:
            err = float(np.linalg.norm(X - W @ H, "fro"))
            if abs(prev_err - err) / (prev_err + eps) < tol:
                break
            prev_err = err

    err = float(np.linalg.norm(X - W @ H, "fro"))
    return W, H, err


def _select_k(
    X: np.ndarray,
    max_k: int,
    max_iter: int = 100,
) -> int:
    """Select optimal K (number of isoforms) using BIC-like criterion.

    Tries K = 1, 2, ..., max_k and picks the K that minimises:
        BIC = m*n*log(MSE) + k*(m+n)*log(m*n)

    Args:
        X: Read-fragment matrix.
        max_k: Maximum K to try.
        max_iter: Max NMF iterations per trial.

    Returns:
        Optimal K.
    """
    m, n = X.shape
    mn = m * n
    if mn == 0:
        return 1

    best_k = 1
    best_bic = float("inf")
    log_mn = np.log(mn + 1)

    for k in range(1, min(max_k + 1, min(m, n) + 1)):
        _, _, err = _nmf_multiplicative(X, k, max_iter=max_iter)
        mse = (err**2) / mn + 1e-10
        n_params = k * (m + n)
        bic = mn * np.log(mse) + n_params * log_mn

        if bic < best_bic:
            best_bic = bic
            best_k = k

        # Early stop: if BIC increases for 2 consecutive K values
        if k > best_k + 2:
            break

    return best_k


def _extract_isoforms_from_H(
    H: np.ndarray,
    fragments: list[tuple[int, int]],
    threshold: float = 0.3,
) -> list[list[tuple[int, int]]]:
    """Extract isoform exon structures from the H matrix.

    Each row of H represents an isoform's usage of fragments.
    Binarise by thresholding and map back to genomic coordinates.
    Adjacent fragments are merged into exons.

    Args:
        H: NMF basis matrix (K x n_fragments).
        fragments: Fragment genomic intervals.
        threshold: Fraction of row-max for binarisation.

    Returns:
        List of isoform exon coordinate lists.
    """
    k, n_frags = H.shape
    isoforms: list[list[tuple[int, int]]] = []

    for i in range(k):
        row = H[i]
        row_max = np.max(row)
        if row_max < 1e-10:
            continue

        used = row > (threshold * row_max)

        exon_coords: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int | None = None

        for j in range(n_frags):
            if not used[j]:
                if current_start is not None:
                    exon_coords.append((current_start, current_end))
                    current_start = None
                    current_end = None
                continue

            fstart, fend = fragments[j]
            if current_start is None:
                current_start = fstart
                current_end = fend
            elif fstart <= current_end + 1:
                current_end = max(current_end, fend)
            else:
                exon_coords.append((current_start, current_end))
                current_start = fstart
                current_end = fend

        if current_start is not None:
            exon_coords.append((current_start, current_end))

        if exon_coords:
            isoforms.append(exon_coords)

    return isoforms


def _estimate_isoform_weights(
    X: np.ndarray,
    isoforms: list[list[tuple[int, int]]],
    fragments: list[tuple[int, int]],
) -> np.ndarray:
    """Estimate isoform abundances using NNLS on fragment coverage.

    Builds an isoform-fragment incidence matrix and solves NNLS to fit
    the observed per-fragment coverage.

    Args:
        X: Read-fragment matrix (n_reads x n_fragments).
        isoforms: List of isoform exon coordinate lists.
        fragments: Fragment intervals.

    Returns:
        Array of isoform weights.
    """
    n_frags = len(fragments)
    n_iso = len(isoforms)

    if n_iso == 0 or n_frags == 0:
        return np.zeros(n_iso, dtype=np.float64)

    frag_cov = np.sum(X, axis=0).astype(np.float64)

    frag_starts = np.array([f[0] for f in fragments])
    frag_ends = np.array([f[1] for f in fragments])

    A = np.zeros((n_frags, n_iso), dtype=np.float64)
    for iso_idx, exons in enumerate(isoforms):
        for es, ee in exons:
            overlap = (frag_starts < ee) & (frag_ends > es)
            A[overlap, iso_idx] = 1.0

    weights, _ = nnls(A, frag_cov)
    return weights


def _path_to_exon_coords(
    path: list[int],
    graph_csr: CSRGraph,
) -> list[tuple[int, int]]:
    """Convert a node-ID path to exon genomic coordinates.

    Merges adjacent sub-exon segments from the segment graph back into
    contiguous exon intervals.

    Args:
        path: Sequence of node IDs.
        graph_csr: CSR graph with node metadata.

    Returns:
        List of (start, end) exon coordinate tuples.
    """
    exon_coords: list[tuple[int, int]] = []
    for nid in path:
        if int(graph_csr.node_types[nid]) == int(NodeType.EXON):
            exon_coords.append(
                (int(graph_csr.node_starts[nid]), int(graph_csr.node_ends[nid]))
            )
    # Merge adjacent sub-exon segments.
    if len(exon_coords) > 1:
        merged: list[tuple[int, int]] = [exon_coords[0]]
        for start, end in exon_coords[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        exon_coords = merged
    return exon_coords


def decompose_nmf(
    graph_csr: CSRGraph,
    graph: SpliceGraph,
    read_positions: np.ndarray,
    read_ends: np.ndarray,
    read_junction_starts: list[list[int]] | None = None,
    read_junction_ends: list[list[int]] | None = None,
    config: NMFDecomposeConfig | None = None,
) -> list[Transcript]:
    """Decompose a splice graph into transcripts using Graph-Guided NMF.

    Hybrid algorithm:
    1. Build read-fragment matrix from reads and graph fragments.
    2. Use NMF + BIC to estimate optimal K (number of isoforms).
    3. Enumerate all source-to-sink paths in the splice graph.
    4. Fit path weights via NNLS on edge coverage.
    5. Retain top-K paths by weight (K from step 2).

    This avoids the pure NMF problem of not linking fragments across
    introns (only ~2.6% of reads are splice-spanning) while using NMF's
    statistical model selection for determining isoform count.

    Args:
        graph_csr: Immutable CSR graph.
        graph: Mutable splice graph (for metadata).
        read_positions: Read start positions overlapping this locus.
        read_ends: Read end positions.
        read_junction_starts: Per-read junction start positions (optional).
        read_junction_ends: Per-read junction end positions (optional).
        config: NMF decomposition config.

    Returns:
        Sorted list of Transcript objects.
    """
    if config is None:
        config = NMFDecomposeConfig()

    n_nodes = graph_csr.n_nodes
    n_edges = graph_csr.n_edges

    if n_nodes < 2 or n_edges == 0:
        return []

    # ---- Step 1: Enumerate all graph paths (topology-aware) ----
    all_paths = _enumerate_all_paths(graph_csr, max_paths=config.max_paths)
    if not all_paths:
        return []

    n_paths = len(all_paths)

    # ---- Step 2: Fit path weights via NNLS on edge coverage ----
    weights = _fit_path_weights_lp(graph_csr, all_paths)

    # ---- Step 3: Determine K using NMF on read-fragment matrix ----
    fragments = _build_fragments(graph, graph_csr)
    n_frags = len(fragments)

    # Determine K: number of isoforms to retain
    if n_frags >= 2 and len(read_positions) > 0:
        X = _build_read_fragment_matrix(
            read_positions,
            read_ends,
            read_junction_starts,
            read_junction_ends,
            fragments,
        )
        n_reads = X.shape[0]

        if n_reads > 0:
            # Subsample for speed
            max_reads_nmf = 2000
            if n_reads > max_reads_nmf:
                rng = np.random.RandomState(42)
                idx = rng.choice(n_reads, max_reads_nmf, replace=False)
                X_nmf = X[idx]
            else:
                X_nmf = X

            max_k = min(config.max_isoforms, n_frags, X_nmf.shape[0], n_paths)
            if max_k > 1:
                k = _select_k(X_nmf, max_k, max_iter=50)
            else:
                k = 1
        else:
            k = min(n_paths, config.max_isoforms)
    else:
        # Simple locus: keep all paths above threshold
        k = min(n_paths, config.max_isoforms)

    # ---- Step 4: Select paths using soft K constraint ----
    # Keep all paths above min_transcript_coverage, then use K-guided
    # relative abundance filtering: if there are more than K paths,
    # apply a stricter relative abundance cut to reduce to ~K.
    weighted_paths: list[tuple[list[int], float]] = []
    for path, w in zip(all_paths, weights):
        if w >= config.min_transcript_coverage:
            weighted_paths.append((path, float(w)))

    # Sort by weight descending
    weighted_paths.sort(key=lambda x: -x[1])

    if not weighted_paths:
        return []

    # Soft K constraint: if we have more than K paths, increase the
    # relative abundance threshold so that only the K-th path's weight
    # ratio becomes the new cutoff.  This preserves well-supported
    # additional isoforms while removing noise.
    max_w = weighted_paths[0][1]
    if len(weighted_paths) > k and k > 0:
        # Use the weight of the K-th path as the relative threshold
        k_weight = weighted_paths[k - 1][1]
        soft_threshold = max(
            k_weight * 0.5,  # At least half the K-th path weight
            max_w * config.min_relative_abundance,
        )
    else:
        soft_threshold = max_w * config.min_relative_abundance

    weighted_paths = [
        (p, w) for p, w in weighted_paths if w >= soft_threshold
    ]

    # ---- Step 5: Build Transcript objects ----
    transcripts: list[Transcript] = []
    for path, w in weighted_paths:
        exon_coords = _path_to_exon_coords(path, graph_csr)
        if exon_coords:
            transcripts.append(
                Transcript(
                    node_ids=list(path),
                    exon_coords=exon_coords,
                    weight=w,
                    is_safe=False,
                )
            )

    # Merge compatible transcripts (same exon structure)
    transcripts = _merge_compatible_transcripts(transcripts)

    # Sort by weight descending
    transcripts.sort(key=lambda t: -t.weight)

    return transcripts


def _merge_adjacent_fragments(
    fragments: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge adjacent or overlapping fragment intervals into exons."""
    if not fragments:
        return []
    sorted_frags = sorted(fragments)
    merged: list[tuple[int, int]] = [sorted_frags[0]]
    for s, e in sorted_frags[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged
