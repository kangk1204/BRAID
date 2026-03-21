"""Bootstrap confidence intervals for transcript abundance estimates.

Provides uncertainty quantification for NNLS-based transcript assembly by
resampling junction read counts and re-solving the decomposition.  This
produces per-transcript confidence intervals and existence probabilities
that no other short-read assembler currently provides.

The key insight is that junction read counts follow a Poisson-like
distribution, so bootstrap resampling of edge weights and re-solving NNLS
gives an empirical posterior over transcript abundances.  Transcripts
that appear consistently across bootstrap replicates are high-confidence;
those that appear sporadically are uncertain.

Reference:
    Efron, B. & Tibshirani, R. (1993). An Introduction to the Bootstrap.
    Chapman & Hall/CRC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from braid.graph.splice_graph import CSRGraph

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BootstrapConfig:
    """Configuration for bootstrap confidence interval estimation.

    Attributes:
        n_replicates: Number of bootstrap resamples (default 100).
        confidence_level: Confidence level for intervals (default 0.95).
        min_presence_rate: Minimum fraction of replicates a transcript must
            appear in to be considered real (default 0.5).
        resample_mode: How to resample edge weights.
            ``"poisson"`` draws from Poisson(observed_count) for junction
            edges (most realistic).
            ``"multinomial"`` redistributes total read count across junctions.
        seed: Random seed for reproducibility (None = random).
        junction_weight: Weight multiplier for intron edges in NNLS.
    """

    n_replicates: int = 100
    confidence_level: float = 0.95
    min_presence_rate: float = 0.5
    resample_mode: str = "poisson"
    seed: int | None = None
    junction_weight: float = 2.0


@dataclass(slots=True)
class TranscriptConfidence:
    """Bootstrap confidence statistics for a single transcript.

    Attributes:
        path_index: Index of this path in the enumerated path list.
        weight_mean: Mean abundance across bootstrap replicates.
        weight_median: Median abundance across bootstrap replicates.
        weight_ci_low: Lower bound of the confidence interval.
        weight_ci_high: Upper bound of the confidence interval.
        presence_rate: Fraction of replicates where weight > 0.
        cv: Coefficient of variation (std / mean).
        weights: Full array of bootstrap weights (n_replicates,).
    """

    path_index: int
    weight_mean: float
    weight_median: float
    weight_ci_low: float
    weight_ci_high: float
    presence_rate: float
    cv: float
    weights: np.ndarray = field(repr=False)


@dataclass(slots=True)
class BootstrapResult:
    """Result of bootstrap confidence interval estimation.

    Attributes:
        transcripts: List of per-transcript confidence statistics.
        n_replicates: Number of bootstrap resamples performed.
        confidence_level: Confidence level used.
        n_stable: Number of transcripts with presence_rate >= min_presence_rate.
        weight_matrix: Full (n_replicates x n_paths) weight matrix.
    """

    transcripts: list[TranscriptConfidence]
    n_replicates: int
    confidence_level: float
    n_stable: int
    weight_matrix: np.ndarray = field(repr=False)


def _build_edge_map(graph_csr: CSRGraph) -> dict[tuple[int, int], int]:
    """Build a mapping from (src, dst) node pairs to edge indices."""
    edge_map: dict[tuple[int, int], int] = {}
    for u in range(graph_csr.n_nodes):
        start = int(graph_csr.row_offsets[u])
        end = int(graph_csr.row_offsets[u + 1])
        for idx in range(start, end):
            v = int(graph_csr.col_indices[idx])
            edge_map[(u, v)] = idx
    return edge_map


def _build_path_edge_matrix(
    graph_csr: CSRGraph,
    paths: list[list[int]],
    edge_map: dict[tuple[int, int], int],
) -> np.ndarray:
    """Build the path-edge incidence matrix A (n_edges x n_paths)."""
    n_edges = graph_csr.n_edges
    n_paths = len(paths)
    A = np.zeros((n_edges, n_paths), dtype=np.float64)
    for pi, path in enumerate(paths):
        for i in range(len(path) - 1):
            edge_idx = edge_map.get((path[i], path[i + 1]))
            if edge_idx is not None:
                A[edge_idx, pi] = 1.0
    return A


def _identify_junction_edges(graph_csr: CSRGraph) -> np.ndarray:
    """Return a boolean mask indicating which edges are junction (intron) edges."""
    n_edges = graph_csr.n_edges
    is_junction = np.zeros(n_edges, dtype=bool)
    if hasattr(graph_csr, "edge_coverages") and graph_csr.edge_coverages is not None:
        for eidx in range(n_edges):
            ew = float(graph_csr.edge_weights[eidx])
            ec = float(graph_csr.edge_coverages[eidx])
            if ew > 0 and abs(ew - ec) < 0.01:
                is_junction[eidx] = True
    return is_junction


def _try_gpu_bootstrap(
    A: np.ndarray,
    W: np.ndarray,
    original_weights: np.ndarray,
    is_junction: np.ndarray,
    config: BootstrapConfig,
    rng: np.random.Generator,
    weight_matrix: np.ndarray,
) -> bool:
    """Try GPU-batched bootstrap using PyTorch.

    Generates all Poisson resamples at once and solves batched least
    squares on GPU.  Falls back to CPU if PyTorch or CUDA unavailable.

    Returns True if GPU path succeeded, False to fall back to CPU.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
    except Exception:
        return False

    n_edges, n_paths = A.shape
    R = config.n_replicates
    device = torch.device("cuda")

    # Convert to torch tensors
    A_t = torch.from_numpy(A).float().to(device)
    W_t = torch.from_numpy(W).float().to(device)
    orig_t = torch.from_numpy(original_weights).float().to(device)
    junc_mask = torch.from_numpy(is_junction).bool().to(device)

    # Weighted A matrix
    A_w = A_t * W_t.unsqueeze(1)  # (E, K)

    # Batch Poisson resampling: (R, E)
    lambdas = orig_t.unsqueeze(0).expand(R, -1)
    junction_expanded = junc_mask.unsqueeze(0).expand(R, -1)

    # Poisson for junction edges
    resampled = orig_t.unsqueeze(0).expand(R, -1).clone()
    poisson_samples = torch.poisson(lambdas)
    resampled[junction_expanded] = poisson_samples[junction_expanded]
    resampled = torch.clamp(resampled, min=0.1)

    # Scale continuation edges
    orig_junc_total = orig_t[junc_mask].sum()
    if orig_junc_total > 0:
        resamp_junc_totals = resampled[:, junc_mask].sum(dim=1, keepdim=True)
        scale = resamp_junc_totals / orig_junc_total
        cont_mask = ~junc_mask
        resampled[:, cont_mask] = orig_t[cont_mask].unsqueeze(0) * scale

    # Weighted b vectors: (R, E)
    b_w = resampled * W_t.unsqueeze(0)

    # Batched least squares (approximate NNLS via lstsq + clamp)
    # Solve A_w @ x = b_w for each replicate
    # A_w is (E, K), b_w is (R, E) -> solve R systems
    A_w_T = A_w.T  # (K, E)
    ATA = A_w_T @ A_w  # (K, K)
    ATb = A_w_T @ b_w.T  # (K, R)

    # Solve via Cholesky or direct
    try:
        L = torch.linalg.cholesky(ATA + 1e-6 * torch.eye(n_paths, device=device))
        x = torch.cholesky_solve(ATb, L)  # (K, R)
        x = torch.clamp(x, min=0)  # Approximate NNLS
        weight_matrix[:] = x.T.cpu().numpy()
        return True
    except Exception:
        return False


def _resample_edge_weights(
    original_weights: np.ndarray,
    is_junction: np.ndarray,
    rng: np.random.Generator,
    mode: str = "poisson",
) -> np.ndarray:
    """Generate a bootstrap resample of edge weights.

    For junction edges, resamples from Poisson(observed_count) to simulate
    sampling variability in splice junction detection.  Continuation edge
    weights are scaled proportionally to maintain graph flow balance.

    Args:
        original_weights: Original edge weights (n_edges,).
        is_junction: Boolean mask for junction edges.
        rng: NumPy random generator.
        mode: Resampling strategy (``"poisson"`` or ``"multinomial"``).

    Returns:
        Resampled edge weights.
    """
    resampled = original_weights.copy()

    if mode == "poisson":
        # Junction edges: Poisson resample (natural model for count data)
        junction_mask = is_junction & (original_weights > 0)
        if np.any(junction_mask):
            lambdas = original_weights[junction_mask]
            resampled[junction_mask] = rng.poisson(lambdas).astype(np.float64)
            # Ensure at least 1 read for resampled junctions
            resampled[junction_mask] = np.maximum(resampled[junction_mask], 1.0)

        # Continuation edges: scale deterministically by the ratio of
        # resampled to original total junction coverage.  No independent
        # noise — continuation edge coverage is derived from junction
        # evidence, not an independent random variable.
        cont_mask = ~is_junction & (original_weights > 0)
        if np.any(junction_mask) and np.any(cont_mask):
            orig_total = np.sum(original_weights[junction_mask])
            resamp_total = np.sum(resampled[junction_mask])
            if orig_total > 0:
                scale = resamp_total / orig_total
                resampled[cont_mask] = original_weights[cont_mask] * scale
                resampled[cont_mask] = np.maximum(resampled[cont_mask], 0.1)

    elif mode == "multinomial":
        junction_mask = is_junction & (original_weights > 0)
        if np.any(junction_mask):
            total = int(np.sum(original_weights[junction_mask]))
            probs = original_weights[junction_mask] / np.sum(
                original_weights[junction_mask]
            )
            counts = rng.multinomial(total, probs)
            resampled[junction_mask] = counts.astype(np.float64)
            resampled[junction_mask] = np.maximum(resampled[junction_mask], 1.0)
    else:
        raise ValueError(f"Unknown resample_mode: {mode!r}")

    return resampled


def bootstrap_confidence(
    graph_csr: CSRGraph,
    paths: list[list[int]],
    config: BootstrapConfig | None = None,
) -> BootstrapResult:
    """Compute bootstrap confidence intervals for transcript abundances.

    Resamples junction read counts from the splice graph, re-solves NNLS
    for each replicate, and computes per-transcript confidence statistics.

    Args:
        graph_csr: CSR splice graph with edge weights.
        paths: List of source-to-sink paths (node ID lists).
        config: Bootstrap configuration.  Uses defaults if ``None``.

    Returns:
        :class:`BootstrapResult` with per-transcript confidence statistics.
    """
    if config is None:
        config = BootstrapConfig()

    n_paths = len(paths)
    n_edges = graph_csr.n_edges

    if n_paths == 0 or n_edges == 0:
        return BootstrapResult(
            transcripts=[],
            n_replicates=config.n_replicates,
            confidence_level=config.confidence_level,
            n_stable=0,
            weight_matrix=np.empty((config.n_replicates, 0)),
        )

    rng = np.random.default_rng(config.seed)
    edge_map = _build_edge_map(graph_csr)
    A = _build_path_edge_matrix(graph_csr, paths, edge_map)
    is_junction = _identify_junction_edges(graph_csr)
    original_weights = graph_csr.edge_weights.astype(np.float64)

    # Build junction weight vector
    W = np.ones(n_edges, dtype=np.float64)
    W[is_junction] = config.junction_weight

    # Run bootstrap replicates
    weight_matrix = np.zeros((config.n_replicates, n_paths), dtype=np.float64)

    # Try GPU-batched bootstrap if available
    if _try_gpu_bootstrap(
        A, W, original_weights, is_junction, config, rng, weight_matrix,
    ):
        pass  # GPU path filled weight_matrix
    else:
        # CPU sequential fallback
        for rep in range(config.n_replicates):
            b_resamp = _resample_edge_weights(
                original_weights, is_junction, rng, mode=config.resample_mode,
            )
            A_w = A * W[:, np.newaxis]
            b_w = b_resamp * W
            try:
                weights_rep, _ = nnls(A_w, b_w)
                weight_matrix[rep, :] = weights_rep
            except Exception:
                pass

    # Compute per-transcript statistics
    alpha = 1.0 - config.confidence_level
    transcripts: list[TranscriptConfidence] = []

    for pi in range(n_paths):
        col = weight_matrix[:, pi]
        presence = np.sum(col > 0) / config.n_replicates
        mean_w = float(np.mean(col))
        median_w = float(np.median(col))
        ci_low = float(np.percentile(col, 100 * alpha / 2))
        ci_high = float(np.percentile(col, 100 * (1 - alpha / 2)))
        std_w = float(np.std(col))
        cv = std_w / mean_w if mean_w > 0 else float("nan")

        transcripts.append(
            TranscriptConfidence(
                path_index=pi,
                weight_mean=mean_w,
                weight_median=median_w,
                weight_ci_low=ci_low,
                weight_ci_high=ci_high,
                presence_rate=presence,
                cv=cv,
                weights=col.copy(),
            )
        )

    n_stable = sum(
        1 for tc in transcripts if tc.presence_rate >= config.min_presence_rate
    )

    logger.info(
        "Bootstrap: %d replicates, %d/%d transcripts stable (presence >= %.0f%%)",
        config.n_replicates,
        n_stable,
        n_paths,
        config.min_presence_rate * 100,
    )

    return BootstrapResult(
        transcripts=transcripts,
        n_replicates=config.n_replicates,
        confidence_level=config.confidence_level,
        n_stable=n_stable,
        weight_matrix=weight_matrix,
    )


def format_confidence_gtf_attributes(tc: TranscriptConfidence) -> str:
    """Format bootstrap confidence as GTF attribute string.

    Produces attributes like:
        bootstrap_ci_low "12.3"; bootstrap_ci_high "45.6";
        bootstrap_presence "0.95"; bootstrap_cv "0.23";

    Args:
        tc: Transcript confidence statistics.

    Returns:
        GTF-format attribute string fragment.
    """
    return (
        f'bootstrap_ci_low "{tc.weight_ci_low:.2f}"; '
        f'bootstrap_ci_high "{tc.weight_ci_high:.2f}"; '
        f'bootstrap_presence "{tc.presence_rate:.3f}"; '
        f'bootstrap_cv "{tc.cv:.3f}";'
    )
