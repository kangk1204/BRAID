"""GPU memory management with pooling and memory estimation.

Provides the :class:`MemoryManager` which tracks available GPU memory,
estimates the device-memory cost of splice-graph data structures, and
determines optimal batch sizes so that the pipeline never exceeds the
physical memory of the GPU.

On CPU-only systems every method degrades gracefully: memory queries
return zeros and batch-size computations fall back to processing all
graphs in a single batch (no memory constraint).
"""

from __future__ import annotations

import logging

from rapidsplice.cuda.backend import _check_gpu, get_backend
from rapidsplice.graph.splice_graph import CSRGraph

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants -- per-element byte costs for CSR arrays on the device
# ---------------------------------------------------------------------------

# Node-level arrays in CSRGraph:
#   row_offsets:    int32 (4 bytes per node, plus 1 sentinel)
#   node_coverages: float32 (4 bytes per node)
#   node_starts:   int64 (8 bytes per node)
#   node_ends:     int64 (8 bytes per node)
#   node_types:    int8  (1 byte per node)
_BYTES_PER_NODE: int = 4 + 4 + 8 + 8 + 1  # 25 bytes

# Edge-level arrays in CSRGraph:
#   col_indices:    int32   (4 bytes per edge)
#   edge_weights:   float32 (4 bytes per edge)
#   edge_coverages: float32 (4 bytes per edge)
_BYTES_PER_EDGE: int = 4 + 4 + 4  # 12 bytes

# Fixed overhead per graph in a batch (offsets, metadata pointers, etc.)
_FIXED_OVERHEAD_PER_GRAPH: int = 64

# Row-offsets sentinel (one extra int32 per graph)
_ROW_OFFSET_SENTINEL_BYTES: int = 4

# Safety margin: reserve 10 % of GPU memory for CUDA runtime bookkeeping.
_MEMORY_SAFETY_FACTOR: float = 0.90


class MemoryManager:
    """Manages GPU memory estimation and batch sizing for graph processing.

    Queries the GPU device (if available) at construction time and caches
    the total device memory.  Provides methods to estimate the memory
    footprint of CSR graphs and to determine how many graphs can fit in
    a single GPU batch without exceeding the available memory.

    Args:
        max_gpu_memory_gb: Maximum GPU memory to use, in GiB.  If ``0.0``
            (the default), the manager queries the device for its total
            memory.  On CPU-only systems the value is always ``0.0``.
    """

    def __init__(self, max_gpu_memory_gb: float = 0.0) -> None:
        self._gpu_available: bool = _check_gpu() and get_backend() == "gpu"
        self._total_memory_bytes: int = 0
        self._max_memory_bytes: int = 0

        if self._gpu_available:
            self._total_memory_bytes = self._query_total_device_memory()
            if max_gpu_memory_gb > 0.0:
                requested_bytes = int(max_gpu_memory_gb * (1024 ** 3))
                self._max_memory_bytes = min(requested_bytes, self._total_memory_bytes)
            else:
                self._max_memory_bytes = self._total_memory_bytes

            # Apply safety factor.
            self._max_memory_bytes = int(self._max_memory_bytes * _MEMORY_SAFETY_FACTOR)

            logger.info(
                "MemoryManager: GPU available, total=%.2f GiB, usable=%.2f GiB",
                self._total_memory_bytes / (1024 ** 3),
                self._max_memory_bytes / (1024 ** 3),
            )
        else:
            logger.info("MemoryManager: no GPU available, operating in CPU mode.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def gpu_available(self) -> bool:
        """Whether a usable GPU is present."""
        return self._gpu_available

    @property
    def total_memory_bytes(self) -> int:
        """Total device memory in bytes (0 if no GPU)."""
        return self._total_memory_bytes

    @property
    def max_memory_bytes(self) -> int:
        """Maximum usable device memory in bytes after safety margin."""
        return self._max_memory_bytes

    # ------------------------------------------------------------------
    # Memory estimation
    # ------------------------------------------------------------------

    def estimate_graph_memory(self, n_nodes: int, n_edges: int) -> int:
        """Estimate the device memory required for a single CSR graph.

        The estimate accounts for all arrays that are transferred to the
        GPU: row offsets, column indices, edge weights, edge coverages,
        node coverages, node coordinates, and node types.

        Args:
            n_nodes: Number of nodes in the graph.
            n_edges: Number of edges in the graph.

        Returns:
            Estimated memory in bytes.
        """
        node_bytes = n_nodes * _BYTES_PER_NODE + _ROW_OFFSET_SENTINEL_BYTES
        edge_bytes = n_edges * _BYTES_PER_EDGE
        return node_bytes + edge_bytes + _FIXED_OVERHEAD_PER_GRAPH

    def estimate_batch_memory(self, graphs: list[CSRGraph]) -> int:
        """Estimate the total device memory required for a batch of graphs.

        Sums the per-graph estimates and adds batch-level overhead for the
        graph-offset and edge-offset arrays.

        Args:
            graphs: List of :class:`CSRGraph` instances to batch.

        Returns:
            Estimated total memory in bytes.
        """
        if not graphs:
            return 0

        total = 0
        for g in graphs:
            total += self.estimate_graph_memory(g.n_nodes, g.n_edges)

        # Batch-level offset arrays: graph_offsets (int64) and edge_offsets (int64).
        n_graphs = len(graphs)
        total += (n_graphs + 1) * 8 * 2  # two int64 offset arrays

        return total

    def can_fit_batch(self, graphs: list[CSRGraph]) -> bool:
        """Check whether a batch of graphs fits in available GPU memory.

        On CPU-only systems this always returns ``True`` (no GPU memory
        constraint).

        Args:
            graphs: List of :class:`CSRGraph` instances.

        Returns:
            ``True`` if the estimated batch memory does not exceed the
            maximum usable device memory.
        """
        if not self._gpu_available:
            return True

        required = self.estimate_batch_memory(graphs)
        free = self._get_free_memory()
        usable = min(free, self._max_memory_bytes)
        return required <= usable

    def optimal_batch_size(self, graphs: list[CSRGraph]) -> int:
        """Determine how many graphs from the list can fit in one GPU batch.

        Starting from the full list, uses a binary-search style approach to
        find the largest prefix of *graphs* (in order) that fits within the
        available device memory.

        On CPU-only systems the full list length is returned.

        Args:
            graphs: List of :class:`CSRGraph` instances, ordered by
                processing priority.

        Returns:
            The number of graphs (from the start of the list) that can
            be processed in a single batch.  Always at least 1 unless
            *graphs* is empty.
        """
        if not graphs:
            return 0

        if not self._gpu_available:
            return len(graphs)

        free = self._get_free_memory()
        usable = min(free, self._max_memory_bytes)

        # Edge case: even a single graph doesn't fit -- return 1 anyway so
        # the caller can still make progress (it will spill to CPU).
        if self.estimate_batch_memory(graphs[:1]) > usable:
            logger.warning(
                "Single graph exceeds available GPU memory (%.2f MiB required, "
                "%.2f MiB free). Returning batch_size=1; caller should use CPU fallback.",
                self.estimate_batch_memory(graphs[:1]) / (1024 ** 2),
                usable / (1024 ** 2),
            )
            return 1

        # Binary search for the largest fitting prefix.
        lo, hi = 1, len(graphs)
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            cost = self.estimate_batch_memory(graphs[:mid])
            if cost <= usable:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return best

    # ------------------------------------------------------------------
    # Memory info
    # ------------------------------------------------------------------

    def get_memory_info(self) -> dict[str, float]:
        """Return a summary of GPU memory usage.

        On CPU-only systems all values are ``0.0``.

        Returns:
            Dictionary with keys ``'total_gb'``, ``'used_gb'``, and
            ``'free_gb'``, all in GiB.
        """
        if not self._gpu_available:
            return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0}

        free_bytes = self._get_free_memory()
        total_bytes = self._total_memory_bytes
        used_bytes = total_bytes - free_bytes

        return {
            "total_gb": total_bytes / (1024 ** 3),
            "used_gb": used_bytes / (1024 ** 3),
            "free_gb": free_bytes / (1024 ** 3),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _query_total_device_memory() -> int:
        """Query total device memory in bytes from CuPy.

        Returns:
            Total GPU memory in bytes, or 0 on failure.
        """
        try:
            import cupy as cp

            free, total = cp.cuda.runtime.memGetInfo()
            return int(total)
        except Exception:
            return 0

    @staticmethod
    def _get_free_memory() -> int:
        """Query currently free device memory in bytes from CuPy.

        Returns:
            Free GPU memory in bytes, or 0 on failure.
        """
        try:
            import cupy as cp

            free, total = cp.cuda.runtime.memGetInfo()
            return int(free)
        except Exception:
            return 0
