"""Batched locus processing -- orchestrates GPU/CPU processing of multiple gene loci.

This module provides the :class:`BatchProcessor` which is the primary interface
between the assembly pipeline and the CUDA/CPU acceleration layer.  It handles:

- Transparent device selection (GPU via CuPy or CPU via NumPy).
- Batching of read-level data for parallel coverage computation.
- Packing multiple CSR splice graphs into the :class:`BatchedCSRGraphs`
  layout for efficient batch kernel launches.
- Host-device memory transfers with automatic fallback.

Typical usage::

    proc = BatchProcessor(use_gpu=True, batch_size=2048)
    coverages = proc.process_coverage_batch(read_data_list, regions)
    batched = proc.process_graph_batch(csr_graphs)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rapidsplice.cuda.backend import get_array_module, get_backend, to_device, to_host
from rapidsplice.cuda.kernels import parallel_coverage_scan
from rapidsplice.graph.splice_graph import BatchedCSRGraphs, CSRGraph

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Orchestrates batched computation across multiple gene loci.

    Provides high-level methods for computing coverage arrays, packing
    splice graphs, and transferring data between host and device memory.
    All operations have CPU fallback paths so the assembler functions
    identically on machines without a GPU.

    Args:
        use_gpu: If ``True`` and a CUDA-capable GPU is available (as
            reported by the backend module), arrays will be transferred
            to the GPU for processing.  Otherwise all computation
            stays on the CPU.
        batch_size: Maximum number of loci to process in a single
            kernel launch / batch.  Larger batches amortise launch
            overhead but require more device memory.
    """

    def __init__(self, use_gpu: bool = False, batch_size: int = 1024) -> None:
        self._use_gpu: bool = use_gpu and get_backend() == "gpu"
        self._batch_size: int = batch_size
        self._xp: Any = get_array_module()

        if self._use_gpu:
            logger.info(
                "BatchProcessor initialised with GPU backend (batch_size=%d)",
                batch_size,
            )
        else:
            logger.info(
                "BatchProcessor initialised with CPU backend (batch_size=%d)",
                batch_size,
            )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def use_gpu(self) -> bool:
        """Whether the processor is using the GPU backend."""
        return self._use_gpu

    @property
    def batch_size(self) -> int:
        """Maximum loci per batch."""
        return self._batch_size

    # ------------------------------------------------------------------
    # Coverage computation
    # ------------------------------------------------------------------

    def process_coverage_batch(
        self,
        read_data_list: list[Any],
        regions: list[tuple[int, int]],
    ) -> list[np.ndarray]:
        """Compute per-base coverage arrays for multiple genomic regions.

        Each entry in *read_data_list* is expected to expose ``.positions``
        and ``.end_positions`` attributes (both ``int64`` NumPy arrays), as
        returned by :class:`rapidsplice.io.bam_reader.ReadData`.

        On the GPU path, read position arrays are transferred to device
        memory and processed with a CuPy-backed coverage kernel.  On the
        CPU path, the Numba JIT :func:`parallel_coverage_scan` kernel is
        used directly.

        Args:
            read_data_list: One :class:`ReadData` per region.  Must be the
                same length as *regions*.
            regions: List of ``(region_start, region_end)`` tuples where
                ``region_start`` is 0-based inclusive and ``region_end``
                is 0-based exclusive.

        Returns:
            List of int64 NumPy arrays, one per region, each of length
            ``region_end - region_start`` containing per-base coverage.

        Raises:
            ValueError: If *read_data_list* and *regions* differ in length.
        """
        if len(read_data_list) != len(regions):
            raise ValueError(
                f"read_data_list length ({len(read_data_list)}) does not match "
                f"regions length ({len(regions)})."
            )

        results: list[np.ndarray] = []

        for read_data, (region_start, region_end) in zip(
            read_data_list, regions, strict=True
        ):
            positions = np.asarray(read_data.positions, dtype=np.int64)
            end_positions = np.asarray(read_data.end_positions, dtype=np.int64)

            if positions.shape != end_positions.shape:
                raise ValueError(
                    "positions and end_positions must have identical shape "
                    f"(got {positions.shape} vs {end_positions.shape})."
                )

            if positions.shape[0] == 0:
                # No reads -- zero coverage.
                results.append(np.zeros(region_end - region_start, dtype=np.int64))
                continue

            if self._use_gpu:
                coverage = self._gpu_coverage(
                    positions, end_positions, region_start, region_end
                )
            else:
                coverage = parallel_coverage_scan(
                    positions, end_positions, region_start, region_end
                )

            results.append(np.asarray(coverage, dtype=np.int64))

        return results

    def _gpu_coverage(
        self,
        positions: np.ndarray,
        end_positions: np.ndarray,
        region_start: int,
        region_end: int,
    ) -> np.ndarray:
        """Compute coverage on the GPU using CuPy.

        Falls back to the CPU Numba kernel if the GPU transfer or
        computation fails for any reason.

        Args:
            positions: int64 read start positions (host).
            end_positions: int64 read end positions (host).
            region_start: Region start coordinate.
            region_end: Region end coordinate.

        Returns:
            int64 coverage array on the host.
        """
        try:
            xp = self._xp
            region_len = region_end - region_start

            d_positions = xp.asarray(positions)
            d_end_positions = xp.asarray(end_positions)

            # Delta-encoding on device.
            delta = xp.zeros(region_len + 1, dtype=xp.int64)

            clipped_starts = xp.clip(d_positions, region_start, region_end) - region_start
            clipped_ends = xp.clip(d_end_positions, region_start, region_end) - region_start

            mask = clipped_starts < clipped_ends

            # Use scatter_add for atomic increments on device.
            if xp.__name__ == "cupy":
                import cupyx
                cupyx.scatter_add(delta, clipped_starts[mask], 1)
                cupyx.scatter_add(delta, clipped_ends[mask], -1)
            else:
                # NumPy: use np.add.at for unbuffered in-place addition
                xp.add.at(delta, clipped_starts[mask], 1)
                xp.add.at(delta, clipped_ends[mask], -1)

            coverage = xp.cumsum(delta[:region_len])
            return to_host(coverage)

        except Exception:
            logger.warning(
                "GPU coverage computation failed, falling back to CPU kernel."
            )
            return parallel_coverage_scan(
                positions, end_positions, region_start, region_end
            )

    # ------------------------------------------------------------------
    # Graph batching
    # ------------------------------------------------------------------

    def process_graph_batch(
        self,
        graphs_csr: list[CSRGraph],
    ) -> BatchedCSRGraphs:
        """Pack multiple CSR graphs into a :class:`BatchedCSRGraphs`.

        Iterates over the input list, adds each graph to a new batch
        container, finalises it, and optionally transfers the resulting
        contiguous arrays to the GPU.

        Args:
            graphs_csr: List of :class:`CSRGraph` instances, one per
                gene locus.

        Returns:
            A finalised :class:`BatchedCSRGraphs` instance.  If the GPU
            backend is active, the internal arrays have been transferred
            to device memory.

        Raises:
            ValueError: If *graphs_csr* is empty.
        """
        if not graphs_csr:
            raise ValueError("Cannot batch an empty list of graphs.")

        batch = BatchedCSRGraphs()
        for graph in graphs_csr:
            batch.add_graph(graph)
        batch.finalize()

        if self._use_gpu:
            self._transfer_batch_to_device(batch)

        return batch

    def _transfer_batch_to_device(self, batch: BatchedCSRGraphs) -> None:
        """Transfer all arrays in a finalised batch to the GPU.

        This mutates the batch object's internal arrays in-place by
        replacing them with their device-resident counterparts.  If the
        transfer fails, the batch remains on the CPU and a warning is
        logged.

        Args:
            batch: A finalised :class:`BatchedCSRGraphs`.
        """
        try:
            # Stage all transfers first so we can atomically swap only when all
            # device copies are successful.
            row_offsets = to_device(batch.row_offsets)
            col_indices = to_device(batch.col_indices)
            edge_weights = to_device(batch.edge_weights)
            node_coverages = to_device(batch.node_coverages)
            graph_offsets = to_device(batch.graph_offsets)
            edge_offsets = to_device(batch.edge_offsets)

            edge_coverages = (
                to_device(batch._edge_coverages)
                if batch._edge_coverages is not None
                else None
            )
            node_starts = (
                to_device(batch._node_starts)
                if batch._node_starts is not None
                else None
            )
            node_ends = (
                to_device(batch._node_ends)
                if batch._node_ends is not None
                else None
            )
            node_types = (
                to_device(batch._node_types)
                if batch._node_types is not None
                else None
            )

            batch._row_offsets = row_offsets
            batch._col_indices = col_indices
            batch._edge_weights = edge_weights
            batch._node_coverages = node_coverages
            batch._graph_offsets = graph_offsets
            batch._edge_offsets = edge_offsets
            batch._edge_coverages = edge_coverages
            batch._node_starts = node_starts
            batch._node_ends = node_ends
            batch._node_types = node_types

            logger.debug("Transferred batched graphs to GPU.")
        except Exception:
            logger.warning(
                "Failed to transfer batched graphs to GPU; staying on CPU."
            )

    # ------------------------------------------------------------------
    # Device transfer helpers
    # ------------------------------------------------------------------

    def to_device(self, arr: np.ndarray) -> Any:
        """Transfer a NumPy array to the active compute device.

        On the GPU backend this produces a CuPy device array.  On the
        CPU backend the input array is returned unchanged.

        Args:
            arr: Host-side NumPy array.

        Returns:
            Array on the active device (CuPy ndarray or NumPy ndarray).
        """
        if self._use_gpu:
            return to_device(arr)
        return arr

    def to_host(self, arr: Any) -> np.ndarray:
        """Transfer an array from any device back to the CPU.

        If the array is already a NumPy array it is returned as-is.
        CuPy arrays are copied back to host memory.

        Args:
            arr: Device-resident or host-resident array.

        Returns:
            NumPy array on the host.
        """
        return to_host(arr)
