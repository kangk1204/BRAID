"""CUDA kernels and GPU memory management."""

from braid.cuda.backend import (
    get_array_module,
    get_backend,
    get_device_info,
    is_gpu_array,
    set_backend,
    to_device,
    to_host,
)
from braid.cuda.batch import BatchProcessor
from braid.cuda.kernels import (
    batch_coverage_uniformity,
    batch_dag_longest_path,
    batch_topological_sort,
    parallel_coverage_scan,
    parallel_edge_flow_residual,
    parallel_junction_count,
)
from braid.cuda.memory import MemoryManager

__all__ = [
    "BatchProcessor",
    "MemoryManager",
    "batch_coverage_uniformity",
    "batch_dag_longest_path",
    "batch_topological_sort",
    "get_array_module",
    "get_backend",
    "get_device_info",
    "is_gpu_array",
    "parallel_coverage_scan",
    "parallel_edge_flow_residual",
    "parallel_junction_count",
    "set_backend",
    "to_device",
    "to_host",
]
