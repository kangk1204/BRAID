"""CPU/GPU backend abstraction layer.

Provides transparent dispatch between CuPy (GPU) and NumPy (CPU) backends.
All array operations in the pipeline go through this module so that the
assembler works identically on machines with or without NVIDIA GPUs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_GPU_AVAILABLE: bool | None = None
_BACKEND: str = "cpu"


def _check_gpu() -> bool:
    """Check if a CuPy-compatible CUDA GPU is available.

    The runtime array backend in this module is CuPy-based.  We therefore
    only report GPU availability when CuPy itself can be imported and sees
    at least one CUDA device.
    """
    global _GPU_AVAILABLE
    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    # Try CuPy (NVIDIA CUDA)
    try:
        import cupy as cp  # noqa: F401

        device_count = cp.cuda.runtime.getDeviceCount()
        _GPU_AVAILABLE = device_count > 0
        if _GPU_AVAILABLE:
            logger.info(
                "GPU backend available (CuPy %s, %d device(s))",
                cp.__version__, device_count,
            )
        else:
            logger.info("CuPy installed but no CUDA devices found")
        return _GPU_AVAILABLE
    except Exception:
        pass

    _GPU_AVAILABLE = False
    logger.info("CuPy CUDA backend not available, falling back to CPU backend")
    return _GPU_AVAILABLE


def set_backend(backend: str, threads: int = 0) -> None:
    """Force a specific backend ('cpu' or 'gpu') and configure parallelism.

    Args:
        backend: One of 'cpu', 'gpu', or 'auto'.
        threads: Number of threads for Numba parallel kernels.  0 means
            use all available cores (Numba default).
    """
    global _BACKEND
    if backend == "auto":
        _BACKEND = "gpu" if _check_gpu() else "cpu"
    elif backend in ("cpu", "gpu"):
        if backend == "gpu" and not _check_gpu():
            logger.warning("GPU requested but not available, falling back to CPU")
            _BACKEND = "cpu"
        else:
            _BACKEND = backend
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'cpu', 'gpu', or 'auto'.")

    # Configure Numba thread count for prange kernels.
    if threads > 0:
        try:
            import numba
            numba.set_num_threads(min(threads, numba.config.NUMBA_DEFAULT_NUM_THREADS))
            logger.info("Numba threads set to %d", threads)
        except (ImportError, Exception) as exc:
            logger.debug("Could not set Numba threads: %s", exc)

    logger.info("Backend set to: %s", _BACKEND)


def get_backend() -> str:
    """Return the currently active backend name."""
    return _BACKEND


def get_array_module() -> Any:
    """Return the array module for the active backend (NumPy or CuPy)."""
    if _BACKEND == "gpu" and _check_gpu():
        import cupy as cp

        return cp
    return np


def get_sparse_module() -> Any:
    """Return the sparse matrix module for the active backend."""
    if _BACKEND == "gpu" and _check_gpu():
        import cupyx.scipy.sparse as cusparse

        return cusparse
    import scipy.sparse

    return scipy.sparse


def to_device(arr: np.ndarray) -> Any:
    """Transfer a NumPy array to the active device (GPU or keep on CPU).

    Args:
        arr: Input NumPy array.

    Returns:
        Array on the active device.
    """
    if _BACKEND == "gpu" and _check_gpu():
        import cupy as cp

        return cp.asarray(arr)
    return arr


def to_host(arr: Any) -> np.ndarray:
    """Transfer an array from any device back to CPU as NumPy.

    Args:
        arr: Input array (NumPy or CuPy).

    Returns:
        NumPy array on CPU.
    """
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def is_gpu_array(arr: Any) -> bool:
    """Check if an array is a CuPy GPU array."""
    try:
        import cupy as cp

        return isinstance(arr, cp.ndarray)
    except ImportError:
        return False


@dataclass
class DeviceInfo:
    """Information about the compute device."""

    backend: str
    device_name: str
    total_memory_gb: float
    compute_capability: str


def get_device_info() -> DeviceInfo:
    """Return information about the active compute device."""
    if _BACKEND == "gpu" and _check_gpu():
        import cupy as cp

        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        mem_gb = props["totalGlobalMem"] / (1024**3)
        cc = f"{props['major']}.{props['minor']}"
        return DeviceInfo(backend="gpu", device_name=name, total_memory_gb=mem_gb,
                          compute_capability=cc)
    return DeviceInfo(backend="cpu", device_name="CPU", total_memory_gb=0.0,
                      compute_capability="N/A")
