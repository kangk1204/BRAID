"""Regression tests for CUDA batch fallback behavior."""

from __future__ import annotations

from unittest import mock

import numpy as np

import braid.cuda.batch as batch_module
from braid.cuda.batch import BatchProcessor


def test_gpu_coverage_fallback_logs_exception_context(monkeypatch) -> None:
    """GPU fallback should preserve exception context for diagnosis."""

    class BrokenXP:
        __name__ = "cupy"
        int64 = np.int64

        def asarray(self, arr):
            raise RuntimeError("simulated GPU transfer failure")

    def fake_cpu_scan(positions, end_positions, region_start, region_end):
        assert region_start == 10
        assert region_end == 13
        return np.array([1, 1, 0], dtype=np.int64)

    processor = BatchProcessor(use_gpu=False)
    processor._xp = BrokenXP()
    monkeypatch.setattr(batch_module, "parallel_coverage_scan", fake_cpu_scan)

    with mock.patch.object(batch_module.logger, "warning") as warn:
        coverage = processor._gpu_coverage(
            np.array([10], dtype=np.int64),
            np.array([12], dtype=np.int64),
            10,
            13,
        )

    np.testing.assert_array_equal(coverage, np.array([1, 1, 0], dtype=np.int64))
    warn.assert_called_once()
    assert warn.call_args.kwargs["exc_info"] is True
