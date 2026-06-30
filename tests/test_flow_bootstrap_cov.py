"""Coverage-focused tests for braid/flow/bootstrap.py.

Targets uncovered lines:
  - _try_gpu_bootstrap (176-237): covered via torch import mocks + cuda-unavailable mocks
  - zero-edges early return (320): zero-edge CSR with non-empty paths
  - allow_approximate_gpu branch entry (355): mock _try_gpu_bootstrap return values
  - NNLS failure path: failed[rep]=True + logger.debug (367-369)
  - logger.warning when fail_pct >= 5% (373-374)
  - n_valid==0 path (all replicates failed, 392-404)

All tests use synthetic in-memory fixtures; no external data.
"""

from __future__ import annotations

import sys
import types
import unittest.mock as mock

import numpy as np
import pytest

from braid.flow.bootstrap import (
    BootstrapConfig,
    BootstrapResult,
    TranscriptConfidence,
    _build_edge_map,
    _build_path_edge_matrix,
    _identify_junction_edges,
    _resample_edge_weights,
    _try_gpu_bootstrap,
    bootstrap_confidence,
    format_confidence_gtf_attributes,
)
from braid.graph.splice_graph import CSRGraph, EdgeType, NodeType, SpliceGraph

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_two_path_csr() -> tuple[CSRGraph, list[list[int]]]:
    """Build a two-path CSR graph via SpliceGraph.to_csr() so edge_types is set.

    Topology (topological order guaranteed by to_csr):
        SOURCE -> exon1 -[INTRON]-> exon2 -> SINK   (path 0)
                        -[INTRON]-> exon3 -> SINK   (path 1)
    """
    g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=900)
    src = g.add_node(start=0, end=0, node_type=NodeType.SOURCE, coverage=0)
    e1 = g.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=100)
    e2 = g.add_node(start=300, end=400, node_type=NodeType.EXON, coverage=70)
    e3 = g.add_node(start=500, end=600, node_type=NodeType.EXON, coverage=30)
    snk = g.add_node(start=900, end=900, node_type=NodeType.SINK, coverage=0)

    g.add_edge(src, e1, EdgeType.SOURCE_LINK, weight=100, coverage=100)
    g.add_edge(e1, e2, EdgeType.INTRON, weight=70, coverage=70)
    g.add_edge(e1, e3, EdgeType.INTRON, weight=30, coverage=30)
    g.add_edge(e2, snk, EdgeType.SINK_LINK, weight=70, coverage=70)
    g.add_edge(e3, snk, EdgeType.SINK_LINK, weight=30, coverage=30)

    csr = g.to_csr()
    # to_csr reorders nodes in topo order: SOURCE=0, exon1=1, exon2=2, exon3=3, SINK=4
    paths = [[0, 1, 2, 4], [0, 1, 3, 4]]
    return csr, paths


def _make_zero_edge_csr() -> CSRGraph:
    """CSR with 2 nodes but zero edges — triggers the n_edges==0 early return."""
    return CSRGraph(
        row_offsets=np.array([0, 0, 0], dtype=np.int32),
        col_indices=np.empty(0, dtype=np.int32),
        edge_weights=np.empty(0, dtype=np.float32),
        edge_coverages=np.empty(0, dtype=np.float32),
        node_coverages=np.zeros(2, dtype=np.float32),
        node_starts=np.array([0, 100], dtype=np.int64),
        node_ends=np.array([0, 200], dtype=np.int64),
        node_types=np.array([NodeType.SOURCE, NodeType.SINK], dtype=np.int8),
        n_nodes=2,
        n_edges=0,
    )


def _make_single_path_csr() -> tuple[CSRGraph, list[list[int]]]:
    """Linear graph: SOURCE -> exon -> SINK with one INTRON edge."""
    g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=500)
    src = g.add_node(start=0, end=0, node_type=NodeType.SOURCE, coverage=0)
    e1 = g.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=50)
    snk = g.add_node(start=500, end=500, node_type=NodeType.SINK, coverage=0)
    g.add_edge(src, e1, EdgeType.SOURCE_LINK, weight=50, coverage=50)
    g.add_edge(e1, snk, EdgeType.INTRON, weight=50, coverage=50)
    csr = g.to_csr()
    return csr, [[0, 1, 2]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _real_nnls(A, b):
    from scipy.optimize import nnls
    return nnls(A, b)


# ---------------------------------------------------------------------------
# 1. Zero-edges early return (line 320)
# ---------------------------------------------------------------------------


class TestZeroEdgesEarlyReturn:
    def test_zero_edges_with_nonempty_paths_returns_empty_result(self) -> None:
        """n_edges==0 must trigger early return even when paths is non-empty."""
        csr = _make_zero_edge_csr()
        result = bootstrap_confidence(csr, [[0, 1]], BootstrapConfig(n_replicates=10, seed=0))
        assert isinstance(result, BootstrapResult)
        assert result.transcripts == []
        assert result.n_stable == 0
        assert result.n_replicates == 10
        assert result.weight_matrix.shape[1] == 0

    def test_zero_edges_with_empty_paths_also_returns_empty(self) -> None:
        """Both n_paths==0 and n_edges==0 independently trigger the early return."""
        csr = _make_zero_edge_csr()
        result = bootstrap_confidence(csr, [], BootstrapConfig(n_replicates=5, seed=0))
        assert result.transcripts == []

    def test_none_config_uses_default_bootstrap_config(self) -> None:
        """Passing config=None must create a default BootstrapConfig (line 320)."""
        csr, paths = _make_two_path_csr()
        # config=None triggers the `config = BootstrapConfig()` branch
        result = bootstrap_confidence(csr, paths, None)
        assert isinstance(result, BootstrapResult)
        assert len(result.transcripts) == 2
        # Default n_replicates is 100
        assert result.n_replicates == 100


# ---------------------------------------------------------------------------
# 2. NNLS failure path (lines 367-369, 373-374) + failed-replicate exclusion
# ---------------------------------------------------------------------------


class TestNNLSFailurePaths:

    def test_all_replicates_fail_gives_nan_stats(self) -> None:
        """When every replicate fails NNLS, all per-transcript stats are NaN."""
        csr, paths = _make_two_path_csr()
        cfg = BootstrapConfig(n_replicates=15, seed=42)

        with mock.patch("braid.flow.bootstrap.nnls", side_effect=ValueError("mock fail")):
            result = bootstrap_confidence(csr, paths, cfg)

        assert result.n_replicates == 0
        for tc in result.transcripts:
            assert np.isnan(tc.weight_mean)
            assert np.isnan(tc.weight_ci_low)
            assert np.isnan(tc.weight_ci_high)
            assert np.isnan(tc.presence_rate)
            assert len(tc.weights) == 0

    def test_all_fail_n_stable_is_zero(self) -> None:
        """n_stable must be 0 when all replicates fail (NaN >= threshold is False)."""
        csr, paths = _make_two_path_csr()
        with mock.patch("braid.flow.bootstrap.nnls", side_effect=ValueError("fail")):
            result = bootstrap_confidence(csr, paths, BootstrapConfig(n_replicates=6, seed=0))
        assert result.n_stable == 0

    def test_high_fail_rate_logs_warning(self) -> None:
        """100% failure rate (>= 5%) must call logger.warning, not just debug."""
        csr, paths = _make_two_path_csr()
        cfg = BootstrapConfig(n_replicates=10, seed=0)

        with mock.patch("braid.flow.bootstrap.nnls", side_effect=ValueError("forced")):
            with mock.patch("braid.flow.bootstrap.logger") as mock_log:
                bootstrap_confidence(csr, paths, cfg)
                assert mock_log.warning.called

    def test_low_fail_rate_uses_debug_not_warning(self) -> None:
        """When < 5% of replicates fail, only debug is called (no warning)."""
        csr, paths = _make_two_path_csr()
        n_reps = 100
        call_count = [0]

        def sometimes_fail(A, b):
            call_count[0] += 1
            if call_count[0] <= 4:  # 4% -> below threshold
                raise ValueError("partial fail")
            return _real_nnls(A, b)

        cfg = BootstrapConfig(n_replicates=n_reps, seed=7)
        with mock.patch("braid.flow.bootstrap.nnls", side_effect=sometimes_fail):
            with mock.patch("braid.flow.bootstrap.logger") as mock_log:
                result = bootstrap_confidence(csr, paths, cfg)
                assert not mock_log.warning.called
                assert mock_log.debug.called

        assert result.n_replicates == 96  # 96 successes

    def test_partial_failure_excludes_failed_rows(self) -> None:
        """Failed replicates are excluded: weight_matrix has only success rows."""
        csr, paths = _make_two_path_csr()
        fail_on = {0, 2, 4}  # 3 of 20 -> 17 valid
        call_count = [0]

        def selective_fail(A, b):
            idx = call_count[0]
            call_count[0] += 1
            if idx in fail_on:
                raise ValueError("selective")
            return _real_nnls(A, b)

        cfg = BootstrapConfig(n_replicates=20, seed=11)
        with mock.patch("braid.flow.bootstrap.nnls", side_effect=selective_fail):
            result = bootstrap_confidence(csr, paths, cfg)

        assert result.n_replicates == 17
        assert result.weight_matrix.shape == (17, 2)
        for tc in result.transcripts:
            assert tc.weight_ci_low <= tc.weight_mean <= tc.weight_ci_high


# ---------------------------------------------------------------------------
# 3. n_valid == 0 path fields (lines 392-404)
# ---------------------------------------------------------------------------


class TestNValidZero:

    def _run_all_fail(self, n_paths: int = 2) -> BootstrapResult:
        csr, paths = _make_two_path_csr()
        cfg = BootstrapConfig(n_replicates=5, seed=0)
        with mock.patch("braid.flow.bootstrap.nnls", side_effect=RuntimeError("fail")):
            return bootstrap_confidence(csr, paths[:n_paths], cfg)

    def test_weights_array_empty_on_all_fail(self) -> None:
        result = self._run_all_fail()
        for tc in result.transcripts:
            assert len(tc.weights) == 0

    def test_path_index_correct_on_all_fail(self) -> None:
        result = self._run_all_fail(n_paths=2)
        assert result.transcripts[0].path_index == 0
        assert result.transcripts[1].path_index == 1

    def test_n_replicates_zero_on_all_fail(self) -> None:
        result = self._run_all_fail()
        assert result.n_replicates == 0

    def test_weight_cv_is_nan_on_all_fail(self) -> None:
        result = self._run_all_fail()
        for tc in result.transcripts:
            assert np.isnan(tc.cv)

    def test_weight_median_is_nan_on_all_fail(self) -> None:
        result = self._run_all_fail()
        for tc in result.transcripts:
            assert np.isnan(tc.weight_median)


# ---------------------------------------------------------------------------
# 4. allow_approximate_gpu branch (line 355)
# ---------------------------------------------------------------------------


class TestGPUBranchEntry:

    def test_default_config_does_not_call_try_gpu(self) -> None:
        """Default (allow_approximate_gpu=False) must never call _try_gpu_bootstrap."""
        from braid.flow import bootstrap as bsmod

        csr, paths = _make_two_path_csr()
        cfg = BootstrapConfig(n_replicates=5, seed=0, allow_approximate_gpu=False)

        with mock.patch.object(bsmod, "_try_gpu_bootstrap", return_value=False) as mock_gpu:
            bootstrap_confidence(csr, paths, cfg)
            assert not mock_gpu.called

    def test_allow_gpu_true_calls_try_gpu_bootstrap(self) -> None:
        """Setting allow_approximate_gpu=True must call _try_gpu_bootstrap."""
        from braid.flow import bootstrap as bsmod

        csr, paths = _make_two_path_csr()
        cfg = BootstrapConfig(n_replicates=5, seed=0, allow_approximate_gpu=True)

        with mock.patch.object(bsmod, "_try_gpu_bootstrap", return_value=False) as mock_gpu:
            bootstrap_confidence(csr, paths, cfg)
            assert mock_gpu.called

    def test_when_try_gpu_returns_true_weight_matrix_is_used_directly(self) -> None:
        """If _try_gpu_bootstrap returns True, the weight_matrix it wrote is used as-is."""
        from braid.flow import bootstrap as bsmod

        csr, paths = _make_two_path_csr()
        n_reps, n_paths = 8, len(paths)
        sentinel = np.full((n_reps, n_paths), 42.0)

        def fake_gpu(A, W, orig, is_junc, config, rng, weight_matrix):
            weight_matrix[:] = sentinel
            return True

        cfg = BootstrapConfig(n_replicates=n_reps, seed=0, allow_approximate_gpu=True)
        with mock.patch.object(bsmod, "_try_gpu_bootstrap", side_effect=fake_gpu):
            result = bootstrap_confidence(csr, paths, cfg)

        np.testing.assert_array_equal(result.weight_matrix, sentinel)

    def test_allow_gpu_true_but_no_cuda_falls_back_to_cpu(self) -> None:
        """CUDA unavailable -> _try_gpu_bootstrap returns False -> CPU path runs."""
        csr, paths = _make_two_path_csr()
        cfg = BootstrapConfig(n_replicates=20, seed=3, allow_approximate_gpu=True)
        result = bootstrap_confidence(csr, paths, cfg)
        assert isinstance(result, BootstrapResult)
        assert len(result.transcripts) == 2
        assert result.n_replicates > 0


# ---------------------------------------------------------------------------
# 5. _try_gpu_bootstrap internal branches (lines 176-237)
# ---------------------------------------------------------------------------


class TestTryGpuBootstrapBranches:
    """Test _try_gpu_bootstrap directly via module-level mocking of 'torch'."""

    def _base_args(self, n_reps: int = 10, n_paths: int = 2):
        n_edges = 5
        rng = np.random.default_rng(0)
        A = rng.random((n_edges, n_paths))
        W = np.ones(n_edges)
        orig = np.array([70.0, 70.0, 30.0, 70.0, 30.0])
        is_junc = np.array([False, True, True, False, False])
        cfg = BootstrapConfig(n_replicates=n_reps, seed=1, allow_approximate_gpu=True)
        rng2 = np.random.default_rng(1)
        wmat = np.zeros((n_reps, n_paths))
        return A, W, orig, is_junc, cfg, rng2, wmat

    def test_returns_false_when_torch_import_raises(self) -> None:
        """_try_gpu_bootstrap returns False when torch is not importable."""
        A, W, orig, is_junc, cfg, rng, wmat = self._base_args()
        # Inject a broken module so 'import torch' raises
        broken = types.ModuleType("torch")
        broken.__spec__ = None  # type: ignore[attr-defined]

        saved = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore[assignment]
        try:
            result = _try_gpu_bootstrap(A, W, orig, is_junc, cfg, rng, wmat)
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

        assert result is False

    def test_returns_false_when_cuda_not_available(self) -> None:
        """_try_gpu_bootstrap returns False when torch.cuda.is_available() is False."""
        A, W, orig, is_junc, cfg, rng, wmat = self._base_args()

        fake_torch = mock.MagicMock()
        fake_torch.cuda.is_available.return_value = False

        saved = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            result = _try_gpu_bootstrap(A, W, orig, is_junc, cfg, rng, wmat)
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

        assert result is False
        fake_torch.cuda.is_available.assert_called_once()

    def test_returns_false_when_cholesky_raises(self) -> None:
        """Returns False if the Cholesky solve raises (fallback to CPU)."""
        A, W, orig, is_junc, cfg, rng, wmat = self._base_args()
        n_reps = cfg.n_replicates
        n_edges, n_paths = A.shape

        # Build a minimal torch mock that reaches the cholesky call and then raises
        fake_torch = _build_minimal_torch_mock(n_reps, n_paths, raise_on_cholesky=True)

        saved = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            result = _try_gpu_bootstrap(A, W, orig, is_junc, cfg, rng, wmat)
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

        assert result is False

    def test_returns_true_when_solve_succeeds(self) -> None:
        """Returns True and fills weight_matrix when the GPU solve path works."""
        A, W, orig, is_junc, cfg, rng, wmat = self._base_args()
        n_reps = cfg.n_replicates
        n_edges, n_paths = A.shape

        fake_torch = _build_minimal_torch_mock(n_reps, n_paths, raise_on_cholesky=False)

        saved = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            result = _try_gpu_bootstrap(A, W, orig, is_junc, cfg, rng, wmat)
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

        assert result is True
        # weight_matrix should have been written (all ones from our mock solve)
        assert wmat.shape == (n_reps, n_paths)
        assert np.all(wmat >= 0)

    def test_gpu_resample_floor_matches_cpu_junction_floor(self) -> None:
        """GPU path must floor resampled junction edges at 1.0, not 0.1."""
        n_reps = 1
        n_paths = 2
        A = np.eye(2, dtype=np.float64)
        W = np.ones(2, dtype=np.float64)
        orig = np.array([5.0, 5.0], dtype=np.float64)
        is_junc = np.array([True, False])
        cfg = BootstrapConfig(n_replicates=n_reps, seed=1, allow_approximate_gpu=True)
        rng = np.random.default_rng(1)
        wmat = np.zeros((n_reps, n_paths))
        fake_torch = _build_minimal_torch_mock(
            n_reps, n_paths, raise_on_cholesky=False,
        )
        captured: dict[str, np.ndarray] = {}

        def zero_poisson(lam, generator=None):
            return type(lam)(np.zeros_like(lam.d))

        def capture_solve(ATb, L):
            captured["ATb"] = ATb.d.copy()
            return type(ATb)(np.ones((n_paths, n_reps)))

        fake_torch.poisson.side_effect = zero_poisson
        fake_torch.cholesky_solve.side_effect = capture_solve

        saved = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            ok = _try_gpu_bootstrap(A, W, orig, is_junc, cfg, rng, wmat)
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved

        assert ok is True
        np.testing.assert_allclose(captured["ATb"], np.array([[1.0], [1.0]]))


def _build_minimal_torch_mock(
    n_reps: int, n_paths: int, *, raise_on_cholesky: bool
) -> mock.MagicMock:
    """Build the simplest possible torch mock that drives _try_gpu_bootstrap through its body.

    Uses numpy arrays wrapped in a thin proxy so tensor operations work correctly.
    """
    import numpy as np

    class T:
        """Thin numpy-backed tensor proxy."""

        def __init__(self, arr):
            # Preserve bool dtype; convert everything else to float64
            a = np.asarray(arr)
            self.d = a if a.dtype == bool else a.astype(np.float64)

        # ----- shape / iteration -----
        @property
        def shape(self):
            return self.d.shape

        def __len__(self):
            return len(self.d)

        # ----- indexing -----
        @staticmethod
        def _unwrap_key(key):
            """Recursively unwrap T instances in index keys (including tuples)."""
            if isinstance(key, T):
                return key.d
            if isinstance(key, tuple):
                return tuple(T._unwrap_key(k) for k in key)
            return key

        def __getitem__(self, key):
            return T(self.d[T._unwrap_key(key)])

        def __setitem__(self, key, val):
            self.d[T._unwrap_key(key)] = val.d if isinstance(val, T) else val

        # ----- arithmetic -----
        def __mul__(self, other):
            return T(self.d * (other.d if isinstance(other, T) else other))

        def __rmul__(self, other):
            return self.__mul__(other)

        def __truediv__(self, other):
            return T(self.d / (other.d if isinstance(other, T) else other))

        def __add__(self, other):
            return T(self.d + (other.d if isinstance(other, T) else other))

        def __matmul__(self, other):
            return T(self.d @ other.d)

        def __neg__(self):
            return T(-self.d)

        def __invert__(self):
            return T(~self.d)

        def __gt__(self, other):
            val = other.d if isinstance(other, T) else other
            return T(self.d > val)

        def __lt__(self, other):
            val = other.d if isinstance(other, T) else other
            return T(self.d < val)

        # ----- reductions -----
        def sum(self, dim=None, keepdim=False):
            if dim is None:
                return T(np.array(self.d.sum()))
            return T(self.d.sum(axis=dim, keepdims=keepdim))

        # ----- shape ops -----
        def unsqueeze(self, dim):
            return T(np.expand_dims(self.d, axis=dim))

        def expand(self, *sizes):
            shape = tuple(
                self.d.shape[i] if s == -1 else s for i, s in enumerate(sizes)
            )
            return T(np.broadcast_to(self.d, shape).copy())

        def clone(self):
            return T(self.d.copy())

        @property
        def T(self):  # noqa: N802
            return T(self.d.T)

        # ----- device / dtype transfer -----
        def float(self):
            return T(self.d.astype(np.float64))

        def bool(self):
            # Return a new T but with bool numpy array preserved
            arr = self.d.astype(bool)
            obj = object.__new__(T)
            obj.d = arr
            return obj

        def to(self, device):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.d.copy()

        # ----- bool indexing support -----
        def __bool__(self):
            return bool(self.d.any())

    # ----- factory functions -----
    ft = mock.MagicMock(name="torch")
    ft.cuda.is_available.return_value = True
    ft.device.return_value = "cpu"

    ft.from_numpy.side_effect = lambda a: T(a)
    ft.eye.side_effect = lambda n, device=None: T(np.eye(n))

    # Generator (just needs manual_seed)
    gen = mock.MagicMock()
    gen.manual_seed.return_value = None
    ft.Generator.return_value = gen

    # torch.poisson: return lambdas unchanged (mean-preserving stand-in)
    ft.poisson.side_effect = lambda lam, generator=None: T(lam.d.copy())

    # torch.clamp
    ft.clamp.side_effect = lambda t, min=None: T(
        np.maximum(t.d, min) if min is not None else t.d.copy()
    )

    if raise_on_cholesky:
        ft.linalg.cholesky.side_effect = RuntimeError("singular mock")
    else:
        # Return identity (K x K); cholesky_solve returns (K x R) of ones
        ft.linalg.cholesky.side_effect = lambda m: T(np.eye(m.d.shape[0]))
        ft.cholesky_solve.side_effect = lambda ATb, L: T(np.ones((n_paths, n_reps)))

    return ft


# ---------------------------------------------------------------------------
# 6. CI ordering and numeric correctness
# ---------------------------------------------------------------------------


class TestCIOrdering:

    def test_ci_low_le_ci_high(self) -> None:
        csr, paths = _make_two_path_csr()
        result = bootstrap_confidence(csr, paths, BootstrapConfig(n_replicates=80, seed=99))
        for tc in result.transcripts:
            assert tc.weight_ci_low <= tc.weight_ci_high

    def test_ci_bounds_contain_mean(self) -> None:
        csr, paths = _make_two_path_csr()
        result = bootstrap_confidence(csr, paths, BootstrapConfig(n_replicates=200, seed=42))
        for tc in result.transcripts:
            assert tc.weight_ci_low <= tc.weight_mean <= tc.weight_ci_high

    def test_dominant_path_has_higher_mean(self) -> None:
        """Path carrying 70% of reads should have higher mean than 30% path."""
        csr, paths = _make_two_path_csr()
        result = bootstrap_confidence(csr, paths, BootstrapConfig(n_replicates=100, seed=1))
        assert result.transcripts[0].weight_mean > result.transcripts[1].weight_mean

    def test_hand_computed_presence_rate(self) -> None:
        """presence_rate = fraction of valid rows with weight > 0; verify by hand."""
        csr, paths = _make_two_path_csr()
        result = bootstrap_confidence(csr, paths, BootstrapConfig(n_replicates=50, seed=5))
        for tc in result.transcripts:
            col = result.weight_matrix[:, tc.path_index]
            expected = float(np.sum(col > 0) / len(col))
            assert tc.presence_rate == pytest.approx(expected)

    def test_ci_percentiles_match_weight_matrix(self) -> None:
        """CI bounds must equal the alpha/2 and 1-alpha/2 percentiles of the weight column."""
        csr, paths = _make_two_path_csr()
        cfg = BootstrapConfig(n_replicates=60, confidence_level=0.95, seed=77)
        result = bootstrap_confidence(csr, paths, cfg)
        alpha = 1.0 - cfg.confidence_level
        for tc in result.transcripts:
            col = result.weight_matrix[:, tc.path_index]
            expected_low = float(np.percentile(col, 100 * alpha / 2))
            expected_high = float(np.percentile(col, 100 * (1 - alpha / 2)))
            assert tc.weight_ci_low == pytest.approx(expected_low)
            assert tc.weight_ci_high == pytest.approx(expected_high)


# ---------------------------------------------------------------------------
# 7. Determinism under fixed seed
# ---------------------------------------------------------------------------


class TestDeterminism:

    def test_same_seed_yields_identical_matrix(self) -> None:
        csr, paths = _make_two_path_csr()
        r1 = bootstrap_confidence(csr, paths, BootstrapConfig(n_replicates=50, seed=7))
        r2 = bootstrap_confidence(csr, paths, BootstrapConfig(n_replicates=50, seed=7))
        np.testing.assert_array_equal(r1.weight_matrix, r2.weight_matrix)

    def test_different_seeds_yield_different_matrices(self) -> None:
        csr, paths = _make_two_path_csr()
        r1 = bootstrap_confidence(csr, paths, BootstrapConfig(n_replicates=50, seed=1))
        r2 = bootstrap_confidence(csr, paths, BootstrapConfig(n_replicates=50, seed=2))
        assert not np.array_equal(r1.weight_matrix, r2.weight_matrix)


# ---------------------------------------------------------------------------
# 8. Edge cases: zero reads, single isoform
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_zero_weight_edges_do_not_crash(self) -> None:
        """Graph with all-zero edge weights should run without error."""
        g = SpliceGraph(chrom="chr1", strand="+", locus_start=0, locus_end=400)
        src = g.add_node(start=0, end=0, node_type=NodeType.SOURCE, coverage=0)
        e1 = g.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=0)
        snk = g.add_node(start=400, end=400, node_type=NodeType.SINK, coverage=0)
        g.add_edge(src, e1, EdgeType.SOURCE_LINK, weight=0, coverage=0)
        g.add_edge(e1, snk, EdgeType.INTRON, weight=0, coverage=0)
        csr = g.to_csr()
        result = bootstrap_confidence(csr, [[0, 1, 2]], BootstrapConfig(n_replicates=10, seed=0))
        assert len(result.transcripts) == 1
        assert result.transcripts[0].weight_mean >= 0.0

    def test_single_isoform_high_presence_rate(self) -> None:
        csr, paths = _make_single_path_csr()
        result = bootstrap_confidence(csr, paths, BootstrapConfig(n_replicates=50, seed=5))
        assert result.transcripts[0].presence_rate > 0.8

    def test_cv_nan_when_mean_zero(self) -> None:
        """CV should be NaN when the mean weight across replicates is 0."""
        csr, paths = _make_two_path_csr()
        n_reps = 20

        # Force all weights to zero so mean=0 -> cv=nan
        def zero_nnls(A, b):
            return np.zeros(A.shape[1]), 0.0

        cfg = BootstrapConfig(n_replicates=n_reps, seed=0)
        with mock.patch("braid.flow.bootstrap.nnls", side_effect=zero_nnls):
            result = bootstrap_confidence(csr, paths, cfg)

        for tc in result.transcripts:
            # mean=0 -> cv must be nan
            if tc.weight_mean == 0.0:
                assert np.isnan(tc.cv)


# ---------------------------------------------------------------------------
# 9. _identify_junction_edges — both code paths
# ---------------------------------------------------------------------------


class TestIdentifyJunctionEdges:

    def test_intron_only_flagged_when_edge_types_present(self) -> None:
        """With edge_types populated, only INTRON edges are True."""
        csr, _ = _make_two_path_csr()
        mask = _identify_junction_edges(csr)
        assert mask.dtype == bool
        for i, et in enumerate(np.asarray(csr.edge_types)):
            if int(et) == int(EdgeType.INTRON):
                assert mask[i]
            else:
                assert not mask[i]

    def test_legacy_heuristic_weight_equals_coverage(self) -> None:
        """Without edge_types, edges with weight==coverage are flagged."""
        csr_leg = CSRGraph(
            row_offsets=np.array([0, 1, 2, 2], dtype=np.int32),
            col_indices=np.array([1, 2], dtype=np.int32),
            edge_weights=np.array([10.0, 5.0], dtype=np.float32),
            edge_coverages=np.array([10.0, 8.0], dtype=np.float32),  # edge0: eq, edge1: neq
            node_coverages=np.zeros(3, dtype=np.float32),
            node_starts=np.array([0, 100, 300], dtype=np.int64),
            node_ends=np.array([0, 200, 400], dtype=np.int64),
            node_types=np.array([NodeType.SOURCE, NodeType.EXON, NodeType.SINK], dtype=np.int8),
            n_nodes=3,
            n_edges=2,
            edge_types=None,
        )
        mask = _identify_junction_edges(csr_leg)
        assert bool(mask[0]) is True   # weight(10)==coverage(10)
        assert bool(mask[1]) is False  # weight(5)!=coverage(8)

    def test_legacy_no_edge_coverages_returns_all_false(self) -> None:
        """Legacy path: when edge_coverages is None, mask is all-False."""
        csr_no_cov = CSRGraph(
            row_offsets=np.array([0, 1, 1], dtype=np.int32),
            col_indices=np.array([1], dtype=np.int32),
            edge_weights=np.array([5.0], dtype=np.float32),
            edge_coverages=np.array([5.0], dtype=np.float32),
            node_coverages=np.zeros(2, dtype=np.float32),
            node_starts=np.zeros(2, dtype=np.int64),
            node_ends=np.zeros(2, dtype=np.int64),
            node_types=np.array([NodeType.SOURCE, NodeType.SINK], dtype=np.int8),
            n_nodes=2,
            n_edges=1,
            edge_types=None,
        )
        object.__setattr__(csr_no_cov, "edge_coverages", None)
        mask = _identify_junction_edges(csr_no_cov)
        assert not np.any(mask)


# ---------------------------------------------------------------------------
# 10. _build_edge_map and _build_path_edge_matrix
# ---------------------------------------------------------------------------


class TestBuildHelpers:

    def test_edge_map_has_exactly_n_edges_entries(self) -> None:
        csr, _ = _make_two_path_csr()
        assert len(_build_edge_map(csr)) == csr.n_edges

    def test_path_edge_matrix_shape(self) -> None:
        csr, paths = _make_two_path_csr()
        A = _build_path_edge_matrix(csr, paths, _build_edge_map(csr))
        assert A.shape == (csr.n_edges, len(paths))

    def test_path_edge_matrix_binary(self) -> None:
        csr, paths = _make_two_path_csr()
        A = _build_path_edge_matrix(csr, paths, _build_edge_map(csr))
        assert set(A.flatten().tolist()).issubset({0.0, 1.0})

    def test_each_path_activates_at_least_one_edge(self) -> None:
        csr, paths = _make_two_path_csr()
        A = _build_path_edge_matrix(csr, paths, _build_edge_map(csr))
        for pi in range(len(paths)):
            assert A[:, pi].sum() > 0

    def test_empty_edge_map_yields_zero_matrix(self) -> None:
        """Missing edges in map are silently skipped — all-zero matrix."""
        csr, paths = _make_two_path_csr()
        A = _build_path_edge_matrix(csr, paths, {})
        assert np.all(A == 0.0)


# ---------------------------------------------------------------------------
# 11. _resample_edge_weights
# ---------------------------------------------------------------------------


class TestResampleEdgeWeights:

    def test_poisson_junction_mean_preserved(self) -> None:
        """Poisson resampling approximately preserves per-edge mean."""
        rng = np.random.default_rng(42)
        orig = np.array([100.0, 50.0, 200.0])
        is_junc = np.array([True, True, True])
        samples = [_resample_edge_weights(orig, is_junc, rng, "poisson") for _ in range(2000)]
        mean_arr = np.mean(samples, axis=0)
        np.testing.assert_allclose(mean_arr, orig, rtol=0.05)

    def test_non_junction_edges_scaled_not_noise(self) -> None:
        """Non-junction edges should be deterministically scaled, so all >= 0.1."""
        rng = np.random.default_rng(1)
        orig = np.array([100.0, 50.0, 200.0])
        is_junc = np.array([True, True, False])
        for _ in range(100):
            r = _resample_edge_weights(orig, is_junc, rng, "poisson")
            assert r[2] >= 0.1

    def test_no_junction_edges_returns_original(self) -> None:
        rng = np.random.default_rng(0)
        orig = np.array([10.0, 20.0])
        r = _resample_edge_weights(orig, np.array([False, False]), rng, "poisson")
        np.testing.assert_array_equal(r, orig)

    def test_multinomial_junction_total_preserved(self) -> None:
        """Multinomial should sum to original junction total (before floor)."""
        rng = np.random.default_rng(0)
        orig = np.array([40.0, 60.0])
        is_junc = np.array([True, True])
        expected = int(orig.sum())
        # After flooring each to >=1, sum can only be >= expected
        for _ in range(30):
            r = _resample_edge_weights(orig, is_junc, rng, "multinomial")
            assert r.sum() >= expected

    def test_multinomial_no_junction_returns_original(self) -> None:
        rng = np.random.default_rng(2)
        orig = np.array([5.0, 15.0])
        r = _resample_edge_weights(orig, np.array([False, False]), rng, "multinomial")
        np.testing.assert_array_equal(r, orig)

    def test_invalid_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown resample_mode"):
            _resample_edge_weights(np.array([1.0]), np.array([True]),
                                   np.random.default_rng(0), "bad_mode")

    def test_all_weights_positive_after_resample(self) -> None:
        """All resampled weights must be strictly positive."""
        rng = np.random.default_rng(99)
        orig = np.array([10.0, 5.0, 1.0, 50.0])
        is_junc = np.array([True, True, True, False])
        for _ in range(100):
            r = _resample_edge_weights(orig, is_junc, rng, "poisson")
            assert np.all(r > 0)


# ---------------------------------------------------------------------------
# 12. format_confidence_gtf_attributes — including NaN / zero
# ---------------------------------------------------------------------------


class TestFormatGTFAttributes:

    def test_normal_values(self) -> None:
        tc = TranscriptConfidence(
            path_index=0, weight_mean=50.0, weight_median=48.0,
            weight_ci_low=30.5, weight_ci_high=72.1,
            presence_rate=0.95, cv=0.23, weights=np.array([50.0]),
        )
        s = format_confidence_gtf_attributes(tc)
        assert 'bootstrap_ci_low "30.50"' in s
        assert 'bootstrap_ci_high "72.10"' in s
        assert 'bootstrap_presence "0.950"' in s
        assert 'bootstrap_cv "0.230"' in s

    def test_nan_values_do_not_crash(self) -> None:
        tc = TranscriptConfidence(
            path_index=0, weight_mean=float("nan"), weight_median=float("nan"),
            weight_ci_low=float("nan"), weight_ci_high=float("nan"),
            presence_rate=float("nan"), cv=float("nan"), weights=np.empty(0),
        )
        s = format_confidence_gtf_attributes(tc)
        assert "bootstrap_ci_low" in s
        assert "nan" in s.lower()

    def test_zero_values(self) -> None:
        tc = TranscriptConfidence(
            path_index=1, weight_mean=0.0, weight_median=0.0,
            weight_ci_low=0.0, weight_ci_high=0.0,
            presence_rate=0.0, cv=0.0, weights=np.array([0.0]),
        )
        s = format_confidence_gtf_attributes(tc)
        assert 'bootstrap_ci_low "0.00"' in s
        assert 'bootstrap_presence "0.000"' in s
