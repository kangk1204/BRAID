"""Tests for the conditional-ridge NNLS solver (numerical conditioning guard).

The path-edge incidence matrix used by the flow decomposer is frequently
rank-deficient (condition numbers ~1e16-1e17), which leaves plain NNLS weights
in the null space arbitrary. ``solve_nnls_regularized`` must (a) be bit-identical
to ``scipy.optimize.nnls`` for well-conditioned systems and (b) return a stable,
non-negative, good-fit solution for severely ill-conditioned ones.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

from braid.flow.decompose import solve_nnls_regularized


def test_well_conditioned_matches_plain_nnls() -> None:
    rng = np.random.default_rng(0)
    A = rng.random((8, 3))
    b = rng.random(8)

    w, cond = solve_nnls_regularized(A, b)
    w_plain, _ = nnls(A, b)

    assert np.allclose(w, w_plain)  # behaviour-preserving below the threshold
    assert np.isfinite(cond) and cond < 1e12
    assert np.all(w >= 0)


def test_rank_deficient_is_finite_nonnegative_and_fits() -> None:
    # Duplicate a column -> exact rank deficiency -> huge condition number.
    base = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0]])
    A = np.hstack([base, base[:, :1]])  # column 0 duplicated as column 2
    b = np.array([2.0, 3.0, 1.0, 3.0])

    # Force the ridge branch with a low threshold.
    w, cond = solve_nnls_regularized(A, b, cond_threshold=1e6)

    assert np.all(np.isfinite(w))
    assert np.all(w >= 0)
    assert cond > 1e6  # genuinely ill-conditioned
    # The ridge is tiny, so the data fit is essentially preserved.
    assert np.linalg.norm(A @ w - b) < 0.5
    # Deterministic / reproducible.
    w2, _ = solve_nnls_regularized(A, b, cond_threshold=1e6)
    assert np.allclose(w, w2)


def test_empty_and_degenerate_inputs() -> None:
    w, cond = solve_nnls_regularized(np.zeros((0, 0)), np.zeros(0))
    assert w.shape == (0,)
    assert np.isnan(cond)
