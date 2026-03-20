"""Tests for multi-replicate bootstrap helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from rapidsplice.target.multi_replicate_bootstrap import (
    combine_isoform_bootstrap_results,
)


def test_combine_isoform_bootstrap_results_falls_back_from_nan_cv() -> None:
    """Combine helper should use CI width when per-replicate CV is undefined."""
    rep1 = [
        SimpleNamespace(
            gene_id="G1",
            isoforms=[
                SimpleNamespace(
                    transcript_id="tx1",
                    nnls_weight=10.0,
                    cv=float("nan"),
                    ci_low=8.0,
                    ci_high=12.0,
                ),
            ],
        ),
    ]
    rep2 = [
        SimpleNamespace(
            gene_id="G1",
            isoforms=[
                SimpleNamespace(
                    transcript_id="tx1",
                    nnls_weight=14.0,
                    cv=0.1,
                    ci_low=13.0,
                    ci_high=15.0,
                ),
            ],
        ),
    ]

    combined = combine_isoform_bootstrap_results([rep1, rep2])

    assert len(combined) == 1
    row = combined[0]
    assert row["gene_id"] == "G1"
    assert row["transcript_id"] == "tx1"
    assert row["mean_weight"] == pytest.approx(12.0)
    assert row["bio_std"] == pytest.approx(2.82842712, rel=1e-6)
    assert row["sampling_std"] > 0.0
    assert row["n_replicates"] == 2
