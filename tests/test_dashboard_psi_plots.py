"""Tests for the dashboard PSI-plot preparation helpers.

The dashboard loader only requires ``gene_id``/``psi``/``event_type``, so a
partial events TSV (``ci_width`` present but no ``total_reads``) used to pass the
loader and then crash inside Plotly with ``Value of 'size' is not the name of a
column`` (scatter), or inside pandas with a ``KeyError`` (summary aggregation).
The plot-prep helpers now build their column references from the columns the
frame actually carries, so the page renders instead of crashing.
"""

import pandas as pd

from braid.dashboard.components.psi_plots import (
    _ci_scatter_size_kwargs,
    _psi_summary_agg,
)


def test_scatter_size_omitted_when_total_reads_absent():
    cols = ["gene_id", "event_id", "psi", "event_type", "ci_width"]
    assert _ci_scatter_size_kwargs(cols) == {}


def test_scatter_size_used_when_total_reads_present():
    cols = ["gene_id", "psi", "event_type", "ci_width", "total_reads"]
    kwargs = _ci_scatter_size_kwargs(cols)
    assert kwargs["size"] == "total_reads"
    assert kwargs["size_max"] == 15


def test_summary_agg_skips_missing_optional_columns():
    cols = ["gene_id", "psi", "event_type", "ci_width"]  # no total_reads
    agg = _psi_summary_agg(cols)
    assert "mean_reads" not in agg
    assert "mean_ci_width" in agg
    assert agg["count"] == ("psi", "count")


def test_summary_agg_runs_on_partial_frame_without_keyerror():
    """The exact reproducer: ci_width present, total_reads absent."""
    df = pd.DataFrame({
        "gene_id": ["G1", "G1", "G2"],
        "event_id": ["e1", "e2", "e3"],
        "psi": [0.5, 0.6, 0.7],
        "event_type": ["SE", "SE", "A3SS"],
        "ci_width": [0.1, 0.2, 0.3],
    })
    summary = df.groupby("event_type").agg(**_psi_summary_agg(df.columns))
    assert "mean_psi" in summary.columns
    assert "mean_ci_width" in summary.columns
    assert "mean_reads" not in summary.columns
