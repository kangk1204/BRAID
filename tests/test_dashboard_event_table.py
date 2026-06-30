"""Regression tests for dashboard event-table filtering."""

from __future__ import annotations

import sys
import types

import pandas as pd

from braid.dashboard.components.event_table import render_event_table


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_gene_search_treats_regex_metacharacters_as_literal(monkeypatch):
    """A literal ``[`` gene search must filter, not crash via pandas regex parsing."""
    seen: dict[str, pd.DataFrame] = {}

    fake_st = types.SimpleNamespace(
        header=lambda *a, **k: None,
        info=lambda *a, **k: None,
        columns=lambda n: [_Ctx() for _ in range(n)],
        multiselect=lambda *a, **k: ["SE"],
        slider=lambda *a, **k: (0.0, 1.0),
        number_input=lambda *a, **k: 0,
        text_input=lambda *a, **k: "[",
        write=lambda *a, **k: None,
        dataframe=lambda df, **k: seen.setdefault("df", df.copy()),
        download_button=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)

    events = pd.DataFrame({
        "event_id": ["e1", "e2"],
        "event_type": ["SE", "SE"],
        "gene_id": ["GENE[1]", "GENE2"],
        "psi": [0.4, 0.6],
    })

    render_event_table(events)

    assert seen["df"]["gene_id"].tolist() == ["GENE[1]"]
