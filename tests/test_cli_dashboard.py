"""Tests for `braid dashboard` failure propagation.

The dashboard launcher shells out to Streamlit. Previously it called
``subprocess.run(cmd)`` without inspecting the return code, so a missing
Streamlit (or any launch failure) printed an error yet exited 0 — a false
success that hides the failure from CI smoke tests and workflow managers. It now
preflights for Streamlit and propagates a non-zero subprocess exit.
"""

from __future__ import annotations

import importlib.util
import subprocess
from argparse import Namespace

import pytest

from braid.cli import _run_dashboard


def _dash_args(**overrides: object) -> Namespace:
    base = {
        "events_tsv": "events.tsv",
        "gtf": "ann.gtf",
        "bam": None,
        "port": 8599,
    }
    base.update(overrides)
    return Namespace(**base)


def test_missing_streamlit_exits_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Streamlit installed -> clear error and non-zero exit, never 0."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    # subprocess must NOT even be reached; if it is, fail loudly.
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: pytest.fail("subprocess.run reached despite missing Streamlit"),
    )
    with pytest.raises(SystemExit) as exc:
        _run_dashboard(_dash_args())
    assert exc.value.code != 0


def test_streamlit_launch_failure_propagates_returncode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-zero Streamlit exit must surface as a non-zero CLI exit."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 3),
    )
    with pytest.raises(SystemExit) as exc:
        _run_dashboard(_dash_args())
    assert exc.value.code == 3


def test_streamlit_clean_exit_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clean (0) Streamlit exit must NOT raise."""
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0),
    )
    _run_dashboard(_dash_args())  # no raise
