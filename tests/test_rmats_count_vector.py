"""Tests for rMATS per-replicate count-vector parsing robustness.

Real rMATS IJC_SAMPLE_1 / SJC_SAMPLE_1 fields are gapless integer CSVs, one value
per replicate. ``_parse_count_vector`` drops empty/``NA`` cells, but an *interior*
gap (a non-trailing empty/NA) would silently re-index the later replicates and
desynchronise inclusion/exclusion pairing — a confidently-wrong number. The parser
rejects it as a hard error, while leaving a benign trailing comma quiet.
"""

import logging

import pytest

from braid.target.rmats_bootstrap import _parse_count_vector


def test_gapless_vector_parses_unchanged():
    assert _parse_count_vector("10,20,30") == (10, 20, 30)
    assert _parse_count_vector("10") == (10,)
    assert _parse_count_vector(" 10 , 20 ") == (10, 20)


def test_trailing_empty_is_quiet(caplog):
    """A trailing comma is benign and must NOT warn."""
    with caplog.at_level(logging.WARNING):
        assert _parse_count_vector("10,20,") == (10, 20)
    assert not any("interior" in r.message for r in caplog.records)


def test_interior_gap_raises():
    """An interior empty cell must raise (it would shift replicate pairing)."""
    with pytest.raises(ValueError, match="interior"):
        _parse_count_vector("10,,30")


def test_interior_na_raises():
    """An interior NA cell must raise (it would shift replicate pairing)."""
    with pytest.raises(ValueError, match="interior"):
        _parse_count_vector("10,NA,30")


def test_empty_and_none_return_empty():
    assert _parse_count_vector("") == ()
    assert _parse_count_vector(None) == ()
