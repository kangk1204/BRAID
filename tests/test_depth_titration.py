"""In-memory regression tests for the depth-titration thinning math.

The depth-titration benchmark recomputes every interval from binomially-thinned
rMATS count tables. A wrong thinning formula would silently corrupt every depth
point without crashing, so these tests lock the load-bearing properties with a
hand-built SEEvent (no data-tree dependency, per the in-memory test convention):

* full-depth thinning is an exact identity (this is what makes the f=1.0 column
  reproduce the canonical head-to-head numbers);
* sub-sampling is bounded, seed-deterministic, and binomial in expectation;
* the per-replicate IncLevel recompute matches the rMATS length-normalised PSI
  formula, including 0/0 drop-out;
* the recomputed ΔPSI follows the rMATS IncLevelDifference = IL1 - IL2 sign
  convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmarks.headtohead.depth_titration import (
    _recompute_inclevel,
    thin_event,
)
from benchmarks.headtohead.head_to_head_coverage import SEEvent


def _event() -> SEEvent:
    """A hand-built 3-replicate SE event with known counts."""
    return SEEvent(
        gene="G", chrom="chr1", strand="+", exon_start=100, exon_end=200,
        ijc1=(40, 50, 60), sjc1=(10, 12, 8),
        ijc2=(5, 6, 4), sjc2=(45, 44, 46),
        inc_form_len=148.0, skip_form_len=100.0,
        inclevel1=(0.0, 0.0, 0.0), inclevel2=(0.0, 0.0, 0.0),
        rmats_dpsi=0.0, rmats_fdr=0.01,
    )


def test_thin_event_full_depth_is_exact_identity() -> None:
    """frac>=1.0 must return the counts unchanged (the f=1.0 identity anchor)."""
    ev = _event()
    out = thin_event(ev, 1.0, np.random.default_rng(0))
    assert out.ijc1 == ev.ijc1
    assert out.sjc1 == ev.sjc1
    assert out.ijc2 == ev.ijc2
    assert out.sjc2 == ev.sjc2


def test_thin_event_subsample_is_bounded_and_seed_deterministic() -> None:
    """Thinned counts never exceed the original and are reproducible per seed."""
    ev = _event()
    a = thin_event(ev, 0.5, np.random.default_rng(123))
    b = thin_event(ev, 0.5, np.random.default_rng(123))
    assert a.ijc1 == b.ijc1 and a.sjc2 == b.sjc2  # same seed -> identical
    for thinned, original in (
        (a.ijc1, ev.ijc1), (a.sjc1, ev.sjc1),
        (a.ijc2, ev.ijc2), (a.sjc2, ev.sjc2),
    ):
        for t, o in zip(thinned, original):
            assert 0 <= t <= o


def test_thin_event_binomial_in_expectation() -> None:
    """Sub-sampling a large count averages to frac * count (binomial mean)."""
    big = SEEvent(
        gene="G", chrom="chr1", strand="+", exon_start=1, exon_end=2,
        ijc1=(10000,), sjc1=(0,), ijc2=(0,), sjc2=(10000,),
        inc_form_len=100.0, skip_form_len=100.0,
        inclevel1=(1.0,), inclevel2=(0.0,), rmats_dpsi=1.0, rmats_fdr=0.0,
    )
    draws = [thin_event(big, 0.3, np.random.default_rng(k)).ijc1[0] for k in range(300)]
    assert abs(np.mean(draws) - 0.3 * 10000) < 60  # ~3 SE of Binomial(10000,0.3)


def test_recompute_inclevel_matches_rmats_length_normalised_psi() -> None:
    """IncLevel = (IJC/incLen) / (IJC/incLen + SJC/skipLen) per replicate."""
    il = _recompute_inclevel((40,), (10,), 148.0, 100.0)
    # ni = 40/148 = 0.27027, ns = 10/100 = 0.10 -> 0.27027 / 0.37027 = 0.72993
    assert il[0] == pytest.approx(0.72993, abs=1e-4)


def test_recompute_inclevel_drops_zero_over_zero_replicates() -> None:
    """A replicate with no inclusion and no skipping reads is dropped (NA)."""
    il = _recompute_inclevel((5, 0, 7), (3, 0, 1), 100.0, 100.0)
    assert len(il) == 2  # the middle 0/0 replicate is dropped


def test_thin_event_dpsi_follows_inclevel_difference_convention() -> None:
    """Recomputed ΔPSI must equal mean(IncLevel1) - mean(IncLevel2)."""
    ev = _event()
    out = thin_event(ev, 0.5, np.random.default_rng(7))
    expected = float(np.mean(out.inclevel1) - np.mean(out.inclevel2))
    assert out.rmats_dpsi == pytest.approx(expected)
    # group 1 is inclusion-heavy, group 2 exclusion-heavy -> positive ΔPSI
    assert out.rmats_dpsi > 0
