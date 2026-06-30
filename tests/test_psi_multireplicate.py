"""Regression tests for multi-replicate PSI replicate-count handling.

``braid psi`` multi-replicate mode previously used the BAM count as the
replicate count and silently fell back to the group-sum total when the rMATS
table carried fewer per-replicate counts. That duplicated group totals across
phantom replicates, inflating output counts and understating biological
variance. The fix caps the replicate count at the rMATS table's per-replicate
vector length and hard-fails on a BAM/table mismatch unless the caller opts into
an explicit degraded mode.
"""

import pytest

from braid.commands.psi import (
    _bootstrap_per_replicate,
    _combine_replicate_results,
    _resolve_effective_replicate_count,
)
from braid.target.rmats_bootstrap import RmatsEvent


def _event(inc_reps, exc_reps, group_inc=None, group_exc=None):
    """Build a minimal SE RmatsEvent with the given per-replicate vectors."""
    return RmatsEvent(
        event_id="chr1:SE:100:200:+",
        event_type="SE",
        chrom="chr1",
        strand="+",
        gene="GENEX",
        inc_count=sum(inc_reps) if inc_reps else (group_inc or 0),
        exc_count=sum(exc_reps) if exc_reps else (group_exc or 0),
        rmats_psi=0.5,
        rmats_fdr=1.0,
        rmats_dpsi=0.0,
        sample_1_inc_count=group_inc if group_inc is not None else sum(inc_reps),
        sample_1_exc_count=group_exc if group_exc is not None else sum(exc_reps),
        sample_1_inc_replicates=tuple(inc_reps),
        sample_1_exc_replicates=tuple(exc_reps),
    )


def test_mismatch_more_bams_than_table_reps_hard_fails():
    """2 BAMs but only 1 per-replicate count must be a hard error by default."""
    events = [_event((10,), (10,))]
    with pytest.raises(SystemExit):
        _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=False)


def test_mismatch_degrades_to_real_replicates_with_flag():
    """--allow-replicate-fallback caps at the real replicate count (no phantom)."""
    events = [_event((10,), (10,))]
    n_eff = _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=True)
    assert n_eff == 1


def test_matching_replicate_count_passes_through():
    events = [_event((10, 12), (5, 7))]
    assert _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=False) == 2


def test_no_per_replicate_counts_hard_fails():
    """Multi-replicate requested but the table has no per-replicate vectors."""
    events = [_event((), (), group_inc=20, group_exc=10)]
    with pytest.raises(SystemExit):
        _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=False)


def test_min_across_heterogeneous_events_is_used():
    """The cap is the *minimum* table vector length so no event is forced to fall back."""
    events = [_event((10, 12), (5, 7)), _event((9,), (4,))]
    with pytest.raises(SystemExit):
        _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=False)


def test_degraded_single_replicate_does_not_duplicate_group_sum():
    """End-to-end: capping at 1 real replicate must NOT double the output counts.

    This is the exact regression symptom — old behaviour produced combined_counts
    (20, 20) from a single (10, 10) replicate. With the cap, only the real
    replicate is bootstrapped and the combine step keeps its true counts.
    """
    events = [_event((10,), (10,))]
    n_eff = _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=True)
    rep_lists = [
        _bootstrap_per_replicate(
            events, i, n_replicates=50, confidence_level=0.95, seed=1 + i,
            conformal_calibrator=None,
        )
        for i in range(n_eff)
    ]
    combined = _combine_replicate_results(rep_lists, 0.95)
    assert len(combined) == 1
    assert combined[0].inclusion_count == 10
    assert combined[0].exclusion_count == 10


def test_fewer_bams_than_table_reps_hard_fails():
    """Mirror-image guard: 2 BAMs but 3 table replicates must hard-fail by default.

    Previously only the surplus direction (more BAMs than table reps) errored;
    the deficit direction silently bootstrapped only the first ``n_bams`` table
    replicates, undercounting reads/variance. Any BAM/table mismatch is now an
    error unless the caller opts into the degraded mode.
    """
    events = [_event((10, 20, 30), (5, 10, 15))]
    with pytest.raises(SystemExit):
        _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=False)


def test_fewer_bams_fallback_uses_all_table_replicates():
    """Degraded mode must use ALL table replicates, never a truncated prefix."""
    events = [_event((10, 20, 30), (5, 10, 15))]
    n_eff = _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=True)
    assert n_eff == 3


def test_fewer_bams_fallback_combines_full_group_total():
    """End-to-end deficit case: combined counts must be the full 60/30 table total.

    With 3 table replicates (10,20,30)/(5,10,15) and only 2 BAMs, the old code
    bootstrapped replicates 0 and 1 only, yielding combined counts 30/15 instead
    of the table's 60/30. Using all three table replicates restores the total.
    """
    events = [_event((10, 20, 30), (5, 10, 15))]
    n_eff = _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=True)
    rep_lists = [
        _bootstrap_per_replicate(
            events, i, n_replicates=50, confidence_level=0.95, seed=1 + i,
            conformal_calibrator=None,
        )
        for i in range(n_eff)
    ]
    combined = _combine_replicate_results(rep_lists, 0.95)
    assert len(combined) == 1
    assert combined[0].inclusion_count == 60
    assert combined[0].exclusion_count == 30


def test_unequal_inc_exc_lengths_do_not_double_count_in_fallback():
    """A row with len(IJC) != len(SJC) must not double-count via the group fallback.

    Residual of the missing-exclusion fix: the per-replicate completeness is
    ``min(len(inc), len(exc))``, but if the table-wide replicate count were taken
    from the inclusion vector alone, the surplus index would be drawn, fall back
    to the group-sum total, and ``_combine_replicate_results`` would then sum BOTH
    the per-replicate count AND the group total (e.g. 10 + 30 = 40 != 30). The
    table-wide count must use the same complete-replicate definition so that
    surplus index is never drawn.
    """
    # inc has 2 entries, exc has 1 -> the complete replicate count is 1.
    events = [_event((10, 20), (5,), group_inc=30, group_exc=5)]
    # Default: 2 BAMs vs 1 complete replicate must hard-fail.
    with pytest.raises(SystemExit):
        _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=False)
    # The complete replicate count is 1, not 2.
    n_eff = _resolve_effective_replicate_count(events, n_bams=2, allow_fallback=True)
    assert n_eff == 1
    rep_lists = [
        _bootstrap_per_replicate(
            events, i, n_replicates=50, confidence_level=0.95, seed=1 + i,
            conformal_calibrator=None,
        )
        for i in range(n_eff)
    ]
    combined = _combine_replicate_results(rep_lists, 0.95)
    # Uses only the single complete replicate (10/5); never the doubled 40.
    assert combined[0].inclusion_count == 10
    assert combined[0].exclusion_count == 5


def test_short_exclusion_vector_does_not_fabricate_zero_exclusion():
    """A replicate with inclusion but missing exclusion must NOT become exc=0.

    rMATS rows carry parallel IJC_SAMPLE_1 / SJC_SAMPLE_1 vectors. If the
    exclusion vector is shorter than the inclusion vector, the missing exclusion
    must not be read as an observed zero (which would push PSI toward 1 with
    false confidence). Such a replicate falls back to the group-sum total.
    """
    ev = _event((10, 20), (5,), group_inc=30, group_exc=5)
    # replicate_index 1 has inclusion (20) but no exclusion in the table vector.
    results = _bootstrap_per_replicate(
        [ev], replicate_index=1, n_replicates=50, confidence_level=0.95, seed=1,
        conformal_calibrator=None,
    )
    assert len(results) == 1
    # Must use the group-sum total (30/5), NOT the fabricated (20, 0).
    assert results[0].exclusion_count == 5
    assert results[0].inclusion_count == 30
