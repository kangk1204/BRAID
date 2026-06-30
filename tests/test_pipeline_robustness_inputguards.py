"""Input-guard robustness for the rMATS calibration pipeline.

Covers two verified gaps left by the earlier robustness pass, using the same
``S{stage}_{failure-mode}`` convention as the other ``test_pipeline_robustness*``
suites:

  S3 = rMATS table parse (``braid/target/rmats_bootstrap.py``)
  S5 = conformal calibrator (``braid/target/conformal.py``)

* S3 x A6 -- a negative junction count (corrupted/edited rMATS row) must be
  rejected at the parse boundary. Left unguarded it reaches ``rng.beta(a<=0)`` and
  aborts the whole ``braid differential`` run with an opaque ``a <= 0``.
* S5 x A5/A6 -- a custom calibrator JSON with a ``bin_edges`` length other than 4
  (the fixed 5-label support scheme) crashed ``q_for`` with ``IndexError`` on
  high-support events; an out-of-range ``alpha`` was accepted silently.
"""
from __future__ import annotations

import numpy as np
import pytest

from braid.adapters.base import confidence_tier
from braid.target import conformal as C
from braid.target.psi_bootstrap import classify_support_bin
from braid.target.rmats_bootstrap import (
    _parse_count_vector,
    _parse_float_na,
    parse_rmats_output,
)

_SE_HEADER = [
    "ID", "GeneID", "geneSymbol", "chr", "strand", "exonStart_0base", "exonEnd",
    "upstreamES", "upstreamEE", "downstreamES", "downstreamEE",
    "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
]


def _write_se_table(directory, ijc1: str) -> None:
    """Write a one-row SE.MATS.JC.txt whose IJC_SAMPLE_1 vector is *ijc1*."""
    row = [
        "1", "GENE", "GENE", "chr1", "+", "100", "200", "0", "100", "200", "300",
        ijc1, "40,40,40", "60,60,60", "20,20,20",
        "100", "100", "0.01", "0.01", "0.6,0.6,0.6", "0.8,0.8,0.8", "-0.2",
    ]
    table = directory / "SE.MATS.JC.txt"
    table.write_text("\t".join(_SE_HEADER) + "\n" + "\t".join(row) + "\n")


# --- S3 x A6: out-of-range (negative) junction counts -----------------------


def test_S3_A6_negative_count_vector_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        _parse_count_vector("-5,30,30")


def test_S3_A6_nonnegative_count_vector_still_parses():
    assert _parse_count_vector("5,30,30") == (5, 30, 30)
    assert _parse_count_vector("0,0,0") == (0, 0, 0)


def test_S3_A6_negative_count_row_skipped_nonstrict(tmp_path):
    """Non-strict parse drops the corrupted row rather than keeping a negative count."""
    d = tmp_path / "rmats"
    d.mkdir()
    _write_se_table(d, ijc1="-5,30,30")
    events = parse_rmats_output(str(d), min_total_count=1, strict=False)
    assert events == []


def test_S3_A6_negative_count_row_fails_fast_strict(tmp_path):
    """Strict parse names the corrupted row instead of aborting deep in rng.beta."""
    d = tmp_path / "rmats"
    d.mkdir()
    _write_se_table(d, ijc1="-5,30,30")
    with pytest.raises(ValueError, match="negative"):
        parse_rmats_output(str(d), min_total_count=1, strict=True)


def test_S3_A6_clean_table_unaffected(tmp_path):
    """The guard does not perturb a well-formed (non-negative) table."""
    d = tmp_path / "rmats"
    d.mkdir()
    _write_se_table(d, ijc1="10,30,30")
    events = parse_rmats_output(str(d), min_total_count=1, strict=False)
    assert len(events) == 1
    assert events[0].sample_1_inc_count == 70


# --- S5 x A5: custom calibrator bin_edges length ----------------------------


def test_S5_A5_assign_support_bins_rejects_mismatched_edges():
    with pytest.raises(ValueError, match="bin edges"):
        C.assign_support_bins(np.array([300.0]), edges=(10, 20, 30, 40, 50))


def test_S5_A5_assign_support_bins_default_edges_ok():
    out = C.assign_support_bins(np.array([5.0, 300.0]))
    assert list(out) == ["<20", "250+"]


def test_S5_A5_calibrator_from_dict_rejects_mismatched_bin_edges():
    with pytest.raises(ValueError, match="bin_edges"):
        C.ConformalCalibrator.from_dict(
            {
                "alpha": 0.05,
                "q_global": 0.3,
                "q_by_bin": {},
                "bin_edges": [10, 20, 30, 40, 50],
                "scale_kind": "posterior_std",
            }
        )


def test_S5_A5_calibrator_from_dict_default_edges_loads_and_q_for_finite():
    cal = C.ConformalCalibrator.from_dict(
        {
            "alpha": 0.05,
            "q_global": 0.3,
            "q_by_bin": {"<20": 0.3, "250+": 0.2},
            "bin_edges": [20, 50, 100, 250],
            "scale_kind": "posterior_std",
        }
    )
    # The crash was here: high support -> top bin lookup.
    assert np.isfinite(cal.q_for(300.0))


# --- S5 x A6: calibrator alpha out of range ---------------------------------


@pytest.mark.parametrize("bad_alpha", [-0.5, 0.0, 1.0, 5.0])
def test_S5_A6_calibrator_alpha_out_of_range_rejected(bad_alpha):
    with pytest.raises(ValueError, match="alpha"):
        C.ConformalCalibrator.from_dict(
            {
                "alpha": bad_alpha,
                "q_global": 0.3,
                "q_by_bin": {},
                "scale_kind": "posterior_std",
            }
        )


# --- S5 x A4/A6: bin_edges that pass the length check but are still malformed ----
# A 4-entry bin_edges can still be non-numeric or non-monotonic; both must be caught
# at load time (the --calibration boundary), not crash later inside ``q_for`` ->
# ``assign_support_bins`` -> ``np.digitize``.


def test_S5_A4_calibrator_from_dict_rejects_non_numeric_bin_edges():
    with pytest.raises(ValueError, match="bin_edges"):
        C.ConformalCalibrator.from_dict(
            {
                "alpha": 0.05,
                "q_global": 0.3,
                "q_by_bin": {},
                "bin_edges": ["a", "b", "c", "d"],
                "scale_kind": "posterior_std",
            }
        )


@pytest.mark.parametrize("edges", [[100, 20, 50, 250], [20, 20, 50, 100], [250, 100, 50, 20]])
def test_S5_A6_calibrator_from_dict_rejects_non_monotonic_bin_edges(edges):
    with pytest.raises(ValueError, match="bin_edges"):
        C.ConformalCalibrator.from_dict(
            {
                "alpha": 0.05,
                "q_global": 0.3,
                "q_by_bin": {},
                "bin_edges": edges,
                "scale_kind": "posterior_std",
            }
        )


def test_S5_A6_calibrator_from_dict_rejects_nonfinite_bin_edges():
    with pytest.raises(ValueError, match="bin_edges"):
        C.ConformalCalibrator.from_dict(
            {
                "alpha": 0.05,
                "q_global": 0.3,
                "q_by_bin": {},
                "bin_edges": [20, 50, float("inf"), 250],
                "scale_kind": "posterior_std",
            }
        )


def test_S5_A4_numeric_string_bin_edges_are_coerced_not_stored_as_str():
    """JSON numeric-string edges (e.g. ["20", ...]) coerce float() so they pass the
    numeric/monotonic checks; they must be STORED coerced, else np.digitize crashes at
    q_for time on the string array."""
    cal = C.ConformalCalibrator.from_dict(
        {
            "alpha": 0.05,
            "q_global": 0.3,
            "q_by_bin": {"<20": 0.3, "250+": 0.2},
            "bin_edges": ["20", "50", "100", "250"],
            "scale_kind": "posterior_std",
        }
    )
    assert all(not isinstance(e, str) for e in cal.bin_edges)
    assert np.isfinite(cal.q_for(300.0))  # previously raised TypeError
    assert tuple(cal.to_dict()["bin_edges"]) == (20, 50, 100, 250)


def test_S5_valid_increasing_bin_edges_round_trip_and_q_for():
    """A valid 4-edge calibrator still loads, computes q_for, and round-trips."""
    cal = C.ConformalCalibrator.from_dict(
        {
            "alpha": 0.05,
            "q_global": 0.3,
            "q_by_bin": {"<20": 0.3, "250+": 0.2},
            "bin_edges": [20, 50, 100, 250],
            "scale_kind": "posterior_std",
        }
    )
    assert np.isfinite(cal.q_for(300.0))
    assert tuple(cal.to_dict()["bin_edges"]) == (20, 50, 100, 250)


# ---------------------------------------------------------------------------
# Review-pass fixes (2026-06-26): findings from the multi-perspective review.
# S4 = PSI posterior binning; S5 = conformal calibrator; S6 = confidence tier.
# ---------------------------------------------------------------------------


# --- S4 x A4: classify_support_bin must handle non-integer totals -----------
# Form-length normalization makes inc + exc*ratio a float, so a total in a unit
# gap (19<t<20 etc.) must NOT misroute to the least-conservative "250+" bin.


@pytest.mark.parametrize(
    "total, expected",
    [
        (19.5, "<20"), (19, "<20"), (0.0, "<20"), (-5.0, "<20"),
        (20.0, "20-49"), (49.5, "20-49"),
        (50.0, "50-99"), (99.5, "50-99"),
        (100.0, "100-249"), (249.5, "100-249"),
        (250.0, "250+"), (300.0, "250+"),
    ],
)
def test_S4_A4_classify_support_bin_float_total_routes_correctly(total, expected):
    assert classify_support_bin(total) == expected


def test_S4_A4_gap_floats_never_route_to_top_bin():
    # The exact regression: every unit-gap float used to return "250+".
    for gap_total in (19.5, 49.5, 99.5, 249.5):
        assert classify_support_bin(gap_total) != "250+"


# --- S6 x consistency: confidence_tier None vs bool branch agree ------------
# (reliable=True, effect=False) must map to the SAME tier whether the caller
# reports significance (False) or not (None) -- previously None gave "supported".


def test_S6_tier_reliable_no_effect_consistent_across_none_and_false():
    assert confidence_tier(True, False, None) == "not-significant"
    assert confidence_tier(True, False, False) == "not-significant"


def test_S6_tier_supported_still_requires_effect():
    # "supported" must require a reliable *effect*, in both branches.
    assert confidence_tier(True, True, None) == "high-confidence"
    assert confidence_tier(True, True, False) == "supported"
    assert confidence_tier(True, False, None) != "supported"


def test_S6_tier_unreliable_none_still_not_reliable():
    # The !reliable + None case keeps its distinct "not-reliable" label.
    assert confidence_tier(False, True, None) == "not-reliable"
    assert confidence_tier(False, False, None) == "not-reliable"


# --- S3 x A10: _parse_float_na rejects non-finite, mirroring inc-level parse -


def test_S3_A10_parse_float_na_rejects_infinity():
    assert np.isnan(_parse_float_na("inf"))
    assert np.isnan(_parse_float_na("-inf"))
    assert np.isnan(_parse_float_na("Infinity"))


def test_S3_A10_parse_float_na_keeps_finite_values():
    assert _parse_float_na("0.05") == 0.05
    assert _parse_float_na("-0.6") == -0.6
    assert np.isnan(_parse_float_na("NA"))


# --- S5 x E1: from_json caps file size before reading -----------------------


def test_S5_E1_from_json_rejects_oversized_file(tmp_path):
    big = tmp_path / "huge_calibrator.json"
    # Write just over the 10 MB cap so stat() trips before any parse.
    big.write_bytes(b"0" * (C.ConformalCalibrator._MAX_CALIBRATION_BYTES + 1))
    with pytest.raises(ValueError, match="exceeding|limit"):
        C.ConformalCalibrator.from_json(big)


def test_S5_E1_from_json_loads_normal_sized_file(tmp_path):
    cal = C.ConformalCalibrator(
        alpha=0.05, q_global=0.3, q_by_bin={"<20": 0.3}, scale_kind="posterior_std",
    )
    path = tmp_path / "ok.json"
    cal.to_json(path)
    loaded = C.ConformalCalibrator.from_json(path)
    assert loaded.q_global == 0.3
