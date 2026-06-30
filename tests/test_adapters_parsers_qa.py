"""QA edge-case regression tests for the `braid filter` caller adapters.

Found by adversarial probing of braid/adapters/parsers.py (the filter-CLI path; not
on any benchmark/paper-number path). Three silent-wrong-output bugs:

* SUPPA2: a single trailing tab on the first data row tripped the unnamed-id-column
  heuristic, shifting EVERY column and reading ΔPSI from the p-value column.
* MAJIQ: a corrupt probability outside [0, 1] mapped to pvalue = 1 - P < 0, silently
  flagging the LSV as caller-significant.
* betAS: an inverted native interval (upper < lower) was carried forward with a
  negative width.
"""
from __future__ import annotations

from pathlib import Path

from braid.adapters.parsers import parse_betas, parse_majiq, parse_suppa2


def _w(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


# --- SUPPA2: trailing-tab must not shift columns -----------------------------


def test_suppa2_trailing_tab_on_first_row_does_not_corrupt_dpsi(tmp_path):
    """Named-id header (3 cols) + a trailing tab on row 1 must still read ΔPSI from
    the dPSI column, not the p-value column."""
    content = (
        "event_id\tCtrl-KD_dPSI\tCtrl-KD_p-val\n"
        "ENSG;SE:chr1:1-2:+\t0.50\t0.01\t\n"   # trailing tab -> 4 fields
        "ENSG2;SE:chr1:3-4:+\t0.30\t0.02\n"
    )
    events = parse_suppa2(_w(tmp_path, "s.dpsi", content))
    by_id = {e.event_id: e for e in events}
    assert by_id["ENSG;SE:chr1:1-2:+"].dpsi == 0.50
    assert by_id["ENSG;SE:chr1:1-2:+"].pvalue == 0.01
    assert by_id["ENSG2;SE:chr1:3-4:+"].dpsi == 0.30


def test_suppa2_genuinely_unnamed_id_column_still_works(tmp_path):
    """The intended unnamed-id case (header 2 cols, data 3 fields) must still parse."""
    content = (
        "Ctrl-KD_dPSI\tCtrl-KD_p-val\n"
        "ENSG;SE:chr1:1-2:+\t0.50\t0.01\n"
    )
    events = parse_suppa2(_w(tmp_path, "s2.dpsi", content))
    assert len(events) == 1
    assert events[0].dpsi == 0.50
    assert events[0].pvalue == 0.01


# --- MAJIQ: probability must be in [0, 1] ------------------------------------


def test_majiq_out_of_domain_probability_does_not_flag_significant(tmp_path):
    """P = 1.8 must NOT become pvalue = -0.8 (which is < every threshold)."""
    content = (
        "gene_name\tlsv_id\tE(dPSI)\tP(|dPSI|>=0.20)\n"
        "GENE\tLSV1\t0.50\t1.80\n"
    )
    events = parse_majiq(_w(tmp_path, "m.tsv", content))
    assert len(events) == 1
    assert events[0].dpsi == 0.50
    assert events[0].pvalue is None  # corrupt probability -> no significance call


def test_majiq_valid_probability_maps_to_pvalue(tmp_path):
    """A valid P in [0, 1] still maps to pvalue = 1 - P."""
    content = (
        "gene_name\tlsv_id\tE(dPSI)\tP(|dPSI|>=0.20)\n"
        "GENE\tLSV1\t0.50\t0.97\n"
    )
    events = parse_majiq(_w(tmp_path, "m2.tsv", content))
    assert abs(events[0].pvalue - 0.03) < 1e-9


# --- betAS: inverted / partial native interval -------------------------------


def test_betas_inverted_interval_is_dropped_not_stored(tmp_path):
    """upper < lower is corrupt: no std, and no negative-width comparison interval."""
    content = (
        "event_id\tdpsi\tlower\tupper\n"
        "E1\t0.4\t0.6\t0.2\n"   # inverted
    )
    events = parse_betas(_w(tmp_path, "b.tsv", content))
    assert events[0].sampling_std is None
    assert events[0].caller_low is None
    assert events[0].caller_high is None


def test_betas_valid_interval_yields_std_and_bounds(tmp_path):
    content = (
        "event_id\tdpsi\tlower\tupper\n"
        "E1\t0.4\t0.2\t0.6\n"
    )
    events = parse_betas(_w(tmp_path, "b2.tsv", content))
    assert events[0].caller_low == 0.2
    assert events[0].caller_high == 0.6
    assert events[0].sampling_std is not None and events[0].sampling_std > 0


def test_betas_zero_width_interval_is_valid(tmp_path):
    content = (
        "event_id\tdpsi\tlower\tupper\n"
        "E1\t0.4\t0.3\t0.3\n"
    )
    events = parse_betas(_w(tmp_path, "b3.tsv", content))
    assert events[0].sampling_std == 0.0
    assert events[0].caller_low == 0.3 and events[0].caller_high == 0.3


def test_suppa2_no_placeholder_header_with_trailing_tabs_still_parses(tmp_path):
    """No-placeholder SUPPA2 (unnamed id column) with a trailing tab on BOTH header and
    data row (an Excel round-trip pattern) must still parse. The earlier trailing-tab fix
    trimmed only the data side and compared to the raw header, so trimmed_data == raw_header
    made id_in_data False and every row was silently dropped (0 events). The symmetric trim
    (both sides) fixes this without regressing the named-id case."""
    content = (
        "Ctrl-KD_dPSI\tCtrl-KD_p-val\t\n"                 # header + trailing tab
        "ENSG;SE:chr1:1-2:+\t0.50\t0.01\t\n"               # data + trailing tab
        "ENSG2;SE:chr1:3-4:+\t0.30\t0.02\t\n"
    )
    events = parse_suppa2(_w(tmp_path, "s_notab.dpsi", content))
    by_id = {e.event_id: e for e in events}
    assert len(events) == 2
    assert by_id["ENSG;SE:chr1:1-2:+"].dpsi == 0.50
    assert by_id["ENSG;SE:chr1:1-2:+"].pvalue == 0.01
    assert by_id["ENSG2;SE:chr1:3-4:+"].dpsi == 0.30
