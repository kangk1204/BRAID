"""Tests for dashboard event-TSV schema validation.

Feeding a ``braid psi``/``braid differential`` TSV to the dashboard previously
raised a downstream ``KeyError 'gene_id'`` because the dashboard
addresses the ``analyze`` schema. ``load_events`` now fails fast with a clear
message naming the expected source command.
"""

import pandas as pd
import pytest

from braid.dashboard.data_loader import load_events, load_gtf_transcripts


def _write_tsv(tmp_path, df):
    p = tmp_path / "events.tsv"
    df.to_csv(p, sep="\t", index=False)
    return str(p)


def test_load_events_accepts_analyze_schema(tmp_path):
    path = _write_tsv(tmp_path, pd.DataFrame({
        "gene_id": ["G1"], "psi": [0.5], "event_type": ["SE"],
        "inclusion_count": [10], "exclusion_count": [10],
    }))
    out = load_events(path)
    assert list(out["gene_id"]) == ["G1"]
    assert out["psi"].iloc[0] == 0.5


def test_load_events_rejects_psi_output_schema(tmp_path):
    # `braid psi` dialect: gene / PSI (uppercase), no gene_id
    path = _write_tsv(tmp_path, pd.DataFrame({
        "gene": ["G1"], "PSI": [0.5], "event_type": ["SE"],
        "CI_low": [0.4], "CI_high": [0.6],
    }))
    with pytest.raises(ValueError, match="braid analyze"):
        load_events(path)


def test_load_gtf_transcripts_preserves_exon_first_records(tmp_path):
    """A later transcript feature must not erase exons already seen for that ID."""
    gtf = tmp_path / "exon_first.gtf"
    gtf.write_text(
        'chr1\tsrc\texon\t10\t20\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
        'chr1\tsrc\ttranscript\t5\t25\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n',
        encoding="utf-8",
    )

    out = load_gtf_transcripts(gtf)

    assert out.loc[0, "transcript_id"] == "T1"
    assert out.loc[0, "n_exons"] == 1
    assert out.loc[0, "exon_starts"] == [9]
    assert out.loc[0, "exon_ends"] == [20]


def test_event_table_download_neutralises_formula_injection():
    """The dashboard's filtered-events TSV download must neutralise formula leaders
    in string cells (it may render an arbitrary/attacker-influenced table)."""
    from braid.dashboard.components.event_table import _csv_safe_frame
    df = pd.DataFrame(
        {"gene": ["=EVIL", "@GENE", "+chr1", "normal"], "psi": [0.5, 0.6, 0.7, 0.8]}
    )
    out = _csv_safe_frame(df)
    assert out["gene"].tolist() == ["'=EVIL", "'@GENE", "'+chr1", "normal"]
    assert out["psi"].tolist() == [0.5, 0.6, 0.7, 0.8]   # numeric untouched
    tsv = out.to_csv(index=False, sep="\t")
    assert not any(
        cell[:1] in "=+-@"
        for line in tsv.splitlines()[1:]
        for cell in line.split("\t")
    )


def test_event_table_download_neutralises_pandas_string_dtype_columns():
    """csv-safety must cover pandas extension ``string`` dtype columns, not only
    ``object`` dtype. A StringDtype column's dtype is not ``object``, so an
    object-only guard would skip it and pass a formula-injection payload straight
    into the download. Locks the is_string_dtype path."""
    from braid.dashboard.components.event_table import _csv_safe_frame
    df = pd.DataFrame(
        {
            "gene": pd.array(["=EVIL", "@GENE", "+chr1", "normal"], dtype="string"),
            "psi": [0.5, 0.6, 0.7, 0.8],
        }
    )
    # Guard: the column must genuinely be the non-object string dtype, otherwise
    # this test would silently stop exercising the path a01cd35 added.
    assert str(df["gene"].dtype) == "string"
    assert df["gene"].dtype != object
    out = _csv_safe_frame(df)
    assert out["gene"].tolist() == ["'=EVIL", "'@GENE", "'+chr1", "normal"]
    assert out["psi"].tolist() == [0.5, 0.6, 0.7, 0.8]   # numeric untouched


def test_get_gene_list_tolerates_missing_labels():
    """A gene_id column with a blank/NaN cell must not crash the selector sort:
    mixing float(nan) with str makes sorted() raise TypeError. NaN is dropped and
    values are sorted as strings; a missing column returns []."""
    from braid.dashboard.data_loader import get_gene_list
    df = pd.DataFrame({"gene_id": ["GENEB", None, "GENEA"], "psi": [0.1, 0.2, 0.3]})
    assert get_gene_list(df) == ["GENEA", "GENEB"]
    assert get_gene_list(pd.DataFrame({"psi": [0.1]})) == []


def test_get_event_type_list_tolerates_missing_labels():
    """The Event Explorer event-type filter must tolerate a NaN event_type
    (otherwise sorted() raises TypeError comparing float and str and crashes the
    table). NaN is dropped and the rest sorted as strings; no column returns []."""
    from braid.dashboard.data_loader import get_event_type_list
    df = pd.DataFrame({"event_type": ["SE", None, "RI"], "gene_id": ["A", "B", "C"]})
    assert get_event_type_list(df) == ["RI", "SE"]
    assert get_event_type_list(pd.DataFrame({"x": [1]})) == []


def test_filter_by_event_types_matches_numeric_event_type():
    """get_event_type_list stringifies the options, so the Event Explorer filter must
    compare event_type coerced to str. A numeric event_type column must not silently
    filter to 0 rows (the regression that the str-coercing options would otherwise
    introduce against a raw int column)."""
    from braid.dashboard.data_loader import (
        filter_by_event_types,
        get_event_type_list,
    )
    df = pd.DataFrame({"event_type": [1, 2, 1], "gene_id": ["A", "B", "C"]})
    options = get_event_type_list(df)  # ["1", "2"]
    assert filter_by_event_types(df, options).shape[0] == 3   # all retained, not 0
    assert filter_by_event_types(df, ["1"]).shape[0] == 2     # selective subset works
    # string event_type still works; empty selection / missing column pass through
    sdf = pd.DataFrame({"event_type": ["SE", "RI"], "gene_id": ["A", "B"]})
    assert filter_by_event_types(sdf, get_event_type_list(sdf)).shape[0] == 2
    assert filter_by_event_types(df, []).shape[0] == 3
    assert filter_by_event_types(pd.DataFrame({"x": [1]}), ["1"]).shape[0] == 1
