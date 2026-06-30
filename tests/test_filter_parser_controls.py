from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pytest

from braid.adapters.base import CallerEvent
from braid.adapters.parsers import ParserConfig, ParseResult, parse_betas, parse_suppa2
from braid.commands.filter_cmd import _flip_event_sign, run_filter


def _w(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def test_suppa2_contrast_selects_matching_dpsi_and_pvalue(tmp_path: Path) -> None:
    path = _w(
        tmp_path,
        "diff.dpsi",
        (
            "A-B_dPSI\tA-B_p-val\tC-D_dPSI\tC-D_p-val\n"
            "ENSG;SE:chr1:1-2:+\t0.30\t0.01\t0.90\t0.50\n"
        ),
    )

    parsed = parse_suppa2(
        path, config=ParserConfig(contrast="C-D"), return_summary=True
    )

    assert isinstance(parsed, ParseResult)
    assert parsed.events[0].dpsi == pytest.approx(0.90)
    assert parsed.events[0].pvalue == pytest.approx(0.50)
    assert parsed.summary.contrast == "C-D"
    assert parsed.summary.dpsi_column == "C-D_dPSI"
    assert parsed.summary.pvalue_column == "C-D_p-val"


def test_suppa2_dpsi_override_infers_same_contrast_pvalue(tmp_path: Path) -> None:
    path = _w(
        tmp_path,
        "diff.dpsi",
        (
            "A-B_dPSI\tA-B_p-val\tC-D_dPSI\tC-D_p-val\n"
            "ENSG;SE:chr1:1-2:+\t0.30\t0.01\t0.90\t0.50\n"
        ),
    )

    events = parse_suppa2(path, config=ParserConfig(dpsi_column="C-D_dPSI"))

    assert events[0].dpsi == pytest.approx(0.90)
    assert events[0].pvalue == pytest.approx(0.50)


def test_column_override_raises_when_header_is_absent(tmp_path: Path) -> None:
    path = _w(
        tmp_path,
        "diff.dpsi",
        "A-B_dPSI\tA-B_p-val\nENSG;SE:chr1:1-2:+\t0.30\t0.01\n",
    )

    with pytest.raises(ValueError, match="missing_dpsi"):
        parse_suppa2(path, config=ParserConfig(dpsi_column="missing_dpsi"))


def test_betas_column_overrides_map_nonstandard_headers(tmp_path: Path) -> None:
    path = _w(
        tmp_path,
        "betas.tsv",
        (
            "gene_custom\tid_custom\tdelta_custom\tq_custom\treads_custom\n"
            "GENE\tEV1\t0.42\t0.04\t123\n"
        ),
    )

    parsed = parse_betas(
        path,
        config=ParserConfig(
            gene_column="gene_custom",
            event_id_column="id_custom",
            dpsi_column="delta_custom",
            fdr_column="q_custom",
            support_column="reads_custom",
        ),
        return_summary=True,
    )

    assert isinstance(parsed, ParseResult)
    ev = parsed.events[0]
    assert ev.gene == "GENE"
    assert ev.event_id == "EV1"
    assert ev.dpsi == pytest.approx(0.42)
    assert ev.fdr == pytest.approx(0.04)
    assert ev.total_support == pytest.approx(123.0)
    assert parsed.summary.dpsi_column == "delta_custom"
    assert parsed.summary.fdr_column == "q_custom"


def test_flip_sign_mirrors_native_interval_and_swaps_group_psi() -> None:
    ev = CallerEvent(
        event_id="E1",
        gene="G",
        event_type="SE",
        dpsi=0.40,
        total_support=50.0,
        caller="betas",
        group1_psi=0.70,
        group2_psi=0.30,
        caller_low=0.20,
        caller_high=0.60,
    )

    flipped = _flip_event_sign(ev)

    assert flipped.dpsi == pytest.approx(-0.40)
    assert flipped.group1_psi == pytest.approx(0.30)
    assert flipped.group2_psi == pytest.approx(0.70)
    assert flipped.caller_low == pytest.approx(-0.60)
    assert flipped.caller_high == pytest.approx(-0.20)


def test_filter_verbose_summary_and_flip_sign_reach_tsv(tmp_path: Path, capsys) -> None:
    path = _w(
        tmp_path,
        "diff.dpsi",
        "Ctrl-KD_dPSI\tCtrl-KD_p-val\nENSG;SE:chr1:1-2:+\t0.50\t0.01\n",
    )
    out = tmp_path / "filter_out"
    args = argparse.Namespace(
        input=path,
        caller="suppa2",
        example=False,
        output=str(out),
        effect_cutoff=0.1,
        sig_threshold=0.05,
        min_support=20,
        strict=False,
        calibration=None,
        top_n=5,
        make_figure=False,
        verbose=True,
        flip_sign=True,
        contrast="Ctrl-KD",
        dpsi_column=None,
        pvalue_column=None,
        fdr_column=None,
        support_column=None,
        event_id_column=None,
        gene_column=None,
    )

    run_filter(args)

    stdout = capsys.readouterr().out
    assert "Parse summary:" in stdout
    assert "dpsi_column=Ctrl-KD_dPSI" in stdout
    assert "pvalue_column=Ctrl-KD_p-val" in stdout
    assert "sign=flipped" in stdout
    assert "support=unknown -> global calibrator" in stdout

    with open(f"{out}.tsv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert float(rows[0]["dpsi"]) == pytest.approx(-0.50)
