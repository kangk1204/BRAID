#!/usr/bin/env python3
"""Prepare metadata and truth tables for public RNA-seq/qRT-PCR benchmarks.

This script normalizes the benchmark metadata gathered from GEO/SRA and
extracts RT-PCR validation tables from the SUPPA2 supplementary workbook.

It does not download raw RNA-seq reads. The output is a set of compact TSV/JSON
files under ``data/public_benchmarks/meta`` that downstream download and
analysis steps can consume.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "public_benchmarks" / "meta"
SUPPA_XLSX = OUT_DIR / "suppa2_tables.xlsx"


def _find_header_row(df: pd.DataFrame, first_cell: str) -> int:
    for idx, value in enumerate(df.iloc[:, 0].tolist()):
        if str(value).strip() == first_cell:
            return idx
    raise ValueError(f"Header row starting with {first_cell!r} not found")


def _read_sheet(sheet_name: str, first_cell: str = "Source") -> pd.DataFrame:
    raw = pd.read_excel(SUPPA_XLSX, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(raw, first_cell)
    header = [
        str(value).strip()
        for value in raw.iloc[header_row].tolist()
    ]
    data = raw.iloc[header_row + 1:].copy()
    data.columns = header
    data = data.dropna(how="all").reset_index(drop=True)
    return data


def _write_tsv(df: pd.DataFrame, name: str) -> str:
    path = OUT_DIR / name
    df.to_csv(path, sep="\t", index=False)
    return str(path.relative_to(ROOT))


def build_truth_tables() -> dict:
    tra2_positive = _read_sheet("Table S4")
    tra2_positive = tra2_positive.loc[
        tra2_positive["Source"].astype(str).eq("RT-PCR"),
        ["Gene", "chr", "exon_start", "exon_end", "strand", "deltaPSI", "Event_id"],
    ].copy()
    tra2_positive.columns = [
        "gene",
        "chrom",
        "exon_start",
        "exon_end",
        "strand",
        "delta_psi_rtpcr",
        "suppa_event_id",
    ]

    tra2_negative = _read_sheet("Table S10")
    tra2_negative = tra2_negative.loc[
        tra2_negative["Source"].astype(str).eq("RT-PCR_negative"),
        ["Gene_symbol", "Chr", "Exon_start", "Exon_end"],
    ].copy()
    tra2_negative.columns = ["gene", "chrom", "exon_start", "exon_end"]
    tra2_negative["delta_psi_rtpcr"] = 0.0

    mouse_positive = _read_sheet("Table S6")
    mouse_positive = mouse_positive.loc[
        mouse_positive["Source"].astype(str).eq("RT-PCR"),
        [
            "Gene ID",
            "chr",
            "strand",
            "InclusionJunction",
            "ExclusionJunction",
            "deltaPSI",
        ],
    ].copy()
    mouse_positive.columns = [
        "gene",
        "chrom",
        "strand",
        "inclusion_junction",
        "exclusion_junction",
        "delta_psi_rtpcr",
    ]

    outputs = {
        "gse59335_positive_tsv": _write_tsv(
            tra2_positive,
            "gse59335_tra2_positive_events.tsv",
        ),
        "gse59335_negative_tsv": _write_tsv(
            tra2_negative,
            "gse59335_tra2_negative_events.tsv",
        ),
        "gse54651_positive_tsv": _write_tsv(
            mouse_positive,
            "gse54651_circadian_positive_events.tsv",
        ),
        "gse59335_positive_count": int(len(tra2_positive)),
        "gse59335_negative_count": int(len(tra2_negative)),
        "gse54651_positive_count": int(len(mouse_positive)),
    }
    return outputs


def _truth_file_entry(name: str, *, canonical: bool, note: str) -> dict:
    """Build a ``truth_tables`` entry that reflects the meta file's actual state.

    These tables previously stayed hardcoded as
    ``pending_direct_extraction_from_paper_supplement`` even after the files were
    produced. We now report what is on disk and whether it is a canonical,
    manuscript-cited truth set or analysis-only.
    """
    rel = f"data/public_benchmarks/meta/{name}"
    if not (OUT_DIR / name).exists():
        return {"status": "pending_direct_extraction_from_paper_supplement"}
    return {
        "status": "available" if canonical else "analysis_only_not_canonical",
        "positive": rel,
        "note": note,
    }


def build_dataset_metadata(truth_tables: dict) -> list[dict]:
    return [
        {
            "priority": 1,
            "dataset_id": "GSE59335_TRA2",
            "species": "human",
            "coordinate_build": "hg19",
            "evidence": [
                "SUPPA2 benchmark uses GSE59335 for TRA2A/B knockdown in MDA-MB-231",
                "GEO supplementary filenames explicitly include hg19",
            ],
            "study_accessions": {
                "geo": "GSE59335",
                "sra": "SRP044265",
                "bioproject": "PRJNA255099",
            },
            "conditions": {
                "control": ["SRR1513329", "SRR1513330", "SRR1513331"],
                "knockdown": ["SRR1513332", "SRR1513333", "SRR1513334"],
            },
            "truth_tables": {
                "positive": truth_tables["gse59335_positive_tsv"],
                "negative": truth_tables["gse59335_negative_tsv"],
            },
        },
        {
            "priority": 2,
            "dataset_id": "GSE54651_CIRCADIAN",
            "species": "mouse",
            "coordinate_build": "mm10",
            "evidence": [
                "SUPPA2 methods state mouse datasets were mapped to mm10",
                "Benchmark uses cerebellum vs liver at CT28/CT40/CT52",
            ],
            "study_accessions": {
                "geo": "GSE54651",
                "sra": "SRP036186",
                "bioproject": "PRJNA237293",
            },
            "conditions": {
                "cerebellum": ["SRR1158546", "SRR1158548", "SRR1158550"],
                "liver": ["SRR1158578", "SRR1158580", "SRR1158582"],
            },
            "truth_tables": {
                "positive": truth_tables["gse54651_positive_tsv"],
            },
        },
        {
            "priority": 3,
            "dataset_id": "GSE55215_QKI",
            "species": "human",
            "coordinate_build": "hg19",
            "evidence": [
                "QKI paper methods state reads were aligned to the human genome hg19",
                "GEO/SRA expose four RNA-seq runs covering sh-Ctrl and sh-QKI",
            ],
            "study_accessions": {
                "geo": "GSE55215",
                "sra": "SRP038702",
                "bioproject": "PRJNA238897",
            },
            "conditions": {
                "control": ["SRR1173996", "SRR1173997"],
                "knockdown": ["SRR1173998", "SRR1173999"],
            },
            "truth_tables": _truth_file_entry(
                "qki_positive_events.tsv",
                canonical=False,
                note=(
                    "QKI RT-PCR section was removed from the manuscript (rMATS FPR "
                    "did not reproduce on GRCh38). Retained for exploratory analysis "
                    "only; NOT a canonical, manuscript-cited truth set."
                ),
            ),
        },
        {
            "priority": 4,
            "dataset_id": "SRS354082_RMATS",
            "species": "human",
            "coordinate_build": "hg19",
            "evidence": [
                "rMATS paper methods state reads were mapped to hg19",
                "Paper reports 34 RT-PCR validated SE events with 32/34 confirmed",
            ],
            "study_accessions": {
                "sra_sample": "SRS354082",
            },
            "conditions": {
                "PC3E": ["SRR536348", "SRR536350", "SRR536352"],
                "GS689.Li": ["SRR536342", "SRR536344", "SRR536346"],
            },
            "truth_tables": _truth_file_entry(
                "rmats_pc3e_gs689_positive_events.tsv",
                canonical=True,
                note=(
                    "Cassette-exon RT-PCR positive events; the PC3E/GS689 "
                    "head-to-head coverage dataset "
                    "(benchmarks/headtohead/comprehensive_benchmark.py). "
                    "Coordinates hg19; ΔPSI orientation PC3E - GS689."
                ),
            ),
        },
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    truth_tables = build_truth_tables()
    metadata = build_dataset_metadata(truth_tables)

    summary = {
        "truth_tables": truth_tables,
        "datasets": metadata,
    }

    out_json = OUT_DIR / "public_validation_datasets.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {out_json.relative_to(ROOT)}")
    print(
        "Prepared truth tables:",
        truth_tables["gse59335_positive_count"],
        "TRA2 positives,",
        truth_tables["gse59335_negative_count"],
        "TRA2 negatives,",
        truth_tables["gse54651_positive_count"],
        "mouse positives",
    )


if __name__ == "__main__":
    main()
