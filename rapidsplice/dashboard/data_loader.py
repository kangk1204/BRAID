"""Data loading utilities for the dashboard.

Loads events TSV and GTF files into pandas DataFrames for visualization.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_events(events_path: str | Path) -> pd.DataFrame:
    """Load events TSV into a DataFrame.

    Args:
        events_path: Path to the events TSV file.

    Returns:
        DataFrame with typed columns for event data.
    """
    df = pd.read_csv(events_path, sep="\t")

    # Convert numeric columns
    numeric_cols = [
        "inclusion_count", "exclusion_count", "total_reads",
        "ci_low", "ci_high", "ci_width", "confidence_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert PSI (may contain "NA")
    if "psi" in df.columns:
        df["psi"] = pd.to_numeric(df["psi"], errors="coerce")

    return df


def load_gtf_transcripts(gtf_path: str | Path) -> pd.DataFrame:
    """Load transcripts from a GTF file into a DataFrame.

    Args:
        gtf_path: Path to the GTF file.

    Returns:
        DataFrame with columns: transcript_id, gene_id, chrom, strand,
        start, end, exon_starts, exon_ends.
    """
    transcripts: dict[str, dict] = {}

    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            if feature_type not in ("transcript", "exon"):
                continue

            chrom = parts[0]
            start = int(parts[3]) - 1
            end = int(parts[4])
            strand = parts[6]

            attrs = {}
            for attr in parts[8].rstrip(";").split(";"):
                attr = attr.strip()
                if not attr:
                    continue
                if " " in attr:
                    key, val = attr.split(" ", 1)
                    attrs[key] = val.strip('"')
                elif "=" in attr:
                    key, val = attr.split("=", 1)
                    attrs[key] = val.strip('"')

            tid = attrs.get("transcript_id", "")
            gid = attrs.get("gene_id", "")

            if feature_type == "transcript":
                transcripts[tid] = {
                    "transcript_id": tid,
                    "gene_id": gid,
                    "chrom": chrom,
                    "strand": strand,
                    "start": start,
                    "end": end,
                    "exon_starts": [],
                    "exon_ends": [],
                }
            elif feature_type == "exon":
                if tid not in transcripts:
                    transcripts[tid] = {
                        "transcript_id": tid,
                        "gene_id": gid,
                        "chrom": chrom,
                        "strand": strand,
                        "start": start,
                        "end": end,
                        "exon_starts": [],
                        "exon_ends": [],
                    }
                transcripts[tid]["exon_starts"].append(start)
                transcripts[tid]["exon_ends"].append(end)
                if start < transcripts[tid]["start"]:
                    transcripts[tid]["start"] = start
                if end > transcripts[tid]["end"]:
                    transcripts[tid]["end"] = end

    rows = []
    for info in transcripts.values():
        es = sorted(info["exon_starts"])
        ee = sorted(info["exon_ends"])
        rows.append({
            "transcript_id": info["transcript_id"],
            "gene_id": info["gene_id"],
            "chrom": info["chrom"],
            "strand": info["strand"],
            "start": info["start"],
            "end": info["end"],
            "exon_starts": es,
            "exon_ends": ee,
            "n_exons": len(es),
        })

    return pd.DataFrame(rows)


def get_gene_list(events_df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique gene IDs from events.

    Args:
        events_df: Events DataFrame.

    Returns:
        Sorted list of gene IDs.
    """
    if "gene_id" in events_df.columns:
        return sorted(events_df["gene_id"].unique().tolist())
    return []


def get_event_type_counts(events_df: pd.DataFrame) -> pd.Series:
    """Get counts of each event type.

    Args:
        events_df: Events DataFrame.

    Returns:
        Series with event type counts.
    """
    if "event_type" in events_df.columns:
        return events_df["event_type"].value_counts()
    return pd.Series(dtype=int)
