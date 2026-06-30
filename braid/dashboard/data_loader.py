"""Data loading utilities for the dashboard.

Loads events TSV and GTF files into pandas DataFrames for visualization.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns the dashboard components address directly (lowercase ``analyze``
# schema). ``braid psi``/``braid differential`` emit a different dialect
# (``gene``/``PSI`` etc.), so we validate up front and fail with a clear message
# instead of a downstream ``KeyError``.
_REQUIRED_DASHBOARD_COLUMNS = ("gene_id", "psi", "event_type")


def load_events(events_path: str | Path) -> pd.DataFrame:
    """Load an ``analyze`` events TSV into a DataFrame.

    Args:
        events_path: Path to the events TSV produced by ``braid analyze``.

    Returns:
        DataFrame with typed columns for event data.

    Raises:
        ValueError: if the TSV is missing the canonical ``analyze`` columns the
            dashboard requires (e.g. a ``braid psi`` or ``braid differential``
            output, which use a different schema, was supplied).
    """
    df = pd.read_csv(events_path, sep="\t")

    missing = [c for c in _REQUIRED_DASHBOARD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{events_path}: missing dashboard column(s) {missing}. The dashboard "
            f"consumes the TSV from `braid analyze` (columns: gene_id, psi, "
            f"event_type, ...). `braid psi`/`braid differential` outputs use a "
            f"different schema (gene, PSI, ...) and are not dashboard inputs."
        )

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
            if not tid:
                continue

            if feature_type == "transcript":
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
                else:
                    # GTF feature order is not guaranteed. If exon rows arrive
                    # before the transcript row, preserve the accumulated exon
                    # coordinates and fill/update transcript-level metadata.
                    transcripts[tid]["gene_id"] = gid or transcripts[tid]["gene_id"]
                    transcripts[tid]["chrom"] = chrom
                    transcripts[tid]["strand"] = strand
                    transcripts[tid]["start"] = min(transcripts[tid]["start"], start)
                    transcripts[tid]["end"] = max(transcripts[tid]["end"], end)
            elif feature_type == "exon":
                rec = transcripts.setdefault(tid, {
                    "transcript_id": tid,
                    "gene_id": gid,
                    "chrom": chrom,
                    "strand": strand,
                    "start": start,
                    "end": end,
                    "exon_starts": [],
                    "exon_ends": [],
                })
                rec["exon_starts"].append(start)
                rec["exon_ends"].append(end)
                if start < rec["start"]:
                    rec["start"] = start
                if end > rec["end"]:
                    rec["end"] = end

    rows = []
    for info in transcripts.values():
        exon_pairs = sorted(zip(info["exon_starts"], info["exon_ends"]))
        es = [start for start, _ in exon_pairs]
        ee = [end for _, end in exon_pairs]
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
        # Drop missing labels and coerce to str before sorting: a column with any
        # NaN (e.g. a blank cell in an external/contaminated TSV) mixes float(nan)
        # with str and makes sorted() raise TypeError.
        return sorted(str(g) for g in events_df["gene_id"].dropna().unique())
    return []


def get_event_type_list(events_df: pd.DataFrame) -> list[str]:
    """Get the sorted unique event types as strings.

    Missing labels are dropped and values coerced to str before sorting, so a
    blank cell (NaN) cannot mix float and str and raise TypeError in ``sorted`` —
    mirrors :func:`get_gene_list`.
    """
    if "event_type" in events_df.columns:
        return sorted(str(t) for t in events_df["event_type"].dropna().unique())
    return []


def filter_by_event_types(
    events_df: pd.DataFrame, selected_types: list[str]
) -> pd.DataFrame:
    """Keep rows whose ``event_type`` is in ``selected_types``.

    :func:`get_event_type_list` returns event types coerced to str (a numeric
    event_type column yields ``["1", "2"]``), so the filter must compare the column
    coerced to str too — otherwise a numeric event_type matches nothing and the
    table silently empties. An empty selection or a missing column passes the frame
    through unchanged.
    """
    if not selected_types or "event_type" not in events_df.columns:
        return events_df
    return events_df[events_df["event_type"].astype(str).isin(selected_types)]


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
