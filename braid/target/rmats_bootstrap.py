"""Add BRAID calibrated CI to rMATS-detected AS events.

Takes rMATS output (SE/A3SS/A5SS/MXE/RI junction count files) and
computes per-event PSI confidence intervals using an overdispersed
Beta posterior with support-adaptive scaling.

This combines rMATS's proven event detection with BRAID's calibrated
CI — the best of both tools.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from braid.target.psi_bootstrap import (
    CONFIDENT_CI_WIDTH_THRESHOLD,
    CONFIDENT_CV_THRESHOLD,
    PSIResult,
    bootstrap_psi,
)

logger = logging.getLogger(__name__)


@dataclass
class RmatsEvent:
    """An rMATS-detected AS event with junction counts."""

    event_id: str
    event_type: str
    chrom: str
    strand: str
    gene: str
    # Junction counts from rMATS
    inc_count: int  # Inclusion junction count
    exc_count: int  # Exclusion (skip) junction count
    # rMATS statistics
    rmats_psi: float
    rmats_fdr: float
    rmats_dpsi: float
    # Coordinates
    exon_start: int = 0
    exon_end: int = 0
    sample_1_inc_count: int = 0
    sample_1_exc_count: int = 0
    sample_2_inc_count: int = 0
    sample_2_exc_count: int = 0
    sample_1_inc_replicates: tuple[int, ...] = ()
    sample_1_exc_replicates: tuple[int, ...] = ()
    sample_2_inc_replicates: tuple[int, ...] = ()
    sample_2_exc_replicates: tuple[int, ...] = ()
    sample_1_psi: float | None = None
    sample_2_psi: float | None = None
    upstream_es: int | None = None
    upstream_ee: int | None = None
    downstream_es: int | None = None
    downstream_ee: int | None = None


def _strip_quotes(value: str) -> str:
    """Remove optional double quotes emitted by rMATS tables."""
    return value.strip().strip('"')


def _get_field(fields: list[str], cols: dict[str, int], name: str) -> str | None:
    """Return one column value if present."""
    idx = cols.get(name)
    if idx is None or idx < 0 or idx >= len(fields):
        return None
    return fields[idx]


def _parse_count_sum(value: str | None) -> int:
    """Parse one comma-separated rMATS count field."""
    return sum(_parse_count_vector(value))


def _parse_count_vector(value: str | None) -> tuple[int, ...]:
    """Parse one comma-separated rMATS count field into replicate counts."""
    if not value:
        return ()
    counts: list[int] = []
    for part in value.split(","):
        part = _strip_quotes(part)
        if not part or part == "NA":
            continue
        counts.append(int(part))
    return tuple(counts)


def _parse_inc_level_mean(value: str | None) -> float | None:
    """Parse one comma-separated inclusion-level field."""
    if not value:
        return None
    psi_vals = [
        float(part)
        for part in value.split(",")
        if part and part != "NA"
    ]
    if not psi_vals:
        return None
    return float(np.mean(psi_vals))


def _select_gene(fields: list[str], cols: dict[str, int]) -> str:
    """Prefer gene symbols for downstream target matching when available."""
    gene_symbol = _get_field(fields, cols, "geneSymbol")
    if gene_symbol:
        gene_symbol = _strip_quotes(gene_symbol)
        if gene_symbol and gene_symbol != "NA":
            return gene_symbol
    gene_id = _get_field(fields, cols, "GeneID")
    return _strip_quotes(gene_id or "")


def get_group_counts(
    event: RmatsEvent,
    sample: str = "sample_1",
) -> tuple[int, int]:
    """Return inclusion/exclusion counts for the requested rMATS group.

    Args:
        event: An rMATS-detected AS event.
        sample: Which group to return counts for (``"sample_1"`` or ``"sample_2"``).

    Raises:
        ValueError: If *sample* is not ``"sample_1"`` or ``"sample_2"``.
    """
    if sample == "sample_1":
        return event.sample_1_inc_count, event.sample_1_exc_count
    if sample == "sample_2":
        return event.sample_2_inc_count, event.sample_2_exc_count
    raise ValueError(
        f"Invalid sample label {sample!r}; expected 'sample_1' or 'sample_2'"
    )


def _build_event_id(
    event_type: str,
    chrom: str,
    es: int,
    ee: int,
    up_es: int | None,
    up_ee: int | None,
    dn_es: int | None,
    dn_ee: int | None,
    fields: list[str],
    cols: dict[str, int],
) -> str:
    """Build a collision-resistant event identifier.

    For SE events the upstream and downstream exon coordinates are
    appended so that two SE events sharing the same skipped exon but
    different flanking exons produce distinct IDs.  For A3SS/A5SS the
    long and short exon boundaries disambiguate.  For MXE both
    alternative exons are included.  For RI the flanking exon
    boundaries are appended.
    """
    base = f"{event_type}:{chrom}:{es}-{ee}"

    if event_type == "SE":
        parts: list[str] = []
        if up_ee is not None:
            parts.append(f"u{up_ee}")
        if dn_es is not None:
            parts.append(f"d{dn_es}")
        if parts:
            return f"{base}:{'/'.join(parts)}"
        return base

    if event_type in ("A3SS", "A5SS"):
        long_es = _get_field(fields, cols, "longExonStart_0base")
        long_ee = _get_field(fields, cols, "longExonEnd")
        short_es = _get_field(fields, cols, "shortES")
        short_ee = _get_field(fields, cols, "shortEE")
        extras: list[str] = []
        if long_es is not None and long_ee is not None:
            extras.append(f"l{long_es}-{long_ee}")
        if short_es is not None and short_ee is not None:
            extras.append(f"s{short_es}-{short_ee}")
        if extras:
            return f"{base}:{'/'.join(extras)}"
        return base

    if event_type == "MXE":
        e1_es = _get_field(fields, cols, "1stExonStart_0base")
        e1_ee = _get_field(fields, cols, "1stExonEnd")
        e2_es = _get_field(fields, cols, "2ndExonStart_0base")
        e2_ee = _get_field(fields, cols, "2ndExonEnd")
        mxe_parts: list[str] = []
        if e1_es is not None and e1_ee is not None:
            mxe_parts.append(f"e1={e1_es}-{e1_ee}")
        if e2_es is not None and e2_ee is not None:
            mxe_parts.append(f"e2={e2_es}-{e2_ee}")
        if up_ee is not None:
            mxe_parts.append(f"u{up_ee}")
        if dn_es is not None:
            mxe_parts.append(f"d{dn_es}")
        if mxe_parts:
            return f"{base}:{'/'.join(mxe_parts)}"
        return base

    if event_type == "RI":
        ri_parts: list[str] = []
        if up_es is not None:
            ri_parts.append(f"u{up_es}-{up_ee}")
        if dn_es is not None:
            ri_parts.append(f"d{dn_es}-{dn_ee}")
        if ri_parts:
            return f"{base}:{'/'.join(ri_parts)}"
        return base

    return base


def _resolve_rmats_table(rmats_dir: str, event_type: str) -> str | None:
    """Return the first supported rMATS output table for one event type."""
    for suffix in (
        "MATS.JunctionCountOnly.txt",
        "MATS.JC.txt",
        "MATS.JCEC.txt",
    ):
        fname = os.path.join(rmats_dir, f"{event_type}.{suffix}")
        if os.path.exists(fname):
            return fname
    return None


def parse_rmats_output(
    rmats_dir: str,
    event_types: list[str] | None = None,
    min_total_count: int = 10,
) -> list[RmatsEvent]:
    """Parse rMATS junction count output files.

    Args:
        rmats_dir: Directory containing rMATS *.MATS.JunctionCountOnly.txt
        event_types: Event types to parse (default: all)
        min_total_count: Minimum total junction count to include

    Returns:
        List of parsed events with junction counts.
    """
    if event_types is None:
        event_types = ["SE", "A3SS", "A5SS", "MXE", "RI"]

    events: list[RmatsEvent] = []

    for et in event_types:
        fname = _resolve_rmats_table(rmats_dir, et)
        if fname is None:
            continue

        with open(fname) as f:
            header = f.readline().strip().split("\t")

            # Column indices
            cols = {h: i for i, h in enumerate(header)}

            for line_no, line in enumerate(f, start=2):
                fields = line.strip().split("\t")
                if len(fields) < len(header):
                    continue

                gene = ""
                try:
                    chrom = fields[cols.get("chr", 3)]
                    strand = fields[cols.get("strand", 4)]
                    gene = _select_gene(fields, cols)

                    # Junction counts (per group)
                    inc_1_parts = _parse_count_vector(_get_field(fields, cols, "IJC_SAMPLE_1"))
                    exc_1_parts = _parse_count_vector(_get_field(fields, cols, "SJC_SAMPLE_1"))
                    inc_2_parts = _parse_count_vector(_get_field(fields, cols, "IJC_SAMPLE_2"))
                    exc_2_parts = _parse_count_vector(_get_field(fields, cols, "SJC_SAMPLE_2"))
                    inc_1 = sum(inc_1_parts)
                    exc_1 = sum(exc_1_parts)
                    inc_2 = sum(inc_2_parts)
                    exc_2 = sum(exc_2_parts)
                    total_count = inc_1 + exc_1 + inc_2 + exc_2

                    if total_count < min_total_count:
                        continue

                    # PSI
                    psi_1 = _parse_inc_level_mean(_get_field(fields, cols, "IncLevel1"))
                    psi_2 = _parse_inc_level_mean(_get_field(fields, cols, "IncLevel2"))

                    # FDR and dPSI
                    fdr = float(_get_field(fields, cols, "FDR") or "nan")
                    dpsi = float(
                        _get_field(fields, cols, "IncLevelDifference") or "nan"
                    )

                    # Exon coordinates
                    es = int(fields[cols.get("exonStart_0base", 5)])
                    ee = int(fields[cols.get("exonEnd", 6)])

                    # Flanking exon coordinates (when present)
                    up_es = (
                        int(_get_field(fields, cols, "upstreamES") or "0")
                        if _get_field(fields, cols, "upstreamES") is not None
                        else None
                    )
                    up_ee = (
                        int(_get_field(fields, cols, "upstreamEE") or "0")
                        if _get_field(fields, cols, "upstreamEE") is not None
                        else None
                    )
                    dn_es = (
                        int(_get_field(fields, cols, "downstreamES") or "0")
                        if _get_field(fields, cols, "downstreamES") is not None
                        else None
                    )
                    dn_ee = (
                        int(_get_field(fields, cols, "downstreamEE") or "0")
                        if _get_field(fields, cols, "downstreamEE") is not None
                        else None
                    )

                    eid = _build_event_id(
                        et, chrom, es, ee,
                        up_es, up_ee, dn_es, dn_ee,
                        fields, cols,
                    )

                    events.append(RmatsEvent(
                        event_id=eid,
                        event_type=et,
                        chrom=chrom,
                        strand=strand,
                        gene=gene,
                        inc_count=inc_1,
                        exc_count=exc_1,
                        sample_1_inc_count=inc_1,
                        sample_1_exc_count=exc_1,
                        sample_2_inc_count=inc_2,
                        sample_2_exc_count=exc_2,
                        sample_1_inc_replicates=inc_1_parts,
                        sample_1_exc_replicates=exc_1_parts,
                        sample_2_inc_replicates=inc_2_parts,
                        sample_2_exc_replicates=exc_2_parts,
                        rmats_psi=psi_1 if psi_1 is not None else 0.5,
                        sample_1_psi=psi_1,
                        sample_2_psi=psi_2,
                        rmats_fdr=fdr,
                        rmats_dpsi=dpsi,
                        exon_start=es,
                        exon_end=ee,
                        upstream_es=up_es,
                        upstream_ee=up_ee,
                        downstream_es=dn_es,
                        downstream_ee=dn_ee,
                    ))
                except (ValueError, IndexError, KeyError) as exc:
                    logger.warning(
                        "Skipping %s line %d (gene=%s): %s",
                        fname, line_no, gene or "unknown", exc,
                    )
                    continue

    logger.info(
        "Parsed %d rMATS events from %s", len(events), rmats_dir,
    )
    return events


def add_bootstrap_ci(
    events: list[RmatsEvent],
    n_replicates: int = 500,
    confidence_level: float = 0.95,
    seed: int | None = None,
    sample: str = "sample_1",
) -> list[PSIResult]:
    """Add BRAID bootstrap CI to rMATS events.

    For each rMATS event, uses the inclusion/exclusion junction
    counts directly (from rMATS output) and applies overdispersed Beta posterior
    bootstrap to compute CI.

    Args:
        events: Parsed rMATS events.
        n_replicates: Bootstrap replicates.
        confidence_level: CI confidence level.
        seed: Random seed.
        sample: Which rMATS group to summarize (``"sample_1"`` or ``"sample_2"``).

    Returns:
        List of PSI results with bootstrap CI.
    """
    rng = np.random.default_rng(seed)
    results: list[PSIResult] = []

    for ev in events:
        inc_count, exc_count = get_group_counts(ev, sample=sample)
        psi, ci_low, ci_high, cv = bootstrap_psi(
            inc_count,
            exc_count,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            seed=int(rng.integers(0, 2**31)),
            event_type=ev.event_type,
        )

        ci_width = ci_high - ci_low
        is_confident = (
            ci_width < CONFIDENT_CI_WIDTH_THRESHOLD
            and np.isfinite(cv)
            and cv <= CONFIDENT_CV_THRESHOLD
        )
        results.append(PSIResult(
            event_id=ev.event_id,
            event_type=ev.event_type,
            gene=ev.gene,
            chrom=ev.chrom,
            psi=psi,
            ci_low=ci_low,
            ci_high=ci_high,
            cv=cv,
            inclusion_count=inc_count,
            exclusion_count=exc_count,
            event_start=ev.exon_start,
            event_end=ev.exon_end,
            ci_width=ci_width,
            is_confident=is_confident,
        ))

    n_conf = sum(1 for r in results if r.is_confident)
    logger.info(
        "Bootstrap CI added to %d events (%d confident)",
        len(results), n_conf,
    )
    return results
