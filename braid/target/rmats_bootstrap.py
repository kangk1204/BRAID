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
from typing import TYPE_CHECKING

import numpy as np

from braid.target.psi_bootstrap import (
    CONFIDENT_CI_WIDTH_THRESHOLD,
    CONFIDENT_CV_THRESHOLD,
    PSIResult,
    bootstrap_psi,
)

if TYPE_CHECKING:
    from braid.target.conformal import ConformalCalibrator

logger = logging.getLogger(__name__)

DEFAULT_RMATS_EVENT_TYPES = ("SE", "A3SS", "A5SS", "MXE", "RI")
RMATS_TABLE_SUFFIXES = (
    "MATS.JunctionCountOnly.txt",
    "MATS.JC.txt",
    "MATS.JCEC.txt",
)


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
    # rMATS effective isoform lengths, used to convert raw junction counts to the
    # length-normalized molecular-PSI (IncLevel) scale. Default 1.0/1.0 == identity
    # (no normalization) so existing callers and fixtures are unaffected.
    inc_form_len: float = 1.0
    skip_form_len: float = 1.0


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
    """Parse one comma-separated rMATS count field into replicate counts.

    Real rMATS IJC/SJC vectors are gapless integer CSVs (one value per replicate,
    ``0`` when a replicate has no reads). A trailing empty cell (benign trailing
    comma) is dropped quietly, but an *interior* gap (a non-trailing empty/NA) is a
    hard error: dropping it would re-index the later replicates and desynchronise the
    inclusion/exclusion pairing — a confidently-wrong number. Malformed input is
    rejected here rather than silently producing shifted replicate pairs.
    """
    if not value:
        return ()
    parts = [_strip_quotes(p) for p in value.split(",")]
    counts: list[int] = []
    for idx, part in enumerate(parts):
        if not part or part == "NA":
            # Interior gap iff a real value still follows: that is where the drop
            # would shift subsequent replicate indices and desynchronise the
            # inclusion/exclusion pairing. Reject rather than silently re-index.
            if any(p and p != "NA" for p in parts[idx + 1:]):
                raise ValueError(
                    f"rMATS count vector {value!r} has an interior empty/NA cell; "
                    "dropping it would shift replicate indices and misalign "
                    "inclusion/exclusion counts. Fix the truncated/merged rMATS row."
                )
            continue
        count = int(part)
        if count < 0:
            # rMATS junction counts are non-negative by construction; a negative value
            # is a corrupted/edited row. Reject it at the parse boundary so BOTH the PSI
            # and differential paths are protected at the origin -- otherwise a negative
            # count flows into rng.beta(a <= 0) and aborts the whole `braid differential`
            # run with an opaque "a <= 0". parse_rmats_output's per-row try/except turns
            # this into a skip-with-warning (or a clear failure under --strict).
            raise ValueError(
                f"rMATS count vector {value!r} has a negative count {count}; junction "
                "counts must be >= 0. Fix the corrupted/edited rMATS row."
            )
        counts.append(count)
    return tuple(counts)


def _parse_inc_level_mean(value: str | None) -> float | None:
    """Parse one comma-separated inclusion-level field."""
    if not value:
        return None
    psi_vals: list[float] = []
    for raw_part in value.split(","):
        part = _strip_quotes(raw_part)
        if not part or part.upper() in ("NA", "NAN", "NULL"):
            continue
        try:
            psi = float(part)
        except ValueError:
            continue
        if np.isfinite(psi):
            psi_vals.append(psi)
    if not psi_vals:
        return None
    return float(np.mean(psi_vals))


def _parse_float_na(value: str | None) -> float:
    """Parse an rMATS float field, returning NaN for missing/NA values.

    rMATS emits ``NA`` for FDR / IncLevelDifference on some events (e.g. when the
    statistical test could not be run). Such events still carry valid junction
    counts and must be kept so they receive a BRAID CI -- previously
    ``float('NA')`` raised and the entire event was silently dropped.
    """
    if value is None:
        return float("nan")
    value = _strip_quotes(value)
    if not value or value.upper() in ("NA", "NAN", "NULL"):
        return float("nan")
    try:
        parsed = float(value)
    except ValueError:
        return float("nan")
    # Treat a non-finite literal ("inf"/"-inf") as missing, mirroring
    # _parse_inc_level_mean (which filters non-finite). A real rMATS table never emits
    # these; returning NaN keeps FDR/dPSI/form-len handling uniform (np.isfinite guards
    # downstream) instead of letting an infinity leak into an output column.
    return parsed if np.isfinite(parsed) else float("nan")


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
    strand: str,
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

    The strand is part of the base so that two events sharing identical
    coordinates on opposite strands (overlapping antisense loci) do not
    collide.  For SE events the upstream and downstream exon coordinates are
    appended so that two SE events sharing the same skipped exon but
    different flanking exons produce distinct IDs.  For A3SS/A5SS the
    long and short exon boundaries plus the constitutive flanking exon
    disambiguate.  For MXE both alternative exons are included.  For RI the flanking exon
    boundaries are appended.
    """
    base = f"{event_type}:{chrom}:{strand}:{es}-{ee}"

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
        flank_es = _get_field(fields, cols, "flankingES")
        flank_ee = _get_field(fields, cols, "flankingEE")
        extras: list[str] = []
        if long_es is not None and long_ee is not None:
            extras.append(f"l{long_es}-{long_ee}")
        if short_es is not None and short_ee is not None:
            extras.append(f"s{short_es}-{short_ee}")
        if flank_es is not None and flank_ee is not None:
            extras.append(f"f{flank_es}-{flank_ee}")
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
    for suffix in RMATS_TABLE_SUFFIXES:
        fname = os.path.join(rmats_dir, f"{event_type}.{suffix}")
        if os.path.exists(fname):
            return fname
    return None


def find_rmats_tables(
    rmats_dir: str,
    event_types: list[str] | tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Return supported rMATS event tables found in *rmats_dir*."""
    if event_types is None:
        event_types = DEFAULT_RMATS_EVENT_TYPES
    tables: dict[str, str] = {}
    for et in event_types:
        fname = _resolve_rmats_table(rmats_dir, et)
        if fname is not None:
            tables[et] = fname
    return tables


def parse_rmats_output(
    rmats_dir: str,
    event_types: list[str] | None = None,
    min_total_count: int = 10,
    strict: bool = False,
) -> list[RmatsEvent]:
    """Parse rMATS junction count output files.

    Args:
        rmats_dir: Directory containing rMATS *.MATS.JunctionCountOnly.txt
        event_types: Event types to parse (default: all)
        min_total_count: Minimum total junction count to include
        strict: If True, raise on the first malformed row instead of skipping it
            (data-integrity mode; default skips with a warning and a final count).

    Returns:
        List of parsed events with junction counts.
    """
    if event_types is None:
        event_types = list(DEFAULT_RMATS_EVENT_TYPES)

    events: list[RmatsEvent] = []
    skipped = 0

    for et in event_types:
        fname = _resolve_rmats_table(rmats_dir, et)
        if fname is None:
            continue

        with open(fname) as f:
            header = f.readline().rstrip("\r\n").split("\t")

            # Column indices
            cols = {h: i for i, h in enumerate(header)}

            # An empty file or a blank header row has no usable columns at all
            # (readline on a 0-byte file yields "" -> split -> [""]). Distinguish it
            # from a column-drop so the error names the real problem instead of
            # reporting every count column as "missing".
            if not any(h.strip() for h in header):
                raise ValueError(
                    f"rMATS table {fname} is empty or has no header row; expected an "
                    "rMATS junction-count table (a header line plus one row per event)."
                )

            # Schema-drift guard: rMATS always emits these four count columns. If one
            # is absent the header is still self-consistent (len(fields) == len(header)),
            # so the per-row truncation guard below never fires -- but _get_field would
            # return None for the missing column, _parse_count_vector(None) -> (), and
            # that group's count silently becomes 0, fabricating PSI == 1.0 and a
            # spurious ΔPSI for EVERY event in the table. There is no per-row recovery
            # (the whole table is unusable), so fail fast regardless of --strict.
            missing_count_cols = [
                c for c in ("IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2")
                if c not in cols
            ]
            if missing_count_cols:
                raise ValueError(
                    f"rMATS table {fname} is missing required count column(s) "
                    f"{missing_count_cols}; every PSI would be computed from fabricated "
                    f"zero counts. Header columns present: {sorted(cols)}."
                )

            for line_no, line in enumerate(f, start=2):
                fields = line.rstrip("\r\n").split("\t")
                if len(fields) < len(header):
                    if strict:
                        raise ValueError(
                            f"Truncated rMATS row at {fname} line {line_no}: "
                            f"{len(fields)} fields < {len(header)} header columns. "
                            "Omit --strict to skip malformed rows instead of failing."
                        )
                    skipped += 1
                    logger.warning(
                        "Skipping %s line %d: truncated row (%d < %d columns).",
                        fname, line_no, len(fields), len(header),
                    )
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

                    # FDR and dPSI (NA-tolerant: keep the event, mark as NaN so
                    # it still receives a CI; tier logic guards on np.isfinite).
                    fdr = _parse_float_na(_get_field(fields, cols, "FDR"))
                    dpsi = _parse_float_na(_get_field(fields, cols, "IncLevelDifference"))

                    # Effective isoform lengths for molecular-PSI normalization
                    # (absent/NA -> 1.0/1.0 == identity, no normalization).
                    inc_fl = _parse_float_na(_get_field(fields, cols, "IncFormLen"))
                    skip_fl = _parse_float_na(_get_field(fields, cols, "SkipFormLen"))
                    inc_fl = inc_fl if np.isfinite(inc_fl) and inc_fl > 0 else 1.0
                    skip_fl = skip_fl if np.isfinite(skip_fl) and skip_fl > 0 else 1.0

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
                        et, chrom, strand, es, ee,
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
                        rmats_psi=psi_1 if psi_1 is not None else float("nan"),
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
                        inc_form_len=inc_fl,
                        skip_form_len=skip_fl,
                    ))
                except (ValueError, IndexError, KeyError) as exc:
                    if strict:
                        raise ValueError(
                            f"Malformed rMATS row at {fname} line {line_no} "
                            f"(gene={gene or 'unknown'}): {exc}. Omit --strict to "
                            "skip malformed rows instead of failing."
                        ) from exc
                    skipped += 1
                    logger.warning(
                        "Skipping %s line %d (gene=%s): %s",
                        fname, line_no, gene or "unknown", exc,
                    )
                    continue

    if skipped:
        logger.warning(
            "Skipped %d malformed rMATS row(s) while parsing %s; the affected events "
            "are ABSENT from the output. Inspect the table if this is unexpected.",
            skipped, rmats_dir,
        )
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
    *,
    use_conformal: bool = True,
    conformal_calibrator: ConformalCalibrator | None = None,
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
    if conformal_calibrator is None and use_conformal:
        from braid.target.conformal import load_default_conformal_calibrator
        try:
            conformal_calibrator = load_default_conformal_calibrator()
        except (FileNotFoundError, OSError) as exc:
            logger.warning(
                "No default conformal calibrator (%s); using legacy intervals", exc
            )

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
            conformal_calibrator=conformal_calibrator,
            inc_form_len=ev.inc_form_len,
            skip_form_len=ev.skip_form_len,
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
