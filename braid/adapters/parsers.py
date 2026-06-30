"""Native-output parsers that normalise each caller into :class:`CallerEvent`.

Each parser reads the *standard differential table* a caller emits and maps it to
the common record. Column detection is tolerant (callers and versions vary); the
minimal columns each parser needs are documented in its docstring and the README.

Supported callers:
  * rMATS  -- ``*.MATS.JC.txt`` count tables (full support + sampling SD).
  * MAJIQ  -- voila ``deltapsi.tsv`` (LSV ΔPSI + P(|dPSI|)).
  * SUPPA2 -- ``diffSplice`` ``.dpsi`` table (ΔPSI + p-value).
  * betAS  -- differential TSV (ΔPSI + native Beta interval).
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass

from braid.adapters.base import (
    CallerEvent,
    beta_dpsi_std,
)

logger = logging.getLogger(__name__)

_Z95 = 1.959963984540054
_SUPPA_TYPE = {
    "SE": "SE", "A3": "A3SS", "A5": "A5SS", "RI": "RI",
    "MX": "MXE", "AF": "AF", "AL": "AL",
}


@dataclass(frozen=True)
class ParserConfig:
    """User-supplied parser controls for ``braid filter``.

    Column overrides are exact header names after surrounding whitespace is stripped.
    ``contrast`` is currently SUPPA2-specific and selects ``<contrast>_dPSI`` with
    the matching ``<contrast>_p-val`` column.
    """

    contrast: str | None = None
    dpsi_column: str | None = None
    pvalue_column: str | None = None
    fdr_column: str | None = None
    support_column: str | None = None
    event_id_column: str | None = None
    gene_column: str | None = None


@dataclass(frozen=True)
class ParseSummary:
    """Machine-readable description of what a caller parser consumed."""

    caller: str
    parsed: int
    dpsi_column: str | None = None
    pvalue_column: str | None = None
    fdr_column: str | None = None
    support_column: str | None = None
    event_id_column: str | None = None
    gene_column: str | None = None
    contrast: str | None = None
    dropped_missing_dpsi: int = 0
    dropped_out_of_range: int = 0


@dataclass(frozen=True)
class ParseResult:
    """Parser output plus the column/drop summary used by ``braid filter -v``."""

    events: list[CallerEvent]
    summary: ParseSummary


def _to_float(value: str | None) -> float | None:
    """Parse a numeric cell; ``None`` for NA/empty/non-numeric."""
    if value is None:
        return None
    v = value.strip().strip('"')
    if v == "" or v.upper() in ("NA", "NAN", "NULL", "."):
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    # Non-finite (inf/-inf, or "1e999") is corrupt for every field we read.
    return f if math.isfinite(f) else None


def _in_dpsi_domain(d: float | None) -> bool:
    """ΔPSI is a difference of inclusion fractions, so ``|ΔPSI| <= 1`` always.

    A present value outside ``[-1, 1]`` (or, after :func:`_to_float`, a non-finite
    one) indicates a corrupt caller row and is dropped rather than emitted.
    """
    return d is not None and -1.0 <= d <= 1.0


def _warn_dropped(path: str, n: int) -> None:
    """Log when caller rows were dropped for an out-of-domain ΔPSI (not silent)."""
    if n:
        logger.warning(
            "%s: dropped %d row(s) with out-of-range ΔPSI (|ΔPSI| > 1); "
            "check the caller output for corruption.", path, n,
        )


def _warn_missing_dpsi(path: str, n: int) -> None:
    """Log dropped rows whose ΔPSI cell was missing, non-numeric, or non-finite."""
    if n:
        logger.warning(
            "%s: dropped %d row(s) with missing/non-finite ΔPSI; check the caller "
            "output for incomplete differential rows.", path, n,
        )


def _read_rows(path: str) -> tuple[list[str], list[list[str]]]:
    """Read a tab-separated table -> (header, rows).

    Uses ``utf-8-sig`` so a leading byte-order mark (BOM) from a Windows/Excel-
    exported caller table is stripped transparently; otherwise the BOM would cling
    to the first header cell (``\\ufeff`` survives ``str.strip()``) and break column
    detection, e.g. a ΔPSI column going unrecognised. Plain UTF-8 files read
    identically.
    """
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        rows = [r for r in reader if r and not (len(r) == 1 and r[0].strip() == "")]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _find_col(header: list[str], *candidates: str, contains: bool = False) -> int | None:
    """Index of the first header cell matching any candidate (exact or substring)."""
    low = [h.strip().lower() for h in header]
    for cand in candidates:
        c = cand.lower()
        for i, h in enumerate(low):
            if (c in h) if contains else (h == c):
                return i
    return None


def _column_name(header: list[str], idx: int | None, *, fallback: str | None = None) -> str | None:
    if idx is None:
        return fallback
    if 0 <= idx < len(header):
        name = header[idx].strip()
        return name or fallback
    return fallback


def _override_col(
    header: list[str], name: str | None, *, role: str, path: str
) -> int | None:
    """Resolve an explicit column override by exact stripped header name."""
    if name is None:
        return None
    target = name.strip()
    matches = [i for i, h in enumerate(header) if h.strip() == target]
    if not matches:
        raise ValueError(f"{path}: --{role}-column {name!r} was not found in the header.")
    if len(matches) > 1:
        raise ValueError(f"{path}: --{role}-column {name!r} matches duplicate headers.")
    return matches[0]


def _select_col(
    header: list[str],
    override: str | None,
    *,
    role: str,
    path: str,
    candidates: tuple[str, ...],
    contains: bool = True,
) -> int | None:
    """Resolve an override when supplied, otherwise fall back to fuzzy matching."""
    if override is not None:
        return _override_col(header, override, role=role, path=path)
    return _find_col(header, *candidates, contains=contains)


def _maybe_result(
    events: list[CallerEvent], summary: ParseSummary, return_summary: bool
) -> list[CallerEvent] | ParseResult:
    return ParseResult(events=events, summary=summary) if return_summary else events


def _cell(row: list[str], idx: int | None) -> str | None:
    if idx is None or idx >= len(row):
        return None
    return row[idx]


def _trailing_trimmed_len(row: list[str]) -> int:
    """Field count of *row* ignoring trailing empty cells.

    A trailing tab (common in hand-edited or Excel-exported TSVs) adds an empty cell
    that would otherwise inflate the row width and trip width-based layout heuristics.
    """
    n = len(row)
    while n > 0 and row[n - 1].strip() == "":
        n -= 1
    return n


def _majiq_pick(
    dpsi_value: str | None, prob_value: str | None
) -> tuple[float | None, float | None]:
    """The most-changing junction's (dPSI, P) as a *pair*, from the SAME junction.

    MAJIQ reports per-junction ';'-separated E(dPSI) and P(|dPSI|>=cutoff). We select
    the junction with the largest |dPSI| and return ITS probability, so a large effect
    on one junction is never mixed with a high probability on a different junction.
    """
    if dpsi_value is None:
        return None, None
    dvals = [_to_float(p) for p in str(dpsi_value).replace(",", ";").split(";")]
    pvals = (
        [_to_float(p) for p in str(prob_value).replace(",", ";").split(";")]
        if prob_value is not None else []
    )
    best_i: int | None = None
    for i, d in enumerate(dvals):
        if d is None:
            continue
        if best_i is None or abs(d) > abs(dvals[best_i]):
            best_i = i
    if best_i is None:
        return None, None
    prob = pvals[best_i] if best_i < len(pvals) else None
    return dvals[best_i], prob


# --------------------------------------------------------------------------- #
# rMATS                                                                        #
# --------------------------------------------------------------------------- #
def parse_rmats(
    rmats_dir: str,
    *,
    min_support: int = 0,
    strict: bool = False,
    config: ParserConfig | None = None,
    return_summary: bool = False,
) -> list[CallerEvent] | ParseResult:
    """Parse an rMATS output directory (``*.MATS.JC.txt``).

    Junction counts give real read support and a closed-form ΔPSI sampling SD, so
    rMATS events get the depth-robust calibrated interval. ΔPSI is rMATS
    ``IncLevelDifference`` (sample_1 − sample_2).
    """
    from braid.target.rmats_bootstrap import get_group_counts, parse_rmats_output

    if config is not None:
        unsupported = {
            "contrast": config.contrast,
            "dpsi": config.dpsi_column,
            "pvalue": config.pvalue_column,
            "fdr": config.fdr_column,
            "support": config.support_column,
            "event-id": config.event_id_column,
            "gene": config.gene_column,
        }
        used = [name for name, value in unsupported.items() if value is not None]
        if used:
            raise ValueError(
                "rMATS filter input uses the native rMATS schema; column overrides "
                f"are not supported for --caller rmats ({', '.join(used)})."
            )

    events = parse_rmats_output(rmats_dir, min_total_count=min_support, strict=strict)
    out: list[CallerEvent] = []
    dropped = 0
    missing = 0
    for ev in events:
        # rMATS IncLevelDifference can be NA (non-finite) for degenerate events;
        # drop such rows with a warning -- matching the MAJIQ/SUPPA2/betAS adapters
        # -- rather than emitting a NaN ΔPSI that becomes a full [-1, 1] interval
        # and a literal "nan" in the output TSV.
        if ev.rmats_dpsi is None or not math.isfinite(float(ev.rmats_dpsi)):
            missing += 1
            continue
        if not _in_dpsi_domain(ev.rmats_dpsi):
            dropped += 1
            continue
        inc1, exc1 = get_group_counts(ev, sample="sample_1")
        inc2, exc2 = get_group_counts(ev, sample="sample_2")
        support = float(inc1 + exc1 + inc2 + exc2)
        # Length-normalize the skipping counts to the molecular-PSI (IncLevel) scale
        # before the closed-form ΔPSI sampling SD, matching `braid differential`
        # (differential.py rescales exclusion by IncFormLen/SkipFormLen) so the two
        # rMATS paths report the same interval width for the same event.
        ratio = ev.inc_form_len / ev.skip_form_len if ev.skip_form_len > 0 else 1.0
        std = beta_dpsi_std(inc1, exc1 * ratio, inc2, exc2 * ratio)
        out.append(CallerEvent(
            event_id=ev.event_id,
            gene=ev.gene,
            event_type=ev.event_type,
            dpsi=float(ev.rmats_dpsi),
            total_support=support,
            caller="rmats",
            fdr=float(ev.rmats_fdr) if ev.rmats_fdr == ev.rmats_fdr else None,
            sampling_std=std,
            group1_psi=ev.sample_1_psi,
            group2_psi=ev.sample_2_psi,
            chrom=ev.chrom,
        ))
    _warn_missing_dpsi(rmats_dir, missing)
    _warn_dropped(rmats_dir, dropped)
    summary = ParseSummary(
        caller="rmats",
        parsed=len(out),
        dpsi_column="IncLevelDifference",
        fdr_column="FDR",
        support_column="IJC_SAMPLE_1/SJC_SAMPLE_1/IJC_SAMPLE_2/SJC_SAMPLE_2",
        event_id_column="ID",
        gene_column="geneSymbol/GeneID",
        dropped_missing_dpsi=missing,
        dropped_out_of_range=dropped,
    )
    return _maybe_result(out, summary, return_summary)


# --------------------------------------------------------------------------- #
# MAJIQ                                                                        #
# --------------------------------------------------------------------------- #
def parse_majiq(
    tsv: str,
    *,
    config: ParserConfig | None = None,
    return_summary: bool = False,
) -> list[CallerEvent] | ParseResult:
    """Parse a MAJIQ voila ``deltapsi.tsv``.

    Minimal columns: a gene name, an LSV id, an ``E(dPSI)``/mean-dPSI column, and a
    ``P(|dPSI|>=...)`` column. The dPSI and its probability are taken from the SAME
    (most-changing) junction. MAJIQ reports a probability, not a p-value; it is mapped
    to ``pvalue = 1 - P``, so an LSV is caller-significant when ``pvalue`` passes the
    shared significance threshold (default 0.05), i.e. ``P > 1 - threshold`` (P > 0.95
    at the default). Event type is ``LSV`` (MAJIQ is not cassette-typed), which
    cascades to the support/global conformal quantile.
    """
    config = config or ParserConfig()
    if config.contrast is not None:
        raise ValueError("--contrast is only supported for --caller suppa2.")
    header, rows = _read_rows(tsv)
    if not header:
        summary = ParseSummary(caller="majiq", parsed=0)
        return _maybe_result([], summary, return_summary)
    gi = _select_col(
        header, config.gene_column, role="gene", path=tsv,
        candidates=("gene_name", "gene_id", "gene"),
    )
    li = _select_col(
        header, config.event_id_column, role="event-id", path=tsv,
        candidates=("lsv_id", "lsv id", "lsv"),
    )
    di = _select_col(
        header, config.dpsi_column, role="dpsi", path=tsv,
        candidates=("e(dpsi)", "mean_dpsi", "median_dpsi", "dpsi"),
    )
    pi = _select_col(
        header, config.pvalue_column, role="pvalue", path=tsv,
        candidates=("p(|dpsi|", "probability", "prob"),
    )
    fi = _select_col(
        header, config.fdr_column, role="fdr", path=tsv,
        candidates=("fdr", "padj", "qval"),
    )
    ri = _select_col(
        header, config.support_column, role="support", path=tsv,
        candidates=("num_reads", "reads"),
    )
    si = _find_col(header, "stdev", "std", "sd", contains=True)
    if di is None:
        raise ValueError(
            f"MAJIQ table {tsv}: no ΔPSI column found "
            "(expected one containing 'E(dPSI)' or 'dpsi')."
        )
    out: list[CallerEvent] = []
    dropped_oor = 0
    missing = 0
    for i, row in enumerate(rows):
        dpsi, prob = _majiq_pick(_cell(row, di), _cell(row, pi))
        if dpsi is None:
            missing += 1
            continue
        if not _in_dpsi_domain(dpsi):
            dropped_oor += 1
            continue
        reads = _to_float(_cell(row, ri))
        std = _to_float(_cell(row, si))
        eid = (_cell(row, li) or _cell(row, gi) or f"lsv_{i}").strip().strip('"')
        out.append(CallerEvent(
            event_id=eid,
            gene=(_cell(row, gi) or "").strip().strip('"') or eid.split(";")[0],
            event_type="LSV",
            dpsi=float(dpsi),
            total_support=float(reads) if reads is not None else None,
            caller="majiq",
            # P(|dPSI|>=cutoff) is a probability in [0, 1]; map to pvalue = 1 - P. A
            # corrupt value outside [0, 1] (e.g. P=1.8 -> pvalue=-0.8 < any threshold)
            # would silently flag the LSV as caller-significant, so treat it as missing.
            pvalue=(1.0 - prob) if (prob is not None and 0.0 <= prob <= 1.0) else None,
            fdr=_to_float(_cell(row, fi)),
            sampling_std=std,
        ))
    _warn_missing_dpsi(tsv, missing)
    _warn_dropped(tsv, dropped_oor)
    summary = ParseSummary(
        caller="majiq",
        parsed=len(out),
        dpsi_column=_column_name(header, di),
        pvalue_column=_column_name(header, pi),
        fdr_column=_column_name(header, fi),
        support_column=_column_name(header, ri),
        event_id_column=_column_name(header, li),
        gene_column=_column_name(header, gi),
        dropped_missing_dpsi=missing,
        dropped_out_of_range=dropped_oor,
    )
    return _maybe_result(out, summary, return_summary)


# --------------------------------------------------------------------------- #
# SUPPA2                                                                       #
# --------------------------------------------------------------------------- #
def _exact_header_col(header: list[str], name: str) -> int | None:
    target = name.strip()
    for i, h in enumerate(header):
        if h.strip() == target:
            return i
    return None


def _suppa2_matching_pvalue_col(header: list[str], dpsi_idx: int | None) -> int | None:
    """Infer the SUPPA2 p-value column paired with a selected ``*_dPSI`` column."""
    if dpsi_idx is None or dpsi_idx >= len(header):
        return None
    name = header[dpsi_idx].strip()
    low = name.lower()
    for suffix in ("_dpsi", "dpsi"):
        if low.endswith(suffix):
            prefix = name[: len(name) - len(suffix)]
            for p_suffix in ("_p-val", "_pval", "_p_val"):
                j = _exact_header_col(header, f"{prefix}{p_suffix}")
                if j is not None:
                    return j
    return None


def _suppa2_contrast_col(header: list[str], contrast: str, suffixes: tuple[str, ...]) -> int | None:
    for suffix in suffixes:
        j = _exact_header_col(header, f"{contrast}{suffix}")
        if j is not None:
            return j
    return None


def parse_suppa2(
    dpsi_path: str,
    *,
    config: ParserConfig | None = None,
    return_summary: bool = False,
) -> list[CallerEvent] | ParseResult:
    """Parse a SUPPA2 ``diffSplice`` ``.dpsi`` table.

    Format: an event id in column 0, then a ``*_dPSI`` and a ``*_p-val`` column.
    SUPPA2 works from transcript abundance, not junction counts, so there is no
    read support; ``total_support`` is ``None`` and events use the pooled global
    quantile. Event type is read from the SUPPA2 event id (``SE``/``A3``/``A5``/...).
    If the file carries more than one contrast (multiple ``*_dPSI`` columns), the
    first contrast is used.
    """
    config = config or ParserConfig()
    if config.contrast is not None and (
        config.dpsi_column is not None or config.pvalue_column is not None
    ):
        raise ValueError(
            "Use either --contrast or explicit --dpsi-column/--pvalue-column, not both."
        )
    header, rows = _read_rows(dpsi_path)
    if not header:
        summary = ParseSummary(caller="suppa2", parsed=0)
        return _maybe_result([], summary, return_summary)
    # SUPPA2 headers name the contrast, e.g. "Ctrl-KD_dPSI" / "Ctrl-KD_p-val".
    if config.contrast is not None:
        di = _suppa2_contrast_col(header, config.contrast, ("_dPSI", "_dpsi", "_DPSI"))
        if di is None:
            raise ValueError(
                f"SUPPA2 table {dpsi_path}: contrast {config.contrast!r} has no "
                f"{config.contrast}_dPSI column."
            )
        pj = _suppa2_contrast_col(
            header, config.contrast, ("_p-val", "_pval", "_p_val", "_P-Val")
        )
        if pj is None:
            raise ValueError(
                f"SUPPA2 table {dpsi_path}: contrast {config.contrast!r} has no "
                "matching p-value column."
            )
    else:
        di = _select_col(
            header, config.dpsi_column, role="dpsi", path=dpsi_path,
            candidates=("_dpsi", "dpsi"),
        )
        pj = _select_col(
            header, config.pvalue_column, role="pvalue", path=dpsi_path,
            candidates=("p-val", "pval", "p_val"),
        )
        if config.dpsi_column is not None and config.pvalue_column is None:
            paired = _suppa2_matching_pvalue_col(header, di)
            if paired is not None:
                pj = paired
    if di is None:
        raise ValueError(
            f"SUPPA2 table {dpsi_path}: no dPSI column found "
            "(expected one containing '_dPSI' or 'dpsi')."
        )
    fi = _select_col(
        header, config.fdr_column, role="fdr", path=dpsi_path,
        candidates=("fdr", "padj", "qval"),
    )
    si = _select_col(
        header, config.support_column, role="support", path=dpsi_path,
        candidates=("total_support", "support", "reads"),
    )
    ii = _override_col(
        header, config.event_id_column, role="event-id", path=dpsi_path
    ) if config.event_id_column is not None else None
    gi = _override_col(
        header, config.gene_column, role="gene", path=dpsi_path
    ) if config.gene_column is not None else None
    n_dpsi = sum(1 for h in header if "dpsi" in h.strip().lower())
    if n_dpsi > 1 and config.contrast is None and config.dpsi_column is None:
        logger.warning(
            "SUPPA2 table %s has %d dPSI contrasts; using the first (%s). Split the "
            "file by contrast or verify the group orientation.",
            dpsi_path, n_dpsi, header[di] if di is not None else "?",
        )
    # The event id is the first column; SUPPA2 sometimes leaves its header blank.
    out: list[CallerEvent] = []
    dropped_oor = 0
    missing = 0
    # When the id column is unnamed, the data rows have one extra leading field.
    # Compare widths ignoring trailing empty cells on BOTH sides: a trailing tab on the
    # data row alone would otherwise make a NORMAL (named-id) row look wider than the
    # header and shift every column by one; trimming only the data side (and comparing to
    # the RAW header) instead breaks the supported no-placeholder format when both header
    # and data carry a trailing tab (e.g. an Excel round-trip), dropping every row. Trim
    # both sides so the comparison reflects real field counts in all four combinations.
    id_in_data = (
        _trailing_trimmed_len(rows[0]) > _trailing_trimmed_len(header) if rows else False
    )
    def shifted(idx: int | None) -> int | None:
        return (idx + 1) if (id_in_data and idx is not None) else idx

    for row in rows:
        eid = (
            (_cell(row, shifted(ii)) or "").strip().strip('"')
            if ii is not None else
            (row[0].strip().strip('"') if row else "")
        )
        if not eid:
            continue
        # When the id column is unnamed, every value sits one field right of its header.
        d_idx = shifted(di)
        p_idx = shifted(pj)
        dpsi = _to_float(_cell(row, d_idx))
        pval = _to_float(_cell(row, p_idx))
        if dpsi is None:
            missing += 1
            continue
        if not _in_dpsi_domain(dpsi):
            dropped_oor += 1
            continue
        etype = "SE"
        gene = (_cell(row, shifted(gi)) or "").strip().strip('"') if gi is not None else ""
        gene = gene or eid.split(";")[0]
        if ";" in eid:
            tag = eid.split(";", 1)[1].split(":", 1)[0].strip().upper()[:2]
            etype = _SUPPA_TYPE.get(tag, "SE")
        support = _to_float(_cell(row, shifted(si)))
        out.append(CallerEvent(
            event_id=eid,
            gene=gene,
            event_type=etype,
            dpsi=float(dpsi),
            total_support=float(support) if support is not None else None,
            caller="suppa2",
            pvalue=pval,
            fdr=_to_float(_cell(row, shifted(fi))),
        ))
    _warn_missing_dpsi(dpsi_path, missing)
    _warn_dropped(dpsi_path, dropped_oor)
    summary = ParseSummary(
        caller="suppa2",
        parsed=len(out),
        dpsi_column=_column_name(header, di),
        pvalue_column=_column_name(header, pj),
        fdr_column=_column_name(header, fi),
        support_column=_column_name(header, si),
        event_id_column=_column_name(header, ii, fallback="column0"),
        gene_column=_column_name(header, gi),
        contrast=config.contrast,
        dropped_missing_dpsi=missing,
        dropped_out_of_range=dropped_oor,
    )
    return _maybe_result(out, summary, return_summary)


# --------------------------------------------------------------------------- #
# betAS                                                                        #
# --------------------------------------------------------------------------- #
def parse_betas(
    tsv: str,
    *,
    config: ParserConfig | None = None,
    return_summary: bool = False,
) -> list[CallerEvent] | ParseResult:
    """Parse a betAS differential TSV.

    Minimal columns: a ΔPSI column (``deltapsi``/``dpsi``) and an event/gene id.
    Optional: a native Beta interval (``lower``/``upper``), from which a sampling
    SD is derived; ``support``/``reads``; ``event_type``; a p-value/FDR.
    """
    config = config or ParserConfig()
    if config.contrast is not None:
        raise ValueError("--contrast is only supported for --caller suppa2.")
    header, rows = _read_rows(tsv)
    if not header:
        summary = ParseSummary(caller="betas", parsed=0)
        return _maybe_result([], summary, return_summary)
    gi = _select_col(
        header, config.gene_column, role="gene", path=tsv,
        candidates=("gene", "gene_name"),
    )
    ti = _find_col(header, "event_type", "type", contains=True)
    # Event id: prefer specific names; never reuse the event_type column. A bare
    # "event" substring would otherwise match "event_type" and give every event the
    # same id (e.g. all "A3SS"), corrupting the identifier and collapsing rows in
    # any downstream join.
    ii = _select_col(
        header, config.event_id_column, role="event-id", path=tsv,
        candidates=("event_id", "key"),
    )
    if ii is None:
        for _cand in ("event", "id"):
            j = _find_col(header, _cand, contains=True)
            if j is not None and j != ti:
                ii = j
                break
    di = _select_col(
        header, config.dpsi_column, role="dpsi", path=tsv,
        candidates=("deltapsi", "delta_psi", "dpsi", "median_dpsi"),
    )
    lo = _find_col(header, "lower", "ci_low", "low", contains=True)
    hi = _find_col(header, "upper", "ci_high", "high", contains=True)
    fi = _select_col(
        header, config.fdr_column, role="fdr", path=tsv,
        candidates=("fdr", "padj", "qval"),
    )
    pi = _select_col(
        header, config.pvalue_column, role="pvalue", path=tsv,
        candidates=("pval", "p_value", "p-val"),
    )
    si = _select_col(
        header, config.support_column, role="support", path=tsv,
        candidates=("total_support", "support", "reads"),
    )
    if di is None:
        raise ValueError(
            f"betAS table {tsv}: no ΔPSI column found "
            "(expected one containing 'deltapsi' or 'dpsi')."
        )
    out: list[CallerEvent] = []
    dropped_oor = 0
    missing = 0
    for i, row in enumerate(rows):
        dpsi = _to_float(_cell(row, di))
        if dpsi is None:
            missing += 1
            continue
        if not _in_dpsi_domain(dpsi):
            dropped_oor += 1
            continue
        low = _to_float(_cell(row, lo))
        high = _to_float(_cell(row, hi))
        std = None
        if low is not None and high is not None and high >= low:
            std = (high - low) / (2.0 * _Z95)
        else:
            # An incomplete (one bound missing) or inverted (upper < lower) native
            # interval is corrupt: derive no std from it and do not carry it forward as
            # a comparison interval with a negative width.
            low = high = None
        support = _to_float(_cell(row, si))
        gene = (_cell(row, gi) or "").strip().strip('"')
        eid = (_cell(row, ii) or gene or f"event_{i}").strip().strip('"')
        etype = (_cell(row, ti) or "SE").strip().strip('"') or "SE"
        out.append(CallerEvent(
            event_id=eid,
            gene=gene or eid.split(";")[0],
            event_type=etype,
            dpsi=float(dpsi),
            total_support=float(support) if support is not None else None,
            caller="betas",
            fdr=_to_float(_cell(row, fi)),
            pvalue=_to_float(_cell(row, pi)),
            sampling_std=std,
            caller_low=low,
            caller_high=high,
        ))
    _warn_missing_dpsi(tsv, missing)
    _warn_dropped(tsv, dropped_oor)
    summary = ParseSummary(
        caller="betas",
        parsed=len(out),
        dpsi_column=_column_name(header, di),
        pvalue_column=_column_name(header, pi),
        fdr_column=_column_name(header, fi),
        support_column=_column_name(header, si),
        event_id_column=_column_name(header, ii),
        gene_column=_column_name(header, gi),
        dropped_missing_dpsi=missing,
        dropped_out_of_range=dropped_oor,
    )
    return _maybe_result(out, summary, return_summary)


PARSERS = {
    "rmats": parse_rmats,
    "majiq": parse_majiq,
    "suppa2": parse_suppa2,
    "betas": parse_betas,
}
