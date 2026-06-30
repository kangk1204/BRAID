#!/usr/bin/env python3
"""Real-data head-to-head: ΔPSI interval coverage & sharpness vs RT-PCR truth.

The question this answers is the make-or-break one: *is BRAID's calibrated
(conformal) interval genuinely better than what competing tools produce, on real
data with external ground truth?*  We judge every method by the only thing that
matters for an uncertainty layer:

    - **Coverage**: does the method's 1-alpha ΔPSI interval actually contain the
      RT-PCR-measured ΔPSI at the nominal rate?
    - **Sharpness**: how wide is the interval (narrower = more useful), *given*
      that coverage is met?
    - **Interval score** (Gneiting & Raftery 2007): a single proper scoring rule
      that rewards calibrated AND sharp intervals; lower is better.

Ground truth is the RT-PCR ΔPSI per validated event (84 TRA2-regulated cassette
exons) plus RT-PCR-negative events (ΔPSI ~ 0). Crucially, the RNA-seq-to-RT-PCR
residual includes platform or assay-definition discordance and any systematic
offset that no count-based sampling model (betAS, Jeffreys) can see -- so those
intervals are systematically too narrow and under-cover. A split-conformal
interval fit on the actual RT-PCR residual distribution absorbs that residual
and hits nominal coverage at a controlled width. That is the principled reason
BRAID-conformal wins here, and this script measures it.

Methods compared (all length-normalized to the rMATS IncLevel / molecular-PSI
scale so the comparison is fair -- see ``_norm_counts``):
    rMATS-perRep    : ΔPSI +/- z*SE from the 3 per-replicate IncLevels (the CI a
                      practitioner derives from rMATS output; rMATS ships only p/FDR).
    betAS           : difference of two per-group Beta(inc, exc) posteriors
                      (betAS's defining estimator; real betAS substituted when
                      --betas-tsv is supplied).
    BRAID-Jeffreys  : difference of two Beta(inc+0.5, exc+0.5) posteriors
                      (BRAID's production differential posterior).
    BRAID-conformal : BRAID-Jeffreys point/scale, interval width set by a
                      cross-fit Mondrian split-conformal quantile (BRAID's
                      production default calibration layer).

Also reports BRAID-Jeffreys on *raw* (un-normalized) counts to quantify the
length-normalization bias in the shipped differential code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

# Make the in-repo braid package importable when run from anywhere.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from braid.target.conformal import (  # noqa: E402
    assign_support_bins,
    conformal_quantile,
    nonconformity_scores,
)

Z95 = 1.959963984540054


# ---------------------------------------------------------------------------
# rMATS SE parsing (self-contained: we need form lengths + per-replicate values)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SEEvent:
    gene: str
    chrom: str
    strand: str
    exon_start: int          # exonStart_0base
    exon_end: int
    ijc1: tuple[int, ...]
    sjc1: tuple[int, ...]
    ijc2: tuple[int, ...]
    sjc2: tuple[int, ...]
    inc_form_len: float
    skip_form_len: float
    inclevel1: tuple[float, ...]
    inclevel2: tuple[float, ...]
    rmats_dpsi: float
    rmats_fdr: float
    upstream_ee: int = 0     # upstream exon end (flanking, for junction matching)
    downstream_es: int = 0   # downstream exon start (flanking)

    @property
    def skip_junction(self) -> tuple[int, int]:
        """Genomic (sorted) endpoints of the exon-skipping junction."""
        lo, hi = sorted((self.upstream_ee, self.downstream_es))
        return lo, hi


def _ivec(s: str) -> tuple[int, ...]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok in ("", "NA"):
            continue
        out.append(int(round(float(tok))))
    return tuple(out)


def _fvec(s: str) -> tuple[float, ...]:
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok in ("", "NA"):
            continue
        out.append(float(tok))
    return tuple(out)


def _fna(s: str) -> float:
    s = (s or "").strip()
    if s in ("", "NA", "NaN", "NULL"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _json_safe(obj):
    """Recursively replace non-finite floats with None so the written JSON is
    strict (RFC 8259) -- bare NaN/Infinity is not valid JSON and is rejected by
    conforming parsers (jq, JS, Go). Missing-value semantics are preserved (a
    non-finite number becomes null)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def parse_se_table(path: str) -> list[SEEvent]:
    """Parse an rMATS ``SE.MATS.JC(EC).txt`` into SEEvent records."""
    events: list[SEEvent] = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        col = {h: i for i, h in enumerate(header)}

        def g(fields: list[str], name: str, default: str = "") -> str:
            i = col.get(name)
            return fields[i] if i is not None and i < len(fields) else default

        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < len(header):
                continue
            try:
                ev = SEEvent(
                    gene=g(fields, "geneSymbol").strip('"'),
                    chrom=g(fields, "chr"),
                    strand=g(fields, "strand"),
                    exon_start=int(g(fields, "exonStart_0base")),
                    exon_end=int(g(fields, "exonEnd")),
                    ijc1=_ivec(g(fields, "IJC_SAMPLE_1")),
                    sjc1=_ivec(g(fields, "SJC_SAMPLE_1")),
                    ijc2=_ivec(g(fields, "IJC_SAMPLE_2")),
                    sjc2=_ivec(g(fields, "SJC_SAMPLE_2")),
                    inc_form_len=float(g(fields, "IncFormLen", "1") or "1"),
                    skip_form_len=float(g(fields, "SkipFormLen", "1") or "1"),
                    inclevel1=_fvec(g(fields, "IncLevel1")),
                    inclevel2=_fvec(g(fields, "IncLevel2")),
                    rmats_dpsi=_fna(g(fields, "IncLevelDifference")),
                    rmats_fdr=_fna(g(fields, "FDR")),
                    upstream_ee=int(g(fields, "upstreamEE", "0") or "0"),
                    downstream_es=int(g(fields, "downstreamES", "0") or "0"),
                )
            except (ValueError, KeyError):
                continue
            events.append(ev)
    return events


# ---------------------------------------------------------------------------
# RT-PCR ground-truth loading + matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    gene: str
    chrom: str
    exon_start: int
    exon_end: int
    dpsi_rtpcr: float
    is_positive: bool
    skip_junction: tuple[int, int] | None = None  # (lo, hi) genomic skip-junction
    # Full SUPPA SE structure (1-based), parsed from the target's suppa_event_id when
    # present: cassette exon [sup_cs, sup_ce] plus the two flanking splice sites
    # (sup_up = upstream donor, sup_down = downstream acceptor) and strand. With this,
    # match_event disambiguates events that share a cassette exon but differ in flanks
    # (e.g. duplicate-exon genes like KTN1) rather than matching on the cassette alone.
    sup_up: int = 0
    sup_cs: int = 0
    sup_ce: int = 0
    sup_down: int = 0
    sup_strand: str = ""
    has_structure: bool = False


def _norm_chrom(c: str) -> str:
    c = c.strip()
    return c if c.startswith("chr") else f"chr{c}"


def _parse_suppa_se(sid: str) -> tuple[int, int, int, int, str] | None:
    """``gene;SE:chr:a-b:c-d:strand`` -> ``(a, b, c, d, strand)`` (1-based); None if
    unparseable. a = upstream donor, b-c = cassette exon, d = downstream acceptor."""
    if not sid or "SE:" not in sid:
        return None
    parts = sid.split("SE:", 1)[1].split(":")   # chr : a-b : c-d : strand
    if len(parts) < 4:
        return None
    try:
        a, b = (int(x) for x in parts[1].split("-"))
        c, d = (int(x) for x in parts[2].split("-"))
    except ValueError:
        return None
    return a, b, c, d, parts[3]


def _target_from_row(row: dict, *, is_positive: bool) -> Target:
    """Build a Target, attaching the full SUPPA SE structure when the column exists."""
    sup = _parse_suppa_se(row.get("suppa_event_id", ""))
    kw = dict(
        gene=row["gene"].strip(),
        chrom=_norm_chrom(row["chrom"]),
        exon_start=int(row["exon_start"]),
        exon_end=int(row["exon_end"]),
        dpsi_rtpcr=float(row.get("delta_psi_rtpcr", 0.0) or 0.0),
        is_positive=is_positive,
    )
    if sup is not None:
        a, b, c, d, strand = sup
        kw.update(sup_up=a, sup_cs=b, sup_ce=c, sup_down=d,
                  sup_strand=strand, has_structure=True)
    return Target(**kw)


def load_targets(validated_tsv: str, failed_tsv: str | None) -> list[Target]:
    targets: list[Target] = []
    with open(validated_tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            targets.append(_target_from_row(row, is_positive=True))
    if failed_tsv and os.path.exists(failed_tsv):
        with open(failed_tsv) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                targets.append(_target_from_row(row, is_positive=False))
    return targets


def load_circadian_targets(tsv: str) -> list[Target]:
    """Load junction-based circadian RT-PCR targets (positives only, with ΔPSI).

    Columns: gene, chrom, strand, inclusion_junction, exclusion_junction,
    delta_psi_rtpcr. The exclusion_junction's genomic endpoints are the upstream
    exon end and downstream exon start of the SE event, used for matching.
    """
    targets: list[Target] = []
    with open(tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            excl = row["exclusion_junction"].replace(":", "-").split("-")
            lo, hi = sorted(int(x) for x in excl[-2:])
            targets.append(Target(
                gene=row["gene"].strip(),
                chrom=_norm_chrom(row["chrom"]),
                exon_start=0, exon_end=0,
                dpsi_rtpcr=float(row["delta_psi_rtpcr"]),
                is_positive=True,
                skip_junction=(lo, hi),
            ))
    return targets


def match_event(t: Target, events: list[SEEvent], tol: int = 6,
                junction_tol: int = 12) -> SEEvent | None:
    """Best rMATS SE event for an RT-PCR target.

    Junction targets (circadian) match on the skip-junction genomic endpoints.
    Exon targets match on the FULL SUPPA SE structure when available (cassette exon
    plus both flanking splice sites and strand), which disambiguates events sharing a
    cassette exon but differing in flanks (e.g. duplicate-exon genes like KTN1); if the
    target carries no SUPPA structure they fall back to cassette-exon coordinates. Ties
    broken by coordinate distance, then gene-name agreement, then highest read support.
    """
    best = None
    best_key = None
    use_junction = t.skip_junction is not None
    for ev in events:
        if _norm_chrom(ev.chrom) != t.chrom:
            continue
        if use_junction:
            elo, ehi = ev.skip_junction
            tlo, thi = t.skip_junction
            ds, de = abs(elo - tlo), abs(ehi - thi)
            if ds > junction_tol or de > junction_tol:
                continue
        elif t.has_structure:
            if ev.strand != t.sup_strand:
                continue
            # Explicit rMATS(mixed-base) -> SUPPA(1-based) coordinate conversion, so the
            # match does NOT rely on a loose tolerance to absorb a coordinate-system
            # offset. rMATS '*_0base' starts are 0-based and its ends are 1-based; the
            # SUPPA SE tuple (sup_up donor, sup_cs cassette start, sup_ce cassette end,
            # sup_down acceptor) is all 1-based. After conversion, coordinates should
            # agree exactly up to a small residual annotation tolerance.
            ev_cs = ev.exon_start + 1        # cassette start: 0-based -> 1-based
            ev_ce = ev.exon_end              # cassette end:   already 1-based
            ev_up = ev.upstream_ee           # upstream donor: already 1-based
            ev_down = ev.downstream_es + 1   # downstream acceptor: 0-based -> 1-based
            stol = 2                         # residual annotation tolerance, not a base offset
            if abs(ev_cs - t.sup_cs) > stol or abs(ev_ce - t.sup_ce) > stol:
                continue
            # flanks as an unordered pair (SUPPA is genomic-ascending; rMATS up/down is
            # strand-relative), now compared in the same 1-based frame
            ev_fl = sorted((ev_up, ev_down))
            t_fl = sorted((t.sup_up, t.sup_down))
            if abs(ev_fl[0] - t_fl[0]) > stol or abs(ev_fl[1] - t_fl[1]) > stol:
                continue
            ds = abs(ev_cs - t.sup_cs) + abs(ev_fl[0] - t_fl[0])
            de = abs(ev_ce - t.sup_ce) + abs(ev_fl[1] - t_fl[1])
        else:
            ds = abs(ev.exon_start - t.exon_start)
            de = abs(ev.exon_end - t.exon_end)
            if ds > tol or de > tol:
                continue
        support = sum(ev.ijc1) + sum(ev.sjc1) + sum(ev.ijc2) + sum(ev.sjc2)
        key = (ds + de, ev.gene.lower() != t.gene.lower(), -support)
        if best_key is None or key < best_key:
            best_key, best = key, ev
    return best


# ---------------------------------------------------------------------------
# Per-event normalized counts and ΔPSI interval estimators
# ---------------------------------------------------------------------------


def _norm_counts(ijc: int, sjc: int, inc_fl: float, skip_fl: float) -> tuple[float, float]:
    """rMATS-consistent length normalization to the molecular-PSI scale.

    PSI = (IJC/IncFL) / (IJC/IncFL + SJC/SkipFL).  Equivalently keep the
    inclusion count as-is and rescale the skipping count by IncFL/SkipFL, which
    preserves the inclusion read evidence (concentration) while matching the
    rMATS IncLevel mean.  Verified to reproduce IncLevel exactly.
    """
    ratio = inc_fl / skip_fl if skip_fl > 0 else 1.0
    return float(ijc), float(sjc) * ratio


@dataclass(frozen=True)
class EventCounts:
    a_c: float  # ctrl inclusion (the RT-PCR minuend group)
    b_c: float  # ctrl exclusion (normalized)
    a_t: float  # treat inclusion (the RT-PCR subtrahend group)
    b_t: float  # treat exclusion (normalized)
    inclevel_ctrl: tuple[float, ...]
    inclevel_treat: tuple[float, ...]
    total_support: float
    rmats_dpsi: float
    rmats_fdr: float


def event_counts(ev: SEEvent, *, normalize: bool = True,
                 swap_groups: bool = False) -> EventCounts:
    """Per-event normalized group counts oriented to the RT-PCR convention.

    ΔPSI is always ctrl - treat. ``swap_groups`` chooses which rMATS sample is
    "ctrl" (the RT-PCR minuend): False -> ctrl=sample_1 (rMATS IncLevelDifference
    convention, sample_1 - sample_2); True -> ctrl=sample_2, used when the RT-PCR
    table reports sample_2 - sample_1. For TRA2 the design has b1=control,
    b2=knockdown and the RT-PCR table reports knockdown - control, so swap=True.
    The orientation is fixed by experiment design and applied identically to every
    method, so it cannot advantage one method over another.
    """
    ijc1, sjc1 = sum(ev.ijc1), sum(ev.sjc1)
    ijc2, sjc2 = sum(ev.ijc2), sum(ev.sjc2)
    if normalize:
        a1, b1 = _norm_counts(ijc1, sjc1, ev.inc_form_len, ev.skip_form_len)
        a2, b2 = _norm_counts(ijc2, sjc2, ev.inc_form_len, ev.skip_form_len)
    else:
        a1, b1, a2, b2 = float(ijc1), float(sjc1), float(ijc2), float(sjc2)
    if swap_groups:
        a_c, b_c, il_c = a2, b2, ev.inclevel2
        a_t, b_t, il_t = a1, b1, ev.inclevel1
        rmats_dpsi = -ev.rmats_dpsi
    else:
        a_c, b_c, il_c = a1, b1, ev.inclevel1
        a_t, b_t, il_t = a2, b2, ev.inclevel2
        rmats_dpsi = ev.rmats_dpsi
    return EventCounts(
        a_c=a_c, b_c=b_c, a_t=a_t, b_t=b_t,
        inclevel_ctrl=il_c, inclevel_treat=il_t,
        total_support=float(ijc1 + sjc1 + ijc2 + sjc2),
        rmats_dpsi=rmats_dpsi, rmats_fdr=ev.rmats_fdr,
    )


# --- ΔPSI posterior sampling (shared by betAS / Jeffreys / conformal) -------


def _dpsi_samples(ec: EventCounts, jeffreys: float, rng: np.random.Generator,
                  n: int = 6000) -> np.ndarray:
    """Difference of two per-group Beta posteriors.

    ``jeffreys=0.5`` -> Beta(inc+0.5, exc+0.5)  (BRAID).
    ``jeffreys=0.0`` -> Beta(inc, exc)           (betAS-style; +eps to stay valid).
    Convention: ctrl - treat (sample_1 - sample_2), matching rMATS
    IncLevelDifference = IncLevel1 - IncLevel2.
    """
    eps = 1e-6 if jeffreys == 0.0 else 0.0
    ctrl = rng.beta(ec.a_c + jeffreys + eps, ec.b_c + jeffreys + eps, size=n)
    treat = rng.beta(ec.a_t + jeffreys + eps, ec.b_t + jeffreys + eps, size=n)
    return ctrl - treat


def beta_interval(ec: EventCounts, alpha: float, jeffreys: float,
                  rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Return (dpsi_mean, dpsi_std, low, high) for a difference-of-Betas method."""
    s = _dpsi_samples(ec, jeffreys, rng)
    mean = float(np.mean(s))
    std = float(np.std(s))
    low = float(np.percentile(s, 100 * alpha / 2))
    high = float(np.percentile(s, 100 * (1 - alpha / 2)))
    return mean, std, low, high


def rmats_interval(ec: EventCounts, alpha: float) -> tuple[float, float, float, float]:
    """rMATS per-replicate Welch-t CI from the IncLevel values per group.

    A naive per-event ΔPSI CI from the replicate IncLevels (rMATS itself reports a
    likelihood-ratio test, not an interval). With only 3 replicates/group a normal
    (z) quantile is anti-conservative; we use the **Student-t** quantile at the
    Welch-Satterthwaite df, which is the correct small-sample choice for a difference
    of two means with possibly unequal variance. Orientation-corrected so ΔPSI =
    ctrl - treat matches the RT-PCR convention (see ``event_counts``).
    """
    i1 = np.asarray(ec.inclevel_ctrl, dtype=float)
    i2 = np.asarray(ec.inclevel_treat, dtype=float)
    if i1.size == 0 or i2.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    n1, n2 = i1.size, i2.size
    mean = float(i1.mean() - i2.mean())
    # Per-group variance of the mean: s^2 / n.
    v1 = float(i1.var(ddof=1)) / n1 if n1 > 1 else 0.0
    v2 = float(i2.var(ddof=1)) / n2 if n2 > 1 else 0.0
    se = math.sqrt(v1 + v2)
    # Welch-Satterthwaite degrees of freedom (falls back to n1+n2-2 when variances
    # are degenerate / zero, e.g. all replicates identical).
    denom = (v1 * v1 / (n1 - 1) if n1 > 1 else 0.0) + (v2 * v2 / (n2 - 1) if n2 > 1 else 0.0)
    df = ((v1 + v2) ** 2 / denom) if denom > 0 else float(max(n1 + n2 - 2, 1))
    # Guard against degenerate zero-SE (all reps identical) with a small floor so
    # rMATS isn't handed an impossible 0-width interval.
    se = max(se, 1e-3)
    from scipy.stats import t as _student_t
    tq = float(_student_t.ppf(1 - alpha / 2, df))
    half = tq * se
    return mean, se, mean - half, mean + half


def _z_for(alpha: float) -> float:
    # Inverse normal CDF via rational approximation (avoid scipy dependency).
    from statistics import NormalDist
    return NormalDist().inv_cdf(1 - alpha / 2)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def interval_score(y: float, low: float, high: float, alpha: float) -> float:
    """Gneiting-Raftery interval score (lower is better)."""
    width = high - low
    penalty = 0.0
    if y < low:
        penalty += (2.0 / alpha) * (low - y)
    elif y > high:
        penalty += (2.0 / alpha) * (y - high)
    return width + penalty


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (labels: 1=positive, 0=negative)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    pos_n = int(np.sum(labels == 1))
    neg_n = int(np.sum(labels == 0))
    if pos_n == 0 or neg_n == 0:
        return float("nan")
    ranks = _rankdata(scores)
    rank_pos = float(ranks[labels == 1].sum())
    return float((rank_pos - pos_n * (pos_n + 1) / 2.0) / (pos_n * neg_n))


def _rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    sa = a[order]
    i = 0
    cur = np.empty(a.size, dtype=float)
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        cur[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    ranks[order] = cur
    return ranks


# ---------------------------------------------------------------------------
# Cross-fit split-conformal (honest: held-out intervals, no test leakage)
# ---------------------------------------------------------------------------


def conformal_crossfit(points: np.ndarray, truths: np.ndarray, scales: np.ndarray,
                       supports: np.ndarray, alpha: float, *, k: int = 5,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """K-fold cross-fit Mondrian split-conformal ΔPSI intervals.

    Each event's interval is built from a conformal quantile fit ONLY on the
    other folds, so the reported coverage is an honest held-out estimate. The
    quantile is stratified by support bin (Mondrian) with a pooled global
    fallback for sparse bins. Clipped to the ΔPSI range [-1, 1].
    """
    n = points.size
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    bins = assign_support_bins(supports)
    low = np.empty(n)
    high = np.empty(n)
    for fi in range(k):
        test = folds[fi]
        train = np.concatenate([folds[g] for g in range(k) if g != fi])
        gq = conformal_quantile(
            nonconformity_scores(points[train], truths[train], scales[train]), alpha)
        qbin: dict[object, float] = {}
        for b in np.unique(bins[train]):
            m = bins[train] == b
            qb = conformal_quantile(
                nonconformity_scores(points[train][m], truths[train][m], scales[train][m]),
                alpha)
            qbin[b] = qb if np.isfinite(qb) else gq
        for i in test:
            qb = qbin.get(bins[i], gq)
            if not np.isfinite(qb):
                qb = gq
            half = qb * scales[i]
            low[i] = np.clip(points[i] - half, -1.0, 1.0)
            high[i] = np.clip(points[i] + half, -1.0, 1.0)
    return low, high


def conformal_crossfit_grouped(points: np.ndarray, truths: np.ndarray, scales: np.ndarray,
                               group_labels: np.ndarray, alpha: float, *, k: int = 5,
                               min_bin: int = 20, seed: int = 0,
                               clip: tuple[float, float] = (-1.0, 1.0),
                               ) -> tuple[np.ndarray, np.ndarray]:
    """K-fold cross-fit Mondrian split-conformal intervals grouped by ``group_labels``.

    Like :func:`conformal_crossfit`, but the Mondrian partition is an arbitrary
    per-event label (e.g. rMATS event type, or an ``"event_type|support_bin"``
    composite) instead of the support bin. Each group's quantile is fit ONLY on the
    other folds (leakage-free); groups with fewer than ``min_bin`` training events
    fall back to the pooled global quantile, so the adaptive band never has fewer
    calibration points than the global constant band. ``clip`` bounds the interval
    ([-1, 1] for ΔPSI, [0, 1] for PSI).
    """
    n = points.size
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    gl = np.asarray(group_labels)
    low = np.empty(n)
    high = np.empty(n)
    for fi in range(k):
        test = folds[fi]
        train = np.concatenate([folds[g] for g in range(k) if g != fi])
        gq = conformal_quantile(
            nonconformity_scores(points[train], truths[train], scales[train]), alpha)
        qg: dict[object, float] = {}
        for b in np.unique(gl[train]):
            m = gl[train] == b
            if int(m.sum()) < min_bin:
                continue  # sparse group -> pooled global fallback (no leakage, no thin fit)
            qb = conformal_quantile(
                nonconformity_scores(points[train][m], truths[train][m], scales[train][m]), alpha)
            qg[b] = qb if np.isfinite(qb) else gq
        for i in test:
            qb = qg.get(gl[i], gq)
            if not np.isfinite(qb):
                qb = gq
            half = qb * scales[i]
            low[i] = float(np.clip(points[i] - half, clip[0], clip[1]))
            high[i] = float(np.clip(points[i] + half, clip[0], clip[1]))
    return low, high


def boundary_proximity(ec: EventCounts) -> float:
    """Closest-to-boundary group PSI: ``min`` over groups of ``min(psi, 1-psi)``.

    Uses the Jeffreys posterior mean per group from the (length-normalized)
    EventCounts. Small values flag near-0/near-1 (boundary) events, where a
    symmetric constant band is most likely to mis-cover.
    """
    psi_c = (ec.a_c + 0.5) / (ec.a_c + ec.b_c + 1.0)
    psi_t = (ec.a_t + 0.5) / (ec.a_t + ec.b_t + 1.0)
    return float(min(min(psi_c, 1.0 - psi_c), min(psi_t, 1.0 - psi_t)))


# ---------------------------------------------------------------------------
# Per-dataset head-to-head
# ---------------------------------------------------------------------------

NOMINAL_LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)


def _stable_method_seed_offset(method_name: str, modulo: int = 1000) -> int:
    """Deterministic per-method seed offset, independent of Python hash salt."""
    digest = hashlib.blake2s(method_name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def _load_betas_intervals(path: str, n: int) -> dict[str, tuple]:
    """Load real-betAS intervals (from run_betas.R) keyed by row order ev0..ev{n-1}.

    Returns {level: (lows[n], highs[n], means[n])}. The TSV must contain exactly
    the expected row-order keys. Missing rows used to be filled with vacuous
    [-1, 1] intervals, which silently inflated coverage; fail fast instead.
    """
    rows: dict[int, dict] = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = row.get("key", "")
            if not key.startswith("ev"):
                raise ValueError(f"Invalid betAS interval key {key!r} in {path}; expected evN.")
            try:
                idx = int(key[2:])  # "ev123" -> 123
            except ValueError as exc:
                raise ValueError(
                    f"Invalid betAS interval key {key!r} in {path}; expected evN."
                ) from exc
            if idx in rows:
                raise ValueError(
                    f"Duplicate betAS interval row {key!r} in {path}; row-order alignment failed."
                )
            rows[idx] = row
    expected = set(range(n))
    found = set(rows)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing or extra:
        parts = []
        if missing:
            preview = ", ".join(f"ev{i}" for i in missing[:5])
            suffix = "..." if len(missing) > 5 else ""
            parts.append(f"missing betAS interval rows: {preview}{suffix}")
        if extra:
            preview = ", ".join(f"ev{i}" for i in extra[:5])
            suffix = "..." if len(extra) > 5 else ""
            parts.append(f"unexpected betAS interval rows: {preview}{suffix}")
        raise ValueError(
            f"betAS interval row-order alignment failed for {path}: expected ev0..ev{n - 1}; "
            + "; ".join(parts)
        )
    out: dict[str, tuple] = {}
    for lvl in NOMINAL_LEVELS:
        lo = np.empty(n)
        hi = np.empty(n)
        mean = np.empty(n)
        for i in range(n):
            r = rows[i]
            lo[i] = float(r[f"low_{lvl:.2f}"])
            hi[i] = float(r[f"high_{lvl:.2f}"])
            mean[i] = float(r["dpsi_mean"])
        out[f"{lvl:.2f}"] = (lo, hi, mean)
    return out


def run_dataset(rmats_se_path: str, validated_tsv: str | None, failed_tsv: str | None,
                name: str, *, seed: int = 7, betas_tsv: str | None = None,
                swap_groups: bool = False,
                targets: list[Target] | None = None) -> dict:
    events = parse_se_table(rmats_se_path)
    if targets is None:
        targets = load_targets(validated_tsv, failed_tsv)

    rng = np.random.default_rng(seed)
    rows = []
    unmatched = 0
    for t in targets:
        ev = match_event(t, events)
        if ev is None:
            unmatched += 1
            continue
        ec = event_counts(ev, normalize=True, swap_groups=swap_groups)
        ec_raw = event_counts(ev, normalize=False, swap_groups=swap_groups)
        # Skip events with no usable support in either group.
        if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
            continue
        bj_mean, bj_std, _, _ = beta_interval(ec, 0.05, 0.5, rng)
        raw_mean, _, _, _ = beta_interval(ec_raw, 0.05, 0.5, rng)
        rows.append({
            "gene": t.gene, "chrom": t.chrom,
            "truth": t.dpsi_rtpcr, "is_positive": t.is_positive,
            "ec": ec, "ec_raw": ec_raw,
            "support": ec.total_support,
            "bj_mean": bj_mean, "bj_std": bj_std, "raw_mean": raw_mean,
        })

    n = len(rows)
    truths = np.array([r["truth"] for r in rows])
    labels = np.array([1 if r["is_positive"] else 0 for r in rows])
    supports = np.array([r["support"] for r in rows])
    points = np.array([r["bj_mean"] for r in rows])
    scales = np.maximum(np.array([r["bj_std"] for r in rows]), 1e-6)

    # Optionally load REAL betAS intervals (per-event evN keyed by row order).
    betas_real = _load_betas_intervals(betas_tsv, n) if betas_tsv else None
    betas_label = "betAS(real)" if betas_real else "betAS"

    # Parametric methods: build intervals at every nominal level.
    methods_param = {
        "rMATS-perRep": ("rmats", None),
        betas_label: ("beta", 0.0),
        "BRAID-Jeffreys": ("beta", 0.5),
    }
    results: dict[str, dict] = {}

    for mname, (kind, jeff) in methods_param.items():
        is_betas = mname == betas_label
        per_level = {}
        # one rng per method for reproducibility
        mrng = np.random.default_rng(seed + _stable_method_seed_offset(mname))
        for lvl in NOMINAL_LEVELS:
            a = 1.0 - lvl
            if is_betas and betas_real is not None:
                lows, highs, means = betas_real[f"{lvl:.2f}"]
            else:
                lows = np.empty(n)
                highs = np.empty(n)
                means = np.empty(n)
                for i, r in enumerate(rows):
                    if kind == "rmats":
                        m, _s, lo, hi = rmats_interval(r["ec"], a)
                    else:
                        m, _s, lo, hi = beta_interval(r["ec"], a, jeff, mrng)
                    lows[i], highs[i], means[i] = lo, hi, m
            cov = float(np.mean((truths >= lows) & (truths <= highs)))
            width = float(np.mean(highs - lows))
            iscore = float(np.mean([
                interval_score(truths[i], lows[i], highs[i], a) for i in range(n)]))
            per_level[f"{lvl:.2f}"] = {"coverage": cov, "width": width,
                                       "interval_score": iscore}
        # detection at 95%: interval excludes zero
        a = 0.05
        if is_betas and betas_real is not None:
            lows95, highs95, means95 = betas_real["0.95"]
            excl = (lows95 > 0) | (highs95 < 0)
        else:
            excl = np.empty(n, dtype=bool)
            means95 = np.empty(n)
            for i, r in enumerate(rows):
                if kind == "rmats":
                    m, _s, lo, hi = rmats_interval(r["ec"], a)
                else:
                    m, _s, lo, hi = beta_interval(r["ec"], a, jeff, mrng)
                excl[i] = (lo > 0) or (hi < 0)
                means95[i] = m
        results[mname] = _finalize(mname, per_level, means95, excl, truths, labels, scales)

    # Conformal variants (cross-fit, honest held-out). All share the BRAID-Jeffreys
    # point estimate; they differ only in the nonconformity SCALE:
    #   - "BRAID-conformal"     : sigma = posterior std (BRAID's production scale).
    #   - "BRAID-conformal-abs" : sigma = 1 (absolute-residual conformal). Motivated
    #     by the empirical finding that real ΔPSI-vs-RT-PCR error is platform-
    #     dominated (nearly independent of read depth / posterior std), so a
    #     constant-width schedule is better calibrated per support bin and sharper.
    for cname, cscale in (("BRAID-conformal", scales),
                          ("BRAID-conformal-abs", np.ones(n))):
        per_level_c = {}
        excl_c = np.empty(n, dtype=bool)
        for lvl in NOMINAL_LEVELS:
            a = 1.0 - lvl
            lows, highs = conformal_crossfit(points, truths, cscale, supports, a, seed=seed)
            cov = float(np.mean((truths >= lows) & (truths <= highs)))
            width = float(np.mean(highs - lows))
            iscore = float(np.mean([
                interval_score(truths[i], lows[i], highs[i], a) for i in range(n)]))
            per_level_c[f"{lvl:.2f}"] = {"coverage": cov, "width": width,
                                         "interval_score": iscore}
            if abs(lvl - 0.95) < 1e-9:
                excl_c = (lows > 0) | (highs < 0)
        results[cname] = _finalize(
            cname, per_level_c, points, excl_c, truths, labels, scales)

    # Length-normalization bias diagnostic (raw vs normalized point estimate).
    raw_means = np.array([r["raw_mean"] for r in rows])
    norm_rmse = float(np.sqrt(np.mean((points - truths) ** 2)))
    raw_rmse = float(np.sqrt(np.mean((raw_means - truths) ** 2)))

    return {
        "dataset": name,
        "n_targets": len(targets),
        "n_matched": n,
        "n_unmatched": unmatched,
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "methods": results,
        "length_norm_diagnostic": {
            "braid_jeffreys_normalized_rmse_vs_rtpcr": norm_rmse,
            "braid_jeffreys_raw_rmse_vs_rtpcr": raw_rmse,
        },
    }


def _finalize(mname: str, per_level: dict, means95: np.ndarray, excl: np.ndarray,
              truths: np.ndarray, labels: np.ndarray, scales: np.ndarray) -> dict:
    pos = labels == 1
    neg = labels == 0
    sens = float(np.mean(excl[pos])) if pos.any() else float("nan")
    fpr = float(np.mean(excl[neg])) if neg.any() else float("nan")
    # direction accuracy on clear positives (|truth| > 0.05)
    clear = pos & (np.abs(truths) > 0.05)
    diracc = float(np.mean(np.sign(means95[clear]) == np.sign(truths[clear]))) \
        if clear.any() else float("nan")
    rmse = float(np.sqrt(np.mean((means95 - truths) ** 2)))
    det_score = np.abs(means95) / np.maximum(scales, 1e-6)
    det_auc = auc(det_score, labels)
    return {
        "per_level": per_level,
        "coverage95": per_level["0.95"]["coverage"],
        "width95": per_level["0.95"]["width"],
        "interval_score95": per_level["0.95"]["interval_score"],
        "rmse_point": rmse,
        "direction_acc": diracc,
        "sensitivity95": sens,
        "fpr95": fpr,
        "detection_auc": det_auc,
    }


def _fmt(x: float, w: int = 7, p: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return f"{'NA':>{w}}"
    return f"{x:>{w}.{p}f}"


def print_report(res: dict) -> None:
    d = res
    print(f"\n{'='*92}")
    print(f"HEAD-TO-HEAD ΔPSI coverage vs RT-PCR  —  {d['dataset']}")
    print(f"  matched {d['n_matched']}/{d['n_targets']} targets "
          f"({d['n_positive']} positive, {d['n_negative']} negative); "
          f"unmatched={d['n_unmatched']}")
    diag = d["length_norm_diagnostic"]
    print(f"  length-norm diagnostic: ΔPSI RMSE vs RT-PCR — "
          f"normalized={diag['braid_jeffreys_normalized_rmse_vs_rtpcr']:.3f}  "
          f"raw={diag['braid_jeffreys_raw_rmse_vs_rtpcr']:.3f}")
    print(f"{'-'*92}")
    print(f"{'method':<17}{'cov@95':>8}{'width95':>9}{'iscore95':>10}"
          f"{'pt-RMSE':>9}{'dir-acc':>9}{'sens95':>8}{'fpr95':>8}{'detAUC':>8}")
    print(f"{'-'*92}")
    for m, r in d["methods"].items():
        print(f"{m:<17}{_fmt(r['coverage95'],8)}{_fmt(r['width95'],9)}"
              f"{_fmt(r['interval_score95'],10)}{_fmt(r['rmse_point'],9)}"
              f"{_fmt(r['direction_acc'],9)}{_fmt(r['sensitivity95'],8)}"
              f"{_fmt(r['fpr95'],8)}{_fmt(r['detection_auc'],8)}")
    print(f"{'-'*92}")
    print("calibration (empirical coverage at each nominal level):")
    print(f"{'method':<17}" + "".join(f"{f'{lv:.2f}':>9}" for lv in NOMINAL_LEVELS))
    for m, r in d["methods"].items():
        print(f"{m:<17}" + "".join(
            _fmt(r['per_level'][f'{lv:.2f}']['coverage'], 9) for lv in NOMINAL_LEVELS))
    print(f"{'='*92}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rmats-se", required=True, help="rMATS SE.MATS.JC(EC).txt")
    ap.add_argument("--validated", default=None, help="validated_events.tsv (positives)")
    ap.add_argument("--failed", default=None, help="failed_events.tsv (negatives)")
    ap.add_argument("--circadian-tsv", default=None,
                    help="junction-based circadian positives TSV (alt to --validated)")
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--betas-tsv", default=None,
                    help="optional real-betAS output to substitute (gene\\tdpsi\\tlow\\thigh)")
    ap.add_argument("--swap-groups", action="store_true",
                    help="orient ΔPSI as sample_2 - sample_1 (RT-PCR convention "
                         "when rMATS b1=control, b2=treatment, e.g. TRA2)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    targets = None
    if args.circadian_tsv:
        targets = load_circadian_targets(args.circadian_tsv)
    res = run_dataset(args.rmats_se, args.validated, args.failed, args.name,
                      seed=args.seed, betas_tsv=args.betas_tsv,
                      swap_groups=args.swap_groups, targets=targets)
    print_report(res)
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(_json_safe(res), f, indent=2, default=str, allow_nan=False)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
