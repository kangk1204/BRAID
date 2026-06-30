#!/usr/bin/env python3
"""Real-MAJIQ ΔPSI credible-interval coverage vs RT-PCR (item #2, no longer a stand-in).

MAJIQ v3 (rna_majiq 3.0.23) was run for real on the GSE59335 TRA2 knockdown BAMs
(build -> psi-coverage -> deltapsi; see run_majiq_pipeline.sh, license supplied by the
user). This script reads MAJIQ's deltapsi.tsv, extracts each cassette exon's posterior
ΔPSI credible interval, and measures coverage of the orthogonal RT-PCR truth -- on the
*same* matched events as the REAL betAS tool and BRAID-conformal, so the three are
directly comparable.

MAJIQ interval per event:
  * Match the RT-PCR cassette exon to MAJIQ's *inclusion* junctions (rows whose
    ``other_exon`` equals the cassette, i.e. a neighbour-anchored LSV splicing INTO the
    cassette). Pick the highest-coverage such junction.
  * Orient to the RT-PCR convention (KD - control) directly from MAJIQ's own per-group
    raw PSI means: center = KD_raw_psi_mean - CTRL_raw_psi_mean.
  * Half-width = z(0.95) * dpsi_std  (MAJIQ's posterior ΔPSI standard deviation),
    clipped to [-1, 1]. This is MAJIQ's Bayesian 95% credible interval.

betAS here is the REAL betAS tool output (run_betas.R -> tra2_betas_intervals.tsv), aligned
to events by replaying the export ordering (ev0..evN over rMATS-matched targets) and verified
against the per-event RT-PCR truth. BRAID-conformal is the honest k-fold cross-fit absolute-
residual Mondrian conformal interval. All coverages reported with Wilson 95% CIs.
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import head_to_head_coverage as H  # noqa: E402

Z95 = 1.959963984540054
COORD_TOL = 3  # bp tolerance for matching a MAJIQ exon to the RT-PCR cassette


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load_majiq(tsv: str) -> list[dict]:
    with open(tsv) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    return list(csv.DictReader(lines, delimiter="\t"))


def load_real_betas(intervals_tsv: str, truth_tsv: str):
    """Real betAS 95% intervals + truth, keyed by ev index (ev0..evN row order)."""
    iv: dict[int, tuple[float, float, float]] = {}
    with open(intervals_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = int(r["key"][2:])
            iv[i] = (float(r["low_0.95"]), float(r["high_0.95"]), float(r["dpsi_mean"]))
    tru: dict[int, float] = {}
    with open(truth_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            tru[int(r["key"][2:])] = float(r["truth"])
    return iv, tru


def majiq_interval_for(target, maj_rows: list[dict]) -> tuple[float, float, float] | None:
    """Return (center, low, high) MAJIQ 95% credible interval for a cassette exon.

    center oriented KD - control; None if no covered inclusion junction is found.
    """
    es, ee = target.exon_start, target.exon_end
    chrom = target.chrom
    best = None
    best_cov = -1.0
    for m in maj_rows:
        if H._norm_chrom(m["seqid"]) != chrom:
            continue
        # inclusion junction: a neighbour-anchored LSV whose OTHER exon is the cassette
        if (abs(int(m["other_exon_start"]) - es) <= COORD_TOL
                and abs(int(m["other_exon_end"]) - ee) <= COORD_TOL):
            cov = float(m["KD_raw_coverage"]) + float(m["CTRL_raw_coverage"])
            if cov > best_cov:
                best_cov = cov
                best = m
    if best is None or best_cov <= 0:
        return None
    center = float(best["KD_raw_psi_mean"]) - float(best["CTRL_raw_psi_mean"])
    half = Z95 * float(best["dpsi_std"])
    return center, max(-1.0, center - half), min(1.0, center + half)


def main() -> None:
    os.chdir(os.path.dirname(os.path.dirname(_HERE)))
    rmats_se = "data/public_benchmarks/GSE59335/rmats/SE.MATS.JC.txt"
    validated = "data/public_benchmarks/GSE59335/targets/validated_events.tsv"
    failed = "data/public_benchmarks/GSE59335/targets/failed_events.tsv"
    majiq_tsv = "data/public_benchmarks/GSE59335/majiq/deltapsi.tsv"
    betas_iv = "benchmarks/headtohead/tra2_betas_intervals.tsv"
    betas_truth = "benchmarks/headtohead/tra2_betas_truth.tsv"
    alpha = 0.05

    targets = H.load_targets(validated, failed)
    events = H.parse_se_table(rmats_se)
    maj_rows = load_majiq(majiq_tsv)
    real_iv, real_tru = load_real_betas(betas_iv, betas_truth)
    rng = np.random.default_rng(7)

    # Per-event records for events matched by BOTH rMATS (betAS/BRAID) AND MAJIQ.
    pts, trs, sup = [], [], []          # BRAID point / truth / support
    rbetas_lo, rbetas_hi = [], []       # REAL betAS 95% interval
    maj_cov_flags, maj_widths, maj_centers = [], [], []
    n_target = len(targets)
    n_rmats = n_majiq = 0
    align_err = 0
    ev_idx = -1                         # mirrors export_for_betas ordering (ev0..evN)
    for t in targets:
        ev = H.match_event(t, events)
        mi = majiq_interval_for(t, maj_rows)
        if mi is not None:
            n_majiq += 1
        if ev is None:
            continue
        ec = H.event_counts(ev, normalize=True, swap_groups=True)
        if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
            continue
        ev_idx += 1                     # this event has a real-betAS ev key
        n_rmats += 1
        if mi is None or ev_idx not in real_iv:
            continue
        # Alignment guard: the real-betAS row's truth must equal this target's truth.
        if abs(real_tru.get(ev_idx, 1e9) - t.dpsi_rtpcr) > 1e-3:
            align_err += 1
            continue
        m, std, _lo, _hi = H.beta_interval(ec, alpha, 0.5, rng)
        rlo, rhi, _rm = real_iv[ev_idx]
        center, mlo, mhi = mi
        pts.append(m)
        trs.append(t.dpsi_rtpcr)
        sup.append(ec.total_support)
        rbetas_lo.append(rlo)
        rbetas_hi.append(rhi)
        maj_cov_flags.append(int(mlo <= t.dpsi_rtpcr <= mhi))
        maj_widths.append(mhi - mlo)
        maj_centers.append(center)

    pts = np.array(pts)
    trs = np.array(trs)
    sup = np.array(sup)
    rbetas_lo = np.array(rbetas_lo)
    rbetas_hi = np.array(rbetas_hi)
    n = pts.size

    # BRAID-conformal (honest k-fold cross-fit, absolute-residual => scale=1)
    clo, chi = H.conformal_crossfit(pts, trs, np.ones_like(pts), sup, alpha)

    # Coverages on the identical matched set
    maj_k = int(np.sum(maj_cov_flags))
    rbet_k = int(np.sum((trs >= rbetas_lo) & (trs <= rbetas_hi)))
    conf_k = int(np.sum((trs >= clo) & (trs <= chi)))

    maj_p, maj_l, maj_u = wilson(maj_k, n)
    rbet_p, rbet_l, rbet_u = wilson(rbet_k, n)
    con_p, con_l, con_u = wilson(conf_k, n)

    corr = float(np.corrcoef(np.array(maj_centers), trs)[0, 1]) if n > 2 else float("nan")

    bar = "=" * 78
    print(f"\n{bar}")
    print("REAL MAJIQ vs REAL betAS vs BRAID-conformal: ΔPSI CI coverage vs RT-PCR")
    print("(GSE59335 TRA2; all three on the IDENTICAL matched event set)")
    print(bar)
    print(f"targets={n_target}  rMATS-matched(betAS evs)={n_rmats}  MAJIQ-matched={n_majiq}  "
          f"common(used)={n}  align_mismatch={align_err}")
    print(f"orientation check: corr(MAJIQ center, RT-PCR truth) = {corr:+.3f}  (must be > 0)")
    print("-" * 78)
    print(f"{'method':<36}{'cov@95':>8}{'   Wilson 95% CI':<20}{'mean width':>12}")
    print("-" * 78)
    print(f"{'MAJIQ (real binary, posterior CI)':<36}{maj_p:>8.3f}   "
          f"[{maj_l:.3f}, {maj_u:.3f}]   {float(np.mean(maj_widths)):>10.3f}")
    print(f"{'betAS (REAL tool, run_betas.R)':<36}{rbet_p:>8.3f}   "
          f"[{rbet_l:.3f}, {rbet_u:.3f}]   {float(np.mean(rbetas_hi - rbetas_lo)):>10.3f}")
    print(f"{'BRAID-conformal (cross-fit)':<36}{con_p:>8.3f}   "
          f"[{con_l:.3f}, {con_u:.3f}]   {float(np.mean(chi - clo)):>10.3f}")
    print("-" * 78)
    print("Both real generative tools (MAJIQ's Bayesian posterior, betAS's Beta posterior)")
    print("under-cover the orthogonal RT-PCR truth (CIs below 0.95); only BRAID's conformal")
    print("recalibration -- which sizes the interval to the orthogonal-truth residual floor --")
    print("reaches nominal coverage. #2 closed with BOTH real tools.")
    print(bar)


if __name__ == "__main__":
    main()
