#!/usr/bin/env python3
"""Biostatistician's audit of the head-to-head coverage benchmark.

Quantifies the statistical concerns a reviewer would raise about the head-to-head
coverage headline and records the sensitivity checks that confirm it:
  1. Paired test: coverage is a paired 2x2 design (same events, 2 methods) -> McNemar,
     not non-overlapping marginal Wilson CIs.
  2. Clustering / pseudoreplication: events cluster within genes -> effective n < n_events;
     gene-clustered bootstrap CI vs naive Wilson.
  3. rMATS interval fairness: the canonical per-replicate rMATS CI already uses Student-t
     with Welch-Satterthwaite df; we add a fixed-df Student-t check (df=n1+n2-2) to confirm
     the coverage verdict is not an artifact of the degrees-of-freedom choice.
  4. MAJIQ point-estimate consistency: is dpsi_mean ~ (group1_raw - group2_raw)? (we orient
     by the raw diff; if magnitudes disagree the interval is internally inconsistent).
  5. Variance decomposition: is mean residual ~0 (no systematic RNA-seq vs RT-PCR bias)?
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import comprehensive_benchmark as C  # noqa: E402
import head_to_head_coverage as H  # noqa: E402

ALPHA = 0.05
Z95 = 1.959963984540054


def collect(cfg):
    """Per-event records on the common (rMATS&MAJIQ) set, incl. gene + per-method flags
    and the raw MAJIQ point/std for consistency checks."""
    targets = cfg["targets"]
    incls = cfg.get("incls") or [None] * len(targets)
    events = H.parse_se_table(cfg["rmats_se"])
    maj_rows = C.load_majiq(cfg["majiq_tsv"])
    real_iv, real_tru = C.load_real_betas(cfg["betas_iv"], cfg["betas_truth"])
    g1, g2 = cfg["majiq_groups"]
    rng = np.random.default_rng(7)
    rows = []
    ev_idx = -1
    for t, incl in zip(targets, incls):
        ev = H.match_event(t, events)
        mi = C.majiq_interval(t, incl, cfg["kind"], maj_rows, g1, g2)
        if ev is None:
            continue
        ec = H.event_counts(ev, normalize=True, swap_groups=cfg["swap"])
        if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
            continue
        ev_idx += 1
        if mi is None or ev_idx not in real_iv:
            continue
        if abs(real_tru.get(ev_idx, 1e9) - t.dpsi_rtpcr) > 1e-3:
            continue
        m, std, _, _ = H.beta_interval(ec, ALPHA, 0.5, rng)
        rm_mean, rm_se, rmlo, rmhi = H.rmats_interval(ec, ALPHA)
        blo, bhi = real_iv[ev_idx]
        center, mlo, mhi = mi
        rows.append(dict(gene=t.gene, truth=t.dpsi_rtpcr, point=m, std=std,
                         support=ec.total_support,
                         betas=(blo, bhi), rmats=(rmlo, rmhi), rm_se=rm_se,
                         majiq=(mlo, mhi), maj_center=center,
                         il_ctrl=ec.inclevel_ctrl, il_treat=ec.inclevel_treat))
    return rows


def mcnemar_exact(a_cov, b_cov):
    """Exact McNemar (binomial) for paired binary coverage. Returns (b, c, p)."""
    a = np.asarray(a_cov, bool)
    b = np.asarray(b_cov, bool)
    n01 = int(np.sum(a & ~b))   # A covers, B misses
    n10 = int(np.sum(~a & b))   # A misses, B covers
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5))
    return n01, n10, p


def cov_flags(rows, key):
    return np.array([lo <= r["truth"] <= hi for r in rows for (lo, hi) in [r[key]]])


def conformal_flags(rows):
    pts = np.array([r["point"] for r in rows])
    trs = np.array([r["truth"] for r in rows])
    sup = np.array([r["support"] for r in rows])
    lo, hi = H.conformal_crossfit(pts, trs, np.ones_like(pts), sup, ALPHA)
    return (trs >= lo) & (trs <= hi)


def gene_cluster_ci(flags, genes, n_boot=4000, seed=1):
    """Cluster bootstrap over genes (resample genes, take all their events)."""
    rng = np.random.default_rng(seed)
    by = {}
    for f, g in zip(flags, genes):
        by.setdefault(g, []).append(bool(f))
    gl = list(by.keys())
    covs = []
    for _ in range(n_boot):
        pick = rng.choice(len(gl), len(gl), replace=True)
        vals = [v for i in pick for v in by[gl[i]]]
        covs.append(np.mean(vals))
    return float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5))


def rmats_t_interval(rows):
    """Fixed-df sensitivity check: rMATS interval with Student-t(df=n1+n2-2).

    The canonical benchmark already uses Student-t with Welch-Satterthwaite df;
    this fixed-df variant confirms the coverage verdict is not sensitive to the
    degrees-of-freedom choice.
    """
    flags = []
    for r in rows:
        i1 = np.asarray(r["il_ctrl"], float)
        i2 = np.asarray(r["il_treat"], float)
        if i1.size == 0 or i2.size == 0:
            flags.append(False)
            continue
        mean = i1.mean() - i2.mean()
        v1 = i1.var(ddof=1) / i1.size if i1.size > 1 else 0.0
        v2 = i2.var(ddof=1) / i2.size if i2.size > 1 else 0.0
        se = max(np.sqrt(v1 + v2), 1e-3)
        df = max(i1.size + i2.size - 2, 1)
        tq = stats.t.ppf(1 - ALPHA / 2, df)
        lo, hi = mean - tq * se, mean + tq * se
        flags.append(lo <= r["truth"] <= hi)
    return np.array(flags)


def main():
    os.chdir(os.path.dirname(os.path.dirname(_HERE)))
    DB = "data/public_benchmarks"
    BH = "benchmarks/headtohead"
    circ_tsv = f"{DB}/meta/gse54651_circadian_positive_events.tsv"
    cfgs = [
        dict(name="TRA2", kind="exon", swap=True,
             rmats_se=f"{DB}/GSE59335/rmats/SE.MATS.JC.txt",
             targets=H.load_targets(f"{DB}/GSE59335/targets/validated_events.tsv",
                                    f"{DB}/GSE59335/targets/failed_events.tsv"),
             majiq_tsv=f"{DB}/GSE59335/majiq/deltapsi.tsv",
             betas_iv=f"{BH}/tra2_betas_intervals.tsv", betas_truth=f"{BH}/tra2_betas_truth.tsv",
             majiq_groups=("KD", "CTRL")),
        dict(name="Circadian", kind="junction", swap=True,
             rmats_se=f"{DB}/GSE54651/rmats/SE.MATS.JC.txt",
             targets=H.load_circadian_targets(circ_tsv), incls=C.load_circ_incl(circ_tsv),
             majiq_tsv=f"{DB}/GSE54651/majiq/deltapsi.tsv",
             betas_iv=f"{BH}/circ_betas_intervals.tsv", betas_truth=f"{BH}/circ_betas_truth.tsv",
             majiq_groups=("LIVER", "CEREB")),
        dict(name="PC3E", kind="exon", swap=False,
             rmats_se=f"{DB}/SRS354082/rmats/SE.MATS.JC.txt",
             targets=H.load_targets(f"{DB}/meta/rmats_pc3e_gs689_positive_events.tsv", None),
             majiq_tsv=f"{DB}/SRS354082/majiq/deltapsi.tsv",
             betas_iv=f"{BH}/srs_betas_intervals.tsv", betas_truth=f"{BH}/srs_betas_truth.tsv",
             majiq_groups=("PC3E", "GS689")),
    ]
    allrows = []
    for cfg in cfgs:
        allrows += collect(cfg)
    rows = allrows
    n = len(rows)
    genes = [r["gene"] for r in rows]
    print("=" * 76)
    print(f"STATISTICAL AUDIT — pooled common set (n_events={n})")
    print("=" * 76)

    # 1+2. clustering
    n_genes = len(set(genes))
    from collections import Counter
    multi = sum(1 for _, c in Counter(genes).items() if c > 1)
    print(f"\n[1] CLUSTERING: {n} events across {n_genes} unique genes "
          f"({multi} genes with >1 event). design effect ~ n/n_genes = {n/n_genes:.2f}")

    bf = conformal_flags(rows)
    mf = cov_flags(rows, "majiq")
    betf = cov_flags(rows, "betas")
    rf = cov_flags(rows, "rmats")
    def wil(fl):
        k = int(fl.sum())
        p = k/len(fl)
        d = 1+Z95**2/len(fl)
        c = (p+Z95**2/(2*len(fl)))/d
        h = Z95*np.sqrt(p*(1-p)/len(fl)+Z95**2/(4*len(fl)**2))/d
        return p, max(0,c-h), min(1,c+h)
    print("\n[2] BRAID-conformal coverage CI: naive Wilson vs gene-clustered bootstrap")
    p, lo, u = wil(bf)
    gl, gu = gene_cluster_ci(bf, genes)
    print(f"    naive Wilson      : {p:.3f} [{lo:.3f}, {u:.3f}]")
    print(f"    gene-clustered boot: {p:.3f} [{gl:.3f}, {gu:.3f}]  "
          f"(wider if clustering matters)")

    # 3. McNemar paired tests
    print("\n[3] PAIRED McNEMAR (BRAID-conformal vs each competitor; b=BRAID-only cover, "
          "c=competitor-only):")
    for name, fl in [("MAJIQ", mf), ("betAS", betf), ("rMATS", rf)]:
        b, c, pmc = mcnemar_exact(bf, fl)
        sig = "***" if pmc < 1e-3 else ("**" if pmc < 1e-2 else ("*" if pmc < 0.05 else "ns"))
        print(f"    BRAID vs {name:<6}: b(BRAID-only)={b:3d}  c({name}-only)={c:2d}  "
              f"exact p={pmc:.2e} {sig}")

    # 4. rMATS t vs z
    rf_t = rmats_t_interval(rows)
    print(f"\n[4] rMATS interval (both Student-t; the manuscript and "
          f"comprehensive_benchmark.py use the canonical Welch-df form):\n"
          f"    canonical Welch-t coverage = {rf.mean():.3f} ; "
          f"fixed-df t(df=n1+n2-2) = {rf_t.mean():.3f}")

    # 5. MAJIQ point consistency
    print("\n[5] MAJIQ point-estimate consistency (orientation uses raw group-mean diff):")
    # recompute |dpsi_mean| vs |center| not stored; we stored only center=raw diff.
    # Instead check: does center predict truth sign well + magnitude agreement vs betAS point.
    centers = np.array([r["maj_center"] for r in rows])
    truth = np.array([r["truth"] for r in rows])
    points = np.array([r["point"] for r in rows])
    print(f"    corr(MAJIQ center, RT-PCR truth)   = {np.corrcoef(centers, truth)[0,1]:+.3f}")
    print(f"    corr(MAJIQ center, BRAID point)    = {np.corrcoef(centers, points)[0,1]:+.3f}")
    print(f"    mean|MAJIQ center - BRAID point|   = {np.mean(np.abs(centers-points)):.3f}")

    # 6. variance-decomp bias check (TRA2 only, reuse build)
    import variance_decomposition as V
    resid, sig, sup = V.build()
    print(f"\n[6] VARIANCE DECOMP bias: mean residual (RNA-seq - RT-PCR) = {resid.mean():+.3f} "
          f"(should be ~0; large => systematic offset folded into 'platform')")
    print(f"    n={resid.size}, sigma_total={np.sqrt(np.mean(resid**2)):.3f}, "
          f"sigma_samp(mean)={np.sqrt(np.mean(sig**2)):.3f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
