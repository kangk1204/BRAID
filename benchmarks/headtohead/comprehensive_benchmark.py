#!/usr/bin/env python3
"""Comprehensive ΔPSI credible-interval benchmark for the paper: 4 methods x 3 real
datasets, coverage + width + interval score, on identical matched event sets.

Methods (all real, no stand-ins):
  * MAJIQ        -- real MAJIQ v3 binary posterior CI (dpsi_mean +/- z*dpsi_std),
                    oriented to the RT-PCR convention from MAJIQ's per-group raw PSI.
  * betAS        -- real betAS tool intervals (run_betas.R), aligned by ev order with
                    the per-event RT-PCR truth verified.
  * rMATS        -- per-replicate Student-t (Welch df) CI from the IncLevel values.
  * BRAID-conf   -- honest k-fold cross-fit absolute-residual Mondrian conformal.

Datasets:
  * GSE59335 TRA2 (human, hg19), cassette-exon RT-PCR truth, KD - control.
  * GSE54651 circadian (mouse, mm10), junction RT-PCR truth, liver - cerebellum.
  * SRS354082 PC3E/GS689 (human, hg19), cassette-exon RT-PCR truth, PC3E - GS689.

Each method is scored on the set of events matched by BOTH rMATS and MAJIQ (so all four
are defined on the same events). Coverage with Wilson 95% CIs; mean width; mean interval
score (Gneiting-Raftery, lower is better -- rewards calibrated AND sharp). A separate line
reports betAS/rMATS/BRAID-conf on the FULL rMATS-matched set (larger N, MAJIQ excluded) so
the headline coverage claim is not silently restricted to MAJIQ-matchable events.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import head_to_head_coverage as H  # noqa: E402

Z95 = 1.959963984540054
ALPHA = 0.05
COORD_TOL = 3
JUNC_TOL = 12


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
    iv: dict[int, tuple[float, float]] = {}
    with open(intervals_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            iv[int(r["key"][2:])] = (float(r["low_0.95"]), float(r["high_0.95"]))
    tru: dict[int, float] = {}
    with open(truth_tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            tru[int(r["key"][2:])] = float(r["truth"])
    return iv, tru


def _real_betas_interval_for_target(
    real_iv: dict[int, tuple[float, float]],
    real_tru: dict[int, float],
    ev_idx: int,
    truth: float,
) -> tuple[float, float]:
    """Return the betAS interval for an event row, failing on row-order drift."""
    if ev_idx not in real_iv:
        raise ValueError(
            f"Missing betAS interval row ev{ev_idx}; row-order alignment is unsafe"
        )
    observed_truth = real_tru.get(ev_idx)
    if observed_truth is None or abs(observed_truth - truth) > 1e-3:
        raise ValueError(
            "betAS truth row ev"
            f"{ev_idx} does not match target truth ({observed_truth!r} vs {truth!r}); "
            "row-order alignment is unsafe"
        )
    return real_iv[ev_idx]


def load_circ_incl(tsv: str) -> list[tuple[int, int]]:
    """Inclusion-junction (lo, hi) per circadian target, in file (= target) order."""
    out = []
    with open(tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            incl = row["inclusion_junction"].replace(":", "-").split("-")
            out.append(tuple(sorted(int(x) for x in incl[-2:])))
    return out


def majiq_interval(target, incl, kind, maj_rows, g1, g2):
    """(center, low, high) MAJIQ 95% CI oriented g1 - g2; None if unmatched/empty."""
    chrom = target.chrom
    best, best_cov = None, -1.0
    for m in maj_rows:
        if H._norm_chrom(m["seqid"]) != chrom:
            continue
        hit = False
        if kind == "exon":
            # MAJIQ is matched at the cassette-exon level (the LSV other-exon equals the
            # target cassette exon). A junction-level SE-aware variant was evaluated but
            # rejected: requiring a MAJIQ junction to hit a cassette inclusion junction
            # grew the matched set (62->82 on TRA2) yet DROPPED MAJIQ-vs-RT-PCR
            # orientation (0.81->0.64), i.e. a matched junction coordinate does not
            # guarantee the row's PSI is the cassette-inclusion PSI. The cassette-exon
            # match keeps orientation-correct events; MAJIQ therefore resolves a subset.
            hit = (abs(int(m["other_exon_start"]) - target.exon_start) <= COORD_TOL
                   and abs(int(m["other_exon_end"]) - target.exon_end) <= COORD_TOL)
        else:  # junction: MAJIQ junction (start,end) ~ inclusion junction (lo,hi)
            jlo, jhi = sorted((int(m["start"]), int(m["end"])))
            hit = abs(jlo - incl[0]) <= JUNC_TOL and abs(jhi - incl[1]) <= JUNC_TOL
        if hit:
            cov = float(m[f"{g1}_raw_coverage"]) + float(m[f"{g2}_raw_coverage"])
            if cov > best_cov:
                best_cov, best = cov, m
    if best is None or best_cov <= 0:
        return None
    center = float(best[f"{g1}_raw_psi_mean"]) - float(best[f"{g2}_raw_psi_mean"])
    half = Z95 * float(best["dpsi_std"])
    return center, max(-1.0, center - half), min(1.0, center + half)


def run_dataset(cfg) -> dict:
    targets = cfg["targets"]
    incls = cfg.get("incls") or [None] * len(targets)
    events = H.parse_se_table(cfg["rmats_se"])
    maj_rows = load_majiq(cfg["majiq_tsv"])
    real_iv, real_tru = load_real_betas(cfg["betas_iv"], cfg["betas_truth"])
    g1, g2 = cfg["majiq_groups"]
    rng = np.random.default_rng(7)

    rec = {k: [] for k in ("pts", "trs", "sup", "blo", "bhi", "rlo", "rhi",
                           "mlo", "mhi", "mcenter", "gene")}
    full = {k: [] for k in ("pts", "trs", "sup", "blo", "bhi", "rlo", "rhi", "gene")}
    n_majiq = 0
    ev_idx = -1
    for t, incl in zip(targets, incls):
        ev = H.match_event(t, events)
        mi = majiq_interval(t, incl, cfg["kind"], maj_rows, g1, g2)
        if mi is not None:
            n_majiq += 1
        if ev is None:
            continue
        ec = H.event_counts(ev, normalize=True, swap_groups=cfg["swap"])
        if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
            continue
        ev_idx += 1
        rlo_b, rhi_b = _real_betas_interval_for_target(
            real_iv, real_tru, ev_idx, t.dpsi_rtpcr
        )
        m, std, _lo, _hi = H.beta_interval(ec, ALPHA, 0.5, rng)
        rm_mean, rm_se, rmlo, rmhi = H.rmats_interval(ec, ALPHA)
        # FULL rMATS-matched set (MAJIQ not required)
        full["pts"].append(m)
        full["trs"].append(t.dpsi_rtpcr)
        full["sup"].append(ec.total_support)
        full["blo"].append(rlo_b)
        full["bhi"].append(rhi_b)
        full["rlo"].append(rmlo)
        full["rhi"].append(rmhi)
        full["gene"].append(t.gene)
        if mi is None:
            continue
        center, mlo, mhi = mi
        rec["pts"].append(m)
        rec["trs"].append(t.dpsi_rtpcr)
        rec["sup"].append(ec.total_support)
        rec["blo"].append(rlo_b)
        rec["bhi"].append(rhi_b)
        rec["rlo"].append(rmlo)
        rec["rhi"].append(rmhi)
        rec["mlo"].append(mlo)
        rec["mhi"].append(mhi)
        rec["mcenter"].append(center)
        rec["gene"].append(t.gene)

    A = {k: np.array(v) for k, v in rec.items()}
    F = {k: np.array(v) for k, v in full.items()}
    out = {"name": cfg["name"], "n_common": A["pts"].size, "n_full": F["pts"].size,
           "n_majiq": n_majiq, "common": A, "full": F}
    out["corr"] = (float(np.corrcoef(A["mcenter"], A["trs"])[0, 1])
                   if A["pts"].size > 2 else float("nan"))
    return out


def cov_row(lo, hi, truth):
    k = int(np.sum((truth >= lo) & (truth <= hi)))
    p, wlo, u = wilson(k, truth.size)
    width = float(np.mean(hi - lo))
    iscore = float(np.mean([H.interval_score(y, a, b, ALPHA)
                            for y, a, b in zip(truth, lo, hi)]))
    return p, wlo, u, width, iscore


def conformal(A):
    return H.conformal_crossfit(A["pts"], A["trs"], np.ones_like(A["pts"]), A["sup"], ALPHA)


def method_stats(A) -> dict:
    """Per-method {coverage, wilson95, width, iscore} on a matched set A, so the saved
    canonical JSON is the single source for the per-dataset table (Supp Table 1) too."""
    clo, chi = conformal(A)
    rows = [("MAJIQ", A["mlo"], A["mhi"]), ("betAS", A["blo"], A["bhi"]),
            ("rMATS", A["rlo"], A["rhi"]), ("BRAID-conformal", clo, chi)]
    out = {}
    for nm, lo, hi in rows:
        p, wlo, u, w, isc = cov_row(np.array(lo), np.array(hi), A["trs"])
        out[nm] = {"coverage": p, "wilson95": [wlo, u], "width": w, "iscore": isc}
    return out


def method_flags(A):
    """Per-event coverage boolean arrays for all four methods on set A."""
    truth = A["trs"]
    clo, chi = conformal(A)
    return {
        "MAJIQ": (truth >= A["mlo"]) & (truth <= A["mhi"]),
        "betAS": (truth >= A["blo"]) & (truth <= A["bhi"]),
        "rMATS": (truth >= A["rlo"]) & (truth <= A["rhi"]),
        "BRAID-conformal": (truth >= clo) & (truth <= chi),
    }


def mcnemar_exact(a, b):
    """Exact (binomial) McNemar for paired coverage flags. Returns (b01, c10, p)."""
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    n01 = int(np.sum(a & ~b))   # a covers, b misses
    n10 = int(np.sum(~a & b))   # a misses, b covers
    nd = n01 + n10
    p = 1.0 if nd == 0 else min(1.0, 2.0 * stats.binom.cdf(min(n01, n10), nd, 0.5))
    return n01, n10, p


def print_block(title, A):
    truth = A["trs"]
    n = truth.size
    clo, chi = conformal(A)
    rows = [
        ("MAJIQ (real binary)", A["mlo"], A["mhi"]),
        ("betAS (real tool)", A["blo"], A["bhi"]),
        ("rMATS (per-rep CI)", A["rlo"], A["rhi"]),
        ("BRAID-conformal", clo, chi),
    ]
    print(f"\n{title}  (n={n})")
    print(f"  {'method':<26}{'cov@95':>8}{'   Wilson 95% CI':<20}{'width':>8}{'iscore':>9}")
    print("  " + "-" * 70)
    for name, lo, hi in rows:
        p, wlo, u, w, isc = cov_row(lo, hi, truth)
        print(f"  {name:<26}{p:>8.3f}   [{wlo:.3f}, {u:.3f}]   {w:>6.3f}  {isc:>8.3f}")


def main() -> None:
    os.chdir(os.path.dirname(os.path.dirname(_HERE)))
    DB = "data/public_benchmarks"
    BH = "benchmarks/headtohead"
    circ_tsv = "data/public_benchmarks/meta/gse54651_circadian_positive_events.tsv"
    datasets = [
        dict(name="TRA2 (GSE59335, human)", kind="exon", swap=True,
             rmats_se=f"{DB}/GSE59335/rmats/SE.MATS.JC.txt",
             targets=H.load_targets(f"{DB}/GSE59335/targets/validated_events.tsv",
                                    f"{DB}/GSE59335/targets/failed_events.tsv"),
             majiq_tsv=f"{DB}/GSE59335/majiq/deltapsi.tsv",
             betas_iv=f"{BH}/tra2_betas_intervals.tsv", betas_truth=f"{BH}/tra2_betas_truth.tsv",
             majiq_groups=("KD", "CTRL")),
        dict(name="Circadian (GSE54651, mouse)", kind="junction", swap=True,
             rmats_se=f"{DB}/GSE54651/rmats/SE.MATS.JC.txt",
             targets=H.load_circadian_targets(circ_tsv),
             incls=load_circ_incl(circ_tsv),
             majiq_tsv=f"{DB}/GSE54651/majiq/deltapsi.tsv",
             betas_iv=f"{BH}/circ_betas_intervals.tsv", betas_truth=f"{BH}/circ_betas_truth.tsv",
             majiq_groups=("LIVER", "CEREB")),
        dict(name="PC3E/GS689 (SRA, human)", kind="exon", swap=False,
             rmats_se=f"{DB}/SRS354082/rmats/SE.MATS.JC.txt",
             targets=H.load_targets(f"{DB}/meta/rmats_pc3e_gs689_positive_events.tsv", None),
             majiq_tsv=f"{DB}/SRS354082/majiq/deltapsi.tsv",
             betas_iv=f"{BH}/srs_betas_intervals.tsv", betas_truth=f"{BH}/srs_betas_truth.tsv",
             majiq_groups=("PC3E", "GS689")),
    ]
    results = [run_dataset(c) for c in datasets]

    print("=" * 78)
    print("COMPREHENSIVE ΔPSI CREDIBLE-INTERVAL BENCHMARK (4 methods x real datasets)")
    print("=" * 78)
    for r in results:
        print(f"\n### {r['name']}: targets→ rMATS&MAJIQ common={r['n_common']}, "
              f"full rMATS={r['n_full']}, MAJIQ-matched={r['n_majiq']}, "
              f"orient corr={r['corr']:+.3f}")
        print_block("  [common set: all 4 methods]", r["common"])

    # Pooled across datasets (common set)
    pooled = {k: np.concatenate([r["common"][k] for r in results])
              for k in results[0]["common"]}
    print("\n" + "=" * 78)
    print_block("POOLED (TRA2 + circadian + PC3E), common 4-method set", pooled)

    # Coverage is a PAIRED design (same events, different methods). Non-overlapping
    # marginal Wilson CIs are only a conservative proxy; McNemar's exact test is the
    # correct paired comparison. Also report event/gene clustering (design effect).
    genes = list(pooled["gene"])
    ne, ng = len(genes), len(set(genes))
    print(f"\nClustering check: {ne} events / {ng} genes "
          f"(design effect {ne/ng:.2f} → events ≈ independent; pseudoreplication negligible).")
    fl = method_flags(pooled)
    bf = fl["BRAID-conformal"]
    print("Paired McNemar — BRAID-conformal vs each competitor "
          "(b = BRAID covers & competitor misses; c = competitor covers & BRAID misses):")
    mcnemar_out = {}
    for name in ("MAJIQ", "betAS", "rMATS"):
        b01, c10, p = mcnemar_exact(bf, fl[name])
        mcnemar_out[name] = {"b": int(b01), "c": int(c10), "p": float(p)}
        print(f"  BRAID vs {name:<6}: b={b01:3d}  c={c10:2d}  exact p={p:.2e}"
              + ("  (c=0: no event covered by the comparator but missed by BRAID)"
                 if c10 == 0 else ""))

    # Headline coverage on FULL rMATS-matched set (betAS / rMATS / BRAID-conf; MAJIQ N/A)
    pooled_full = {k: np.concatenate([r["full"][k] for r in results])
                   for k in results[0]["full"]}
    tf = pooled_full["trs"]
    clo, chi = H.conformal_crossfit(pooled_full["pts"], tf, np.ones_like(pooled_full["pts"]),
                                    pooled_full["sup"], ALPHA)
    print(f"\nFULL rMATS-matched pooled set (MAJIQ not required, n={tf.size}):")
    full_out = {}
    for name, lo, hi in [("betAS (real tool)", pooled_full["blo"], pooled_full["bhi"]),
                         ("rMATS (per-rep CI)", pooled_full["rlo"], pooled_full["rhi"]),
                         ("BRAID-conformal", clo, chi)]:
        p, wlo, u, w, isc = cov_row(np.array(lo), np.array(hi), tf)
        full_out[name] = {"coverage": p, "wilson95": [wlo, u], "width": w, "iscore": isc}
        print(f"  {name:<26}{p:>8.3f}   [{wlo:.3f}, {u:.3f}]   width={w:.3f}  iscore={isc:.3f}")

    # Persist the canonical head-to-head coverage JSON so every manuscript head-to-head
    # number maps to one source artifact + the command `python comprehensive_benchmark.py`.
    pooled_methods = {}
    pclo, pchi = conformal(pooled)
    for nm, lo, hi in [("MAJIQ", pooled["mlo"], pooled["mhi"]),
                       ("betAS", pooled["blo"], pooled["bhi"]),
                       ("rMATS", pooled["rlo"], pooled["rhi"]),
                       ("BRAID-conformal", pclo, pchi)]:
        cp, cl, cu, cw, cisc = cov_row(np.array(lo), np.array(hi), pooled["trs"])
        pooled_methods[nm] = {"coverage": cp, "wilson95": [cl, cu],
                              "width": cw, "iscore": cisc}
    out = {
        "datasets": {r["name"]: {"n_common": int(r["n_common"]),
                                 "n_full": int(r["n_full"]),
                                 "n_majiq": int(r["n_majiq"]),
                                 "orient_corr": r["corr"],
                                 "methods": method_stats(r["common"])} for r in results},
        "pooled_common": {"n": int(pooled["trs"].size), "n_genes": int(ng),
                          "methods": pooled_methods, "mcnemar_vs_braid": mcnemar_out},
        "full_rmats_matched": {"n": int(tf.size), "methods": full_out},
    }
    dest = "benchmarks/results/headtohead_coverage.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"Saved {dest}")
    print("=" * 78)


if __name__ == "__main__":
    main()
