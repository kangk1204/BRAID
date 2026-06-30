#!/usr/bin/env python3
# ruff: noqa: I001, E402
"""Why BRAID over aggressively filtering an existing caller? (TRA2 RT-PCR).

A reviewer's natural objection: "Just take each tool's calls at padj < 0.05 AND
|delta PSI| >= cutoff and you have high-confidence events -- what does BRAID's
calibrated interval add?"  This script answers it on the one benchmark with both
RT-PCR-positive (76) and RT-PCR-negative (36) cassette-exon events, GSE59335 TRA2.

Two orthogonal axes are measured, both on real committed per-event data
(no fabricated numbers):

  A. CALIBRATION is not a filtering knob.  For each caller we take exactly the
     events that pass that caller's own padj<0.05 AND |dPSI|>=0.1 filter and
     measure the coverage of the *native* interval on those confident calls.
     Filtering selects events; it does not widen the interval to include the
     RNA-seq->RT-PCR residual, so the native interval still under-covers even on
     the most confident subset, whereas BRAID's conformal interval on the SAME
     events is near-nominal.

  B. DETECTION parity.  We sweep each caller's full (significance x |dPSI|) grid
     and record the BEST achievable MCC / the precision-recall frontier on the
     76+36 panel.  BRAID's operating points are overlaid.  The point is that no
     filter strength buys detection performance beyond BRAID's tier -- so one
     does not give up detection by adopting the calibrated layer.

Everything reuses head_to_head_coverage (H) so orientation, matching, length
normalization, and the interval definitions are byte-for-byte those behind
Table 1.  TRA2 design is b1=control, b2=treatment -> swap_groups=True.

Outputs: outputs/headtohead/detection_filter_sweep.json  (+ stdout summary).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import head_to_head_coverage as H
import majiq_coverage as MJ
from cross_dataset_transfer import apply_abs_conformal, build_arrays, fit_abs_conformal

from braid.target.conformal import load_differential_conformal_calibrator

ROOT = _HERE.parents[1]
Z95 = 1.959963984540054
NEG_SUPPORT_FLOOR = 1  # keep H's "usable support in either group" gate

RMATS_SE = str(ROOT / "data/public_benchmarks/GSE59335/rmats/SE.MATS.JC.txt")
VALIDATED = str(ROOT / "data/public_benchmarks/GSE59335/targets/validated_events.tsv")
FAILED = str(ROOT / "data/public_benchmarks/GSE59335/targets/failed_events.tsv")
MAJIQ_TSV = str(ROOT / "data/public_benchmarks/GSE59335/majiq/deltapsi.tsv")
BETAS_IV = str(_HERE / "tra2_betas_intervals.tsv")
BETAS_TRUTH = str(_HERE / "tra2_betas_truth.tsv")
# Circadian (GSE54651) -- used ONLY to fit a transfer calibrator that never sees
# any TRA2 RT-PCR, emulating "deploy BRAID on a new experiment with no RT-PCR".
CIRC_SE = str(ROOT / "data/public_benchmarks/GSE54651/rmats/SE.MATS.JC.txt")
CIRC_TSV = str(ROOT / "data/public_benchmarks/meta/gse54651_circadian_positive_events.tsv")
OUT_JSON = _HERE / "detection_filter_sweep.json"


# --------------------------------------------------------------------------
# Per-event table, reusing H's exact matched-row construction (swap_groups=True)
# --------------------------------------------------------------------------
def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _two_sided_p(mean: float, std: float) -> float:
    """Bayesian two-sided tail mass at 0 for a Gaussian posterior (betAS analog of padj)."""
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        return float("nan")
    z = abs(mean) / std
    return 2.0 * (1.0 - _normal_cdf(z))


def _prob_abs_ge(mean: float, std: float, cut: float) -> float:
    """P(|X| >= cut) for X ~ N(mean, std) -- posterior effect probability."""
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        return float("nan")
    hi = (cut - mean) / std
    lo = (-cut - mean) / std
    return (1.0 - _normal_cdf(hi)) + _normal_cdf(lo)


def _load_betas_meanstd(path: str) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[int(r["key"][2:])] = (float(r["dpsi_mean"]), float(r["dpsi_std"]))
    return out


def build() -> dict:
    """Reconstruct the canonical TRA2 matched rows + attach every caller's
    per-event (significance, dPSI, native 95% interval)."""
    events = H.parse_se_table(RMATS_SE)
    targets = H.load_targets(VALIDATED, FAILED)
    maj_rows = MJ.load_majiq(MAJIQ_TSV)
    betas_meanstd = _load_betas_meanstd(BETAS_IV)
    rng = np.random.default_rng(7)

    rows: list[dict] = []
    for t in targets:
        ev = H.match_event(t, events)
        if ev is None:
            continue
        ec = H.event_counts(ev, normalize=True, swap_groups=True)
        if (ec.a_c + ec.b_c) < NEG_SUPPORT_FLOOR or (ec.a_t + ec.b_t) < NEG_SUPPORT_FLOOR:
            continue
        bj_mean, bj_std, _, _ = H.beta_interval(ec, 0.05, 0.5, rng)
        # native rMATS per-replicate Student-t 95% interval (Table-1 definition)
        r_m, _r_s, r_lo, r_hi = H.rmats_interval(ec, 0.05)
        rows.append({
            "gene": t.gene, "chrom": t.chrom,
            "truth": float(t.dpsi_rtpcr), "is_positive": bool(t.is_positive),
            "support": float(ec.total_support),
            "bj_mean": float(bj_mean), "bj_std": float(max(bj_std, 1e-6)),
            "rmats_fdr": float(ec.rmats_fdr), "rmats_dpsi": float(ec.rmats_dpsi),
            "rmats_lo": float(r_lo), "rmats_hi": float(r_hi),
            "_target": t,
        })

    n = len(rows)
    # betAS native 95% interval (real betAS output), keyed by ev row order
    betas_iv = H._load_betas_intervals(BETAS_IV, n)  # per-level lows/highs/means
    blo, bhi, bmean = betas_iv["0.95"]
    for i, r in enumerate(rows):
        bm, bs = betas_meanstd[i]
        r["betas_dpsi"] = float(bm)
        r["betas_std"] = float(bs)
        r["betas_p"] = _two_sided_p(bm, bs)
        r["betas_lo"] = float(blo[i])
        r["betas_hi"] = float(bhi[i])

    # MAJIQ: match each row's cassette exon to MAJIQ inclusion junctions.
    for r in rows:
        t = r["_target"]
        es, ee, chrom = t.exon_start, t.exon_end, t.chrom
        best, best_cov = None, -1.0
        for m in maj_rows:
            if H._norm_chrom(m["seqid"]) != chrom:
                continue
            if (abs(int(m["other_exon_start"]) - es) <= MJ.COORD_TOL
                    and abs(int(m["other_exon_end"]) - ee) <= MJ.COORD_TOL):
                cov = float(m["KD_raw_coverage"]) + float(m["CTRL_raw_coverage"])
                if cov > best_cov:
                    best_cov, best = cov, m
        if best is None or best_cov <= 0:
            r["majiq_dpsi"] = float("nan")
            r["majiq_prob"] = float("nan")
            r["majiq_lo"] = float("nan")
            r["majiq_hi"] = float("nan")
            continue
        center = float(best["KD_raw_psi_mean"]) - float(best["CTRL_raw_psi_mean"])
        half = Z95 * float(best["dpsi_std"])
        r["majiq_dpsi"] = center
        r["majiq_prob"] = float(best["probability_changing"])
        r["majiq_lo"] = max(-1.0, center - half)
        r["majiq_hi"] = min(1.0, center + half)

    # BRAID conformal-abs 95% interval (k-fold cross-fit, honest held-out).
    # NOTE: this variant USES TRA2 RT-PCR truth to calibrate; it is the
    # RT-PCR-calibrated upper reference, NOT what a user gets without RT-PCR.
    truths = np.array([r["truth"] for r in rows])
    supports = np.array([r["support"] for r in rows])
    points = np.array([r["bj_mean"] for r in rows])
    blo_c, bhi_c = H.conformal_crossfit(
        points, truths, np.ones(n), supports, 0.05, seed=7)

    # --- No-TRA2-RT-PCR variants (the realistic "no RT-PCR for this experiment") ---
    # (1) Transfer: fit the absolute-residual conformal quantile on circadian
    #     (GSE54651) RT-PCR ONLY, then deploy unchanged on TRA2. Never sees TRA2 truth.
    circ_t = H.load_circadian_targets(CIRC_TSV)
    cp, ct, cs = build_arrays(CIRC_SE, circ_t, swap_groups=True)
    qg_t, qb_t = fit_abs_conformal(cp, ct, cs, 0.05)
    tlo, thi = apply_abs_conformal(points, supports, qg_t, qb_t)

    # (2) Shipped default: the packaged `braid differential` calibrator (absolute
    #     dPSI half-width). Provenance: real_rtpcr_tra2_circadian_n162 -- it DID see
    #     TRA2, so for TRA2 it is mildly in-sample; reported only as the default the
    #     user actually gets out of the box. The clean no-RT-PCR number is transfer (1).
    cal = load_differential_conformal_calibrator()
    for i, r in enumerate(rows):
        r["braid_lo"] = float(blo_c[i])
        r["braid_hi"] = float(bhi_c[i])
        r["braid_prob_large"] = _prob_abs_ge(r["bj_mean"], r["bj_std"], 0.1)
        r["braid_transfer_lo"] = float(tlo[i])
        r["braid_transfer_hi"] = float(thi[i])
        dlo, dhi = cal.interval(r["bj_mean"], 1.0, r["support"], clip=(-1.0, 1.0))
        r["braid_default_lo"] = float(dlo)
        r["braid_default_hi"] = float(dhi)
    for r in rows:
        r.pop("_target", None)
    return {"rows": rows, "n_pos": int(truths.size and sum(x["is_positive"] for x in rows)),
            "n_neg": int(sum(not x["is_positive"] for x in rows)), "n": n}


# --------------------------------------------------------------------------
# Analysis A -- filtering does not fix interval coverage
# --------------------------------------------------------------------------
def _cov(rows: list[dict], lo_key: str, hi_key: str) -> float:
    ok = [(r["truth"] >= r[lo_key]) and (r["truth"] <= r[hi_key])
          for r in rows if math.isfinite(r[lo_key]) and math.isfinite(r[hi_key])]
    return float(np.mean(ok)) if ok else float("nan")


def analysis_a(rows: list[dict]) -> dict:
    """For each caller's own padj<0.05 AND |dPSI|>=0.1 confident calls, native
    interval coverage vs BRAID conformal coverage on the SAME events."""
    callers = {
        "rMATS": dict(sig=lambda r: r["rmats_fdr"] < 0.05, dpsi="rmats_dpsi",
                      lo="rmats_lo", hi="rmats_hi"),
        "betAS": dict(sig=lambda r: r["betas_p"] < 0.05, dpsi="betas_dpsi",
                      lo="betas_lo", hi="betas_hi"),
        "MAJIQ": dict(sig=lambda r: r["majiq_prob"] >= 0.90, dpsi="majiq_dpsi",
                      lo="majiq_lo", hi="majiq_hi"),
    }
    out: dict[str, dict] = {}
    for name, c in callers.items():
        conf = [r for r in rows
                if math.isfinite(r[c["dpsi"]]) and c["sig"](r)
                and abs(r[c["dpsi"]]) >= 0.10]
        out[name] = {
            "n_confident": len(conf),
            "native_coverage": _cov(conf, c["lo"], c["hi"]),
            # RT-PCR-calibrated upper reference (uses TRA2 truth):
            "braid_crossfit_coverage_same_events": _cov(conf, "braid_lo", "braid_hi"),
            # No-TRA2-RT-PCR variants (what you get without RT-PCR on this experiment):
            "braid_transfer_coverage_same_events": _cov(
                conf, "braid_transfer_lo", "braid_transfer_hi"),
            "braid_default_coverage_same_events": _cov(
                conf, "braid_default_lo", "braid_default_hi"),
            "native_mean_width": (
                float(np.mean([r[c["hi"]] - r[c["lo"]] for r in conf]))
                if conf else float("nan")),
            "braid_transfer_mean_width_same_events": (
                float(np.mean([r["braid_transfer_hi"] - r["braid_transfer_lo"]
                               for r in conf])) if conf else float("nan")),
        }
    return out


def full_panel_coverage(rows: list[dict]) -> dict:
    """Coverage + mean width of every interval on the full matched panel (n = 112)."""
    variants = {
        "rMATS_native": ("rmats_lo", "rmats_hi"),
        "betAS_native": ("betas_lo", "betas_hi"),
        "MAJIQ_native": ("majiq_lo", "majiq_hi"),
        "BRAID_crossfit_rtpcr": ("braid_lo", "braid_hi"),
        "BRAID_transfer_no_tra2_rtpcr": ("braid_transfer_lo", "braid_transfer_hi"),
        "BRAID_default_shipped": ("braid_default_lo", "braid_default_hi"),
    }
    out: dict[str, dict] = {}
    for name, (lo, hi) in variants.items():
        usable = [r for r in rows if math.isfinite(r[lo]) and math.isfinite(r[hi])]
        out[name] = {
            "n": len(usable),
            "coverage": _cov(usable, lo, hi),
            "mean_width": (float(np.mean([r[hi] - r[lo] for r in usable]))
                           if usable else float("nan")),
        }
    return out


# --------------------------------------------------------------------------
# Analysis B -- detection frontier: best (sig x |dPSI|) per caller vs BRAID
# --------------------------------------------------------------------------
def _mcc(tp: int, fp: int, fn: int, tn: int) -> float:
    num = tp * tn - fp * fn
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return num / den if den > 0 else 0.0


def _confusion(rows: list[dict], pred) -> tuple[int, int, int, int]:
    tp = fp = fn = tn = 0
    for r in rows:
        p = pred(r)
        y = r["is_positive"]
        if p and y:
            tp += 1
        elif p and not y:
            fp += 1
        elif (not p) and y:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _metrics(rows: list[dict], pred) -> dict:
    tp, fp, fn, tn = _confusion(rows, pred)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec,
            "recall": rec, "fpr": fpr, "mcc": _mcc(tp, fp, fn, tn)}


def analysis_b(rows: list[dict]) -> dict:
    dpsi_cuts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    grids = {
        "rMATS": {"dpsi": "rmats_dpsi",
                  "sigs": [("FDR", t, (lambda r, t=t: r["rmats_fdr"] < t))
                           for t in (1.0, 0.25, 0.10, 0.05, 0.01)]},
        "betAS": {"dpsi": "betas_dpsi",
                  "sigs": [("p", t, (lambda r, t=t: r["betas_p"] < t))
                           for t in (1.0, 0.25, 0.10, 0.05, 0.01)]},
        "MAJIQ": {"dpsi": "majiq_dpsi",
                  "sigs": [("prob", t, (lambda r, t=t: r["majiq_prob"] >= t))
                           for t in (0.0, 0.50, 0.70, 0.90, 0.95)]},
    }
    out: dict[str, dict] = {}
    for name, g in grids.items():
        matched = [r for r in rows if math.isfinite(r[g["dpsi"]])]
        npos = sum(r["is_positive"] for r in matched)
        nneg = sum(not r["is_positive"] for r in matched)
        cells = []
        best = None
        for slabel, sval, sfn in g["sigs"]:
            for cut in dpsi_cuts:
                def pred(r, sfn=sfn, cut=cut, dk=g["dpsi"]):
                    return sfn(r) and abs(r[dk]) >= cut
                m = _metrics(matched, pred)
                m.update({"sig_field": slabel, "sig_thr": sval, "dpsi_cut": cut})
                cells.append(m)
                if best is None or m["mcc"] > best["mcc"]:
                    best = m
        out[name] = {"n_matched": len(matched), "n_pos": int(npos),
                     "n_neg": int(nneg), "best_by_mcc": best, "grid": cells}
    return out


def braid_operating_points(rows: list[dict], detection_subsets: dict) -> dict:
    """BRAID's operating points, scored on the full panel AND on each caller's
    matched subset for an apples-to-apples overlay."""
    def supported(r):
        return (math.isfinite(r["rmats_fdr"]) and r["rmats_fdr"] < 0.05
                and math.isfinite(r["braid_prob_large"]) and r["braid_prob_large"] >= 0.5)

    def _excl0(lo_key, hi_key):
        def fn(r):
            return (math.isfinite(r[lo_key]) and math.isfinite(r[hi_key])
                    and (r[lo_key] > 0 or r[hi_key] < 0))
        return fn

    pts = {
        "BRAID_supported": supported,
        "BRAID_crossfit_excl0": _excl0("braid_lo", "braid_hi"),
        "BRAID_transfer_excl0": _excl0("braid_transfer_lo", "braid_transfer_hi"),
        "BRAID_default_excl0": _excl0("braid_default_lo", "braid_default_hi"),
    }
    out: dict[str, dict] = {"full_panel": {}, "on_caller_subset": {}}
    for label, fn in pts.items():
        out["full_panel"][label] = _metrics(rows, fn)
    for caller, info in detection_subsets.items():
        dk = {"rMATS": "rmats_dpsi", "betAS": "betas_dpsi", "MAJIQ": "majiq_dpsi"}[caller]
        sub = [r for r in rows if math.isfinite(r[dk])]
        out["on_caller_subset"][caller] = {
            label: _metrics(sub, fn) for label, fn in pts.items()}
    return out


def main() -> None:
    data = build()
    rows = data["rows"]
    a = analysis_a(rows)
    fp = full_panel_coverage(rows)
    b = analysis_b(rows)
    bop = braid_operating_points(rows, b)

    result = {
        "dataset": "GSE59335 TRA2 (RT-PCR positives + negatives)",
        "n_events": data["n"], "n_positive": data["n_pos"], "n_negative": data["n_neg"],
        "full_panel_coverage": fp,
        "analysis_a_filtering_does_not_fix_coverage": a,
        "analysis_b_detection_frontier": {
            k: {kk: vv for kk, vv in v.items() if kk != "grid"} for k, v in b.items()},
        "analysis_b_full_grid": {k: v["grid"] for k, v in b.items()},
        "braid_operating_points": bop,
        "notes": {
            "orientation": "swap_groups=True (TRA2 b1=control, b2=treatment); dPSI=treat-control",
            "betas_significance": "Bayesian two-sided Gaussian tail at 0 from betAS (mean,std)",
            "majiq_significance": "MAJIQ probability_changing (P(|dPSI|>=0.2))",
            "suppa2": "excluded from detection frontier: committed S5 (n=66) is gene-matched "
                      "positives-only with no RT-PCR negatives, so FPR/MCC are not defined",
            "interval_defs": "native intervals = Table-1 definitions reused from "
                             "head_to_head_coverage",
            "braid_variants": (
                "BRAID_crossfit = TRA2-RT-PCR cross-fit (calibrated upper reference, "
                "USES TRA2 truth); BRAID_transfer = absolute-residual conformal fit on "
                "circadian RT-PCR only, deployed on TRA2 with NO TRA2 RT-PCR (the "
                "realistic no-RT-PCR-for-this-experiment case); BRAID_default = packaged "
                "differential calibrator (real_rtpcr_tra2_circadian_n162, mildly in-sample "
                "for TRA2, shown as the literal out-of-box default)"),
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # Sanitize non-finite metrics (e.g. precision when a grid cell predicts no
    # positives) to null so the artifact is strict JSON on any input, matching the
    # head_to_head_coverage writer contract.
    OUT_JSON.write_text(json.dumps(H._json_safe(result), indent=2, allow_nan=False))

    # ---- stdout summary ----
    print(f"TRA2 panel: n={data['n']}  positives={data['n_pos']}  negatives={data['n_neg']}")
    print("\nFull-panel coverage of RT-PCR truth (n=112; nominal 0.95):")
    print(f"  {'interval':32} {'n':>4} {'cov':>6} {'width':>6}")
    for name, v in fp.items():
        print(f"  {name:32} {v['n']:>4d} {v['coverage']:>6.3f} {v['mean_width']:>6.3f}")
    print("\nA. Caller's own padj<0.05 AND |dPSI|>=0.1 confident calls -- native vs BRAID")
    print("   (cross-fit USES TRA2 truth; transfer/default do NOT):")
    print(f"  {'caller':7} {'n_conf':>6} {'native':>7} {'BRAID_xfit':>10} "
          f"{'BRAID_transfer':>14} {'BRAID_default':>13}")
    for name, v in a.items():
        print(f"  {name:7} {v['n_confident']:>6d} {v['native_coverage']:>7.3f} "
              f"{v['braid_crossfit_coverage_same_events']:>10.3f} "
              f"{v['braid_transfer_coverage_same_events']:>14.3f} "
              f"{v['braid_default_coverage_same_events']:>13.3f}")
    print("\nB. Best achievable detection (max MCC over the full sig x |dPSI| grid):")
    print(f"  {'caller':7} {'n(pos/neg)':>11} {'bestMCC':>8}  best filter")
    for name, v in b.items():
        bm = v["best_by_mcc"]
        print(f"  {name:7} {v['n_pos']:>4d}/{v['n_neg']:<5d} {bm['mcc']:>8.3f}  "
              f"{bm['sig_field']}<{bm['sig_thr']} & |dPSI|>={bm['dpsi_cut']} "
              f"(P={bm['precision']:.2f} R={bm['recall']:.2f} FPR={bm['fpr']:.2f})")
    print("\n  BRAID operating points on the full panel:")
    for label, m in bop["full_panel"].items():
        print(f"    {label:22} MCC={m['mcc']:.3f}  P={m['precision']:.2f} "
              f"R={m['recall']:.2f} FPR={m['fpr']:.2f}")
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
