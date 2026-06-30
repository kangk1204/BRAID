#!/usr/bin/env python3
"""Sum (pooled group counts) vs Rep (replicate-aware bootstrap) on the RT-PCR head-to-head set.

Question: on the real RT-PCR-truth benchmark, does the differential ΔPSI *model*
(``--differential-model sum`` vs ``rep``) change coverage, sharpness, or point
accuracy? Both posteriors are fed through the SAME leakage-free cross-fit conformal
procedure (``head_to_head_coverage.conformal_crossfit``) and scored against the same
RT-PCR ΔPSI truth, so the only thing that varies is how the per-event point estimate
and posterior scale are derived:

  * sum : pooled Jeffreys-Beta on summed, length-normalized group counts
          (Beta(sum_inc+0.5, sum_exc*ratio+0.5)) -- the paper's BRAID-Jeffreys point,
          and `braid differential --differential-model sum`.
  * rep : resample the rMATS per-replicate count vectors with replacement, draw a
          per-replicate Beta PSI, average within group -- mirrors the production
          `_draw_replicate_mean_psi` used by `--differential-model rep/auto`.

Datasets, orientation, event matching, and length normalization are reused verbatim
from head_to_head_coverage, so this is the full rMATS-matched surface (n=196; the
139-event four-method common set reported in the manuscript abstract is the
MAJIQ-matched subset of it).
Output: benchmarks/results/sum_vs_rep_eval.json + a printed table. Read-only on all
committed artifacts.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy import stats

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import head_to_head_coverage as H  # noqa: E402

_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
DB = os.path.join(_REPO, "data", "public_benchmarks")
RESULTS = os.path.join(_REPO, "benchmarks", "results", "sum_vs_rep_eval.json")

DRAWS = 4000
ALPHA = 0.05
SEED = 7


def _datasets() -> list[dict]:
    circ_tsv = os.path.join(DB, "meta", "gse54651_circadian_positive_events.tsv")
    return [
        dict(name="TRA2 (GSE59335)", swap=True,
             rmats_se=os.path.join(DB, "GSE59335", "rmats", "SE.MATS.JC.txt"),
             targets=H.load_targets(
                 os.path.join(DB, "GSE59335", "targets", "validated_events.tsv"),
                 os.path.join(DB, "GSE59335", "targets", "failed_events.tsv"))),
        dict(name="Circadian (GSE54651)", swap=True,
             rmats_se=os.path.join(DB, "GSE54651", "rmats", "SE.MATS.JC.txt"),
             targets=H.load_circadian_targets(circ_tsv)),
        dict(name="SRS354082 (PC3E/GS689)", swap=False,
             rmats_se=os.path.join(DB, "SRS354082", "rmats", "SE.MATS.JC.txt"),
             targets=H.load_targets(
                 os.path.join(DB, "meta", "rmats_pc3e_gs689_positive_events.tsv"), None)),
    ]


def _rep_group_mean_psi(inc_vec, exc_vec, ratio: float, draws: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Replicate-aware group mean PSI (mirror of production _draw_replicate_mean_psi)."""
    inc = np.asarray(inc_vec, dtype=float)
    exc = np.asarray(exc_vec, dtype=float) * ratio
    n = inc.size
    idx = rng.integers(0, n, size=(draws, n))
    return rng.beta(inc[idx] + 0.5, exc[idx] + 0.5).mean(axis=1)


def _oriented_vectors(ev, swap: bool):
    """Return (ctrl_inc, ctrl_exc, treat_inc, treat_exc) replicate vectors per the
    RT-PCR orientation -- identical assignment to head_to_head event_counts()."""
    if swap:  # ctrl = sample_2, treat = sample_1
        return ev.ijc2, ev.sjc2, ev.ijc1, ev.sjc1
    return ev.ijc1, ev.sjc1, ev.ijc2, ev.sjc2


def _collect() -> dict:
    """Build the matched-event records with sum + rep point/scale per event."""
    rng = np.random.default_rng(SEED)
    per_ds: dict[str, list[dict]] = {}
    fell_back = 0
    for cfg in _datasets():
        events = H.parse_se_table(cfg["rmats_se"])
        rows = []
        for t in cfg["targets"]:
            ev = H.match_event(t, events)
            if ev is None:
                continue
            ec = H.event_counts(ev, normalize=True, swap_groups=cfg["swap"])
            if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
                continue
            # sum: the paper's BRAID-Jeffreys point/scale (pooled, normalized).
            sum_mean, sum_std, _, _ = H.beta_interval(ec, ALPHA, 0.5, rng)

            # rep: replicate bootstrap on the oriented per-replicate vectors.
            ci, ce, ti, te = _oriented_vectors(ev, cfg["swap"])
            ratio = ev.inc_form_len / ev.skip_form_len if ev.skip_form_len > 0 else 1.0
            complete = (len(ci) == len(ce) >= 2) and (len(ti) == len(te) >= 2)
            if complete:
                ctrl = _rep_group_mean_psi(ci, ce, ratio, DRAWS, rng)
                treat = _rep_group_mean_psi(ti, te, ratio, DRAWS, rng)
                d = ctrl - treat
                rep_mean, rep_std, model = float(d.mean()), float(d.std()), "rep"
            else:  # auto fallback to sum for incomplete/ single-replicate rows
                rep_mean, rep_std, model = sum_mean, sum_std, "sum(fallback)"
                fell_back += 1

            rows.append(dict(
                gene=t.gene, truth=float(t.dpsi_rtpcr), support=float(ec.total_support),
                sum_mean=float(sum_mean), sum_std=float(max(sum_std, 1e-6)),
                rep_mean=float(rep_mean), rep_std=float(max(rep_std, 1e-6)),
                rep_model=model, n_ctrl=len(ci), n_treat=len(ti),
            ))
        per_ds[cfg["name"]] = rows
    return {"per_dataset": per_ds, "n_fell_back": fell_back}


def _score(rows: list[dict]) -> dict:
    """Coverage / width / iscore for sum and rep via the same cross-fit conformal."""
    if not rows:
        return {}
    truths = np.array([r["truth"] for r in rows])
    supports = np.array([r["support"] for r in rows])
    out: dict = {"n": len(rows)}
    cover = {}
    for tag, mkey, skey in (("sum", "sum_mean", "sum_std"), ("rep", "rep_mean", "rep_std")):
        points = np.array([r[mkey] for r in rows])
        scales = np.maximum(np.array([r[skey] for r in rows]), 1e-6)
        rmse = float(np.sqrt(np.mean((points - truths) ** 2)))
        block = {"point_rmse_vs_rtpcr": rmse}
        for variant, sc in (("conformal_abs", np.ones(len(rows))),  # paper headline (scale=1)
                            ("conformal_std", scales)):              # posterior-std scale
            lows, highs = H.conformal_crossfit(points, truths, sc, supports, ALPHA, seed=SEED)
            covered = (truths >= lows) & (truths <= highs)
            block[variant] = {
                "coverage95": float(covered.mean()),
                "mean_width": float(np.mean(highs - lows)),
                "interval_score": float(np.mean([
                    H.interval_score(truths[i], lows[i], highs[i], ALPHA)
                    for i in range(len(rows))])),
            }
            cover[(tag, variant)] = covered
        out[tag] = block
    # McNemar: sum-covered vs rep-covered (paper-headline conformal_abs at 95%)
    cs, cr = cover[("sum", "conformal_abs")], cover[("rep", "conformal_abs")]
    b = int(np.sum(cs & ~cr))   # sum covers, rep misses
    c = int(np.sum(~cs & cr))   # rep covers, sum misses
    if b + c > 0:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p = float(stats.chi2.sf(stat, 1))
    else:
        stat, p = 0.0, 1.0
    out["mcnemar_sum_vs_rep_conformal_abs"] = {
        "b_sum_covers_rep_misses": b, "c_rep_covers_sum_misses": c,
        "stat": float(stat), "p": p}
    return out


def main() -> None:
    data = _collect()
    per_ds = data["per_dataset"]
    pooled = [r for rows in per_ds.values() for r in rows]
    report = {
        "draws": DRAWS, "alpha": ALPHA, "seed": SEED,
        "n_fell_back_to_sum": data["n_fell_back"],
        "pooled": _score(pooled),
        "per_dataset": {name: _score(rows) for name, rows in per_ds.items()},
    }
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    def line(tag_block: dict, label: str) -> None:
        ab, st = tag_block["conformal_abs"], tag_block["conformal_std"]
        print(f"  {label:5s} | RMSE {tag_block['point_rmse_vs_rtpcr']:.3f} | "
              f"abs: cov {ab['coverage95']:.3f} w {ab['mean_width']:.3f} "
              f"is {ab['interval_score']:.3f} | "
              f"std: cov {st['coverage95']:.3f} w {st['mean_width']:.3f} "
              f"is {st['interval_score']:.3f}")

    print("=" * 78)
    print("SUM vs REP differential model on RT-PCR head-to-head (same conformal cross-fit)")
    print(f"draws={DRAWS} alpha={ALPHA} seed={SEED} | fell_back_to_sum={data['n_fell_back']}")
    print("=" * 78)
    print(f"POOLED (n={report['pooled']['n']})")
    line(report["pooled"]["sum"], "sum")
    line(report["pooled"]["rep"], "rep")
    mc = report["pooled"]["mcnemar_sum_vs_rep_conformal_abs"]
    print(f"  McNemar (conformal_abs): b(sum-only)={mc['b_sum_covers_rep_misses']} "
          f"c(rep-only)={mc['c_rep_covers_sum_misses']} p={mc['p']:.3g}")
    for name, sc in report["per_dataset"].items():
        if not sc:
            continue
        print(f"{name} (n={sc['n']})")
        line(sc["sum"], "sum")
        line(sc["rep"], "rep")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
