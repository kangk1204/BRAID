"""Sequencing-depth titration of ΔPSI interval coverage vs RT-PCR truth.

*How does each method's calibration respond as sequencing depth drops?*

Read-level subsampling of an RNA-seq library to fraction ``f`` retains each
junction-supporting read independently with probability ``f``; the recounted
inclusion/skipping junction count is therefore ``Binomial(N, f)``. This is the
exact statistical model of downsampling for the count-driven head-to-head
pipeline (which derives every interval directly from the rMATS IJC/SJC tables),
so we *thin* the published count tables instead of re-running rMATS on
downsampled BAMs. Binomial thinning ADDS the correct sampling noise -- it is not
the naive deterministic count-scaling (which would merely rescale and omit the
growth of sampling variance at low depth).

The orthogonal RT-PCR ΔPSI truth is fixed (never downsampled). At each depth we
recompute, from the thinned counts, every method that is a deterministic
function of those counts: rMATS per-replicate t-CI, betAS-style difference of
Betas, BRAID-Jeffreys, and BRAID-conformal / conformal-abs (cross-fit). Real
MAJIQ and the real-betAS R tool cannot be re-depthed without re-running them, so
they are out of scope here (their full-depth coverage is the documented
reference in ``headtohead_coverage.json``).

Pre-registered predictions:
  H1  BRAID-conformal coverage ~ flat at nominal across depth; the sampling-only
      methods' coverage DROPS as depth rises (their shrinking sampling intervals
      fall below the depth-invariant orthogonal-truth error floor).
  H2  BRAID width decreases then plateaus (knee at the floor); sampling-only
      widths decrease monotonically (overconfidence).
  H3  the sampling-variance fraction (posterior std vs point-RMSE-to-RT-PCR)
      shrinks as depth rises -- the ~94% out-of-sampling error floor is itself
      depth-dependent (lower at shallow depth).

Run from anywhere::

    python benchmarks/headtohead/depth_titration.py
    python benchmarks/headtohead/depth_titration.py --fractions 0.05,0.25,1.0 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import head_to_head_coverage as H  # noqa: E402  (sys.path set just above)

FRACTIONS_DEFAULT = (0.05, 0.10, 0.25, 0.50, 0.75, 1.0)
TRACKED_METHODS = (
    "BRAID-conformal",
    "BRAID-conformal-abs",
    "rMATS-perRep",
    "betAS",
    "BRAID-Jeffreys",
)


# ---------------------------------------------------------------------------
# Binomial thinning of a parsed SEEvent
# ---------------------------------------------------------------------------


def _recompute_inclevel(
    ijc: tuple[int, ...], sjc: tuple[int, ...], inc_len: float, skip_len: float,
) -> tuple[float, ...]:
    """rMATS length-normalized PSI per replicate; 0/0 replicates drop out (NA)."""
    out: list[float] = []
    for i, s in zip(ijc, sjc):
        ni = i / inc_len
        ns = s / skip_len
        denom = ni + ns
        if denom > 0:
            out.append(ni / denom)
    return tuple(out)


def thin_event(ev: H.SEEvent, frac: float, rng: np.random.Generator) -> H.SEEvent:
    """Return a copy of *ev* with junction counts binomially thinned to *frac*."""
    if frac >= 1.0:
        return ev

    def th(vec: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(int(rng.binomial(c, frac)) for c in vec)

    ijc1, sjc1, ijc2, sjc2 = th(ev.ijc1), th(ev.sjc1), th(ev.ijc2), th(ev.sjc2)
    il1 = _recompute_inclevel(ijc1, sjc1, ev.inc_form_len, ev.skip_form_len)
    il2 = _recompute_inclevel(ijc2, sjc2, ev.inc_form_len, ev.skip_form_len)
    dpsi = float(np.mean(il1) - np.mean(il2)) if il1 and il2 else float("nan")
    return replace(
        ev, ijc1=ijc1, sjc1=sjc1, ijc2=ijc2, sjc2=sjc2,
        inclevel1=il1, inclevel2=il2, rmats_dpsi=dpsi,
    )


# ---------------------------------------------------------------------------
# Sampling-scale / support diagnostics over the matched events (for H3)
# ---------------------------------------------------------------------------


def matched_stats(events: list[H.SEEvent], targets: list, swap: bool) -> dict:
    """Mean posterior std (sampling scale), mean support, point-RMSE on matches."""
    rng = np.random.default_rng(12345)
    stds, sups, pts, truths = [], [], [], []
    for t in targets:
        ev = H.match_event(t, events)
        if ev is None:
            continue
        ec = H.event_counts(ev, normalize=True, swap_groups=swap)
        if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
            continue
        m, s, _, _ = H.beta_interval(ec, 0.05, 0.5, rng)
        stds.append(s)
        sups.append(ec.total_support)
        pts.append(m)
        truths.append(t.dpsi_rtpcr)
    pts_a, truths_a = np.array(pts), np.array(truths)
    rmse = float(np.sqrt(np.mean((pts_a - truths_a) ** 2))) if pts_a.size else float("nan")
    mean_std = float(np.mean(stds)) if stds else float("nan")
    samp_frac = float(mean_std**2 / rmse**2) if (rmse and not math.isnan(rmse)) else float("nan")
    return {
        "n_matched": len(pts),
        "mean_support": float(np.mean(sups)) if sups else float("nan"),
        "mean_post_std": mean_std,
        "point_rmse": rmse,
        "sampling_var_fraction": samp_frac,
    }


# ---------------------------------------------------------------------------
# Titration driver
# ---------------------------------------------------------------------------


def datasets_config() -> list[dict]:
    db = "data/public_benchmarks"
    circ = f"{db}/meta/gse54651_circadian_positive_events.tsv"
    return [
        dict(name="TRA2 (GSE59335, human)", swap=True,
             rmats_se=f"{db}/GSE59335/rmats/SE.MATS.JC.txt",
             targets=H.load_targets(f"{db}/GSE59335/targets/validated_events.tsv",
                                    f"{db}/GSE59335/targets/failed_events.tsv")),
        dict(name="Circadian (GSE54651, mouse)", swap=True,
             rmats_se=f"{db}/GSE54651/rmats/SE.MATS.JC.txt",
             targets=H.load_circadian_targets(circ)),
        dict(name="PC3E/GS689 (SRA, human)", swap=False,
             rmats_se=f"{db}/SRS354082/rmats/SE.MATS.JC.txt",
             targets=H.load_targets(f"{db}/meta/rmats_pc3e_gs689_positive_events.tsv", None)),
    ]


def run_one(thinned: list[H.SEEvent], cfg: dict) -> dict:
    """Run the published coverage computation on a thinned event list."""
    orig = H.parse_se_table
    H.parse_se_table = lambda *a, **k: thinned  # type: ignore[assignment]
    try:
        res = H.run_dataset(
            cfg["rmats_se"], None, None, cfg["name"],
            seed=7, betas_tsv=None, swap_groups=cfg["swap"], targets=cfg["targets"],
        )
    finally:
        H.parse_se_table = orig
    return res


def titrate(fractions: tuple[float, ...], n_seeds: int) -> dict:
    os.chdir(_REPO)
    configs = datasets_config()
    out: dict = {
        "config": {
            "fractions": list(fractions),
            "n_seeds": n_seeds,
            "tracked_methods": list(TRACKED_METHODS),
            "method": "binomial thinning of rMATS IJC/SJC count tables "
                      "(exact read-subsampling model); RT-PCR truth fixed.",
            "out_of_scope": "real MAJIQ + real-betAS(R) cannot be re-depthed "
                            "without re-running them; see headtohead_coverage.json.",
        },
        "per_dataset": {},
        "pooled": {},
    }

    # pooled accumulators: {frac: {method: {"covered": [...per seed], "n": [...]}}}
    pooled_acc: dict = {}

    for di, cfg in enumerate(configs):
        full = H.parse_se_table(cfg["rmats_se"])
        ds_out: dict = {}
        for frac in fractions:
            seeds = [0] if frac >= 1.0 else list(range(n_seeds))
            per_method: dict = {m: {"cov": [], "width": [], "iscore": [], "n": []}
                                for m in TRACKED_METHODS}
            diag_runs: list[dict] = []
            for s in seeds:
                rng = np.random.default_rng([1000 + s, int(round(frac * 1000)), di])
                thinned = [thin_event(ev, frac, rng) for ev in full]
                res = run_one(thinned, cfg)
                n_matched = res["n_matched"]
                for m in TRACKED_METHODS:
                    r = res["methods"].get(m)
                    if r is None:
                        continue
                    per_method[m]["cov"].append(r["coverage95"])
                    per_method[m]["width"].append(r["width95"])
                    per_method[m]["iscore"].append(r["interval_score95"])
                    per_method[m]["n"].append(n_matched)
                diag_runs.append(matched_stats(thinned, cfg["targets"], cfg["swap"]))

                fk = f"{frac:.2f}"
                pa = pooled_acc.setdefault(fk, {})
                for m in TRACKED_METHODS:
                    r = res["methods"].get(m)
                    if r is None:
                        continue
                    slot = pa.setdefault(m, {"covered": 0.0, "n": 0.0,
                                             "wsum": 0.0, "isum": 0.0})
                    slot["covered"] += r["coverage95"] * n_matched
                    slot["n"] += n_matched
                    slot["wsum"] += r["width95"] * n_matched
                    slot["isum"] += r["interval_score95"] * n_matched

            fk = f"{frac:.2f}"
            ds_out[fk] = {
                "methods": {
                    m: {
                        "coverage95_mean": float(np.mean(d["cov"])) if d["cov"] else None,
                        "coverage95_sd": float(np.std(d["cov"])) if len(d["cov"]) > 1 else 0.0,
                        "width95_mean": float(np.mean(d["width"])) if d["width"] else None,
                        "iscore95_mean": float(np.mean(d["iscore"])) if d["iscore"] else None,
                    }
                    for m, d in per_method.items()
                },
                "n_matched_mean": float(np.mean([d["n_matched"] for d in diag_runs])),
                "diagnostics": {
                    k: float(np.mean([d[k] for d in diag_runs]))
                    for k in ("mean_support", "mean_post_std", "point_rmse",
                              "sampling_var_fraction")
                },
            }
        out["per_dataset"][cfg["name"]] = ds_out

    for fk, pa in pooled_acc.items():
        out["pooled"][fk] = {}
        for m, slot in pa.items():
            n = slot["n"]
            out["pooled"][fk][m] = {
                "coverage95": float(slot["covered"] / n) if n else None,
                "width95": float(slot["wsum"] / n) if n else None,
                "iscore95": float(slot["isum"] / n) if n else None,
                "n_total": float(n),
            }
    return out


def print_summary(out: dict) -> None:
    fr = [f"{f:.2f}" for f in out["config"]["fractions"]]
    print("\n" + "=" * 88)
    print("DEPTH TITRATION — pooled ΔPSI coverage@95 vs sequencing depth (RT-PCR truth)")
    print("  method = binomial thinning of rMATS count tables; n_seeds="
          f"{out['config']['n_seeds']}")
    print("=" * 88)
    header = f"{'method':<20}" + "".join(f"{f:>9}" for f in fr)
    print(header)
    print("-" * len(header))
    for m in TRACKED_METHODS:
        cells = []
        for f in fr:
            v = out["pooled"].get(f, {}).get(m, {})
            cov = v.get("coverage95")
            cells.append(f"{cov:>9.3f}" if cov is not None else f"{'NA':>9}")
        print(f"{m:<20}" + "".join(cells))
    print("-" * len(header))
    print("pooled n per depth:", {f: int(out["pooled"].get(f, {}).get(
        "BRAID-conformal", {}).get("n_total", 0)) for f in fr})
    print("\nWidth@95 (pooled):")
    for m in TRACKED_METHODS:
        cells = []
        for f in fr:
            v = out["pooled"].get(f, {}).get(m, {})
            w = v.get("width95")
            cells.append(f"{w:>9.3f}" if w is not None else f"{'NA':>9}")
        print(f"{m:<20}" + "".join(cells))
    print("\nSampling-variance fraction vs depth (BRAID, pooled-ds mean of per-ds):")
    for name, ds in out["per_dataset"].items():
        cells = []
        for f in fr:
            d = ds.get(f, {}).get("diagnostics", {})
            sf = d.get("sampling_var_fraction")
            cells.append(f"{sf:>9.3f}" if sf is not None and not math.isnan(sf) else f"{'NA':>9}")
        print(f"  {name:<28}" + "".join(cells))
    print("=" * 88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fractions", default=",".join(str(f) for f in FRACTIONS_DEFAULT),
                    help="comma-separated library fractions (1.0 = full depth)")
    ap.add_argument("--seeds", type=int, default=5, help="thinning seeds per fraction")
    ap.add_argument("--out-json", default="benchmarks/results/depth_titration.json")
    args = ap.parse_args()

    fractions = tuple(sorted(float(x) for x in args.fractions.split(",")))
    out = titrate(fractions, args.seeds)
    print_summary(out)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
