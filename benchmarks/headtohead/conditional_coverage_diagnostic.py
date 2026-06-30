"""Stage 0 — quantify the constant band's CONDITIONAL miscalibration (GO/NO-GO gate).

The shipped differential calibrator has a near-constant half-width (q_by_bin
0.28-0.34). It is *marginally* calibrated (~0.95 pooled) but may be
*conditionally* miscalibrated: over-covering "easy" subgroups and under-covering
"hard" ones. A single cutoff cannot fix conditional coverage; a group-conditional
(Mondrian) calibrator can. This script measures whether that conditional gap
actually exists, with leakage-free cross-fit coverage and Wilson 95% CIs.

Baseline ("constant band"): the EXISTING leakage-free
``conformal_crossfit(points, truths, np.ones(n), supports, alpha, k=5)`` — a
scale-constant, support-Mondrian split-conformal band (matches deployment, where
support>250 for nearly all events so it collapses to a near-global constant q).

PRE-REGISTERED ENDPOINT (confirmatory):
  GO  if >=1 partition has two (n>=10) bins with DISJOINT Wilson 95% CIs
      AND >=1 (n>=10) bin whose Wilson 95% CI EXCLUDES 0.95.
  NO-GO if every (n>=10) bin's Wilson 95% CI contains 0.95 -> STOP and write up
      "the constant band is already conditionally calibrated within power"
      (this refutes the reviewer premise; still a result).

Confirmatory partitions: dataset, boundary-proximity (within-SE), event-type
(A3SS/A5SS, PacBio surface), PSI-level (PacBio). Exploratory: support, |dPSI|.
Bins with n<10 are reported descriptive-only and excluded from the gate.

Output: benchmarks/results/conditional_coverage_diagnostic.json

Usage:  python benchmarks/headtohead/conditional_coverage_diagnostic.py
"""

from __future__ import annotations

# ruff: noqa: I001  (third-party + local imports are split by the sys.path setup below)

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from braid.target.conformal import assign_support_bins  # noqa: E402
from comprehensive_benchmark import wilson  # noqa: E402
from head_to_head_coverage import conformal_crossfit, interval_score  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

ALPHA = 0.05
SEEDS = list(range(10))
MIN_BIN_N = 10
NOMINAL = 1.0 - ALPHA
DPSI_DATASETS = ("tra2", "circ", "srs", "jurkat")  # head-to-head set (qki/pc3e excluded; see notes)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _read_tsv(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        return [
            dict(zip(header, ln.rstrip("\n").split("\t")))
            for ln in fh
            if ln.strip()
        ]


def load_dpsi_dataset(name: str) -> dict:
    """Reconstruct ΔPSI point/truth/support/boundary from committed betas TSVs.

    point = Jeffreys-Beta posterior mean per group (Sum_inc+0.5)/(Sum_inc+Sum_exc+1),
    ΔPSI = group0 - group1, oriented to truth by per-dataset sign-correlation
    (orientation is a global experiment-design property, not per-event fitting).
    boundary = min over groups of min(psi, 1-psi)  (orientation-invariant).
    """
    counts = _read_tsv(os.path.join(_HERE, f"{name}_betas_counts.tsv"))
    truth = {r["key"]: r for r in _read_tsv(os.path.join(_HERE, f"{name}_betas_truth.tsv"))}
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for r in counts:
        g = agg[r["key"]][r["group"]]
        g[0] += float(r["inc"])
        g[1] += float(r["exc"])

    points, truths, supports, boundary, keys = [], [], [], [], []
    for key, groups in agg.items():
        if key not in truth or len(groups) != 2:
            continue
        glabels = sorted(groups)
        psis = [(groups[gl][0] + 0.5) / (groups[gl][0] + groups[gl][1] + 1.0) for gl in glabels]
        points.append(psis[0] - psis[1])
        boundary.append(min(min(p, 1.0 - p) for p in psis))
        truths.append(float(truth[key]["truth"]))
        supports.append(float(truth[key]["support"]))
        keys.append(key)

    pts = np.asarray(points)
    trs = np.asarray(truths)
    corr = float(np.corrcoef(pts, trs)[0, 1]) if len(pts) >= 3 and pts.std() > 0 else float("nan")
    flipped = corr < 0
    if flipped:
        pts = -pts
    return {
        "points": pts, "truths": trs, "supports": np.asarray(supports),
        "boundary": np.asarray(boundary), "dataset": [name] * len(pts),
        "keys": keys, "orient_corr": corr, "orient_flipped": bool(flipped),
    }


def load_dpsi_surface() -> dict:
    parts = [load_dpsi_dataset(n) for n in DPSI_DATASETS]
    return {
        "points": np.concatenate([p["points"] for p in parts]),
        "truths": np.concatenate([p["truths"] for p in parts]),
        "supports": np.concatenate([p["supports"] for p in parts]),
        "boundary": np.concatenate([p["boundary"] for p in parts]),
        "dataset": [d for p in parts for d in p["dataset"]],
        "orient": {n: {"corr": p["orient_corr"], "flipped": p["orient_flipped"]}
                   for n, p in zip(DPSI_DATASETS, parts)},
    }


def load_pacbio_surface() -> dict:
    path = os.path.join(_REPO, "benchmarks/results/rtpcr_benchmark.json")
    pe = json.load(open(path, encoding="utf-8"))["pacbio_psi"]["per_event"]
    return {
        "points": np.array([float(e["psi_hat"]) for e in pe]),
        "truths": np.array([float(e["lr_psi"]) for e in pe]),
        "supports": np.array([float(e["support"]) for e in pe]),
        "event_type": [str(e["event_id"]).split(":")[0] for e in pe],
    }


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #
def _labels_support(s: np.ndarray) -> list[str]:
    return [f"support:{b}" for b in assign_support_bins(np.asarray(s, dtype=float)).astype(str)]


def _labels_edges(x: np.ndarray, edges: tuple[float, ...], prefix: str) -> list[str]:
    out = []
    for v in np.asarray(x, dtype=float):
        j = int(np.searchsorted(edges, v, side="right")) - 1
        j = min(max(j, 0), len(edges) - 2)
        out.append(f"{prefix}:[{edges[j]:g},{edges[j + 1]:g})")
    return out


def partitions_dpsi(d: dict) -> dict[str, list[str]]:
    return {
        "dataset": [f"dataset:{x}" for x in d["dataset"]],
        "boundary_proximity": _labels_edges(d["boundary"], (0.0, 0.05, 0.15, 0.5001), "bd"),
        "abs_dpsi": _labels_edges(np.abs(d["points"]), (0.0, 0.1, 0.3, 1.0001), "adp"),
        "support": _labels_support(d["supports"]),
    }


def partitions_pacbio(d: dict) -> dict[str, list[str]]:
    return {
        "event_type": [f"etype:{x}" for x in d["event_type"]],
        "psi_level": _labels_edges(d["points"], (0.0, 0.1, 0.9, 1.0001), "psi"),
        "support": _labels_support(d["supports"]),
    }


# --------------------------------------------------------------------------- #
# Coverage + gate
# --------------------------------------------------------------------------- #
def constant_band(d: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = d["points"].size
    return conformal_crossfit(d["points"], d["truths"], np.ones(n), d["supports"], ALPHA, seed=seed)


def per_bin_table(covered: np.ndarray, widths: np.ndarray, iscores: np.ndarray,
                  labels: list[str]) -> list[dict]:
    labels = np.asarray(labels)
    rows = []
    for b in sorted(set(labels.tolist())):
        m = labels == b
        n = int(m.sum())
        k = int(covered[m].sum())
        _, lo, hi = wilson(k, n)
        rows.append({
            "bin": b, "k": k, "n": n,
            "coverage": k / n if n else float("nan"),
            "wilson": [lo, hi],
            "mean_width": float(widths[m].mean()) if n else float("nan"),
            "mean_iscore": float(iscores[m].mean()) if n else float("nan"),
            "binom_p": float(binomtest(k, n, NOMINAL).pvalue) if n else float("nan"),
            "descriptive_only": n < MIN_BIN_N,
        })
    elig = [r for r in rows if not r["descriptive_only"]]
    if elig:
        _, holm, _, _ = multipletests([r["binom_p"] for r in elig], method="holm")
        for r, hp in zip(elig, holm):
            r["holm_p"] = float(hp)
    for r in rows:
        r.setdefault("holm_p", None)
    return rows


def gate_for_partition(rows: list[dict]) -> dict:
    elig = [r for r in rows if not r["descriptive_only"]]
    excl = [r for r in elig if r["wilson"][0] > NOMINAL or r["wilson"][1] < NOMINAL]
    disjoint = any(
        a["wilson"][1] < b["wilson"][0] or b["wilson"][1] < a["wilson"][0]
        for i, a in enumerate(elig) for b in elig[i + 1:]
    )
    gaps = [abs(r["coverage"] - NOMINAL) for r in elig]
    return {
        "n_eligible_bins": len(elig),
        "has_disjoint_pair": bool(disjoint),
        "n_bins_excluding_0.95": len(excl),
        "bins_excluding_0.95": [r["bin"] for r in excl],
        "max_abs_gap": max(gaps) if gaps else None,
        "worst_bin_coverage": min((r["coverage"] for r in elig), default=None),
        "best_bin_coverage": max((r["coverage"] for r in elig), default=None),
        "partition_go": bool(disjoint and excl),
    }


def evaluate_surface(d: dict, parts: dict[str, list[str]]) -> dict:
    # seed=0 reference tables (honest n for Wilson); seed sweep for stability.
    low0, high0 = constant_band(d, 0)
    cov0 = (d["truths"] >= low0) & (d["truths"] <= high0)
    width0 = high0 - low0
    isc0 = np.array([interval_score(y, lo, hi, ALPHA)
                     for y, lo, hi in zip(d["truths"], low0, high0)])
    out = {"n": int(d["points"].size), "partitions": {}}
    for pname, labels in parts.items():
        rows = per_bin_table(cov0, width0, isc0, labels)
        out["partitions"][pname] = {"bins": rows, "gate": gate_for_partition(rows)}

    # seed sweep: max_abs_gap distribution + fraction of seeds with partition_go
    sweep: dict[str, dict] = {p: {"max_abs_gap": [], "partition_go": []} for p in parts}
    for s in SEEDS:
        lo, hi = constant_band(d, s)
        cov = (d["truths"] >= lo) & (d["truths"] <= hi)
        w = hi - lo
        isc = np.array([interval_score(y, a, b, ALPHA) for y, a, b in zip(d["truths"], lo, hi)])
        for pname, labels in parts.items():
            g = gate_for_partition(per_bin_table(cov, w, isc, labels))
            sweep[pname]["max_abs_gap"].append(g["max_abs_gap"])
            sweep[pname]["partition_go"].append(g["partition_go"])
    for pname in parts:
        gaps = [x for x in sweep[pname]["max_abs_gap"] if x is not None]
        gos = sweep[pname]["partition_go"]
        out["partitions"][pname]["seed_sweep"] = {
            "max_abs_gap_mean": float(np.mean(gaps)) if gaps else None,
            "max_abs_gap_min": float(np.min(gaps)) if gaps else None,
            "max_abs_gap_max": float(np.max(gaps)) if gaps else None,
            "frac_seeds_partition_go": float(np.mean(gos)) if gos else 0.0,
        }
    return out


def main() -> None:
    dpsi = load_dpsi_surface()
    pacbio = load_pacbio_surface()
    result = {
        "alpha": ALPHA, "k": 5, "seeds": SEEDS, "min_bin_n": MIN_BIN_N,
        "nominal": NOMINAL,
        "dpsi_datasets": list(DPSI_DATASETS),
        "n_dpsi": int(dpsi["points"].size),
        "n_pacbio": int(pacbio["points"].size),
        "orientation": dpsi["orient"],
        "dpsi": evaluate_surface(dpsi, partitions_dpsi(dpsi)),
        "pacbio": evaluate_surface(pacbio, partitions_pacbio(pacbio)),
    }

    # Global GO/NO-GO across both surfaces (seed=0 gate; report seed stability).
    passing = []
    for surf in ("dpsi", "pacbio"):
        for pname, pdata in result[surf]["partitions"].items():
            if pdata["gate"]["partition_go"]:
                passing.append(f"{surf}.{pname}")
    go = len(passing) > 0
    result["go_decision"] = {
        "GO": go,
        "passing_partitions_seed0": passing,
        "rule": "GO if >=1 partition has disjoint (n>=10) Wilson CIs AND >=1 bin "
                "whose Wilson CI excludes 0.95",
    }

    out_path = os.path.join(_REPO, "benchmarks/results/conditional_coverage_diagnostic.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # human summary
    print(f"n_dpsi={result['n_dpsi']}  n_pacbio={result['n_pacbio']}")
    print("orientation (corr, flipped):",
          {k: (round(v["corr"], 3), v["flipped"]) for k, v in dpsi["orient"].items()})
    for surf in ("dpsi", "pacbio"):
        print(f"\n=== {surf} surface ===")
        for pname, pdata in result[surf]["partitions"].items():
            g = pdata["gate"]
            ss = pdata["seed_sweep"]
            print(f"  [{pname}] eligible_bins={g['n_eligible_bins']} "
                  f"max_abs_gap={g['max_abs_gap']} worst={g['worst_bin_coverage']} "
                  f"best={g['best_bin_coverage']} disjoint={g['has_disjoint_pair']} "
                  f"excl0.95={g['n_bins_excluding_0.95']} GO={g['partition_go']} "
                  f"frac_seeds_go={ss['frac_seeds_partition_go']:.2f}")
            for r in pdata["bins"]:
                tag = " (n<10)" if r["descriptive_only"] else ""
                print(f"      {r['bin']:<26} n={r['n']:<4} cov={r['coverage']:.3f} "
                      f"CI=[{r['wilson'][0]:.3f},{r['wilson'][1]:.3f}] "
                      f"w={r['mean_width']:.3f}{tag}")
    print(f"\n>>> GO/NO-GO: {'GO' if go else 'NO-GO'}  passing={passing}")
    print(f">>> wrote {os.path.relpath(out_path, _REPO)}")


if __name__ == "__main__":
    main()
