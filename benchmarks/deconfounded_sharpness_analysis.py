#!/usr/bin/env python3
"""Deconfounded sharpness analysis + conformal recalibration for PSI intervals.

Consumes a minimal per-event fixture (produced by regenerate_pacbio_validation.py)
with fields: psi_hat, lr_psi, support, ci_low, ci_high, gene (optional event_type),
and reports, per support bin and overall:

  1. BRAID interval coverage and mean width (the committed result).
  2. The DECONFOUNDED comparison: coverage of a width-matched interval centred on
     the SAME point estimate (the fair baseline) vs centred at random. The lift
     over the estimate-centred baseline isolates genuine width calibration; the
     (larger) lift over the random baseline is the potentially misleading number
     reported by the original analysis.
  3. CONFORMAL recalibration: leave-one-gene-out split/Mondrian conformal
     intervals built from BRAID's per-event uncertainty scale, which carry a
     distribution-free coverage guarantee. We report whether conformal reaches
     nominal coverage at comparable or better sharpness (interval score) than the
     fit-to-target BRAID intervals.

Use --demo to run on synthetic data when the real fixture is unavailable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from braid.target.calibration_metrics import (  # noqa: E402
    interval_score,
    sharpness_report,
)
from braid.target.conformal import (  # noqa: E402
    assign_support_bins,
    mondrian_conformal_intervals,
)

_Z = 1.959963984540054  # 97.5th percentile of N(0,1)


def _scale_from_ci(ci_low: np.ndarray, ci_high: np.ndarray) -> np.ndarray:
    """Approximate the per-event posterior sd from a 95% interval width."""
    return np.maximum((ci_high - ci_low) / (2.0 * _Z), 1e-3)


def load_fixture(path: Path) -> dict[str, np.ndarray]:
    raw = json.loads(path.read_text())
    # Accept either a flat ``per_event``/``events`` list or the canonical
    # rtpcr_benchmark.json layout, where the records live under
    # ``pacbio_psi.per_event`` — so the shipped artifact is directly consumable.
    events = (
        raw.get("per_event")
        or raw.get("events")
        or raw.get("pacbio_psi", {}).get("per_event")
        or []
    )
    if not events:
        raise SystemExit(f"No per-event records in {path}")
    def col(key: str, default=np.nan) -> np.ndarray:
        return np.array([ev.get(key, default) for ev in events], dtype=float)
    genes = [str(ev.get("gene", ev.get("gene_id", i))) for i, ev in enumerate(events)]
    return {
        "psi_hat": col("psi_hat"),
        "lr_psi": col("lr_psi"),
        "support": col("support"),
        "ci_low": col("ci_low"),
        "ci_high": col("ci_high"),
        "gene": np.array(genes, dtype=object),
    }


def make_demo(n: int = 204, n_genes: int = 15, seed: int = 0) -> dict[str, np.ndarray]:
    """Synthetic data mimicking the PacBio benchmark shape (overdispersed, wide
    low-support intervals)."""
    rng = np.random.default_rng(seed)
    support = rng.choice([10, 30, 70, 150, 400], size=n, p=[0.1, 0.1, 0.2, 0.2, 0.4]).astype(float)
    truth = rng.uniform(0.05, 0.95, size=n)
    noise = np.clip(0.5 / np.sqrt(support), 0.02, 0.35)
    psi_hat = np.clip(truth + rng.normal(0, noise), 0, 1)
    # BRAID-like fit-to-target intervals: very wide at low support, tight at high
    half = np.where(support >= 250, 1.5 * noise, 0.5)  # ~[0,1] for low support
    ci_low = np.clip(psi_hat - half, 0, 1)
    ci_high = np.clip(psi_hat + half, 0, 1)
    gene = np.array([f"G{i % n_genes}" for i in range(n)], dtype=object)
    return {"psi_hat": psi_hat, "lr_psi": truth, "support": support,
            "ci_low": ci_low, "ci_high": ci_high, "gene": gene}


def logo_conformal(d: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    """Leave-one-gene-out Mondrian conformal intervals on BRAID's per-event scale."""
    scale = _scale_from_ci(d["ci_low"], d["ci_high"])
    bins = assign_support_bins(d["support"])
    low = np.full(d["psi_hat"].shape, np.nan)
    high = np.full(d["psi_hat"].shape, np.nan)
    for g in np.unique(d["gene"]):
        test = d["gene"] == g
        train = ~test
        iv = mondrian_conformal_intervals(
            d["psi_hat"][train], d["lr_psi"][train], scale[train], bins[train],
            d["psi_hat"][test], scale[test], bins[test], alpha=alpha,
        )
        low[test] = iv.low
        high[test] = iv.high
    return {"low": low, "high": high}


def report(d: dict[str, np.ndarray], alpha: float = 0.05) -> None:
    bins = assign_support_bins(d["support"])
    order = ["<20", "20-49", "50-99", "100-249", "250+"]
    print("=" * 92)
    print("DECONFOUNDED SHARPNESS (BRAID intervals)")
    print("=" * 92)
    hdr = (f"{'bin':>9} {'n':>4} {'cov':>6} {'width':>7} {'IS':>7} "
           f"{'estCtrCov':>10} {'lift_est':>9} {'lift_rnd':>9}")
    print(hdr)
    for b in order:
        m = bins == b
        if not np.any(m):
            continue
        rep = sharpness_report(d["ci_low"][m], d["ci_high"][m], d["psi_hat"][m],
                               d["lr_psi"][m], alpha=alpha)
        print(f"{b:>9} {rep.n:>4} {rep.coverage:>6.3f} {rep.mean_width:>7.3f} "
              f"{rep.interval_score:>7.3f} {rep.estimate_centered_coverage:>10.3f} "
              f"{rep.lift_over_estimate_centered:>+9.3f} {rep.lift_over_random:>+9.3f}")
    overall = sharpness_report(d["ci_low"], d["ci_high"], d["psi_hat"], d["lr_psi"], alpha=alpha)
    print(f"{'overall':>9} {overall.n:>4} {overall.coverage:>6.3f} {overall.mean_width:>7.3f} "
          f"{overall.interval_score:>7.3f} {overall.estimate_centered_coverage:>10.3f} "
          f"{overall.lift_over_estimate_centered:>+9.3f} {overall.lift_over_random:>+9.3f}")
    print("\nINTERPRETATION: lift_est ~ 0 means BRAID's interval shape adds little beyond a")
    print("width-matched interval around the point estimate; the larger lift_rnd is the")
    print("confounded number (it rewards centring on an informative estimate).")

    print("\n" + "=" * 92)
    print("CONFORMAL RECALIBRATION (leave-one-gene-out, Mondrian; guaranteed coverage)")
    print("=" * 92)
    cf = logo_conformal(d, alpha)
    print(f"{'bin':>9} {'n':>4} {'cov':>6} {'width':>7} {'IS':>7}   vs BRAID IS")
    for b in order:
        m = (bins == b) & np.isfinite(cf["low"])
        if not np.any(m):
            continue
        cov = float(np.mean((d["lr_psi"][m] >= cf["low"][m]) & (d["lr_psi"][m] <= cf["high"][m])))
        width = float(np.mean(cf["high"][m] - cf["low"][m]))
        is_cf = interval_score(cf["low"][m], cf["high"][m], d["lr_psi"][m], alpha)
        is_braid = interval_score(d["ci_low"][m], d["ci_high"][m], d["lr_psi"][m], alpha)
        flag = "(sharper)" if is_cf < is_braid else ""
        print(f"{b:>9} {int(m.sum()):>4} {cov:>6.3f} {width:>7.3f} "
              f"{is_cf:>7.3f}   {is_braid:>7.3f} {flag}")
    ok = np.isfinite(cf["low"])
    cov_all = float(np.mean((d["lr_psi"][ok] >= cf["low"][ok])
                            & (d["lr_psi"][ok] <= cf["high"][ok])))
    is_cf_all = interval_score(cf["low"][ok], cf["high"][ok], d["lr_psi"][ok], alpha)
    is_braid_all = interval_score(d["ci_low"][ok], d["ci_high"][ok], d["lr_psi"][ok], alpha)
    width_all = float(np.mean(cf["high"][ok] - cf["low"][ok]))
    print(f"{'overall':>9} {int(ok.sum()):>4} {cov_all:>6.3f} "
          f"{width_all:>7.3f} {is_cf_all:>7.3f}   {is_braid_all:>7.3f}")
    print(f"\nTarget coverage {1-alpha:.0%}. Conformal carries a finite-sample guarantee; "
          "lower interval score (IS) = better calibration+sharpness than the "
          "fit-to-target intervals.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixture", type=Path, help="per-event fixture JSON")
    p.add_argument("--demo", action="store_true", help="run on synthetic data")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args(argv)
    if args.demo or args.fixture is None:
        if args.fixture is None and not args.demo:
            print("No --fixture given; running --demo synthetic data.\n", file=sys.stderr)
        d = make_demo()
    else:
        d = load_fixture(args.fixture)
    report(d, alpha=args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
