#!/usr/bin/env python3
"""Fit and write BRAID's shipped default conformal PSI calibrator.

The artifact braid/target/calibration_artifacts/default_psi_conformal.json is the
support-stratified conformal schedule loaded by load_default_conformal_calibrator()
so users get calibrated-AND-sharp intervals out of the box.

Provenance:
  - With --fixture FILE (a per-event JSON from regenerate_pacbio_validation.py:
    fields psi_hat, lr_psi, support and either ci_low/ci_high or scale), the
    calibrator is fit on real long-read-validated PSI (training_scope='pacbio_lr')
    and the coverage guarantee holds for deployment exchangeable with that data.
  - Without a fixture, it is fit on a DETERMINISTIC Beta-Binomial synthetic set
    that mimics RNA-seq overdispersion (training_scope='synthetic_betabinom'). This
    is a sensible default (q ~ the parametric 1.96 per bin); the finite-sample
    guarantee strictly holds only when refit on data exchangeable with deployment.

Run:  python benchmarks/build_default_conformal_calibrator.py [--fixture FILE] [--alpha 0.05]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from braid.target.conformal import fit_conformal_calibrator  # noqa: E402
from braid.target.psi_bootstrap import bootstrap_psi, sample_psi_posterior  # noqa: E402

_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "braid" / "target" / "calibration_artifacts" / "default_psi_conformal.json"
)
_Z = 1.959963984540054


def _synthetic_calibration_set(n: int = 6000, seed: int = 20240601):
    """Deterministic Beta-Binomial calibration set scored through BRAID's posterior.

    Overdispersion (rho) makes the per-event residual structure closer to real
    RNA-seq than a pure binomial, so the fitted quantiles transfer better.
    """
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.02, 0.98, n)
    support = rng.integers(5, 800, n)
    rho = 0.02  # modest overdispersion
    conc = (1.0 - rho) / rho
    a = truth * conc
    b = (1.0 - truth) * conc
    p = rng.beta(a, b)  # per-event realized inclusion rate
    inc = rng.binomial(support, p)
    exc = support - inc

    est = np.empty(n)
    scale = np.empty(n)
    for i in range(n):
        est[i] = bootstrap_psi(int(inc[i]), int(exc[i]), seed=int(i), event_type="SE")[0]
        scale[i] = float(np.std(
            sample_psi_posterior(int(inc[i]), int(exc[i]), seed=int(i), event_type="SE")
        ))
    return est, truth, scale, support.astype(float), "synthetic_betabinom"


def _fixture_calibration_set(path: Path):
    raw = json.loads(path.read_text())
    events = raw.get("per_event") or raw.get("events") or []
    if not events:
        raise SystemExit(f"No per-event records in {path}")
    est = np.array([e["psi_hat"] for e in events], dtype=float)
    truth = np.array([e["lr_psi"] for e in events], dtype=float)
    support = np.array(
        [e.get("support", e.get("inclusion", 0) + e.get("exclusion", 0)) for e in events],
        dtype=float,
    )
    if all("scale" in e for e in events):
        scale = np.array([e["scale"] for e in events], dtype=float)
    else:
        scale = np.array([(e["ci_high"] - e["ci_low"]) / (2 * _Z) for e in events], dtype=float)
    return est, truth, np.maximum(scale, 1e-3), support, "pacbio_lr"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--fixture", type=Path, help="per-event PSI fixture (real long-read data)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=_ARTIFACT)
    args = p.parse_args(argv)

    if args.fixture:
        est, truth, scale, support, scope = _fixture_calibration_set(args.fixture)
    else:
        print(
            "No --fixture; fitting on deterministic synthetic Beta-Binomial set.",
            file=sys.stderr,
        )
        est, truth, scale, support, scope = _synthetic_calibration_set()

    cal = fit_conformal_calibrator(
        est, truth, scale, support, alpha=args.alpha, training_scope=scope
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cal.to_json(args.out)
    print(f"Wrote {args.out}")
    print(f"  training_scope={cal.training_scope}  alpha={cal.alpha}  q_global={cal.q_global:.4f}")
    print(f"  q_by_bin={ {k: round(v, 3) for k, v in cal.q_by_bin.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
