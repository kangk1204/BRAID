"""Reproducibility lock for the paper's PacBio CI-sharpness headline.

The manuscript's load-bearing "sharpness" result (the 250+ support bin: median
CI width 0.485, coverage 0.907, ~48pp lift over a matched-width random baseline)
must be regenerable from committed in-repo data, not from a developer's external
tree. This test recomputes the random-interval baseline and coverage lift from
the committed ``rtpcr_benchmark.json`` bin summary and asserts it matches the
committed ``pacbio_sharpness.json`` within 1e-3, mirroring the deterministic
seed-42 Monte Carlo in ``benchmarks/scripts/pacbio_sharpness_analysis.py``.

NOTE (scope): only bin-level *medians* are committed in-repo, so this verifies
the as-published numbers are reproducible. It does NOT deconfound the random
baseline (which assumes uniform truth + random-centered intervals); a fair,
estimate-centered comparator requires the per-event records, which are not yet
committed. See the SOTA plan's Phase 1 (sharpness deconfound).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RTPCR = _REPO_ROOT / "benchmarks" / "results" / "rtpcr_benchmark.json"
_SHARPNESS = _REPO_ROOT / "benchmarks" / "results" / "pacbio_sharpness.json"

_BIN_ORDER = ["<20", "20-49", "50-99", "100-249", "250+"]

pytestmark = pytest.mark.skipif(
    not (_RTPCR.exists() and _SHARPNESS.exists()),
    reason="committed PacBio benchmark result JSONs not present",
)


def test_sharpness_table_reproduces_in_repo() -> None:
    bins = json.loads(_RTPCR.read_text())["pacbio_psi"]["support_bin_summary"]
    committed = {r["bin"]: r for r in json.loads(_SHARPNESS.read_text())["sharpness_table"]}

    # The script seeds once (np.random.seed(42)) and draws per bin in order,
    # so reproduce the same draw sequence with a single shared generator.
    rng = np.random.RandomState(42)
    n = 100_000
    for bn in _BIN_ORDER:
        w = min(bins[bn]["median_ci_width"], 1.0)
        centers = rng.uniform(0, 1, n)
        lo = np.clip(centers - w / 2, 0, 1)
        hi = np.clip(centers + w / 2, 0, 1)
        true_psi = rng.uniform(0, 1, n)
        rand_cov = float(np.mean((true_psi >= lo) & (true_psi <= hi)))
        braid_cov = bins[bn]["ci_coverage"]
        lift = braid_cov - rand_cov

        c = committed[bn]
        assert abs(rand_cov - c["random_coverage"]) < 1e-3, bn
        assert abs(lift - c["coverage_lift"]) < 1e-3, bn
        # committed table rounds braid_coverage to 4 dp
        assert abs(braid_cov - c["braid_coverage"]) < 1e-3, bn


def test_high_support_headline_values() -> None:
    """Lock the specific 250+ headline the manuscript leads with.

    Values are the public-accession 252-event reproduction (SRR387661 STAR/GRCh38
    vs ENCFF652QLH long-read), which is the version reported in the manuscript.
    """
    bins = json.loads(_RTPCR.read_text())["pacbio_psi"]["support_bin_summary"]
    top = bins["250+"]
    assert top["n_events"] == 107
    assert abs(top["median_ci_width"] - 0.4851) < 1e-3
    assert abs(top["ci_coverage"] - 0.9065) < 1e-3
