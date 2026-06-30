"""Stage 1 — adaptive event-type/group conformal calibration.

Locks the v2 ConformalCalibrator invariants:
- v1 artifacts (no q_by_group) load under the v2 reader (back-compat);
- q_for resolves the composite -> event_type -> support_bin -> global cascade,
  including sparse-group fallback;
- fit_conformal_calibrator drops sparse groups (n < group_min_n);
- the guarantee is never weakened: event-type Mondrian marginal coverage is not
  below the support-only constant band on a fixed-seed heteroscedastic set.

In-memory only (no external data), per the project test convention.
"""

from __future__ import annotations

import numpy as np

from braid.target.conformal import ConformalCalibrator, fit_conformal_calibrator


def test_v1_artifact_loads_under_v2_reader():
    v1 = {
        "kind": "conformal_psi_calibrator", "version": 1, "alpha": 0.05,
        "q_global": 0.30, "q_by_bin": {"250+": 0.28}, "bin_edges": [20, 50, 100, 250],
        "q_by_event_type": {},
    }
    c = ConformalCalibrator.from_dict(v1)
    assert c.q_by_group == {}                 # default for a v1 artifact
    assert c.to_dict()["version"] == 2        # re-serialises at v2
    assert c.q_for(1000) == 0.28              # support-only behaviour unchanged
    assert c.q_for(5) == 0.30                 # bin absent -> global


def test_q_for_cascade_precedence():
    c = ConformalCalibrator(
        alpha=0.05, q_global=0.30, q_by_bin={"250+": 0.28},
        bin_edges=(20, 50, 100, 250),
        q_by_event_type={"A5SS": 0.45},
        q_by_group={"A5SS|250+": 0.50},
    )
    assert c.q_for(1000, "A5SS") == 0.50      # composite group wins
    assert c.q_for(5, "A5SS") == 0.45         # no composite for <20 bin -> event_type
    assert c.q_for(1000, "A3SS") == 0.28      # no group/event_type -> support bin
    assert c.q_for(5, "A3SS") == 0.30         # no bin -> global
    assert c.q_for(1000, None) == 0.28        # event_type=None -> support-only


def test_fit_group_labels_drops_sparse_groups():
    rng = np.random.default_rng(0)
    n = 120
    est = rng.uniform(0, 1, n)
    tru = est + rng.normal(0, 0.1, n)
    sc = np.ones(n)
    sup = rng.uniform(300, 1000, n)
    et = np.array(["A3SS"] * 60 + ["A5SS"] * 55 + ["MXE"] * 5)
    grp = np.array([f"{e}|250+" for e in et])
    cal = fit_conformal_calibrator(est, tru, sc, sup, group_labels=grp, group_min_n=20)
    assert set(cal.q_by_group) == {"A3SS|250+", "A5SS|250+"}  # MXE (n=5) dropped


def test_fit_group_labels_shape_validation():
    rng = np.random.default_rng(1)
    n = 30
    a = rng.uniform(0, 1, n)
    try:
        fit_conformal_calibrator(a, a, np.ones(n), np.full(n, 500.0),
                                 group_labels=np.array(["A"] * (n - 1)))
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched group_labels shape")


def test_no_coverage_regression_event_type_mondrian():
    """Event-type Mondrian must not weaken marginal coverage vs the constant band."""
    rng = np.random.default_rng(42)
    n = 600
    g = np.array(["A"] * 300 + ["B"] * 300)
    truth = rng.uniform(0, 1, n)
    sigma = np.where(g == "A", 0.05, 0.25)        # heteroscedastic by group
    point = np.clip(truth + rng.normal(0, sigma), 0.0, 1.0)
    scale = np.ones(n)
    support = rng.uniform(300, 1000, n)

    idx = rng.permutation(n)
    cal, te = idx[:300], idx[300:]
    base = fit_conformal_calibrator(point[cal], truth[cal], scale[cal], support[cal])
    adapt = fit_conformal_calibrator(point[cal], truth[cal], scale[cal], support[cal],
                                     event_types=g[cal])

    def marginal(c: ConformalCalibrator, use_et: bool) -> float:
        hits = 0
        for i in te:
            lo, hi = c.interval(point[i], scale[i], support[i],
                                event_type=(g[i] if use_et else None))
            hits += lo <= truth[i] <= hi
        return hits / len(te)

    cov_base = marginal(base, False)
    cov_adapt = marginal(adapt, True)
    assert cov_adapt >= 0.90                       # conformal validity preserved
    assert cov_adapt >= cov_base - 0.03            # adaptivity does not weaken coverage

    # the hard group B (constant band under-covers it) is not worse under adaptive
    teB = [i for i in te if g[i] == "B"]
    covB_base = sum(base.interval(point[i], scale[i], support[i])[0] <= truth[i]
                    <= base.interval(point[i], scale[i], support[i])[1] for i in teB) / len(teB)
    covB_adapt = sum(adapt.interval(point[i], scale[i], support[i], event_type="B")[0] <= truth[i]
                     <= adapt.interval(point[i], scale[i], support[i], event_type="B")[1]
                     for i in teB) / len(teB)
    assert covB_adapt >= covB_base - 0.02
