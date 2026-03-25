"""BRAID v2 differential / replicate-level feature extractor.

Extracts 8 features per rMATS splicing event:
  B5  replicate_psi_variance    Variance of per-replicate PSI (sample_1 IncLevel)
  B6  replicate_psi_range       max - min per-replicate PSI
  D1  dpsi_ctrl_replicates      Range of PSI within control (sample_1) replicates
  D2  total_support_ctrl        Total junction reads in control (sample_1)
  D3  total_support_kd          Total junction reads in treatment (sample_2)
  D4  support_asymmetry         abs(log2(ctrl_total / kd_total))
  D5  rmats_fdr                 Passthrough from event
  D6  abs_dpsi                  abs(rmats_dpsi) passthrough
"""

from __future__ import annotations

import math

import numpy as np

from braid.target.rmats_bootstrap import RmatsEvent


def _replicate_psi_values(
    inc_replicates: tuple[int, ...],
    exc_replicates: tuple[int, ...],
) -> list[float]:
    """Compute per-replicate PSI = inc / (inc + exc).

    Returns a list of PSI values, one per replicate. Replicates with
    zero total reads are skipped.
    """
    n = min(len(inc_replicates), len(exc_replicates))
    psi_vals: list[float] = []
    for i in range(n):
        total = inc_replicates[i] + exc_replicates[i]
        if total > 0:
            psi_vals.append(inc_replicates[i] / total)
    return psi_vals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_differential_features(event: RmatsEvent) -> dict[str, float]:
    """Extract 8 differential/replicate features for one rMATS event.

    Parameters
    ----------
    event:
        An ``RmatsEvent`` with per-replicate counts and rMATS statistics.

    Returns
    -------
    dict with keys B5, B6, D1..D6 (descriptive names). Missing features are ``NaN``.
    """
    nan = float("nan")

    # ------------------------------------------------------------------
    # B5-B6, D1: Per-replicate PSI statistics (sample_1 = control)
    # ------------------------------------------------------------------
    ctrl_psi = _replicate_psi_values(
        event.sample_1_inc_replicates,
        event.sample_1_exc_replicates,
    )

    if len(ctrl_psi) >= 2:
        replicate_psi_variance = float(np.var(ctrl_psi, ddof=1))
        replicate_psi_range = float(max(ctrl_psi) - min(ctrl_psi))
        dpsi_ctrl_replicates = replicate_psi_range  # Same as range within ctrl
    elif len(ctrl_psi) == 1:
        replicate_psi_variance = 0.0
        replicate_psi_range = 0.0
        dpsi_ctrl_replicates = 0.0
    else:
        replicate_psi_variance = nan
        replicate_psi_range = nan
        dpsi_ctrl_replicates = nan

    # ------------------------------------------------------------------
    # D2-D4: Support counts and asymmetry
    # ------------------------------------------------------------------
    ctrl_total = event.sample_1_inc_count + event.sample_1_exc_count
    kd_total = event.sample_2_inc_count + event.sample_2_exc_count

    total_support_ctrl = float(ctrl_total)
    total_support_kd = float(kd_total)

    if ctrl_total > 0 and kd_total > 0:
        support_asymmetry = abs(math.log2(ctrl_total / kd_total))
    else:
        support_asymmetry = nan

    # ------------------------------------------------------------------
    # D5-D6: rMATS passthrough
    # ------------------------------------------------------------------
    rmats_fdr = event.rmats_fdr
    abs_dpsi = abs(event.rmats_dpsi) if not math.isnan(event.rmats_dpsi) else nan

    return {
        "replicate_psi_variance": replicate_psi_variance,
        "replicate_psi_range": replicate_psi_range,
        "dpsi_ctrl_replicates": dpsi_ctrl_replicates,
        "total_support_ctrl": total_support_ctrl,
        "total_support_kd": total_support_kd,
        "support_asymmetry": support_asymmetry,
        "rmats_fdr": rmats_fdr,
        "abs_dpsi": abs_dpsi,
    }
