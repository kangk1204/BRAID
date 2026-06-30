#!/usr/bin/env python3
"""QKI (GSE55215) — a 4th orthogonal-validation dataset (detection axis).

QKI knockdown (2 ctrl vs 2 KD, hg19). The orthogonal validation is a list of 81
QKI-regulated cassette exons confirmed by RT-PCR (no quantitative ΔPSI, so this is
a DETECTION benchmark, not a coverage one). We match the 81 validated exons to the
rMATS SE events run here and ask, per method, how many are flagged significant
(sensitivity / recall on RT-PCR-confirmed targets).
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import head_to_head_coverage as H  # noqa: E402

from braid.target.conformal import load_differential_conformal_calibrator  # noqa: E402

RMATS = "data/public_benchmarks/GSE55215/rmats/SE.MATS.JC.txt"
TRUTH = "data/public_benchmarks/meta/qki_positive_events.tsv"
CUTOFF = 0.1


def load_qki_targets(path):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out.append(H.Target(
                gene=row["gene"].strip(), chrom=H._norm_chrom(row["chrom"]),
                exon_start=int(row["exon_start"]), exon_end=int(row["exon_end"]),
                dpsi_rtpcr=float("nan"), is_positive=True))
    return out


def main():
    events = H.parse_se_table(RMATS)
    targets = load_qki_targets(TRUTH)
    cal = load_differential_conformal_calibrator()
    rng = np.random.default_rng(7)
    matched = 0
    rmats_sig = 0
    braid_supported = 0
    braid_conf_excl0 = 0
    for t in targets:
        ev = H.match_event(t, events)
        if ev is None:
            continue
        ec = H.event_counts(ev, normalize=True, swap_groups=False)
        if (ec.a_c + ec.b_c) < 1 or (ec.a_t + ec.b_t) < 1:
            continue
        matched += 1
        s = H._dpsi_samples(ec, 0.5, rng, n=4000)
        m = float(s.mean())
        prob_large = float(np.mean(np.abs(s) >= CUTOFF))
        is_rmats_sig = np.isfinite(ec.rmats_fdr) and ec.rmats_fdr < 0.05
        rmats_sig += int(is_rmats_sig)
        braid_supported += int(is_rmats_sig and prob_large >= 0.5)
        lo, hi = cal.interval(m, 1.0, ec.total_support, clip=(-1.0, 1.0))
        braid_conf_excl0 += int((lo > 0) or (hi < 0))
    n = len(targets)
    print(f"\n{'='*64}")
    print(f"QKI (GSE55215) detection — {matched}/{n} RT-PCR exons matched to rMATS SE")
    print(f"{'-'*64}")
    print(f"{'method':<34}{'detected':>10}{'sensitivity':>14}")
    print(f"{'-'*64}")
    den = max(matched, 1)
    print(f"{'rMATS FDR<0.05':<34}{rmats_sig:>10}{rmats_sig/den:>14.3f}")
    print(f"{'BRAID supported (FDR & effect)':<34}"
          f"{braid_supported:>10}{braid_supported/den:>14.3f}")
    print(f"{'BRAID conformal interval excl 0':<34}"
          f"{braid_conf_excl0:>10}{braid_conf_excl0/den:>14.3f}")
    print(f"{'='*64}")
    print("Note: 2v2 design -> limited statistical power; QKI gives a positives-only RT-PCR")
    print("list (no quantitative ΔPSI), so this is detection/recall, not interval coverage.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(_HERE)))
    main()
