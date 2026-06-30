#!/usr/bin/env python3
"""Export per-event, per-replicate normalized inc/exc counts for the real betAS bridge.

Writes two TSVs consumed by ``run_betas.R``:
  - counts long table: key, group(ctrl|treat), rep, inc, exc   (length-normalized
    effective counts, so betAS is judged on the same molecular-PSI scale as every
    other method -- isolating the *interval-construction* question from the
    orthogonal length-normalization question).
  - truth table: key, truth, is_positive, support.

``ctrl``/``treat`` are oriented to the RT-PCR convention (ctrl = minuend), matching
``head_to_head_coverage.event_counts(swap_groups=...)``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from head_to_head_coverage import (  # noqa: E402
    load_circadian_targets,
    load_targets,
    match_event,
    parse_se_table,
)


def _norm_rep(ijc: int, sjc: int, inc_fl: float, skip_fl: float) -> tuple[float, float]:
    ratio = inc_fl / skip_fl if skip_fl > 0 else 1.0
    return float(ijc), float(sjc) * ratio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rmats-se", required=True)
    ap.add_argument("--validated", default=None)
    ap.add_argument("--failed", default=None)
    ap.add_argument("--circadian-tsv", default=None)
    ap.add_argument("--swap-groups", action="store_true")
    ap.add_argument("--out-counts", required=True)
    ap.add_argument("--out-truth", required=True)
    args = ap.parse_args()

    events = parse_se_table(args.rmats_se)
    if args.circadian_tsv:
        targets = load_circadian_targets(args.circadian_tsv)
    else:
        targets = load_targets(args.validated, args.failed)

    counts_rows = []
    truth_rows = []
    idx = 0
    for t in targets:
        ev = match_event(t, events)
        if ev is None:
            continue
        # per-replicate normalized effective counts
        c_inc = [_norm_rep(i, s, ev.inc_form_len, ev.skip_form_len)
                 for i, s in zip(ev.ijc1, ev.sjc1)]
        t_inc = [_norm_rep(i, s, ev.inc_form_len, ev.skip_form_len)
                 for i, s in zip(ev.ijc2, ev.sjc2)]
        # orient: ctrl = RT-PCR minuend
        if args.swap_groups:
            ctrl_reps, treat_reps = t_inc, c_inc
        else:
            ctrl_reps, treat_reps = c_inc, t_inc
        support = (sum(ev.ijc1) + sum(ev.sjc1) + sum(ev.ijc2) + sum(ev.sjc2))
        ctrl_tot = sum(a + b for a, b in ctrl_reps)
        treat_tot = sum(a + b for a, b in treat_reps)
        if ctrl_tot < 1 or treat_tot < 1:
            continue
        key = f"ev{idx}"
        idx += 1
        for rep, (inc, exc) in enumerate(ctrl_reps):
            counts_rows.append((key, "ctrl", rep, f"{inc:.4f}", f"{exc:.4f}"))
        for rep, (inc, exc) in enumerate(treat_reps):
            counts_rows.append((key, "treat", rep, f"{inc:.4f}", f"{exc:.4f}"))
        truth_rows.append((key, f"{t.dpsi_rtpcr:.5f}",
                           "1" if t.is_positive else "0", f"{support:.0f}"))

    with open(args.out_counts, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["key", "group", "rep", "inc", "exc"])
        w.writerows(counts_rows)
    with open(args.out_truth, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["key", "truth", "is_positive", "support"])
        w.writerows(truth_rows)
    print(f"exported {len(truth_rows)} events, {len(counts_rows)} count rows")
    print(f"  counts -> {args.out_counts}")
    print(f"  truth  -> {args.out_truth}")


if __name__ == "__main__":
    main()
