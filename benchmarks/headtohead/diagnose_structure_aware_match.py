"""READ-ONLY diagnostic: how much does structure-aware event matching change the
TRA2 head-to-head set?  Changes NO committed result.

The current matcher (head_to_head_coverage.match_event) matches an RT-PCR target to
an rMATS SE event on the CASSETTE-EXON coordinates only (exon_start/exon_end, tol 6),
ignoring the flanking splice sites. The target table already carries the canonical
SUPPA SE id (gene;SE:chr:a-b:c-d:strand) with the full structure: a=upstream donor,
b-c=cassette exon, d=downstream acceptor. This script re-matches each positive target
using the FULL SE structure (cassette + both flanks + strand) and reports how many
targets get a different rMATS event, and the rMATS ΔPSI change for those (which is
what would move coverage).
"""
from __future__ import annotations

import csv
import math
import os

DB = "data/public_benchmarks/GSE59335"
VAL = f"{DB}/targets/validated_events.tsv"
RMATS = f"{DB}/rmats/SE.MATS.JC.txt"
TOL = 6


def parse_suppa(sid: str):
    """gene;SE:chr:a-b:c-d:strand -> (a, b, c, d, strand); None if unparseable."""
    if not sid or "SE:" not in sid:
        return None
    body = sid.split("SE:", 1)[1]          # chr:a-b:c-d:strand
    parts = body.split(":")
    if len(parts) < 4:
        return None
    _chr, j1, j2, strand = parts[0], parts[1], parts[2], parts[3]
    try:
        a, b = (int(x) for x in j1.split("-"))
        c, d = (int(x) for x in j2.split("-"))
    except ValueError:
        return None
    return a, b, c, d, strand


def load_rmats():
    evs = []
    with open(RMATS) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            inc = sum(int(x) for x in row["IJC_SAMPLE_1"].split(",")) \
                + sum(int(x) for x in row["IJC_SAMPLE_2"].split(","))
            skp = sum(int(x) for x in row["SJC_SAMPLE_1"].split(",")) \
                + sum(int(x) for x in row["SJC_SAMPLE_2"].split(","))
            evs.append(dict(
                ID=row["ID"], gene=row["geneSymbol"].strip('"'),
                chrom=row["chr"], strand=row["strand"],
                es=int(row["exonStart_0base"]), ee=int(row["exonEnd"]),
                up_ee=int(row["upstreamEE"]), down_es=int(row["downstreamES"]),
                dpsi=(float(row["IncLevelDifference"])
                      if row["IncLevelDifference"] not in ("", "NA") else float("nan")),
                support=inc + skp,
            ))
    return evs


def current_match(t, evs):
    """Replicates head_to_head_coverage.match_event for exon targets."""
    best, best_key = None, None
    for ev in evs:
        if ev["chrom"] != t["chrom"]:
            continue
        ds, de = abs(ev["es"] - t["exon_start"]), abs(ev["ee"] - t["exon_end"])
        if ds > TOL or de > TOL:
            continue
        key = (ds + de, ev["gene"].lower() != t["gene"].lower(), -ev["support"])
        if best_key is None or key < best_key:
            best_key, best = key, ev
    return best


def structural_match(t, evs):
    """Match on cassette exon AND both flanks AND strand (the full SE structure)."""
    a, b, c, d, strand = t["sup"]
    best, best_key = None, None
    for ev in evs:
        if ev["chrom"] != t["chrom"] or ev["strand"] != strand:
            continue
        # cassette: SUPPA b-c (1-based) vs rMATS es(0-based)/ee; flanks: a vs up_ee, d vs down_es
        dcas = abs(ev["es"] - b) + abs(ev["ee"] - c)
        dfl = abs(ev["up_ee"] - a) + abs(ev["down_es"] - d)
        if abs(ev["es"] - b) > TOL or abs(ev["ee"] - c) > TOL:
            continue
        if abs(ev["up_ee"] - a) > TOL or abs(ev["down_es"] - d) > TOL:
            continue
        key = (dcas + dfl, ev["gene"].lower() != t["gene"].lower(), -ev["support"])
        if best_key is None or key < best_key:
            best_key, best = key, ev
    return best


def main():
    if not (os.path.exists(VAL) and os.path.exists(RMATS)):
        print("data not present; cannot run diagnostic")
        return
    targets, no_suppa = [], 0
    with open(VAL) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row["exon_start"] = int(row["exon_start"])
            row["exon_end"] = int(row["exon_end"])
            row["sup"] = parse_suppa(row.get("suppa_event_id", ""))
            if row["sup"] is None:
                no_suppa += 1
                continue
            targets.append(row)
    evs = load_rmats()

    n = len(targets)
    print(f"targets without a parseable SUPPA SE id (skipped): {no_suppa}")
    col_vs_suppa_mismatch = 0
    changed, cur_only, struct_only = [], 0, 0
    for t in targets:
        a, b, c, d, strand = t["sup"]
        # (a) is the exon_start/end column even consistent with the SUPPA cassette?
        if abs(t["exon_start"] - b) > TOL or abs(t["exon_end"] - c) > TOL:
            col_vs_suppa_mismatch += 1
        cur = current_match(t, evs)
        st = structural_match(t, evs)
        if cur is None and st is None:
            continue
        if cur is None:
            struct_only += 1
        if st is None:
            cur_only += 1
        cid = cur["ID"] if cur else None
        sid = st["ID"] if st else None
        if cid != sid:
            changed.append((t["gene"], t["delta_psi_rtpcr"],
                            cur["dpsi"] if cur else None, st["dpsi"] if st else None,
                            cid, sid))

    print(f"positive targets: {n}")
    print(f"exon_start/end column INCONSISTENT with SUPPA cassette (>tol): {col_vs_suppa_mismatch}")
    print(f"current-only matched (struct finds none): {cur_only}")
    print(f"struct-only matched (current finds none): {struct_only}")
    print(f"targets whose matched rMATS event CHANGES under structure-aware: {len(changed)}")

    # (b) Decisive headline check: BRAID-style conformal coverage of RT-PCR truth,
    # SAME calibrated half-width q (fit on the CURRENT-matching residuals so current
    # ~= nominal by construction), applied to current vs structure-aware matching.
    # TRA2 is swap=True, so the RT-PCR-oriented ΔPSI centre = -IncLevelDifference.
    res_old, res_new = [], []
    for t in targets:
        cur, st = current_match(t, evs), structural_match(t, evs)
        if cur is None or st is None:
            continue
        rt = float(t["delta_psi_rtpcr"])
        res_old.append(abs(-cur["dpsi"] - rt))
        res_new.append(abs(-st["dpsi"] - rt))

    def conf_q(res, alpha=0.05):
        s = sorted(res)
        k = min(math.ceil((len(s) + 1) * (1 - alpha)), len(s))
        return s[k - 1]

    q = conf_q(res_old)
    cov_old = sum(r <= q for r in res_old) / len(res_old)
    cov_new = sum(r <= q for r in res_new) / len(res_new)
    print("\n(b) BRAID-style conformal coverage of RT-PCR truth on TRA2 positives "
          f"(same q={q:.3f} from current-matching residuals, n={len(res_old)}):")
    print(f"    current matching     : {cov_old:.3f}")
    print(f"    structure-aware match: {cov_new:.3f}   "
          f"({'HOLDS/IMPROVES' if cov_new >= cov_old else 'DROPS'})")
    print(f"    mean |residual| centre-vs-RTPCR : {sum(res_old)/len(res_old):.3f} "
          f"-> {sum(res_new)/len(res_new):.3f}")
    print("\ngene        RTPCR    cur_rMATSdpsi  struct_rMATSdpsi   cur_ID -> struct_ID")
    for g, rt, cd, sd, ci, si in sorted(changed):
        cds = f"{cd:+.3f}" if cd is not None else "  none"
        sds = f"{sd:+.3f}" if sd is not None else "  none"
        print(f"{g:11s} {float(rt):+.2f}   {cds:>10s}     {sds:>10s}     {ci} -> {si}")


if __name__ == "__main__":
    main()
