"""SG-NEx long-read ΔPSI validation: event-type Mondrian on the headline estimand.

Builds an independent ΔPSI truth surface (the headline estimand) so the Stage-1
event-type conditional-coverage result can be tested on ΔPSI, not just PSI:

  - SHORT read (BRAID point + support): rMATS on SG-NEx Illumina BAMs, HepG2 (b1)
    vs K562 (b2); BRAID ΔPSI = Jeffreys posterior-mean difference, ctrl - treat.
  - LONG read (independent truth): pool SG-NEx ONT genome BAMs per condition and
    count inclusion/exclusion junction reads at each rMATS event; truth ΔPSI =
    PSI_longread(HepG2) - PSI_longread(K562).

Then run the same leakage-free Stage-0/Stage-1 conditional-coverage analysis by
event type (SE / A3SS / A5SS) as conditional_coverage_diagnostic.py.

Stages (idempotent; --stage prep|rmats|lrpsi|surface|analyze|all):
  prep    verify BAMs + indexes + GTF present (downloaded by sgnex/download.sh)
  rmats   run rMATS turbo (Illumina HepG2 vs K562) -> SE/A3SS/A5SS.MATS.JC.txt
  lrpsi   long-read PSI per rMATS event per condition (pysam junction counts)
  surface combine -> benchmarks/results/sgnex_dpsi_surface.json
  analyze leakage-free conditional coverage by event type -> sgnex_conditional_eval.json

Usage: python benchmarks/sgnex_dpsi_validation.py --stage all
"""

from __future__ import annotations

# ruff: noqa: I001
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pysam

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
for _p in (os.path.join(_HERE, "headtohead"), _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SGNEX = os.path.join(_REPO, "data/public_benchmarks/sgnex")
GTF = os.path.join(SGNEX, "ref/ensembl.GRCh38.110.gtf")  # Ensembl naming matches SG-NEx BAMs
RMATS_OUT = os.path.join(SGNEX, "rmats_out")
RMATS_DIR = "/home/keunsoo/mambaforge/envs/rmats/rMATS"      # holds rmatspipeline.*.so (cp312)
RMATS_BIN = os.path.join(RMATS_DIR, "rmats.py")
RMATS_PY = "/home/keunsoo/mambaforge/envs/rmats/bin/python"  # 3.12 env interpreter for the .so
SURFACE = os.path.join(_REPO, "benchmarks/results/sgnex_dpsi_surface.json")
EVAL = os.path.join(_REPO, "benchmarks/results/sgnex_conditional_eval.json")
COND1, COND2 = "HepG2", "K562"   # ctrl (b1) vs treat (b2); ΔPSI = ctrl - treat
TOL = 5                          # junction match tolerance (bp)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _bams(kind: str, cond: str) -> list[str]:
    d = os.path.join(SGNEX, kind)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.endswith(".bam") and f"_{cond}_" in f)


def read_introns(read: pysam.AlignedSegment) -> list[tuple[int, int]]:
    """Genomic (donor, acceptor) for every N (ref-skip) in the read's CIGAR."""
    pos = read.reference_start
    out = []
    for op, ln in read.cigartuples or []:
        if op in (0, 2, 7, 8):     # M D = X consume reference
            pos += ln
        elif op == 3:              # N intron
            out.append((pos, pos + ln))
            pos += ln
    return out


def _match(introns: list[tuple[int, int]], j: tuple[int, int]) -> bool:
    return any(abs(a - j[0]) <= TOL and abs(b - j[1]) <= TOL for a, b in introns)


def _norm_chrom(bam: pysam.AlignmentFile, chrom: str) -> str | None:
    """Map a chrom to this BAM's convention (rMATS emits 'chr1'; SG-NEx BAMs use '1')."""
    refs = bam.references
    if chrom in refs:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in refs:
        return chrom[3:]
    if ("chr" + chrom) in refs:
        return "chr" + chrom
    return None


def count_psi(bams: list[pysam.AlignmentFile], chrom: str, region: tuple[int, int],
              inc_juncs: list[tuple[int, int]], exc_junc: tuple[int, int]) -> tuple[int, int]:
    """Pooled long-read inclusion/exclusion read counts at one event."""
    inc = exc = 0
    lo, hi = region
    for bam in bams:
        c = _norm_chrom(bam, chrom)
        if c is None:
            continue
        try:
            it = bam.fetch(c, max(0, lo), hi)
        except ValueError:
            continue
        for r in it:
            if r.is_secondary or r.is_supplementary or r.is_unmapped:
                continue
            ints = read_introns(r)
            if not ints:
                continue
            has_exc = _match(ints, exc_junc)
            has_inc = any(_match(ints, j) for j in inc_juncs)
            if has_exc and not has_inc:
                exc += 1
            elif has_inc and not has_exc:
                inc += 1
    return inc, exc


# --- rMATS event -> junction definitions (genomic, 0-based half-open) -------
def se_junctions(r: dict) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    uee, es, ee, ds = (int(r["upstreamEE"]), int(r["exonStart_0base"]),
                       int(r["exonEnd"]), int(r["downstreamES"]))
    inc = [(uee, es), (ee, ds)]            # spliced-in exon: two flanking introns
    exc = (uee, ds)                        # skipping junction
    return inc, exc


def alt_junctions(r: dict) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    """A3SS/A5SS: long (inclusion) vs short (exclusion) isoform differ by one junction."""
    les, lee = int(r["longExonStart_0base"]), int(r["longExonEnd"])
    ses, see = int(r["shortES"]), int(r["shortEE"])
    fes, fee = int(r["flankingES"]), int(r["flankingEE"])
    # the flanking exon joins either upstream or downstream of the alt region
    if fes >= lee:                         # flanking is downstream
        inc = [(lee, fes)]
        exc = (see, fes)
    else:                                  # flanking is upstream
        inc = [(fee, les)]
        exc = (fee, ses)
    return inc, exc


def jeffreys_dpsi(r: dict) -> tuple[float, float]:
    """BRAID ΔPSI point (Jeffreys posterior mean, ctrl - treat) and raw support."""
    def grp(ijc: str, sjc: str, il: float, sl: float) -> tuple[float, float]:
        i = sum(float(x) for x in ijc.split(",") if x not in ("", "NA"))
        s = sum(float(x) for x in sjc.split(",") if x not in ("", "NA"))
        s_eff = s * (il / sl) if sl > 0 else s
        return (i + 0.5) / (i + s_eff + 1.0), i + s
    il, sl = float(r["IncFormLen"]), float(r["SkipFormLen"])
    p1, n1 = grp(r["IJC_SAMPLE_1"], r["SJC_SAMPLE_1"], il, sl)
    p2, n2 = grp(r["IJC_SAMPLE_2"], r["SJC_SAMPLE_2"], il, sl)
    return p1 - p2, n1 + n2


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def stage_prep() -> None:
    miss = []
    for cond in (COND1, COND2):
        if not _bams("illumina", cond):
            miss.append(f"illumina/{cond}")
        if not _bams("ont", cond):
            miss.append(f"ont/{cond}")
    if not os.path.exists(GTF):
        miss.append("ref/gencode.v44.gtf")
    print("prep:", "MISSING " + ", ".join(miss) if miss else "all inputs present")
    if miss:
        raise SystemExit("prep incomplete (download still running?)")


def stage_rmats(read_length: int = 150) -> None:
    os.makedirs(RMATS_OUT, exist_ok=True)
    b1 = os.path.join(SGNEX, "b1_hepg2.txt")
    b2 = os.path.join(SGNEX, "b2_k562.txt")
    open(b1, "w").write(",".join(_bams("illumina", COND1)))
    open(b2, "w").write(",".join(_bams("illumina", COND2)))
    cmd = [RMATS_PY, RMATS_BIN, "--b1", b1, "--b2", b2, "--gtf", GTF, "-t", "paired",
           "--readLength", str(read_length), "--variable-read-length", "--nthread", "8",
           "--od", RMATS_OUT, "--tmp", os.path.join(RMATS_OUT, "tmp"), "--task", "both"]
    print("rmats:", " ".join(cmd))
    env = {**os.environ, "PYTHONPATH": RMATS_DIR}  # make rmatspipeline.so importable
    subprocess.run(cmd, check=True, env=env)


def stage_lrpsi() -> dict:
    ont = {c: [pysam.AlignmentFile(b, "rb") for b in _bams("ont", c)] for c in (COND1, COND2)}
    specs = {"SE": se_junctions, "A3SS": alt_junctions, "A5SS": alt_junctions}
    out: dict[str, list[dict]] = {}
    for et, junc_fn in specs.items():
        path = os.path.join(RMATS_OUT, f"{et}.MATS.JC.txt")
        if not os.path.exists(path):
            continue
        import csv
        rows = list(csv.DictReader(open(path), delimiter="\t"))
        recs = []
        for r in rows:
            inc_j, exc_j = junc_fn(r)
            chrom = r["chr"]
            region = (min(min(j) for j in inc_j + [exc_j]) - 10,
                      max(max(j) for j in inc_j + [exc_j]) + 10)
            psi = {}
            ok = True
            for c in (COND1, COND2):
                inc, exc = count_psi(ont[c], chrom, region, inc_j, exc_j)
                if inc + exc < 10:        # require long-read depth for a truth value
                    ok = False
                    break
                psi[c] = inc / (inc + exc)
            if not ok:
                continue
            dpsi_point, support = jeffreys_dpsi(r)
            anchor = r["exonStart_0base"] if et == "SE" else r["longExonStart_0base"]
            recs.append({
                "event_id": f"{et}:{chrom}:{anchor}",
                "event_type": et, "gene": r.get("geneSymbol", ""),
                "braid_dpsi": dpsi_point, "lr_dpsi": psi[COND1] - psi[COND2],
                "support": support, "rmats_inclevel_diff": float(r["IncLevelDifference"]),
            })
        out[et] = recs
        print(f"lrpsi {et}: {len(recs)} events with long-read depth >= 10")
    return out


def stage_validate(n_per_type: int = 400, seed: int = 0) -> dict:
    """QC: long-read junction PSI vs rMATS IncLevel on a RANDOM sample of depth>=10
    events (representative of the analysis set, not a high-coverage tail). Confirms the
    junction-counting logic and is written to a committed artifact so the manuscript's
    validation correlations are reproducible.
    """
    import csv
    ont = [pysam.AlignmentFile(b, "rb") for b in _bams("ont", COND1)]
    rng = np.random.default_rng(seed)
    specs = {"SE": se_junctions, "A3SS": alt_junctions, "A5SS": alt_junctions}
    out: dict[str, dict] = {}
    for et, junc_fn in specs.items():
        path = os.path.join(RMATS_OUT, f"{et}.MATS.JC.txt")
        if not os.path.exists(path):
            continue
        rows = [r for r in csv.DictReader(open(path), delimiter="\t")
                if r["IncLevel1"] not in ("", "NA")]
        lr, sr = [], []
        for i in rng.permutation(len(rows)):
            if len(lr) >= n_per_type:
                break
            r = rows[i]
            inc_j, exc_j = junc_fn(r)
            region = (min(min(j) for j in inc_j + [exc_j]) - 10,
                      max(max(j) for j in inc_j + [exc_j]) + 10)
            inc, exc = count_psi(ont, r["chr"], region, inc_j, exc_j)
            if inc + exc < 10:
                continue
            lr.append(inc / (inc + exc))
            sr.append(float(np.mean([float(x) for x in r["IncLevel1"].split(",")
                                     if x not in ("", "NA")])))
        rr = float(np.corrcoef(lr, sr)[0, 1]) if len(lr) > 2 else float("nan")
        out[et] = {"n": len(lr), "pearson_r": rr}
        print(f"validate {et}: n={len(lr)}  long-read-PSI vs rMATS IncLevel1 r={rr:.3f}")
    res = {"condition": COND1, "n_per_type": n_per_type, "seed": seed, "by_type": out}
    json.dump(res, open(os.path.join(_REPO, "benchmarks/results/sgnex_validation.json"), "w"),
              indent=2)
    return res


def stage_surface() -> dict:
    data = stage_lrpsi()
    flat = [r for recs in data.values() for r in recs]
    # orientation sanity vs rMATS IncLevelDifference
    bp = np.array([r["braid_dpsi"] for r in flat])
    rl = np.array([r["rmats_inclevel_diff"] for r in flat])
    corr = float(np.corrcoef(bp, rl)[0, 1]) if len(bp) > 2 else float("nan")
    surface = {"cond_ctrl": COND1, "cond_treat": COND2, "n": len(flat),
               "braid_vs_rmats_corr": corr,
               "by_type": {k: len(v) for k, v in data.items()}, "events": flat}
    os.makedirs(os.path.dirname(SURFACE), exist_ok=True)
    json.dump(surface, open(SURFACE, "w"), indent=2)
    print(f"surface: {len(flat)} events  braid/rmats corr {corr:.3f}  "
          f"by_type { {k: len(v) for k, v in data.items()} }")
    return surface


def stage_analyze() -> dict:
    from braid.target.conformal import assign_support_bins  # noqa: E402
    from comprehensive_benchmark import wilson  # noqa: E402
    from head_to_head_coverage import (  # noqa: E402
        conformal_crossfit, conformal_crossfit_grouped, interval_score)

    s = json.load(open(SURFACE))
    ev = s["events"]
    pts = np.array([e["braid_dpsi"] for e in ev])
    tru = np.array([e["lr_dpsi"] for e in ev])
    sup = np.array([e["support"] for e in ev])
    et = np.array([e["event_type"] for e in ev])
    if pts.std() > 0 and np.corrcoef(pts, tru)[0, 1] < 0:
        pts = -pts
    n = pts.size
    sb = assign_support_bins(sup).astype(str)
    comp = np.array([f"{e}|{b}" for e, b in zip(et, sb)])  # event-type x support composite
    bands = {
        "constant_support": conformal_crossfit(pts, tru, np.ones(n), sup, 0.05),
        "adaptive_event_type": conformal_crossfit_grouped(
            pts, tru, np.ones(n), et, 0.05, clip=(-1.0, 1.0)),
        "adaptive_composite": conformal_crossfit_grouped(
            pts, tru, np.ones(n), comp, 0.05, min_bin=20, clip=(-1.0, 1.0)),
    }
    res = {"n": int(n), "alpha": 0.05, "by_type": s["by_type"],
           "braid_vs_rmats_corr": s.get("braid_vs_rmats_corr"),
           "median_abs_residual": float(np.median(np.abs(tru - pts)))}
    for name, (lo, hi) in bands.items():
        cov = (tru >= lo) & (tru <= hi)
        isc = float(np.mean([interval_score(y, a, b, 0.05) for y, a, b in zip(tru, lo, hi)]))
        bins = {}
        for t in sorted(set(et.tolist())):
            m = et == t
            k, nn = int(cov[m].sum()), int(m.sum())
            _, wl, wh = wilson(k, nn)
            bins[t] = {"n": nn, "coverage": k / nn if nn else float("nan"),
                       "wilson": [wl, wh], "mean_width": float((hi - lo)[m].mean()),
                       "excludes_0.95": bool(wl > 0.95 or wh < 0.95)}
        gaps = [abs(b["coverage"] - 0.95) for b in bins.values() if b["n"] >= 10]
        res[name] = {"by_type": bins, "max_abs_gap": max(gaps) if gaps else None,
                     "pooled_interval_score": isc, "mean_width": float((hi - lo).mean()),
                     "n_types_excluding_0.95": sum(b["excludes_0.95"] for b in bins.values())}
    json.dump(res, open(EVAL, "w"), indent=2)
    for name in bands:
        r = res[name]
        covs = " ".join(f"{r['by_type'][t]['coverage']:.3f}" for t in ("SE", "A3SS", "A5SS"))
        print(f"{name:<22} max_gap={r['max_abs_gap']:.4f} iscore={r['pooled_interval_score']:.4f} "
              f"meanW={r['mean_width']:.4f}  SE/A3SS/A5SS={covs}  "
              f"CI-excl-0.95={r['n_types_excluding_0.95']}")
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="all",
                   choices=["prep", "rmats", "lrpsi", "validate", "surface", "analyze", "all"])
    p.add_argument("--read-length", type=int, default=150)
    a = p.parse_args(argv)
    if a.stage in ("prep", "all"):
        stage_prep()
    if a.stage in ("rmats", "all"):
        stage_rmats(a.read_length)
    if a.stage in ("validate", "all"):
        stage_validate()
    if a.stage in ("surface", "all"):
        stage_surface()
    elif a.stage == "lrpsi":
        stage_lrpsi()
    if a.stage in ("analyze", "all"):
        stage_analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
