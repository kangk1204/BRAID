#!/usr/bin/env python3
"""POC: does BRAID's flow decomposition beat junction-pair PSI on COMPLEX loci?

The diagnosis (ALGORITHM_HEADROOM.md) established that for a simple cassette exon
PSI is junction-defined and flow decomposition is mathematically identical to the
junction ratio — zero headroom. The ONE structural place the splice graph can win
is a **multi-junction-context exon**: an exon that is included via more than one
downstream junction (and skipped via more than one), so a single inclusion/skip
junction pair — the unit a predefined-event caller scores — structurally mis-counts
the inclusion fraction. BRAID sums ALL flow through the exon, so it should recover
the true PSI where the single-pair estimator is biased.

We build random 4-isoform loci with that structure (exon E2 reachable through two
downstream exons E3/E4, and skipped through both), set edge weights / node coverages
to the true (abundance-proportional) values, run ``decompose_graph``, and compare:
  - flow_psi   : sum of decomposed transcript weight through E2 / total
  - naive_psi  : one inclusion junction / (that inclusion + one skip junction)
against the known true PSI. Honest: this is a constructed structural demonstration
(no real ground truth exists in-repo for complex events), not a real-data head-to-head.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from braid.flow.decompose import DecomposeConfig, decompose_graph  # noqa: E402
from braid.graph.splice_graph import EdgeType, NodeType, SpliceGraph  # noqa: E402


def build_locus(t1, t2, t3, t4, noise_rng=None):
    """4-isoform locus; E2 included via E3 (T1) and via E4 (T2), skipped via E3 (T3)
    and via E4 (T4). Returns (csr, true_psi_E2, naive_psi, node_ids).

    Coordinates (constitutive E1 -> [cassette E2] -> alt E3/E4):
      E1 100-200 ; E2 300-400 ; E3 500-600 ; E4 700-800
    With ``noise_rng`` the OBSERVED junction counts / exon coverages are Poisson
    draws around the abundance-proportional expectation (finite-read sampling); the
    graph and naive PSI use those noisy observations, while true_psi uses the true
    abundances — exactly the real situation a caller faces.
    """
    def obs(x):
        return float(noise_rng.poisson(max(x, 0.0))) if noise_rng is not None else float(x)

    j_e1e2 = obs(t1 + t2)   # E1->E2 inclusion
    j_e1e3 = obs(t3)        # E1->E3 skip
    j_e1e4 = obs(t4)        # E1->E4 skip
    j_e2e3 = obs(t1)        # E2->E3
    j_e2e4 = obs(t2)        # E2->E4
    c_e1 = obs(t1 + t2 + t3 + t4)
    c_e2 = obs(t1 + t2)
    c_e3 = obs(t1 + t3)
    c_e4 = obs(t2 + t4)

    g = SpliceGraph(chrom="chr1", strand="+", locus_start=100, locus_end=900)
    src = g.add_node(start=100, end=100, node_type=NodeType.SOURCE, coverage=0)
    e1 = g.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=c_e1)
    e2 = g.add_node(start=300, end=400, node_type=NodeType.EXON, coverage=c_e2)
    e3 = g.add_node(start=500, end=600, node_type=NodeType.EXON, coverage=c_e3)
    e4 = g.add_node(start=700, end=800, node_type=NodeType.EXON, coverage=c_e4)
    snk = g.add_node(start=900, end=900, node_type=NodeType.SINK, coverage=0)

    g.add_edge(src, e1, EdgeType.SOURCE_LINK, weight=c_e1, coverage=c_e1)
    g.add_edge(e1, e2, EdgeType.INTRON, weight=j_e1e2, coverage=j_e1e2)
    g.add_edge(e1, e3, EdgeType.INTRON, weight=j_e1e3, coverage=j_e1e3)
    g.add_edge(e1, e4, EdgeType.INTRON, weight=j_e1e4, coverage=j_e1e4)
    g.add_edge(e2, e3, EdgeType.INTRON, weight=j_e2e3, coverage=j_e2e3)
    g.add_edge(e2, e4, EdgeType.INTRON, weight=j_e2e4, coverage=j_e2e4)
    g.add_edge(e3, snk, EdgeType.SINK_LINK, weight=j_e2e3 + j_e1e3, coverage=j_e2e3 + j_e1e3)
    g.add_edge(e4, snk, EdgeType.SINK_LINK, weight=j_e2e4 + j_e1e4, coverage=j_e2e4 + j_e1e4)

    total = t1 + t2 + t3 + t4
    true_psi = (t1 + t2) / total if total else 0.0
    # naive single-pair PSI as a predefined-event caller scores it: one inclusion
    # junction (E2->E3) over (that inclusion + one skip junction E1->E3), observed.
    naive_psi = j_e2e3 / (j_e2e3 + j_e1e3) if (j_e2e3 + j_e1e3) > 0 else 0.0
    return g.to_csr(), true_psi, naive_psi, {"e2": e2}


def flow_psi(csr, node_ids):
    cfg = DecomposeConfig(min_transcript_coverage=0.5, min_relative_abundance=0.0,
                          min_relative_isoform_weight=0.0)
    transcripts = decompose_graph(csr, cfg)
    e2 = node_ids["e2"]
    total = sum(t.weight for t in transcripts)
    incl = sum(t.weight for t in transcripts if e2 in t.node_ids)
    return (incl / total) if total > 0 else 0.0


def _run(n, noisy, seed=0):
    rng = np.random.default_rng(seed)
    nrng = np.random.default_rng(seed + 1) if noisy else None
    flow_err, naive_err = [], []
    for _ in range(n):
        depth = rng.integers(60, 600)        # total reads at the locus
        w = rng.dirichlet([1.0, 1.0, 1.0, 1.0]) * depth
        t1, t2, t3, t4 = (float(x) for x in np.round(w))
        if min(t1, t2, t3, t4) < 1:
            continue
        csr, true_psi, naive_psi, nid = build_locus(t1, t2, t3, t4, noise_rng=nrng)
        flow_err.append(abs(flow_psi(csr, nid) - true_psi))
        naive_err.append(abs(naive_psi - true_psi))
    return np.array(flow_err), np.array(naive_err)


def main():
    print(f"\n{'='*72}")
    print("COMPLEX-LOCUS POC — exon in 2 inclusion + 2 skip junction contexts")
    print("(random 4-isoform loci, PSI truth known; constructed structural test)")
    for noisy, tag in [(False, "noiseless (expected counts)"),
                       (True, "Poisson read noise (finite depth 60-600)")]:
        fe, ne = _run(220, noisy)
        win = float(np.mean(fe < ne - 1e-6))
        print(f"\n--- {tag}: n={fe.size} ---")
        print(f"{'estimator':<28}{'mean|PSIerr|':>14}{'median':>9}{'P90':>8}")
        print(f"{'naive junction-pair':<28}{ne.mean():>14.3f}{np.median(ne):>9.3f}"
              f"{np.percentile(ne,90):>8.3f}")
        print(f"{'BRAID flow decomposition':<28}{fe.mean():>14.3f}{np.median(fe):>9.3f}"
              f"{np.percentile(fe,90):>8.3f}")
        print(f"flow better on {win:.0%} of loci; "
              f"mean error reduction {1 - fe.mean()/max(ne.mean(),1e-9):.0%}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
