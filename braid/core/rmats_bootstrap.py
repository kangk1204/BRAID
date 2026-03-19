"""Add BRAID bootstrap CI to rMATS-detected AS events.

Takes rMATS output (SE/A3SS/A5SS/MXE/RI junction count files) and
a BAM file, then computes per-event bootstrap PSI confidence intervals
using Poisson resampling of the junction read counts.

This combines rMATS's proven event detection with BRAID's bootstrap
CI — the best of both tools.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from braid.core.psi_bootstrap import PSIResult, bootstrap_psi

logger = logging.getLogger(__name__)


@dataclass
class RmatsEvent:
    """An rMATS-detected AS event with junction counts."""

    event_id: str
    event_type: str
    chrom: str
    strand: str
    gene: str
    # Junction counts from rMATS
    inc_count: int  # Inclusion junction count
    exc_count: int  # Exclusion (skip) junction count
    # rMATS statistics
    rmats_psi: float
    rmats_fdr: float
    rmats_dpsi: float
    # Coordinates
    exon_start: int = 0
    exon_end: int = 0


def parse_rmats_output(
    rmats_dir: str,
    event_types: list[str] | None = None,
    min_total_count: int = 10,
) -> list[RmatsEvent]:
    """Parse rMATS junction count output files.

    Args:
        rmats_dir: Directory containing rMATS *.MATS.JunctionCountOnly.txt
        event_types: Event types to parse (default: all)
        min_total_count: Minimum total junction count to include

    Returns:
        List of parsed events with junction counts.
    """
    if event_types is None:
        event_types = ["SE", "A3SS", "A5SS", "MXE", "RI"]

    events: list[RmatsEvent] = []

    for et in event_types:
        fname = os.path.join(rmats_dir, f"{et}.MATS.JunctionCountOnly.txt")
        if not os.path.exists(fname):
            continue

        with open(fname) as f:
            header = f.readline().strip().split("\t")

            # Column indices
            cols = {h: i for i, h in enumerate(header)}

            for line in f:
                fields = line.strip().split("\t")
                if len(fields) < len(header):
                    continue

                try:
                    chrom = fields[cols.get("chr", 3)].replace("chr", "")
                    strand = fields[cols.get("strand", 4)]
                    gene = fields[cols.get("GeneID", 1)]

                    # Junction counts (sample 1)
                    inc_str = fields[cols.get("IJC_SAMPLE_1", -1)]
                    exc_str = fields[cols.get("SJC_SAMPLE_1", -1)]
                    inc = sum(int(x) for x in inc_str.split(",") if x)
                    exc = sum(int(x) for x in exc_str.split(",") if x)

                    if inc + exc < min_total_count:
                        continue

                    # PSI
                    psi_str = fields[cols.get("IncLevel1", -1)]
                    psi_vals = [
                        float(x) for x in psi_str.split(",")
                        if x and x != "NA"
                    ]
                    psi = float(np.mean(psi_vals)) if psi_vals else 0.5

                    # FDR and dPSI
                    fdr = float(fields[cols.get("FDR", -1)])
                    dpsi = float(
                        fields[cols.get("IncLevelDifference", -1)]
                    )

                    # Exon coordinates
                    es = int(fields[cols.get("exonStart_0base", 5)])
                    ee = int(fields[cols.get("exonEnd", 6)])

                    eid = f"{et}:{chrom}:{es}-{ee}"

                    events.append(RmatsEvent(
                        event_id=eid,
                        event_type=et,
                        chrom=chrom,
                        strand=strand,
                        gene=gene,
                        inc_count=inc,
                        exc_count=exc,
                        rmats_psi=psi,
                        rmats_fdr=fdr,
                        rmats_dpsi=dpsi,
                        exon_start=es,
                        exon_end=ee,
                    ))
                except (ValueError, IndexError, KeyError):
                    continue

    logger.info(
        "Parsed %d rMATS events from %s", len(events), rmats_dir,
    )
    return events


def add_bootstrap_ci(
    events: list[RmatsEvent],
    n_replicates: int = 500,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> list[PSIResult]:
    """Add BRAID bootstrap CI to rMATS events.

    For each rMATS event, uses the inclusion/exclusion junction
    counts directly (from rMATS output) and applies Poisson
    bootstrap to compute CI.

    Args:
        events: Parsed rMATS events.
        n_replicates: Bootstrap replicates.
        confidence_level: CI confidence level.
        seed: Random seed.

    Returns:
        List of PSI results with bootstrap CI.
    """
    rng = np.random.default_rng(seed)
    results: list[PSIResult] = []

    for ev in events:
        psi, ci_low, ci_high, cv = bootstrap_psi(
            ev.inc_count,
            ev.exc_count,
            n_replicates=n_replicates,
            confidence_level=confidence_level,
            seed=int(rng.integers(0, 2**31)),
        )

        ci_width = ci_high - ci_low
        results.append(PSIResult(
            event_id=ev.event_id,
            event_type=ev.event_type,
            gene=ev.gene,
            chrom=ev.chrom,
            psi=psi,
            ci_low=ci_low,
            ci_high=ci_high,
            cv=cv,
            inclusion_count=ev.inc_count,
            exclusion_count=ev.exc_count,
            is_confident=ci_width < 0.2,
        ))

    n_conf = sum(1 for r in results if r.is_confident)
    logger.info(
        "Bootstrap CI added to %d events (%d confident)",
        len(results), n_conf,
    )
    return results
