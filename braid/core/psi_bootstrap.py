"""PRISM: Post-assembly Resampling for Isoform Stability Measurement.

Bootstrap confidence intervals for PSI (Percent Spliced In) using
actual junction read counts from BAM.  Provides per-event PSI
confidence intervals from a single sample by Poisson resampling.
No biological replicates required.

Supports standard AS event types:
- SE (Skipped Exon / Exon Skipping)
- A5SS (Alternative 5' Splice Site)
- A3SS (Alternative 3' Splice Site)
- MXE (Mutually Exclusive Exons)
- RI (Retained Intron)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PSIResult:
    """PSI estimate with bootstrap confidence interval.

    Attributes:
        event_id: Event identifier.
        event_type: AS event type (SE, A5SS, A3SS, MXE, RI).
        gene: Gene name or ID.
        chrom: Chromosome.
        psi: Point estimate of PSI.
        ci_low: Lower bound of confidence interval.
        ci_high: Upper bound of confidence interval.
        cv: Coefficient of variation of PSI across replicates.
        inclusion_count: Junction reads supporting inclusion.
        exclusion_count: Junction reads supporting exclusion.
        is_confident: Whether CI width is below a threshold.
    """

    event_id: str
    event_type: str
    gene: str
    chrom: str
    psi: float
    ci_low: float
    ci_high: float
    cv: float
    inclusion_count: int
    exclusion_count: int
    is_confident: bool = False


def bootstrap_psi(
    inclusion_count: int,
    exclusion_count: int,
    n_replicates: int = 500,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float, float]:
    """Compute PSI with bootstrap confidence interval.

    Resamples inclusion and exclusion junction counts from Poisson
    distributions and computes PSI for each replicate.

    Args:
        inclusion_count: Junction reads supporting inclusion.
        exclusion_count: Junction reads supporting exclusion.
        n_replicates: Number of bootstrap replicates.
        confidence_level: Confidence level for interval.
        seed: Random seed.

    Returns:
        Tuple of (psi, ci_low, ci_high, cv).
    """
    total = inclusion_count + exclusion_count
    if total == 0:
        return 0.0, 0.0, 0.0, float("nan")

    psi = inclusion_count / total

    rng = np.random.default_rng(seed)

    # Poisson resample
    inc_samples = rng.poisson(max(inclusion_count, 0.5), size=n_replicates)
    exc_samples = rng.poisson(max(exclusion_count, 0.5), size=n_replicates)

    totals = inc_samples + exc_samples
    # Avoid division by zero
    valid = totals > 0
    psi_samples = np.zeros(n_replicates)
    psi_samples[valid] = inc_samples[valid] / totals[valid]

    alpha = 1.0 - confidence_level
    ci_low = float(np.percentile(psi_samples, 100 * alpha / 2))
    ci_high = float(np.percentile(psi_samples, 100 * (1 - alpha / 2)))

    mean_psi = float(np.mean(psi_samples))
    std_psi = float(np.std(psi_samples))
    cv = std_psi / mean_psi if mean_psi > 0 else float("nan")

    return psi, ci_low, ci_high, cv


def compute_psi_from_junctions(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
    n_replicates: int = 500,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> list[PSIResult]:
    """Compute PSI + CI for AS events using actual junction read counts.

    Extracts junction counts from the BAM, identifies alternative
    splice site events (junctions sharing a donor or acceptor),
    and computes PSI with bootstrap CI for each event.

    Args:
        bam_path: Path to indexed BAM file.
        chrom: Chromosome name.
        start: Region start coordinate.
        end: Region end coordinate.
        n_replicates: Bootstrap replicates.
        confidence_level: CI confidence level.
        seed: Random seed.

    Returns:
        List of PSI results for detected AS events.
    """
    import pysam

    # Extract junction read counts from BAM
    junction_counts: dict[tuple[int, int], int] = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom, start, end):
            if read.is_unmapped or read.cigartuples is None:
                continue
            pos = read.reference_start
            for op, length in read.cigartuples:
                if op == 3:  # N = intron
                    jstart, jend = pos, pos + length
                    if jstart >= start and jend <= end:
                        key = (jstart, jend)
                        junction_counts[key] = junction_counts.get(key, 0) + 1
                if op in (0, 2, 3, 7, 8):
                    pos += length

    if not junction_counts:
        return []

    # Group junctions by shared donor (A5SS) or shared acceptor (A3SS)
    results: list[PSIResult] = []
    rng = np.random.default_rng(seed)

    # A3SS: same donor, different acceptors
    donors: dict[int, list[tuple[int, int]]] = {}
    for jstart, jend in junction_counts:
        donors.setdefault(jstart, []).append((jstart, jend))

    for donor, juncs in donors.items():
        if len(juncs) < 2:
            continue
        # Sort by count descending
        juncs.sort(key=lambda j: junction_counts[j], reverse=True)
        total_reads = sum(junction_counts[j] for j in juncs)
        if total_reads < 5:
            continue

        for junc in juncs:
            count = junction_counts[junc]
            other_count = total_reads - count

            psi, ci_low, ci_high, cv = bootstrap_psi(
                count, other_count,
                n_replicates=n_replicates,
                confidence_level=confidence_level,
                seed=int(rng.integers(0, 2**31)),
            )
            ci_width = ci_high - ci_low
            event_id = f"A3SS:{junc[0]}-{junc[1]}"
            results.append(PSIResult(
                event_id=event_id,
                event_type="A3SS",
                gene="",
                chrom=chrom,
                psi=psi,
                ci_low=ci_low,
                ci_high=ci_high,
                cv=cv,
                inclusion_count=count,
                exclusion_count=other_count,
                is_confident=ci_width < 0.2,
            ))

    # A5SS: same acceptor, different donors
    acceptors: dict[int, list[tuple[int, int]]] = {}
    for jstart, jend in junction_counts:
        acceptors.setdefault(jend, []).append((jstart, jend))

    for acceptor, juncs in acceptors.items():
        if len(juncs) < 2:
            continue
        total_reads = sum(junction_counts[j] for j in juncs)
        if total_reads < 5:
            continue

        for junc in juncs:
            count = junction_counts[junc]
            other_count = total_reads - count

            psi, ci_low, ci_high, cv = bootstrap_psi(
                count, other_count,
                n_replicates=n_replicates,
                confidence_level=confidence_level,
                seed=int(rng.integers(0, 2**31)),
            )
            ci_width = ci_high - ci_low
            event_id = f"A5SS:{junc[0]}-{junc[1]}"
            results.append(PSIResult(
                event_id=event_id,
                event_type="A5SS",
                gene="",
                chrom=chrom,
                psi=psi,
                ci_low=ci_low,
                ci_high=ci_high,
                cv=cv,
                inclusion_count=count,
                exclusion_count=other_count,
                is_confident=ci_width < 0.2,
            ))

    # SE (Skipped Exon): junction d1→a2 (skip) exists AND
    # d1→a1 + d2→a2 exist where a1 < d2 (exon [a1,d2] is skipped)
    all_juncs = sorted(junction_counts.keys())
    se_seen: set[str] = set()
    for d1, a1 in all_juncs:
        for d2, a2 in all_juncs:
            if d2 <= a1 or d2 - a1 > 50000:
                continue  # exon must be between a1 and d2
            skip_key = (d1, a2)
            if skip_key not in junction_counts:
                continue
            se_key = f"{a1}-{d2}"
            if se_key in se_seen:
                continue
            se_seen.add(se_key)

            inc_count = min(
                junction_counts[(d1, a1)], junction_counts[(d2, a2)],
            )
            exc_count = junction_counts[skip_key]
            total = inc_count + exc_count
            if total < 5:
                continue

            psi, ci_low, ci_high, cv = bootstrap_psi(
                inc_count, exc_count,
                n_replicates=n_replicates,
                confidence_level=confidence_level,
                seed=int(rng.integers(0, 2**31)),
            )
            ci_width = ci_high - ci_low
            event_id = f"SE:{a1}-{d2}"
            results.append(PSIResult(
                event_id=event_id,
                event_type="SE",
                gene="",
                chrom=chrom,
                psi=psi,
                ci_low=ci_low,
                ci_high=ci_high,
                cv=cv,
                inclusion_count=inc_count,
                exclusion_count=exc_count,
                is_confident=ci_width < 0.2,
            ))

    # RI (Retained Intron): intron body has reads (no junction = retention)
    # Compare junction-spanning reads vs intron-body coverage
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for (jstart, jend), jcount in junction_counts.items():
            if jend - jstart > 50000:
                continue  # skip very long introns
            if jend - jstart < 50:
                continue  # skip very short
            # Count reads fully within the intron body (retained intron evidence)
            ri_count = 0
            for read in bam.fetch(chrom, jstart, jend):
                if read.is_unmapped:
                    continue
                # Read must span a significant portion of the intron
                if (read.reference_start <= jstart + 10
                        and read.reference_end >= jend - 10):
                    ri_count += 1
                elif (read.reference_start >= jstart
                      and read.reference_end <= jend
                      and read.reference_end - read.reference_start > 50):
                    ri_count += 1

            if ri_count < 3:
                continue
            total = jcount + ri_count
            if total < 5:
                continue

            psi_ri, ci_low, ci_high, cv = bootstrap_psi(
                ri_count, jcount,
                n_replicates=n_replicates,
                confidence_level=confidence_level,
                seed=int(rng.integers(0, 2**31)),
            )
            ci_width = ci_high - ci_low
            event_id = f"RI:{jstart}-{jend}"
            results.append(PSIResult(
                event_id=event_id,
                event_type="RI",
                gene="",
                chrom=chrom,
                psi=psi_ri,
                ci_low=ci_low,
                ci_high=ci_high,
                cv=cv,
                inclusion_count=ri_count,
                exclusion_count=jcount,
                is_confident=ci_width < 0.2,
            ))

    return results


def format_psi_report(results: list[PSIResult]) -> str:
    """Format PSI results as a text report."""
    lines: list[str] = []
    lines.append(f"{'Event':<30} {'PSI':>6} {'CI_low':>7} {'CI_high':>7} "
                 f"{'CV':>6} {'Inc':>5} {'Exc':>5} {'Conf':>5}")
    lines.append("-" * 80)
    for r in results:
        conf = "Y" if r.is_confident else "N"
        lines.append(
            f"{r.event_id:<30} {r.psi:>5.1%} {r.ci_low:>6.1%} "
            f"{r.ci_high:>6.1%} {r.cv:>6.2f} {r.inclusion_count:>5} "
            f"{r.exclusion_count:>5} {conf:>5}"
        )
    return "\n".join(lines)
