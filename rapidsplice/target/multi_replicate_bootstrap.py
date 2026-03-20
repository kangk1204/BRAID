"""Multi-replicate bootstrap CI for isoforms and AS events.

Combines within-replicate sampling variance (Poisson bootstrap)
with between-replicate biological variance to produce properly
calibrated confidence intervals.

This is the recommended mode when biological replicates are
available, which is the standard in RNA-seq experiments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pysam

from rapidsplice.target.psi_bootstrap import (
    CONFIDENT_CI_WIDTH_THRESHOLD,
    CONFIDENT_CV_THRESHOLD,
    OVERDISPERSED_COUNT_SCALE,
)

logger = logging.getLogger(__name__)

DEFAULT_MIN_MAPQ = 1


@dataclass
class MultiRepPSIResult:
    """PSI estimate from multiple biological replicates.

    Attributes:
        event_id: Event identifier.
        event_type: AS event type.
        chrom: Chromosome.
        mean_psi: Mean PSI across replicates.
        bio_std: Standard deviation across replicates (biological).
        sampling_std: Mean within-replicate bootstrap std (sampling).
        total_std: Combined std (sqrt of bio² + sampling²).
        ci_low: Lower bound of combined CI.
        ci_high: Upper bound of combined CI.
        cv: Coefficient of variation (total_std / mean_psi).
        n_replicates: Number of replicates.
        per_rep_psi: PSI per replicate.
        per_rep_ci: CI per replicate [(low, high), ...].
        is_confident: Whether combined CI width < threshold.
    """

    event_id: str = ""
    event_type: str = ""
    chrom: str = ""
    mean_psi: float = 0.0
    bio_std: float = 0.0
    sampling_std: float = 0.0
    total_std: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    cv: float = 0.0
    n_replicates: int = 0
    per_rep_psi: list[float] = field(default_factory=list)
    per_rep_ci: list[tuple[float, float]] = field(default_factory=list)
    is_confident: bool = False
    inclusion_counts: list[int] = field(default_factory=list)
    exclusion_counts: list[int] = field(default_factory=list)


def extract_junction_counts_from_bam(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
    min_mapq: int = DEFAULT_MIN_MAPQ,
) -> dict[tuple[int, int], int]:
    """Extract junction read counts from a BAM region."""
    counts: dict[tuple[int, int], int] = {}
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(chrom, start, end):
            if read.is_unmapped or read.cigartuples is None:
                continue
            if read.is_secondary or read.is_supplementary:
                continue
            if read.is_duplicate or read.is_qcfail:
                continue
            if read.mapping_quality < min_mapq:
                continue
            pos = read.reference_start
            for op, length in read.cigartuples:
                if op == 3:  # N = intron
                    jstart, jend = pos, pos + length
                    if jstart >= start and jend <= end:
                        key = (jstart, jend)
                        counts[key] = counts.get(key, 0) + 1
                if op in (0, 2, 3, 7, 8):
                    pos += length
    return counts


def multi_replicate_psi(
    bam_paths: list[str],
    chrom: str,
    start: int,
    end: int,
    n_bootstrap: int = 200,
    confidence_level: float = 0.95,
    min_total_count: int = 5,
    min_mapq: int = DEFAULT_MIN_MAPQ,
    seed: int | None = None,
) -> list[MultiRepPSIResult]:
    """Compute PSI with multi-replicate + bootstrap CI.

    For each AS event detected across replicates:
    1. Extract junction counts per replicate
    2. Compute PSI per replicate
    3. Bootstrap within each replicate (sampling variance)
    4. Compute between-replicate variance (biological)
    5. Combine into total CI

    Args:
        bam_paths: List of BAM file paths (one per replicate).
        chrom: Chromosome.
        start: Region start.
        end: Region end.
        n_bootstrap: Bootstrap replicates per sample.
        confidence_level: CI confidence level.
        min_total_count: Minimum junction reads across replicates.
        min_mapq: Minimum alignment MAPQ to count.
        seed: Random seed.

    Returns:
        List of multi-replicate PSI results.
    """
    rng = np.random.default_rng(seed)
    n_reps = len(bam_paths)
    alpha = 1.0 - confidence_level

    # Step 1: Extract junction counts from each replicate
    rep_junctions: list[dict[tuple[int, int], int]] = []
    for bam_path in bam_paths:
        jc = extract_junction_counts_from_bam(
            bam_path,
            chrom,
            start,
            end,
            min_mapq=min_mapq,
        )
        rep_junctions.append(jc)

    # Step 2: Find all junctions across replicates
    all_junctions: set[tuple[int, int]] = set()
    for jc in rep_junctions:
        all_junctions.update(jc.keys())

    if not all_junctions:
        return []

    # Step 3: Identify A3SS events (same donor, different acceptors)
    results: list[MultiRepPSIResult] = []

    donors: dict[int, set[int]] = {}
    for jstart, jend in all_junctions:
        donors.setdefault(jstart, set()).add(jend)

    for donor, acceptors in donors.items():
        if len(acceptors) < 2:
            continue

        for acc in sorted(acceptors):
            junc = (donor, acc)
            other_juncs = [(donor, a) for a in acceptors if a != acc]

            # Per-replicate PSI
            rep_psis = []
            rep_ci_lows = []
            rep_ci_highs = []
            rep_sampling_vars = []
            inc_counts = []
            exc_counts = []

            for ri, jc in enumerate(rep_junctions):
                inc = jc.get(junc, 0)
                exc = sum(jc.get(oj, 0) for oj in other_juncs)
                total = inc + exc

                inc_counts.append(inc)
                exc_counts.append(exc)

                if total < 2:
                    continue

                psi = inc / total

                # Posterior predictive uncertainty for this replicate.
                boot_psis = rng.beta(
                    inc * OVERDISPERSED_COUNT_SCALE + 0.5,
                    exc * OVERDISPERSED_COUNT_SCALE + 0.5,
                    size=n_bootstrap,
                )

                rep_psis.append(psi)
                ci_lo = float(np.percentile(boot_psis, 100 * alpha / 2))
                ci_hi = float(np.percentile(boot_psis, 100 * (1 - alpha / 2)))
                rep_ci_lows.append(ci_lo)
                rep_ci_highs.append(ci_hi)
                rep_sampling_vars.append(float(np.var(boot_psis)))

            if len(rep_psis) < 2:
                continue

            # Total counts across replicates
            total_inc = sum(inc_counts)
            total_exc = sum(exc_counts)
            if total_inc + total_exc < min_total_count:
                continue

            # Biological variance (between replicates)
            bio_var = float(np.var(rep_psis, ddof=1))
            bio_std = float(np.sqrt(bio_var))

            # Mean sampling variance (within replicates)
            sampling_var = float(np.mean(rep_sampling_vars))
            sampling_std = float(np.sqrt(sampling_var))

            # Combined
            total_var = bio_var + sampling_var
            total_std = float(np.sqrt(total_var))

            mean_psi = float(np.mean(rep_psis))
            z = 1.96 if confidence_level == 0.95 else 2.576

            combined_lo = max(0.0, mean_psi - z * total_std)
            combined_hi = min(1.0, mean_psi + z * total_std)
            ci_width = combined_hi - combined_lo

            cv = total_std / mean_psi if mean_psi > 0 else float("nan")

            is_confident = (
                ci_width < CONFIDENT_CI_WIDTH_THRESHOLD
                and np.isfinite(cv)
                and cv <= CONFIDENT_CV_THRESHOLD
            )

            results.append(MultiRepPSIResult(
                event_id=f"A3SS:{donor}-{acc}",
                event_type="A3SS",
                chrom=chrom,
                mean_psi=mean_psi,
                bio_std=bio_std,
                sampling_std=sampling_std,
                total_std=total_std,
                ci_low=combined_lo,
                ci_high=combined_hi,
                cv=cv,
                n_replicates=len(rep_psis),
                per_rep_psi=rep_psis,
                per_rep_ci=list(zip(rep_ci_lows, rep_ci_highs)),
                is_confident=is_confident,
                inclusion_counts=inc_counts,
                exclusion_counts=exc_counts,
            ))

    # A5SS events (same acceptor, different donors)
    acceptors_map: dict[int, set[int]] = {}
    for jstart, jend in all_junctions:
        acceptors_map.setdefault(jend, set()).add(jstart)

    for acc, donor_set in acceptors_map.items():
        if len(donor_set) < 2:
            continue

        for don in sorted(donor_set):
            junc = (don, acc)
            other_juncs = [(d, acc) for d in donor_set if d != don]

            rep_psis = []
            rep_ci_lows = []
            rep_ci_highs = []
            rep_sampling_vars = []
            inc_counts = []
            exc_counts = []

            for ri, jc in enumerate(rep_junctions):
                inc = jc.get(junc, 0)
                exc = sum(jc.get(oj, 0) for oj in other_juncs)
                total = inc + exc
                inc_counts.append(inc)
                exc_counts.append(exc)

                if total < 2:
                    continue

                psi = inc / total
                boot_psis = rng.beta(
                    inc * OVERDISPERSED_COUNT_SCALE + 0.5,
                    exc * OVERDISPERSED_COUNT_SCALE + 0.5,
                    size=n_bootstrap,
                )

                rep_psis.append(psi)
                ci_lo = float(np.percentile(boot_psis, 100 * alpha / 2))
                ci_hi = float(np.percentile(boot_psis, 100 * (1 - alpha / 2)))
                rep_ci_lows.append(ci_lo)
                rep_ci_highs.append(ci_hi)
                rep_sampling_vars.append(float(np.var(boot_psis)))

            if len(rep_psis) < 2:
                continue

            total_inc = sum(inc_counts)
            total_exc = sum(exc_counts)
            if total_inc + total_exc < min_total_count:
                continue

            bio_var = float(np.var(rep_psis, ddof=1))
            sampling_var = float(np.mean(rep_sampling_vars))
            total_var = bio_var + sampling_var
            total_std = float(np.sqrt(total_var))
            mean_psi = float(np.mean(rep_psis))

            z = 1.96
            combined_lo = max(0.0, mean_psi - z * total_std)
            combined_hi = min(1.0, mean_psi + z * total_std)
            ci_width = combined_hi - combined_lo

            cv = total_std / mean_psi if mean_psi > 0 else float("nan")

            is_confident = (
                ci_width < CONFIDENT_CI_WIDTH_THRESHOLD
                and np.isfinite(cv)
                and cv <= CONFIDENT_CV_THRESHOLD
            )

            results.append(MultiRepPSIResult(
                event_id=f"A5SS:{don}-{acc}",
                event_type="A5SS",
                chrom=chrom,
                mean_psi=mean_psi,
                bio_std=float(np.sqrt(bio_var)),
                sampling_std=float(np.sqrt(sampling_var)),
                total_std=total_std,
                ci_low=combined_lo,
                ci_high=combined_hi,
                cv=cv,
                n_replicates=len(rep_psis),
                per_rep_psi=rep_psis,
                per_rep_ci=list(zip(rep_ci_lows, rep_ci_highs)),
                is_confident=is_confident,
                inclusion_counts=inc_counts,
                exclusion_counts=exc_counts,
            ))

    logger.info(
        "Multi-replicate PSI: %d events from %d replicates, "
        "%d confident",
        len(results), n_reps,
        sum(1 for r in results if r.is_confident),
    )

    return results


def multi_replicate_isoform_bootstrap(
    stringtie_gtf: str,
    bam_paths: list[str],
    reference_path: str | None = None,
    n_bootstrap: int = 200,
    seed: int | None = None,
) -> list[dict]:
    """Isoform-level bootstrap with multi-replicate support."""
    from rapidsplice.target.stringtie_bootstrap import (
        STBootstrapConfig,
        run_stringtie_bootstrap,
    )

    all_rep_results = []
    for i, bam in enumerate(bam_paths):
        config = STBootstrapConfig(
            stringtie_gtf=stringtie_gtf,
            bam_path=bam,
            reference_path=reference_path,
            n_replicates=n_bootstrap,
            seed=(seed + i) if seed else None,
        )
        rep_result = run_stringtie_bootstrap(config)
        all_rep_results.append(rep_result)

    return combine_isoform_bootstrap_results(all_rep_results)


def combine_isoform_bootstrap_results(
    all_rep_results: list[list[object]],
) -> list[dict]:
    """Combine precomputed StringTie bootstrap results across replicates."""
    # Match genes by gene_id
    gene_ids = set()
    for rep in all_rep_results:
        for gr in rep:
            gene_ids.add(gr.gene_id)

    combined = []
    for gid in gene_ids:
        rep_gene_results = []
        for rep in all_rep_results:
            gr = next((g for g in rep if g.gene_id == gid), None)
            if gr and gr.isoforms:
                rep_gene_results.append(gr)

        if len(rep_gene_results) < 2:
            continue

        # Match isoforms by transcript_id
        all_tids = set()
        for gr in rep_gene_results:
            for iso in gr.isoforms:
                all_tids.add(iso.transcript_id)

        for tid in all_tids:
            rep_weights = []
            rep_sampling_stds = []

            for gr in rep_gene_results:
                iso = next(
                    (i for i in gr.isoforms if i.transcript_id == tid),
                    None,
                )
                if iso:
                    rep_weights.append(iso.nnls_weight)
                    if np.isfinite(iso.cv):
                        rep_sampling_stds.append(iso.cv * iso.nnls_weight)
                    else:
                        # Fall back to the bootstrap CI width when CV is
                        # undefined (usually mean weight == 0).
                        approx_std = (iso.ci_high - iso.ci_low) / (2 * 1.96)
                        if np.isfinite(approx_std):
                            rep_sampling_stds.append(float(max(0.0, approx_std)))

            if len(rep_weights) < 2:
                continue

            bio_std = float(np.std(rep_weights, ddof=1))
            mean_weight = float(np.mean(rep_weights))
            sampling_std = (
                float(np.mean(rep_sampling_stds))
                if rep_sampling_stds
                else 0.0
            )

            total_std = float(np.sqrt(bio_std**2 + sampling_std**2))
            total_cv = total_std / mean_weight if mean_weight > 0 else float("nan")

            combined.append({
                "gene_id": gid,
                "transcript_id": tid,
                "mean_weight": mean_weight,
                "bio_std": bio_std,
                "sampling_std": sampling_std,
                "total_std": total_std,
                "total_cv": total_cv,
                "ci_low": max(0, mean_weight - 1.96 * total_std),
                "ci_high": mean_weight + 1.96 * total_std,
                "n_replicates": len(rep_weights),
                "per_rep_weights": rep_weights,
            })

    logger.info(
        "Multi-replicate isoform bootstrap: %d isoforms from %d replicates",
        len(combined), len(all_rep_results),
    )

    return combined


def format_multi_rep_report(results: list[MultiRepPSIResult]) -> str:
    """Format multi-replicate PSI results."""
    lines: list[str] = []
    lines.append(
        f"{'Event':<30} {'PSI':>6} {'CI':>15} "
        f"{'Bio σ':>6} {'Samp σ':>7} {'Total σ':>7} {'Reps':>5}"
    )
    lines.append("-" * 80)
    for r in sorted(results, key=lambda x: -x.mean_psi):
        lines.append(
            f"{r.event_id:<30} {r.mean_psi:>5.1%} "
            f"[{r.ci_low:.1%},{r.ci_high:.1%}] "
            f"{r.bio_std:>6.3f} {r.sampling_std:>7.3f} "
            f"{r.total_std:>7.3f} {r.n_replicates:>5}"
        )
    return "\n".join(lines)
