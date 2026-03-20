"""Bootstrap confidence intervals for StringTie assembled isoforms.

Takes StringTie output (GTF) and the original BAM, reconstructs the
splice graph for each gene locus, fits NNLS weights to the StringTie
isoform paths, and runs bootstrap resampling to provide per-isoform
confidence intervals.

This combines StringTie's proven assembly accuracy with statistical
uncertainty quantification that no existing assembler provides.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from rapidsplice.io.bam_reader import extract_junctions_from_bam
from rapidsplice.io.reference import ReferenceGenome

logger = logging.getLogger(__name__)


@dataclass
class STBootstrapConfig:
    """Configuration for StringTie bootstrap analysis."""

    stringtie_gtf: str
    bam_path: str
    reference_path: str | None = None
    n_replicates: int = 500
    confidence_level: float = 0.95
    min_junction_support: int = 2
    min_anchor_length: int = 8
    seed: int | None = None


@dataclass
class IsoformCI:
    """Per-isoform confidence interval result."""

    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    exons: list[tuple[int, int]]
    n_exons: int
    # Original StringTie values
    stringtie_cov: float
    stringtie_fpkm: float
    stringtie_tpm: float
    # Bootstrap results
    nnls_weight: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    presence_rate: float = 0.0
    cv: float = 0.0
    is_stable: bool = False


@dataclass
class GeneBootstrapResult:
    """Bootstrap result for one gene."""

    gene_id: str
    chrom: str
    n_isoforms: int
    n_junctions: int
    isoforms: list[IsoformCI] = field(default_factory=list)
    assembly_time: float = 0.0


def run_stringtie_bootstrap(
    config: STBootstrapConfig,
) -> list[GeneBootstrapResult]:
    """Run bootstrap CI analysis on StringTie output.

    Parses StringTie GTF, groups transcripts by gene, reconstructs
    junction evidence from the BAM for each gene locus, fits NNLS
    weights to the StringTie isoform structures, and runs bootstrap
    resampling.

    Args:
        config: Configuration.

    Returns:
        List of per-gene bootstrap results.
    """
    t0 = time.perf_counter()

    # Parse StringTie GTF
    genes = _parse_stringtie_gtf(config.stringtie_gtf)
    logger.info("Parsed %d genes from StringTie GTF", len(genes))

    reference = None
    if config.reference_path:
        reference = ReferenceGenome(config.reference_path)

    rng = np.random.default_rng(config.seed)
    results: list[GeneBootstrapResult] = []

    for gene_id, gene_data in genes.items():
        chrom = gene_data["chrom"]
        gene_start = gene_data["start"]
        gene_end = gene_data["end"]
        transcripts = gene_data["transcripts"]

        # Skip single-exon-only genes
        multi_exon = [t for t in transcripts if len(t["exons"]) >= 2]
        if not multi_exon:
            continue

        # Extract junctions for this gene region
        try:
            junctions, n_spliced, _ = extract_junctions_from_bam(
                config.bam_path,
                chrom,
                min_mapq=0,
                min_anchor_length=config.min_anchor_length,
                reference=reference,
                return_stats=True,
                region_start=gene_start,
                region_end=gene_end,
            )
        except Exception as exc:
            logger.debug("Junction extraction failed for %s: %s", gene_id, exc)
            continue

        n_junctions = len(junctions.starts)
        if n_junctions == 0:
            continue

        # Build junction-to-index mapping
        junc_map: dict[tuple[int, int], int] = {}
        junc_weights = np.zeros(n_junctions, dtype=np.float64)
        for i in range(n_junctions):
            key = (int(junctions.starts[i]), int(junctions.ends[i]))
            junc_map[key] = i
            junc_weights[i] = float(junctions.counts[i])

        # Build path-edge incidence matrix for StringTie isoforms
        # Each path is defined by its intron chain
        n_paths = len(multi_exon)
        A = np.zeros((n_junctions, n_paths), dtype=np.float64)
        valid_paths: list[int] = []

        for pi, tx in enumerate(multi_exon):
            exons = tx["exons"]
            introns = [
                (exons[i][1], exons[i + 1][0])
                for i in range(len(exons) - 1)
            ]
            path_valid = True
            for donor, acceptor in introns:
                jidx = junc_map.get((donor, acceptor))
                if jidx is not None:
                    A[jidx, pi] = 1.0
                else:
                    # Junction not found in BAM — check nearby
                    found = False
                    for tol in range(1, 6):
                        for d_off in range(-tol, tol + 1):
                            for a_off in range(-tol, tol + 1):
                                jidx2 = junc_map.get(
                                    (donor + d_off, acceptor + a_off),
                                )
                                if jidx2 is not None:
                                    A[jidx2, pi] = 1.0
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if not found:
                        path_valid = False
            if path_valid:
                valid_paths.append(pi)

        if not valid_paths:
            continue

        # Keep only valid paths
        A_valid = A[:, valid_paths]
        valid_txs = [multi_exon[i] for i in valid_paths]

        # NNLS fit
        try:
            weights, _ = nnls(A_valid, junc_weights)
        except Exception:
            continue

        # Bootstrap
        n_valid = len(valid_paths)
        weight_matrix = np.zeros(
            (config.n_replicates, n_valid), dtype=np.float64,
        )

        alpha = 1.0 - config.confidence_level
        for rep in range(config.n_replicates):
            b_resamp = rng.poisson(junc_weights).astype(np.float64)
            b_resamp = np.maximum(b_resamp, 0.1)
            try:
                w_rep, _ = nnls(A_valid, b_resamp)
                weight_matrix[rep, :] = w_rep
            except Exception:
                pass

        # Build results
        gene_result = GeneBootstrapResult(
            gene_id=gene_id,
            chrom=chrom,
            n_isoforms=n_valid,
            n_junctions=n_junctions,
        )

        for pi, tx in enumerate(valid_txs):
            col = weight_matrix[:, pi]
            presence = float(np.sum(col > 0) / config.n_replicates)
            mean_w = float(np.mean(col))
            ci_low = float(np.percentile(col, 100 * alpha / 2))
            ci_high = float(np.percentile(col, 100 * (1 - alpha / 2)))
            std_w = float(np.std(col))
            cv = std_w / mean_w if mean_w > 0 else float("nan")

            iso = IsoformCI(
                transcript_id=tx["transcript_id"],
                gene_id=gene_id,
                chrom=chrom,
                strand=gene_data.get("strand", "."),
                exons=tx["exons"],
                n_exons=len(tx["exons"]),
                stringtie_cov=tx.get("cov", 0.0),
                stringtie_fpkm=tx.get("fpkm", 0.0),
                stringtie_tpm=tx.get("tpm", 0.0),
                nnls_weight=float(weights[pi]),
                ci_low=ci_low,
                ci_high=ci_high,
                presence_rate=presence,
                cv=cv,
                is_stable=presence >= 0.5,
            )
            gene_result.isoforms.append(iso)

        results.append(gene_result)

    if reference:
        reference.close()

    elapsed = time.perf_counter() - t0
    total_isoforms = sum(r.n_isoforms for r in results)
    n_stable = sum(
        sum(1 for i in r.isoforms if i.is_stable) for r in results
    )
    logger.info(
        "StringTie bootstrap: %d genes, %d isoforms (%d stable), %.1fs",
        len(results), total_isoforms, n_stable, elapsed,
    )

    return results


def _parse_stringtie_gtf(
    gtf_path: str,
) -> dict[str, dict]:
    """Parse StringTie GTF output into gene-grouped transcripts."""
    genes: dict[str, dict] = {}

    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue

            chrom = fields[0]
            feat_type = fields[2]
            start = int(fields[3]) - 1  # 1-based → 0-based
            end = int(fields[4])
            strand = fields[6]
            attrs = fields[8]

            gene_id = _parse_attr(attrs, "gene_id")
            tid = _parse_attr(attrs, "transcript_id")

            if not gene_id:
                continue

            if gene_id not in genes:
                genes[gene_id] = {
                    "chrom": chrom,
                    "strand": strand,
                    "start": start,
                    "end": end,
                    "transcripts": {},
                }

            gene = genes[gene_id]
            gene["start"] = min(gene["start"], start)
            gene["end"] = max(gene["end"], end)

            if feat_type == "transcript" and tid:
                cov = _parse_attr_float(attrs, "cov")
                fpkm = _parse_attr_float(attrs, "FPKM")
                tpm = _parse_attr_float(attrs, "TPM")
                gene["transcripts"][tid] = {
                    "transcript_id": tid,
                    "exons": [],
                    "cov": cov,
                    "fpkm": fpkm,
                    "tpm": tpm,
                }
            elif feat_type == "exon" and tid:
                if tid in gene.get("transcripts", {}):
                    gene["transcripts"][tid]["exons"].append((start, end))

    # Sort exons and convert transcripts dict to list
    for gene in genes.values():
        tx_list = []
        for tx in gene["transcripts"].values():
            tx["exons"].sort(key=lambda e: e[0])
            tx_list.append(tx)
        gene["transcripts"] = tx_list

    return genes


def _parse_attr(attrs: str, key: str) -> str | None:
    """Extract string attribute from GTF attributes."""
    for part in attrs.split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(None, 1)
        if len(tokens) == 2 and tokens[0] == key:
            return tokens[1].strip('"')
    return None


def _parse_attr_float(attrs: str, key: str) -> float:
    """Extract float attribute from GTF attributes."""
    val = _parse_attr(attrs, key)
    if val is not None:
        try:
            return float(val)
        except ValueError:
            pass
    return 0.0


def format_bootstrap_report(
    results: list[GeneBootstrapResult],
    top_n: int = 20,
) -> str:
    """Format a text report of bootstrap results."""
    lines: list[str] = []
    lines.append(f"{'='*75}")
    lines.append("  StringTie + Bootstrap CI Report")
    lines.append(f"{'='*75}")

    total_genes = len(results)
    total_iso = sum(r.n_isoforms for r in results)
    total_stable = sum(
        sum(1 for i in r.isoforms if i.is_stable) for r in results
    )

    lines.append(f"  Genes analyzed:    {total_genes}")
    lines.append(f"  Total isoforms:    {total_iso}")
    lines.append(f"  Stable (pres≥50%): {total_stable}")
    lines.append(
        f"  Unstable:          {total_iso - total_stable}"
    )
    lines.append("")

    # Top isoforms by CV (most uncertain)
    all_isoforms = []
    for r in results:
        all_isoforms.extend(r.isoforms)

    lines.append(
        f"  {'Transcript':<25} {'Gene':<15} {'Exons':>5} "
        f"{'Weight':>7} {'CI_low':>7} {'CI_high':>7} "
        f"{'Pres':>6} {'CV':>6}"
    )
    lines.append(
        f"  {'-'*25} {'-'*15} {'-'*5} {'-'*7} {'-'*7} "
        f"{'-'*7} {'-'*6} {'-'*6}"
    )

    # Sort by weight descending
    sorted_iso = sorted(all_isoforms, key=lambda x: x.nnls_weight, reverse=True)
    for iso in sorted_iso[:top_n]:
        lines.append(
            f"  {iso.transcript_id:<25} {iso.gene_id:<15} "
            f"{iso.n_exons:>5} {iso.nnls_weight:>7.1f} "
            f"{iso.ci_low:>7.1f} {iso.ci_high:>7.1f} "
            f"{iso.presence_rate:>5.0%} {iso.cv:>6.2f}"
        )

    lines.append(f"\n{'='*75}")
    return "\n".join(lines)
