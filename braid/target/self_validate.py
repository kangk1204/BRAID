"""Self-validation of assembled isoforms via de novo assembly.

Validates reference-guided isoforms by checking whether they can be
independently reconstructed from reads without reference bias.

Pipeline:
1. Extract reads from target region
2. De novo assemble (de Bruijn graph, no reference)
3. Map reference-guided isoform sequences against de novo contigs
4. Isoforms supported by de novo contigs are independently validated

This provides a reference-free validation signal complementary to
bootstrap confidence intervals.
"""

from __future__ import annotations

import logging
import os
import tempfile

import pysam

logger = logging.getLogger(__name__)


def self_validate_isoforms(
    bam_path: str,
    reference_path: str,
    chrom: str,
    start: int,
    end: int,
    isoform_exons: list[list[tuple[int, int]]],
    k: int = 21,
    min_kmer_count: int = 2,
    min_match_fraction: float = 0.8,
) -> list[dict]:
    """Validate assembled isoforms via de novo assembly.

    For each reference-guided isoform, extracts its spliced sequence
    from the reference, then checks whether this sequence (or major
    portions of it) can be found in de novo assembled contigs.

    Args:
        bam_path: Path to indexed BAM file.
        reference_path: Path to reference genome FASTA.
        chrom: Chromosome name.
        start: Region start coordinate.
        end: Region end coordinate.
        isoform_exons: List of exon coordinate lists for each isoform.
        k: K-mer size for de novo assembly.
        min_kmer_count: Minimum k-mer count.
        min_match_fraction: Minimum fraction of isoform k-mers found
            in de novo contigs to count as validated.

    Returns:
        List of validation dicts, one per isoform, with keys:
        - validated: bool
        - match_fraction: float (fraction of k-mers found in de novo)
        - n_denovo_contigs: int
        - best_contig_coverage: float
    """
    # Step 1: Extract isoform sequences from reference
    isoform_seqs: list[str] = []
    try:
        with pysam.FastaFile(reference_path) as fasta:
            for exons in isoform_exons:
                seq_parts = []
                for estart, eend in exons:
                    seq_parts.append(fasta.fetch(chrom, estart, eend))
                isoform_seqs.append("".join(seq_parts).upper())
    except Exception as exc:
        logger.warning("Failed to extract isoform sequences: %s", exc)
        return [{"validated": False, "match_fraction": 0.0,
                 "n_denovo_contigs": 0, "best_contig_coverage": 0.0}
                for _ in isoform_exons]

    # Step 2: De novo assembly of reads from the region
    denovo_contigs = _run_denovo_for_validation(
        bam_path, chrom, start, end, k, min_kmer_count,
    )

    if not denovo_contigs:
        logger.info("No de novo contigs assembled")
        return [{"validated": False, "match_fraction": 0.0,
                 "n_denovo_contigs": 0, "best_contig_coverage": 0.0}
                for _ in isoform_exons]

    # Step 3: Build k-mer index from de novo contigs
    kmer_k = min(k, 31)  # Use same k as assembly
    contig_kmers: set[str] = set()
    for contig_seq in denovo_contigs:
        seq = contig_seq.upper()
        for i in range(len(seq) - kmer_k + 1):
            contig_kmers.add(seq[i:i + kmer_k])

    logger.info(
        "De novo: %d contigs, %d unique %d-mers",
        len(denovo_contigs), len(contig_kmers), kmer_k,
    )

    # Step 4: Validate each isoform
    results: list[dict] = []
    for iso_seq in isoform_seqs:
        if len(iso_seq) < kmer_k:
            results.append({
                "validated": False,
                "match_fraction": 0.0,
                "n_denovo_contigs": len(denovo_contigs),
                "best_contig_coverage": 0.0,
            })
            continue

        # Count how many isoform k-mers are in de novo contigs
        total_kmers = len(iso_seq) - kmer_k + 1
        matched_kmers = 0
        for i in range(total_kmers):
            kmer = iso_seq[i:i + kmer_k]
            if kmer in contig_kmers:
                matched_kmers += 1

        match_frac = matched_kmers / total_kmers if total_kmers > 0 else 0.0

        # Also check best single-contig coverage
        best_coverage = 0.0
        for contig_seq in denovo_contigs:
            cseq = contig_seq.upper()
            # Check how much of the isoform sequence is covered by this contig
            covered = 0
            for i in range(0, len(iso_seq) - kmer_k + 1, kmer_k):
                kmer = iso_seq[i:i + kmer_k]
                if kmer in cseq:
                    covered += 1
            total_checks = max(1, (len(iso_seq) - kmer_k + 1) // kmer_k)
            cov = covered / total_checks
            best_coverage = max(best_coverage, cov)

        validated = match_frac >= min_match_fraction
        results.append({
            "validated": validated,
            "match_fraction": match_frac,
            "n_denovo_contigs": len(denovo_contigs),
            "best_contig_coverage": best_coverage,
        })

        logger.debug(
            "Isoform %d/%d: %.1f%% k-mer match, validated=%s",
            len(results), len(isoform_exons),
            match_frac * 100, validated,
        )

    n_validated = sum(1 for r in results if r["validated"])
    logger.info(
        "Self-validation: %d/%d isoforms validated (%.0f%% k-mer threshold)",
        n_validated, len(results), min_match_fraction * 100,
    )

    return results


def _run_denovo_for_validation(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
    k: int,
    min_kmer_count: int,
) -> list[str]:
    """Run de novo assembly and return contig sequences."""
    from braid.denovo.pipeline import DeNovoConfig, run_denovo_assembly

    with tempfile.TemporaryDirectory(prefix="ts_selfval_") as tmpdir:
        # Extract reads as FASTQ
        fq_path = os.path.join(tmpdir, "reads.fq")
        n_reads = 0
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            with open(fq_path, "w") as fq:
                for read in bam.fetch(chrom, start, end):
                    if read.is_unmapped or read.is_secondary:
                        continue
                    seq = read.query_sequence
                    if not seq or len(seq) < k:
                        continue
                    qual = read.qual or ("I" * len(seq))
                    fq.write(f"@{read.query_name}\n{seq}\n+\n{qual}\n")
                    n_reads += 1

        if n_reads < 20:
            return []

        contigs_path = os.path.join(tmpdir, "contigs.fa")
        config = DeNovoConfig(
            fastq_paths=[fq_path],
            output_path=contigs_path,
            k=k,
            min_kmer_count=min_kmer_count,
            canonical_kmers=False,
            min_transcript_length=50,
            min_transcript_coverage=1.0,
        )

        try:
            transcripts, _ = run_denovo_assembly(config)
        except Exception as exc:
            logger.warning("De novo assembly failed: %s", exc)
            return []

        # Read contig sequences
        seqs: list[str] = []
        if os.path.exists(contigs_path):
            with open(contigs_path) as f:
                current: list[str] = []
                for line in f:
                    if line.startswith(">"):
                        if current:
                            seqs.append("".join(current))
                        current = []
                    else:
                        current.append(line.strip())
                if current:
                    seqs.append("".join(current))

        return seqs
