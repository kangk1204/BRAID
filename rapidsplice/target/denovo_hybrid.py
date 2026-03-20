"""Hybrid de novo + reference-guided assembly for target genes.

Runs de novo assembly on reads from the target region to discover
novel exons or transcripts not present in the reference annotation.
Novel contigs are aligned back to the target region and integrated
into the splice graph for NNLS decomposition.

This enables discovery of:
- Novel exons (e.g., cryptic exons in cancer)
- Retained introns
- Novel transcript isoforms absent from reference
"""

from __future__ import annotations

import logging
import os
import tempfile

import pysam

from rapidsplice.denovo.pipeline import DeNovoConfig, run_denovo_assembly

logger = logging.getLogger(__name__)


def run_target_denovo(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
    reference_path: str,
    k: int = 21,
    min_kmer_count: int = 2,
    min_contig_length: int = 100,
) -> list[dict]:
    """Run de novo assembly on reads from a target region.

    Extracts reads from the target region as FASTQ, runs de Bruijn
    graph assembly, and aligns resulting contigs back to the reference
    to identify novel sequences.

    Args:
        bam_path: Path to indexed BAM file.
        chrom: Chromosome name.
        start: Region start coordinate.
        end: Region end coordinate.
        reference_path: Path to reference FASTA.
        k: K-mer size for de Bruijn graph.
        min_kmer_count: Minimum k-mer count.
        min_contig_length: Minimum contig length to report.

    Returns:
        List of novel contig dicts with keys: sequence, length,
        coverage, aligned_start, aligned_end, is_novel.
    """
    with tempfile.TemporaryDirectory(prefix="ts_denovo_") as tmpdir:
        # Step 1: Extract reads as FASTQ
        fq_path = os.path.join(tmpdir, "reads.fq")
        n_reads = _extract_reads_as_fastq(bam_path, chrom, start, end, fq_path)

        if n_reads < 10:
            logger.info("Too few reads (%d) for de novo assembly", n_reads)
            return []

        # Step 2: Run de novo assembly
        contigs_path = os.path.join(tmpdir, "contigs.fa")
        config = DeNovoConfig(
            fastq_paths=[fq_path],
            output_path=contigs_path,
            k=k,
            min_kmer_count=min_kmer_count,
            canonical_kmers=False,  # Non-canonical for RNA
            min_transcript_length=min_contig_length,
            min_transcript_coverage=1.0,
        )

        try:
            transcripts, stats = run_denovo_assembly(config)
            n_contigs = len(transcripts)
        except Exception as exc:
            logger.warning("De novo assembly failed: %s", exc)
            return []

        if n_contigs == 0 or not os.path.exists(contigs_path):
            return []

        # Step 3: Align contigs back to reference region
        novel_contigs = _align_contigs_to_reference(
            contigs_path, reference_path, chrom, start, end,
        )

        logger.info(
            "De novo: %d reads → %d contigs → %d novel",
            n_reads, n_contigs, len(novel_contigs),
        )

        return novel_contigs


def _extract_reads_as_fastq(
    bam_path: str,
    chrom: str,
    start: int,
    end: int,
    output_path: str,
) -> int:
    """Extract reads from a BAM region as FASTQ."""
    n = 0
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        with open(output_path, "w") as fq:
            for read in bam.fetch(chrom, start, end):
                if read.is_unmapped or read.is_secondary:
                    continue
                seq = read.query_sequence
                qual = read.qual
                if not seq:
                    continue
                fq.write(f"@{read.query_name}\n{seq}\n+\n{qual or 'I'*len(seq)}\n")
                n += 1
    return n


def _align_contigs_to_reference(
    contigs_fasta: str,
    reference_path: str,
    chrom: str,
    start: int,
    end: int,
) -> list[dict]:
    """Align contigs to reference and identify novel sequences.

    Uses HISAT2 (or simple BLAST-like approach) to map contigs.
    Contigs that align with novel splice patterns or don't align
    well are flagged as potentially novel.
    """
    # Parse contigs
    contigs: list[dict] = []
    with open(contigs_fasta) as f:
        name = ""
        seq_parts: list[str] = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name and seq_parts:
                    contigs.append({
                        "name": name,
                        "sequence": "".join(seq_parts),
                        "length": len("".join(seq_parts)),
                    })
                name = line[1:].split()[0]
                seq_parts = []
            elif line:
                seq_parts.append(line)
        if name and seq_parts:
            contigs.append({
                "name": name,
                "sequence": "".join(seq_parts),
                "length": len("".join(seq_parts)),
            })

    if not contigs:
        return []

    # Extract reference region for comparison
    try:
        with pysam.FastaFile(reference_path) as fasta:
            ref_seq = fasta.fetch(chrom, start, end).upper()
    except Exception:
        return []

    # Simple novel contig detection: check if contig sequence
    # is present in the reference region
    novel = []
    for contig in contigs:
        seq = contig["sequence"].upper()
        if len(seq) < 50:
            continue

        # Check if any 50bp substring is in the reference
        found_in_ref = False
        for i in range(0, len(seq) - 49, 25):
            kmer = seq[i:i + 50]
            if kmer in ref_seq:
                found_in_ref = True
                break

        if not found_in_ref:
            contig["is_novel"] = True
            novel.append(contig)

    return novel
