"""Benchmark runner for RapidSplice transcript assembler.

Downloads or generates synthetic RNA-seq data, runs RapidSplice and baseline tools
(StringTie, Scallop2) on the same input, and evaluates assembly quality using
GFFcompare-style metrics (transcript/exon/intron sensitivity and precision).
Runtime and peak memory usage are measured for each tool.

Usage:
    python benchmarks/run_benchmark.py --n-genes 100 --n-reads 1000000
    python benchmarks/run_benchmark.py --output-dir my_results --threads 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pysam

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark run.

    Attributes:
        output_dir: Directory where benchmark results, synthetic data, and
            intermediate files are written.
        n_genes: Number of synthetic genes to generate.
        n_reads: Total number of simulated reads.
        read_length: Simulated read length in base pairs.
        run_stringtie: Whether to run StringTie as a baseline.
        run_scallop: Whether to run Scallop2 as a baseline.
        threads: Number of threads for tools that support multi-threading.
    """

    output_dir: str = "benchmark_results"
    n_genes: int = 100
    n_reads: int = 1_000_000
    read_length: int = 100
    run_stringtie: bool = True
    run_scallop: bool = True
    threads: int = 4


@dataclass
class EvaluationMetrics:
    """GFFcompare-style evaluation metrics for a single tool.

    All values are fractions in [0, 1].

    Attributes:
        transcript_sensitivity: Fraction of true transcripts recovered (exact
            intron-chain match).
        transcript_precision: Fraction of predicted transcripts matching a true
            transcript (exact intron-chain match).
        exon_sensitivity: Fraction of true exons that overlap a predicted exon
            by at least 50% reciprocal.
        exon_precision: Fraction of predicted exons that overlap a true exon
            by at least 50% reciprocal.
        intron_sensitivity: Fraction of true introns (junction donor-acceptor
            pairs) recovered exactly.
        intron_precision: Fraction of predicted introns matching a true intron
            exactly.
    """

    transcript_sensitivity: float = 0.0
    transcript_precision: float = 0.0
    exon_sensitivity: float = 0.0
    exon_precision: float = 0.0
    intron_sensitivity: float = 0.0
    intron_precision: float = 0.0


# ---------------------------------------------------------------------------
# Gene / Transcript data structures for synthetic data generation
# ---------------------------------------------------------------------------


@dataclass
class _ExonDef:
    """Internal definition of one exon in a synthetic transcript."""

    start: int
    end: int


@dataclass
class _TranscriptDef:
    """Internal definition of a synthetic transcript."""

    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    exons: list[_ExonDef] = field(default_factory=list)
    abundance: float = 1.0


@dataclass
class _GeneDef:
    """Internal definition of a synthetic gene locus."""

    gene_id: str
    chrom: str
    strand: str
    start: int
    end: int
    transcripts: list[_TranscriptDef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------


class SyntheticDataGenerator:
    """Generates synthetic RNA-seq data with known ground-truth transcripts.

    Creates:
    - A FASTA reference genome with one or more chromosomes containing gene regions.
    - A truth GTF file describing known transcript structures.
    - A BAM file with simulated reads placed according to the known transcripts.

    The generator creates multi-exon genes with variable numbers of alternative
    isoforms to test the assembler's ability to recover complex transcript
    structures.

    Args:
        config: Benchmark configuration controlling data size parameters.
    """

    # Exon and intron length ranges (base pairs)
    _MIN_EXON_LEN: int = 80
    _MAX_EXON_LEN: int = 600
    _MIN_INTRON_LEN: int = 200
    _MAX_INTRON_LEN: int = 5000
    _INTERGENIC_GAP: int = 10_000
    _CHROM_PAD: int = 5000

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config: BenchmarkConfig = config
        self._rng: random.Random = random.Random(42)
        self._np_rng: np.random.Generator = np.random.default_rng(42)
        self._genes: list[_GeneDef] = []
        self._chrom_name: str = "chr1"
        self._chrom_length: int = 0

    def generate(self) -> tuple[str, str, str]:
        """Generate synthetic data and return paths to the output files.

        Returns:
            A 3-tuple ``(bam_path, reference_path, truth_gtf_path)`` pointing
            to the generated files inside the configured output directory.
        """
        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        data_dir = out_dir / "synthetic_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Generating synthetic data: %d genes, %d reads, read length %d",
            self._config.n_genes,
            self._config.n_reads,
            self._config.read_length,
        )

        # Step 1: Design gene structures
        self._design_genes()

        # Step 2: Write reference FASTA
        ref_path = str(data_dir / "reference.fa")
        self._write_reference(ref_path)

        # Step 3: Write truth GTF
        truth_gtf_path = str(data_dir / "truth.gtf")
        self._write_truth_gtf(truth_gtf_path)

        # Step 4: Simulate reads and write BAM
        bam_path = str(data_dir / "simulated.bam")
        self._simulate_reads(bam_path, ref_path)

        logger.info(
            "Synthetic data generated: BAM=%s, REF=%s, GTF=%s",
            bam_path, ref_path, truth_gtf_path,
        )

        return bam_path, ref_path, truth_gtf_path

    # ------------------------------------------------------------------
    # Gene design
    # ------------------------------------------------------------------

    def _design_genes(self) -> None:
        """Design synthetic gene structures with multiple alternative isoforms.

        Each gene has 2-8 exons and 1-4 alternative transcript isoforms
        created by exon skipping. Gene positions are laid out sequentially
        on a single chromosome with intergenic gaps.
        """
        self._genes = []
        current_pos: int = self._CHROM_PAD

        for gene_idx in range(self._config.n_genes):
            gene_id = f"GENE{gene_idx + 1:06d}"
            strand = self._rng.choice(["+", "-"])

            # Number of exons for this gene
            n_exons = self._rng.randint(2, 8)

            # Generate exon boundaries
            exons: list[_ExonDef] = []
            pos = current_pos
            for _ in range(n_exons):
                exon_len = self._rng.randint(self._MIN_EXON_LEN, self._MAX_EXON_LEN)
                exons.append(_ExonDef(start=pos, end=pos + exon_len))
                pos += exon_len
                # Add intron gap (except after the last exon)
                if len(exons) < n_exons:
                    intron_len = self._rng.randint(
                        self._MIN_INTRON_LEN, self._MAX_INTRON_LEN
                    )
                    pos += intron_len

            gene_start = exons[0].start
            gene_end = exons[-1].end

            # Create alternative isoforms by exon skipping
            n_isoforms = min(self._rng.randint(1, 4), 2 ** (n_exons - 2))
            transcripts: list[_TranscriptDef] = []

            # Full-length transcript is always included
            full_tx = _TranscriptDef(
                transcript_id=f"{gene_id}.1",
                gene_id=gene_id,
                chrom=self._chrom_name,
                strand=strand,
                exons=list(exons),
                abundance=self._rng.uniform(5.0, 50.0),
            )
            transcripts.append(full_tx)

            # Additional isoforms with exon skipping
            used_patterns: set[tuple[int, ...]] = {tuple(range(n_exons))}
            for iso_idx in range(1, n_isoforms):
                # Skip 1-2 internal exons (not the first or last)
                if n_exons <= 2:
                    break
                n_skip = self._rng.randint(1, min(2, n_exons - 2))
                skippable = list(range(1, n_exons - 1))
                if len(skippable) < n_skip:
                    break
                skipped = set(self._rng.sample(skippable, n_skip))
                kept_indices = tuple(
                    i for i in range(n_exons) if i not in skipped
                )
                if kept_indices in used_patterns:
                    continue
                used_patterns.add(kept_indices)

                iso_exons = [exons[i] for i in kept_indices]
                iso_tx = _TranscriptDef(
                    transcript_id=f"{gene_id}.{iso_idx + 1}",
                    gene_id=gene_id,
                    chrom=self._chrom_name,
                    strand=strand,
                    exons=iso_exons,
                    abundance=self._rng.uniform(2.0, 30.0),
                )
                transcripts.append(iso_tx)

            gene = _GeneDef(
                gene_id=gene_id,
                chrom=self._chrom_name,
                strand=strand,
                start=gene_start,
                end=gene_end,
                transcripts=transcripts,
            )
            self._genes.append(gene)

            current_pos = gene_end + self._INTERGENIC_GAP

        self._chrom_length = current_pos + self._CHROM_PAD
        logger.info(
            "Designed %d genes with %d total transcripts on %s (length %d bp)",
            len(self._genes),
            sum(len(g.transcripts) for g in self._genes),
            self._chrom_name,
            self._chrom_length,
        )

    # ------------------------------------------------------------------
    # Reference genome
    # ------------------------------------------------------------------

    def _write_reference(self, ref_path: str) -> None:
        """Write a synthetic reference FASTA file.

        Generates a random nucleotide sequence of the required length and
        writes it with standard FASTA line wrapping. Creates a samtools
        index (.fai) alongside the FASTA.

        Args:
            ref_path: Output FASTA file path.
        """
        bases = "ACGT"
        seq_arr = self._np_rng.choice(list(bases), size=self._chrom_length)
        sequence = "".join(seq_arr)

        line_width = 80
        with open(ref_path, "w", encoding="utf-8") as fh:
            fh.write(f">{self._chrom_name}\n")
            for i in range(0, len(sequence), line_width):
                fh.write(sequence[i : i + line_width])
                fh.write("\n")

        # Create FASTA index
        pysam.faidx(ref_path)
        logger.info("Reference FASTA written: %s (%d bp)", ref_path, self._chrom_length)

    # ------------------------------------------------------------------
    # Truth GTF
    # ------------------------------------------------------------------

    def _write_truth_gtf(self, gtf_path: str) -> None:
        """Write a ground-truth GTF file describing all synthetic transcripts.

        Coordinates follow GTF conventions: 1-based, inclusive on both ends.

        Args:
            gtf_path: Output GTF file path.
        """
        with open(gtf_path, "w", encoding="utf-8") as fh:
            fh.write("# Synthetic ground-truth GTF for RapidSplice benchmarking\n")

            for gene in self._genes:
                for tx in gene.transcripts:
                    # Transcript line
                    tx_start = tx.exons[0].start + 1  # 0-based -> 1-based
                    tx_end = tx.exons[-1].end  # half-open -> inclusive
                    attrs = (
                        f'gene_id "{tx.gene_id}"; '
                        f'transcript_id "{tx.transcript_id}"; '
                        f'abundance "{tx.abundance:.2f}";'
                    )
                    fh.write(
                        f"{tx.chrom}\ttruth\ttranscript\t{tx_start}\t{tx_end}\t"
                        f"1000\t{tx.strand}\t.\t{attrs}\n"
                    )

                    # Exon lines
                    sorted_exons = sorted(tx.exons, key=lambda e: e.start)
                    for exon_num, exon in enumerate(sorted_exons, start=1):
                        e_start = exon.start + 1
                        e_end = exon.end
                        exon_attrs = (
                            f'gene_id "{tx.gene_id}"; '
                            f'transcript_id "{tx.transcript_id}"; '
                            f'exon_number "{exon_num}";'
                        )
                        fh.write(
                            f"{tx.chrom}\ttruth\texon\t{e_start}\t{e_end}\t"
                            f"1000\t{tx.strand}\t.\t{exon_attrs}\n"
                        )

        logger.info("Truth GTF written: %s", gtf_path)

    # ------------------------------------------------------------------
    # Read simulation
    # ------------------------------------------------------------------

    def _simulate_reads(self, bam_path: str, ref_path: str) -> None:
        """Simulate reads from known transcripts and write a sorted, indexed BAM.

        Reads are placed uniformly along each transcript's spliced sequence.
        The number of reads per transcript is proportional to its abundance
        and spliced length. CIGAR strings encode alignment matches (M) and
        intron skips (N).

        Args:
            bam_path: Output BAM file path.
            ref_path: Reference FASTA path (used for BAM header).
        """
        read_length = self._config.read_length
        n_reads = self._config.n_reads

        # Compute per-transcript read counts proportional to abundance * length
        all_transcripts: list[_TranscriptDef] = []
        weights: list[float] = []
        for gene in self._genes:
            for tx in gene.transcripts:
                spliced_len = sum(e.end - e.start for e in tx.exons)
                all_transcripts.append(tx)
                weights.append(tx.abundance * spliced_len)

        total_weight = sum(weights)
        if total_weight == 0.0:
            logger.warning("No transcripts to simulate reads from.")
            return

        read_counts = [max(1, int(n_reads * w / total_weight)) for w in weights]

        # Prepare SAM header
        header = pysam.AlignmentHeader.from_dict({
            "HD": {"VN": "1.6", "SO": "unsorted"},
            "SQ": [{"SN": self._chrom_name, "LN": self._chrom_length}],
            "PG": [{"ID": "braid_bench", "PN": "SyntheticDataGenerator"}],
        })

        # Write unsorted SAM to a temporary file, then sort + index
        tmp_dir = tempfile.mkdtemp(prefix="braid_bench_")
        unsorted_bam = os.path.join(tmp_dir, "unsorted.bam")

        try:
            read_counter = 0
            with pysam.AlignmentFile(unsorted_bam, "wb", header=header) as out:
                for tx, count in zip(all_transcripts, read_counts):
                    reads = self._generate_reads_for_transcript(
                        tx, count, read_length, read_counter,
                    )
                    for seg in reads:
                        out.write(seg)
                    read_counter += count

            # Sort by coordinate
            pysam.sort("-o", bam_path, unsorted_bam)
            # Index the sorted BAM
            pysam.index(bam_path)

            logger.info(
                "Simulated BAM written: %s (%d reads)",
                bam_path, sum(read_counts),
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _generate_reads_for_transcript(
        self,
        tx: _TranscriptDef,
        n_reads: int,
        read_length: int,
        start_read_id: int,
    ) -> list[pysam.AlignedSegment]:
        """Generate simulated aligned reads for a single transcript.

        Each read is placed at a random position along the transcript's
        spliced coordinate space, then converted to genomic coordinates
        with a CIGAR string that encodes matches (M) and intron skips (N).

        Args:
            tx: Transcript definition with exon coordinates.
            n_reads: Number of reads to generate for this transcript.
            read_length: Read length in base pairs.
            start_read_id: Starting integer for read name generation.

        Returns:
            List of pysam AlignedSegment objects ready for BAM writing.
        """
        sorted_exons = sorted(tx.exons, key=lambda e: e.start)
        exon_lengths = [e.end - e.start for e in sorted_exons]
        spliced_length = sum(exon_lengths)

        if spliced_length < read_length:
            effective_read_len = spliced_length
        else:
            effective_read_len = read_length

        # Build cumulative exon lengths for spliced-to-genomic mapping
        cum_lens = np.cumsum([0] + exon_lengths)

        segments: list[pysam.AlignedSegment] = []

        for read_idx in range(n_reads):
            # Random position along the spliced transcript
            max_start = max(0, spliced_length - effective_read_len)
            spliced_pos = self._rng.randint(0, max_start) if max_start > 0 else 0
            spliced_end = min(spliced_pos + effective_read_len, spliced_length)

            # Convert spliced coordinates to genomic CIGAR
            cigar_ops, ref_start = self._spliced_to_cigar(
                spliced_pos, spliced_end, sorted_exons, cum_lens,
            )

            if not cigar_ops or ref_start < 0:
                continue

            seg = pysam.AlignedSegment()
            seg.query_name = f"read_{start_read_id + read_idx}"
            seg.query_sequence = "A" * (spliced_end - spliced_pos)
            seg.flag = 0
            seg.reference_id = 0
            seg.reference_start = ref_start
            seg.mapping_quality = 60
            seg.cigar = cigar_ops
            seg.query_qualities = pysam.qualitystring_to_array(
                "I" * (spliced_end - spliced_pos)
            )

            # Add XS strand tag for spliced reads (required by StringTie)
            has_intron = any(op == 3 for op, _ in cigar_ops)
            if has_intron:
                seg.set_tag("XS", tx.strand, value_type="A")
                seg.set_tag("NH", 1, value_type="i")

            segments.append(seg)

        return segments

    def _spliced_to_cigar(
        self,
        spliced_start: int,
        spliced_end: int,
        sorted_exons: list[_ExonDef],
        cum_lens: np.ndarray,
    ) -> tuple[list[tuple[int, int]], int]:
        """Convert spliced coordinates to a genomic CIGAR string.

        Maps a read interval in spliced (exonic) space back to reference
        coordinates, inserting N (intron skip) operations between exons.

        Args:
            spliced_start: Read start in spliced coordinates (0-based).
            spliced_end: Read end in spliced coordinates (exclusive).
            sorted_exons: Exons sorted by genomic start.
            cum_lens: Cumulative exon lengths (length n_exons + 1).

        Returns:
            A tuple of ``(cigar_tuples, reference_start)`` where cigar_tuples
            is a list of ``(op, length)`` pairs and reference_start is the
            0-based reference start position. Returns ``([], -1)`` if the
            read does not map to any exon.
        """
        cigar: list[tuple[int, int]] = []
        ref_start: int = -1
        read_remaining = spliced_end - spliced_start

        for i, exon in enumerate(sorted_exons):
            exon_spliced_start = int(cum_lens[i])
            exon_spliced_end = int(cum_lens[i + 1])

            # Does the read overlap this exon in spliced space?
            overlap_start = max(spliced_start, exon_spliced_start)
            overlap_end = min(spliced_end, exon_spliced_end)

            if overlap_start >= overlap_end:
                continue

            # Offset within this exon
            offset_in_exon = overlap_start - exon_spliced_start
            match_len = overlap_end - overlap_start

            genomic_start = exon.start + offset_in_exon

            if ref_start == -1:
                ref_start = genomic_start
            else:
                # Insert intron skip (N) from previous exon end to current exon start
                prev_exon = sorted_exons[i - 1]
                intron_len = exon.start - prev_exon.end
                if intron_len > 0:
                    cigar.append((3, intron_len))  # 3 = N (skip)

            cigar.append((0, match_len))  # 0 = M (match)
            read_remaining -= match_len

            if read_remaining <= 0:
                break

        return cigar, ref_start


# ---------------------------------------------------------------------------
# GTF parser for evaluation
# ---------------------------------------------------------------------------


def _parse_gtf_transcripts(
    gtf_path: str,
) -> dict[str, list[tuple[int, int]]]:
    """Parse a GTF file and return transcript exon structures.

    Reads exon features from the GTF, groups them by transcript_id, and
    returns a mapping from transcript_id to a sorted list of exon
    intervals in 0-based half-open coordinates.

    Args:
        gtf_path: Path to the GTF file.

    Returns:
        Dictionary mapping transcript IDs to sorted exon coordinate lists.
    """
    transcripts: dict[str, list[tuple[int, int]]] = {}

    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 9:
                continue

            feature = parts[2]
            if feature != "exon":
                continue

            # 1-based inclusive -> 0-based half-open
            start = int(parts[3]) - 1
            end = int(parts[4])

            # Parse transcript_id from attributes
            attrs = parts[8]
            tx_id = _extract_attribute(attrs, "transcript_id")
            if tx_id is None:
                continue

            if tx_id not in transcripts:
                transcripts[tx_id] = []
            transcripts[tx_id].append((start, end))

    # Sort exons within each transcript
    for tx_id in transcripts:
        transcripts[tx_id].sort()

    return transcripts


def _extract_attribute(attrs: str, key: str) -> str | None:
    """Extract a named attribute value from a GTF attribute string.

    Args:
        attrs: The ninth column of a GTF line (semicolon-separated key-value
            pairs).
        key: The attribute key to search for.

    Returns:
        The attribute value (with quotes stripped), or ``None`` if not found.
    """
    for item in attrs.split(";"):
        item = item.strip()
        if item.startswith(key):
            parts = item.split('"')
            if len(parts) >= 2:
                return parts[1]
            # Try unquoted
            parts = item.split()
            if len(parts) >= 2:
                return parts[1].strip('"')
    return None


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Runs benchmarks comparing RapidSplice against baseline assemblers.

    Orchestrates the full benchmark pipeline: synthetic data generation,
    tool execution, output evaluation, and results collection.

    Args:
        config: Benchmark configuration.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config: BenchmarkConfig = config
        self._out_dir: Path = Path(config.output_dir)
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def run_braid(
        self, bam_path: str, reference_path: str
    ) -> tuple[str, float, float]:
        """Run RapidSplice on the input BAM and measure performance.

        Args:
            bam_path: Path to the input BAM file.
            reference_path: Path to the reference FASTA.

        Returns:
            A tuple ``(gtf_path, elapsed_seconds, peak_memory_mb)`` with the
            output GTF path, wall-clock time, and peak resident memory.
        """
        gtf_path = str(self._out_dir / "braid_output.gtf")

        cmd = [
            sys.executable, "-m", "braid.cli",
            bam_path,
            "-o", gtf_path,
            "-r", reference_path,
            "-t", str(self._config.threads),
            "-c", "0.5",
            "-s", "0.1",
        ]

        logger.info("Running RapidSplice: %s", " ".join(cmd))
        elapsed, peak_mb = self._run_command(cmd)
        logger.info(
            "RapidSplice completed: %.2f s, %.1f MB peak", elapsed, peak_mb,
        )

        # Verify output exists; if CLI run failed, create a minimal output
        if not os.path.exists(gtf_path):
            logger.warning(
                "RapidSplice did not produce output; creating empty GTF."
            )
            with open(gtf_path, "w", encoding="utf-8") as fh:
                fh.write("# RapidSplice - no output\n")

        return gtf_path, elapsed, peak_mb

    def run_stringtie(
        self, bam_path: str, reference_path: str
    ) -> tuple[str, float, float] | None:
        """Run StringTie on the input BAM if the tool is installed.

        Args:
            bam_path: Path to the input BAM file.
            reference_path: Path to the reference FASTA (unused by StringTie
                but kept for API symmetry).

        Returns:
            A tuple ``(gtf_path, elapsed_seconds, peak_memory_mb)`` or
            ``None`` if StringTie is not installed.
        """
        if not self._config.run_stringtie:
            return None

        if shutil.which("stringtie") is None:
            logger.warning("StringTie not found in PATH; skipping.")
            return None

        gtf_path = str(self._out_dir / "stringtie_output.gtf")
        cmd = [
            "stringtie",
            bam_path,
            "-o", gtf_path,
            "-p", str(self._config.threads),
        ]

        logger.info("Running StringTie: %s", " ".join(cmd))
        elapsed, peak_mb = self._run_command(cmd)
        logger.info(
            "StringTie completed: %.2f s, %.1f MB peak", elapsed, peak_mb,
        )

        if not os.path.exists(gtf_path):
            logger.warning("StringTie did not produce output.")
            return None

        return gtf_path, elapsed, peak_mb

    def run_scallop(
        self, bam_path: str, reference_path: str
    ) -> tuple[str, float, float] | None:
        """Run Scallop2 on the input BAM if the tool is installed.

        Args:
            bam_path: Path to the input BAM file.
            reference_path: Path to the reference FASTA (unused by Scallop2
                but kept for API symmetry).

        Returns:
            A tuple ``(gtf_path, elapsed_seconds, peak_memory_mb)`` or
            ``None`` if Scallop2 is not installed.
        """
        if not self._config.run_scallop:
            return None

        scallop_bin = shutil.which("scallop2") or shutil.which("scallop")
        if scallop_bin is None:
            logger.warning("Scallop2 / Scallop not found in PATH; skipping.")
            return None

        gtf_path = str(self._out_dir / "scallop_output.gtf")
        cmd = [
            scallop_bin,
            "-i", bam_path,
            "-o", gtf_path,
        ]

        logger.info("Running Scallop2: %s", " ".join(cmd))
        elapsed, peak_mb = self._run_command(cmd)
        logger.info(
            "Scallop2 completed: %.2f s, %.1f MB peak", elapsed, peak_mb,
        )

        if not os.path.exists(gtf_path):
            logger.warning("Scallop2 did not produce output.")
            return None

        return gtf_path, elapsed, peak_mb

    def evaluate_gtf(
        self, predicted_gtf: str, truth_gtf: str
    ) -> dict[str, float]:
        """Evaluate a predicted GTF against the ground-truth GTF.

        Computes transcript-level, exon-level, and intron-level sensitivity
        and precision using coordinate comparison. Transcript matching
        requires an exact intron chain match. Exon matching uses 50%
        reciprocal overlap. Intron matching requires exact donor-acceptor
        coordinate agreement.

        Args:
            predicted_gtf: Path to the predicted GTF file.
            truth_gtf: Path to the ground-truth GTF file.

        Returns:
            Dictionary with keys matching :class:`EvaluationMetrics` field
            names and float values in [0, 1].
        """
        pred_tx = _parse_gtf_transcripts(predicted_gtf)
        true_tx = _parse_gtf_transcripts(truth_gtf)

        metrics = EvaluationMetrics()

        # ------------------------------------------------------------------
        # Transcript-level (exact intron chain match)
        # ------------------------------------------------------------------
        true_intron_chains = self._get_intron_chains(true_tx)
        pred_intron_chains = self._get_intron_chains(pred_tx)

        true_chain_set = set(true_intron_chains.values())
        pred_chain_set = set(pred_intron_chains.values())

        matched_true = sum(1 for c in true_chain_set if c in pred_chain_set)
        matched_pred = sum(1 for c in pred_chain_set if c in true_chain_set)

        if true_chain_set:
            metrics.transcript_sensitivity = matched_true / len(true_chain_set)
        if pred_chain_set:
            metrics.transcript_precision = matched_pred / len(pred_chain_set)

        # ------------------------------------------------------------------
        # Exon-level (50% reciprocal overlap)
        # ------------------------------------------------------------------
        true_exons = self._collect_all_exons(true_tx)
        pred_exons = self._collect_all_exons(pred_tx)

        exon_sn, exon_pr = self._compute_overlap_metrics(
            true_exons, pred_exons, min_reciprocal_overlap=0.5,
        )
        metrics.exon_sensitivity = exon_sn
        metrics.exon_precision = exon_pr

        # ------------------------------------------------------------------
        # Intron-level (exact donor-acceptor match)
        # ------------------------------------------------------------------
        true_introns = self._collect_all_introns(true_tx)
        pred_introns = self._collect_all_introns(pred_tx)

        true_intron_set = set(true_introns)
        pred_intron_set = set(pred_introns)

        if true_intron_set:
            metrics.intron_sensitivity = len(
                true_intron_set & pred_intron_set
            ) / len(true_intron_set)
        if pred_intron_set:
            metrics.intron_precision = len(
                true_intron_set & pred_intron_set
            ) / len(pred_intron_set)

        return asdict(metrics)

    def run_all(self) -> dict:
        """Execute the full benchmark pipeline.

        Generates synthetic data, runs all tools, evaluates outputs, and
        collects results into a single dictionary that is also saved as
        JSON.

        Returns:
            Dictionary with benchmark configuration, per-tool results
            (metrics, runtime, memory), and dataset metadata.
        """
        # Generate synthetic data
        generator = SyntheticDataGenerator(self._config)
        bam_path, ref_path, truth_gtf = generator.generate()

        results: dict = {
            "config": asdict(self._config),
            "dataset": {
                "type": "synthetic",
                "n_genes": self._config.n_genes,
                "n_reads": self._config.n_reads,
                "read_length": self._config.read_length,
                "bam_path": bam_path,
                "reference_path": ref_path,
                "truth_gtf": truth_gtf,
            },
            "tools": {},
        }

        # Run RapidSplice
        rs_gtf, rs_time, rs_mem = self.run_braid(bam_path, ref_path)
        rs_metrics = self.evaluate_gtf(rs_gtf, truth_gtf)
        results["tools"]["RapidSplice"] = {
            "gtf_path": rs_gtf,
            "runtime_seconds": rs_time,
            "peak_memory_mb": rs_mem,
            "metrics": rs_metrics,
        }

        # Run StringTie
        st_result = self.run_stringtie(bam_path, ref_path)
        if st_result is not None:
            st_gtf, st_time, st_mem = st_result
            st_metrics = self.evaluate_gtf(st_gtf, truth_gtf)
            results["tools"]["StringTie"] = {
                "gtf_path": st_gtf,
                "runtime_seconds": st_time,
                "peak_memory_mb": st_mem,
                "metrics": st_metrics,
            }

        # Run Scallop2
        sc_result = self.run_scallop(bam_path, ref_path)
        if sc_result is not None:
            sc_gtf, sc_time, sc_mem = sc_result
            sc_metrics = self.evaluate_gtf(sc_gtf, truth_gtf)
            results["tools"]["Scallop2"] = {
                "gtf_path": sc_gtf,
                "runtime_seconds": sc_time,
                "peak_memory_mb": sc_mem,
                "metrics": sc_metrics,
            }

        # Save results JSON
        results_path = str(self._out_dir / "results.json")
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, default=str)
        logger.info("Results saved to %s", results_path)

        # Print summary table
        self._print_summary(results)

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_command(self, cmd: list[str]) -> tuple[float, float]:
        """Execute an external command and measure time and peak memory.

        Args:
            cmd: Command and arguments to execute.

        Returns:
            Tuple of ``(elapsed_seconds, peak_memory_mb)``.
        """
        start_time = time.perf_counter()

        # Record memory before
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if result.returncode != 0:
                logger.warning(
                    "Command exited with code %d: %s\nstderr: %s",
                    result.returncode,
                    " ".join(cmd),
                    result.stderr[:2000] if result.stderr else "(none)",
                )
        except subprocess.TimeoutExpired:
            logger.error("Command timed out after 3600s: %s", " ".join(cmd))
        except FileNotFoundError:
            logger.error("Command not found: %s", cmd[0])

        elapsed = time.perf_counter() - start_time

        # Peak memory from resource usage (children)
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        # ru_maxrss is in kilobytes on Linux
        peak_kb = usage_after.ru_maxrss - usage_before.ru_maxrss
        peak_mb = max(0.0, peak_kb / 1024.0)

        return elapsed, peak_mb

    @staticmethod
    def _get_intron_chains(
        transcripts: dict[str, list[tuple[int, int]]],
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        """Extract intron chains from transcript exon structures.

        An intron chain is the tuple of (donor, acceptor) pairs derived from
        consecutive exon boundaries. Single-exon transcripts have an empty
        chain.

        Args:
            transcripts: Mapping from transcript ID to sorted exon list.

        Returns:
            Mapping from transcript ID to intron chain tuple.
        """
        chains: dict[str, tuple[tuple[int, int], ...]] = {}
        for tx_id, exons in transcripts.items():
            if len(exons) < 2:
                chains[tx_id] = ()
                continue
            introns = []
            for i in range(len(exons) - 1):
                donor = exons[i][1]
                acceptor = exons[i + 1][0]
                introns.append((donor, acceptor))
            chains[tx_id] = tuple(introns)
        return chains

    @staticmethod
    def _collect_all_exons(
        transcripts: dict[str, list[tuple[int, int]]],
    ) -> list[tuple[int, int]]:
        """Collect all unique exon intervals across transcripts.

        Args:
            transcripts: Mapping from transcript ID to sorted exon list.

        Returns:
            Sorted list of unique (start, end) exon intervals.
        """
        exon_set: set[tuple[int, int]] = set()
        for exons in transcripts.values():
            for exon in exons:
                exon_set.add(exon)
        return sorted(exon_set)

    @staticmethod
    def _collect_all_introns(
        transcripts: dict[str, list[tuple[int, int]]],
    ) -> list[tuple[int, int]]:
        """Collect all unique intron intervals across transcripts.

        Args:
            transcripts: Mapping from transcript ID to sorted exon list.

        Returns:
            Sorted list of unique (donor, acceptor) intron intervals.
        """
        intron_set: set[tuple[int, int]] = set()
        for exons in transcripts.values():
            if len(exons) < 2:
                continue
            for i in range(len(exons) - 1):
                donor = exons[i][1]
                acceptor = exons[i + 1][0]
                intron_set.add((donor, acceptor))
        return sorted(intron_set)

    @staticmethod
    def _compute_overlap_metrics(
        true_intervals: list[tuple[int, int]],
        pred_intervals: list[tuple[int, int]],
        min_reciprocal_overlap: float = 0.5,
    ) -> tuple[float, float]:
        """Compute sensitivity and precision based on reciprocal overlap.

        Two intervals match if their overlap divided by each interval's
        length is at least ``min_reciprocal_overlap``.

        Args:
            true_intervals: Ground-truth intervals (sorted).
            pred_intervals: Predicted intervals (sorted).
            min_reciprocal_overlap: Minimum reciprocal overlap fraction.

        Returns:
            Tuple ``(sensitivity, precision)`` as floats in [0, 1].
        """
        if not true_intervals and not pred_intervals:
            return 1.0, 1.0
        if not true_intervals:
            return 1.0, 0.0
        if not pred_intervals:
            return 0.0, 1.0

        true_matched = 0
        pred_matched = 0

        # For each true interval, find the best matching predicted interval
        for t_start, t_end in true_intervals:
            t_len = t_end - t_start
            for p_start, p_end in pred_intervals:
                p_len = p_end - p_start
                overlap_start = max(t_start, p_start)
                overlap_end = min(t_end, p_end)
                overlap = max(0, overlap_end - overlap_start)

                if t_len > 0 and p_len > 0:
                    frac_true = overlap / t_len
                    frac_pred = overlap / p_len
                    if (
                        frac_true >= min_reciprocal_overlap
                        and frac_pred >= min_reciprocal_overlap
                    ):
                        true_matched += 1
                        break

        # For each predicted interval, find the best matching true interval
        for p_start, p_end in pred_intervals:
            p_len = p_end - p_start
            for t_start, t_end in true_intervals:
                t_len = t_end - t_start
                overlap_start = max(t_start, p_start)
                overlap_end = min(t_end, p_end)
                overlap = max(0, overlap_end - overlap_start)

                if t_len > 0 and p_len > 0:
                    frac_true = overlap / t_len
                    frac_pred = overlap / p_len
                    if (
                        frac_true >= min_reciprocal_overlap
                        and frac_pred >= min_reciprocal_overlap
                    ):
                        pred_matched += 1
                        break

        sensitivity = true_matched / len(true_intervals)
        precision = pred_matched / len(pred_intervals)

        return sensitivity, precision

    @staticmethod
    def _print_summary(results: dict) -> None:
        """Print a formatted summary table of benchmark results.

        Args:
            results: The results dictionary produced by :meth:`run_all`.
        """
        print("\n" + "=" * 90)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 90)
        print(
            f"Dataset: {results['dataset']['type']} | "
            f"Genes: {results['dataset']['n_genes']} | "
            f"Reads: {results['dataset']['n_reads']:,}"
        )
        print("-" * 90)

        header = (
            f"{'Tool':<15} {'TxSn':>6} {'TxPr':>6} {'ExSn':>6} "
            f"{'ExPr':>6} {'InSn':>6} {'InPr':>6} "
            f"{'Time(s)':>9} {'Mem(MB)':>9}"
        )
        print(header)
        print("-" * 90)

        for tool_name, tool_data in results["tools"].items():
            m = tool_data["metrics"]
            print(
                f"{tool_name:<15} "
                f"{m['transcript_sensitivity']:>6.1%} "
                f"{m['transcript_precision']:>6.1%} "
                f"{m['exon_sensitivity']:>6.1%} "
                f"{m['exon_precision']:>6.1%} "
                f"{m['intron_sensitivity']:>6.1%} "
                f"{m['intron_precision']:>6.1%} "
                f"{tool_data['runtime_seconds']:>9.2f} "
                f"{tool_data['peak_memory_mb']:>9.1f}"
            )

        print("=" * 90)
        print(
            f"\nFull results saved to: "
            f"{results['config']['output_dir']}/results.json"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and run the benchmark suite.

    Configures logging, builds a :class:`BenchmarkConfig` from command-line
    arguments, and executes :meth:`BenchmarkRunner.run_all`.
    """
    parser = argparse.ArgumentParser(
        description="RapidSplice benchmark runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=100,
        help="Number of synthetic genes to generate.",
    )
    parser.add_argument(
        "--n-reads",
        type=int,
        default=1_000_000,
        help="Number of simulated reads.",
    )
    parser.add_argument(
        "--read-length",
        type=int,
        default=100,
        help="Simulated read length (bp).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads for multi-threaded tools.",
    )
    parser.add_argument(
        "--no-stringtie",
        action="store_true",
        default=False,
        help="Skip StringTie baseline.",
    )
    parser.add_argument(
        "--no-scallop",
        action="store_true",
        default=False,
        help="Skip Scallop2 baseline.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose logging.",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = BenchmarkConfig(
        output_dir=args.output_dir,
        n_genes=args.n_genes,
        n_reads=args.n_reads,
        read_length=args.read_length,
        run_stringtie=not args.no_stringtie,
        run_scallop=not args.no_scallop,
        threads=args.threads,
    )

    runner = BenchmarkRunner(config)
    results = runner.run_all()

    # Attempt to generate PDF report
    try:
        from benchmarks.generate_report import generate_report

        report_path = str(Path(config.output_dir) / "benchmark_report.pdf")
        generate_report(results, report_path)
        logger.info("PDF report generated: %s", report_path)
    except ImportError:
        logger.info(
            "Report generation skipped (run benchmarks/generate_report.py "
            "separately)."
        )
    except Exception as exc:
        logger.warning("Report generation failed: %s", exc)


if __name__ == "__main__":
    main()
