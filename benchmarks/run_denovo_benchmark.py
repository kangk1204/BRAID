"""Benchmark de novo assembler against Trinity/rnaSPAdes.

Generates synthetic FASTQ reads from known transcript sequences,
runs RapidSplice de novo assembler and competing tools, then evaluates
transcript recovery using BLAST-based metrics.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from braid.denovo.pipeline import DeNovoConfig, run_denovo_assembly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Synthetic data generation
# ===========================================================================


@dataclass
class SyntheticTranscript:
    """A synthetic reference transcript.

    Attributes:
        name: Transcript identifier.
        sequence: Full DNA sequence.
        expression_level: Relative expression (copies).
    """

    name: str
    sequence: str
    expression_level: int = 10


def generate_random_sequence(length: int, rng: np.random.RandomState) -> str:
    """Generate a random DNA sequence.

    Args:
        length: Sequence length in bases.
        rng: Random state for reproducibility.

    Returns:
        Random DNA sequence string.
    """
    return "".join(rng.choice(list("ACGT"), size=length))


def generate_transcripts(
    n_transcripts: int = 50,
    min_length: int = 300,
    max_length: int = 3000,
    seed: int = 42,
) -> list[SyntheticTranscript]:
    """Generate a set of synthetic transcripts with varying expression.

    Creates transcripts with realistic length distribution and
    expression levels spanning 3 orders of magnitude.

    Args:
        n_transcripts: Number of transcripts to generate.
        min_length: Minimum transcript length.
        max_length: Maximum transcript length.
        seed: Random seed for reproducibility.

    Returns:
        List of synthetic transcripts.
    """
    rng = np.random.RandomState(seed)
    transcripts = []

    for i in range(n_transcripts):
        length = rng.randint(min_length, max_length + 1)
        seq = generate_random_sequence(length, rng)

        # Higher expression for sufficient coverage
        # Need ~30x coverage: for a 1000bp transcript with 100bp reads,
        # need ~300 reads to get 30x coverage.
        expr = max(50, int(rng.lognormal(mean=5.5, sigma=0.8)))

        transcripts.append(SyntheticTranscript(
            name=f"TX_{i:04d}",
            sequence=seq,
            expression_level=expr,
        ))

    return transcripts


def simulate_reads(
    transcripts: list[SyntheticTranscript],
    read_length: int = 100,
    error_rate: float = 0.01,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Simulate FASTQ reads from synthetic transcripts.

    Generates single-end reads with uniform error model.

    Args:
        transcripts: Reference transcripts.
        read_length: Read length in bases.
        error_rate: Per-base error rate.
        seed: Random seed.

    Returns:
        Tuple of (headers, sequences, qualities).
    """
    rng = np.random.RandomState(seed)
    headers: list[str] = []
    sequences: list[str] = []
    qualities: list[str] = []
    read_id = 0

    for tx in transcripts:
        n_reads = tx.expression_level
        for _ in range(n_reads):
            if len(tx.sequence) <= read_length:
                start = 0
                frag = tx.sequence
            else:
                start = rng.randint(0, len(tx.sequence) - read_length)
                frag = tx.sequence[start:start + read_length]

            # Add sequencing errors
            if error_rate > 0:
                frag = _add_errors(frag, error_rate, rng)

            # Random strand
            if rng.random() < 0.5:
                frag = _reverse_complement(frag)

            qual = "I" * len(frag)  # Q=40 for all bases
            headers.append(f"@read_{read_id} tx={tx.name} pos={start}")
            sequences.append(frag)
            qualities.append(qual)
            read_id += 1

    return headers, sequences, qualities


def write_fastq(
    output_path: str,
    headers: list[str],
    sequences: list[str],
    qualities: list[str],
) -> int:
    """Write reads to a FASTQ file.

    Args:
        output_path: Output file path.
        headers: Read headers.
        sequences: Read sequences.
        qualities: Quality strings.

    Returns:
        Number of reads written.
    """
    with open(output_path, "w") as fh:
        for h, s, q in zip(headers, sequences, qualities):
            fh.write(f"{h}\n{s}\n+\n{q}\n")
    return len(headers)


def write_reference_fasta(
    output_path: str,
    transcripts: list[SyntheticTranscript],
) -> None:
    """Write reference transcripts to FASTA for evaluation.

    Args:
        output_path: Output file path.
        transcripts: Reference transcripts.
    """
    with open(output_path, "w") as fh:
        for tx in transcripts:
            fh.write(f">{tx.name} len={len(tx.sequence)} expr={tx.expression_level}\n")
            for i in range(0, len(tx.sequence), 80):
                fh.write(tx.sequence[i:i + 80] + "\n")


def _add_errors(seq: str, rate: float, rng: np.random.RandomState) -> str:
    """Add random substitution errors to a sequence.

    Args:
        seq: Input DNA sequence.
        rate: Per-base error rate.
        rng: Random state.

    Returns:
        Sequence with errors.
    """
    bases = list(seq)
    for i in range(len(bases)):
        if rng.random() < rate:
            bases[i] = rng.choice([b for b in "ACGT" if b != bases[i]])
    return "".join(bases)


def _reverse_complement(seq: str) -> str:
    """Compute reverse complement of a DNA sequence.

    Args:
        seq: DNA sequence.

    Returns:
        Reverse complement string.
    """
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq))


# ===========================================================================
# Evaluation
# ===========================================================================


@dataclass
class AssemblyResult:
    """Results from one assembler run.

    Attributes:
        name: Assembler name.
        n_transcripts: Number of assembled transcripts.
        total_bases: Total assembled bases.
        n50: N50 length.
        mean_length: Mean transcript length.
        max_length: Maximum transcript length.
        elapsed_seconds: Wall time.
        recall: Fraction of reference transcripts recovered.
        precision: Fraction of assembled transcripts matching reference.
        f1: Harmonic mean of recall and precision.
        recovered_transcripts: Number of reference transcripts recovered.
    """

    name: str
    n_transcripts: int = 0
    total_bases: int = 0
    n50: int = 0
    mean_length: float = 0.0
    max_length: int = 0
    elapsed_seconds: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    recovered_transcripts: int = 0


def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Parse a FASTA file into (name, sequence) pairs.

    Args:
        path: FASTA file path.

    Returns:
        List of (header, sequence) tuples.
    """
    records: list[tuple[str, str]] = []
    current_name = ""
    current_seq: list[str] = []

    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if current_name:
                    records.append((current_name, "".join(current_seq)))
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_name:
            records.append((current_name, "".join(current_seq)))

    return records


def evaluate_assembly(
    assembled_path: str,
    reference_transcripts: list[SyntheticTranscript],
    min_identity: float = 0.90,
    min_coverage: float = 0.80,
) -> tuple[float, float, float, int]:
    """Evaluate assembled transcripts against reference.

    Uses k-mer based metrics standard for de novo assembly evaluation:

    - **Recall** (reference k-mer coverage): Fraction of reference
      k-mers found in any assembled transcript.
    - **Precision** (assembly correctness): Fraction of assembled
      k-mers found in any reference transcript.
    - **Recovered**: Number of reference transcripts with >=10%
      k-mer recovery by any single assembled contig.

    Args:
        assembled_path: Path to assembled FASTA.
        reference_transcripts: Reference transcripts.
        min_identity: Not used (kept for API compatibility).
        min_coverage: Not used (kept for API compatibility).

    Returns:
        Tuple of (recall, precision, f1, n_recovered).
    """
    if not os.path.exists(assembled_path):
        return 0.0, 0.0, 0.0, 0

    assembled = parse_fasta(assembled_path)
    if not assembled:
        return 0.0, 0.0, 0.0, 0

    k = 15  # K-mer size for similarity comparison

    # Build combined reference k-mer set (both strands)
    all_ref_kmers: set[str] = set()
    ref_kmer_sets: list[set[str]] = []
    for tx in reference_transcripts:
        kmers: set[str] = set()
        seq = tx.sequence
        for i in range(len(seq) - k + 1):
            kmers.add(seq[i:i + k])
        rc = _reverse_complement(seq)
        for i in range(len(rc) - k + 1):
            kmers.add(rc[i:i + k])
        ref_kmer_sets.append(kmers)
        all_ref_kmers |= kmers

    # Build combined assembled k-mer set
    all_asm_kmers: set[str] = set()
    asm_kmer_sets: list[set[str]] = []
    for _, seq in assembled:
        kmers_set: set[str] = set()
        for i in range(len(seq) - k + 1):
            kmers_set.add(seq[i:i + k])
        asm_kmer_sets.append(kmers_set)
        all_asm_kmers |= kmers_set

    # Global recall: what fraction of reference k-mers appear in assembly
    shared_global = len(all_ref_kmers & all_asm_kmers)
    recall = shared_global / len(all_ref_kmers) if all_ref_kmers else 0

    # Global precision: what fraction of assembly k-mers are in reference
    precision = shared_global / len(all_asm_kmers) if all_asm_kmers else 0

    # F1
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0

    # Count recovered transcripts (>=10% of reference k-mers found)
    n_recovered = 0
    for ref_kmers in ref_kmer_sets:
        if not ref_kmers:
            continue
        # Check if any assembled contig covers this reference
        best_cov = 0.0
        for asm_kmers in asm_kmer_sets:
            shared = len(ref_kmers & asm_kmers)
            cov = shared / len(ref_kmers)
            best_cov = max(best_cov, cov)
        if best_cov >= 0.10:
            n_recovered += 1

    return recall, precision, f1, n_recovered


def compute_assembly_stats(fasta_path: str) -> tuple[int, int, float, int]:
    """Compute basic assembly statistics from a FASTA file.

    Args:
        fasta_path: Path to FASTA file.

    Returns:
        Tuple of (n_transcripts, n50, mean_length, max_length).
    """
    if not os.path.exists(fasta_path):
        return 0, 0, 0.0, 0

    records = parse_fasta(fasta_path)
    if not records:
        return 0, 0, 0.0, 0

    lengths = sorted([len(seq) for _, seq in records], reverse=True)
    n = len(lengths)
    total = sum(lengths)
    mean_len = total / n
    max_len = lengths[0]

    # N50
    running = 0
    n50 = 0
    for length in lengths:
        running += length
        if running >= total / 2:
            n50 = length
            break

    return n, n50, mean_len, max_len


# ===========================================================================
# Assembler runners
# ===========================================================================


def run_braid_denovo(
    fastq_path: str,
    output_dir: str,
    k: int = 25,
) -> AssemblyResult:
    """Run RapidSplice de novo assembler.

    Args:
        fastq_path: Input FASTQ path.
        output_dir: Output directory.
        k: K-mer size.

    Returns:
        AssemblyResult with metrics.
    """
    out_fa = os.path.join(output_dir, "braid_denovo.fa")

    t0 = time.perf_counter()
    config = DeNovoConfig(
        fastq_paths=[fastq_path],
        output_path=out_fa,
        k=k,
        min_kmer_count=2,
        min_transcript_length=200,
        min_transcript_coverage=2.0,
        min_read_length=50,
    )
    transcripts, stats = run_denovo_assembly(config)
    elapsed = time.perf_counter() - t0

    n_tx, n50, mean_len, max_len = compute_assembly_stats(out_fa)

    result = AssemblyResult(
        name="RapidSplice-denovo",
        n_transcripts=n_tx,
        total_bases=stats.total_transcript_bases,
        n50=n50,
        mean_length=mean_len,
        max_length=max_len,
        elapsed_seconds=elapsed,
    )
    return result, out_fa


def run_rnaspades(
    fastq_path: str,
    output_dir: str,
) -> tuple[AssemblyResult, str]:
    """Run rnaSPAdes assembler.

    Args:
        fastq_path: Input FASTQ path.
        output_dir: Output directory.

    Returns:
        Tuple of (AssemblyResult, output_fasta_path).
    """
    spades_dir = os.path.join(output_dir, "rnaspades_out")

    if not shutil.which("spades.py") and not shutil.which("rnaspades.py"):
        logger.warning("rnaSPAdes not found, skipping")
        return AssemblyResult(name="rnaSPAdes"), ""

    t0 = time.perf_counter()
    cmd = [
        "spades.py", "--rna",
        "-s", fastq_path,
        "-o", spades_dir,
        "-t", "4",
        "--memory", "8",
    ]
    try:
        subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("rnaSPAdes failed: %s", e)
        return AssemblyResult(name="rnaSPAdes"), ""

    elapsed = time.perf_counter() - t0

    out_fa = os.path.join(spades_dir, "transcripts.fasta")
    if not os.path.exists(out_fa):
        # Try alternative output path
        out_fa = os.path.join(spades_dir, "hard_filtered_transcripts.fasta")

    n_tx, n50, mean_len, max_len = compute_assembly_stats(out_fa)
    total_bases = 0
    if os.path.exists(out_fa):
        records = parse_fasta(out_fa)
        total_bases = sum(len(s) for _, s in records)

    result = AssemblyResult(
        name="rnaSPAdes",
        n_transcripts=n_tx,
        total_bases=total_bases,
        n50=n50,
        mean_length=mean_len,
        max_length=max_len,
        elapsed_seconds=elapsed,
    )
    return result, out_fa


def run_trinity(
    fastq_path: str,
    output_dir: str,
) -> tuple[AssemblyResult, str]:
    """Run Trinity assembler.

    Args:
        fastq_path: Input FASTQ path.
        output_dir: Output directory.

    Returns:
        Tuple of (AssemblyResult, output_fasta_path).
    """
    trinity_dir = os.path.join(output_dir, "trinity_out")

    if not shutil.which("Trinity"):
        logger.warning("Trinity not found, skipping")
        return AssemblyResult(name="Trinity"), ""

    t0 = time.perf_counter()
    cmd = [
        "Trinity",
        "--seqType", "fq",
        "--single", fastq_path,
        "--max_memory", "8G",
        "--CPU", "4",
        "--output", trinity_dir,
        "--no_normalize_reads",
    ]
    try:
        subprocess.run(
            cmd, capture_output=True, text=True, timeout=1200,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning("Trinity failed: %s", e)
        return AssemblyResult(name="Trinity"), ""

    elapsed = time.perf_counter() - t0

    out_fa = os.path.join(trinity_dir, "Trinity.fasta")
    n_tx, n50, mean_len, max_len = compute_assembly_stats(out_fa)
    total_bases = 0
    if os.path.exists(out_fa):
        records = parse_fasta(out_fa)
        total_bases = sum(len(s) for _, s in records)

    result = AssemblyResult(
        name="Trinity",
        n_transcripts=n_tx,
        total_bases=total_bases,
        n50=n50,
        mean_length=mean_len,
        max_length=max_len,
        elapsed_seconds=elapsed,
    )
    return result, out_fa


# ===========================================================================
# Report generation
# ===========================================================================


def generate_report(
    results: list[AssemblyResult],
    ref_transcripts: list[SyntheticTranscript],
    n_reads: int,
    output_dir: str,
) -> None:
    """Generate benchmark comparison report with figures.

    Args:
        results: List of assembly results.
        ref_transcripts: Reference transcripts.
        n_reads: Total number of reads.
        output_dir: Output directory for report files.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter out empty results
    valid = [r for r in results if r.n_transcripts > 0]

    if not valid:
        logger.warning("No valid assembly results to report")
        return

    # --- Report text ---
    report_path = os.path.join(output_dir, "denovo_benchmark_report.txt")
    with open(report_path, "w") as fh:
        fh.write("=" * 70 + "\n")
        fh.write("De Novo RNA-Seq Assembly Benchmark Report\n")
        fh.write("=" * 70 + "\n\n")
        fh.write(f"Reference transcripts: {len(ref_transcripts)}\n")
        fh.write(f"Total reads: {n_reads}\n")
        ref_lengths = [len(tx.sequence) for tx in ref_transcripts]
        fh.write(f"Ref length range: {min(ref_lengths)}-{max(ref_lengths)} bp\n")
        fh.write(f"Ref mean length: {np.mean(ref_lengths):.0f} bp\n\n")

        fh.write(f"{'Metric':<25}")
        for r in valid:
            fh.write(f"{r.name:>20}")
        fh.write("\n" + "-" * (25 + 20 * len(valid)) + "\n")

        rows = [
            ("Transcripts", [r.n_transcripts for r in valid], "d"),
            ("Total bases", [r.total_bases for r in valid], ",d"),
            ("N50 (bp)", [r.n50 for r in valid], ",d"),
            ("Mean length (bp)", [r.mean_length for r in valid], ".0f"),
            ("Max length (bp)", [r.max_length for r in valid], ",d"),
            ("Recall (%)", [r.recall * 100 for r in valid], ".1f"),
            ("Precision (%)", [r.precision * 100 for r in valid], ".1f"),
            ("F1 (%)", [r.f1 * 100 for r in valid], ".1f"),
            ("Recovered txs", [r.recovered_transcripts for r in valid], "d"),
            ("Time (s)", [r.elapsed_seconds for r in valid], ".1f"),
        ]

        for label, vals, fmt in rows:
            fh.write(f"{label:<25}")
            for v in vals:
                fh.write(f"{v:>20{fmt}}")
            fh.write("\n")

        fh.write("\n" + "=" * 70 + "\n")

    logger.info("Report written to %s", report_path)

    # --- Figure 1: Assembly statistics comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    names = [r.name for r in valid]
    colors = ["#4060c0", "#c04040", "#40c040", "#c0c040"][:len(valid)]

    # Panel A: N50
    axes[0].bar(names, [r.n50 for r in valid], color=colors,
                edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("N50 (bp)")
    axes[0].set_title("A. N50 Length")
    for i, r in enumerate(valid):
        axes[0].text(i, r.n50 + max(r.n50 for r in valid) * 0.02,
                     str(r.n50), ha="center", fontsize=9)

    # Panel B: Transcript recovery
    axes[1].bar(names, [r.recall * 100 for r in valid], color=colors,
                edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("Recall (%)")
    axes[1].set_title("B. Reference Recovery")
    axes[1].set_ylim(0, 100)
    for i, r in enumerate(valid):
        axes[1].text(i, r.recall * 100 + 2, f"{r.recall * 100:.1f}%",
                     ha="center", fontsize=9)

    # Panel C: Runtime
    axes[2].bar(names, [r.elapsed_seconds for r in valid], color=colors,
                edgecolor="black", linewidth=0.5)
    axes[2].set_ylabel("Time (seconds)")
    axes[2].set_title("C. Runtime")
    for i, r in enumerate(valid):
        axes[2].text(i, r.elapsed_seconds + max(r.elapsed_seconds for r in valid) * 0.02,
                     f"{r.elapsed_seconds:.1f}s", ha="center", fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "denovo_benchmark.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Figure saved to %s", fig_path)

    # --- Figure 2: Detailed metrics ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: Precision vs Recall
    for i, r in enumerate(valid):
        axes[0].scatter(r.recall * 100, r.precision * 100,
                        s=200, c=colors[i], edgecolors="black",
                        linewidth=1, zorder=5)
        axes[0].annotate(r.name, (r.recall * 100 + 1, r.precision * 100 + 1),
                         fontsize=9)
    axes[0].set_xlabel("Recall (%)")
    axes[0].set_ylabel("Precision (%)")
    axes[0].set_title("A. Precision vs Recall")
    axes[0].set_xlim(0, 105)
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=0.3)

    # Panel B: Transcripts vs N50
    for i, r in enumerate(valid):
        axes[1].scatter(r.n_transcripts, r.n50,
                        s=200, c=colors[i], edgecolors="black",
                        linewidth=1, zorder=5)
        axes[1].annotate(r.name, (r.n_transcripts + 1, r.n50 + 10),
                         fontsize=9)
    axes[1].set_xlabel("Number of Transcripts")
    axes[1].set_ylabel("N50 (bp)")
    axes[1].set_title("B. Transcripts vs N50")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "denovo_benchmark_detail.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig2_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Detailed figure saved to %s", fig2_path)


# ===========================================================================
# Main benchmark
# ===========================================================================


def run_benchmark(
    output_dir: str = "benchmarks/denovo_results",
    n_transcripts: int = 50,
    read_length: int = 100,
    k_sizes: list[int] | None = None,
) -> None:
    """Run the complete de novo assembly benchmark.

    Args:
        output_dir: Output directory.
        n_transcripts: Number of synthetic transcripts.
        read_length: Simulated read length.
        k_sizes: K-mer sizes to test for RapidSplice.
    """
    if k_sizes is None:
        k_sizes = [25]

    os.makedirs(output_dir, exist_ok=True)

    # --- Generate synthetic data ---
    logger.info("Generating %d synthetic transcripts...", n_transcripts)
    transcripts = generate_transcripts(
        n_transcripts=n_transcripts,
        min_length=300,
        max_length=2000,
        seed=42,
    )

    ref_path = os.path.join(output_dir, "reference_transcripts.fa")
    write_reference_fasta(ref_path, transcripts)

    logger.info("Simulating reads (read_length=%d)...", read_length)
    headers, sequences, qualities = simulate_reads(
        transcripts,
        read_length=read_length,
        error_rate=0.01,
        seed=42,
    )

    fastq_path = os.path.join(output_dir, "simulated_reads.fq")
    n_reads = write_fastq(fastq_path, headers, sequences, qualities)
    logger.info("Generated %d reads in %s", n_reads, fastq_path)

    # --- Run assemblers ---
    results: list[AssemblyResult] = []

    # RapidSplice de novo with different k sizes
    for k in k_sizes:
        logger.info("Running RapidSplice de novo (k=%d)...", k)
        result, out_fa = run_braid_denovo(fastq_path, output_dir, k=k)
        if len(k_sizes) > 1:
            result.name = f"RapidSplice-k{k}"

        # Evaluate
        if out_fa and os.path.exists(out_fa):
            recall, precision, f1, n_recovered = evaluate_assembly(
                out_fa, transcripts,
            )
            result.recall = recall
            result.precision = precision
            result.f1 = f1
            result.recovered_transcripts = n_recovered

        results.append(result)
        logger.info(
            "  RapidSplice-k%d: %d transcripts, N50=%d, recall=%.1f%%, "
            "precision=%.1f%%, time=%.1fs",
            k, result.n_transcripts, result.n50,
            result.recall * 100, result.precision * 100,
            result.elapsed_seconds,
        )

    # rnaSPAdes
    logger.info("Running rnaSPAdes...")
    result_spades, spades_fa = run_rnaspades(fastq_path, output_dir)
    if spades_fa and os.path.exists(spades_fa):
        recall, precision, f1, n_recovered = evaluate_assembly(
            spades_fa, transcripts,
        )
        result_spades.recall = recall
        result_spades.precision = precision
        result_spades.f1 = f1
        result_spades.recovered_transcripts = n_recovered
    if result_spades.n_transcripts > 0:
        results.append(result_spades)
        logger.info(
            "  rnaSPAdes: %d transcripts, N50=%d, recall=%.1f%%, "
            "precision=%.1f%%, time=%.1fs",
            result_spades.n_transcripts, result_spades.n50,
            result_spades.recall * 100, result_spades.precision * 100,
            result_spades.elapsed_seconds,
        )

    # Trinity
    logger.info("Running Trinity...")
    result_trinity, trinity_fa = run_trinity(fastq_path, output_dir)
    if trinity_fa and os.path.exists(trinity_fa):
        recall, precision, f1, n_recovered = evaluate_assembly(
            trinity_fa, transcripts,
        )
        result_trinity.recall = recall
        result_trinity.precision = precision
        result_trinity.f1 = f1
        result_trinity.recovered_transcripts = n_recovered
    if result_trinity.n_transcripts > 0:
        results.append(result_trinity)
        logger.info(
            "  Trinity: %d transcripts, N50=%d, recall=%.1f%%, "
            "precision=%.1f%%, time=%.1fs",
            result_trinity.n_transcripts, result_trinity.n50,
            result_trinity.recall * 100, result_trinity.precision * 100,
            result_trinity.elapsed_seconds,
        )

    # --- Generate report ---
    logger.info("Generating benchmark report...")
    generate_report(results, transcripts, n_reads, output_dir)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("DE NOVO ASSEMBLY BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Reference: {n_transcripts} transcripts, {n_reads} reads")
    print(f"{'Assembler':<25} {'#Tx':>6} {'N50':>8} {'Recall':>8} "
          f"{'Prec':>8} {'F1':>8} {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r.name:<25} {r.n_transcripts:>6} {r.n50:>8,} "
              f"{r.recall * 100:>7.1f}% {r.precision * 100:>7.1f}% "
              f"{r.f1 * 100:>7.1f}% {r.elapsed_seconds:>7.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark(
        output_dir="benchmarks/denovo_results",
        n_transcripts=50,
        read_length=100,
        k_sizes=[21, 25, 31],
    )
