"""GTF and GFF3 output writers for assembled transcripts.

This module provides writers that convert internal transcript representations
into standard GTF2.2 and GFF3 annotation formats, following StringTie output
conventions. Coordinates are stored internally as 0-based half-open intervals
and converted to 1-based inclusive (GTF) or 1-based inclusive/half-open (GFF3)
on output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class TranscriptRecord:
    """A single assembled transcript with its exon structure and expression.

    All coordinates are stored in 0-based half-open format internally.
    Writers handle conversion to the appropriate output coordinate system.

    Attributes:
        transcript_id: Unique identifier for this transcript.
        gene_id: Identifier for the parent gene locus.
        chrom: Reference sequence name (chromosome / contig).
        strand: Genomic strand, either '+' or '-'.
        start: Transcript start position, 0-based inclusive.
        end: Transcript end position, 0-based exclusive.
        exons: List of (start, end) exon intervals in 0-based half-open
            coordinates. Exons are expected to be non-overlapping and sorted
            by start position.
        score: Assembly confidence or coverage score (default 0.0).
        fpkm: Fragments Per Kilobase of transcript per Million mapped
            fragments (default 0.0).
        tpm: Transcripts Per Million expression level (default 0.0).
        coverage: Average per-base read coverage across the transcript
            (default 0.0).
    """

    transcript_id: str
    gene_id: str
    chrom: str
    strand: str
    start: int
    end: int
    exons: list[tuple[int, int]] = field(default_factory=list)
    score: float = 0.0
    fpkm: float = 0.0
    tpm: float = 0.0
    coverage: float = 0.0
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None
    bootstrap_presence: float | None = None
    bootstrap_cv: float | None = None

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if self.strand not in ("+", "-"):
            raise ValueError(
                f"strand must be '+' or '-', got {self.strand!r}"
            )
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(
                f"end ({self.end}) must be greater than start ({self.start})"
            )
        for i, (estart, eend) in enumerate(self.exons):
            if estart < self.start or eend > self.end:
                raise ValueError(
                    f"exon {i} [{estart}, {eend}) falls outside transcript "
                    f"bounds [{self.start}, {self.end})"
                )
            if eend <= estart:
                raise ValueError(
                    f"exon {i} end ({eend}) must be > start ({estart})"
                )

    @property
    def transcript_length(self) -> int:
        """Return the total spliced length of the transcript in bases."""
        if not self.exons:
            return self.end - self.start
        return sum(eend - estart for estart, eend in self.exons)


def _format_score(value: float) -> str:
    """Format a floating-point score for GTF/GFF3 output.

    Produces up to six decimal places, stripping unnecessary trailing zeros
    while always keeping at least one decimal digit (e.g. ``1000.0`` not
    ``1000``).

    Args:
        value: The numeric value to format.

    Returns:
        The formatted string representation.
    """
    if value == 0.0:
        return "0.000000"
    formatted = f"{value:.6f}"
    # Strip trailing zeros but keep at least one decimal place.
    if "." in formatted:
        formatted = formatted.rstrip("0")
        if formatted.endswith("."):
            formatted += "0"
    return formatted


class GtfWriter:
    """Write assembled transcripts in GTF2.2 format.

    Follows StringTie GTF output conventions:
    - Feature types ``transcript`` and ``exon``.
    - Attributes include ``gene_id``, ``transcript_id``, ``cov``, ``FPKM``,
      ``TPM`` on transcript lines, and ``gene_id``, ``transcript_id``,
      ``exon_number`` on exon lines.
    - Coordinates are 1-based inclusive on both ends.

    Example::

        writer = GtfWriter("output.gtf")
        writer.write_transcripts(transcript_list)
    """

    def __init__(self, output_path: str, source: str = "RapidSplice") -> None:
        """Initialize the GTF writer.

        Args:
            output_path: Filesystem path for the output GTF file.
            source: Value for the *source* column (column 2). Defaults to
                ``"RapidSplice"``.
        """
        self.output_path: str = output_path
        self.source: str = source

    def write_transcripts(self, transcripts: list[TranscriptRecord]) -> None:
        """Write all transcripts to the GTF file.

        Transcripts are sorted lexicographically by chromosome and then by
        start position before writing. A header comment is emitted first.

        Args:
            transcripts: The transcript records to write.
        """
        sorted_transcripts = sorted(
            transcripts, key=lambda t: (t.chrom, t.start)
        )
        with open(self.output_path, "w", encoding="utf-8") as fh:
            fh.write(
                f"# GTF file produced by {self.source}\n"
            )
            for record in sorted_transcripts:
                self._write_transcript_entry(fh, record)

    def _write_transcript_entry(
        self, fh: TextIO, record: TranscriptRecord
    ) -> None:
        """Write a single transcript and its exon features.

        The method emits one ``transcript`` line followed by one ``exon`` line
        per exon. Exons are numbered starting from 1 in genomic order
        (ascending by start coordinate regardless of strand).

        Args:
            fh: An open, writable text file handle.
            record: The transcript record to write.
        """
        # Convert 0-based half-open to 1-based inclusive.
        gtf_start = record.start + 1
        gtf_end = record.end  # half-open end == inclusive end in 1-based

        # Build transcript-level attributes.
        transcript_attrs = (
            f'gene_id "{record.gene_id}"; '
            f'transcript_id "{record.transcript_id}"; '
            f'cov "{_format_score(record.coverage)}"; '
            f'FPKM "{_format_score(record.fpkm)}"; '
            f'TPM "{_format_score(record.tpm)}";'
        )
        if record.bootstrap_ci_low is not None:
            transcript_attrs += (
                f' bootstrap_ci_low "{record.bootstrap_ci_low:.2f}";'
                f' bootstrap_ci_high "{record.bootstrap_ci_high:.2f}";'
                f' bootstrap_presence "{record.bootstrap_presence:.3f}";'
                f' bootstrap_cv "{record.bootstrap_cv:.3f}";'
            )

        # Transcript line.
        fh.write(
            f"{record.chrom}\t"
            f"{self.source}\t"
            f"transcript\t"
            f"{gtf_start}\t"
            f"{gtf_end}\t"
            f"{_format_score(record.score)}\t"
            f"{record.strand}\t"
            f".\t"
            f"{transcript_attrs}\n"
        )

        # Sort exons by start position (ascending in genomic coordinates).
        sorted_exons = sorted(record.exons, key=lambda e: e[0])

        for exon_num, (estart, eend) in enumerate(sorted_exons, start=1):
            exon_gtf_start = estart + 1
            exon_gtf_end = eend  # half-open to inclusive

            exon_attrs = (
                f'gene_id "{record.gene_id}"; '
                f'transcript_id "{record.transcript_id}"; '
                f'exon_number "{exon_num}";'
            )

            fh.write(
                f"{record.chrom}\t"
                f"{self.source}\t"
                f"exon\t"
                f"{exon_gtf_start}\t"
                f"{exon_gtf_end}\t"
                f"{_format_score(record.score)}\t"
                f"{record.strand}\t"
                f".\t"
                f"{exon_attrs}\n"
            )


class Gff3Writer:
    """Write assembled transcripts in GFF3 format.

    Follows the GFF3 specification (Sequence Ontology):
    - Directives: ``##gff-version 3`` header.
    - Feature types ``mRNA`` for transcripts, ``exon`` for exon features.
    - Parent/ID relationships encoded in the ninth column.
    - Coordinates are 1-based inclusive on both ends (same as GTF).

    Example::

        writer = Gff3Writer("output.gff3")
        writer.write_transcripts(transcript_list)
    """

    def __init__(self, output_path: str, source: str = "RapidSplice") -> None:
        """Initialize the GFF3 writer.

        Args:
            output_path: Filesystem path for the output GFF3 file.
            source: Value for the *source* column (column 2). Defaults to
                ``"RapidSplice"``.
        """
        self.output_path: str = output_path
        self.source: str = source

    def write_transcripts(self, transcripts: list[TranscriptRecord]) -> None:
        """Write all transcripts to the GFF3 file.

        Emits gene, mRNA, and exon features. Transcripts are sorted
        lexicographically by chromosome and then by start position. A
        ``##gff-version 3`` directive is written as the first line.

        Args:
            transcripts: The transcript records to write.
        """
        sorted_transcripts = sorted(
            transcripts, key=lambda t: (t.chrom, t.start)
        )

        # Collect unique gene_ids to emit gene features.
        gene_spans: dict[str, _GeneSpan] = {}
        for rec in sorted_transcripts:
            if rec.gene_id not in gene_spans:
                gene_spans[rec.gene_id] = _GeneSpan(
                    chrom=rec.chrom,
                    strand=rec.strand,
                    start=rec.start,
                    end=rec.end,
                )
            else:
                span = gene_spans[rec.gene_id]
                span.start = min(span.start, rec.start)
                span.end = max(span.end, rec.end)

        with open(self.output_path, "w", encoding="utf-8") as fh:
            fh.write("##gff-version 3\n")
            fh.write(f"# GFF3 file produced by {self.source}\n")

            emitted_genes: set[str] = set()

            for record in sorted_transcripts:
                # Emit the gene feature once per gene_id.
                if record.gene_id not in emitted_genes:
                    emitted_genes.add(record.gene_id)
                    span = gene_spans[record.gene_id]
                    gene_start = span.start + 1
                    gene_end = span.end
                    gene_attrs = (
                        f"ID={_gff3_encode(record.gene_id)};"
                        f"Name={_gff3_encode(record.gene_id)}"
                    )
                    fh.write(
                        f"{span.chrom}\t"
                        f"{self.source}\t"
                        f"gene\t"
                        f"{gene_start}\t"
                        f"{gene_end}\t"
                        f".\t"
                        f"{span.strand}\t"
                        f".\t"
                        f"{gene_attrs}\n"
                    )

                self._write_transcript_entry(fh, record)

    def _write_transcript_entry(
        self, fh: TextIO, record: TranscriptRecord
    ) -> None:
        """Write a single transcript (mRNA) and its exon features in GFF3.

        Args:
            fh: An open, writable text file handle.
            record: The transcript record to write.
        """
        gff_start = record.start + 1
        gff_end = record.end

        mrna_attrs = (
            f"ID={_gff3_encode(record.transcript_id)};"
            f"Parent={_gff3_encode(record.gene_id)};"
            f"Name={_gff3_encode(record.transcript_id)};"
            f"coverage={_format_score(record.coverage)};"
            f"FPKM={_format_score(record.fpkm)};"
            f"TPM={_format_score(record.tpm)}"
        )

        fh.write(
            f"{record.chrom}\t"
            f"{self.source}\t"
            f"mRNA\t"
            f"{gff_start}\t"
            f"{gff_end}\t"
            f"{_format_score(record.score)}\t"
            f"{record.strand}\t"
            f".\t"
            f"{mrna_attrs}\n"
        )

        sorted_exons = sorted(record.exons, key=lambda e: e[0])

        for exon_num, (estart, eend) in enumerate(sorted_exons, start=1):
            exon_gff_start = estart + 1
            exon_gff_end = eend

            exon_id = f"{record.transcript_id}.exon{exon_num}"
            exon_attrs = (
                f"ID={_gff3_encode(exon_id)};"
                f"Parent={_gff3_encode(record.transcript_id)}"
            )

            fh.write(
                f"{record.chrom}\t"
                f"{self.source}\t"
                f"exon\t"
                f"{exon_gff_start}\t"
                f"{exon_gff_end}\t"
                f"{_format_score(record.score)}\t"
                f"{record.strand}\t"
                f".\t"
                f"{exon_attrs}\n"
            )


@dataclass
class _GeneSpan:
    """Internal helper to track the genomic extent of a gene locus.

    Attributes:
        chrom: Chromosome name.
        strand: Genomic strand.
        start: Minimum start across all transcripts (0-based).
        end: Maximum end across all transcripts (0-based exclusive).
    """

    chrom: str
    strand: str
    start: int
    end: int


def _gff3_encode(value: str) -> str:
    """Percent-encode reserved characters for GFF3 attribute values.

    GFF3 reserves ``; = & ,`` and control characters. This function encodes
    those characters using standard percent-encoding (RFC 3986 style).

    Args:
        value: The raw string to encode.

    Returns:
        The encoded string safe for use in GFF3 column 9.
    """
    replacements = {
        "%": "%25",  # Must be first to avoid double-encoding.
        ";": "%3B",
        "=": "%3D",
        "&": "%26",
        ",": "%2C",
        "\t": "%09",
        "\n": "%0A",
        "\r": "%0D",
    }
    result = value
    for char, encoded in replacements.items():
        result = result.replace(char, encoded)
    return result


def compute_expression(
    transcripts: list[TranscriptRecord],
    total_mapped_reads: int,
    read_length: int = 100,
) -> None:
    """Compute FPKM and TPM for all transcripts in-place.

    FPKM (Fragments Per Kilobase of transcript per Million mapped fragments)
    is calculated as::

        FPKM_i = (coverage_i * transcript_length_i / read_length) /
                 (total_mapped_reads / 1e6) / (transcript_length_i / 1e3)

    which simplifies to::

        FPKM_i = (coverage_i * 1e9) / (total_mapped_reads * read_length)

    TPM (Transcripts Per Million) normalizes per-base coverage rates so that
    they sum to 1 million across all transcripts::

        rate_i = coverage_i / read_length
        TPM_i  = (rate_i / sum(rate_j for all j)) * 1e6

    If *total_mapped_reads* is zero or there are no transcripts, all values
    are set to ``0.0``.

    Args:
        transcripts: List of transcript records to update. The ``fpkm`` and
            ``tpm`` fields are modified in-place.
        total_mapped_reads: Total number of mapped reads (or fragments) in
            the experiment.
        read_length: Average sequencing read length in bases. Defaults to
            ``100``.

    Raises:
        ValueError: If *total_mapped_reads* is negative or *read_length* is
            not positive.
    """
    if total_mapped_reads < 0:
        raise ValueError(
            f"total_mapped_reads must be >= 0, got {total_mapped_reads}"
        )
    if read_length <= 0:
        raise ValueError(f"read_length must be > 0, got {read_length}")

    if not transcripts or total_mapped_reads == 0:
        for rec in transcripts:
            rec.fpkm = 0.0
            rec.tpm = 0.0
        return

    # Compute FPKM for each transcript.
    # FPKM_i = (read_count_i * 1e9) / (total_mapped_reads * length_i)
    # where read_count_i = coverage_i * length_i / read_length
    # so FPKM_i = coverage_i * 1e9 / (total_mapped_reads * read_length)
    for rec in transcripts:
        length = rec.transcript_length
        if length <= 0:
            rec.fpkm = 0.0
            continue
        read_count = rec.coverage * length / read_length
        rec.fpkm = (read_count * 1.0e9) / (total_mapped_reads * length)

    # Compute TPM.
    # rate_i = coverage_i / read_length  (proportional to expression)
    # TPM_i  = rate_i / sum_of_rates * 1e6
    rates: list[float] = []
    for rec in transcripts:
        length = rec.transcript_length
        if length <= 0:
            rates.append(0.0)
            continue
        rates.append(rec.coverage / read_length)

    total_rate = sum(rates)

    if total_rate == 0.0:
        for rec in transcripts:
            rec.tpm = 0.0
        return

    for rec, rate in zip(transcripts, rates):
        rec.tpm = (rate / total_rate) * 1.0e6
