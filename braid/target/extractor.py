"""Target region read extraction and gene coordinate lookup.

Extracts reads from a BAM file that map to a specified gene or genomic
region, including flanking regions to capture reads that span into
adjacent intergenic space.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pysam

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TargetRegion:
    """A genomic region of interest for targeted assembly.

    Attributes:
        chrom: Chromosome name.
        start: 0-based start coordinate.
        end: 0-based exclusive end coordinate.
        strand: Strand ('+', '-', or '.' for unstranded).
        gene_name: Optional gene name for display.
        gene_id: Optional gene identifier.
    """

    chrom: str
    start: int
    end: int
    strand: str = "."
    gene_name: str | None = None
    gene_id: str | None = None

    @property
    def length(self) -> int:
        """Region length in base pairs."""
        return self.end - self.start

    def with_flank(self, flank: int) -> TargetRegion:
        """Return a new region expanded by *flank* bp on each side."""
        return TargetRegion(
            chrom=self.chrom,
            start=max(0, self.start - flank),
            end=self.end + flank,
            strand=self.strand,
            gene_name=self.gene_name,
            gene_id=self.gene_id,
        )


def lookup_gene(
    gtf_path: str,
    gene_name: str,
) -> TargetRegion | None:
    """Look up a gene's coordinates from a GTF annotation file.

    Searches for a gene feature matching the given name (case-insensitive)
    in the ``gene_name`` or ``gene_id`` GTF attributes.

    Args:
        gtf_path: Path to the GTF annotation file.
        gene_name: Gene symbol or ID to search for.

    Returns:
        A :class:`TargetRegion` for the gene, or ``None`` if not found.
    """
    query = gene_name.upper()

    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != "gene":
                continue

            attrs = fields[8]
            gname = _parse_attr(attrs, "gene_name")
            gid = _parse_attr(attrs, "gene_id")

            if (gname and gname.upper() == query) or (gid and gid.upper() == query):
                chrom = fields[0]
                start = int(fields[3]) - 1  # GTF 1-based → 0-based
                end = int(fields[4])
                strand = fields[6] if fields[6] in ("+", "-") else "."
                return TargetRegion(
                    chrom=chrom,
                    start=start,
                    end=end,
                    strand=strand,
                    gene_name=gname or gene_name,
                    gene_id=gid,
                )

    logger.warning("Gene %r not found in %s", gene_name, gtf_path)
    return None


def parse_region_string(region: str) -> TargetRegion:
    """Parse a UCSC-style region string into a TargetRegion.

    Accepts formats:
        ``chr17:7668402-7687538``
        ``17:7668402-7687538``

    Args:
        region: Region string.

    Returns:
        Parsed :class:`TargetRegion`.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        chrom, coords = region.rsplit(":", 1)
        start_str, end_str = coords.replace(",", "").split("-")
        return TargetRegion(
            chrom=chrom,
            start=int(start_str) - 1,  # Assume 1-based input
            end=int(end_str),
        )
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"Cannot parse region string {region!r}. "
            f"Expected format: chrom:start-end (e.g. chr17:7668402-7687538)"
        ) from exc


@dataclass
class ExtractionStats:
    """Statistics from read extraction."""

    total_reads: int = 0
    spliced_reads: int = 0
    unique_junctions: int = 0
    mean_coverage: float = 0.0


def extract_target_reads(
    bam_path: str,
    region: TargetRegion,
    min_mapq: int = 0,
) -> tuple[list[pysam.AlignedSegment], ExtractionStats]:
    """Extract all reads overlapping a target region from a BAM file.

    Args:
        bam_path: Path to indexed BAM file.
        region: Target region to extract reads from.
        min_mapq: Minimum mapping quality filter.

    Returns:
        Tuple of (list of aligned reads, extraction statistics).
    """
    reads: list[pysam.AlignedSegment] = []
    stats = ExtractionStats()
    junctions: set[tuple[int, int]] = set()

    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for read in bam.fetch(region.chrom, region.start, region.end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.mapping_quality < min_mapq:
                continue

            reads.append(read)
            stats.total_reads += 1

            # Count splice junctions
            if read.cigartuples:
                ref_pos = read.reference_start
                for op, length in read.cigartuples:
                    if op == 3:  # N = intron
                        junctions.add((ref_pos, ref_pos + length))
                        stats.spliced_reads += 1
                    if op in (0, 2, 3, 7, 8):  # M, D, N, =, X
                        ref_pos += length

    stats.unique_junctions = len(junctions)

    # Estimate mean coverage
    if reads and region.length > 0:
        total_bases = sum(
            read.query_alignment_length or 0 for read in reads
        )
        stats.mean_coverage = total_bases / region.length

    logger.info(
        "Target extraction: %s %s:%d-%d — %d reads, %d spliced, "
        "%d junctions, %.1fx coverage",
        region.gene_name or "",
        region.chrom,
        region.start,
        region.end,
        stats.total_reads,
        stats.spliced_reads,
        stats.unique_junctions,
        stats.mean_coverage,
    )

    return reads, stats


def _parse_attr(attrs: str, key: str) -> str | None:
    """Extract an attribute value from a GTF attribute string."""
    for part in attrs.split(";"):
        part = part.strip()
        if not part:
            continue
        tokens = part.split(None, 1)
        if len(tokens) == 2 and tokens[0] == key:
            return tokens[1].strip('"').strip("'")
    return None
