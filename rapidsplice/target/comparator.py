"""Reference isoform comparison for targeted assembly.

Classifies each assembled isoform against known reference transcripts
(e.g., GENCODE) as exact match, novel junction combination, novel
junction, or novel exon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IsoformClassification:
    """Classification of an assembled isoform against a reference.

    Attributes:
        category: One of 'exact_match', 'novel_combination',
            'novel_junction', 'novel_exon', 'single_exon'.
        matched_transcript_id: Reference transcript ID for exact matches.
        matched_gene: Reference gene name.
        n_known_junctions: Number of junctions found in reference.
        n_novel_junctions: Number of junctions not in reference.
        description: Human-readable classification description.
    """

    category: str
    matched_transcript_id: str | None = None
    matched_gene: str | None = None
    n_known_junctions: int = 0
    n_novel_junctions: int = 0
    description: str = ""


def _exons_to_intron_chain(
    exons: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Convert sorted exon list to intron chain (donor, acceptor pairs)."""
    introns = []
    for i in range(len(exons) - 1):
        donor = exons[i][1]
        acceptor = exons[i + 1][0]
        introns.append((donor, acceptor))
    return introns


def classify_isoform(
    exons: list[tuple[int, int]],
    ref_transcripts: list[dict],
    tolerance: int = 0,
) -> IsoformClassification:
    """Classify an assembled isoform against reference transcripts.

    Args:
        exons: Assembled isoform exon coordinates [(start, end), ...].
        ref_transcripts: List of reference transcript dicts with keys
            'exons' (list of (start,end)), 'transcript_id', 'gene_name'.
        tolerance: Coordinate matching tolerance in bp.

    Returns:
        Classification result.
    """
    if len(exons) <= 1:
        return IsoformClassification(
            category="single_exon",
            description="Single-exon transcript (no intron chain to compare)",
        )

    query_introns = _exons_to_intron_chain(exons)

    # Collect all known junctions from reference
    all_ref_junctions: set[tuple[int, int]] = set()
    ref_intron_chains: list[tuple[list[tuple[int, int]], dict]] = []

    for ref_tx in ref_transcripts:
        ref_exons = ref_tx["exons"]
        if len(ref_exons) < 2:
            continue
        ref_introns = _exons_to_intron_chain(ref_exons)
        all_ref_junctions.update(ref_introns)
        ref_intron_chains.append((ref_introns, ref_tx))

    # Check for exact intron chain match (terminal exon boundaries may differ)
    for ref_introns, ref_tx in ref_intron_chains:
        if _chains_match(query_introns, ref_introns, tolerance):
            return IsoformClassification(
                category="exact_match",
                matched_transcript_id=ref_tx.get("transcript_id"),
                matched_gene=ref_tx.get("gene_name"),
                n_known_junctions=len(query_introns),
                n_novel_junctions=0,
                description=f"Exact match to {ref_tx.get('transcript_id', '?')}",
            )
    # Also check with small tolerance for near-exact matches
    if tolerance == 0:
        for ref_introns, ref_tx in ref_intron_chains:
            if _chains_match(query_introns, ref_introns, 5):
                return IsoformClassification(
                    category="exact_match",
                    matched_transcript_id=ref_tx.get("transcript_id"),
                    matched_gene=ref_tx.get("gene_name"),
                    n_known_junctions=len(query_introns),
                    n_novel_junctions=0,
                    description=(
                        f"Near-exact match to "
                        f"{ref_tx.get('transcript_id', '?')} (≤5bp)"
                    ),
                )

    # Check if query is a reference chain with extra terminal introns
    # (flanking noise). If all ref introns appear as a contiguous
    # subsequence of query introns, classify as exact_match.
    for ref_introns, ref_tx in ref_intron_chains:
        if len(ref_introns) < 2:
            continue
        ref_set = set(ref_introns)
        # Check if all ref introns are in query (subset match)
        matched = sum(1 for j in query_introns if j in ref_set)
        if matched == len(ref_introns) and matched >= len(query_introns) - 2:
            return IsoformClassification(
                category="exact_match",
                matched_transcript_id=ref_tx.get("transcript_id"),
                matched_gene=ref_tx.get("gene_name"),
                n_known_junctions=matched,
                n_novel_junctions=len(query_introns) - matched,
                description=(
                    f"Superset match to "
                    f"{ref_tx.get('transcript_id', '?')} "
                    f"({matched}/{len(query_introns)} introns)"
                ),
            )

    # Count known vs novel junctions
    n_known = 0
    n_novel = 0
    for junction in query_introns:
        if _junction_in_set(junction, all_ref_junctions, tolerance):
            n_known += 1
        else:
            n_novel += 1

    if n_novel == 0:
        return IsoformClassification(
            category="novel_combination",
            n_known_junctions=n_known,
            n_novel_junctions=0,
            description=(
                f"Novel combination of {n_known} known junctions"
            ),
        )

    # Check for novel exons (exons not overlapping any reference exon)
    all_ref_exons: list[tuple[int, int]] = []
    for ref_tx in ref_transcripts:
        all_ref_exons.extend(ref_tx["exons"])

    novel_exon_count = 0
    for estart, eend in exons:
        if not _overlaps_any(estart, eend, all_ref_exons):
            novel_exon_count += 1

    if novel_exon_count > 0:
        return IsoformClassification(
            category="novel_exon",
            n_known_junctions=n_known,
            n_novel_junctions=n_novel,
            description=(
                f"{novel_exon_count} novel exon(s), "
                f"{n_novel} novel junction(s)"
            ),
        )

    return IsoformClassification(
        category="novel_junction",
        n_known_junctions=n_known,
        n_novel_junctions=n_novel,
        description=f"{n_novel} novel junction(s), {n_known} known",
    )


def classify_all_isoforms(
    isoform_exons: list[list[tuple[int, int]]],
    gtf_path: str,
    region_chrom: str,
    region_start: int,
    region_end: int,
    tolerance: int = 0,
) -> list[IsoformClassification]:
    """Classify all assembled isoforms against a GTF reference.

    Args:
        isoform_exons: List of exon lists for each assembled isoform.
        gtf_path: Path to reference GTF annotation.
        region_chrom: Target chromosome.
        region_start: Target region start.
        region_end: Target region end.
        tolerance: Coordinate matching tolerance.

    Returns:
        List of classifications, one per isoform.
    """
    ref_transcripts = _load_reference_transcripts(
        gtf_path, region_chrom, region_start, region_end,
    )

    logger.info(
        "Loaded %d reference transcripts for comparison in %s:%d-%d",
        len(ref_transcripts), region_chrom, region_start, region_end,
    )

    return [
        classify_isoform(exons, ref_transcripts, tolerance)
        for exons in isoform_exons
    ]


def _load_reference_transcripts(
    gtf_path: str,
    chrom: str,
    start: int,
    end: int,
) -> list[dict]:
    """Load reference transcripts overlapping a region from GTF."""
    tx_data: dict[str, dict] = {}

    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[0] != chrom:
                continue
            if fields[2] not in ("exon", "transcript"):
                continue

            feat_start = int(fields[3]) - 1
            feat_end = int(fields[4])

            attrs = fields[8]
            tid = _parse_attr(attrs, "transcript_id")
            if not tid:
                continue

            if fields[2] == "transcript":
                # Check overlap with region
                if feat_end < start or feat_start > end:
                    continue
                gname = _parse_attr(attrs, "gene_name")
                tx_data[tid] = {
                    "transcript_id": tid,
                    "gene_name": gname,
                    "exons": [],
                }
            elif fields[2] == "exon":
                if tid in tx_data:
                    tx_data[tid]["exons"].append((feat_start, feat_end))

    # Sort exons
    for tx in tx_data.values():
        tx["exons"].sort(key=lambda e: e[0])

    return list(tx_data.values())


def _chains_match(
    chain_a: list[tuple[int, int]],
    chain_b: list[tuple[int, int]],
    tolerance: int,
) -> bool:
    """Check if two intron chains match within tolerance."""
    if len(chain_a) != len(chain_b):
        return False
    for (a_donor, a_acc), (b_donor, b_acc) in zip(chain_a, chain_b):
        if abs(a_donor - b_donor) > tolerance or abs(a_acc - b_acc) > tolerance:
            return False
    return True


def _junction_in_set(
    junction: tuple[int, int],
    ref_set: set[tuple[int, int]],
    tolerance: int,
) -> bool:
    """Check if a junction matches any in the reference set."""
    if tolerance == 0:
        return junction in ref_set
    donor, acceptor = junction
    for rd, ra in ref_set:
        if abs(donor - rd) <= tolerance and abs(acceptor - ra) <= tolerance:
            return True
    return False


def _overlaps_any(
    start: int,
    end: int,
    intervals: list[tuple[int, int]],
) -> bool:
    """Check if an interval overlaps any in a list."""
    for istart, iend in intervals:
        if start < iend and end > istart:
            return True
    return False


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
