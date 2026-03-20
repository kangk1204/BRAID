"""Splice graph builder for GPU-accelerated RNA-seq transcript assembly.

Constructs splice graphs from aligned RNA-seq reads. Takes ReadData and
JunctionEvidence produced by the BAM reader and builds SpliceGraph objects
suitable for downstream flow decomposition on the GPU.

The builder implements a three-phase pipeline:

1. **Locus identification** -- Junctions are clustered into independent gene
   loci using a union-find strategy on overlapping junction spans.
2. **Exon boundary determination** -- Junction donor/acceptor coordinates
   define the boundaries between exonic regions.
3. **Graph construction** -- Exon nodes are connected by intron (junction)
   edges and continuation edges, with virtual SOURCE and SINK nodes to
   guarantee a single-entry, single-exit DAG.

The algorithm follows the splice graph formulation described in Pertea et al.
(StringTie, Nature Biotechnology 2015) and Shao & Kingsford (Scallop, Nature
Biotechnology 2017), adapted for batch GPU processing.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from rapidsplice.graph.splice_graph import (
    BatchedCSRGraphs,
    EdgeType,
    NodeType,
    SpliceGraph,
)
from rapidsplice.io.bam_reader import JunctionEvidence, ReadData
from rapidsplice.utils.cigar import CIGAR_D, CIGAR_EQ, CIGAR_M, CIGAR_N, CIGAR_X
from rapidsplice.utils.interval import compute_coverage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LocusDefinition:
    """Definition of a gene locus derived from clustered splice junctions.

    A locus is a contiguous genomic region containing one or more splice
    junctions that have been grouped together because their exonic spans
    overlap.  Each locus is processed independently during graph construction.

    Attributes:
        chrom: Chromosome / contig name.
        start: 0-based start coordinate of the locus (inclusive).
        end: 0-based end coordinate of the locus (exclusive).
        strand: Strand of the locus (``'+'``, ``'-'``, or ``'.'`` for unknown).
        junction_indices: Indices into the :class:`JunctionEvidence` arrays
            for the junctions belonging to this locus.
    """

    chrom: str
    start: int
    end: int
    strand: str
    junction_indices: list[int]


@dataclass(slots=True)
class GraphBuilderConfig:
    """Configuration parameters for splice graph construction.

    Attributes:
        locus_flank: Number of bases to extend on both sides of each junction
            span when clustering loci from junction-only evidence. This must
            be large enough to bridge typical internal exon lengths; too small
            fragments one transcript into multiple loci.
        min_junction_support: Minimum number of reads required to support a
            splice junction before it is included in the graph.
        min_exon_coverage: Minimum average read coverage across an exon for
            it to be retained.
        min_intron_length: Minimum length of an intron (junction span) in
            base pairs.  Junctions shorter than this are treated as deletions
            and discarded.
        max_intron_length: Maximum intron length.  Junctions exceeding this
            are likely alignment artefacts and are discarded.
        min_exon_length: Minimum exon length in base pairs.  Exon intervals
            shorter than this threshold are dropped.
        coverage_resolution: Base-pair resolution for coverage computation.
            A value of 1 means per-base resolution; higher values reduce
            memory usage for very large loci.
        merge_distance: Merge exon boundaries that are within this many
            base pairs of each other.  Set to 0 to disable merging.
        terminal_coverage_dropoff: Fraction of peak coverage at which the
            terminal exon boundary is placed.  Higher values produce
            shorter terminal exons.
        max_terminal_extension: Maximum base-pair extension from the
            outermost splice site to the terminal exon boundary.
        min_relative_exon_coverage: Minimum coverage of an exon expressed
            as a fraction of the locus maximum exon coverage.  Exons below
            this threshold are removed, breaking inter-gene bridges.
    """

    locus_flank: int = 800
    min_junction_support: int = 3
    min_exon_coverage: float = 1.0
    min_intron_length: int = 50
    max_intron_length: int = 500_000
    min_exon_length: int = 10
    coverage_resolution: int = 1
    merge_distance: int = 0
    junction_merge_distance: int = 10
    min_relative_junction_support: float = 0.05
    terminal_coverage_dropoff: float = 0.30
    max_terminal_extension: int = 5_000
    min_relative_exon_coverage: float = 0.01
    assign_ambiguous_junctions_to_dominant_strand: bool = True
    add_fallback_terminal_edges: bool = True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _merge_nearby_junctions(
    j_starts: list[int],
    j_ends: list[int],
    j_counts: list[float],
    merge_dist: int = 20,
) -> tuple[list[int], list[int], list[float]]:
    """Merge junctions with nearby donor or acceptor sites.

    When multiple junctions share similar donor (or acceptor) positions
    within ``merge_dist`` bp, keep only the highest-support one per cluster.
    This eliminates alignment artefacts that create false exon boundaries.

    Args:
        j_starts: Junction donor positions.
        j_ends: Junction acceptor positions.
        j_counts: Junction read support counts.
        merge_dist: Maximum distance to consider as same splice site.

    Returns:
        Filtered (starts, ends, counts) with nearby sites merged.
    """
    if len(j_starts) <= 1:
        return j_starts, j_ends, j_counts

    n = len(j_starts)
    keep = [True] * n

    # --- Merge donors (junction starts) sharing similar acceptors ---
    # Cluster donors pointing to nearby acceptors
    # Sort by start position
    indices = sorted(range(n), key=lambda i: (j_ends[i], j_starts[i]))

    # Cluster by acceptor proximity
    acc_clusters: list[list[int]] = []
    cur_cluster: list[int] = [indices[0]]
    for k in range(1, n):
        if abs(j_ends[indices[k]] - j_ends[cur_cluster[-1]]) <= merge_dist:
            cur_cluster.append(indices[k])
        else:
            acc_clusters.append(cur_cluster)
            cur_cluster = [indices[k]]
    acc_clusters.append(cur_cluster)

    # Within each acceptor cluster, sub-cluster by donor proximity
    for cluster in acc_clusters:
        if len(cluster) <= 1:
            continue
        cluster.sort(key=lambda i: j_starts[i])
        sub: list[list[int]] = [[cluster[0]]]
        for k in range(1, len(cluster)):
            if abs(j_starts[cluster[k]] - j_starts[sub[-1][-1]]) <= merge_dist:
                sub[-1].append(cluster[k])
            else:
                sub.append([cluster[k]])
        for group in sub:
            if len(group) <= 1:
                continue
            best = max(group, key=lambda i: j_counts[i])
            for i in group:
                if i != best:
                    keep[i] = False

    return (
        [j_starts[i] for i in range(n) if keep[i]],
        [j_ends[i] for i in range(n) if keep[i]],
        [j_counts[i] for i in range(n) if keep[i]],
    )


def _filter_relative_support(
    j_starts: list[int],
    j_ends: list[int],
    j_counts: list[float],
    min_relative: float = 0.01,
) -> tuple[list[int], list[int], list[float]]:
    """Remove junctions with support far below the locus maximum.

    Args:
        j_starts: Junction donor positions.
        j_ends: Junction acceptor positions.
        j_counts: Junction read support counts.
        min_relative: Minimum fraction of max count to retain.

    Returns:
        Filtered (starts, ends, counts).
    """
    if not j_counts:
        return j_starts, j_ends, j_counts
    max_count = max(j_counts)
    threshold = max_count * min_relative
    keep = [i for i in range(len(j_counts)) if j_counts[i] >= threshold]
    return (
        [j_starts[i] for i in keep],
        [j_ends[i] for i in keep],
        [j_counts[i] for i in keep],
    )


def _compute_terminal_boundary(
    read_data: "ReadData",
    chrom_id: int,
    anchor_pos: int,
    direction: str,
    dropoff_threshold: float = 0.10,
    max_extend: int = 10_000,
) -> int:
    """Determine terminal exon boundary using coverage drop-off.

    Instead of using the outermost read position, walk outward from the
    anchor (first donor or last acceptor) and find where coverage drops
    below ``dropoff_threshold`` fraction of the peak.

    Args:
        read_data: Read alignment data.
        chrom_id: Chromosome ID.
        anchor_pos: The splice site position to walk from.
        direction: ``"left"`` (toward 5') or ``"right"`` (toward 3').
        dropoff_threshold: Fraction of peak coverage for boundary.
        max_extend: Maximum extension in bp from anchor.

    Returns:
        The boundary position.
    """
    if read_data.n_reads == 0:
        if direction == "left":
            return max(0, anchor_pos - 100)
        return anchor_pos + 100

    mask = read_data.chrom_ids == chrom_id
    positions = read_data.positions[mask]
    end_positions = read_data.end_positions[mask]

    if len(positions) == 0:
        if direction == "left":
            return max(0, anchor_pos - 100)
        return anchor_pos + 100

    if direction == "left":
        region_start = max(int(np.min(positions)), anchor_pos - max_extend)
        region_end = anchor_pos
    else:
        region_start = anchor_pos
        region_end = min(int(np.max(end_positions)), anchor_pos + max_extend)

    region_len = region_end - region_start
    if region_len <= 0:
        return anchor_pos

    # Compute coverage in bins of 10bp for efficiency
    bin_size = max(1, min(10, region_len // 10))
    n_bins = max(1, region_len // bin_size)
    coverage = np.zeros(n_bins, dtype=np.int32)

    # Build coverage from aligned reference blocks only, so introns (CIGAR N)
    # do not inflate terminal boundary coverage.
    read_indices = np.where(mask)[0]
    for ridx in read_indices:
        pos = int(read_data.positions[ridx])
        cig_start = int(read_data.cigar_offsets[ridx])
        cig_end = int(read_data.cigar_offsets[ridx + 1])

        for ci in range(cig_start, cig_end):
            op = int(read_data.cigar_ops[ci])
            length = int(read_data.cigar_lens[ci])

            if op in (CIGAR_M, CIGAR_EQ, CIGAR_X):
                block_start = max(pos, region_start)
                block_end = min(pos + length, region_end)
                if block_start < block_end:
                    s = max(0, (block_start - region_start) // bin_size)
                    e = min(n_bins, (block_end - region_start) // bin_size + 1)
                    if s < n_bins and e > 0:
                        coverage[max(0, s):min(n_bins, e)] += 1
                pos += length
            elif op in (CIGAR_D, CIGAR_N):
                pos += length

    if np.max(coverage) == 0:
        return anchor_pos

    # Find peak near anchor
    anchor_bin = min(n_bins - 1, max(0, (anchor_pos - region_start) // bin_size))
    window = min(5, n_bins)
    if direction == "left":
        peak_region = coverage[max(0, anchor_bin - window):anchor_bin + 1]
    else:
        peak_region = coverage[anchor_bin:min(n_bins, anchor_bin + window + 1)]

    peak = int(np.max(peak_region)) if len(peak_region) > 0 else 1
    threshold = max(1, int(peak * dropoff_threshold))

    if direction == "left":
        for b in range(anchor_bin, -1, -1):
            if coverage[b] < threshold:
                return region_start + (b + 1) * bin_size
        return region_start
    else:
        for b in range(anchor_bin, n_bins):
            if coverage[b] < threshold:
                return region_start + b * bin_size
        return region_end


def _determine_exon_boundaries(
    junction_starts: np.ndarray,
    junction_ends: np.ndarray,
    locus_start: int,
    locus_end: int,
    min_exon_length: int,
) -> list[tuple[int, int]]:
    """Determine exon intervals from splice junction coordinates.

    Uses a **segment graph** approach: all unique donor and acceptor
    coordinates serve as boundary points that split the locus into
    segments.  Each segment between consecutive boundary points is
    included as an exon node unless it lies entirely within a single
    intron (junction span) *and* no junction connects to or from it.

    This correctly represents alternative 3' splice sites (A3SS) and
    alternative 5' splice sites (A5SS) that the previous merge-based
    algorithm collapsed.

    Args:
        junction_starts: Donor (5') positions of filtered junctions.
        junction_ends: Acceptor (3') positions of filtered junctions, in the
            same order as ``junction_starts``.
        locus_start: Genomic start of the locus.
        locus_end: Genomic end of the locus.
        min_exon_length: Exons shorter than this are discarded, unless
            they are needed for junction connectivity.

    Returns:
        Sorted list of ``(start, end)`` tuples for each exon segment.
        Adjacent segments that belong to the same exon are NOT merged
        here; downstream path-to-exon conversion handles that.
    """
    if len(junction_starts) == 0:
        # No junctions -- the entire locus is one exon.
        length = locus_end - locus_start
        if length >= min_exon_length:
            return [(locus_start, locus_end)]
        return []

    donors = np.unique(junction_starts)
    acceptors = np.unique(junction_ends)
    donor_set: set[int] = set(donors.tolist())
    acceptor_set: set[int] = set(acceptors.tolist())

    # All unique boundary points including locus bounds.
    all_points: list[int] = sorted(set(
        [locus_start, locus_end]
        + donors.tolist()
        + acceptors.tolist()
    ))

    # Build junction span set for intronic checking.
    junction_spans: list[tuple[int, int]] = [
        (int(js), int(je))
        for js, je in zip(junction_starts, junction_ends)
    ]

    # Create segments between consecutive boundary points.
    # A segment is KEPT if:
    #   1. It is NOT entirely within any single junction span (exonic), OR
    #   2. It IS within a junction span but is needed for junction
    #      connectivity: its start is an acceptor or its end is a donor.
    segments: list[tuple[int, int]] = []
    for i in range(len(all_points) - 1):
        seg_start = all_points[i]
        seg_end = all_points[i + 1]
        if seg_end <= seg_start:
            continue

        # Check if entirely within any junction (intron) span.
        is_within_intron = False
        for js, je in junction_spans:
            if js <= seg_start and je >= seg_end:
                is_within_intron = True
                break

        if is_within_intron:
            # Keep only if needed for junction connectivity.
            if seg_start in acceptor_set or seg_end in donor_set:
                segments.append((seg_start, seg_end))
        else:
            segments.append((seg_start, seg_end))

    # Filter by minimum exon length, but keep short segments that are
    # junction connection points (needed for graph wiring).
    filtered: list[tuple[int, int]] = []
    for s, e in segments:
        length = e - s
        if length >= min_exon_length:
            filtered.append((s, e))
        elif s in acceptor_set or e in donor_set:
            # Keep short segments needed for junction connectivity.
            filtered.append((s, e))

    return filtered


def _compute_exon_coverage(
    read_data: ReadData,
    chrom_id: int,
    exon_start: int,
    exon_end: int,
) -> float:
    """Compute average read coverage over an exon region.

    Uses the per-read start and end positions from ``ReadData`` to build a
    coverage profile for the exon and returns the mean.

    Args:
        read_data: Columnar read data.
        chrom_id: Numeric chromosome ID to filter reads.
        exon_start: 0-based start of the exon (inclusive).
        exon_end: 0-based end of the exon (exclusive).

    Returns:
        Average per-base read coverage across the exon.  Returns 0.0 if the
        exon has zero length or no reads overlap.
    """
    exon_length = exon_end - exon_start
    if exon_length <= 0 or read_data.n_reads == 0:
        return 0.0

    # Select reads on the correct chromosome that overlap the exon.
    chrom_mask = read_data.chrom_ids == chrom_id
    overlap_mask = (read_data.positions < exon_end) & (read_data.end_positions > exon_start)
    mask = chrom_mask & overlap_mask

    selected_starts = read_data.positions[mask]
    selected_ends = read_data.end_positions[mask]

    if len(selected_starts) == 0:
        return 0.0

    # Use the interval module's compute_coverage for per-base computation.
    cov_array = compute_coverage(selected_starts, selected_ends, exon_start, exon_end)
    return float(np.mean(cov_array))


def _compute_exon_coverages_batch(
    read_data: ReadData,
    chrom_id: int,
    exons: list[tuple[int, int]],
) -> list[float]:
    """Compute average read coverage for multiple exons in a single pass.

    Applies the chromosome filter once and reuses it across all exons,
    avoiding redundant mask creation per exon.

    Args:
        read_data: Columnar read data.
        chrom_id: Numeric chromosome ID to filter reads.
        exons: List of ``(start, end)`` tuples for each exon.

    Returns:
        List of average per-base coverages, one per exon.
    """
    if read_data.n_reads == 0 or not exons:
        return [0.0] * len(exons)

    # Single chromosome filter for the whole locus.
    chrom_mask = read_data.chrom_ids == chrom_id
    chrom_positions = read_data.positions[chrom_mask]
    chrom_end_positions = read_data.end_positions[chrom_mask]

    if len(chrom_positions) == 0:
        return [0.0] * len(exons)

    coverages: list[float] = []
    for exon_start, exon_end in exons:
        exon_length = exon_end - exon_start
        if exon_length <= 0:
            coverages.append(0.0)
            continue
        overlap = (chrom_positions < exon_end) & (chrom_end_positions > exon_start)
        sel_starts = chrom_positions[overlap]
        sel_ends = chrom_end_positions[overlap]
        if len(sel_starts) == 0:
            coverages.append(0.0)
            continue
        cov_array = compute_coverage(sel_starts, sel_ends, exon_start, exon_end)
        coverages.append(float(np.mean(cov_array)))

    return coverages


# ---------------------------------------------------------------------------
# SpliceGraphBuilder
# ---------------------------------------------------------------------------


class SpliceGraphBuilder:
    """Builds splice graphs from aligned RNA-seq read data.

    The builder takes pre-extracted :class:`ReadData` and
    :class:`JunctionEvidence` and constructs one :class:`SpliceGraph` per
    gene locus.  Loci are identified by clustering overlapping splice
    junctions, exon boundaries are derived from junction coordinates, and
    edges are weighted by junction support counts.

    Args:
        config: Builder configuration.  If ``None``, default parameters are
            used.
    """

    def __init__(self, config: GraphBuilderConfig | None = None) -> None:
        self._config: GraphBuilderConfig = config or GraphBuilderConfig()

    @property
    def config(self) -> GraphBuilderConfig:
        """Active builder configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Locus identification
    # ------------------------------------------------------------------

    def identify_loci(
        self,
        junctions: JunctionEvidence,
        chrom_length: int,
        read_data: ReadData | None = None,
    ) -> list[LocusDefinition]:
        """Cluster splice junctions into independent gene loci.

        Two junctions belong to the same locus if their spans (with flanking
        exonic regions) overlap.  The algorithm sorts junctions by start
        position and then performs a linear sweep, merging overlapping spans
        into contiguous loci.

        When ``read_data`` is provided, the locus boundaries are extended to
        the actual coverage extent of reads spanning the junctions, producing
        accurate terminal exon boundaries.

        Args:
            junctions: Junction evidence for a single chromosome.
            chrom_length: Length of the chromosome in base pairs, used to
                clamp locus boundaries.
            read_data: Optional read data used to extend locus boundaries to
                the actual read coverage extent.

        Returns:
            Sorted list of :class:`LocusDefinition` objects, one per
            identified locus.
        """
        n_junctions = len(junctions.starts)
        if n_junctions == 0:
            return []

        cfg = self._config

        # Sort junction indices by start position.
        order = np.argsort(junctions.starts)

        # Junction-only locus discovery needs enough flank to bridge internal
        # exons between consecutive introns.  The previous fixed 200bp flank
        # fragmented transcripts with longer exons into multiple loci.
        flank = max(cfg.min_exon_length, cfg.locus_flank)

        loci: list[LocusDefinition] = []
        current_indices: list[int] = [int(order[0])]
        current_start = max(0, int(junctions.starts[order[0]]) - flank)
        current_end = min(chrom_length, int(junctions.ends[order[0]]) + flank)

        for k in range(1, n_junctions):
            idx = int(order[k])
            j_start = max(0, int(junctions.starts[idx]) - flank)
            j_end = min(chrom_length, int(junctions.ends[idx]) + flank)

            if j_start <= current_end:
                # Overlaps with the current locus -- extend.
                current_end = max(current_end, j_end)
                current_indices.append(idx)
            else:
                # No overlap -- emit the current locus and start a new one.
                strand = self._consensus_strand(junctions.strands, current_indices)
                loci.append(LocusDefinition(
                    chrom=junctions.chrom,
                    start=current_start,
                    end=current_end,
                    strand=strand,
                    junction_indices=current_indices,
                ))
                current_indices = [idx]
                current_start = j_start
                current_end = j_end

        # Emit the last locus.
        strand = self._consensus_strand(junctions.strands, current_indices)
        loci.append(LocusDefinition(
            chrom=junctions.chrom,
            start=current_start,
            end=current_end,
            strand=strand,
            junction_indices=current_indices,
        ))

        # Split loci containing junctions on both strands into per-strand loci.
        loci = self._split_by_strand(loci, junctions, chrom_length)

        # Refine locus boundaries using actual read positions when available.
        if read_data is not None and read_data.n_reads > 0:
            loci = self._refine_locus_boundaries(loci, read_data, chrom_length)

        logger.debug(
            "Identified %d loci on %s from %d junctions",
            len(loci), junctions.chrom, n_junctions,
        )
        return loci

    # ------------------------------------------------------------------
    # Single-locus graph construction
    # ------------------------------------------------------------------

    def build_graph(
        self,
        locus: LocusDefinition,
        read_data: ReadData,
        junctions: JunctionEvidence,
    ) -> SpliceGraph | None:
        """Build a splice graph for a single gene locus.

        The construction proceeds in seven stages:

        1. Filter junctions by support count and intron length.
        2. Determine exon boundaries from junction coordinates.
        3. Compute per-exon read coverage.
        4. Discard low-coverage exons.
        5. Create the graph topology (SOURCE, EXON nodes, SINK).
        6. Wire edges: source links, intron edges, continuation edges,
           sink links.
        7. Set edge weights from junction support counts.

        Args:
            locus: The locus definition specifying the genomic region and
                the junction indices to use.
            read_data: Columnar read data covering at least the locus region.
            junctions: Full junction evidence for the chromosome.

        Returns:
            A populated :class:`SpliceGraph`, or ``None`` if the graph would
            be trivially empty (no exons survive filtering).
        """
        cfg = self._config

        # --- Stage 1: Filter junctions -----------------------------------
        locus_j_starts: list[int] = []
        locus_j_ends: list[int] = []
        locus_j_counts: list[float] = []

        for idx in locus.junction_indices:
            count = int(junctions.counts[idx])
            if count < cfg.min_junction_support:
                continue
            j_start = int(junctions.starts[idx])
            j_end = int(junctions.ends[idx])
            intron_len = j_end - j_start
            if intron_len < cfg.min_intron_length or intron_len > cfg.max_intron_length:
                continue
            locus_j_starts.append(j_start)
            locus_j_ends.append(j_end)
            locus_j_counts.append(float(count))

        if len(locus_j_starts) == 0:
            logger.debug(
                "Locus %s:%d-%d has no junctions after filtering; skipping.",
                locus.chrom, locus.start, locus.end,
            )
            return None

        # --- Stage 1b: Merge nearby junctions ----------------------------
        # Cluster donor/acceptor sites within merge_distance bp and keep
        # the highest-support one per cluster to eliminate alignment artefacts.
        if cfg.junction_merge_distance > 0:
            locus_j_starts, locus_j_ends, locus_j_counts = _merge_nearby_junctions(
                locus_j_starts, locus_j_ends, locus_j_counts,
                cfg.junction_merge_distance,
            )

        # --- Stage 1c: Relative support filter ----------------------------
        # Discard junctions far below the locus maximum support.
        if cfg.min_relative_junction_support > 0:
            locus_j_starts, locus_j_ends, locus_j_counts = _filter_relative_support(
                locus_j_starts, locus_j_ends, locus_j_counts,
                cfg.min_relative_junction_support,
            )

        if len(locus_j_starts) == 0:
            return None

        j_starts_arr = np.array(locus_j_starts, dtype=np.int64)
        j_ends_arr = np.array(locus_j_ends, dtype=np.int64)

        # --- Stage 2: Determine exon boundaries --------------------------
        # Use coverage drop-off for terminal exon boundaries instead of
        # min/max read extent which over-extends into adjacent genes.
        chrom_id_early = self._find_chrom_id(read_data, locus)

        first_donor = int(np.min(j_starts_arr))
        last_acceptor = int(np.max(j_ends_arr))

        if chrom_id_early >= 0 and read_data.n_reads > 0:
            effective_start = _compute_terminal_boundary(
                read_data, chrom_id_early, first_donor, "left",
                cfg.terminal_coverage_dropoff,
                cfg.max_terminal_extension,
            )
            effective_end = _compute_terminal_boundary(
                read_data, chrom_id_early, last_acceptor, "right",
                cfg.terminal_coverage_dropoff,
                cfg.max_terminal_extension,
            )
        else:
            effective_start = locus.start
            effective_end = locus.end

        exons = _determine_exon_boundaries(
            j_starts_arr, j_ends_arr,
            effective_start, effective_end,
            cfg.min_exon_length,
        )

        if len(exons) == 0:
            logger.debug(
                "Locus %s:%d-%d has no exons after boundary determination.",
                locus.chrom, locus.start, locus.end,
            )
            return None

        # Apply merge_distance: merge exons that are within the threshold.
        if cfg.merge_distance > 0 and len(exons) > 1:
            exons = self._merge_close_exons(exons, cfg.merge_distance)

        # --- Stage 3: Compute per-exon coverage --------------------------
        chrom_id = chrom_id_early

        exon_coverages = _compute_exon_coverages_batch(
            read_data, chrom_id, exons,
        )

        # --- Stage 4: Filter low-coverage exons --------------------------
        filtered_exons: list[tuple[int, int]] = []
        filtered_coverages: list[float] = []
        for (exon_start, exon_end), cov in zip(exons, exon_coverages):
            if cov >= cfg.min_exon_coverage:
                filtered_exons.append((exon_start, exon_end))
                filtered_coverages.append(cov)

        # --- Stage 4b: Relative exon coverage filter ----------------------
        # Remove exons whose coverage is far below the locus maximum.
        # This breaks low-coverage bridges between merged gene loci.
        if cfg.min_relative_exon_coverage > 0 and len(filtered_coverages) > 0:
            max_cov = max(filtered_coverages)
            rel_threshold = max_cov * cfg.min_relative_exon_coverage
            # Keep exons needed for junction connectivity even if low-cov.
            junction_endpoints: set[int] = set()
            for js, je in zip(locus_j_starts, locus_j_ends):
                junction_endpoints.add(js)
                junction_endpoints.add(je)
            rel_exons: list[tuple[int, int]] = []
            rel_covs: list[float] = []
            for (es, ee), cov in zip(filtered_exons, filtered_coverages):
                if cov >= rel_threshold:
                    rel_exons.append((es, ee))
                    rel_covs.append(cov)
                elif es in junction_endpoints or ee in junction_endpoints:
                    # Keep junction-connected exons even with low coverage.
                    rel_exons.append((es, ee))
                    rel_covs.append(cov)
            filtered_exons = rel_exons
            filtered_coverages = rel_covs

        if len(filtered_exons) == 0:
            logger.debug(
                "Locus %s:%d-%d has no exons after coverage filtering.",
                locus.chrom, locus.start, locus.end,
            )
            return None

        # --- Stage 5: Build graph nodes ----------------------------------
        graph = SpliceGraph(
            chrom=locus.chrom,
            strand=locus.strand,
            locus_start=locus.start,
            locus_end=locus.end,
        )
        graph.runtime_diagnostics = {
            "fallback_source_edges_added": 0,
            "fallback_sink_edges_added": 0,
        }

        # Virtual source node.
        source_id = graph.add_node(
            start=locus.start,
            end=locus.start,
            node_type=NodeType.SOURCE,
            coverage=0.0,
        )

        # Exon nodes, sorted by genomic position.
        exon_node_ids: list[int] = []
        for (exon_start, exon_end), cov in zip(filtered_exons, filtered_coverages):
            nid = graph.add_node(
                start=exon_start,
                end=exon_end,
                node_type=NodeType.EXON,
                coverage=cov,
            )
            exon_node_ids.append(nid)

        # Virtual sink node.
        sink_id = graph.add_node(
            start=locus.end,
            end=locus.end,
            node_type=NodeType.SINK,
            coverage=0.0,
        )

        # --- Stage 6: Wire edges -----------------------------------------
        # Build lookup structures for connecting junctions to exon nodes.
        # Use lists to handle multiple exons sharing the same boundary
        # coordinate (e.g. segment-graph splits at alternative splice sites).
        exon_start_map: dict[int, list[int]] = defaultdict(list)  # exon_start -> [node_ids]
        exon_end_map: dict[int, list[int]] = defaultdict(list)    # exon_end   -> [node_ids]
        for i, (exon_start, exon_end) in enumerate(filtered_exons):
            exon_start_map[exon_start].append(exon_node_ids[i])
            exon_end_map[exon_end].append(exon_node_ids[i])

        # Track which exon nodes have incoming intron edges (not first exons)
        # and which have outgoing intron edges (not last exons).
        has_incoming_junction: set[int] = set()
        has_outgoing_junction: set[int] = set()

        # Intron (junction) edges: exon ending at junction_start -> exon
        # starting at junction_end.  When multiple exons share a boundary,
        # create edges for all matching source/destination pairs.
        for j_start, j_end, j_count in zip(
            locus_j_starts, locus_j_ends, locus_j_counts,
        ):
            src_nids = exon_end_map.get(j_start, [])
            dst_nids = exon_start_map.get(j_end, [])
            for src_nid in src_nids:
                for dst_nid in dst_nids:
                    graph.add_edge(
                        src=src_nid,
                        dst=dst_nid,
                        edge_type=EdgeType.INTRON,
                        weight=j_count,
                        coverage=j_count,
                    )
                    has_outgoing_junction.add(src_nid)
                    has_incoming_junction.add(dst_nid)

        # Continuation edges: consecutive exon segments that abut (share a
        # boundary point).  With the segment graph approach, sub-exon
        # segments always have next_start == current_end.
        has_incoming_continuation: set[int] = set()
        has_outgoing_continuation: set[int] = set()

        for i in range(len(exon_node_ids) - 1):
            current_end = filtered_exons[i][1]
            next_start = filtered_exons[i + 1][0]
            src_nid = exon_node_ids[i]
            dst_nid = exon_node_ids[i + 1]
            # Add continuation if the two exons are adjacent (no gap or small
            # gap) and there is no intron edge already connecting them.
            if next_start <= current_end:
                # Overlapping or abutting exons -- always add continuation.
                existing_edge = graph.get_edge(src_nid, dst_nid)
                if existing_edge is None:
                    graph.add_edge(
                        src=src_nid,
                        dst=dst_nid,
                        edge_type=EdgeType.CONTINUATION,
                        weight=min(filtered_coverages[i], filtered_coverages[i + 1]),
                        coverage=min(filtered_coverages[i], filtered_coverages[i + 1]),
                    )
                    has_incoming_continuation.add(dst_nid)
                    has_outgoing_continuation.add(src_nid)

        # Source links: SOURCE -> exon nodes that have no incoming intron
        # edge AND no incoming continuation edge (true first exons).
        for i, nid in enumerate(exon_node_ids):
            if nid not in has_incoming_junction and nid not in has_incoming_continuation:
                graph.add_edge(
                    src=source_id,
                    dst=nid,
                    edge_type=EdgeType.SOURCE_LINK,
                    weight=filtered_coverages[i],
                    coverage=filtered_coverages[i],
                )

        # Sink links: exon nodes that have no outgoing intron edge AND no
        # outgoing continuation edge -> SINK (true last exons).
        for i, nid in enumerate(exon_node_ids):
            if nid not in has_outgoing_junction and nid not in has_outgoing_continuation:
                graph.add_edge(
                    src=nid,
                    dst=sink_id,
                    edge_type=EdgeType.SINK_LINK,
                    weight=filtered_coverages[i],
                    coverage=filtered_coverages[i],
                )

        # Ensure SOURCE can reach at least one node.
        if cfg.add_fallback_terminal_edges and len(graph.get_successors(source_id)) == 0:
            # Fallback: connect SOURCE to the first exon.
            graph.add_edge(
                src=source_id,
                dst=exon_node_ids[0],
                edge_type=EdgeType.SOURCE_LINK,
                weight=filtered_coverages[0],
                coverage=filtered_coverages[0],
            )
            graph.runtime_diagnostics["fallback_source_edges_added"] = 1

        # Ensure at least one node reaches SINK.
        if cfg.add_fallback_terminal_edges and len(graph.get_predecessors(sink_id)) == 0:
            # Fallback: connect the last exon to SINK.
            graph.add_edge(
                src=exon_node_ids[-1],
                dst=sink_id,
                edge_type=EdgeType.SINK_LINK,
                weight=filtered_coverages[-1],
                coverage=filtered_coverages[-1],
            )
            graph.runtime_diagnostics["fallback_sink_edges_added"] = 1

        logger.debug(
            "Built graph for %s:%d-%d (%s): %d nodes, %d edges",
            locus.chrom, locus.start, locus.end, locus.strand,
            graph.n_nodes, graph.n_edges,
        )
        return graph

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def build_all_graphs(
        self,
        read_data: ReadData,
        junctions: JunctionEvidence,
        chrom: str,
        chrom_length: int,
    ) -> list[SpliceGraph]:
        """Build splice graphs for all loci on a chromosome.

        Identifies loci from the junction evidence and constructs a graph
        for each.  Loci that produce empty graphs are silently skipped.

        Args:
            read_data: Columnar read data for the chromosome.
            junctions: Junction evidence for the chromosome.
            chrom: Chromosome name.
            chrom_length: Length of the chromosome in base pairs.

        Returns:
            List of :class:`SpliceGraph` objects, one per non-trivial locus.
        """
        loci = self.identify_loci(junctions, chrom_length, read_data=read_data)
        graphs: list[SpliceGraph] = []

        for locus in loci:
            graph = self.build_graph(locus, read_data, junctions)
            if graph is not None:
                graphs.append(graph)

        logger.info(
            "Built %d graphs on %s from %d loci",
            len(graphs), chrom, len(loci),
        )
        return graphs

    def build_batched(self, graphs: list[SpliceGraph]) -> BatchedCSRGraphs:
        """Convert a list of splice graphs to a batched CSR for GPU processing.

        Each graph is converted to CSR format and packed into a single
        :class:`BatchedCSRGraphs` instance with contiguous arrays.

        Args:
            graphs: List of :class:`SpliceGraph` objects to batch.

        Returns:
            A finalized :class:`BatchedCSRGraphs` containing all input graphs.

        Raises:
            ValueError: If the input list is empty.
        """
        if len(graphs) == 0:
            raise ValueError("Cannot build a batched CSR from an empty graph list.")

        batch = BatchedCSRGraphs()
        for graph in graphs:
            csr = graph.to_csr()
            batch.add_graph(csr, {
                "chrom": graph.chrom,
                "strand": graph.strand,
                "locus_start": graph.locus_start,
                "locus_end": graph.locus_end,
            })
        batch.finalize()

        logger.info(
            "Batched %d graphs: %d total nodes, %d total edges",
            batch.n_graphs, batch.total_nodes, batch.total_edges,
        )
        return batch

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _refine_locus_boundaries(
        loci: list[LocusDefinition],
        read_data: ReadData,
        chrom_length: int,
    ) -> list[LocusDefinition]:
        """Refine locus boundaries to the actual extent of read coverage.

        For each locus, finds the minimum and maximum positions of reads
        that span the locus junctions, extending the locus to capture the
        full terminal exons.

        Args:
            loci: Initial locus definitions.
            read_data: Columnar read data.
            chrom_length: Chromosome length for clamping.

        Returns:
            Updated list of locus definitions with refined boundaries.
        """
        refined: list[LocusDefinition] = []
        for locus in loci:
            # Find reads overlapping the junction region (the core of the locus)
            j_region_start = locus.start
            j_region_end = locus.end
            overlap_mask = (
                (read_data.positions < j_region_end)
                & (read_data.end_positions > j_region_start)
            )
            if not np.any(overlap_mask):
                refined.append(locus)
                continue

            # Extend to actual read extent
            read_min = int(np.min(read_data.positions[overlap_mask]))
            read_max = int(np.max(read_data.end_positions[overlap_mask]))
            new_start = max(0, min(locus.start, read_min))
            new_end = min(chrom_length, max(locus.end, read_max))

            refined.append(LocusDefinition(
                chrom=locus.chrom,
                start=new_start,
                end=new_end,
                strand=locus.strand,
                junction_indices=locus.junction_indices,
            ))
        return refined

    @staticmethod
    def _consensus_strand(
        strands: np.ndarray,
        indices: list[int],
    ) -> str:
        """Determine the consensus strand for a set of junctions.

        Uses majority vote among the junction strand assignments.  If there
        is an unresolvable tie or all strands are ambiguous, returns ``'.'``.

        Args:
            strands: Full strand array from :class:`JunctionEvidence`.
            indices: Indices of junctions belonging to the locus.

        Returns:
            ``'+'``, ``'-'``, or ``'.'``.
        """
        fwd_count = 0
        rev_count = 0
        for idx in indices:
            s = int(strands[idx])
            if s == 0:
                fwd_count += 1
            elif s == 1:
                rev_count += 1
        if fwd_count > rev_count:
            return "+"
        if rev_count > fwd_count:
            return "-"
        return "."

    def _split_by_strand(
        self,
        loci: list[LocusDefinition],
        junctions: JunctionEvidence,
        chrom_length: int,
    ) -> list[LocusDefinition]:
        """Split loci containing junctions on both strands into per-strand loci.

        Overlapping opposite-strand genes can merge into mega-loci that
        confuse decomposition.  This method detects loci with mixed-strand
        junctions and creates separate per-strand loci.

        Args:
            loci: Initial locus definitions.
            junctions: Full junction evidence for the chromosome.
            chrom_length: Length of the chromosome in base pairs, used to
                clamp end coordinates.

        Returns:
            Updated list of locus definitions, potentially with more loci
            than the input (split loci replace their parent).
        """
        result: list[LocusDefinition] = []

        for locus in loci:
            fwd_indices: list[int] = []
            rev_indices: list[int] = []
            ambig_indices: list[int] = []

            for idx in locus.junction_indices:
                s = int(junctions.strands[idx])
                if s == 0:
                    fwd_indices.append(idx)
                elif s == 1:
                    rev_indices.append(idx)
                else:
                    ambig_indices.append(idx)

            # Only split if both strands have junctions.
            if not fwd_indices or not rev_indices:
                result.append(locus)
                continue

            # Always assign ambiguous junctions to the dominant strand.
            if len(fwd_indices) >= len(rev_indices):
                fwd_indices.extend(ambig_indices)
            else:
                rev_indices.extend(ambig_indices)

            # Create per-strand loci.
            for strand_char, indices in [("+", fwd_indices), ("-", rev_indices)]:
                if not indices:
                    continue
                starts = [int(junctions.starts[i]) for i in indices]
                ends = [int(junctions.ends[i]) for i in indices]
                flank = self._config.locus_flank
                result.append(LocusDefinition(
                    chrom=locus.chrom,
                    start=max(0, min(starts) - flank),
                    end=min(max(ends) + flank, chrom_length),
                    strand=strand_char,
                    junction_indices=indices,
                ))

        # Re-sort by start position.
        result.sort(key=lambda loc: loc.start)
        return result

    @staticmethod
    def _merge_close_exons(
        exons: list[tuple[int, int]],
        merge_distance: int,
    ) -> list[tuple[int, int]]:
        """Merge exon intervals that are within *merge_distance* of each other.

        Args:
            exons: Sorted list of ``(start, end)`` exon intervals.
            merge_distance: Maximum gap between exons to merge.

        Returns:
            Merged exon list.
        """
        merged: list[tuple[int, int]] = [exons[0]]
        for start, end in exons[1:]:
            prev_start, prev_end = merged[-1]
            if start - prev_end <= merge_distance:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _find_chrom_id(read_data: ReadData, locus: LocusDefinition) -> int:
        """Find the numeric chromosome ID used in ReadData for this locus.

        When reads are fetched per-chromosome (the common case), all reads
        share the same ``chrom_id`` and we return it immediately.  Otherwise,
        falls back to a majority-vote among reads overlapping the locus region.

        Args:
            read_data: Columnar read data.
            locus: The locus to look up.

        Returns:
            The integer chromosome ID.  Returns ``-1`` if no reads overlap
            the locus.
        """
        if read_data.n_reads == 0:
            return -1

        # Fast path: if all reads share the same chrom_id (per-chromosome
        # fetching), return it directly without scanning.
        first_id = int(read_data.chrom_ids[0])
        if read_data.chrom_ids[-1] == first_id:
            return first_id

        # Slow path: find reads that overlap the locus region.
        overlap_mask = (
            (read_data.positions < locus.end) &
            (read_data.end_positions > locus.start)
        )

        overlapping_chroms = read_data.chrom_ids[overlap_mask]
        if len(overlapping_chroms) == 0:
            return -1

        # Return the most common chrom_id among overlapping reads.
        unique_ids, counts = np.unique(overlapping_chroms, return_counts=True)
        return int(unique_ids[np.argmax(counts)])
