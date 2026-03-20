"""Transcript filtering and post-processing.

Applies a cascade of quality filters to scored candidate transcripts, removing
low-confidence, redundant, and structurally implausible candidates. The
filtering pipeline follows the Aletsch / Beaver approach (Shao & Kingsford,
2019; Gatter & Stadler, 2023) with the following stages applied in order:

1. **Score threshold** -- Remove transcripts whose ML (or heuristic) score
   falls below a configurable minimum.
2. **Coverage threshold** -- Remove transcripts with insufficient average
   read coverage.
3. **Junction support threshold** -- Remove multi-exon transcripts that have
   any junction with support below a minimum.
4. **Length threshold** -- Remove transcripts whose total spliced length is
   below a minimum.
5. **Redundancy removal** -- Remove transcripts whose exon set is a proper
   subset of another higher-scoring transcript.
6. **Identical intron chain merging** -- Merge transcripts that share the
   same intron chain, summing their weights and extending exon boundaries
   to the union.
7. **Per-locus cap** -- Limit the number of reported transcripts per gene
   locus.

The Transcript type used throughout this module is
:class:`rapidsplice.flow.decompose.Transcript`, which carries ``exon_coords``,
``node_ids``, ``weight``, and ``is_safe``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from rapidsplice.scoring.features import TranscriptFeatures

if TYPE_CHECKING:
    from rapidsplice.flow.decompose import Transcript

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filter configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FilterConfig:
    """Configuration parameters for transcript filtering.

    All thresholds are inclusive lower bounds (a transcript must meet or
    exceed the threshold to pass).

    Attributes:
        min_score: Minimum ML / heuristic quality score to retain a
            transcript. Range [0, 1].
        min_coverage: Minimum average per-base read coverage.
        min_junction_support: Minimum read-support count required for
            every junction in a multi-exon transcript.
        min_exon_length: Minimum total spliced length (sum of exon lengths)
            in bases.
        max_transcripts_per_locus: Maximum number of transcripts to report
            per gene locus after all other filters.
        remove_redundant: Whether to remove transcripts whose exon set is a
            subset of a higher-scoring transcript in the same locus.
        merge_similar: Whether to merge transcripts that share an identical
            intron chain (keeping the highest-coverage representative).
    """

    min_score: float = 0.3
    min_coverage: float = 1.0
    min_junction_support: int = 2
    min_exon_length: int = 50
    max_transcripts_per_locus: int = 30
    remove_redundant: bool = True
    merge_similar: bool = True


@dataclass(slots=True)
class FilterDiagnostics:
    """Stage counts from the transcript filtering cascade."""

    initial: int = 0
    after_score: int = 0
    after_coverage: int = 0
    after_junction_support: int = 0
    after_length: int = 0
    after_redundancy: int = 0
    after_cap: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _exon_set(transcript: Transcript) -> frozenset[tuple[int, int]]:
    """Return the transcript's exon intervals as a frozen set.

    Args:
        transcript: A transcript object.

    Returns:
        A frozenset of (start, end) tuples.
    """
    return frozenset(transcript.exon_coords)


def _intron_chain_key(transcript: Transcript) -> tuple[tuple[int, int], ...]:
    """Return a hashable key for the transcript's intron chain.

    Two transcripts with the same intron chain key have identical splice
    sites, even if their terminal exon boundaries differ.

    Args:
        transcript: A transcript object.

    Returns:
        A tuple of (intron_start, intron_end) pairs, or an empty tuple
        for single-exon transcripts.
    """
    exons = transcript.exon_coords
    introns: list[tuple[int, int]] = []
    for i in range(len(exons) - 1):
        introns.append((exons[i][1], exons[i + 1][0]))
    return tuple(introns)


# ---------------------------------------------------------------------------
# Transcript filter
# ---------------------------------------------------------------------------


class TranscriptFilter:
    """Filter and post-process scored transcript candidates.

    Applies a configurable cascade of quality, redundancy, and per-locus
    filters to a set of scored transcripts. The filter is stateless -- all
    configuration is provided via :class:`FilterConfig`.

    Args:
        config: Filter configuration. Uses defaults if ``None``.
    """

    def __init__(self, config: FilterConfig | None = None) -> None:
        self._config: FilterConfig = config if config is not None else FilterConfig()

    @property
    def config(self) -> FilterConfig:
        """The active filter configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Main filtering pipeline
    # ------------------------------------------------------------------

    def filter_transcripts(
        self,
        transcripts: list[Transcript],
        scores: np.ndarray,
        features_list: list[TranscriptFeatures],
    ) -> list[int]:
        """Apply the full filtering cascade and return surviving indices."""
        surviving, _ = self.filter_transcripts_with_diagnostics(
            transcripts, scores, features_list,
        )
        return surviving

    def filter_transcripts_with_diagnostics(
        self,
        transcripts: list[Transcript],
        scores: np.ndarray,
        features_list: list[TranscriptFeatures],
    ) -> tuple[list[int], FilterDiagnostics]:
        """Apply the full filtering cascade and return surviving indices.

        Filters are applied in the following order:
        1. Score threshold
        2. Coverage threshold
        3. Junction support threshold
        4. Length threshold
        5. Redundancy removal (if enabled)
        6. Per-locus cap

        Identical intron chain merging is *not* performed here because it
        modifies transcript objects; use :meth:`merge_identical_intron_chains`
        separately before or after filtering.

        Args:
            transcripts: List of candidate transcripts (instances of
                :class:`~rapidsplice.flow.decompose.Transcript`).
            scores: 1-D array of shape ``(len(transcripts),)`` with per-
                transcript quality scores.
            features_list: List of :class:`TranscriptFeatures` instances,
                one per transcript (same order as *transcripts*).

        Returns:
            Tuple of:
            - Sorted list of surviving 0-based indices into *transcripts*
            - :class:`FilterDiagnostics` with stage-by-stage candidate counts

        Raises:
            ValueError: If the lengths of inputs are inconsistent.
        """
        scores = np.asarray(scores, dtype=np.float64)
        n = len(transcripts)

        if scores.shape != (n,):
            raise ValueError(
                f"scores length {scores.shape[0]} != transcripts length {n}"
            )
        if len(features_list) != n:
            raise ValueError(
                f"features_list length {len(features_list)} != transcripts length {n}"
            )

        if n == 0:
            return [], FilterDiagnostics()

        cfg = self._config
        diagnostics = FilterDiagnostics(initial=n)

        # Start with all indices
        surviving: list[int] = list(range(n))

        # 1. Score threshold — Relaxed for high-coverage transcripts.
        # Transcripts with strong read support (coverage >= 10.0 AND >= 10x
        # min_coverage) get a lower score requirement (0.5 * min_score).
        high_cov_threshold = max(20.0, cfg.min_coverage * 10)
        relaxed_score = cfg.min_score * 0.5
        surviving = [
            i for i in surviving
            if scores[i] >= cfg.min_score
            or (
                features_list[i].mean_coverage >= high_cov_threshold
                and scores[i] >= relaxed_score
            )
        ]
        diagnostics.after_score = len(surviving)

        # 2. Coverage threshold (use mean_coverage from features)
        surviving = [
            i for i in surviving
            if features_list[i].mean_coverage >= cfg.min_coverage
        ]
        diagnostics.after_coverage = len(surviving)

        # 3. Junction support threshold (only for multi-exon transcripts)
        surviving = [
            i for i in surviving
            if (
                features_list[i].n_junctions == 0
                or features_list[i].min_junction_support >= cfg.min_junction_support
            )
        ]
        diagnostics.after_junction_support = len(surviving)

        # 4. Length threshold
        surviving = [
            i for i in surviving
            if features_list[i].total_length >= cfg.min_exon_length
        ]
        diagnostics.after_length = len(surviving)

        n_before_redundancy = len(surviving)

        # 5. Redundancy removal
        if cfg.remove_redundant and len(surviving) > 1:
            surviving = self._remove_redundant_indices(
                transcripts, scores, surviving,
            )

        n_after_redundancy = len(surviving)
        diagnostics.after_redundancy = n_after_redundancy

        # 6. Per-locus cap
        if cfg.max_transcripts_per_locus > 0:
            surviving = self._apply_locus_cap(
                transcripts, scores, surviving, cfg.max_transcripts_per_locus,
            )
        diagnostics.after_cap = len(surviving)

        logger.debug(
            "Filtered %d -> %d transcripts (%d by thresholds, %d by redundancy, "
            "%d by locus cap)",
            n,
            len(surviving),
            n - n_before_redundancy,
            n_before_redundancy - n_after_redundancy,
            n_after_redundancy - len(surviving),
        )

        return sorted(surviving), diagnostics

    # ------------------------------------------------------------------
    # Redundancy removal
    # ------------------------------------------------------------------

    def remove_redundant_transcripts(
        self,
        transcripts: list[Transcript],
    ) -> list[int]:
        """Remove transcripts whose exon set is a subset of a higher-scoring transcript.

        For each pair of transcripts in the same locus, if the exon set of
        transcript A is a proper subset of the exon set of transcript B, and
        B has higher weight (coverage), A is removed. This is applied greedily
        starting from the highest-weight transcript.

        Args:
            transcripts: List of candidate transcripts.

        Returns:
            Sorted list of 0-based indices of transcripts that survive
            redundancy removal.
        """
        if len(transcripts) <= 1:
            return list(range(len(transcripts)))

        # Use weight as the score proxy
        weights = np.array(
            [t.weight for t in transcripts], dtype=np.float64,
        )
        return self._remove_redundant_indices(
            transcripts, weights, list(range(len(transcripts))),
        )

    def _remove_redundant_indices(
        self,
        transcripts: list[Transcript],
        scores: np.ndarray,
        indices: list[int],
    ) -> list[int]:
        """Internal redundancy removal on a subset of indices.

        Transcripts are grouped by locus (approximated by checking for
        shared genomic overlap). Within each group, transcripts with
        identical intron chains (same splice sites) are considered redundant;
        only the highest-scored representative is kept.

        Transcripts with different intron chains represent distinct
        alternative splicing isoforms and are never considered redundant,
        even if one transcript's exon set is a subset of another's.

        Args:
            transcripts: Full transcript list.
            scores: Score array (same length as *transcripts*).
            indices: Subset of indices to consider.

        Returns:
            Filtered list of indices with redundant transcripts removed.
        """
        if len(indices) <= 1:
            return list(indices)

        # Group by locus -- since the Transcript dataclass from decompose.py
        # does not carry a locus_id, we group by overlapping genomic span.
        locus_groups = self._group_by_overlap(transcripts, indices)

        kept: list[int] = []

        for group_indices in locus_groups:
            if len(group_indices) <= 1:
                kept.extend(group_indices)
                continue

            # Sort by score descending (highest score = hardest to remove)
            sorted_indices = sorted(
                group_indices, key=lambda i: scores[i], reverse=True,
            )

            # Compute intron chain keys for each transcript
            chain_keys: dict[int, tuple[tuple[int, int], ...]] = {}
            for i in sorted_indices:
                chain_keys[i] = _intron_chain_key(transcripts[i])

            # Keep only the highest-scored transcript per unique intron chain
            seen_chains: set[tuple[tuple[int, int], ...]] = set()
            removed: set[int] = set()

            for idx in sorted_indices:
                chain = chain_keys[idx]
                if chain in seen_chains:
                    # Duplicate intron chain — remove lower-scored copy
                    removed.add(idx)
                else:
                    seen_chains.add(chain)

            kept.extend(i for i in sorted_indices if i not in removed)

        return sorted(kept)

    # ------------------------------------------------------------------
    # Intron chain merging
    # ------------------------------------------------------------------

    def merge_identical_intron_chains(
        self,
        transcripts: list[Transcript],
    ) -> list[Transcript]:
        """Merge transcripts with identical intron chains.

        Transcripts that share the exact same intron chain (same splice
        donor and acceptor positions for every intron) are merged into a
        single representative. The representative retains the node_ids of
        the highest-weight member, with the merged weight set to the sum
        of all member weights and is_safe set if any member is safe. Its
        terminal exon boundaries are extended to the union (outermost
        start and end coordinates) of all merged transcripts.

        Single-exon transcripts are not merged (they have no intron chain).

        Args:
            transcripts: List of candidate transcripts. Objects are not
                modified in place; new :class:`Transcript` instances are
                returned for merged groups.

        Returns:
            A new list of transcripts with identical intron chains merged.
            The order follows the order of the first occurrence of each
            unique intron chain.
        """
        from rapidsplice.flow.decompose import Transcript as TranscriptCls

        if not self._config.merge_similar:
            return list(transcripts)

        if len(transcripts) <= 1:
            return list(transcripts)

        # Group transcripts by intron chain key
        groups: dict[
            tuple[tuple[int, int], ...], list[int]
        ] = defaultdict(list)
        insertion_order: list[tuple[tuple[int, int], ...]] = []

        for idx, tx in enumerate(transcripts):
            chain = _intron_chain_key(tx)
            if chain not in groups:
                insertion_order.append(chain)
            groups[chain].append(idx)

        result: list[Transcript] = []

        for chain in insertion_order:
            member_indices = groups[chain]

            if len(member_indices) == 1 or len(chain) == 0:
                # Single member or single-exon: keep as-is
                for i in member_indices:
                    result.append(transcripts[i])
                continue

            # Multiple transcripts with the same intron chain -- merge
            members = [transcripts[i] for i in member_indices]

            # Pick the highest-weight transcript as representative
            best = max(members, key=lambda t: t.weight)

            # Compute union of terminal exon boundaries
            # First exon: minimum start, same end (defined by first splice site)
            # Last exon: same start (defined by last splice site), maximum end
            all_first_starts = [m.exon_coords[0][0] for m in members]
            all_last_ends = [m.exon_coords[-1][1] for m in members]

            merged_start = min(all_first_starts)
            merged_end = max(all_last_ends)

            # Build merged exon list from the best representative
            merged_exon_coords: list[tuple[int, int]] = []
            for i, (ex_start, ex_end) in enumerate(best.exon_coords):
                if i == 0:
                    merged_exon_coords.append((merged_start, ex_end))
                elif i == len(best.exon_coords) - 1:
                    merged_exon_coords.append((ex_start, merged_end))
                else:
                    merged_exon_coords.append((ex_start, ex_end))

            # Sum weights across merged members
            merged_weight = sum(m.weight for m in members)

            merged_tx = TranscriptCls(
                node_ids=list(best.node_ids),
                exon_coords=merged_exon_coords,
                weight=merged_weight,
                is_safe=any(m.is_safe for m in members),
            )
            result.append(merged_tx)

        logger.debug(
            "Intron chain merging: %d -> %d transcripts",
            len(transcripts),
            len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Per-locus cap
    # ------------------------------------------------------------------

    def _apply_locus_cap(
        self,
        transcripts: list[Transcript],
        scores: np.ndarray,
        indices: list[int],
        max_per_locus: int,
    ) -> list[int]:
        """Limit the number of transcripts per locus, keeping highest-scored.

        Args:
            transcripts: Full transcript list.
            scores: Score array.
            indices: Subset of indices to consider.
            max_per_locus: Maximum transcripts to retain per locus.

        Returns:
            Filtered list of indices respecting the per-locus cap.
        """
        locus_groups = self._group_by_overlap(transcripts, indices)

        kept: list[int] = []
        for group_indices in locus_groups:
            # Sort by score descending and keep top max_per_locus
            sorted_group = sorted(
                group_indices, key=lambda i: scores[i], reverse=True,
            )
            kept.extend(sorted_group[:max_per_locus])

        return sorted(kept)

    # ------------------------------------------------------------------
    # Locus grouping helper
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_overlap(
        transcripts: list[Transcript],
        indices: list[int],
    ) -> list[list[int]]:
        """Group transcript indices by overlapping genomic span.

        Since the ``Transcript`` dataclass from ``flow.decompose`` does not
        carry an explicit ``locus_id``, this method approximates locus
        grouping by sorting transcripts by their genomic start coordinate
        and merging overlapping spans into groups.

        Args:
            transcripts: Full transcript list.
            indices: Subset of indices to group.

        Returns:
            List of lists, where each inner list contains indices of
            transcripts that share an overlapping genomic span.
        """
        if not indices:
            return []

        # Sort indices by transcript start coordinate
        def _tx_start(i: int) -> int:
            exons = transcripts[i].exon_coords
            return exons[0][0] if exons else 0

        def _tx_end(i: int) -> int:
            exons = transcripts[i].exon_coords
            return exons[-1][1] if exons else 0

        sorted_indices = sorted(indices, key=_tx_start)

        groups: list[list[int]] = []
        current_group: list[int] = [sorted_indices[0]]
        current_end = _tx_end(sorted_indices[0])

        for idx in sorted_indices[1:]:
            start = _tx_start(idx)
            end = _tx_end(idx)
            if start < current_end:
                # Overlaps with current group
                current_group.append(idx)
                current_end = max(current_end, end)
            else:
                # No overlap -- start new group
                groups.append(current_group)
                current_group = [idx]
                current_end = end

        groups.append(current_group)
        return groups
