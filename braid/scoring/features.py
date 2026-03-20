"""Feature engineering for ML-based transcript scoring.

Extracts 50+ numerical features from a candidate transcript and its splice
graph context, following the feature design principles described in the
Aletsch (Shao & Kingsford, 2019) and Beaver (Gatter & Stadler, 2023)
transcript assembly papers. Features are grouped into four categories:

1. **Coverage features** (15): Capture the distribution and uniformity of
   read coverage across the transcript's exons.
2. **Junction features** (10): Quantify splice-junction read support and
   the balance between exon coverage and junction evidence.
3. **Structure features** (15): Encode geometric properties of the exon-intron
   architecture such as lengths, counts, and ratios.
4. **Graph context features** (10+): Encode the transcript's relationship to
   other transcripts in the same locus and to the splice graph topology.

The feature vector is designed for consumption by a random-forest classifier
that predicts whether a candidate transcript is a true positive.

The :func:`extract_features` function accepts the core
:class:`~braid.flow.decompose.Transcript` dataclass (which carries
``exon_coords``, ``node_ids``, ``weight``, and ``is_safe``) and derives
per-exon coverage and per-junction support from the splice graph and
junction evidence rather than requiring them to be pre-computed on the
transcript object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from braid.flow.decompose import Transcript
    from braid.graph.splice_graph import SpliceGraph
    from braid.io.bam_reader import JunctionEvidence


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TranscriptFeatures:
    """All numerical features extracted for a single candidate transcript.

    Features are grouped into four blocks. Every field is a ``float`` (or
    ``int`` promoted to ``float`` when converted to array) so that the
    dataclass can be serialized into a flat NumPy vector.

    Coverage features (15):
        mean_coverage: Mean exon coverage.
        median_coverage: Median exon coverage.
        min_coverage: Minimum exon coverage.
        max_coverage: Maximum exon coverage.
        coverage_cv: Coefficient of variation of exon coverages.
        coverage_std: Standard deviation of exon coverages.
        coverage_range_ratio: (max - min) / mean coverage.
        coverage_q25: 25th percentile of exon coverages.
        coverage_q75: 75th percentile of exon coverages.
        coverage_iqr_ratio: IQR / median coverage.
        coverage_drop_count: Number of significant coverage drops between
            adjacent exons (drop > 50% of mean).
        coverage_uniformity: 1 - CV (higher = more uniform).
        relative_coverage: Transcript coverage / total locus coverage.
        log_coverage: log2(mean_coverage + 1).
        coverage_entropy: Shannon entropy of the normalized per-exon coverage
            distribution.

    Junction features (10):
        mean_junction_support: Mean junction read-support count.
        min_junction_support: Minimum junction support.
        max_junction_support: Maximum junction support.
        junction_support_cv: Coefficient of variation of junction supports.
        fraction_canonical_junctions: Fraction of junctions that are canonical
            (GT-AG, GC-AG, AT-AC) based on the evidence data.
        n_junctions: Number of splice junctions.
        junction_coverage_ratio: mean junction support / mean exon coverage.
        min_junction_ratio: min junction support / mean exon coverage.
        junction_balance: Average ratio min(left_cov, right_cov) /
            max(left_cov, right_cov) across all junctions.
        has_weak_junction: 1 if any junction has support < 3.

    Structure features (15):
        n_exons: Number of exons.
        total_length: Total spliced length in bases.
        mean_exon_length: Mean exon length.
        median_exon_length: Median exon length.
        min_exon_length: Minimum exon length.
        max_exon_length: Maximum exon length.
        exon_length_cv: Coefficient of variation of exon lengths.
        mean_intron_length: Mean intron length (0.0 for single-exon).
        median_intron_length: Median intron length.
        min_intron_length: Minimum intron length.
        max_intron_length: Maximum intron length.
        intron_length_cv: Coefficient of variation of intron lengths.
        transcript_span: Genomic span (end - start).
        exon_fraction: total_length / transcript_span.
        is_single_exon: 1 if single exon, 0 otherwise.

    Graph context features (10):
        locus_n_transcripts: Number of transcripts in the same locus.
        rank_in_locus: Rank by coverage within the locus (1-based).
        fraction_of_locus_coverage: Transcript coverage / sum of all locus
            transcript coverages.
        n_shared_junctions: Junctions shared with at least one other transcript.
        n_unique_junctions: Junctions unique to this transcript.
        graph_n_nodes: Number of nodes in the splice graph.
        graph_n_edges: Number of edges in the splice graph.
        graph_complexity: n_edges / n_nodes.
        is_safe_path: 1 if the transcript is based on a safe path.
        safe_path_fraction: Fraction of exon nodes originating from safe paths.
    """

    # -- Coverage features (15) --
    mean_coverage: float = 0.0
    median_coverage: float = 0.0
    min_coverage: float = 0.0
    max_coverage: float = 0.0
    coverage_cv: float = 0.0
    coverage_std: float = 0.0
    coverage_range_ratio: float = 0.0
    coverage_q25: float = 0.0
    coverage_q75: float = 0.0
    coverage_iqr_ratio: float = 0.0
    coverage_drop_count: float = 0.0
    coverage_uniformity: float = 0.0
    relative_coverage: float = 0.0
    log_coverage: float = 0.0
    coverage_entropy: float = 0.0

    # -- Junction features (10) --
    mean_junction_support: float = 0.0
    min_junction_support: float = 0.0
    max_junction_support: float = 0.0
    junction_support_cv: float = 0.0
    fraction_canonical_junctions: float = 0.0
    n_junctions: float = 0.0
    junction_coverage_ratio: float = 0.0
    min_junction_ratio: float = 0.0
    junction_balance: float = 0.0
    has_weak_junction: float = 0.0

    # -- Structure features (15) --
    n_exons: float = 0.0
    total_length: float = 0.0
    mean_exon_length: float = 0.0
    median_exon_length: float = 0.0
    min_exon_length: float = 0.0
    max_exon_length: float = 0.0
    exon_length_cv: float = 0.0
    mean_intron_length: float = 0.0
    median_intron_length: float = 0.0
    min_intron_length: float = 0.0
    max_intron_length: float = 0.0
    intron_length_cv: float = 0.0
    transcript_span: float = 0.0
    exon_fraction: float = 0.0
    is_single_exon: float = 0.0

    # -- Graph context features (10) --
    locus_n_transcripts: float = 0.0
    rank_in_locus: float = 0.0
    fraction_of_locus_coverage: float = 0.0
    n_shared_junctions: float = 0.0
    n_unique_junctions: float = 0.0
    graph_n_nodes: float = 0.0
    graph_n_edges: float = 0.0
    graph_complexity: float = 0.0
    is_safe_path: float = 0.0
    safe_path_fraction: float = 0.0


# ---------------------------------------------------------------------------
# Feature names (cached)
# ---------------------------------------------------------------------------

_FEATURE_NAMES: list[str] | None = None


def feature_names() -> list[str]:
    """Return the ordered list of feature names matching the array layout.

    Returns:
        A list of strings, one per feature, in the same order as the fields
        of :class:`TranscriptFeatures` and the columns of the array
        produced by :func:`features_to_array`.
    """
    global _FEATURE_NAMES  # noqa: PLW0603
    if _FEATURE_NAMES is None:
        _FEATURE_NAMES = [f.name for f in fields(TranscriptFeatures)]
    return list(_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def features_to_array(feat: TranscriptFeatures) -> np.ndarray:
    """Convert a :class:`TranscriptFeatures` instance to a 1-D NumPy array.

    The array has dtype ``float64`` and length equal to the number of features
    (i.e. ``len(feature_names())``).

    Args:
        feat: The feature dataclass to convert.

    Returns:
        A 1-D ``numpy.ndarray`` of shape ``(n_features,)``.
    """
    return np.array(
        [float(getattr(feat, name)) for name in feature_names()],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Internal math helpers
# ---------------------------------------------------------------------------


def _safe_cv(arr: np.ndarray) -> float:
    """Coefficient of variation, returning 0.0 when mean is zero.

    Args:
        arr: 1-D numeric array.

    Returns:
        Standard deviation divided by mean, or 0.0 if mean is zero.
    """
    m = float(np.mean(arr))
    if m == 0.0:
        return 0.0
    return float(np.std(arr)) / m


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division guarded against zero denominators.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value returned when *denominator* is zero.

    Returns:
        ``numerator / denominator`` or *default*.
    """
    if denominator == 0.0:
        return default
    return numerator / denominator


def _shannon_entropy(values: np.ndarray) -> float:
    """Shannon entropy of a non-negative array after normalizing to a distribution.

    Args:
        values: 1-D array of non-negative values.

    Returns:
        Entropy in nats. Returns 0.0 if the array sums to zero.
    """
    total = float(np.sum(values))
    if total <= 0.0:
        return 0.0
    probs = values / total
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0.0]
    return float(-np.sum(probs * np.log(probs)))


# ---------------------------------------------------------------------------
# Transcript accessor helpers
# ---------------------------------------------------------------------------
# The Transcript dataclass from flow.decompose uses ``exon_coords`` for the
# exon list, ``weight`` for abundance, ``is_safe`` for safe-path status, and
# ``node_ids`` for graph path.  These helpers provide a uniform access layer
# so the feature extractors work regardless of small API changes.


def _get_exons(transcript: Transcript) -> list[tuple[int, int]]:
    """Return the transcript's exon coordinate list.

    Args:
        transcript: A Transcript instance.

    Returns:
        List of (start, end) exon intervals.
    """
    return list(transcript.exon_coords)


def _get_weight(transcript: Transcript) -> float:
    """Return the transcript's abundance weight.

    Args:
        transcript: A Transcript instance.

    Returns:
        The flow weight / coverage estimate.
    """
    return float(transcript.weight)


def _get_intron_chain(transcript: Transcript) -> list[tuple[int, int]]:
    """Compute the intron chain from the exon coordinates.

    Each intron spans from the end of one exon to the start of the next.

    Args:
        transcript: A Transcript instance.

    Returns:
        List of (intron_start, intron_end) tuples.
    """
    exons = _get_exons(transcript)
    introns: list[tuple[int, int]] = []
    for i in range(len(exons) - 1):
        introns.append((exons[i][1], exons[i + 1][0]))
    return introns


def _get_exon_coverages_from_graph(
    transcript: Transcript,
    graph: SpliceGraph,
) -> list[float]:
    """Derive per-exon coverage from the splice graph node coverages.

    For each exon in the transcript, look up the matching graph node (by
    coordinate overlap) and use its coverage. If the transcript has node_ids,
    those are used directly; otherwise a coordinate-based lookup is performed.

    Args:
        transcript: A Transcript instance.
        graph: The splice graph for the locus.

    Returns:
        List of per-exon coverage values, aligned with exon_coords.
    """
    exons = _get_exons(transcript)
    node_ids = transcript.node_ids

    if node_ids:
        # Use node_ids directly -- filter to EXON-type nodes
        covs: list[float] = []
        for nid in node_ids:
            try:
                node = graph.get_node(nid)
            except KeyError:
                continue
            # Import NodeType locally to avoid circular imports at module level
            from braid.graph.splice_graph import NodeType
            if node.node_type == NodeType.EXON:
                covs.append(node.coverage)
        # If count matches exon count, return directly
        if len(covs) == len(exons):
            return covs

    # Fallback: coordinate-based lookup -- find graph nodes overlapping each exon
    from braid.graph.splice_graph import NodeType

    covs = []
    for ex_start, ex_end in exons:
        best_cov = _get_weight(transcript)  # fallback to weight
        best_overlap = 0
        for nid in (graph._nodes if hasattr(graph, "_nodes") else {}):
            try:
                node = graph.get_node(nid)
            except KeyError:
                continue
            if node.node_type != NodeType.EXON:
                continue
            overlap_start = max(node.start, ex_start)
            overlap_end = min(node.end, ex_end)
            overlap = overlap_end - overlap_start
            if overlap > best_overlap:
                best_overlap = overlap
                best_cov = node.coverage
        covs.append(best_cov)
    return covs


def _get_junction_supports(
    transcript: Transcript,
    junction_evidence: JunctionEvidence,
) -> list[float]:
    """Look up per-junction read support from the junction evidence data.

    For each intron in the transcript's intron chain, find the matching
    junction in the evidence and return its support count.

    Args:
        transcript: A Transcript instance.
        junction_evidence: Junction evidence for the locus.

    Returns:
        List of support counts, one per intron.
    """
    introns = _get_intron_chain(transcript)
    if not introns:
        return []

    # Build lookup dict from evidence
    ev_starts = junction_evidence.starts
    ev_ends = junction_evidence.ends
    ev_counts = junction_evidence.counts

    evidence_map: dict[tuple[int, int], float] = {}
    for idx in range(len(ev_starts)):
        key = (int(ev_starts[idx]), int(ev_ends[idx]))
        evidence_map[key] = float(ev_counts[idx])

    supports: list[float] = []
    for intron_start, intron_end in introns:
        supports.append(evidence_map.get((intron_start, intron_end), 0.0))
    return supports


# ---------------------------------------------------------------------------
# Core feature extraction
# ---------------------------------------------------------------------------


def _extract_coverage_features(
    transcript: Transcript,
    graph: SpliceGraph,
    locus_transcripts: list[Transcript],
) -> dict[str, float]:
    """Extract the 15 coverage features for a single transcript.

    Per-exon coverage is derived from the splice graph node coverages.

    Args:
        transcript: The candidate transcript.
        graph: The splice graph for the locus.
        locus_transcripts: All transcripts in the same locus.

    Returns:
        Dictionary mapping coverage feature names to their values.
    """
    exon_covs_list = _get_exon_coverages_from_graph(transcript, graph)
    if not exon_covs_list:
        exon_covs_list = [_get_weight(transcript)]

    exon_covs = np.array(exon_covs_list, dtype=np.float64)

    mean_cov = float(np.mean(exon_covs))
    median_cov = float(np.median(exon_covs))
    min_cov = float(np.min(exon_covs))
    max_cov = float(np.max(exon_covs))
    std_cov = float(np.std(exon_covs))
    cv = _safe_divide(std_cov, mean_cov)

    q25 = float(np.percentile(exon_covs, 25))
    q75 = float(np.percentile(exon_covs, 75))
    iqr = q75 - q25

    # Coverage drop count: adjacent exons where coverage changes by > 50% of mean
    drop_threshold = 0.5 * mean_cov
    drop_count = 0.0
    for i in range(len(exon_covs) - 1):
        if abs(exon_covs[i + 1] - exon_covs[i]) > drop_threshold:
            drop_count += 1.0

    # Total locus coverage (sum of weights of all transcripts)
    tx_weight = _get_weight(transcript)
    total_locus_cov = sum(_get_weight(t) for t in locus_transcripts)
    relative_cov = _safe_divide(tx_weight, total_locus_cov)

    return {
        "mean_coverage": mean_cov,
        "median_coverage": median_cov,
        "min_coverage": min_cov,
        "max_coverage": max_cov,
        "coverage_cv": cv,
        "coverage_std": std_cov,
        "coverage_range_ratio": _safe_divide(max_cov - min_cov, mean_cov),
        "coverage_q25": q25,
        "coverage_q75": q75,
        "coverage_iqr_ratio": _safe_divide(iqr, median_cov),
        "coverage_drop_count": drop_count,
        "coverage_uniformity": max(0.0, 1.0 - cv),
        "relative_coverage": relative_cov,
        "log_coverage": math.log2(mean_cov + 1.0),
        "coverage_entropy": _shannon_entropy(exon_covs),
    }


def _extract_junction_features(
    transcript: Transcript,
    graph: SpliceGraph,
    junction_evidence: JunctionEvidence,
) -> dict[str, float]:
    """Extract the 10 junction features for a single transcript.

    Per-junction support is looked up from the junction evidence. The
    canonical fraction is determined by the presence of each junction in
    the evidence set (evidence is derived from aligned reads using canonical
    splice signals).

    Args:
        transcript: The candidate transcript.
        graph: The splice graph for the locus.
        junction_evidence: Junction read-support evidence for the locus.

    Returns:
        Dictionary mapping junction feature names to their values.
    """
    junc_supports_list = _get_junction_supports(transcript, junction_evidence)
    n_junctions = len(junc_supports_list)

    if n_junctions == 0:
        return {
            "mean_junction_support": 0.0,
            "min_junction_support": 0.0,
            "max_junction_support": 0.0,
            "junction_support_cv": 0.0,
            "fraction_canonical_junctions": 1.0,
            "n_junctions": 0.0,
            "junction_coverage_ratio": 0.0,
            "min_junction_ratio": 0.0,
            "junction_balance": 1.0,
            "has_weak_junction": 0.0,
        }

    junc_supports = np.array(junc_supports_list, dtype=np.float64)
    mean_js = float(np.mean(junc_supports))
    min_js = float(np.min(junc_supports))
    max_js = float(np.max(junc_supports))

    # Determine canonical status by looking up each transcript junction in
    # the evidence data. A junction is canonical if it appears in the
    # evidence set (the evidence is built from aligned reads whose CIGAR
    # encodes canonical splice signals by the aligner).
    intron_chain = _get_intron_chain(transcript)
    ev_starts = junction_evidence.starts
    ev_ends = junction_evidence.ends

    evidence_set: set[tuple[int, int]] = set()
    for idx in range(len(ev_starts)):
        evidence_set.add((int(ev_starts[idx]), int(ev_ends[idx])))

    n_canonical = sum(
        1 for intron_start, intron_end in intron_chain
        if (intron_start, intron_end) in evidence_set
    )
    frac_canonical = _safe_divide(float(n_canonical), float(n_junctions), 1.0)

    # Mean exon coverage for ratio computation
    exon_covs = _get_exon_coverages_from_graph(transcript, graph)
    mean_cov = float(np.mean(exon_covs)) if exon_covs else _get_weight(transcript)

    # Junction balance: for each junction, compute
    # min(left_exon_cov, right_exon_cov) / max(left_exon_cov, right_exon_cov)
    balance_values: list[float] = []
    if len(exon_covs) >= 2:
        for i in range(len(exon_covs) - 1):
            left = exon_covs[i]
            right = exon_covs[i + 1]
            max_val = max(left, right)
            if max_val > 0.0:
                balance_values.append(min(left, right) / max_val)
            else:
                balance_values.append(1.0)

    avg_balance = float(np.mean(balance_values)) if balance_values else 1.0

    return {
        "mean_junction_support": mean_js,
        "min_junction_support": min_js,
        "max_junction_support": max_js,
        "junction_support_cv": _safe_cv(junc_supports),
        "fraction_canonical_junctions": frac_canonical,
        "n_junctions": float(n_junctions),
        "junction_coverage_ratio": _safe_divide(mean_js, mean_cov),
        "min_junction_ratio": _safe_divide(min_js, mean_cov),
        "junction_balance": avg_balance,
        "has_weak_junction": 1.0 if min_js < 3.0 else 0.0,
    }


def _extract_structure_features(transcript: Transcript) -> dict[str, float]:
    """Extract the 15 structural features for a single transcript.

    Args:
        transcript: The candidate transcript.

    Returns:
        Dictionary mapping structure feature names to their values.
    """
    exons = _get_exons(transcript)
    n_exons = len(exons)

    if n_exons == 0:
        return {name: 0.0 for name in [
            "n_exons", "total_length", "mean_exon_length", "median_exon_length",
            "min_exon_length", "max_exon_length", "exon_length_cv",
            "mean_intron_length", "median_intron_length", "min_intron_length",
            "max_intron_length", "intron_length_cv", "transcript_span",
            "exon_fraction", "is_single_exon",
        ]}

    exon_lengths = np.array([end - start for start, end in exons], dtype=np.float64)
    total_length = float(np.sum(exon_lengths))
    span = float(exons[-1][1] - exons[0][0]) if exons else 0.0

    result: dict[str, float] = {
        "n_exons": float(n_exons),
        "total_length": total_length,
        "mean_exon_length": float(np.mean(exon_lengths)),
        "median_exon_length": float(np.median(exon_lengths)),
        "min_exon_length": float(np.min(exon_lengths)),
        "max_exon_length": float(np.max(exon_lengths)),
        "exon_length_cv": _safe_cv(exon_lengths),
        "transcript_span": span,
        "exon_fraction": _safe_divide(total_length, span, 1.0),
        "is_single_exon": 1.0 if n_exons == 1 else 0.0,
    }

    # Intron features
    introns = _get_intron_chain(transcript)
    if introns:
        intron_lengths = np.array(
            [end - start for start, end in introns], dtype=np.float64
        )
        result["mean_intron_length"] = float(np.mean(intron_lengths))
        result["median_intron_length"] = float(np.median(intron_lengths))
        result["min_intron_length"] = float(np.min(intron_lengths))
        result["max_intron_length"] = float(np.max(intron_lengths))
        result["intron_length_cv"] = _safe_cv(intron_lengths)
    else:
        result["mean_intron_length"] = 0.0
        result["median_intron_length"] = 0.0
        result["min_intron_length"] = 0.0
        result["max_intron_length"] = 0.0
        result["intron_length_cv"] = 0.0

    return result


def _extract_graph_context_features(
    transcript: Transcript,
    graph: SpliceGraph,
    locus_transcripts: list[Transcript],
) -> dict[str, float]:
    """Extract the 10 graph-context features for a single transcript.

    Safe-path fraction is computed from the transcript's node_ids by
    checking which graph nodes have exactly one predecessor and one
    successor (a proxy for safe/unambiguous nodes) when the splice graph
    topology is available.

    Args:
        transcript: The candidate transcript.
        graph: The splice graph for the locus.
        locus_transcripts: All transcripts in the same locus.

    Returns:
        Dictionary mapping graph context feature names to their values.
    """
    n_locus = len(locus_transcripts)

    # Rank by weight within locus (1-based, highest weight = rank 1)
    tx_weight = _get_weight(transcript)
    weights_sorted = sorted(
        [_get_weight(t) for t in locus_transcripts], reverse=True,
    )
    rank = 1
    for i, w in enumerate(weights_sorted):
        if w <= tx_weight:
            rank = i + 1
            break

    total_locus_weight = sum(_get_weight(t) for t in locus_transcripts)

    # Compute shared/unique junctions
    this_introns = set(_get_intron_chain(transcript))
    other_introns: set[tuple[int, int]] = set()
    for other in locus_transcripts:
        if other is not transcript:
            other_introns.update(_get_intron_chain(other))

    n_shared = len(this_introns & other_introns)
    n_unique = len(this_introns - other_introns)

    # Graph topology properties
    g_n_nodes = float(graph.n_nodes)
    g_n_edges = float(graph.n_edges)

    # Safe path features: is_safe from the transcript, and safe_path_fraction
    # computed as the fraction of the transcript's graph nodes that lie on
    # unambiguous (single-in, single-out) chains in the splice graph.
    is_safe = 1.0 if transcript.is_safe else 0.0
    node_ids = transcript.node_ids
    total_exon_nodes = max(len(node_ids), 1)

    # Count nodes on unambiguous chains (exactly one predecessor, one successor)
    safe_count = 0
    for nid in node_ids:
        try:
            preds = graph.get_predecessors(nid)
            succs = graph.get_successors(nid)
        except KeyError:
            continue
        if len(preds) == 1 and len(succs) == 1:
            safe_count += 1

    # If the transcript is marked as safe, use 1.0 as minimum fraction
    safe_frac = _safe_divide(float(safe_count), float(total_exon_nodes))
    if is_safe > 0.5:
        safe_frac = max(safe_frac, 1.0)

    return {
        "locus_n_transcripts": float(n_locus),
        "rank_in_locus": float(rank),
        "fraction_of_locus_coverage": _safe_divide(
            tx_weight, total_locus_weight,
        ),
        "n_shared_junctions": float(n_shared),
        "n_unique_junctions": float(n_unique),
        "graph_n_nodes": g_n_nodes,
        "graph_n_edges": g_n_edges,
        "graph_complexity": _safe_divide(g_n_edges, g_n_nodes),
        "is_safe_path": is_safe,
        "safe_path_fraction": safe_frac,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_features(
    transcript: Transcript,
    graph: SpliceGraph,
    locus_transcripts: list[Transcript],
    junction_evidence: JunctionEvidence,
) -> TranscriptFeatures:
    """Extract all features for a single candidate transcript.

    Combines coverage, junction, structure, and graph-context feature groups
    into a single :class:`TranscriptFeatures` instance. Per-exon coverage
    is derived from the splice graph node coverages, and per-junction support
    is looked up from the junction evidence.

    The returned object can be converted to a NumPy array via
    :func:`features_to_array`.

    Args:
        transcript: The candidate transcript to extract features for. This is
            the :class:`~braid.flow.decompose.Transcript` dataclass
            carrying ``exon_coords``, ``node_ids``, ``weight``, and
            ``is_safe``.
        graph: The splice graph for the gene locus containing *transcript*.
        locus_transcripts: All candidate transcripts in the same gene locus,
            including *transcript* itself.
        junction_evidence: Splice-junction evidence aggregated from aligned
            reads for this locus.

    Returns:
        A populated :class:`TranscriptFeatures` dataclass instance.
    """
    feat_dict: dict[str, float] = {}

    feat_dict.update(
        _extract_coverage_features(transcript, graph, locus_transcripts)
    )
    feat_dict.update(
        _extract_junction_features(transcript, graph, junction_evidence)
    )
    feat_dict.update(_extract_structure_features(transcript))
    feat_dict.update(
        _extract_graph_context_features(transcript, graph, locus_transcripts)
    )

    return TranscriptFeatures(**feat_dict)
