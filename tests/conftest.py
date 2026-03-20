"""Shared fixtures for RapidSplice test suite.

Provides reusable synthetic test data fixtures for splice graph assembly tests.
All data is created in-memory with no external file dependencies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rapidsplice.graph.splice_graph import (
    CSRGraph,
    EdgeType,
    NodeType,
    SpliceGraph,
)
from rapidsplice.io.bam_reader import JunctionEvidence, ReadData

# ---------------------------------------------------------------------------
# Fixture 1: simple_read_data
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_read_data() -> ReadData:
    """Create a minimal ReadData with 10 synthetic reads on chr1.

    Layout:
      - 5 reads spanning 100-500 with one junction at 200-300
        (CIGAR: 100M 100N 100M -> ops [0,3,0], lens [100,100,100])
      - 3 reads spanning 100-200 (CIGAR: 100M -> ops [0], lens [100])
      - 2 reads spanning 300-500 (CIGAR: 200M -> ops [0], lens [200])

    All reads are mapped (mapq=60), forward strand, chromosome index 0.
    """
    n_reads = 10

    chrom_ids = np.zeros(n_reads, dtype=np.int32)
    positions = np.array(
        [100, 100, 100, 100, 100, 100, 100, 100, 300, 300],
        dtype=np.int64,
    )
    end_positions = np.array(
        [500, 500, 500, 500, 500, 200, 200, 200, 500, 500],
        dtype=np.int64,
    )
    strands = np.zeros(n_reads, dtype=np.int8)
    mapping_qualities = np.full(n_reads, 60, dtype=np.uint8)
    is_paired = np.zeros(n_reads, dtype=np.bool_)
    is_read1 = np.zeros(n_reads, dtype=np.bool_)
    mate_positions = np.full(n_reads, -1, dtype=np.int64)
    mate_chrom_ids = np.full(n_reads, -1, dtype=np.int32)
    query_names = [f"read_{i}" for i in range(n_reads)]

    # Build flat CIGAR arrays.
    # Reads 0-4: 100M 100N 100M  (3 ops each)
    # Reads 5-7: 100M             (1 op each)
    # Reads 8-9: 200M             (1 op each)
    cigar_ops_list: list[int] = []
    cigar_lens_list: list[int] = []
    cigar_offsets_list: list[int] = [0]

    for _ in range(5):
        cigar_ops_list.extend([0, 3, 0])  # M, N, M
        cigar_lens_list.extend([100, 100, 100])
        cigar_offsets_list.append(len(cigar_ops_list))

    for _ in range(3):
        cigar_ops_list.extend([0])  # M
        cigar_lens_list.extend([100])
        cigar_offsets_list.append(len(cigar_ops_list))

    for _ in range(2):
        cigar_ops_list.extend([0])  # M
        cigar_lens_list.extend([200])
        cigar_offsets_list.append(len(cigar_ops_list))

    cigar_ops = np.array(cigar_ops_list, dtype=np.uint8)
    cigar_lens = np.array(cigar_lens_list, dtype=np.int32)
    cigar_offsets = np.array(cigar_offsets_list, dtype=np.int64)

    return ReadData(
        chrom_ids=chrom_ids,
        positions=positions,
        end_positions=end_positions,
        strands=strands,
        mapping_qualities=mapping_qualities,
        is_paired=is_paired,
        is_read1=is_read1,
        mate_positions=mate_positions,
        mate_chrom_ids=mate_chrom_ids,
        cigar_ops=cigar_ops,
        cigar_lens=cigar_lens,
        cigar_offsets=cigar_offsets,
        query_names=query_names,
        n_reads=n_reads,
    )


# ---------------------------------------------------------------------------
# Fixture 2: simple_junction_evidence
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_junction_evidence() -> JunctionEvidence:
    """JunctionEvidence for chr1 with one junction at (200, 300) supported by 5 reads."""
    return JunctionEvidence(
        chrom="chr1",
        starts=np.array([200], dtype=np.int64),
        ends=np.array([300], dtype=np.int64),
        counts=np.array([5], dtype=np.int32),
        strands=np.array([0], dtype=np.int8),
    )


# ---------------------------------------------------------------------------
# Fixture 3: simple_splice_graph
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_splice_graph() -> SpliceGraph:
    """A SpliceGraph for chr1:100-500 with SOURCE, 2 EXON nodes, and SINK.

    Topology:
      SOURCE(0) --source_link--> EXON(100-200) --intron--> EXON(300-500) --sink_link--> SINK(500)
    """
    graph = SpliceGraph(chrom="chr1", strand="+", locus_start=100, locus_end=500)

    source_id = graph.add_node(start=100, end=100, node_type=NodeType.SOURCE, coverage=0.0)
    exon1_id = graph.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=8.0)
    exon2_id = graph.add_node(start=300, end=500, node_type=NodeType.EXON, coverage=7.0)
    sink_id = graph.add_node(start=500, end=500, node_type=NodeType.SINK, coverage=0.0)

    graph.add_edge(source_id, exon1_id, EdgeType.SOURCE_LINK, weight=8.0, coverage=8.0)
    graph.add_edge(exon1_id, exon2_id, EdgeType.INTRON, weight=5.0, coverage=5.0)
    graph.add_edge(exon2_id, sink_id, EdgeType.SINK_LINK, weight=7.0, coverage=7.0)

    return graph


# ---------------------------------------------------------------------------
# Fixture 4: simple_csr_graph
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_csr_graph(simple_splice_graph: SpliceGraph) -> CSRGraph:
    """CSR version of the simple_splice_graph."""
    return simple_splice_graph.to_csr()


# ---------------------------------------------------------------------------
# Fixture 5: complex_splice_graph
# ---------------------------------------------------------------------------


@pytest.fixture
def complex_splice_graph() -> SpliceGraph:
    """A complex splice graph with alternative splicing (2 possible transcripts).

    Topology:
      SOURCE
        |-- source_link --> EXON_A (100-200)
        |                     |-- intron --> EXON_B (300-400) -- intron --> EXON_D (500-600) --\\
        |                     |-- intron --> EXON_C (350-450) --intron--> EXON_E --+--> SINK
        |
    Transcript 1: SOURCE -> A -> B -> D -> SINK
    Transcript 2: SOURCE -> A -> C -> E -> SINK
    """
    graph = SpliceGraph(chrom="chr1", strand="+", locus_start=100, locus_end=700)

    source_id = graph.add_node(start=100, end=100, node_type=NodeType.SOURCE, coverage=0.0)
    exon_a = graph.add_node(start=100, end=200, node_type=NodeType.EXON, coverage=10.0)
    exon_b = graph.add_node(start=300, end=400, node_type=NodeType.EXON, coverage=6.0)
    exon_c = graph.add_node(start=350, end=450, node_type=NodeType.EXON, coverage=4.0)
    exon_d = graph.add_node(start=500, end=600, node_type=NodeType.EXON, coverage=6.0)
    exon_e = graph.add_node(start=550, end=650, node_type=NodeType.EXON, coverage=4.0)
    sink_id = graph.add_node(start=700, end=700, node_type=NodeType.SINK, coverage=0.0)

    # SOURCE -> A
    graph.add_edge(source_id, exon_a, EdgeType.SOURCE_LINK, weight=10.0, coverage=10.0)

    # A -> B (intron)
    graph.add_edge(exon_a, exon_b, EdgeType.INTRON, weight=6.0, coverage=6.0)
    # A -> C (intron)
    graph.add_edge(exon_a, exon_c, EdgeType.INTRON, weight=4.0, coverage=4.0)

    # B -> D (intron)
    graph.add_edge(exon_b, exon_d, EdgeType.INTRON, weight=6.0, coverage=6.0)
    # C -> E (intron)
    graph.add_edge(exon_c, exon_e, EdgeType.INTRON, weight=4.0, coverage=4.0)

    # D -> SINK
    graph.add_edge(exon_d, sink_id, EdgeType.SINK_LINK, weight=6.0, coverage=6.0)
    # E -> SINK
    graph.add_edge(exon_e, sink_id, EdgeType.SINK_LINK, weight=4.0, coverage=4.0)

    return graph


# ---------------------------------------------------------------------------
# Fixture 6: tmp_output_dir
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """A temporary directory for output files."""
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    return out_dir


# ---------------------------------------------------------------------------
# Fixture 7: as_transcripts — transcripts with alternative splicing
# ---------------------------------------------------------------------------


@pytest.fixture
def as_transcripts() -> list:
    """Transcripts exhibiting SE, A5SS, and RI alternative splicing.

    Gene g1 (chr1, + strand):
      tx1: [100,200) -- [300,400) -- [500,600)  (3-exon canonical)
      tx2: [100,200) -------------- [500,600)  (skips middle exon → SE)
      tx3: [100,250) -- [300,400) -- [500,600)  (alt 5'ss on exon1 → A5SS)

    Gene g2 (chr1, + strand):
      tx4: [1000,1500)                          (retained intron → RI)
      tx5: [1000,1200) -- [1300,1500)            (spliced form)
    """
    from rapidsplice.io.gtf_writer import TranscriptRecord

    return [
        TranscriptRecord(
            transcript_id="tx1", gene_id="g1", chrom="chr1", strand="+",
            start=100, end=600, exons=[(100, 200), (300, 400), (500, 600)],
        ),
        TranscriptRecord(
            transcript_id="tx2", gene_id="g1", chrom="chr1", strand="+",
            start=100, end=600, exons=[(100, 200), (500, 600)],
        ),
        TranscriptRecord(
            transcript_id="tx3", gene_id="g1", chrom="chr1", strand="+",
            start=100, end=600, exons=[(100, 250), (300, 400), (500, 600)],
        ),
        TranscriptRecord(
            transcript_id="tx4", gene_id="g2", chrom="chr1", strand="+",
            start=1000, end=1500, exons=[(1000, 1500)],
        ),
        TranscriptRecord(
            transcript_id="tx5", gene_id="g2", chrom="chr1", strand="+",
            start=1000, end=1500, exons=[(1000, 1200), (1300, 1500)],
        ),
    ]


# ---------------------------------------------------------------------------
# Fixture 8: as_junction_evidence — junction evidence for AS transcripts
# ---------------------------------------------------------------------------


@pytest.fixture
def as_junction_evidence() -> JunctionEvidence:
    """Junction evidence matching the as_transcripts fixture."""
    return JunctionEvidence(
        chrom="chr1",
        starts=np.array([200, 200, 250, 400, 1200], dtype=np.int64),
        ends=np.array([300, 500, 300, 500, 1300], dtype=np.int64),
        counts=np.array([20, 15, 10, 20, 25], dtype=np.int32),
        strands=np.array([0, 0, 0, 0, 0], dtype=np.int8),
    )
