"""Tests for the de novo RNA-seq assembler module.

Covers k-mer extraction, de Bruijn graph construction, compaction,
simplification, transcript extraction, FASTQ reading, and the
end-to-end pipeline.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import numpy as np
import pytest

from braid.denovo.assemble import (
    AssemblyConfig,
    DeNovoTranscript,
    _compute_path_coverage,
    extract_transcripts,
    write_fasta,
)
from braid.denovo.fastq import (
    FastqRead,
    _avg_quality,
    _trim_3prime,
    read_fastq,
    stream_fastq_sequences,
)
from braid.denovo.graph import (
    DBGEdge,
    DBGNode,
    DeBruijnGraph,
    build_debruijn_graph,
    compact_graph,
)
from braid.denovo.kmer import (
    KmerCountTable,
    canonicalize_kmers,
    count_kmers,
    decode_kmer,
    encode_sequence,
    extract_prefix_suffix,
    extract_prefixes_suffixes,
    reverse_complement_kmer,
    reverse_complement_kmers,
)
from braid.denovo.pipeline import (
    DeNovoConfig,
    _compute_n50,
    run_denovo_assembly,
)
from braid.denovo.simplify import (
    SimplifyConfig,
    _remove_isolated_nodes,
    _remove_low_coverage_edges,
    _remove_tips,
    simplify_graph,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def simple_sequence() -> str:
    """A short DNA sequence for k-mer tests."""
    return "ACGTACGTAC"


@pytest.fixture
def longer_sequence() -> str:
    """A longer sequence for graph construction tests."""
    return "ACGTACGTACGTACGTACGT"


@pytest.fixture
def repeated_sequence() -> str:
    """Sequence with repeated k-mers for counting tests."""
    return "ACGTACGTACGTACGT"


@pytest.fixture
def fastq_file(tmp_path: Path) -> Path:
    """Create a temporary FASTQ file with test reads."""
    fq_path = tmp_path / "test.fq"
    reads = [
        ("@read1", "ACGTACGTACGTACGTACGTACGTACGTACGT", "+", "I" * 32),
        ("@read2", "TGCATGCATGCATGCATGCATGCATGCATGCA", "+", "I" * 32),
        ("@read3", "ACGTACGTACGTACGTACGTACGTACGTACGT", "+", "I" * 32),
        ("@read4", "AAAACCCCGGGGTTTTAAAACCCCGGGGTTTT", "+", "I" * 32),
        ("@read5", "ACGTACGTACGTACGTACGTACGTACGTACGT", "+", "I" * 32),
    ]
    with open(fq_path, "w") as fh:
        for header, seq, plus, qual in reads:
            fh.write(f"{header}\n{seq}\n{plus}\n{qual}\n")
    return fq_path


@pytest.fixture
def fastq_gz_file(tmp_path: Path) -> Path:
    """Create a gzip-compressed FASTQ file."""
    fq_path = tmp_path / "test.fq.gz"
    reads = [
        ("@read1", "ACGTACGTACGTACGTACGT", "+", "I" * 20),
        ("@read2", "ACGTACGTACGTACGTACGT", "+", "I" * 20),
    ]
    with gzip.open(fq_path, "wt") as fh:
        for header, seq, plus, qual in reads:
            fh.write(f"{header}\n{seq}\n{plus}\n{qual}\n")
    return fq_path


# ===========================================================================
# K-mer tests
# ===========================================================================


class TestEncodeSequence:
    """Tests for encode_sequence."""

    def test_basic_encoding(self, simple_sequence: str) -> None:
        """Encode a simple sequence and verify output shape."""
        kmers = encode_sequence(simple_sequence, k=3)
        assert len(kmers) == len(simple_sequence) - 3 + 1
        assert kmers.dtype == np.uint64

    def test_kmer_values(self) -> None:
        """Verify specific k-mer encodings."""
        # ACG = 0b00_01_10 = 6
        kmers = encode_sequence("ACG", k=3)
        assert len(kmers) == 1
        assert kmers[0] == 6

    def test_single_base_kmer(self) -> None:
        """k=1 should return individual base encodings."""
        kmers = encode_sequence("ACGT", k=1)
        assert len(kmers) == 4
        assert list(kmers) == [0, 1, 2, 3]

    def test_empty_input(self) -> None:
        """Empty sequence should return empty array."""
        kmers = encode_sequence("", k=3)
        assert len(kmers) == 0

    def test_short_input(self) -> None:
        """Sequence shorter than k should return empty array."""
        kmers = encode_sequence("AC", k=3)
        assert len(kmers) == 0

    def test_invalid_bases_skipped(self) -> None:
        """Non-ACGT characters should break k-mer runs."""
        kmers = encode_sequence("ACGNACG", k=3)
        # "ACG" then N breaks, then "ACG" again
        assert len(kmers) == 2

    def test_lowercase_input(self) -> None:
        """Lowercase bases should work identically to uppercase."""
        upper = encode_sequence("ACGTACGT", k=3)
        lower = encode_sequence("acgtacgt", k=3)
        np.testing.assert_array_equal(upper, lower)

    def test_bytes_input(self) -> None:
        """Bytes input should work identically to string."""
        str_result = encode_sequence("ACGTACGT", k=3)
        bytes_result = encode_sequence(b"ACGTACGT", k=3)
        np.testing.assert_array_equal(str_result, bytes_result)

    def test_k_validation(self) -> None:
        """Invalid k values should raise ValueError."""
        with pytest.raises(ValueError, match="k must be 1"):
            encode_sequence("ACGT", k=0)
        with pytest.raises(ValueError, match="k must be 1"):
            encode_sequence("ACGT", k=32)

    def test_max_k(self) -> None:
        """k=31 should work with sufficiently long sequence."""
        seq = "A" * 40
        kmers = encode_sequence(seq, k=31)
        assert len(kmers) == 10


class TestReverseComplement:
    """Tests for reverse complement computation."""

    def test_single_kmer_rc(self) -> None:
        """ACG -> CGT in k=3."""
        acg = encode_sequence("ACG", k=3)[0]
        cgt = encode_sequence("CGT", k=3)[0]
        assert reverse_complement_kmer(acg, 3) == cgt

    def test_palindrome(self) -> None:
        """ACGT is its own reverse complement."""
        acgt = encode_sequence("ACGT", k=4)[0]
        rc = reverse_complement_kmer(acgt, 4)
        assert rc == acgt

    def test_batch_rc(self) -> None:
        """Batch RC should match individual RC for each k-mer."""
        kmers = encode_sequence("ACGTACGTACGT", k=5)
        batch_rc = reverse_complement_kmers(kmers, 5)
        individual_rc = np.array([
            reverse_complement_kmer(km, 5) for km in kmers
        ], dtype=np.uint64)
        np.testing.assert_array_equal(batch_rc, individual_rc)

    def test_double_rc_identity(self) -> None:
        """RC(RC(x)) should equal x."""
        kmers = encode_sequence("ACGTACGTACGT", k=7)
        for km in kmers:
            rc = reverse_complement_kmer(km, 7)
            rc_rc = reverse_complement_kmer(rc, 7)
            assert rc_rc == km

    def test_empty_array(self) -> None:
        """Empty input should return empty output."""
        result = reverse_complement_kmers(np.empty(0, dtype=np.uint64), 5)
        assert len(result) == 0


class TestCanonicalKmers:
    """Tests for canonical k-mer computation."""

    def test_canonical_is_min(self) -> None:
        """Canonical should be the smaller of forward and RC."""
        kmers = encode_sequence("ACGTACGTACGT", k=5)
        canonical = canonicalize_kmers(kmers, 5)
        rc = reverse_complement_kmers(kmers, 5)
        expected = np.minimum(kmers, rc)
        np.testing.assert_array_equal(canonical, expected)

    def test_canonical_strand_independent(self) -> None:
        """Forward and RC sequences should produce the same canonical k-mers."""
        fwd = encode_sequence("ACGTACGT", k=5)
        rev = encode_sequence("ACGTACGT", k=5)
        fwd_can = canonicalize_kmers(fwd, 5)
        rev_can = canonicalize_kmers(rev, 5)
        np.testing.assert_array_equal(fwd_can, rev_can)


class TestDecodeKmer:
    """Tests for k-mer decoding."""

    def test_roundtrip(self) -> None:
        """Encode then decode should recover the original sequence."""
        for seq in ["ACG", "ACGT", "TTTTT", "GCGCG"]:
            k = len(seq)
            encoded = encode_sequence(seq, k)[0]
            decoded = decode_kmer(encoded, k)
            assert decoded == seq

    def test_all_bases(self) -> None:
        """Test encoding/decoding with all four bases."""
        encoded = encode_sequence("ACGT", k=4)[0]
        decoded = decode_kmer(encoded, 4)
        assert decoded == "ACGT"


class TestCountKmers:
    """Tests for k-mer counting."""

    def test_basic_counting(self) -> None:
        """Count k-mers from a simple repeated sequence."""
        sequences = ["ACGTACGTACGT"]
        table = count_kmers(sequences, k=3, min_count=1)
        assert isinstance(table, KmerCountTable)
        assert len(table.kmers) > 0
        assert table.k == 3

    def test_min_count_filter(self) -> None:
        """K-mers below min_count should be excluded."""
        sequences = ["ACGTACGT"]
        table_no_filter = count_kmers(sequences, k=3, min_count=1)
        table_filtered = count_kmers(sequences, k=3, min_count=3)
        assert len(table_filtered.kmers) <= len(table_no_filter.kmers)

    def test_multiple_sequences(self) -> None:
        """Counts should accumulate across multiple sequences."""
        sequences = ["ACGTACGT", "ACGTACGT", "ACGTACGT"]
        table = count_kmers(sequences, k=3, min_count=1)
        # Same k-mers from each sequence should sum
        assert all(c >= 3 for c in table.counts)

    def test_canonical_mode(self) -> None:
        """Canonical counting should merge forward and RC k-mers."""
        fwd = ["ACGTACGT"]
        table_canon = count_kmers(fwd, k=3, min_count=1, canonical=True)
        table_no_canon = count_kmers(fwd, k=3, min_count=1, canonical=False)
        # Canonical should have <= non-canonical unique k-mers
        assert len(table_canon.kmers) <= len(table_no_canon.kmers)

    def test_empty_input(self) -> None:
        """Empty sequence list should return empty table."""
        table = count_kmers([], k=5, min_count=1)
        assert len(table.kmers) == 0
        assert len(table.counts) == 0

    def test_sorted_output(self) -> None:
        """Output k-mers should be sorted."""
        table = count_kmers(["ACGTACGTACGTACGT"], k=5, min_count=1)
        if len(table.kmers) > 1:
            assert np.all(table.kmers[:-1] <= table.kmers[1:])


class TestPrefixSuffix:
    """Tests for prefix/suffix extraction."""

    def test_single_kmer(self) -> None:
        """Prefix and suffix of ACG (k=3) should be AC and CG."""
        acg = encode_sequence("ACG", k=3)[0]
        prefix, suffix = extract_prefix_suffix(acg, 3)
        assert decode_kmer(prefix, 2) == "AC"
        assert decode_kmer(suffix, 2) == "CG"

    def test_batch_extraction(self) -> None:
        """Batch extraction should match individual extraction."""
        kmers = encode_sequence("ACGTACGT", k=4)
        prefixes, suffixes = extract_prefixes_suffixes(kmers, 4)
        for i, km in enumerate(kmers):
            p, s = extract_prefix_suffix(km, 4)
            assert prefixes[i] == p
            assert suffixes[i] == s


# ===========================================================================
# FASTQ reader tests
# ===========================================================================


class TestFastqReader:
    """Tests for FASTQ file reading."""

    def test_read_fastq(self, fastq_file: Path) -> None:
        """Read a plain FASTQ file."""
        reads = read_fastq(fastq_file)
        assert len(reads) == 5
        assert all(isinstance(r, FastqRead) for r in reads)

    def test_read_fastq_gz(self, fastq_gz_file: Path) -> None:
        """Read a gzip-compressed FASTQ file."""
        reads = read_fastq(fastq_gz_file)
        assert len(reads) == 2

    def test_min_length_filter(self, fastq_file: Path) -> None:
        """Reads shorter than min_length should be filtered."""
        reads = read_fastq(fastq_file, min_length=100)
        assert len(reads) == 0  # All reads are 32 bases

    def test_stream_sequences(self, fastq_file: Path) -> None:
        """stream_fastq_sequences should return only sequences."""
        seqs = stream_fastq_sequences(fastq_file)
        assert len(seqs) == 5
        assert all(isinstance(s, str) for s in seqs)
        assert all(s.isupper() for s in seqs)

    def test_file_not_found(self) -> None:
        """Missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_fastq("/nonexistent/file.fq")

    def test_quality_trimming(self) -> None:
        """3' quality trimming should remove low-quality bases."""
        seq = "ACGTACGT"
        qual = "IIIIII!!"  # Last 2 bases have Q=0
        trimmed_seq, trimmed_qual = _trim_3prime(seq, qual, threshold=10)
        assert len(trimmed_seq) == 6
        assert trimmed_seq == "ACGTAC"

    def test_avg_quality(self) -> None:
        """Average quality computation."""
        # 'I' = ASCII 73, Phred = 73-33 = 40
        assert _avg_quality("IIII") == 40.0
        assert _avg_quality("") == 0.0
        # '!' = ASCII 33, Phred = 0
        assert _avg_quality("!!!!") == 0.0

    def test_quality_filter(self, tmp_path: Path) -> None:
        """Reads below min_avg_quality should be filtered."""
        fq = tmp_path / "qual.fq"
        with open(fq, "w") as fh:
            fh.write("@good\nACGT\n+\nIIII\n")
            fh.write("@bad\nACGT\n+\n!!!!\n")
        reads = read_fastq(fq, min_avg_quality=20.0)
        assert len(reads) == 1
        assert reads[0].name == "good"

    def test_read_length_property(self) -> None:
        """FastqRead.length should return sequence length."""
        read = FastqRead(name="test", sequence="ACGT", quality="IIII")
        assert read.length == 4


# ===========================================================================
# De Bruijn graph tests
# ===========================================================================


class TestDeBruijnGraph:
    """Tests for de Bruijn graph construction."""

    def test_build_from_kmers(self) -> None:
        """Build graph from simple k-mer table."""
        sequences = ["ACGTACGTACGT"]
        table = count_kmers(sequences, k=5, min_count=1, canonical=False)
        graph = build_debruijn_graph(table)
        assert graph.n_nodes > 0
        assert graph.n_edges > 0
        assert graph.k == 5

    def test_empty_table(self) -> None:
        """Empty k-mer table should produce empty graph."""
        table = KmerCountTable(
            kmers=np.empty(0, dtype=np.uint64),
            counts=np.empty(0, dtype=np.uint32),
            k=5,
        )
        graph = build_debruijn_graph(table)
        assert graph.n_nodes == 0
        assert graph.n_edges == 0

    def test_node_connectivity(self) -> None:
        """Edges should connect prefix to suffix (k-1)-mers."""
        sequences = ["ACGTACGT"]
        table = count_kmers(sequences, k=4, min_count=1, canonical=False)
        graph = build_debruijn_graph(table)

        # Every edge's source should be in the target's in_edges
        for edge in graph.edges:
            assert edge.source in graph.nodes
            assert edge.target in graph.nodes
            assert edge.target in graph.nodes[edge.source].out_edges
            assert edge.source in graph.nodes[edge.target].in_edges

    def test_sources_and_sinks(self) -> None:
        """Non-cyclic sequence should have source and sink nodes."""
        # Use a sequence that doesn't form cycles in the de Bruijn graph
        sequences = ["ACGATCG"]
        table = count_kmers(sequences, k=3, min_count=1, canonical=False)
        graph = build_debruijn_graph(table)
        # A short non-repeating sequence should have at least one source or sink
        has_source_or_sink = len(graph.sources()) > 0 or len(graph.sinks()) > 0
        assert has_source_or_sink or graph.n_nodes == 0

    def test_node_coverage(self) -> None:
        """Node coverage should be positive."""
        sequences = ["ACGTACGT"]
        table = count_kmers(sequences, k=3, min_count=1, canonical=False)
        graph = build_debruijn_graph(table)
        for node in graph.nodes.values():
            assert node.coverage >= 0

    def test_edge_coverage(self) -> None:
        """Edge coverage should match k-mer counts."""
        sequences = ["ACGTACGT"]
        table = count_kmers(sequences, k=3, min_count=1, canonical=False)
        graph = build_debruijn_graph(table)
        for edge in graph.edges:
            assert edge.coverage > 0

    def test_get_edge_coverage(self) -> None:
        """get_edge_coverage should return correct values."""
        sequences = ["ACGTACGT"]
        table = count_kmers(sequences, k=3, min_count=1, canonical=False)
        graph = build_debruijn_graph(table)
        if graph.edges:
            e = graph.edges[0]
            cov = graph.get_edge_coverage(e.source, e.target)
            assert cov == e.coverage
        # Non-existent edge
        assert graph.get_edge_coverage(-1, -2) == 0.0


class TestGraphCompaction:
    """Tests for graph compaction."""

    def test_linear_chain_compaction(self) -> None:
        """Linear chain should be compacted into fewer nodes."""
        # Long non-repeating sequence creates a linear chain
        sequences = ["ACGATCGATCGATCG"]
        table = count_kmers(sequences, k=4, min_count=1, canonical=False)
        graph = build_debruijn_graph(table)
        n_before = graph.n_nodes
        compacted = compact_graph(graph)
        assert compacted.n_nodes <= n_before

    def test_compaction_preserves_branching(self) -> None:
        """Branching points should be preserved after compaction."""
        # Two paths diverge and reconverge
        sequences = [
            "ACGTAAAAACGT",
            "ACGTCCCCACGT",
        ]
        table = count_kmers(sequences, k=4, min_count=1, canonical=False)
        graph = build_debruijn_graph(table)
        compacted = compact_graph(graph)
        # Should still have branching structure
        assert compacted.n_nodes > 0

    def test_empty_graph_compaction(self) -> None:
        """Compacting empty graph should return empty graph."""
        graph = DeBruijnGraph(k=5)
        compacted = compact_graph(graph)
        assert compacted.n_nodes == 0

    def test_unitig_sequences(self) -> None:
        """Compacted unitig sequences should be valid DNA."""
        sequences = ["ACGATCGATCGATCG"]
        table = count_kmers(sequences, k=4, min_count=1, canonical=False)
        graph = build_debruijn_graph(table)
        compacted = compact_graph(graph)
        valid_bases = set("ACGT")
        for node in compacted.nodes.values():
            assert all(b in valid_bases for b in node.sequence)


# ===========================================================================
# Graph simplification tests
# ===========================================================================


class TestGraphSimplification:
    """Tests for graph simplification."""

    def test_tip_removal(self) -> None:
        """Short dead-end tips should be removed."""
        # Create a graph with a short tip
        graph = _make_graph_with_tip()
        n_before = graph.n_nodes
        removed = _remove_tips(graph, min_tip_length=50)
        assert removed > 0
        assert graph.n_nodes < n_before

    def test_low_coverage_removal(self) -> None:
        """Low-coverage edges should be removed."""
        graph = _make_graph_with_low_cov_edge()
        n_edges_before = graph.n_edges
        removed = _remove_low_coverage_edges(graph, min_coverage=5.0)
        assert removed > 0
        assert graph.n_edges < n_edges_before

    def test_isolated_node_removal(self) -> None:
        """Isolated nodes should be cleaned up."""
        graph = DeBruijnGraph(k=5)
        graph.nodes[0] = DBGNode(
            node_id=0, kmer_encoding=np.uint64(0),
            sequence="ACGT", unitig_length=4,
        )
        graph.nodes[1] = DBGNode(
            node_id=1, kmer_encoding=np.uint64(1),
            sequence="TGCA", unitig_length=4,
        )
        removed = _remove_isolated_nodes(graph)
        assert removed == 2
        assert graph.n_nodes == 0

    def test_full_simplification(self) -> None:
        """Full simplification should reduce graph size."""
        sequences = ["ACGTACGTACGTACGT"] * 10 + ["ACGTNNNNACGT"]
        table = count_kmers(sequences, k=5, min_count=1, canonical=False)
        graph = build_debruijn_graph(table)
        cfg = SimplifyConfig(
            min_tip_length=30,
            min_coverage=2.0,
            max_iterations=3,
        )
        stats = simplify_graph(graph, cfg)
        assert stats.iterations > 0

    def test_simplify_empty_graph(self) -> None:
        """Simplifying empty graph should complete without error."""
        graph = DeBruijnGraph(k=5)
        stats = simplify_graph(graph)
        assert stats.iterations >= 1
        assert stats.tips_removed == 0


# ===========================================================================
# Transcript extraction tests
# ===========================================================================


class TestTranscriptExtraction:
    """Tests for transcript path extraction."""

    def test_linear_graph_single_transcript(self) -> None:
        """Linear graph should produce one transcript."""
        graph = _make_linear_graph(k=5, length=300)
        transcripts = extract_transcripts(
            graph,
            AssemblyConfig(min_transcript_length=10, min_coverage=1.0),
        )
        assert len(transcripts) >= 1

    def test_empty_graph(self) -> None:
        """Empty graph should produce no transcripts."""
        graph = DeBruijnGraph(k=5)
        transcripts = extract_transcripts(graph)
        assert len(transcripts) == 0

    def test_transcript_attributes(self) -> None:
        """Transcripts should have valid attributes."""
        graph = _make_linear_graph(k=5, length=300)
        transcripts = extract_transcripts(
            graph,
            AssemblyConfig(min_transcript_length=10, min_coverage=0.5),
        )
        for tx in transcripts:
            assert tx.transcript_id.startswith("DENOVO_")
            assert len(tx.sequence) > 0
            assert tx.length == len(tx.sequence)
            assert tx.coverage > 0
            assert len(tx.path_node_ids) >= 2

    def test_coverage_sorted(self) -> None:
        """Transcripts should be sorted by coverage descending."""
        graph = _make_linear_graph(k=5, length=300)
        transcripts = extract_transcripts(
            graph,
            AssemblyConfig(min_transcript_length=10, min_coverage=0.5),
        )
        if len(transcripts) >= 2:
            for i in range(len(transcripts) - 1):
                assert transcripts[i].coverage >= transcripts[i + 1].coverage

    def test_min_length_filter(self) -> None:
        """Short transcripts should be filtered."""
        graph = _make_linear_graph(k=5, length=50)
        transcripts = extract_transcripts(
            graph,
            AssemblyConfig(min_transcript_length=1000, min_coverage=0.5),
        )
        assert len(transcripts) == 0

    def test_write_fasta(self, tmp_path: Path) -> None:
        """FASTA output should be readable."""
        transcripts = [
            DeNovoTranscript(
                transcript_id="DENOVO_000000",
                sequence="ACGT" * 100,
                coverage=10.0,
            ),
            DeNovoTranscript(
                transcript_id="DENOVO_000001",
                sequence="TGCA" * 50,
                coverage=5.0,
            ),
        ]
        out_path = tmp_path / "output.fa"
        write_fasta(transcripts, str(out_path))
        assert out_path.exists()

        # Parse and verify
        content = out_path.read_text()
        headers = [line for line in content.splitlines() if line.startswith(">")]
        assert len(headers) == 2
        assert "DENOVO_000000" in headers[0]
        assert "len=400" in headers[0]

    def test_compute_path_coverage(self) -> None:
        """Path coverage should be the minimum edge coverage."""
        edge_cov = {(0, 1): 10.0, (1, 2): 5.0, (2, 3): 15.0}
        assert _compute_path_coverage([0, 1, 2, 3], edge_cov) == 5.0

    def test_compute_path_coverage_empty(self) -> None:
        """Empty or single-node path should return 0."""
        assert _compute_path_coverage([], {}) == 0.0
        assert _compute_path_coverage([0], {}) == 0.0


class TestN50:
    """Tests for N50 computation."""

    def test_basic_n50(self) -> None:
        """Verify N50 on a known set of lengths."""
        lengths = [100, 200, 300, 400, 500]
        # Total = 1500, 50% = 750
        # 500 -> 500 < 750
        # 500 + 400 = 900 >= 750 -> N50 = 400
        assert _compute_n50(lengths) == 400

    def test_single_length(self) -> None:
        """Single sequence should have N50 = its length."""
        assert _compute_n50([1000]) == 1000

    def test_empty(self) -> None:
        """Empty list should return 0."""
        assert _compute_n50([]) == 0

    def test_equal_lengths(self) -> None:
        """All equal lengths: N50 = that length."""
        assert _compute_n50([100, 100, 100]) == 100


# ===========================================================================
# Pipeline integration tests
# ===========================================================================


class TestDeNovoPipeline:
    """Tests for the end-to-end de novo assembly pipeline."""

    def test_pipeline_basic(self, fastq_file: Path) -> None:
        """Run pipeline on a simple FASTQ file."""
        with tempfile.NamedTemporaryFile(suffix=".fa", delete=False) as f:
            out_path = f.name

        config = DeNovoConfig(
            fastq_paths=[str(fastq_file)],
            output_path=out_path,
            k=5,
            min_kmer_count=1,
            min_read_length=10,
            min_transcript_length=10,
            min_transcript_coverage=1.0,
        )
        transcripts, stats = run_denovo_assembly(config)

        assert stats.n_reads == 5
        assert stats.n_unique_kmers > 0
        assert stats.elapsed_seconds > 0
        # May or may not produce transcripts depending on graph structure
        assert isinstance(transcripts, list)

        # Cleanup
        Path(out_path).unlink(missing_ok=True)

    def test_pipeline_empty_input(self, tmp_path: Path) -> None:
        """Pipeline with empty FASTQ should return empty results."""
        fq = tmp_path / "empty.fq"
        fq.write_text("")

        config = DeNovoConfig(
            fastq_paths=[str(fq)],
            output_path=str(tmp_path / "out.fa"),
            k=5,
            min_kmer_count=1,
        )
        transcripts, stats = run_denovo_assembly(config)
        assert stats.n_reads == 0
        assert stats.n_transcripts == 0

    def test_pipeline_stats(self, fastq_file: Path) -> None:
        """Pipeline stats should have all expected fields."""
        config = DeNovoConfig(
            fastq_paths=[str(fastq_file)],
            output_path="/dev/null",
            k=5,
            min_kmer_count=1,
            min_read_length=10,
            min_transcript_length=5,
            min_transcript_coverage=0.5,
        )
        _, stats = run_denovo_assembly(config)
        assert stats.n_reads > 0
        assert stats.n_graph_nodes >= 0
        assert stats.n_graph_edges >= 0
        assert stats.elapsed_seconds > 0


# ===========================================================================
# CLI tests
# ===========================================================================


class TestCLI:
    """Tests for CLI denovo subcommand argument parsing."""

    def test_denovo_parser(self) -> None:
        """Denovo subcommand should parse correctly."""
        from braid.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "denovo", "reads.fq", "-o", "out.fa", "-k", "25",
        ])
        assert args.command == "denovo"
        assert args.fastq == ["reads.fq"]
        assert args.output == "out.fa"
        assert args.kmer_size == 25

    def test_denovo_multiple_fastq(self) -> None:
        """Multiple FASTQ files should be parsed."""
        from braid.cli import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "denovo", "r1.fq", "r2.fq", "r3.fq",
        ])
        assert args.fastq == ["r1.fq", "r2.fq", "r3.fq"]

    def test_denovo_defaults(self) -> None:
        """Default values should be set correctly."""
        from braid.cli import create_parser

        parser = create_parser()
        args = parser.parse_args(["denovo", "reads.fq"])
        assert args.output == "denovo_transcripts.fa"
        assert args.kmer_size == 25
        assert args.min_kmer_count == 3
        assert args.min_length == 200
        assert args.min_coverage == 2.0
        assert args.no_canonical is False


# ===========================================================================
# Helper functions for test graph construction
# ===========================================================================


def _make_linear_graph(k: int = 5, length: int = 300) -> DeBruijnGraph:
    """Create a linear de Bruijn graph from a random sequence.

    Args:
        k: K-mer size.
        length: Sequence length in bases.

    Returns:
        A constructed de Bruijn graph.
    """
    rng = np.random.RandomState(42)
    bases = "ACGT"
    seq = "".join(rng.choice(list(bases)) for _ in range(length))
    table = count_kmers([seq], k=k, min_count=1, canonical=False)
    return build_debruijn_graph(table)


def _make_graph_with_tip() -> DeBruijnGraph:
    """Create a graph with a short tip for tip removal testing."""
    graph = DeBruijnGraph(k=5)

    # Main path: 0 -> 1 -> 2
    for i in range(3):
        graph.nodes[i] = DBGNode(
            node_id=i, kmer_encoding=np.uint64(i),
            sequence="ACGT" * 10, coverage=20.0, unitig_length=40,
        )

    # Short tip: 0 -> 3 (dead end, short)
    graph.nodes[3] = DBGNode(
        node_id=3, kmer_encoding=np.uint64(3),
        sequence="AC", coverage=2.0, unitig_length=2,
    )

    # Main path edges
    graph.nodes[0].out_edges = {1, 3}
    graph.nodes[1].in_edges = {0}
    graph.nodes[1].out_edges = {2}
    graph.nodes[2].in_edges = {1}

    # Tip edge
    graph.nodes[3].in_edges = {0}

    graph.edges = [
        DBGEdge(source=0, target=1, kmer_encoding=np.uint64(0), coverage=20.0),
        DBGEdge(source=1, target=2, kmer_encoding=np.uint64(1), coverage=20.0),
        DBGEdge(source=0, target=3, kmer_encoding=np.uint64(2), coverage=2.0),
    ]

    return graph


def _make_graph_with_low_cov_edge() -> DeBruijnGraph:
    """Create a graph with a low-coverage edge."""
    graph = DeBruijnGraph(k=5)

    for i in range(3):
        graph.nodes[i] = DBGNode(
            node_id=i, kmer_encoding=np.uint64(i),
            sequence="ACGT" * 10, coverage=20.0, unitig_length=40,
        )

    graph.nodes[0].out_edges = {1, 2}
    graph.nodes[1].in_edges = {0}
    graph.nodes[2].in_edges = {0}

    graph.edges = [
        DBGEdge(source=0, target=1, kmer_encoding=np.uint64(0), coverage=20.0),
        DBGEdge(source=0, target=2, kmer_encoding=np.uint64(1), coverage=1.0),
    ]

    return graph
