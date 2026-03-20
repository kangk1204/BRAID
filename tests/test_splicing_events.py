"""Tests for alternative splicing event detection."""

from __future__ import annotations

from rapidsplice.io.gtf_writer import TranscriptRecord
from rapidsplice.splicing.events import (
    EventType,
    _detect_alternative_3ss,
    _detect_alternative_5ss,
    _detect_alternative_first_exons,
    _detect_alternative_last_exons,
    _detect_mutually_exclusive_exons,
    _detect_retained_introns,
    _detect_skipped_exons,
    detect_all_events,
    detect_events_for_gene,
)

# ---------------------------------------------------------------------------
# Helper to make transcripts quickly
# ---------------------------------------------------------------------------

def _tx(
    tid: str,
    gene_id: str,
    exons: list[tuple[int, int]],
    strand: str = "+",
    chrom: str = "chr1",
) -> TranscriptRecord:
    """Create a TranscriptRecord from exon list."""
    return TranscriptRecord(
        transcript_id=tid,
        gene_id=gene_id,
        chrom=chrom,
        strand=strand,
        start=exons[0][0],
        end=exons[-1][1],
        exons=exons,
    )


# ---------------------------------------------------------------------------
# Test SE: Skipped Exon
# ---------------------------------------------------------------------------


class TestSkippedExon:
    """Tests for exon skipping (SE) event detection."""

    def test_basic_se(self) -> None:
        """Detect a simple exon skipping event."""
        # Tx A: exon1 -- exon2 -- exon3  (exon2 is the skipped exon)
        # Tx B: exon1 ----------- exon3  (skips exon2)
        tx_a = _tx("tx1", "g1", [(100, 200), (300, 400), (500, 600)])
        tx_b = _tx("tx2", "g1", [(100, 200), (500, 600)])
        events = _detect_skipped_exons([tx_a, tx_b], "g1")
        assert len(events) == 1
        e = events[0]
        assert e.event_type == EventType.SE
        assert e.coordinates["skipped_exon_start"] == 300
        assert e.coordinates["skipped_exon_end"] == 400
        assert "tx1" in e.inclusion_transcripts
        assert "tx2" in e.exclusion_transcripts

    def test_no_se_with_single_exon(self) -> None:
        """No SE events when transcripts have < 3 exons."""
        tx_a = _tx("tx1", "g1", [(100, 200), (300, 400)])
        tx_b = _tx("tx2", "g1", [(100, 200), (300, 400)])
        events = _detect_skipped_exons([tx_a, tx_b], "g1")
        assert len(events) == 0

    def test_multiple_se(self) -> None:
        """Detect multiple SE events with multiple skipped exons."""
        tx_a = _tx("tx1", "g1", [(100, 200), (300, 400), (500, 600), (700, 800)])
        tx_b = _tx("tx2", "g1", [(100, 200), (500, 600), (700, 800)])
        tx_c = _tx("tx3", "g1", [(100, 200), (300, 400), (700, 800)])
        events = _detect_skipped_exons([tx_a, tx_b, tx_c], "g1")
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# Test A5SS: Alternative 5' Splice Site
# ---------------------------------------------------------------------------


class TestAlternative5SS:
    """Tests for alternative 5' splice site detection."""

    def test_basic_a5ss(self) -> None:
        """Detect A5SS: same acceptor, different donor."""
        # Both introns end at 400, but start at different positions
        tx_a = _tx("tx1", "g1", [(100, 250), (400, 600)])
        tx_b = _tx("tx2", "g1", [(100, 300), (400, 600)])
        events = _detect_alternative_5ss([tx_a, tx_b], "g1")
        assert len(events) == 1
        e = events[0]
        assert e.event_type == EventType.A5SS
        assert e.coordinates["acceptor"] == 400

    def test_no_a5ss_identical_introns(self) -> None:
        """No A5SS when introns are identical."""
        tx_a = _tx("tx1", "g1", [(100, 300), (400, 600)])
        tx_b = _tx("tx2", "g1", [(100, 300), (400, 600)])
        events = _detect_alternative_5ss([tx_a, tx_b], "g1")
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Test A3SS: Alternative 3' Splice Site
# ---------------------------------------------------------------------------


class TestAlternative3SS:
    """Tests for alternative 3' splice site detection."""

    def test_basic_a3ss(self) -> None:
        """Detect A3SS: same donor, different acceptor."""
        # Both introns start at 300, but end at different positions
        tx_a = _tx("tx1", "g1", [(100, 300), (400, 600)])
        tx_b = _tx("tx2", "g1", [(100, 300), (450, 600)])
        events = _detect_alternative_3ss([tx_a, tx_b], "g1")
        assert len(events) == 1
        e = events[0]
        assert e.event_type == EventType.A3SS
        assert e.coordinates["donor"] == 300

    def test_no_a3ss_identical_introns(self) -> None:
        """No A3SS when introns are identical."""
        tx_a = _tx("tx1", "g1", [(100, 300), (400, 600)])
        tx_b = _tx("tx2", "g1", [(100, 300), (400, 600)])
        events = _detect_alternative_3ss([tx_a, tx_b], "g1")
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Test MXE: Mutually Exclusive Exons
# ---------------------------------------------------------------------------


class TestMutuallyExclusiveExons:
    """Tests for mutually exclusive exon detection."""

    def test_basic_mxe(self) -> None:
        """Detect MXE between same flanking exons."""
        # Tx A: U -- X -- D
        # Tx B: U -- Y -- D  (X and Y non-overlapping)
        tx_a = _tx("tx1", "g1", [(100, 200), (300, 400), (600, 700)])
        tx_b = _tx("tx2", "g1", [(100, 200), (450, 550), (600, 700)])
        events = _detect_mutually_exclusive_exons([tx_a, tx_b], "g1")
        assert len(events) == 1
        e = events[0]
        assert e.event_type == EventType.MXE

    def test_no_mxe_overlapping_exons(self) -> None:
        """No MXE when internal exons overlap."""
        tx_a = _tx("tx1", "g1", [(100, 200), (300, 450), (600, 700)])
        tx_b = _tx("tx2", "g1", [(100, 200), (400, 550), (600, 700)])
        events = _detect_mutually_exclusive_exons([tx_a, tx_b], "g1")
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Test RI: Retained Intron
# ---------------------------------------------------------------------------


class TestRetainedIntron:
    """Tests for retained intron detection."""

    def test_basic_ri(self) -> None:
        """Detect RI: one exon spans another's intron."""
        # Tx A: single exon spanning [100, 500)
        # Tx B: two exons [100, 300), [400, 500) with intron [300, 400)
        tx_a = _tx("tx1", "g1", [(100, 500)])
        tx_b = _tx("tx2", "g1", [(100, 300), (400, 500)])
        events = _detect_retained_introns([tx_a, tx_b], "g1")
        assert len(events) == 1
        e = events[0]
        assert e.event_type == EventType.RI
        assert e.coordinates["intron_start"] == 300
        assert e.coordinates["intron_end"] == 400

    def test_no_ri_no_spanning(self) -> None:
        """No RI when no exon spans an intron."""
        tx_a = _tx("tx1", "g1", [(100, 200), (300, 400)])
        tx_b = _tx("tx2", "g1", [(100, 200), (300, 400)])
        events = _detect_retained_introns([tx_a, tx_b], "g1")
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Test AFE: Alternative First Exon
# ---------------------------------------------------------------------------


class TestAlternativeFirstExon:
    """Tests for alternative first exon detection."""

    def test_basic_afe(self) -> None:
        """Detect AFE: different first exons, same second exon."""
        tx_a = _tx("tx1", "g1", [(100, 200), (400, 600)])
        tx_b = _tx("tx2", "g1", [(250, 350), (400, 600)])
        events = _detect_alternative_first_exons([tx_a, tx_b], "g1")
        assert len(events) == 1
        e = events[0]
        assert e.event_type == EventType.AFE

    def test_no_afe_same_first_exon(self) -> None:
        """No AFE when first exons are identical."""
        tx_a = _tx("tx1", "g1", [(100, 200), (400, 600)])
        tx_b = _tx("tx2", "g1", [(100, 200), (400, 600)])
        events = _detect_alternative_first_exons([tx_a, tx_b], "g1")
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Test ALE: Alternative Last Exon
# ---------------------------------------------------------------------------


class TestAlternativeLastExon:
    """Tests for alternative last exon detection."""

    def test_basic_ale(self) -> None:
        """Detect ALE: same upstream, different last exons."""
        tx_a = _tx("tx1", "g1", [(100, 300), (400, 500)])
        tx_b = _tx("tx2", "g1", [(100, 300), (450, 600)])
        events = _detect_alternative_last_exons([tx_a, tx_b], "g1")
        assert len(events) == 1
        e = events[0]
        assert e.event_type == EventType.ALE

    def test_no_ale_same_last_exon(self) -> None:
        """No ALE when last exons are identical."""
        tx_a = _tx("tx1", "g1", [(100, 300), (400, 500)])
        tx_b = _tx("tx2", "g1", [(100, 300), (400, 500)])
        events = _detect_alternative_last_exons([tx_a, tx_b], "g1")
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDetectEventsForGene:
    """Tests for gene-level event detection."""

    def test_single_transcript(self) -> None:
        """No events with a single transcript."""
        tx = _tx("tx1", "g1", [(100, 200), (300, 400)])
        events = detect_events_for_gene([tx], "g1")
        assert len(events) == 0

    def test_mixed_events(self) -> None:
        """Detect multiple event types for a single gene."""
        tx_a = _tx("tx1", "g1", [(100, 200), (300, 400), (500, 600)])
        tx_b = _tx("tx2", "g1", [(100, 200), (500, 600)])  # SE
        tx_c = _tx("tx3", "g1", [(100, 250), (500, 600)])  # A5SS vs tx_a
        events = detect_events_for_gene([tx_a, tx_b, tx_c], "g1")
        types = {e.event_type for e in events}
        assert EventType.SE in types


class TestDetectAllEvents:
    """Tests for multi-gene event detection."""

    def test_multiple_genes(self) -> None:
        """Detect events across multiple genes."""
        tx_a = _tx("tx1", "g1", [(100, 200), (300, 400), (500, 600)])
        tx_b = _tx("tx2", "g1", [(100, 200), (500, 600)])
        tx_c = _tx("tx3", "g2", [(1000, 1100), (1200, 1300)])
        tx_d = _tx("tx4", "g2", [(1000, 1150), (1200, 1300)])  # A5SS
        events = detect_all_events([tx_a, tx_b, tx_c, tx_d])
        assert len(events) >= 2
        gene_ids = {e.gene_id for e in events}
        assert "g1" in gene_ids
        assert "g2" in gene_ids

    def test_empty_input(self) -> None:
        """No events from empty input."""
        events = detect_all_events([])
        assert len(events) == 0

    def test_event_deduplication(self) -> None:
        """Events are deduplicated by coordinate-based ID."""
        tx_a = _tx("tx1", "g1", [(100, 200), (300, 400), (500, 600)])
        tx_b = _tx("tx2", "g1", [(100, 200), (500, 600)])
        events = detect_all_events([tx_a, tx_b])
        event_ids = [e.event_id for e in events]
        assert len(event_ids) == len(set(event_ids)), "Duplicate event IDs found"
