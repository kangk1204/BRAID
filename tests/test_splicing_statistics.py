"""Tests for the splicing classifier and I/O modules."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from braid.splicing.classifier import (
    EventClassifier,
    EventFeatures,
    extract_event_features,
    features_to_array,
    heuristic_score,
)
from braid.splicing.events import ASEvent, EventType
from braid.splicing.io import read_events_tsv, write_events_tsv, write_ioe
from braid.splicing.psi import PSIResult

# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------


class TestHeuristicScore:
    """Tests for the heuristic scoring fallback."""

    def test_high_confidence(self) -> None:
        """High reads, narrow CI, balanced junctions → high score."""
        features = EventFeatures(
            total_reads=100,
            ci_width=0.1,
            junction_balance=0.8,
            log_total_reads=math.log1p(100),
        )
        score = heuristic_score(features)
        assert score > 0.6

    def test_low_confidence(self) -> None:
        """Low reads, wide CI, imbalanced junctions → lower score."""
        features = EventFeatures(
            total_reads=2,
            ci_width=0.9,
            junction_balance=0.1,
            log_total_reads=math.log1p(2),
        )
        score = heuristic_score(features)
        assert score < 0.7

    def test_score_range(self) -> None:
        """Score is always in [0, 1]."""
        for reads in [0, 1, 10, 100, 10000]:
            for ci in [0.0, 0.5, 1.0]:
                for bal in [0.0, 0.5, 1.0]:
                    features = EventFeatures(
                        total_reads=reads,
                        ci_width=ci,
                        junction_balance=bal,
                        log_total_reads=math.log1p(reads),
                    )
                    s = heuristic_score(features)
                    assert 0.0 <= s <= 1.0


class TestExtractEventFeatures:
    """Tests for feature extraction from events."""

    def test_feature_extraction(self) -> None:
        """Correctly extracts features from event and PSI result."""
        event = ASEvent(
            event_id="SE:chr1:+:100_200_300_400",
            event_type=EventType.SE,
            gene_id="g1",
            chrom="chr1",
            strand="+",
            inclusion_junctions=[(100, 200), (300, 400)],
            exclusion_junctions=[(100, 400)],
            inclusion_transcripts=["tx1", "tx2"],
            exclusion_transcripts=["tx3"],
        )
        psi = PSIResult("SE:chr1:+:100_200_300_400", 0.7, 70, 30, 2, 1, 100, 0.6, 0.8)

        features = extract_event_features(event, psi)
        assert features.inclusion_count == 70
        assert features.exclusion_count == 30
        assert features.total_reads == 100
        assert abs(features.psi - 0.7) < 0.01
        assert features.n_inclusion_junctions == 2
        assert features.n_exclusion_junctions == 1
        assert features.n_inclusion_transcripts == 2
        assert features.n_exclusion_transcripts == 1
        assert features.event_type_onehot[0] == 1.0  # SE

    def test_nan_psi_replaced(self) -> None:
        """NaN PSI is replaced with 0.5 in features."""
        event = ASEvent(
            event_id="test",
            event_type=EventType.A5SS,
            gene_id="g1",
            chrom="chr1",
            strand="+",
        )
        psi = PSIResult("test", float("nan"), 0, 0, 1, 1, 0)
        features = extract_event_features(event, psi)
        assert features.psi == 0.5


class TestFeaturesToArray:
    """Tests for feature vector conversion."""

    def test_array_shape(self) -> None:
        """Feature array has expected shape."""
        features = EventFeatures()
        arr = features_to_array(features)
        assert arr.shape == (21,)
        assert arr.dtype == np.float64


class TestEventClassifier:
    """Tests for the EventClassifier class."""

    def test_untrained_uses_heuristic(self) -> None:
        """Untrained classifier falls back to heuristic."""
        clf = EventClassifier()
        assert not clf.is_trained
        features = EventFeatures(
            total_reads=50,
            ci_width=0.2,
            junction_balance=0.7,
            log_total_reads=math.log1p(50),
        )
        score = clf.score(features)
        assert 0.0 <= score <= 1.0

    def test_too_few_samples_stays_heuristic(self) -> None:
        """Training with < 10 samples keeps heuristic fallback."""
        clf = EventClassifier()
        X = np.random.rand(5, 21)
        y = np.array([0, 1, 0, 1, 0])
        clf.train(X, y)
        assert not clf.is_trained

    def test_train_and_score(self) -> None:
        """Train with sufficient data, then score."""
        clf = EventClassifier(n_estimators=10, max_depth=2)
        rng = np.random.RandomState(42)
        X = rng.rand(50, 21)
        y = (X[:, 2] > 0.5).astype(int)  # Use total_reads column as signal
        clf.train(X, y)
        assert clf.is_trained

        features = EventFeatures(total_reads=80, log_total_reads=math.log1p(80))
        score = clf.score(features)
        assert 0.0 <= score <= 1.0

    def test_score_batch(self) -> None:
        """Score a batch of events."""
        clf = EventClassifier()
        events = [
            ASEvent("e1", EventType.SE, "g1", "chr1", "+"),
            ASEvent("e2", EventType.A5SS, "g1", "chr1", "+"),
        ]
        psi_results = [
            PSIResult("e1", 0.7, 70, 30, 2, 1, 100, 0.6, 0.8),
            PSIResult("e2", 0.3, 30, 70, 1, 1, 100, 0.2, 0.4),
        ]
        scores = clf.score_batch(events, psi_results)
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)


# ---------------------------------------------------------------------------
# I/O tests
# ---------------------------------------------------------------------------


class TestEventsTSV:
    """Tests for TSV I/O."""

    def test_write_and_read(self, tmp_path: Path) -> None:
        """Round-trip write and read events TSV."""
        events = [
            ASEvent(
                event_id="SE:chr1:+:200_300_400_500",
                event_type=EventType.SE,
                gene_id="g1",
                chrom="chr1",
                strand="+",
                coordinates={"upstream_exon_end": 200, "skipped_exon_start": 300},
                inclusion_transcripts=["tx1"],
                exclusion_transcripts=["tx2"],
                inclusion_junctions=[(200, 300), (400, 500)],
                exclusion_junctions=[(200, 500)],
            ),
        ]
        psi_results = [
            PSIResult("SE:chr1:+:200_300_400_500", 0.75, 75, 25, 2, 1, 100, 0.65, 0.85),
        ]
        scores = [0.92]

        tsv_path = tmp_path / "events.tsv"
        write_events_tsv(tsv_path, events, psi_results, scores)

        rows = read_events_tsv(tsv_path)
        assert len(rows) == 1
        row = rows[0]
        assert row["event_id"] == "SE:chr1:+:200_300_400_500"
        assert row["event_type"] == "SE"
        assert row["gene_id"] == "g1"
        assert float(row["psi"]) == pytest.approx(0.75, abs=0.01)
        assert float(row["confidence_score"]) == pytest.approx(0.92, abs=0.01)

    def test_empty_events(self, tmp_path: Path) -> None:
        """Writing empty events creates a header-only file."""
        tsv_path = tmp_path / "empty.tsv"
        write_events_tsv(tsv_path, [], [])
        rows = read_events_tsv(tsv_path)
        assert len(rows) == 0


class TestIOE:
    """Tests for SUPPA2-compatible IOE output."""

    def test_write_ioe(self, tmp_path: Path) -> None:
        """Write events in IOE format."""
        events = [
            ASEvent(
                event_id="SE:chr1:+:200_300_400_500",
                event_type=EventType.SE,
                gene_id="g1",
                chrom="chr1",
                strand="+",
                inclusion_transcripts=["tx1", "tx3"],
                exclusion_transcripts=["tx2"],
            ),
        ]
        ioe_path = tmp_path / "events.ioe"
        write_ioe(ioe_path, events)

        with open(ioe_path) as fh:
            lines = fh.readlines()
        assert len(lines) == 2  # header + 1 event
        assert "tx1" in lines[1]
        assert "tx2" in lines[1]
