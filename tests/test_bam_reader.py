"""Tests for BAM reader filtering behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import braid.io.bam_reader as bam_reader_module


class _FakeRead:
    """Minimal pysam-aligned-segment stand-in."""

    def __init__(
        self,
        *,
        start: int,
        cigar: list[tuple[int, int]],
        flag: int = 0,
        mapq: int = 60,
        name: str = "read",
    ) -> None:
        self.flag = flag
        self.mapping_quality = mapq
        self.reference_start = start
        self.reference_id = 0
        self.query_name = name
        self.cigartuples = cigar
        self.is_unmapped = bool(flag & 0x4)
        self.is_reverse = bool(flag & 0x10)
        self.is_paired = False
        self.is_read1 = False
        self.mate_is_unmapped = True
        self.next_reference_start = -1
        self.next_reference_id = -1
        pos = start
        for op, length in cigar:
            if op in (0, 2, 3, 7, 8):
                pos += length
        self.reference_end = pos

    def has_tag(self, tag: str) -> bool:
        return False

    def get_tag(self, tag: str):
        raise KeyError(tag)


class _FakeAlignmentFile:
    """Fake pysam.AlignmentFile with a fixed header and read set."""

    def __init__(self, bam_path: str, mode: str) -> None:
        self.header = SimpleNamespace(references=("chr1",), lengths=(1000,))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def fetch(self, *args, **kwargs):
        return [
            _FakeRead(
                start=100,
                cigar=[(0, 50), (3, 100), (0, 50)],
                name="primary",
            ),
            _FakeRead(
                start=100,
                cigar=[(0, 50), (3, 100), (0, 50)],
                flag=0x800,
                name="supplementary",
            ),
        ]


def test_extract_junctions_from_bam_ignores_supplementary(monkeypatch, tmp_path: Path) -> None:
    """Supplementary alignments must not contribute duplicate junction support."""
    bam_path = tmp_path / "synthetic.bam"
    bam_path.write_bytes(b"")
    (tmp_path / "synthetic.bam.bai").write_bytes(b"")

    monkeypatch.setattr(
        bam_reader_module.pysam,
        "AlignmentFile",
        _FakeAlignmentFile,
    )

    evidence, n_spliced = bam_reader_module.extract_junctions_from_bam(
        str(bam_path),
        "chr1",
    )

    assert n_spliced == 1
    assert evidence.starts.tolist() == [150]
    assert evidence.ends.tolist() == [250]
    assert evidence.counts.tolist() == [1]


def test_bam_reader_accepts_csi_index(monkeypatch, tmp_path: Path) -> None:
    """CSI-indexed BAMs should pass the upfront index check."""
    bam_path = tmp_path / "synthetic.bam"
    bam_path.write_bytes(b"")
    (tmp_path / "synthetic.bam.csi").write_bytes(b"")

    monkeypatch.setattr(
        bam_reader_module.pysam,
        "AlignmentFile",
        _FakeAlignmentFile,
    )

    reader = bam_reader_module.BamReader(str(bam_path))

    assert reader.chromosomes == ["chr1"]


def test_bam_reader_accepts_cram_crai_index(monkeypatch, tmp_path: Path) -> None:
    """CRAM inputs routed by the CLI should accept the standard .crai index."""
    cram_path = tmp_path / "synthetic.cram"
    cram_path.write_bytes(b"")
    (tmp_path / "synthetic.cram.crai").write_bytes(b"")

    monkeypatch.setattr(
        bam_reader_module.pysam,
        "AlignmentFile",
        _FakeAlignmentFile,
    )

    reader = bam_reader_module.BamReader(str(cram_path))

    assert reader.chromosomes == ["chr1"]


def test_count_reads_honours_flag_filters(monkeypatch, tmp_path: Path) -> None:
    """count_reads must apply the configured flag/quality filters, not return the
    raw index ``mapped`` total.

    The fake BAM yields one primary plus one supplementary (0x800) read. The
    default filters exclude supplementary alignments, so the documented filtered
    count is 1, even though the index fast path (stat.mapped) would report 2.
    Regression for count_reads over-counting pipeline total_reads.
    """
    bam_path = tmp_path / "synthetic.bam"
    bam_path.write_bytes(b"")
    (tmp_path / "synthetic.bam.bai").write_bytes(b"")
    monkeypatch.setattr(bam_reader_module.pysam, "AlignmentFile", _FakeAlignmentFile)

    reader = bam_reader_module.BamReader(str(bam_path))  # default filter_flags
    assert reader.count_reads() == 1

    # With every flag/quality filter disabled, both reads are counted, confirming
    # the difference is the filter contract and not the read set.
    reader_unfiltered = bam_reader_module.BamReader(
        str(bam_path), min_mapq=0, required_flags=0, filter_flags=0
    )
    assert reader_unfiltered.count_reads() == 2


class _FakeAlignmentFileWithIndex(_FakeAlignmentFile):
    """Fake AlignmentFile that also exposes BAM index statistics."""

    def get_index_statistics(self):
        # Both fetched reads (primary + supplementary) are "mapped" in the index.
        return [SimpleNamespace(contig="chr1", mapped=2, unmapped=0, total=2)]


def test_count_reads_approximate_uses_index_total_over_filtered(
    monkeypatch, tmp_path: Path
) -> None:
    """approximate=True returns the fast index mapped total even when filters are
    active, trading exactness for startup speed; the default stays exact."""
    bam_path = tmp_path / "synthetic.bam"
    bam_path.write_bytes(b"")
    (tmp_path / "synthetic.bam.bai").write_bytes(b"")
    monkeypatch.setattr(
        bam_reader_module.pysam, "AlignmentFile", _FakeAlignmentFileWithIndex
    )

    reader = bam_reader_module.BamReader(str(bam_path))  # default filters drop supp.
    # Exact (default): supplementary filtered out -> 1.
    assert reader.count_reads() == 1
    # Approximate: index mapped total counts both reads -> 2 (fast upper bound).
    assert reader.count_reads(approximate=True) == 2


def test_single_end_strandedness_does_not_invent_read1_status() -> None:
    """Single-end rf/fr orientation should use the SE protocol directly."""
    infer = bam_reader_module._infer_strand_from_orientation

    assert infer(0x10, True, "rf") == 0
    assert infer(0, False, "rf") == 1
    assert infer(0x10, True, "fr") == 1
    assert infer(0, False, "fr") == 0
