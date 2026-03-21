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
