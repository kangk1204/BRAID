"""Regression tests for single-process vs multiprocess assembly parity.

The per-chromosome assembly logic is implemented twice: ``_process_chromosome``
(single-process method) and ``_process_chromosome_worker`` (multiprocess worker).
Two real divergences were found and fixed:

1. The worker raised ``UnboundLocalError: se_records`` and crashed the whole
   process pool when ``disable_single_exon=True`` (the method handled it).
2. The worker honoured ``cfg.min_relative_abundance`` (default 0.01) while the
   method silently used the ``DecomposeConfig`` default (0.02), so the two paths
   pruned transcripts at different thresholds depending on thread count.

Both paths now build their decomposition config via the shared
``_make_decompose_config`` so they cannot drift again.
"""
from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest

from braid.pipeline import (
    AssemblyPipeline,
    PipelineConfig,
    _make_decompose_config,
    _process_chromosome_worker,
)

pysam = pytest.importorskip("pysam")


def _write_tiny_bam(path: Path) -> str:
    bam = str(path / "tiny.bam")
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.0", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 3000}],
        }
    )

    def seg(name: str, start: int, cigar: list[tuple[int, int]]) -> "pysam.AlignedSegment":
        a = pysam.AlignedSegment(header)
        a.query_name = name
        a.reference_id = 0
        a.reference_start = start
        a.mapping_quality = 60
        a.flag = 0
        a.cigartuples = cigar
        n = sum(length for op, length in cigar if op in (0, 1, 4))
        a.query_sequence = "A" * n
        a.query_qualities = pysam.qualitystring_to_array("I" * n)
        return a

    with pysam.AlignmentFile(bam, "wb", header=header) as fh:
        for i in range(20):
            fh.write(seg(f"r{i}", 100, [(0, 100), (3, 100), (0, 100)]))
    pysam.index(bam)
    return bam


def _config(bam: str, **overrides: object) -> PipelineConfig:
    base = dict(
        bam_path=bam,
        enable_motif_validation=False,
        enable_bootstrap=False,
        min_junction_support=1,
        min_coverage=0.5,
        min_transcript_score=0.0,
        use_ml_scoring=False,
    )
    base.update(overrides)
    return PipelineConfig(**base)


@pytest.mark.parametrize("disable_single_exon", [True, False])
def test_worker_does_not_crash_on_single_exon_flag(
    tmp_path: Path, disable_single_exon: bool
) -> None:
    """The worker must run to completion for both single-exon settings.

    Regression for the UnboundLocalError that crashed the ProcessPoolExecutor
    whenever single-exon detection was disabled.
    """
    bam = _write_tiny_bam(tmp_path)
    cfg = _config(bam, disable_single_exon=disable_single_exon)

    records = _process_chromosome_worker(cfg, "chr1")

    assert isinstance(records, list)


def _write_two_chrom_bam(path: Path) -> str:
    """Tiny 2-chromosome BAM so threads>1 routes through the parallel worker."""
    bam = str(path / "two.bam")
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.0", "SO": "coordinate"},
            "SQ": [{"SN": "chr1", "LN": 3000}, {"SN": "chr2", "LN": 3000}],
        }
    )

    def seg(name: str, ref_id: int, start: int,
            cigar: list[tuple[int, int]]) -> "pysam.AlignedSegment":
        a = pysam.AlignedSegment(header)
        a.query_name = name
        a.reference_id = ref_id
        a.reference_start = start
        a.mapping_quality = 60
        a.flag = 0
        a.cigartuples = cigar
        n = sum(length for op, length in cigar if op in (0, 1, 4))
        a.query_sequence = "A" * n
        a.query_qualities = pysam.qualitystring_to_array("I" * n)
        return a

    with pysam.AlignmentFile(bam, "wb", header=header) as fh:
        for ref_id in (0, 1):
            for i in range(20):
                fh.write(seg(f"c{ref_id}r{i}", ref_id, 100,
                             [(0, 100), (3, 100), (0, 100)]))
    pysam.index(bam)
    return bam


def _parse_transcript_structures(gtf_path: str) -> set:
    """Structural transcript set from a GTF: {(chrom, strand, frozenset(exons))}.

    Compares by structure (coords/exons/strand), not transcript_id ordering, so
    it is robust to the order in which parallel workers return chromosomes.
    """
    exons_by_tx: dict[str, list[tuple[int, int]]] = {}
    meta: dict[str, tuple[str, str]] = {}
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "exon":
                continue
            tid = ""
            for attr in f[8].split(";"):
                attr = attr.strip()
                if attr.startswith("transcript_id"):
                    tid = attr.split('"')[1] if '"' in attr else attr.split()[-1]
                    break
            exons_by_tx.setdefault(tid, []).append((int(f[3]), int(f[4])))
            meta[tid] = (f[0], f[6])
    return {
        (meta[tid][0], meta[tid][1], frozenset(exons))
        for tid, exons in exons_by_tx.items()
    }


def test_single_process_and_parallel_produce_identical_transcripts(
    tmp_path: Path,
) -> None:
    """Single-process (method) and parallel (worker) paths must agree.

    This is the parity guarantee behind the two duplicated per-chromosome
    implementations. Under default config (no NMF, no shadow decomposer) the
    single-process ``_process_chromosome`` and the parallel
    ``_process_chromosome_worker`` must produce the SAME transcripts regardless of
    thread count. It locks the shared assembly contract so the two paths cannot
    silently drift — the class of bug behind the prior se_records crash and the
    min_relative_abundance threshold mismatch. (The worker is intentionally a
    reduced-feature subset: it omits the NMF/shadow/diagnostics paths the method
    adds, so this asserts equality only on the shared default-config behaviour.)
    """
    bam = _write_two_chrom_bam(tmp_path)

    out1 = str(tmp_path / "single.gtf")
    AssemblyPipeline(_config(bam, output_path=out1, threads=1)).run()
    single = _parse_transcript_structures(out1)

    out2 = str(tmp_path / "parallel.gtf")
    AssemblyPipeline(_config(bam, output_path=out2, threads=2)).run()
    parallel = _parse_transcript_structures(out2)

    assert single, "single-process run produced no transcripts (test BAM too sparse)"
    assert single == parallel


def test_shared_decompose_config_honours_min_relative_abundance() -> None:
    """Both paths build DecomposeConfig via _make_decompose_config, which must
    propagate the user-facing pruning threshold (previously dropped by the
    single-process method)."""
    cfg = PipelineConfig(bam_path="unused.bam", min_relative_abundance=0.123)

    dc = _make_decompose_config(cfg)

    assert dc.min_relative_abundance == 0.123
    assert dc.max_paths == cfg.max_paths
    assert dc.candidate_budget == cfg.candidate_budget


def test_parallel_processing_raises_on_partial_chromosome_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed worker chromosome must not be silently dropped from output."""

    class FakeFuture:
        def __init__(self, chrom: str) -> None:
            self.chrom = chrom

        def result(self):
            if self.chrom == "chr2":
                raise RuntimeError("boom")
            return []

    class FakePool:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def submit(self, fn, cfg, chrom):
            return FakeFuture(chrom)

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(
        concurrent.futures,
        "as_completed",
        lambda futures: list(futures),
    )
    pipeline = AssemblyPipeline(_config("unused.bam", threads=2))

    with pytest.raises(RuntimeError, match="chr2"):
        pipeline._process_chromosomes_parallel(["chr1", "chr2"], n_workers=2)


def test_nmf_with_threads_falls_back_to_single_threaded(tmp_path: Path, caplog) -> None:
    """A semantic flag (--nmf) must not be silently dropped by the parallel path.

    Parallel workers do not apply NMF, so requesting it with threads>1 must fall
    back to single-threaded with a warning rather than silently changing the
    decomposition algorithm.
    """
    import logging

    bam = _write_two_chrom_bam(tmp_path)
    out = str(tmp_path / "nmf.gtf")
    with caplog.at_level(logging.WARNING):
        AssemblyPipeline(
            _config(bam, output_path=out, threads=2, use_nmf_decomposition=True)
        ).run()
    assert "running single-threaded to preserve" in caplog.text


def test_parallel_warns_diagnostics_not_aggregated(tmp_path: Path, caplog) -> None:
    """Parallel assembly must flag that per-chromosome diagnostics and summary
    stats are not aggregated, so the dashboard's zero is not silently misleading."""
    import logging

    bam = _write_two_chrom_bam(tmp_path)
    out = str(tmp_path / "par.gtf")
    with caplog.at_level(logging.WARNING):
        AssemblyPipeline(_config(bam, output_path=out, threads=2)).run()
    assert "not aggregated across workers" in caplog.text


@pytest.mark.parametrize(
    "flag",
    ["use_neural_decomposition", "use_junction_scoring", "use_boundary_detection"],
)
def test_unwired_ml_flags_fail_fast(flag: str) -> None:
    """Flags whose feature is constructed but never consumed must fail fast at
    config time, consistent with use_gnn_scoring, rather than silently running a
    plain assembly that ignores the requested feature."""
    with pytest.raises(NotImplementedError, match=flag):
        PipelineConfig(bam_path="unused.bam", **{flag: True})


def test_default_config_does_not_fail_fast() -> None:
    """The fail-fast guard must not fire for a default configuration."""
    cfg = PipelineConfig(bam_path="unused.bam")
    assert cfg.use_neural_decomposition is False
    assert cfg.use_junction_scoring is False
    assert cfg.use_boundary_detection is False
