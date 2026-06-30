"""Regression tests for fastq-target alignment subprocess error handling.

``align_fastq_to_target`` piped hisat2 -> samtools sort but previously
never inspected either process's return code, so an external-tool failure was
indistinguishable from a successful run that produced zero alignments (both
returned ``(0, 0, elapsed)``). The fix raises ``RuntimeError`` with captured
stderr on a non-zero exit, and only treats a clean exit with an empty BAM as the
genuine zero-alignment case.
"""

import subprocess

import pytest

import braid.target.fastq_pipeline as fp


class _FakeStream:
    def __init__(self, data: bytes = b""):
        self._data = data

    def close(self):
        pass

    def read(self):
        return self._data


class _FakeProc:
    """Minimal stand-in for subprocess.Popen handles."""

    def __init__(self, returncode: int, stderr: bytes = b""):
        self.returncode = returncode
        self.stdout = _FakeStream()
        self.stderr = _FakeStream(stderr)

    def communicate(self):
        return (b"", self.stderr.read())

    def wait(self):
        return self.returncode


def _install_fakes(monkeypatch, hisat2_rc: int, sort_rc: int,
                   hisat2_err=b"", sort_err=b""):
    # hisat2-build runs via subprocess.run(check=True): make it a no-op success.
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a[0] if a else [], 0, b"", b""),
    )

    procs = iter([
        _FakeProc(hisat2_rc, hisat2_err),  # hisat2
        _FakeProc(sort_rc, sort_err),      # samtools sort
    ])
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: next(procs))


def test_align_rejects_empty_fastq_list_before_external_tools(monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be reached")

    monkeypatch.setattr(subprocess, "run", fail_run)
    with pytest.raises(ValueError, match="At least one FASTQ"):
        fp.align_fastq_to_target([], "tgt.fa", "out.bam", threads=1)


def test_align_rejects_more_than_two_fastqs_before_external_tools(monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be reached")

    monkeypatch.setattr(subprocess, "run", fail_run)
    with pytest.raises(ValueError, match="Expected one single-end FASTQ"):
        fp.align_fastq_to_target(["r1.fq", "r2.fq", "r3.fq"], "tgt.fa", "out.bam")


def test_clamp_region_to_reference_caps_at_contig_end(monkeypatch):
    class _FakeFasta:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get_reference_length(self, chrom):
            assert chrom == "chr1"
            return 100

    monkeypatch.setattr(fp.pysam, "FastaFile", lambda path: _FakeFasta())

    assert fp._clamp_region_to_reference("ref.fa", "chr1", 90, 150) == (90, 100)
    assert fp._clamp_region_to_reference("ref.fa", "chr1", 120, 150) == (100, 100)


def test_hisat2_nonzero_exit_raises(monkeypatch):
    _install_fakes(monkeypatch, hisat2_rc=1, sort_rc=0,
                   hisat2_err=b"hisat2: fatal index error")
    with pytest.raises(RuntimeError, match="hisat2 alignment failed"):
        fp.align_fastq_to_target(["r1.fq"], "tgt.fa", "out.bam", threads=1)


def test_samtools_sort_nonzero_exit_raises(monkeypatch):
    _install_fakes(monkeypatch, hisat2_rc=0, sort_rc=139,
                   sort_err=b"samtools sort: truncated input")
    with pytest.raises(RuntimeError, match="samtools sort failed"):
        fp.align_fastq_to_target(["r1.fq"], "tgt.fa", "out.bam", threads=1)


def test_clean_exit_empty_bam_is_zero_alignment(monkeypatch, tmp_path):
    """Both tools exit 0 but the BAM is missing -> genuine zero alignments."""
    _install_fakes(monkeypatch, hisat2_rc=0, sort_rc=0)
    out_bam = str(tmp_path / "missing.bam")  # never created by the fakes
    n_aligned, n_spliced, elapsed = fp.align_fastq_to_target(
        ["r1.fq"], "tgt.fa", out_bam, threads=1,
    )
    assert (n_aligned, n_spliced) == (0, 0)
    assert elapsed >= 0.0


def test_pipe_helper_reaps_aligner_when_sorter_launch_fails(monkeypatch):
    """If the sorter fails to launch, the already-started aligner must be reaped.

    Otherwise the aligner keeps running, fills its stdout pipe (no reader), and
    blocks until non-deterministic GC closes the fd — a process leak in a
    long-running host. The helper must terminate and wait the aligner before
    re-raising the launch error.
    """
    events: dict[str, bool] = {}

    class _Stream:
        def close(self):
            events["stdout_closed"] = True

    class _Aligner:
        def __init__(self):
            self.stdout = _Stream()

        def terminate(self):
            events["terminated"] = True

        def wait(self):
            events["waited"] = True
            return -15

    calls = {"n": 0}

    def fake_popen(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Aligner()  # aligner launches
        raise FileNotFoundError("samtools: command not found")  # sorter fails

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    with pytest.raises(FileNotFoundError):
        fp._pipe_aligner_to_sorter(["aligner"], ["sorter"])
    assert events.get("terminated") is True
    assert events.get("waited") is True
    assert events.get("stdout_closed") is True


def test_pipe_helper_drains_large_aligner_stderr_without_deadlock():
    """A flood of aligner stderr must not deadlock the downstream sorter.

    The producer writes ~2 MB to stderr — far above the ~64 KB OS pipe buffer —
    BEFORE emitting its single stdout line, while the consumer only reads stdout.
    If stderr were an unread ``PIPE`` the producer would block on a full stderr
    buffer, never close stdout, and ``communicate()`` on the consumer would hang
    forever. Draining stderr to a temp file keeps both stages live. Uses real
    ``sh``/``cat`` subprocesses because pipe backpressure cannot be faked.
    """
    producer = [
        "sh", "-c",
        "i=0; while [ $i -lt 40000 ]; do "
        "echo xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx 1>&2; "
        "i=$((i+1)); done; echo PAYLOAD",
    ]
    consumer = ["cat"]
    aligner_ret, aligner_err, sort_ret, sort_err = fp._pipe_aligner_to_sorter(
        producer, consumer,
    )
    assert aligner_ret == 0
    assert sort_ret == 0
    # The full stderr flood was captured (proves it was drained, not dropped).
    assert len(aligner_err) > 100_000
