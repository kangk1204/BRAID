"""Tests for the BRAID QKI benchmark runner."""

from __future__ import annotations

import json

from benchmarks import qki_benchmark
from rapidsplice.target.psi_bootstrap import PSIResult


def _make_se_result(
    *,
    gene: str,
    chrom: str,
    start: int,
    end: int,
    event_id: str,
    support_total: int,
    confident: bool = True,
) -> PSIResult:
    """Construct one synthetic SE benchmark result."""
    inc = support_total // 2
    exc = support_total - inc
    return PSIResult(
        event_id=event_id,
        event_type="SE",
        gene=gene,
        chrom=chrom,
        psi=0.6,
        ci_low=0.5,
        ci_high=0.6 if confident else 0.85,
        cv=0.2 if confident else 0.8,
        inclusion_count=inc,
        exclusion_count=exc,
        event_start=start,
        event_end=end,
        ci_width=0.1 if confident else 0.35,
        is_confident=confident,
    )


def test_run_qki_benchmark_tracks_overlap_and_matches(tmp_path, monkeypatch) -> None:
    """Overlapping failed targets should be reported separately, not as negatives."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t100\t200\n"
        "G2\tchr1\t300\t350\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t100\t200\n"
        "G3\tchr2\t500\t600\n",
        encoding="utf-8",
    )

    def fake_compute(
        bam_path: str,
        chrom: str,
        start: int,
        end: int,
        *,
        gene: str | None = None,
        n_replicates: int = 0,
        confidence_level: float = 0.95,
        min_mapq: int = 1,
        seed: int | None = None,
        annotation_gtf: str | None = None,
        event_source: str = "hybrid",
        uncertainty_model: str = "overdispersed",
        min_event_support: int = 5,
    ) -> list[PSIResult]:
        del (
            bam_path,
            start,
            end,
            n_replicates,
            confidence_level,
            min_mapq,
            seed,
            annotation_gtf,
            event_source,
            uncertainty_model,
            min_event_support,
        )
        if gene == "G1":
            return [
                _make_se_result(
                    gene="G1",
                    chrom=chrom,
                    start=100,
                    end=200,
                    event_id="SE:100-200",
                    support_total=20,
                ),
                _make_se_result(
                    gene="G1",
                    chrom=chrom,
                    start=101,
                    end=199,
                    event_id="SE:101-199",
                    support_total=99,
                ),
            ]
        if gene == "G3":
            return [
                _make_se_result(
                    gene="G3",
                    chrom=chrom,
                    start=500,
                    end=600,
                    event_id="SE:500-600",
                    support_total=8,
                    confident=False,
                ),
            ]
        return []

    monkeypatch.setattr(qki_benchmark, "compute_psi_from_junctions", fake_compute)

    output_path = tmp_path / "results" / "braid.json"
    results = qki_benchmark.run_qki_benchmark(
        bam_path="dummy.bam",
        qki_dir=str(qki_dir),
        gtf_path="dummy.gtf",
        output_path=str(output_path),
        window=100,
        tolerance=10,
        n_replicates=25,
        confidence_level=0.95,
        min_mapq=1,
        seed=42,
    )

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert results["metadata"]["label_overlap_count"] == 1
    assert saved["metadata"]["label_overlap_count"] == 1
    assert results["validated_summary"]["matched_targets"] == 1
    assert results["failed_summary"]["matched_targets"] == 2
    assert results["overlap_failed_summary"]["total_targets"] == 1
    assert results["overlap_failed_summary"]["matched_targets"] == 1
    assert results["exclusive_failed_summary"]["total_targets"] == 1
    assert results["exclusive_failed_summary"]["matched_targets"] == 1

    overlap_failed = next(
        row for row in results["failed"]
        if row["gene"] == "G1"
    )
    assert overlap_failed["overlap_with_validated"] is True
    assert overlap_failed["matched"] is True
    assert overlap_failed["event_id"] == "SE:100-200"
    assert overlap_failed["n_matching_candidates"] == 2


def test_run_qki_multi_sample_benchmark_aggregates_matches(
    tmp_path,
    monkeypatch,
) -> None:
    """Aggregate output should mark targets detected in either sample."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t100\t200\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G3\tchr2\t500\t600\n",
        encoding="utf-8",
    )

    def fake_compute(
        bam_path: str,
        chrom: str,
        start: int,
        end: int,
        *,
        gene: str | None = None,
        n_replicates: int = 0,
        confidence_level: float = 0.95,
        min_mapq: int = 1,
        seed: int | None = None,
        annotation_gtf: str | None = None,
        event_source: str = "hybrid",
        uncertainty_model: str = "overdispersed",
        min_event_support: int = 5,
    ) -> list[PSIResult]:
        del (
            start,
            end,
            n_replicates,
            confidence_level,
            min_mapq,
            seed,
            annotation_gtf,
            event_source,
            uncertainty_model,
            min_event_support,
        )
        if bam_path == "ctrl.bam" and gene == "G1":
            return [
                _make_se_result(
                    gene="G1",
                    chrom=chrom,
                    start=100,
                    end=200,
                    event_id="SE:100-200",
                    support_total=10,
                ),
            ]
        if bam_path == "kd.bam" and gene == "G3":
            return [
                _make_se_result(
                    gene="G3",
                    chrom=chrom,
                    start=500,
                    end=600,
                    event_id="SE:500-600",
                    support_total=12,
                ),
            ]
        return []

    monkeypatch.setattr(qki_benchmark, "compute_psi_from_junctions", fake_compute)

    output_path = tmp_path / "results" / "multi.json"
    results = qki_benchmark.run_qki_multi_sample_benchmark(
        sample_bams={"ctrl": "ctrl.bam", "kd": "kd.bam"},
        qki_dir=str(qki_dir),
        gtf_path="dummy.gtf",
        output_path=str(output_path),
        window=100,
        tolerance=10,
        n_replicates=25,
        confidence_level=0.95,
        min_mapq=1,
        seed=42,
    )

    aggregate = results["detected_in_either_sample"]
    assert output_path.exists()
    assert sorted(results["samples"]) == ["ctrl", "kd"]
    assert aggregate["validated_summary"]["matched_targets"] == 1
    assert aggregate["failed_summary"]["matched_targets"] == 1
    validated_row = aggregate["validated"][0]
    failed_row = aggregate["failed"][0]
    assert validated_row["matched_samples"] == ["ctrl"]
    assert validated_row["best_sample"] == "ctrl"
    assert failed_row["matched_samples"] == ["kd"]
    assert failed_row["best_sample"] == "kd"


def test_run_qki_benchmark_prefers_lifted_hg38_target_tables(
    tmp_path,
    monkeypatch,
) -> None:
    """Lifted hg38 target tables should override raw source tables."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t100\t200\n",
        encoding="utf-8",
    )
    (qki_dir / "validated_events.hg38.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t500\t600\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.hg38.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n",
        encoding="utf-8",
    )

    def fake_compute(
        bam_path: str,
        chrom: str,
        start: int,
        end: int,
        *,
        gene: str | None = None,
        n_replicates: int = 0,
        confidence_level: float = 0.95,
        min_mapq: int = 1,
        seed: int | None = None,
        annotation_gtf: str | None = None,
        event_source: str = "hybrid",
        uncertainty_model: str = "overdispersed",
        min_event_support: int = 5,
    ) -> list[PSIResult]:
        del (
            bam_path,
            start,
            end,
            n_replicates,
            confidence_level,
            min_mapq,
            seed,
            annotation_gtf,
            event_source,
            uncertainty_model,
            min_event_support,
        )
        if gene == "G1":
            return [
                _make_se_result(
                    gene="G1",
                    chrom=chrom,
                    start=500,
                    end=600,
                    event_id="SE:500-600",
                    support_total=11,
                ),
            ]
        return []

    monkeypatch.setattr(qki_benchmark, "compute_psi_from_junctions", fake_compute)

    results = qki_benchmark.run_qki_benchmark(
        bam_path="dummy.bam",
        qki_dir=str(qki_dir),
        gtf_path="dummy.gtf",
        output_path=str(tmp_path / "results.json"),
        window=100,
        tolerance=10,
        n_replicates=25,
        confidence_level=0.95,
        min_mapq=1,
        seed=42,
    )

    row = results["validated"][0]
    assert row["matched"] is True
    assert row["target_start"] == 500
    assert row["target_end"] == 600
    assert results["metadata"]["validated_targets_build"] == "hg38_lifted_from_hg19"


def test_run_qki_benchmark_can_force_raw_target_tables(
    tmp_path,
    monkeypatch,
) -> None:
    """Raw target mode should ignore lifted hg38 tables."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t100\t200\n",
        encoding="utf-8",
    )
    (qki_dir / "validated_events.hg38.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t500\t600\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n",
        encoding="utf-8",
    )

    def fake_compute(
        bam_path: str,
        chrom: str,
        start: int,
        end: int,
        *,
        gene: str | None = None,
        n_replicates: int = 0,
        confidence_level: float = 0.95,
        min_mapq: int = 1,
        seed: int | None = None,
        annotation_gtf: str | None = None,
        event_source: str = "hybrid",
        uncertainty_model: str = "overdispersed",
        min_event_support: int = 5,
    ) -> list[PSIResult]:
        del (
            bam_path,
            start,
            end,
            n_replicates,
            confidence_level,
            min_mapq,
            seed,
            annotation_gtf,
            event_source,
            uncertainty_model,
            min_event_support,
        )
        if gene == "G1":
            return [
                _make_se_result(
                    gene="G1",
                    chrom=chrom,
                    start=100,
                    end=200,
                    event_id="SE:100-200",
                    support_total=11,
                ),
            ]
        return []

    monkeypatch.setattr(qki_benchmark, "compute_psi_from_junctions", fake_compute)

    results = qki_benchmark.run_qki_benchmark(
        bam_path="dummy.bam",
        qki_dir=str(qki_dir),
        gtf_path="dummy.gtf",
        output_path=str(tmp_path / "results.json"),
        window=100,
        tolerance=10,
        n_replicates=25,
        confidence_level=0.95,
        min_mapq=1,
        seed=42,
        prefer_lifted_targets=False,
    )

    row = results["validated"][0]
    assert row["matched"] is True
    assert row["target_start"] == 100
    assert row["target_end"] == 200
    assert results["metadata"]["validated_targets_build"] == "source_table"
