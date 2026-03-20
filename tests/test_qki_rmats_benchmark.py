"""Tests for the QKI rMATS + BRAID benchmark runner."""

from __future__ import annotations

import json

from benchmarks import qki_rmats_benchmark
from rapidsplice.target.rmats_bootstrap import RmatsEvent


def _write_rmats_se(
    path,
    *,
    gene: str,
    chrom: str,
    exon_start: int,
    exon_end: int,
    inc1: str,
    exc1: str,
    inc2: str,
    exc2: str,
    inc_level1: str = "0.8,0.7",
    inc_level2: str = "0.3",
    fdr: str = "0.01",
    dpsi: str = "0.4",
) -> None:
    """Write one minimal SE.MATS.JunctionCountOnly.txt file."""
    path.write_text(
        "\t".join([
            "ID", "GeneID", "geneSymbol", "chr", "strand",
            "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE",
            "downstreamES", "downstreamEE", "ID", "IJC_SAMPLE_1",
            "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
            "IncFormLen", "SkipFormLen", "PValue", "FDR",
            "IncLevel1", "IncLevel2", "IncLevelDifference",
        ]) + "\n" + "\t".join([
            "1", '"ENSG0001"', gene, chrom, "+",
            str(exon_start), str(exon_end),
            "50", "99", "201", "250", "1",
            inc1, exc1, inc2, exc2,
            "197", "99", "0.001", fdr,
            inc_level1, inc_level2, dpsi,
        ]) + "\n",
        encoding="utf-8",
    )


def test_run_qki_rmats_benchmark_tracks_overlap(tmp_path) -> None:
    """QKI rMATS benchmark should keep failed/validated overlap separate."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t100\t200\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t100\t200\n"
        "G2\tchr1\t300\t350\n",
        encoding="utf-8",
    )

    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    _write_rmats_se(
        rmats_dir / "SE.MATS.JunctionCountOnly.txt",
        gene="G1",
        chrom="chr1",
        exon_start=100,
        exon_end=200,
        inc1="10,8",
        exc1="1,0",
        inc2="3",
        exc2="7",
    )

    output_path = tmp_path / "results" / "qki_rmats.json"
    results = qki_rmats_benchmark.run_qki_rmats_benchmark(
        rmats_dir=str(rmats_dir),
        qki_dir=str(qki_dir),
        output_path=str(output_path),
        tolerance=10,
        min_overlap_fraction=0.5,
        target_normalization_max_delta=500,
        min_total_count=1,
        n_replicates=25,
        confidence_level=0.95,
        fdr_threshold=0.05,
        high_confidence_dpsi=0.1,
        high_confidence_min_total_support=20,
        seed=42,
    )

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert results["validated_summary"]["matched_targets"] == 1
    assert results["failed_summary"]["matched_targets"] == 1
    assert saved["overlap_failed_summary"]["matched_targets"] == 1

    row = results["validated"][0]
    assert row["matched"] is True
    assert row["significant"] is True
    assert row["ctrl_support_total"] == 19
    assert row["kd_support_total"] == 10
    assert row["either_confident"] in {True, False}
    assert "null_calibration" in results
    assert "supported_probability_threshold" in results["metadata"]


def test_run_qki_rmats_benchmark_reports_high_confidence_matches(tmp_path) -> None:
    """High-confidence calls should require FDR, dPSI, and a posterior that excludes zero."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t100\t200\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n",
        encoding="utf-8",
    )

    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    _write_rmats_se(
        rmats_dir / "SE.MATS.JunctionCountOnly.txt",
        gene="G1",
        chrom="chr1",
        exon_start=100,
        exon_end=200,
        inc1="300,280",
        exc1="1,0",
        inc2="0,1",
        exc2="310,290",
        inc_level1="0.96,0.94",
        inc_level2="0.01,0.03",
        fdr="0.0001",
        dpsi="0.93",
    )

    results = qki_rmats_benchmark.run_qki_rmats_benchmark(
        rmats_dir=str(rmats_dir),
        qki_dir=str(qki_dir),
        output_path=str(tmp_path / "qki_high_conf.json"),
        tolerance=10,
        min_overlap_fraction=0.5,
        target_normalization_max_delta=500,
        min_total_count=1,
        n_replicates=200,
        confidence_level=0.95,
        fdr_threshold=0.05,
        high_confidence_dpsi=0.1,
        high_confidence_min_total_support=20,
        seed=42,
    )

    row = results["validated"][0]
    assert row["matched"] is True
    assert row["significant"] is True
    assert row["braid_dpsi_excludes_zero"] is True
    assert row["supported_differential"] is True
    assert row["high_confidence"] is True
    assert results["validated_summary"]["supported_matches"] == 1
    assert results["validated_summary"]["high_confidence_matches"] == 1


def test_run_qki_rmats_benchmark_can_match_by_overlap(tmp_path) -> None:
    """Overlap-based matching should rescue small coordinate shifts."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\t1\t100\t200\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n",
        encoding="utf-8",
    )

    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    _write_rmats_se(
        rmats_dir / "SE.MATS.JunctionCountOnly.txt",
        gene="G1",
        chrom="chr1",
        exon_start=140,
        exon_end=240,
        inc1="5,5",
        exc1="0,1",
        inc2="1",
        exc2="4",
    )

    results = qki_rmats_benchmark.run_qki_rmats_benchmark(
        rmats_dir=str(rmats_dir),
        qki_dir=str(qki_dir),
        output_path=str(tmp_path / "qki_overlap.json"),
        tolerance=10,
        min_overlap_fraction=0.5,
        target_normalization_max_delta=500,
        min_total_count=1,
        n_replicates=20,
        confidence_level=0.95,
        fdr_threshold=0.05,
        high_confidence_dpsi=0.1,
        high_confidence_min_total_support=20,
        seed=7,
    )

    row = results["validated"][0]
    assert row["matched"] is True
    assert row["overlap_bp"] == 60
    assert row["coord_delta"] == 80


def test_run_qki_rmats_benchmark_can_match_via_template_normalization(tmp_path) -> None:
    """Nearest fromGTF template should rescue moderate coordinate offsets."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\t1\t100\t150\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n",
        encoding="utf-8",
    )

    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    _write_rmats_se(
        rmats_dir / "SE.MATS.JC.txt",
        gene="G1",
        chrom="chr1",
        exon_start=200,
        exon_end=250,
        inc1="5,5",
        exc1="0,1",
        inc2="1",
        exc2="4",
    )
    (rmats_dir / "fromGTF.SE.txt").write_text(
        "\t".join([
            "ID", "GeneID", "geneSymbol", "chr", "strand",
            "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE",
            "downstreamES", "downstreamEE",
        ]) + "\n" + "\t".join([
            "1", '"ENSG0001"', "G1", "chr1", "+",
            "200", "250", "50", "99", "301", "350",
        ]) + "\n",
        encoding="utf-8",
    )

    results = qki_rmats_benchmark.run_qki_rmats_benchmark(
        rmats_dir=str(rmats_dir),
        qki_dir=str(qki_dir),
        output_path=str(tmp_path / "qki_norm.json"),
        tolerance=10,
        min_overlap_fraction=0.5,
        target_normalization_max_delta=250,
        min_total_count=1,
        n_replicates=20,
        confidence_level=0.95,
        fdr_threshold=0.05,
        high_confidence_dpsi=0.1,
        high_confidence_min_total_support=20,
        seed=7,
    )

    row = results["validated"][0]
    assert row["matched"] is True
    assert row["match_basis"] == "normalized"
    assert row["target_was_normalized"] is True
    assert row["normalized_target_start"] == 200
    assert row["normalized_target_end"] == 250


def test_run_qki_rmats_benchmark_prefers_lifted_hg38_target_tables(tmp_path) -> None:
    """rMATS benchmark should also resolve lifted hg38 target tables first."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\t1\t100\t150\n",
        encoding="utf-8",
    )
    (qki_dir / "validated_events.hg38.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\t1\t200\t250\n",
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

    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    _write_rmats_se(
        rmats_dir / "SE.MATS.JC.txt",
        gene="G1",
        chrom="chr1",
        exon_start=200,
        exon_end=250,
        inc1="5,5",
        exc1="0,1",
        inc2="1",
        exc2="4",
    )

    results = qki_rmats_benchmark.run_qki_rmats_benchmark(
        rmats_dir=str(rmats_dir),
        qki_dir=str(qki_dir),
        output_path=str(tmp_path / "qki_lifted.json"),
        tolerance=10,
        min_overlap_fraction=0.5,
        target_normalization_max_delta=250,
        min_total_count=1,
        n_replicates=20,
        confidence_level=0.95,
        fdr_threshold=0.05,
        high_confidence_dpsi=0.1,
        high_confidence_min_total_support=20,
        seed=7,
    )

    row = results["validated"][0]
    assert row["matched"] is True
    assert row["target_start"] == 200
    assert row["target_end"] == 250
    assert results["metadata"]["validated_targets_build"] == "hg38_lifted_from_hg19"


def test_run_qki_rmats_benchmark_can_force_raw_target_tables(tmp_path) -> None:
    """Raw target mode should bypass lifted hg38 tables."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\t1\t100\t150\n",
        encoding="utf-8",
    )
    (qki_dir / "validated_events.hg38.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\t1\t200\t250\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n",
        encoding="utf-8",
    )

    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    _write_rmats_se(
        rmats_dir / "SE.MATS.JC.txt",
        gene="G1",
        chrom="chr1",
        exon_start=100,
        exon_end=150,
        inc1="5,5",
        exc1="0,1",
        inc2="1",
        exc2="4",
    )

    results = qki_rmats_benchmark.run_qki_rmats_benchmark(
        rmats_dir=str(rmats_dir),
        qki_dir=str(qki_dir),
        output_path=str(tmp_path / "qki_raw.json"),
        tolerance=10,
        min_overlap_fraction=0.5,
        target_normalization_max_delta=250,
        min_total_count=1,
        n_replicates=20,
        confidence_level=0.95,
        fdr_threshold=0.05,
        high_confidence_dpsi=0.1,
        high_confidence_min_total_support=20,
        seed=7,
        prefer_lifted_targets=False,
    )

    row = results["validated"][0]
    assert row["matched"] is True
    assert row["target_start"] == 100
    assert row["target_end"] == 150
    assert results["metadata"]["validated_targets_build"] == "source_table"


def test_build_null_calibration_uses_support_bin_fallback() -> None:
    """Empty low-support bins should fall back to the next larger populated bin."""
    events = []
    for idx in range(30):
        events.append(RmatsEvent(
            event_id=f"SE:chr1:{100 + idx}-{150 + idx}",
            event_type="SE",
            chrom="chr1",
            strand="+",
            gene="G1",
            inc_count=0,
            exc_count=0,
            rmats_psi=0.5,
            rmats_fdr=0.8,
            rmats_dpsi=0.0,
            exon_start=100 + idx,
            exon_end=150 + idx,
            sample_1_inc_count=18,
            sample_1_exc_count=12,
            sample_2_inc_count=10,
            sample_2_exc_count=20,
            sample_1_inc_replicates=(18, 18),
            sample_1_exc_replicates=(12, 12),
            sample_2_inc_replicates=(10,),
            sample_2_exc_replicates=(20,),
        ))

    calibration = qki_rmats_benchmark._build_null_calibration(
        events,
        n_replicates=20,
        confidence_level=0.95,
        effect_cutoff=0.1,
        min_total_support=20,
        target_null_rate=0.05,
        minimum_threshold=0.5,
        sparse_bin_min_events=25,
        seed=42,
    )

    assert calibration["bins"]["50-99"]["event_count"] == 30
    assert calibration["bins"]["20-49"]["threshold_source"] == "fallback:50-99"


def test_run_qki_rmats_benchmark_reports_matched_null_controls(tmp_path) -> None:
    """Matched null controls should be emitted separately from failed cohorts."""
    qki_dir = tmp_path / "qki"
    qki_dir.mkdir()
    (qki_dir / "validated_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n"
        "G1\tchr1\t100\t200\n",
        encoding="utf-8",
    )
    (qki_dir / "failed_events.tsv").write_text(
        "gene\tchrom\texon_start\texon_end\n",
        encoding="utf-8",
    )

    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join([
            "ID", "GeneID", "geneSymbol", "chr", "strand",
            "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE",
            "downstreamES", "downstreamEE", "IJC_SAMPLE_1",
            "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
            "IncLevel1", "IncLevel2", "FDR", "IncLevelDifference",
        ]) + "\n" + "\n".join([
            "\t".join([
                "1", "G1", "G1", "chr1", "+",
                "100", "200", "50", "99", "201", "260",
                "10,8", "1,0", "3", "7", "0.8,0.7", "0.3", "0.01", "0.4",
            ]),
            "\t".join([
                "2", "G2", "G2", "chr1", "+",
                "400", "500", "320", "399", "501", "560",
                "12,10", "11,9", "9", "10", "0.5,0.5", "0.47", "0.8", "0.02",
            ]),
        ]) + "\n",
        encoding="utf-8",
    )

    results = qki_rmats_benchmark.run_qki_rmats_benchmark(
        rmats_dir=str(rmats_dir),
        qki_dir=str(qki_dir),
        output_path=str(tmp_path / "qki_null.json"),
        tolerance=10,
        min_overlap_fraction=0.5,
        target_normalization_max_delta=500,
        min_total_count=1,
        n_replicates=20,
        confidence_level=0.95,
        fdr_threshold=0.05,
        high_confidence_dpsi=0.1,
        high_confidence_min_total_support=20,
        seed=42,
    )

    assert results["matched_null_control_summary"]["total_targets"] == 1
    assert len(results["matched_null_control"]) == 1
    assert results["matched_null_control"][0]["cohort"] == "matched_null_control"
    assert results["matched_null_control_selection"]["selected_count"] == 1
