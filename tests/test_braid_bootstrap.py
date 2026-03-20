"""Regression tests for BRAID PSI bootstrap utilities."""

from __future__ import annotations

import sys
import types

import pytest

from rapidsplice.target import psi_bootstrap as braid
from rapidsplice.target.rmats_bootstrap import parse_rmats_output


class FakeRead:
    """Minimal read stub for pysam-based unit tests."""

    def __init__(
        self,
        *,
        reference_start: int,
        cigartuples: list[tuple[int, int]] | None,
        mapping_quality: int = 60,
        is_secondary: bool = False,
        is_supplementary: bool = False,
        is_duplicate: bool = False,
        is_qcfail: bool = False,
        is_unmapped: bool = False,
        reference_end: int | None = None,
    ) -> None:
        self.reference_start = reference_start
        self.cigartuples = cigartuples
        self.mapping_quality = mapping_quality
        self.is_secondary = is_secondary
        self.is_supplementary = is_supplementary
        self.is_duplicate = is_duplicate
        self.is_qcfail = is_qcfail
        self.is_unmapped = is_unmapped

        if reference_end is None:
            pos = reference_start
            for op, length in cigartuples or []:
                if op in (0, 2, 3, 7, 8):
                    pos += length
            reference_end = pos
        self.reference_end = reference_end


def _install_fake_pysam(monkeypatch, fetch_map: dict[tuple[str, int, int], list]) -> None:
    """Install a fake pysam module that serves reads from a static mapping."""

    class FakeAlignmentFile:
        def __init__(self, bam_path: str, mode: str) -> None:
            self.bam_path = bam_path
            self.mode = mode

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def fetch(self, chrom: str, start: int, end: int):
            return list(fetch_map.get((chrom, start, end), []))

    fake_module = types.SimpleNamespace(AlignmentFile=FakeAlignmentFile)
    monkeypatch.setitem(sys.modules, "pysam", fake_module)


def test_compute_psi_filters_secondary_duplicate_and_low_mapq(monkeypatch) -> None:
    """Junction counts should ignore non-primary, duplicate, and low-MAPQ reads."""
    fetch_map = {
        ("chr1", 0, 500): [
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(
                reference_start=50,
                cigartuples=[(0, 50), (3, 100), (0, 10)],
                is_secondary=True,
            ),
            FakeRead(
                reference_start=50,
                cigartuples=[(0, 50), (3, 100), (0, 10)],
                is_duplicate=True,
            ),
            FakeRead(
                reference_start=50,
                cigartuples=[(0, 50), (3, 100), (0, 10)],
                mapping_quality=0,
            ),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 150), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 150), (0, 10)]),
        ],
    }
    _install_fake_pysam(monkeypatch, fetch_map)

    results = braid.compute_psi_from_junctions(
        "dummy.bam",
        "chr1",
        0,
        500,
        gene="GENE1",
        n_replicates=25,
        seed=7,
    )

    a3ss = {result.event_id: result for result in results if result.event_type == "A3SS"}
    assert set(a3ss) == {"A3SS:100-200", "A3SS:100-250"}
    assert a3ss["A3SS:100-200"].inclusion_count == 3
    assert a3ss["A3SS:100-200"].exclusion_count == 2
    assert a3ss["A3SS:100-250"].inclusion_count == 2
    assert a3ss["A3SS:100-250"].exclusion_count == 3
    assert all(result.gene == "GENE1" for result in a3ss.values())


def test_compute_psi_ri_excludes_spliced_reads(monkeypatch) -> None:
    """Spliced reads crossing an intron must not count as retained intron evidence."""
    fetch_map = {
        ("chr1", 0, 400): [
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
        ],
        ("chr1", 100, 200): [
            FakeRead(reference_start=95, cigartuples=[(0, 120)]),
            FakeRead(reference_start=100, cigartuples=[(0, 80)]),
            FakeRead(reference_start=95, cigartuples=[(0, 10), (3, 100), (0, 20)]),
        ],
    }
    _install_fake_pysam(monkeypatch, fetch_map)

    results = braid.compute_psi_from_junctions(
        "dummy.bam",
        "chr1",
        0,
        400,
        n_replicates=25,
        seed=11,
    )

    assert results == []


def test_confident_interval_requires_low_cv() -> None:
    """A narrow CI alone is not enough when PSI variability is high."""
    ci_width, is_confident = braid._is_confident_interval(0.20, 0.30, 0.75)
    assert ci_width == pytest.approx(0.10)
    assert is_confident is False


def test_effective_count_scale_support_schedule_is_native_for_a3_a5() -> None:
    """A3/A5 native mode should compress effective counts more at high support."""
    low_native = braid.effective_count_scale(4, 4, event_type="A3SS", schedule_mode="native")
    high_native = braid.effective_count_scale(200, 200, event_type="A3SS", schedule_mode="native")
    high_legacy = braid.effective_count_scale(200, 200, event_type="A3SS", schedule_mode="legacy")

    assert high_native < low_native
    assert high_native < high_legacy


def test_native_count_scale_schedule_exposes_audit_derived_bins() -> None:
    """The public native schedule helper should expose the calibrated support bins."""
    schedule = braid.native_count_scale_schedule(event_type="A3SS", base_scale=0.01)

    assert schedule["20-49"]["mode"] == "native_support_schedule"
    assert schedule["250+"]["effective_scale"] < schedule["20-49"]["effective_scale"]
    assert schedule["250+"]["audit_interval_factor"] > 1.0


def test_native_interval_inflation_schedule_is_support_aware() -> None:
    """Native interval inflation should also tighten by support bin."""
    schedule = braid.native_interval_inflation_schedule(event_type="A3SS")

    assert schedule["20-49"]["interval_inflation_factor"] > 1.0
    assert schedule["250+"]["interval_inflation_factor"] >= schedule["20-49"]["interval_inflation_factor"]


def test_native_confidence_width_schedule_relaxes_high_support_a3_a5() -> None:
    """High-support native A3/A5 should get a looser confidence width threshold."""
    schedule = braid.native_confidence_width_schedule(event_type="A3SS")
    ci_width, is_confident = braid._is_confident_interval(
        0.2,
        0.75,
        0.2,
        event_type="A3SS",
        inclusion_count=200,
        exclusion_count=200,
        schedule_mode="native",
    )

    assert schedule["250+"]["confidence_width_threshold"] == pytest.approx(0.60)
    assert ci_width == pytest.approx(0.55)
    assert is_confident is True


def test_native_confidence_cv_schedule_defaults_to_global_threshold() -> None:
    """Native CV schedule should default to the global confidence CV threshold."""
    schedule = braid.native_confidence_cv_schedule(event_type="A3SS")

    assert schedule["100-249"]["confidence_cv_threshold"] == pytest.approx(
        braid.CONFIDENT_CV_THRESHOLD
    )


def test_bootstrap_psi_uses_support_aware_scale() -> None:
    """Native high-support intervals should not exceed low-support width."""
    low = braid.bootstrap_psi(
        6,
        6,
        n_replicates=2000,
        seed=5,
        event_type="A3SS",
    )
    high = braid.bootstrap_psi(
        60,
        60,
        n_replicates=2000,
        seed=5,
        event_type="A3SS",
    )

    low_width = low[2] - low[1]
    high_width = high[2] - high[1]
    assert high_width <= low_width


def test_bootstrap_psi_native_is_wider_than_legacy_for_high_support_a3ss() -> None:
    """The native schedule should widen high-support A3SS intervals relative to legacy mode."""
    native = braid.bootstrap_psi(
        120,
        120,
        n_replicates=2000,
        seed=5,
        event_type="A3SS",
        schedule_mode="native",
    )
    legacy = braid.bootstrap_psi(
        120,
        120,
        n_replicates=2000,
        seed=5,
        event_type="A3SS",
        schedule_mode="legacy",
    )

    native_width = native[2] - native[1]
    legacy_width = legacy[2] - legacy[1]
    assert native_width > legacy_width


def test_fixed_schedule_mode_uses_unadjusted_base_scale() -> None:
    """Fixed mode should bypass support-aware compression."""
    fixed = braid.effective_count_scale(
        200,
        200,
        event_type="A3SS",
        base_scale=0.01,
        schedule_mode="fixed",
    )

    assert fixed == pytest.approx(0.01)


def test_custom_native_calibration_override_controls_scale_and_threshold() -> None:
    """Custom native schedules should override scale, interval factor, and confidence width."""
    override = {
        "base_scale": 0.01,
        "scale_by_bin": {"250+": 0.004},
        "interval_inflation_by_bin": {"250+": 1.5},
        "confidence_width_by_bin": {"250+": 0.45},
        "confidence_cv_by_bin": {"250+": 0.3},
        "training_scope": "unit_test",
    }

    scale = braid.effective_count_scale(
        200,
        200,
        event_type="A3SS",
        base_scale=0.01,
        schedule_mode="native",
        calibration_schedule=override,
    )
    ci_width, is_confident = braid._is_confident_interval(
        0.2,
        0.62,
        0.25,
        event_type="A3SS",
        inclusion_count=200,
        exclusion_count=200,
        schedule_mode="native",
        calibration_schedule=override,
    )
    widened = braid.bootstrap_psi(
        120,
        120,
        n_replicates=2000,
        seed=5,
        event_type="A3SS",
        schedule_mode="native",
        calibration_schedule=override,
    )
    baseline = braid.bootstrap_psi(
        120,
        120,
        n_replicates=2000,
        seed=5,
        event_type="A3SS",
        schedule_mode="fixed",
        base_scale=0.01,
    )

    assert scale == pytest.approx(0.004)
    assert ci_width == pytest.approx(0.42)
    assert is_confident is True
    assert (widened[2] - widened[1]) > (baseline[2] - baseline[1])

    _, rejected = braid._is_confident_interval(
        0.2,
        0.62,
        0.35,
        event_type="A3SS",
        inclusion_count=200,
        exclusion_count=200,
        schedule_mode="native",
        calibration_schedule=override,
    )
    assert rejected is False


def test_effect_aware_native_override_can_rescue_extreme_mid_support_event() -> None:
    """Effect-aware native gates should allow extreme mid-support events to be confident."""
    override = {
        "base_scale": 0.01,
        "confidence_effect_by_bin": {"50-99": 0.45},
        "confidence_effect_snr_by_bin": {"50-99": 0.50},
        "training_scope": "unit_test",
    }

    ci_width, rescued = braid._is_confident_interval(
        0.05,
        0.95,
        0.95,
        psi=0.99,
        event_type="A3SS",
        inclusion_count=40,
        exclusion_count=20,
        schedule_mode="native",
        calibration_schedule=override,
    )
    _, rejected = braid._is_confident_interval(
        0.05,
        0.95,
        0.95,
        psi=0.72,
        event_type="A3SS",
        inclusion_count=40,
        exclusion_count=20,
        schedule_mode="native",
        calibration_schedule=override,
    )

    assert ci_width == pytest.approx(0.90)
    assert rescued is True
    assert rejected is False


def test_effect_aware_native_override_can_apply_optional_cv_ceiling() -> None:
    """Effect-aware overrides may optionally keep a separate CV ceiling."""
    override = {
        "base_scale": 0.01,
        "confidence_effect_by_bin": {"100-249": 0.45},
        "confidence_effect_snr_by_bin": {"100-249": 0.40},
        "confidence_effect_cv_by_bin": {"100-249": 0.35},
        "training_scope": "unit_test",
    }

    _, accepted = braid._is_confident_interval(
        0.10,
        0.90,
        0.30,
        psi=0.98,
        event_type="A3SS",
        inclusion_count=120,
        exclusion_count=30,
        schedule_mode="native",
        calibration_schedule=override,
    )
    _, rejected = braid._is_confident_interval(
        0.10,
        0.90,
        0.45,
        psi=0.98,
        event_type="A3SS",
        inclusion_count=120,
        exclusion_count=30,
        schedule_mode="native",
        calibration_schedule=override,
    )

    assert accepted is True
    assert rejected is False


def test_annotation_guided_se_proposal_matches_exon_coords(
    tmp_path,
    monkeypatch,
) -> None:
    """Annotation-guided SE proposal should emit the annotated skipped exon."""
    gtf_path = tmp_path / "genes.gtf"
    gtf_path.write_text(
        "\n".join([
            'chr1\ttest\texon\t51\t100\t.\t+\t.\tgene_id "G1"; gene_name "G1"; transcript_id "TX1";',
            'chr1\ttest\texon\t201\t250\t.\t+\t.\tgene_id "G1"; gene_name "G1"; transcript_id "TX1";',
            'chr1\ttest\texon\t351\t400\t.\t+\t.\tgene_id "G1"; gene_name "G1"; transcript_id "TX1";',
        ]) + "\n",
        encoding="utf-8",
    )
    fetch_map = {
        ("chr1", 0, 500): [
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=200, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=200, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=200, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=200, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=200, cigartuples=[(0, 50), (3, 100), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 250), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 250), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 250), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 250), (0, 10)]),
            FakeRead(reference_start=50, cigartuples=[(0, 50), (3, 250), (0, 10)]),
        ],
    }
    _install_fake_pysam(monkeypatch, fetch_map)

    results = braid.compute_psi_from_junctions(
        "dummy.bam",
        "chr1",
        0,
        500,
        gene="G1",
        n_replicates=25,
        seed=13,
        annotation_gtf=str(gtf_path),
        event_source="annotation",
    )

    se_results = [result for result in results if result.event_type == "SE"]
    assert len(se_results) == 1
    assert se_results[0].event_start == 200
    assert se_results[0].event_end == 250
    assert se_results[0].proposal_source == "annotation"
    assert se_results[0].n_supported_junctions == 3


def test_parse_rmats_output_preserves_chr_prefix(tmp_path) -> None:
    """rMATS parser should preserve chromosome names as emitted by input files."""
    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    (rmats_dir / "SE.MATS.JunctionCountOnly.txt").write_text(
        "\t".join([
            "ID", "GeneID", "geneSymbol", "chr", "strand",
            "exonStart_0base", "exonEnd", "IJC_SAMPLE_1",
            "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
            "IncLevel1", "IncLevel2", "FDR", "IncLevelDifference",
        ]) + "\n" + "\t".join([
            "1", '"ENSG0001"', "GENE1", "chr1", "+", "100", "200",
            "3,2", "1,4", "7,1", "2,0", "0.40,0.60", "0.30,0.50", "0.01", "0.15",
        ]) + "\n",
        encoding="utf-8",
    )

    events = parse_rmats_output(str(rmats_dir), event_types=["SE"], min_total_count=1)

    assert len(events) == 1
    assert events[0].chrom == "chr1"
    assert events[0].gene == "GENE1"
    assert events[0].event_id == "SE:chr1:100-200"
    assert events[0].sample_1_inc_count == 5
    assert events[0].sample_1_exc_count == 5
    assert events[0].sample_2_inc_count == 8
    assert events[0].sample_2_exc_count == 2
    assert events[0].sample_1_inc_replicates == (3, 2)
    assert events[0].sample_1_exc_replicates == (1, 4)


def test_parse_rmats_output_supports_jc_fallback(tmp_path) -> None:
    """Parser should read standard rMATS JC outputs when JunctionCountOnly is absent."""
    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join([
            "ID", "GeneID", "geneSymbol", "chr", "strand",
            "exonStart_0base", "exonEnd", "IJC_SAMPLE_1",
            "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
            "IncLevel1", "IncLevel2", "FDR", "IncLevelDifference",
        ]) + "\n" + "\t".join([
            "1", "GENE1", "GENE1", "1", "+", "100", "200",
            "4,1", "2", "3", "1", "0.60,0.50", "0.20", "0.02", "0.35",
        ]) + "\n",
        encoding="utf-8",
    )

    events = parse_rmats_output(str(rmats_dir), event_types=["SE"], min_total_count=1)

    assert len(events) == 1
    assert events[0].event_id == "SE:1:100-200"
    assert events[0].sample_1_inc_count == 5
    assert events[0].sample_2_exc_count == 1
    assert events[0].sample_1_inc_replicates == (4, 1)
    assert events[0].sample_2_exc_replicates == (1,)


def test_parse_rmats_output_preserves_se_flanking_coordinates(tmp_path) -> None:
    """SE parser should retain upstream/downstream exon boundaries for recount."""
    rmats_dir = tmp_path / "rmats"
    rmats_dir.mkdir()
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join([
            "ID", "GeneID", "geneSymbol", "chr", "strand",
            "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE",
            "downstreamES", "downstreamEE", "IJC_SAMPLE_1",
            "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
            "IncLevel1", "IncLevel2", "FDR", "IncLevelDifference",
        ]) + "\n" + "\t".join([
            "1", "GENE1", "GENE1", "1", "+", "100", "150",
            "40", "80", "200", "240", "4,1", "2", "3", "1",
            "0.60,0.50", "0.20", "0.02", "0.35",
        ]) + "\n",
        encoding="utf-8",
    )

    events = parse_rmats_output(str(rmats_dir), event_types=["SE"], min_total_count=1)

    assert len(events) == 1
    assert events[0].upstream_es == 40
    assert events[0].upstream_ee == 80
    assert events[0].downstream_es == 200
    assert events[0].downstream_ee == 240


def test_extract_event_evidence_from_bam_counts_se_components(monkeypatch) -> None:
    """Direct BAM recount should recover SE left/right/skip/body evidence."""
    event = braid.build_se_splice_event(
        event_id="SE:100-150",
        gene="GENE1",
        chrom="chr1",
        exon_start=100,
        exon_end=150,
        upstream_ee=80,
        downstream_es=200,
    )
    fetch_map = {
        ("chr1", 79, 201): [
            FakeRead(reference_start=50, cigartuples=[(0, 30), (3, 20), (0, 20)]),
            FakeRead(reference_start=50, cigartuples=[(0, 30), (3, 20), (0, 20)]),
            FakeRead(reference_start=120, cigartuples=[(0, 30), (3, 50), (0, 20)]),
            FakeRead(reference_start=50, cigartuples=[(0, 30), (3, 120), (0, 20)]),
            FakeRead(reference_start=105, cigartuples=[(0, 40)]),
        ],
    }
    _install_fake_pysam(monkeypatch, fetch_map)

    evidence = braid.extract_event_evidence_from_bam(
        "dummy.bam",
        event,
        min_mapq=1,
    )

    assert evidence.inclusion_count == 1
    assert evidence.exclusion_count == 1
    assert evidence.body_count == 1
    assert evidence.evidence_breakdown["left_junction"] == 2
    assert evidence.evidence_breakdown["right_junction"] == 1
    assert evidence.evidence_breakdown["skip_junction"] == 1
