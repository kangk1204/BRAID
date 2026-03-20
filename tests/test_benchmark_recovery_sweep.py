"""Tests for synthetic recovery sweep reporting."""

from __future__ import annotations

from pathlib import Path

from benchmarks.run_recovery_sweep import _flatten_rows


def test_flatten_rows_uses_variant_builder_profile_as_authoritative_value() -> None:
    """Sweep summary should report the actual executed builder profile."""
    comparison = {
        "dataset": {"motif_compatibility": {"canonical_truth_fraction": 0.031549}},
        "variants": [
            {
                "variant": "iterative_v2_relaxed_no_motif_validation",
                "decomposer": "iterative_v2",
                "builder_profile": "aggressive_recall",
                "motif_validation": "disabled",
                "status": "ok",
                "runtime_seconds": 5.0,
                "peak_memory_mb": 256.0,
                "metrics": {
                    "transcript_sensitivity": 0.5,
                    "transcript_precision": 0.4,
                    "intron_sensitivity": 0.9,
                    "intron_precision": 1.0,
                },
                "diagnostics": {"summary": {"total_surviving_transcripts": 10, "graphs_built": 3}},
            }
        ],
    }

    rows = _flatten_rows(
        comparison,
        requested_builder_profile="default",
        min_junction_support=2,
        comparison_json=Path("/tmp/comparison.json"),
    )

    assert len(rows) == 1
    assert rows[0]["requested_builder_profile"] == "default"
    assert rows[0]["builder_profile"] == "aggressive_recall"
