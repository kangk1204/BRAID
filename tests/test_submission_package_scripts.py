"""Regression tests for manuscript submission helper scripts."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "benchmarks" / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deposition_local_only_matcher_honours_globs() -> None:
    """Local-only glob patterns must actually exclude data artifacts, not be
    silently dropped because they contain ``*`` (latent data-leak regression)."""
    script = _load_script("make_deposition_archive")
    patterns = [
        "data/public_benchmarks/**/*.bam",
        "data/public_benchmarks/**/rmats/",
        "data/public_benchmarks/**/majiq/",
        "outputs/",
    ]
    # Glob-matched data artifacts are local-only (must be excluded from the archive).
    assert script._violates_local_only("data/public_benchmarks/GSE59335/bam/x.bam", patterns)
    assert script._violates_local_only(
        "data/public_benchmarks/GSE59335/rmats/SE.MATS.JC.txt", patterns
    )
    assert script._violates_local_only("data/public_benchmarks/GSE54651/majiq/d.tsv", patterns)
    # Source code must NOT be treated as local-only.
    assert not script._violates_local_only("braid/io/bam_reader.py", patterns)
    assert not script._violates_local_only(
        "benchmarks/headtohead/head_to_head_coverage.py", patterns
    )


def test_tracked_json_artifacts_are_strict() -> None:
    """Every tracked JSON must be RFC 8259 strict (no bare NaN/Infinity), so it is
    readable by conforming parsers (jq, JS, Go), not just Python's lenient loader."""
    import subprocess

    listed = subprocess.run(
        ["git", "ls-files", "*.json"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    files = [f for f in listed.stdout.split() if f]
    assert files, "no tracked JSON files found"

    def _reject(token: str):
        raise ValueError(token)

    bad = []
    for rel in files:
        try:
            with open(ROOT / rel, encoding="utf-8") as fh:
                json.load(fh, parse_constant=_reject)
        except ValueError as exc:
            bad.append(f"{rel}: {exc}")
        except (OSError, UnicodeDecodeError):
            pass
    assert not bad, "non-strict JSON (bare NaN/Infinity): " + "; ".join(bad)


def test_manuscript_tables_preserve_tra2_mcc_contract(tmp_path, monkeypatch) -> None:
    # The manuscript-table writer round-trips through an .xlsx workbook, so the
    # test needs openpyxl. It ships in the dev extra; skip (not fail) for an
    # install that omits it rather than erroring in a github-only checkout.
    pytest.importorskip("openpyxl")
    script = _load_script("make_manuscript_tables")
    source = tmp_path / "s1_source_data.xlsx"
    panel = pd.DataFrame(
        [
            {
                "rule_id": "rmats_fdr",
                "rule_label": "rMATS FDR < 0.05",
                "n_positive": 76,
                "n_negative": 36,
                "tp": 52,
                "fn": 24,
                "fp": 8,
                "tn": 28,
                "sensitivity": 0.684211,
                "false_positive_rate": 0.222222,
                "specificity": 0.777778,
                "mcc": 0.432625,
            },
            {
                "rule_id": "braid_supported",
                "rule_label": "BRAID effect-supported",
                "n_positive": 76,
                "n_negative": 36,
                "tp": 50,
                "fn": 26,
                "fp": 2,
                "tn": 34,
                "sensitivity": 0.657895,
                "false_positive_rate": 0.055556,
                "specificity": 0.944444,
                "mcc": 0.564056,
            },
        ]
    )
    with pd.ExcelWriter(source, engine="openpyxl") as writer:
        panel.to_excel(writer, sheet_name="panel_B_mcc", index=False)

    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "S1_SOURCE", source)

    out = script.build_supp_table_3()

    assert list(out["rule_id"]) == ["rmats_fdr", "braid_supported"]
    assert out.loc[out["rule_id"] == "braid_supported", "production_braid_tier"].item()
    assert (
        out.loc[out["rule_id"] == "braid_supported", "mcc"].item()
        > out.loc[out["rule_id"] == "rmats_fdr", "mcc"].item()
    )
    assert set(out["source"]) == {"s1_source_data.xlsx"}
    assert set(out["source_sheet"]) == {"panel_B_mcc"}


def test_manuscript_table_manifest_records_mcc_limitations(monkeypatch) -> None:
    script = _load_script("make_manuscript_tables")
    missing = ROOT / "__missing_submission_test_input__"
    for name in [
        "F2_SOURCE",
        "S1_SOURCE",
        "HEADTOHEAD_JSON",
        "DM1_SUMMARY_JSON",
        "DM1_ANCHORS",
        "DM1_CANDIDATES",
        "DM1_CATEGORIES",
        "TRA2_SE",
        "TRA2_VALIDATED",
        "TRA2_FAILED",
        "TRA2_BETAS",
        "CIRC_SE",
        "CIRC_TSV",
        "CIRC_BETAS",
        "SRS_SE",
        "SRS_TRUTH",
        "SRS_BETAS",
    ]:
        monkeypatch.setattr(script, name, missing / name)

    tables = {
        "table_1_common_coverage": pd.DataFrame(
            [{"method": "BRAID conformal", "coverage_at_95": 0.9712230215827338}]
        ),
        "table_2_srs354082_full": pd.DataFrame(
            [{"method": "BRAID conformal-abs", "coverage_at_95": 0.9705882352941176}]
        ),
        "supp_table_3_tra2_detection_mcc": pd.DataFrame(
            [
                {"rule_id": "rmats_fdr", "mcc": 0.4326251176569044},
                {"rule_id": "braid_supported", "mcc": 0.5640555331476095},
            ]
        ),
    }

    manifest = script.build_manifest(tables, [], {})

    assert "tra2_rmats_mcc" not in manifest["central_claims"]
    assert manifest["auxiliary_metrics"]["tra2_rmats_mcc"] == 0.4326251176569044
    assert manifest["auxiliary_metrics"]["tra2_braid_supported_mcc"] == 0.5640555331476095
    assert any(
        "MCC is a secondary metric reported only for TRA2/GSE59335" in limitation
        for limitation in manifest["known_limitations"]
    )


def test_release_plan_excludes_its_own_release_support_outputs(tmp_path, monkeypatch) -> None:
    script = _load_script("make_release_deposition_plan")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    normal = tmp_path / "outputs" / "submission" / "plos_one" / "manuscript" / "paper.md"
    normal.parent.mkdir(parents=True)
    normal.write_text("paper\n", encoding="utf-8")
    manifest = {
        "outputs": [
            {
                "path": "outputs/submission/plos_one/release_support/plan.json",
                "bytes": 10,
                "sha256": "release-hash",
            },
            {
                "path": "outputs/submission/plos_one/metadata/plos_submission_metadata.json",
                "bytes": 20,
                "sha256": "metadata-hash",
            },
            {
                "path": "outputs/submission/plos_one/manuscript/paper.md",
                "bytes": normal.stat().st_size,
                "sha256": "paper-hash",
            },
        ]
    }

    outputs = script._submission_outputs(manifest)

    assert [item["path"] for item in outputs] == [
        "outputs/submission/plos_one/manuscript/paper.md"
    ]
    assert outputs[0]["exists_now"] is True


def test_release_plan_marks_dm1_generated_dirs_local_only() -> None:
    script = _load_script("make_release_deposition_plan")

    assert {
        "benchmarks/application_dm1/raw/",
        "benchmarks/application_dm1/rmats/",
        "benchmarks/application_dm1/results/",
    }.issubset(set(script.LOCAL_ONLY_PATHS))


def test_dm1_application_gitignore_excludes_generated_dirs() -> None:
    gitignore_text = (ROOT / "benchmarks" / "application_dm1" / ".gitignore").read_text(
        encoding="utf-8"
    )

    for generated_dir in ["raw/", "rmats/", "results/"]:
        assert generated_dir in gitignore_text


def test_audit_table_manifest_locks_mcc_auxiliary_metrics(tmp_path, monkeypatch) -> None:
    script = _load_script("audit_submission_readiness")
    manifest = tmp_path / "manuscript_tables_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "central_claims": {
                    "common_set_braid_coverage": 0.9712230215827338,
                    "srs354082_braid_abs_coverage": 0.9705882352941176,
                },
                "auxiliary_metrics": {
                    "tra2_rmats_mcc": 0.4326251176569044,
                    "tra2_braid_supported_mcc": 0.5640555331476095,
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "TABLE_MANIFEST", manifest)

    checks: list[dict[str, str]] = []
    script._audit_table_manifest(checks)

    mcc_checks = [check for check in checks if "mcc" in check["item"]]
    assert {check["status"] for check in mcc_checks} == {"PASS"}


def test_audit_release_plan_detects_package_manifest_hash_drift() -> None:
    script = _load_script("audit_submission_readiness")
    plan = {
        "submission_outputs": [
            {
                "path": "outputs/submission/plos_one/manuscript/paper.md",
                "bytes": 100,
                "sha256": "old-hash",
            },
            {
                "path": "outputs/submission/plos_one/manuscript/paper.docx",
                "bytes": 200,
                "sha256": "same-hash",
            },
        ]
    }
    package_manifest = {
        "outputs": [
            {
                "path": "outputs/submission/plos_one/manuscript/paper.md",
                "bytes": 101,
                "sha256": "new-hash",
            },
            {
                "path": "outputs/submission/plos_one/manuscript/paper.docx",
                "bytes": 200,
                "sha256": "same-hash",
            },
        ]
    }

    mismatches = script._release_plan_package_mismatches(plan, package_manifest)

    assert mismatches == [
        "outputs/submission/plos_one/manuscript/paper.md: "
        "release plan bytes/hash do not match current package manifest"
    ]


def test_audit_release_plan_accepts_matching_package_manifest() -> None:
    script = _load_script("audit_submission_readiness")
    plan = {
        "submission_outputs": [
            {
                "path": "outputs/submission/plos_one/manuscript/paper.md",
                "bytes": 100,
                "sha256": "same-hash",
            }
        ]
    }
    package_manifest = {
        "outputs": [
            {
                "path": "outputs/submission/plos_one/manuscript/paper.md",
                "bytes": 100,
                "sha256": "same-hash",
            },
            {
                "path": "outputs/submission/plos_one/release_support/plan.json",
                "bytes": 10,
                "sha256": "release-hash",
            },
        ]
    }

    assert script._release_plan_package_mismatches(plan, package_manifest) == []


def test_audit_release_plan_requires_dm1_generated_dirs_local_only(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("audit_submission_readiness")
    release_json = tmp_path / "release_plan.json"
    release_md = tmp_path / "release_plan.md"
    release_validation = tmp_path / "release_validation.txt"
    package_manifest = tmp_path / "package_manifest.json"
    release_json.write_text(
        json.dumps(
            {
                "release_blockers": ["Create public repository."],
                "datasets": [{}, {}, {}, {}],
                "local_only_paths": [
                    "benchmarks/application_dm1/raw/",
                    "benchmarks/application_dm1/rmats/",
                    "benchmarks/application_dm1/results/",
                    "benchmarks/application_esrp/raw/",
                    "benchmarks/application_esrp/rmats/",
                    "benchmarks/application_esrp/results/",
                ],
                "git": {"commit": "abc1234", "remote_appears_private": False},
                "submission_outputs": [],
            }
        ),
        encoding="utf-8",
    )
    release_md.write_text("release\n", encoding="utf-8")
    release_validation.write_text("PASS\n", encoding="utf-8")
    package_manifest.write_text(json.dumps({"outputs": []}), encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "RELEASE_PLAN_JSON", release_json)
    monkeypatch.setattr(script, "RELEASE_PLAN_MD", release_md)
    monkeypatch.setattr(script, "RELEASE_PLAN_VALIDATION", release_validation)
    monkeypatch.setattr(script, "PACKAGE_MANIFEST", package_manifest)
    monkeypatch.setattr(script, "_git", lambda _args: "abc1234")

    checks: list[dict[str, str]] = []
    script._audit_release_plan(checks)

    dm1_check = next(
        check
        for check in checks
        if check["item"] == "release/deposition plan marks DM1/Esrp generated dirs local-only"
    )
    assert dm1_check["status"] == "PASS"
    head_check = next(
        check for check in checks if check["item"] == "release/deposition plan records current HEAD"
    )
    assert head_check["status"] == "PASS"


def test_audit_release_plan_blocks_when_esrp_dirs_not_local_only(tmp_path, monkeypatch) -> None:
    """If the release plan marks only DM1 (not Esrp) generated dirs local-only, the
    audit must block, so Esrp raw/rMATS/results cannot silently leak into the public
    archive."""
    script = _load_script("audit_submission_readiness")
    release_json = tmp_path / "release_plan.json"
    release_md = tmp_path / "release_plan.md"
    release_validation = tmp_path / "release_validation.txt"
    package_manifest = tmp_path / "package_manifest.json"
    release_json.write_text(
        json.dumps(
            {
                "release_blockers": ["Create public repository."],
                "datasets": [{}, {}, {}, {}],
                "local_only_paths": [
                    "benchmarks/application_dm1/raw/",
                    "benchmarks/application_dm1/rmats/",
                    "benchmarks/application_dm1/results/",
                ],
                "git": {"commit": "abc1234", "remote_appears_private": False},
                "submission_outputs": [],
            }
        ),
        encoding="utf-8",
    )
    release_md.write_text("release\n", encoding="utf-8")
    release_validation.write_text("PASS\n", encoding="utf-8")
    package_manifest.write_text(json.dumps({"outputs": []}), encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "RELEASE_PLAN_JSON", release_json)
    monkeypatch.setattr(script, "RELEASE_PLAN_MD", release_md)
    monkeypatch.setattr(script, "RELEASE_PLAN_VALIDATION", release_validation)
    monkeypatch.setattr(script, "PACKAGE_MANIFEST", package_manifest)
    monkeypatch.setattr(script, "_git", lambda _args: "abc1234")

    checks: list[dict[str, str]] = []
    script._audit_release_plan(checks)

    dm1_check = next(
        check
        for check in checks
        if check["item"] == "release/deposition plan marks DM1/Esrp generated dirs local-only"
    )
    assert dm1_check["status"] == "BLOCKER"


def test_audit_release_plan_blocks_stale_head_commit(tmp_path, monkeypatch) -> None:
    script = _load_script("audit_submission_readiness")
    release_json = tmp_path / "release_plan.json"
    release_md = tmp_path / "release_plan.md"
    release_validation = tmp_path / "release_validation.txt"
    package_manifest = tmp_path / "package_manifest.json"
    release_json.write_text(
        json.dumps(
            {
                "release_blockers": ["Create public repository."],
                "datasets": [{}, {}, {}, {}],
                "local_only_paths": [
                    "benchmarks/application_dm1/raw/",
                    "benchmarks/application_dm1/rmats/",
                    "benchmarks/application_dm1/results/",
                ],
                "git": {"commit": "old1234", "remote_appears_private": False},
                "submission_outputs": [],
            }
        ),
        encoding="utf-8",
    )
    release_md.write_text("release\n", encoding="utf-8")
    release_validation.write_text("PASS\n", encoding="utf-8")
    package_manifest.write_text(json.dumps({"outputs": []}), encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "RELEASE_PLAN_JSON", release_json)
    monkeypatch.setattr(script, "RELEASE_PLAN_MD", release_md)
    monkeypatch.setattr(script, "RELEASE_PLAN_VALIDATION", release_validation)
    monkeypatch.setattr(script, "PACKAGE_MANIFEST", package_manifest)
    monkeypatch.setattr(script, "_git", lambda _args: "new1234")

    checks: list[dict[str, str]] = []
    script._audit_release_plan(checks)

    head_check = next(
        check for check in checks if check["item"] == "release/deposition plan records current HEAD"
    )
    assert head_check["status"] == "BLOCKER"


def test_deposition_archive_filters_local_only_source_paths(monkeypatch) -> None:
    script = _load_script("make_deposition_archive")
    monkeypatch.setattr(
        script,
        "_git_ls_files",
        lambda _paths: [
            "README.md",
            "benchmarks/application_dm1/run_dm1_application.py",
            "benchmarks/application_dm1/raw/GSE201255_caseVScontrol_SE.MATS.JC.txt.gz",
            "benchmarks/application_dm1/rmats/SE.MATS.JC.txt",
            "benchmarks/application_dm1/results/dm1_application_summary.json",
        ],
    )
    release_plan = {
        "source_paths_for_public_release": ["README.md", "benchmarks/application_dm1/"],
        "local_only_paths": [
            "benchmarks/application_dm1/raw/",
            "benchmarks/application_dm1/rmats/",
            "benchmarks/application_dm1/results/",
        ],
    }

    files = script._source_files(release_plan)

    assert files == [
        "README.md",
        "benchmarks/application_dm1/run_dm1_application.py",
    ]


def test_deposition_archive_detects_dirty_release_source(monkeypatch) -> None:
    script = _load_script("make_deposition_archive")
    monkeypatch.setattr(
        script,
        "_git_status_porcelain",
        lambda _paths: [
            " M README.md",
            "?? benchmarks/scripts/new_release_helper.py",
            "?? benchmarks/application_dm1/results/dm1_application_summary.json",
        ],
    )
    release_plan = {
        "source_paths_for_public_release": ["README.md", "benchmarks/scripts/"],
        "local_only_paths": ["benchmarks/application_dm1/results/"],
    }

    dirty = script._dirty_release_source_status(release_plan)

    assert dirty == [" M README.md", "?? benchmarks/scripts/new_release_helper.py"]


def test_deposition_archive_refuses_dirty_release_source(tmp_path, monkeypatch) -> None:
    script = _load_script("make_deposition_archive")
    release_plan = tmp_path / "release_plan.json"
    submission_manifest = tmp_path / "submission_manifest.json"
    release_plan.write_text(
        json.dumps({"source_paths_for_public_release": ["README.md"]}),
        encoding="utf-8",
    )
    submission_manifest.write_text(json.dumps({"outputs": []}), encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "OUT_DIR", tmp_path)
    monkeypatch.setattr(script, "RELEASE_PLAN", release_plan)
    monkeypatch.setattr(script, "SUBMISSION_MANIFEST", submission_manifest)
    monkeypatch.setattr(script, "_git_status_porcelain", lambda _paths: [" M README.md"])

    with pytest.raises(RuntimeError, match="Release source changes must be committed"):
        script.build_deposition_archive()


def test_deposition_archive_refuses_stale_release_plan_commit(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("make_deposition_archive")
    release_plan = tmp_path / "release_plan.json"
    submission_manifest = tmp_path / "submission_manifest.json"
    release_plan.write_text(
        json.dumps(
            {
                "git": {"commit": "old1234"},
                "source_paths_for_public_release": ["README.md"],
            }
        ),
        encoding="utf-8",
    )
    submission_manifest.write_text(json.dumps({"outputs": []}), encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "OUT_DIR", tmp_path)
    monkeypatch.setattr(script, "RELEASE_PLAN", release_plan)
    monkeypatch.setattr(script, "SUBMISSION_MANIFEST", submission_manifest)
    monkeypatch.setattr(script, "_git_status_porcelain", lambda _paths: [])
    monkeypatch.setattr(script, "_git", lambda _args: "new1234")

    with pytest.raises(RuntimeError, match="Release plan git commit must match HEAD"):
        script.build_deposition_archive()


def test_deposition_archive_zip_is_deterministic(tmp_path, monkeypatch) -> None:
    script = _load_script("make_deposition_archive")
    source = tmp_path / "README.md"
    source.write_text("stable content\n", encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    manifest = {
        "files": [
            {
                "path": "README.md",
                "archive_path": "BRAID_PLOSOne_deposition_package/README.md",
            }
        ]
    }
    archive_a = tmp_path / "a.zip"
    archive_b = tmp_path / "b.zip"

    manifest_a = script._write_zip(archive_a, manifest)
    manifest_b = script._write_zip(archive_b, manifest)

    assert manifest_a == manifest_b
    assert archive_a.read_bytes() == archive_b.read_bytes()


def test_audit_deposition_archive_locks_checksum_and_exclusions(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("audit_submission_readiness")
    archive = tmp_path / "archive.zip"
    manifest = tmp_path / "archive_manifest.json"
    validation = tmp_path / "archive_validation.txt"
    archive.write_bytes(b"archive bytes\n")
    archive_sha = script._sha256(archive)
    manifest.write_text(
        json.dumps(
            {
                "archive": {
                    "path": "outputs/submission/release_deposition/archive.zip",
                    "bytes": archive.stat().st_size,
                    "sha256": archive_sha,
                },
                "git": {"commit": "abc1234"},
                "source_file_count": 2,
                "submission_file_count": 3,
                "files": [
                    {"path": "README.md"},
                    {"path": "outputs/submission/plos_one/manuscript/paper.md"},
                ],
            }
        ),
        encoding="utf-8",
    )
    validation.write_text("PASS\tarchive exists\n", encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "DEPOSITION_ARCHIVE", archive)
    monkeypatch.setattr(script, "DEPOSITION_ARCHIVE_MANIFEST", manifest)
    monkeypatch.setattr(script, "DEPOSITION_ARCHIVE_VALIDATION", validation)
    # The archive's recorded commit must match HEAD; pin HEAD to the manifest's.
    monkeypatch.setattr(script, "_git", lambda args: "abc1234")

    checks: list[dict[str, str]] = []
    script._audit_deposition_archive(checks)

    assert {
        check["status"]
        for check in checks
        if check["item"].startswith("deposition archive")
    } == {"PASS"}
    exclusion_check = next(
        check
        for check in checks
        if check["item"] == "deposition archive excludes local-only benchmark artifacts"
    )
    assert exclusion_check["status"] == "PASS"
    head_check = next(
        check
        for check in checks
        if check["item"] == "deposition archive manifest records current HEAD"
    )
    assert head_check["status"] == "PASS"


def test_audit_deposition_archive_blocks_stale_commit(tmp_path, monkeypatch) -> None:
    """A deposition archive whose manifest records a commit other than HEAD must
    be flagged BLOCKER, so a stale archive cannot pass readiness (HIGH finding)."""
    script = _load_script("audit_submission_readiness")
    archive = tmp_path / "archive.zip"
    manifest = tmp_path / "archive_manifest.json"
    validation = tmp_path / "archive_validation.txt"
    archive.write_bytes(b"archive bytes\n")
    manifest.write_text(
        json.dumps(
            {
                "archive": {"sha256": script._sha256(archive)},
                "git": {"commit": "0ldc0de"},
                "source_file_count": 2,
                "submission_file_count": 3,
                "files": [{"path": "README.md"}],
            }
        ),
        encoding="utf-8",
    )
    validation.write_text("PASS\tarchive exists\n", encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "DEPOSITION_ARCHIVE", archive)
    monkeypatch.setattr(script, "DEPOSITION_ARCHIVE_MANIFEST", manifest)
    monkeypatch.setattr(script, "DEPOSITION_ARCHIVE_VALIDATION", validation)
    monkeypatch.setattr(script, "_git", lambda args: "newhead")

    checks: list[dict[str, str]] = []
    script._audit_deposition_archive(checks)

    head_check = next(
        check
        for check in checks
        if check["item"] == "deposition archive manifest records current HEAD"
    )
    assert head_check["status"] == "BLOCKER"


def test_audit_manuscript_flags_wrapped_repository_url_placeholder(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("audit_submission_readiness")
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "## Code availability\n"
        "The repository URL\n"
        "should be inserted before submission.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "MANUSCRIPT", manuscript)

    checks: list[dict[str, str]] = []
    script._audit_manuscript(checks, manuscript.read_text(encoding="utf-8"))

    repo_check = next(
        check for check in checks if check["item"] == "public repository URL finalized"
    )
    assert repo_check["status"] == "BLOCKER"


def test_audit_manuscript_requires_submission_facing_figure_legends(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("audit_submission_readiness")
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "## Figure plan\n"
        "Figure 1. BRAID workflow and operating principle.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "MANUSCRIPT", manuscript)
    monkeypatch.setattr(script, "REQUIRED_FIGURE_LEGENDS", [])
    monkeypatch.setattr(script, "CLAIM_STRINGS", [])

    checks: list[dict[str, str]] = []
    script._audit_manuscript(checks, manuscript.read_text(encoding="utf-8"))

    stale_check = next(
        check for check in checks if check["item"] == "stale token absent: Figure plan"
    )
    legends_check = next(
        check for check in checks if check["item"] == "figure legend section is submission-facing"
    )
    assert stale_check["status"] == "BLOCKER"
    assert legends_check["status"] == "BLOCKER"


def test_audit_manuscript_enforces_plos_abstract_word_limit(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("audit_submission_readiness")
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "## Abstract\n\n"
        + " ".join(["word"] * 301)
        + "\n\nKeywords: alternative splicing; RNA-seq\n\n"
        "## Code availability\n\n"
        "The repository URL should be inserted before submission.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "MANUSCRIPT", manuscript)
    monkeypatch.setattr(script, "STALE_TOKENS", [])
    monkeypatch.setattr(script, "REQUIRED_FIGURE_LEGENDS", [])
    monkeypatch.setattr(script, "CLAIM_STRINGS", [])

    checks: list[dict[str, str]] = []
    script._audit_manuscript(checks, manuscript.read_text(encoding="utf-8"))

    abstract_check = next(
        check for check in checks if check["item"] == "abstract length within PLOS limit"
    )
    assert abstract_check["status"] == "BLOCKER"
    assert "301 words" in abstract_check["detail"]


def test_audit_manuscript_requires_s4_detection_counts(tmp_path, monkeypatch) -> None:
    script = _load_script("audit_submission_readiness")
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "## Figure legends\n"
        "Supplementary Figure 4. Auxiliary TRA2 positive/negative detection check.\n"
        "MCC = 0.541.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "MANUSCRIPT", manuscript)
    monkeypatch.setattr(script, "STALE_TOKENS", [])
    monkeypatch.setattr(script, "REQUIRED_FIGURE_LEGENDS", [])
    monkeypatch.setattr(
        script,
        "CLAIM_STRINGS",
        [
            ("TRA2 matched positive targets", "76 RT-PCR-positive"),
            ("TRA2 matched negative targets", "36 RT-PCR-negative"),
            ("TRA2 BRAID effect-supported MCC", "0.541"),
        ],
    )

    checks: list[dict[str, str]] = []
    script._audit_manuscript(checks, manuscript.read_text(encoding="utf-8"))

    required = {
        check["item"]: check["status"]
        for check in checks
        if check["item"].startswith("secondary benchmark string present")
    }
    assert (
        required["secondary benchmark string present: TRA2 matched positive targets"]
        == "BLOCKER"
    )
    assert (
        required["secondary benchmark string present: TRA2 matched negative targets"]
        == "BLOCKER"
    )
    assert (
        required["secondary benchmark string present: TRA2 BRAID effect-supported MCC"] == "PASS"
    )


def test_audit_claim_strings_use_current_tra2_operating_point_label() -> None:
    script = _load_script("audit_submission_readiness")

    labels = [
        label for label, _ in script.CLAIM_STRINGS if label.startswith("TRA2 BRAID ")
    ]

    assert labels == [
        "TRA2 BRAID effect-supported true positives",
        "TRA2 BRAID effect-supported false positives",
        "TRA2 BRAID effect-supported MCC",
    ]


def test_audit_manuscript_requires_plos_software_methods_contract(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("audit_submission_readiness")
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "## Methods\n"
        "### Software implementation, dependencies, and test data\n"
        "BRAID is distributed under the MIT License. "
        "The smoke test is braid example.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "MANUSCRIPT", manuscript)
    monkeypatch.setattr(script, "STALE_TOKENS", [])
    monkeypatch.setattr(script, "REQUIRED_FIGURE_LEGENDS", [])
    monkeypatch.setattr(
        script,
        "CLAIM_STRINGS",
        [
            ("software methods section", "Software implementation, dependencies, and test data"),
            ("software license", "MIT License"),
            ("python version requirement", "Python >=3.10"),
            ("core install command", "python -m pip install -e ."),
            ("supplied smoke test", "braid example"),
        ],
    )

    checks: list[dict[str, str]] = []
    script._audit_manuscript(checks, manuscript.read_text(encoding="utf-8"))

    required = {
        check["item"]: check["status"]
        for check in checks
        if check["item"].startswith("central claim string present")
    }
    assert required["central claim string present: software methods section"] == "PASS"
    assert required["central claim string present: software license"] == "PASS"
    assert required["central claim string present: supplied smoke test"] == "PASS"
    assert required["central claim string present: python version requirement"] == "BLOCKER"
    assert required["central claim string present: core install command"] == "BLOCKER"


def test_audit_manuscript_requires_public_data_ethics_statement(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("audit_submission_readiness")
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "## Methods\n"
        "### Ethics statement\n"
        "The current study is a secondary computational analysis. "
        "No new human participants were recruited.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "MANUSCRIPT", manuscript)
    monkeypatch.setattr(script, "STALE_TOKENS", [])
    monkeypatch.setattr(script, "REQUIRED_FIGURE_LEGENDS", [])
    monkeypatch.setattr(
        script,
        "CLAIM_STRINGS",
        [
            ("ethics statement section", "Ethics statement"),
            ("ethics public data reuse", "secondary computational analysis"),
            ("ethics no new participants", "No new human participants were recruited"),
            ("ethics no new specimens", "no new biospecimens"),
            ("ethics no identifying data", "no protected health information"),
            ("ethics no additional consent", "no additional informed consent"),
        ],
    )

    checks: list[dict[str, str]] = []
    script._audit_manuscript(checks, manuscript.read_text(encoding="utf-8"))

    required = {
        check["item"]: check["status"]
        for check in checks
        if check["item"].startswith("central claim string present")
    }
    assert required["central claim string present: ethics statement section"] == "PASS"
    assert required["central claim string present: ethics public data reuse"] == "PASS"
    assert required["central claim string present: ethics no new participants"] == "PASS"
    assert required["central claim string present: ethics no new specimens"] == "BLOCKER"
    assert required["central claim string present: ethics no identifying data"] == "BLOCKER"
    assert required["central claim string present: ethics no additional consent"] == "BLOCKER"


def test_audit_manuscript_checks_author_year_reference_resolution(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("audit_submission_readiness")
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "# BRAID title\n\n"
        "Short title: BRAID intervals\n\n"
        "## Abstract\n\n"
        "Abstract text.\n\n"
        "## Introduction\n\n"
        "Supported claim (Best et al., 2014; Missing et al., 2022).\n\n"
        "## References\n\n"
        "Best A. Supported paper. Journal. 2014.\n\n"
        "Uncited U. Extra paper. Journal. 2020.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "MANUSCRIPT", manuscript)
    monkeypatch.setattr(script, "STALE_TOKENS", [])
    monkeypatch.setattr(script, "REQUIRED_FIGURE_LEGENDS", [])
    monkeypatch.setattr(script, "CLAIM_STRINGS", [])

    checks: list[dict[str, str]] = []
    script._audit_manuscript(checks, manuscript.read_text(encoding="utf-8"))

    statuses = {check["item"]: check for check in checks}
    assert statuses["author-year citations resolve to references"]["status"] == "BLOCKER"
    assert "Missing 2022" in statuses["author-year citations resolve to references"]["detail"]
    assert statuses["reference entries are cited in text"]["status"] == "BLOCKER"
    assert "Uncited 2020" in statuses["reference entries are cited in text"]["detail"]


def test_audit_manuscript_requires_submission_facing_data_availability(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("audit_submission_readiness")
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "# BRAID title\n\n"
        "Short title: BRAID intervals\n\n"
        "## Abstract\n\n"
        "Abstract text.\n\n"
        "## Data availability\n\n"
        "The minimal data set is provided as figure source-data workbooks. "
        "The final public deposition identifier will be inserted before publication.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "MANUSCRIPT", manuscript)
    monkeypatch.setattr(script, "STALE_TOKENS", [])
    monkeypatch.setattr(script, "REQUIRED_FIGURE_LEGENDS", [])
    monkeypatch.setattr(script, "CLAIM_STRINGS", [])

    checks: list[dict[str, str]] = []
    script._audit_manuscript(checks, manuscript.read_text(encoding="utf-8"))

    statuses = {check["item"]: check["status"] for check in checks}
    assert statuses["data availability states minimal data set"] == "PASS"
    assert statuses["data availability states figure source-data workbooks"] == "PASS"
    assert statuses["data availability deposition finalization state"] == "INFO"
    assert statuses["large benchmark artifacts publicly resolved"] == "PASS"


def test_audit_manuscript_accepts_github_only_archive(
    tmp_path,
    monkeypatch,
) -> None:
    """GitHub-only policy: a public repository URL satisfies code/data availability
    without an external archival DOI, and a repo-pointing data-availability statement
    is a finalized archival state (not a pending deposition)."""
    script = _load_script("audit_submission_readiness")
    manuscript = tmp_path / "manuscript.md"
    manuscript.write_text(
        "# BRAID title\n\n"
        "Short title: BRAID intervals\n\n"
        "## Abstract\n\n"
        "Abstract text.\n\n"
        "## Data availability\n\n"
        "The minimal data set is provided as figure source-data workbooks. All source "
        "data are openly mirrored in the public Git repository at "
        "https://github.com/kangk1204/BRAID.\n\n"
        "## Code availability\n\n"
        "BRAID is openly available under the MIT License in the public Git repository "
        "at https://github.com/kangk1204/BRAID.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "MANUSCRIPT", manuscript)
    monkeypatch.setattr(script, "STALE_TOKENS", [])
    monkeypatch.setattr(script, "REQUIRED_FIGURE_LEGENDS", [])
    monkeypatch.setattr(script, "CLAIM_STRINGS", [])

    checks: list[dict[str, str]] = []
    script._audit_manuscript(checks, manuscript.read_text(encoding="utf-8"))

    statuses = {check["item"]: check["status"] for check in checks}
    assert statuses["code archive is publicly resolvable"] == "PASS"
    assert statuses["data availability deposition finalization state"] == "PASS"


def test_plos_submission_metadata_records_finalization_fields(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("make_plos_submission_metadata")
    manuscript = tmp_path / "paper" / "manuscript.md"
    manuscript.parent.mkdir(parents=True)
    manuscript.write_text(
        "# BRAID title\n\n"
        "Short title: BRAID calibrated intervals\n\n"
        "Keunsoo Kang<sup>1,*</sup>\n\n"
        "<sup>1</sup>Department, University\n\n"
        "<sup>*</sup>Correspondence: kang@example.edu\n\n"
        "## Abstract\n\n"
        "Abstract text.\n\n"
        "Keywords: alternative splicing; RNA-seq; uncertainty calibration\n\n"
        "## Methods\n\n"
        "### Ethics statement\n\n"
        "Secondary public-data analysis.\n\n"
        "### Software implementation, dependencies, and test data\n\n"
        "For long-term utility and maintenance, BRAID is kept as a small command-line "
        "post-processing layer. The release archive and public repository are stable "
        "hosting surfaces.\n\n"
        "## Data availability\n\n"
        "Public accession data are used.\n\n"
        "## Code availability\n\n"
        "The exact public repository URL should be inserted at submission time.\n\n"
        "## Competing interests\n\n"
        "The author declares no competing interests.\n",
        encoding="utf-8",
    )
    release_plan = tmp_path / "outputs" / "release_plan.json"
    release_plan.parent.mkdir(parents=True)
    release_plan.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-19T11:00:00+00:00",
                "git": {
                    "commit": "abc1234",
                    "branch": "release-test",
                    "remote_appears_private": True,
                },
                "release_blockers": ["publish public remote"],
            }
        ),
        encoding="utf-8",
    )
    package_manifest = tmp_path / "outputs" / "package_manifest.json"
    package_manifest.write_text(json.dumps({"outputs": [{"path": "paper.md"}]}), encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "MANUSCRIPT", manuscript)
    monkeypatch.setattr(script, "RELEASE_PLAN", release_plan)

    metadata = script.build_metadata()

    assert metadata["target_journal"] == "PLOS ONE"
    assert metadata["title"] == "BRAID title"
    assert metadata["short_title"] == "BRAID calibrated intervals"
    assert metadata["title_lengths"]["full_title_limit"] == 250
    assert metadata["title_lengths"]["short_title_limit"] == 100
    assert metadata["abstract"] == "Abstract text."
    assert metadata["keywords"] == [
        "alternative splicing",
        "RNA-seq",
        "uncertainty calibration",
    ]
    assert metadata["corresponding_author"]["email"] == "kang@example.edu"
    assert metadata["corresponding_author"]["name"] == "Keunsoo Kang"
    assert metadata["corresponding_author"]["affiliation"] == "Department, University"
    assert metadata["funding_current"].startswith("Not stored in manuscript")
    assert metadata["competing_interests_current"] == "The author declares no competing interests."
    assert metadata["release_context"]["release_plan_generated_at"] == "2026-06-19T11:00:00+00:00"
    assert metadata["release_context"]["git"]["commit"] == "abc1234"
    assert metadata["release_context"]["release_blockers"] == ["publish public remote"]
    fields = {item["field"]: item["status"] for item in metadata["finalization_fields"]}
    assert fields["public_repository_url"] == "needs_author_action"
    assert fields["archive_doi_or_private_reviewer_url"] == "needs_author_action"
    assert fields["funding_statement"] == "needs_author_confirmation"
    assert fields["competing_interests"] == "needs_author_confirmation"
    assert fields["public_release_remote"] == "needs_author_action"
    scope = {item["criterion"]: item["status"] for item in metadata["scope_compliance_matrix"]}
    assert scope["utility"] == "addressed_in_manuscript"
    assert scope["validation"] == "addressed_in_manuscript"
    assert scope["availability"] == "needs_external_finalization"
    assert scope["long_term_utility_and_maintenance"] == "addressed_in_manuscript"


def test_audit_requires_plos_submission_metadata_fields(tmp_path, monkeypatch) -> None:
    script = _load_script("audit_submission_readiness")
    metadata = tmp_path / "plos_submission_metadata.json"
    release_plan = tmp_path / "release_plan.json"
    release_plan.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-19T11:00:00+00:00",
                "git": {"commit": "abc1234"},
            }
        ),
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-19T12:00:00+00:00",
                "release_context": {"git": {"commit": "abc1234"}},
                "target_journal": "PLOS ONE",
                "short_title": "BRAID intervals",
                "title": "BRAID title",
                "scope_compliance_matrix": [
                    {"criterion": "utility", "status": "addressed_in_manuscript"},
                    {"criterion": "validation", "status": "addressed_in_manuscript"},
                    {"criterion": "availability", "status": "needs_external_finalization"},
                    {"criterion": "algorithm_details", "status": "addressed_in_manuscript"},
                    {
                        "criterion": "open_source_and_dependencies",
                        "status": "addressed_in_manuscript",
                    },
                    {
                        "criterion": "test_data_install_and_run",
                        "status": "addressed_in_manuscript",
                    },
                    {
                        "criterion": "long_term_utility_and_maintenance",
                        "status": "addressed_in_manuscript",
                    },
                ],
                "finalization_fields": [
                    {"field": "public_repository_url", "status": "needs_author_action"},
                    {
                        "field": "archive_doi_or_private_reviewer_url",
                        "status": "needs_author_action",
                    },
                    {"field": "funding_statement", "status": "needs_author_confirmation"},
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata_md = tmp_path / "plos_submission_metadata.md"
    metadata_md.write_text("# Metadata\n", encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "SUBMISSION_METADATA_JSON", metadata)
    monkeypatch.setattr(script, "SUBMISSION_METADATA_MD", metadata_md)
    monkeypatch.setattr(script, "RELEASE_PLAN_JSON", release_plan)

    checks: list[dict[str, str]] = []
    script._audit_submission_metadata(checks)

    statuses = {check["item"]: check["status"] for check in checks}
    assert statuses["PLOS metadata target journal recorded"] == "PASS"
    assert statuses["PLOS metadata is current with release/deposition plan"] == "PASS"
    assert statuses["PLOS metadata records current release commit"] == "PASS"
    assert (
        statuses["PLOS metadata finalization field recorded: public_repository_url"] == "PASS"
    )
    assert statuses["PLOS metadata finalization field recorded: competing_interests"] == "BLOCKER"


def test_audit_requires_public_release_remote_for_private_release_plan(
    tmp_path,
    monkeypatch,
) -> None:
    script = _load_script("audit_submission_readiness")
    metadata = tmp_path / "plos_submission_metadata.json"
    release_plan = tmp_path / "release_plan.json"
    release_plan.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-19T11:00:00+00:00",
                "git": {"commit": "abc1234", "remote_appears_private": True},
            }
        ),
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-19T12:00:00+00:00",
                "release_context": {"git": {"commit": "abc1234"}},
                "target_journal": "PLOS ONE",
                "short_title": "BRAID intervals",
                "title": "BRAID title",
                "scope_compliance_matrix": [],
                "finalization_fields": [
                    {"field": "public_repository_url", "status": "needs_author_action"},
                    {
                        "field": "archive_doi_or_private_reviewer_url",
                        "status": "needs_author_action",
                    },
                    {"field": "funding_statement", "status": "needs_author_confirmation"},
                    {"field": "competing_interests", "status": "needs_author_confirmation"},
                ],
            }
        ),
        encoding="utf-8",
    )
    metadata_md = tmp_path / "plos_submission_metadata.md"
    metadata_md.write_text("# Metadata\n", encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "SUBMISSION_METADATA_JSON", metadata)
    monkeypatch.setattr(script, "SUBMISSION_METADATA_MD", metadata_md)
    monkeypatch.setattr(script, "RELEASE_PLAN_JSON", release_plan)

    checks: list[dict[str, str]] = []
    script._audit_submission_metadata(checks)

    statuses = {check["item"]: check["status"] for check in checks}
    assert statuses["PLOS metadata finalization field recorded: public_repository_url"] == "PASS"
    assert statuses["PLOS metadata finalization field recorded: competing_interests"] == "PASS"
    assert statuses["PLOS metadata finalization field recorded: public_release_remote"] == "BLOCKER"


def test_audit_blocks_stale_plos_submission_metadata(tmp_path, monkeypatch) -> None:
    script = _load_script("audit_submission_readiness")
    metadata = tmp_path / "plos_submission_metadata.json"
    release_plan = tmp_path / "release_plan.json"
    release_plan.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-19T12:00:00+00:00",
                "git": {"commit": "new1234"},
            }
        ),
        encoding="utf-8",
    )
    metadata.write_text(
        json.dumps(
            {
                "generated_at": "2026-06-19T11:00:00+00:00",
                "release_context": {"git": {"commit": "old1234"}},
                "target_journal": "PLOS ONE",
                "short_title": "BRAID intervals",
                "title": "BRAID title",
                "scope_compliance_matrix": [],
                "finalization_fields": [],
            }
        ),
        encoding="utf-8",
    )
    metadata_md = tmp_path / "plos_submission_metadata.md"
    metadata_md.write_text("# Metadata\n", encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "SUBMISSION_METADATA_JSON", metadata)
    monkeypatch.setattr(script, "SUBMISSION_METADATA_MD", metadata_md)
    monkeypatch.setattr(script, "RELEASE_PLAN_JSON", release_plan)

    checks: list[dict[str, str]] = []
    script._audit_submission_metadata(checks)

    statuses = {check["item"]: check["status"] for check in checks}
    assert statuses["PLOS metadata is current with release/deposition plan"] == "BLOCKER"
    assert statuses["PLOS metadata records current release commit"] == "BLOCKER"


def test_audit_blocks_tracked_source_changes_before_release_commit(tmp_path, monkeypatch) -> None:
    script = _load_script("audit_submission_readiness")
    readme = tmp_path / "README.md"
    gitignore = tmp_path / ".gitignore"
    readme.write_text("# BRAID\n", encoding="utf-8")
    gitignore.write_text("outputs/tables/\noutputs/submission/\npaper/\n", encoding="utf-8")
    monkeypatch.setattr(script, "ROOT", tmp_path)
    monkeypatch.setattr(script, "README", readme)
    monkeypatch.setattr(script, "GITIGNORE", gitignore)
    monkeypatch.setattr(
        script,
        "_git_status",
        lambda: [" M benchmarks/scripts/audit_submission_readiness.py", "?? scratch.txt"],
    )

    checks: list[dict[str, str]] = []
    status_lines = script._audit_repo_surface(checks)

    statuses = {check["item"]: check["status"] for check in checks}
    assert status_lines == [" M benchmarks/scripts/audit_submission_readiness.py", "?? scratch.txt"]
    assert statuses["tracked source changes"] == "BLOCKER"
    assert statuses["untracked files before release commit"] == "WARN"


def test_missing_release_support_is_a_validation_check_not_copy_failure(tmp_path) -> None:
    script = _load_script("build_submission_package")
    checks: list[tuple[str, bool]] = []

    script._require_file(tmp_path / "missing-plan.json", "release support", checks)

    assert checks == [
        ("release support exists", False),
        ("release support non-empty", False),
    ]


def test_build_docx_reuses_newer_existing_docx(tmp_path) -> None:
    script = _load_script("build_submission_package")
    source = tmp_path / "manuscript.md"
    docx = tmp_path / "manuscript.docx"
    source.write_text("# Manuscript\n", encoding="utf-8")
    docx.write_bytes(b"stable-docx")
    os.utime(source, (1_700_000_000, 1_700_000_000))
    os.utime(docx, (1_700_000_100, 1_700_000_100))

    result = script._build_docx(source, docx)

    assert result.returncode == 0
    assert "skipped" in result.stdout
    assert docx.read_bytes() == b"stable-docx"
