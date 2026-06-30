#!/usr/bin/env python3
"""Write a release/deposition plan for the BRAID PLOS One package."""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "submission" / "release_deposition"
PUBLIC_DATASETS = ROOT / "data" / "public_benchmarks" / "meta" / "public_validation_datasets.json"
DM1_SUMMARY = ROOT / "benchmarks" / "application_dm1" / "results" / "dm1_application_summary.json"
SUBMISSION_MANIFEST = (
    ROOT / "outputs" / "submission" / "plos_one" / "submission_package_manifest.json"
)
TABLE_MANIFEST = ROOT / "paper" / "supplementary_data" / "manuscript_tables_manifest.json"

REPRODUCTION_COMMANDS = [
    "python benchmarks/scripts/make_f1_scheme.py",
    "python benchmarks/scripts/make_f2_benchmark.py",
    "python benchmarks/headtohead/make_tool_vs_braid_figure.py",
    "python benchmarks/headtohead/make_within_study_figure.py",
    "python benchmarks/application_dm1/run_dm1_application.py",
    "python benchmarks/scripts/make_s1_detection_validation.py",
    "python benchmarks/headtohead/make_adaptive_scaling_figure.py",
    "python benchmarks/headtohead/make_sgnex_conditional_figure.py",
    "python benchmarks/headtohead/detection_filter_sweep.py",
    "python benchmarks/headtohead/make_detection_filter_figure.py",
    "python benchmarks/application_esrp/run_esrp_application.py",
    "python benchmarks/scripts/make_manuscript_tables.py",
    "python benchmarks/scripts/make_release_deposition_plan.py",
    "python benchmarks/scripts/make_plos_submission_metadata.py",
    "python benchmarks/scripts/build_submission_package.py",
    "python benchmarks/scripts/make_deposition_archive.py",
    "python benchmarks/scripts/audit_submission_readiness.py",
]
SOURCE_PATHS_FOR_PUBLIC_RELEASE = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "environment-benchmark.yml",
    "braid/",
    "benchmarks/headtohead/",
    "benchmarks/scripts/",
    "benchmarks/results/",
    "benchmarks/application_dm1/",
    "benchmarks/application_esrp/",
    "tests/",
]
LOCAL_ONLY_PATHS = [
    "benchmarks/application_dm1/raw/",
    "benchmarks/application_dm1/rmats/",
    "benchmarks/application_dm1/results/",
    "benchmarks/application_esrp/raw/",
    "benchmarks/application_esrp/rmats/",
    "benchmarks/application_esrp/results/",
    "data/public_benchmarks/**/*.bam",
    "data/public_benchmarks/**/rmats/",
    "data/public_benchmarks/**/majiq/",
    "outputs/",
    "paper/",
]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _remote_urls() -> list[str]:
    remotes = []
    for line in _git(["remote", "-v"]).splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] not in remotes:
            remotes.append(parts[1])
    return remotes


def _submission_outputs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = []
    for item in manifest.get("outputs", []):
        rel_path = item.get("path", "")
        if "/release_support/" in rel_path or "/metadata/" in rel_path:
            continue
        path = ROOT / rel_path
        outputs.append(
            {
                "path": rel_path,
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
                "exists_now": path.exists(),
            }
        )
    return outputs


def _dataset_summary(
    public_datasets: dict[str, Any],
    dm1_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for dataset in public_datasets.get("datasets", []):
        status = dataset.get("truth_tables", {}).get("status", "canonical")
        if status == "analysis_only_not_canonical":
            continue
        datasets.append(
            {
                "dataset_id": dataset.get("dataset_id"),
                "species": dataset.get("species"),
                "coordinate_build": dataset.get("coordinate_build"),
                "study_accessions": dataset.get("study_accessions", {}),
                "conditions": dataset.get("conditions", {}),
                "truth_tables": dataset.get("truth_tables", {}),
            }
        )
    if dm1_summary:
        datasets.append(
            {
                "dataset_id": dm1_summary.get("dataset"),
                "species": "human",
                "coordinate_build": "as supplied by deposited GSE201255 rMATS output",
                "study_accessions": {"geo": dm1_summary.get("source", {}).get("geo")},
                "conditions": dm1_summary.get("sample_counts", {}),
                "truth_tables": {"status": "application_dataset_not_rt_pcr_truth"},
            }
        )
    return datasets


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# BRAID Release And Deposition Plan",
        "",
        f"Generated: {payload['generated_at']}",
        f"Git commit: `{payload['git']['commit'] or 'uncommitted'}`",
        "",
        "## Current Release Blockers",
        "",
    ]
    for blocker in payload["release_blockers"]:
        lines.append(f"- {blocker}")
    lines.extend(["", "## Public Release Source Paths", ""])
    for source_path in payload["source_paths_for_public_release"]:
        lines.append(f"- `{source_path}`")
    lines.extend(["", "## Local-Only Or Regenerable Paths", ""])
    for local_path in payload["local_only_paths"]:
        lines.append(f"- `{local_path}`")
    lines.extend(["", "## Reproduction Commands", ""])
    for command in payload["reproduction_commands"]:
        lines.append(f"- `{command}`")
    lines.extend(["", "## Public Datasets", ""])
    for dataset in payload["datasets"]:
        lines.append(f"- `{dataset['dataset_id']}`: {json.dumps(dataset['study_accessions'])}")
    lines.extend(["", "## Submission Package Outputs", ""])
    for item in payload["submission_outputs"]:
        lines.append(f"- `{item['path']}` ({item['bytes']} bytes, sha256 `{item['sha256']}`)")
    lines.extend(["", "## Table Central Claims", ""])
    for key, value in payload.get("table_central_claims", {}).items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Table Auxiliary Metrics", ""])
    for key, value in payload.get("table_auxiliary_metrics", {}).items():
        lines.append(f"- `{key}`: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    public_datasets = _read_json(PUBLIC_DATASETS)
    dm1_summary = _read_json(DM1_SUMMARY)
    submission_manifest = _read_json(SUBMISSION_MANIFEST)
    table_manifest = _read_json(TABLE_MANIFEST)
    remote_urls = _remote_urls()
    private_remote = any("private" in remote.lower() for remote in remote_urls)

    payload: dict[str, Any] = {
        "plan_id": "BRAID-PLOSOne-release-deposition-plan",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "benchmarks/scripts/make_release_deposition_plan.py",
        "git": {
            "commit": _git(["rev-parse", "--short", "HEAD"]),
            "branch": _git(["branch", "--show-current"]),
            "remotes": remote_urls,
            "remote_appears_private": private_remote,
        },
        "release_blockers": [
            "Verify the public GitHub repository (https://github.com/kangk1204/BRAID) "
            "reflects the submitted release commit.",
            "Enter author-confirmed financial disclosure and competing-interest "
            "statements in the PLOS submission system.",
        ],
        "source_paths_for_public_release": SOURCE_PATHS_FOR_PUBLIC_RELEASE,
        "local_only_paths": LOCAL_ONLY_PATHS,
        "reproduction_commands": REPRODUCTION_COMMANDS,
        "datasets": _dataset_summary(public_datasets, dm1_summary),
        "submission_outputs": _submission_outputs(submission_manifest),
        "table_central_claims": table_manifest.get("central_claims", {}),
        "table_auxiliary_metrics": table_manifest.get("auxiliary_metrics", {}),
    }

    json_path = OUT_DIR / "braid_release_deposition_plan.json"
    md_path = OUT_DIR / "braid_release_deposition_plan.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(payload, md_path)
    validation_path = OUT_DIR / "release_deposition_plan_validation.txt"
    validation_lines = [
        "BRAID release/deposition plan validation",
        f"PASS\t{_rel(json_path)} exists\tsha256={_sha256(json_path)}",
        f"PASS\t{_rel(md_path)} exists\tsha256={_sha256(md_path)}",
        f"PASS\tcanonical dataset count\t{len(payload['datasets'])}",
        f"PASS\tsubmission output count\t{len(payload['submission_outputs'])}",
    ]
    validation_path.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")
    print(f"Wrote {_rel(json_path)}")
    print(f"Wrote {_rel(md_path)}")
    print(f"Wrote {_rel(validation_path)}")


if __name__ == "__main__":
    main()
