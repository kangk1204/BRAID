#!/usr/bin/env python3
"""Build a deterministic local deposition archive for the BRAID PLOS package."""
from __future__ import annotations

import fnmatch
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "submission" / "release_deposition"
RELEASE_PLAN = OUT_DIR / "braid_release_deposition_plan.json"
SUBMISSION_MANIFEST = (
    ROOT / "outputs" / "submission" / "plos_one" / "submission_package_manifest.json"
)
ARCHIVE = OUT_DIR / "BRAID_PLOSOne_deposition_package.zip"
ARCHIVE_MANIFEST = OUT_DIR / "BRAID_PLOSOne_deposition_package_manifest.json"
ARCHIVE_VALIDATION = OUT_DIR / "BRAID_PLOSOne_deposition_package_validation.txt"
ARCHIVE_ROOT = "BRAID_PLOSOne_deposition_package"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _git_ls_files(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git ls-files failed")
    return sorted(line for line in result.stdout.splitlines() if line)


def _git_status_porcelain(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git status failed")
    return sorted(line for line in result.stdout.splitlines() if line)


def _status_path(line: str) -> str:
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip('"')


def _is_within(path: str, prefix: str) -> bool:
    return path == prefix.rstrip("/") or path.startswith(prefix.rstrip("/") + "/")


def _matches_local_only(path: str, pattern: str) -> bool:
    """Match a path against a local-only entry, honouring glob patterns.

    Literal entries (no ``*``) match as directory/file prefixes; glob entries
    (e.g. ``data/public_benchmarks/**/*.bam`` or ``**/rmats/``) match via
    ``fnmatch`` so large data artifacts are actually excluded rather than
    silently dropped because they contained a ``*``.
    """
    if "*" in pattern:
        pat = pattern.rstrip("/")
        return fnmatch.fnmatch(path, pat) or fnmatch.fnmatch(path, pat + "/*")
    return _is_within(path, pattern)


def _violates_local_only(path: str, local_only_paths: list[str]) -> bool:
    return any(_matches_local_only(path, pattern) for pattern in local_only_paths)


def _source_files(release_plan: dict[str, Any]) -> list[str]:
    source_paths = list(release_plan.get("source_paths_for_public_release", []))
    local_only_paths = list(release_plan.get("local_only_paths", []))
    files = _git_ls_files(source_paths)
    return [path for path in files if not _violates_local_only(path, local_only_paths)]


def _dirty_release_source_status(release_plan: dict[str, Any]) -> list[str]:
    source_paths = list(release_plan.get("source_paths_for_public_release", []))
    local_only_paths = list(release_plan.get("local_only_paths", []))
    status_lines = _git_status_porcelain(source_paths)
    return [
        line
        for line in status_lines
        if not _violates_local_only(_status_path(line), local_only_paths)
    ]


def _ensure_release_plan_matches_head(release_plan: dict[str, Any]) -> None:
    plan_commit = str(release_plan.get("git", {}).get("commit", ""))
    head_commit = _git(["rev-parse", "--short", "HEAD"])
    if not plan_commit or plan_commit != head_commit:
        raise RuntimeError(
            "Release plan git commit must match HEAD before building deposition archive: "
            f"release_plan={plan_commit or '<missing>'}, HEAD={head_commit or '<unknown>'}"
        )


def _submission_files(submission_manifest: dict[str, Any]) -> list[str]:
    files = []
    for item in submission_manifest.get("outputs", []):
        path = str(item.get("path", ""))
        if path:
            files.append(path)
    return sorted(set(files))


def _archive_entry(path: str) -> str:
    return f"{ARCHIVE_ROOT}/{path}"


def _file_record(role: str, path: str) -> dict[str, Any]:
    source = ROOT / path
    return {
        "role": role,
        "path": path,
        "archive_path": _archive_entry(path),
        "bytes": source.stat().st_size,
        "sha256": _sha256(source),
    }


def _manifest_payload(
    release_plan: dict[str, Any],
    source_files: list[str],
    submission_files: list[str],
) -> dict[str, Any]:
    records = [_file_record("release_source", path) for path in source_files]
    records.extend(_file_record("submission_package", path) for path in submission_files)
    records = sorted(records, key=lambda item: (item["role"], item["path"]))
    return {
        "package_id": "BRAID-PLOSOne-deposition-package",
        "script": "benchmarks/scripts/make_deposition_archive.py",
        "git": {
            "commit": _git(["rev-parse", "--short", "HEAD"]),
            "branch": _git(["branch", "--show-current"]),
        },
        "archive_root": ARCHIVE_ROOT,
        "release_plan": _rel(RELEASE_PLAN),
        "submission_manifest": _rel(SUBMISSION_MANIFEST),
        "source_file_count": len(source_files),
        "submission_file_count": len(submission_files),
        "files": records,
        "known_limitations": [
            "This archive is a self-contained convenience bundle of the public source "
            "and submission files; the canonical public archive is the GitHub repository "
            "(https://github.com/kangk1204/BRAID). An external DOI deposit (Zenodo, "
            "Figshare, OSF) is optional and recommended only for long-term citability.",
            "Large BAM, rMATS, MAJIQ, and regenerated paper/output directories are not "
            "included; public accessions and tracked scripts regenerate those artifacts.",
        ],
        "release_blockers": release_plan.get("release_blockers", []),
    }


def _write_zip(archive_path: Path, manifest: dict[str, Any]) -> bytes:
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zf:
        for item in manifest["files"]:
            source = ROOT / item["path"]
            info = ZipInfo(item["archive_path"], ZIP_TIMESTAMP)
            info.compress_type = ZIP_DEFLATED
            zf.writestr(info, source.read_bytes())
        info = ZipInfo(f"{ARCHIVE_ROOT}/release_manifest.json", ZIP_TIMESTAMP)
        info.compress_type = ZIP_DEFLATED
        zf.writestr(info, manifest_bytes)
    return manifest_bytes


def build_deposition_archive() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    release_plan = _read_json(RELEASE_PLAN)
    submission_manifest = _read_json(SUBMISSION_MANIFEST)
    dirty_source = _dirty_release_source_status(release_plan)
    if dirty_source:
        raise RuntimeError(
            "Release source changes must be committed before building deposition archive: "
            + "; ".join(dirty_source[:10])
        )
    _ensure_release_plan_matches_head(release_plan)
    source_files = _source_files(release_plan)
    submission_files = _submission_files(submission_manifest)
    missing = [path for path in source_files + submission_files if not (ROOT / path).exists()]
    if missing:
        raise RuntimeError("Deposition archive inputs are missing: " + ", ".join(missing[:10]))
    manifest = _manifest_payload(release_plan, source_files, submission_files)
    manifest_bytes = _write_zip(ARCHIVE, manifest)
    external_manifest = {
        **manifest,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": {
            "path": _rel(ARCHIVE),
            "bytes": ARCHIVE.stat().st_size,
            "sha256": _sha256(ARCHIVE),
        },
        "embedded_manifest": {
            "archive_path": f"{ARCHIVE_ROOT}/release_manifest.json",
            "bytes": len(manifest_bytes),
            "sha256": _sha256_bytes(manifest_bytes),
        },
    }
    ARCHIVE_MANIFEST.write_text(
        json.dumps(external_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation_lines = [
        "BRAID PLOS One deposition archive validation",
        f"PASS\tarchive exists\t{_rel(ARCHIVE)}",
        f"PASS\tarchive sha256\t{external_manifest['archive']['sha256']}",
        f"PASS\trelease source files\t{len(source_files)}",
        f"PASS\tsubmission package files\t{len(submission_files)}",
        f"PASS\tembedded manifest sha256\t{external_manifest['embedded_manifest']['sha256']}",
    ]
    ARCHIVE_VALIDATION.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")
    return external_manifest


def main() -> None:
    manifest = build_deposition_archive()
    print(f"Wrote {_rel(ARCHIVE)}")
    print(f"Wrote {_rel(ARCHIVE_MANIFEST)}")
    print(f"Wrote {_rel(ARCHIVE_VALIDATION)}")
    print(f"Archive sha256: {manifest['archive']['sha256']}")


if __name__ == "__main__":
    main()
