#!/usr/bin/env python3
"""Audit the local BRAID PLOS One submission package for final-surface blockers."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "submission" / "plos_one"
MANUSCRIPT = ROOT / "paper" / "braid_calibrated_dpsi_manuscript.md"
PACKAGE_MANIFEST = OUT_DIR / "submission_package_manifest.json"
PACKAGE_VALIDATION = OUT_DIR / "submission_package_validation.txt"
TABLE_MANIFEST = ROOT / "paper" / "supplementary_data" / "manuscript_tables_manifest.json"
TABLE_VALIDATION = ROOT / "paper" / "supplementary_data" / "manuscript_tables_validation.txt"
RELEASE_PLAN_JSON = ROOT / "outputs" / "submission" / "release_deposition" / (
    "braid_release_deposition_plan.json"
)
RELEASE_PLAN_MD = ROOT / "outputs" / "submission" / "release_deposition" / (
    "braid_release_deposition_plan.md"
)
RELEASE_PLAN_VALIDATION = ROOT / "outputs" / "submission" / "release_deposition" / (
    "release_deposition_plan_validation.txt"
)
DEPOSITION_ARCHIVE = ROOT / "outputs" / "submission" / "release_deposition" / (
    "BRAID_PLOSOne_deposition_package.zip"
)
DEPOSITION_ARCHIVE_MANIFEST = ROOT / "outputs" / "submission" / "release_deposition" / (
    "BRAID_PLOSOne_deposition_package_manifest.json"
)
DEPOSITION_ARCHIVE_VALIDATION = ROOT / "outputs" / "submission" / "release_deposition" / (
    "BRAID_PLOSOne_deposition_package_validation.txt"
)
SUBMISSION_METADATA_JSON = OUT_DIR / "metadata" / "plos_submission_metadata.json"
SUBMISSION_METADATA_MD = OUT_DIR / "metadata" / "plos_submission_metadata.md"
README = ROOT / "README.md"
GITIGNORE = ROOT / ".gitignore"
ABSTRACT_MAX_WORDS = 300
FULL_TITLE_MAX_CHARS = 250
SHORT_TITLE_MAX_CHARS = 100
AUTHOR_YEAR_RE = re.compile(
    r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]+)"
    r"(?:\s+et\s+al\.|\s+and\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]+)?,\s*"
    r"(\d{4}[a-z]?)"
)
DEPOSITION_PLACEHOLDER_RE = re.compile(
    r"\bfinal public deposition identifier\b|\[(?:ARCHIVE_DOI|ARCHIVE_DOI_OR_REVIEWER_URL)\]",
    re.I,
)

STALE_TOKENS = [
    "StringTie",
    "Pertea",
    "Figure plan",
    "fig2_pacbio",
    "fig3_qki",
    "fig4_calibration",
    "fig5_positioning",
    "fig_roc_qki",
]
REQUIRED_FIGURE_LEGENDS = [
    "Figure 1. BRAID calculation scheme",
    "Figure 2. RT-PCR benchmark",
    "Figure 3. BRAID recalibration is caller-agnostic",
    "Figure 4. Filtering an rMATS operating point",
    "Figure 5. Active-learning calibration",
    "Figure 6. DM1 public rMATS application",
    "Figure 7. BRAID score-based candidate prioritization",
    "S1 Fig. Auxiliary TRA2 positive/negative detection check",
    # Supplementary figures S2-S5 are shipped in the submission package, so the
    # audit must independently catch a missing/renamed legend for any of them.
    "S2 Fig. Per-event adaptive interval widths",
    "S3 Fig. An event-type/support composite Mondrian calibrator",
    "S4 Fig. BRAID recovers literature-curated Esrp1/2 targets",
    "S5 Fig. Strict filtering",
    "S6 Fig. The confidence tiers stratify orthogonal RT-PCR agreement",
]
CLAIM_STRINGS = [
    ("pooled BRAID coverage", "0.971"),
    ("pooled MAJIQ coverage", "0.518"),
    ("pooled betAS coverage", "0.734"),
    ("pooled rMATS coverage", "0.633"),
    ("pooled BRAID interval score", "0.720"),
    ("pooled betAS interval score", "1.414"),
    ("pooled rMATS interval score", "1.625"),
    ("full rMATS-matched BRAID coverage", "0.949"),
    ("full rMATS-matched BRAID interval score", "0.754"),
    ("TRA2 matched positive targets", "76 RT-PCR-positive"),
    ("TRA2 matched negative targets", "36 RT-PCR-negative"),
    ("TRA2 rMATS true positives", "52 true positives"),
    ("TRA2 rMATS false positives", "8 false positives"),
    ("TRA2 rMATS MCC", "0.433"),
    ("TRA2 BRAID effect-supported true positives", "50 true positives"),
    ("TRA2 BRAID effect-supported false positives", "2 false positives"),
    ("TRA2 BRAID effect-supported MCC", "0.564"),
    ("DM1 post-support event count", "144,975"),
    ("DM1 rMATS-significant large-effect count", "967"),
    ("DM1 high-confidence tier count", "68"),
    ("software methods section", "Software implementation, dependencies, and test data"),
    ("software license", "MIT License"),
    ("python version requirement", "Python >=3.10"),
    ("commercial dependency disclosure", "No commercial"),
    ("core install command", "python -m pip install -e ."),
    ("supplied smoke test", "braid example"),
    ("regression test command", "python -m pytest tests/ -v"),
    ("benchmark environment file", "environment-benchmark.yml"),
    ("long-term utility and maintenance", "For long-term utility and maintenance"),
    ("stable hosting surfaces", "release archive and public repository"),
    ("ethics statement section", "Ethics statement"),
    ("ethics public data reuse", "secondary computational analysis"),
    ("ethics no new participants", "No new human participants were recruited"),
    ("ethics no new specimens", "no new biospecimens"),
    ("ethics no identifying data", "no protected health information"),
    ("ethics no additional consent", "no additional informed consent"),
]


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return [f"git status failed: {result.stderr.strip()}"]
    return [line for line in result.stdout.splitlines() if line.strip()]


def _has_fail_marker(path: Path) -> bool:
    text = _read_text(path)
    return any(line.startswith("FAIL") for line in text.splitlines())


def _section(text: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return ""
    next_heading = re.search(r"^## ", text[match.end() :], flags=re.MULTILINE)
    if not next_heading:
        return text[match.end() :]
    return text[match.end() : match.end() + next_heading.start()]


def _subsection(text: str, heading: str) -> str:
    """Return the body of a ``### heading`` subsection (h3), up to the next ``###``.

    Mirrors :func:`_section` but at the h3 level, used to carve out a single
    Results subsection (e.g. the differential-model robustness block) from the
    surrounding text. Returns "" when the heading is absent (so callers fail
    closed: a renamed heading leaves the carve-out empty rather than silently
    matching the whole document).
    """
    pattern = rf"^### {re.escape(heading)}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return ""
    next_heading = re.search(r"^### ", text[match.end() :], flags=re.MULTILINE)
    if not next_heading:
        return text[match.end() :]
    return text[match.end() : match.end() + next_heading.start()]


def _abstract_body(text: str) -> str:
    section = _section(text, "Abstract")
    return re.split(r"^Keywords:\s*", section, flags=re.MULTILINE)[0].strip()


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_title(text: str) -> str:
    first = _first_nonempty_line(text)
    return first[2:].strip() if first.startswith("# ") else first


def _extract_short_title(text: str) -> str:
    match = re.search(r"^Short title:\s*(.+)$", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _normalized_author_year_pairs(text: str) -> set[str]:
    normalized = re.sub(r"\s+", " ", text)
    return {f"{author} {year}" for author, year in AUTHOR_YEAR_RE.findall(normalized)}


def _reference_author_year_pairs(text: str) -> set[str]:
    references = _section(text, "References")
    references = references.split("## Figure legends", 1)[0]
    pairs = set()
    for entry in re.split(r"\n\s*\n", references.strip()):
        entry = re.sub(r"\s+", " ", entry.strip())
        if not entry:
            continue
        author_match = re.match(r"([A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]+)\b", entry)
        year_match = re.search(r"\b(19|20)\d{2}[a-z]?\b", entry)
        if author_match and year_match:
            pairs.add(f"{author_match.group(1)} {year_match.group(0)}")
    return pairs


def _word_count(text: str) -> int:
    # Whitespace-delimited count (PLOS/word-processor convention: a hyphenated
    # compound or a decimal number such as "RNA-seq" or "0.971" counts as one word).
    return len(text.split())


def _add(
    checks: list[dict[str, str]],
    status: str,
    item: str,
    detail: str,
    evidence: str | Path | None = None,
) -> None:
    record = {"status": status, "item": item, "detail": detail}
    if evidence is not None:
        record["evidence"] = _rel(evidence) if isinstance(evidence, Path) else evidence
    checks.append(record)


def _add_bool(
    checks: list[dict[str, str]],
    condition: bool,
    item: str,
    pass_detail: str,
    fail_status: str,
    fail_detail: str,
    evidence: str | Path | None = None,
) -> None:
    _add(
        checks,
        "PASS" if condition else fail_status,
        item,
        pass_detail if condition else fail_detail,
        evidence,
    )


def _audit_manuscript(checks: list[dict[str, str]], manuscript_text: str) -> None:
    _add_bool(
        checks,
        MANUSCRIPT.exists() and bool(manuscript_text.strip()),
        "canonical manuscript present",
        "canonical Markdown manuscript exists and is non-empty",
        "BLOCKER",
        "canonical Markdown manuscript is missing or empty",
        MANUSCRIPT,
    )
    title = _extract_title(manuscript_text)
    short_title = _extract_short_title(manuscript_text)
    _add_bool(
        checks,
        bool(title),
        "full title present",
        "title page contains a full title",
        "BLOCKER",
        "title page is missing a full title",
        MANUSCRIPT,
    )
    _add_bool(
        checks,
        bool(title) and len(title) <= FULL_TITLE_MAX_CHARS,
        "full title within PLOS limit",
        f"{len(title)} characters; PLOS limit is {FULL_TITLE_MAX_CHARS}",
        "BLOCKER",
        f"{len(title)} characters; PLOS limit is {FULL_TITLE_MAX_CHARS}",
        MANUSCRIPT,
    )
    _add_bool(
        checks,
        bool(short_title),
        "short title present",
        "title page contains a short title",
        "BLOCKER",
        "title page is missing a short title",
        MANUSCRIPT,
    )
    _add_bool(
        checks,
        bool(short_title) and len(short_title) <= SHORT_TITLE_MAX_CHARS,
        "short title within PLOS limit",
        f"{len(short_title)} characters; PLOS limit is {SHORT_TITLE_MAX_CHARS}",
        "BLOCKER",
        f"{len(short_title)} characters; PLOS limit is {SHORT_TITLE_MAX_CHARS}",
        MANUSCRIPT,
    )
    abstract_words = _word_count(_abstract_body(manuscript_text))
    _add_bool(
        checks,
        0 < abstract_words <= ABSTRACT_MAX_WORDS,
        "abstract length within PLOS limit",
        f"{abstract_words} words",
        "BLOCKER",
        f"{abstract_words} words; PLOS limit is {ABSTRACT_MAX_WORDS}",
        MANUSCRIPT,
    )
    for token in STALE_TOKENS:
        _add_bool(
            checks,
            token not in manuscript_text,
            f"stale token absent: {token}",
            "token not found in canonical manuscript",
            "BLOCKER",
            "stale or deprecated analysis surface remains in canonical manuscript",
            MANUSCRIPT,
        )
    for legend in REQUIRED_FIGURE_LEGENDS:
        _add_bool(
            checks,
            legend in manuscript_text,
            f"figure legend present: {legend}",
            "required figure legend is present",
            "BLOCKER",
            "required figure legend is missing",
            MANUSCRIPT,
        )
    _add_bool(
        checks,
        "## Figure legends" in manuscript_text,
        "figure legend section is submission-facing",
        "canonical manuscript contains a Figure legends section",
        "BLOCKER",
        "canonical manuscript still lacks a Figure legends section",
        MANUSCRIPT,
    )
    for label, value in CLAIM_STRINGS:
        claim_scope = (
            "secondary benchmark string present"
            if label.startswith("TRA2 ")
            else "central claim string present"
        )
        _add_bool(
            checks,
            value in manuscript_text,
            f"{claim_scope}: {label}",
            f"found {value}",
            "BLOCKER",
            f"expected claim string {value} was not found",
            MANUSCRIPT,
        )

    body_text = manuscript_text.split("## References", 1)[0]
    cited_pairs = _normalized_author_year_pairs(body_text)
    reference_pairs = _reference_author_year_pairs(manuscript_text)
    missing_refs = sorted(cited_pairs - reference_pairs)
    uncited_refs = sorted(reference_pairs - cited_pairs)
    _add_bool(
        checks,
        not missing_refs,
        "author-year citations resolve to references",
        f"{len(cited_pairs)} cited author-year pair(s) resolve to the reference list",
        "BLOCKER",
        "citation(s) missing from reference list: " + ", ".join(missing_refs),
        MANUSCRIPT,
    )
    _add_bool(
        checks,
        not uncited_refs,
        "reference entries are cited in text",
        f"{len(reference_pairs)} reference entry author-year pair(s) are cited",
        "BLOCKER",
        "reference entry author-year pair(s) not cited: " + ", ".join(uncited_refs),
        MANUSCRIPT,
    )

    pmids = re.findall(r"PMID:\d+\.", manuscript_text)
    duplicate_pmids = sorted(pmid for pmid, count in Counter(pmids).items() if count > 1)
    _add_bool(
        checks,
        not duplicate_pmids,
        "reference PMID entries are unique",
        "no duplicate PMID tokens detected",
        "WARN",
        "duplicate PMID token(s): " + ", ".join(duplicate_pmids),
        MANUSCRIPT,
    )
    # The superseded headline was a *pooled* BRAID coverage of 0.964 over the old
    # head-to-head set; the reframe removed it. The literal "0.964" now appears
    # only as a legitimate per-dataset TRA2 value (sum and rep) in the
    # differential-model robustness Supplementary Table, so exempt that one
    # subsection and keep blocking any reappearance in the abstract / main Results.
    diff_model_block = _subsection(
        manuscript_text,
        "The replicate-aware differential model leaves calibrated coverage unchanged",
    )
    manuscript_text_excl_diff_model = (
        manuscript_text.replace(diff_model_block, "") if diff_model_block
        else manuscript_text
    )
    _add_bool(
        checks,
        "0.964" not in manuscript_text_excl_diff_model,
        "stale pooled BRAID coverage absent",
        "abstract/results no longer carry the superseded 0.964 pooled BRAID coverage "
        "(per-dataset TRA2 value in the differential-model Supplementary Table is exempt)",
        "BLOCKER",
        "superseded pooled BRAID coverage 0.964 remains outside the "
        "differential-model robustness Supplementary Table",
        MANUSCRIPT,
    )

    code_availability = _section(manuscript_text, "Code availability")
    code_placeholder = bool(
        re.search(r"repository\s+URL\s+should\s+be\s+inserted", code_availability, re.I)
    )
    code_has_url = bool(re.search(r"https?://", code_availability))
    _add_bool(
        checks,
        code_has_url and not code_placeholder,
        "public repository URL finalized",
        "code availability contains a public URL and no placeholder language",
        "BLOCKER",
        "code availability still needs final public repository URL",
        MANUSCRIPT,
    )
    # PLOS code-availability compliance is satisfied by a public repository URL; an
    # external archival DOI (Zenodo/Figshare/OSF) is recommended best practice but not
    # required. The project archives via the public GitHub repository (GitHub-only), so
    # a finalized public repo URL clears this gate; a DOI, if present, also clears it.
    archive_doi_present = bool(
        re.search(r"(zenodo|figshare|osf|software\s+heritage)", code_availability, re.I)
    )
    if archive_doi_present:
        _add(
            checks,
            "PASS",
            "code archive is publicly resolvable",
            "code availability resolves to an external archival DOI",
            MANUSCRIPT,
        )
    elif code_has_url:
        _add(
            checks,
            "PASS",
            "code archive is publicly resolvable",
            "code availability resolves to a public Git repository URL; an external "
            "archival DOI (e.g. Zenodo) is recommended but not required for PLOS "
            "code-availability compliance",
            MANUSCRIPT,
        )
    else:
        _add(
            checks,
            "BLOCKER",
            "code archive is publicly resolvable",
            "neither a public repository URL nor an external archival DOI is present",
            MANUSCRIPT,
        )
    _add_bool(
        checks,
        "[Insert funding statement before submission.]" not in manuscript_text,
        "funding placeholder absent from manuscript",
        "financial disclosure placeholder is not in the manuscript body",
        "BLOCKER",
        "funding statement placeholder remains",
        MANUSCRIPT,
    )
    _add_bool(
        checks,
        "## Funding" not in manuscript_text,
        "funding statement kept out of manuscript file",
        "PLOS financial disclosure will be entered in the submission system",
        "BLOCKER",
        "funding section remains in the manuscript file",
        MANUSCRIPT,
    )
    competing_interests = _section(manuscript_text, "Competing interests")
    _add_bool(
        checks,
        "[Confirm before submission.]" not in competing_interests,
        "competing-interest confirmation placeholder absent",
        "competing-interest section no longer contains confirmation placeholder",
        "BLOCKER",
        "competing-interest confirmation placeholder remains",
        MANUSCRIPT,
    )
    _add_bool(
        checks,
        bool(competing_interests.strip()),
        "competing-interest statement present",
        "competing-interest section contains a manuscript statement",
        "BLOCKER",
        "competing-interest section is missing or empty",
        MANUSCRIPT,
    )
    _add_bool(
        checks,
        "large BAM, rMATS, and MAJIQ output directories are stored locally" not in manuscript_text,
        "large benchmark artifacts publicly resolved",
        "data availability no longer depends on local regeneration language",
        "WARN",
        "large BAM/rMATS/MAJIQ artifacts still need public-release regeneration/deposition plan",
        MANUSCRIPT,
    )
    data_availability = re.sub(r"\s+", " ", _section(manuscript_text, "Data availability"))
    for label, pattern, value in [
        ("minimal data set", r"\bminimal data set\b", "minimal data set"),
        (
            "figure source-data workbooks",
            r"\bfigure\s+source-data workbooks\b",
            "figure source-data workbooks",
        ),
    ]:
        _add_bool(
            checks,
            bool(re.search(pattern, data_availability)),
            f"data availability states {label}",
            f"found {value}",
            "BLOCKER",
            f"data availability is missing {value}",
            MANUSCRIPT,
        )
    has_pending_deposition_language = bool(DEPOSITION_PLACEHOLDER_RE.search(data_availability))
    data_points_to_repo = bool(
        re.search(r"https?://\S*github|public Git repository", data_availability, re.I)
    )
    if data_points_to_repo and not has_pending_deposition_language:
        _add(
            checks,
            "PASS",
            "data availability deposition finalization state",
            "data availability points to the public Git repository as the archive "
            "(GitHub-only; no external deposition pending)",
            MANUSCRIPT,
        )
    elif has_pending_deposition_language:
        _add(
            checks,
            "INFO",
            "data availability deposition finalization state",
            "deposition identifier is explicitly pending insertion",
            MANUSCRIPT,
        )
    else:
        _add(
            checks,
            "WARN",
            "data availability deposition finalization state",
            "data availability neither points to a public repository nor marks a "
            "pending deposition",
            MANUSCRIPT,
        )


def _audit_validation_files(checks: list[dict[str, str]]) -> None:
    for label, path in [
        ("submission package validation", PACKAGE_VALIDATION),
        ("manuscript table validation", TABLE_VALIDATION),
    ]:
        _add_bool(
            checks,
            path.exists() and path.stat().st_size > 0,
            f"{label} file present",
            "validation file exists and is non-empty",
            "BLOCKER",
            "validation file is missing or empty",
            path,
        )
        _add_bool(
            checks,
            path.exists() and not _has_fail_marker(path),
            f"{label} has no FAIL marker",
            "no line starts with FAIL",
            "BLOCKER",
            "validation file contains a FAIL marker",
            path,
        )


def _audit_package_manifest(checks: list[dict[str, str]]) -> None:
    manifest = _read_json(PACKAGE_MANIFEST)
    _add_bool(
        checks,
        bool(manifest),
        "submission package manifest present",
        "manifest JSON loaded",
        "BLOCKER",
        "submission package manifest is missing or invalid",
        PACKAGE_MANIFEST,
    )
    for item in manifest.get("outputs", []):
        path = ROOT / item.get("path", "")
        expected_hash = item.get("sha256", "")
        exists = path.exists() and path.stat().st_size > 0
        _add_bool(
            checks,
            exists,
            f"packaged output exists: {item.get('path', '<missing path>')}",
            "output exists and is non-empty",
            "BLOCKER",
            "manifest-declared output is missing or empty",
            path,
        )
        if exists and expected_hash:
            actual_hash = _sha256(path)
            _add_bool(
                checks,
                actual_hash == expected_hash,
                f"packaged output hash matches: {item.get('path', '<missing path>')}",
                "current file hash matches package manifest",
                "BLOCKER",
                f"hash drift: expected {expected_hash}, observed {actual_hash}",
                path,
            )

    package_md = OUT_DIR / "manuscript" / "BRAID_PLOSOne_manuscript.md"
    _add_bool(
        checks,
        package_md.exists() and package_md.read_text(encoding="utf-8") == _read_text(MANUSCRIPT),
        "packaged Markdown matches canonical manuscript",
        "packaged manuscript Markdown is byte-for-byte equal to canonical Markdown",
        "BLOCKER",
        "packaged Markdown manuscript has drifted from canonical Markdown",
        package_md,
    )


def _audit_table_manifest(checks: list[dict[str, str]]) -> None:
    manifest = _read_json(TABLE_MANIFEST)
    _add_bool(
        checks,
        bool(manifest),
        "manuscript table manifest present",
        "table manifest JSON loaded",
        "BLOCKER",
        "manuscript table manifest is missing or invalid",
        TABLE_MANIFEST,
    )
    central = manifest.get("central_claims", {})
    auxiliary = manifest.get("auxiliary_metrics", {})
    expected_claims = {
        "common_set_braid_coverage": 0.9712230215827338,
        "srs354082_braid_abs_coverage": 0.9705882352941176,
    }
    expected_auxiliary_metrics = {
        "tra2_rmats_mcc": 0.4326251176569044,
        "tra2_braid_supported_mcc": 0.5640555331476095,
    }
    for key, expected in expected_claims.items():
        actual = central.get(key)
        _add_bool(
            checks,
            isinstance(actual, (int, float)) and abs(actual - expected) < 1e-12,
            f"table central claim matches: {key}",
            f"{actual}",
            "BLOCKER",
            f"expected {expected}, observed {actual}",
            TABLE_MANIFEST,
        )
    for key, expected in expected_auxiliary_metrics.items():
        actual = auxiliary.get(key)
        _add_bool(
            checks,
            isinstance(actual, (int, float)) and abs(actual - expected) < 1e-12,
            f"table auxiliary metric matches: {key}",
            f"{actual}",
            "BLOCKER",
            f"expected {expected}, observed {actual}",
            TABLE_MANIFEST,
        )


def _release_plan_package_mismatches(
    plan: dict[str, Any],
    package_manifest: dict[str, Any],
) -> list[str]:
    manifest_outputs = {
        item.get("path", ""): item for item in package_manifest.get("outputs", [])
    }
    mismatches: list[str] = []
    for item in plan.get("submission_outputs", []):
        path = item.get("path", "")
        current = manifest_outputs.get(path)
        if not current:
            mismatches.append(f"{path}: missing from current package manifest")
            continue
        if item.get("bytes") != current.get("bytes") or item.get("sha256") != current.get(
            "sha256"
        ):
            mismatches.append(
                f"{path}: release plan bytes/hash do not match current package manifest"
            )
    return mismatches


def _audit_release_plan(checks: list[dict[str, str]]) -> None:
    for label, path in [
        ("release/deposition plan JSON", RELEASE_PLAN_JSON),
        ("release/deposition plan Markdown", RELEASE_PLAN_MD),
        ("release/deposition plan validation", RELEASE_PLAN_VALIDATION),
    ]:
        _add_bool(
            checks,
            path.exists() and path.stat().st_size > 0,
            f"{label} present",
            "release/deposition support file exists and is non-empty",
            "BLOCKER",
            "release/deposition support file is missing or empty",
            path,
        )
    plan = _read_json(RELEASE_PLAN_JSON)
    package_manifest = _read_json(PACKAGE_MANIFEST)
    head_commit = _git(["rev-parse", "--short", "HEAD"])
    plan_commit = str(plan.get("git", {}).get("commit", ""))
    _add_bool(
        checks,
        bool(head_commit) and plan_commit == head_commit,
        "release/deposition plan records current HEAD",
        f"release plan commit {plan_commit}",
        "BLOCKER",
        f"release plan commit {plan_commit or '<missing>'} does not match HEAD {head_commit}",
        RELEASE_PLAN_JSON,
    )
    _add_bool(
        checks,
        bool(plan.get("release_blockers")),
        "release/deposition plan records blockers",
        "release/deposition plan carries explicit finalization blockers",
        "BLOCKER",
        "release/deposition plan is missing blocker list",
        RELEASE_PLAN_JSON,
    )
    _add_bool(
        checks,
        len(plan.get("datasets", [])) >= 4,
        "release/deposition plan records public datasets",
        f"{len(plan.get('datasets', []))} dataset entries",
        "BLOCKER",
        "release/deposition plan does not record the expected dataset set",
        RELEASE_PLAN_JSON,
    )
    local_only_paths = set(plan.get("local_only_paths", []))
    expected_app_local_only = {
        "benchmarks/application_dm1/raw/",
        "benchmarks/application_dm1/rmats/",
        "benchmarks/application_dm1/results/",
        "benchmarks/application_esrp/raw/",
        "benchmarks/application_esrp/rmats/",
        "benchmarks/application_esrp/results/",
    }
    missing_dm1_local_only = sorted(expected_app_local_only - local_only_paths)
    _add_bool(
        checks,
        not missing_dm1_local_only,
        "release/deposition plan marks DM1/Esrp generated dirs local-only",
        "DM1 and Esrp raw, rMATS, and result directories are recorded as regenerable",
        "BLOCKER",
        "missing local-only path(s): " + ", ".join(missing_dm1_local_only),
        RELEASE_PLAN_JSON,
    )
    private_remote = bool(plan.get("git", {}).get("remote_appears_private"))
    _add(
        checks,
        "WARN" if private_remote else "PASS",
        "current git remote is public-release appropriate",
        "remote appears private" if private_remote else "remote does not look private by name",
        RELEASE_PLAN_JSON,
    )
    mismatches = _release_plan_package_mismatches(plan, package_manifest)
    _add_bool(
        checks,
        not mismatches,
        "release/deposition plan matches current package manifest",
        "release plan submission outputs match current package manifest bytes and hashes",
        "BLOCKER",
        f"{len(mismatches)} mismatch(es): " + "; ".join(mismatches[:5]),
        RELEASE_PLAN_JSON,
    )


def _audit_deposition_archive(checks: list[dict[str, str]]) -> None:
    for label, path in [
        ("deposition archive zip", DEPOSITION_ARCHIVE),
        ("deposition archive manifest", DEPOSITION_ARCHIVE_MANIFEST),
        ("deposition archive validation", DEPOSITION_ARCHIVE_VALIDATION),
    ]:
        _add_bool(
            checks,
            path.exists() and path.stat().st_size > 0,
            f"{label} present",
            "deposition archive support file exists and is non-empty",
            "BLOCKER",
            "deposition archive support file is missing or empty",
            path,
        )
    manifest = _read_json(DEPOSITION_ARCHIVE_MANIFEST)
    archive = manifest.get("archive", {})
    expected_sha = archive.get("sha256")
    actual_sha = _sha256(DEPOSITION_ARCHIVE) if DEPOSITION_ARCHIVE.exists() else ""
    _add_bool(
        checks,
        bool(expected_sha) and expected_sha == actual_sha,
        "deposition archive checksum matches manifest",
        f"{actual_sha}",
        "BLOCKER",
        f"expected {expected_sha}, observed {actual_sha}",
        DEPOSITION_ARCHIVE_MANIFEST,
    )
    validation_text = _read_text(DEPOSITION_ARCHIVE_VALIDATION)
    _add_bool(
        checks,
        bool(validation_text) and "FAIL" not in validation_text,
        "deposition archive validation has no FAIL marker",
        "no FAIL marker detected",
        "BLOCKER",
        "deposition archive validation contains FAIL or is empty",
        DEPOSITION_ARCHIVE_VALIDATION,
    )
    source_count = int(manifest.get("source_file_count", 0) or 0)
    submission_count = int(manifest.get("submission_file_count", 0) or 0)
    _add_bool(
        checks,
        source_count > 0 and submission_count > 0,
        "deposition archive includes source and submission package files",
        f"{source_count} source file(s), {submission_count} submission file(s)",
        "BLOCKER",
        "deposition archive manifest has missing source or submission files",
        DEPOSITION_ARCHIVE_MANIFEST,
    )
    local_only_prefixes = [
        "benchmarks/application_dm1/raw/",
        "benchmarks/application_dm1/rmats/",
        "benchmarks/application_dm1/results/",
        "benchmarks/application_esrp/raw/",
        "benchmarks/application_esrp/rmats/",
        "benchmarks/application_esrp/results/",
        "data/",
    ]
    included_local_only = [
        item.get("path", "")
        for item in manifest.get("files", [])
        if any(str(item.get("path", "")).startswith(prefix) for prefix in local_only_prefixes)
    ]
    _add_bool(
        checks,
        not included_local_only,
        "deposition archive excludes local-only benchmark artifacts",
        "no local-only benchmark artifact paths included",
        "BLOCKER",
        "included local-only path(s): " + ", ".join(included_local_only[:5]),
        DEPOSITION_ARCHIVE_MANIFEST,
    )
    head_commit = _git(["rev-parse", "--short", "HEAD"])
    archive_commit = str(manifest.get("git", {}).get("commit", ""))
    _add_bool(
        checks,
        bool(head_commit) and archive_commit == head_commit,
        "deposition archive manifest records current HEAD",
        f"deposition archive commit {archive_commit}",
        "BLOCKER",
        f"deposition archive commit {archive_commit or '<missing>'} does not match "
        f"HEAD {head_commit}; rebuild the archive at the current commit",
        DEPOSITION_ARCHIVE_MANIFEST,
    )


def _audit_submission_metadata(checks: list[dict[str, str]]) -> None:
    for label, path in [
        ("PLOS submission metadata JSON", SUBMISSION_METADATA_JSON),
        ("PLOS submission metadata Markdown", SUBMISSION_METADATA_MD),
    ]:
        _add_bool(
            checks,
            path.exists() and path.stat().st_size > 0,
            f"{label} present",
            "submission metadata support file exists and is non-empty",
            "BLOCKER",
            "submission metadata support file is missing or empty",
            path,
        )

    metadata = _read_json(SUBMISSION_METADATA_JSON)
    release_plan = _read_json(RELEASE_PLAN_JSON)
    metadata_generated_at = _parse_timestamp(metadata.get("generated_at"))
    release_plan_generated_at = _parse_timestamp(release_plan.get("generated_at"))
    _add_bool(
        checks,
        bool(metadata_generated_at and release_plan_generated_at)
        and metadata_generated_at >= release_plan_generated_at,
        "PLOS metadata is current with release/deposition plan",
        "metadata generated at or after the current release plan",
        "BLOCKER",
        (
            "metadata generated_at is older than release plan or one timestamp is missing: "
            f"metadata={metadata.get('generated_at')}, "
            f"release_plan={release_plan.get('generated_at')}"
        ),
        SUBMISSION_METADATA_JSON,
    )
    release_context = metadata.get("release_context", {})
    _add_bool(
        checks,
        release_context.get("git", {}).get("commit")
        == release_plan.get("git", {}).get("commit")
        and bool(release_plan.get("git", {}).get("commit")),
        "PLOS metadata records current release commit",
        f"release commit {release_plan.get('git', {}).get('commit')}",
        "BLOCKER",
        (
            "metadata release_context git commit does not match release plan: "
            f"metadata={release_context.get('git', {}).get('commit')}, "
            f"release_plan={release_plan.get('git', {}).get('commit')}"
        ),
        SUBMISSION_METADATA_JSON,
    )
    finalization_fields = {
        field.get("field"): field.get("status") for field in metadata.get("finalization_fields", [])
    }
    required_finalization_fields = [
        "public_repository_url",
        "archive_doi_or_private_reviewer_url",
        "funding_statement",
        "competing_interests",
    ]
    if release_plan.get("git", {}).get("remote_appears_private"):
        required_finalization_fields.append("public_release_remote")
    for field in required_finalization_fields:
        _add_bool(
            checks,
            finalization_fields.get(field) in {"needs_author_action", "needs_author_confirmation"},
            f"PLOS metadata finalization field recorded: {field}",
            f"{finalization_fields.get(field)}",
            "BLOCKER",
            "required PLOS finalization field is not recorded",
            SUBMISSION_METADATA_JSON,
        )
    _add_bool(
        checks,
        metadata.get("target_journal") == "PLOS ONE",
        "PLOS metadata target journal recorded",
        "target journal is PLOS ONE",
        "BLOCKER",
        "target journal is missing or not PLOS ONE",
        SUBMISSION_METADATA_JSON,
    )
    title = str(metadata.get("title", ""))
    short_title = str(metadata.get("short_title", ""))
    _add_bool(
        checks,
        bool(title) and len(title) <= FULL_TITLE_MAX_CHARS,
        "PLOS metadata full title within limit",
        f"{len(title)} characters; PLOS limit is {FULL_TITLE_MAX_CHARS}",
        "BLOCKER",
        f"{len(title)} characters; PLOS limit is {FULL_TITLE_MAX_CHARS}",
        SUBMISSION_METADATA_JSON,
    )
    _add_bool(
        checks,
        bool(short_title) and len(short_title) <= SHORT_TITLE_MAX_CHARS,
        "PLOS metadata short title within limit",
        f"{len(short_title)} characters; PLOS limit is {SHORT_TITLE_MAX_CHARS}",
        "BLOCKER",
        f"{len(short_title)} characters; PLOS limit is {SHORT_TITLE_MAX_CHARS}",
        SUBMISSION_METADATA_JSON,
    )
    matrix = metadata.get("scope_compliance_matrix", [])
    matrix_by_criterion = {
        str(row.get("criterion")): str(row.get("status")) for row in matrix if isinstance(row, dict)
    }
    for criterion in [
        "utility",
        "validation",
        "availability",
        "algorithm_details",
        "open_source_and_dependencies",
        "test_data_install_and_run",
        "long_term_utility_and_maintenance",
    ]:
        expected_statuses = (
            {"addressed_in_manuscript", "needs_external_finalization"}
            if criterion == "availability"
            else {"addressed_in_manuscript"}
        )
        _add_bool(
            checks,
            matrix_by_criterion.get(criterion) in expected_statuses,
            f"PLOS scope matrix criterion recorded: {criterion}",
            f"{matrix_by_criterion.get(criterion)}",
            "BLOCKER",
            f"criterion missing or incomplete: {matrix_by_criterion.get(criterion)}",
            SUBMISSION_METADATA_JSON,
        )


def _audit_repo_surface(checks: list[dict[str, str]]) -> list[str]:
    gitignore_text = _read_text(GITIGNORE)
    for ignored in ["outputs/tables/", "outputs/submission/", "paper/"]:
        _add_bool(
            checks,
            ignored in gitignore_text,
            f"local artifact path ignored: {ignored}",
            "artifact path is listed in .gitignore",
            "WARN",
            "local artifact path is not listed in .gitignore",
            GITIGNORE,
        )

    readme_text = _read_text(README)
    _add_bool(
        checks,
        "StringTie" not in readme_text,
        "README excludes StringTie",
        "README no longer advertises the removed StringTie surface",
        "BLOCKER",
        "README still contains StringTie",
        README,
    )

    for stale_path in [ROOT / "paper" / "braid.tex", ROOT / "paper" / "_build_calibrated.md"]:
        if stale_path.exists() and any(token in _read_text(stale_path) for token in STALE_TOKENS):
            _add(
                checks,
                "WARN",
                f"stale non-canonical artifact present: {_rel(stale_path)}",
                "ignored paper build artifact still contains stale tokens; "
                "submission package uses canonical Markdown",
                stale_path,
            )

    status_lines = _git_status()
    untracked = [line for line in status_lines if line.startswith("??")]
    tracked_changed = [line for line in status_lines if not line.startswith("??")]
    _add(
        checks,
        "BLOCKER" if tracked_changed else "PASS",
        "tracked source changes",
        f"{len(tracked_changed)} tracked status line(s)",
        "; ".join(tracked_changed) if tracked_changed else "clean tracked status",
    )
    _add(
        checks,
        "WARN" if untracked else "PASS",
        "untracked files before release commit",
        f"{len(untracked)} untracked status line(s)",
        "; ".join(untracked) if untracked else "none",
    )
    return status_lines


def _write_reports(checks: list[dict[str, str]], git_status: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = Counter(check["status"] for check in checks)
    if counts.get("BLOCKER"):
        verdict = "BLOCKED"
    elif counts.get("WARN"):
        verdict = "READY_WITH_WARNINGS"
    else:
        verdict = "READY"
    payload = {
        "audit_id": "BRAID-PLOSOne-submission-readiness-audit",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "benchmarks/scripts/audit_submission_readiness.py",
        "verdict": verdict,
        "status_counts": dict(sorted(counts.items())),
        "checks": checks,
        "git_status_short": git_status,
    }
    json_path = OUT_DIR / "submission_readiness_audit.json"
    md_path = OUT_DIR / "submission_readiness_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# BRAID PLOS One Submission Readiness Audit",
        "",
        f"Generated: {payload['generated_at']}",
        f"Verdict: **{verdict}**",
        "",
        "| Status | Item | Detail | Evidence |",
        "|---|---|---|---|",
    ]
    for check in checks:
        lines.append(
            "| {status} | {item} | {detail} | {evidence} |".format(
                status=check["status"],
                item=check["item"].replace("|", "\\|"),
                detail=check["detail"].replace("|", "\\|"),
                evidence=check.get("evidence", "").replace("|", "\\|"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {_rel(json_path)}")
    print(f"Wrote {_rel(md_path)}")
    print(f"Verdict: {verdict}")
    print("Status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def main() -> None:
    checks: list[dict[str, str]] = []
    manuscript_text = _read_text(MANUSCRIPT)
    _audit_manuscript(checks, manuscript_text)
    _audit_validation_files(checks)
    _audit_package_manifest(checks)
    _audit_table_manifest(checks)
    _audit_release_plan(checks)
    _audit_deposition_archive(checks)
    _audit_submission_metadata(checks)
    git_status = _audit_repo_surface(checks)
    _write_reports(checks, git_status)


if __name__ == "__main__":
    main()
