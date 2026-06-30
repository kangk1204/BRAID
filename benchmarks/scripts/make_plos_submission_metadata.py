#!/usr/bin/env python3
"""Build PLOS One submission-form metadata from the canonical BRAID manuscript."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = ROOT / "paper" / "braid_calibrated_dpsi_manuscript.md"
RELEASE_PLAN = ROOT / "outputs" / "submission" / "release_deposition" / (
    "braid_release_deposition_plan.json"
)
OUT_DIR = ROOT / "outputs" / "submission" / "plos_one" / "metadata"
FULL_TITLE_MAX_CHARS = 250
SHORT_TITLE_MAX_CHARS = 100

PLOS_POLICY_BASIS = [
    {
        "source": "PLOS ONE Getting Started",
        "url": "https://journals.plos.org/plosone/s/getting-started",
        "relevance": (
            "submission preparation requires funding, competing interests, "
            "data availability, author information, and license readiness"
        ),
    },
    {
        "source": "PLOS ONE Data Availability",
        "url": "https://journals.plos.org/plosone/s/data-availability",
        "relevance": (
            "data needed to replicate findings must be publicly available "
            "without restriction at publication unless a restriction is justified"
        ),
    },
    {
        "source": "PLOS ONE Submission Guidelines",
        "url": "https://journals.plos.org/plosone/s/submission-guidelines",
        "relevance": (
            "software submissions require algorithm, dependency, test-data, "
            "installation, and run details"
        ),
    },
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _section(text: str, heading: str, level: int = 2) -> str:
    marker = "#" * level
    pattern = rf"^{re.escape(marker)} {re.escape(heading)}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return ""
    next_heading = re.search(rf"^{re.escape(marker)} ", text[match.end() :], flags=re.MULTILINE)
    if not next_heading:
        return text[match.end() :].strip()
    return text[match.end() : match.end() + next_heading.start()].strip()


def _plain(section: str) -> str:
    section = re.sub(r"<[^>]+>", "", section)
    section = re.sub(r"`([^`]+)`", r"\1", section)
    section = re.sub(r"\s+", " ", section)
    return section.strip()


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_title(manuscript: str) -> str:
    first = _first_nonempty_line(manuscript)
    return first[2:].strip() if first.startswith("# ") else first


def _extract_short_title(manuscript: str) -> str:
    match = re.search(r"^Short title:\s*(.+)$", manuscript, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def _strip_superscripts(line: str) -> str:
    return re.sub(r"<sup>.*?</sup>", "", line).strip()


def _extract_keywords(manuscript: str) -> list[str]:
    match = re.search(r"^Keywords:\s*(.+)$", manuscript, flags=re.MULTILINE)
    if not match:
        return []
    return [item.strip() for item in match.group(1).split(";") if item.strip()]


def _abstract_body(manuscript: str) -> str:
    return re.split(r"^Keywords:\s*", _section(manuscript, "Abstract"), flags=re.MULTILINE)[
        0
    ].strip()


def _extract_corresponding_author(manuscript: str) -> dict[str, str]:
    author_line = ""
    for line in manuscript.splitlines():
        if "<sup>1,*</sup>" in line:
            author_line = _strip_superscripts(line)
            break
    email_match = re.search(r"[\w.+-]+@[\w.-]+", manuscript)
    affiliation = ""
    for line in manuscript.splitlines():
        if line.startswith("<sup>1</sup>"):
            affiliation = _strip_superscripts(line)
            break
    return {
        "name": author_line,
        "email": email_match.group(0) if email_match else "",
        "affiliation": affiliation,
    }


def _finalization_fields(manuscript: str, release_plan: dict[str, Any]) -> list[dict[str, str]]:
    funding_evidence = _plain(_section(manuscript, "Funding")) or (
        "Financial disclosure is intentionally not stored in the manuscript body; "
        "enter the author-approved statement in the PLOS submission system."
    )
    fields = [
        {
            "field": "public_repository_url",
            "status": "needs_author_action",
            "current_evidence": (
                "Code availability cites the public repository URL "
                "https://github.com/kangk1204/BRAID."
            ),
            "required_action": (
                "Confirm the public repository hosts the submitted release commit; the "
                "URL is already inserted in Code availability."
            ),
        },
        {
            "field": "archive_doi_or_private_reviewer_url",
            "status": "needs_author_action",
            "current_evidence": "No archive DOI or deposition record is present in the manuscript.",
            "required_action": (
                "Deposit the release archive and supplementary package, then paste the DOI "
                "or reviewer-access URL supplied by the repository."
            ),
        },
        {
            "field": "funding_statement",
            "status": "needs_author_confirmation",
            "current_evidence": funding_evidence,
            "required_action": (
                "Enter the final author-approved financial disclosure in the PLOS "
                "submission system."
            ),
        },
        {
            "field": "competing_interests",
            "status": "needs_author_confirmation",
            "current_evidence": _plain(_section(manuscript, "Competing interests"))
            or "Competing interests section missing.",
            "required_action": (
                "Confirm whether the current no-competing-interests statement is final."
            ),
        },
    ]
    if release_plan.get("git", {}).get("remote_appears_private"):
        fields.append(
            {
                "field": "public_release_remote",
                "status": "needs_author_action",
                "current_evidence": "Current git remote appears private in the release plan.",
                "required_action": "Push or mirror the release commit to the final public remote.",
            }
        )
    return fields


def _scope_compliance_matrix(manuscript: str) -> list[dict[str, str]]:
    return [
        {
            "criterion": "utility",
            "status": "addressed_in_manuscript",
            "evidence": (
                "Abstract, Introduction, Results, and Discussion define BRAID as a "
                "caller-agnostic calibrated Delta PSI uncertainty layer and benchmark "
                "its advantage over existing RNA-seq-derived intervals."
            ),
        },
        {
            "criterion": "validation",
            "status": "addressed_in_manuscript",
            "evidence": (
                "Results, Figure 2, Supplementary Figure 1, and Supplementary Tables "
                "report RT-PCR benchmark coverage, interval score, cross-dataset "
                "transfer, recalibration, and secondary TRA2 MCC checks."
            ),
        },
        {
            "criterion": "availability",
            "status": "needs_external_finalization",
            "evidence": (
                "Code and Data availability sections plus the release/deposition plan "
                "define the public repository URL and archive DOI fields still needed "
                "before submission."
            ),
        },
        {
            "criterion": "algorithm_details",
            "status": "addressed_in_manuscript",
            "evidence": (
                "Methods sections describe Delta PSI construction, conformal "
                "calibration, comparator intervals, statistical evaluation, and the "
                "DM1 rMATS application workflow."
            ),
        },
        {
            "criterion": "open_source_and_dependencies",
            "status": "addressed_in_manuscript",
            "evidence": (
                "Software implementation states MIT License, Python >=3.10, open-source "
                "core dependencies, benchmark environment file, package extras, and no "
                "commercial dependency for the documented rMATS workflow."
            ),
        },
        {
            "criterion": "test_data_install_and_run",
            "status": "addressed_in_manuscript",
            "evidence": (
                "Software implementation gives pip install commands, the dependency-free "
                "braid example smoke test, regression-test command, and paper figure/table "
                "regeneration scripts."
            ),
        },
        {
            "criterion": "long_term_utility_and_maintenance",
            "status": (
                "addressed_in_manuscript"
                if "For long-term utility and maintenance" in manuscript
                else "missing_from_manuscript"
            ),
            "evidence": (
                "Software implementation discusses the command-line maintenance model, "
                "non-commercial core workflow, regression checks, extension path, and "
                "release archive/public repository as stable hosting surfaces."
            ),
        },
    ]


def _release_context(release_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "release_plan": str(RELEASE_PLAN.relative_to(ROOT)),
        "release_plan_generated_at": release_plan.get("generated_at", ""),
        "git": release_plan.get("git", {}),
        "release_blockers": release_plan.get("release_blockers", []),
    }


def build_metadata() -> dict[str, Any]:
    manuscript = _read_text(MANUSCRIPT)
    release_plan = _read_json(RELEASE_PLAN)
    data_availability = _plain(_section(manuscript, "Data availability"))
    code_availability = _plain(_section(manuscript, "Code availability"))
    title = _extract_title(manuscript)
    short_title = _extract_short_title(manuscript)
    return {
        "metadata_id": "BRAID-PLOSOne-submission-form-support",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "benchmarks/scripts/make_plos_submission_metadata.py",
        "source_manuscript": str(MANUSCRIPT.relative_to(ROOT)),
        "target_journal": "PLOS ONE",
        "article_type": "Methods, software, databases, and tools",
        "title": title,
        "short_title": short_title,
        "title_lengths": {
            "full_title_characters": len(title),
            "short_title_characters": len(short_title),
            "full_title_limit": FULL_TITLE_MAX_CHARS,
            "short_title_limit": SHORT_TITLE_MAX_CHARS,
        },
        "corresponding_author": _extract_corresponding_author(manuscript),
        "abstract": _plain(_abstract_body(manuscript)),
        "keywords": _extract_keywords(manuscript),
        "ethics_statement": _plain(_section(manuscript, "Ethics statement", level=3)),
        "data_availability_current": data_availability,
        "code_availability_current": code_availability,
        "data_availability_final_template": (
            data_availability
            + " The public release archive and supplementary package are available at "
            "[ARCHIVE_DOI_OR_REVIEWER_URL]."
        ).strip(),
        "code_availability_final_template": (
            "BRAID source code is available at [PUBLIC_REPOSITORY_URL] and archived at "
            "[ARCHIVE_DOI]. The submitted release package includes figure source data, "
            "supplementary tables, manifests, and validation reports."
        ),
        "funding_current": _plain(_section(manuscript, "Funding"))
        or "Not stored in manuscript; enter final financial disclosure in PLOS submission system.",
        "competing_interests_current": _plain(_section(manuscript, "Competing interests")),
        "release_context": _release_context(release_plan),
        "finalization_fields": _finalization_fields(manuscript, release_plan),
        "scope_compliance_matrix": _scope_compliance_matrix(manuscript),
        "plos_policy_basis": PLOS_POLICY_BASIS,
    }


def _write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# BRAID PLOS ONE Submission Metadata Support",
        "",
        f"Generated: {payload['generated_at']}",
        f"Target journal: {payload['target_journal']}",
        f"Article type: {payload['article_type']}",
        "",
        "## Title",
        "",
        payload["title"],
        "",
        "## Short Title",
        "",
        payload["short_title"],
        "",
        "## Corresponding Author",
        "",
        f"- Name: {payload['corresponding_author']['name']}",
        f"- Email: {payload['corresponding_author']['email']}",
        f"- Affiliation: {payload['corresponding_author']['affiliation']}",
        "",
        "## Keywords",
        "",
    ]
    for keyword in payload["keywords"]:
        lines.append(f"- {keyword}")
    lines.extend(
        [
            "",
            "## Current Data Availability",
            "",
            payload["data_availability_current"],
            "",
            "## Current Code Availability",
            "",
            payload["code_availability_current"],
            "",
            "## Release Context",
            "",
            f"- Release plan: `{payload['release_context']['release_plan']}`",
            f"- Release plan generated: {payload['release_context']['release_plan_generated_at']}",
            f"- Git commit: `{payload['release_context']['git'].get('commit', '')}`",
            f"- Git branch: `{payload['release_context']['git'].get('branch', '')}`",
            f"- Remaining release blockers: {len(payload['release_context']['release_blockers'])}",
            "",
            "## Final Data Availability Template",
            "",
            payload["data_availability_final_template"],
            "",
            "## Final Code Availability Template",
            "",
            payload["code_availability_final_template"],
            "",
            "## Finalization Fields",
            "",
        ]
    )
    for field in payload["finalization_fields"]:
        lines.append(f"- `{field['field']}` ({field['status']}): {field['required_action']}")
    lines.extend(["", "## Scope Compliance Matrix", ""])
    lines.append("| Criterion | Status | Evidence |")
    lines.append("|---|---|---|")
    for row in payload["scope_compliance_matrix"]:
        evidence = row["evidence"].replace("|", "\\|")
        lines.append(f"| {row['criterion']} | {row['status']} | {evidence} |")
    lines.extend(["", "## PLOS Policy Basis", ""])
    for source in payload["plos_policy_basis"]:
        lines.append(f"- {source['source']}: {source['url']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_metadata()
    json_path = OUT_DIR / "plos_submission_metadata.json"
    md_path = OUT_DIR / "plos_submission_metadata.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown(payload, md_path)
    print(f"Wrote {json_path.relative_to(ROOT)}")
    print(f"Wrote {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
