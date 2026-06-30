#!/usr/bin/env python3
"""Lift QKI validation target tables from hg19 to hg38."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Iterable

try:
    from pyliftover import LiftOver
except ModuleNotFoundError as exc:  # pragma: no cover - user-facing import guard.
    raise SystemExit(
        "pyliftover is required for QKI target liftover. "
        "Install the BRAID benchmark extra with `python -m pip install -e \".[benchmark]\"` "
        "or `python -m pip install -r requirements-benchmark.txt`."
    ) from exc


DEFAULT_CHAIN = "real_benchmark/reference/liftover/hg19ToHg38.over.chain.gz"
DEFAULT_QKI_DIR = "real_benchmark/rtpcr_benchmark/qki"


def _add_chr_prefix(chrom: str) -> str:
    """Return one UCSC-style chromosome label."""
    if chrom.startswith("chr"):
        return chrom
    return f"chr{chrom}"


def _strip_chr_prefix(chrom: str) -> str:
    """Return one repo-style chromosome label."""
    return chrom[3:] if chrom.startswith("chr") else chrom


def _best_interval_mapping(
    lifter: LiftOver,
    chrom: str,
    start_1based: int,
    end_1based: int,
) -> tuple[str, int, int, str] | None:
    """Lift one 1-based inclusive exon interval to 0-based half-open hg38."""
    source_chrom = _add_chr_prefix(chrom)
    start_hits = lifter.convert_coordinate(source_chrom, start_1based - 1)
    end_hits = lifter.convert_coordinate(source_chrom, end_1based - 1)
    if not start_hits or not end_hits:
        return None

    source_len = end_1based - start_1based + 1
    best = None
    for start_hit in start_hits:
        for end_hit in end_hits:
            if start_hit[0] != end_hit[0] or start_hit[2] != end_hit[2]:
                continue
            target_start = min(start_hit[1], end_hit[1])
            target_end = max(start_hit[1], end_hit[1]) + 1
            length_delta = abs((target_end - target_start) - source_len)
            # Prefer near-length-preserving mappings with the strongest chain score.
            candidate = (
                length_delta,
                -(start_hit[3] + end_hit[3]),
                _strip_chr_prefix(start_hit[0]),
                target_start,
                target_end,
                start_hit[2],
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return None
    return best[2], best[3], best[4], best[5]


def _lift_rows(
    rows: Iterable[dict[str, str]],
    lifter: LiftOver,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Lift one TSV payload and split successes from failures."""
    lifted_rows: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []

    for row in rows:
        chrom = row.get("chrom", "")
        start_text = row.get("exon_start")
        end_text = row.get("exon_end")
        if not chrom or start_text is None or end_text is None:
            failed_rows.append(dict(row, liftover_status="missing_coordinates"))
            continue

        lifted = _best_interval_mapping(
            lifter,
            chrom=chrom,
            start_1based=int(start_text),
            end_1based=int(end_text),
        )
        if lifted is None:
            failed_rows.append(dict(row, liftover_status="unmapped"))
            continue

        target_chrom, target_start, target_end, target_strand = lifted
        lifted_row = dict(row)
        lifted_row["chrom"] = target_chrom
        lifted_row["exon_start"] = str(target_start)
        lifted_row["exon_end"] = str(target_end)
        lifted_row["source_build"] = "hg19"
        lifted_row["target_build"] = "hg38"
        lifted_row["source_coord_system"] = "1-based_inclusive"
        lifted_row["target_coord_system"] = "0-based_start_1-based_end"
        lifted_row["source_chrom"] = chrom
        lifted_row["source_exon_start"] = start_text
        lifted_row["source_exon_end"] = end_text
        lifted_row["lifted_strand"] = target_strand
        lifted_row["liftover_status"] = "mapped"
        lifted_rows.append(lifted_row)

    return lifted_rows, failed_rows


def _write_tsv(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    """Write one TSV table."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def _process_table(
    *,
    input_path: str,
    output_path: str,
    unmapped_path: str,
    lifter: LiftOver,
) -> dict:
    """Lift one input target table and persist results."""
    with open(input_path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        source_rows = list(reader)
        source_fields = list(reader.fieldnames or [])

    extra_fields = [
        "source_build",
        "target_build",
        "source_coord_system",
        "target_coord_system",
        "source_chrom",
        "source_exon_start",
        "source_exon_end",
        "lifted_strand",
        "liftover_status",
    ]
    fieldnames = source_fields + [
        field
        for field in extra_fields
        if field not in source_fields
    ]

    lifted_rows, failed_rows = _lift_rows(source_rows, lifter)
    _write_tsv(output_path, lifted_rows, fieldnames)
    _write_tsv(
        unmapped_path,
        failed_rows,
        source_fields + ["liftover_status"],
    )
    return {
        "input_path": input_path,
        "output_path": output_path,
        "unmapped_path": unmapped_path,
        "source_rows": len(source_rows),
        "lifted_rows": len(lifted_rows),
        "unmapped_rows": len(failed_rows),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--qki-dir", default=DEFAULT_QKI_DIR)
    parser.add_argument("--chain", default=DEFAULT_CHAIN)
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Optional summary JSON path. Defaults to <qki-dir>/qki_target_liftover_summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    qki_dir = args.qki_dir
    summary_path = (
        args.summary_json
        or os.path.join(qki_dir, "qki_target_liftover_summary.json")
    )
    lifter = LiftOver(args.chain)

    validated_summary = _process_table(
        input_path=os.path.join(qki_dir, "validated_events.tsv"),
        output_path=os.path.join(qki_dir, "validated_events.hg38.tsv"),
        unmapped_path=os.path.join(qki_dir, "validated_events.hg38.unmapped.tsv"),
        lifter=lifter,
    )
    failed_summary = _process_table(
        input_path=os.path.join(qki_dir, "failed_events.tsv"),
        output_path=os.path.join(qki_dir, "failed_events.hg38.tsv"),
        unmapped_path=os.path.join(qki_dir, "failed_events.hg38.unmapped.tsv"),
        lifter=lifter,
    )

    summary = {
        "chain_path": args.chain,
        "source_build": "hg19",
        "target_build": "hg38",
        "source_coord_system": "1-based_inclusive",
        "target_coord_system": "0-based_start_1-based_end",
        "validated": validated_summary,
        "failed": failed_summary,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(
        "Lifted QKI targets:",
        f"validated {validated_summary['lifted_rows']}/{validated_summary['source_rows']},",
        f"failed {failed_summary['lifted_rows']}/{failed_summary['source_rows']}",
    )
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
