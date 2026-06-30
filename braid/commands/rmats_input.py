"""Shared rMATS input validation for CLI commands."""

from __future__ import annotations

import logging
import os

from braid.target.rmats_bootstrap import (
    DEFAULT_RMATS_EVENT_TYPES,
    RMATS_TABLE_SUFFIXES,
    find_rmats_tables,
)


def require_rmats_tables(rmats_dir: str, logger: logging.Logger) -> None:
    """Fail the CLI early when *rmats_dir* cannot feed an rMATS-based command."""
    if not os.path.isdir(rmats_dir):
        logger.error("rMATS directory not found: %s", rmats_dir)
        raise SystemExit(1)

    if find_rmats_tables(rmats_dir):
        return

    examples = ", ".join(
        f"{event}.{suffix}"
        for event in DEFAULT_RMATS_EVENT_TYPES[:2]
        for suffix in RMATS_TABLE_SUFFIXES[:2]
    )
    logger.error(
        "No supported rMATS event tables found in %s. Expected files such as %s.",
        rmats_dir,
        examples,
    )
    raise SystemExit(1)
