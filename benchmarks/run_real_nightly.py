"""Preset wrapper for full-BAM nightly RapidSplice real-data runs."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.run_real_variant_matrix import main

if __name__ == "__main__":
    main(default_mode="nightly")
