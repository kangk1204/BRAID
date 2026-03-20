"""Assembly statistics and reporting utilities."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AssemblyStats:
    """Aggregate statistics from a transcript assembly run."""

    total_reads: int = 0
    mapped_reads: int = 0
    spliced_reads: int = 0
    total_junctions: int = 0
    unique_junctions: int = 0
    total_loci: int = 0
    assembled_transcripts: int = 0
    multi_exon_transcripts: int = 0
    single_exon_transcripts: int = 0
    filtered_transcripts: int = 0
    elapsed_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    gpu_time_seconds: float = 0.0
    per_stage_times: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a formatted summary string."""
        lines = [
            "=" * 60,
            "RapidSplice Assembly Summary",
            "=" * 60,
            f"  Reads:       {self.total_reads:>12,} total",
            f"               {self.mapped_reads:>12,} mapped",
            f"               {self.spliced_reads:>12,} spliced",
            f"  Junctions:   {self.total_junctions:>12,} total",
            f"               {self.unique_junctions:>12,} unique",
            f"  Loci:        {self.total_loci:>12,}",
            f"  Transcripts: {self.assembled_transcripts:>12,} assembled",
            f"               {self.multi_exon_transcripts:>12,} multi-exon",
            f"               {self.single_exon_transcripts:>12,} single-exon",
            f"               {self.filtered_transcripts:>12,} filtered out",
            f"  Time:        {self.elapsed_seconds:>12.1f}s total",
        ]
        if self.gpu_time_seconds > 0:
            lines.append(f"               {self.gpu_time_seconds:>12.1f}s GPU")
        if self.per_stage_times:
            lines.append("  Stages:")
            for stage, t in self.per_stage_times.items():
                lines.append(f"    {stage:<30s} {t:>8.2f}s")
        lines.append("=" * 60)
        return "\n".join(lines)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, stats: AssemblyStats | None = None) -> None:
        self.name = name
        self.stats = stats
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        logger.info("Starting: %s", self.name)
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger.info("Finished: %s (%.2fs)", self.name, self.elapsed)
        if self.stats is not None:
            self.stats.per_stage_times[self.name] = self.elapsed
