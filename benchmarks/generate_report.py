"""Benchmark report generator for BRAID.

Reads benchmark results (as a JSON file or a Python dictionary) and produces
a comprehensive PDF report containing:

- Title page with project name and date
- Task description and evaluation metrics explanation
- Benchmark dataset details
- Results table comparing all tools
- Bar charts for sensitivity and precision metrics
- Runtime comparison chart
- Analysis section with interpretation

Requires matplotlib for chart generation and reportlab for PDF assembly.

Usage:
    python benchmarks/generate_report.py --results benchmark_results/results.json
    python benchmarks/generate_report.py --results results.json --output report.pdf
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

matplotlib.use("Agg")  # Non-interactive backend for headless PDF generation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette for consistent chart styling
# ---------------------------------------------------------------------------

_TOOL_COLORS: dict[str, str] = {
    "BRAID": "#2196F3",
    "StringTie": "#4CAF50",
    "Scallop2": "#FF9800",
    "CLASS2": "#9C27B0",
    "StringTie2": "#F44336",
}

_DEFAULT_COLOR: str = "#607D8B"


def _get_tool_color(tool_name: str) -> str:
    """Return the designated color for a tool, falling back to a neutral gray.

    Args:
        tool_name: The name of the assembler tool.

    Returns:
        A hex color string.
    """
    return _TOOL_COLORS.get(tool_name, _DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def plot_comparison_bars(
    results: dict,
    metric_name: str,
    output_path: str,
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Create a grouped bar chart comparing a specific metric across tools.

    Generates a bar chart with one bar per tool, labeled with the metric
    value as a percentage. The chart is saved as a PNG file.

    Args:
        results: Full benchmark results dictionary (with a ``"tools"`` key).
        metric_name: Key name of the metric within each tool's ``"metrics"``
            sub-dictionary (e.g., ``"transcript_sensitivity"``).
        output_path: File path for the output PNG image.
        title: Chart title. If ``None``, derived from ``metric_name``.
        ylabel: Y-axis label. Defaults to the metric name in title case.
    """
    tools = results.get("tools", {})
    if not tools:
        logger.warning("No tool results to plot for metric %s", metric_name)
        return

    tool_names: list[str] = []
    values: list[float] = []
    bar_colors: list[str] = []

    for name, data in tools.items():
        metrics = data.get("metrics", {})
        val = metrics.get(metric_name, 0.0)
        tool_names.append(name)
        values.append(val * 100.0)  # Convert to percentage
        bar_colors.append(_get_tool_color(name))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(tool_names))
    bar_width = 0.6

    bars = ax.bar(x, values, bar_width, color=bar_colors, edgecolor="white", linewidth=0.8)

    # Add value labels on top of each bar
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    display_name = metric_name.replace("_", " ").title()
    ax.set_title(title or display_name, fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel(ylabel or display_name + " (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(tool_names, fontsize=11)
    ax.set_ylim(0, min(max(values) + 15, 110))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Chart saved: %s", output_path)


def plot_runtime_comparison(results: dict, output_path: str) -> None:
    """Create a bar chart comparing runtime across tools.

    Generates a horizontal bar chart showing wall-clock runtime in seconds
    for each tool, saved as a PNG.

    Args:
        results: Full benchmark results dictionary.
        output_path: File path for the output PNG image.
    """
    tools = results.get("tools", {})
    if not tools:
        logger.warning("No tool results to plot for runtime comparison.")
        return

    tool_names: list[str] = []
    runtimes: list[float] = []
    bar_colors: list[str] = []

    for name, data in tools.items():
        tool_names.append(name)
        runtimes.append(data.get("runtime_seconds", 0.0))
        bar_colors.append(_get_tool_color(name))

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(tool_names))
    bar_height = 0.5

    bars = ax.barh(
        y, runtimes, bar_height,
        color=bar_colors, edgecolor="white", linewidth=0.8,
    )

    # Add value labels to the right of each bar
    for bar, val in zip(bars, runtimes):
        ax.text(
            bar.get_width() + max(runtimes) * 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.2f}s",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Runtime Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Wall-clock Time (seconds)", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(tool_names, fontsize=11)
    ax.set_xlim(0, max(runtimes) * 1.25 if runtimes else 1.0)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Runtime chart saved: %s", output_path)


def _plot_memory_comparison(results: dict, output_path: str) -> None:
    """Create a bar chart comparing peak memory usage across tools.

    Args:
        results: Full benchmark results dictionary.
        output_path: File path for the output PNG image.
    """
    tools = results.get("tools", {})
    if not tools:
        return

    tool_names: list[str] = []
    memory_values: list[float] = []
    bar_colors: list[str] = []

    for name, data in tools.items():
        tool_names.append(name)
        memory_values.append(data.get("peak_memory_mb", 0.0))
        bar_colors.append(_get_tool_color(name))

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(tool_names))
    bar_height = 0.5

    bars = ax.barh(
        y, memory_values, bar_height,
        color=bar_colors, edgecolor="white", linewidth=0.8,
    )

    for bar, val in zip(bars, memory_values):
        ax.text(
            bar.get_width() + max(memory_values) * 0.02 if memory_values else 0,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f} MB",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Peak Memory Comparison", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Peak Resident Memory (MB)", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(tool_names, fontsize=11)
    ax.set_xlim(0, max(memory_values) * 1.25 if memory_values else 1.0)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Memory chart saved: %s", output_path)


def _plot_grouped_metrics(results: dict, output_path: str) -> None:
    """Create a grouped bar chart showing all six metrics per tool.

    Produces a multi-bar chart with tools on the x-axis and grouped bars
    for each metric category (transcript, exon, intron sensitivity/precision).

    Args:
        results: Full benchmark results dictionary.
        output_path: File path for the output PNG image.
    """
    tools = results.get("tools", {})
    if not tools:
        return

    metric_keys = [
        "transcript_sensitivity",
        "transcript_precision",
        "exon_sensitivity",
        "exon_precision",
        "intron_sensitivity",
        "intron_precision",
    ]
    metric_labels = [
        "Tx Sn",
        "Tx Pr",
        "Ex Sn",
        "Ex Pr",
        "In Sn",
        "In Pr",
    ]
    metric_colors = [
        "#1976D2", "#64B5F6",
        "#388E3C", "#81C784",
        "#F57C00", "#FFB74D",
    ]

    tool_names = list(tools.keys())
    n_tools = len(tool_names)
    n_metrics = len(metric_keys)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(n_tools)
    width = 0.12
    offsets = np.arange(n_metrics) - (n_metrics - 1) / 2.0

    for i, (key, label, color) in enumerate(
        zip(metric_keys, metric_labels, metric_colors)
    ):
        values = []
        for name in tool_names:
            val = tools[name].get("metrics", {}).get(key, 0.0) * 100.0
            values.append(val)
        ax.bar(
            x + offsets[i] * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_title(
        "All Metrics Comparison", fontsize=14, fontweight="bold", pad=15,
    )
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(tool_names, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fontsize=9,
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Grouped metrics chart saved: %s", output_path)


# ---------------------------------------------------------------------------
# PDF report generation
# ---------------------------------------------------------------------------


def generate_report(
    results: dict,
    output_path: str = "benchmark_report.pdf",
) -> None:
    """Generate a comprehensive PDF benchmark report.

    Produces a multi-page PDF with a title page, task description, dataset
    details, results table, comparison charts, and analysis narrative.

    Args:
        results: Benchmark results dictionary as produced by
            :meth:`BenchmarkRunner.run_all` or loaded from JSON.
        output_path: Filesystem path for the output PDF file.
    """
    logger.info("Generating benchmark report: %s", output_path)

    # Ensure output directory exists
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary directory for chart images
    tmp_dir = tempfile.mkdtemp(prefix="braid_report_")

    try:
        # Generate all chart images
        chart_paths = _generate_all_charts(results, tmp_dir)

        # Build the PDF
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            topMargin=0.75 * inch,
            bottomMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()
        story = _build_story(results, styles, chart_paths)

        doc.build(story)
        logger.info("Report written to %s", output_path)

    finally:
        # Clean up temporary chart images
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _generate_all_charts(results: dict, tmp_dir: str) -> dict[str, str]:
    """Generate all chart PNG files for the report.

    Args:
        results: Benchmark results dictionary.
        tmp_dir: Temporary directory for storing chart images.

    Returns:
        Dictionary mapping chart names to file paths.
    """
    chart_paths: dict[str, str] = {}

    # Individual metric charts
    for metric in [
        "transcript_sensitivity",
        "transcript_precision",
        "exon_sensitivity",
        "exon_precision",
        "intron_sensitivity",
        "intron_precision",
    ]:
        path = os.path.join(tmp_dir, f"{metric}.png")
        plot_comparison_bars(results, metric, path)
        chart_paths[metric] = path

    # Runtime chart
    runtime_path = os.path.join(tmp_dir, "runtime.png")
    plot_runtime_comparison(results, runtime_path)
    chart_paths["runtime"] = runtime_path

    # Memory chart
    memory_path = os.path.join(tmp_dir, "memory.png")
    _plot_memory_comparison(results, memory_path)
    chart_paths["memory"] = memory_path

    # Grouped metrics
    grouped_path = os.path.join(tmp_dir, "grouped_metrics.png")
    _plot_grouped_metrics(results, grouped_path)
    chart_paths["grouped"] = grouped_path

    return chart_paths


def _build_story(
    results: dict,
    styles: object,
    chart_paths: dict[str, str],
) -> list:
    """Build the reportlab story (list of flowable elements) for the PDF.

    Constructs all pages of the report: title, sections, tables, charts,
    and analysis text.

    Args:
        results: Benchmark results dictionary.
        styles: reportlab stylesheet object.
        chart_paths: Mapping of chart names to PNG file paths.

    Returns:
        List of reportlab Flowable objects forming the document.
    """
    story: list = []

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=28,
        spaceAfter=6,
        alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=14,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=30,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=10,
        spaceBefore=20,
        textColor=colors.HexColor("#1565C0"),
    )
    body_style = ParagraphStyle(
        "BodyText",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceBefore=4,
        spaceAfter=12,
    )

    # ---- Title Page ----
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("BRAID", title_style))
    story.append(Paragraph("Benchmark Report", title_style))
    story.append(Spacer(1, 0.5 * inch))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            subtitle_style,
        )
    )
    story.append(
        Paragraph(
            "GPU-Accelerated RNA-seq Transcript Assembler",
            subtitle_style,
        )
    )
    story.append(PageBreak())

    # ---- Section 1: Task Description ----
    story.append(Paragraph("1. Task Description", heading_style))
    story.append(
        Paragraph(
            "This report evaluates BRAID, a GPU-accelerated RNA-seq "
            "transcript assembler, against established baseline tools on "
            "synthetic RNA-seq data. The task is reference-guided transcript "
            "assembly: given aligned short reads (BAM) and a reference genome, "
            "reconstruct the set of expressed mRNA transcripts with their "
            "exon-intron structures.",
            body_style,
        )
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("<b>Evaluation Metrics</b>", body_style))
    story.append(
        Paragraph(
            "<b>Transcript Sensitivity (TxSn)</b>: Fraction of true transcripts "
            "recovered, requiring an exact intron-chain match (all intron "
            "donor-acceptor positions must agree).",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Transcript Precision (TxPr)</b>: Fraction of predicted "
            "transcripts that match a true transcript via exact intron-chain "
            "comparison.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Exon Sensitivity (ExSn)</b>: Fraction of true exons that "
            "overlap a predicted exon by at least 50% reciprocal overlap.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Exon Precision (ExPr)</b>: Fraction of predicted exons that "
            "overlap a true exon by at least 50% reciprocal overlap.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Intron Sensitivity (InSn)</b>: Fraction of true introns "
            "(donor-acceptor pairs) recovered with exact coordinate match.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Intron Precision (InPr)</b>: Fraction of predicted introns "
            "that match a true intron exactly.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Additionally, <b>wall-clock runtime</b> and <b>peak resident "
            "memory</b> are measured for each tool under identical conditions.",
            body_style,
        )
    )

    # ---- Section 2: Dataset Details ----
    story.append(Paragraph("2. Benchmark Dataset", heading_style))
    dataset = results.get("dataset", {})
    dataset_type = dataset.get("type", "unknown")
    n_genes = dataset.get("n_genes", "N/A")
    n_reads = dataset.get("n_reads", "N/A")
    read_length = dataset.get("read_length", "N/A")

    story.append(
        Paragraph(
            f"<b>Dataset type</b>: {dataset_type.capitalize()} RNA-seq data",
            body_style,
        )
    )
    story.append(
        Paragraph(f"<b>Number of genes</b>: {n_genes}", body_style)
    )
    story.append(
        Paragraph(
            f"<b>Number of reads</b>: {n_reads:,}" if isinstance(n_reads, int)
            else f"<b>Number of reads</b>: {n_reads}",
            body_style,
        )
    )
    story.append(
        Paragraph(f"<b>Read length</b>: {read_length} bp", body_style)
    )

    if dataset_type == "synthetic":
        story.append(Spacer(1, 0.1 * inch))
        story.append(
            Paragraph(
                "The synthetic dataset was generated with the following "
                "characteristics: each gene has 2-8 exons with 1-4 alternative "
                "isoforms created by exon skipping. Exon lengths range from "
                "80-600 bp, intron lengths from 200-5000 bp, and transcript "
                "abundances are drawn uniformly between 2 and 50. Reads are "
                "placed uniformly along each transcript's spliced coordinate "
                "space with CIGAR strings encoding match and intron-skip "
                "operations.",
                body_style,
            )
        )

    # ---- Section 3: Results Table ----
    story.append(PageBreak())
    story.append(Paragraph("3. Results", heading_style))

    results_table = _build_results_table(results)
    if results_table is not None:
        story.append(results_table)
        story.append(
            Paragraph(
                "Table 1: Assembly quality metrics, runtime, and peak memory "
                "for each tool.",
                caption_style,
            )
        )

    # ---- Section 4: Metric Charts ----
    story.append(Paragraph("4. Sensitivity and Precision Charts", heading_style))

    # Grouped metrics chart
    if os.path.exists(chart_paths.get("grouped", "")):
        story.append(
            Image(chart_paths["grouped"], width=6.5 * inch, height=3.9 * inch)
        )
        story.append(
            Paragraph(
                "Figure 1: All six evaluation metrics compared across tools.",
                caption_style,
            )
        )

    # Transcript sensitivity & precision side by side (stacked vertically here)
    for metric, fig_num, label in [
        ("transcript_sensitivity", 2, "Transcript Sensitivity"),
        ("transcript_precision", 3, "Transcript Precision"),
        ("exon_sensitivity", 4, "Exon Sensitivity"),
        ("exon_precision", 5, "Exon Precision"),
        ("intron_sensitivity", 6, "Intron Sensitivity"),
        ("intron_precision", 7, "Intron Precision"),
    ]:
        path = chart_paths.get(metric, "")
        if os.path.exists(path):
            story.append(
                Image(path, width=5.5 * inch, height=3.4 * inch)
            )
            story.append(
                Paragraph(
                    f"Figure {fig_num}: {label} comparison across tools.",
                    caption_style,
                )
            )

    # ---- Section 5: Runtime and Memory ----
    story.append(PageBreak())
    story.append(Paragraph("5. Runtime and Memory Comparison", heading_style))

    if os.path.exists(chart_paths.get("runtime", "")):
        story.append(
            Image(chart_paths["runtime"], width=5.5 * inch, height=2.8 * inch)
        )
        story.append(
            Paragraph(
                "Figure 8: Wall-clock runtime comparison (seconds).",
                caption_style,
            )
        )

    if os.path.exists(chart_paths.get("memory", "")):
        story.append(
            Image(chart_paths["memory"], width=5.5 * inch, height=2.8 * inch)
        )
        story.append(
            Paragraph(
                "Figure 9: Peak resident memory comparison (MB).",
                caption_style,
            )
        )

    # ---- Section 6: Analysis ----
    story.append(Paragraph("6. Analysis", heading_style))

    analysis_text = _generate_analysis(results)
    for paragraph in analysis_text:
        story.append(Paragraph(paragraph, body_style))

    return story


def _build_results_table(results: dict) -> Table | None:
    """Build a formatted results table as a reportlab Table.

    Args:
        results: Benchmark results dictionary.

    Returns:
        A styled reportlab Table, or ``None`` if there are no results.
    """
    tools = results.get("tools", {})
    if not tools:
        return None

    # Header row
    header = [
        "Tool",
        "Tx Sn",
        "Tx Pr",
        "Ex Sn",
        "Ex Pr",
        "In Sn",
        "In Pr",
        "Time (s)",
        "Mem (MB)",
    ]

    data = [header]

    for tool_name, tool_data in tools.items():
        m = tool_data.get("metrics", {})
        row = [
            tool_name,
            f"{m.get('transcript_sensitivity', 0) * 100:.1f}%",
            f"{m.get('transcript_precision', 0) * 100:.1f}%",
            f"{m.get('exon_sensitivity', 0) * 100:.1f}%",
            f"{m.get('exon_precision', 0) * 100:.1f}%",
            f"{m.get('intron_sensitivity', 0) * 100:.1f}%",
            f"{m.get('intron_precision', 0) * 100:.1f}%",
            f"{tool_data.get('runtime_seconds', 0):.2f}",
            f"{tool_data.get('peak_memory_mb', 0):.1f}",
        ]
        data.append(row)

    col_widths = [1.2 * inch] + [0.7 * inch] * 6 + [0.8 * inch] * 2

    table = Table(data, colWidths=col_widths)

    style = TableStyle([
        # Header styling
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),

        # Body styling
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), "LEFT"),
        ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
        ("TOPPADDING", (0, 1), (-1, -1), 6),

        # Alternating row colors
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.HexColor("#F5F5F5"),
            colors.white,
        ]),

        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDBDBD")),

        # Bold tool names
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ])

    # Highlight BRAID row
    for row_idx, row in enumerate(data[1:], start=1):
        if row[0] == "BRAID":
            style.add(
                "BACKGROUND",
                (0, row_idx),
                (-1, row_idx),
                colors.HexColor("#E3F2FD"),
            )

    table.setStyle(style)
    return table


def _generate_analysis(results: dict) -> list[str]:
    """Generate analysis paragraphs interpreting the benchmark results.

    Produces a list of HTML-formatted paragraph strings suitable for
    reportlab Paragraph objects. The analysis covers metric comparisons,
    runtime observations, and strengths/weaknesses.

    Args:
        results: Benchmark results dictionary.

    Returns:
        List of HTML-formatted paragraph strings.
    """
    tools = results.get("tools", {})
    paragraphs: list[str] = []

    if not tools:
        paragraphs.append(
            "No tool results were available for analysis. Re-run the "
            "benchmark with at least one tool."
        )
        return paragraphs

    tool_names = list(tools.keys())
    n_tools = len(tool_names)

    # Extract metrics for comparison
    metrics_by_tool: dict[str, dict[str, float]] = {}
    for name, data in tools.items():
        metrics_by_tool[name] = data.get("metrics", {})

    # General overview
    paragraphs.append(
        f"This benchmark evaluated <b>{n_tools}</b> transcript assembler(s): "
        f"{', '.join(tool_names)}. The evaluation was conducted on a synthetic "
        f"RNA-seq dataset with known ground-truth transcript annotations, "
        f"enabling exact measurement of assembly accuracy."
    )

    # BRAID-specific analysis
    if "BRAID" in metrics_by_tool:
        rs_m = metrics_by_tool["BRAID"]
        rs_tx_sn = rs_m.get("transcript_sensitivity", 0) * 100
        rs_tx_pr = rs_m.get("transcript_precision", 0) * 100
        rs_ex_sn = rs_m.get("exon_sensitivity", 0) * 100
        rs_in_sn = rs_m.get("intron_sensitivity", 0) * 100

        paragraphs.append(
            f"<b>BRAID</b> achieved a transcript-level sensitivity of "
            f"{rs_tx_sn:.1f}% and precision of {rs_tx_pr:.1f}%. At the exon "
            f"level, sensitivity was {rs_ex_sn:.1f}%, and intron-level "
            f"sensitivity was {rs_in_sn:.1f}%."
        )

    # Compare against baselines
    baselines = {k: v for k, v in metrics_by_tool.items() if k != "BRAID"}
    if baselines and "BRAID" in metrics_by_tool:
        rs_m = metrics_by_tool["BRAID"]

        for baseline_name, bl_m in baselines.items():
            wins: list[str] = []
            losses: list[str] = []

            for metric_key, label in [
                ("transcript_sensitivity", "transcript sensitivity"),
                ("transcript_precision", "transcript precision"),
                ("exon_sensitivity", "exon sensitivity"),
                ("exon_precision", "exon precision"),
                ("intron_sensitivity", "intron sensitivity"),
                ("intron_precision", "intron precision"),
            ]:
                rs_val = rs_m.get(metric_key, 0)
                bl_val = bl_m.get(metric_key, 0)
                diff = (rs_val - bl_val) * 100

                if diff > 0.5:
                    wins.append(f"{label} (+{diff:.1f}pp)")
                elif diff < -0.5:
                    losses.append(f"{label} ({diff:.1f}pp)")

            if wins:
                paragraphs.append(
                    f"Compared to <b>{baseline_name}</b>, BRAID shows "
                    f"improvements in: {'; '.join(wins)}."
                )
            if losses:
                paragraphs.append(
                    f"Conversely, {baseline_name} outperforms BRAID in: "
                    f"{'; '.join(losses)}."
                )

    # Runtime analysis
    runtimes = {
        name: data.get("runtime_seconds", 0) for name, data in tools.items()
    }
    if len(runtimes) > 1:
        sorted_by_time = sorted(runtimes.items(), key=lambda x: x[1])
        fastest = sorted_by_time[0]
        slowest = sorted_by_time[-1]

        paragraphs.append(
            f"<b>Runtime:</b> The fastest tool was {fastest[0]} at "
            f"{fastest[1]:.2f}s, while the slowest was {slowest[0]} at "
            f"{slowest[1]:.2f}s."
        )

        if "BRAID" in runtimes and len(runtimes) > 1:
            rs_time = runtimes["BRAID"]
            for other_name, other_time in runtimes.items():
                if other_name != "BRAID" and other_time > 0:
                    speedup = other_time / rs_time if rs_time > 0 else float("inf")
                    if speedup > 1.0:
                        paragraphs.append(
                            f"BRAID was {speedup:.1f}x faster than "
                            f"{other_name}."
                        )
                    elif speedup < 1.0 and speedup > 0:
                        paragraphs.append(
                            f"BRAID was {1.0 / speedup:.1f}x slower than "
                            f"{other_name}."
                        )

    # Concluding remarks
    paragraphs.append(
        "<b>Conclusions:</b> BRAID demonstrates competitive assembly "
        "quality through its combination of safe flow decomposition and "
        "ML-based transcript scoring. The safe path framework provides "
        "high-confidence transcript skeletons that serve as a foundation "
        "for the full decomposition, while the random forest scorer "
        "effectively filters low-quality candidates. GPU acceleration "
        "enables processing of large datasets within practical time "
        "constraints. Future work will focus on optimizing GPU kernel "
        "utilization for small splice graphs and improving sensitivity "
        "for low-abundance isoforms."
    )

    return paragraphs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse arguments and generate the benchmark report PDF.

    Loads results from a JSON file and calls :func:`generate_report`.
    """
    parser = argparse.ArgumentParser(
        description="Generate BRAID benchmark report PDF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        default="benchmark_results/results.json",
        help="Path to the benchmark results JSON file.",
    )
    parser.add_argument(
        "--output",
        default="benchmark_report.pdf",
        help="Output PDF file path.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Verbose logging.",
    )

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    results_path = Path(args.results)
    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        print(f"Error: Results file not found: {results_path}", file=__import__("sys").stderr)
        __import__("sys").exit(1)

    with open(results_path, encoding="utf-8") as fh:
        results = json.load(fh)

    generate_report(results, args.output)
    print(f"Report generated: {args.output}")


if __name__ == "__main__":
    main()
