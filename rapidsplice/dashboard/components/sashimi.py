"""Sashimi plot component with coverage track and junction arcs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def _draw_arc(
    x_start: float,
    x_end: float,
    height: float,
    n_points: int = 50,
) -> tuple[list[float], list[float]]:
    """Generate x, y coordinates for a parabolic arc.

    Args:
        x_start: Arc start x position.
        x_end: Arc end x position.
        height: Peak height of the arc.
        n_points: Number of interpolation points.

    Returns:
        Tuple of (x_coords, y_coords) lists.
    """
    import numpy as np

    t = np.linspace(0, 1, n_points)
    x = x_start + (x_end - x_start) * t
    y = 4 * height * t * (1 - t)
    return x.tolist(), y.tolist()


def render_sashimi(
    events_df: "pd.DataFrame",
    transcripts_df: "pd.DataFrame",
    bam_path: str | None = None,
) -> None:
    """Render Sashimi plot page with coverage and junction arcs.

    Args:
        events_df: Events DataFrame.
        transcripts_df: Transcripts DataFrame.
        bam_path: Optional path to BAM file for coverage data.
    """
    import plotly.graph_objects as go
    import streamlit as st

    st.header("Sashimi Plots")

    if len(transcripts_df) == 0:
        st.warning("No transcripts loaded.")
        return

    # Gene selector
    gene_ids = sorted(transcripts_df["gene_id"].unique().tolist())
    selected_gene = st.selectbox("Select Gene", gene_ids, key="sashimi_gene")

    gene_txs = transcripts_df[transcripts_df["gene_id"] == selected_gene]
    if len(events_df) > 0:
        gene_events = events_df[events_df["gene_id"] == selected_gene]
    else:
        gene_events = events_df

    if len(gene_txs) == 0:
        st.info(f"No transcripts for gene {selected_gene}.")
        return

    # Determine region
    region_start = int(gene_txs["start"].min())
    region_end = int(gene_txs["end"].max())

    fig = go.Figure()

    # Coverage track (from BAM if available)
    coverage_y_offset = 0
    if bam_path is not None:
        try:
            import numpy as np
            import pysam

            chrom = gene_txs.iloc[0]["chrom"]
            with pysam.AlignmentFile(bam_path, "rb") as af:
                coverage = np.zeros(region_end - region_start, dtype=np.int32)
                for col in af.pileup(chrom, region_start, region_end, truncate=True):
                    pos = col.reference_pos - region_start
                    if 0 <= pos < len(coverage):
                        coverage[pos] = col.nsegments

            positions = list(range(region_start, region_end))
            fig.add_trace(go.Scatter(
                x=positions,
                y=coverage.tolist(),
                fill="tozeroy",
                fillcolor="rgba(99,110,250,0.3)",
                line={"color": "#636EFA", "width": 1},
                name="Coverage",
            ))
            coverage_y_offset = max(coverage) if len(coverage) > 0 else 0
        except Exception:
            st.info("Could not load BAM coverage. Showing junctions only.")

    # Junction arcs from events
    junctions_drawn: set[tuple[int, int]] = set()

    if len(gene_events) > 0:
        for _, event in gene_events.iterrows():
            # Parse junction strings
            for junc_col, color, label_prefix in [
                ("inclusion_junctions", "#00CC96", "Inc"),
                ("exclusion_junctions", "#EF553B", "Exc"),
            ]:
                junc_str = event.get(junc_col, "")
                if not isinstance(junc_str, str) or not junc_str:
                    continue

                for junc in junc_str.split(";"):
                    if "-" not in junc:
                        continue
                    parts = junc.split("-")
                    if len(parts) != 2:
                        continue
                    try:
                        j_start = int(parts[0])
                        j_end = int(parts[1])
                    except ValueError:
                        continue

                    if (j_start, j_end) in junctions_drawn:
                        continue
                    junctions_drawn.add((j_start, j_end))

                    arc_height = max(
                        10,
                        (j_end - j_start) * 0.15 + coverage_y_offset * 0.3,
                    )
                    arc_x, arc_y = _draw_arc(j_start, j_end, arc_height)
                    # Offset above coverage
                    arc_y = [y + coverage_y_offset * 0.1 for y in arc_y]

                    reads = event.get("total_reads", "?")
                    fig.add_trace(go.Scatter(
                        x=arc_x,
                        y=arc_y,
                        mode="lines",
                        line={"color": color, "width": 2},
                        name=f"{label_prefix} {j_start}-{j_end}",
                        hovertext=f"{label_prefix}: {j_start}-{j_end} ({reads} reads)",
                    ))

    # Gene model at bottom
    y_model = -max(10, coverage_y_offset * 0.15)
    for _, tx in gene_txs.iterrows():
        exon_starts = tx["exon_starts"]
        exon_ends = tx["exon_ends"]
        if not exon_starts:
            continue

        fig.add_trace(go.Scatter(
            x=[tx["start"], tx["end"]],
            y=[y_model, y_model],
            mode="lines",
            line={"color": "#888", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        ))

        for es, ee in zip(exon_starts, exon_ends):
            fig.add_shape(
                type="rect",
                x0=es, x1=ee,
                y0=y_model - 3, y1=y_model + 3,
                fillcolor="#2196F3",
                line={"color": "#1565C0"},
            )
        y_model -= 8

    fig.update_layout(
        height=500,
        xaxis_title="Genomic Position",
        yaxis_title="Coverage / Junction Reads",
        margin={"l": 60, "r": 20, "t": 30, "b": 50},
        plot_bgcolor="white",
        showlegend=True,
        legend={"orientation": "h", "y": 1.05},
    )

    st.plotly_chart(fig, use_container_width=True)

    # Export button
    if st.button("Export as PNG"):
        try:

            img_bytes = fig.to_image(format="png", width=1200, height=500)
            st.download_button(
                "Download PNG",
                data=img_bytes,
                file_name=f"sashimi_{selected_gene}.png",
                mime="image/png",
            )
        except Exception:
            st.warning("Install kaleido for image export: pip install kaleido")
