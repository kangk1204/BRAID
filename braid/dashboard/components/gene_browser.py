"""Gene browser page with transcript tracks and AS event overlays."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# Event type colors
EVENT_COLORS: dict[str, str] = {
    "SE": "#e41a1c",
    "A5SS": "#377eb8",
    "A3SS": "#4daf4a",
    "MXE": "#984ea3",
    "RI": "#ff7f00",
    "AFE": "#a65628",
    "ALE": "#f781bf",
}


def render_gene_browser(
    events_df: "pd.DataFrame",
    transcripts_df: "pd.DataFrame",
) -> None:
    """Render the gene browser page with transcript models and event overlays.

    Args:
        events_df: Events DataFrame.
        transcripts_df: Transcripts DataFrame.
    """
    import plotly.graph_objects as go
    import streamlit as st

    st.header("Gene Browser")

    if len(transcripts_df) == 0:
        st.warning("No transcripts loaded.")
        return

    # Gene selector
    gene_ids = sorted(transcripts_df["gene_id"].unique().tolist())
    selected_gene = st.selectbox("Select Gene", gene_ids, index=0)

    gene_txs = transcripts_df[transcripts_df["gene_id"] == selected_gene]
    if len(events_df) > 0:
        gene_events = events_df[events_df["gene_id"] == selected_gene]
    else:
        gene_events = events_df

    if len(gene_txs) == 0:
        st.info(f"No transcripts for gene {selected_gene}.")
        return

    # Build transcript model figure
    fig = go.Figure()

    # Draw transcripts
    for idx, (_, tx) in enumerate(gene_txs.iterrows()):
        y_pos = -(idx + 1)
        exon_starts = tx["exon_starts"]
        exon_ends = tx["exon_ends"]

        if not exon_starts:
            continue

        # Draw intron lines
        fig.add_trace(go.Scatter(
            x=[tx["start"], tx["end"]],
            y=[y_pos, y_pos],
            mode="lines",
            line={"color": "#888", "width": 1},
            showlegend=False,
            hoverinfo="skip",
        ))

        # Draw exon boxes
        for es, ee in zip(exon_starts, exon_ends):
            fig.add_shape(
                type="rect",
                x0=es, x1=ee,
                y0=y_pos - 0.3, y1=y_pos + 0.3,
                fillcolor="#2196F3",
                line={"color": "#1565C0", "width": 1},
            )

        # Label
        fig.add_annotation(
            x=tx["start"] - 50,
            y=y_pos,
            text=tx["transcript_id"],
            showarrow=False,
            xanchor="right",
            font={"size": 10},
        )

    # Overlay AS events
    if len(gene_events) > 0:
        for _, event in gene_events.iterrows():
            et = event["event_type"]
            color = EVENT_COLORS.get(et, "#999")

            # Parse coordinates
            coords_str = event.get("coordinates", "")
            if isinstance(coords_str, str) and coords_str:
                coord_pairs = coords_str.split(";")
                for pair in coord_pairs:
                    if "=" in pair:
                        key, val = pair.split("=", 1)
                        try:
                            pos = int(val)
                            fig.add_vline(
                                x=pos,
                                line_dash="dot",
                                line_color=color,
                                opacity=0.5,
                            )
                        except ValueError:
                            pass

    n_txs = len(gene_txs)
    fig.update_layout(
        height=max(200, 50 * n_txs + 100),
        xaxis_title="Genomic Position",
        yaxis={"visible": False, "range": [-(n_txs + 1), 0.5]},
        margin={"l": 150, "r": 20, "t": 30, "b": 50},
        plot_bgcolor="white",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Events table for this gene
    if len(gene_events) > 0:
        st.subheader(f"Events for {selected_gene}")
        display_cols = [c for c in ["event_id", "event_type", "psi", "total_reads",
                                     "ci_low", "ci_high", "confidence_score"]
                        if c in gene_events.columns]
        st.dataframe(gene_events[display_cols], use_container_width=True)
