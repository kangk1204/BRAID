"""Overview summary page with metric cards and pie charts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def render_summary(events_df: "pd.DataFrame", transcripts_df: "pd.DataFrame") -> None:
    """Render the overview summary page.

    Shows metric cards, event type distribution pie chart, PSI histogram,
    and top genes by event count.

    Args:
        events_df: Events DataFrame.
        transcripts_df: Transcripts DataFrame.
    """
    import plotly.express as px
    import streamlit as st

    st.header("Overview")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events", len(events_df))
    col2.metric("Genes with Events", events_df["gene_id"].nunique() if len(events_df) > 0 else 0)
    col3.metric("Total Transcripts", len(transcripts_df))

    valid_psi = events_df["psi"].dropna()
    col4.metric("Median PSI", f"{valid_psi.median():.3f}" if len(valid_psi) > 0 else "N/A")

    # Event type counts
    if len(events_df) > 0:
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Event Type Distribution")
            type_counts = events_df["event_type"].value_counts().reset_index()
            type_counts.columns = ["Event Type", "Count"]
            fig_pie = px.pie(
                type_counts,
                names="Event Type",
                values="Count",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.subheader("PSI Distribution")
            if len(valid_psi) > 0:
                fig_hist = px.histogram(
                    events_df.dropna(subset=["psi"]),
                    x="psi",
                    nbins=30,
                    labels={"psi": "PSI"},
                    color_discrete_sequence=["#636EFA"],
                )
                fig_hist.update_layout(height=350, xaxis_title="PSI", yaxis_title="Count")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No valid PSI values to display.")

        # Top genes by event count
        st.subheader("Top Genes by Event Count")
        gene_counts = (
            events_df.groupby("gene_id")
            .size()
            .reset_index(name="event_count")
            .sort_values("event_count", ascending=False)
            .head(20)
        )
        fig_bar = px.bar(
            gene_counts,
            x="gene_id",
            y="event_count",
            labels={"gene_id": "Gene", "event_count": "Event Count"},
            color_discrete_sequence=["#EF553B"],
        )
        fig_bar.update_layout(height=350, xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No events detected.")
