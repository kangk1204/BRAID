"""PSI analysis plots: violin plots, scatter, and histograms."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def render_psi_analysis(events_df: "pd.DataFrame") -> None:
    """Render PSI analysis page with violin plots, scatter, and histograms.

    Args:
        events_df: Events DataFrame with PSI values.
    """
    import plotly.express as px
    import streamlit as st

    st.header("PSI Analysis")

    if len(events_df) == 0:
        st.info("No events to analyze.")
        return

    valid_df = events_df.dropna(subset=["psi"])
    if len(valid_df) == 0:
        st.warning("No events with valid PSI values.")
        return

    # Violin plots by event type
    st.subheader("PSI Distribution by Event Type")
    fig_violin = px.violin(
        valid_df,
        x="event_type",
        y="psi",
        color="event_type",
        box=True,
        points="outliers",
        labels={"event_type": "Event Type", "psi": "PSI"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_violin.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_violin, use_container_width=True)

    # CI width vs PSI scatter
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("CI Width vs PSI")
        if "ci_width" in valid_df.columns:
            fig_scatter = px.scatter(
                valid_df,
                x="psi",
                y="ci_width",
                color="event_type",
                size="total_reads",
                size_max=15,
                labels={
                    "psi": "PSI",
                    "ci_width": "CI Width",
                    "event_type": "Type",
                    "total_reads": "Total Reads",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
                hover_data=["gene_id", "event_id"],
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No CI width data available.")

    with col_right:
        st.subheader("Events per Gene")
        gene_event_counts = (
            events_df.groupby("gene_id")
            .size()
            .reset_index(name="count")
        )
        fig_gene_hist = px.histogram(
            gene_event_counts,
            x="count",
            nbins=20,
            labels={"count": "Number of Events per Gene"},
            color_discrete_sequence=["#AB63FA"],
        )
        fig_gene_hist.update_layout(height=400, yaxis_title="Number of Genes")
        st.plotly_chart(fig_gene_hist, use_container_width=True)

    # Confidence score distribution
    if "confidence_score" in valid_df.columns:
        st.subheader("Confidence Score Distribution")
        fig_conf = px.histogram(
            valid_df,
            x="confidence_score",
            color="event_type",
            nbins=30,
            barmode="overlay",
            opacity=0.7,
            labels={"confidence_score": "Confidence Score", "event_type": "Type"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_conf.update_layout(height=350)
        st.plotly_chart(fig_conf, use_container_width=True)

    # Summary statistics table
    st.subheader("Summary Statistics by Event Type")
    agg_dict: dict[str, tuple[str, str]] = {
        "count": ("psi", "count"),
        "mean_psi": ("psi", "mean"),
        "median_psi": ("psi", "median"),
        "mean_reads": ("total_reads", "mean"),
    }
    if "ci_width" in valid_df.columns:
        agg_dict["mean_ci_width"] = ("ci_width", "mean")
    summary = (
        valid_df.groupby("event_type")
        .agg(**agg_dict)
        .round(3)
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)
