"""Filterable and sortable AS event table component."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def render_event_table(events_df: "pd.DataFrame") -> None:
    """Render a filterable, sortable event explorer table.

    Provides multi-select for event type, PSI range slider, minimum reads
    filter, and gene search.

    Args:
        events_df: Events DataFrame.
    """
    import streamlit as st

    st.header("Event Explorer")

    if len(events_df) == 0:
        st.info("No events to display.")
        return

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        event_types = sorted(events_df["event_type"].unique().tolist())
        selected_types = st.multiselect(
            "Event Types",
            options=event_types,
            default=event_types,
        )

    with col2:
        psi_range = st.slider(
            "PSI Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05,
        )

    with col3:
        min_reads = st.number_input(
            "Min Total Reads",
            min_value=0,
            value=0,
            step=1,
        )

    # Gene search
    gene_search = st.text_input("Search Gene ID", "")

    # Apply filters
    filtered = events_df.copy()

    if selected_types:
        filtered = filtered[filtered["event_type"].isin(selected_types)]

    if "psi" in filtered.columns:
        valid_psi = filtered["psi"].notna()
        filtered = filtered[
            (~valid_psi) |
            ((filtered["psi"] >= psi_range[0]) & (filtered["psi"] <= psi_range[1]))
        ]

    if min_reads > 0 and "total_reads" in filtered.columns:
        filtered = filtered[filtered["total_reads"] >= min_reads]

    if gene_search:
        filtered = filtered[
            filtered["gene_id"].str.contains(gene_search, case=False, na=False)
        ]

    # Display
    st.write(f"Showing {len(filtered)} of {len(events_df)} events")

    display_cols = [
        c for c in [
            "event_id", "event_type", "gene_id", "chrom", "strand",
            "psi", "total_reads", "inclusion_count", "exclusion_count",
            "ci_low", "ci_high", "confidence_score",
        ]
        if c in filtered.columns
    ]

    st.dataframe(
        filtered[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=min(600, 35 * len(filtered) + 38),
    )

    # Download filtered data
    if len(filtered) > 0:
        csv_data = filtered.to_csv(index=False, sep="\t")
        st.download_button(
            "Download Filtered Events (TSV)",
            data=csv_data,
            file_name="filtered_events.tsv",
            mime="text/tab-separated-values",
        )
