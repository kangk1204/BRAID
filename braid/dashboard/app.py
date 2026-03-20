"""Multi-page Streamlit dashboard for RapidSplice AS event analysis.

Pages:
1. Overview - metric cards, event type pie chart, PSI histogram
2. Gene Browser - transcript tracks with AS event overlays
3. Sashimi Plots - coverage + junction arcs
4. Event Explorer - filterable/sortable event table
5. PSI Analysis - violin plots, scatter, histograms
"""

from __future__ import annotations

import argparse


def parse_dashboard_args() -> argparse.Namespace:
    """Parse dashboard-specific command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="RapidSplice Dashboard")
    parser.add_argument("--events-tsv", required=True, help="Events TSV file.")
    parser.add_argument("--gtf", required=True, help="GTF file.")
    parser.add_argument("--bam", default=None, help="Optional BAM file.")
    return parser.parse_args()


def main() -> None:
    """Run the Streamlit dashboard application."""
    import streamlit as st

    st.set_page_config(
        page_title="RapidSplice Dashboard",
        page_icon=":dna:",
        layout="wide",
    )

    # Parse arguments
    args = parse_dashboard_args()

    # Load data
    from braid.dashboard.data_loader import load_events, load_gtf_transcripts

    @st.cache_data
    def _load_events(path: str):  # type: ignore[no-untyped-def]
        return load_events(path)

    @st.cache_data
    def _load_transcripts(path: str):  # type: ignore[no-untyped-def]
        return load_gtf_transcripts(path)

    events_df = _load_events(args.events_tsv)
    transcripts_df = _load_transcripts(args.gtf)

    # Sidebar navigation
    st.sidebar.title("RapidSplice")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigate",
        ["Overview", "Gene Browser", "Sashimi Plots", "Event Explorer", "PSI Analysis"],
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"**Events:** {len(events_df)}\n\n"
        f"**Transcripts:** {len(transcripts_df)}\n\n"
        f"**Genes:** {events_df['gene_id'].nunique() if len(events_df) > 0 else 0}"
    )

    # Page dispatch
    if page == "Overview":
        from braid.dashboard.components.summary import render_summary
        render_summary(events_df, transcripts_df)

    elif page == "Gene Browser":
        from braid.dashboard.components.gene_browser import render_gene_browser
        render_gene_browser(events_df, transcripts_df)

    elif page == "Sashimi Plots":
        from braid.dashboard.components.sashimi import render_sashimi
        render_sashimi(events_df, transcripts_df, bam_path=args.bam)

    elif page == "Event Explorer":
        from braid.dashboard.components.event_table import render_event_table
        render_event_table(events_df)

    elif page == "PSI Analysis":
        from braid.dashboard.components.psi_plots import render_psi_analysis
        render_psi_analysis(events_df)


if __name__ == "__main__":
    main()
