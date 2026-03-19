"""BRAID Interactive Dashboard — Browse all genes genome-wide.

Loads pre-computed bootstrap results and lets users browse any gene
with interactive sashimi plots, confidence tables, and PSI charts.

Usage:
    # First: run genome-wide bootstrap
    python benchmarks/bootstrap_benchmark.py

    # Then: launch dashboard
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)


# ─── Page Config ───
st.set_page_config(
    page_title="BRAID Dashboard",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 BRAID: Isoform Confidence Dashboard")
st.markdown("*Browse bootstrap confidence for all assembled isoforms*")


# ─── Color helper ───
def cv_color(cv: float) -> str:
    """Map CV to traffic-light color."""
    if cv <= 0.1:
        return "#2ecc71"
    if cv <= 0.3:
        return "#f39c12"
    return "#e74c3c"


# ─── Load Data ───
@st.cache_resource(show_spinner="Loading bootstrap results...")
def load_bootstrap_results(path: str):
    """Load pre-computed genome-wide bootstrap results."""
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner="Loading annotation index...")
def load_gene_names(gtf_path: str) -> dict[str, tuple[str, int, int]]:
    """Build gene_name → (chrom, start, end) mapping from GTF."""
    genes: dict[str, tuple[str, int, int]] = {}
    with open(gtf_path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = fields[8]
            gname = None
            for part in attrs.split(";"):
                part = part.strip()
                if part.startswith("gene_name"):
                    gname = part.split('"')[1]
                    break
            if gname:
                chrom = fields[0]
                start = int(fields[3]) - 1
                end = int(fields[4])
                genes[gname] = (chrom, start, end)
    return genes


# ─── Sidebar ───
st.sidebar.header("📂 Data")

pkl_path = st.sidebar.text_input(
    "Bootstrap results (.pkl)",
    value="benchmarks/results/k562_bootstrap_results.pkl",
)
gtf_path = st.sidebar.text_input(
    "Annotation GTF",
    value="real_benchmark/annotation/gencode.v38.nochr.gtf",
)
bam_path = st.sidebar.text_input(
    "BAM (for PSI)", value="real_benchmark/bam/SRR387661.bam",
)

if not os.path.exists(pkl_path):
    st.error(f"File not found: {pkl_path}")
    st.info("Run `python benchmarks/bootstrap_benchmark.py` first.")
    st.stop()

results = load_bootstrap_results(pkl_path)
gene_names = load_gene_names(gtf_path) if os.path.exists(gtf_path) else {}

# Build gene lookup from bootstrap results
gene_lookup: dict[str, int] = {}
for i, gr in enumerate(results):
    gene_lookup[gr.gene_id] = i

# Match StringTie gene IDs to gene names
st_to_name: dict[str, str] = {}
name_to_st: dict[str, str] = {}
for gr in results:
    if not gr.isoforms:
        continue
    gs = min(e[0] for iso in gr.isoforms for e in iso.exons)
    ge = max(e[1] for iso in gr.isoforms for e in iso.exons)
    for gname, (chrom, start, end) in gene_names.items():
        if chrom == gr.chrom and start < ge and end > gs:
            st_to_name[gr.gene_id] = gname
            name_to_st[gname] = gr.gene_id
            break

# ─── Gene Selection ───
st.sidebar.header("🔍 Gene Selection")
st.sidebar.markdown(
    f"**{len(results):,}** genes, **{sum(len(r.isoforms) for r in results):,}** isoforms",
)

# Search by gene name
search_names = sorted(name_to_st.keys())
selected_gene = st.sidebar.selectbox(
    "Search gene name",
    options=[""] + search_names,
    index=0,
    help="Type to search (e.g. TP53, BRCA1, EZH2)",
)

cv_threshold = st.sidebar.slider("CV threshold", 0.01, 1.0, 0.1, 0.01)

if not selected_gene:
    # Show overview
    st.header("📊 Genome-Wide Overview")

    total_iso = sum(len(r.isoforms) for r in results)
    n_stable = sum(
        sum(1 for iso in r.isoforms if iso.is_stable)
        for r in results
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Genes", f"{len(results):,}")
    col2.metric("Isoforms", f"{total_iso:,}")
    col3.metric("Stable (pres≥50%)", f"{n_stable:,}")
    col4.metric("Named genes", f"{len(name_to_st):,}")

    # CV distribution
    all_cvs = [
        iso.cv for r in results for iso in r.isoforms
        if not np.isnan(iso.cv)
    ]
    fig_hist = go.Figure(go.Histogram(
        x=all_cvs, nbinsx=100,
        marker_color="#3498db",
    ))
    fig_hist.add_vline(x=cv_threshold, line_dash="dash", line_color="red",
                       annotation_text=f"CV={cv_threshold}")
    fig_hist.update_layout(
        title="Bootstrap CV Distribution (all isoforms)",
        xaxis_title="CV", yaxis_title="Count",
        height=400,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Top genes by isoform count
    st.subheader("Top Genes by Isoform Count")
    gene_data = []
    for gr in sorted(results, key=lambda r: len(r.isoforms), reverse=True)[:30]:
        gname = st_to_name.get(gr.gene_id, gr.gene_id)
        n_conf = sum(1 for iso in gr.isoforms if iso.cv <= cv_threshold)
        gene_data.append({
            "Gene": gname,
            "Isoforms": len(gr.isoforms),
            f"CV≤{cv_threshold}": n_conf,
            "Chrom": gr.chrom,
        })
    if gene_data:
        import pandas as pd
        st.dataframe(pd.DataFrame(gene_data), use_container_width=True)

    st.info("👈 Select a gene from the sidebar to see detailed plots.")
    st.stop()


# ─── Gene Detail View ───
st_gid = name_to_st.get(selected_gene)
if st_gid is None or st_gid not in gene_lookup:
    st.error(f"Gene '{selected_gene}' not found in bootstrap results.")
    st.stop()

gr = results[gene_lookup[st_gid]]
isoforms = gr.isoforms

if not isoforms:
    st.warning(f"No multi-exon isoforms for {selected_gene}.")
    st.stop()

st.header(f"🧬 {selected_gene} ({gr.chrom})")

col1, col2, col3 = st.columns(3)
col1.metric("Isoforms", len(isoforms))
col2.metric(f"CV ≤ {cv_threshold}", sum(1 for i in isoforms if i.cv <= cv_threshold))
col3.metric("Junctions", gr.n_junctions)

tab1, tab2, tab3 = st.tabs(["🧬 Sashimi Plot", "📊 Isoform Table", "🔀 PSI Events"])

# ─── Sashimi Plot ───
with tab1:
    n_iso = len(isoforms)
    fig = make_subplots(
        rows=n_iso + 1, cols=1, shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.3] + [0.7 / max(n_iso, 1)] * n_iso,
        subplot_titles=["Junction Arcs"] + [
            f"{iso.transcript_id} — CV={iso.cv:.3f} "
            f"({'✓ ' + (iso.classification or '') if iso.cv <= cv_threshold else '?'})"
            for iso in isoforms
        ],
    )

    # Junction arcs
    junctions = set()
    for iso in isoforms:
        for i in range(len(iso.exons) - 1):
            junctions.add((iso.exons[i][1], iso.exons[i + 1][0]))

    for d, a in sorted(junctions):
        x_arc = np.linspace(d, a, 50)
        h = (a - d) / 3000
        y_arc = h * np.sin(np.pi * (x_arc - d) / (a - d))
        fig.add_trace(
            go.Scatter(
                x=x_arc, y=y_arc, mode="lines",
                line=dict(color="#3498db", width=2),
                showlegend=False,
                hovertext=f"{d:,} → {a:,} ({a - d:,} bp)",
            ),
            row=1, col=1,
        )

    # Exon tracks
    for idx, iso in enumerate(isoforms):
        row = idx + 2
        color = cv_color(iso.cv)
        for es, ee in iso.exons:
            fig.add_trace(
                go.Scatter(
                    x=[es, ee, ee, es, es],
                    y=[0, 0, 1, 1, 0],
                    fill="toself", fillcolor=color,
                    line=dict(color="black", width=1),
                    showlegend=False,
                    hovertext=(
                        f"Exon: {es:,}–{ee:,} ({ee - es} bp)<br>"
                        f"Weight: {iso.nnls_weight:.1f}<br>"
                        f"CI: [{iso.ci_low:.1f}, {iso.ci_high:.1f}]<br>"
                        f"CV: {iso.cv:.3f}<br>"
                        f"Presence: {iso.presence_rate:.0%}"
                    ),
                ),
                row=row, col=1,
            )
        for i in range(len(iso.exons) - 1):
            fig.add_trace(
                go.Scatter(
                    x=[iso.exons[i][1], iso.exons[i + 1][0]],
                    y=[0.5, 0.5],
                    mode="lines",
                    line=dict(color=color, width=1, dash="dash"),
                    showlegend=False,
                ),
                row=row, col=1,
            )

    fig.update_layout(
        height=200 + 80 * n_iso,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "🟢 CV ≤ 0.1 &nbsp; 🟠 0.1 < CV ≤ 0.3 &nbsp; 🔴 CV > 0.3",
    )

# ─── Isoform Table ───
with tab2:
    import pandas as pd

    data = []
    for iso in isoforms:
        data.append({
            "ID": iso.transcript_id,
            "Exons": iso.n_exons,
            "Weight": round(iso.nnls_weight, 1),
            "CI Low": round(iso.ci_low, 1),
            "CI High": round(iso.ci_high, 1),
            "Presence": f"{iso.presence_rate:.0%}",
            "CV": round(iso.cv, 3),
            "Stable": "✓" if iso.is_stable else "",
            "StringTie Cov": round(iso.stringtie_cov, 1),
        })

    df = pd.DataFrame(data)
    st.dataframe(
        df.style.background_gradient(subset=["CV"], cmap="RdYlGn_r"),
        use_container_width=True,
    )

    # Abundance bar chart
    fig_bar = go.Figure()
    for iso in isoforms:
        fig_bar.add_trace(go.Bar(
            x=[iso.transcript_id],
            y=[iso.nnls_weight],
            error_y=dict(
                type="data", symmetric=False,
                array=[iso.ci_high - iso.nnls_weight],
                arrayminus=[iso.nnls_weight - iso.ci_low],
            ),
            marker_color=cv_color(iso.cv),
            hovertext=(
                f"Weight: {iso.nnls_weight:.1f}<br>"
                f"CI: [{iso.ci_low:.1f}, {iso.ci_high:.1f}]<br>"
                f"CV: {iso.cv:.3f}"
            ),
        ))
    fig_bar.update_layout(
        title="Isoform Abundance + 95% Bootstrap CI",
        yaxis_title="NNLS Weight",
        showlegend=False, height=400,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ─── PSI Events ───
with tab3:
    gene_info = gene_names.get(selected_gene)
    if gene_info and os.path.exists(bam_path):
        from braid.target.psi_bootstrap import compute_psi_from_junctions

        chrom, start, end = gene_info
        try:
            psi_results = compute_psi_from_junctions(
                bam_path, chrom, start, end, 500, 0.95, 42,
            )
        except Exception:
            psi_results = []

        if psi_results:
            sorted_psi = sorted(
                psi_results, key=lambda r: r.psi, reverse=True,
            )
            fig_psi = go.Figure()
            for r in sorted_psi:
                fig_psi.add_trace(go.Bar(
                    x=[r.event_id],
                    y=[r.psi],
                    error_y=dict(
                        type="data", symmetric=False,
                        array=[r.ci_high - r.psi],
                        arrayminus=[r.psi - r.ci_low],
                    ),
                    marker_color="#2ecc71" if r.is_confident else "#e74c3c",
                    hovertext=(
                        f"PSI: {r.psi:.1%}<br>"
                        f"CI: [{r.ci_low:.1%}, {r.ci_high:.1%}]<br>"
                        f"CV: {r.cv:.3f}<br>"
                        f"Inc: {r.inclusion_count}, Exc: {r.exclusion_count}"
                    ),
                ))
            fig_psi.update_layout(
                title=f"Alternative Splicing PSI + Bootstrap CI — {selected_gene}",
                yaxis_title="PSI",
                yaxis=dict(range=[0, 1.05]),
                showlegend=False, height=500,
            )
            fig_psi.update_xaxes(tickangle=45)
            st.plotly_chart(fig_psi, use_container_width=True)

            n_conf = sum(1 for r in psi_results if r.is_confident)
            st.markdown(
                f"**{len(psi_results)} events**, "
                f"**{n_conf} confident** (CI < 20%)",
            )
        else:
            st.info("No AS events detected for this gene.")
    else:
        st.info("Provide BAM path and annotation GTF for PSI analysis.")

# ─── Footer ───
st.sidebar.markdown("---")
st.sidebar.markdown("**BRAID** v1.0")
