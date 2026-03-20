#!/usr/bin/env python3
"""BRAID quick-start demo — generates synthetic data and runs full pipeline.

Usage:
    python demo/run_demo.py

Produces:
    demo/results/demo_report.html   — interactive HTML report
    demo/results/demo_results.json  — raw results
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from braid.target.psi_bootstrap import (
    bootstrap_psi,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_synthetic_events(
    n_events: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic AS events with known ground-truth PSI."""
    rng = np.random.default_rng(seed)
    events = []
    for i in range(n_events):
        true_psi = rng.beta(2, 2)
        total_reads = int(rng.exponential(200) + 10)
        inc = rng.binomial(total_reads, true_psi)
        exc = total_reads - inc

        # Add noise (overdispersion)
        if rng.random() < 0.3:
            noise = int(rng.normal(0, total_reads * 0.1))
            inc = max(0, inc + noise)
            exc = max(0, total_reads - inc)

        event_type = rng.choice(["SE", "A3SS", "A5SS", "RI"], p=[0.5, 0.2, 0.2, 0.1])
        chrom = str(rng.integers(1, 23))
        start = rng.integers(1_000_000, 200_000_000)

        events.append({
            "event_id": f"{event_type}:{chrom}:{start}-{start+100}",
            "event_type": event_type,
            "chrom": chrom,
            "true_psi": float(true_psi),
            "inc_count": int(inc),
            "exc_count": int(exc),
            "total": int(inc + exc),
        })
    return events


def run_braid_on_events(
    events: list[dict],
    n_replicates: int = 500,
    seed: int = 42,
) -> list[dict]:
    """Run BRAID bootstrap on synthetic events."""
    results = []
    rng = np.random.default_rng(seed)

    for ev in events:
        psi, ci_low, ci_high, cv = bootstrap_psi(
            ev["inc_count"],
            ev["exc_count"],
            n_replicates=n_replicates,
            seed=int(rng.integers(0, 2**31)),
        )

        ci_width = ci_high - ci_low
        covers_true = ci_low <= ev["true_psi"] <= ci_high

        results.append({
            **ev,
            "braid_psi": float(psi),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_width": float(ci_width),
            "cv": float(cv),
            "covers_true": bool(covers_true),
            "is_confident": ci_width < 0.2 and np.isfinite(cv) and cv <= 0.5,
        })

    return results


def compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics."""
    n = len(results)
    coverage = sum(r["covers_true"] for r in results) / n
    confident = [r for r in results if r["is_confident"]]
    n_confident = len(confident)
    conf_accuracy = (
        sum(r["covers_true"] for r in confident) / n_confident
        if n_confident > 0 else 0
    )

    psi_errors = [abs(r["braid_psi"] - r["true_psi"]) for r in results]
    correlation = float(np.corrcoef(
        [r["true_psi"] for r in results],
        [r["braid_psi"] for r in results],
    )[0, 1])

    # By event type
    by_type = {}
    for et in ["SE", "A3SS", "A5SS", "RI"]:
        subset = [r for r in results if r["event_type"] == et]
        if subset:
            by_type[et] = {
                "count": len(subset),
                "ci_coverage": sum(r["covers_true"] for r in subset) / len(subset),
                "median_ci_width": float(np.median([r["ci_width"] for r in subset])),
                "n_confident": sum(r["is_confident"] for r in subset),
            }

    # By support bin
    by_support = {}
    bins = [("<50", 0, 50), ("50-99", 50, 100), ("100-249", 100, 250), ("250+", 250, 999999)]
    for label, lo, hi in bins:
        subset = [r for r in results if lo <= r["total"] < hi]
        if subset:
            by_support[label] = {
                "count": len(subset),
                "ci_coverage": sum(r["covers_true"] for r in subset) / len(subset),
                "median_ci_width": float(np.median([r["ci_width"] for r in subset])),
            }

    return {
        "n_events": n,
        "ci_coverage": float(coverage),
        "correlation": float(correlation),
        "mae": float(np.mean(psi_errors)),
        "median_ci_width": float(np.median([r["ci_width"] for r in results])),
        "n_confident": n_confident,
        "confident_accuracy": float(conf_accuracy),
        "by_event_type": by_type,
        "by_support_bin": by_support,
    }


def generate_html_report(results: list[dict], summary: dict) -> str:
    """Generate interactive HTML report with Plotly."""
    true_psis = [r["true_psi"] for r in results]
    braid_psis = [r["braid_psi"] for r in results]
    ci_lows = [r["ci_low"] for r in results]
    ci_highs = [r["ci_high"] for r in results]
    ci_widths = [r["ci_width"] for r in results]
    covers = [r["covers_true"] for r in results]
    types = [r["event_type"] for r in results]
    totals = [r["total"] for r in results]
    event_ids = [r["event_id"] for r in results]

    type_colors = {"SE": "#3498db", "A3SS": "#e74c3c", "A5SS": "#2ecc71", "RI": "#f39c12"}
    _colors = [type_colors.get(t, "#999") for t in types]
    _marker_symbols = ["circle" if c else "x" for c in covers]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>BRAID Demo Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; background: #fafafa; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                 text-align: center; }}
        .card .value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .card .label {{ font-size: 0.9em; color: #7f8c8d; margin-top: 5px; }}
        .card.green .value {{ color: #27ae60; }}
        .card.blue .value {{ color: #2980b9; }}
        .card.orange .value {{ color: #e67e22; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white;
                 border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        th, td {{ padding: 10px 15px; text-align: center; border-bottom: 1px solid #ecf0f1; }}
        th {{ background: #2c3e50; color: white; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .plot {{ margin: 20px 0; background: white; border-radius: 8px; padding: 15px;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .footer {{ margin-top: 40px; padding: 20px; text-align: center; color: #95a5a6; font-size: 0.9em; }}
    </style>
</head>
<body>
<h1>BRAID Demo Report</h1>
<p>Synthetic benchmark: {summary['n_events']} AS events with known ground-truth PSI.</p>

<div class="summary-grid">
    <div class="card green">
        <div class="value">{summary['ci_coverage']:.1%}</div>
        <div class="label">CI Coverage</div>
    </div>
    <div class="card blue">
        <div class="value">{summary['correlation']:.3f}</div>
        <div class="label">Correlation (r)</div>
    </div>
    <div class="card orange">
        <div class="value">{summary['n_confident']}</div>
        <div class="label">Confident Events</div>
    </div>
    <div class="card green">
        <div class="value">{summary['confident_accuracy']:.0%}</div>
        <div class="label">Confident Accuracy</div>
    </div>
</div>

<h2>By Event Type</h2>
<table>
<tr><th>Type</th><th>Count</th><th>CI Coverage</th><th>Median CI Width</th><th>Confident</th></tr>
"""
    for et, data in summary["by_event_type"].items():
        html += f"<tr><td>{et}</td><td>{data['count']}</td>"
        html += f"<td>{data['ci_coverage']:.1%}</td>"
        html += f"<td>{data['median_ci_width']:.3f}</td>"
        html += f"<td>{data['n_confident']}</td></tr>\n"

    html += """</table>

<h2>By Support Bin</h2>
<table>
<tr><th>Support</th><th>Count</th><th>CI Coverage</th><th>Median CI Width</th></tr>
"""
    for label, data in summary["by_support_bin"].items():
        html += f"<tr><td>{label}</td><td>{data['count']}</td>"
        html += f"<td>{data['ci_coverage']:.1%}</td>"
        html += f"<td>{data['median_ci_width']:.3f}</td></tr>\n"

    html += """</table>

<h2>PSI Scatter Plot (BRAID vs Ground Truth)</h2>
<div class="plot" id="scatter"></div>

<h2>CI Width vs Read Support</h2>
<div class="plot" id="ci_support"></div>

<h2>CI Coverage by Support Bin</h2>
<div class="plot" id="coverage_bar"></div>

<h2>PSI Distribution with CI</h2>
<div class="plot" id="ci_forest"></div>

<div class="footer">
    Generated by BRAID v0.1.0 | <a href="https://github.com/kangk1204/BRAID">GitHub</a>
</div>

<script>
"""

    # Scatter plot
    html += f"""
var true_psi = {json.dumps(true_psis)};
var braid_psi = {json.dumps(braid_psis)};
var ci_low = {json.dumps(ci_lows)};
var ci_high = {json.dumps(ci_highs)};
var types = {json.dumps(types)};
var totals = {json.dumps(totals)};
var covers = {json.dumps(covers)};
var event_ids = {json.dumps(event_ids)};

// Scatter: BRAID PSI vs True PSI
var traces = [];
['SE','A3SS','A5SS','RI'].forEach(function(et) {{
    var colors = {json.dumps(type_colors)};
    var idx = [];
    for (var i=0; i<types.length; i++) if (types[i]===et) idx.push(i);
    traces.push({{
        x: idx.map(i => true_psi[i]),
        y: idx.map(i => braid_psi[i]),
        mode: 'markers',
        type: 'scatter',
        name: et,
        marker: {{
            color: colors[et],
            size: idx.map(i => Math.min(12, Math.max(4, Math.sqrt(totals[i])/3))),
            opacity: 0.7,
            symbol: idx.map(i => covers[i] ? 'circle' : 'x'),
        }},
        text: idx.map(i => event_ids[i] + '<br>True=' + true_psi[i].toFixed(3) +
            '<br>BRAID=' + braid_psi[i].toFixed(3) +
            '<br>CI=[' + ci_low[i].toFixed(3) + ',' + ci_high[i].toFixed(3) + ']' +
            '<br>Reads=' + totals[i]),
        hoverinfo: 'text',
    }});
}});
traces.push({{x:[0,1], y:[0,1], mode:'lines', line:{{color:'gray',dash:'dash'}},
    showlegend:false, hoverinfo:'none'}});

Plotly.newPlot('scatter', traces, {{
    title: 'BRAID PSI vs Ground Truth (r={summary["correlation"]:.3f})',
    xaxis: {{title: 'True PSI'}}, yaxis: {{title: 'BRAID PSI'}},
    height: 500, hovermode: 'closest',
}});

// CI width vs support
Plotly.newPlot('ci_support', [{{
    x: totals, y: {json.dumps(ci_widths)},
    mode: 'markers', type: 'scatter',
    marker: {{color: covers.map(c => c ? '#27ae60' : '#e74c3c'), size: 5, opacity: 0.6}},
    text: event_ids.map((id,i) => id + '<br>Width=' + {json.dumps(ci_widths)}[i].toFixed(3)),
    hoverinfo: 'text',
}}], {{
    title: 'CI Width vs Read Support (green=covers true, red=misses)',
    xaxis: {{title: 'Total Reads', type: 'log'}}, yaxis: {{title: 'CI Width'}},
    height: 400,
}});

// Coverage bar chart
var support_labels = {json.dumps(list(summary['by_support_bin'].keys()))};
var support_coverage = {json.dumps([d['ci_coverage'] for d in summary['by_support_bin'].values()])};
Plotly.newPlot('coverage_bar', [{{
    x: support_labels, y: support_coverage, type: 'bar',
    marker: {{color: support_coverage.map(c => c >= 0.9 ? '#27ae60' : c >= 0.8 ? '#f39c12' : '#e74c3c')}},
    text: support_coverage.map(c => (c*100).toFixed(1) + '%'), textposition: 'auto',
}}], {{
    title: 'CI Coverage by Support Bin (target: 95%)',
    yaxis: {{title: 'Coverage', range: [0,1.05]}},
    shapes: [{{type:'line', x0:-0.5, x1:support_labels.length-0.5, y0:0.95, y1:0.95,
        line:{{color:'red',dash:'dash',width:2}}}}],
    height: 350,
}});

// Forest plot (first 30 events sorted by PSI)
var sorted_idx = Array.from(Array(Math.min(30, true_psi.length)).keys())
    .sort((a,b) => true_psi[a] - true_psi[b]);
Plotly.newPlot('ci_forest', [
    {{x: sorted_idx.map(i => braid_psi[i]), y: sorted_idx.map((_,j) => j),
      error_x: {{
          type: 'data', symmetric: false,
          array: sorted_idx.map(i => ci_high[i] - braid_psi[i]),
          arrayminus: sorted_idx.map(i => braid_psi[i] - ci_low[i]),
          color: '#3498db', thickness: 1.5,
      }},
      mode: 'markers', marker: {{color: '#2c3e50', size: 6}},
      text: sorted_idx.map(i => event_ids[i]),
      hoverinfo: 'text+x', name: 'BRAID CI'}},
    {{x: sorted_idx.map(i => true_psi[i]), y: sorted_idx.map((_,j) => j),
      mode: 'markers', marker: {{color: '#e74c3c', size: 8, symbol: 'diamond'}},
      name: 'True PSI', hoverinfo: 'x'}},
], {{
    title: 'Forest Plot: BRAID CI vs True PSI (first 30 events)',
    xaxis: {{title: 'PSI', range: [-0.05, 1.05]}},
    yaxis: {{title: 'Event', showticklabels: false}},
    height: 600, hovermode: 'closest',
}});
</script>
</body>
</html>"""
    return html


def main():
    """Run BRAID demo."""
    print("=" * 50)
    print("  BRAID Quick-Start Demo")
    print("=" * 50)

    # Generate synthetic data
    print("\n1. Generating 200 synthetic AS events...")
    events = generate_synthetic_events(n_events=200, seed=42)
    print(f"   {len(events)} events generated")

    # Run BRAID
    print("\n2. Running BRAID bootstrap (500 replicates)...")
    t0 = time.time()
    results = run_braid_on_events(events, n_replicates=500, seed=42)
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.2f}s")

    # Compute summary
    summary = compute_summary(results)
    print(f"\n3. Results:")
    print(f"   CI Coverage:         {summary['ci_coverage']:.1%}")
    print(f"   Correlation:         {summary['correlation']:.3f}")
    print(f"   MAE:                 {summary['mae']:.3f}")
    print(f"   Confident events:    {summary['n_confident']}")
    print(f"   Confident accuracy:  {summary['confident_accuracy']:.0%}")

    # Save JSON
    json_path = os.path.join(OUTPUT_DIR, "demo_results.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "events": results}, f, indent=2)
    print(f"\n4. Saved JSON: {json_path}")

    # Generate HTML
    print("\n5. Generating interactive HTML report...")
    html = generate_html_report(results, summary)
    html_path = os.path.join(OUTPUT_DIR, "demo_report.html")
    with open(html_path, "w") as f:
        f.write(html)
    print(f"   Saved: {html_path}")
    print(f"   Open in browser to explore interactive plots!")

    print(f"\n{'='*50}")
    print(f"  Demo complete!")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
