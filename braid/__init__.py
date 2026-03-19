"""BRAID: Bootstrap Resampling for Assembly Isoform Dependability.

A post-assembly tool that adds per-isoform bootstrap confidence
intervals to RNA-seq transcript assembly output. Works with
StringTie, Scallop, or any assembler that produces GTF output.

Key features:
- Post-assembly bootstrap CV scoring (precision +35-53pp)
- Single-sample PSI confidence intervals (no replicates needed)
- BAM-free mode (113s genome-wide)
- Interactive Streamlit dashboard
"""

__version__ = "1.0.0"
