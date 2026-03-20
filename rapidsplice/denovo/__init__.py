"""GPU-native de novo RNA-seq transcript assembler.

This module implements a de Bruijn graph-based de novo assembler that
operates directly on FASTQ reads without a reference genome.  All core
operations are designed for GPU acceleration with transparent CPU fallback.

Pipeline stages:

1. **FASTQ reading** -- Load and optionally quality-filter reads.
2. **K-mer extraction** -- 2-bit encoded k-mers from reads.
3. **K-mer counting** -- Frequency table via sort-and-count on GPU.
4. **De Bruijn graph** -- Nodes are (k-1)-mers, edges are k-mers.
5. **Graph compaction** -- Merge linear chains into unitigs.
6. **Graph simplification** -- Tip removal, bubble collapsing, coverage
   filtering to produce a clean assembly graph.
7. **Transcript extraction** -- Coverage-guided path enumeration through
   branching points to reconstruct full-length transcripts.

Modules:

- ``fastq`` -- FASTQ file reading with quality filtering.
- ``kmer`` -- K-mer extraction, encoding, and counting.
- ``graph`` -- De Bruijn graph construction and compaction.
- ``simplify`` -- Graph simplification algorithms.
- ``assemble`` -- Transcript path extraction.
- ``pipeline`` -- End-to-end de novo assembly pipeline.
"""
