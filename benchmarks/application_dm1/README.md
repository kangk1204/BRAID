# DM1 Application Analysis

This directory builds the disease-application evidence layer for the BRAID
manuscript. The analysis uses public rMATS skipped-exon output from GSE201255,
a human skeletal-muscle myotonic dystrophy type 1 dataset, then applies BRAID as
an rMATS post-processing confidence layer.

Primary command:

```bash
python benchmarks/application_dm1/run_dm1_application.py
```

The script defaults to `--differential-model sum`, the pooled-count BRAID
estimator used for the manuscript DM1 numbers. To run the newer replicate-aware
sensitivity analysis instead, pass `--differential-model auto`.

Generated outputs:

- `raw/GSE201255_caseVScontrol_SE.MATS.JC.txt.gz`: downloaded GEO rMATS table.
- `rmats/SE.MATS.JC.txt`: rMATS table in BRAID CLI layout.
- `results/dm1_braid_differential.tsv`: BRAID differential output.
- `results/dm1_application_summary.json`: analysis summary and provenance.
- `results/dm1_anchor_gene_summary.tsv`: known DM1/muscle-splicing anchor recovery.
- `results/dm1_top_braid_candidates.tsv`: non-anchor high-confidence candidates.
- `results/dm1_top_braid_candidate_genes.tsv`: non-anchor candidates aggregated by gene.

Important convention: the GSE201255 supplementary rMATS table labels sample 1 as
case and sample 2 as control. BRAID's differential CLI preserves the rMATS
sample_1 minus sample_2 sign convention. In these application outputs,
`disease_minus_control_dpsi` therefore corresponds to sample_1 minus sample_2.

Expected manuscript checks with the default command:

- `events_after_braid_min_support`: 144,975
- `rmats_big_events_fdr_lt_0_05_abs_dpsi_ge_0_1`: 967
- `tier_counts.high-confidence`: 68
- `anchor_genes_with_braid_supported_event`: 12
- `anchor_genes_with_braid_high_confidence_rmats_significant_event`: 10
- `primary_anchor_genes_with_braid_high_confidence_rmats_significant_event`: 7
- `top_non_anchor_candidates_written`: 49
