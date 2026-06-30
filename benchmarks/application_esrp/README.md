# ESRP Regulator-Target Recovery Analysis

This directory builds the cross-species regulator-loss recovery evidence for the
BRAID manuscript, complementing the human disease application in
`../application_dm1`. Where DM1 asks whether BRAID recovers known *disease*
mis-splicing anchors, this analysis asks the regulator-knockout version: in an
Esrp1/Esrp2 double-knockout (DKO) vs wild-type (WT) mouse comparison, does BRAID
confidently recover the canonical epithelial cassette exons that Esrp1/2 are
established to regulate?

## Dataset

- GEO: [GSE64357](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE64357)
  (Bebee et al., 2015, *eLife*) — Esrp1/Esrp2 double-knockout mouse.
- DKO replicates (rMATS `--b1`): SRR1725976, SRR1725977.
- WT replicates (rMATS `--b2`): SRR1725983, SRR1725984.

Unlike DM1 (which post-processes a published rMATS table), GSE64357 has no
published skipped-exon table, so the rMATS output is regenerated from the raw SRA
reads: HISAT2 → mm10 (GENCODE vM2) → rMATS turbo v4.3.0 (paired). That pipeline
lives under `data/public_benchmarks/GSE64357_esrp/` (gitignored — BAMs/FASTQs are
large). The rMATS `SE.MATS.JC.txt` and the other event tables are the input to
this analysis.

## Command

```bash
# Requires the rMATS output dir at data/public_benchmarks/GSE64357_esrp/rmats/
python benchmarks/application_esrp/run_esrp_application.py
```

The script defaults to `--differential-model auto`, matching the manuscript
application run. The selected model is recorded in
`results/esrp_application_summary.json`.

## Generated outputs (gitignored under results/)

- `results/esrp_braid_differential.tsv`: BRAID differential output (all event types).
- `results/esrp_application_summary.json`: analysis summary and provenance.
- `results/esrp_anchor_gene_summary.tsv`: per-anchor-gene recovery (best event,
  calibrated dPSI, 95% CI, tier).

## Conventions and scope

- rMATS sample_1 = DKO (`--b1`), sample_2 = WT (`--b2`); BRAID `dpsi_mean =
  PSI(DKO) - PSI(WT)`.
- Anchors are a literature-curated set of canonical Esrp1/2-regulated cassette
  exons (Warzecha 2009 *Mol Cell*; Warzecha 2010 *EMBO J*; Dittmar 2012 *MCB*;
  Bebee 2015 *eLife*). These are a published gene set, not fabricated data; every
  PSI, interval, and tier is computed by BRAID on the real rMATS table.
- Recovery metric (mirroring DM1): gene-level. For each anchor gene the best
  event is taken across all rMATS event types (by tier, then |dPSI|, then
  support), and the gene is recovered when that event is BRAID high-confidence.
  The high-confidence tier already requires the calibrated 95% interval to
  exclude zero (and rMATS BH-FDR < 0.05), so "high-confidence" and "interval
  excludes zero" are the same criterion, not independent evidence. Denominators
  are the genes with a testable event, since genes with no event cannot be
  recovered. The reported per-gene dPSI/interval is the max-selected best event
  (an upper bound on that gene's effect, not an unbiased estimate). Direction is
  reported, not asserted per gene — Esrp can promote either inclusion or exclusion
  depending on binding position.
- This is regulator-target recovery evidence, not orthogonal RT-PCR validation.
  MXE targets (e.g. Fgfr2 IIIb/IIIc) are not in the SE table by construction.

Expected manuscript checks with the default command:

- `events_after_braid_min_support`: 27,169
- `rmats_big_events_fdr_lt_0_05_abs_dpsi_ge_0_1`: 724
- `tier_counts.high-confidence`: 114
- `anchor_genes_with_se_event`: 12
- `anchor_genes_with_braid_high_confidence_event`: 9
- `primary_anchor_genes_with_testable_event`: 8
- `primary_anchor_genes_with_braid_high_confidence_event`: 7
