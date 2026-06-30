# Additional Benchmark Datasets for BRAID

## 1. rMATS Validation Dataset (Shen et al. 2014, PNAS)

### Publication
- **Paper**: Shen S et al. "rMATS: Robust and Flexible Detection of Differential Alternative Splicing from Replicate RNA-Seq Data" PNAS 2014; 111(51):E5593-601
- **PMID**: 25480548
- **PMCID**: PMC4280593

### Validated Events
- **34 exon skipping (SE) events** selected randomly from differential splicing results
- RT-PCR validation in prostate cancer cell lines
- **94% validation rate** (32/34 confirmed by RT-PCR)
- PSI values span |delta-PSI| from 0.10 to 0.90
- Pearson correlation r = 0.96 between RNA-seq and RT-PCR PSI estimates

### Supplementary Data
- **SD1 & SD2**: RT-PCR validation results for the 34 selected exons (genomic coordinates + PSI values)
- **SD3**: 315 differential SE events from TCGA ccRCC data
- **SD4**: 11 SE events unique to paired rMATS analysis
- **SD5**: Sample IDs for 65 tumor-normal matched pairs
- Available from PMC supplementary files: https://pmc.ncbi.nlm.nih.gov/articles/PMC4280593/

### RNA-Seq Data
- **Cell lines**: PC3E (epithelial) vs GS689.Li (mesenchymal) prostate cancer cells
- **Replicates**: 3 biological replicates per condition
- **Platform**: Illumina HiSeq 2000, paired-end
- **SRA accession**: SRS354082

| Cell Line | SRA Experiment | SRR Runs | Total Bases |
|-----------|---------------|----------|-------------|
| PC3E | SRX174803 | SRR536348, SRR536350, SRR536352 | 76.9 Gb |
| GS689.Li | SRX174805 | SRR536342, SRR536344, SRR536346 | 73.8 Gb |

### Download Commands
```bash
# Download PC3E replicates
fastq-dump --split-files --gzip SRR536348
fastq-dump --split-files --gzip SRR536350
fastq-dump --split-files --gzip SRR536352

# Download GS689.Li replicates
fastq-dump --split-files --gzip SRR536342
fastq-dump --split-files --gzip SRR536344
fastq-dump --split-files --gzip SRR536346

# Or use fasterq-dump (faster)
fasterq-dump --split-files -e 8 SRR536348 SRR536350 SRR536352
fasterq-dump --split-files -e 8 SRR536342 SRR536344 SRR536346

# Download supplementary tables from PMC
wget "https://pmc.ncbi.nlm.nih.gov/articles/PMC4280593/bin/pnas.1419161111.sd01.xlsx" -O rmats_SD1_rtpcr_validation.xlsx
wget "https://pmc.ncbi.nlm.nih.gov/articles/PMC4280593/bin/pnas.1419161111.sd02.xlsx" -O rmats_SD2_rtpcr_validation.xlsx
```

### Downloadability
- **RNA-seq FASTQ**: Yes, publicly available on SRA (~150 Gb total)
- **Supplementary tables**: Yes, available from PMC (Excel format)
- **Status**: Ready to download

---

## 2. SUPPA2 Validation Dataset (Trincado et al. 2018, Genome Biology)

### Publication
- **Paper**: Trincado JL et al. "SUPPA2: fast, accurate, and uncertainty-aware differential splicing analysis across multiple conditions" Genome Biology 2018; 19:40
- **PMID**: 29571299
- **PMCID**: PMC5866513
- **DOI**: 10.1186/s13059-018-1417-1

### Validated Events (TRA2A/B Knockdown)
- **83 cassette exon (SE) events** validated by RT-PCR as differentially spliced upon TRA2A/B double knockdown
- **44 RT-PCR negative cassette exon events** (no significant change upon knockdown) -- serves as negative control set
- **Additional validation**: 15 SE events + 7 A3SS events with CLIP tags and Tra2 motifs (9/15 SE confirmed, 4/6 A3SS confirmed)
- Cell line: **MDA-MB-231** breast cancer cells
- Condition: TRA2A + TRA2B double siRNA knockdown vs. negative control siRNA
- 3 biological replicates per condition

### Additional SUPPA2 Validation Datasets
| Dataset | GEO | Description | Validated Events |
|---------|-----|-------------|-----------------|
| TRA2A/B KD | GSE59335 | MDA-MB-231 TRA2 double knockdown | 83 SE + 44 negative controls |
| Mouse circadian | GSE54651 | Cerebellum + liver circadian expression | 50 RT-PCR validated events |
| iPSC differentiation | GSE60548 | Bipolar neuron differentiation (4 days) | Multi-condition time course |
| Erythropoiesis | GSE53635 | Human erythroblast differentiation (5 stages) | Multi-condition time course |

### RNA-Seq Data (TRA2A/B dataset)
- **GEO accession**: GSE59335
- **SRA project**: SRP044265
- **Platform**: Illumina HiSeq 2000

| Sample | Type | GEO | SRA Experiment | Example SRR |
|--------|------|-----|---------------|-------------|
| Control rep1 | Neg. control siRNA | GSM1435246 | - | - |
| Control rep2 | Neg. control siRNA | GSM1435247 | - | - |
| Control rep3 | Neg. control siRNA | GSM1435248 | - | - |
| KD rep1 | TRA2A/B siRNA | GSM1435249 | SRX651011 | SRR1513332 |
| KD rep2 | TRA2A/B siRNA | GSM1435250 | SRX651012 | - |
| KD rep3 | TRA2A/B siRNA | GSM1435251 | SRX651013 | - |

Note: GSM1435252-GSM1435254 are iCLIP samples (not RNA-seq), skip for splicing analysis.

### Supplementary Data
- **Supplementary Tables S1-S17**: Excel file from PMC (3 MB)
- **GitHub repository**: https://github.com/comprna/SUPPA_supplementary_data
  - `supplementary_data/tra2/` -- SUPPA2/rMATS/MAJIQ/DEXSeq results on TRA2 data
  - `additional_files/TRA2_diffSplice.dpsi.gz` -- Differential PSI results
  - `additional_files/TRA2_diffSplice_iso.dpsi.gz` -- Isoform-level differential PSI

### Download Commands
```bash
# Download TRA2A/B knockdown RNA-seq (6 samples: 3 control + 3 knockdown)
# Get SRR accessions from SRA Run Selector
# https://www.ncbi.nlm.nih.gov/Traces/study/?acc=SRP044265

# Example for one knockdown replicate:
fasterq-dump --split-files -e 8 SRR1513332

# Download all RNA-seq runs for the project (excludes iCLIP by filtering):
# Use SRA Run Selector to get the full list, then:
# prefetch SRR1513329 SRR1513330 SRR1513331 SRR1513332 SRR1513333 SRR1513334
# (Exact SRR numbers for controls need verification via Run Selector)

# Clone SUPPA2 supplementary data (includes pre-computed results)
git clone https://github.com/comprna/SUPPA_supplementary_data.git

# Download supplementary tables from PMC
wget "https://pmc.ncbi.nlm.nih.gov/articles/PMC5866513/bin/13059_2018_1417_MOESM2_ESM.xlsx" -O suppa2_supplementary_tables.xlsx
```

### Downloadability
- **RNA-seq FASTQ**: Yes, publicly available on SRA (GSE59335 / SRP044265)
- **Supplementary tables**: Yes, from PMC (Excel)
- **Pre-computed results**: Yes, from GitHub (comprna/SUPPA_supplementary_data)
- **Status**: Ready to download

---

## Summary Table

| Dataset | Paper | Events | Type | Accession | Downloadable |
|---------|-------|--------|------|-----------|-------------|
| rMATS SE validation | Shen 2014 PNAS | 34 SE (32 confirmed) | RT-PCR validated | SRS354082 | Yes |
| rMATS TCGA ccRCC | Shen 2014 PNAS | 315 SE | Computational | TCGA | Yes (via GDC) |
| SUPPA2 TRA2 KD | Trincado 2018 Genome Biol | 83 SE + 44 neg | RT-PCR validated | GSE59335 | Yes |
| SUPPA2 mouse circadian | Trincado 2018 Genome Biol | 50 events | RT-PCR validated | GSE54651 | Yes |
| SUPPA2 iPSC diff | Trincado 2018 Genome Biol | Multi-condition | Time course | GSE60548 | Yes |
| SUPPA2 erythropoiesis | Trincado 2018 Genome Biol | Multi-condition | Time course | GSE53635 | Yes |

## Recommended Priority for BRAID Benchmarking

1. **SUPPA2 TRA2 KD (GSE59335)** -- Best validation dataset: 83 positive + 44 negative RT-PCR validated SE events with publicly available RNA-seq. Clear ground truth for precision/recall evaluation.

2. **rMATS SE validation (SRS354082)** -- 34 RT-PCR validated SE events with high-quality RNA-seq. Smaller but well-characterized, with published PSI correlation (r=0.96).

3. **SUPPA2 mouse circadian (GSE54651)** -- 50 validated events, useful for cross-species generalization testing.
