# Additional Benchmark Attempts — Status

## Attempted but Unusable

### PC3E/GS689 (Shen 2014, PNAS)
- **Status**: UNUSABLE
- **Reason**: Gene-level matching fails (r=-0.079). Each gene has dozens of SE events.
  Exact exon coordinates needed from PNAS supplementary, but Cloudflare blocks download.
- **Data available**: rMATS output from Zenodo (283K SE events)
- **Fix needed**: Manual coordinate extraction from PNAS PDF or author contact

### MAJIQ Mouse (Vaquero-Garcia 2016, eLife)  
- **Status**: UNUSABLE
- **Reason**: RT-PCR quantitative PSI values not in supplementary tables.
  Primer sequences and event IDs available (73 events), but actual PSI measurements
  only in figure panels. GitHub data/ directory not included in repo.
- **Data available**: Event coordinates (supp2), primer sequences, SRA accessions
- **Fix needed**: Author contact for raw RT-PCR PSI data, or RNA-seq BAM download + rMATS

## Successfully Completed

### TRA2 KD (Trincado 2018) — 42,443 SE events
### TARDBP KD (ENCODE) — 84,838 events (5 types)
### QKI RT-PCR (Hall 2013) — 80 validated + 25 failed
### PacBio K562 (ENCODE) — 204 A3SS/A5SS events
### GM12878 Multi-Replicate — 32,771 isoforms
### 3 Cell Line Isoform Confidence — 128,878 isoforms
