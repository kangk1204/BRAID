# `braid filter` sample data

One tiny, synthetic differential-splicing table per caller, so you can run
`braid filter` end to end without any real data. Each file contains the same three
illustrative events:

| gene | meaning | expected BRAID tier |
|---|---|---|
| BIGSWITCH | large, significant ΔPSI | high-confidence |
| MODEST | significant but small ΔPSI (calibrated interval crosses 0) | caller-significant-only |
| NULLEVENT | no change, not significant | not-significant |

These files are bundled in the installed package, so `--example` runs them with no
path needed (works straight after `pip install braid`):

| File | Caller | Command |
|---|---|---|
| `rmats_output/SE.MATS.JC.txt` | rMATS | `braid filter --caller rmats --example -o demo` |
| `majiq_deltapsi.tsv` | MAJIQ | `braid filter --caller majiq --example -o demo` |
| `suppa2_diffSplice.dpsi` | SUPPA2 | `braid filter --caller suppa2 --example -o demo` |
| `betas_differential.tsv` | betAS | `braid filter --caller betas --example -o demo` |

To run on a copy in the repo instead, pass the path under `braid/examples/filter/`.

Each run writes `demo.tsv`, `demo.xlsx`, and `demo.png/.pdf/.svg` (the Excel and
figure need `pip install -e ".[report]"`). The data are fabricated for illustration
only — they are not a benchmark.

The samples intentionally differ in support metadata. rMATS has junction counts,
MAJIQ includes a `num_reads` column, SUPPA2 has no countable read support, and the
betAS sample includes a native lower/upper interval but no support column. BRAID
marks this in `support_known` and uses the pooled global quantile when support is
missing.
