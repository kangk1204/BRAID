# Failure-Mode Matrix — `braid differential` ΔPSI pipeline

Tested: 2026-06-23 (input-guard pass 2026-06-25; review-pass 2026-06-26) | Stages: 7 | Fail: 0
Suites: `tests/test_pipeline_robustness_differential.py` (15) +
`tests/test_pipeline_robustness_inputguards.py` (40) +
`tests/test_differential_swap.py` (4) +
`tests/test_adapters_parsers_qa.py` (8, `braid filter` caller adapters) | Full repo: 1046 passed
(Companion: `tests/test_pipeline_robustness.py` covers the `braid filter` path.)

Pipeline (execution order): parse rMATS → PSI/ΔPSI posterior → conformal interval
→ confidence tier → write TSV. Cells below were ❌ before this session's two guards
and are ✅ after (see "Fixed cells").

| Stage \ Failure mode | A1 NA/empty cell | A2 empty input | A4 type mismatch | A5 schema drift | A6 out-of-range | A9 extreme scale | A10 inf/NaN | B4 row misalign | C1 file-not-found | F4 contract |
|---|---|---|---|---|---|---|---|---|---|---|
| `S1 cli-validate`        | — | — | ✅ | — | ✅ | — | — | — | — | — |
| `S2 table-discover`      | — | ✅ | — | — | — | — | — | — | ✅ | — |
| `S3 parse`               | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | ✅ | — | — |
| `S4 psi-posterior`       | — | ✅ | ✅ | — | — | ✅ | ✅ | — | — | — |
| `S5 conformal`           | — | — | ✅ | ✅ | ✅ | — | ✅ | — | ✅ | ✅ |
| `S6 tier`                | — | — | — | — | — | — | ✅ | — | — | — |
| `S7 write`               | ✅ | ✅ | — | — | — | — | ✅ | — | — | — |

Legend: ✅ handled (typed error or valid output) · ⚠️ degraded · ❌ crashes/corrupts · — N/A.

## Fixed cells (were ❌ → now ✅)

- **S3 × A5** (`test_S3_A5_missing_required_count_column_fails_fast`): a table missing
  a required count column (e.g. `SJC_SAMPLE_2`) is self-consistent
  (`len(fields)==len(header)`), so the row-truncation guard never fires. The missing
  group silently parsed as `exc=0`, fabricating `treat_psi=1.0` and a spurious large
  ΔPSI for **every** event — silent corruption. **Fix:** `parse_rmats_output` now
  validates the four count columns are present in the header and fails fast (table-level,
  regardless of `--strict`). `braid/target/rmats_bootstrap.py`.
- **S5 × C1** (`test_S5_C1_missing_calibration_file_clear_error`): `--calibration
  <missing>.json` raised a raw `FileNotFoundError [Errno 2]` (the user path wasn't
  wrapped; only the default-artifact path caught it). **Fix:** clear,
  `--calibration`-scoped `FileNotFoundError`. `braid/commands/differential.py`.
- **S5 × F4** (`test_S5_F4_malformed_calibration_json_clear_error`): a garbage or
  wrong-schema `--calibration` JSON leaked a raw `json.JSONDecodeError` / `KeyError`.
  **Fix:** re-raised as a clear `ValueError` naming the file. `braid/commands/differential.py`.

## Fixed cells — 2026-06-25 input-guard pass (were ❌ → now ✅)

- **S3 × A6** (`test_S3_A6_negative_count_vector_is_rejected`,
  `..._row_skipped_nonstrict`, `..._row_fails_fast_strict`): a **negative** junction count
  (corrupted/edited rMATS row) parsed straight through (`int("-5")`), then reached
  `rng.beta(a<=0)` and **aborted the whole `braid differential` run** with an opaque
  `Error: a <= 0` — no row/gene context, no per-row recovery in the differential loop.
  **Fix:** `_parse_count_vector` rejects negatives at the parse boundary, so the existing
  per-row try/except skips the row with a named warning (non-strict) or fails fast
  (`--strict`) — protecting both the PSI and differential paths at the origin.
  `braid/target/rmats_bootstrap.py`.
- **S5 × A4/A5** (`test_S5_A5_assign_support_bins_rejects_mismatched_edges`,
  `..._calibrator_from_dict_rejects_mismatched_bin_edges`,
  `..._rejects_non_numeric_bin_edges`, `..._rejects_non_monotonic_bin_edges`,
  `..._rejects_nonfinite_bin_edges`): a custom `--calibration` JSON whose `bin_edges` was
  the wrong length crashed `assign_support_bins`/`q_for` with a bare `IndexError` on the
  first high-support event; a right-length-but-malformed set (non-numeric, non-finite, or
  non-monotonic) passed the length check and instead crashed mid-run inside `np.digitize`
  (`TypeError` for strings, "bins must be monotonically increasing" for out-of-order).
  **Fix:** `ConformalCalibrator.from_dict` now validates the full edge contract at the
  JSON boundary — length == 4, numeric, finite, strictly increasing — **and stores the
  coerced numeric edges** (a numeric string like `"20"` passes `float()` but, left as a
  str, would still crash `np.digitize` at `q_for` time); integer-ness is preserved so the
  shipped all-integer edge set serializes unchanged. `assign_support_bins` keeps the
  length invariant defensively. `braid/target/conformal.py`.
- **S5 × A6** (`test_S5_A6_calibrator_alpha_out_of_range_rejected`): `from_dict` accepted a
  calibrator `alpha` outside `(0, 1)` (a meaningless miscoverage level marking a corrupted
  artifact). Now rejected at the JSON boundary alongside the existing negative-`q` guard.
  `braid/target/conformal.py`.

## Fixed cells — 2026-06-26 multi-perspective review pass (were ❌/inconsistent → now ✅)

Findings surfaced by a 5-perspective review (stat-correctness, security, code-quality,
numerical, adversarial), each re-verified before fixing.

- **S4 × A4** (`test_S4_A4_classify_support_bin_*`): `classify_support_bin` returned the
  least-conservative `250+` bin for any non-integer total in a unit gap (19<t<20, etc.).
  Reachable via `braid psi`, because form-length normalization makes `inc + exc*ratio` a
  float. Now bins by the first exclusive upper edge (matching `assign_support_bins`).
  `braid/target/psi_bootstrap.py`.
- **S6 tier consistency** (`test_S6_tier_*`): `confidence_tier` mapped the identical
  `(reliable=True, effect=False)` pair to `supported` in the `caller_sig is None` branch
  but `not-significant` in the bool branch. The None branch now mirrors the bool branch
  (reliable-without-effect is never `supported`). Differential is unaffected (always
  passes a bool FDR flag); this only tightened the `braid filter` cross-caller path.
  `braid/adapters/base.py`.
- **S3 × A10** (`test_S3_A10_parse_float_na_*`): `_parse_float_na("inf")` kept `inf`,
  inconsistent with `_parse_inc_level_mean` (filters non-finite). Now returns NaN.
  `braid/target/rmats_bootstrap.py`.
- **S5 × E1** (`test_S5_E1_from_json_*`): `ConformalCalibrator.from_json` read a
  `--calibration` path with no size cap (memory-exhaustion / device-hang surface). Now
  capped at 10 MB before read. `braid/target/conformal.py`.
- Type-annotation hygiene (no runtime change): `dict[str, object]` → `dict[str, Any]`
  (18 sites) and the `calibrator` parameter annotated in `adapters/base.py`; mypy errors
  in `conformal.py` dropped 10 → 2 (the residual 2 are unrelated importlib `joinpath`
  stub strictness).
- **S3 × A8 (duplicate event_id)** (`test_a3ss_includes_flanking_coords_to_avoid_rmats_collisions`):
  A3SS/A5SS event IDs carried the long/short exon boundaries but omitted the constitutive
  **flanking** exon, so two distinct rMATS events sharing the same long/short exon but a
  different flanking exon collapsed to one `event_id`. Real collisions on the paper
  datasets (639 / 193 / 1635 across GSE59335 / GSE54651 / SRS354082 at `min_total_count=0`);
  distinct events could be lost in any key-based downstream. **Fix:** append
  `f{flankingES}-{flankingEE}` to the A3SS/A5SS ID → rows == unique IDs (dup=0) on all three
  datasets. No published number changes: SE headline reproduces byte-exact (max coverage
  diff 0.00e+00), the SG-NEx surface builds its own event_id, and the committed non-SE
  fixture (`rtpcr_benchmark.json`, n=252) had zero duplicate IDs (no realized loss). The
  A3SS/A5SS ID *string* is now more specific, so it will not string-match IDs stored by the
  old format. `braid/target/rmats_bootstrap.py`.

### Feature — `braid differential --swap-groups` (`tests/test_differential_swap.py`)

Exposes the ΔPSI direction as an explicit parameter (was hardcoded sample_1 − sample_2).
`--swap-groups` reports sample_2 − sample_1: control↔treatment counts and replicate
vectors swap, `ctrl/treat_psi` + `ctrl/treat_support` columns swap, `rmats_dpsi` is
negated to stay sign-consistent, and direction-agnostic fields (`reliable`, `tier`,
`caller_significant`) are unchanged. Mirrors the benchmark `event_counts(swap_groups=...)`
semantics. `braid/commands/differential.py`, documented in `README.md`.

## Already-robust cells verified by test (12 ✅, no change needed)

- S3 × A1 — trailing comma / trailing NA dropped quietly; **interior** gap hard-rejected
  (would desync replicate pairing). S3 × A4 — non-int count skips row (non-strict) /
  fails fast (`--strict`). S3 × B4 — truncated row skipped.
- S4 × A2 — all-below-support writes header-only TSV (no crash, no missing file).
  S4 × A9 — extreme `IncFormLen/SkipFormLen` ratio stays finite in [-1,1].
- S5 × A10 — `robust_interval` with NaN/inf scale or point → conservative full clip,
  never `(nan,nan)`. S5 × A6 — negative/NaN support → widest (most conservative) bin,
  never the tightest `250+`. S5 × F4 — scale-kind mismatch rejected (`require_scale_kind`).
- S6 × A10 — NaN rMATS FDR keeps the event but sets `caller_significant=no`.
- S7 × A7 — CSV formula-injection in gene symbol apostrophe-guarded.

## Not-applicable rationale

- S1/S2 content pathologies (A1/A4/A9/A10): S1 only validates scalar CLI knobs; S2 only
  checks directory/table presence — content is parsed at S3+.
- S3 × A9/A10: scale/inf are numeric properties tested at the posterior (S4) and interval
  (S5) stages where they bite, not at the text-parse layer. (A6 negative counts ARE now
  rejected at parse — see "Fixed cells — 2026-06-25".)
- S4/S5/S6/S7 × C1: the input file is opened once at S2/S3; later stages receive
  in-memory objects.
- S6/S7 × A5/B4: schema/row issues are caught upstream at S3 before tiering/writing.

## Coverage notes

- Categories tested: A (input pathologies), B4 (row/header), C1 (file), F4 (contract).
- NOT tested (recommend future): C2/C3 (permissions/disk), E (resource/OOM), F3
  (mutable-default bleed). The differential path is single-process CPU, so D2/D4
  (singular matrix / non-convergence) and E2/E3 (GPU) do not apply.
- Scope: `braid differential` only. `braid filter` is covered by the companion suite;
  `braid psi` and `braid run`/assemble share S3 (parse) but their PSI/flow stages were
  not swept this session.

---

# Failure-Mode Matrix — replicate-aware differential stage

Tested: 2026-06-23 | Functions: 6 | Tests: 10 (new) | Pass: 10 | Fail: 0
Suite: `tests/test_pipeline_robustness_replicate.py`
Covers the per-replicate machinery behind `--differential-model {auto,rep,sum}`
(commits 0c3eefb + dcc1d71). Every cell came back ✅ — no bug, no missing guard.

| Function \ Failure mode | A2 degenerate | A3 single-rep | A8 length mismatch | A10 inf/NaN | F1 contract |
|---|---|---|---|---|---|
| `_complete_replicate_arrays` | — | ✅ | ✅ | — | — |
| `_draw_replicate_mean_psi`   | ✅ | — | — | ✅ | — |
| `_resolve_differential_model`| — | — | — | — | ✅ |
| `_draw_dpsi_samples` (dispatch) | — | ✅ | ✅ | — | ✅ |

Cell provenance: A8 `test_complete_arrays_rejects_inc_exc_length_mismatch` +
`test_auto_falls_back_to_sum_on_vector_length_mismatch`; A3
`test_complete_arrays_rejects_single_replicate`; A2/A10
`test_draw_replicate_mean_psi_all_zero_is_finite` +
`test_draw_replicate_mean_psi_extreme_ratio_is_finite`; F1
`test_resolve_model_rejects_invalid_value` +
`test_resolve_model_legacy_replicate_aware_flag` +
`test_no_replicate_aware_alias_resolves_to_sum`; rep-fail-fast (no partial output)
`test_rep_mode_writes_no_partial_output_on_error`; over-rejection guard
`test_complete_arrays_accepts_matched_pairs`.

## Verified behaviour
- inc/exc vector **length mismatch** or **single replicate** → `None` → `auto` falls
  back to the pooled `sum` model (finite output), `rep` raises a clear ValueError and
  writes **no partial TSV**.
- `_draw_replicate_mean_psi` is numerically safe: an all-zero replicate gives
  `Beta(0.5,0.5)` (no NaN/inf), and an extreme length ratio keeps PSI in [0,1].
- model resolver rejects unknown strings, honours the legacy `replicate_aware` flag,
  and the `--no-replicate-aware` alias resolves to `sum`.

## Residual risk (low, not defects — no fix applied)
- `_available_replicates` returns `max(len(inc), len(exc))` when exactly one side is
  empty, so the informational `ctrl_replicates`/`treat_replicates` column can report a
  count while the model fell back to `sum`. Cosmetic (reporting only; never affects the
  ΔPSI computation), and only reachable on a corrupted empty count cell that real rMATS
  does not emit.
- `auto`'s fallback to `sum` on a vector mismatch is silent (no log line). The output
  is correct (pooled estimate); surfacing it would be a convenience, not a correctness
  fix.
