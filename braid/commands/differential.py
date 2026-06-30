"""BRAID differential subcommand: two-group ΔPSI confidence intervals.

Usage:
    braid differential --rmats-dir rMATS_output/ -o tiers.tsv

Group convention (IMPORTANT): PSI counts come from the rMATS tables, not from any
BAMs, so the control/treatment mapping is fixed entirely by how rMATS was run.
BRAID maps **control = rMATS sample_1 (rMATS
``--b1``)** and **treatment = rMATS sample_2 (rMATS ``--b2``)**. ΔPSI is reported
as control − treatment = sample_1 − sample_2, matching rMATS IncLevelDifference
(so ``dpsi_mean`` and ``rmats_dpsi`` always carry the same sign). Run rMATS with
control as ``--b1`` so the ``ctrl_psi``/``treat_psi`` output columns are labelled
as intended.

If you cannot (or do not want to) re-run rMATS with the groups in that order, pass
``--swap-groups`` to reverse the direction at the BRAID layer: control becomes rMATS
sample_2 and ΔPSI is reported as sample_2 − sample_1 (treatment − control). The
reported ``rmats_dpsi`` is negated so it stays sign-consistent with the swapped
``dpsi``; FDR-based ``caller_significant`` is direction-agnostic and unchanged. This
is the same orientation the head-to-head benchmark applies via
``event_counts(swap_groups=...)`` for datasets whose RT-PCR truth runs the opposite
way (TRA2/GSE59335, circadian/GSE54651).
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from braid.adapters.base import confidence_tier
from braid.output_safety import csv_safe

logger = logging.getLogger(__name__)

_DIFFERENTIAL_MODEL_CHOICES = {"auto", "rep", "sum"}


def _resolve_differential_calibrator(args: argparse.Namespace):
    """Resolve the ΔPSI conformal calibrator (None when disabled/unavailable).

    Returns ``None`` if ``--no-conformal`` was passed or the packaged artifact is
    missing (in which case the differential path falls back to the Jeffreys-Beta
    posterior percentile interval).
    """
    if not getattr(args, "use_conformal", True):
        if getattr(args, "calibration", None):
            logger.warning(
                "--calibration %s is ignored because --no-conformal disables the "
                "conformal layer; drop one of the two conflicting options.",
                args.calibration,
            )
        return None
    from braid.target.conformal import (
        ConformalCalibrator,
        load_differential_conformal_calibrator,
        require_scale_kind,
    )
    path = getattr(args, "calibration", None)
    if path:
        try:
            cal = ConformalCalibrator.from_json(path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"--calibration file not found: {path}") from exc
        except OSError as exc:
            # A directory, a permission error, etc. (FileNotFoundError is handled
            # above; this catches the other unreadable-path cases so the operator
            # gets a --calibration-scoped message, not a raw IsADirectoryError).
            raise OSError(f"--calibration path could not be read: {path} ({exc})") from exc
        except (ValueError, KeyError, TypeError) as exc:
            # json.JSONDecodeError subclasses ValueError; KeyError/TypeError surface
            # from a valid-JSON-but-wrong-schema artifact (e.g. missing 'alpha').
            # Re-raise with the --calibration scope so the operator knows which file
            # is bad, instead of a raw decoder/KeyError traceback.
            raise ValueError(
                f"--calibration {path} is not a valid ConformalCalibrator JSON: {exc}"
            ) from exc
        return require_scale_kind(cal, "absolute_dpsi", f"--calibration {path}")
    try:
        return load_differential_conformal_calibrator()
    except (FileNotFoundError, OSError):
        logger.warning(
            "Differential conformal calibrator artifact not found; "
            "falling back to the Jeffreys-Beta percentile interval."
        )
        return None


def _replicate_vectors(ev, sample: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return inclusion/exclusion count vectors for an rMATS sample group."""
    if sample == "sample_1":
        return ev.sample_1_inc_replicates, ev.sample_1_exc_replicates
    if sample == "sample_2":
        return ev.sample_2_inc_replicates, ev.sample_2_exc_replicates
    raise ValueError(f"unknown rMATS sample group: {sample}")


def _available_replicates(ev, sample: str) -> int:
    """Best-effort replicate count for reporting, even when vectors are incomplete."""
    inc, exc = _replicate_vectors(ev, sample)
    if inc and exc:
        return min(len(inc), len(exc))
    return max(len(inc), len(exc))


def _informative_replicates(ev, sample: str) -> int:
    """Replicate pairs that carry reads (the count the rep model can actually use).

    Counts matched-length replicate pairs with inc + exc > 0; zero-support pairs are
    excluded, mirroring the drop in :func:`_complete_replicate_arrays`.
    """
    inc, exc = _replicate_vectors(ev, sample)
    if len(inc) != len(exc):
        return 0
    return int(sum(1 for a, b in zip(inc, exc) if a + b > 0))


def _complete_replicate_arrays(
    ev, sample: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return complete, informative replicate arrays, or None for pooled fallback.

    A replicate with no reads (inc + exc == 0) carries no PSI information, so it is
    dropped rather than entering the equal-weight replicate mean as a spurious
    Beta(0.5, 0.5) ~ 0.5 draw. Fewer than two informative replicates -> None (the
    caller then falls back to the pooled sum model, or errors in rep mode).
    """
    inc, exc = _replicate_vectors(ev, sample)
    if len(inc) != len(exc) or len(inc) < 2:
        return None
    inc_a = np.asarray(inc, dtype=float)
    exc_a = np.asarray(exc, dtype=float)
    informative = (inc_a + exc_a) > 0.0
    inc_a, exc_a = inc_a[informative], exc_a[informative]
    if inc_a.size < 2:
        return None
    return inc_a, exc_a


def _draw_replicate_mean_psi(
    inc_reps: np.ndarray,
    exc_reps: np.ndarray,
    *,
    ratio: float,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw group mean PSI while preserving biological-replicate structure."""
    n_reps = int(inc_reps.size)
    idx = rng.integers(0, n_reps, size=(draws, n_reps))
    inc = inc_reps[idx]
    exc = (exc_reps * ratio)[idx]
    return rng.beta(inc + 0.5, exc + 0.5).mean(axis=1)


def _draw_dpsi_samples(
    ev,
    *,
    ctrl_inc: int,
    ctrl_exc: int,
    treat_inc: int,
    treat_exc: int,
    ratio: float,
    draws: int,
    rng: np.random.Generator,
    differential_model: str,
    ctrl_sample: str = "sample_1",
    treat_sample: str = "sample_2",
) -> tuple[np.ndarray, float, float, str, int, int]:
    """Draw ΔPSI samples and per-group PSI points for the requested model.

    Returns ``(dpsi_samples, ctrl_psi, treat_psi, model, ctrl_reps, treat_reps)``.
    ``ctrl_psi``/``treat_psi`` are the per-group posterior means of the SAME draws,
    so the reported ``dpsi == ctrl_psi - treat_psi`` for every model -- the
    equal-weight rep mean and the pooled sum mean each stay internally consistent
    with their own ΔPSI (no estimand mismatch between the columns).

    ``ctrl_sample``/``treat_sample`` select which rMATS group is the ΔPSI minuend
    (control) and subtrahend (treatment). The default (``sample_1``/``sample_2``)
    yields ΔPSI = sample_1 - sample_2 = rMATS IncLevelDifference; ``--swap-groups``
    passes them reversed so the counts AND the replicate vectors are pulled from the
    intended group -- the caller also negates the reported ``rmats_dpsi`` so its sign
    keeps matching BRAID's ``dpsi``.
    """
    ctrl_n = _available_replicates(ev, ctrl_sample)
    treat_n = _available_replicates(ev, treat_sample)

    if differential_model in {"auto", "rep"}:
        ctrl_reps = _complete_replicate_arrays(ev, ctrl_sample)
        treat_reps = _complete_replicate_arrays(ev, treat_sample)
        if ctrl_reps is not None and treat_reps is not None:
            ctrl_samples = _draw_replicate_mean_psi(
                ctrl_reps[0], ctrl_reps[1], ratio=ratio, draws=draws, rng=rng
            )
            treat_samples = _draw_replicate_mean_psi(
                treat_reps[0], treat_reps[1], ratio=ratio, draws=draws, rng=rng
            )
            return (
                ctrl_samples - treat_samples,
                float(ctrl_samples.mean()), float(treat_samples.mean()),
                "rep", int(ctrl_reps[0].size), int(treat_reps[0].size),
            )
        if differential_model == "rep":
            ctrl_inf = _informative_replicates(ev, ctrl_sample)
            treat_inf = _informative_replicates(ev, treat_sample)
            raise ValueError(
                "differential_model=rep requires at least two complete, informative "
                f"inclusion/exclusion replicate count pairs in both groups for "
                f"{ev.event_id} (informative {ctrl_sample}={ctrl_inf}, "
                f"{treat_sample}={treat_inf}; raw vectors ctrl={ctrl_n}/treat={treat_n}). "
                "Labels track the control/treatment mapping, which --swap-groups reverses."
            )

    ctrl_exc_n = ctrl_exc * ratio
    treat_exc_n = treat_exc * ratio
    ctrl_samples = rng.beta(ctrl_inc + 0.5, ctrl_exc_n + 0.5, size=draws)
    treat_samples = rng.beta(treat_inc + 0.5, treat_exc_n + 0.5, size=draws)
    return (
        ctrl_samples - treat_samples,
        float(ctrl_samples.mean()), float(treat_samples.mean()),
        "sum", ctrl_n, treat_n,
    )


def _resolve_differential_model(args: argparse.Namespace) -> str:
    """Resolve differential model while preserving the old programmatic flag."""
    mode = getattr(args, "differential_model", None)
    if mode is None:
        mode = "auto" if getattr(args, "replicate_aware", True) else "sum"
    if mode not in _DIFFERENTIAL_MODEL_CHOICES:
        choices = ", ".join(sorted(_DIFFERENTIAL_MODEL_CHOICES))
        raise ValueError(f"differential_model must be one of {{{choices}}}, got {mode!r}")
    return mode


def add_differential_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``differential`` subcommand."""
    parser = subparsers.add_parser(
        "differential",
        aliases=["diff"],
        help="Main rMATS workflow: two-group calibrated ΔPSI intervals + tiers.",
        description=(
            "The main rMATS two-group workflow. Post-processes rMATS output, "
            "bootstraps a posterior ΔPSI from the junction counts, uses rMATS "
            "per-replicate count vectors when available, and writes the "
            "most detailed rMATS-specific table: the canonical columns shared with "
            "`braid filter` (dpsi, ci_low, ci_high, reliable, caller_significant, "
            "tier) plus rMATS-specific detail (prob_large_effect, ctrl_psi/treat_psi, "
            "per-group support, raw rmats_dpsi). For a unified cross-caller report "
            "with Excel + figure, use `braid filter` instead."
        ),
    )
    parser.add_argument(
        "--rmats-dir", required=True,
        help="rMATS output directory.",
    )
    parser.add_argument(
        "-o", "--output", default="braid_differential.tsv",
        help="Output TSV (default: braid_differential.tsv).",
    )
    parser.add_argument(
        "--replicates", type=int, default=500,
        help="Posterior draws, not biological replicate count (default: 500).",
    )
    parser.add_argument(
        "--differential-model",
        choices=sorted(_DIFFERENTIAL_MODEL_CHOICES),
        default="auto",
        help="ΔPSI posterior model: auto uses per-replicate rMATS vectors when "
             "complete and otherwise falls back to sum; rep requires the "
             "per-replicate model; sum uses legacy pooled group counts "
             "(default: auto).",
    )
    parser.add_argument(
        "--no-replicate-aware",
        dest="differential_model",
        action="store_const",
        const="sum",
        help="Compatibility alias for --differential-model sum.",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level for the legacy --no-conformal percentile fallback "
             "(default: 0.95). In default conformal mode, interval alpha comes "
             "from the calibrator JSON.",
    )
    parser.add_argument(
        "--fdr", type=float, default=0.05,
        help="rMATS FDR significance threshold (default: 0.05).",
    )
    parser.add_argument(
        "--effect-cutoff", type=float, default=0.1,
        help="|ΔPSI| effect-size cutoff (default: 0.1).",
    )
    parser.add_argument(
        "--min-support", type=int, default=20,
        help="Minimum total support per group (default: 20).",
    )
    parser.add_argument(
        "--swap-groups", action="store_true",
        help="Reverse the ΔPSI direction: report sample_2 - sample_1 (treatment - "
             "control) instead of the default sample_1 - sample_2 (control - "
             "treatment = rMATS IncLevelDifference). Use when your RT-PCR/ground-truth "
             "convention is the opposite of how rMATS was run (--b1/--b2). rmats_dpsi "
             "is negated in the output so it stays sign-consistent with BRAID's dpsi; "
             "FDR-based significance is direction-agnostic and unchanged.",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail on the first malformed rMATS row instead of skipping it "
             "(data-integrity mode).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--no-conformal", dest="use_conformal", action="store_false",
        help="Disable the conformal ΔPSI calibration layer and use the raw "
             "Jeffreys-Beta posterior percentile interval instead.",
    )
    parser.add_argument(
        "--calibration", default=None, metavar="PATH",
        help="Path to a ΔPSI ConformalCalibrator JSON (default: packaged "
             "differential calibrator fit on real RT-PCR residuals).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output.",
    )
    parser.set_defaults(func=run_differential, use_conformal=True)


def run_differential(args: argparse.Namespace) -> None:
    """Execute the differential subcommand."""
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.replicates < 1:
        raise ValueError(f"replicates must be >= 1, got {args.replicates}")
    if not 0.0 < args.confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {args.confidence}")
    fdr_threshold = getattr(args, "fdr", 0.05)
    if not 0.0 <= fdr_threshold <= 1.0:
        raise ValueError(f"fdr must be in [0, 1], got {fdr_threshold}")
    if args.min_support < 0:
        raise ValueError(f"min_support must be >= 0, got {args.min_support}")
    if getattr(args, "effect_cutoff", 0.1) < 0.0:
        # A negative |ΔPSI| cutoff makes abs(dpsi) >= cutoff vacuously true for every
        # event, silently promoting tiers (not-significant -> supported). Reject it
        # here, matching `braid filter` (filter_cmd.py).
        raise ValueError(f"effect-cutoff must be >= 0, got {args.effect_cutoff}")

    from braid.commands.rmats_input import require_rmats_tables
    from braid.target.rmats_bootstrap import (
        get_group_counts,
        parse_rmats_output,
    )

    logger.info(
        "Note: PSI computed from rMATS junction count tables, not directly from BAM."
    )
    logger.info(
        "Group convention: control = rMATS sample_1 (--b1), treatment = rMATS "
        "sample_2 (--b2); ΔPSI = control − treatment (matches rMATS "
        "IncLevelDifference). Ensure rMATS was run with control as --b1, or use "
        "--swap-groups to reverse the direction at the BRAID layer."
    )

    require_rmats_tables(args.rmats_dir, logger)
    events = parse_rmats_output(
        args.rmats_dir,
        min_total_count=args.min_support,
        strict=getattr(args, "strict", False),
    )
    logger.info("Parsed %d rMATS events", len(events))

    calibrator = _resolve_differential_calibrator(args)
    if calibrator is not None:
        logger.info(
            "Differential ΔPSI conformal calibration ON (scope=%s); the conformal "
            "residual quantile carries the distribution-free finite-sample guarantee "
            "under exchangeability, combined with a conservative depth-robust "
            "sampling term.",
            calibrator.training_scope,
        )
    else:
        logger.info("Differential ΔPSI conformal calibration OFF (Jeffreys percentile).")

    requested_differential_model = _resolve_differential_model(args)
    # Only mention the conformal calibrator when it is actually in use; under
    # --no-conformal (calibrator is None) the rep/auto path uses the raw Jeffreys
    # percentile, so a "calibrated by transfer" note would misstate the state.
    transfer_note = (
        " Events that resolve to the rep model use the pooled-fit conformal "
        "calibrator by transfer (not a rep-specific refit); sum is the estimator the "
        "coverage claims rest on."
        if calibrator is not None
        else ""
    )
    if requested_differential_model == "auto":
        logger.info(
            "Differential model auto: use rMATS per-replicate count vectors when "
            "both groups have at least two complete replicate pairs; otherwise "
            "fall back to pooled group sums.%s", transfer_note,
        )
    elif requested_differential_model == "rep":
        logger.info(
            "Differential model rep: require per-replicate rMATS count vectors.%s",
            transfer_note,
        )
    else:
        logger.info("Differential model sum: use pooled group counts.")

    alpha = 1.0 - args.confidence
    rng = np.random.default_rng(args.seed)
    results = []

    # Direction: by default control = rMATS sample_1 (--b1) and ΔPSI = sample_1 -
    # sample_2 = rMATS IncLevelDifference. --swap-groups reverses it (control =
    # sample_2), for when the ground-truth/RT-PCR convention is treatment - control.
    swap_groups = getattr(args, "swap_groups", False)
    ctrl_sample = "sample_2" if swap_groups else "sample_1"
    treat_sample = "sample_1" if swap_groups else "sample_2"
    if swap_groups:
        logger.info(
            "Direction: --swap-groups ON -> control = rMATS sample_2 (--b2), "
            "treatment = rMATS sample_1 (--b1); ΔPSI = sample_2 - sample_1 "
            "(rmats_dpsi negated to stay sign-consistent with BRAID dpsi)."
        )

    # Differential uses rMATS sample_1/sample_2 count vectors when biological
    # replicates are present; single-replicate or incomplete-vector rows keep the
    # legacy pooled-count posterior so older and sparse rMATS tables remain valid.
    for ev in events:
        ctrl_inc, ctrl_exc = get_group_counts(ev, sample=ctrl_sample)
        treat_inc, treat_exc = get_group_counts(ev, sample=treat_sample)

        # Raw read totals (the actual evidence) gate inclusion and choose the
        # conformal support bin.
        ctrl_total = ctrl_inc + ctrl_exc
        treat_total = treat_inc + treat_exc

        if ctrl_total < args.min_support or treat_total < args.min_support:
            continue

        # Length-normalize the skipping counts to the molecular-PSI (rMATS IncLevel)
        # scale: PSI = (inc/IncFormLen) / (inc/IncFormLen + exc/SkipFormLen). Keeping
        # the inclusion count and rescaling exclusion by IncFormLen/SkipFormLen
        # preserves read evidence while matching the IncLevel mean. Default form
        # lengths are 1.0/1.0 (identity) for fixtures that omit them.
        ratio = ev.inc_form_len / ev.skip_form_len if ev.skip_form_len > 0 else 1.0

        # Per-group PSI posteriors. We deliberately use concentrated Jeffreys-Beta
        # posteriors -- Beta(inc + 1/2, exc + 1/2) -- here, NOT the single-group
        # coverage-tuned overdispersed posterior. The overdispersed posterior is so
        # wide (scale ~0.01) that the difference of two near-uniform draws almost
        # never excludes zero, so genuine large effects would be missed. Biological
        # replicate heterogeneity is represented by resampling the rMATS replicate
        # count vectors when they are complete in both groups.
        group_rng = np.random.default_rng(int(rng.integers(0, 2**31)))

        # ΔPSI posterior. Convention: sample_1 - sample_2 (control - treatment),
        # matching rMATS IncLevelDifference (IncLevel1 - IncLevel2), so that
        # dpsi_mean and rmats_dpsi carry the same sign.
        (dpsi_samples, ctrl_psi, treat_psi, event_differential_model,
         ctrl_replicates, treat_replicates) = (
            _draw_dpsi_samples(
                ev,
                ctrl_inc=ctrl_inc,
                ctrl_exc=ctrl_exc,
                treat_inc=treat_inc,
                treat_exc=treat_exc,
                ratio=ratio,
                draws=args.replicates,
                rng=group_rng,
                differential_model=requested_differential_model,
                ctrl_sample=ctrl_sample,
                treat_sample=treat_sample,
            )
        )
        dpsi_mean = float(np.mean(dpsi_samples))
        dpsi_std = float(np.std(dpsi_samples))
        prob_large = float(np.mean(np.abs(dpsi_samples) >= args.effect_cutoff))

        if calibrator is not None:
            # Depth-robust conformal ΔPSI interval: the calibrated half-width q covers
            # the orthogonal-truth residual floor (fit on real RT-PCR residuals), combined
            # in quadrature with the within-sample sampling spread z*dpsi_std so that
            # coverage stays at nominal on lower-depth data without a refit (at fit
            # depth dpsi_std is negligible and this reduces to +/- q).
            dpsi_ci_low, dpsi_ci_high = calibrator.robust_interval(
                dpsi_mean, dpsi_std, float(ctrl_total + treat_total),
                event_type=ev.event_type, clip=(-1.0, 1.0),
            )
        else:
            dpsi_ci_low = float(np.percentile(dpsi_samples, 100 * alpha / 2))
            dpsi_ci_high = float(np.percentile(dpsi_samples, 100 * (1 - alpha / 2)))
        excludes_zero = dpsi_ci_low > 0 or dpsi_ci_high < 0

        # ctrl_psi / treat_psi come back from _draw_dpsi_samples as the per-group
        # posterior means of the SAME draws that produced dpsi (length-normalized,
        # IncLevel scale), so dpsi == ctrl_psi - treat_psi holds for every model
        # (rep equal-weight mean and sum pooled mean each stay self-consistent).

        # Tiers use the same confidence_tier() decision rule as `braid filter`
        # (braid/adapters/base.py). The commands can still diverge on borderline
        # rMATS events because `filter` uses the caller-native estimate while this
        # path uses the posterior-sampled center/interval. reliable = the calibrated
        # interval excludes 0; effect = |Delta PSI| >= cutoff; caller_significant =
        # rMATS FDR < threshold. (prob_large and min_support no longer gate the tier
        # -- min_support already pre-filters events above, and prob_large is reported
        # separately as prob_large_effect.)
        rmats_sig = (
            ev.rmats_fdr < fdr_threshold
            if np.isfinite(ev.rmats_fdr)
            else False
        )
        tier = confidence_tier(
            reliable=excludes_zero,
            effect=abs(dpsi_mean) >= args.effect_cutoff,
            caller_sig=rmats_sig,
        )

        results.append({
            "event_id": ev.event_id,
            "event_type": ev.event_type,
            "gene": ev.gene,
            "chrom": ev.chrom,
            "caller": "rmats",
            # Canonical columns shared with `braid filter`, so a user who learned the
            # filter schema can read this table directly. `dpsi` is BRAID's posterior
            # ΔPSI mean (the calibrated point estimate); `reliable` means the interval
            # excludes 0; `caller_significant` is the upstream rMATS FDR call. The raw
            # rMATS IncLevelDifference is preserved separately below as `rmats_dpsi`.
            "dpsi": dpsi_mean,
            "ci_low": dpsi_ci_low,
            "ci_high": dpsi_ci_high,
            "reliable": excludes_zero,
            "caller_significant": rmats_sig,
            "tier": tier,
            # rMATS-specific detail columns (no caller-agnostic equivalent).
            "ctrl_psi": ctrl_psi,
            "treat_psi": treat_psi,
            "prob_large_effect": prob_large,
            "rmats_fdr": ev.rmats_fdr,
            # Negate rMATS IncLevelDifference under --swap-groups so its sign keeps
            # matching BRAID's swapped dpsi (sample_2 - sample_1).
            "rmats_dpsi": -ev.rmats_dpsi if swap_groups else ev.rmats_dpsi,
            "ctrl_support": ctrl_total,
            "treat_support": treat_total,
            "differential_model": event_differential_model,
            "ctrl_replicates": ctrl_replicates,
            "treat_replicates": treat_replicates,
        })

    # Distribution-shift check: the conformal coverage guarantee is conditional on
    # exchangeability with the calibration set. Warn if the input read-support regime
    # is materially shallower than what the calibrator was fit on.
    if calibrator is not None and results:
        supports = np.array([r["ctrl_support"] + r["treat_support"] for r in results])
        ok, msg = calibrator.check_applicability(supports)
        if not ok:
            logger.warning("Calibration applicability: %s", msg)

    # Write output (always creates the file, even if empty)
    _write_differential_tsv(results, args.output)

    if not results:
        logger.warning(
            "No events passed filters (min_support=%d). "
            "Output file %s contains only the header.",
            args.min_support,
            args.output,
        )

    # Summary
    tiers: dict[str, int] = {}
    for r in results:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1

    # Always state the direction that was actually computed, at the point of use, so a
    # reader never has to guess (and is never misled by --ctrl/--treat).
    direction = (
        "sample_2 − sample_1 (treatment − control) [--swap-groups]"
        if swap_groups
        else "sample_1 − sample_2 (control − treatment, = rMATS IncLevelDifference)"
    )
    print(f"BRAID differential: {len(results)} events → {args.output}")
    print(f"  ΔPSI direction: {direction}")
    for tier_name in [
        "high-confidence", "supported", "caller-significant-only", "not-significant"
    ]:
        if tier_name in tiers:
            print(f"  {tier_name}: {tiers[tier_name]}")


def _write_differential_tsv(results: list[dict], output: str) -> None:
    """Write differential results to TSV.

    Always writes the header so that the output file exists even when
    *results* is empty.

    Args:
        results: List of result dictionaries.
        output: Path for the output TSV file.
    """
    cols = [
        # Canonical columns (shared with `braid filter`).
        "event_id", "event_type", "gene", "chrom", "caller",
        "dpsi", "ci_low", "ci_high", "reliable", "caller_significant", "tier",
        # rMATS-specific detail columns.
        "ctrl_psi", "treat_psi", "prob_large_effect",
        "rmats_fdr", "rmats_dpsi", "ctrl_support", "treat_support",
        "differential_model", "ctrl_replicates", "treat_replicates",
    ]
    with open(output, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in results:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, (bool, np.bool_)):
                    # Check bool before float: numpy bools are neither a Python
                    # bool nor a float, so without np.bool_ here a numpy flag would
                    # fall through to str() and serialize as "True"/"False"
                    # instead of "yes"/"no".
                    vals.append("yes" if v else "no")
                elif isinstance(v, float):
                    vals.append(f"{v:.6f}")
                else:
                    vals.append(csv_safe(str(v)))
            f.write("\t".join(vals) + "\n")
