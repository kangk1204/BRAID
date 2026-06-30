"""Direction control for ``braid differential`` (--swap-groups).

Default direction is ΔPSI = sample_1 - sample_2 = rMATS IncLevelDifference. The
``--swap-groups`` flag reverses it to sample_2 - sample_1 (treatment - control),
mirroring the head-to-head benchmark's ``event_counts(swap_groups=...)``. The flip
must be sign-consistent across BRAID's dpsi, the reported rmats_dpsi, and the
ctrl/treat PSI columns, while direction-agnostic fields (tier, caller_significant)
stay unchanged.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from braid.commands.differential import run_differential

_HEADER = [
    "ID", "GeneID", "geneSymbol", "chr", "strand", "exonStart_0base", "exonEnd",
    "upstreamES", "upstreamEE", "downstreamES", "downstreamEE",
    "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
    "IncFormLen", "SkipFormLen", "PValue", "FDR",
    "IncLevel1", "IncLevel2", "IncLevelDifference",
]


def _write_se_table(rmats_dir: Path) -> None:
    rmats_dir.mkdir(parents=True, exist_ok=True)
    # sample_1 PSI ~0.8, sample_2 PSI ~0.3 -> IncLevelDifference = +0.5, significant.
    row = [
        "1", "GENE", "GENE", "chr1", "+", "100", "200", "0", "100", "200", "300",
        "80,80,80", "20,20,20", "30,30,30", "70,70,70",
        "100", "100", "0.001", "0.001", "0.8,0.8,0.8", "0.3,0.3,0.3", "0.5",
    ]
    (rmats_dir / "SE.MATS.JC.txt").write_text(
        "\t".join(_HEADER) + "\n" + "\t".join(row) + "\n"
    )


def _args(rmats_dir: Path, out: Path, *, swap_groups: bool) -> argparse.Namespace:
    return argparse.Namespace(
        rmats_dir=str(rmats_dir), output=str(out), replicates=4000, confidence=0.95,
        fdr=0.05, effect_cutoff=0.1, min_support=20, seed=42, verbose=False,
        ctrl=None, treat=None, use_conformal=True, calibration=None,
        differential_model="auto", swap_groups=swap_groups,
    )


def _row(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))[0]


def test_swap_groups_flips_dpsi_and_rmats_dpsi_sign(tmp_path: Path) -> None:
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)
    noswap = tmp_path / "noswap.tsv"
    swap = tmp_path / "swap.tsv"
    run_differential(_args(rmats_dir, noswap, swap_groups=False))
    run_differential(_args(rmats_dir, swap, swap_groups=True))

    a, b = _row(noswap), _row(swap)
    da, db = float(a["dpsi"]), float(b["dpsi"])

    # Default direction matches rMATS IncLevelDifference (+0.5); swap reverses it.
    assert da > 0.4
    assert db < -0.4
    # The two dpsi are mirror images (within posterior noise).
    assert abs(da + db) < 0.05
    # rmats_dpsi is negated under swap so it stays sign-consistent with dpsi.
    assert float(a["rmats_dpsi"]) > 0
    assert float(b["rmats_dpsi"]) < 0
    assert abs(float(a["rmats_dpsi"]) + float(b["rmats_dpsi"])) < 1e-9


def test_swap_groups_swaps_ctrl_treat_psi_columns(tmp_path: Path) -> None:
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)
    noswap = tmp_path / "noswap.tsv"
    swap = tmp_path / "swap.tsv"
    run_differential(_args(rmats_dir, noswap, swap_groups=False))
    run_differential(_args(rmats_dir, swap, swap_groups=True))

    a, b = _row(noswap), _row(swap)
    # Default: ctrl = sample_1 (high PSI ~0.8) > treat (sample_2 ~0.3).
    assert float(a["ctrl_psi"]) > float(a["treat_psi"])
    # Swap: ctrl = sample_2 (~0.3) < treat = sample_1 (~0.8). The columns swap.
    assert float(b["ctrl_psi"]) < float(b["treat_psi"])
    assert abs(float(a["ctrl_psi"]) - float(b["treat_psi"])) < 0.05
    assert abs(float(a["treat_psi"]) - float(b["ctrl_psi"])) < 0.05
    # ctrl/treat SUPPORT also swaps (sample_1 total != sample_2 total here).
    assert a["ctrl_support"] == b["treat_support"]
    assert a["treat_support"] == b["ctrl_support"]


def test_swap_groups_leaves_direction_agnostic_fields_unchanged(tmp_path: Path) -> None:
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)
    noswap = tmp_path / "noswap.tsv"
    swap = tmp_path / "swap.tsv"
    run_differential(_args(rmats_dir, noswap, swap_groups=False))
    run_differential(_args(rmats_dir, swap, swap_groups=True))

    a, b = _row(noswap), _row(swap)
    # |ΔPSI|, reliability (interval excludes 0), the tier, and the FDR-based caller
    # flag do not depend on direction.
    assert a["tier"] == b["tier"]
    assert a["reliable"] == b["reliable"]
    assert a["caller_significant"] == b["caller_significant"]
    assert a["rmats_fdr"] == b["rmats_fdr"]


def test_default_is_unswapped_when_flag_absent(tmp_path: Path) -> None:
    """getattr default: an args object without swap_groups behaves as no-swap."""
    rmats_dir = tmp_path / "rmats"
    _write_se_table(rmats_dir)
    out = tmp_path / "out.tsv"
    args = _args(rmats_dir, out, swap_groups=False)
    del args.swap_groups  # simulate an older caller that never set the attribute
    run_differential(args)
    assert float(_row(out)["dpsi"]) > 0.4  # default sample_1 - sample_2 direction


def test_swap_groups_rep_error_message_labels_track_true_groups() -> None:
    """Under --swap-groups, the rep-model 'too few replicates' error must label the
    informative counts by their true rMATS group (not the literal sample_1/sample_2),
    so a user is not misled about which group is replicate-short."""
    import numpy as np
    import pytest

    from braid.commands.differential import _draw_dpsi_samples
    from braid.target.rmats_bootstrap import RmatsEvent

    # sample_1 has 2 informative replicate pairs, sample_2 has 1 -> rep errors.
    ev = RmatsEvent(
        event_id="EV1", event_type="SE", chrom="1", strand="+", gene="g",
        inc_count=0, exc_count=0, rmats_psi=0.5, rmats_fdr=0.01, rmats_dpsi=0.1,
        sample_1_inc_replicates=(10, 12), sample_1_exc_replicates=(5, 6),
        sample_2_inc_replicates=(8,), sample_2_exc_replicates=(4,),
    )
    with pytest.raises(ValueError) as exc:
        _draw_dpsi_samples(
            ev, ctrl_inc=22, ctrl_exc=11, treat_inc=8, treat_exc=4, ratio=1.0,
            draws=16, rng=np.random.default_rng(0), differential_model="rep",
            ctrl_sample="sample_2", treat_sample="sample_1",  # the --swap-groups mapping
        )
    msg = str(exc.value)
    # The true informative counts are sample_1=2, sample_2=1; the message must say so.
    assert "sample_1=2" in msg
    assert "sample_2=1" in msg
