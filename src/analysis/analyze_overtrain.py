"""Overtraining and selection diagnostic for the leader-follower cell.

Reproduces Section 5.1 and Figure 4 (``paper/figures/diag_overlay_clean.png``) of
"Equal Accuracy, Unequal Cost: Channel Dependence in PatchTST under Controlled
Coupling". The experiment instruments one faithful training trajectory per
(mode, seed) at the leader-follower gamma = 0.6, C = 21 cell, logging validation
MSE several times per epoch and training past the early-stopping horizon for
seeds {42, 123, 456, 789, 1011}.

Two committed CSVs are read:

  results/results_overtrain_summary.csv
      Per-seed test MSE under each post-hoc selection rule (10 rows: 5 seeds x
      {CI, CD}). Source of every statistic in Section 5.1.
  results/results_overtrain_diag.csv
      Per-checkpoint validation curves (mode, seed, update, epoch, epoch_frac,
      is_boundary, lr, val_mse). Source of the overlay figure and the val-curve
      descriptives only.

Statistics reuse the shared paired-difference machinery in ``paired_stats``: the
matched unit is the per-seed difference d_s = MSE_CD - MSE_CI, summarised by a
grand mean with a 95% paired-t CI (df = 4, t = 2.776 at n = 5). Selection always
happens on validation MSE; the reported number is the test MSE at the selected
checkpoint. Test MSE is never used to choose a checkpoint.

The selection-rule -> summary-column map is fixed by the upstream logging schema
and is used verbatim; the column names do not describe their meaning and must not
be inferred:

  main early-stop    -> main_test_mse
  early-stop replay  -> test_earlystop
  best-val-min       -> test_upd_global
  capped U = 3150    -> test_upd_within_U
  cadence            -> cadence_gap (reported as a per-mode mean, not a CD-CI
                        difference)

Run from a clean checkout:

    python analyze_overtrain.py
    python analyze_overtrain.py --summary-csv path/to/summary.csv \\
                                --diag-csv path/to/diag.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import.

import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)
import pandas as pd  # noqa: E402

import paired_stats as ps  # noqa: E402  (sibling module; on sys.path when run as a script)

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "results"
_FIGURES_DIR = _ROOT / "paper" / "figures"

_SUMMARY_CSV = _RESULTS_DIR / "results_overtrain_summary.csv"
_DIAG_CSV = _RESULTS_DIR / "results_overtrain_diag.csv"
_FIG_PATH = _FIGURES_DIR / "diag_overlay_clean.png"

_BUDGET_UPDATES = 3150  # equal-compute budget U for the C = 21 leader-follower cell.
_SEED_ORDER: tuple[int, ...] = (42, 123, 456, 789, 1011)
_BASE_MODE = "CI"
_ALT_MODE = "CD"

# Selection rule -> summary column. Verbatim; do not infer from column names.
_RULE_COLUMNS: dict[str, str] = {
    "main early-stop": "main_test_mse",
    "early-stop replay": "test_earlystop",
    "best-val-min": "test_upd_global",
    "capped U=3150": "test_upd_within_U",
}

# Oracle targets from main.tex Section 5.1 (mean, ci_lo, ci_hi), 4 decimal places.
_ORACLE_RULES: dict[str, tuple[float, float, float]] = {
    "main early-stop": (0.0079, 0.0046, 0.0113),
    "early-stop replay": (-0.0019, -0.0105, 0.0066),
    "best-val-min": (-0.0031, -0.0109, 0.0046),
    "capped U=3150": (0.0059, 0.0003, 0.0114),
}
_ORACLE_CADENCE: dict[str, float] = {"CD": 0.0005, "CI": -0.0007}

_SUMMARY_REQUIRED: set[str] = {"mode", "seed", "cadence_gap", *_RULE_COLUMNS.values()}
_DIAG_REQUIRED: set[str] = {"mode", "seed", "update", "epoch", "epoch_frac", "val_mse"}


def load_summary(path: Path) -> pd.DataFrame:
    """Load and validate the per-seed selection-rule summary.

    Args:
        path: Path to results_overtrain_summary.csv.

    Returns:
        The summary frame, unmodified.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If required columns, modes, or the balanced 5-seed design are
            missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    df = pd.read_csv(path)
    missing = _SUMMARY_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Summary CSV missing columns: {sorted(missing)}")
    modes = set(df["mode"].unique())
    if {_BASE_MODE, _ALT_MODE} - modes:
        raise ValueError(f"Summary CSV must contain modes {{{_BASE_MODE}, {_ALT_MODE}}}; found {sorted(modes)}")
    for mode in (_BASE_MODE, _ALT_MODE):
        seeds = sorted(df.loc[df["mode"] == mode, "seed"].unique())
        if seeds != sorted(_SEED_ORDER):
            raise ValueError(f"{mode} seeds {seeds} do not match the expected {sorted(_SEED_ORDER)}")
    return df


def load_diag(path: Path) -> pd.DataFrame:
    """Load and validate the per-checkpoint validation curves.

    Args:
        path: Path to results_overtrain_diag.csv.

    Returns:
        The validation-curve frame, unmodified.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Diagnostic CSV not found: {path}")
    df = pd.read_csv(path)
    missing = _DIAG_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Diagnostic CSV missing columns: {sorted(missing)}")
    return df


def selection_stats(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-seed paired CD-CI grand means under each selection rule.

    Args:
        summary: Output of load_summary.

    Returns:
        Mapping rule name -> grand_mean_diff result (mean, se, ci_lo, ci_hi,
        base_mean, n). The difference is CD - CI on the rule's mapped column.
    """
    out: dict[str, dict[str, float]] = {}
    for rule, column in _RULE_COLUMNS.items():
        diff = ps.make_diff_frame(summary, cell_cols=[], mode_base=_BASE_MODE, mode_alt=_ALT_MODE, value=column)
        out[rule] = ps.grand_mean_diff(diff)
    return out


def cadence_means(summary: pd.DataFrame) -> dict[str, float]:
    """Per-mode mean of the boundary-vs-update cadence gap.

    The cadence gap is already a within-mode quantity (best epoch boundary minus
    best mid-epoch update), so it is averaged per mode rather than differenced
    across modes.

    Args:
        summary: Output of load_summary.

    Returns:
        Mapping mode -> mean cadence_gap, for CD and CI.
    """
    means = summary.groupby("mode")["cadence_gap"].mean()
    return {mode: float(means[mode]) for mode in (_ALT_MODE, _BASE_MODE)}


def steps_per_epoch(diag: pd.DataFrame, mode: str) -> int:
    """Infer updates per epoch for a mode from the logged update/epoch ratio.

    Args:
        diag: Output of load_diag.
        mode: "CD" or "CI".

    Returns:
        The rounded median of update / epoch_frac over checkpoints with
        epoch_frac > 0.
    """
    sub = diag[(diag["mode"] == mode) & (diag["epoch_frac"] > 0)]
    if sub.empty:
        raise ValueError(f"No positive-epoch checkpoints for mode {mode}; cannot infer steps per epoch.")
    return int(round(float((sub["update"] / sub["epoch_frac"]).median())))


def val_descriptives(diag: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Per-mode validation-curve descriptives averaged over seeds.

    For each mode: the mean over seeds of each seed's minimum validation MSE, the
    epoch range over which those minima occur, and the mean over seeds of each
    seed's final validation MSE (at its largest logged epoch_frac).

    Args:
        diag: Output of load_diag.

    Returns:
        Mapping mode -> {val_min_mean, min_epoch_lo, min_epoch_hi, val_end_mean}.
    """
    out: dict[str, dict[str, float]] = {}
    for mode, grp in diag.groupby("mode"):
        min_idx = grp.groupby("seed")["val_mse"].idxmin()
        min_rows = grp.loc[min_idx]
        end_idx = grp.groupby("seed")["epoch_frac"].idxmax()
        end_rows = grp.loc[end_idx]
        out[str(mode)] = {
            "val_min_mean": float(min_rows["val_mse"].mean()),
            "min_epoch_lo": int(min_rows["epoch"].min()),
            "min_epoch_hi": int(min_rows["epoch"].max()),
            "val_end_mean": float(end_rows["val_mse"].mean()),
        }
    return out


def build_overlay(diag: pd.DataFrame, out_path: Path, budget_updates: int = _BUDGET_UPDATES) -> Path:
    """Render the two-panel validation-overlay figure (CD left, CI right).

    One line per seed (val_mse versus epoch via epoch_frac), a star at each seed's
    validation minimum, and a dashed vertical line on the CD panel at the
    equal-compute budget expressed in CD epochs.

    Args:
        diag: Output of load_diag.
        out_path: Destination PNG path.
        budget_updates: Equal-compute budget U in optimiser updates.

    Returns:
        out_path.
    """
    panels = (_ALT_MODE, _BASE_MODE)
    titles = {_ALT_MODE: "CD (channel-dependent)", _BASE_MODE: "CI (channel-independent)"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)

    for ax, mode in zip(axes, panels):
        mode_df = diag[diag["mode"] == mode]
        for seed in _SEED_ORDER:
            curve = mode_df[mode_df["seed"] == seed].sort_values("epoch_frac")
            if curve.empty:
                continue
            line = ax.plot(curve["epoch_frac"], curve["val_mse"], linewidth=1.2, label=f"seed {seed}")[0]
            argmin = curve["val_mse"].idxmin()
            ax.plot(
                curve.loc[argmin, "epoch_frac"],
                curve.loc[argmin, "val_mse"],
                marker="*",
                markersize=12,
                color=line.get_color(),
                markeredgecolor="black",
                markeredgewidth=0.4,
                linestyle="none",
                zorder=5,
            )
        if mode == _ALT_MODE:
            budget_epoch = budget_updates / steps_per_epoch(diag, _ALT_MODE)
            ax.axvline(
                budget_epoch,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"$U={budget_updates}$ budget",
            )
        ax.set_title(titles[mode])
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    axes[0].set_ylabel("validation MSE")
    fig.suptitle("Validation-MSE trajectories, leader-follower $\\gamma=0.6$, $C=21$")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _format_check(stats: dict[str, dict[str, float]], cadence: dict[str, float]) -> tuple[list[str], bool]:
    """Build the target-vs-reproduced check table for the five Section 5.1 lines.

    Args:
        stats: Output of selection_stats.
        cadence: Output of cadence_means.

    Returns:
        (lines, all_pass). Comparison is to 4 decimal places, the precision at
        which main.tex prints these numbers.
    """
    lines: list[str] = []
    all_pass = True
    for rule, (o_mean, o_lo, o_hi) in _ORACLE_RULES.items():
        res = stats[rule]
        got = (round(res["mean"], 4), round(res["ci_lo"], 4), round(res["ci_hi"], 4))
        ok = got == (o_mean, o_lo, o_hi)
        all_pass = all_pass and ok
        lines.append(
            f"  [{'PASS' if ok else 'FAIL'}] {rule:18s} "
            f"CD-CI {got[0]:+.4f} CI [{got[1]:+.4f}, {got[2]:+.4f}]  "
            f"target {o_mean:+.4f} [{o_lo:+.4f}, {o_hi:+.4f}]"
        )
    for mode, o_val in _ORACLE_CADENCE.items():
        got = round(cadence[mode], 4)
        ok = got == o_val
        all_pass = all_pass and ok
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] cadence {mode:14s} {got:+.4f}  target {o_val:+.4f}")
    return lines, all_pass


def main(argv: list[str] | None = None) -> int:
    """Reproduce Section 5.1 statistics and regenerate the overlay figure.

    Args:
        argv: Optional argument vector (defaults to sys.argv).

    Returns:
        Process exit code: 0 if every statistic matches the oracle, else 1.
    """
    parser = argparse.ArgumentParser(description="Overtraining/selection diagnostic (Section 5.1, Figure 4).")
    parser.add_argument("--summary-csv", type=Path, default=_SUMMARY_CSV, help="Per-seed selection-rule summary CSV.")
    parser.add_argument("--diag-csv", type=Path, default=_DIAG_CSV, help="Per-checkpoint validation-curve CSV.")
    parser.add_argument("--fig", type=Path, default=_FIG_PATH, help="Output PNG for the overlay figure.")
    args = parser.parse_args(argv)

    summary = load_summary(args.summary_csv)
    diag = load_diag(args.diag_csv)

    stats = selection_stats(summary)
    cadence = cadence_means(summary)
    descriptives = val_descriptives(diag)

    print("=== SECTION 5.1 SELECTION-RULE EFFECT SIZES (CD - CI, n=5, df=4) ===")
    for rule, column in _RULE_COLUMNS.items():
        res = stats[rule]
        print(f"  {rule:18s} [{column:18s}] {res['mean']:+.4f}  " f"95% CI [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]")
    print(f"  cadence (per-mode mean) CD {cadence['CD']:+.4f}  CI {cadence['CI']:+.4f}")

    print("\n=== VALIDATION-CURVE DESCRIPTIVES (mean over seeds) ===")
    for mode in (_ALT_MODE, _BASE_MODE):
        d = descriptives[mode]
        print(
            f"  {mode}: val_min {d['val_min_mean']:.3f} (epoch {d['min_epoch_lo']}-{d['min_epoch_hi']}) "
            f"-> val_end {d['val_end_mean']:.3f}"
        )

    fig_path = build_overlay(diag, args.fig)
    print(f"\nFigure written: {fig_path}")

    print("\n=== ORACLE CHECK (4 dp vs main.tex Section 5.1) ===")
    check_lines, all_pass = _format_check(stats, cadence)
    print("\n".join(check_lines))
    print(f"\nRESULT: {'PASS - all five lines reproduce' if all_pass else 'FAIL - see lines above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
