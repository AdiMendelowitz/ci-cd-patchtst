"""C=84 three-arm lightweight-CD ablation for the AR(1) grid.

The analysis was specified before the underlying data existed, so the
methodology could not be shaped by which direction the numbers landed. Oracle values below were frozen at the first run against the
committed results file.

Reads a self-contained results file for the AR(1) grid at C=84, modes
{CI, CD, CD_Block}, 5 seeds each ({42, 123, 456, 789, 1011}),
pred_len=96. The launch covered the single cell rho=0.5, with all five
seeds generated at 14,400 timesteps (8,033 training windows). This is
deliberately not results_grid.csv: that file's existing C=84 CD/CI rows
were trained under the earlier environment at forced batch 1, and its
seeds 42/123/456 use a 13,400-timestep generation, so rows from the two
files are never paired. Seeds 789/1011 share the 14,400-timestep
generation across both files, and the CI rows for those two seeds
reproduce the committed grid values exactly.

CD_Block partition: necessarily neutral (contiguous groups), since the
AR(1) grid's compound-symmetry covariance has no leader-follower or
block structure for a partition to align with or against -- unlike the
leader-follower sweep's CD_Block, which preserves real pair structure.
This is a structural fact about the grid, not a modelling choice.

Outputs, matching analyze_block_attention.py's structure so the two
three-arm ablations in this paper report in a consistent shape: the
per-rho pivot (mean test MSE per arm, CD/CI, Blk/CI, Blk/CD ratios), the
three paired per-seed contrasts with 95% paired-t CIs (df=4) via the
shared paired_stats machinery, and per-arm best_epoch ranges.

ORACLE
------
Pinned values (pivot to 4dp, paired contrasts, sign patterns, exact
one-sided sign-test p for CD-CI, best_epoch ranges) were frozen at the
first run against the committed results file and are committed together
with it. The check runs on every invocation; rho cells present in the
file but absent from the oracle tables fail the run until frozen, so
future sweep extensions cannot enter reporting unverified.

Run from the repository root:

    python src/analysis/analyze_boundary_c84.py
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

import paired_stats as ps  # sibling module; on sys.path when run as a script

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSV = _ROOT / "results" / "Revision" / "train_grid_c84_block_attn" / "results_grid_C84_block_attn.csv"

_MODES: tuple[str, ...] = ("CI", "CD", "CD_Block")
_SEED_ORDER: tuple[int, ...] = (42, 123, 456, 789, 1011)
_C = 84
_PRED_LEN = 96
_REQUIRED_COLS: set[str] = {"C", "rho", "mode", "pred_len", "seed", "test_mse", "best_epoch", "batch_size"}
# Launch batch policy, asserted in load_results: CI at batch 32; CD and
# CD_Block at batch 8 over the flattened C*N=5292 token sequence.
_BATCH_POLICY: dict[str, int] = {"CI": 32, "CD": 8, "CD_Block": 8}


def load_results(path: Path, expected_rhos: set[float] | None = None) -> pd.DataFrame:
    """Load and validate the C=84 three-arm sweep results.

    Args:
        path: Path to the self-contained C=84 results CSV.
        expected_rhos: If given, the exact rho set the file must contain
            (e.g. {0.5} for a single-cell launch, or {0.1, 0.5, 0.9} for
            the full sweep). If None, whatever rho values are present are
            accepted and reported, deferring the scope decision to the
            caller rather than baking one choice into this script.

    Returns:
        The results frame, unmodified.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If required columns, modes, C, pred_len, the balanced
            5-seed design, the per-mode batch policy, the total_steps
            identity, or (if given) the expected rho set are violated.
    """
    if not path.exists():
        raise FileNotFoundError(f"C=84 results CSV not found: {path}")
    df = pd.read_csv(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    if (df["C"] != _C).any():
        raise ValueError(f"CSV contains C values other than {_C}: {sorted(df['C'].unique())}")
    if (df["pred_len"] != _PRED_LEN).any():
        raise ValueError(f"CSV contains pred_len values other than {_PRED_LEN}: {sorted(df['pred_len'].unique())}")
    modes = set(df["mode"].unique())
    if set(_MODES) - modes:
        raise ValueError(f"CSV must contain modes {sorted(_MODES)}; found {sorted(modes)}")
    rhos = sorted(df["rho"].unique())
    if expected_rhos is not None and set(rhos) != expected_rhos:
        raise ValueError(f"rho values {rhos} do not match expected {sorted(expected_rhos)}")
    for mode in _MODES:
        for rho in rhos:
            cell = df[(df["mode"] == mode) & (df["rho"] == rho)]
            seeds = sorted(cell["seed"].unique())
            if seeds != sorted(_SEED_ORDER) or len(cell) != len(_SEED_ORDER):
                raise ValueError(
                    f"{mode} rho={rho}: seeds {seeds} over {len(cell)} rows "
                    f"do not match one row per seed in {sorted(_SEED_ORDER)}"
                )
    for mode, batch in _BATCH_POLICY.items():
        found = sorted(df.loc[df["mode"] == mode, "batch_size"].unique())
        if found != [batch]:
            raise ValueError(f"{mode} batch_size must be {batch}; found {found}")
    if {"total_steps", "steps_per_epoch"} <= set(df.columns):
        bad = df["total_steps"] != df["best_epoch"] * df["steps_per_epoch"]
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} rows violate total_steps == best_epoch * "
                f"steps_per_epoch; validation-best restore is not verifiable."
            )
    return df


def pivot_body(df: pd.DataFrame) -> pd.DataFrame:
    """Per-rho mean test MSE for each arm and the three ratios.

    Means are taken over seeds at full precision; ratios are formed from
    the unrounded means, not from rounded display values.

    Args:
        df: Output of load_results.

    Returns:
        One row per rho with columns rho, ci, cd, blk, cd_ci, blk_ci, blk_cd.
    """
    rows: list[dict[str, float]] = []
    for rho in sorted(df["rho"].unique()):
        means = {
            mode: df[(df["mode"] == mode) & (df["rho"] == rho)]["test_mse"].mean()
            for mode in _MODES
        }
        rows.append({
            "rho": rho,
            "ci": means["CI"], "cd": means["CD"], "blk": means["CD_Block"],
            "cd_ci": means["CD"] / means["CI"],
            "blk_ci": means["CD_Block"] / means["CI"],
            "blk_cd": means["CD_Block"] / means["CD"],
        })
    return pd.DataFrame(rows)


def sign_test_p(signs: str) -> float:
    """Exact one-sided sign-test p for a paired sign string.

    Matches analyze_block_attention.py's convention exactly (binomial tail
    P(K >= n_pos) at 0.5 over the nonzero differences), so sign-test p-values
    are comparable across both three-arm ablations in this paper -- a
    two-sided test here would silently produce a different, inconsistent
    statistic for the same underlying design.

    Args:
        signs: String over {+, -, 0} from paired_differences.

    Returns:
        The exact one-sided p-value; nan if every difference is zero.
    """
    n_pos = signs.count("+")
    n_nonzero = len(signs) - signs.count("0")
    if n_nonzero == 0:
        return float("nan")
    return float(stats.binom.sf(n_pos - 1, n_nonzero, 0.5))


def contrasts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The three paired per-seed contrasts with 95% paired-t CIs (df=4).

    Delegates to the shared paired_stats machinery (make_diff_frame +
    paired_differences) rather than reimplementing the same arithmetic --
    the module exists precisely so this computation has one verified
    implementation, not one per script.

    Sign convention follows paired_stats: diff = alt - base, positive = the
    mode under scrutiny is worse. Blk-CD negative therefore means block
    attention narrows the CD penalty, consistent with
    analyze_block_attention.py's reading of the same sign convention.

    Args:
        df: Output of load_results.

    Returns:
        Mapping from contrast label ("CD-CI", "Blk-CI", "Blk-CD") to the
        paired_differences frame keyed by rho.
    """
    pairs = {"CD-CI": ("CI", "CD"), "Blk-CI": ("CI", "CD_Block"), "Blk-CD": ("CD", "CD_Block")}
    out: dict[str, pd.DataFrame] = {}
    for label, (base, alt) in pairs.items():
        diff = ps.make_diff_frame(df, cell_cols=["rho"], mode_base=base, mode_alt=alt)
        out[label] = ps.paired_differences(diff, cell_cols=["rho"])
    return out


def best_epoch_ranges(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Min/max best_epoch per arm across every rho and seed present.

    Args:
        df: Output of load_results.

    Returns:
        {mode: (min_best_epoch, max_best_epoch)}.
    """
    return {
        mode: (int(df.loc[df["mode"] == mode, "best_epoch"].min()),
               int(df.loc[df["mode"] == mode, "best_epoch"].max()))
        for mode in _MODES
    }


# Oracle values frozen at the first run against the committed results file
# (15 rows, single cell rho=0.5). Pivot entries are (ci, cd, blk, cd_ci,
# blk_ci, blk_cd) to 4dp. Contrast entries are (mean_diff, ci_lo, ci_hi)
# to 4dp, the relative mean difference in percent of the base arm to 2dp,
# and the per-seed sign string in _SEED_ORDER order. Rho keys are compared
# after rounding to 4dp.
_ORACLE_PIVOT: dict[float, tuple[float, float, float, float, float, float]] = {
    0.5: (1.0076, 1.0045, 1.0050, 0.9969, 0.9974, 1.0005),
}
_ORACLE_CONTRASTS: dict[str, dict[float, tuple[float, float, float, float, str]]] = {
    "CD-CI":  {0.5: (-0.0031, -0.0105, +0.0042, -0.31, "+-+--")},
    "Blk-CI": {0.5: (-0.0027, -0.0093, +0.0039, -0.26, "+----")},
    "Blk-CD": {0.5: (+0.0005, -0.0022, +0.0032, +0.05, "-+-++")},
}
_ORACLE_CDCI_SIGN_P: dict[float, float] = {0.5: 0.812}
_ORACLE_BEST_EPOCH: dict[str, tuple[int, int]] = {
    "CI": (9, 17), "CD": (6, 7), "CD_Block": (6, 10),
}
_ORACLE_ROWS = 15


def oracle_check(
    df: pd.DataFrame,
    body: pd.DataFrame,
    contr: dict[str, pd.DataFrame],
    epochs: dict[str, tuple[int, int]],
) -> bool:
    """Compare recomputed values against the frozen oracle tables.

    Every rho present in the data must have a frozen entry in every table;
    an unfrozen rho is a failure, not a skip, so extending the sweep forces
    a deliberate re-freeze rather than silently passing new cells through.

    Args:
        df: Output of load_results.
        body: Output of pivot_body.
        contr: Output of contrasts.
        epochs: Output of best_epoch_ranges.

    Returns:
        True if every check passes.
    """
    failures: list[str] = []

    if len(df) != _ORACLE_ROWS:
        failures.append(f"row count {len(df)} != {_ORACLE_ROWS}")

    for _, row in body.iterrows():
        rho = round(float(row["rho"]), 4)
        pinned = _ORACLE_PIVOT.get(rho)
        if pinned is None:
            failures.append(f"rho={rho}: no frozen pivot entry")
            continue
        got = tuple(round(float(row[k]), 4) for k in ("ci", "cd", "blk", "cd_ci", "blk_ci", "blk_cd"))
        if got != pinned:
            failures.append(f"rho={rho}: pivot {got} != {pinned}")

    for label, frame in contr.items():
        for _, row in frame.iterrows():
            rho = round(float(row["rho"]), 4)
            pinned = _ORACLE_CONTRASTS[label].get(rho)
            if pinned is None:
                failures.append(f"{label} rho={rho}: no frozen contrast entry")
                continue
            got = (round(float(row["mean_diff"]), 4), round(float(row["ci_lo"]), 4),
                   round(float(row["ci_hi"]), 4), round(float(row["rel_pct"]), 2),
                   str(row["signs"]))
            if got != pinned:
                failures.append(f"{label} rho={rho}: {got} != {pinned}")
            if label == "CD-CI":
                p_pinned = _ORACLE_CDCI_SIGN_P.get(rho)
                p_got = round(sign_test_p(str(row["signs"])), 3)
                if p_pinned is None or p_got != p_pinned:
                    failures.append(f"CD-CI rho={rho}: sign p {p_got} != {p_pinned}")

    for mode, pinned_range in _ORACLE_BEST_EPOCH.items():
        if epochs.get(mode) != pinned_range:
            failures.append(f"{mode} best_epoch {epochs.get(mode)} != {pinned_range}")

    for msg in failures:
        print(f"  [ORACLE FAIL] {msg}")
    return not failures


def main(argv: list[str] | None = None) -> int:
    """Report the C=84 three-arm pivot, paired contrasts, and best_epoch ranges.

    Ends with the frozen-oracle check (see the ORACLE section in the
    module docstring). Structural and schema failures raise directly.

    Args:
        argv: Optional argument vector (defaults to sys.argv).

    Returns:
        0 if the oracle check passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="C=84 three-arm lightweight-CD ablation (pivot, paired contrasts, best_epoch)."
    )
    parser.add_argument("--csv", type=Path, default=_DEFAULT_CSV, help="C=84 results CSV.")
    parser.add_argument("--rhos", type=str, default=None,
                        help="Comma-separated expected rho values, e.g. '0.5' or '0.1,0.5,0.9'. "
                             "If omitted, whatever rho values are present are accepted.")
    args = parser.parse_args(argv)

    expected_rhos = (
        {float(r) for r in args.rhos.split(",")} if args.rhos is not None else None
    )
    df = load_results(args.csv, expected_rhos)
    body = pivot_body(df)
    contr = contrasts(df)
    epochs = best_epoch_ranges(df)

    print(f"=== C=84 THREE-ARM PIVOT (mean test MSE over seeds {sorted(_SEED_ORDER)}) ===")
    for _, row in body.iterrows():
        print(
            f"  rho={row['rho']}: CI {row['ci']:.4f}  CD {row['cd']:.4f}  Blk {row['blk']:.4f}  "
            f"CD/CI {row['cd_ci']:.4f}  Blk/CI {row['blk_ci']:.4f}  Blk/CD {row['blk_cd']:.4f}"
        )

    base_of = {"CD-CI": "CI", "Blk-CI": "CI", "Blk-CD": "CD"}
    for label, frame in contr.items():
        print(f"\n=== PAIRED {label} (95% paired-t CI, df=4) ===")
        for _, row in frame.iterrows():
            # Sign-test note restricted to CD-CI, matching
            # analyze_block_attention.py exactly: the one-sided test asks
            # "is the mode under scrutiny significantly worse", which only
            # has a clean reading for that contrast -- printing it for
            # Blk-CD, say, would report p=1.000 whenever block attention is
            # clearly better, which reads as "no effect" rather than the
            # true "effect is in the other direction."
            sign_note = (
                f"  exact sign p={sign_test_p(str(row['signs'])):.3f}" if label == "CD-CI" else ""
            )
            print(
                f"  rho={row['rho']}: {label} {row['mean_diff']:+.4f}  "
                f"95% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  signs {row['signs']}  "
                f"({row['rel_pct']:+.2f}% of {base_of[label]}){sign_note}"
            )

    print("\n=== BEST_EPOCH RANGES ===")
    for mode, (lo, hi) in epochs.items():
        print(f"  {mode}: {lo}-{hi}")

    print()
    passed = oracle_check(df, body, contr, epochs)
    print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())