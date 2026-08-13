"""C=84 three-arm lightweight-CD ablation (reviewer yc7L Recommended 2), B10.

Pre-registered before the underlying data exists, per this project's P4
discipline (analysis scripts written before results, so methodology
cannot be shaped by which direction the numbers happen to land). Do not
add oracle target values to this file until a real committed CSV exists
and the analysis has run against it once -- see FREEZING THE ORACLE below.

Reads a self-contained results file for the AR(1) grid at C=84, modes
{CI, CD, CD_Block}, rho in whichever subset of {0.1, 0.5, 0.9} was
launched (OQ8, still open: one rho or all three), 5 seeds each
({42, 123, 456, 789, 1011}), pred_len=96. This is DELIBERATELY not
results_grid.csv: that file's existing C=84 CD/CI rows were trained under
the pre-fusion environment at forced batch 1 (a hardware-forced artefact,
not the current protocol), so per P1 (new rows never pair against
committed rows run under a different environment) this script expects a
new, separate file -- naming convention below assumes
results_boundary_c84.csv, adjust --csv if the actual launch used a
different name.

CD_Block partition: necessarily neutral (contiguous groups), since the
AR(1) grid's compound-symmetry covariance has no leader-follower or
block structure for a partition to align with or against -- unlike the
leader-follower sweep's CD_Block, which preserves real pair structure.
This is a structural fact about the grid, not a modelling choice; the
neutral-partition framing belongs in the writeup regardless of the
rho scope decided at launch.

Outputs, matching analyze_block_attention.py's structure so the two
three-arm ablations in this paper report in a consistent shape: the
per-rho pivot (mean test MSE per arm, CD/CI, Blk/CI, Blk/CD ratios), the
three paired per-seed contrasts with 95% paired-t CIs (df=4) via the
shared paired_stats machinery, and per-arm best_epoch ranges (useful
since C=84's convergence behaviour is a live open question -- CI's
C=84 s/epoch was UNMEASURED as of the B10 decision, gating the sweep on
a first-session measurement per P2).

FREEZING THE ORACLE
--------------------
This script currently has NO oracle section -- there is nothing to freeze
against real data that does not exist yet. Once the C=84 run completes
and this script has been run against it once, add an oracle_check
function following analyze_block_attention.py's exact pattern (pinned
pivot values to 4dp, paired sign-test outcomes, best_epoch ranges) and
freeze it in the same commit as the results CSV, per this project's
provenance discipline (a number enters a response only via an
oracle-passing script). Do not add oracle numbers speculatively before
that point.

Run:

    python src/analysis/analyze_boundary_c84.py --csv path/to/results_boundary_c84.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

import paired_stats as ps  # sibling module; on sys.path when run as a script

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSV = _ROOT / "results" / "Revision" / "train_boundary_c84" / "results_boundary_c84.csv"

_MODES: tuple[str, ...] = ("CI", "CD", "CD_Block")
_SEED_ORDER: tuple[int, ...] = (42, 123, 456, 789, 1011)
_C = 84
_PRED_LEN = 96
_REQUIRED_COLS: set[str] = {"C", "rho", "mode", "pred_len", "seed", "test_mse", "best_epoch", "batch_size"}
# Committed protocol: CD-family arms and CI share one batch policy decision
# at C=84, unlike the leader-follower sweep where CI and CD differ (128 vs
# 8) -- confirm against the launch notebook's CFG once it exists; this is
# a placeholder pending that confirmation (OQ8), not an assumption to trust
# blindly. Left permissive (no batch-policy assertion) until then.


def load_results(path: Path, expected_rhos: set[float] | None = None) -> pd.DataFrame:
    """Load and validate the C=84 three-arm sweep results.

    Args:
        path: Path to the self-contained C=84 results CSV.
        expected_rhos: If given, the exact rho set the file must contain
            (e.g. {0.5} for a single-cell launch, or {0.1, 0.5, 0.9} for
            the full sweep). If None, whatever rho values are present are
            accepted and reported, deferring the OQ8 scope decision to
            the caller rather than baking one choice into this script.

    Returns:
        The results frame, unmodified.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If required columns, modes, C, pred_len, the balanced
            5-seed design, or (if given) the expected rho set are violated.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"C=84 results CSV not found: {path}. This is expected until the "
            f"B10 launch completes -- this script is pre-registered ahead of "
            f"the data, per P4."
        )
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


def main(argv: list[str] | None = None) -> int:
    """Report the C=84 three-arm pivot, paired contrasts, and best_epoch ranges.

    No oracle check: see the FREEZING THE ORACLE section in the module
    docstring. Exit code is always 0 (structural/schema failures raise
    directly; there is no PASS/FAIL oracle gate yet to report).

    Args:
        argv: Optional argument vector (defaults to sys.argv).

    Returns:
        0.
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

    print("\n=== BEST_EPOCH RANGES (convergence check -- C=84 CI s/epoch was UNMEASURED pre-launch) ===")
    for mode, (lo, hi) in epochs.items():
        print(f"  {mode}: {lo}-{hi}")

    print(
        "\nNo oracle check: this script is pre-registered ahead of the C=84 "
        "data. Once real results exist, run this once, freeze the printed "
        "values into an oracle_check function following "
        "analyze_block_attention.py's pattern, and commit both together."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
