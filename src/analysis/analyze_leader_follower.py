"""Leader-follower P = 16 sweep analysis (Table 6 and Section 4.3 slope).

Reproduces Table 6 and the gamma-slope of Section 4.3 of "Equal Accuracy, Unequal
Cost: Channel Dependence in PatchTST under Controlled Coupling". The dedicated
leader-follower sweep fixes C = 21, rho = 0.5, patch size P = 16 and varies the
lag-1 coupling strength gamma in {0.0, 0.3, 0.6, 0.9} across modes {CI, CD,
DLinear} and seeds {42, 123, 456, 789, 1011} (60 rows).

This is a different experiment from the boundary patch-size sweep in
analyze_boundary.py. The dedicated sweep here gives a CD-CI gap that rises with
gamma (slope +0.0072, from results_leader_follower.csv); the boundary sweep gives
a flat slope (-0.0006, from results_boundary.csv). The two slopes are never pooled
and never reported in place of one another.

Statistics reuse the shared paired-difference machinery in ``paired_stats``: the
matched unit is the per-seed difference d_s = MSE_CD - MSE_CI within each gamma.
Per-gamma paired CD-CI means carry 95% paired-t CIs (df = 4, t = 2.776 at n = 5);
the gamma slope is an ordinary least squares fit of those per-seed differences on
gamma (20 points, df = 18 for the slope CI).

Run from a clean checkout:

    python analyze_leader_follower.py
    python analyze_leader_follower.py --csv path/to/results_leader_follower.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

import paired_stats as ps  # sibling module; on sys.path when run as a script

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "results"
_CSV = _RESULTS_DIR / "results_leader_follower.csv"

_BASE_MODE = "CI"
_ALT_MODE = "CD"
_DLINEAR_MODE = "DLinear"
_GAMMAS: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)
_SEED_ORDER: tuple[int, ...] = (42, 123, 456, 789, 1011)
_REQUIRED_COLS: set[str] = {"gamma", "mode", "seed", "test_mse"}

# Oracle targets from main.tex Table 6 and Section 4.3.
# gamma -> (CI mean, CD mean, DLinear mean, CD/CI ratio), 4 decimal places.
_ORACLE_BODY: dict[float, tuple[float, float, float, float]] = {
    0.0: (1.0270, 1.0287, 1.0301, 1.0016),
    0.3: (1.0326, 1.0393, 1.0330, 1.0065),
    0.6: (1.0298, 1.0377, 1.0301, 1.0077),
    0.9: (1.0278, 1.0362, 1.0284, 1.0082),
}
# slope, ci_lo, ci_hi (4 dp), p (3 dp), r2 (2 dp).
_ORACLE_SLOPE: dict[str, float] = {
    "slope": 0.0072,
    "ci_lo": 0.0027,
    "ci_hi": 0.0117,
    "p": 0.004,
    "r2": 0.38,
}


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate the leader-follower sweep results.

    Args:
        path: Path to results_leader_follower.csv.

    Returns:
        The results frame, unmodified.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If required columns, modes, gammas, or the balanced 5-seed
            design are missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Leader-follower CSV not found: {path}")
    df = pd.read_csv(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    modes = set(df["mode"].unique())
    expected_modes = {_BASE_MODE, _ALT_MODE, _DLINEAR_MODE}
    if expected_modes - modes:
        raise ValueError(f"CSV must contain modes {sorted(expected_modes)}; found {sorted(modes)}")
    if sorted(df["gamma"].unique()) != sorted(_GAMMAS):
        raise ValueError(f"gamma values {sorted(df['gamma'].unique())} do not match {sorted(_GAMMAS)}")
    for mode in expected_modes:
        for gamma in _GAMMAS:
            seeds = sorted(df.loc[(df["mode"] == mode) & (df["gamma"] == gamma), "seed"].unique())
            if seeds != sorted(_SEED_ORDER):
                raise ValueError(f"{mode} gamma={gamma} seeds {seeds} do not match {sorted(_SEED_ORDER)}")
    return df


def table6_body(df: pd.DataFrame) -> pd.DataFrame:
    """Per-gamma mean test MSE for each mode and the CD/CI ratio.

    Means are taken over seeds at full precision; the ratio is formed from the
    unrounded means, not from rounded display values.

    Args:
        df: Output of load_results.

    Returns:
        One row per gamma with columns gamma, ci, cd, dlinear, ratio.
    """
    rows: list[dict[str, float]] = []
    for gamma in _GAMMAS:
        sub = df[df["gamma"] == gamma]
        ci = float(sub.loc[sub["mode"] == _BASE_MODE, "test_mse"].mean())
        cd = float(sub.loc[sub["mode"] == _ALT_MODE, "test_mse"].mean())
        dlinear = float(sub.loc[sub["mode"] == _DLINEAR_MODE, "test_mse"].mean())
        rows.append({"gamma": gamma, "ci": ci, "cd": cd, "dlinear": dlinear, "ratio": cd / ci})
    return pd.DataFrame(rows)


def gamma_slope(df: pd.DataFrame) -> dict[str, float]:
    """OLS slope of the per-seed paired CD-CI difference on gamma.

    The matched unit d_s = MSE_CD - MSE_CI is formed per (gamma, seed) by the
    shared paired-difference helper, then regressed on gamma. The slope CI is the
    ordinary 95% OLS interval (df = n - 2).

    Args:
        df: Output of load_results.

    Returns:
        Mapping with slope, ci_lo, ci_hi, p, r2, and n (number of paired points).
    """
    diff = ps.make_diff_frame(df, cell_cols=["gamma"], mode_base=_BASE_MODE, mode_alt=_ALT_MODE)
    model = smf.ols("diff ~ gamma", data=diff).fit()
    ci_lo, ci_hi = model.conf_int().loc["gamma"]
    return {
        "slope": float(model.params["gamma"]),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p": float(model.pvalues["gamma"]),
        "r2": float(model.rsquared),
        "n": int(diff.shape[0]),
    }


def paired_by_gamma(df: pd.DataFrame) -> pd.DataFrame:
    """Per-gamma paired CD-CI mean difference with 95% paired-t CIs (df = 4).

    Supporting detail for Table 6; reuses paired_stats so the per-cell CIs share
    one verified implementation with the other control analyses.

    Args:
        df: Output of load_results.

    Returns:
        paired_differences output keyed by gamma.
    """
    diff = ps.make_diff_frame(df, cell_cols=["gamma"], mode_base=_BASE_MODE, mode_alt=_ALT_MODE)
    return ps.paired_differences(diff, cell_cols=["gamma"])


def _check(body: pd.DataFrame, slope: dict[str, float]) -> tuple[list[str], bool]:
    """Build the target-vs-reproduced check for the four rows and the slope.

    Comparison precision matches how main.tex prints each quantity: means and
    ratio and slope and slope CI to 4 dp, p to 3 dp, R^2 to 2 dp.

    Args:
        body: Output of table6_body.
        slope: Output of gamma_slope.

    Returns:
        (lines, all_pass).
    """
    lines: list[str] = []
    all_pass = True
    body_indexed = body.set_index("gamma")
    for gamma in _GAMMAS:
        row = body_indexed.loc[gamma]
        got = (round(row["ci"], 4), round(row["cd"], 4), round(row["dlinear"], 4), round(row["ratio"], 4))
        target = _ORACLE_BODY[gamma]
        ok = got == target
        all_pass = all_pass and ok
        lines.append(
            f"  [{'PASS' if ok else 'FAIL'}] gamma={gamma}: "
            f"CI {got[0]:.4f} CD {got[1]:.4f} DLinear {got[2]:.4f} ratio {got[3]:.4f}  "
            f"target CI {target[0]:.4f} CD {target[1]:.4f} DLinear {target[2]:.4f} ratio {target[3]:.4f}"
        )
    got_slope = (
        round(slope["slope"], 4),
        round(slope["ci_lo"], 4),
        round(slope["ci_hi"], 4),
        round(slope["p"], 3),
        round(slope["r2"], 2),
    )
    target_slope = (
        _ORACLE_SLOPE["slope"],
        _ORACLE_SLOPE["ci_lo"],
        _ORACLE_SLOPE["ci_hi"],
        _ORACLE_SLOPE["p"],
        _ORACLE_SLOPE["r2"],
    )
    ok = got_slope == target_slope
    all_pass = all_pass and ok
    lines.append(
        f"  [{'PASS' if ok else 'FAIL'}] slope {got_slope[0]:+.4f} "
        f"95% CI [{got_slope[1]:+.4f}, {got_slope[2]:+.4f}] p={got_slope[3]:.3f} R2={got_slope[4]:.2f}  "
        f"target {target_slope[0]:+.4f} [{target_slope[1]:+.4f}, {target_slope[2]:+.4f}] "
        f"p={target_slope[3]:.3f} R2={target_slope[4]:.2f}"
    )
    return lines, all_pass


def main(argv: list[str] | None = None) -> int:
    """Reproduce Table 6 and the Section 4.3 gamma slope.

    Args:
        argv: Optional argument vector (defaults to sys.argv).

    Returns:
        Process exit code: 0 if every value matches the oracle, else 1.
    """
    parser = argparse.ArgumentParser(description="Leader-follower P=16 sweep (Table 6, Section 4.3 slope).")
    parser.add_argument("--csv", type=Path, default=_CSV, help="Leader-follower results CSV.")
    args = parser.parse_args(argv)

    df = load_results(args.csv)
    body = table6_body(df)
    slope = gamma_slope(df)
    paired = paired_by_gamma(df)

    print("=== TABLE 6 BODY (mean test MSE over seeds {42,123,456,789,1011}) ===")
    for _, row in body.iterrows():
        print(
            f"  gamma={row['gamma']}: CI {row['ci']:.4f}  CD {row['cd']:.4f}  "
            f"DLinear {row['dlinear']:.4f}  CD/CI ratio {row['ratio']:.4f}"
        )

    print("\n=== PER-GAMMA PAIRED CD-CI (95% paired-t CI, df=4) ===")
    for _, row in paired.iterrows():
        print(
            f"  gamma={row['gamma']}: CD-CI {row['mean_diff']:+.4f}  "
            f"95% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  ({row['rel_pct']:+.2f}% of CI)"
        )

    print("\n=== GAMMA SLOPE (OLS of paired CD-CI on gamma) ===")
    print(
        f"  slope {slope['slope']:+.4f}  95% CI [{slope['ci_lo']:+.4f}, {slope['ci_hi']:+.4f}]  "
        f"p={slope['p']:.3f}  R2={slope['r2']:.2f}  n={slope['n']}"
    )

    print("\n=== ORACLE CHECK (main.tex Table 6 and Section 4.3) ===")
    check_lines, all_pass = _check(body, slope)
    print("\n".join(check_lines))
    print(f"\nRESULT: {'PASS - Table 6 and slope reproduce' if all_pass else 'FAIL - see lines above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
