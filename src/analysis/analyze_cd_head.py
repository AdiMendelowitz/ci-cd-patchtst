"""Cross-variate head (CD_Head) control.

Tests whether the per-variate prediction head, rather than the encoder, is
what stops CD from exploiting lagged cross-channel structure. If CD_Head
still does not beat CI, the head-bottleneck objection is closed.

Reads
-----
results/results_cd_head.csv
    One row per (cell, mode, seed). Required columns: cell, mode, seed,
    test_mse, gamma. Modes: CI, CD, CD_Head. Cells: the leader-follower
    cells lf_gamma{0.0, 0.6, 0.9}, five seeds each, with gamma=0 serving as
    the AR(1) floor.

Console
-------
Per-cell paired differences for three contrasts (CD_Head-CI, CD-CI,
CD_Head-CD); the seed-clustered grand mean for each, with an oracle check;
a regression of CD_Head-CI on gamma over the leader-follower cells; and a
LaTeX paragraph for the CD_Head-CI headline result.

Usage
-----
python analyze_cd_head.py [path/to/results_cd_head.csv]
"""

import sys
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

import paired_stats as ps

_RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "results_cd_head.csv"

_LF_PREFIX = "lf_"
_SEEDS: tuple[int, ...] = (42, 123, 456, 789, 1011)
_CELLS: tuple[str, ...] = ("lf_gamma0.0", "lf_gamma0.6", "lf_gamma0.9")
_MODES: tuple[str, ...] = ("CI", "CD", "CD_Head")

_WORD: dict[int, str] = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

# Oracle values for the seed-clustered grand mean of each contrast, frozen
# against the committed results CSV (3 cells, 5 seeds, clustered by seed so
# the same seed's data draw across cells is not double-counted as
# independent evidence). mean/ci_lo/ci_hi to 4dp, rel_pct to 2dp.
_ORACLE_GRAND_MEAN: dict[tuple[str, str], dict[str, float]] = {
    ("CI", "CD_Head"): {"mean": 0.0023, "ci_lo": -0.0023, "ci_hi": 0.0070, "rel_pct": 0.23},
    ("CI", "CD"): {"mean": 0.0060, "ci_lo": 0.0032, "ci_hi": 0.0088, "rel_pct": 0.59},
    ("CD", "CD_Head"): {"mean": -0.0037, "ci_lo": -0.0089, "ci_hi": 0.0014, "rel_pct": -0.36},
}


def _w(k: int) -> str:
    return _WORD.get(k, str(k))


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate the CD_Head results CSV.

    Validates the full balanced design (exactly the three leader-follower
    cells, five seeds, three modes, no duplicates) rather than only checking
    that the required columns and modes are present: the grand-mean
    contrasts assume this shape, and a silently unbalanced input would
    change what "grand mean" and "clustered by seed" mean without any
    signal that the input had drifted from the expected design.

    Args:
        path: Path to the results CSV.

    Returns:
        The validated results frame, unmodified.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If required columns or modes are missing, any cell is
            not one of the three expected leader-follower cells, rows are
            duplicated on (cell, mode, seed), or any (cell, mode) does not
            cover exactly the five expected seeds.
    """
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)

    required = {"cell", "mode", "seed", "test_mse", "gamma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results CSV missing columns: {missing}")

    present_modes = set(df["mode"].unique())
    missing_modes = set(_MODES) - present_modes
    if missing_modes:
        raise ValueError(f"Expected mode(s) {sorted(missing_modes)} absent from results.")

    bad_cells = set(df["cell"].unique()) - set(_CELLS)
    if bad_cells:
        raise ValueError(
            f"Unexpected cell(s) {sorted(bad_cells)}; this script assumes exactly "
            f"the leader-follower cells {_CELLS}."
        )

    dup = df.duplicated(subset=["cell", "mode", "seed"])
    if dup.any():
        raise ValueError(f"Results contain {int(dup.sum())} duplicate (cell, mode, seed) rows.")

    counts = df.groupby(["cell", "mode"])["seed"].apply(lambda s: tuple(sorted(s)))
    bad_seeds = counts[counts != _SEEDS]
    if not bad_seeds.empty:
        raise ValueError(f"(cell, mode) group(s) with unexpected seed coverage: {bad_seeds.to_dict()}")

    return df


def print_contrast(label: str, paired: pd.DataFrame, gm: dict, oracle: dict) -> bool:
    """Print one contrast's per-cell table and seed-clustered grand mean.

    Args:
        label: Human-readable contrast name for the console header.
        paired: Output of paired_stats.paired_differences, one row per cell.
        gm: Output of paired_stats.grand_mean_diff_clustered.
        oracle: Frozen target for gm, from _ORACLE_GRAND_MEAN.

    Returns:
        True if gm matches oracle at the stated precision.
    """
    print(f"\n=== {label} ===")
    print("Positive = alt worse. 'alt beats base' means the CI on the difference lies fully below zero.")
    print(f"{'cell':>20}  {'mean_diff':>10} {'rel_%':>8}  {'95%CI_lo':>10} {'95%CI_hi':>10}  signs")
    print("-" * 78)
    for _, r in paired.iterrows():
        print(
            f"{str(r.cell):>20}  {r.mean_diff:>+10.6f} {r.rel_pct:>+8.4f}%  "
            f"{r.ci_lo:>+10.6f} {r.ci_hi:>+10.6f}  {r.signs}"
        )
    n_alt_wins = int((paired["ci_hi"] < 0).sum())
    pct = 100.0 * gm["mean"] / gm["base_mean"]
    print(f"Cells where alt beats base (CI below zero): {n_alt_wins} / {len(paired)}")
    print(
        f"Grand mean (clustered by seed, n={gm['n']}): {gm['mean']:+.6f}  "
        f"({pct:+.4f}% of base mean)  95% CI [{gm['ci_lo']:+.6f}, {gm['ci_hi']:+.6f}]"
    )

    got = {
        "mean": round(gm["mean"], 4), "ci_lo": round(gm["ci_lo"], 4),
        "ci_hi": round(gm["ci_hi"], 4), "rel_pct": round(pct, 2),
    }
    ok = got == oracle
    print(f"  [{'PASS' if ok else 'FAIL'}] oracle: {got}" + ("" if ok else f" != {oracle}"))
    return ok


def gamma_trend(df: pd.DataFrame, mode_base: str, mode_alt: str) -> dict:
    """Regress the paired (alt - base) difference on gamma over leader-follower cells.

    The regression's standard errors are not clustered by seed, even though
    the same seed set recurs at every gamma; treat the reported CI and
    p-value as indicative of the trend's direction and rough magnitude, not
    as a rigorously independent-samples inference. This script does not
    currently feed this regression's output into the paper -- if that
    changes, cluster-robust standard errors should be computed first.

    Args:
        df: Full results frame (only leader-follower cells are used).
        mode_base: Reference mode (subtracted).
        mode_alt: Mode under scrutiny (minuend).

    Returns:
        Dict with slope, ci_lo, ci_hi, pval, and r2 for the gamma coefficient.
    """
    lf = df[df["cell"].str.startswith(_LF_PREFIX, na=False)]
    diff = ps.make_diff_frame(lf, ["gamma"], mode_base, mode_alt)
    model = smf.ols("diff ~ gamma", data=diff).fit()
    return {
        "slope": float(model.params["gamma"]),
        "ci_lo": float(model.conf_int().loc["gamma", 0]),
        "ci_hi": float(model.conf_int().loc["gamma", 1]),
        "pval": float(model.pvalues["gamma"]),
        "r2": float(model.rsquared),
    }


def print_gamma_trend(label: str, trend: dict) -> None:
    """Print a gamma-trend regression line."""
    print(
        f"\n{label}: slope {trend['slope']:+.4f} MSE per unit gamma "
        f"(95% CI [{trend['ci_lo']:+.4f}, {trend['ci_hi']:+.4f}], p = {trend['pval']:.3f}, R2 = {trend['r2']:.2f}) "
        f"-- not seed-clustered, see gamma_trend docstring"
    )


def print_latex_prose(head_vs_ci: pd.DataFrame, gm_head_ci: dict, trend_head_ci: dict) -> None:
    """Print the CD_Head paragraph for the paper.

    Args:
        head_vs_ci: Per-cell paired_differences output for CD_Head-CI.
        gm_head_ci: Seed-clustered grand mean for CD_Head-CI.
        trend_head_ci: gamma_trend output for CD_Head-CI.
    """
    n_cells = len(head_vs_ci)
    n_wins = int((head_vs_ci["ci_hi"] < 0).sum())
    pct = abs(100.0 * gm_head_ci["mean"] / gm_head_ci["base_mean"])
    print("\n=== PAPER PROSE (CD_Head control paragraph) ===")
    print(
        f"To test whether the per-variate head, rather than the encoder, limits "
        f"CD, we add a variant (CD\\_Head) that pools each variate's encoder output "
        f"into a summary token, attends across the C summaries, and concatenates "
        f"the resulting per-variate context to the flattened encoder output before "
        f"the shared projection; the encoder is unchanged. Across the {_w(n_cells)} "
        f"leader-follower cells (with $\\gamma = 0$ as the AR(1) floor), CD\\_Head "
        f"beats CI in {_w(n_wins)} of them, counting a cell as a win only when the "
        f"95\\% CI on the paired difference lies entirely below zero. Pooled across "
        f"$\\gamma$ and clustered by seed, the CD\\_Head$-$CI grand mean is "
        f"${gm_head_ci['mean']:+.4f}$ MSE (95\\% CI "
        f"$[{gm_head_ci['ci_lo']:.4f}, {gm_head_ci['ci_hi']:.4f}]$, ${pct:.2f}\\%$ "
        f"of the CI mean). Explicitly pooling cross-variate information at the "
        f"head does not overturn the null, so the result is not a head artefact."
    )


def main() -> int:
    """Run all three contrasts, the gamma trend, and print the paper paragraph.

    Returns:
        0 if every contrast's grand mean matches its frozen oracle, 1
        otherwise. Schema and design violations in load_results raise
        directly rather than returning non-zero.
    """
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _RESULTS_PATH
    df = load_results(csv_path)

    contrasts = [
        ("CD_Head - CI  (headline: does the better head beat CI?)", "CI", "CD_Head"),
        ("CD - CI       (canonical null, reproduced here)", "CI", "CD"),
        ("CD_Head - CD  (does the head help at all?)", "CD", "CD_Head"),
    ]
    results: dict[tuple[str, str], tuple[pd.DataFrame, dict]] = {}
    all_passed = True
    for label, base, alt in contrasts:
        diff = ps.make_diff_frame(df, ["cell"], base, alt)
        paired = ps.paired_differences(diff, ["cell"])
        gm = ps.grand_mean_diff_clustered(diff, "seed")
        passed = print_contrast(label, paired, gm, _ORACLE_GRAND_MEAN[(base, alt)])
        all_passed = all_passed and passed
        results[(base, alt)] = (paired, gm)

    print("\n=== GAMMA TREND OVER LEADER-FOLLOWER CELLS ===")
    trend_head_ci = gamma_trend(df, "CI", "CD_Head")
    print_gamma_trend("CD_Head - CI", trend_head_ci)
    print_gamma_trend("CD - CI", gamma_trend(df, "CI", "CD"))

    head_vs_ci_paired, head_vs_ci_gm = results[("CI", "CD_Head")]
    print_latex_prose(head_vs_ci_paired, head_vs_ci_gm, trend_head_ci)

    print(f"\nRESULT: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())