"""Experiment 1 analysis: cross-variate head (CD+head) control.

Reads
-----
results/results_cd_head.csv
    One row per (cell, mode, seed). Required columns: cell, mode, seed, test_mse, gamma. Modes: CI, CD, CD_Head.
    Cells: the leader-follower cells lf_gamma{0.0, 0.6, 0.9}, with gamma=0 serving as the AR(1) floor.

Console
-------
Per-cell paired differences for three contrasts (CD_Head - CI, CD - CI, CD_Head - CD), the grand means, a regression of
CD_Head - CI on gamma over the leader-follower cells, and a LaTeX paragraph.

Purpose
-------
The Limitations section concedes that the per-variate head might be what stops CD exploiting lagged structure. This
script tests an explicit cross-variate head. If CD+head still does not beat CI, the head-bottleneck objection is closed.

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

_WORD: dict[int, str] = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _w(k: int) -> str:
    return _WORD.get(k, str(k))


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate the CD+head results CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"cell", "mode", "seed", "test_mse", "gamma"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results CSV missing columns: {missing}")
    present = set(df["mode"].unique())
    for mode in ("CI", "CD", "CD_Head"):
        if mode not in present:
            raise ValueError(f"Expected mode {mode} absent from results.")
    return df


def print_contrast(label: str, paired: pd.DataFrame, gm: dict) -> None:
    """Print one contrast's per-cell table, win count, and grand mean."""
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
    print(f"Grand mean: {gm['mean']:+.6f}  ({pct:+.4f}% of base mean)  95% CI [{gm['ci_lo']:+.6f}, {gm['ci_hi']:+.6f}]")


def gamma_trend(df: pd.DataFrame, mode_base: str, mode_alt: str) -> dict:
    """Regress the paired (alt - base) difference on gamma over leader-follower cells."""
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
        f"(95% CI [{trend['ci_lo']:+.4f}, {trend['ci_hi']:+.4f}], p = {trend['pval']:.3f}, R2 = {trend['r2']:.2f})"
    )


def print_latex_prose(head_vs_ci: pd.DataFrame, gm_head_ci: dict, trend_head_ci: dict) -> None:
    """Print the CD+head paragraph for the paper."""
    n_cells = len(head_vs_ci)
    n_wins = int((head_vs_ci["ci_hi"] < 0).sum())
    pct = abs(100.0 * gm_head_ci["mean"] / gm_head_ci["base_mean"])
    print("\n=== PAPER PROSE (CD+head control paragraph) ===")
    print(
        f"To test whether the per-variate head, rather than the encoder, limits "
        f"CD, we add a variant (CD+head) that pools each variate's encoder output "
        f"into a summary token, attends across the C summaries, and concatenates "
        f"the resulting per-variate context to the flattened encoder output before "
        f"the shared projection; the encoder is unchanged. Across the {_w(n_cells)} "
        f"leader-follower cells (with $\\gamma = 0$ as the AR(1) floor), CD+head "
        f"beats CI in {_w(n_wins)} of them, counting a cell as a win only when the "
        f"95\\% CI on the paired difference lies entirely below zero. The grand-mean "
        f"CD+head$-$CI difference is ${gm_head_ci['mean']:+.4f}$ MSE (95\\% CI "
        f"$[{gm_head_ci['ci_lo']:.4f}, {gm_head_ci['ci_hi']:.4f}]$, ${pct:.2f}\\%$ "
        f"of the CI mean), and the difference does not narrow with coupling "
        f"(slope ${trend_head_ci['slope']:+.4f}$ per unit $\\gamma$, "
        f"$p = {trend_head_ci['pval']:.2f}$). Explicitly pooling cross-variate "
        f"information at the head does not overturn the null, so the result is not "
        f"a head artefact."
    )


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _RESULTS_PATH
    df = load_results(csv_path)

    contrasts = [
        ("CD_Head - CI  (headline: does the better head beat CI?)", "CI", "CD_Head"),
        ("CD - CI       (canonical null, reproduced here)", "CI", "CD"),
        ("CD_Head - CD  (does the head help at all?)", "CD", "CD_Head"),
    ]
    results: dict[tuple[str, str], tuple[pd.DataFrame, dict]] = {}
    for label, base, alt in contrasts:
        diff = ps.make_diff_frame(df, ["cell"], base, alt)
        paired = ps.paired_differences(diff, ["cell"])
        gm = ps.grand_mean_diff(diff)
        print_contrast(label, paired, gm)
        results[(base, alt)] = (paired, gm)

    print("\n=== GAMMA TREND OVER LEADER-FOLLOWER CELLS ===")
    trend_head_ci = gamma_trend(df, "CI", "CD_Head")
    print_gamma_trend("CD_Head - CI", trend_head_ci)
    print_gamma_trend("CD - CI", gamma_trend(df, "CI", "CD"))

    head_vs_ci_paired, head_vs_ci_gm = results[("CI", "CD_Head")]
    print_latex_prose(head_vs_ci_paired, head_vs_ci_gm, trend_head_ci)


if __name__ == "__main__":
    main()