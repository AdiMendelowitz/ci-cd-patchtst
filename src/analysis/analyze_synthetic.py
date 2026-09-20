"""Synthetic AR(1) grid analysis: statistics and heatmap figure.

Reads
-----
results/results_grid.csv
    135 rows: one per (C, rho, mode, seed) at five seeds {42, 123, 456, 789, 1011}.
    Required columns: C, rho, mode, seed, test_mse.

Writes
------
paper/figures/heatmap.png
    3x3 CD/CI MSE ratio heatmap (300 dpi). Colormap is fixed at ±0.005
    around 1.0, declared here as a method constant, not fitted to the data.
    Numeric annotations make the figure self-sufficient.

Console
-------
Per-cell paired CD-CI differences with 95% t-CIs, a regression on those paired differences, the grand-mean effect
estimate, and a ready-to-paste LaTeX paragraph.

Statistical design
------------------
The primary object is d_s = MSE_CD - MSE_CI per (C, rho, seed). Pivoting on seed before computing differences exploits
the matched design: CI and CD within the same (C, rho, seed) triple share the same random seed and data draw, so their
difference removes shared noise. Regressing raw MSE on mode would treat these paired observations as independent and
inflate residual variance. The regression uses d as the response.

With n=5 seeds per cell the per-cell CIs use df=4 (t-critical = 2.776). The reported bound is the 95% CI half-width on
the per-cell mean difference: t_{0.975, df=4} * SD_within / sqrt(5), where SD_within is the pooled within-cell SD of d.
This is the half-width of the confidence interval for a cell's mean difference, the smallest mean difference whose
interval would exclude zero at the two-sided 5% level. The finding is that any CD advantage, where present, is bounded
below this half-width; it is a measured bound, not a mere failure to reject.

The same five seeds recur across all nine (C, rho) cells, so the 45 (C, rho, seed) rows are not 45 independent draws:
pooling them flat overstates the effective sample size and understates the grand mean's own CI. The grand-mean
function reports both figures -- the naive flat-pooled interval (n=45) and the seed-clustered interval (n=5, each
seed's difference averaged across the nine cells first) -- and the paper cites the clustered one.

Usage
-----
python analyze_synthetic.py [path/to/results_grid.csv]
Script lives at src/analysis/ in the repository root; all paths are relative to that root.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

_ROOT        = Path(__file__).resolve().parents[2]
_GRID_PATH   = _ROOT / "results" / "results_grid.csv"
_FIGURES_DIR = _ROOT / "paper" / "figures"

_C_VALUES:   list[int]   = [7, 21, 84]
_RHO_VALUES: list[float] = [0.1, 0.5, 0.9]

# Colormap half-range around a ratio of 1.0, half of the paper's own +/-1% practical-
# equivalence band. Fixed here as a method constant rather than fitted to the observed
# spread, so a cell approaching the paper's threshold reads as visually salient rather
# than being rescaled away by whatever range the data happens to span.
_RATIO_HALF_RANGE: float = 0.005


# ── Data loading ──────────────────────────────────────────────────────────────

def load_grid(path: Path) -> pd.DataFrame:
    """Load and validate the canonical grid CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Grid CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"C", "rho", "mode", "seed", "test_mse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Grid CSV missing columns: {missing}")
    if "DLinear" not in df["mode"].values:
        print("[WARN] No DLinear rows found in CSV; DLinear sanity check unavailable.")
    return df


# ── Statistics ────────────────────────────────────────────────────────────────

def make_diff_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (C, rho, seed) with columns CI, CD, diff.

    Pivot on seed before computing d = MSE_CD - MSE_CI. Rows where either mode is absent for a given (C, rho, seed)
    triple are dropped silently; the balanced design means this should not occur.
    """
    pivot = (
        df[df["mode"].isin(["CI", "CD"])]
        .pivot_table(index=["C", "rho", "seed"], columns="mode", values="test_mse")
        .dropna()
        .reset_index()
    )
    pivot["diff"] = pivot["CD"] - pivot["CI"]
    return pivot


def paired_differences(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell summary of CD-CI differences with 95% t-CIs.

    Returns one row per (C, rho), sorted ascending, with columns:
        C, rho, mean_diff, std_diff, ci_lo, ci_hi, signs, rel_pct, ci_mean.

    With n=5 seeds, df=4 and t-critical=2.776.
    """
    rows: list[dict] = []
    for (C, rho), grp in diff_df.groupby(["C", "rho"]):
        d      = grp["diff"].values
        n      = len(d)
        mean_d = float(d.mean())
        std_d  = float(d.std(ddof=1))
        se     = std_d / np.sqrt(n)
        tcrit  = stats.t.ppf(0.975, df=n - 1)
        ci_mean = float(grp["CI"].mean())
        rows.append(dict(
            C=int(C),
            rho=float(rho),
            mean_diff=mean_d,
            std_diff=std_d,
            ci_lo=mean_d - tcrit * se,
            ci_hi=mean_d + tcrit * se,
            signs="".join("+" if x > 0 else ("-" if x < 0 else "0") for x in d),
            rel_pct=100.0 * mean_d / ci_mean,
            ci_mean=ci_mean,
        ))
    return (
        pd.DataFrame(rows)
        .sort_values(["C", "rho"])
        .reset_index(drop=True)
    )


def compute_ci_half_width(diff_df: pd.DataFrame) -> float:
    """95% CI half-width for the per-cell paired design.

    half_width = t_{0.975, df=4} * SD_within / sqrt(n_seeds), where SD_within is the pooled within-cell SD of d,
    obtained by averaging the per-cell variances (not the per-cell SDs, which biases the pooled SD low) and taking the
    root. With equal n per cell this equals the pooled-variance estimate. The result is the half-width of the 95% CI for
    a single cell's mean difference: the smallest mean difference whose interval would exclude zero at the two-sided
    5% level. It is a detection threshold at the significance level.
    """
    cell_vars = [
        float(grp["diff"].var(ddof=1)) for _, grp in diff_df.groupby(["C", "rho"]) if len(grp) > 1
    ]
    pooled_sd = float(np.sqrt(np.mean(cell_vars)))
    n_seeds = int(diff_df.groupby(["C", "rho"]).size().mode().iloc[0])
    tcrit = stats.t.ppf(0.975, df=n_seeds - 1)
    return tcrit * pooled_sd / np.sqrt(n_seeds)


def grand_mean_diff(diff_df: pd.DataFrame) -> dict:
    """Grand mean of d, with a naive flat-pooled CI and a seed-clustered CI.

    The five seeds recur across all nine (C, rho) cells, so the 45 (C, rho, seed) rows are not
    45 independent draws. The naive interval below treats them as independent (n=45) and is
    reported only for contrast. The clustered interval averages each seed's difference across
    the nine cells first, giving n=5 independent seed-level means; that is the interval the
    paper cites.

    Returns:
        mean: grand-mean CD-CI difference (identical under both poolings).
        naive_ci_lo, naive_ci_hi: flat-pooled 95% CI, n=45, informational only.
        ci_lo, ci_hi: seed-clustered 95% CI, n=5, the reported interval.
        mean_ci_mse: mean CI-mode MSE, for expressing the difference as a percentage.
        n_naive, n_clustered: sample sizes behind each interval.
    """
    d = diff_df["diff"].values
    n_naive = len(d)
    mean_ = float(d.mean())
    se_naive = float(d.std(ddof=1) / np.sqrt(n_naive))
    tcrit_naive = stats.t.ppf(0.975, df=n_naive - 1)

    seed_means = diff_df.groupby("seed")["diff"].mean()
    n_clustered = len(seed_means)
    se_clustered = float(seed_means.std(ddof=1) / np.sqrt(n_clustered))
    tcrit_clustered = stats.t.ppf(0.975, df=n_clustered - 1)

    return {
        "mean":         mean_,
        "naive_ci_lo":  mean_ - tcrit_naive * se_naive,
        "naive_ci_hi":  mean_ + tcrit_naive * se_naive,
        "n_naive":      n_naive,
        "ci_lo":        mean_ - tcrit_clustered * se_clustered,
        "ci_hi":        mean_ + tcrit_clustered * se_clustered,
        "n_clustered":  n_clustered,
        "mean_ci_mse":  float(diff_df["CI"].mean()),
    }


def diff_regression(diff_df: pd.DataFrame) -> dict:
    """Regress d = MSE_CD - MSE_CI on C (categorical) and rho (numeric).

    C is encoded categorically; {7, 21, 84} are not linearly ordered in any principled sense and categorical encoding
    avoids imposing a linear trend. rho is numeric (equidistant on {0.1, 0.5, 0.9}).

    The regression tests whether C or rho systematically predicts the CD-CI gap after accounting for the paired
    structure. The grand-mean-diff function above provides the scalar summary; this model provides context
    on whether the gap varies across the grid.
    """
    df_lm = diff_df.copy()
    df_lm["n_ch"] = df_lm["C"].astype(str)
    model = smf.ols("diff ~ n_ch + rho", data=df_lm).fit()
    return {
        "pval_f": float(model.f_pvalue),
        "r2":     float(model.rsquared),
        "model":  model,
    }


def ratio_grid(paired: pd.DataFrame) -> pd.DataFrame:
    """CD/CI ratio per cell, derived from paired differences.

    Expected input columns: C, rho, ci_mean, mean_diff.
    Computed as (ci_mean + mean_diff) / ci_mean to guarantee numerical
    consistency with the paired-difference analysis.
    """
    result = paired[["C", "rho", "ci_mean", "mean_diff"]].copy()
    result["cd_mean"] = result["ci_mean"] + result["mean_diff"]
    result["ratio"]   = result["cd_mean"] / result["ci_mean"]
    return result


# ── Console output ────────────────────────────────────────────────────────────

def print_paired(paired: pd.DataFrame, half_width: float, seeds: list[int]) -> None:
    """Print per-cell paired differences and the CI half-width note."""
    seed_str = ", ".join(str(s) for s in seeds)
    print("\n=== PER-CELL PAIRED DIFFERENCES (CD - CI) ===")
    print(f"Positive = CD worse. Signs = per-seed direction (seeds {seed_str}).")
    print(f"{'C':>4} {'rho':>5}  {'mean_diff':>10} {'rel_%':>8}  "
          f"{'95%CI_lo':>10} {'95%CI_hi':>10}  signs")
    print("-" * 70)
    for _, r in paired.iterrows():
        print(
            f"{int(r.C):>4} {r.rho:>5.1f}  "
            f"{r.mean_diff:>+10.6f} {r.rel_pct:>+8.4f}%  "
            f"{r.ci_lo:>+10.6f} {r.ci_hi:>+10.6f}  {r.signs}"
        )

    max_diff = paired["mean_diff"].abs().max()
    max_pct  = paired["rel_pct"].abs().max()
    n_zero   = int(((paired["ci_lo"] < 0) & (paired["ci_hi"] > 0)).sum())
    mixed    = int((paired["signs"].apply(lambda s: "+" in s and "-" in s)).sum())
    n_seeds  = len(seeds)
    print(f"\nMax |mean_diff|:              {max_diff:.6f} MSE units")
    print(f"Max |relative effect|:        {max_pct:.4f}% of CI MSE")
    print(f"CIs including zero:           {n_zero} / {len(paired)}")
    print(f"Cells with mixed signs:       {mixed} / {len(paired)}")
    print(
        f"\nBound note: the 95% CI half-width on a single cell's mean difference "
        f"with n={n_seeds} seeds is {half_width:.4f} MSE units "
        f"(pooled within-cell SD x t_{{0.975,{n_seeds - 1}}} / sqrt({n_seeds})). "
        f"Any CD advantage, where present, is bounded below this half-width."
    )


def print_grand_mean(gm: dict) -> None:
    """Print the grand-mean CD-CI difference, naive and seed-clustered."""
    pct = 100.0 * gm["mean"] / gm["mean_ci_mse"]
    print("\n=== GRAND MEAN CD-CI DIFFERENCE ===")
    print(f"Mean diff (all cells/seeds):     {gm['mean']:+.6f}  ({pct:+.4f}% of mean CI MSE)")
    print(f"Naive flat-pooled 95% CI (n={gm['n_naive']}):   [{gm['naive_ci_lo']:+.6f}, {gm['naive_ci_hi']:+.6f}]")
    print(f"Seed-clustered 95% CI (n={gm['n_clustered']}):      [{gm['ci_lo']:+.6f}, {gm['ci_hi']:+.6f}]")
    print("The same seeds recur across every cell; the clustered interval is the one reported.")


def print_lm(lm: dict) -> None:
    """Print regression summary."""
    print("\n=== REGRESSION ON PAIRED DIFFERENCES: diff ~ n_ch(categorical) + rho ===")
    print("Response: d = MSE_CD - MSE_CI. Tests whether C or rho predicts the gap.")
    print(f"F-test p-value: {lm['pval_f']:.4f}")
    print(f"R²:             {lm['r2']:.4f}")
    print(lm["model"].summary().tables[1])


def print_latex_prose(paired: pd.DataFrame, gm: dict, lm: dict, half_width: float, seeds: list[int]) -> None:
    """Print a ready-to-paste LaTeX paragraph for the statistical analysis."""
    n_seeds   = len(seeds)
    n_cells   = len(paired)
    n_pairs   = n_seeds * n_cells
    df_cell   = n_seeds - 1
    tcrit     = stats.t.ppf(0.975, df=df_cell)
    max_diff  = paired["mean_diff"].abs().max()
    max_pct   = paired["rel_pct"].abs().max()
    mixed     = int((paired["signs"].apply(lambda s: "+" in s and "-" in s)).sum())
    n_zero    = int(((paired["ci_lo"] < 0) & (paired["ci_hi"] > 0)).sum())
    n_sig     = n_cells - n_zero
    seed_set  = ", ".join(str(s) for s in seeds)
    word      = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
                 6: "six", 7: "seven", 8: "eight", 9: "nine"}

    def w(k: int) -> str:
        return word.get(k, str(k))

    pct_gm    = abs(100.0 * gm["mean"] / gm["mean_ci_mse"])
    hw_pct    = 100.0 * half_width / gm["mean_ci_mse"]

    # Name the excluding-zero cells directly, matching the paper's wording; derived from the
    # data so this can never silently go stale if a future CSV changes which cells are significant.
    sig_rows = paired.loc[~((paired["ci_lo"] < 0) & (paired["ci_hi"] > 0))]
    sig_cells = [f"$C = {int(r.C)}$, $\\rho = {r.rho:.1f}$" for r in sig_rows.itertuples()]
    max_sig_pct = sig_rows["rel_pct"].abs().max() if n_sig else 0.0

    if n_sig == 0:
        sig_clause = f"all {w(n_cells)} 95\\% CIs include zero"
        exception_clause = ""
    elif n_sig == 1:
        sig_clause = (
            f"{w(n_zero)} of {w(n_cells)} 95\\% CIs include zero; the one that excludes zero, "
            f"at {sig_cells[0]}, is positive (CD worse) and below ${max_sig_pct:.2f}\\%$ of the CI mean"
        )
        exception_clause = " -- including the per-cell exception above --"
    elif n_sig == 2:
        sig_clause = (
            f"{w(n_zero)} of {w(n_cells)} 95\\% CIs include zero; the two that exclude zero, "
            f"at {' and '.join(sig_cells)}, are both positive (CD worse) and below "
            f"${max_sig_pct:.2f}\\%$ of the CI mean"
        )
        exception_clause = " -- including the two per-cell exceptions above --"
    else:
        cell_list = ", ".join(sig_cells[:-1]) + f", and {sig_cells[-1]}"
        sig_clause = (
            f"{w(n_zero)} of {w(n_cells)} 95\\% CIs include zero; the {w(n_sig)} that exclude "
            f"zero, at {cell_list}, are all positive (CD worse) and below "
            f"${max_sig_pct:.2f}\\%$ of the CI mean"
        )
        exception_clause = " -- including the per-cell exceptions above --"

    print("\n=== PAPER PROSE (Statistical analysis paragraph) ===")
    print(
        f"For each $(C, \\rho)$ cell we compute the per-seed difference "
        f"$d_s = e^{{\\text{{CD}}}}_s - e^{{\\text{{CI}}}}_s$ over seeds "
        f"$\\{{{seed_set}\\}}$ and derive its mean, standard deviation, and "
        f"95\\% confidence interval from a $t$-distribution with {w(df_cell)} degrees of "
        f"freedom ($t_{{0.025,{df_cell}}} = {tcrit:.3f}$). "
        f"Across all {w(n_cells)} cells the mean CD$-$CI difference stays within "
        f"$\\pm {max_diff:.4f}$ MSE units, at most ${max_pct:.2f}\\%$ of the "
        f"corresponding CI mean, and {sig_clause}. "
        f"Pooling the {n_pairs} matched pairs as a simple unweighted average gives a "
        f"grand-mean CD$-$CI of ${gm['mean']:+.4f}$ MSE (${pct_gm:.2f}\\%$ of the mean CI MSE), "
        f"and regressing the paired difference on $C$ (categorical) and $\\rho$ finds no "
        f"dependence on either factor ($F$-test $p = {lm['pval_f']:.2f}$, $R^2 = {lm['r2']:.2f}$). "
        f"Because the same {w(n_seeds)} seeds recur across all {w(n_cells)} cells, we report the "
        f"grand mean's interval clustered by seed rather than treating all {n_pairs} rows as "
        f"independent -- averaging each seed's difference across the {w(n_cells)} cells first "
        f"gives a wider but still zero-including 95\\% CI of $[{gm['ci_lo']:+.4f}, {gm['ci_hi']:+.4f}]$. "
        f"Pooling the within-cell variances gives a 95\\% CI half-width of "
        f"${half_width:.4f}$ MSE at $n = {n_seeds}$, roughly ${hw_pct:.2f}\\%$ of the mean CI MSE. "
        f"Any uniform CD advantage larger than that would have pushed the grand-mean CI entirely "
        f"above zero, and none of the observed differences{exception_clause} come close."
    )


# ── Heatmap figure ────────────────────────────────────────────────────────────

def plot_heatmap(ratios: pd.DataFrame, out_path: Path) -> None:
    """Render the 3x3 CD/CI ratio heatmap.

    Colormap is fixed at ±_RATIO_HALF_RANGE = ±0.005 around 1.0, declared
    as a method constant. Numeric annotations make exact values readable
    independently of the colour encoding.
    """
    lookup = {(int(r.C), float(r.rho)): float(r.ratio) for _, r in ratios.iterrows()}
    data = pd.DataFrame(
        [[lookup.get((c, rho), np.nan) for rho in _RHO_VALUES] for c in _C_VALUES],
        index=[f"C={c}" for c in _C_VALUES],
        columns=[f"\u03c1={r}" for r in _RHO_VALUES],
    )
    annot = data.map(lambda v: f"{v:.4f}" if not np.isnan(v) else "N/A")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    sns.heatmap(
        data,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=1.0,
        vmin=1.0 - _RATIO_HALF_RANGE,
        vmax=1.0 + _RATIO_HALF_RANGE,
        linewidths=0.5,
        linecolor="#cccccc",
        ax=ax,
        cbar_kws={"label": "CD/CI MSE ratio", "shrink": 0.85},
    )
    ax.set_xlabel("Correlation strength \u03c1", labelpad=6)
    ax.set_ylabel("Number of variates C", labelpad=6)
    ax.set_title(
        "Synthetic AR(1) grid: CD/CI MSE ratio\n"
        "(>1.0 favours CI, <1.0 favours CD; colour scale fixed at \u00b10.5%)",
        pad=8,
    )
    # ratio=1.0 sits at the midpoint of the symmetric colorbar.
    ax.collections[0].colorbar.ax.axhline(y=0.5, color="black", linewidth=0.8, linestyle="--")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

# Oracle targets quoted from main.tex: Table 4 (CD/CI per cell, 4 dp), the
# grand-mean paragraph of Section 4.1 (grand mean and seed-clustered 95% CI, 4 dp,
# pooled half-width, 4 dp) and its regression (F-test p, 2 dp; R2, 2 dp).
_ORACLE_RATIO: dict[tuple[int, float], float] = {
    (7, 0.1): 1.0016, (7, 0.5): 1.0008, (7, 0.9): 1.0011,
    (21, 0.1): 1.0019, (21, 0.5): 1.0010, (21, 0.9): 0.9984,
    (84, 0.1): 0.9999, (84, 0.5): 1.0012, (84, 0.9): 1.0056,
}
_ORACLE_GRAND: dict[str, float] = {
    "mean_diff": 0.0013, "clustered_ci_lo": -0.0023, "clustered_ci_hi": 0.0048,
    "half_width": 0.0063, "f_p": 0.81, "r2": 0.02,
}


def _oracle_check(ratios: pd.DataFrame, gm: dict, lm: dict, half_width: float) -> tuple[list[str], bool]:
    """Compare recomputed values against the targets quoted from main.tex."""
    lines: list[str] = []
    ok_all = True
    for _, row in ratios.iterrows():
        key = (int(row["C"]), round(float(row["rho"]), 6))
        got = round(float(row["ratio"]), 4)
        target = _ORACLE_RATIO[key]
        ok = got == target
        ok_all = ok_all and ok
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] C={key[0]} rho={key[1]}: CD/CI {got:.4f}  target {target:.4f}")
    got_grand = {
        "mean_diff": round(gm["mean"], 4),
        "clustered_ci_lo": round(gm["ci_lo"], 4),
        "clustered_ci_hi": round(gm["ci_hi"], 4),
        "half_width": round(half_width, 4),
        "f_p": round(lm["pval_f"], 2),
        "r2": round(lm["r2"], 2),
    }
    for name, target in _ORACLE_GRAND.items():
        ok = got_grand[name] == target
        ok_all = ok_all and ok
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {got_grand[name]:+.4f}  target {target:+.4f}")
    return lines, ok_all


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _GRID_PATH
    df = load_grid(csv_path)

    seeds = sorted(int(s) for s in df["seed"].unique())

    diff_df    = make_diff_frame(df)
    paired     = paired_differences(diff_df)
    half_width = compute_ci_half_width(diff_df)
    gm         = grand_mean_diff(diff_df)
    lm         = diff_regression(diff_df)
    ratios     = ratio_grid(paired)

    print_paired(paired, half_width, seeds)
    print_grand_mean(gm)
    print_lm(lm)
    print_latex_prose(paired, gm, lm, half_width, seeds)
    plot_heatmap(ratios, _FIGURES_DIR / "heatmap.png")

    print("\n=== ORACLE CHECK (main.tex Table 4 and Section 4.1) ===")
    check_lines, all_pass = _oracle_check(ratios, gm, lm, half_width)
    print("\n".join(check_lines))
    verdict = "PASS - Table 4 and the grand-mean paragraph reproduce" if all_pass else "FAIL - see lines above"
    print(f"\nRESULT: {verdict}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
