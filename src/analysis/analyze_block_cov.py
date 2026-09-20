"""Covariance-family control: block-diagonal-covariance AR(1) grid.

Reads
-----
results/results_block_cov.csv
    One row per (C, rho_in, mode, seed). Required columns: C, rho_in, mode, seed, test_mse. Modes: CI, CD, and
    optionally DLinear.

Writes
------
paper/figures/heatmap_block_cov.png
    CD/CI MSE ratio heatmap over (C, rho_in), fixed colour scale at +/-0.5%.
    Written only when every cell is complete (see the completeness gate below).

Console
-------
A completeness report, per-cell paired CD-CI differences with 95% t-CIs, the pooled CI half-width, the grand-mean
effect, a regression of the paired difference on C and rho_in, a DLinear sanity column, and a LaTeX paragraph. Cells
with fewer than two paired seeds are excluded from every statistic. The figure and the paragraph are emitted only when
all cells of the design are at the canonical seed count; until then the diagnostics still print for inspection.

Purpose
-------
The compound-symmetry grid (analyze_synthetic.py) shows no CD advantage from instantaneous correlation. This script
tests the same claim under block-diagonal covariance, the second covariance family, to replace the assertion that
compound symmetry is "most hostile to CD" with evidence across two families.

Usage
-----
python analyze_block_cov.py [path/to/results_block_cov.csv]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import paired_stats as ps

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_PATH = _ROOT / "results" / "results_block_cov.csv"
_FIGURES_DIR = _ROOT / "paper" / "figures"

_CELL_COLS = ["C", "rho_in"]
_RATIO_HALF_RANGE = 0.005  # heatmap colour scale, +/- this fraction around 1.0

_MIN_SEEDS = 2  # fewest paired seeds that still admit a t-CI
_EXPECTED_N = 5  # canonical seeds per cell; prose and figure are withheld until every cell reaches it
_EQUIV_BAND_PCT = 1.0  # pre-specified practical-equivalence band, percent of CI MSE

_WORD: dict[int, str] = {
    0: "none",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


def _w(k: int) -> str:
    return _WORD.get(k, str(k))


def _cell_key(c: object, r: object) -> tuple[int, float]:
    """Normalise a (C, rho_in) pair to native Python types for set membership."""
    return (int(c), float(r))


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate the block-covariance results CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"C", "rho_in", "mode", "seed", "test_mse"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results CSV missing columns: {missing}")
    present = set(df["mode"].unique())
    for mode in ("CI", "CD"):
        if mode not in present:
            raise ValueError(f"Expected mode {mode} absent from results.")
    if "DLinear" not in present:
        print("[WARN] No DLinear rows found; DLinear sanity check unavailable.")
    return df


def assess_completeness(df: pd.DataFrame, diff: pd.DataFrame) -> dict:
    """Classify every design cell by how many paired seeds it carries.

    A cell is usable for statistics when it has at least _MIN_SEEDS paired seeds.
    The design is ready for paper output when every cell present in the raw data
    is paired and at the canonical _EXPECTED_N.

    Args:
        df: Raw long-format results (defines the set of expected cells).
        diff: Paired-difference frame from paired_stats.make_diff_frame.

    Returns:
        Dict with usable (list of cell keys), and missing, too_few, incomplete
        (lists of (cell_key, n)), plus ready (bool).
    """
    counts = {_cell_key(idx[0], idx[1]): int(n) for idx, n in diff.groupby(_CELL_COLS).size().items()}
    expected = {_cell_key(r[0], r[1]) for r in df[_CELL_COLS].drop_duplicates().to_numpy()}
    present = set(counts)

    missing = sorted(expected - present)
    too_few = sorted((c, counts[c]) for c in present if counts[c] < _MIN_SEEDS)
    incomplete = sorted((c, counts[c]) for c in present if _MIN_SEEDS <= counts[c] < _EXPECTED_N)
    usable = sorted(c for c in present if counts[c] >= _MIN_SEEDS)
    ready = not missing and bool(present) and all(n == _EXPECTED_N for n in counts.values())

    return {"usable": usable, "missing": missing, "too_few": too_few, "incomplete": incomplete, "ready": ready}


def print_completeness(report: dict) -> None:
    """Print the completeness report and any exclusions."""
    print("\n=== COMPLETENESS ===")
    print(f"Cells usable for statistics (n>={_MIN_SEEDS} paired seeds): {len(report['usable'])}")
    for cell in report["missing"]:
        print(f"  [WARN] cell C={cell[0]} rho_in={cell[1]} absent from paired data (a mode is missing).")
    for cell, n in report["too_few"]:
        print(f"  [WARN] cell C={cell[0]} rho_in={cell[1]} has n={n} paired seed(s); excluded (no CI).")
    for cell, n in report["incomplete"]:
        print(f"  [INFO] cell C={cell[0]} rho_in={cell[1]} at n={n} of {_EXPECTED_N}; shown but not paper-ready.")
    print(f"Paper prose and figure: {'EMITTED' if report['ready'] else 'WITHHELD (design incomplete)'}")


def filter_to_cells(frame: pd.DataFrame, cells: list[tuple[int, float]]) -> pd.DataFrame:
    """Return the rows of frame whose (C, rho_in) is in cells."""
    keep = set(cells)
    mask = [_cell_key(c, r) in keep for c, r in zip(frame["C"], frame["rho_in"])]
    return frame[mask].reset_index(drop=True)


def diff_regression(diff_df: pd.DataFrame) -> dict:
    """Regress d = MSE_CD - MSE_CI on C (categorical) and rho_in (numeric).

    C is encoded categorically because {21, 84} carry no principled linear
    ordering; rho_in is numeric. The additive form matches analyze_synthetic.py.
    Returns the overall F-test, R^2, and the per-factor significance so the prose
    can report what the fit actually finds rather than asserting a fixed verdict.
    """
    df_lm = diff_df.copy()
    df_lm["n_ch"] = df_lm["C"].astype(str)
    model = smf.ols("diff ~ n_ch + rho_in", data=df_lm).fit()
    nch_terms = [t for t in model.params.index if t.startswith("n_ch")]
    p_c = float(model.f_test(" = 0, ".join(nch_terms) + " = 0").pvalue) if nch_terms else float("nan")
    return {
        "pval_f": float(model.f_pvalue),
        "r2": float(model.rsquared),
        "p_rho_in": float(model.pvalues["rho_in"]),
        "coef_rho_in": float(model.params["rho_in"]),
        "p_C": p_c,
        "model": model,
    }


def dlinear_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell CD/CI and DLinear/CI mean-MSE ratios for the sanity check."""
    means = df.groupby(_CELL_COLS + ["mode"])["test_mse"].mean().unstack("mode")
    out = pd.DataFrame(index=means.index)
    out["CD/CI"] = means["CD"] / means["CI"]
    if "DLinear" in means.columns:
        out["DLinear/CI"] = means["DLinear"] / means["CI"]
    return out.reset_index()


def print_paired(paired: pd.DataFrame, half_width: float, seeds: list[int]) -> None:
    """Print per-cell paired differences and the half-width bound note."""
    seed_str = ", ".join(str(s) for s in seeds)
    print("\n=== PER-CELL PAIRED DIFFERENCES (CD - CI) ===")
    print(f"Positive = CD worse. Signs = per-seed direction (seeds {seed_str}).")
    print(f"{'C':>4} {'rho_in':>7}  {'mean_diff':>10} {'rel_%':>8}  {'95%CI_lo':>10} {'95%CI_hi':>10}  signs")
    print("-" * 74)
    for _, r in paired.iterrows():
        print(
            f"{int(r.C):>4} {r.rho_in:>7.1f}  {r.mean_diff:>+10.6f} {r.rel_pct:>+8.4f}%  "
            f"{r.ci_lo:>+10.6f} {r.ci_hi:>+10.6f}  {r.signs}"
        )
    n_zero = int(((paired["ci_lo"] < 0) & (paired["ci_hi"] > 0)).sum())
    n_cd_wins = int((paired["ci_hi"] < 0).sum())
    n_seeds = len(seeds)
    print(f"\nMax |mean_diff|:        {paired['mean_diff'].abs().max():.6f} MSE units")
    print(f"Max |relative effect|:  {paired['rel_pct'].abs().max():.4f}% of CI MSE")
    print(f"CIs including zero:      {n_zero} / {len(paired)}")
    print(f"Cells where CD beats CI (CI excludes zero, negative): {n_cd_wins} / {len(paired)}")
    print(
        f"Bound: 95% CI half-width on a cell mean difference at n={n_seeds} is "
        f"{half_width:.4f} MSE (pooled within-cell SD x t_{{0.975,{n_seeds - 1}}} / sqrt({n_seeds}))."
    )


def print_grand_mean(gm: dict) -> None:
    """Print the grand-mean CD-CI difference."""
    pct = 100.0 * gm["mean"] / gm["base_mean"]
    print("\n=== GRAND MEAN CD-CI DIFFERENCE (block covariance) ===")
    print(f"Mean diff (all cells/seeds):  {gm['mean']:+.6f}  ({pct:+.4f}% of mean CI MSE)")
    print(f"95% CI:                       [{gm['ci_lo']:+.6f}, {gm['ci_hi']:+.6f}]")


def print_lm(lm: dict) -> None:
    """Print the regression summary."""
    print("\n=== REGRESSION: diff ~ n_ch(categorical) + rho_in ===")
    print(f"F-test p-value: {lm['pval_f']:.4f}   R2: {lm['r2']:.4f}")
    print(lm["model"].summary().tables[1])


def print_dlinear(ratios: pd.DataFrame) -> None:
    """Print the DLinear sanity ratios."""
    print("\n=== DLinear SANITY (per-cell mean-MSE ratios) ===")
    print(ratios.to_string(index=False))


def _regression_clause(lm: dict) -> str:
    """Describe what the regression finds, conditioned on the per-factor p-values."""
    common = f"$F$-test $p = {lm['pval_f']:.3f}$, $R^2 = {lm['r2']:.2f}$"
    factors: list[str] = []
    detail: list[str] = []
    if lm["p_rho_in"] < 0.05:
        factors.append("$\\rho_{\\text{in}}$")
        detail.append(f"slope ${lm['coef_rho_in']:+.4f}$, $p = {lm['p_rho_in']:.3f}$")
    if lm["p_C"] < 0.05:
        factors.append("$C$")
        detail.append(f"$C$ $p = {lm['p_C']:.3f}$")
    if not factors:
        return f"finds no dependence on either factor ({common})"
    inner = "; ".join(detail + [common])
    return f"finds it varies with {' and '.join(factors)} ({inner})"


def _equivalence_clause(paired: pd.DataFrame, lm: dict) -> str:
    """Closing sentence, conditioned on the equivalence band and the rho_in trend."""
    max_pct = float(paired["rel_pct"].abs().max())
    within = max_pct <= _EQUIV_BAND_PCT
    trend = ""
    if lm["p_rho_in"] < 0.05:
        direction = "favours CD" if lm["coef_rho_in"] < 0 else "favours CI"
        trend = f", though within the band the gap {direction} as within-group correlation rises"
    if within:
        return (
            f"Every cell stays within the {_EQUIV_BAND_PCT:.0f}\\% practical-equivalence band "
            f"(largest ${max_pct:.2f}\\%$){trend}, so neither mode gains a practically relevant "
            f"advantage under block structure, as under compound symmetry."
        )
    return (
        f"The largest per-cell gap is ${max_pct:.2f}\\%$, past the {_EQUIV_BAND_PCT:.0f}\\% "
        f"equivalence band{trend}, the one departure from the compound-symmetry result."
    )


def print_latex_prose(paired: pd.DataFrame, gm: dict, lm: dict, half_width: float, n_pairs: int) -> None:
    """Print a ready-to-paste LaTeX paragraph for the block-covariance result."""
    n_cells = len(paired)
    n_zero = int(((paired["ci_lo"] < 0) & (paired["ci_hi"] > 0)).sum())
    n_cd_wins = int((paired["ci_hi"] < 0).sum())
    max_pct = float(paired["rel_pct"].abs().max())
    pct_gm = abs(100.0 * gm["mean"] / gm["base_mean"])
    hw_pct = 100.0 * half_width / gm["base_mean"]

    print("\n=== PAPER PROSE (block-covariance robustness paragraph) ===")
    print(
        f"To test whether the compound-symmetry result is an artefact of that covariance shape, "
        f"we repeat the experiment with block-diagonal covariance (groups of seven channels, "
        f"within-group correlation $\\rho_{{\\text{{in}}}} \\in \\{{0.5, 0.9\\}}$, zero "
        f"between-group correlation) at $C \\in \\{{21, 84\\}}$ and five seeds, holding the "
        f"transition diagonal so Granger non-causality is preserved. Across {_w(n_cells)} cells "
        f"the grand-mean CD$-$CI difference is ${gm['mean']:+.4f}$ MSE (95\\% CI "
        f"$[{gm['ci_lo']:.4f}, {gm['ci_hi']:.4f}]$, ${pct_gm:.2f}\\%$ of the mean CI MSE), no "
        f"cell exceeds ${max_pct:.2f}\\%$ of its CI mean, {_w(n_zero)} of {_w(n_cells)} per-cell "
        f"95\\% CIs include zero, and CD beats CI in {_w(n_cd_wins)} of {_w(n_cells)} cells. "
        f"Regressing the paired difference on $C$ (categorical) and $\\rho_{{\\text{{in}}}}$ "
        f"{_regression_clause(lm)}; the pooled 95\\% CI half-width is ${half_width:.4f}$ MSE "
        f"(${hw_pct:.2f}\\%$ of the CI mean) over {n_pairs} matched pairs. "
        f"{_equivalence_clause(paired, lm)}"
    )


def plot_heatmap(paired: pd.DataFrame, out_path: Path) -> None:
    """Render the CD/CI ratio heatmap over (C, rho_in)."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    paired = paired.copy()
    paired["ratio"] = (paired["base_mean"] + paired["mean_diff"]) / paired["base_mean"]
    c_values = sorted(paired["C"].unique())
    rho_values = sorted(paired["rho_in"].unique())
    lookup = {(int(r.C), float(r.rho_in)): float(r.ratio) for _, r in paired.iterrows()}
    grid = pd.DataFrame(
        [[lookup.get((c, rho), np.nan) for rho in rho_values] for c in c_values],
        index=[f"C={c}" for c in c_values],
        columns=[f"\u03c1_in={r}" for r in rho_values],
    )
    annot = grid.map(lambda v: f"{v:.4f}" if not np.isnan(v) else "N/A")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    sns.heatmap(
        grid,
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
    ax.set_xlabel("Within-group correlation \u03c1_in", labelpad=6)
    ax.set_ylabel("Number of variates C", labelpad=6)
    ax.set_title("Block-covariance AR(1): CD/CI MSE ratio\n(>1.0 favours CI; scale fixed at \u00b10.5%)", pad=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _RESULTS_PATH
    df = load_results(csv_path)

    diff = ps.make_diff_frame(df, _CELL_COLS, "CI", "CD")
    report = assess_completeness(df, diff)
    print_completeness(report)

    if not report["usable"]:
        print("\nNo cell has enough paired seeds to analyse yet.")
        return

    diff_usable = filter_to_cells(diff, report["usable"])
    df_usable = filter_to_cells(df, report["usable"])
    seeds = sorted(int(s) for s in diff_usable["seed"].unique())

    paired = ps.paired_differences(diff_usable, _CELL_COLS)
    half_width = ps.compute_ci_half_width(diff_usable, _CELL_COLS)
    gm = ps.grand_mean_diff(diff_usable)
    lm = diff_regression(diff_usable)

    print_paired(paired, half_width, seeds)
    print_grand_mean(gm)
    print_lm(lm)
    if "DLinear" in set(df_usable["mode"].unique()):
        print_dlinear(dlinear_ratios(df_usable))

    if not report["ready"]:
        print("\n=== PAPER PROSE AND FIGURE WITHHELD ===")
        print("The design is not yet complete at the canonical seed count; rerun once every")
        print("cell is filled to emit the paragraph and write the figure.")
        return

    print_latex_prose(paired, gm, lm, half_width, len(diff_usable))
    plot_heatmap(paired, _FIGURES_DIR / "heatmap_block_cov.png")


if __name__ == "__main__":
    main()