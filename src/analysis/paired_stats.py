"""Shared paired-difference statistics for the CI/CD control experiments.

Every experiment in this set compares two PatchTST modes that share a seed and a
data draw within each cell, so the matched unit of analysis is the per-seed
difference d_s = MSE_alt - MSE_base. This module factors out the per-cell t-CIs,
the pooled CI half-width, and the grand-mean estimate so the experiment scripts
(analyze_cd_head.py, analyze_equal_compute.py, analyze_block_cov.py) apply one
verified implementation rather than three copies. The formulas are identical to
analyze_synthetic.py; only the cell key and the pair of modes are parameterised.

Sign convention: diff = MSE_alt - MSE_base, so a positive difference means the
alternative mode is worse. Choose base and alt so that "positive = the mode under
scrutiny is worse" reads naturally (for example base="CI", alt="CD").
"""

import numpy as np
import pandas as pd
from scipy import stats

_BASE = "_base"
_ALT = "_alt"


def make_diff_frame(
    df: pd.DataFrame,
    cell_cols: list[str],
    mode_base: str,
    mode_alt: str,
    value: str = "test_mse",
) -> pd.DataFrame:
    """Pivot to one row per (cell, seed) carrying base, alt, and their difference.

    Rows where either mode is missing for a (cell, seed) combination are dropped;
    a balanced design should not trigger that.

    Args:
        df: Long-format results with columns covering cell_cols, "seed", "mode",
            and the value column.
        cell_cols: Columns that define a cell (for example ["C", "rho_in"] or
            ["gamma"]).
        mode_base: Reference mode (subtracted).
        mode_alt: Mode under scrutiny (minuend).
        value: Metric column to difference. Defaults to "test_mse".

    Returns:
        DataFrame with cell_cols, "seed", "_base", "_alt", and "diff" columns,
        where diff = alt - base.

    Raises:
        ValueError: If a requested mode is absent from df.
    """
    present = set(df["mode"].unique())
    missing = {mode_base, mode_alt} - present
    if missing:
        raise ValueError(f"Modes absent from results: {missing}. Present: {sorted(present)}.")

    pivot = (
        df[df["mode"].isin([mode_base, mode_alt])]
        .pivot_table(index=cell_cols + ["seed"], columns="mode", values=value)
        .dropna()
        .reset_index()
        .rename(columns={mode_base: _BASE, mode_alt: _ALT})
    )
    pivot["diff"] = pivot[_ALT] - pivot[_BASE]
    return pivot


def paired_differences(diff_df: pd.DataFrame, cell_cols: list[str]) -> pd.DataFrame:
    """Per-cell paired difference summary with 95% t-CIs.

    Args:
        diff_df: Output of make_diff_frame.
        cell_cols: Columns that define a cell.

    Returns:
        One row per cell, sorted by cell_cols, with columns: cell_cols,
        mean_diff, std_diff, ci_lo, ci_hi, signs, rel_pct, base_mean, n.
        rel_pct is the mean difference as a percentage of the base-mode mean.
    """
    rows: list[dict] = []
    for key, grp in diff_df.groupby(cell_cols):
        d = grp["diff"].to_numpy()
        n = len(d)
        mean_d = float(d.mean())
        std_d = float(d.std(ddof=1))
        se = std_d / np.sqrt(n)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        base_mean = float(grp[_BASE].mean())
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(cell_cols, key_tuple))
        row.update(
            mean_diff=mean_d,
            std_diff=std_d,
            ci_lo=mean_d - tcrit * se,
            ci_hi=mean_d + tcrit * se,
            signs="".join("+" if x > 0 else ("-" if x < 0 else "0") for x in d),
            rel_pct=100.0 * mean_d / base_mean,
            base_mean=base_mean,
            n=n,
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(cell_cols).reset_index(drop=True)


def compute_ci_half_width(diff_df: pd.DataFrame, cell_cols: list[str]) -> float:
    """Pooled 95% CI half-width for a single cell's mean difference.

    half_width = t_{0.975, n-1} * pooled_SD / sqrt(n), where pooled_SD is the root
    of the mean per-cell variance of d. Averaging variances rather than SDs avoids
    biasing the pooled SD low. With equal n per cell this equals the
    pooled-variance estimate. The result is the smallest per-cell mean difference
    whose interval would exclude zero at the two-sided 5% level: a measured
    detection bound, not a mere failure to reject.

    Args:
        diff_df: Output of make_diff_frame.
        cell_cols: Columns that define a cell.

    Returns:
        The half-width in MSE units.
    """
    cell_vars = [float(grp["diff"].var(ddof=1)) for _, grp in diff_df.groupby(cell_cols) if len(grp) > 1]
    if not cell_vars:
        raise ValueError("No cell has more than one seed; cannot pool a within-cell variance.")
    pooled_sd = float(np.sqrt(np.mean(cell_vars)))
    n_seeds = int(diff_df.groupby(cell_cols).size().mode().iloc[0])
    tcrit = float(stats.t.ppf(0.975, df=n_seeds - 1))
    return tcrit * pooled_sd / np.sqrt(n_seeds)


def grand_mean_diff(diff_df: pd.DataFrame) -> dict:
    """Grand mean of d over all (cell, seed) observations with a 95% t-CI.

    The simplest unbiased scalar summary of the difference across the design. It
    requires no extrapolation, unlike an OLS intercept.

    Valid only when every (cell, seed) row is an independent observation. If the
    same seed value recurs across multiple cells -- as it does whenever a design
    reuses one seed set across several settings of a swept parameter -- those
    rows share a common source (the seed's own training/data-draw randomness) and
    are not independent; use grand_mean_diff_clustered instead in that case, or
    this will understate the interval width.

    Args:
        diff_df: Output of make_diff_frame.

    Returns:
        Dict with mean, se, ci_lo, ci_hi, base_mean, and n.
    """
    d = diff_df["diff"].to_numpy()
    n = len(d)
    mean_ = float(d.mean())
    se_ = float(d.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return {
        "mean": mean_,
        "se": se_,
        "ci_lo": mean_ - tcrit * se_,
        "ci_hi": mean_ + tcrit * se_,
        "base_mean": float(diff_df[_BASE].mean()),
        "n": n,
    }


def grand_mean_diff_clustered(diff_df: pd.DataFrame, cluster_col: str) -> dict:
    """Grand mean of d, clustered by an identifier shared across cells, with a 95% t-CI.

    Use this instead of grand_mean_diff whenever the same cluster_col value (for
    example a seed) recurs across multiple cells: pooling every (cell, cluster)
    row as an independent draw then understates the interval, since rows sharing
    a cluster are not independent. This averages diff within each cluster first,
    giving one independent unit per cluster, then takes a paired t-CI over those
    units -- the cluster, not the (cell, cluster) row, is the independent unit.

    Args:
        diff_df: Output of make_diff_frame.
        cluster_col: Column identifying the independent unit, typically "seed".

    Returns:
        Dict with mean, se, ci_lo, ci_hi, base_mean, and n (clusters, not rows).

    Raises:
        ValueError: If clusters do not all cover the same number of rows, since
            an uneven cluster size would silently change what "clustered" means
            rather than signal that the design is unbalanced.
    """
    counts = diff_df.groupby(cluster_col).size()
    if counts.nunique() > 1:
        raise ValueError(
            f"Unequal row count per {cluster_col}; every cluster must cover the "
            f"same cells for a clustered mean to be well defined: {counts.to_dict()}"
        )
    per_cluster = diff_df.groupby(cluster_col)["diff"].mean()
    n = len(per_cluster)
    mean_ = float(per_cluster.mean())
    se_ = float(per_cluster.std(ddof=1) / np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return {
        "mean": mean_,
        "se": se_,
        "ci_lo": mean_ - tcrit * se_,
        "ci_hi": mean_ + tcrit * se_,
        "base_mean": float(diff_df[_BASE].mean()),
        "n": n,
    }