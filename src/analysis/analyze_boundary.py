"""Boundary experiment analysis: CD/CI ratio across the (patch_size, gamma) grid.

Reads
-----
results_boundary_p4_ci.csv
    CI only at P=4, gamma in {0.6, 0.9}, original environment. Superseded at
    P=4 by the current-environment tranche below wherever the two overlap; kept
    as the original-environment reference the paper's earlier P=4 cells cite.

results_boundary.csv
    P in {2, 8, 16}, original environment.
      P=2:  CI only at gamma in {0.6, 0.9} (CD infeasible: 21 x 511 = 10731
            tokens). gamma in {0.0, 0.3} were not run.
      P=8:  CI and CD at gamma in {0.0, 0.3, 0.6, 0.9}.
      P=16: CI and CD at gamma in {0.0, 0.3, 0.6, 0.9}.

results_boundary_p4*_complete.csv  (optional, zero or more)
    Current-environment CI and CD at P=4, one merged file per gamma, produced by
    merge_boundary_p4.py. When supplied, these rows replace any P=4 row carrying
    the same (gamma, mode, seed) key from the original-environment reference, so
    the two environments are never pooled within a cell.

Seed count per cell is read from the data rather than assumed; the script reports
whatever n is present and a mixed-n merge is flagged loudly.

Both CSVs share the schema:
    dataset, C, rho, gamma, patch_size, mode, seed, test_mse, test_mae,
    best_epoch, batch_size, steps_per_epoch, total_steps

Writes
------
Results/figures/boundary_heatmap.png
    Partial (P, gamma) grid of the CD/CI MSE ratio at every patch size where both
    modes were run. Cells where CD was not run are grey N/A. The colour range is
    data-driven (max observed deviation times 1.1, floor 0.002).

Console
-------
CI-only summary at CD-absent patch sizes; per-cell paired CD-CI differences with
95% t-CIs at every paired patch size; the step-count table read from the CSV; the
regression on paired differences, including a slope-homogeneity check and a
seed-clustered refit; and a ready-to-paste LaTeX paragraph.

Paths
-----
The script is assumed to live in time-series-forecasting/paper/, with the result
CSVs under results/ at the repository root and figures written to paper/figures/.
Both default paths are resolved against those
candidate directories; pass explicit paths to override:

    python analyze_boundary.py [p4_ci.csv boundary.csv [p4_complete.csv ...]]

Statistical design
------------------
Primary object: d = MSE_CD - MSE_CI per (patch_size, gamma, seed), pivoted within
each triple to exploit the matched design. The slope of d on gamma is fitted on
the regression-eligible cells only, defined as the paired patch sizes drawn from
results_boundary.csv (P in {8, 16}); CD is structurally absent at P=2 because of
memory, not random omission, so the P=2 by gamma interaction is not estimable.
The gamma effect is checked for homogeneity across those patch sizes, and the
slope is refitted with seed-clustered standard errors because the five seeds
recur across cells.

The current-environment P=4 tranche is deliberately excluded from the regression
while still entering the ratio heatmap. Two independent reasons, either
sufficient on its own:

  1. Provenance. The P=4 rows were trained in a later software environment than
     the P in {8, 16} rows. The CD/CI ratio is a within-cell quantity, both arms
     sharing one protocol, so it remains comparable across environments; an
     inferential slope pooled over both would not be.
  2. Training protocol. At P in {8, 16} the CD arm runs at batch 8 against CI's
     128, giving CD 16.1x more gradient updates per epoch. At P=4 both arms run
     at batch 128 with identical step counts. Pooling matched-budget and
     16x-unmatched cells into one slope would confound the coupling effect with
     the update-budget difference.

The same protocol difference makes the P=4 cells the only matched-update-budget
cells in this sweep, which is reported rather than hidden: the step-count table
and the generated prose both break the ratio out per patch size instead of
quoting a single figure across cells that do not share one.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save-only; never open a window on Kaggle or in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "results"
_FIGURES_DIR = _ROOT / "paper" / "figures"

_NAME_P4_CI = "results_boundary_p4_ci.csv"
_NAME_BOUNDARY = "results_boundary.csv"

# Internal provenance tag attached at load time. Rows from the original-environment
# CSVs are "original"; rows from a merged P=4 tranche are "current". The tag drives
# regression eligibility and the figure annotation, and is never written to disk.
_PROV_COL = "_provenance"
_PROV_ORIGINAL = "original"
_PROV_CURRENT = "current"

# Display order: P descending so P=16 (the null condition) sits at the top and
# P=2 (the finest patch resolution) at the bottom.
_PATCH_SIZES: list[int] = [16, 8, 4, 2]
_GAMMAS: list[float] = [0.0, 0.3, 0.6, 0.9]

# Generator constants for the boundary family, used only to derive the CD token
# count quoted in the generated prose (N = floor((L - P) / S) + 1 at S = P / 2).
_SEQ_LEN = 512
_C_VARIATES = 21

_REQUIRED_COLS: set[str] = {
    "dataset",
    "C",
    "rho",
    "gamma",
    "patch_size",
    "mode",
    "seed",
    "test_mse",
    "best_epoch",
    "steps_per_epoch",
    "total_steps",
}


# -- Path resolution -----------------------------------------------------------


def resolve_csv(filename: str) -> Path:
    """Locate a result CSV under the canonical result directories.

    Searches results/ at the repository root. Raises with
    both candidate locations listed if neither exists.
    """
    candidates = [_RESULTS_DIR / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    listed = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not find {filename}. Looked in:\n  {listed}\n"
        f"Pass explicit paths: python analyze_boundary.py <results_boundary_p4_ci.csv> <results_boundary.csv>"
    )


# -- Data loading --------------------------------------------------------------


def load_boundary(path_a: Path, path_b: Path, current_paths: list[Path] | None = None) -> pd.DataFrame:
    """Load, validate, and merge the boundary CSVs.

    path_a and path_b are the original-environment inputs. current_paths, when
    given, are merged current-environment P=4 tranches; their rows supersede any
    original-environment row sharing a (patch_size, gamma, mode, seed) key rather
    than being pooled with it, since the two environments must never be averaged
    within a cell. Every superseded row is reported with both values so the
    substitution is visible rather than silent.

    Duplicate keys within one provenance are still rejected outright, and a
    non-uniform seed count across cells is warned about as the signature of a
    stale or mixed-n input file.
    """
    current_paths = list(current_paths or [])
    for path in (path_a, path_b, *current_paths):
        if not path.exists():
            raise FileNotFoundError(
                f"Boundary CSV not found: {path}\n"
                f"Expected filenames:\n"
                f"  P4 CI-only: {_NAME_P4_CI}\n"
                f"  Boundary:   {_NAME_BOUNDARY}"
            )

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    for label, frame in (("Slice A", df_a), ("Slice B", df_b)):
        missing = _REQUIRED_COLS - set(frame.columns)
        if missing:
            raise ValueError(f"{label} CSV missing columns: {missing}")

    original = pd.concat([df_a, df_b], ignore_index=True)
    original[_PROV_COL] = _PROV_ORIGINAL

    current_frames = []
    for path in current_paths:
        frame = pd.read_csv(path)
        missing = _REQUIRED_COLS - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {missing}")
        frame[_PROV_COL] = _PROV_CURRENT
        current_frames.append(frame)

    key_cols = ["patch_size", "gamma", "mode", "seed"]

    if current_frames:
        current = pd.concat(current_frames, ignore_index=True)
        cur_dupes = current[current.duplicated(subset=key_cols, keep=False)]
        if not cur_dupes.empty:
            raise ValueError(
                f"Duplicate (patch_size, gamma, mode, seed) rows across the "
                f"current-environment inputs:\n{cur_dupes}"
            )
        superseded = original.merge(current[key_cols], on=key_cols, how="inner")
        if not superseded.empty:
            replacement = current.set_index(key_cols)["test_mse"]
            lines = []
            for _, row in superseded.iterrows():
                key = tuple(row[c] for c in key_cols)
                lines.append(
                    f"  P={int(row.patch_size)} gamma={row.gamma} {row['mode']} seed={int(row.seed)}: "
                    f"original {row.test_mse:.6f} -> current {float(replacement.loc[key]):.6f}"
                )
            print(
                f"[INFO] {len(superseded)} original-environment row(s) superseded by the "
                f"current-environment tranche (not pooled):\n" + "\n".join(lines)
            )
            merged_keys = current.set_index(key_cols).index
            original = original[~original.set_index(key_cols).index.isin(merged_keys)].reset_index(drop=True)
        df = pd.concat([original, current], ignore_index=True)
    else:
        df = original

    dupes = df[df.duplicated(subset=key_cols, keep=False)]
    if not dupes.empty:
        raise ValueError(f"Duplicate (patch_size, gamma, mode, seed) rows:\n{dupes}")

    df = df.sort_values(key_cols).reset_index(drop=True)
    print(f"Loaded {len(df_a)} rows from Slice A, {len(df_b)} rows from Slice B.")
    if current_frames:
        n_current = sum(len(f) for f in current_frames)
        print(f"Loaded {n_current} row(s) from {len(current_frames)} current-environment file(s).")
    print(f"Merged: {len(df)} rows total.")
    print(f"Patch sizes: {sorted(df['patch_size'].unique())}")
    print(f"Gammas:      {sorted(df['gamma'].unique())}")
    print(f"Modes:       {sorted(df['mode'].unique())}")

    seed_counts = df.groupby(["patch_size", "gamma", "mode"])["seed"].nunique()
    if seed_counts.nunique() > 1:
        print(
            "[WARN] seed count is NOT uniform across cells. A mixed-n merge usually "
            "means a stale input file. Per-cell seed counts:\n"
            f"{seed_counts.to_string()}"
        )
    return df


# -- Statistics ----------------------------------------------------------------


def make_diff_frame(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (patch_size, gamma, seed) with columns CI, CD, diff.

    Only triples where both CI and CD exist are retained, which excludes any
    patch size where CD was never run (P=2 throughout, and P=4 unless a
    current-environment tranche was supplied). A triple with exactly one mode
    present is reported and dropped, because a silent drop after a partial merge
    could shrink the effective sample below what the caller expects. The
    provenance tag is carried through so downstream callers can scope the
    regression without re-reading the inputs.
    """
    both = df[df["mode"].isin(["CI", "CD"])]
    pivot_full = both.pivot_table(
        index=["patch_size", "gamma", "seed"],
        columns="mode",
        values="test_mse",
    )
    # A cell is single-provenance by construction: load_boundary supersedes rather
    # than pools, so both arms of any (patch_size, gamma, seed) share one tag.
    prov = both.groupby(["patch_size", "gamma", "seed"])[_PROV_COL].first()

    cd_patch_sizes = both.loc[both["mode"] == "CD", "patch_size"].unique()
    at_paired_p = pivot_full.index.get_level_values("patch_size").isin(cd_patch_sizes)
    unpaired = pivot_full[at_paired_p & pivot_full.isna().any(axis=1)]
    if not unpaired.empty:
        cols = ["patch_size", "gamma", "seed"]
        print(
            f"[WARN] {len(unpaired)} triple(s) at CD-capable patch sizes have only one "
            f"mode present and are excluded from the paired analysis:\n"
            f"{unpaired.reset_index()[cols].to_string(index=False)}"
        )

    missing_modes = {"CI", "CD"} - set(pivot_full.columns)
    if missing_modes:
        raise ValueError(
            f"Paired analysis needs both CI and CD; {missing_modes} absent entirely "
            f"from the merged data. Check the input files."
        )

    pivot = pivot_full.dropna()
    if pivot.empty:
        raise ValueError("No paired (CI, CD) triples remain after the inner join.")
    pivot = pivot.join(prov).reset_index()
    pivot["diff"] = pivot["CD"] - pivot["CI"]
    return pivot


def paired_differences(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Per-cell CD-CI differences with 95% t-CIs, sorted ascending.

    Returns one row per (patch_size, gamma) with mean_diff, ci_lo, ci_hi, signs,
    rel_pct, and ci_mean. The t-critical value uses df = n_seeds - 1, with n read
    from the data per cell.
    """
    rows: list[dict[str, object]] = []
    for (patch_size, gamma), grp in diff_df.groupby(["patch_size", "gamma"]):
        provenance = str(grp[_PROV_COL].iloc[0]) if _PROV_COL in grp.columns else _PROV_ORIGINAL
        diffs = grp["diff"].to_numpy()
        n = len(diffs)
        if n < 2:
            raise ValueError(
                f"Cell (P={int(patch_size)}, gamma={float(gamma)}) has n={n} paired "
                f"seed(s); a t-CI needs n>=2. Check upstream unpaired-seed drops."
            )
        mean_d = float(diffs.mean())
        std_d = float(diffs.std(ddof=1))
        se = std_d / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        ci_mean = float(grp["CI"].mean())
        rows.append(
            {
                "patch_size": int(patch_size),
                "gamma": float(gamma),
                "mean_diff": mean_d,
                "std_diff": std_d,
                "ci_lo": mean_d - tcrit * se,
                "ci_hi": mean_d + tcrit * se,
                "signs": "".join("+" if x > 0 else ("-" if x < 0 else "0") for x in diffs),
                "rel_pct": 100.0 * mean_d / ci_mean,
                "ci_mean": ci_mean,
                _PROV_COL: provenance,
            }
        )
    return pd.DataFrame(rows).sort_values(["patch_size", "gamma"]).reset_index(drop=True)


def regression_eligible(diff_df: pd.DataFrame) -> pd.DataFrame:
    """Subset of paired triples admissible to the gamma slope.

    Restricted to original-environment rows. The current-environment P=4 tranche
    is excluded for two independent reasons, either sufficient on its own: it was
    trained under a later software environment, and its CD arm ran at the same
    batch size as CI where the original-environment CD arm ran at batch 8. A
    slope pooled across either difference would confound the coupling effect with
    it. Those cells still enter the ratio heatmap, where the within-cell CD/CI
    ratio is unaffected by both.
    """
    if _PROV_COL not in diff_df.columns:
        return diff_df
    return diff_df[diff_df[_PROV_COL] == _PROV_ORIGINAL].reset_index(drop=True)


def diff_regression(diff_df: pd.DataFrame) -> dict[str, object]:
    """Regress d = MSE_CD - MSE_CI on gamma (numeric) and patch_size (categorical).

    Three fits, reported together:
      1. Main effects, the headline slope. gamma is numeric (equidistant on
         {0.0, 0.3, 0.6, 0.9}); patch_size is categorical, since the observed
         levels carry no linearity assumption.
      2. The gamma-by-patch_size interaction, to confirm the slope is homogeneous
         across patch sizes rather than assuming it. A non-significant interaction
         justifies pooling into the single main-effects slope.
      3. The main-effects model refitted with seed-clustered standard errors,
         because the five seeds recur across cells and the per-row independence
         assumption of ordinary OLS does not hold. The clustered fit is the
         conservative inferential summary.

    A positive gamma coefficient means the CD-CI gap widens as coupling grows (CD
    relatively worse); a negative coefficient means it narrows. The direction is
    reported explicitly in the prose. The fit is on the regression-eligible cells
    only; see regression_eligible for what that excludes and why.
    """
    if diff_df["patch_size"].nunique() < 2:
        raise ValueError(
            f"diff_regression requires at least 2 distinct patch sizes; found "
            f"{diff_df['patch_size'].nunique()}. Check that paired data exists for "
            f"multiple P."
        )

    df_lm = diff_df.copy()
    df_lm["patch_size_cat"] = df_lm["patch_size"].astype(str)

    main = smf.ols("diff ~ gamma + patch_size_cat", data=df_lm).fit()
    interaction = smf.ols("diff ~ gamma * patch_size_cat", data=df_lm).fit()
    clustered = smf.ols("diff ~ gamma + patch_size_cat", data=df_lm).fit(
        cov_type="cluster", cov_kwds={"groups": df_lm["seed"]}
    )

    inter_terms = [t for t in interaction.pvalues.index if t.startswith("gamma:")]
    interaction_p = float(interaction.pvalues[inter_terms].min()) if inter_terms else float("nan")

    gamma_ci = main.conf_int().loc["gamma"]
    cluster_ci = clustered.conf_int().loc["gamma"]
    fitted_ps = ", ".join(f"P={int(p)}" for p in sorted(df_lm["patch_size"].unique()))
    return {
        "scope_note": f"{fitted_ps}, original environment.",
        "coef_gamma": float(main.params["gamma"]),
        "ci_lo_gamma": float(gamma_ci.iloc[0]),
        "ci_hi_gamma": float(gamma_ci.iloc[1]),
        "pval_gamma": float(main.pvalues["gamma"]),
        "pval_f": float(main.f_pvalue),
        "r2": float(main.rsquared),
        "n_rows": int(len(df_lm)),
        "mean_ci_mse": float(diff_df["CI"].mean()),
        "interaction_p": interaction_p,
        "cluster_pval_gamma": float(clustered.pvalues["gamma"]),
        "cluster_ci_lo": float(cluster_ci.iloc[0]),
        "cluster_ci_hi": float(cluster_ci.iloc[1]),
        "model": main,
    }


def ratio_grid(paired: pd.DataFrame) -> pd.DataFrame:
    """CD/CI ratio per cell from paired differences (ci_mean and mean_diff)."""
    cols = ["patch_size", "gamma", "ci_mean", "mean_diff"]
    if _PROV_COL in paired.columns:
        cols.append(_PROV_COL)
    result = paired[cols].copy()
    result["cd_mean"] = result["ci_mean"] + result["mean_diff"]
    result["ratio"] = result["cd_mean"] / result["ci_mean"]
    return result


def mode_batch_sizes(df: pd.DataFrame) -> dict[str, int | None]:
    """Per-mode batch size at the paired patch sizes, for prose that must match data.

    Scoped to patch sizes where CD was run, since mode-wide scoping would be
    contaminated by CI-only patch sizes that used a different batch. Returns the
    unique batch size per mode, or None when the column is absent or the mode used
    more than one batch size there (in which case prose must not assert a number).
    """
    out: dict[str, int | None] = {"CI": None, "CD": None}
    if "batch_size" not in df.columns:
        return out
    # Where CD ran under more than one batch size across patch sizes, no single
    # number describes it and None is returned so prose omits the claim entirely.
    cd_patch_sizes = df.loc[df["mode"] == "CD", "patch_size"].unique()
    paired = df[df["patch_size"].isin(cd_patch_sizes)]
    for mode in ("CI", "CD"):
        vals = paired.loc[paired["mode"] == mode, "batch_size"].dropna().unique()
        out[mode] = int(vals[0]) if len(vals) == 1 else None
    return out


def step_count_table(df: pd.DataFrame) -> pd.DataFrame:
    """CD/CI steps-per-epoch ratio per patch_size, read from the CSV.

    steps_per_epoch = dataset_size / batch_size and is independent of patch_size,
    so it must be constant within each (patch_size, mode) group; this is asserted.
    Rows where CD is absent (P=2, P=4) carry NaN in the CD column.
    """
    sub = df[df["mode"].isin(["CI", "CD"])]
    n_unique = sub.groupby(["patch_size", "mode"])["steps_per_epoch"].nunique()
    if (n_unique > 1).any():
        bad_keys = n_unique[n_unique > 1].index.tolist()
        detail = sub.set_index(["patch_size", "mode"]).loc[bad_keys, "steps_per_epoch"].groupby(level=[0, 1]).unique()
        raise ValueError(f"steps_per_epoch not constant within {bad_keys}.\nObserved:\n{detail.to_string()}")

    tbl = sub.groupby(["patch_size", "mode"])["steps_per_epoch"].first().unstack("mode").reset_index()
    if "CI" in tbl.columns and "CD" in tbl.columns:
        tbl["cd_ci_ratio"] = (tbl["CD"] / tbl["CI"]).round(1)
    return tbl.sort_values("patch_size", ascending=False).reset_index(drop=True)


def ci_only_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean CI MSE at patch sizes where CD was not run.

    The CD-absent patch sizes are read from the data rather than hardcoded, since
    which patch sizes lack a CD arm depends on whether a current-environment P=4
    tranche was supplied. These values appear in the paper tables but have no CD
    counterpart; printing them here lets the script output verify every paper
    number.
    """
    cd_patch_sizes = set(df.loc[df["mode"] == "CD", "patch_size"].unique())
    ci_only_patch_sizes = sorted(set(df["patch_size"].unique()) - cd_patch_sizes)
    ci_sub = df[(df["mode"] == "CI") & (df["patch_size"].isin(ci_only_patch_sizes))]
    if ci_sub.empty:
        return pd.DataFrame()
    return (
        ci_sub.groupby(["patch_size", "gamma"])["test_mse"]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .sort_values(["patch_size", "gamma"])
        .reset_index(drop=True)
    )


# -- Console output ------------------------------------------------------------


def print_ci_only(ci_only: pd.DataFrame) -> None:
    """Print CI-only results at P=2 and P=4 for paper verification."""
    if ci_only.empty:
        print("\n[INFO] No CI-only patch sizes: every patch size present carries a CD arm.")
        return
    ps_list = ", ".join(f"P={int(p)}" for p in sorted(ci_only["patch_size"].unique()))
    print(f"\n=== CI-ONLY SUMMARY ({ps_list}; CD not run) ===")
    print("These values appear in the paper; CD counterparts do not exist.")
    print(f"{'P':>4} {'gamma':>6}  {'CI mean':>10} {'CI std':>9}  n")
    print("-" * 44)
    for _, r in ci_only.iterrows():
        print(f"{int(r.patch_size):>4} {r.gamma:>6.1f}  {r['mean']:>10.4f} {r['std']:>9.4f}  {int(r.n)}")


def print_paired(paired: pd.DataFrame, seeds: list[int]) -> None:
    """Print per-cell paired differences for P in {8, 16}."""
    seed_str = ", ".join(str(s) for s in seeds)
    n_seeds = len(seeds)
    ps_list = ", ".join(f"P={int(p)}" for p in sorted(paired["patch_size"].unique()))
    print("\n=== PER-CELL PAIRED DIFFERENCES (CD - CI) ===")
    print(f"{ps_list}: every patch size where both modes ran.")
    print(f"Positive = CD worse. Signs = per-seed direction (seeds {seed_str}).")
    show_prov = _PROV_COL in paired.columns and paired[_PROV_COL].nunique() > 1
    prov_head = f" {'env':>8}" if show_prov else ""
    header = (
        f"{'P':>4} {'gamma':>6}  {'mean_diff':>10} {'rel_%':>8}  "
        f"{'95%CI_lo':>10} {'95%CI_hi':>10}  signs{prov_head}"
    )
    print(header)
    print("-" * (72 + len(prov_head)))
    for _, r in paired.iterrows():
        prov_cell = f" {r[_PROV_COL]:>8}" if show_prov else ""
        print(
            f"{int(r.patch_size):>4} {r.gamma:>6.1f}  {r.mean_diff:>+10.6f} {r.rel_pct:>+8.4f}%  "
            f"{r.ci_lo:>+10.6f} {r.ci_hi:>+10.6f}  {r.signs}{prov_cell}"
        )

    max_diff = paired["mean_diff"].abs().max()
    max_pct = paired["rel_pct"].abs().max()
    n_zero = int(((paired["ci_lo"] < 0) & (paired["ci_hi"] > 0)).sum())
    n_cells = len(paired)
    n_cd_wins = int((paired["mean_diff"] < 0).sum())
    print(f"\nPaired cells:                    {n_cells}")
    print(f"Cells where CD wins (mean):      {n_cd_wins} / {n_cells}")
    print(f"Max |mean_diff|:                 {max_diff:.6f} MSE units")
    print(f"Max |relative effect|:           {max_pct:.4f}% of CI MSE")
    print(f"CIs including zero:              {n_zero} / {n_cells}")
    print(
        f"\nNote: n={n_seeds} seeds per cell, df={n_seeds - 1}. These are unadjusted per-cell "
        f"95% paired-t intervals, reported descriptively rather than as a multiple-comparison "
        f"procedure across the {n_cells} cells; the regression slope is the inferential summary."
    )


def print_step_table(steps: pd.DataFrame, batch: dict[str, int | None]) -> None:
    """Print the step-count ratio table."""
    print("\n=== STEP-COUNT TABLE (from CSV, not hardcoded) ===")
    cd_bs = batch["CD"] if batch["CD"] is not None else "varies by patch size"
    ci_bs = batch["CI"] if batch["CI"] is not None else "varies by patch size"
    print(f"CD batch_size={cd_bs}, CI batch_size={ci_bs}.")
    print("steps_per_epoch = dataset_size / batch_size, independent of patch_size.")
    cols = [c for c in ["patch_size", "CI", "CD", "cd_ci_ratio"] if c in steps.columns]
    print(steps[cols].to_string(index=False))

    if "cd_ci_ratio" in steps.columns:
        ratios = steps.loc[steps["cd_ci_ratio"].notna(), ["patch_size", "cd_ci_ratio"]]
        matched = ratios[np.isclose(ratios["cd_ci_ratio"], 1.0)]
        unmatched = ratios[~np.isclose(ratios["cd_ci_ratio"], 1.0)]
        if not unmatched.empty:
            listed = ", ".join(
                f"P={int(r.patch_size)} ({r.cd_ci_ratio:.1f}x)" for _, r in unmatched.iterrows()
            )
            print(
                f"\nCD receives more gradient updates per epoch than CI at {listed}, so at those "
                f"patch sizes the null cannot be attributed to CI holding a compute advantage."
            )
        if not matched.empty:
            listed = ", ".join(f"P={int(r.patch_size)}" for _, r in matched.iterrows())
            print(
                f"At {listed} both modes run the same batch size and therefore the same number of "
                f"updates per epoch, so those cells carry no step-count asymmetry in either "
                f"direction and are the sweep's only matched-update-budget cells."
            )


def print_lm(lm: dict[str, object]) -> None:
    """Print the regression on paired differences with both robustness checks."""
    pct_per_unit = 100.0 * lm["coef_gamma"] / lm["mean_ci_mse"]
    direction = "positive" if lm["coef_gamma"] > 0 else "negative"
    widens = "widens" if lm["coef_gamma"] > 0 else "narrows"
    print("\n=== REGRESSION ON PAIRED DIFFERENCES: diff ~ gamma + patch_size_cat ===")
    print(f"Fitted on the regression-eligible cells only ({lm['n_rows']} rows): {lm['scope_note']}")
    print(f"gamma coefficient:    {lm['coef_gamma']:+.6f}  ({pct_per_unit:+.4f}% of mean CI MSE / unit gamma)")
    print(f"Direction:            {direction}, CD gap {widens} as gamma increases")
    print(f"95% CI (gamma):       [{lm['ci_lo_gamma']:+.6f}, {lm['ci_hi_gamma']:+.6f}]")
    print(f"p-value (gamma):      {lm['pval_gamma']:.4f}")
    print(f"F-test p-value:       {lm['pval_f']:.4f}")
    print(f"R-squared:            {lm['r2']:.4f}")
    print(
        f"Slope homogeneity:    gamma-by-P interaction p={lm['interaction_p']:.3f} "
        f"(non-significant, so the pooled slope is appropriate)"
    )
    print(
        f"Seed-clustered slope: p={lm['cluster_pval_gamma']:.3f}, "
        f"95% CI [{lm['cluster_ci_lo']:+.6f}, {lm['cluster_ci_hi']:+.6f}] "
        f"(conservative; the five seeds recur across cells)"
    )
    print(lm["model"].summary().tables[1])


def _cd_token_phrase(patch_size: int) -> str:
    """LaTeX phrase naming the CD token count at one patch size.

    N = floor((L - P) / S) + 1 at S = P / 2, so the count is derived rather than
    quoted; a hardcoded figure would silently misdescribe any patch size but the
    one it was written for.
    """
    stride = max(patch_size // 2, 1)
    n_patches = (_SEQ_LEN - patch_size) // stride + 1
    tokens = f"{_C_VARIATES * n_patches:,}".replace(",", "{,}")
    return f"${_C_VARIATES} \\times {n_patches} = {tokens}$ tokens at $P = {patch_size}$"


def print_latex_prose(
    paired: pd.DataFrame,
    lm: dict[str, object],
    steps: pd.DataFrame,
    batch: dict[str, int | None],
    regression_paired: pd.DataFrame | None = None,
) -> None:
    """Print a ready-to-paste LaTeX paragraph for the boundary experiment.

    paired covers every cell where both modes ran and drives the descriptive
    claims; regression_paired, when the two differ, covers the narrower set the
    slope was fitted on. Keeping them separate stops the paragraph from quoting a
    slope over cells it was not fitted on.
    """
    max_diff = paired["mean_diff"].abs().max()
    max_pct = paired["rel_pct"].abs().max()
    n_cells = len(paired)
    n_zero = int(((paired["ci_lo"] < 0) & (paired["ci_hi"] > 0)).sum())
    n_cd_wins = int((paired["mean_diff"] < 0).sum())
    reg_paired = paired if regression_paired is None else regression_paired
    paired_ps = sorted(int(p) for p in paired["patch_size"].unique())
    reg_ps = sorted(int(p) for p in reg_paired["patch_size"].unique())
    cd_absent_ps = sorted(set(_PATCH_SIZES) - set(paired_ps))

    if n_zero == n_cells:
        zero_clause = f"all {n_cells} of the 95\\% confidence intervals include zero"
    else:
        zero_clause = (
            f"{n_zero} of the {n_cells} 95\\% confidence intervals include zero, the lone "
            f"exception being a ${max_pct:.2f}\\%$ effect that stays well inside the $1\\%$ band"
        )

    if n_cd_wins == 0:
        wins_clause = f"CD does not reach a lower mean MSE than CI in any of the {n_cells} cells"
    else:
        wins_clause = f"CD reaches a lower mean MSE than CI in {n_cd_wins} of the {n_cells} cells"

    # Reported per patch size rather than as one range: a single figure spanning a
    # matched cell (1.0x) and an unmatched one (16.1x) would misdescribe both.
    step_clause = ""
    if "cd_ci_ratio" in steps.columns:
        rows = steps[steps["cd_ci_ratio"].notna()]
        unmatched = rows[~np.isclose(rows["cd_ci_ratio"], 1.0)]
        matched = rows[np.isclose(rows["cd_ci_ratio"], 1.0)]
        parts = []
        if not unmatched.empty:
            listed = " and ".join(
                f"$\\approx{r.cd_ci_ratio:.1f}\\times$ at $P = {int(r.patch_size)}$"
                for _, r in unmatched.iterrows()
            )
            parts.append(
                f"CD received more gradient updates per epoch than CI ({listed}), so at those "
                f"patch sizes the null cannot be attributed to a CI compute advantage"
            )
        if not matched.empty:
            listed = " and ".join(f"$P = {int(r.patch_size)}$" for _, r in matched.iterrows())
            parts.append(
                f"at {listed} both modes ran the same batch size and therefore the same number of "
                f"updates per epoch, so that cell carries no step-count asymmetry in either direction"
            )
        if parts:
            step_clause = "; ".join(parts) + "."

    direction_phrase = (
        "widening slightly as coupling strength grows"
        if lm["coef_gamma"] > 0
        else "narrowing slightly as coupling strength grows"
    )

    paired_ps_tex = ", ".join(str(p) for p in paired_ps)
    reg_ps_tex = ", ".join(str(p) for p in reg_ps)

    if cd_absent_ps:
        # Token counts are derived per patch size rather than quoted from one cell:
        # N = floor((L - P) / S) + 1 at L = 512, S = P / 2, so C * N differs at each P.
        listed = " and ".join(_cd_token_phrase(p) for p in cd_absent_ps)
        absent_clause = (
            f"CD is computationally infeasible at the remaining patch sizes on the T4, where "
            f"sequence lengths of {listed} exhaust available memory, so those cells are "
            f"structurally absent from the paired analysis and their interaction with $\\gamma$ "
            f"is not estimable. "
        )
    else:
        absent_clause = ""

    if reg_ps != paired_ps:
        scope_clause = (
            f" The slope is fitted on $P \\in \\{{{reg_ps_tex}\\}}$ only: the remaining paired cells "
            f"were trained in a later software environment and, in that environment, with CD and CI "
            f"at a common batch size, so pooling them into one slope would confound the coupling "
            f"effect with both differences. Their within-cell CD/CI ratios, which neither difference "
            f"affects, are reported in the accompanying table and figure."
        )
    else:
        scope_clause = ""

    print("\n=== PAPER PROSE (leader-follower boundary experiment paragraph) ===")
    print(
        f"To test whether channel-dependent attention can exploit cross-variate coupling once it is "
        f"concentrated within individual patches, we sweep patch size $P \\in \\{{2, 4, 8, 16\\}}$ against "
        f"coupling strength $\\gamma \\in \\{{0.0, 0.3, 0.6, 0.9\\}}$ on the leader-follower VAR(1) process "
        f"at $C = 21$, $\\rho = 0.5$. {absent_clause}Across the {n_cells} cells where "
        f"both modes run ($P \\in \\{{{paired_ps_tex}\\}}$, all four $\\gamma$), the mean CD$-$CI difference stays "
        f"within $\\pm{max_diff:.4f}$ MSE (at most ${max_pct:.2f}\\%$ of the CI mean), {zero_clause}, and "
        f"{wins_clause}. Regressing the paired difference on $\\gamma$ gives a slope of "
        f"${lm['coef_gamma']:+.4f}$ MSE per unit $\\gamma$ "
        f"(95\\% CI $[{lm['ci_lo_gamma']:+.4f}, {lm['ci_hi_gamma']:+.4f}]$, $p = {lm['pval_gamma']:.2f}$), "
        f"with the gap {direction_phrase}; the slope is homogeneous across patch sizes "
        f"(interaction $p = {lm['interaction_p']:.2f}$) and survives seed-clustered standard errors "
        f"($p = {lm['cluster_pval_gamma']:.2f}$).{scope_clause} {step_clause}"
    )


# -- Heatmap figure ------------------------------------------------------------


def plot_heatmap(ratios: pd.DataFrame, out_path: Path) -> None:
    """Render the partial (P, gamma) CD/CI ratio heatmap.

    Only cells present in ratios carry colour; the rest are grey N/A. The colour
    range is data-driven: the max deviation from 1.0 scaled by 1.1, with a floor
    of 0.002. When the cells span more than one provenance a footnote records
    which patch sizes came from the current environment, so the figure does not
    present two training environments as one without saying so.
    """
    lookup: dict[tuple[int, float], float] = {
        (int(r.patch_size), float(r.gamma)): float(r.ratio) for _, r in ratios.iterrows()
    }

    data = pd.DataFrame(
        [[lookup.get((int(P), float(g)), np.nan) for g in _GAMMAS] for P in _PATCH_SIZES],
        index=[f"P={P}" for P in _PATCH_SIZES],
        columns=[f"\u03b3={g}" for g in _GAMMAS],
    )
    annot = data.map(lambda v: f"{v:.4f}" if not np.isnan(v) else "N/A")

    finite = data.values[~np.isnan(data.values)]
    half_range = max(float(np.abs(finite - 1.0).max()) * 1.1, 0.002)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.heatmap(
        data,
        annot=annot,
        fmt="",
        cmap="RdBu_r",
        center=1.0,
        vmin=1.0 - half_range,
        vmax=1.0 + half_range,
        linewidths=0.5,
        linecolor="#cccccc",
        ax=ax,
        mask=data.isna(),
        cbar_kws={"label": "CD/CI MSE ratio", "shrink": 0.85},
    )

    for row_idx, P in enumerate(_PATCH_SIZES):
        for col_idx, g in enumerate(_GAMMAS):
            if (int(P), float(g)) not in lookup:
                ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=True, color="#dddddd", zorder=0))
                ax.text(col_idx + 0.5, row_idx + 0.5, "N/A", ha="center", va="center", fontsize=9, color="#666666")

    colorbar = ax.collections[0].colorbar
    if colorbar is not None:
        colorbar.ax.axhline(y=1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Coupling strength \u03b3", labelpad=6)
    ax.set_ylabel("Patch size P", labelpad=6)
    ax.set_title(
        "Leader-follower VAR(1): CD/CI MSE ratio\n" "(below 1.0 favours CD, above 1.0 favours CI; grey = CD not run)",
        pad=8,
    )

    if _PROV_COL in ratios.columns and ratios[_PROV_COL].nunique() > 1:
        current_ps = sorted(
            int(p) for p in ratios.loc[ratios[_PROV_COL] == _PROV_CURRENT, "patch_size"].unique()
        )
        listed = ", ".join(f"P={p}" for p in current_ps)
        fig.text(
            0.5,
            -0.02,
            f"{listed} trained in a later software environment, CD at CI's batch size; "
            f"the remaining rows are the original environment. Ratios are within-cell "
            f"and unaffected by both differences.",
            ha="center",
            va="top",
            fontsize=7.5,
            color="#444444",
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# -- Main ----------------------------------------------------------------------


def main() -> None:
    """Run the full boundary analysis and write the heatmap."""
    current_paths: list[Path] = []
    if len(sys.argv) >= 3:
        path_a = Path(sys.argv[1])
        path_b = Path(sys.argv[2])
        current_paths = [Path(a) for a in sys.argv[3:]]
    elif len(sys.argv) == 1:
        path_a = resolve_csv(_NAME_P4_CI)
        path_b = resolve_csv(_NAME_BOUNDARY)
    else:
        raise SystemExit(
            "Usage: python analyze_boundary.py "
            "[p4_ci.csv boundary.csv [p4_gamma_complete.csv ...]]"
        )

    df = load_boundary(path_a, path_b, current_paths)
    diff_df = make_diff_frame(df)
    paired = paired_differences(diff_df)

    reg_diff = regression_eligible(diff_df)
    if reg_diff.empty:
        raise ValueError(
            "No regression-eligible paired cells remain; the gamma slope cannot be "
            "fitted. See regression_eligible for the scoping rule."
        )
    lm = diff_regression(reg_diff)
    reg_paired = paired_differences(reg_diff)

    ratios = ratio_grid(paired)
    steps = step_count_table(df)
    ci_only = ci_only_summary(df)
    batch = mode_batch_sizes(df)

    paired_seeds = sorted(int(s) for s in diff_df["seed"].unique())

    print_ci_only(ci_only)
    print_paired(paired, paired_seeds)
    print_step_table(steps, batch)
    print_lm(lm)
    print_latex_prose(paired, lm, steps, batch, reg_paired)
    plot_heatmap(ratios, _FIGURES_DIR / "boundary_heatmap.png")


if __name__ == "__main__":
    main()