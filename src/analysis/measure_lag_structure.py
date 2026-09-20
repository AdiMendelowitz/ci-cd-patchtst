"""Measure contemporaneous and lagged cross-correlation structure in ETTh1 and ECL.

Computes Pearson cross-correlation at lags 0 through max_lag on the training split
of each dataset, using the same split protocol as the paper's main experiments
(Section 2.2: ETTh1 train_rows=8640; ECL train_rows=15813, the proportional
60/20/20 split of 26,304 timesteps). Produces two output CSVs, read by
analyze_realdata.py to print the "Correlation and lead-lag structure" paragraph
(Section 3.2) of "Equal Accuracy, Unequal Cost: Channel Dependence in PatchTST
under Controlled Coupling":

  realdata_corr_summary.csv   One row per dataset; contemporaneous and lagged
                               summary statistics side by side.
  realdata_lag_summary.csv    One row per (dataset, lag); mean/median/max
                               absolute cross-correlation at each lag.

Cross-correlation at lag k is defined as:
    cross_corr(i, j, k) = corr(x_i[t+k], x_j[t])
which measures how well channel j at time t predicts channel i at time t+k.
This is directional: cross_corr(i, j, k) != cross_corr(j, i, k) in general.

For the symmetric summary statistic "max dependence between pair (i, j) over
lags", we take:
    sym_max_lag(i, j) = max over k in [1, max_lag] of
                         max(|cross_corr(i, j, k)|, |cross_corr(j, i, k)|)

The paper question this script answers is whether the max-over-lags dependence
substantially exceeds the lag-0 contemporaneous correlation for the average
pair, indicating that lead-lag structure carries additional predictive signal
beyond what contemporaneous correlation alone captures. A gain of 5 percentage
points over lag-0 is used as the reporting threshold; this is a descriptive
cutoff, not a statistically motivated one, and is reported as such downstream.

Implementation uses np.corrcoef on stacked lagged matrices to vectorise over
all channel pairs simultaneously. For each lag k, a single (2C x 2C)
correlation matrix is computed; the top-right (C x C) block gives
cross_corr(i, j, k).

Memory note: ECL has C=321, T_train=15813. The stacked matrix is
(2*321, 15813-k) = (642, ~15813) float64. np.corrcoef allocates a (642, 642)
output, on the order of 3 MB. Peak memory is well within typical constraints.

Pipeline position: this script generates the raw correlation statistics; it
does not itself verify them against the paper. analyze_realdata.py is the
downstream consumer responsible for cross-checking its output against the
values quoted in main.tex against a pinned oracle. After the
first run against the current data files, review the printed summary by hand
and pin the resulting values as the oracle in analyze_realdata.py.

Usage:
    uv run src/analysis/measure_lag_structure.py
    Run from the repository root; all paths resolve relative to it.

Output:
    results/realdata_corr_summary.csv
    results/realdata_lag_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"

CONFIG = {
    "datasets": [
        {
            "name": "ETTh1",
            "csv_path": _DATA_DIR / "ETTh1.csv",
            "train_rows": 8640,
            "date_col": "date",
            "freq_label": "hourly",
        },
        {
            "name": "ECL",
            "csv_path": _DATA_DIR / "electricity.csv",
            # Matches the proportional 60/20/20 split of 26,304 timesteps used
            # throughout the paper (Section 2.2): 15,813 train rows. Must be
            # kept in sync with that split; a mismatch here computes the
            # correlation structure on a different window than the one the
            # models were actually trained on.
            "train_rows": 15813,
            "date_col": "date",
            "freq_label": "hourly",
        },
    ],
    "max_lag": 48,  # 48 hourly lags = 2 days; covers daily and sub-daily cycles.
    "top_k": 5,  # Number of most-correlated pairs to report.
    "lag0_threshold": 0.05,  # Min. gain over lag-0 flagged as "lead-lag adds signal".
    "results_dir": _REPO_ROOT / "results",
}


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def load_train_split(csv_path: Path, train_rows: int, date_col: str) -> np.ndarray:
    """Load the training split as a float64 array of shape (train_rows, C).

    Args:
        csv_path: Path to the dataset CSV.
        train_rows: Number of leading rows that constitute the training split,
            matching the protocol used to train the paper's models.
        date_col: Name of the date/timestamp column to exclude from the array.

    Returns:
        Float64 array of shape (train_rows, C).

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If the CSV contains NaN values, or has fewer than
            train_rows rows.
    """
    df = pd.read_csv(csv_path, usecols=lambda c: c != date_col)
    if df.isnull().any().any():
        raise ValueError(f"{csv_path.name}: NaN values found; preprocessing required.")
    if len(df) < train_rows:
        raise ValueError(
            f"{csv_path.name}: expected at least {train_rows} rows, got {len(df)}."
        )
    return df.iloc[:train_rows].values.astype(np.float64)


def cross_corr_matrix(data: np.ndarray, lag: int) -> np.ndarray:
    """Compute the (C x C) cross-correlation matrix at a given lag.

    cross_corr_matrix[i, j] = corr(data[lag:, i], data[:-lag, j])
                             = corr(channel i at t+lag, channel j at t)

    For lag=0, returns the standard contemporaneous Pearson correlation matrix.

    Args:
        data: Float64 array of shape (T, C).
        lag: Non-negative integer lag in timesteps.

    Returns:
        Float64 array of shape (C, C).

    Raises:
        ValueError: If lag is negative.
    """
    if lag < 0:
        raise ValueError(f"lag must be non-negative; got {lag}.")
    C = data.shape[1]
    if lag == 0:
        return np.corrcoef(data.T)
    # Stack future (data[lag:]) on top of past (data[:-lag]). np.corrcoef
    # expects rows as variables and columns as observations.
    stacked = np.vstack([data[lag:].T, data[:-lag].T])  # shape (2C, T-lag)
    full = np.corrcoef(stacked)  # shape (2C, 2C)
    return full[:C, C:]  # top-right block: (C, C)


def off_diagonal_abs(matrix: np.ndarray) -> np.ndarray:
    """Return the absolute values of all off-diagonal entries of a square matrix."""
    C = matrix.shape[0]
    mask = ~np.eye(C, dtype=bool)
    return np.abs(matrix[mask])


def symmetric_pair_max(matrices: list[np.ndarray]) -> np.ndarray:
    """For each unique unordered pair (i, j), take the max over all lags of
    max(|M[i,j]|, |M[j,i]|).

    Args:
        matrices: One (C x C) cross-correlation matrix per lag.

    Returns:
        Float64 array of shape (n_pairs,), where n_pairs = C * (C - 1) / 2.
    """
    C = matrices[0].shape[0]
    row_idx, col_idx = np.triu_indices(C, k=1)
    n_pairs = len(row_idx)
    pair_max = np.zeros(n_pairs)
    for M in matrices:
        for p, (i, j) in enumerate(zip(row_idx, col_idx)):
            val = max(abs(M[i, j]), abs(M[j, i]))
            if val > pair_max[p]:
                pair_max[p] = val
    return pair_max


def top_k_pairs(
    matrix: np.ndarray,
    column_names: list[str],
    k: int,
    symmetric: bool = True,
) -> list[tuple[str, str, float]]:
    """Return the top-k channel pairs by absolute correlation value.

    Args:
        matrix: (C x C) correlation matrix.
        column_names: List of C channel names.
        k: Number of pairs to return.
        symmetric: If True, use only the upper triangle (unordered pairs).

    Returns:
        List of (name_i, name_j, abs_corr) tuples, sorted descending.
    """
    C = matrix.shape[0]
    if symmetric:
        row_idx, col_idx = np.triu_indices(C, k=1)
    else:
        mask = ~np.eye(C, dtype=bool)
        row_idx, col_idx = np.where(mask)
    values = np.abs(matrix[row_idx, col_idx])
    order = np.argsort(-values)[:k]
    return [
        (column_names[row_idx[o]], column_names[col_idx[o]], round(float(values[o]), 4))
        for o in order
    ]


def analyze_dataset(
    name: str,
    data: np.ndarray,
    column_names: list[str],
    max_lag: int,
    top_k: int,
    lag0_threshold: float,
) -> tuple[dict, list[dict]]:
    """Run the full contemporaneous and lagged cross-correlation analysis.

    Args:
        name: Dataset name, used as the identifying key in both output CSVs.
        data: Float64 array of shape (T, C), the training split.
        column_names: Channel names, length C.
        max_lag: Maximum lag in timesteps to compute.
        top_k: Number of most-correlated pairs to report at lag 0.
        lag0_threshold: Minimum gain over lag-0 correlation for a pair to be
            counted in the reported "gained signal from lags" percentage.

    Returns:
        summary_row: One dict, written as one row of realdata_corr_summary.csv.
        lag_rows: One dict per lag, written as rows of realdata_lag_summary.csv.
    """
    T, C = data.shape
    print(f"\n  Computing lag-0 contemporaneous correlation ({C} variates)...")
    lag0_matrix = cross_corr_matrix(data, lag=0)
    lag0_off = off_diagonal_abs(lag0_matrix)

    lag_rows: list[dict] = [
        {
            "dataset": name,
            "lag": 0,
            "mean_abs_r": round(float(lag0_off.mean()), 6),
            "median_abs_r": round(float(np.median(lag0_off)), 6),
            "max_abs_r": round(float(lag0_off.max()), 6),
        }
    ]

    print(f"  Computing lags 1-{max_lag}...")
    lagged_matrices: list[np.ndarray] = []
    for lag in range(1, max_lag + 1):
        M = cross_corr_matrix(data, lag)
        off_abs = off_diagonal_abs(M)
        lag_rows.append(
            {
                "dataset": name,
                "lag": lag,
                "mean_abs_r": round(float(off_abs.mean()), 6),
                "median_abs_r": round(float(np.median(off_abs)), 6),
                "max_abs_r": round(float(off_abs.max()), 6),
            }
        )
        lagged_matrices.append(M)
        if lag % 12 == 0:
            print(f"    lag {lag}/{max_lag} done")

    # Symmetric max-over-lags per pair.
    pair_max_over_lags = symmetric_pair_max(lagged_matrices)
    mean_max_over_lags = float(pair_max_over_lags.mean())

    # Fraction of pairs gaining at least lag0_threshold over their lag-0 value.
    ri, ci_idx = np.triu_indices(C, k=1)
    lag0_per_pair = np.abs(lag0_matrix[ri, ci_idx])
    n_pairs_gain = int(np.sum(pair_max_over_lags - lag0_per_pair >= lag0_threshold))
    pct_gain = 100.0 * n_pairs_gain / len(ri)

    # Top-k pairs at lag 0.
    top_lag0 = top_k_pairs(lag0_matrix, column_names, top_k)

    # Lag with the highest mean absolute cross-correlation.
    mean_abs_by_lag = [row["mean_abs_r"] for row in lag_rows]
    best_lag = int(np.argmax(mean_abs_by_lag))  # 0-indexed matches lag value.

    summary_row = {
        "dataset": name,
        "num_variates": C,
        "num_timesteps_train": T,
        # Contemporaneous.
        "lag0_mean_abs_r": round(float(lag0_off.mean()), 6),
        "lag0_median_abs_r": round(float(np.median(lag0_off)), 6),
        "lag0_min_abs_r": round(float(lag0_off.min()), 6),
        "lag0_max_abs_r": round(float(lag0_off.max()), 6),
        # Lagged.
        "mean_max_over_lags": round(mean_max_over_lags, 6),
        "best_lag": best_lag,
        "mean_abs_r_at_best_lag": round(mean_abs_by_lag[best_lag], 6),
        # Lead-lag signal.
        f"pct_pairs_gain_ge_{int(lag0_threshold * 100)}pct_from_lags": round(pct_gain, 2),
        # Top pairs.
        "top_pairs_lag0": str(top_lag0),
    }

    print(f"\n  {name} summary:")
    print(f"    Lag-0 mean|r|:          {summary_row['lag0_mean_abs_r']:.4f}")
    print(f"    Mean max-over-lags|r|:  {summary_row['mean_max_over_lags']:.4f}")
    print(f"    Best lag (mean|r|):     {best_lag}")
    print(f"    % pairs gaining >={int(lag0_threshold * 100)}pp from lead-lag: {pct_gain:.1f}%")
    print(f"    Top-{top_k} pairs at lag 0: {top_lag0}")

    return summary_row, lag_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_dir: Path = CONFIG["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict] = []
    all_lag_rows: list[dict] = []

    for ds in CONFIG["datasets"]:
        name = ds["name"]
        csv_path = ds["csv_path"]

        if not csv_path.exists():
            print(f"\n[SKIP] {name}: CSV not found at {csv_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Dataset: {name}  ({ds['freq_label']})")
        print(f"{'=' * 60}")

        data = load_train_split(csv_path, ds["train_rows"], ds["date_col"])
        df_full = pd.read_csv(csv_path, usecols=lambda c: c != ds["date_col"])
        column_names = list(df_full.columns)

        print(f"  Train split: {data.shape[0]} timesteps x {data.shape[1]} variates")

        summary_row, lag_rows = analyze_dataset(
            name=name,
            data=data,
            column_names=column_names,
            max_lag=CONFIG["max_lag"],
            top_k=CONFIG["top_k"],
            lag0_threshold=CONFIG["lag0_threshold"],
        )
        all_summary_rows.append(summary_row)
        all_lag_rows.extend(lag_rows)

    if not all_summary_rows:
        print("\nNo datasets processed. Check that CSV paths in CONFIG are correct.")
    else:
        summary_csv = results_dir / "realdata_corr_summary.csv"
        lag_csv = results_dir / "realdata_lag_summary.csv"

        pd.DataFrame(all_summary_rows).to_csv(summary_csv, index=False)
        pd.DataFrame(all_lag_rows).to_csv(lag_csv, index=False)

        print(f"\n{'=' * 60}")
        print(f"Summary written to {summary_csv}")
        print(f"Lag detail written to {lag_csv}")

        print("\n--- Cross-dataset comparison ---")
        pct_col = f"pct_pairs_gain_ge_{int(CONFIG['lag0_threshold'] * 100)}pct_from_lags"
        print(
            pd.DataFrame(all_summary_rows)[
                ["dataset", "lag0_mean_abs_r", "mean_max_over_lags", "best_lag", pct_col]
            ].to_string(index=False)
        )

        print(
            "\nInterpretation: if mean_max_over_lags substantially exceeds lag0_mean_abs_r "
            "for a dataset, lead-lag structure carries additional signal beyond "
            "contemporaneous correlation. A high pct_pairs_gain_ge_5pct_from_lags "
            "indicates that architectures exploiting temporal ordering (e.g. LIFT) "
            "may have a structural advantage over CI on that dataset.\n"
            "This run has not yet been cross-checked against a pinned oracle. Review "
            "the summary above by hand, compare it against the values quoted in "
            "main.tex Section 3.2, and pin the reviewed values as the oracle in "
            "analyze_realdata.py before treating this run as verified."
        )
