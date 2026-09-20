"""Partition ETTh1 test windows by lagged cross-channel coupling strength.

Purpose
-------
Supports a per-subset accuracy comparison: CI and CD MSE on high-coupling
versus low-coupling subsets of the ETTh1 test set. This script computes the
partition: one row per test input window, carrying two window-level coupling
metrics and a per-horizon high/low flag at that horizon's median. The
per-window forecast errors come separately from an instrumented rerun of the
ETTh1 notebook; the subgroup analysis script joins the two on window_start.

Metrics
-------
Both metrics are the mean absolute lag-1 cross-correlation over ordered
off-diagonal channel pairs within the 512-step input window, computed on the
train-normalised series exactly as the model sees it.

  raw     lag-1 cross-correlation of the levels. On persistent series this is
          dominated by within-channel autocorrelation times contemporaneous
          correlation, the same spurious lagged signal the paper's Granger
          validation documents in its omitted-variable note, so it measures
          mostly instantaneous structure.
  diffed  lag-1 cross-correlation of first differences, which suppresses the
          autocorrelation channel and is closer to genuine lag-1 dynamics.

On the ETTh1 test split the two rank windows in near-opposite order (Spearman
about -0.28), so the choice is substantive, not cosmetic. diffed is the
primary metric, chosen before the error data existed, because the paper's question is about dynamical
(lagged) coupling, not instantaneous correlation; raw is reported as a
robustness column.

Known limitation, stated up front: both metrics are highly autocorrelated
across window starts (about 0.97 at a 24-step offset), so a median split yields
contiguous time blocks and any subgroup gap is partially confounded with
regime. Downstream inference aggregates errors to run level per subset before
the usual per-seed pairing, which handles window dependence; the regime
confound is reported as a limitation.

Splits
------
The high flag at each horizon is metric >= the median over the windows that
exist at that horizon (window_start <= T_test - 512 - H), so subsets are
balanced per horizon rather than at a global median that H=720 truncation
would skew.

Output
------
results/etth1_coupling_partition.csv with columns: window_start, raw, diffed,
and high_{metric}_H{horizon} flags for both metrics at horizons 96, 192, 336,
720. Also prints distribution summaries, the metric-disagreement check, and
an oracle self-check against the canonical-ETTh1 values (exit code 1 on
mismatch, matching the repo's analysis-script idiom).

Flag round-trip contract for consumers: the high_* columns are pandas
nullable booleans and degrade to True/False/empty through CSV. Read them back
with  pd.read_csv(path).assign(**{c: lambda d, c=c: d[c].astype("boolean")
for c in flag_cols})  or equivalently astype("boolean") per flag column;
never compare the raw parsed column against Python booleans directly.

Usage
-----
    python src/analysis/partition_etth1_coupling.py path/to/ETTh1.csv
    python src/analysis/partition_etth1_coupling.py path/to/ETTh1.csv --out results/etth1_coupling_partition.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_LEN: int = 512
TRAIN_END: int = 8640
VAL_END: int = 11520
PRED_LENS: tuple[int, ...] = (96, 192, 336, 720)
_EPS: float = 1e-12
_CANONICAL_ROWS: int = 17_420  # ETTh1 data rows; guards against sibling ETT files.
_CANONICAL_C: int = 7


def load_test_split(csv_path: Path) -> np.ndarray:
    """Load ETTh1 and return the train-normalised test split.

    Mirrors the training notebook exactly: numeric columns only, z-score fitted
    on the first TRAIN_END rows, test split from VAL_END onward.

    Args:
        csv_path: Path to ETTh1.csv.

    Returns:
        Float64 array of shape (T_test, 7).
    """
    df = pd.read_csv(csv_path)
    data = df.select_dtypes(include=np.number).values.astype(np.float64)
    if data.shape != (_CANONICAL_ROWS, _CANONICAL_C):
        raise ValueError(
            f"Input has shape {data.shape}, expected {(_CANONICAL_ROWS, _CANONICAL_C)} for canonical ETTh1. "
            f"The ETT-small files share one schema; this guard exists so ETTh2/ETTm1 cannot be partitioned "
            f"silently under ETTh1's split constants."
        )
    mean = data[:TRAIN_END].mean(axis=0, keepdims=True)
    std = data[:TRAIN_END].std(axis=0, keepdims=True)
    z = (data - mean) / np.where(std == 0, 1.0, std)
    return z[VAL_END:]


def _mean_abs_lag1_xcorr(past: np.ndarray, present: np.ndarray) -> float:
    """Mean |corr(x_i[t-1], x_j[t])| over ordered off-diagonal pairs."""
    c = past.shape[1]
    a = (past - past.mean(axis=0)) / (past.std(axis=0) + _EPS)
    b = (present - present.mean(axis=0)) / (present.std(axis=0) + _EPS)
    m = a.T @ b / len(a)
    off = ~np.eye(c, dtype=bool)
    return float(np.abs(m[off]).mean())


def window_metrics(window: np.ndarray) -> tuple[float, float]:
    """Raw and first-differenced lagged-coupling metrics for one input window.

    Args:
        window: Array of shape (SEQ_LEN, C), train-normalised.

    Returns:
        (raw, diffed) as defined in the module docstring.
    """
    raw = _mean_abs_lag1_xcorr(window[:-1], window[1:])
    d = np.diff(window, axis=0)
    diffed = _mean_abs_lag1_xcorr(d[:-1], d[1:])
    return raw, diffed


def build_partition(test: np.ndarray) -> pd.DataFrame:
    """Compute both metrics per test window and per-horizon high/low flags.

    Args:
        test: Train-normalised test split, shape (T_test, C).

    Returns:
        One row per window start at the shortest horizon, with per-horizon
        balanced median-split flags for both metrics.
    """
    n_max = len(test) - SEQ_LEN - min(PRED_LENS) + 1
    if n_max <= 0:
        raise ValueError(f"Test split too short: {len(test)} rows for seq_len {SEQ_LEN}.")
    # Double slice keeps black and flake8 (E203) in agreement on slice spacing.
    vals = np.array([window_metrics(test[start:][:SEQ_LEN]) for start in range(n_max)])
    out = pd.DataFrame({"window_start": np.arange(n_max), "raw": vals[:, 0], "diffed": vals[:, 1]})
    for horizon in PRED_LENS:
        n_h = len(test) - SEQ_LEN - horizon + 1
        for metric in ("raw", "diffed"):
            median = float(out.loc[: n_h - 1, metric].median())
            flag = (out[metric] >= median).astype("boolean")
            flag[n_h:] = pd.NA
            out[f"high_{metric}_H{horizon}"] = flag
    return out


def print_summary(part: pd.DataFrame) -> None:
    """Print distribution, split balance, and the metric-disagreement checks."""
    for metric in ("diffed", "raw"):
        v = part[metric].to_numpy()
        q = np.percentile(v, [25, 50, 75])
        hi = v >= q[1]
        label = "primary" if metric == "diffed" else "robustness"
        print(
            f"{metric:>6} ({label}): n={len(v)}  mean={v.mean():.4f}  sd={v.std():.4f}  "
            f"q25/50/75 = {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}  "
            f"high/low means {v[hi].mean():.4f}/{v[~hi].mean():.4f} (ratio {v[hi].mean() / v[~hi].mean():.2f})"
        )
    spear = part["raw"].corr(part["diffed"], method="spearman")
    step = 24
    ac = float(np.corrcoef(part["diffed"][:-step], part["diffed"][step:])[0, 1])
    print(f"spearman(raw, diffed) = {spear:+.3f}  (metric choice is substantive)")
    print(f"diffed autocorr at {step}-step offset = {ac:.3f}  (splits form contiguous regimes; see docstring)")
    for horizon in PRED_LENS:
        flags = part[f"high_diffed_H{horizon}"].dropna()
        print(f"H={horizon}: {len(flags)} windows, high/low = {int(flags.sum())}/{int((~flags).sum())}")


_ORACLE: dict[str, float | int] = {
    # Canonical-ETTh1 targets (4 dp where float).
    "diffed_mean": 0.0828,
    "diffed_sd": 0.0177,
    "spearman": -0.2791,
    "n_windows": 5293,
    "h96_high": 2647,
    "h720_high": 2335,
}


def oracle_check(part: pd.DataFrame) -> bool:
    """Self-check against canonical-ETTh1 targets; prints PASS/FAIL lines.

    Args:
        part: Output of build_partition on canonical ETTh1.

    Returns:
        True when every target reproduces at its stated precision.
    """
    got: dict[str, float | int] = {
        "diffed_mean": round(float(part["diffed"].mean()), 4),
        "diffed_sd": round(float(part["diffed"].std(ddof=0)), 4),
        "spearman": round(float(part["raw"].corr(part["diffed"], method="spearman")), 4),
        "n_windows": int(len(part)),
        "h96_high": int(part["high_diffed_H96"].dropna().sum()),
        "h720_high": int(part["high_diffed_H720"].dropna().sum()),
    }
    ok = True
    print("\n=== ORACLE CHECK (canonical ETTh1) ===")
    for key, target in _ORACLE.items():
        match = got[key] == target
        ok = ok and match
        print(f"  [{'PASS' if match else 'FAIL'}] {key}: got {got[key]}  target {target}")
    return ok


def main() -> int:
    """Compute the partition, print the summary and oracle check, write the CSV."""
    parser = argparse.ArgumentParser(description="Partition ETTh1 test windows by lagged coupling strength.")
    parser.add_argument("csv", type=Path, help="Path to ETTh1.csv.")
    parser.add_argument("--out", type=Path, default=Path("results/etth1_coupling_partition.csv"))
    args = parser.parse_args()

    test = load_test_split(args.csv)
    part = build_partition(test)
    print_summary(part)
    ok = oracle_check(part)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    part.to_csv(args.out, index=False)
    print(f"\nPartition written: {args.out}  ({len(part)} rows)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
