"""ETTh1 coupling-subgroup analysis.

Reads the rerun of the committed ETTh1 protocol (CI and CD at horizons
{96, 192, 336, 720}, seeds {42, 123, 456, 789, 1011}; 40 aggregate rows and
202,120 per-test-window error rows) and joins the per-window errors to the
committed coupling partition, so the CD-CI contrast can be split into
high-coupling and low-coupling test windows.

The question this answers: the aggregate ETTh1 contrast cannot distinguish
"CD is worse everywhere" from "CD is worse only where cross-variate coupling
is absent". Partitioning the same test windows by a coupling statistic
computed from the data alone (never from model errors) separates those.

WINDOW-KEY CONVENTION: the load-bearing detail of this script.
The two files use different, both-internally-consistent conventions:

    results_etth1_b4_windows.csv  window_start = ABSOLUTE ETTh1 row index,
                                  i.e. VAL_END + i for the i-th test window,
                                  so it runs 11520 .. 11520+n_windows-1.
    etth1_coupling_partition.csv  window_start = 0-INDEXED within the test
                                  region, so it runs 0 .. n_windows-1.

Joining these without the VAL_END shift silently produces an EMPTY overlap at
every horizon (the ranges are disjoint), and joining with a wrong shift would
misalign every row while still "succeeding". Rather than trusting the constant,
``resolve_offset`` derives it from the data and verifies it structurally: both
frames must have contiguous per-horizon window_start ranges of identical
length, and the implied offset must be identical at all four horizons and equal
VAL_END. Any deviation raises before a single statistic is computed.

Coupling metric: the partition supplies both a raw and a first-differenced
lag-1 cross-correlation flag per horizon (high_raw_H*, high_diffed_H*). The
DIFFED flag is the primary (differencing removes the shared trend that
otherwise dominates the raw statistic and makes nearly every window look
coupled); the raw flag is reported alongside as a robustness read, never as the
headline. Both are median splits computed from the input series alone.

Sign convention follows paired_stats throughout: diff = CD - CI, so POSITIVE
means CD is worse.

The wording ceiling applies: any subgroup asymmetry is reported as consistent
with the stated hypothesis, never as a mechanism claim.

Run from a clean checkout:

    python src/analysis/analyze_etth1_subgroup.py
    python src/analysis/analyze_etth1_subgroup.py --windows path/to/results_etth1_b4_windows.csv \
        --partition path/to/etth1_coupling_partition.csv --main path/to/results_etth1_b4.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import paired_stats as ps  # sibling module; on sys.path when run as a script

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "results"
_WINDOWS = _RESULTS_DIR / "results_etth1_b4_windows.csv"
_MAIN = _RESULTS_DIR / "results_etth1_b4.csv"
_PARTITION = _ROOT / "results" / "etth1_coupling_partition.csv"

_MODES: tuple[str, ...] = ("CI", "CD")
_HORIZONS: tuple[int, ...] = (96, 192, 336, 720)
_SEED_ORDER: tuple[int, ...] = (42, 123, 456, 789, 1011)

# Committed ETTh1 split (train [0,8640), val [8640,11520), test [11520,17420)).
# VAL_END is the absolute row at which the test region begins, and therefore the
# expected offset between the two window_start conventions.
_VAL_END = 11520
_T_TOTAL = 17420
_SEQ_LEN = 512

# Test-window count per horizon. Computed from the committed split AND pinned
# to literals, with the two cross-checked at import. Computing alone would
# silently absorb a change to the split constants; pinning alone leaves the
# split constants dead and unverifiable. Holding both makes a change to either
# fail loudly at import rather than at a statistic.
_PINNED_WINDOWS: dict[int, int] = {96: 5293, 192: 5197, 336: 5053, 720: 4669}
_EXPECTED_WINDOWS: dict[int, int] = {
    h: (_T_TOTAL - _VAL_END) - _SEQ_LEN - h + 1 for h in _HORIZONS
}
if _EXPECTED_WINDOWS != _PINNED_WINDOWS:  # pragma: no cover - import-time guard
    raise AssertionError(
        f"Window counts derived from the committed split {_EXPECTED_WINDOWS} disagree with "
        f"the pinned literals {_PINNED_WINDOWS}; one of the split constants has changed."
    )

_WINDOW_REQUIRED_COLS: set[str] = {"mode", "pred_len", "seed", "window_start", "sq_err", "abs_err"}
_MAIN_REQUIRED_COLS: set[str] = {"mode", "pred_len", "seed", "test_mse", "best_epoch"}

# Oracle. Every value below is recomputed by this script from the committed
# CSVs; the pins exist so a silent change in an input file fails loudly.
#
# Aggregate CD-CI per horizon (mean over 5 seeds, 4 dp): (mean_diff, ci_lo, ci_hi).
_ORACLE_AGG: dict[int, tuple[float, float, float]] = {
    96: (0.0254, 0.0026, 0.0482),
    192: (0.0325, -0.0204, 0.0853),
    336: (0.0068, -0.0171, 0.0307),
    720: (0.0283, 0.0245, 0.0320),
}
# Primary (diffed) subgroup CD-CI, (horizon, group) -> (mean_diff, ci_lo, ci_hi), 4 dp.
_ORACLE_DIFFED: dict[tuple[int, str], tuple[float, float, float]] = {
    (96, "high"): (0.0058, -0.0159, 0.0276),
    (96, "low"): (0.0450, 0.0208, 0.0692),
    (192, "high"): (0.0183, -0.0151, 0.0517),
    (192, "low"): (0.0466, -0.0258, 0.1190),
    (336, "high"): (-0.0030, -0.0259, 0.0199),
    (336, "low"): (0.0167, -0.0089, 0.0423),
    (720, "high"): (0.0263, 0.0196, 0.0329),
    (720, "low"): (0.0302, 0.0275, 0.0330),
}
# Robustness (raw) subgroup CD-CI, same (mean, ci_lo, ci_hi) coverage as the
# primary split. Pinning the mean alone would let an interval change pass
# unnoticed, and the raw split is the check that decides whether the primary
# split's ordering is metric-invariant, so it needs equal, not lesser, cover.
_ORACLE_RAW: dict[tuple[int, str], tuple[float, float, float]] = {
    (96, "high"): (0.0186, -0.0079, 0.0450),
    (96, "low"): (0.0322, 0.0119, 0.0525),
    (192, "high"): (0.0347, -0.0281, 0.0974),
    (192, "low"): (0.0302, -0.0129, 0.0734),
    (336, "high"): (0.0107, -0.0203, 0.0417),
    (336, "low"): (0.0029, -0.0139, 0.0198),
    (720, "high"): (0.0431, 0.0355, 0.0508),
    (720, "low"): (0.0134, 0.0113, 0.0155),
}
# Every contrast cell must rest on the full 5-seed pairing. paired_stats drops a
# (cell, seed) row when either mode is missing, which would shrink n silently;
# this is pinned so that shrinkage prints as FAIL instead of as a plausible number.
_ORACLE_N_PER_CELL = len(_SEED_ORDER)
# Structural integrity, deterministic from the committed CSVs.
_ORACLE_MAIN_ROWS = 40
_ORACLE_WINDOW_ROWS = 202_120
_ORACLE_OFFSET = _VAL_END
# Reconciliation tolerance between the per-window mean of sq_err and the
# aggregate test_mse. Two float32 summation orders over thousands of windows
# disagree at ~1e-8 relative even when both are correct; a real aggregation
# bug is off by a large factor, not a rounding-sized amount.
_RECON_TOL = 1e-3


def load_windows(path: Path) -> pd.DataFrame:
    """Load and validate the per-test-window error log.

    Args:
        path: Path to results_etth1_b4_windows.csv.

    Returns:
        The window frame, unmodified.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If required columns, modes, horizons, the balanced 5-seed
            design, per-run window counts, or finiteness are violated.
    """
    if not path.exists():
        raise FileNotFoundError(f"window CSV not found: {path}")
    df = pd.read_csv(path)
    missing = _WINDOW_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Window CSV missing columns: {sorted(missing)}")
    modes = set(df["mode"].unique())
    if modes != set(_MODES):
        raise ValueError(f"Window CSV modes {sorted(modes)} do not match {sorted(_MODES)}")
    horizons = sorted(df["pred_len"].unique())
    if horizons != sorted(_HORIZONS):
        raise ValueError(f"Window CSV horizons {horizons} do not match {sorted(_HORIZONS)}")
    if not np.isfinite(df[["sq_err", "abs_err"]].to_numpy()).all():
        raise ValueError("Window CSV contains non-finite sq_err or abs_err.")
    for horizon in _HORIZONS:
        for mode in _MODES:
            for seed in _SEED_ORDER:
                run = df[
                    (df["pred_len"] == horizon) & (df["mode"] == mode) & (df["seed"] == seed)
                ]
                if len(run) != _EXPECTED_WINDOWS[horizon]:
                    raise ValueError(
                        f"H={horizon} {mode} seed={seed}: {len(run)} window rows, "
                        f"expected {_EXPECTED_WINDOWS[horizon]}"
                    )
                if run["window_start"].duplicated().any():
                    raise ValueError(f"H={horizon} {mode} seed={seed}: duplicate window_start values")
    return df


def load_main(path: Path) -> pd.DataFrame:
    """Load and validate the aggregate per-run results.

    Args:
        path: Path to results_etth1_b4.csv.

    Returns:
        The aggregate frame, unmodified.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If required columns or the balanced 40-run design are violated.
    """
    if not path.exists():
        raise FileNotFoundError(f"aggregate CSV not found: {path}")
    df = pd.read_csv(path)
    missing = _MAIN_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Aggregate CSV missing columns: {sorted(missing)}")
    for horizon in _HORIZONS:
        for mode in _MODES:
            cell = df[(df["pred_len"] == horizon) & (df["mode"] == mode)]
            seeds = sorted(cell["seed"].unique())
            if seeds != sorted(_SEED_ORDER) or len(cell) != len(_SEED_ORDER):
                raise ValueError(
                    f"H={horizon} {mode}: seeds {seeds} over {len(cell)} rows do not match "
                    f"one row per seed in {sorted(_SEED_ORDER)}"
                )
    return df


def load_partition(path: Path) -> pd.DataFrame:
    """Load and validate the committed coupling partition.

    Args:
        path: Path to etth1_coupling_partition.csv.

    Returns:
        The partition frame, unmodified.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If required columns are missing, the per-horizon valid
            extent does not match the committed window counts, or a flag column
            is not a clean two-group split.
    """
    if not path.exists():
        raise FileNotFoundError(f"Coupling partition CSV not found: {path}")
    part = pd.read_csv(path)
    required = {"window_start"} | {
        f"high_{kind}_H{h}" for kind in ("raw", "diffed") for h in _HORIZONS
    }
    missing = required - set(part.columns)
    if missing:
        raise ValueError(f"Partition CSV missing columns: {sorted(missing)}")
    for horizon in _HORIZONS:
        for kind in ("raw", "diffed"):
            col = f"high_{kind}_H{horizon}"
            valid = part[part[col].notna()]
            if len(valid) != _EXPECTED_WINDOWS[horizon]:
                raise ValueError(
                    f"Partition {col}: {len(valid)} non-null rows, "
                    f"expected {_EXPECTED_WINDOWS[horizon]}"
                )
            groups = set(valid[col].astype(bool).unique())
            if groups != {True, False}:
                raise ValueError(f"Partition {col} is not a two-group split; found {groups}")
    return part


def resolve_offset(windows: pd.DataFrame, part: pd.DataFrame) -> int:
    """Derive and structurally verify the window_start offset between the files.

    The two files index test windows differently (see the module docstring).
    This derives the offset from the data instead of trusting a constant, then
    demands it be consistent at every horizon and equal to the committed
    VAL_END. Both conventions are internally contiguous, so a mismatch here
    means one of the inputs is not the file this script was written against.

    Args:
        windows: Output of load_windows.
        part: Output of load_partition.

    Returns:
        The verified offset (partition key + offset == window key).

    Raises:
        ValueError: If either range is non-contiguous, the implied offsets
            disagree across horizons, or the offset is not VAL_END.
    """
    offsets: dict[int, int] = {}
    for horizon in _HORIZONS:
        w = windows.loc[windows["pred_len"] == horizon, "window_start"]
        w_lo, w_hi, w_n = int(w.min()), int(w.max()), int(w.nunique())
        if w_hi - w_lo + 1 != w_n or w_n != _EXPECTED_WINDOWS[horizon]:
            raise ValueError(
                f"H={horizon}: window_start in the log is not a contiguous run of "
                f"{_EXPECTED_WINDOWS[horizon]} values (got {w_n} distinct spanning {w_lo}..{w_hi})"
            )
        p = part.loc[part[f"high_diffed_H{horizon}"].notna(), "window_start"]
        p_lo, p_hi, p_n = int(p.min()), int(p.max()), int(p.nunique())
        if p_hi - p_lo + 1 != p_n or p_n != _EXPECTED_WINDOWS[horizon]:
            raise ValueError(
                f"H={horizon}: window_start in the partition is not a contiguous run of "
                f"{_EXPECTED_WINDOWS[horizon]} values (got {p_n} distinct spanning {p_lo}..{p_hi})"
            )
        offsets[horizon] = w_lo - p_lo
    distinct = set(offsets.values())
    if len(distinct) != 1:
        raise ValueError(f"Implied window_start offset differs across horizons: {offsets}")
    offset = distinct.pop()
    if offset != _VAL_END:
        raise ValueError(
            f"Implied window_start offset {offset} != committed VAL_END {_VAL_END}. "
            "One of the inputs does not use the convention this script was written against; "
            "refusing to join on an unverified key."
        )
    return offset


def join_partition(windows: pd.DataFrame, part: pd.DataFrame, offset: int, kind: str) -> pd.DataFrame:
    """Attach the coupling flag to every window row, for one metric family.

    Args:
        windows: Output of load_windows.
        part: Output of load_partition.
        offset: Verified offset from resolve_offset.
        kind: "diffed" (primary) or "raw" (robustness).

    Returns:
        The window frame with a boolean "high_coupling" column added.

    Raises:
        ValueError: If any window row fails to match a partition row.
    """
    if kind not in ("diffed", "raw"):
        raise ValueError(f"kind must be 'diffed' or 'raw'; got {kind!r}")
    pieces: list[pd.DataFrame] = []
    for horizon in _HORIZONS:
        col = f"high_{kind}_H{horizon}"
        keys = part.loc[part[col].notna(), ["window_start", col]].copy()
        keys["window_start"] = keys["window_start"] + offset
        keys = keys.rename(columns={col: "high_coupling"})
        sub = windows[windows["pred_len"] == horizon].merge(keys, on="window_start", how="left")
        if sub["high_coupling"].isna().any():
            n_bad = int(sub["high_coupling"].isna().sum())
            raise ValueError(f"H={horizon} {kind}: {n_bad} window rows matched no partition row")
        sub["high_coupling"] = sub["high_coupling"].astype(bool)
        pieces.append(sub)
    return pd.concat(pieces, ignore_index=True)


def reconcile(windows: pd.DataFrame, main: pd.DataFrame) -> pd.DataFrame:
    """Cross-check the per-window log against the aggregate test_mse.

    The two were produced by separate code paths over the same test pass, so
    agreement is evidence that neither aggregation is mis-axed. Independent of
    the partition entirely.

    Args:
        windows: Output of load_windows.
        main: Output of load_main.

    Structural checks only. The joined row count is asserted in the oracle
    section rather than here, so a mismatch prints as FAIL alongside every
    other check instead of aborting the report: the convention established by
    analyze_block_attention.load_diag.

    Returns:
        One row per run with test_mse, window_mean_mse, and rel_gap.
    """
    win_mean = (
        windows.groupby(["mode", "pred_len", "seed"])["sq_err"].mean().reset_index(name="window_mean_mse")
    )
    merged = main.merge(win_mean, on=["mode", "pred_len", "seed"], how="inner")
    merged["rel_gap"] = (merged["window_mean_mse"] - merged["test_mse"]).abs() / merged["test_mse"].abs()
    return merged


def aggregate_contrast(main: pd.DataFrame) -> pd.DataFrame:
    """Paired CD-CI per horizon on the aggregate test_mse, via paired_stats.

    Args:
        main: Output of load_main.

    Returns:
        paired_differences frame keyed by pred_len.
    """
    diff = ps.make_diff_frame(main, cell_cols=["pred_len"], mode_base="CI", mode_alt="CD")
    return ps.paired_differences(diff, cell_cols=["pred_len"])


def group_sizes(joined: pd.DataFrame) -> pd.DataFrame:
    """Windows per coupling group per horizon, per run.

    A median split should halve each horizon's windows. Reporting the counts
    makes a lopsided or degenerate split visible instead of leaving it to be
    inferred from an unexpectedly wide interval.

    Args:
        joined: Output of join_partition.

    Returns:
        One row per (pred_len, high_coupling) with the per-run window count.
    """
    n_runs = len(_MODES) * len(_SEED_ORDER)
    sizes = joined.groupby(["pred_len", "high_coupling"]).size().reset_index(name="rows")
    sizes["windows_per_run"] = sizes["rows"] // n_runs
    return sizes


def subgroup_contrast(joined: pd.DataFrame) -> pd.DataFrame:
    """Paired CD-CI within each (horizon, coupling group), via paired_stats.

    Each run contributes one number per group: the mean sq_err over that
    group's windows. Those per-seed means are then differenced CD-CI exactly as
    the aggregate contrast is, so the seed remains the unit of analysis and the
    5-seed pairing is preserved. Windows are NOT treated as independent units;
    they are correlated within a run and would badly understate the interval.

    Args:
        joined: Output of join_partition.

    Returns:
        paired_differences frame keyed by pred_len and high_coupling.
    """
    per_seed = (
        joined.groupby(["pred_len", "high_coupling", "mode", "seed"])["sq_err"]
        .mean()
        .reset_index(name="test_mse")
    )
    diff = ps.make_diff_frame(
        per_seed, cell_cols=["pred_len", "high_coupling"], mode_base="CI", mode_alt="CD"
    )
    return ps.paired_differences(diff, cell_cols=["pred_len", "high_coupling"])


def _check_line(ok: bool, text: str, lines: list[str]) -> bool:
    """Append one PASS/FAIL line and return the flag for accumulation."""
    lines.append(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    return ok


def _group_label(flag: bool) -> str:
    return "high" if flag else "low"


def oracle_check(
    windows: pd.DataFrame,
    main: pd.DataFrame,
    offset: int,
    recon: pd.DataFrame,
    agg: pd.DataFrame,
    diffed: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[list[str], bool]:
    """Compare every reported quantity against the pinned oracle.

    Comparison precision matches presentation: MSE differences and interval
    bounds to 4 dp, counts exactly.

    Returns:
        (lines, all_pass).
    """
    lines: list[str] = []
    all_pass = True

    all_pass &= _check_line(
        len(main) == _ORACLE_MAIN_ROWS and len(windows) == _ORACLE_WINDOW_ROWS,
        f"row counts: aggregate {len(main)}/{_ORACLE_MAIN_ROWS}, "
        f"windows {len(windows)}/{_ORACLE_WINDOW_ROWS}",
        lines,
    )
    all_pass &= _check_line(
        len(recon) == _ORACLE_MAIN_ROWS,
        f"reconciliation join: {len(recon)}/{_ORACLE_MAIN_ROWS} runs matched on "
        f"(mode, pred_len, seed)",
        lines,
    )
    # Guards against paired_stats silently dropping a (cell, seed) whose base or
    # alt mode is absent, which would shrink df without changing the output shape.
    short = {
        label: int((frame["n"] != _ORACLE_N_PER_CELL).sum())
        for label, frame in (("aggregate", agg), ("diffed", diffed), ("raw", raw))
    }
    n_offenders = sum(short.values())
    detail = (
        "none short"
        if n_offenders == 0
        else ", ".join(f"{label} {count}" for label, count in short.items() if count)
    )
    all_pass &= _check_line(
        n_offenders == 0,
        f"paired n: every cell across aggregate/diffed/raw pairs all "
        f"{_ORACLE_N_PER_CELL} seeds ({detail})",
        lines,
    )
    all_pass &= _check_line(
        offset == _ORACLE_OFFSET,
        f"window_start offset: derived {offset}, target {_ORACLE_OFFSET} (= VAL_END), "
        f"consistent at all {len(_HORIZONS)} horizons",
        lines,
    )
    max_gap = float(recon["rel_gap"].max())
    all_pass &= _check_line(
        max_gap < _RECON_TOL,
        f"window/aggregate reconciliation: max relative gap {max_gap:.2e} over "
        f"{len(recon)} runs, tolerance {_RECON_TOL:.0e}",
        lines,
    )

    agg_indexed = agg.set_index("pred_len")
    for horizon in _HORIZONS:
        row = agg_indexed.loc[horizon]
        got = (round(row["mean_diff"], 4), round(row["ci_lo"], 4), round(row["ci_hi"], 4))
        target = _ORACLE_AGG[horizon]
        all_pass &= _check_line(
            got == target,
            f"aggregate CD-CI H={horizon}: {got[0]:+.4f} [{got[1]:+.4f}, {got[2]:+.4f}]  "
            f"target {target[0]:+.4f} [{target[1]:+.4f}, {target[2]:+.4f}]",
            lines,
        )

    diffed_indexed = diffed.set_index(["pred_len", "high_coupling"])
    for (horizon, group), target in sorted(_ORACLE_DIFFED.items()):
        row = diffed_indexed.loc[(horizon, group == "high")]
        got = (round(row["mean_diff"], 4), round(row["ci_lo"], 4), round(row["ci_hi"], 4))
        all_pass &= _check_line(
            got == target,
            f"diffed CD-CI H={horizon} {group:>4}: {got[0]:+.4f} [{got[1]:+.4f}, {got[2]:+.4f}]  "
            f"target {target[0]:+.4f} [{target[1]:+.4f}, {target[2]:+.4f}]",
            lines,
        )

    raw_indexed = raw.set_index(["pred_len", "high_coupling"])
    for (horizon, group), target in sorted(_ORACLE_RAW.items()):
        row = raw_indexed.loc[(horizon, group == "high")]
        got = (round(row["mean_diff"], 4), round(row["ci_lo"], 4), round(row["ci_hi"], 4))
        all_pass &= _check_line(
            got == target,
            f"raw CD-CI H={horizon} {group:>4}: {got[0]:+.4f} [{got[1]:+.4f}, {got[2]:+.4f}]  "
            f"target {target[0]:+.4f} [{target[1]:+.4f}, {target[2]:+.4f}]",
            lines,
        )
    return lines, all_pass


def main_(argv: list[str] | None = None) -> int:
    """Reproduce the aggregate and coupling-subgroup contrasts.

    Args:
        argv: Optional argument vector (defaults to sys.argv).

    Returns:
        Process exit code: 0 if every value matches the oracle, else 1.
    """
    parser = argparse.ArgumentParser(
        description="ETTh1 coupling-subgroup analysis (aggregate and high/low-coupling CD-CI)."
    )
    parser.add_argument("--windows", type=Path, default=_WINDOWS, help="per-window error CSV.")
    parser.add_argument("--main", type=Path, default=_MAIN, help="aggregate results CSV.")
    parser.add_argument("--partition", type=Path, default=_PARTITION, help="Coupling partition CSV.")
    args = parser.parse_args(argv)

    windows = load_windows(args.windows)
    main = load_main(args.main)
    part = load_partition(args.partition)

    offset = resolve_offset(windows, part)
    recon = reconcile(windows, main)
    agg = aggregate_contrast(main)
    joined_diffed = join_partition(windows, part, offset, "diffed")
    joined_raw = join_partition(windows, part, offset, "raw")
    diffed = subgroup_contrast(joined_diffed)
    raw = subgroup_contrast(joined_raw)
    sizes = group_sizes(joined_diffed)

    print("=== WINDOW KEY ===")
    print(
        f"  Derived offset {offset} (partition 0-indexed within the test region; "
        f"log absolute from VAL_END={_VAL_END}), verified identical at all horizons."
    )
    print(
        f"  Window/aggregate reconciliation: max relative gap "
        f"{float(recon['rel_gap'].max()):.2e} over {len(recon)} runs."
    )

    print("\n=== AGGREGATE CD-CI (paired over seeds {42,123,456,789,1011}, 95% t-CI) ===")
    for _, row in agg.iterrows():
        print(
            f"  H={int(row['pred_len']):>3}: CI mean {row['base_mean']:.4f}  "
            f"CD-CI {row['mean_diff']:+.4f}  95% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  "
            f"signs {row['signs']}  n={int(row['n'])}  ({row['rel_pct']:+.2f}% of CI)"
        )

    print("\n=== COUPLING SPLIT SIZES (windows per run; median split should halve) ===")
    for _, row in sizes.iterrows():
        print(
            f"  H={int(row['pred_len']):>3} {_group_label(bool(row['high_coupling'])):>4}-coupling: "
            f"{int(row['windows_per_run'])} windows/run"
        )

    for label, frame in (("PRIMARY (diffed)", diffed), ("ROBUSTNESS (raw)", raw)):
        print(f"\n=== SUBGROUP CD-CI, {label} coupling split ===")
        for horizon in _HORIZONS:
            for flag in (True, False):
                row = frame[
                    (frame["pred_len"] == horizon) & (frame["high_coupling"] == flag)
                ].iloc[0]
                print(
                    f"  H={horizon:>3} {_group_label(flag):>4}-coupling: CI mean {row['base_mean']:.4f}  "
                    f"CD-CI {row['mean_diff']:+.4f}  95% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  "
                    f"signs {row['signs']}  n={int(row['n'])}  ({row['rel_pct']:+.2f}% of CI)"
                )

    print("\n=== ORACLE CHECK (pinned against the committed CSVs and partition) ===")
    check_lines, all_pass = oracle_check(windows, main, offset, recon, agg, diffed, raw)
    print("\n".join(check_lines))
    print(
        f"\nRESULT: {'PASS - aggregate and subgroup contrasts reproduce' if all_pass else 'FAIL - see lines above'}"
    )
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main_())