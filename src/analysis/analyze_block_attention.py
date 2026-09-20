"""Block-attention ablation analysis for the leader-follower sweep.

Reproduces the three-arm leader-follower sweep under
results/: modes {CI, CD, CD_Block} at C = 21,
rho = 0.5, P = 16, gamma in {0.0, 0.3, 0.6, 0.9}, seeds {42, 123, 456, 789,
1011} (60 rows). CD_Block restricts attention to within-group blocks
(models_cd_block.py); the CD-family arms run at the
committed batch 8 while CI stays at its committed 128, so the CD-CI contrast
here is directly comparable to the committed Table 7 sweep.

Outputs, in order: the three-arm pivot (per-gamma mean test MSE and the three
ratios CD/CI, Blk/CI, Blk/CD), the three paired per-seed contrasts with 95%
paired-t CIs (df = 4) reusing the shared ``paired_stats`` machinery, and the
diagnostics panel from diag_b5_gamma06.csv (participation ratio of the
encoder-output feature covariance, first -> last epoch, gamma = 0.6 runs only;
final-epoch pre-clip grad-norm means per mode and gamma; nonfinite-step
accounting). The wording ceiling applies throughout: the panel is reported as
consistent with the stated hypothesis, never as a mechanism claim.

The oracle section pins every reported number against a frozen oracle pivot
plus the deterministic supporting facts recomputed from the committed CSVs.
The paper's numbers for this ablation are taken from this script's output.

Run from a clean checkout:

    python src/analysis/analyze_block_attention.py
    python src/analysis/analyze_block_attention.py --csv path/to/results_block_attention.csv --diag path/to/diag_b5_gamma06.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

import paired_stats as ps  # sibling module; on sys.path when run as a script

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "results"
_CSV = _RESULTS_DIR / "results_block_attention.csv"
_DIAG = _RESULTS_DIR / "diag_b5_gamma06.csv"

_MODES: tuple[str, ...] = ("CI", "CD", "CD_Block")
_GAMMAS: tuple[float, ...] = (0.0, 0.3, 0.6, 0.9)
_SEED_ORDER: tuple[int, ...] = (42, 123, 456, 789, 1011)
_REQUIRED_COLS: set[str] = {"gamma", "mode", "seed", "test_mse", "best_epoch", "batch_size"}
_DIAG_REQUIRED_COLS: set[str] = {
    "mode", "gamma", "seed", "epoch", "grad_norm_mean", "nonfinite_steps", "participation_ratio",
}
# CD-family arms at the committed batch 8, CI at its committed 128.
_EXPECTED_BATCH: dict[str, int] = {"CI": 128, "CD": 8, "CD_Block": 8}

# Oracle: the frozen pivot (mean test MSE over 5 seeds, 4 dp).
# gamma -> (CI, CD, CD_Block, CD/CI, Blk/CI, Blk/CD). Ratios are formed from the
# unrounded means, then rounded for comparison.
_ORACLE_BODY: dict[float, tuple[float, float, float, float, float, float]] = {
    0.0: (1.0270, 1.0287, 1.0273, 1.0016, 1.0003, 0.9986),
    0.3: (1.0326, 1.0393, 1.0372, 1.0065, 1.0045, 0.9980),
    0.6: (1.0298, 1.0377, 1.0353, 1.0077, 1.0054, 0.9977),
    0.9: (1.0278, 1.0362, 1.0335, 1.0082, 1.0056, 0.9974),
}
# Paired-sign oracle recomputed from the committed CSV. CD-CI is positive on
# 5/5 seeds at every gamma > 0; the exact one-sided sign-test p (binomial
# tail at 0.5 over the nonzero differences, 1/32 = 0.03125 when 5/5) is
# computed and pinned below.
# Blk-CD negative-seed counts are 4/5 at gamma <= 0.3 and 3/5 at gamma >= 0.6.
_ORACLE_CDCI_POS: dict[float, int] = {0.3: 5, 0.6: 5, 0.9: 5}
_ORACLE_CDCI_SIGN_P: dict[float, float] = {0.3: 0.031, 0.6: 0.031, 0.9: 0.031}
_ORACLE_BLKCD_NEG: dict[float, int] = {0.0: 4, 0.3: 4, 0.6: 3, 0.9: 3}
# Committed early-stopping fingerprint: CD-family best_epoch range vs CI's.
_ORACLE_BEST_EPOCH: dict[str, tuple[int, int]] = {"CI": (17, 31), "CD": (4, 7), "CD_Block": (4, 7)}
# gamma = 0 internal control: every pairwise mean ratio within this tolerance of 1.
_G0_TOL = 0.002
# Diagnostics integrity, deterministic from the committed diag CSV.
_ORACLE_DIAG_ROWS = 1248
_ORACLE_PR_ROWS = 301          # non-null participation-ratio rows, gamma = 0.6 only
_ORACLE_PR_RUNS = 15           # (mode, seed) runs carrying PR
_ORACLE_NONFINITE_TOTAL = 216  # AMP loss-scale adjustment steps; the training
# notebook excludes them from the per-epoch grad-norm means
_ORACLE_NONFINITE_CI = 0       # none occur in CI runs
_ORACLE_NONFINITE_MIN_EPOCH = 6
# Participation ratio, mean over seeds of the first and last non-null epoch
# per seed at gamma = 0.6 (2 dp).
_ORACLE_PR_FIRST_LAST: dict[str, tuple[float, float]] = {
    "CI": (4.21, 2.64),
    "CD": (3.75, 4.42),
    "CD_Block": (3.87, 8.92),
}
# Final-epoch pre-clip grad-norm mean over seeds, per (mode, gamma), 2 dp.
# Training clips at 1.0, so CD's gamma > 0 values sit above the clip threshold.
_ORACLE_GRAD_FINAL: dict[str, tuple[float, float, float, float]] = {
    "CI": (0.80, 0.90, 0.88, 0.87),
    "CD": (0.73, 1.17, 1.21, 1.23),
    "CD_Block": (0.54, 0.82, 0.84, 0.84),
}


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate the three-arm sweep results.

    Args:
        path: Path to results_block_attention.csv.

    Returns:
        The results frame, unmodified.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If required columns, modes, gammas, the balanced 5-seed
            design, or the batch policy are violated.
    """
    if not path.exists():
        raise FileNotFoundError(f"Block-attention CSV not found: {path}")
    df = pd.read_csv(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    modes = set(df["mode"].unique())
    if set(_MODES) - modes:
        raise ValueError(f"CSV must contain modes {sorted(_MODES)}; found {sorted(modes)}")
    if sorted(df["gamma"].unique()) != sorted(_GAMMAS):
        raise ValueError(f"gamma values {sorted(df['gamma'].unique())} do not match {sorted(_GAMMAS)}")
    for mode in _MODES:
        for gamma in _GAMMAS:
            cell = df[(df["mode"] == mode) & (df["gamma"] == gamma)]
            seeds = sorted(cell["seed"].unique())
            if seeds != sorted(_SEED_ORDER) or len(cell) != len(_SEED_ORDER):
                raise ValueError(
                    f"{mode} gamma={gamma}: seeds {seeds} over {len(cell)} rows "
                    f"do not match one row per seed in {sorted(_SEED_ORDER)}"
                )
        batches = set(df.loc[df["mode"] == mode, "batch_size"].unique())
        if batches != {_EXPECTED_BATCH[mode]}:
            raise ValueError(f"{mode} batch sizes {sorted(batches)} violate batch policy {_EXPECTED_BATCH[mode]}")
    return df


def load_diag(path: Path) -> pd.DataFrame:
    """Load and validate the per-epoch diagnostics.

    Structural checks only; exact counts are asserted in the oracle section so
    a mismatch prints as FAIL rather than aborting the report.

    Args:
        path: Path to diag_b5_gamma06.csv.

    Returns:
        The diagnostics frame, unmodified.

    Raises:
        FileNotFoundError: If the file is absent.
        ValueError: If required columns are missing or a participation-ratio
            row exists outside gamma = 0.6.
    """
    if not path.exists():
        raise FileNotFoundError(f"diagnostics CSV not found: {path}")
    diag = pd.read_csv(path)
    missing = _DIAG_REQUIRED_COLS - set(diag.columns)
    if missing:
        raise ValueError(f"Diagnostics CSV missing columns: {sorted(missing)}")
    pr_gammas = set(diag.loc[diag["participation_ratio"].notna(), "gamma"].unique())
    if pr_gammas - {0.6}:
        raise ValueError(f"Participation ratio recorded outside gamma=0.6: {sorted(pr_gammas)}")
    return diag


def pivot_body(df: pd.DataFrame) -> pd.DataFrame:
    """Per-gamma mean test MSE for each arm and the three ratios.

    Means are taken over seeds at full precision; ratios are formed from the
    unrounded means, not from rounded display values.

    Args:
        df: Output of load_results.

    Returns:
        One row per gamma with columns gamma, ci, cd, blk, cd_ci, blk_ci, blk_cd.
    """
    rows: list[dict[str, float]] = []
    for gamma in _GAMMAS:
        sub = df[df["gamma"] == gamma]
        ci = float(sub.loc[sub["mode"] == "CI", "test_mse"].mean())
        cd = float(sub.loc[sub["mode"] == "CD", "test_mse"].mean())
        blk = float(sub.loc[sub["mode"] == "CD_Block", "test_mse"].mean())
        rows.append(
            {
                "gamma": gamma,
                "ci": ci,
                "cd": cd,
                "blk": blk,
                "cd_ci": cd / ci,
                "blk_ci": blk / ci,
                "blk_cd": blk / cd,
            }
        )
    return pd.DataFrame(rows)


def sign_test_p(signs: str) -> float:
    """Exact one-sided sign-test p for a paired sign string.

    Binomial tail P(K >= n_pos) at 0.5 over the nonzero differences. Zero
    differences are excluded from the trial count; with 5 nonzero seeds and
    5 positives this is 1/32 = 0.03125.

    Args:
        signs: String over {+, -, 0} from paired_differences.

    Returns:
        The exact one-sided p-value; nan if every difference is zero.
    """
    n_pos = signs.count("+")
    n_nonzero = len(signs) - signs.count("0")
    if n_nonzero == 0:
        return float("nan")
    return float(stats.binom.sf(n_pos - 1, n_nonzero, 0.5))


def contrasts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The three paired per-seed contrasts with 95% paired-t CIs (df = 4).

    Sign convention follows paired_stats: diff = alt - base, positive = the
    mode under scrutiny is worse. Blk-CD negative therefore means block
    attention narrows the CD penalty.

    Args:
        df: Output of load_results.

    Returns:
        Mapping from contrast label ("CD-CI", "Blk-CI", "Blk-CD") to the
        paired_differences frame keyed by gamma.
    """
    pairs = {"CD-CI": ("CI", "CD"), "Blk-CI": ("CI", "CD_Block"), "Blk-CD": ("CD", "CD_Block")}
    out: dict[str, pd.DataFrame] = {}
    for label, (base, alt) in pairs.items():
        diff = ps.make_diff_frame(df, cell_cols=["gamma"], mode_base=base, mode_alt=alt)
        out[label] = ps.paired_differences(diff, cell_cols=["gamma"])
    return out


def b5_panel(diag: pd.DataFrame) -> dict[str, object]:
    """Diagnostics: PR trajectory, final grad norms, nonfinite accounting.

    Participation ratio is summarised as the mean over seeds of the first and
    last non-null epoch per seed (gamma = 0.6 runs). Grad norms are the mean
    over seeds of each run's final-epoch pre-clip value. Nonfinite steps are
    counted as recorded; they were excluded from the per-epoch grad-norm means
    upstream in the training notebook.

    Args:
        diag: Output of load_diag.

    Returns:
        Dict with keys pr_first_last (mode -> (first, last)), grad_final
        (mode -> tuple over _GAMMAS), nonfinite_total, nonfinite_ci,
        nonfinite_min_epoch, n_rows, pr_rows, pr_runs.
    """
    pr = diag[diag["participation_ratio"].notna()].sort_values("epoch")
    per_seed = pr.groupby(["mode", "seed"])["participation_ratio"].agg(first="first", last="last")
    pr_first_last = {
        mode: (float(grp["first"].mean()), float(grp["last"].mean()))
        for mode, grp in per_seed.groupby(level="mode")
    }
    final_rows = diag.sort_values("epoch").groupby(["mode", "gamma", "seed"]).tail(1)
    grad_by_cell = (
        final_rows.groupby(["mode", "gamma"])["grad_norm_mean"]
        .mean()
        .reindex(pd.MultiIndex.from_product([_MODES, _GAMMAS], names=["mode", "gamma"]))
    )
    grad_final = {
        mode: tuple(float(grad_by_cell.loc[(mode, g)]) for g in _GAMMAS) for mode in _MODES
    }
    nonzero = diag[diag["nonfinite_steps"] > 0]
    return {
        "pr_first_last": pr_first_last,
        "grad_final": grad_final,
        "nonfinite_total": int(diag["nonfinite_steps"].sum()),
        "nonfinite_ci": int(diag.loc[diag["mode"] == "CI", "nonfinite_steps"].sum()),
        "nonfinite_min_epoch": int(nonzero["epoch"].min()) if not nonzero.empty else -1,
        "n_rows": int(len(diag)),
        "pr_rows": int(pr.shape[0]),
        "pr_runs": int(per_seed.shape[0]),
    }


def _check_line(ok: bool, text: str, lines: list[str]) -> bool:
    """Append one PASS/FAIL line and return the flag for accumulation."""
    lines.append(f"  [{'PASS' if ok else 'FAIL'}] {text}")
    return ok


def oracle_check(
    df: pd.DataFrame,
    body: pd.DataFrame,
    contr: dict[str, pd.DataFrame],
    panel: dict[str, object],
) -> tuple[list[str], bool]:
    """Compare every reported quantity against the pinned oracle.

    Comparison precision matches presentation: means and ratios to 4 dp,
    panel values to 2 dp, counts exactly.

    Args:
        df: Output of load_results.
        body: Output of pivot_body.
        contr: Output of contrasts.
        panel: Output of b5_panel.

    Returns:
        (lines, all_pass).
    """
    lines: list[str] = []
    all_pass = True
    body_indexed = body.set_index("gamma")
    for gamma in _GAMMAS:
        row = body_indexed.loc[gamma]
        got = tuple(round(row[c], 4) for c in ("ci", "cd", "blk", "cd_ci", "blk_ci", "blk_cd"))
        target = _ORACLE_BODY[gamma]
        all_pass &= _check_line(
            got == target,
            f"pivot gamma={gamma}: CI {got[0]:.4f} CD {got[1]:.4f} Blk {got[2]:.4f} "
            f"CD/CI {got[3]:.4f} Blk/CI {got[4]:.4f} Blk/CD {got[5]:.4f}  "
            f"target {' '.join(f'{t:.4f}' for t in target)}",
            lines,
        )
    cdci = contr["CD-CI"].set_index("gamma")
    for gamma, want in _ORACLE_CDCI_POS.items():
        signs = str(cdci.loc[gamma, "signs"])
        p_got = round(sign_test_p(signs), 3)
        p_want = _ORACLE_CDCI_SIGN_P[gamma]
        all_pass &= _check_line(
            signs.count("+") == want and p_got == p_want,
            f"CD-CI gamma={gamma}: positive on {signs.count('+')}/5 seeds ({signs}), "
            f"sign p={p_got:.3f}, target {want}/5 at p={p_want:.3f}",
            lines,
        )
    blkcd = contr["Blk-CD"].set_index("gamma")
    for gamma, want in _ORACLE_BLKCD_NEG.items():
        signs = str(blkcd.loc[gamma, "signs"])
        all_pass &= _check_line(
            signs.count("-") == want,
            f"Blk-CD gamma={gamma}: negative on {signs.count('-')}/5 seeds ({signs}), target {want}/5",
            lines,
        )
    for mode, (lo, hi) in _ORACLE_BEST_EPOCH.items():
        got_lo = int(df.loc[df["mode"] == mode, "best_epoch"].min())
        got_hi = int(df.loc[df["mode"] == mode, "best_epoch"].max())
        all_pass &= _check_line(
            (got_lo, got_hi) == (lo, hi),
            f"best_epoch {mode}: range {got_lo}-{got_hi}, target {lo}-{hi}",
            lines,
        )
    g0 = body_indexed.loc[0.0]
    g0_dev = max(abs(g0["cd_ci"] - 1.0), abs(g0["blk_ci"] - 1.0), abs(g0["blk_cd"] - 1.0))
    all_pass &= _check_line(
        g0_dev <= _G0_TOL,
        f"gamma=0 control: max pairwise mean-ratio deviation {100 * g0_dev:.2f}% <= {100 * _G0_TOL:.1f}%",
        lines,
    )
    counts_ok = (
        panel["n_rows"] == _ORACLE_DIAG_ROWS
        and panel["pr_rows"] == _ORACLE_PR_ROWS
        and panel["pr_runs"] == _ORACLE_PR_RUNS
        and panel["nonfinite_total"] == _ORACLE_NONFINITE_TOTAL
        and panel["nonfinite_ci"] == _ORACLE_NONFINITE_CI
        and panel["nonfinite_min_epoch"] == _ORACLE_NONFINITE_MIN_EPOCH
    )
    all_pass &= _check_line(
        counts_ok,
        f"diag integrity: rows {panel['n_rows']}/{_ORACLE_DIAG_ROWS}, "
        f"PR rows {panel['pr_rows']}/{_ORACLE_PR_ROWS} over {panel['pr_runs']}/{_ORACLE_PR_RUNS} runs, "
        f"nonfinite {panel['nonfinite_total']}/{_ORACLE_NONFINITE_TOTAL} "
        f"(CI share {panel['nonfinite_ci']}, earliest epoch {panel['nonfinite_min_epoch']})",
        lines,
    )
    for mode, (t_first, t_last) in _ORACLE_PR_FIRST_LAST.items():
        first, last = panel["pr_first_last"].get(mode, (float("nan"), float("nan")))
        all_pass &= _check_line(
            (round(first, 2), round(last, 2)) == (t_first, t_last),
            f"PR {mode}: {first:.2f} -> {last:.2f}, target {t_first:.2f} -> {t_last:.2f}",
            lines,
        )
    for mode, targets in _ORACLE_GRAD_FINAL.items():
        got_g = tuple(round(v, 2) for v in panel["grad_final"][mode])
        all_pass &= _check_line(
            got_g == targets,
            f"grad-norm final {mode}: {' '.join(f'{v:.2f}' for v in got_g)}  "
            f"target {' '.join(f'{t:.2f}' for t in targets)}",
            lines,
        )
    return lines, all_pass


def main(argv: list[str] | None = None) -> int:
    """Reproduce the three-arm pivot, paired contrasts, and diagnostics panel.

    Args:
        argv: Optional argument vector (defaults to sys.argv).

    Returns:
        Process exit code: 0 if every value matches the oracle, else 1.
    """
    parser = argparse.ArgumentParser(
        description="Block-attention ablation (three-arm pivot, paired contrasts, diagnostics panel)."
    )
    parser.add_argument("--csv", type=Path, default=_CSV, help="Block-attention results CSV.")
    parser.add_argument("--diag", type=Path, default=_DIAG, help="diagnostics CSV.")
    args = parser.parse_args(argv)

    df = load_results(args.csv)
    diag = load_diag(args.diag)
    body = pivot_body(df)
    contr = contrasts(df)
    panel = b5_panel(diag)

    print("=== THREE-ARM PIVOT (mean test MSE over seeds {42,123,456,789,1011}) ===")
    for _, row in body.iterrows():
        print(
            f"  gamma={row['gamma']}: CI {row['ci']:.4f}  CD {row['cd']:.4f}  Blk {row['blk']:.4f}  "
            f"CD/CI {row['cd_ci']:.4f}  Blk/CI {row['blk_ci']:.4f}  Blk/CD {row['blk_cd']:.4f}"
        )

    base_of = {"CD-CI": "CI", "Blk-CI": "CI", "Blk-CD": "CD"}
    for label, frame in contr.items():
        print(f"\n=== PAIRED {label} (95% paired-t CI, df=4) ===")
        for _, row in frame.iterrows():
            sign_note = (
                f"  exact sign p={sign_test_p(str(row['signs'])):.3f}" if label == "CD-CI" else ""
            )
            print(
                f"  gamma={row['gamma']}: {label} {row['mean_diff']:+.4f}  "
                f"95% CI [{row['ci_lo']:+.4f}, {row['ci_hi']:+.4f}]  signs {row['signs']}  "
                f"({row['rel_pct']:+.2f}% of {base_of[label]}){sign_note}"
            )

    print("\n=== DIAGNOSTICS PANEL (gamma=0.6 participation ratio; final-epoch pre-clip grad norms) ===")
    for mode in _MODES:
        first, last = panel["pr_first_last"].get(mode, (float("nan"), float("nan")))
        print(f"  PR {mode}: first {first:.2f} -> last {last:.2f} (mean over 5 seeds)")
    for mode in _MODES:
        vals = "  ".join(
            f"g={g}: {v:.2f}" for g, v in zip(_GAMMAS, panel["grad_final"][mode])
        )
        print(f"  grad-norm final {mode}: {vals}  (clip threshold 1.0)")
    print(
        f"  nonfinite steps: {panel['nonfinite_total']} total, {panel['nonfinite_ci']} in CI runs, "
        f"earliest at epoch {panel['nonfinite_min_epoch']}; "
        f"excluded from grad-norm means by the training notebook"
    )

    print("\n=== ORACLE CHECK (frozen pivot + committed-CSV supporting facts) ===")
    check_lines, all_pass = oracle_check(df, body, contr, panel)
    print("\n".join(check_lines))
    print(f"\nRESULT: {'PASS - pivot and diagnostics panel reproduce' if all_pass else 'FAIL - see lines above'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
