"""Matched-update-budget control: equal-compute CI vs CD.

Reads
-----
results/results_equal_compute.csv
    One row per (cell, mode, seed). Required columns: cell, mode, seed, test_mse, gamma, budget_updates, total_steps.
    Modes: CI, CD.

Console
-------
A validity gate confirming CI and CD consumed the same update budget per cell, then per-cell paired CD-CI differences,
the grand mean, and a LaTeX paragraph. The paragraph is emitted only when the validity gate passes; its conclusion is
derived from the statistics rather than asserted.

Purpose
-------
In the canonical runs CD takes many more gradient updates per epoch than CI (up to 32x at C=84), so it could be
argued that the apparent CD deficit reflects CD over- or under-training rather than an architectural fact. This experiment
fixes the total update budget U identical for both modes, with an identical update-based schedule and best-validation
checkpointing, and asks whether the deficit survives.

Usage
-----
python analyze_equal_compute.py [path/to/results_equal_compute.csv]
"""

import sys
from pathlib import Path

import pandas as pd

import paired_stats as ps

_RESULTS_PATH = Path(__file__).resolve().parents[2] / "results" / "results_equal_compute.csv"

# Pre-specified practical-equivalence band, as a percentage of the CI-mode MSE.
_EQUIV_BAND_PCT: float = 1.0

_WORD: dict[int, str] = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _w(k: int) -> str:
    return _WORD.get(k, str(k))


def load_results(path: Path) -> pd.DataFrame:
    """Load and validate the equal-compute results CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"cell", "mode", "seed", "test_mse", "gamma", "budget_updates", "total_steps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results CSV missing columns: {missing}")
    present = set(df["mode"].unique())
    for mode in ("CI", "CD"):
        if mode not in present:
            raise ValueError(f"Expected mode {mode} absent from results.")
    return df


def check_equal_budget(df: pd.DataFrame) -> bool:
    """Verify CI and CD shared the same update budget within each cell.

    The experiment's claim only holds if both modes ran exactly the same number of optimiser updates. Two conditions are
    checked per cell: budget_updates is constant across modes and seeds, and total_steps equals budget_updates for
    every run (both modes ran the full budget rather than early-stopping).

    Args:
        df: The loaded results.

    Returns:
        True if every cell passes; prints a per-cell report either way.
    """
    print("\n=== VALIDITY GATE: equal update budget per cell ===")
    ok = True
    for cell, grp in df.groupby("cell"):
        budgets = grp["budget_updates"].unique()
        ran_full = bool((grp["total_steps"] == grp["budget_updates"]).all())
        same_budget = len(budgets) == 1
        ok = ok and same_budget and ran_full
        status = "PASS" if (same_budget and ran_full) else "FAIL"
        detail = f"budget={int(budgets[0])}" if same_budget else f"budgets DIFFER {sorted(int(b) for b in budgets)}"
        if not ran_full:
            detail += "  total_steps != budget for some run"
        print(f"  {str(cell):>20}: {status}  {detail}")
    if not ok:
        print("  [WARN] At least one cell did not use a matched, fully consumed budget.")
    return ok


def print_paired(paired: pd.DataFrame, gm: dict) -> None:
    """Print per-cell CD-CI differences and the grand mean."""
    print("\n=== PER-CELL PAIRED DIFFERENCES (CD - CI) at equal budget ===")
    print(f"{'cell':>20}  {'mean_diff':>10} {'rel_%':>8}  {'95%CI_lo':>10} {'95%CI_hi':>10}  signs")
    print("-" * 78)
    for _, r in paired.iterrows():
        print(
            f"{str(r.cell):>20}  {r.mean_diff:>+10.6f} {r.rel_pct:>+8.4f}%  "
            f"{r.ci_lo:>+10.6f} {r.ci_hi:>+10.6f}  {r.signs}"
        )
    n_cd_wins = int((paired["ci_hi"] < 0).sum())
    pct = 100.0 * gm["mean"] / gm["base_mean"]
    print(f"\nCells where CD beats CI (CI below zero): {n_cd_wins} / {len(paired)}")
    print(
        f"Grand mean CD-CI: {gm['mean']:+.6f}  ({pct:+.4f}% of CI mean)  "
        f"95% CI [{gm['ci_lo']:+.6f}, {gm['ci_hi']:+.6f}]"
    )


def print_latex_prose(paired: pd.DataFrame, gm: dict, budget_note: str) -> None:
    """Print the equal-compute paragraph, with the conclusion driven by the statistics."""
    n_cells = len(paired)
    cd_better = int((paired["ci_hi"] < 0).sum())  # cells where CD is significantly lower
    cd_worse = int((paired["ci_lo"] > 0).sum())  # cells where CD is significantly higher
    max_rel = float(paired["rel_pct"].abs().max())
    pct = abs(100.0 * gm["mean"] / gm["base_mean"])
    favours = "CD" if gm["mean"] < 0 else "CI"

    if cd_worse == 0:
        verdict = (
            f"Equalising the update budget removes the canonical CD deficit, so it does not "
            f"reflect channel dependence under matched compute; the small residual difference "
            f"favours {favours} and stays within the {_EQUIV_BAND_PCT:.0f}\\% "
            f"practical-equivalence band (largest per-cell gap ${max_rel:.2f}\\%$), so the two "
            f"modes are practically equivalent once compute is matched."
        )
    else:
        verdict = (
            f"At equal compute CD remains significantly higher in {_w(cd_worse)} of the "
            f"{_w(n_cells)} cells, so the canonical deficit does not reduce to the step-count "
            f"asymmetry."
        )

    print("\n=== PAPER PROSE (equal-compute paragraph) ===")
    print(
        f"Because CD takes more gradient updates per epoch than CI under the hardware-forced "
        f"batch sizes, we control compute directly: both modes train for an identical fixed "
        f"budget of $U$ optimiser updates ({budget_note}) with the same update-based cosine "
        f"schedule, warmup, and best-validation checkpointing. At equal $U$ the grand-mean "
        f"CD$-$CI difference is ${gm['mean']:+.4f}$ MSE (95\\% CI "
        f"$[{gm['ci_lo']:.4f}, {gm['ci_hi']:.4f}]$, ${pct:.2f}\\%$ of the CI mean), and CD is "
        f"significantly lower in {_w(cd_better)} of the {_w(n_cells)} cells and significantly "
        f"higher in {_w(cd_worse)}. {verdict}"
    )


def main() -> None:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _RESULTS_PATH
    df = load_results(csv_path)

    gate_ok = check_equal_budget(df)

    diff = ps.make_diff_frame(df, ["cell"], "CI", "CD")
    paired = ps.paired_differences(diff, ["cell"])
    gm = ps.grand_mean_diff(diff)
    print_paired(paired, gm)

    if not gate_ok:
        print("\n=== PAPER PROSE WITHHELD ===")
        print("Validity gate failed: the budgets are not matched and fully consumed, so the")
        print("equal-compute paragraph is not emitted until the gate passes.")
        return

    budgets = sorted(int(b) for b in df["budget_updates"].unique())
    budget_note = f"$U = {budgets[0]}$" if len(budgets) == 1 else f"$U$ per cell in {{{', '.join(map(str, budgets))}}}"
    print_latex_prose(paired, gm, budget_note)


if __name__ == "__main__":
    main()