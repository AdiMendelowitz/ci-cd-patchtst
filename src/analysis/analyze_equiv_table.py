"""Recompute Table 8 (``tab:equiv``) practical-equivalence effect sizes.

Each row of Table 8 in "Equal Accuracy, Unequal Cost: Channel Dependence in
PatchTST under Controlled Coupling" is a CD-CI (or CD_Head-CI) test-MSE effect
size for one controlled cell under one selection rule. Cells under different
rules are never pooled; every row is recomputed independently from its own
canonical CSV with the shared paired-difference machinery in ``paired_stats``
(matched unit d_s = MSE_alt - MSE_base over shared seeds, 95% paired-t CI).

Row -> source map (column noted where a row reads a non-default metric column):

  AR(1) grid grand mean (9 cells), early-stop      results_grid.csv
  LF gamma=0.6, early-stop                         results_cd_head.csv
  LF gamma=0.6, val-minimum                        results_overtrain_summary.csv [test_upd_global]
  LF gamma=0.6, matched-U                          results_equal_compute.csv
  LF gamma=0.9, early-stop                         results_cd_head.csv
  LF gamma=0.6, cross-variate head                 results_cd_head.csv [CD_Head]
  LF gamma=0.9, cross-variate head                 results_cd_head.csv [CD_Head]
  AR(1) C=84 rho=0.9, matched-U                    results_equal_compute.csv
  Block-cov C=21 rho_in=0.5, early-stop            results_block_cov.csv
  Block-cov C=21 rho_in=0.9, early-stop            results_block_cov.csv
  Block-cov C=84 rho_in=0.5, early-stop            results_block_cov.csv
  Block-cov C=84 rho_in=0.9, early-stop            results_block_cov.csv

The val-minimum row reads test_upd_global (best validation update), not
test_epoch_global. practical_equivalence_tables.md in the source tree is stale on
two points and is not trusted: it cites results_block_cov_v2.csv with the
C=84 rho_in=0.9 cell at n=1; the committed results_block_cov.csv is the n=5 cut,
which gives +0.0000 as in main.tex.

Run from a clean checkout:

    python analyze_equiv_table.py
    python analyze_equiv_table.py --results-dir path/to/results --out-csv out.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import paired_stats as ps  # noqa: E402  (sibling module; on sys.path when run as a script)

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "results"
_THRESHOLD_PCT = 1.0  # pre-registered practical-equivalence band, percent of CI mean.


@dataclass(frozen=True)
class RowSpec:
    """One Table 8 row and how to recompute it from a canonical CSV.

    Attributes:
        family: LaTeX family label (column 1).
        cell: LaTeX cell label (column 2).
        rule: LaTeX selection-rule label (column 3).
        csv: Source CSV filename within the results directory.
        cell_cols: Columns that define a matched cell for paired differencing.
        base_mode: Reference mode (subtracted).
        alt_mode: Mode under scrutiny (minuend).
        query: Optional pandas query selecting this row's cell(s); None uses all.
        value: Metric column to difference.
        oracle_mean: main.tex point estimate, 4 dp, for the self-check.
        group: Group index; a midrule is drawn where the group index increments.
    """

    family: str
    cell: str
    rule: str
    csv: str
    cell_cols: list[str]
    base_mode: str
    alt_mode: str
    oracle_mean: float
    group: int
    query: str | None = None
    value: str = "test_mse"


_ROWS: tuple[RowSpec, ...] = (
    RowSpec(
        "AR(1) grid",
        "grand mean (9 cells)",
        "early-stop",
        "results_grid.csv",
        ["C", "rho"],
        "CI",
        "CD",
        0.0013,
        group=0,
    ),
    RowSpec(
        "Leader-follower",
        "$\\gamma = 0.6$",
        "early-stop",
        "results_cd_head.csv",
        ["gamma"],
        "CI",
        "CD",
        0.0079,
        group=1,
        query="gamma == 0.6",
    ),
    RowSpec(
        "Leader-follower",
        "$\\gamma = 0.6$",
        "val-minimum",
        "results_overtrain_summary.csv",
        [],
        "CI",
        "CD",
        -0.0031,
        group=1,
        value="test_upd_global",
    ),
    RowSpec(
        "Leader-follower",
        "$\\gamma = 0.6$",
        "matched-$U$",
        "results_equal_compute.csv",
        ["cell"],
        "CI",
        "CD",
        -0.0053,
        group=1,
        query="cell == 'lf_gamma0.6'",
    ),
    RowSpec(
        "Leader-follower",
        "$\\gamma = 0.9$",
        "early-stop",
        "results_cd_head.csv",
        ["gamma"],
        "CI",
        "CD",
        0.0085,
        group=1,
        query="gamma == 0.9",
    ),
    RowSpec(
        "Leader-follower",
        "$\\gamma = 0.6$, cross-var.\\ head",
        "early-stop",
        "results_cd_head.csv",
        ["gamma"],
        "CI",
        "CD_Head",
        0.0030,
        group=1,
        query="gamma == 0.6",
    ),
    RowSpec(
        "Leader-follower",
        "$\\gamma = 0.9$, cross-var.\\ head",
        "early-stop",
        "results_cd_head.csv",
        ["gamma"],
        "CI",
        "CD_Head",
        0.0023,
        group=1,
        query="gamma == 0.9",
    ),
    RowSpec(
        "AR(1)",
        "$C{=}84$, $\\rho{=}0.9$",
        "matched-$U$",
        "results_equal_compute.csv",
        ["cell"],
        "CI",
        "CD",
        -0.0028,
        group=2,
        query="cell == 'ar1_C84_rho0.9'",
    ),
    RowSpec(
        "Block-cov.",
        "$C{=}21$, $\\rho_{\\text{in}}{=}0.5$",
        "early-stop",
        "results_block_cov.csv",
        ["C", "rho_in"],
        "CI",
        "CD",
        0.0028,
        group=2,
        query="C == 21 and rho_in == 0.5",
    ),
    RowSpec(
        "Block-cov.",
        "$C{=}21$, $\\rho_{\\text{in}}{=}0.9$",
        "early-stop",
        "results_block_cov.csv",
        ["C", "rho_in"],
        "CI",
        "CD",
        -0.0062,
        group=2,
        query="C == 21 and rho_in == 0.9",
    ),
    RowSpec(
        "Block-cov.",
        "$C{=}84$, $\\rho_{\\text{in}}{=}0.5$",
        "early-stop",
        "results_block_cov.csv",
        ["C", "rho_in"],
        "CI",
        "CD",
        0.0005,
        group=2,
        query="C == 84 and rho_in == 0.5",
    ),
    RowSpec(
        "Block-cov.",
        "$C{=}84$, $\\rho_{\\text{in}}{=}0.9$",
        "early-stop",
        "results_block_cov.csv",
        ["C", "rho_in"],
        "CI",
        "CD",
        0.0000,
        group=2,
        query="C == 84 and rho_in == 0.9",
    ),
)


def _load(results_dir: Path, name: str) -> pd.DataFrame:
    """Load a source CSV, raising a clear error if it is absent."""
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Source CSV not found: {path}")
    return pd.read_csv(path)


def compute_row(spec: RowSpec, results_dir: Path) -> dict[str, object]:
    """Recompute one Table 8 row from its source CSV.

    Args:
        spec: The row specification.
        results_dir: Directory holding the canonical CSVs.

    Returns:
        A flat record with the effect size, CI, relative percent, threshold flag,
        and provenance, plus a normalised absolute-rounding match against the
        oracle point estimate.
    """
    df = _load(results_dir, spec.csv)
    if spec.query is not None:
        df = df.query(spec.query)
    diff = ps.make_diff_frame(
        df, cell_cols=spec.cell_cols, mode_base=spec.base_mode, mode_alt=spec.alt_mode, value=spec.value
    )
    res = ps.grand_mean_diff(diff)
    rel_pct = 100.0 * res["mean"] / res["base_mean"]
    # Normalise -0.0 to 0.0 so the +0.0000 cell prints with a stable sign.
    mean_4dp = round(res["mean"], 4) + 0.0
    return {
        "family": spec.family,
        "cell": spec.cell,
        "selection_rule": spec.rule,
        "source_csv": spec.csv,
        "value_column": spec.value,
        "contrast": f"{spec.alt_mode}-{spec.base_mode}",
        "cd_minus_ci": mean_4dp,
        "ci_lo": round(res["ci_lo"], 4) + 0.0,
        "ci_hi": round(res["ci_hi"], 4) + 0.0,
        "rel_pct": round(rel_pct, 2) + 0.0,
        "within_threshold": bool(abs(rel_pct) <= _THRESHOLD_PCT),
        "n": int(res["n"]),
        "oracle_mean": spec.oracle_mean,
        "match": mean_4dp == round(spec.oracle_mean, 4),
        "group": spec.group,
    }


def build_table(results_dir: Path) -> pd.DataFrame:
    """Recompute every Table 8 row.

    Args:
        results_dir: Directory holding the canonical CSVs.

    Returns:
        One row per Table 8 entry, in display order.
    """
    return pd.DataFrame([compute_row(spec, results_dir) for spec in _ROWS])


def to_latex(table: pd.DataFrame) -> str:
    """Render the recomputed table as the Table 8 ``tabular`` body.

    Args:
        table: Output of build_table.

    Returns:
        LaTeX matching the structure of ``tab:equiv`` in main.tex.
    """
    head = [
        "\\begin{tabular}{lllrr c}",
        "    \\toprule",
        "    Family & Cell & Selection rule & CD$-$CI & Rel & $\\le 1\\%$ \\\\",
        "    \\midrule",
    ]
    body: list[str] = []
    prev_group = int(table.iloc[0]["group"])
    for _, r in table.iterrows():
        if int(r["group"]) != prev_group:
            body.append("    \\midrule")
            prev_group = int(r["group"])
        flag = "yes" if r["within_threshold"] else "no"
        body.append(
            f"    {r['family']} & {r['cell']} & {r['selection_rule']} & "
            f"${r['cd_minus_ci']:+.4f}$ & ${r['rel_pct']:+.2f}\\%$ & {flag} \\\\"
        )
    tail = ["    \\bottomrule", "\\end{tabular}"]
    return "\n".join(head + body + tail)


def main(argv: list[str] | None = None) -> int:
    """Recompute Table 8, write equiv_summary.csv, and print the LaTeX body.

    Args:
        argv: Optional argument vector (defaults to sys.argv).

    Returns:
        Process exit code: 0 if every row matches its oracle point estimate at
        4 dp, else 1.
    """
    parser = argparse.ArgumentParser(description="Recompute Table 8 practical-equivalence effect sizes.")
    parser.add_argument("--results-dir", type=Path, default=_RESULTS_DIR, help="Directory of canonical CSVs.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=_RESULTS_DIR / "equiv_summary.csv",
        help="Destination for the machine-readable summary.",
    )
    args = parser.parse_args(argv)

    table = build_table(args.results_dir)

    out_cols = [
        "family",
        "cell",
        "selection_rule",
        "source_csv",
        "value_column",
        "contrast",
        "cd_minus_ci",
        "ci_lo",
        "ci_hi",
        "rel_pct",
        "within_threshold",
        "n",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table[out_cols].to_csv(args.out_csv, index=False, encoding="utf-8")

    print("=== TABLE 8 PER-ROW RECOMPUTE (CD-CI test MSE) ===")
    for _, r in table.iterrows():
        status = "PASS" if r["match"] else "FAIL"
        print(
            f"  [{status}] {r['family']:16s} {r['cell']:34s} {r['selection_rule']:12s} "
            f"{r['cd_minus_ci']:+.4f} ({r['rel_pct']:+.2f}%) <=1%={'Y' if r['within_threshold'] else 'N'}  "
            f"target {r['oracle_mean']:+.4f}  [{r['source_csv']}]"
        )
    all_pass = bool(table["match"].all())
    print(f"\nSummary CSV: {args.out_csv}")
    print(f"RESULT: {'PASS - all 12 rows reproduce' if all_pass else 'FAIL - see rows above'}\n")

    print("=== LaTeX (tab:equiv body) ===")
    print(to_latex(table))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
