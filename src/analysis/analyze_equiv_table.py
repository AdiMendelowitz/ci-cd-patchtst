"""Recompute Table 10 (``tab:equiv``) practical-equivalence effect sizes.

Each row of Table 10 in "Equal Accuracy, Unequal Cost: Channel Dependence in
PatchTST under Controlled Coupling" is a CD-CI (or CD_Head-CI) test-MSE effect
size for one controlled cell under one selection rule. Cells under different
rules are never pooled; every row is recomputed independently from its own
canonical CSV with the shared paired-difference machinery in ``paired_stats``
(matched unit d_s = MSE_alt - MSE_base over shared seeds, 95% paired-t CI --
clustered by seed instead where a row pools multiple cells that share seeds;
see the "clustered" column below).

Row -> source map (column noted where a row reads a non-default metric column):

  AR(1) grid grand mean (9 cells), early-stop      results_grid.csv            [clustered]
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

Only the AR(1) grid grand-mean row pools observations across multiple cells
that share the same five seeds {42, 123, 456, 789, 1011}; every other row is
already a single-cell paired design (a query narrows cell_cols to one cell
before differencing), so the naive per-row CI is valid for those as-is. The
AR(1) grid row uses paired_stats.grand_mean_diff_clustered instead of
grand_mean_diff for exactly this reason -- pooling all 45 (cell, seed) rows
as independent draws understates the interval, since each seed's nine rows
share that seed's data draw. The point estimate is unaffected either way;
only the CI width changes. Any new row that pools more than one cell should
set clustered=True and justify it in this comment the same way.

The val-minimum row reads test_upd_global (best validation update), not
test_epoch_global. The canonical results_block_cov.csv is the n=5 cut, which
gives +0.0000 for the C=84 rho_in=0.9 cell as in main.tex.

Every row also carries a frozen oracle CI (oracle_ci_lo/oracle_ci_hi, 4 dp,
taken from the committed equiv_summary.csv / main.tex at the time the row was
last verified by hand) alongside the existing point-estimate oracle. The
point-estimate check alone cannot catch a wrong CI, since clustering changes
only the interval, not the mean. Both checks must pass for a row to report
PASS.

Run from a clean checkout:

    python analyze_equiv_table.py
    python analyze_equiv_table.py --results-dir path/to/results --out-csv out.csv
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import paired_stats as ps  # noqa: E402  (sibling module; on sys.path when run as a script)

_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _ROOT / "results"
_THRESHOLD_PCT = 1.0  # pre-specified practical-equivalence band, percent of CI mean.


@dataclass(frozen=True)
class RowSpec:
    """One Table 10 row and how to recompute it from a canonical CSV.

    Attributes:
        family: LaTeX family label (column 1).
        cell: LaTeX cell label (column 2).
        rule: LaTeX selection-rule label (column 3).
        csv: Source CSV filename within the results directory.
        cell_cols: Columns that define a matched cell for paired differencing.
        base_mode: Reference mode (subtracted).
        alt_mode: Mode under scrutiny (minuend).
        oracle_mean: main.tex point estimate, 4 dp, for the self-check.
        oracle_ci_lo: main.tex/equiv_summary.csv CI lower bound, 4 dp, for the
            self-check. None skips the CI check (only for a row never yet
            hand-verified against a committed CI; every current row has one).
        oracle_ci_hi: Companion upper bound to oracle_ci_lo.
        group: Group index; a midrule is drawn where the group index increments.
        query: Optional pandas query selecting this row's cell(s); None uses all.
        value: Metric column to difference.
        clustered: If True, the grand-mean CI is computed clustered by seed
            (paired_stats.grand_mean_diff_clustered) instead of treating every
            (cell, seed) row as independent. Set True only when this row pools
            multiple cells that share seeds; see the module docstring.
    """

    family: str
    cell: str
    rule: str
    csv: str
    cell_cols: list[str]
    base_mode: str
    alt_mode: str
    oracle_mean: float
    oracle_ci_lo: float | None
    oracle_ci_hi: float | None
    group: int
    query: str | None = None
    value: str = "test_mse"
    clustered: bool = False


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
        -0.0023,
        0.0048,
        group=0,
        clustered=True,
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
        0.0046,
        0.0113,
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
        -0.0109,
        0.0046,
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
        -0.0092,
        -0.0014,
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
        0.0050,
        0.0119,
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
        -0.0034,
        0.0094,
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
        -0.0035,
        0.0082,
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
        -0.0138,
        0.0082,
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
        0.0015,
        0.0040,
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
        -0.0137,
        0.0013,
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
        -0.0005,
        0.0016,
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
        -0.0010,
        0.0010,
        group=2,
        query="C == 84 and rho_in == 0.9",
    ),
)


def _load_cached(results_dir: Path, name: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Load a source CSV once per run, raising a clear error if it is absent.

    Several rows share a source CSV (results_cd_head.csv backs four rows,
    results_block_cov.csv backs four more); caching avoids re-reading the same
    file from disk once per row.

    Args:
        results_dir: Directory holding the canonical CSVs.
        name: CSV filename.
        cache: Mutable dict reused across calls within one build_table run.

    Returns:
        The loaded DataFrame (shared; callers must not mutate it in place).
    """
    if name not in cache:
        path = results_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Source CSV not found: {path}")
        cache[name] = pd.read_csv(path)
    return cache[name]


def compute_row(spec: RowSpec, results_dir: Path, cache: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Recompute one Table 10 row from its source CSV.

    Args:
        spec: The row specification.
        results_dir: Directory holding the canonical CSVs.
        cache: Shared CSV cache; see _load_cached.

    Returns:
        A flat record with the effect size, CI, relative percent, threshold
        flag, and provenance, plus normalised match flags against the oracle
        point estimate and oracle CI bounds.

    Raises:
        FileNotFoundError: If the row's source CSV is missing.
        ValueError: If the row's query or mode selection yields no data, or a
            clustered row has fewer than two seed clusters.
    """
    row_id = f"{spec.family} / {spec.cell} / {spec.rule}"
    try:
        df = _load_cached(results_dir, spec.csv, cache)
        if spec.query is not None:
            df = df.query(spec.query)
        diff = ps.make_diff_frame(
            df, cell_cols=spec.cell_cols, mode_base=spec.base_mode, mode_alt=spec.alt_mode, value=spec.value
        )
        res = ps.grand_mean_diff_clustered(diff, cluster_col="seed") if spec.clustered else ps.grand_mean_diff(diff)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise type(exc)(f"Row '{row_id}': {exc}") from exc

    rel_pct = 100.0 * res["mean"] / res["base_mean"]
    # Normalise -0.0 to 0.0 so the +0.0000 cell prints with a stable sign.
    mean_4dp = round(res["mean"], 4) + 0.0
    ci_lo_4dp = round(res["ci_lo"], 4) + 0.0
    ci_hi_4dp = round(res["ci_hi"], 4) + 0.0
    ci_match = (
        spec.oracle_ci_lo is None
        or (ci_lo_4dp == round(spec.oracle_ci_lo, 4) and ci_hi_4dp == round(spec.oracle_ci_hi, 4))
    )
    return {
        "family": spec.family,
        "cell": spec.cell,
        "selection_rule": spec.rule,
        "source_csv": spec.csv,
        "value_column": spec.value,
        "contrast": f"{spec.alt_mode}-{spec.base_mode}",
        "cd_minus_ci": mean_4dp,
        "ci_lo": ci_lo_4dp,
        "ci_hi": ci_hi_4dp,
        "rel_pct": round(rel_pct, 2) + 0.0,
        "within_threshold": bool(abs(rel_pct) <= _THRESHOLD_PCT),
        "n": int(res["n"]),
        "oracle_mean": spec.oracle_mean,
        "oracle_ci_lo": spec.oracle_ci_lo,
        "oracle_ci_hi": spec.oracle_ci_hi,
        "mean_match": mean_4dp == round(spec.oracle_mean, 4),
        "ci_match": ci_match,
        "match": (mean_4dp == round(spec.oracle_mean, 4)) and ci_match,
        "group": spec.group,
    }


def build_table(results_dir: Path) -> pd.DataFrame:
    """Recompute every Table 10 row.

    Args:
        results_dir: Directory holding the canonical CSVs.

    Returns:
        One row per Table 10 entry, in display order.
    """
    cache: dict[str, pd.DataFrame] = {}
    return pd.DataFrame([compute_row(spec, results_dir, cache) for spec in _ROWS])


def to_latex(table: pd.DataFrame) -> str:
    """Render the recomputed table as the Table 10 ``tabular`` body.

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
    """Recompute Table 10, write equiv_summary.csv, and print the LaTeX body.

    Args:
        argv: Optional argument vector (defaults to sys.argv).

    Returns:
        Process exit code: 0 if every row matches both its oracle point
        estimate and its oracle CI at 4 dp, else 1.
    """
    parser = argparse.ArgumentParser(description="Recompute Table 10 practical-equivalence effect sizes.")
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

    print("=== TABLE 10 PER-ROW RECOMPUTE (CD-CI test MSE) ===")
    for _, r in table.iterrows():
        status = "PASS" if r["match"] else "FAIL"
        ci_flag = "" if r["ci_match"] else "  [CI MISMATCH]"
        print(
            f"  [{status}] {r['family']:16s} {r['cell']:34s} {r['selection_rule']:12s} "
            f"{r['cd_minus_ci']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] ({r['rel_pct']:+.2f}%) "
            f"<=1%={'Y' if r['within_threshold'] else 'N'}  target {r['oracle_mean']:+.4f}  "
            f"[{r['source_csv']}]{ci_flag}"
        )
    all_pass = bool(table["match"].all())
    print(f"\nSummary CSV: {args.out_csv}")
    print(f"RESULT: {'PASS - all 12 rows reproduce (mean and CI)' if all_pass else 'FAIL - see rows above'}\n")

    print("=== LaTeX (tab:equiv body) ===")
    print(to_latex(table))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())