"""Analysis: P=4 CD vs CI in the current environment.

The paired contrast, the cross-environment CI reproducibility measurement,
the paste-ready effect-size row (tab:equiv schema) plus a tab:boundary note,
and a frozen oracle are wired to the executed CSV(s). The script exits 0
only when every number reproduces the oracle recomputed from the file(s) of
record; any drift, or any gamma present in the data with no oracle entry
yet, exits 1.

Reads
-----
One or more current-environment CD+CI CSVs, any gamma coverage, unioned and
deduplicated by (gamma, mode, seed) — for example the gamma-0.9 tranche
(results_boundary_p4_complete.csv), the gamma-0 tranche
(results_boundary_p4_gamma0_complete.csv), and the gamma-0.6 tranche
(results_boundary_p4_gamma06_complete.csv) together. CD and CI at a fixed
(gamma, seed) train on the same data draw (generate(gamma, seed) is
deterministic in (gamma, seed)), so the per-seed CD-CI difference is a matched
pair within each gamma. Passing zero files uses the single legacy default
(results_boundary_p4_complete.csv), preserving the original single-tranche
behaviour.

results/results_boundary_p4_ci.csv  (committed, original environment)
    CI only at P=4, gamma in {0.6, 0.9}. Used ONLY as a cross-environment CI-vs-CI
    reproducibility reference; never mixed into the paired contrast. Its per-gamma
    means are the source of the P=4 CI cells already printed in tab:boundary.

Schema (both): dataset, C, rho, gamma, patch_size, mode, seed, test_mse,
test_mae, best_epoch, batch_size, steps_per_epoch, total_steps.

Paths
-----
Candidate directories resolve against the repository root (first parent holding a
results/ directory), so the script runs from any CWD. The boundary P=4 folder is
discovered under results/Revision/ preferring the correctly-spelled
train_boundary_p4, falling back to a train_bound*_p4 glob (covers the known local
"train_boundry_p4" typo; that typo must never reach a
committed path). Explicit paths override discovery entirely:

    python analyze_boundary_p4.py [new_csv ...] [--ci-ref committed_p4_ci.csv]

    # single tranche (legacy-equivalent):
    python analyze_boundary_p4.py results_boundary_p4_complete.csv
    # multiple tranches unioned in one run:
    python analyze_boundary_p4.py results_boundary_p4_complete.csv \\
        results_boundary_p4_gamma0_complete.csv \\
        results_boundary_p4_gamma06_complete.csv --ci-ref results_boundary_p4_ci.csv

Statistical design
------------------
Primary object: d = MSE_CD - MSE_CI per (gamma, seed) at P=4, current environment
only; per-gamma mean with a 95% paired-t CI via paired_stats, plus the exact
one-sided sign test (alternative: CD worse than CI). Seed count per cell is read
from the data. The committed CI-only rows enter a separate environment-comparison
measurement (per-gamma CI means, old vs new), never the paired test.

Outputs (all human-gated for main.tex)
--------------------------------------
One effect-size row per gamma in tab:equiv / equiv_summary.csv schema, and a note
on the corresponding tab:boundary row: in the current environment CD is feasible,
so that cell can carry a CD/CI ratio rather than a CI-only value, and the table
caption's "CD exceeds T4 memory" no longer holds at that patch size. Whether either
edit lands in main.tex is a paper decision, not the script's; the script only
supplies the numbers and flags the mixed-environment caveat. Each pasted row's
source_csv field names the actual input file(s) used for that run, not a fixed
constant.

Protocol identity oracle
------------------------
Every new row must carry batch_size=128, steps_per_epoch=104, patch_size=4,
C=21 (the committed boundary windowing); a row failing this is a protocol break
and the script refuses to analyse. Every gamma present in the unioned input must
have a paired-oracle entry in _ORACLE_PAIRED or the run exits 1 (untested-cell
guard) — add the entry once new numbers are verified by hand, per this project's
standing analysis-before-results discipline; the script never grades its own
freshly-computed numbers as correct by default.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

from paired_stats import make_diff_frame, paired_differences

EXPECTED_SPE = 104
EXPECTED_BATCH = 128
EXPECTED_P = 4
EXPECTED_C = 21
NEW_CSV_NAME = "results_boundary_p4_complete.csv"
THRESHOLD_PCT = 1.0  # pre-registered practical-equivalence band, percent of CI mean.

SCHEMA = ["dataset", "C", "rho", "gamma", "patch_size", "mode", "seed",
          "test_mse", "test_mae", "best_epoch", "batch_size",
          "steps_per_epoch", "total_steps"]

# Frozen oracle, recomputed from the file(s) of record. Precision matches how each
# quantity is printed into main.tex: MSE means / ratio / CD-CI / CI bounds at
# 4 dp, rel_pct at 2 dp, sign-test p at 3 dp. gamma -> targets.
_ORACLE_PAIRED = {
    0.9: {"ci": 0.9749, "cd": 0.9753, "ratio": 1.0004,
          "cd_minus_ci": 0.0004, "ci_lo": -0.0010, "ci_hi": 0.0018,
          "rel_pct": 0.04, "within_threshold": True,
          "k_pos": 3, "n_signs": 5, "sign_p": 0.500, "n": 5},
    # gamma-0 tranche, verified against results_boundary_p4_gamma0_complete.csv
    # (5/5 CD + 5/5 CI seeds, merged from slices a1/a2/a3 with no seed/mode
    # overlap). CD/CI ratio 0.9997 matches the expected internal-control
    # result at gamma=0.
    0.0: {"ci": 0.9836, "cd": 0.9833, "ratio": 0.9997,
          "cd_minus_ci": -0.0003, "ci_lo": -0.0019, "ci_hi": 0.0014,
          "rel_pct": -0.03, "within_threshold": True,
          "k_pos": 3, "n_signs": 5, "sign_p": 0.500, "n": 5},
    # gamma-0.6 tranche, verified against results_boundary_p4_gamma06_complete.csv
    # (5/5 CD + 5/5 CI seeds, merged from slices slice1/slice2/slice3 with no
    # seed/mode overlap).
    0.6: {"ci": 0.9759, "cd": 0.9761, "ratio": 1.0002,
          "cd_minus_ci": 0.0002, "ci_lo": -0.0014, "ci_hi": 0.0019,
          "rel_pct": 0.02, "within_threshold": True,
          "k_pos": 3, "n_signs": 5, "sign_p": 0.500, "n": 5},
}
# gamma -> {"new": new-env CI mean or None (not yet measured), "old": old-env CI
# mean}, 4 dp. The old-env means double as the published tab:boundary P=4 CI
# cells. A gamma with no old-env reference at all (e.g. gamma=0) is absent from
# this dict entirely, not represented with a None old value.
_ORACLE_REPRO = {
    0.6: {"new": 0.9759, "old": 0.9757},
    0.9: {"new": 0.9749, "old": 0.9743},
}


def repo_root() -> Path:
    """First parent of this file containing a results/ directory."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "results").is_dir():
            return parent
    raise FileNotFoundError(
        "No parent of this script contains a results/ directory; run from "
        "inside the repository or pass explicit CSV paths.")


def boundary_p4_root(root: Path) -> Path:
    """Locate the boundary P=4 results folder under results/Revision/.

    Prefers the correctly-spelled train_boundary_p4. Falls back to a
    train_bound*_p4 glob so a local "train_boundry_p4" typo (must never
    reach a committed path) does not break default
    resolution — but fails loudly rather than silently guessing if the glob
    is ambiguous or empty.
    """
    exact = root / "results" / "Revision" / "train_boundary_p4"
    if exact.is_dir():
        return exact
    matches = sorted((root / "results" / "Revision").glob("train_bound*_p4"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            "no train_bound*_p4 folder found under results/Revision/ "
            "(expected train_boundary_p4; pass an explicit CSV path if the "
            "local folder uses a different name)")
    raise FileNotFoundError(
        f"ambiguous boundary P=4 folder — found {matches}; pass an explicit "
        "CSV path instead of relying on discovery")


def resolve_csv(filename: str, rel_candidates: tuple[str, ...],
                override: str | None) -> Path:
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise FileNotFoundError(f"explicit path {p} does not exist")
        return p
    root = repo_root()
    for c in rel_candidates:
        p = root / c / filename
        if p.exists():
            return p
    raise FileNotFoundError(
        f"{filename} not found under {[str(root / c) for c in rel_candidates]}")


def default_new_csv() -> Path:
    """Legacy single-file default, typo-tolerant: the gamma-0.9 file of
    record if no new-CSV paths are given on the command line."""
    return boundary_p4_root(repo_root()) / NEW_CSV_NAME


def load_new(path: Path) -> pd.DataFrame:
    """Schema and protocol-identity checks for one CSV. Cross-gamma balance
    (every gamma has a matched CD/CI seed set) is checked after unioning in
    load_new_multi, since that guard must see the full merged table, not any
    one input file in isolation."""
    df = pd.read_csv(path)
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns {missing}")
    bad = df[(df.batch_size != EXPECTED_BATCH) | (df.steps_per_epoch != EXPECTED_SPE)
             | (df.patch_size != EXPECTED_P) | (df.C != EXPECTED_C)]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} row(s) in {path.name} break boundary-protocol identity "
            f"(expected batch={EXPECTED_BATCH}, spe={EXPECTED_SPE}, "
            f"P={EXPECTED_P}, C={EXPECTED_C}):\n{bad.to_string(index=False)}")
    if df.duplicated(["gamma", "mode", "seed"]).any():
        dups = df[df.duplicated(["gamma", "mode", "seed"], keep=False)]
        raise ValueError(f"duplicate (gamma, mode, seed) rows within {path.name}:\n{dups}")
    return df


def load_new_multi(paths: list[Path]) -> pd.DataFrame:
    """Union any number of current-environment CSVs (any gamma coverage),
    guarding against the same duplicate-key and unbalanced-cell failures a
    single-file load already guarded against, now checked on the merged
    table so a tranche split across files (e.g. gamma-0.9 and gamma-0 in
    separate files) is validated exactly as if it had always been one file."""
    frames = [load_new(p) for p in paths]
    merged = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if len(frames) > 1 and merged.duplicated(["gamma", "mode", "seed"]).any():
        dups = merged[merged.duplicated(["gamma", "mode", "seed"], keep=False)]
        raise ValueError(f"duplicate (gamma, mode, seed) rows across input files:\n{dups}")
    # Balance guard: make_diff_frame silently drops an unmatched (gamma, seed),
    # shrinking n without warning. Refuse a cell that is not a full CD/CI pair.
    for gamma, cell in merged.groupby("gamma"):
        by_mode = {m: sorted(g.seed.tolist()) for m, g in cell.groupby("mode")}
        if set(by_mode) != {"CI", "CD"} or by_mode.get("CI") != by_mode.get("CD"):
            raise ValueError(f"gamma={gamma} is not a balanced CD/CI cell: {by_mode}")
    return merged


def load_committed_ci(path: Path) -> pd.DataFrame:
    """The committed reference must be exactly what it claims to be
    (adjacent-but-wrong-artifact guard, twice bitten in this project)."""
    df = pd.read_csv(path)
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns {missing}")
    problems = []
    if set(df["mode"].unique()) != {"CI"}:
        problems.append(f"modes {sorted(df['mode'].unique())} != ['CI']")
    if not (df.patch_size == EXPECTED_P).all():
        problems.append(f"patch sizes {sorted(df.patch_size.unique())} != [{EXPECTED_P}]")
    if not (df.steps_per_epoch == EXPECTED_SPE).all():
        problems.append(f"spe {sorted(df.steps_per_epoch.unique())} != [{EXPECTED_SPE}]")
    if not (df.batch_size == EXPECTED_BATCH).all():
        problems.append(f"batch {sorted(df.batch_size.unique())} != [{EXPECTED_BATCH}]")
    if problems:
        raise ValueError(
            f"{path.name} is not the committed P=4 CI reference: " + "; ".join(problems))
    return df


def sign_test(diffs: pd.Series) -> tuple[int, int, float]:
    """Exact one-sided sign test that CD is worse than CI (median d > 0).

    Zero differences are excluded from both the count and n, the standard
    treatment. Returns (positives, non-zero n, p-value); with no non-zero
    difference the test is undefined and p is reported as 1.0.
    """
    d = [float(x) for x in diffs]
    k = sum(1 for x in d if x > 0)
    n = sum(1 for x in d if x != 0)
    if n == 0:
        return 0, 0, 1.0
    return k, n, float(stats.binomtest(k, n, 0.5, alternative="greater").pvalue)


def paired_contrast(new: pd.DataFrame) -> pd.DataFrame:
    """Per-gamma matched CD-CI contrast with 95% paired-t CI and sign test.

    Returns one row per gamma: gamma, ci_mean, cd_mean, ratio, cd_minus_ci,
    ci_lo, ci_hi, rel_pct, within_threshold, n, k_pos, n_signs, sign_p.
    """
    diff = make_diff_frame(new, cell_cols=["gamma"], mode_base="CI", mode_alt="CD")
    pf = paired_differences(diff, cell_cols=["gamma"]).set_index("gamma")
    rows = []
    for gamma, grp in diff.groupby("gamma"):
        p = pf.loc[gamma]
        ci_mean = float(p["base_mean"])
        cd_mean = ci_mean + float(p["mean_diff"])
        rel_pct = float(p["rel_pct"])
        k_pos, n_signs, sign_p = sign_test(grp["diff"])
        rows.append({
            "gamma": float(gamma), "ci_mean": ci_mean, "cd_mean": cd_mean,
            "ratio": cd_mean / ci_mean, "cd_minus_ci": float(p["mean_diff"]),
            "ci_lo": float(p["ci_lo"]), "ci_hi": float(p["ci_hi"]),
            "rel_pct": rel_pct, "within_threshold": bool(abs(rel_pct) <= THRESHOLD_PCT),
            "n": int(p["n"]), "k_pos": k_pos, "n_signs": n_signs, "sign_p": sign_p,
        })
    return pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)


def ci_one_sample_halfwidth(values: pd.Series) -> float:
    """95% one-sample t half-width of a set of CI MSEs (a precision scale)."""
    n = int(values.size)
    if n < 2:
        return float("nan")
    se = float(values.std(ddof=1)) / (n ** 0.5)
    return float(stats.t.ppf(0.975, df=n - 1)) * se


def repro_table(new: pd.DataFrame, old_ci: pd.DataFrame) -> pd.DataFrame:
    """Per-gamma CI mean, new env vs old env, with the old-env half-width.

    A measurement, not a test. Old-env-only gammas report new_ci as NaN.
    """
    new_ci = new[new["mode"] == "CI"].groupby("gamma")["test_mse"].mean()
    rows = []
    for gamma, grp in old_ci.groupby("gamma"):
        old_mean = float(grp["test_mse"].mean())
        new_mean = float(new_ci[gamma]) if gamma in new_ci.index else float("nan")
        delta = (new_mean - old_mean) if not pd.isna(new_mean) else float("nan")
        rows.append({"gamma": float(gamma), "new_ci": new_mean, "old_ci": old_mean,
                     "delta": delta, "old_halfwidth": ci_one_sample_halfwidth(grp["test_mse"])})
    return pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)


def paste_fragments(paired: pd.DataFrame, source_label: str) -> str:
    """Human-gated paste material: an equiv-schema effect-size row and a
    tab:boundary note. Neither is applied to main.tex by this script.

    source_label names the actual input file(s) analysed this run."""
    lines = []
    for _, r in paired.iterrows():
        note = "" if bool(r["within_threshold"]) else "  (OUTSIDE 1% band)"
        flag = "yes" if bool(r["within_threshold"]) else "no"
        # One canonical LaTeX cell string, reused verbatim in the CSV row and the
        # rendered row so equiv_summary.csv and tab:equiv cannot drift apart.
        cell = f"$P{{=}}4$, $\\gamma = {r['gamma']:.1f}$ (curr.\\ env.)"
        lines.append("# equiv_summary.csv row (schema: family,cell,selection_rule,source_csv,"
                      "value_column,contrast,cd_minus_ci,ci_lo,ci_hi,rel_pct,within_threshold,n):")
        lines.append(
            f'Leader-follower,"{cell}",early-stop,{source_label},test_mse,CD-CI,'
            f"{r['cd_minus_ci']:.4f},{r['ci_lo']:.4f},{r['ci_hi']:.4f},"
            f"{r['rel_pct']:.2f},{bool(r['within_threshold'])},{int(r['n'])}")
        lines.append("# rendered tab:equiv row (current environment; adoption is a paper decision):")
        lines.append(
            f"Leader-follower & {cell} & early-stop & "
            f"${r['cd_minus_ci']:+.4f}$ & ${r['rel_pct']:+.2f}\\%$ & {flag} \\\\{note}")
        lines.append("# tab:boundary note: at P=4 CD is feasible in the current environment, so the")
        lines.append(f"#   ($P{{=}}4$, $\\gamma = {r['gamma']:.1f}$) cell can carry a CD/CI ratio "
                     f"{r['ratio']:.4f} (curr.-env CI {r['ci_mean']:.4f}, CD {r['cd_mean']:.4f})")
        lines.append("#   rather than the CI-only entry now shown; the caption clause")
        lines.append("#   \"CD exceeds T4 memory\" no longer holds at P=4.")
    return "\n".join(lines)


def _oracle(paired: pd.DataFrame, repro: pd.DataFrame) -> tuple[list[str], bool]:
    lines: list[str] = []
    ok_all = True

    data_gammas = set(round(float(g), 6) for g in paired["gamma"])
    oracle_gammas = set(round(float(g), 6) for g in _ORACLE_PAIRED)
    untested = data_gammas - oracle_gammas
    if untested:
        lines.append(f"  [FAIL] gamma(s) {sorted(untested)} present in the data with no "
                     f"oracle entry yet (known: {sorted(oracle_gammas)}) — add a verified "
                     f"entry to _ORACLE_PAIRED before trusting these numbers")
        ok_all = False

    pindexed = paired.set_index("gamma")
    for gamma in sorted(data_gammas & oracle_gammas):
        tgt = _ORACLE_PAIRED[gamma]
        r = pindexed.loc[gamma]
        got = {
            "ci": round(float(r["ci_mean"]), 4), "cd": round(float(r["cd_mean"]), 4),
            "ratio": round(float(r["ratio"]), 4), "cd_minus_ci": round(float(r["cd_minus_ci"]), 4),
            "ci_lo": round(float(r["ci_lo"]), 4), "ci_hi": round(float(r["ci_hi"]), 4),
            "rel_pct": round(float(r["rel_pct"]), 2), "within_threshold": bool(r["within_threshold"]),
            "k_pos": int(r["k_pos"]), "n_signs": int(r["n_signs"]),
            "sign_p": round(float(r["sign_p"]), 3), "n": int(r["n"]),
        }
        ok = got == tgt
        ok_all = ok_all and ok
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] paired gamma={gamma}: {got}")
        if not ok:
            lines.append(f"           target: {tgt}")

    rindexed = repro.set_index("gamma")
    for gamma in sorted(data_gammas):
        if gamma not in _ORACLE_REPRO:
            continue  # no old-env reference exists for this gamma (e.g. gamma=0);
                      # repro is a cross-environment measurement, nothing to check
        if gamma not in rindexed.index:
            lines.append(f"  [FAIL] repro gamma={gamma}: expected in repro table, absent")
            ok_all = False
            continue
        tgt = _ORACLE_REPRO[gamma]
        r = rindexed.loc[gamma]
        if pd.isna(r["new_ci"]):
            lines.append(f"  [FAIL] repro gamma={gamma}: gamma is in the loaded new-env "
                         f"data but produced no new-env CI mean (unexpected; check for an "
                         f"all-CD or empty-CI cell)")
            ok_all = False
            continue
        got_new = round(float(r["new_ci"]), 4)
        got_old = round(float(r["old_ci"]), 4)
        if tgt["new"] is None:
            lines.append(f"  [FAIL] repro gamma={gamma}: new-env data is now present "
                         f"(new_ci={got_new:.4f}) but _ORACLE_REPRO still has 'new': None "
                         f"— add the verified value before trusting this number")
            ok_all = False
            continue
        ok = (got_new == tgt["new"]) and (got_old == tgt["old"])
        ok_all = ok_all and ok
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] repro gamma={gamma}: "
                     f"new {got_new:.4f} old {got_old:.4f}  "
                     f"target new {tgt['new']:.4f} old {tgt['old']:.4f}")
    return lines, ok_all


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analysis: P=4 CD vs CI in the current environment. "
                     "Accepts any number of current-environment CSVs (any gamma "
                     "coverage); with none given, falls back to the single legacy "
                     f"default ({NEW_CSV_NAME}).")
    parser.add_argument(
        "new_csvs", nargs="*", metavar="NEW_CSV",
        help="Current-environment CD+CI CSV(s) to union. Omit to use the legacy "
             f"single-file default ({NEW_CSV_NAME}, typo-tolerant discovery).")
    parser.add_argument(
        "--ci-ref", dest="ci_ref", default=None, metavar="PATH",
        help="Committed original-environment CI-only reference CSV. "
             "Defaults to results/results_boundary_p4_ci.csv.")
    return parser


def main(argv: list[str]) -> int:
    args = build_argparser().parse_args(argv[1:])

    if args.new_csvs:
        new_paths = [Path(p) for p in args.new_csvs]
        for p in new_paths:
            if not p.exists():
                raise FileNotFoundError(f"explicit path {p} does not exist")
    else:
        new_paths = [default_new_csv()]
    old_ci_path = resolve_csv("results_boundary_p4_ci.csv", ("results",), args.ci_ref)

    new = load_new_multi(new_paths)
    old_ci = load_committed_ci(old_ci_path)
    source_label = ";".join(p.name for p in new_paths)

    paired = paired_contrast(new)
    repro = repro_table(new, old_ci)

    print(f"=== INPUT: {len(new_paths)} file(s) — {source_label} ===")
    print("=== PAIRED CD-CI, P=4, current environment (matched per seed) ===")
    for _, r in paired.iterrows():
        print(f"  gamma={r['gamma']:.1f}: CI {r['ci_mean']:.4f}  CD {r['cd_mean']:.4f}  "
              f"CD/CI {r['ratio']:.4f}  |  CD-CI {r['cd_minus_ci']:+.4f} "
              f"95% CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] ({r['rel_pct']:+.2f}%)  "
              f"sign {int(r['k_pos'])}/{int(r['n_signs'])} p={r['sign_p']:.3f}  n={int(r['n'])}  "
              f"<=1%={'Y' if bool(r['within_threshold']) else 'N'}")

    print("\n=== CROSS-ENVIRONMENT CI REPRODUCIBILITY (measurement, not a test) ===")
    for _, r in repro.iterrows():
        new_s = "  n/a " if pd.isna(r["new_ci"]) else f"{r['new_ci']:.4f}"
        delta_s = "  n/a " if pd.isna(r["delta"]) else f"{r['delta']:+.4f}"
        print(f"  gamma={r['gamma']:.1f}: new-env CI {new_s}  old-env CI {r['old_ci']:.4f}  "
              f"delta {delta_s}  (old-env 95% half-width {r['old_halfwidth']:.4f})")

    print("\n=== PASTE MATERIAL (human-gated; not applied to main.tex) ===")
    print(paste_fragments(paired, source_label))

    print("\n=== ORACLE CHECK (recomputed from the file(s) of record) ===")
    check, ok_all = _oracle(paired, repro)
    print("\n".join(check))
    print(f"\nRESULT: {'PASS - boundary P=4 current-env reproduces' if ok_all else 'FAIL - see lines above'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))