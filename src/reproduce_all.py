"""Run every analysis command from the README reproduction table and report.

The run first verifies every ``results/*.csv`` against ``results/SHA256SUMS``
(line endings normalised), so a changed, missing or unlisted input fails
before any script runs. Each command is then run from the repository root
with the current interpreter. A command fails when it exits non-zero or when
its ``RESULT:`` line does not read ``RESULT: PASS``; every analysis script
prints one, pinned to the values quoted in the paper. The run exits 1 if
anything failed, so a broken reproduction path is caught before a change is
committed. Figure files rewritten by the
scripts are listed for information; their bytes are not expected to match
the committed files across matplotlib builds, so a rewritten figure is not a
failure.

Usage, from the repository root:
    python src/reproduce_all.py            # every command in the table
    python src/reproduce_all.py --tests    # also run the unit tests first
    python src/reproduce_all.py --quiet    # summary only; failed commands still print
    python src/reproduce_all.py --write-sums   # regenerate results/SHA256SUMS
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_FIGURES = _ROOT / "paper" / "figures"
_RESULTS = _ROOT / "results"
_SUMS = _RESULTS / "SHA256SUMS"

# One entry per row of the README reproduction table, in table order.
COMMANDS: list[tuple[str, list[str]]] = [
    ("AR(1) grid table and heatmap", ["src/analysis/analyze_synthetic.py"]),
    ("ETTh1 and ECL tables and figure", ["src/analysis/analyze_realdata.py"]),
    ("ETTh1 coupling-subgroup contrast", ["src/analysis/analyze_etth1_subgroup.py"]),
    ("Leader-follower gamma sweep and slope", ["src/analysis/analyze_leader_follower.py"]),
    ("Boundary patch-size table and heatmap", [
        "src/analysis/analyze_boundary.py",
        "results/results_boundary_p4_ci.csv",
        "results/results_boundary.csv",
        "results/results_boundary_p4_gamma0_complete.csv",
        "results/results_boundary_p4_gamma03_complete.csv",
        "results/results_boundary_p4_gamma06_complete.csv",
        "results/results_boundary_p4_gamma09_complete.csv",
    ]),
    ("Selection-rule diagnostic and trajectories", ["src/analysis/analyze_overtrain.py"]),
    ("Matched-compute control", ["src/analysis/analyze_equal_compute.py"]),
    ("Cross-variate-head control", ["src/analysis/analyze_cd_head.py"]),
    ("Block-covariance family", ["src/analysis/analyze_block_cov.py"]),
    ("Block-wise attention ablation", ["src/analysis/analyze_block_attention.py"]),
    ("P=4 boundary cells (all four gamma)", ["src/analysis/analyze_boundary_p4.py"]),
    ("C=84 three-arm block-attention cell", ["src/analysis/analyze_boundary_c84.py"]),
    ("Compute-versus-accuracy figure", [
        "src/analysis/make_compute_accuracy_fig.py", "--outdir", "paper/figures",
    ]),
    ("Attention-memory bound and measured peak", ["src/analysis/derive_vram_bound.py"]),
    ("Practical-equivalence summary", ["src/analysis/analyze_equiv_table.py"]),
    ("Theoretical forecast-error ceilings", ["src/analysis/derive_theoretical_bounds.py"]),
    ("Granger non-causality", ["src/analysis/validate_granger.py"]),
]


def _figure_digests() -> dict[str, str]:
    if not _FIGURES.is_dir():
        return {}
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(_FIGURES.iterdir()) if p.is_file()
    }


def _sha256(path: Path) -> str:
    """Digest of the file with CRLF normalised to LF, so a checkout with either
    line-ending convention yields the same value."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_input_digests() -> tuple[int, str]:
    """Verify every results CSV against results/SHA256SUMS before anything runs.

    Returns (exit code, report). A missing, extra or altered file is a failure:
    the oracles inside the scripts catch drift in the numbers they pin, this
    check catches any change to any input, including columns no script reads.
    """
    if not _SUMS.exists():
        return 1, f"{_SUMS.relative_to(_ROOT)} is missing\n"
    expected: dict[str, str] = {}
    for line in _SUMS.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split(None, 1)
            expected[name.strip().lstrip("*")] = digest
    present = {p.name for p in _RESULTS.glob("*.csv")}
    lines: list[str] = []
    bad = 0
    for name in sorted(expected):
        path = _RESULTS / name
        if not path.exists():
            lines.append(f"  [FAIL] {name}: listed in SHA256SUMS but missing")
            bad += 1
            continue
        got = _sha256(path)
        ok = got == expected[name]
        bad += 0 if ok else 1
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}" + ("" if ok else f": digest {got[:12]}... differs"))
    for name in sorted(present - set(expected)):
        lines.append(f"  [FAIL] {name}: present but not listed in SHA256SUMS")
        bad += 1
    verified = max(len(expected) - bad, 0)
    report = "\n".join(lines) + f"\n{verified}/{len(expected)} listed files verified\n"
    return (1 if bad else 0), report


def write_input_digests() -> int:
    """Regenerate results/SHA256SUMS from the CSVs present. Run this only when a
    results file has legitimately changed, and commit the new file with it."""
    lines = [
        "# SHA256 of every committed results/*.csv, computed with CRLF normalised to LF.",
        "# Verified by src/reproduce_all.py before any analysis runs. Regenerate with",
        "#   python src/reproduce_all.py --write-sums",
        "# only when a results file has legitimately changed, and commit both together.",
    ]
    paths = sorted(_RESULTS.glob("*.csv"))
    lines += [f"{_sha256(p)}  {p.name}" for p in paths]
    _SUMS.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(paths)} digests to {_SUMS.relative_to(_ROOT)}")
    return 0


def _run(args: list[str], quiet: bool) -> tuple[int, str]:
    # The child's stdout is a pipe, whose default encoding on Windows is the
    # ANSI code page; force UTF-8 so scripts that print non-ASCII do not fail
    # under the runner when they succeed in a terminal.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc = subprocess.run(
        [sys.executable, *args], cwd=_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    output = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
    if not quiet or proc.returncode != 0:
        print(output, end="" if output.endswith("\n") else "\n")
    return proc.returncode, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tests", action="store_true", help="run the unit tests first")
    parser.add_argument("--quiet", action="store_true",
                        help="print the summary only, plus the output of any failed command")
    parser.add_argument("--write-sums", action="store_true",
                        help="regenerate results/SHA256SUMS from the CSVs present and exit")
    opts = parser.parse_args()
    if opts.write_sums:
        return write_input_digests()

    results: list[tuple[str, str, float]] = []
    before = _figure_digests()

    print("===== Input integrity: results/*.csv against results/SHA256SUMS =====")
    t0 = time.time()
    code, report = check_input_digests()
    if not opts.quiet or code != 0:
        print(report, end="")
    results.append(("Input integrity (SHA256SUMS)", "PASS" if code == 0 else "FAIL", time.time() - t0))

    if opts.tests:
        t0 = time.time()
        code, output = _run(["-m", "pytest", "src/tests/", "-q"], opts.quiet)
        results.append(("Unit tests", "PASS" if code == 0 else "FAIL", time.time() - t0))

    for label, args in COMMANDS:
        print(f"\n===== {label}: {' '.join(args)} =====")
        t0 = time.time()
        code, output = _run(args, opts.quiet)
        if code != 0:
            status = f"FAIL (exit {code})"
        elif "RESULT:" in output and "RESULT: PASS" not in output:
            status = "FAIL (RESULT line is not PASS)"
        else:
            status = "PASS"
        results.append((label, status, time.time() - t0))

    after = _figure_digests()
    rewritten = sorted(k for k in after if before.get(k) != after[k])

    width = max(len(r[0]) for r in results)
    print("\n" + "=" * (width + 30))
    print("REPRODUCTION SUMMARY")
    print("=" * (width + 30))
    for label, status, seconds in results:
        print(f"{label:<{width}}  {status:<28}  {seconds:5.1f}s")
    failed = [r for r in results if not r[1].startswith("PASS")]
    if rewritten:
        print(f"\nFigures rewritten (bytes changed, values are fixed by the CSVs): {', '.join(rewritten)}")
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
