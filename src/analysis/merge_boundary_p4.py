"""Merge boundary P=4 slices for one gamma into the file of record.

Generalizes merge_boundary_p4_gamma0.py and merge_boundary_p4_gamma06.py
(both superseded by this script) into a single implementation taking gamma
as a parameter, rather than a third near-identical copy for gamma=0.3.

Reads every results_boundary_p4*.csv found anywhere under the boundary_p4
results root, EXCLUDING any prior *_complete.csv (an output of this script
or a sibling gamma's run, never a source slice) and results_boundary_p4_ci.csv
(a distinct, already-provenanced historical artifact, not a slice output --
see the _EXCLUDED_NAMES comment below). Slices are discovered by
file location rather than by a hardcoded per-gamma tag list, since slice
folder naming has not been consistent across gamma tranches (gamma0_a1/2/3,
gamma06_s1/2/3, and gamma-0.3's own naming may differ again) -- discovering
by glob and filtering by the row data itself is correct regardless of
naming, and does not need updating if a future gamma tranche picks yet
another slice-naming convention.

Each discovered file's rows are filtered to the target gamma before any
validation runs, so a slice folder that happens to hold more than one
gamma's data (or a stray file from a different tranche) cannot corrupt the
merge -- only rows matching --gamma are ever considered.

Validates, per discovered slice (one slice = one distinct source file's
parent folder): schema completeness, protocol identity (gamma, patch_size,
batch_size, steps_per_epoch, C -- constant across every boundary-P4 gamma
tranche), and internal (gamma, mode, seed) uniqueness after an exact-
duplicate drop (a slice may hold more than one CSV across resumed
sessions; repeated rows across those sessions must agree exactly, or the
merge refuses rather than picking one silently).

Then validates across slices: no (gamma, mode, seed) triple appears in
more than one slice, and the full canonical 5 CD + 5 CI seed coverage
{42, 123, 456, 789, 1011} is present for the target gamma. This assertion
is deliberately strict -- a partial gamma (e.g. gamma-0.3 mid-tranche)
correctly makes this script refuse to write a "_complete" file, rather
than merging a partial result under a name that claims completeness.

Usage
-----
    python merge_boundary_p4.py --gamma 0.3
    python merge_boundary_p4.py --gamma 0    # note: 0, not 0.0, on the CLI
                                              # is fine; parsed as float either way

Writes
------
results/Revision/train_bound*_p4/results_boundary_p4_gamma<tag>_complete.csv
    <tag> follows the established naming: 0 -> "0", 0.3 -> "03",
    0.6 -> "06", 0.9 -> "09" (the single post-decimal digit, zero-padded to
    two characters, except gamma=0 itself which keeps the pre-existing
    "gamma0_complete.csv" name rather than "gamma00").
"""

import argparse
from pathlib import Path

import pandas as pd

SCHEMA = ["dataset", "C", "rho", "gamma", "patch_size", "mode", "seed",
          "test_mse", "test_mae", "best_epoch", "batch_size",
          "steps_per_epoch", "total_steps"]

EXPECTED_PATCH_SIZE = 4
EXPECTED_BATCH_SIZE = 128
EXPECTED_STEPS_PER_EPOCH = 104
EXPECTED_C = 21
EXPECTED_SEEDS = {42, 123, 456, 789, 1011}
EXPECTED_ROW_COUNTS = {"CD": 5, "CI": 5}

# Established gamma -> output-filename-tag mapping, read from the two
# existing committed filenames (gamma0_complete.csv, gamma06_complete.csv).
# Gamma=0 is irregular (kept as "0", not zero-padded like the others) because
# that is the file already committed under that name; changing it now would
# orphan the existing file rather than extend the convention.
_GAMMA_TAGS = {0.0: "0", 0.3: "03", 0.6: "06", 0.9: "09"}


def gamma_tag(gamma: float) -> str:
    """Output-filename tag for a committed boundary-P4 gamma value.

    Raises for any gamma not already part of the paper's protocol grid
    (0, 0.3, 0.6, 0.9) -- a typo'd or exploratory gamma should fail loudly
    here, not silently produce a plausible-looking output filename.
    """
    if gamma not in _GAMMA_TAGS:
        raise ValueError(
            f"gamma={gamma} is not one of the committed protocol values "
            f"{sorted(_GAMMA_TAGS)}; refusing to guess an output filename tag."
        )
    return _GAMMA_TAGS[gamma]


def validate_schema(df: pd.DataFrame, source: str) -> None:
    """Raise if df is missing any column of the committed boundary schema."""
    missing = set(SCHEMA) - set(df.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")


def validate_protocol_identity(df: pd.DataFrame, gamma: float, source: str) -> None:
    """Raise if any row deviates from the committed boundary-P4 protocol
    fingerprint for this gamma."""
    bad = df[(df.gamma != gamma) | (df.patch_size != EXPECTED_PATCH_SIZE)
             | (df.batch_size != EXPECTED_BATCH_SIZE)
             | (df.steps_per_epoch != EXPECTED_STEPS_PER_EPOCH) | (df.C != EXPECTED_C)]
    if not bad.empty:
        raise ValueError(
            f"{source}: {len(bad)} row(s) break the boundary-P4 gamma={gamma} protocol "
            f"(expected gamma={gamma}, patch_size={EXPECTED_PATCH_SIZE}, "
            f"batch_size={EXPECTED_BATCH_SIZE}, "
            f"steps_per_epoch={EXPECTED_STEPS_PER_EPOCH}, C={EXPECTED_C}):\n"
            f"{bad.to_string(index=False)}"
        )


def discover_p4_root(results_root: Path) -> Path:
    """Locate the boundary_p4 parent regardless of the "boundary"/"boundry"
    spelling on disk (a known local folder typo; committed paths do not
    carry it)."""
    candidates = [p for p in results_root.glob("train_bound*_p4") if p.is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one train_bound*_p4 directory under {results_root}, "
            f"found: {candidates}"
        )
    return candidates[0]


# Filenames excluded from slice discovery even though they match
# results_boundary_p4*.csv: results_boundary_p4_ci.csv is a distinct,
# already-provenanced historical artifact (P=4 CI-only reference data at
# gamma in {0.6, 0.9}, per analyze_boundary.py's own docstring), not a
# per-slice training output. If it ever sits anywhere under p4_root, its
# rows could pass gamma-filtering and protocol-identity checks and get
# silently absorbed into a merge as if it were a slice's contribution --
# excluded by name here rather than relying on directory scoping alone,
# since the real slice-folder layout for any given gamma has not been
# independently confirmed.
_EXCLUDED_NAMES = {"results_boundary_p4_ci.csv"}


def discover_slices(p4_root: Path, gamma: float) -> dict[str, pd.DataFrame]:
    """Find every results_boundary_p4*.csv under p4_root, excluding prior
    *_complete.csv outputs and other known non-slice artifacts, group by
    parent folder, and filter each group's rows to the target gamma.
    Returns {slice_tag: validated_dataframe}."""
    all_matches = sorted(
        p for p in p4_root.rglob("results_boundary_p4*.csv")
        if "_complete" not in p.name and p.name not in _EXCLUDED_NAMES
    )
    if not all_matches:
        raise FileNotFoundError(f"no results_boundary_p4*.csv found under {p4_root}")

    by_folder: dict[Path, list[Path]] = {}
    for path in all_matches:
        by_folder.setdefault(path.parent, []).append(path)

    slices: dict[str, pd.DataFrame] = {}
    for folder, paths in sorted(by_folder.items()):
        tag = folder.name
        print(f"  {tag}: {len(paths)} file(s) -> {[p.name for p in paths]}")
        frames = []
        for path in paths:
            df = pd.read_csv(path)
            validate_schema(df, str(path))
            df = df[df["gamma"] == gamma]
            if not df.empty:
                frames.append(df)
        if not frames:
            print(f"    (no gamma={gamma} rows in this slice, skipped)")
            continue
        combined = pd.concat(frames, ignore_index=True).drop_duplicates()
        validate_protocol_identity(combined, gamma, tag)

        dupes = combined[combined.duplicated(["gamma", "mode", "seed"], keep=False)]
        if not dupes.empty:
            raise ValueError(
                f"{tag}: conflicting rows for the same (gamma, mode, seed):\n"
                f"{dupes.to_string(index=False)}"
            )
        slices[tag] = combined
    return slices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gamma", type=float, required=True,
                        help="Target gamma value, e.g. 0.3")
    parser.add_argument("--results-root", type=str, default="results/Revision",
                        help="Root containing the train_bound*_p4 folder.")
    args = parser.parse_args()

    tag = gamma_tag(args.gamma)  # validates gamma is a committed protocol value first
    results_root = Path(args.results_root)
    p4_root = discover_p4_root(results_root)
    print(f"using boundary P=4 root: {p4_root}")
    print(f"target gamma: {args.gamma}")

    slices = discover_slices(p4_root, args.gamma)
    if not slices:
        raise FileNotFoundError(f"no slice contained any gamma={args.gamma} rows")

    merged = pd.concat(
        [df.assign(_source=slice_tag) for slice_tag, df in slices.items()],
        ignore_index=True,
    )

    dupes = merged[merged.duplicated(["gamma", "mode", "seed"], keep=False)]
    if not dupes.empty:
        raise ValueError(
            f"unexpected overlap across slices:\n{dupes[['_source', 'gamma', 'mode', 'seed']]}"
        )

    counts = merged.groupby("mode").size().to_dict()
    if counts != EXPECTED_ROW_COUNTS:
        raise ValueError(
            f"row count mismatch for gamma={args.gamma}: got {counts}, "
            f"expected {EXPECTED_ROW_COUNTS} -- this gamma is not yet complete "
            f"across all slices, refusing to write a file claiming it is."
        )

    for mode, group in merged.groupby("mode"):
        seeds = set(group["seed"])
        if seeds != EXPECTED_SEEDS:
            raise ValueError(
                f"mode={mode}: seed set {sorted(seeds)} != expected {sorted(EXPECTED_SEEDS)}"
            )

    merged = merged.drop(columns="_source")
    out_path = p4_root / f"results_boundary_p4_gamma{tag}_complete.csv"
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists, refusing to overwrite silently")
    merged.to_csv(out_path, index=False)
    print(f"wrote {len(merged)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
