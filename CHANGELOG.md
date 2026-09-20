# Changelog

Releases are annotated git tags. `v1.0-tmlr` is the tag cited in the paper and
is not moved; every later change to the public repository is listed here and
released under a new tag.

## v1.2-tmlr (2026-09-20)

Verification now covers every input and every analysis script.

- `results/SHA256SUMS` lists the digest of every committed results CSV, and
  `src/reproduce_all.py` verifies all of them before running anything, so a
  changed, missing or unlisted input fails the sweep whether or not a script
  reads it. `--write-sums` regenerates the file after a legitimate change.
- `analyze_synthetic.py`, `analyze_boundary.py`, `analyze_equal_compute.py`
  and `analyze_block_cov.py` gained oracle checks pinned to the values the
  paper quotes (Tables 4 and 8, Sections 4.1, 5.2, 5.3 and 5.4) and now exit
  non-zero when a value does not reproduce, matching the other analysis
  scripts. Under `v1.1.1-tmlr` those four ran without a pinned target, so a
  corrupted input to them passed the sweep unless a downstream script caught
  it.
- README: quick-start block at the top; the reproduction section describes the
  two verification layers.

## v1.1.1-tmlr (2026-09-20)

- `src/reproduce_all.py` runs its child scripts with UTF-8 stdout, and prints
  a failed command's output even under `--quiet`. Under `v1.1-tmlr` the
  Granger row failed on Windows because `validate_granger.py` printed
  non-ASCII markers into a code-page pipe; those markers are now ASCII.

## v1.1-tmlr (2026-09-20)

Repository hardening after an independent reproduction audit of `v1.0-tmlr`.
No result file, figure value or paper text changed.

- Added `src/generators/generate_ar1_grid.py`, the compound-symmetry AR(1)
  generator that produced the grid series for seeds {42, 123, 456}, with unit
  tests, and `notebooks/train_grid_extra_seeds.ipynb`, the Kaggle notebook
  that trained seeds {789, 1011}. The README describes the grid's two
  provenance paths.
- Added `src/reproduce_all.py`, which runs every command of the README
  reproduction table and fails on any non-zero exit or non-PASS result.
- `src/analysis/derive_vram_bound.py` now compares its output against the
  committed `results/vram_bound.csv` instead of refusing to run when the file
  exists; the CSV itself is now committed.
- Added `results/SCHEMA.md`, documenting every column of every results file,
  and `CITATION.cff`.
- Removed `results/stage0_ecl_check.csv`, a CPU dry-run artefact of the
  stage-0 probe that nothing read; the T4 measurements the paper uses are
  recorded in `derive_vram_bound.py` and `results/logs/`.
- README: data provenance and licences for ETTh1 and ECL, the figure
  byte-reproducibility caveat, the Python version statement, macOS and Linux
  activation line, results inventory completed, layout tree corrected.

## v1.0-tmlr (2026-09-20)

State of the repository at camera-ready submission of the TMLR paper. During
the hours between the repository going public and the camera-ready submission
this tag was re-pointed several times while the release was being prepared;
it has been fixed at the submission-time commit since and will not move again.
