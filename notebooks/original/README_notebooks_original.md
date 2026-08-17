# Original boundary training notebooks

The notebooks in this directory trained the boundary patch-size sweep
(Table 7, Figure 3), on Kaggle T4 GPUs. Their results are committed as
`results/results_boundary.csv` and `results/results_boundary_p4_ci.csv`, and
are analysed by `src/analysis/analyze_boundary.py`.

Vanilla CD is infeasible at the finest patch sizes under the environment
these notebooks ran in, so the CD arm is filtered out at P in {2, 4}; those
cells appear as N/A in Table 7 and grey in Figure 3.

| File | Trains | Feeds |
|---|---|---|
| `train_boundary_grid_n3.ipynb` | P in {8, 16} CI+CD and P=2 CI, gamma {0, 0.3, 0.6, 0.9}, seeds {42, 123, 456} | `results_boundary.csv` (n=3 rows) |
| `train_boundary_grid_n5.ipynb` | the same grid extended to seeds {789, 1011}, plus the P=2 gamma 0.9 CI cells | `results_boundary.csv` (n=5, 90 rows) |
| `train_boundary_grid_n5_resume.ipynb` | as `grid_n5`, with a resume cell for continuing across sessions | `results_boundary.csv` (n=5 extension sessions) |
| `train_boundary_p4_ci.ipynb` | P=4 CI, gamma {0.6, 0.9}, seeds {42, 123, 456} | `results_boundary_p4_ci.csv` (n=3 rows) |
| `train_boundary_p4_ci_dlinear.ipynb` | DLinear at gamma 0.9, then the same P=4 CI cells, seeds {42, 123, 456} | `results_boundary_p4_ci.csv`; the DLinear results were superseded and are not used in the paper |
| `train_boundary_p4_full_plan.ipynb` | the original P=4 plan (CI+CD, all gammas, seeds {42, 123, 456}), narrowed to CI before completion | superseded by `train_boundary_p4_ci.ipynb` |

The notebook version that extended the P=4 CI cells to n=5 (seeds
{789, 1011}) is not included here.

## Protocol

All six define the same data-generating process and training protocol,
verified constant-for-constant:

- **Data:** VAR(1), C=21, phi=0.8, innovation std 0.1, independent
  innovations, T=20,000 timesteps, chronological 70/10/20 split, z-scored on
  training statistics. This differs from the repository's other synthetic
  experiments in series length, split proportions, and innovation structure
  (the leader-follower sweep uses correlated innovations), so results are not
  comparable across the two families in absolute terms.
- **Windows:** lookback 512, horizon 96, giving 13,393 training windows; the
  train loader drops the last partial batch (104 steps/epoch at batch 128,
  1,674 at batch 8). Two implementations appear — dense stacking in the four
  earlier files, lazy views in the two `grid_n5` files — over the same window
  definition, and produce identical batches.
- **Training:** AdamW, lr 1e-4, weight decay 1e-4, gradient clipping at 1.0,
  batch 128 (CI) and 8 (CD), 10 warmup epochs, 50 max, patience 10, cosine
  decay to a 1e-6 floor, mixed precision, evaluation under autocast.

These are Kaggle-environment notebooks (T4, `/kaggle/working` paths) and
retrain their experiments on that platform.

## File hashes

SHA256 of each file as originally committed (04 Aug 2026), for verification
against any copy from that commit. The notebooks are as they ran, except
for a comment reworded in `train_boundary_p4_ci.ipynb` and
`train_boundary_p4_full_plan.ipynb` at retrieval time; no code,
configuration, or output differs from the as-ran files.

VERIFIED CURRENT (17 Aug 2026): SHA256 recomputed fresh against the current
on-disk files and checked against every value in the table below -- all six
match exactly. The STALE NOTICE below, written 09 Aug 2026 when the table
was genuinely out of date, described a state that no longer holds: the
table's own "12 Aug 2026, current" caption reflects a recomputation that
did happen, this notice just wasn't removed afterward. Retained beneath
this note as the historical record of that gap, per this project's own
artefact discipline, rather than deleted.

STALE NOTICE (09 Aug 2026, superseded 12 Aug 2026 -- see note above): a
repository-wide terminology sweep (scripts_provenance.md Part K) subsequently
renamed each file's markdown title cell — `Account A`/`Account B` became
`Slice A`/`Slice B` — in all six files below. Only that markdown text changed;
no code, configuration, or output cell was touched by the sweep, so the
training protocol documented above remains accurate and verified as stated.
But every hash in the table below now describes the pre-sweep file and will
NOT match the current on-disk or currently-committed copy. Recomputation
against the current files is owed before this table can be used for
verification again; the values below are retained as the historical record
of the 04 Aug retrieval commit, not deleted, per this project's own artefact
discipline (a wrong hash left silently in place is worse than no hash —
mark and recompute, never leave standing).

| File | SHA256 (12 Aug 2026, current) |
|---|---|
| `train_boundary_grid_n3.ipynb` | `4831117041703c6e369bdd847c8d0748270e8fa4db565b0ce53e7de3ff52dad9` |
| `train_boundary_grid_n5.ipynb` | `f7e5f0fd31d152a1e88348057eabc3b26c3c7f871b6569f121408fa4c406ddce` |
| `train_boundary_grid_n5_resume.ipynb` | `aed8d4e1fd45e5792aa825e7bd4998688546945554f23c7df38aae6c409795ac` |
| `train_boundary_p4_ci.ipynb` | `25bda49c44df21416ab48dcc03548990113f9febdbd05f8addae386887ef028e` |
| `train_boundary_p4_ci_dlinear.ipynb` | `4d3dfbae2d9c2f02f9150953daa8306864fd28d7118b8c1e92a749ad49bdd950` |
| `train_boundary_p4_full_plan.ipynb` | `d8091503c283d8b610df61c371696596be814d0f06036f972300ee4b3fcec4ab` |