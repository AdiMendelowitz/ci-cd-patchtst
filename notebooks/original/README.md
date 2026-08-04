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

SHA256 of each file, for verification against any copy. The notebooks are as
they ran, except for a comment reworded in `train_boundary_p4_ci.ipynb` and
`train_boundary_p4_full_plan.ipynb`; no code, configuration, or output
differs.

| File | SHA256 |
|---|---|
| `train_boundary_grid_n3.ipynb` | `3fd70587b1004e0d38de6fd5d1321a1e0675f557db00b441574845fdf6d1e2f2` |
| `train_boundary_grid_n5.ipynb` | `fc961690559a6f6c69caee288663253d24ef1719856501bc94e05346b470356d` |
| `train_boundary_grid_n5_resume.ipynb` | `eb784961c818a24b683217ca083e74ca5b18a39635945648dd15824f90283c2b` |
| `train_boundary_p4_ci.ipynb` | `2312893dfcdaa39012d79e0b642697446ae5cf9f8dc86f25669219af71a78e41` |
| `train_boundary_p4_ci_dlinear.ipynb` | `030d4c20bd05a8bd27de969eda151aaa075292c6dea33ef08217f103c3d8b4dd` |
| `train_boundary_p4_full_plan.ipynb` | `e81c8800c56de37d410a6b584876332807cbfe86698f5df0520f2934de708ee9` |