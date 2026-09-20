# Results schema

Every CSV under `results/`, its writer, its readers, and the meaning of every
column. Conventions that hold across files are stated once here; the per-file
sections note only what differs.

## Conventions

All error metrics (`test_mse`, `test_mae`, `val_mse`, `sq_err`, `abs_err`,
`bound`) are on z-scored data, per channel, with the mean and standard
deviation fitted on the training split only, and are element means over the
(horizon x channels) prediction block. A value of 1.0 is therefore the
no-signal ceiling in every file, which is what makes `theoretical_bounds.csv`
comparable with the training results.

`best_epoch` is 1-based in every file. `steps_per_epoch` is the number of
optimiser updates per epoch, `len(train_loader)`: `ceil(n_train / batch)` for
every engine except the boundary P=4 gamma-slice engine, which uses
`drop_last=True` and therefore `floor(n_train / batch)`.

`total_steps` is the cumulative optimiser-update count at the end of the
best-validation epoch, so `total_steps == best_epoch * steps_per_epoch` in every
row of every training-run file, checked over all committed rows when this
document was written; the paper's Section 5.1 relies on it. The ETTh1 files carry the same quantity
under the name `total_steps_to_best`. The one file where `total_steps` means
something else is `results_equal_compute.csv`, where it is the fixed update
budget both arms ran to completion and `best_update` carries the
updates-to-best; see that section.

`test_mse` and `test_mae` are measured with the validation-best weights
restored. Seeds are {42, 123, 456, 789, 1011} unless stated. `mode` is one of
`CI`, `CD`, `DLinear`, `CD_Head`, `CD_Block`; each file lists which appear.

Three files are script outputs committed for reference rather than
experimental inputs: `equiv_summary.csv`, `theoretical_bounds.csv` and
`vram_bound.csv`. Their scripts recompute them and compare against the
committed copy.

Notebooks write to `/kaggle/working/<name>.csv`; the file is then committed
under `results/`.

## Training-run files with the shared schema

`results_grid.csv`, `results_grid_C84_block_attn.csv`,
`results_leader_follower.csv`, `results_boundary.csv`,
`results_boundary_p4_ci.csv`, `results_boundary_p4_gamma{0,03,06,09}_complete.csv`,
`results_cd_head.csv`, `results_block_cov.csv`, `results_block_attention.csv`.

Shared columns:

| Column | Meaning |
| --- | --- |
| `dataset` | Provenance tag: `synthetic_ar1`, `leader_follower_var1`, `block_cov_ar1`. |
| `C` | Number of variates. |
| `mode` | Model arm. |
| `seed` | Integer seed; sets the series and the training RNG. |
| `test_mse`, `test_mae` | Test-split metrics at the validation-best checkpoint. |
| `best_epoch` | 1-based epoch of the validation minimum. |
| `batch_size` | Batch actually used; the non-resumable engines halve it on OOM and record the halved value. |
| `steps_per_epoch` | Optimiser updates per epoch. |
| `total_steps` | Updates at the end of `best_epoch`; equals `best_epoch * steps_per_epoch`. |

Shared protocol unless a file says otherwise: seq_len 512, pred_len 96,
patch 16, stride 8 (63 patches), d_model 64, 8 heads, 3 layers, dropout 0.2,
AdamW at lr 1e-4 and weight decay 1e-4, gradient clip 1.0, AMP, max 50 epochs,
patience 10, 10 warmup epochs then cosine (DLinear has no warmup), 14,400
usable timesteps after 1,000 burn-in, 60/20/20 chronological split.

### results_grid.csv (135 rows)

Writer: `notebooks/train_grid_extra_seeds.ipynb` for seeds {789, 1011}; the
notebook for seeds {42, 123, 456} is not committed (README, Reproducing from
scratch). Readers: `analyze_synthetic.py`, `analyze_equiv_table.py`,
`make_compute_accuracy_fig.py`, `analyze_boundary_c84.py`,
`derive_theoretical_bounds.py`.

Modes CI, CD, DLinear over C in {7, 21, 84} and rho in {0.1, 0.5, 0.9}.
Extra columns: `rho`, the target compound-symmetry innovation correlation;
`empirical_rho`, the mean of the upper-triangle pairwise Pearson correlations
of the z-scored training split, measured per (C, rho, seed); `pred_len`, 96.
Batches: CI and DLinear {7: 128, 21: 128, 84: 32}, CD {7: 64, 21: 8, 84: 1}.

### results_grid_C84_block_attn.csv (15 rows)

Writer: `notebooks/train_grid_c84_block_attn.ipynb`. Reader:
`analyze_boundary_c84.py`. Same columns as `results_grid.csv`; a single cell
C=84, rho=0.5 with modes CI, CD, CD_Block (batches 32, 8, 8). Resumable at epoch
granularity.

### results_leader_follower.csv (60 rows)

Writer: `notebooks/train_leader_follower.ipynb`. Readers:
`analyze_leader_follower.py`, `make_compute_accuracy_fig.py`. Extra columns:
`rho` (0.5 throughout), `gamma`, the leader-to-follower VAR(1) coefficient in
{0.0, 0.3, 0.6, 0.9}. C=21 is 10 leaders, 10 followers and 1 isolate. Modes
CI, CD, DLinear with batches 128, 8, 128.

### results_boundary.csv (90 rows) and results_boundary_p4_ci.csv (10 rows)

Writers: the original notebooks under `notebooks/original/` (see the README
there). Readers: `analyze_boundary.py`; the P=4 CI file is also read by
`analyze_boundary_p4.py`. Extra columns: `rho`, `gamma`, `patch_size` in
{2, 8, 16} (`results_boundary.csv`) or 4 (`results_boundary_p4_ci.csv`, CI
only). Metrics rounded to 8 decimal places.

### results_boundary_p4_gamma{0,03,06,09}_complete.csv (10 rows each)

Writers: the gamma-slice notebooks `notebooks/train_boundary_p4_gamma*.ipynb`,
merged by `src/analysis/merge_boundary_p4.py`, which checks protocol identity
across slices, per-slice uniqueness and full 5 CD + 5 CI seed coverage.
Readers: `analyze_boundary_p4.py`, `analyze_boundary.py`. Same columns as
`results_boundary.csv` with `patch_size` 4. This engine uses `drop_last=True`,
so `steps_per_epoch` is 104 at batch 128; the epoch shuffle generator is seeded
`1_000_003 * seed + epoch` so a resumed run repeats the same epochs.

### results_cd_head.csv (45 rows)

Writer: `notebooks/train_cd_head.ipynb`. Readers: `analyze_cd_head.py`,
`analyze_equiv_table.py`, `derive_theoretical_bounds.py`. Extra columns:
`cell`, one of `lf_gamma0.0`, `lf_gamma0.6`, `lf_gamma0.9`; `rho` (0.5);
`gamma` (the same value as in `cell`). Modes CI, CD, CD_Head with batches
128, 8, 8.

### results_block_cov.csv (60 rows)

Writer: `notebooks/train_block_cov.ipynb`. Readers: `analyze_block_cov.py`,
`analyze_equiv_table.py`. Extra columns: `cell`, one of `block_C21_rhoin0.5`,
`block_C21_rhoin0.9`, `block_C84_rhoin0.5`, `block_C84_rhoin0.9`; `rho_in`,
the within-group innovation correlation; `rho_out`, the between-group
correlation, 0.0 throughout; `group_size`, 7 throughout, the number of
consecutive channels per correlated block (3 groups at C=21, 12 at C=84).
Modes CI, CD, DLinear.

### results_block_attention.csv (60 rows)

Writer: `notebooks/train_block_attention.ipynb`. Readers:
`analyze_block_attention.py`, `derive_theoretical_bounds.py`. Extra columns:
`rho` (0.5), `gamma` in {0.0, 0.3, 0.6, 0.9}. Modes CI, CD, CD_Block (batches
128, 8, 8); CD_Block is the block-diagonal cross-variate attention variant in
`src/models_cd_block.py` with groups from `leader_follower_groups()`. Its
per-epoch diagnostics are in `diag_b5_gamma06.csv`.

## diag_b5_gamma06.csv (1,248 rows)

Writer: `notebooks/train_block_attention.ipynb`. Reader:
`analyze_block_attention.py`. One row per (mode, gamma, seed, epoch). Despite
the filename the file holds all four gamma values for all three arms; only the
gamma=0.6 rows carry `participation_ratio`.

| Column | Meaning |
| --- | --- |
| `mode` | CI, CD or CD_Block. |
| `gamma` | Coupling coefficient, {0.0, 0.3, 0.6, 0.9}. |
| `seed` | Seed. |
| `epoch` | 1-based epoch. |
| `grad_norm_mean` | Mean over the epoch's updates of the pre-clip total gradient L2 norm on unscaled gradients; non-finite steps are excluded; NaN if every step overflowed. Values above 1.0 mean clipping was active. |
| `nonfinite_steps` | Number of updates in the epoch whose pre-clip norm was non-finite (AMP overflow; the scaler skips them). |
| `participation_ratio` | (sum of eigenvalues)^2 / sum of squared eigenvalues of the 64 x 64 encoder-output feature covariance on a fixed probe of the first 32 validation windows, float32 in eval mode; range 1 (collapse) to 64 (isotropic). Populated at gamma=0.6 only. |
| `val_mse` | End-of-epoch validation MSE, the early-stopping quantity. |

## results_overtrain_summary.csv (10 rows) and results_overtrain_diag.csv (1,675 rows)

Writer: an instrumented rerun of the leader-follower gamma=0.6, C=21, rho=0.5
cell whose notebook is not committed. Readers: `analyze_overtrain.py`
(both files); `analyze_equiv_table.py` reads `test_upd_global` from the summary.
The column definitions below were checked numerically against the diagnostic
file: every derived summary column reproduces from the diagnostic rows.

### results_overtrain_diag.csv

One row per validation checkpoint, five per epoch plus the epoch boundary
(the fifth CD checkpoint coincides with the boundary): 210 rows per CI seed
(35 epochs), 125 per CD seed (25 epochs).

| Column | Meaning |
| --- | --- |
| `mode`, `seed` | Arm and seed. |
| `update` | Cumulative optimiser-update index at the checkpoint. |
| `epoch` | 1-based epoch containing the checkpoint; at an exact boundary it is that epoch, not the next. |
| `epoch_frac` | `update / steps_per_epoch`; integral at boundaries; the x-axis of the paper's trajectory figure. |
| `is_boundary` | True when the checkpoint is an exact epoch end. |
| `lr` | Learning rate in force (10-epoch linear warmup then cosine, peak 1e-4); constant within an epoch. |
| `val_mse` | Validation MSE of the live weights, evaluated in eval mode without touching the training RNG. |

### results_overtrain_summary.csv

One row per (mode, seed); `steps_per_epoch` is 63 for CI (batch 128) and 1005
for CD (batch 8). Every test-MSE column is the test error of a checkpoint
chosen by one selection rule; the reader maps rules to columns explicitly
(`_RULE_COLUMNS`) rather than inferring from names.

| Column | Meaning |
| --- | --- |
| `mode`, `seed` | Arm and seed. |
| `spe` | Optimiser updates per epoch. |
| `main_best_epoch` | 1-based best epoch of the original committed run in `results_leader_follower.csv`. |
| `replayed_best_epoch` | 1-based epoch of the validation minimum in this rerun over epoch-boundary checkpoints. |
| `stop_epoch` | `replayed_best_epoch + 10`, where patience would have expired. |
| `stop_total` | `stop_epoch * spe`. |
| `epoch_faithful` | True when `abs(main_best_epoch - replayed_best_epoch) <= 1`. |
| `main_test_mse` | Test MSE of the original committed run (rounded to 2 decimals). |
| `test_earlystop` | Test MSE of the checkpoint selected by the epoch-granularity early-stop rule within the stop horizon. |
| `test_earlystop_delta` | `test_earlystop - main_test_mse`. |
| `test_epoch_global` | Test MSE at the global validation minimum over epoch boundaries, no horizon cap. |
| `test_upd_within_stop` | Test MSE at the best checkpoint on the mid-epoch update grid with `update <= stop_total`. |
| `test_upd_global` | Test MSE at the global validation minimum on the update grid, no cap (the "val-minimum" rule in the equivalence table). |
| `test_upd_within_U` | Test MSE at the best update-grid checkpoint with `update <= 3150`, the matched-compute budget for this cell. |
| `cadence_gap` | `test_earlystop - test_upd_within_stop`: the within-arm cost of evaluating only at epoch boundaries; averaged per arm, never differenced across arms. |
| `upd_global_update`, `upd_global_epoch`, `upd_global_lr` | The `update`, `epoch` and `lr` of the global update-grid minimum. |
| `val_min` | Minimum `val_mse` over all checkpoints. |
| `val_end` | `val_mse` at the last checkpoint. |
| `overtrain_tail` | True when the run continued past the early-stopping horizon (all rows). |
| `n_eval_points` | Number of diagnostic rows for this run. |
| `n_upd_rungs` | Number of checkpoints that set a new running validation minimum on the update grid, including the first. |
| `n_ep_rungs` | The same count over epoch-boundary checkpoints only. |

## results_equal_compute.csv (30 rows)

Writer: `notebooks/train_equal_compute.ipynb`. Readers:
`analyze_equal_compute.py`, `analyze_equiv_table.py`,
`derive_theoretical_bounds.py`. Three cells (`ar1_C84_rho0.9`, `lf_gamma0.0`,
`lf_gamma0.6`) by CI and CD by five seeds. Both arms run a fixed budget of
optimiser updates with no early stopping; the budget is derived from CI's
geometry, `MAX_EPOCHS * ceil(n_train / batch_CI)`, and the learning-rate
schedule is per update.

| Column | Meaning |
| --- | --- |
| `dataset` | `equal_compute`. |
| `cell` | Cell label as above. |
| `C`, `rho`, `gamma` | Cell parameters; `gamma` is blank for the AR(1) cell. |
| `mode`, `seed` | Arm and seed. |
| `test_mse`, `test_mae` | Test metrics at the selected checkpoint. |
| `best_update` | Optimiser-update index of the validation minimum; validation runs every `ceil(n_train / batch_CI)` updates and at the budget, so this is a multiple of that interval. This is the updates-to-best quantity that `total_steps` carries elsewhere. |
| `batch_size` | This arm's batch (CI 128 or 32, CD 8 or 1). |
| `steps_per_epoch` | `len(train_loader)` at this arm's batch; informational, it does not divide `total_steps` here. |
| `total_steps` | The fixed update budget actually run, identical for both arms of a cell (12,600 for C=84, 3,150 for the leader-follower cells). |
| `budget_updates` | The same value; the reader asserts `total_steps == budget_updates` and one budget per cell. |

## ETTh1 files

### results_etth1.csv (40 rows) and results_etth1_b4.csv (40 rows)

Writers: `notebooks/train_etth1.ipynb` and `notebooks/train_etth1_b4.ipynb`
(the matched-budget rerun). Readers: `analyze_realdata.py` and
`analyze_etth1_subgroup.py` respectively. The two files are never pooled.
CI and CD by horizons {96, 192, 336, 720} by five seeds; batch 32 for both
arms; evaluation without autocast.

| Column | Meaning |
| --- | --- |
| `mode`, `pred_len`, `seed` | Arm, horizon in hours, seed. |
| `test_mse`, `test_mae` | Test metrics at the validation-best checkpoint (`results_etth1_b4.csv` rounds to 8 decimals). |
| `best_epoch` | 1-based. |
| `batch_size` | 32. |
| `steps_per_epoch` | `ceil(n_train_windows / 32)`: 252, 249, 244, 232 across the four horizons. |
| `total_steps_to_best` | Updates at the end of `best_epoch`; equals `best_epoch * steps_per_epoch`. Same quantity as `total_steps` in the synthetic files. |
| `warmup_epochs` | 10, linear warmup before cosine decay. |
| `min_epochs` | 15; the patience counter does not start before this 1-based epoch. |
| `max_epochs` | 100, the hard cap and the cosine denominator. |

### results_etth1_b4_windows.csv (202,120 rows)

Writer: `notebooks/train_etth1_b4.ipynb`. Reader:
`analyze_etth1_subgroup.py`. One row per test window per (mode, horizon,
seed), from the validation-best checkpoint.

| Column | Meaning |
| --- | --- |
| `mode`, `pred_len`, `seed` | Arm, horizon, seed. |
| `window_start` | Absolute ETTh1 row index of the first input timestep; the test split starts at row 11,520, so values run from 11,520. |
| `sq_err` | Mean squared error over the window's (horizon x 7) prediction elements. Averaging over a run's windows reproduces its `test_mse`. |
| `abs_err` | Mean absolute error over the same elements. |

### etth1_coupling_partition.csv (5,293 rows)

Writer: `src/analysis/partition_etth1_coupling.py`. Reader:
`analyze_etth1_subgroup.py`. One row per test input window at the shortest
horizon. Windows are scored on the train-normalised input only, before any
error data existed; `diffed` is the pre-specified primary metric and `raw` the
robustness metric.

| Column | Meaning |
| --- | --- |
| `window_start` | Window index within the test split, 0-based. This differs from `results_etth1_b4_windows.csv`, which is absolute; the reader derives and asserts the 11,520 offset before joining. |
| `raw` | Mean absolute lag-1 cross-correlation of the levels over ordered off-diagonal channel pairs within the 512-step window. |
| `diffed` | The same statistic on first differences of the window. |
| `high_{raw,diffed}_H{96,192,336,720}` | Per-horizon balanced median split: True when the metric is at or above the median over the windows that exist at that horizon; blank for windows that do not exist at that horizon. Nullable booleans: read back with `astype("boolean")`. |

## realdata_corr_summary.csv (2 rows) and realdata_lag_summary.csv (98 rows)

Writer: `src/analysis/measure_lag_structure.py`. Reader:
`analyze_realdata.py`. Computed on the training splits (ETTh1 8,640 rows, ECL
15,813 rows) with lags 0 to 48 hours. The directional cross-correlation is
`corr(x_i[t+k], x_j[t])`; "off-diagonal" statistics run over all ordered pairs
i != j. Values rounded to 6 decimals.

`realdata_lag_summary.csv`, one row per (dataset, lag):

| Column | Meaning |
| --- | --- |
| `dataset` | `ETTh1` or `ECL`. |
| `lag` | Lag in hours, 0 to 48. |
| `mean_abs_r`, `median_abs_r`, `max_abs_r` | Mean, median and maximum absolute cross-correlation over ordered off-diagonal pairs at that lag. |

`realdata_corr_summary.csv`, one row per dataset:

| Column | Meaning |
| --- | --- |
| `dataset`, `num_variates`, `num_timesteps_train` | Dataset, C (7 or 321), training rows used. |
| `lag0_mean_abs_r`, `lag0_median_abs_r`, `lag0_min_abs_r`, `lag0_max_abs_r` | Absolute contemporaneous correlation statistics over ordered pairs. |
| `mean_max_over_lags` | Mean over unordered pairs of the maximum absolute cross-correlation over lags 1 to 48 in either direction (lag 0 excluded). |
| `best_lag` | The lag whose `mean_abs_r` is largest; 0 means no lead-lag gain. |
| `mean_abs_r_at_best_lag` | `mean_abs_r` at that lag. |
| `pct_pairs_gain_ge_5pct_from_lags` | Percentage of unordered pairs whose best lagged absolute correlation exceeds their lag-0 value by at least 0.05 in correlation units; a descriptive cutoff. |
| `top_pairs_lag0` | The five most correlated lag-0 pairs as a stringified Python list of `(name_i, name_j, r)` tuples; parse with `ast.literal_eval`. |

## results_ecl.csv (4 rows)

Writer: `notebooks/ecl_ci_cd_train_resumable.ipynb`. Reader:
`analyze_realdata.py`. CI and CD at horizons 96 and 336, seed 456. Protocol:
seq_len 96, C=321, patch 16, stride 8, d_model 128, 16 heads, 3 layers,
dropout 0.2, lr 1e-4, 5 warmup epochs, max 50 epochs, patience 10.

| Column | Meaning |
| --- | --- |
| `mode`, `pred_len`, `seed` | Arm, horizon in hours, seed. |
| `test_mse`, `test_mae` | Test metrics, rounded to 6 decimals. |
| `best_val_mse` | Minimum validation MSE reached. |
| `best_epoch` | 1-based epoch of `best_val_mse`. |
| `num_params` | Total parameters of this arm at this horizon. |
| `batch_size` | Windows per forward pass (8). |
| `accum_steps` | Gradient-accumulation factor (1). |
| `effective_batch` | `batch_size * accum_steps`. |
| `steps_per_epoch` | Forward/backward passes per epoch, `ceil(n_train / batch_size)`. |
| `effective_steps_per_epoch` | Optimiser updates per epoch, `ceil(steps_per_epoch / accum_steps)`. |
| `arch_version` | Integer architecture stamp (2), incremented only when a model change invalidates earlier checkpoints; gates resume from attached prior-session outputs. |

There is no updates-to-best column; it is `best_epoch * effective_steps_per_epoch`.

## equiv_summary.csv (12 rows)

Writer and reader: `src/analysis/analyze_equiv_table.py`, one row per row of
the paper's practical-equivalence table. Statistics from
`src/analysis/paired_stats.py`.

| Column | Meaning |
| --- | --- |
| `family`, `cell`, `selection_rule` | The table's first three columns, as LaTeX labels. Rows under different selection rules are never pooled. |
| `source_csv` | The `results/` file the row was recomputed from. |
| `value_column` | The metric column read from it: `test_mse`, or `test_upd_global` for the overtrain-summary row. |
| `contrast` | `CD-CI` or `CD_Head-CI`; the first arm is the minuend. |
| `cd_minus_ci` | Mean paired difference over (cell, seed) units, rounded to 4 decimals. |
| `ci_lo`, `ci_hi` | 95% paired-t interval on that mean. |
| `rel_pct` | `100 * cd_minus_ci / mean(base arm)`, rounded to 2 decimals. |
| `within_threshold` | True when `abs(rel_pct) <= 1.0`, the pre-specified equivalence band. |
| `n` | Independent units in the interval: the number of seeds (5), which for the AR(1) grand-mean row means seed clusters rather than the 45 pooled rows. |

## theoretical_bounds.csv (62 rows)

Writer and reader: `src/analysis/derive_theoretical_bounds.py`, which reads
no data and compares its closed-form output against the committed file.

| Column | Meaning |
| --- | --- |
| `family` | `AR(1) grid`, `block-covariance`, `boundary sweep` or `leader-follower`. |
| `cell` | Cell label; format varies by family (`C=7, rho=0.1`; `C=21, rho_in=0.5`; `all P, all gamma`; `gamma=0.6, follower`). |
| `h` | Forecast horizon in timesteps; 96 for the exact rows, a sweep for the follower rows. |
| `bound_type` | `CI=CD (exact)`, `CD (sandwich upper=1.0)` or `CI (own-history optimal)`. |
| `bound` | Population-optimal h-step forecast MSE in stationary-variance units. For diagonal-transition families it is `1 - phi^(2h)`. The CD row is the full-information ceiling from the h-step innovation covariance; the CI row is the steady-state Kalman own-history ceiling. |
| `sandwich_width` | On CD rows only: `max(0, 1 - bound)`, the width of the interval between the no-signal ceiling and the CD ceiling. It bounds the CI minus CD gap from above and does not estimate it. |

## vram_bound.csv (2 rows)

Writer and reader: `src/analysis/derive_vram_bound.py`, which reads no data.

| Column | Meaning |
| --- | --- |
| `batch` | Batch size (8 or 128). |
| `tokens_C_N` | Attention sequence length `C * N` for ECL: 321 variates x 11 patches. |
| `num_heads`, `num_layers` | 16 and 3. |
| `bound_1layer_GiB` | Bytes of one layer's materialised FP16 score matrix, `batch * heads * (C*N)^2 * 2`, in GiB. |
| `bound_alllayers_GiB` | The same for all layers retained for backward. |
| `measured_fused_GiB` | Peak allocated memory measured on a T4 under fused attention at that batch. |
| `measured_sec_per_epoch` | Measured seconds per epoch in the same probe. |
| `measured_fused_OOM` | Whether the probe ran out of memory (False in both rows). |
