# Engineering Design Rationale: CI vs CD PatchTST Experiments

This document records the reasoning behind every significant engineering and
coding decision made across the synthetic grid, ETTh1, ECL, and leader-follower
notebooks. The sources are the experiment session logs and the canonical result
CSVs under `results/`.

---

## 1. Architecture: CI vs CD Implementation

### Why PatchTST and not iTransformer or Crossformer

PatchTST's CI and CD modes differ in exactly one architectural decision:
whether self-attention operates within each variate's patch sequence
independently or across all variates jointly. Everything else, meaning patch
extraction, linear projection, Transformer encoder, training procedure, and
hyperparameters, is shared. This makes it the cleanest test of the CI/CD
distinction without entangling it with other architectural choices.
iTransformer would confound the comparison with its inverted embedding design,
and Crossformer would confound it with two-stage routing attention.

### Why seq_len=512, patch_size=16, stride=8

These are the PatchTST paper's benchmark configuration values (the paper's
"PatchTST/64" name refers to approximately 64 patches). With no padding applied,
`(512 - 16) // 8 + 1 = 63` patches per variate. The canonical implementation
computes this directly with no padding and produces 63 patches; a padding
convention that yields 62 is not used. The 63-patch no-padding form is the
implementation of record and the one `src/models.py` follows. This discrepancy
was identified during the architecture comparison review and corrected there.

For ECL, seq_len=96 follows the iTransformer paper's protocol, giving
`(96 - 16) // 8 + 1 = 11` patches and `321 x 11 = 3531` CD tokens.

### Why the CI mode reshapes to (B*C, L)

Reshaping to (B*C, L) forces each variate through the same encoder
independently. Weight sharing across variates acts as a free regulariser,
because the model cannot overfit to variate-specific patterns when it is
literally using the same parameters for all of them.

### Why the CD mode concatenates to (B, C*N, D) before the encoder

Concatenating all variates' patch tokens along the sequence dimension before the
encoder is the simplest implementation that allows self-attention to operate
across variates. After encoding, the output is split back into per-variate
representations before the forecast head.

### Why the CD head is Linear(N*D, pred_len) not Linear(C*N*D, pred_len*C)

The original notebook used a global head `Linear(C*N*D, pred_len*C)`, which is
`Linear(84672, 2016)` at C=21. This caused catastrophic failure: a CD MSE of
about 1.30 against a CI MSE of about 0.99, with best_epoch=1 at every gamma and
every seed. The global head was an 84,672 to 2,016 linear layer that could not
train in 50 epochs with batch size 8.

The correct implementation applies a shared `Linear(N*D, pred_len)`, which is
`Linear(4032, 96)`, per variate after splitting the encoder output back into C
separate representations. This is structurally identical to the CI head and
keeps the CD head tractable. The encoder still mixes all C*N tokens jointly,
which is where CD's cross-variate signal comes from, while the projection back
to the prediction horizon is done per-variate. Verified by:

```python
print(PatchTST_CD(512, 96, 21, 16, 8, 64, 8, 3, 0.2).head)
# Must print: Linear(in_features=4032, out_features=96, bias=True)
```

### Why d_model=64, 8 heads, 3 layers for synthetic/ETTh1; 128/16/3 for ECL

The synthetic and ETTh1 configuration matches Table 1 of the PatchTST paper for
standard benchmark settings. The paper's Appendix A uses d_model=16 and 4 heads
for small datasets, so the choice of d_model=64 here is a deliberate deviation
that gives the model more capacity. This was verified during the architecture
comparison review, and the attribution was corrected from "matches the paper" to
"this implementation's choice".

ECL uses d_model=128 and 16 heads to give the model more capacity per variate
when processing 321 channels.

---

## 2. Training Hyperparameters

### Why AdamW with lr=1e-4, weight_decay=1e-4

AdamW decouples weight decay from the gradient update, unlike Adam, which applies
decay scaled by the adaptive learning rate. This is the Loshchilov and Hutter
(2017) correction and it matters for Transformers. lr=1e-4 is the PatchTST
paper's value. weight_decay=1e-4 is conservative; the original paper uses higher
values, but these experiments use smaller models.

### Why cosine decay with linear warmup

Warmup prevents large unstable gradient updates at initialisation, when
gradients are large and noisy. Cosine decay produces better final
representations than step decay, since it starts slow, decays steeply in the
middle, and flattens near the end, which provides a natural fine-tuning phase in
the final epochs.

### Why warmup_epochs=10 for both CI and CD in the leader-follower experiment

Matching warmup for both modes was an explicit design decision to avoid
repeating the ETTh1 confound. In the ETTh1 runs, CD had warmup_epochs=2 and CI
had warmup_epochs=10. That was a deliberate but poorly motivated choice: the
intent was to prevent CD over-training, but the effect was that CD early-stopped
before its warmup ended. The leader-follower experiment corrects this by giving
both modes identical warmup.

### Why max_epochs=50, patience=10

A 50-epoch cap holds compute at a predictable ceiling consistent with the
12-hour Kaggle session limit (ETTh1 uses 100). patience=10 is long enough to
ride through temporary plateaus without waiting through an entire run for a model
that has already converged.

### Why gradient clipping at max_norm=1.0

Transformers are susceptible to exploding gradients early in training. Clipping
the L2 norm of the full gradient vector at 1.0 prevents catastrophic weight
updates without killing learning.

### Why DLinear was added as a third mode

It was added after hostile review identified the lack of a lower bound as a
weakness. If either PatchTST mode underperforms a linear map from lookback to
horizon, the Transformer is not learning useful temporal structure and the CI/CD
comparison is meaningless. DLinear is a channel-independent
`Linear(seq_len, pred_len)` with shared weights, the minimum sanity check.

### Why 5 seeds {42, 123, 456, 789, 1011}

Single-seed running was the most critical weakness flagged during the first
hostile review of the experiment design: with one seed, variance is invisible
and any cell's CI/CD ordering could flip. Three seeds was the initial minimum
added under the compute budget, and the final runs extend to five so that the
per-cell confidence intervals rest on four degrees of freedom rather than two.

---

## 3. Data Generation

### Why AR(1) with diagonal transition and compound-symmetry covariance

The diagonal transition matrix phi*I gives exact Granger non-causality across all
channel pairs, provably zero from the model definition rather than approximately
zero. Compound-symmetry covariance gives all off-diagonal entries the same value
rho, so a single scalar controls instantaneous correlation across all pairs at
once. This is the simplest family that isolates correlation as a single variable
while holding dynamical coupling at zero.

### Why phi=0.8

Moderate autoregressive persistence, high enough to produce meaningful temporal
structure but below the near-unit-root region (phi >= 0.95) that would cause very
slow mixing and make the burn-in insufficient.

### Why C in {7, 21, 84}

C=7 matches ETTh1 (the real-data anchor). C=21 is intermediate. C=84 is twelve
times ETTh1, large enough to stress the CD attention mechanism and create the
step-count confound at batch=1. These three values span low, medium, and high
dimensionality without being computationally prohibitive on the T4.

### Why rho in {0.1, 0.5, 0.9}

Low, medium, and high instantaneous correlation. rho=0.9 is near-maximal for the
compound-symmetry family to remain positive definite at C=84
(`rho > -1/(C-1) = -1/83 ~ -0.012`). If CD does not win at rho=0.9, it will not
win on instantaneous correlation at any rho.

### Why 14,400 timesteps with 60/20/20 split

14,400 gives 8,640 train / 2,880 val / 2,880 test, matching ETTh1's standard
split exactly. This makes the synthetic and real datasets directly comparable in
training data volume. The 60/20/20 split follows the iTransformer paper's ECL
protocol and is the most common convention in the long-term forecasting
literature.

### Why 1,000 burn-in steps

AR(1) with phi=0.8 mixes fast enough that 1,000 steps provides a comfortable
margin to ensure the process is in its stationary distribution before recording.
The stationarity condition is `|phi| < 1`, satisfied by phi=0.8.

### Why z-score normalisation fit on the training set only

Standard protocol: any normalisation using val or test statistics constitutes
data leakage. Per-channel z-score is the convention used in the original PatchTST
paper and throughout the long-term forecasting literature.

### Why empirical rho is measured and logged per run

A direct response to the critique that the paper should not claim rho=0.9
produces a given correlation without measuring it. The empirical Pearson
correlation of the training split is computed at dataset construction time and
stored as `empirical_rho` in every result row, confirming the data-generating
process behaved as intended.

---

## 4. Leader-Follower VAR(1) Design

### Why leader-follower topology, not a banded chain

A banded chain (each channel coupled to neighbours plus or minus one) is an
arbitrary topology with no scientific motivation for forecasting, and its
coupling mechanism is unclear. The leader-follower structure is directly
motivated by LIFT's empirical finding that lead-lag relationships are the
operative mechanism for CD advantage. When CD wins on leader-follower data there
is a clear mechanistic explanation: CD can attend from follower tokens to leader
tokens across the C*N token sequence, whereas CI cannot recover a leader's past
from a follower's history alone.

### Why 10 leaders, 10 followers, 1 isolate (C=21 total)

The one-to-one leader-follower mapping is the simplest clean design that produces
an unambiguous structural coupling. Channel 20 (the isolate) is a pure AR(1)
variate with no coupling to anything; it serves as a within-experiment negative
control and is used in the conditional Granger tests to verify that spurious
bivariate significance, arising from shared innovation correlation, disappears
once the true cause is conditioned on.

### Why the process is stable for all gamma

A(gamma) is lower-triangular by index ordering: leaders occupy indices 0 to 9
and followers 10 to 19. Lower-triangular matrices have their diagonal entries as
eigenvalues. All diagonal entries equal phi=0.8, so the spectral radius is 0.8
for every finite gamma and no stability constraint on gamma is needed. The
analytic argument covers the full sweep, and the swept values were confirmed
computationally.

### Why gamma in {0.0, 0.3, 0.6, 0.9}

gamma=0.0 reduces the process to the original AR(1) grid at C=21, rho=0.5 and
serves as a cross-experiment sanity check. gamma=0.3 is moderate coupling,
gamma=0.6 is strong, and gamma=0.9 extends the sweep to near the top of the
stable range so any trend in the gap has room to show. With three modes and five
seeds this gives a 60-run grid.

### Why rho=0.5 fixed (not swept)

This experiment isolates gamma as the single new variable; rho was already swept
in the AR(1) grid. Adding a rho sweep here would multiply the run count without
addressing a new question. rho=0.5, the midpoint of the grid range, was chosen
deliberately.

### Why Granger validation uses bivariate tests for designed pairs but conditional tests for isolate pairs

Bivariate Granger tests of (isolate to follower) produce spurious significance
because the isolate shares contemporaneous innovation correlation (rho=0.5) with
leaders, which genuinely cause followers. This is omitted-variable confounding
rather than a generator defect. The conditional F-test includes the true leader
as a control variable, and the isolate's p-value then rises to about 0.22 (not
significant), confirming the spurious bivariate result was entirely attributable
to the omitted leader. In the validation run, all 27 tests passed at gamma=0.0
and gamma=0.3; at gamma=0.6 one false positive appeared (follower 17 to leader 7,
p=0.047, seed 42 only), and seeds 123 and 456 gave p=0.99 and p=0.70 for the same
pair, confirming a seed-specific false positive within the expected rate for ten
tests at alpha=0.05.

---

## 5. Memory and Hardware Engineering

### Why FP16 mixed precision (autocast + GradScaler)

Added to the ECL notebook after an out-of-memory failure at batch_size_cd=2. FP16
halves the attention matrix memory: at B=2, H=16, and 3531 tokens per sequence,
the attention matrix per layer drops from about 3.2 GB to about 1.6 GB. The
GradScaler prevents FP16 underflow in the backward pass. `scaler.unscale_()` must
be called before gradient clipping, because clipping on scaled gradients produces
wrong norms.

The correct import (PyTorch 2.x):
```python
from torch.amp import GradScaler, autocast
scaler = GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))
with autocast('cuda', enabled=(DEVICE.type == 'cuda')):
```

The deprecated form `from torch.cuda.amp import GradScaler, autocast` produces a
FutureWarning on PyTorch 2.x and should not be used.

### Why PYTORCH_ALLOC_CONF is set to expandable_segments before all imports

The out-of-memory error message itself recommended this. After several long
training runs the allocator holds many small reserved-but-unallocated blocks that
cannot be reused for a large contiguous request. Expandable segments let the
allocator grow and shrink segments dynamically. It must be set before any CUDA
allocation, so it is the very first line in Cell 1, ahead of all imports.

### Why del model, optimizer plus torch.cuda.empty_cache() between runs

After a full training run, PyTorch's CUDA allocator holds reserved memory even
after tensors are garbage-collected. Explicitly deleting the model and optimizer
then calling `empty_cache()` releases this back to the device, preventing an
out-of-memory failure on the next run from accumulated fragmentation. Added after
the first such crash during the multi-run grid notebook.

### Why gc.collect() between runs (ECL/grid notebooks)

Python's garbage collector does not run immediately. `gc.collect()` forces
collection before `empty_cache()`, ensuring Python-level references are dropped
before the CUDA memory release.

### Why batch sizes are dicts keyed by C, not scalars

Originally batch sizes were scalars (CI=128, CD=32). This caused an
out-of-memory failure at C=84, where CI at batch=128 processes B*C = 128*84 =
10,752 sequences at once, requiring a large contiguous allocation that failed
after fragmentation from earlier runs. Batch sizes are now per-C dicts:
CI={7:128, 21:128, 84:32}, CD={7:64, 21:8, 84:1}.

### Why CD batch=1 at C=84

The T4 16GB memory constraint. CD at C=84 with L=512 produces 84*63 = 5,292
tokens per sequence, and at batch=2 the attention matrix exceeds available memory
after fragmentation, so batch=1 is the only option that fits. This creates the
31.9x step-count confound between CI and CD at C=84 that the paper discloses.

### Why TF32 blocks were removed

TF32 (TensorFloat-32) is an Ampere GPU feature (sm_80 and above). Kaggle T4s are
Turing architecture (sm_75). The `torch.backends.cuda.matmul.allow_tf32` and
`torch.backends.cudnn.allow_tf32` lines had no effect on the T4 and were dead
code, removed after the hostile tech-lead review.

### Why @torch.inference_mode() instead of @torch.no_grad() on evaluate

`inference_mode` is strictly stronger, disabling both gradient tracking and
version-counter updates on tensors, which is the correct choice for pure
evaluation where no gradient-related operation will follow. The grid notebook
originally used `@torch.no_grad()` and was corrected to
`@torch.inference_mode()` during the hostile code review.

### Why persistent_workers=True and num_workers=2 in DataLoaders

`persistent_workers=True` prevents worker processes from being respawned every
epoch, since each spawn and kill cycle adds overhead; with `num_workers > 0` and
`persistent_workers=True` the workers stay alive between epochs, the recommended
setting per the PyTorch docs. `num_workers` is 2 rather than 4 because T4 Kaggle
environments have shown instability with higher worker counts at the batch sizes
used here.

---

## 6. Resumability and Fault Tolerance

### Why save to CSV after every completed run

Kaggle sessions expire after 12 hours (43,200 seconds). A single CD run at C=84
took about 1,530 seconds (roughly 25 minutes). Saving after every completed run
means an interrupted session loses at most one in-progress run. The pattern is
`pd.DataFrame(results).to_csv(OUT_PATH, index=False)` called immediately after
`results.append(row)`.

### Why the resume key is (gamma, mode, seed) for leader-follower

These three values uniquely identify a run in the 60-run grid. The resume set is
built at startup from the existing CSV. A float-comparison trap (rho read from
CSV as float64 not matching Python float keys) was encountered in the original
AR(1) grid notebook and fixed there with `round(rho, 8)`. The leader-follower
notebook avoids it by leaving rho out of the key, since rho is fixed at 0.5.

### Why OUT_PATH.stat().st_size > 100 guards the CSV load

`pd.read_csv()` raises `EmptyDataError` on a zero-byte or near-empty file. This
was the root cause of repeated crashes in the first leader-follower training
session, where the CSV was created when `OUT_PATH` was defined but no rows had
been written. The size guard prevents the crash and falls through to the `else`
branch that initialises an empty results list.

---

## 7. Statistical Analysis Design

### Why t-distribution CIs with df = n_seeds - 1 = 4

With five seeds, the t-critical value at df=4 is 2.776. The resulting intervals
are deliberately conservative for a five-sample estimate. The claim they support
is modest: even at this width the intervals contain only negligible values. A
normal approximation at this sample size would be unjustified.

### Why C is categorical (not numeric) in the linear model

C takes values {7, 21, 84}, which are not equidistant (gaps of 14 and 63). A
linear-in-C assumption would impose an unjustified dose-response relationship. In
patsy, converting C to string dtype treats it as a categorical factor
automatically. Using the patsy `C()` operator clashes with the column named "C",
so it was avoided in favour of renaming to `n_ch` with string dtype. The encoding
is not cosmetic: treating C as categorical rather than numeric changes the fitted
R-squared, because the numeric form forces a straight-line dose response the data
does not follow.

### Why the mode-by-gamma interaction coefficient sign matters

In the linear model `MSE ~ mode_cd + gamma + mode_cd*gamma`, a negative
mode_cd-by-gamma coefficient means CD's relative performance improves as gamma
rises, with CD's MSE growing more slowly than CI's as coupling is added. A
positive coefficient means CD gets relatively worse. The sign is confirmed from
the actual results rather than assumed.

---

## 8. Notebook Engineering Conventions

### Why GPU training scripts are .ipynb not .py

A standing instruction established after a `train.py` was incorrectly produced
for a GPU task: any script whose primary job is training a neural network on GPU
is delivered as a `.ipynb` notebook from the start, with no intermediate `.py`
step. CPU scripts (analysis, data loading, plotting, model definitions, unit
tests) remain as `.py`.

### Why data paths use Path(__file__).resolve().parents[N] / "data"

This avoids relative-path fragility. A script run from any working directory
resolves to the same absolute path, whereas relative `"./data"` breaks whenever
the working directory is not the script's directory, which is the default in
PyCharm and varies in Kaggle.

### Why generator code is pasted into Kaggle notebook Cell 2, not uploaded as a dataset

Uploading as a Kaggle dataset requires attaching the dataset to the notebook,
adding the correct input path to `sys.path`, and updating that path whenever the
dataset is renamed. Pasting the full generator code into a notebook cell is
simpler, needs no external dependency, and is fully self-contained. The `main()`
function and `argparse` block are kept, but the call to `main()` is removed,
because it would trigger `SystemExit` in a Jupyter kernel, where `sys.argv`
contains kernel launcher arguments that argparse does not recognise.

### Why the argparse import is at module level, not inside main()

Imports inside functions hide dependencies, add per-call overhead, and are not
expected by readers of the module. `argparse` was initially placed inside
`main()` in the generator script and was moved to the top-level imports during
the hostile code review.

---

## 9. Known Confounds and Disclosed Limitations

### Step-count confound at C=84

At C=84, CD uses batch_size=1, giving 7,433 gradient updates per epoch against
CI's 233 updates at batch_size=32, which is 31.9 times more updates per epoch for
CD. The confound is hardware-forced and cannot be eliminated without gradient
accumulation, which reduces update frequency but not forward-pass count and so
does not reduce wall-clock time per epoch. `steps_per_epoch` and `total_steps`
are logged in every result row, and the paper discloses this explicitly. The
matched-compute control in the paper's robustness section caps the update budget
across modes and shows the gap does not survive the cap.

### ETTh1 warmup asymmetry

The early ETTh1 runs used warmup_epochs=2 for CD and warmup_epochs=10 for CI, a
deliberate choice motivated by the belief that CD, with more steps per epoch than
CI, needed a shorter warmup to avoid over-training. The effect was the opposite:
CD early-stopped at epoch 1 to 3, before its warmup completed. The paper
describes this as a deliberate but poorly motivated choice rather than a
configuration error, because it was intentional, and the matched-budget ETTh1
protocol replaces it with identical warmup for both modes.

### ECL CD infeasibility

At C=321 and seq_len=96, the CD encoder processes 3,531 tokens per sequence. Each
training epoch takes about 5,000 seconds on the T4, which exceeds the
43,200-second Kaggle session limit. Gradient accumulation does not help, because
it reduces update frequency but not the forward-pass computation time per epoch.
ECL CD is declared computationally infeasible and is not reported.

### C=84, rho=0.9 CD early overfitting

CD at C=84, rho=0.9 with batch=1 produces 7,433 gradient updates per epoch. The
three seeds examined stop at best_epoch 1 or 2 (7,433 / 14,866 / 7,433 total
steps), with the model overfitting in the first epoch under the extreme step
count. This is a genuine finding to report rather than a bug to hide: the paper
discloses the total step counts and notes that CD used up to 32 times more
compute than CI yet reached essentially the same test MSE.

---

## 10. Canonical Results and Verification

This document records engineering rationale and does not reproduce the
experimental results, which live in two authoritative places: the canonical CSVs
under `results/` and the tables in `paper/main.tex`. Holding a second copy of the
numbers here would only create a surface for the two to drift apart, which is the
failure mode the verification discipline exists to prevent.

The discipline is absolute: no number enters the paper until it reproduces from a
canonical CSV. Each analysis script under `src/analysis/` reads one canonical CSV
and emits the statistics and figures for its part of the paper, so the script
output and the paper text are checked against each other before anything is
written. The synthetic grid is `results_grid.csv` (five seeds, nine
cells, three modes), the leader-follower sweep is `results_leader_follower.csv`,
the matched-budget ETTh1 runs are in the ETTh1 results CSV, the boundary sweep
has its own CSV, and the ECL CI-only runs are in the ECL results CSV. The three
fair-protocol robustness controls (cross-variate head, matched compute, and
block covariance) each have their own canonical CSV and analysis script.

As the single headline anchor: on the AR(1) grid the grand-mean CD minus CI
difference is +0.0013 MSE, with a 95% confidence interval of [-0.0002, +0.0028]
over the five seeds, and every cell ratio falls inside [0.9984, 1.0056]. Both
figures reproduce directly from `results_grid.csv`.
