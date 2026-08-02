# Revision Artefact Ledger — Paper10341 (reviewer yc7L)

Companion to revision_master_plan_v2.md. One row of truth per artefact
produced for the revision: what it is, where it lives, how it was verified,
and what it has produced. Update this file whenever an artefact changes state.
Date of this version: 02 Aug 2026.

Status vocabulary: COMMITTED (on a named branch at a named commit),
DELIVERED (verified and handed over, not yet in git), SCRATCH (probe or
draft, deliberately outside git), PENDING (specified, not yet built).

Update discipline: any commit that changes an artefact's state updates its
entry in the same commit; every Kaggle result table is pasted into its entry
when it lands; UNVERIFIED markers are resolved by checking, never by
deletion.

## 1. Repo code (branch revision/reviewer-yc7L unless noted)

### src/models_cd_block.py — COMMITTED (a60db7c; AMP fix ebba4f2)
Block-wise cross-variate attention CD variant. Same embedding, encoder
configuration, and shared per-variate head as PatchTST_CD; attention runs
within explicit channel groups, computed per group so the (C*N)^2 score
matrix is never allocated (structural saving, no masks). Partition builders:
leader_follower_groups() (10 leader-follower pairs + isolate singleton;
pair-preserving, so the variant can express the lag-1 coupling) and
contiguous_groups(C, b). Parameter count equals vanilla CD by construction.
History: v1 authored and critiqued (5 findings fixed: warmup timing, OOM
ladder, constructor guard, missing tests, docstring); AMP dtype bug found on
first Kaggle run (fp16 embedding vs fp32 LayerNorm-terminated encoder output
at the group scatter) and fixed by allocating the output buffer from the
encoder's dtype (ebba4f2).
Verification: 10 CPU tests passing; flake8/black clean at 120; dry run
completed on T4 under AMP after the fix.

### src/tests/test_cd_block.py — COMMITTED (ebba4f2)
11 tests: partition helpers and guards, output shape, eval determinism,
cross-group independence, within-group (leader-to-follower) dependence,
parameter parity with vanilla CD, singleton-groups-equal-CI scope,
constructor patching guard, forward channel-mismatch guard, and a
CUDA-gated AMP regression test (skips on CPU with stated reason; executes on
any GPU runner).
Verification: 28 passed + 1 CUDA skip in the branch suite.

### src/tests/test_experiments.py — COMMITTED on main (ee28ceb)
Pre-existing suite, repaired this session: the importorskip("models_cd_head")
gate referenced a module that never existed in this repo, silently skipping
all 8 model tests since consolidation into src/models.py. Gate replaced with
a torch-gate plus plain `import models` (skips only when torch is absent;
hard-fails on real defects); stale monorepo run instruction fixed; dead
importorskip calls removed; unknown-mode ValueError test added.
Verification: 18 passed 0 skipped locally and in the clean-clone
certification.

### src/tests/conftest.py — COMMITTED on main (ee28ceb)
Path shim extended with "" so src/ itself is importable (serves `import
models` and the block tests). Docstring updated to name all three imports.

### src/analysis/validate_granger.py — COMMITTED on main (content fixes
ee28ceb; formatting restore c04d3cb)
Section 2.1 Granger battery. This session: black/flake8 formatting fix;
survival function switched from 1-cdf to scipy.stats.f.sf (removes underflow
at tiny p); TypeError fallback for the scheduled removal of statsmodels'
deprecated verbose kwarg; conditional-group strictness documented in the
status-conventions docstring; stale monorepo run instruction corrected.
History note: a copy-paste transfer mutated the formatting once (lesson:
move code by file download, never by pasted text); restored by running black
locally, which is deterministic.
Verification: output byte-identical to pre-fix baselines at gamma 0.0 and
0.6 (27/27 PASS and 26/27 EXPECTED respectively); p-values reproduce across
platforms digit-for-digit; certification formatting gate green.

### src/analysis/partition_etth1_coupling.py — DELIVERED (commit next)
ETTh1 test-window partition by lagged cross-channel coupling. Pre-registers
the first-difference lag-1 |xcorr| metric as primary and raw as robustness
in the module docstring, with the reasoning and the regime-confound
limitation stated before any error data exists. Writes
results/etth1_coupling_partition.csv (window_start, both metrics,
per-horizon balanced median-split flags).
Results (real ETTh1, paper protocol): diffed mean 0.083 sd 0.018, high/low
ratio 1.44; raw ratio 1.24; Spearman(raw, diffed) = -0.28 (metric choice is
substantive); metric autocorr 0.97 at 24 steps (contiguous-regime caveat);
splits 2647/2646 (H=96) down to 2335/2334 (H=720).
Verification: end-to-end run on canonical ETTh1; flake8/black clean.
Action: commit script + its output CSV to the branch (plan §11, week 1).

## 2. Repo configuration and docs

### pyproject.toml + uv.lock + requirements(.dev).txt — COMMITTED (ee28ceb)
uv migration: pyproject as source of truth (runtime deps; dev group pytest/
black/flake8; package=false; black and pytest config; requires-python cap
">=3.12,<3.13" is UNVERIFIED — the cap was instructed but its addition was
never confirmed; check pyproject.toml line 4 and, if absent, add it and
rerun `uv lock` before the next commit), uv.lock as the exact pin, both requirements files generated by
`uv export` (command recorded in their headers) for the certification
script and uv-less reviewers. History: an uv-init stub pyproject hijacked
the first lock (resolved 1 package, emptied the venv, blanked
requirements.txt); recovered via git restore; lesson encoded: check the
resolve count on every `uv lock`.

### README.md — COMMITTED (ee28ceb)
Rebuilt installation (uv-first with pip fallback), test instructions,
layout including the new packaging files, and the uv-run prefix note for
reproduction commands (critique F1). Certification-facing invariants kept:
size, reproduction table, "Table 8" markers.

### docs/revision_master_plan_v2.md — in working tree (docs/)
The governing plan: principles P1-P7, two-phase communication strategy,
Stage 0 probe spec, run matrix B1-B6 with owners, engineering spec, gates
G0-G3, risks R1-R8, timeline. Commit alongside this ledger.

### docs/revision_artifact_ledger.md — this file
Commit to the branch; update at every artefact state change.

## 3. Scratch and probe artefacts (deliberately outside git)

### dryrun_block_attention.py — SCRATCH (Kaggle dataset block-attn-dryrun)
Dataset version state: re-versioned mid-session so the dataset itself
carries the AMP-fixed models_cd_block.py (matching ebba4f2); the copy cell
asserts on "out = enc.new_empty" before %run as a stale-content tripwire.
Any Stage 0 extension ships as a further dataset version, never as
in-notebook patches.
T4 feasibility probe: batch ladder with OOM descent, warmup epoch timed
separately from the steady-state mean, peak memory, projected hours; mirrors
the paper's training step (AdamW, AMP + GradScaler, clip 1.0). Critiqued
(warmup and OOM-traceback fixes applied).
Results (torch 2.13, T4, AMP; the run that resolved [DECIDE] to option B):
  lf_c21   CD       batch 128   38.1 s/epoch   1.55 GiB   0.37 h/35ep
  lf_c21   CD_Block batch 128    9.2 s/epoch   1.61 GiB   0.09 h/35ep
  ar1_c84  CD       batch 128  512.6 s/epoch   6.13 GiB   4.98 h/35ep
  ar1_c84  CD_Block batch 128   69.1 s/epoch   6.21 GiB   0.67 h/35ep
Consequence 1: CD_Block ablation is cheap; option B live (plan G1 path).
Consequence 2: environment finding — committed CSVs record CD batch 1 at
C=84 and batch 8 at C=21 (hardware-forced under materialised attention);
current fused-SDPA torch holds batch 128 at both. Feeds W4 (cost-section
rewrite, VRAM derivation as materialised-attention bound + caveat) and the
Stage 0 re-measurement.
Pending extension (plan §5): CI mode; lf CD_Block at batch 8; boundary P=4
(stride 2 per inference I1) and P=2 (stride 1, confirmed from main.tex);
ECL configs (C=321, split 15840/5256/5256).

### certify_clean_clone.ps1 — local, gitignored
Clean-clone certification harness (fresh clone of committed HEAD, reviewer
venv, canonical row counts, README invariants, pytest-no-skips,
formatting gate, byte-identical equiv_summary.csv, all analyses run, all
figures regenerated). Latest result on main c04d3cb: 29 pass, 0 warn,
0 fail, exit 0. Known trap fixed by procedure (run from main; it clones the
checked-out branch); pending one-word hardening: clone --branch main.
Extension owed at G2: run the new analyze scripts and check the new CSVs.

## 4. Reviewer-response artefacts (local, outside git)

### response_to_reviewer_yc7L_draft.md — DELIVERED (Adi's downloads)
Point-by-point draft in reviewer order. Open [DECIDE] stubs: Critical 1
resolves to option B once B1 is analysed (G1); Critical 2 carries the
optional mechanistic panel now planned as B5; recommended item 1's numbers
await B4; recommended item 2 pairs with the Critical 1 resolution. Perplexity
copy-edits applied (de-escalated Critical 1, toggle stubs, no pre-written
results); its invented subgroup-outcome sentence explicitly rejected.

### Pending per plan: Phase 1 official comment (writing-only items + ETA);
train_block_attention.ipynb (B1, shared engine per plan §7);
train notebooks/engine for B2-B4, B6; analyze_block_attention.py and the
subgroup analysis script (written before results per P4); B5
instrumentation additions.

## 5. Cross-cutting lessons encoded this session

- Transfer code by file download; a pasted transfer mutated a verified file
  once (formatting + a comment fragment) and cost a certification cycle.
- Kaggle module caching: after patching a working-copy module, evict with
  sys.modules.pop or restart; a stale import wasted a GPU cell. Fixed at the
  source by shipping fixed files in the dataset and asserting on content
  before %run.
- CPU verification cannot reach CUDA autocast dtype policy; the class is
  covered by a CUDA-gated regression test that runs wherever a GPU runs.
- uv lock resolve counts are a tripwire: "Resolved 1 package" means the
  wrong pyproject.
- The certification clones the checked-out branch; certify from main.
