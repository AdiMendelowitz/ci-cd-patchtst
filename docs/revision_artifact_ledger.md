# Revision Artefact Ledger — Paper10341 (reviewer yc7L)

Companion to revision_master_plan_v2.md. One row of truth per artefact
produced for the revision: what it is, where it lives, how it was verified,
and what it has produced. Update this file whenever an artefact changes state.
Date of this version: 03 Aug 2026 (B1 complete; G1 evidence in hand).

Status vocabulary: COMMITTED (on a named branch at a named commit),
DELIVERED (verified and handed over, not yet in git), SCRATCH (probe or
draft, deliberately outside git), LOCAL (maintained on Adi's machine only,
by decision, never committed), PENDING (specified, not yet built).

Docs policy (decided 03 Aug): the four files under docs/ (this ledger, the
plan, scripts_provenance, design_rationale) are LOCAL from this version
onward — Adi's eyes only, never added to git. The v2.1 plan and ledger
already inside 8451b5d become frozen snapshots; this local copy is
canonical over them. Consequence for G3: decide docs/ handling at the
mirror cut (the stale committed snapshots either ship scrubbed or the
docs/ directory is excluded from the mirror).

Update discipline: update the changed artefact's entry in this file at the
time of the change (local edit; commits no longer carry the ledger); every
Kaggle result table is pasted into its entry when it lands; UNVERIFIED
markers are resolved by checking, never by deletion.

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

### src/analysis/partition_etth1_coupling.py — COMMITTED (8451b5d)
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
Verification: end-to-end run on canonical ETTh1; flake8/black clean; oracle
self-check PASS lines confirm the committed copy is the verified version.

## 2. Repo configuration and docs

### pyproject.toml + uv.lock + requirements(.dev).txt — COMMITTED (ee28ceb)
uv migration: pyproject as source of truth (runtime deps; dev group pytest/
black/flake8; package=false; black and pytest config; requires-python cap
">=3.12,<3.13" VERIFIED 03 Aug by `git show revision/reviewer-yc7L:pyproject.toml`),
uv.lock as the exact pin, both requirements files generated by
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

### docs/revision_master_plan_v2.md — LOCAL (canonical); v2.1 snapshot
COMMITTED at 8451b5d (confirmed inside the tip by `git show`, 03 Aug) and
now frozen/stale by the docs policy above
The governing plan: principles P1-P7, two-phase communication strategy,
Stage 0 probe spec, run matrix B1-B6 with owners, engineering spec, gates
G0-G3, risks R1-R8, timeline. v2.2 amendment pass 03 Aug (B1 results, I1
and V1 resolutions, Stage 0 status).

### docs/revision_artifact_ledger.md — this file — LOCAL (canonical); v2
snapshot COMMITTED at 8451b5d, now frozen/stale by the docs policy above
Update at every artefact state change (local edit).

### notebooks/train_block_attention.ipynb — COMMITTED (777c462, pushed to
origin; confirmed by git log 03 Aug; staged set was verified pre-commit to
be exactly this file plus the two CSVs below)
B1 three-arm leader-follower sweep (CI/CD/CD_Block, gamma {0,.3,.6,.9} x 5
seeds, committed lf protocol; CD-family arms at committed batch 8 per P3's
B1 exception; DLinear dropped per plan REJECTED list). Models imported from
Kaggle dataset b1-block-attn with tripwire asserts (file set, AMP-fix
marker, build_model); module SHA256s printed at run time match the branch
copies (1eecbddb…, 8703aeed…). Hosts B5 (per-epoch pre-clip grad-norm mean
on every run; participation ratio of encoder-output feature covariance on a
fixed 32-window validation probe at gamma=0.6). Declared engineering-spec
deviation: run-level rather than per-epoch resume, justified in the header.
History: original session's notebook lost (never committed, not in
Downloads); rebuilt 03 Aug from the recovered spec, /critique'd (6 findings
fixed, incl. SEV2 diag-before-results save order), CPU-smoke-tested incl.
resume and B5 hook; one launch-time delta by Adi (hard-coded INPUT_ROOT
containing the Kaggle username — G3 SWEEP ITEM); full cell diff against the
session-verified file confirmed that line is the only difference.
Committed file is the as-ran Kaggle copy (provenance-correct).
Lesson encoded: marker-grep verifies presence, only a full diff verifies
identity.

### results/Revision/train_block_attention/results_block_attention.csv —
COMMITTED (777c462, same commit as the notebook)
60/60 runs, 0 duplicates, schema identical to results_leader_follower.csv.
Run on Kaggle T4 (commit-mode run; version URL to be added to this entry).
Per-run wall clock: CI ~340 s (bs 128), CD ~820 s (bs 8), CD_Block ~340 s
(bs 8); total ~8.4 h. Result table (mean test MSE over 5 seeds):
  gamma    CI      CD      CD_Block  CD/CI   Blk/CI  Blk/CD
  0.0    1.0270  1.0287  1.0273    1.0016  1.0003  0.9986
  0.3    1.0326  1.0393  1.0372    1.0065  1.0045  0.9980
  0.6    1.0298  1.0377  1.0353    1.0077  1.0054  0.9977
  0.9    1.0278  1.0362  1.0335    1.0082  1.0056  0.9974
Paired over seeds: CD-CI positive on 5/5 seeds at every gamma>0 (exact
one-sided sign p=1/32 per cell); CD_Block-CD negative on 4/5 seeds;
CD_Block-CI mixed. gamma=0 internal control passed (three arms within
0.2%). best_epoch: CD family 4-7, CI 17-31 (committed fingerprint
reproduced). G1 evidence: block attention narrows but does not close the
CD penalty; wording freeze awaits analyze_block_attention.py + oracle (P4).

### results/Revision/train_block_attention/diag_b5_gamma06.csv —
COMMITTED (777c462, same commit)
1248 per-epoch rows (grad norms all 60 runs; PR non-null on the 15
gamma=0.6 runs, 301 rows). 216 nonfinite grad steps = AMP scaler warmup,
excluded from means by construction. PR first->last epoch means: CI
4.2->2.6, CD 3.8->4.4, CD_Block 3.9->8.9. Final-epoch grad-norm means grow
with gamma for CD (0.73->1.23, clipping active) vs CI ~0.87. B5 wording
ceiling applies: "consistent with hypothesis", never mechanism.

### Decision: results path convention — results/Revision/<experiment>/
for all revision-produced CSVs (set by the B1 commit). Propagates to the
analyze scripts, certify_clean_clone.ps1 extension, and README at G2.

### docs/scripts_provenance.md — LOCAL (docs policy; the earlier same-day
commit instruction was never executed — git log 03 Aug shows no such
commit — so nothing to remove from the branch)
Pre-revision lineage map: sandbox notebook/CSV/table provenance for every
paper artefact, written in the ml-research-12weeks working tree; paths are
relative to that archive. Names Kaggle accounts throughout — must never
enter the repo or mirror un-scrubbed.

### docs/design_rationale.md — LOCAL (docs policy)
Paper-era engineering rationale (architecture, hyperparameters, data
generation, LF design). Scope note added 03 Aug pointing revision-era
decisions to the plan, this ledger, and module docstrings.

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
Extension status 03 Aug: shipped as a SIBLING SCRIPT (dryrun_stage0.py,
below), not as an edit to this committed file — deliberate deviation from
plan §5's "extend" wording so the committed probe stays untouched.

### dryrun_stage0.py — DELIVERED 03 Aug (chat outputs; not yet uploaded)
Stage 0 probe extension: CI mode; lf CD/CD_Block at batch 8; boundary
P=4/S=2 and P=2/S=1 for CD and CI at ladder + protocol batch; ecl_cd and
ecl_cd_block (C=321, contiguous groups of 3). Committed boundary/lf step
counts pinned; projection epochs per family from committed best_epoch +
patience (lf CD 17, lf CI 41, boundary CD 22, boundary CI 50, ECL 50
worst-case), each labelled with provenance; timed-step cap (60) keeps huge
cells inside one session. /critique'd (2 SEV2 fixed: duplicate-measurement
skip, step cap), CPU-smoke-tested incl. forced-OOM paths. NOTE: the lf
batch-8 cells are now cross-checks only — B1's live run measured them on
real data (CD ~820 s/run, CD_Block ~340 s/run). Remaining G0 value:
boundary and ECL pricing. Action: upload as next block-attn-dryrun dataset
version; run on T4; paste G0 summary into this entry.

### certify_clean_clone.ps1 — local, gitignored
Clean-clone certification harness (fresh clone of committed HEAD, reviewer
venv, canonical row counts, README invariants, pytest-no-skips,
formatting gate, byte-identical equiv_summary.csv, all analyses run, all
figures regenerated). Latest result on main c04d3cb: 29 pass, 0 warn,
0 fail, exit 0. Known trap fixed by procedure (run from main; it clones the
checked-out branch); pending one-word hardening: clone --branch main.
Extension owed at G2: run the new analyze scripts and check the new CSVs.

## 4. Reviewer-response artefacts (local, outside git)

### Phase 1 official comment — POSTED 02 Aug 2026, 19:46
Title "Author response: Resolved items and experiments underway"; 4711
chars; reviewer order; ETA 3-4 weeks stated. Single reviewer confirmed on
the form's reader list at posting (V1 evidence). Source file
phase1_comment_FINAL.md. The "now running" statement was posted while the
B1 launch was believed underway but unconfirmed; the launch was confirmed
03 Aug and the sweep completed the same day, so the claim is discharged.

### response_to_reviewer_yc7L_draft.md — SUPERSEDED by v2; local copy gone
### response_to_reviewer_yc7L_v2.md — LOST LOCALLY, RECOVERABLE
205 lines, operator header, 10 PENDING result slots (Phase 2 response).
Not committed, no longer in Downloads or anywhere under the user profile.
Full text survives in the project chat "Paper review feedback" (03 Aug
session located it); recovery action owed: rebuild from that chat and store
locally with a second copy (docs policy keeps it out of git; a backup
outside this machine is the substitute) — it must not remain chat-only.

### Pending per plan: train notebooks/engine for B2-B4, B6;
analyze_block_attention.py (NEXT — freezes G1 wording) and the subgroup
analysis script (written before results per P4); B4/B6 instrumentation.

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
- Marker-grep verifies presence, only a full cell/file diff verifies
  identity: a grep for the rglob fix passed a staged notebook that differed
  in another line (03 Aug; second instance of the class).
- Commit-instructed is not committed: mark every user-executed git action
  UNVERIFIED until git log/status output confirms it (recurring; third
  occurrence class; the B1 commit resolved cleanly to 777c462 the same day,
  and the scripts_provenance instruction turned out NOT executed — the
  marker earned its keep in both directions).
- Shell dialect: PowerShell cannot run bash heredocs; ship .py files or
  PS-native blocks, never `python - <<EOF` (03 Aug).
- Kaggle mount layout changed to /kaggle/input/datasets/<user>/<dataset>/;
  discover module files by rglob, never fixed-depth glob (03 Aug).