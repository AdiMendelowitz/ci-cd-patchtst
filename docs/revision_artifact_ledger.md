# Revision Artefact Ledger — Paper10341 (reviewer yc7L)

Companion to revision_master_plan_v2.md. One row of truth per artefact
produced for the revision: what it is, where it lives, how it was verified,
and what it has produced. Update this file whenever an artefact changes state.
Version: v4, created 09 Aug 2026 (/critique pass on v3), supersedes v3;
content below extends through 09 Aug 2026 (section 10). Fingerprint of this
file is recorded in its own entry under section 2 as of the v4 creation
date; re-fingerprint at the next version bump. Quote the recorded
fingerprint when handing the file over, so a second copy can never be
mistaken for the canonical one again.
SELF-CHECK GAP FOUND AND FIXED 09 Aug: v3's own header made this same claim
("fingerprint...recorded in its own entry") but section 2's entry only ever
carried v2's baseline fingerprint — v3's own line/byte/sha was never
recorded, despite the instruction to "record...here after the first save."
The gap is exactly the failure mode this mechanism exists to catch (see the
05 Aug duplicate-copy incident below); v3's true baseline fingerprint is
therefore unrecoverable and is not backfilled here. v4's fingerprint is
recorded below at first save, correcting the gap going forward.
Covers: B1 complete; G1 executed; G0 fully closed; boundary notebooks
retrieved and committed; B2 gamma-0.9 tranche complete across accounts
(results_boundary_p4_complete.csv, O-a resolved); B2 gamma-0 tranche
complete across accounts (5/5 CD + 5/5 CI seeds; section 9); P=2 boundary
compute-infeasibility probe decided; response v5 fingerprint-verified;
main.tex protocol disclosure applied; two comment/terminology/tone sweeps
across all working-tree .py/.ipynb files delivered, uncommitted (section 10).

Status vocabulary: COMMITTED (on a named branch at a named commit),
DELIVERED (verified and handed over, not yet in git), SCRATCH (probe or
draft, deliberately outside git), LOCAL (maintained on Adi's machine only,
by decision, never committed), PENDING (specified, not yet built),
SUPERSEDED (replaced by a named later artefact; retained for audit),
WITHDRAWN (an instruction or decision revoked before or after execution;
records what was revoked and why, never deleted).

Docs policy (decided 03 Aug): the four files under docs/ (this ledger, the
plan, scripts_provenance, design_rationale) are LOCAL from this version
onward — Adi's eyes only, never added to git. The v2.1 plan and ledger
already inside 8451b5d become frozen snapshots; this local copy is
canonical over them. Consequence for G3: decide docs/ handling at the
mirror cut (the stale committed snapshots either ship scrubbed or the
docs/ directory is excluded from the mirror).
RISK, unresolved: no off-machine backup is recorded for these four files,
unlike response_to_reviewer_yc7L_*.md (which keeps a second copy in
project knowledge). Loss of the local machine loses local-canonical
revision history for this ledger, the plan, scripts_provenance, and
design_rationale beyond whatever a commit or handoff last captured.

Update discipline: update the changed artefact's entry in this file at the
time of the change (local edit; commits no longer carry the ledger); every
Kaggle result table is pasted into its entry when it lands; UNVERIFIED
markers are resolved by checking, never by deletion.

Duplicate-copy resolution (05 Aug 2026). Two copies existed in docs/:
revision_artifact_ledger.md (27,739 B, sha256 3da29d58…) and
revision_artifact_ledger_v2.md (27,748 B, sha256 9adc30ad…), identical apart
from one line in section 6. The `_v2` copy named the committed record
`notebooks/original/RETRIEVAL.md`; the 04 Aug handoff records the committed
file as `notebooks/original/README.md` (61 lines, "hashes in folder README
are of committed copies"), so the `_v2` line was wrong and that copy is
WITHDRAWN. This file is canonical; delete `revision_artifact_ledger_v2.md`.
Both copies carried section 6, which resolves the standing UNVERIFIED item
"ledger.v2 copied over local canonicals" as DONE. CONFIRMED 05 Aug by
inspecting notebooks/original/README.md directly — the file exists under
that name and contains the roles table, protocol section, and hash table.
Working-tree presence is not proof of tracking; `git ls-files
notebooks/original/` still owed before the mirror cut.

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
and V1 resolutions, Stage 0 status). v2.3 amendment pass 03 Aug (G0
measured costs; B2/B3/B6 trigger applications; G1 status; pre-adoption
/critique, 7 findings fixed).

### docs/revision_artifact_ledger.md — this file — LOCAL (canonical); v2
snapshot COMMITTED at 8451b5d, now frozen/stale by the docs policy above
Update at every artefact state change (local edit). This entry carries the
file's own fingerprint so that two local copies can never again coexist
undetected (the failure that produced the 05 Aug duplicate; see the header).
Base for v3: v2 at 459 lines / 27,739 bytes, sha256 3da29d58…. v3's own
line/byte/sha was never recorded here despite the standing instruction to
do so after first save — found and fixed 09 Aug during a /critique pass;
v3's true baseline is unrecoverable, so this entry now starts a clean
record from v4 rather than backfilling a guess.
Base for v4 (as delivered by the 09 Aug critique pass, immediately before
this fingerprint line was written): 1,114 lines / 71,365 bytes, sha256
962bf3b3d6318cd85ad9827a0d50309b2da9c6366526cce56b5b98296838f3c4 — computed
on the file's content through the end of section 10 with this paragraph
held at its pre-substitution placeholder text (per the note above, a
fingerprint necessarily excludes its own line's final content). This is
v4's true baseline; v3's is not recoverable (see the header). Record the
v5 line/byte/sha here after the first save following the next version
bump. Sibling copy revision_artifact_ledger_v2.md remains WITHDRAWN.

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
reproduced). G1 evidence: block attention narrows but does not close the CD penalty.
G1 precondition MET 03 Aug: src/analysis/analyze_block_attention.py oracle PASS on
Adi's machine (22/22 checks, exit 0; three-arm pivot, three paired
contrasts with exact sign p, B5 panel). Script /critique'd (2 SEV2 +
2 SEV3 fixed, incl. sign-p computed not asserted and diag-absence ->
FAIL not crash); commit to the branch owed at G2 alongside the cert
extension. Prose corrections vs the original entry, per the oracle:
Blk-CD negative on 4/5 seeds at gamma {0,.3} but 3/5 at {.6,.9} (the
blanket 4/5 was wrong); B5 PR CD first-epoch mean is 3.75, not 3.8;
the 216 nonfinite steps span epochs 6-17 in CD-family runs only, i.e.
recurring AMP loss-scale adjustment, not scaler warmup (exclusion from
means unchanged). G1 EXECUTED 03 Aug: the Critical 1 wording landed —
PENDING-B1 resolved in response_to_reviewer_yc7L_v3.md (§4 below) with
numbers taken from this script's output; submission-time re-verifies the
numbers, not the wording. Kaggle version URL: STILL PENDING (placeholder).

### results/Revision/train_block_attention/diag_b5_gamma06.csv —
COMMITTED (777c462, same commit)
1248 per-epoch rows (grad norms all 60 runs; PR non-null on the 15
gamma=0.6 runs, 301 rows). 216 nonfinite grad steps, spanning epochs 6-17
in CD-family runs only — recurring AMP loss-scale adjustment, NOT scaler
warmup; excluded from means by construction. PR first->last epoch means:
CI 4.2->2.6, CD 3.75->4.42, CD_Block 3.9->8.9. Final-epoch grad-norm means
grow with gamma for CD (0.73->1.23, clipping active) vs CI ~0.87. B5
wording ceiling applies: "consistent with hypothesis", never mechanism.
Per-cell Blk-CD sign counts are in the results entry above.

Correction history (03 Aug, per the analyze_block_attention.py oracle;
retained per the update discipline, and folded into the figures above rather
than left standing beside them): the entry originally read "216 nonfinite
grad steps = AMP scaler warmup" and "CD 3.8->4.4". Both were superseded —
the first by the epoch-6-17 CD-family-only finding, the second by the 2-dp
values 3.75 -> 4.42. Exclusion from means unchanged in both cases. Fixed
05 Aug: the superseded figures had remained in the entry's headline with the
correction eight lines below, so a reader quoting the entry got the retracted
version.

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

### dryrun_stage0.py — SCRATCH (block-attn-dryrun dataset version; run
03 Aug)
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
boundary and ECL pricing.
RUN 03 Aug on T4 as a block-attn-dryrun dataset version (torch 2.13,
AMP). Measured (steady s/epoch, peak GiB, projected h/run at the
stated epoch provenance):
  lf_c21      CI  b128   7.2 s/ep  1.55  0.08 h/41ep
  lf_c21      CD  b8    35.8 s/ep  0.12  0.17 h/17ep
  lf_c21      Blk b8    18.1 s/ep  0.13  0.09 h/17ep   (closes P2's
    B1-exemption gap; live-run cross-check: probe projection
    0.17 h = 612 s (steady 35.8 x 17 = 609 s plus rounding) vs live
    820 s for CD (+34-35%, per-epoch val+checkpoint overhead), 324
    vs 340 s for Blk)
  bnd_p4_c21  CI  b128  87.7 s/ep  6.16  1.22 h/50ep
  bnd_p4_c21  CD  b128 915.7 s/ep  6.16  5.60 h/22ep
  bnd_p4_c21  CD  b8   907.5 s/ep  0.42  5.55 h/22ep
  bnd_p2_c21  CI  b128  OOM -> INFEASIBLE at fixed protocol batch
  bnd_p2_c21  CI  b64  261.4 s/ep  6.18  3.63 h/50ep
  bnd_p2_c21  CD  b64 3820.5 s/ep  6.18 23.35 h/22ep
  bnd_p2_c21  CD  b8  3605.1 s/ep  0.82 22.03 h/22ep
  ecl_cd      CD  b128  OOM
  ecl_cd      CD  b64 14803.7 s/ep 11.67 205.61 h/50ep [SUPERSEDED — 60-step probe, over-counts ~2.9x; see measured b2 below]
  ecl_cd      CD  b8  14548.0 s/ep  1.49 202.06 h/50ep [SUPERSEDED — 60-step probe, over-counts ~2.9x; see measured b2 below]
  ecl_cd      CD  b2   5072 s/ep (ep1); ~5073 steady thru ep5; ~62-70 h/converged run (44-50 epochs, per committed CI best_epoch range excluding pred=720, x ~5073 s/ep)  MEASURED, Kaggle v4 ecl_ci_cd_train, single T4, VRAM << 16 GiB
  ecl_blk     Blk b128  OOM
  ecl_blk     Blk b64   351.0 s/ep 11.83   4.88 h/50ep [worst case]
  ecl_blk     Blk b8    340.5 s/ep  1.50   4.73 h/50ep [worst case]
ECL finding: vanilla CD on ECL is memory-FEASIBLE at batch <= 64 under
fused attention (11.67 GiB peak) and wall-clock-infeasible — the committed
infeasibility claim survives as a compute constraint, no longer a memory
constraint (feeds W4 and PENDING-ECL-STATUS). MEASURED (Kaggle v4
ecl_ci_cd_train, single T4, batch 2, VRAM << 16 GiB): 5072 s/ep, 0.641
s/step steady through epoch 5, ~62-70 h/converged run. SUPERSEDED probe
figures (b8/b64, 60-step timed-step probe, over-counts epoch time ~2.9x
from cold-start-dominated 60-step scaling): 61.9 s/step, ~200 h/converged
run ~ 17 sessions — retained here only as the figure the estimate-class
validation below was checked against, never as the current per-step cost.
Estimate-class validation: the p2-scaled quadratic-token prediction made
mid-run (64.6 s/step) landed within 4% of the superseded probe figure
(61.9 s/step); this validates the prediction methodology against the
probe and no longer bears on the real per-step cost, which is the
measured 0.641 s/step above.
I1 validated at the per-step level: P=4 CD 8.7 s/step vs ar1_c84's
8.1 s/step at matched C*N (5355 vs 5292). The pre-G0 ~63 h B2
estimate missed 1.79x = 1.67x window count (boundary epochs carry
13,393 train windows, 13,392 samples consumed per epoch at batch 8 under
drop_last — count pinned 04 Aug from the retrieved boundary notebooks —
vs ar1's 8,033; note 05 Aug that 8,033 is the current-generator figure, which
the committed grid carries only for seeds {789, 1011}, seeds {42, 123, 456}
having 7,433 — see section 7) x 1.07x per-step residual (8.72 vs
8.14 s/step) -- dominated by a window-count error, not an I1 error.
G0 STATUS: FULLY CLOSED 03 Aug (lf, boundary, and ECL all priced;
B2/B3/B6 applications recorded in the plan §6).
Projected epochs borrow committed best_epoch maxima (P=8 CD max 12
stands in for P=4/P=2 CD, no committed cell exists); trigger (b)
re-applies after each block's first seed.
Output artefact: /kaggle/working/stage0_probe.csv — download owed
(scratch, outside git); session ran near the 12 h cap (ECL warmup
epochs ~8 h of it). Kaggle version URL owed for this run (third URL
placeholder alongside the B1 run and the dataset version).

### certify_clean_clone.ps1 — local, gitignored
Clean-clone certification harness (fresh clone of committed HEAD, reviewer
venv, canonical row counts, README invariants, pytest-no-skips,
formatting gate, byte-identical equiv_summary.csv, all analyses run, all
figures regenerated). Latest result on main c04d3cb: 29 pass, 0 warn,
0 fail, exit 0. Known trap fixed by procedure (run from main; it clones the
checked-out branch); one-word hardening APPLIED 04 Aug (--branch main on
the clone line, Select-String-verified). Extension owed at G2: run the new analyze scripts and check the new CSVs.

## 4. Reviewer-response artefacts (local, outside git)

### Phase 1 official comment — POSTED 02 Aug 2026, 19:46
Title "Author response: Resolved items and experiments underway"; 4711
chars; reviewer order; ETA 3-4 weeks stated. Single reviewer confirmed on
the form's reader list at posting (V1 evidence). Source file
phase1_comment_FINAL.md. The "now running" statement was posted while the
B1 launch was believed underway but unconfirmed; the launch was confirmed
03 Aug and the sweep completed the same day, so the claim is discharged.

### response_to_reviewer_yc7L_draft.md — SUPERSEDED by v2; local copy gone
### response_to_reviewer_yc7L_v2.md — RECOVERED 03 Aug
205 lines, operator header, 10 PENDING result slots (Phase 2 response).
Was lost locally (not committed, absent from the user profile); the full
text survived in the project chat "Paper review feedback" and the file
was downloaded from that chat's attachment 03 Aug. Fingerprint verified
against this ledger's record: 205L/12229B, operator header first line,
exactly 10 bracketed PENDING slots (B1, B1-REF, B2B3 x2, B4, B4-REPRO,
B5, B6-IF-RUN, VRAM, ECL-STATUS) — byte-faithful recovery, no
reconstruction. Copies: docs/ locally + project knowledge (off-machine
backup satisfied; docs policy keeps it out of git). Superseded same day
by v3 below; retained as the faithful original.

### response_to_reviewer_yc7L_v3.md — DELIVERED 03 Aug
v2 plus the G1 execution: PENDING-B1 resolved with Critical 1 wording
frozen from analyze_block_attention.py output (oracle PASS 22/22; the
block variant reported as intermediate — narrows but does not close the
CD penalty; equivalence explicitly not extended at n=5); operator header
records the freeze (submission re-verifies numbers, not wording). 220
lines. Remaining slots: B1-REF, B2B3 x2, B4, B4-REPRO, B5, B6-IF-RUN,
VRAM, ECL-STATUS. Proposed ECL-STATUS wording drafted from the G0
measurements 03 Aug — awaiting Adi's approval before it is applied as
v4. Store in docs/ beside v2; second copy in project knowledge.

### response_to_reviewer_yc7L_v4.md — SUPERSEDED by v5 same day
v3 plus the approved ECL-STATUS wording (timed-step projection, W4-checked).
Deletion instructed 04 Aug alongside v3 (keep v2 original + v5 canonical);
deletion UNVERIFIED until a docs/ listing confirms it.

### response_to_reviewer_yc7L_v5.md — DELIVERED 03 Aug; VERIFIED 04 Aug
v4 with the operator-header updates (G1-executed note; ECL-STATUS
resolution recorded as satisfied through W4). Local save fingerprint-
verified 04 Aug against the delivered file: 234 lines / 14,090 bytes,
byte-exact (the PowerShell 207 was Measure-Object skipping blank lines);
8 [PENDING-*] slots remain below the cut line: B1-REF, B2B3 x2, B4,
B4-REPRO, B5, B6-IF-RUN, VRAM. Canonical Phase 2 response going forward.
SUPERSEDED IN PART 06 Aug (section 8): the B2B3 x2 tier assumed a P=2
result; the P=2 CD infeasibility decision replaces those two tiers with a
single P=4 gamma-0.9 existence cell plus a P=2 infeasibility note — see
section 8 for the current slot content.
Count corrected 05 Aug from "7", which contradicted its own list; the
arithmetic is v2's 10 slots, minus B1 resolved at v3 (9), minus ECL-STATUS
resolved at v4 (8). The same wrong count propagated into the 04 Aug handoff
and should be corrected wherever it is quoted, since it is the critical-path
count to G3.

### Pending per plan: train notebooks/engine for B2-B4, B6;
analyze_block_attention.py DELIVERED 03 Aug (oracle PASS local, 22/22;
commit owed at G2 with cert extension, README row, and test smoke); the
subgroup analysis script (written before results per P4); B4/B6
instrumentation.

## 5. Cross-cutting lessons (cumulative; each dated)

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
- A lost chat-produced artefact is recoverable if the chat survives, but
  only download recovers it byte-faithfully; snippet reconstruction cannot
  be diffed against anything (03 Aug, response v2 recovery).
- Generator constants state the intent; the results CSVs state what ran.
  Derive every protocol number from a recorded column (steps_per_epoch,
  batch_size) and only then check it against the generator, never the
  reverse. Deriving 8,033 windows from the generator and skipping the
  CSV check put a wrong figure into a main.tex table that was caught only
  at integration (05 Aug). Corollary: a count that appears in two places
  with different values is a finding, not a rounding difference.
- A count stated beside its own list must be recounted, not trusted: the
  response's "7 PENDING slots" listed eight and propagated into a handoff
  (05 Aug).
- Two copies of a canonical local doc will diverge silently. Record each
  canonical doc's line/byte/sha in its own entry so a duplicate is
  detectable by inspection (05 Aug, the two ledgers).
- `.gitignore` does not untrack anything. A rule matching an already-tracked
  path is inert, and `git check-ignore` returning a match proves only that
  the rule exists. `git ls-files <path>` is the only check that answers
  whether the file is tracked. Three docs stayed in git for two days behind
  a rule that looked like it was working (05 Aug).
- An inference drawn from one command's output must be re-tested before it
  is written down. Three times on 05 Aug a confident reading of a single
  output — a `--stat` line count, a gap in rule line numbers, a claim about
  when rules were committed — was contradicted by the next command. Cheap
  checks are cheaper than corrections in a canonical doc.
- A same-day decision can invalidate a claim written earlier the same day.
  The notebook reword landed hours after Part I asserted byte-identical
  commitment, and the folder README was updated while Part I and this ledger
  were not; the contradiction surfaced only when the README was read
  alongside them (05 Aug). When a decision changes an artefact's bytes, grep
  every doc for the claim it invalidates before closing the session.

## 6. 04 Aug 2026 — retrieval, B2 launch, corrections, open items

### notebooks/original/ — six retrieved boundary training notebooks —
COMMITTED 04 Aug (2b0a885, "notebooks+README"; branch tip fd3a863 confirmed
05 Aug by `git rev-parse HEAD` = fd3a8635fd42feba609a9ac16a0fab588a85e931).
Status corrected 05 Aug from "COMMIT PREPARED … commit pending execution":
the commit executed and the marker was never resolved. The tip is directly
verified; 2b0a885 as the specific commit comes from the 04 Aug handoff, not
from git output seen here — confirm with `git log --oneline` at G2. The
staging package was boundary_notebooks_commit_package_v3.md; the "package
v4" in the earlier wording does not match any local file.
All six boundary training notebooks retrieved from the Kaggle environments
in which they ran; committed under notebooks/original/ with content-based
filenames (p4_full_plan, p4_ci, p4_ci_dlinear, grid_n3, grid_n5,
grid_n5_resume). Identity-free committed record:
notebooks/original/README.md (roles, SHA256s, shared protocol).
BYTE-IDENTITY CORRECTED 05 Aug: four of the six were committed
byte-identical; train_boundary_p4_ci and train_boundary_p4_full_plan carry
the same-day "for this account" -> "for this notebook" reword, so their
committed hashes differ from the retrieved ones (2312893d… / e81c8800…
committed, against f03dcf47… / 6b8235e6… retrieved). Both records are
correct and describe different bytes; the folder README carries the
committed hashes, scripts_provenance Part I the retrieved ones, and Part I
§6 now reconciles them. The earlier blanket "committed byte-identical"
wording here was written before the reword decision and never revisited —
the same stale-prose class as the diag_b5 entry in section 2.
Full mapping with origins: scripts_provenance Part I (LOCAL, per docs
policy — the package's earlier instruction to commit the provenance doc is
WITHDRAWN as a docs-policy violation caught 04 Aug). Reconciliation
findings: grid_n5_resume (retrieved name suffix `_last_run_in_a`) is the
probable executed form of the CSV 14 extension (it alone carries the
resume-seeding cell); NO retrieved P=4 file contains seeds {789, 1011}, so
the executed CSV 15 extension version is an unretrieved Kaggle version.
Hash-pairing verified locally 04 Aug (six distinct SHA256s, correct names).

### Boundary DGP/windowing finding — RECORDED 04 Aug
The boundary generator draws iid innovations (no cross-channel covariance),
NOISE_STD=0.1, T=20,000, 70/10/20, no burn-in — a different DGP from the
leader-follower sweep. The boundary CSVs' rho=0.5 is a hardcoded metadata
constant. Train windows 13,393, drop_last=True (spe 104 b128 / 1674 b8).
Consequences: main.tex's general synthetic protocol sentence is inaccurate
for the boundary rows; the boundary-vs-lf ratio comparison is partly a DGP
difference. Provenance annotated (CSVs 3/4/14/15, Part D, Part I §5).
DECISION TAKEN 05 Aug, ahead of the Friday gate: the fullest disclosure
option was adopted and applied to main.tex (section 7 below records the four
edits and the file fingerprints). New B2 rows keep rho=0.5 for schema
mergeability, with the constant disclosed in the new table's caption rather
than the schema changed — changing it mid-tranche would break the notebook's
SCHEMA assert on a live run. The Friday gate therefore no longer carries the
W-track wording item or the rho-metadata decision.

### notebooks/train_boundary_p4.ipynb (v2) + config_boundary_p4.json (v2) +
analyze_boundary_p4.py (skeleton) — DELIVERED 04 Aug (commit at G2/B2 wrap)
B2 engine: committed boundary protocol verbatim (tripwire-asserted
windows=13393 / spe=104), per-epoch atomic checkpoint+resume (bit-exact
resume verified), session self-budget (11.0 h; a capped commit saves no
output, so the notebook never straddles the cap), cross-session bootstrap
from attached prior-version output, config-driven run list (P5). /critique
x2: 9 + 7 findings fixed, all regression-tested. Skeleton exits 2 by design
until its oracle slot is filled.

### b1-block-attn dataset — updated 04 Aug (version FILL: ____)
Now carries models.py, models_cd_block.py (sha256[:16] 1eecbddb / 8703aeed)
and config_boundary_p4.json (acct2-tranche1-gamma09: gamma 0.9, CD then CI,
seeds {42,123,456,789,1011}, session budget 11.0 h; config sha256[:16]
e40154d741271156).

### B2 gamma-0.9 tranche — LAUNCHED 04 Aug, session 1 (commit mode)
Launched under the per-account compliance model (owners launch from their
own devices; resolved 04 Aug: collaborator accounts are owner-operated,
cross-account handoff of resume state travels via the shared dataset).
Session-1 tripwires confirmed from the live log: module shas match, config
sha e40154d741271156, windows=13393, spe(b128)=104, arms 1,717,088 params,
run 1/10 (CD seed 42) training at launch.
APPEND PER SESSION: session -> account -> version URL -> runs completed.
- Session 1, 04 Aug — account `zoharsf` (established 05 Aug from the next
  session's bootstrap path /kaggle/input/notebooks/zoharsf/train-boundry-p4/,
  not from a session record; log it directly next time). Version URL: FILL
  ____ . Runs completed: CD seeds 42, 123, 456.
- Session 2, 05-06 Aug — third account (Account C, username khasidashvilli).
  Version URL: FILL ____ (v1 and v2). Runs: CD 789 (v1, resumed at 15 epochs,
  best_val 0.934937@12) and CD 1011 (v1 -> v2 resume-chained) both DONE and
  traced; see the RESOLVED (O-a) block.
Dataset URL: FILL ____ .

### Open items (04 Aug snapshot; carried forward, status as of 05 Aug)
- O-a: RESOLVED 06 Aug — CD 789/1011 traced to Account C (khasidashvilli)
  notebook versions 1+2, resume-chained; see the RESOLVED (O-a) block under
  results/.../results_boundary_p4_complete.csv. Version URLs still owed.
- O-b: executing account of the final CSV 14 extension session (retrieved
  filename suffix `_last_run_in_a`); amend CSV 14 if needed. STILL OPEN.
- Kaggle URL placeholders still owed: B1 commit-run version URL; b1-block-attn
  dataset version URL; Stage 0 run version URL (plus the B2 session URLs
  above). STILL OPEN.
- response v3/v4 deletion from docs/: instructed 04 Aug, UNVERIFIED. STILL
  UNVERIFIED.
- pytest run before the retrieval commit (README claims a clean suite).
  STILL OPEN.
- Friday 07 Aug gate: B2 trigger (b) on measured gamma-0.9 convergence;
  PROPOSED threshold amendment; mirror exclusion list for internal docs. The
  W-track disclosure wording and the rho-metadata decision were taken 05 Aug
  and are removed from the gate (section 7).

## 7. 05 Aug 2026 — B2 across accounts, grid-protocol finding, main.tex disclosure

### B2 tranche — RESUMED on a third account 05 Aug
Notebook downloaded from the session-1 account and re-uploaded with the
registry and the surviving checkpoint as a private dataset on the third
account; cross-session, cross-account resume verified end to end from the
live log (registry seeded, ckpt seeded, 3/10 recognised complete, seed 789
resumed at 15 epochs). Tripwires re-passed in the new environment
(windows=13393, spe=104, config sha e40154d741271156, module shas match),
which independently re-confirms the boundary protocol pinned in
scripts_provenance Part I §5.
Results so far (gamma 0.9, CD, batch 128, spe 104): seed 42 MSE 0.95557576 /
MAE 0.77858864 / best_epoch 18 / 1872 steps; seed 123 0.99428057 /
0.79175437 / 10 / 1040; seed 456 0.96999986 / 0.78462838 / 13 / 1352.
Cost re-estimate: completed runs took (best_epoch + patience 10) epochs at
~918 s, i.e. 7.1 / 5.1 / 5.9 h. Remaining tranche prices at ~17.6 h
(CD 789 ~7.4 h, CD 1011 ~6.9 h, five CI runs ~3.3 h combined), against the
~42 h re-estimate that motivated the budget-expansion proposal. The
remainder fits inside one account's weekly quota, which collapses the
scheduling problem behind plan OQ3.

### Notebook final-cell change — DELIVERED 05 Aug
train_boundary_p4.ipynb 682 -> 694 lines. OUT_PATH stays canonical
(results_boundary_p4.csv) because the Cell 6 bootstrap globs that exact
string and asserts a single match; the final cell now additionally writes a
download copy tagged by the resume point,
results_boundary_p4_<MODE>_s<seed>.csv, derived from the first RUN_LIST entry
not in `done` rather than from a filesystem glob over leftover checkpoints.
Local convention: CSV snapshots carry the tag, checkpoints keep the canonical
dot form ckpt_P4_g0.9_CD_s<seed>.pt — an underscore-renamed checkpoint would
be seeded by the bootstrap but never found by the resume lookup, silently
restarting a ~7 h run. Snapshot supersession verified by line comparison:
_CD_s123 is a strict subset of _CD_s789.

### AR(1) grid runs on two series lengths — FINDING 05 Aug
results_grid.csv carries two step-count families splitting cleanly by seed:
{42, 123, 456} at 7,433 training windows (59/117/930/233 steps per epoch)
and {789, 1011} at 8,033 (63/126/1005/252). Every count is ceil(W/batch), so
the recovery is exact. Probable cause, inferred from exact arithmetic
(int(13400*0.6) - 512 - 96 + 1 = 7433, and 13,400 = 14,400 - 1,000 burn-in):
the earlier generator kept 13,400 usable steps, the seed extension 14,400.
Independently confirmed by the gamma=0 cross-experiment anchor, which
reproduces to 4 dp where the window counts match (grid 1.0540 vs LF 1.0540
at seeds {789,1011}) and diverges where they do not (0.9939 vs 1.0090).
Paired per-seed CD-CI differences are unaffected — both modes of a seed share
that seed's series — so every paired statistic in the paper stands; the
absolute cell means in Table 3 mix two series lengths. Recorded in
scripts_provenance Part J §2 and design_rationale sections 3 and 9.
OPEN (J-b): main.tex tab:batch_sizes reports only the 7,433 family and
presents it as the protocol; disclose there, restate per seed group, or rely
on the new table's footnote.

### main.tex protocol disclosure — APPLIED 05 Aug
Base fd3a863, sha256 bd503297009390c7644980366ad7d0a9c94af3659d745e89fef45a63ea8707fd,
794 lines -> 854 lines / 61,554 bytes, sha256
2548bb351e900dc5c86d57cf250d8a93d7daeebd7d277c93757e2110cbd54e2a. Four
edits, each anchor uniqueness-checked; diff is three hunks and five removed
lines. (1) new table tab:synth_protocols over four synthetic families with a
lead-in paragraph; (2) the gamma=0 anchor scoped to the gamma-sweep;
(3) the boundary paragraph's P=16 1.001-vs-1.007 gap no longer attributed to
per-seed spread, since no P=16 cell ran under both generators, and the two
experiments no longer presented as independent replications; (4) the grid
protocol sentence scoped with "in this grid". NOT YET COMMITTED — git
operations are Adi's, and this entry is UNVERIFIED until git output confirms.

### Local docs revised 05 Aug — DELIVERED
scripts_provenance.md 889 -> 1083 lines (critique v2.4, 13 findings, all
patched; new Part J). design_rationale.md 522 -> 632 lines (critique v2.4,
12 findings, all patched; two were mathematically wrong — the
positive-definiteness justification for rho=0.9 and the ECL epoch-vs-session
comparison). This ledger, v3 (critique v2.4, 12 findings, all patched).

### docs/ untracked and history audited — EXECUTED 05 Aug
Closes the G3 docs-handling item ahead of the mirror rather than at it.
- `git rm --cached` on all three tracked docs; `git ls-files docs/` now
  returns nothing. Files remain on disk; they are canonical there. Staged
  set verified as exactly three deletions.
- THE IGNORE RULE WAS NEVER THE MECHANISM. `.gitignore` at HEAD already
  carried `/docs/` (line 65) and `results/Revision/train_boundary_p4/`
  (line 66); `git diff HEAD -- .gitignore` is empty at 66 lines on both
  sides, so nothing this session changed that file. A `.gitignore` rule has
  no effect on a path git already tracks, so `/docs/` sat inert through the
  entire 03 Aug docs-policy period while all three docs stayed tracked. The
  missing step was always `git rm --cached`. Earlier wording in this entry
  said the `docs/` rule was "at .gitignore:64" and that the results rule was
  "added at .gitignore:66"; both were wrong and are corrected here.
- UNRESOLVED, recorded rather than reconstructed: mid-session
  `git check-ignore` reported `docs/` at line 64 — different rule text and a
  different line from the final `/docs/` at 65 — while the file now equals
  HEAD exactly. Two `WriteAllLines` appends were executed and left no net
  change. What happened between those states is not determinable from the
  output captured. Inspect `.gitignore` once directly before the mirror cut;
  the end state is verified and correct, only the path to it is not.
- SPARSE-CHECKOUT / SKIP_WORKTREE DISCOVERY: the first `git rm --cached`
  staged only design_rationale.md and refused the ledger and the plan,
  which carry the skip-worktree bit — the 04 Aug PROPOSED
  `git update-index --skip-worktree` was executed on two of the three.
  Consequence, unnoticed until now: git has been ignoring local edits to
  the ledger and the plan, so their permanent modified-state was silenced
  by hiding the edits rather than by resolving them. `--sparse` completed
  the removal. The skip-worktree proposal is now moot and should not be
  reapplied.
- HISTORY EXPOSURE AUDIT: `git rev-list --all` crossed with `git grep` over
  `docs/` for `zoharsf|mendelowitzadi|kaggle.com/` and for the literal
  Windows user path returned EMPTY at every commit. Three commits ever
  touched docs/ (da46d22, 8451b5d, f0693ed). The tracked snapshots are
  stale, not identifying, so no history rewrite is warranted and the
  hashes every doc cites stay valid.
- COMMIT AUTHOR METADATA — NEW G3 ITEM. `git log --format='%an %ae'` shows
  two identities across all history: a full personal name with a personal
  email, and the GitHub noreply form. File contents are clean; commit
  metadata is not, and no file-level sweep can reach it. CONSEQUENCE: the
  mirror must be created by `git init` on a copy of the working tree under
  an anonymous identity, never by pushing or cloning this history. Whether
  the current origin is public is STILL OPEN and determines whether this is
  a future-exposure item or a present one.
- `results/Revision/train_boundary_p4/` appeared as untracked in the
  `git status` taken before the second `.gitignore` append. In the final
  state it is ignored by a HEAD rule at line 66, verified by
  `git check-ignore -v`. Net effect: it is ignored; the session did not add
  the rule that ignores it.
- paper/main.tex shows modified and unstaged: the four disclosure edits are
  in the working tree, not committed. Git warns LF will become CRLF on this
  path (.gitattributes pins `*.ipynb -text` but not `*.tex`), so the sha
  recorded for the delivered file will not survive a git round-trip. Treat a
  post-checkout mismatch as line-ending normalisation, not corruption; a
  `*.tex text eol=lf` attribute would remove the ambiguity.
- STATUS: the three deletions plus nothing else were staged and reviewed.
  Whether the commit itself executed is UNVERIFIED here; confirm from
  `git log --oneline -1` and record the hash. The removal does not reach
  GitHub until pushed.

### Tracked-file audit — `git ls-files` 05 Aug
Resolves several standing items in one command.
- notebooks/original/ carries the six notebooks AND README.md; no
  RETRIEVAL.md exists. The header's duplicate-copy resolution is CONFIRMED
  from git, not inferred: revision_artifact_ledger_v2.md was wrong and
  remains WITHDRAWN.
- docs/scripts_provenance.md is NOT tracked, confirming §2's record that the
  03 Aug commit instruction was never executed.
- docs/design_rationale.md, docs/revision_artifact_ledger.md and
  docs/revision_master_plan_v2.md ARE tracked — the frozen v2.1 snapshots
  inside 8451b5d. Consequence: every local edit to these three shows as a
  git modification permanently, and the mirror ships them unless docs/ is
  excluded at the G3 cut. The skip-worktree proposal in the 04 Aug handoff
  addresses only the first half.
- src/analysis/analyze_block_attention.py NOT tracked (commit owed at G2),
  notebooks/train_boundary_p4.ipynb NOT tracked, no results_boundary_p4.csv
  yet — all as recorded.
- Response recommended item 3 claims the analysis oracle checks pass from a
  clean clone. Two scripts that claim covers are not yet committed; the
  claim is true only after the G2 commits land. Operator checklist item 6
  covers it, but the dependency is worth naming here.
- RESOLVED 05 Aug: `git log --oneline -- notebooks/original/` returns
  fd3a863 "Reword a comment in two boundary notebooks; update hashes" and
  2b0a885 "Add original boundary training notebooks; update disclosure".
  The 2b0a885 attribution is now confirmed from git rather than carried on
  the 04 Aug handoff's authority, and fd3a863's own message independently
  confirms the byte-identity correction — the reword is a commit, not just a
  decision. Branch history: da46d22, 8451b5d, f0693ed, 777c462, 0cf818e,
  2b0a885, fd3a863.

### Open items opened 05 Aug
- J-a: retrieve the AR(1) grid's original generator and confirm the burn-in
  accounting behind 7,433. Blocks: the mechanism above is stated as inferred.
- J-b: decide the tab:batch_sizes treatment of the seed split.
- J-c: main.tex states CD is excluded at P=4 on VRAM grounds and Table 7
  carries dashes there; B2 is falsifying both. Revisit at B2 wrap, together
  with the Modes row of the new table and notebooks/original/README.md,
  which states that the CD arm is filtered out at P in {2, 4}. The README's
  claim is scoped to "the environment these notebooks ran in" and so remains
  defensible as written, but it is committed, reviewer-visible text about a
  cell the revision is now filling, and should say so explicitly.
- J-e: the notebook's bootstrap prints emit full Kaggle input paths, exposing
  two usernames in cell output; printing .name instead makes the leak
  structurally impossible. Apply before session 3. G3 sweep item, alongside
  the known hard-coded INPUT_ROOT in the B1 notebook.
- B4 build: ETTh1 notebook as a minimum delta off the boundary engine, plus
  its analysis script under a FULL oracle, matching what
  analyze_block_attention.py received (22/22, exit 0). DECIDED 05 Aug. A
  scoped alternative was PROPOSED (6-8 cases over the partition join and the
  subgroup reduction only, engine parts inherited) and REJECTED: it forces a
  caveat into the provenance chain naming which surface the oracle covered,
  and Critical 1 is already a question about whether the controls are what
  the paper says they are. Coverage: schema and load validation; partition
  join key alignment and cardinality with no silent row loss; misalignment
  rejection; per-horizon median-split balance; run-level aggregation before
  per-seed pairing, as the script's docstring pre-registers; paired
  statistics reproducing paired_stats.py at df=4; aggregate reproduction
  against committed results_etth1.csv, which PENDING-B4-REPRO depends on;
  degenerate inputs (missing seed, missing horizon, all-one-subgroup).
  Script before results, per P4.
  CORRECTION 05 Aug: the scoped version was written into this entry and into
  plan §11 as though recorded, before Adi had confirmed anything. It had
  never left PROPOSED. Fixed on the same day it was written; the
  DECIDED/PROPOSED discipline exists for exactly this.
- Correct the "7 PENDING slots" count to 8 wherever it is quoted, including
  the 04 Aug handoff.

## 8. 06 Aug 2026 — B2 boundary tranche complete, O-a resolved, P=2 infeasibility decided

### results/Revision/train_boundary_p4/results_boundary_p4_complete.csv — LOCAL (Kaggle run, 06 Aug 2026; tranche B2)
Boundary P=4 tranche on the leader-follower VAR(1) structure at the patch-size=4
boundary, gamma 0.9: CD and CI × seeds {42,123,456,789,1011}, batch 128. Produced by
train_boundary_p4.ipynb (v2, resumable). Per-run atomic checkpoints
ckpt_P4_g0_9_CD_s{123,789,1011}.pt sit in the same folder (valid torch archives).
Provenance of the pairing: generate(gamma, seed) seeds np.random.default_rng(seed) and
is deterministic in (gamma, seed); the run calls it once per seed, so CD and CI at a
fixed seed train on identical train/val/test series — matched pairs by construction.
Normalisation is fit on the train split only (no leakage). Schedule: warmup 10 /
max 50 / patience 10.

DECIDED (06 Aug 2026): the gamma-0.9 boundary CI of record is this file's CI arm.
Basis for the choice is matched per-seed pairing with CD plus a traceable notebook +
checkpoint chain — decided on provenance, not on effect size.
- results_boundary_p4_ci.csv (in the parent results/ dir), gamma-0.9 CI rows: demoted
  to an unmatched replicate. Retained for cross-check, never pooled with the matched
  CI above. Separate run of unconfirmed provenance (not specified by
  config_boundary_p4.json; no retrieved notebook); best_epoch and test_mse differ from
  the matched CI (up to +0.00264 mse, best_epoch 17->28 on seed 1011).
- results_boundary_p4_ci.csv gamma-0.6 CI rows: sole gamma-0.6 source, retained as
  PROVISIONAL. They inherit that file's unconfirmed provenance and are unmatched;
  superseded once the gamma-0.6 CD tranche runs through train_boundary_p4.ipynb, which
  will emit a matched gamma-0.6 CI.
- The four partial CSVs in train_boundary_p4 (results_boundary_p4.csv,
  results_boundary_p4_CD_s{123,789,1011}.csv) are verified full-row subsets of
  _complete.csv with no unique rows; archived under _superseded_by_complete_* by
  archive_superseded_boundary_p4.ps1 (reversible move, not deleted).

RESULT (within _complete.csv, gamma 0.9, 5 seeds): mean test_mse CD - CI = +0.00041,
sign mixed (2 of 5 seeds negative). The same contrast against the retired ci.csv CI
gives +0.00096 (1 of 5 negative); both are near zero and the qualitative reading is
identical, so the conclusion is robust to the CI choice. Reading: CD is
indistinguishable from CI at the P=4 boundary — consistent with the tranche's
hypothesis. Not evidence of mechanism.

ACTION — DONE 06 Aug: analyze_boundary_p4.py read and run; it pairs CD-CI within
_complete.csv only and loads results_boundary_p4_ci.csv solely as a separate
cross-environment CI reference (never pooled). Oracle PASS, exit 0. The retired
gamma-0.9 ci.csv rows cannot enter the paired contrast.

RESOLVED (O-a, 06 Aug 2026): CD 789 and CD 1011 traced to retrieved account C
(khasidashvilli) notebook runs, versions 1 and 2; both reproduce complete.csv to
4 dp. Per-epoch figures in the logs are validation MSE (early-stopping signal);
the DONE line and the CSV test_mse are the test metric at best_epoch, so the
val-vs-test gap (789: best val 0.9329 vs test 1.0061) is expected, not a mismatch.
- CD 789: account C v1 [4/10], resumed from ckpt_P4_g0.9_CD_s789.pt; new val
  minimum at epoch 18 (0.932928), no improvement over epochs 19-28 (patience 10),
  DONE mse=1.0061 -> complete.csv test_mse 1.006069, best_epoch 18, total_steps
  1872 (=18x104). Epochs 1-15 preceding the v1 resume are checkpoint-backed only
  (their own log not in the retrieved cell); the winning epoch lies inside v1.
- CD 1011: account C v1 [5/10] ran epochs 1-29 (best val 0.997993@20, matching the
  earlier "epoch 29, patience 9/10" sighting), then continued in v2 [5/10], whose
  resume header reads back "29 epochs done, best_val=0.997993 @ epoch 20" -> v1 and
  v2 are one resume-chained run, not independent re-runs. v2 epoch 30 set a new
  minimum (0.997317@30); no improvement over 31-40 (patience 10), stopped at 40;
  DONE mse=0.9506 -> complete.csv test_mse 0.950597, best_epoch 30, total_steps
  3120 (=30x104). The winning epoch 30 was found only by continuing into v2; v1
  alone stopped one epoch short.
- Corroboration: CI 789/1011 also appear in v2 ([9/10] DONE 1.0051@18; [10/10]
  best val 0.996611@28), and v2 wrote the full 10-row CSV matching complete.csv
  value-for-value, incl. CD 42/123/456 (already traced via the earlier run) and the
  printed CD/CI summary 1.000422 -- a cross-check, not a per-seed trace for
  42/123/456.
Provenance recorded as "account C (khasidashvilli) versions 1+2, resume-chained,"
not a single version. All rows carry the committed fingerprint (gamma0.9, P4, C21,
batch128, spe104). O-a closed; the boundary P=4 gamma0.9 CD-vs-CI comparison is
now fully backed.
- P=2 CD compute-infeasibility probe: Kaggle notebook train-boundary-p2 (account mendelowitzadi), version 1 — 6.17 GiB peak alloc / 6.61 GiB reserved at batch 64, 18.8 s/step, spe 209, ~33 GPU-hours/seed projected (timed-step, best-epoch ~30). C×N=10731 confirms the paper's infeasible token count.
- ### src/probes/probe_boundary_p2.py + results/Revision/probe_boundary_p2_stdout.txt + config_boundary_p2.json — LOCAL / DELIVERED (P=2 feasibility probe, 06 Aug 2026)
Purpose: decide P=2 boundary feasibility BEFORE spending quota. Probe reuses the
notebook's real generate/WindowDataset/M.build_model code, overriding only
patch_size=2 (stride 1) and batch 64. spe measured = 209, which equals the derived
config_boundary_p2.json expected_spe (13393 train windows // 64, drop_last); N_PATCHES
= 511, C*N = 10731 — the exact token count the submission calls infeasible.

Measurement (Kaggle notebook train-boundary-p2 v1; account LOCAL-ONLY, G3-scrub, never
committed or reviewer-facing). Timed-step projection, not a completed run (compute-only
lower bound; excludes data-loading and per-epoch validation):
- CD: peak 6.17 GiB alloc / 6.61 GiB reserved [FITS on T4], 18775 ms/step, 1.09 h/epoch,
  ~32.7 h/seed (best-epoch ~30), 54.5 h ceiling.
- CI: 6.17 GiB alloc / 6.63 GiB reserved, 1258 ms/step, ~2.2 h/seed.
- Declared 5-CD + 5-CI run list: ~174 h typical, ~291 h ceiling.

DECIDED (06 Aug 2026): do NOT run P=2 CD. It is memory-feasible under fused SDPA (the
paper's memory wall was environment-bound) but practically infeasible by wall clock at
~33 GPU-hours per single CD seed — a compute constraint, not a memory one, consistent
with the quadratic-in-C asymmetry and mirroring the ECL cost story. Consequences:
- The three pre-declared PENDING-B2B3 tiers in response_to_reviewer_yc7L_v5.md are void;
  each assumed a P=2 result. The slot is filled instead with a single P=4 gamma-0.9
  existence cell plus a P=2 practically-infeasible-by-wall-clock note (GPU-hours only,
  no accounts/quota — that line in the probe stdout is raw output, stays out of the
  response prose).
- Traceability (checklist item 6): the ~6.2 GiB and ~33 GPU-hour figures trace to the
  committed pair src/probes/probe_boundary_p2.py + results/Revision/probe_boundary_p2_stdout.txt
  inside the anonymised repo; the Kaggle v1 identifier is the LOCAL cross-reference only.

Naming: local copies carry the "boundry" folder/notebook typo; committed paths use
"boundary" (script already src/probes/probe_boundary_p2.py). Do not let the typo or the
account username reach a committed path.
OPEN (scheduling, not resource-bounded): remaining P=4 CD gammas {0, 0.3, 0.6} are
affordable (~3 h/seed per this probe); running them before submission is a Friday-gate
choice, kept separate from the P=2 infeasibility language.

## 9. 07-08 Aug 2026 — gamma-0 P=4 tranche complete, HARD STOP data gap closed

STALE-STATUS FIX, 09 Aug /critique pass: this section did not exist before
this pass. The gamma-0 tranche's full completion (known since the 08 Aug
session) had been recorded in the master plan's §6 LAUNCH RECORD but never
carried into this ledger, so this file understated its own "Covers" line by
one full tranche — the exact stale-status defect class section 5's
cross-cutting lessons already names twice. Filed here as the fourth
occurrence rather than a third silent recurrence.

### B2 gamma-0 tranche — LAUNCHED 07 Aug across three accounts (deviation:
CFG embedded per notebook copy, not a shared config dataset — see master
plan §6/§7 exception note), COMPLETE 08 Aug — LOCAL (Kaggle runs; notebook
variants train_boundary_p4_gamma0_a1/_a2/_a3.ipynb, held at
`C:\Users\adime\Documents\ci-cd-patchtst\notebooks\train_boundary_p4_gamma_0\`,
not yet committed)
Seed coverage: CD seeds split acct1 {42,123}, acct2 {1011}, acct3
{456,789} — full canonical set, no gaps or duplicates. CI: all 5 seeds on
acct2. Cross-account consistency verified: models.py/models_cd_block.py
sha256[:16] identical across all three accounts (1eecbddbdf130528 /
8703aeed74877b1a); protocol tripwires (windows 13393, spe 104 at b128,
C*N=5355) identical across all three, as expected (gamma-independent).
All resumes verified clean via matching [resume] log lines against each
notebook's own prior stopping point — not merely re-attached and restarted.

RESULT (5/5 CD seeds, 5/5 CI seeds DONE):
- CD/CI ratio at gamma=0 is 0.9997, matching the expected internal-control
  result (CI, CD, and CD_Block should coincide when gamma=0 removes the
  cross-channel coupling gamma is meant to probe).
- Per-seed convergence-ratio variance against the matched gamma-0.9 row
  (ledger §8) is 0.57x-2.50x, not a clean multiplier. Mean-of-ratios
  (1.24x) and ratio-of-means (1.03x) disagree substantially — a genuine
  finding, not noise, given the spread.
- Known per-seed CD figures (from the master plan's session log, restated
  here as this project's single source for per-run numbers): seed 42
  mse=0.9883, best_epoch 21, 31 epochs total, 8.22 h; seed 456 mse=0.9607,
  best_epoch 16, 26 epochs total, 6.64 h. Seeds 123, 789, 1011 (CD) and all
  five CI seeds finished this session; individual test_mse/test_mae/
  best_epoch values are not narrated row-by-row here, but the paired-cell
  summary they roll up to is now oracle-verified (see ACTION below) — the
  committed CSV plus that check is the record, per P1.
  Do not treat the two-seed figures above as the full row; the 0.57x-2.50x
  variance and the two summary ratios come from the full 5-seed analysis.

ACTION — DONE 09 Aug: the three accounts' output files
(gamma0_a1/a2/a3/results_boundary_p4.csv) merged into
results_boundary_p4_gamma0_complete.csv (10 rows, no cross-account overlap)
and run through analyze_boundary_p4.py alongside the gamma-0.9 file. Both
gammas PASS against the frozen oracle:
  gamma=0.0: CI 0.9836  CD 0.9833  CD/CI 0.9997  CD-CI -0.0003
    95% CI [-0.0019, +0.0014] (-0.03%)  sign 3/5 p=0.500  n=5  within 1%: Y
  gamma=0.9: CI 0.9749  CD 0.9753  CD/CI 1.0004  CD-CI +0.0004
    95% CI [-0.0010, +0.0018] (+0.04%)  sign 3/5 p=0.500  n=5  within 1%: Y
Cross-environment CI reproducibility also PASS at gamma=0.9 (new 0.9749 vs
old 0.9743) and gamma=0.6 (old-only, as expected — no current-env gamma=0.6
data exists yet). Command:
  uv run python src\analysis\analyze_boundary_p4.py
    results\Revision\train_boundry_p4\results_boundary_p4_complete.csv
    results\Revision\train_boundry_p4\results_boundary_p4_gamma0_complete.csv
_ORACLE_PAIRED in analyze_boundary_p4.py carries the gamma=0.0 entry now;
future runs against this file assert without manual re-verification.

HARD STOP STATUS: this section closes the data gap the master plan's HARD
STOP gate (§6) was waiting on. It does NOT resolve the gate — a
re-pricing approach given the 0.57x-2.50x scatter is still an open
decision for Adi (retro task list item 2, 2026-08-09), not something this
ledger entry decides on its own authority. Do not schedule gamma {0.6,
0.3} on the strength of this section alone.

FALSIFIABLE-PREDICTION CHECK, resolved: the CONVERGENCE RE-PRICING GATE's
P=8/P=16-based prediction that gamma-0 converges slower than gamma-0.9 is
confirmed in direction (every seed) but not in the magnitude either the
P=8/P=16 extrapolation (1.4-1.8x) or an earlier n=2 partial read
(~1.14-1.23x) suggested. Per this project's own recurring-pattern lesson
(section 5): a partial-sample read has now undersold real variance twice —
once for the P=8/P=16 projection itself, once for the n=2 gamma-0 read
that preceded this full row. Default to labeling any n<5 read as
provisional-variance-unknown going forward (retro insight, 2026-08-08).

ADDENDUM, 09 Aug 2026 (not a correction — this section's record of what ran
stands as written): the notebooks named above
(train_boundary_p4_gamma0_a1/_a2/_a3.ipynb) and the CFG assignment strings
this section quotes in prose (acct1/acct2/acct3) were subsequently rewritten
by the terminology sweep in section 10 to slice1/slice2/slice3, inside the
notebook files themselves. This section's own prose still reads acct1/
acct2/acct3 above and is left that way deliberately — it describes what
those runs were actually tagged as at execution time, and rewriting it to
match the notebooks' current wording would misrepresent history. Anyone
cross-referencing this section's seed-coverage table against the current
notebook files by assignment name needs the acct-to-slice mapping: acct1 =
slice1, acct2 = slice2, acct3 = slice3 (identity mapping, name only). See
section 10 and scripts_provenance.md Part K item K-a for the open question
this raises: whether the literal strings acct1/acct2/acct3 appear anywhere
else this section doesn't already cover (the resume registry, or the
committed CSV's own row-level metadata, as opposed to this ledger's prose)
— NOT resolved by this addendum, only narrowed to those two remaining
locations.

## 10. 09 Aug 2026 — comment/terminology/tone sweeps, all working-tree .py/.ipynb files

### Comment/terminology/tone sweeps — DELIVERED (two Claude Code sessions;
working-tree edits only, no commit; full file-by-file detail lives in
docs/scripts_provenance.md Part K, not duplicated here per the ledger/
provenance split — this entry is the pointer plus the facts a ledger reader
needs without opening that file)

**Purpose.** Two sequential sweeps across all 47 in-scope .py/.ipynb files
(25 .py: 19 tracked + 6 untracked; 22 .ipynb: 14 tracked + 8 untracked).
Sweep 1: stale account/acctN terminology renamed to slice/sliceN (filename
and CSV-output-path tokens excluded and left verbatim); repeated
protocol-description blocks collapsed to a single pointer line; LOCAL-only
document references (master plan §n, ledger §n, bare P#/I#/B#/G# labels)
stripped from comments and docstrings, broad interpretation (every
plan-internal label, not just literal citations; real filenames like
diag_b5_gamma06.csv and code identifiers like B5_GAMMA never touched).
Sweep 2, run after Sweep 1: narrated, non-declarative prose (third-person
narration by name, self-justifying framing, onboarding explanation,
narrative/slogan framing, residual AI-authorship traces) rewritten to flat
declarative statements; standing project-status vocabulary (PROPOSED,
DECIDED, OPEN, UNVERIFIED) and functional pitfall warnings explicitly
protected and left untouched.

**Scope discipline.** revision_master_plan_v2.md and revision_artifact_ledger.md
(this file) were out of scope for both sweeps and were never opened by
either. No .json config file was opened. The G3 anonymity sweep's subject
matter (real Kaggle usernames, dataset-owner paths, the b1-block-attn
dataset name, the revision/reviewer-yc7L branch name, Critical-N reviewer
IDs) was confirmed untouched by both — a separate, not-yet-run task. Neither
sweep ran git add, git commit, or git push at any point; every edit exists
only as an uncommitted working-tree change or, for untracked files, only on
local disk.

**Files touched.** Sweep 1: 8 .py files, 15 notebooks edited; 24 files
(mostly .py) had no in-scope matches. Sweep 2: 0 .py files required an edit
(all reviewed content judged factual or already terse); 6 notebooks edited,
16 confirmed already clean. Both sweeps touch a strict subset of the same
47-file inventory; scripts_provenance.md Part K §5 gives the exhaustive
per-file table for both.

**Interaction with section 9 (gamma-0 tranche).** The three gamma-0
notebooks this ledger's section 9 documents
(train_boundary_p4_gamma0_a1/_a2/_a3.ipynb) were edited by both sweeps: the
CFG assignment strings acct1/acct2/acct3 quoted in section 9's prose were
renamed to slice1/slice2/slice3 inside the notebook files (Sweep 1), and the
"Deviation" markdown paragraph plus the Session-budget paragraph were
rewritten for tone (Sweep 2). Section 9's own prose was NOT rewritten to
match — see the addendum appended to section 9 above. The gamma-0 tranche's
underlying results, the committed merge (results_boundary_p4_gamma0_complete.csv),
and the oracle PASS recorded in section 9 are unaffected by either sweep;
nothing about the data, the schedule, or the verified numbers changed.

**Recurring tooling defect (scripts_provenance.md Part K §4, summarized
here because it bears directly on an artefact this ledger tracks).**
Claude Code's NotebookEdit tool silently drops a code cell's stored
`outputs`/`execution_count` when replacing that cell's contents, unless
explicitly resupplied. Hit twice, independently, on
notebooks/train_block_attention.ipynb — the one file touched by either
sweep that is both tracked (section 2 above, COMMITTED 777c462) and carries
real execution outputs (9 stored outputs). Both occurrences were caught via
a git-diff check filtered to output_type/execution_count lines and repaired
by transplanting the affected cell's outputs/execution_count from
`git show HEAD:...` into the edited file. Verified via the same filtered
diff post-repair: zero output_type/execution_count changes, all 9 outputs
confirmed matching HEAD's 9 in both cases. Every other notebook either sweep
touched had zero stored outputs at edit time (unrun templates), so the
defect carried no data-loss risk for them. Any future sweep touching an
executed, tracked notebook must anticipate this and verify the same way.

**Known accepted incompleteness.** The Session-budget paragraph's closing
clause ("...so the version saves its checkpoints and registry") was agreed
during Sweep 2 to be flattened further but the fuller wording was settled
only after the first of four affected files was already edited; all four
(train_boundary_p4.ipynb and the three gamma0_a1/a2/a3 files) now carry
identical, internally-consistent, but not-fully-flattened wording. No
functional or provenance consequence — flagged here per the update
discipline (UNVERIFIED/known-gap items are recorded, not silently dropped)
rather than because it blocks anything.

**Open item carried from scripts_provenance.md Part K, sharpened here.**
K-a asked whether the acct1/acct2/acct3 strings renamed by Sweep 1 are
referenced anywhere in the ledger, the resume registry, or any committed
CSV's metadata. This ledger's own section 9 answers the first of those three
directly: yes, in prose, at lines documenting the gamma-0 tranche's seed
coverage — now cross-referenced via the section 9 addendum above rather than
left as an unexamined risk. The resume-registry and committed-CSV-metadata
parts of K-a remain unchecked. Part K's K-a entry in scripts_provenance.md
should be updated to reflect this partial resolution; not done automatically
here since that is a separate LOCAL file and this pass was scoped to the
ledger only — flagged for a deliberate follow-up edit there.

**Verification.** Every edited .py file byte-compiled cleanly after editing.
Every edited .ipynb file re-parsed as valid JSON after editing. A repo-wide
residual-pattern sweep (terminology tokens for Sweep 1; a literal
banned-phrase list for Sweep 2) ran across every edited file at the close of
each sweep, with every hit individually triaged — English-idiom false
positives ("account for", "accounting") and code-identifier false positives
(B5_GAMMA, B5_PROBE_N) were both correctly excluded rather than flagged.
Temporary verification artifacts (signature snapshot files, a pristine-HEAD
backup copy) created during Sweep 1 were deleted before that sweep closed,
confirmed via git status --porcelain. Both sweeps closed with a
git status --porcelain check across the full affected tree confirming no
file was touched beyond its reported edit list and no stray files remained.

**Status.** DELIVERED, uncommitted. No further action required for either
sweep to be considered closed. Remaining open items (the Session-budget
wording gap, the narrowed K-a question, three untracked config_*.json files
of unaccounted-for origin noted during cleanup) are cosmetic or
informational, not blocking, and are tracked in scripts_provenance.md Part K
§7 rather than duplicated here.

K-a — CLOSED 2026-08-09. Confirmed: literal "acct1"/"acct2"/"acct3" strings
absent from results_boundary_p4_gamma0_complete.csv (grepped directly, 0
hits) and from all three pre-merge per-slice registries
(gamma0_a1/a2/a3/results_boundary_p4.csv, grepped directly, 0 hits).
merge_boundary_p4_gamma0.py's own _source tag used "gamma0_a1/a2/a3"
naming, not "acct1/2/3", and that column is dropped before the merged
file is written — no schema field exists in either the per-slice or
merged CSV that could carry an account-style string. Provenance risk for
the closed gamma-0 tranche's self-description vs. actual execution-time
tagging: resolved, no remediation needed.
