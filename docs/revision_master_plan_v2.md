# TMLR Revision Master Plan v2.1 — Paper10341 (reviewer yc7L)

Status: post-critique final; v2.1 amendments (02 Aug, post dry-run): P3 batch
policy for new-environment families; B2 numeric two-tier trigger applied
twice; B5/W6 mechanism wording rule; W4 section title; Stage 0 status split
into completed first half and pending extension. Owner: Adi. Repo: ci-cd-patchtst.
Main certified at c04d3cb (29 pass, 0 warn, 0 fail); work happens on
revision/reviewer-yc7L (tip ebba4f2).

## 1. State of play (verified this session)

- One review received (yc7L, 02 Aug 2026): both TMLR acceptance questions
  Yes; conditional-accept recommendation; three critical revisions, five
  recommended items, two additional-comment asks (related-work subsection;
  formal VRAM calculation).
- Assumption A1: the review set is complete. Verification action V1 (not an
  assumption to rest on): check the OpenReview forum's reviewer-assignment
  status; TMLR normally collects three reviews before an AE recommendation,
  so a lone review is unusual and the forum will show whether more are
  pending. Re-triage before any submission if A1 falsifies.
- Response draft exists with [DECIDE] stubs; Critical 1 resolves to option B
  (evidence-primary) pending B1 numbers; the variate-token reimplementation
  remains declined on scope, quoting the submission.
- Dry-run measurements (torch 2.13, Kaggle T4, AMP):
  lf_c21: CD 38.1 s/epoch, CD_Block 9.2 s/epoch, both batch 128.
  ar1_c84: CD 512.6 s/epoch peak 6.13 GiB, CD_Block 69.1 s/epoch peak
  6.21 GiB, both batch 128.
- Environment finding: committed CSVs record CD batch 1 at C=84 (8033
  steps/epoch) and batch 8 at C=21; current torch holds batch 128 at both.
  Cause: fused SDPA no longer materialises the (C*N)^2 score matrix. Memory
  and batch claims are environment-bound; the quadratic-in-C compute
  asymmetry survives (38 -> 513 s/epoch for C 21 -> 84 at fixed batch).
- Boundary protocol: P=2 uses stride 1 (stated in main.tex; N=511,
  C*N=10,731). Working inference I1: P=4 uses stride 2 (the P/2 convention;
  N=255, C*N=5355, nearly identical to the measured ar1_c84 token count, so
  P=4 CD cost is approximately already measured at ~513 s/epoch). Confirm I1
  from train_boundary notes at Stage 0; the probe measures it regardless.
- ECL protocol (from train_ecl.ipynb): proportional split 15840/5256/5256 of
  26352 rows, z-score fit on train only, ddof=0. Test windows at H=96: 4649.
- ETTh1 coupling partition computed and pre-registered in the script's
  docstring: diffed metric primary (high/low ratio 1.44), raw as robustness
  (Spearman -0.28 makes the choice substantive), regime-confound limitation
  stated, balanced per-horizon splits.

## 2. Goals, ranked

1. The most rigorous revision available: complete the reviewer's primary
   critical ask (P in {2,4} cells) if Stage 0 prices it in; the block-wise
   CD ablation; mechanistic instrumentation; ETTh1 subgroup analysis; an
   honest environment-dependence treatment of the cost claims.
2. Every number traceable to a committed CSV plus an analysis script with an
   oracle check (the repo's native pre-registration).
3. Anonymity protected for the mirror; certified main protected.
4. Real budgets respected: ~30 GPU-h/week per account across three accounts
   (Adi plus two collaborators, each operating their own account); Adi at
   roughly two evenings plus one weekend slot per week; the job search keeps
   the standing priority claim on prime hours. Every track boundary is a
   safe pause point.

## 3. Binding design principles

- P1 Within-environment contrasts only. Every new table pairs modes trained
  on torch 2.13 in the same session family; new rows never pair against
  committed rows. A cell existing in both environments is reported as a
  reproducibility measurement, never pooled.
- P2 Probe before commit. No run block starts before its per-run cost is
  measured. B1 is exempt on stated grounds: its configurations were measured
  by the completed dry run, except CD_Block at batch 8, which Stage 0 adds.
- P3 Protocol identity for comparability, amended for batch under P1. Since
  new tables never pair against committed rows, new torch-2.13 families
  standardise on the measured-feasible batch (128) rather than inheriting the
  legacy hardware-forced batches (8 at C=21, 1 at C=84), which were artefacts
  of the materialised-attention environment. Exception: B1's CD-family arms
  run at the committed batch 8 to stay closest to the committed sweep, with
  the choice stated in the paper text; CI in B1 stays at its committed 128.
  Schedule, early stopping, and seeds {42,123,456,789,1011} remain identical
  everywhere. Within any one table, all arms share one batch policy.
- P4 Analysis before results. Each experiment's analyze script is written
  before its runs finish; oracle checks added once numbers exist.
- P5 Zero-interaction collaborator notebooks: find inputs, consult registry,
  resume or start, checkpoint atomically, emit results as outputs.
  Collaborators press Run All and never edit code.
- P6 One mirror cut, at the endgame trigger, after the anonymity sweep.
- P7 The response promises only completed work.

## 4. Communication strategy (two-phase, clock-driven)

- Phase 1 (target: within ~1 week of the review): post an official comment
  answering every writing-only item in reviewer order (Critical 3 accepted
  with the hardened scope text quoted; Critical 2's formalised hypotheses
  and behavioural-validity argument; recommended items 3-5 acceptances; the
  related-work and VRAM commitments), stating plainly that the block-wise
  ablation and grid-completion experiments are running and giving the
  revision ETA. This respects the discussion window while the compute lands
  and converts the reviewer's fallback offers into acknowledged agreements.
- Phase 2: the full revision plus final point-by-point response once G2
  passes. Nothing in Phase 1 promises specific numbers; nothing in Phase 2
  is new to the reviewer in kind.

## 5. Stage 0 — measurement probe (gate G0; one session, Adi's account)

Status: first half complete (the original probe measured lf_c21 and ar1_c84,
CD and CD_Block, batch 128; results in the artefact ledger). The extension
below remains, and it ships as a new version of the block-attn-dryrun Kaggle
dataset, never as in-notebook patches.

Extend dryrun_block_attention.py with: CI mode; lf_c21 CD_Block at batch 8
(closes the B1 gap); boundary_p4_c21 (stride 2 per I1) and boundary_p2_c21
(stride 1) for CD and CI; ecl_cd and ecl_cd_block (C=321, contiguous groups
of 3, ECL window counts). Output per (config, mode): feasible batch, warmup
and steady s/epoch, peak GiB, projected hours per converged run. G0 output:
the finalised run matrix below with measured costs and any triggered
fallbacks.

## 6. Run matrix (priority order; owner; estimate until G0 replaces it)

- B1 Leader-follower three-arm: CD_Block + CI + CD, gamma {0,.3,.6,.9} x 5
  seeds, committed lf protocol. Owner: Adi. ~10-20 h. Serves Critical 1
  fallback, the P1 baseline for all lf-family contrasts, and hosts B5.
- B2 Boundary P=4 completion: CD + CI, gamma grid x 5 seeds, boundary
  protocol, batch 128 per P3. Owner: collaborator C1. Estimate via I1: ~35
  converged epochs x ~513 s x 20 CD runs plus cheap CI runs, ceiling
  60-100 h; split across accounts by gamma if OQ3 requires. Numeric trigger,
  applied twice and allowed to improve but never worsen mid-block:
  (a) at G0 on the 35-epoch ceiling, (b) after the first seed's runs reveal
  realistic early-stopped convergence. Tiers: total <= 70 collaborator-hours
  -> full grid; 70-100 -> gamma {0,.6,.9}; > 100 -> existence-proof runs at
  one gamma, remainder argued on the environment-dependence ground.
- B3 Boundary P=2 partial: CD + CI at gamma {.6,.9} x 5 seeds if priced in;
  else one gamma as an existence proof that the cell completes under fused
  attention. Owner: C2. Gradient checkpointing flag available if the probe
  shows P=2 needs it.
- B4 ETTh1 rerun with per-window error logging: CI + CD x 4 horizons x 5
  seeds, committed ETTh1 protocol (batch 32 both modes). Owner: Adi.
  ~6-12 h. Joined to the committed partition CSV by the subgroup analysis.
- B5 Mechanistic piggyback on B1's gamma=0.6 cell: per-epoch gradient norms
  and encoder-output participation ratio on a fixed probe batch. Near-zero
  marginal cost. No attention-weight extraction (it forces the materialising
  path and reintroduces the memory wall the paper no longer needs to pay).
- B6 ECL: CD_Block x 4 horizons x 5 seeds at near-CI cost, pre-declared as
  an exploratory arm because 321 channels admit no natural grouping and the
  contiguous groups-of-3 partition is neutral rather than structure-aligned;
  plus 1-2 multi-session vanilla-CD runs at H=96 to settle the current-torch
  status of the infeasibility claim. Owner: C2 after B3.

## 7. Engineering spec (one engine, all notebooks)

- Atomic per-epoch checkpointing to /kaggle/working: model, optimiser,
  scaler, epoch, and RNG states (torch, cuda, numpy, python).
- Session resume reconstructing mid-run state; safe against the 12 h cap;
  multi-session runs (B3, B6 CD) depend on it.
- Idempotent run registry: CSV of (cell, mode, seed, status, best_val,
  checkpoint path); a restarted notebook skips completed runs; results
  appended via temp-write-and-rename.
- Registry is per-account; the static block assignment is the cross-account
  coordination; Adi merges result CSVs manually and the merge script checks
  for duplicate (cell, mode, seed) keys.
- Distribution: one shared private Kaggle dataset (code + configs),
  collaborators added as collaborators; results return as notebook outputs.
- Memory levers: AMP plus fused SDPA (automatic); gradient checkpointing
  behind a flag, off unless the probe shows P=2 needs it.
- Compliance: each collaborator operates their own account personally.

## 8. Writing track (parallel, no GPU)

- W1 Critical 3 scope hardening: abstract, introduction, conclusion,
  limitations.
- W2 Section 6 expansion: untested regimes enumerated; the CD-overfitting
  mechanism named as the primary future-work target.
- W3 Related-work subsection consolidating iTransformer, LIFT, Abdelmalak
  et al. 2026 (all already cited; complementarity framing per the earlier
  analysis).
- W4 Cost-section rewrite: pin the training-time environment; measured
  batch/steps remain facts of the runs that produced the accuracy numbers;
  the requested VRAM derivation written as the materialised-attention bound
  with an explicit fused-kernel caveat; current-torch footnote sourced from
  Stage 0 and B6; section note titled "environment dependence of resource
  constraints".
- W5 Figure and table annotations (recommended item 4) implemented inside
  the analyze scripts so every printed value regenerates from canonical
  CSVs; no hand-typed numbers.
- W6 Response finalisation: Critical 1 stub resolved to option B once B1 is
  analysed; Critical 2 upgraded to partial-accept with B5 evidence plus the
  behavioural-validity paragraph; reviewer-order preserved throughout.
  Wording rule for all mechanism text: the B5 panel is presented as
  "consistent with hypothesis Hx" and never as explaining the mechanism; the
  reviewer's formalised-hypotheses framing remains the ceiling of the claim
  even with the panel in hand.

## 9. Gates

- G0 Probe complete -> run matrix finalised with measured costs; fallbacks
  triggered or dismissed; I1 confirmed or corrected.
- G1 B1 analysed -> Critical 1 response wording frozen.
- G2 All planned CSVs committed; analyze scripts with passing oracle checks;
  certification script extended to run the new analyses; certification
  green on the branch.
- G3 Branch merged to main; certification green on main; anonymity sweep
  (name, GitHub URL, Kaggle usernames, local paths) passes; single mirror
  cut; Phase 2 submission.

## 10. Risks and mitigations

- R1 Estimates off by factor two -> P2 gates every block on measurement;
  fallbacks pre-declared per block.
- R2 Session death mid-run -> checkpoint/resume engine; registry
  idempotence; multi-session runs designed for it.
- R3 Collaborator unavailability or account trouble -> static assignment
  makes any block reassignable; checkpoints travel via dataset if needed.
- R4 A1 falsifies (new review arrives) -> V1 monitors; re-triage before
  submission; spent compute remains useful as robustness evidence.
- R5 Mixed-provenance statistics -> P1 forbids; every new table audited
  against P1 at G2.
- R6 Deanonymisation in the mirror -> G3 sweep list executed on the exact
  merge commit.
- R7 ETTh1 rerun drifts from committed Table 4 -> reported as a
  reproducibility measurement under P1; drift beyond committed CIs becomes
  an explicit environment note, never a silent replacement.
- R8 TMLR clock outpaces experiments -> Phase 1 comment lands within the
  week regardless of GPU progress.

## 11. Timeline (Adi-hours capped at ~2 evenings + 1 weekend slot per week)

- Week 1: V1 forum check; Stage 0; Phase 1 comment drafted and posted; B1
  started; W1-W3 drafted; partition script + CSV committed to the branch.
- Week 2: B2 (C1) and B3 (C2) running; B4; W4-W5; analysis scripts for all
  blocks written per P4.
- Week 3: B6; oracle checks; W6; G2 certification on the branch.
- Week 4 buffer: G3 endgame (merge, certify, sweep, mirror, Phase 2
  submission). If experiments finish early, the buffer collapses.

## 12. Remaining open items

- OQ1' Confirm I1 (P=4 stride 2) from train_boundary notes; the probe
  measures the config either way.
- OQ3 Collaborator availability windows (shapes B2/B3 scheduling and the
  gamma-split decision).
- V1 OpenReview reviewer-assignment status (scheduled, week 1).

## 13. Deliverables checklist (endgame)

- [ ] Phase 1 official comment posted (writing-only items + ETA)
- [ ] results_block_attention.csv + analyze_block_attention.py + oracle
- [ ] results_boundary_p4_full.csv (and P=2 partial) + boundary analysis
      update
- [ ] results_etth1_rerun.csv + etth1_coupling_partition.csv + subgroup
      analyze script
- [ ] results_ecl_block.csv + ECL vanilla-CD current-torch status note
- [ ] B5 instrumentation CSV + mechanism panel + Section 5 text
- [ ] main.tex: W1-W5 integrated
- [ ] README, tests, and certify_clean_clone.ps1 extended to cover every
      new script and CSV
- [ ] response_to_reviewer_yc7L final, stubs resolved, reviewer order
- [ ] G3: merge, certification 0, anonymity sweep, mirror cut, submission
