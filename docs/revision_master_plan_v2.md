# TMLR Revision Master Plan v2.9 — Paper10341 (reviewer yc7L)

Status: post-critique final; v2.1 amendments (02 Aug, post dry-run): P3 batch
policy for new-environment families; B2 numeric two-tier trigger applied
twice; B5/W6 mechanism wording rule; W4 section title; Stage 0 status split
into completed first half and pending extension. v2.2 amendments (03 Aug,
post B1): B1 complete with results recorded in the ledger; I1 confirmed from
main.tex; V1 resolved; Stage 0 extension delivered as a sibling script; B2
trigger note from committed best_epoch data; G1 evidence in hand pending the
analyze script. v2.3 amendments (03 Aug, post Stage 0 T4 run): G0 measured
costs recorded (boundary and lf closed, ECL pending); B2 trigger (a)
applied -> tier 3, with a PROPOSED (undecided) budget-expansion reopening
of (b); B3 fallback triggered, P=2 family batch 64, checkpointing
dismissed; G1 precondition met (analyze_block_attention.py oracle PASS
local); OQ3 scope shrunk by the tier outcome; pre-adoption /critique
pass on this draft (7 findings fixed). v2.4 amendments (04 Aug): G0 FULLY
CLOSED per the ledger (ECL priced late 03 Aug; OQ4 resolved); the v2-draft
recovery completed 03 Aug and the response chain now runs v2 -> v5 (ledger
§4; v5 fingerprint-verified 04 Aug); boundary training notebooks retrieved
from Kaggle and committed under notebooks/original/ (I1's no-notebook
clause superseded; ledger §6, provenance Part I); boundary window count
precised to 13,393 (13,392 consumed per epoch at batch 8); boundary
DGP/windowing finding recorded (W-track disclosure joins the Friday gate);
B2 launched 04 Aug at the standing tier 3 (gamma 0.9, CD then CI, session-1
tripwires green). v2.5 amendments (05 Aug): title version corrected (v2.3
stood on a document already carrying v2.4 amendments); branch tip updated;
B2 measured-convergence data recorded as trigger-(b) input, NOT applied;
cross-account resume verified and the §7 registry line corrected to match;
AR(1) grid two-series-length finding recorded; W-track boundary-DGP
disclosure EXECUTED and removed from the Friday gate; G3 sweep list extended;
B4 analysis script confirmed at full-oracle scope, not the scoped
narrowing. v2.6 amendments (07-08 Aug): gamma-0 P=4 tranche launched across
three accounts (LAUNCH RECORD); CONVERGENCE RE-PRICING GATE and HARD STOP
added ahead of gamma {0.6, 0.3} scheduling; gamma-0 progress updated from
"no completed runs" to 2/5 CD seeds DONE with the remaining 3 verified
cleanly resumed; the CONVERGENCE RE-PRICING GATE's P=8/P=16-based
slower-convergence prediction checked against the two completed gamma-0
seeds — direction confirmed, magnitude (~1.14-1.23x) lower than the
1.4-1.8x extrapolation; internal line-number cross-references replaced
with named-section references so future edits cannot desync them. v2.7
amendments (09 Aug, /critique pass): STALE-STATUS FIX — the gamma-0 P=4
CD+CI tranche is now complete at 5/5 seeds (not the 2/5-DONE, n=2
partial-read state this document carried); LAUNCH RECORD and the
CONVERGENCE RE-PRICING GATE below updated accordingly, the HARD STOP
re-pricing decision itself left OPEN, undecided by this pass. MISSING-REVIEW
FIX — a second TMLR review (reviewer MoZG) arrived 08 Aug, requesting a
closed-form theoretical-optimal-MSE addition and an accessibility rewrite of
the abstract/introduction/terminology; this falsifies Assumption A1 below and
was previously unrecorded in this document. Both fixes source from
retro_paper10341-revision_20260808.md and the same-day handoff, the most
recent artefacts available at patch time; neither introduces new
experimental numbers beyond what those two documents already state. v2.8
amendments (09 Aug): gamma-0.6 LAUNCH PLAN added to §6 — decided, seed
split confirmed, pending execution; budget re-priced off the completed
gamma-0 row (ledger §9) as a bracket rather than a point estimate, with an
explicit per-account risk check and the standing cross-account resume
mitigation; the CI arm's non-optional status stated explicitly against the
existing committed gamma-0.6 CI rows, which remain a reproducibility
cross-check only, never a substitute. v2.9 amendments (09 Aug, /critique
pass): STALE-REFERENCE FIX — §6's gamma-0 LAUNCH RECORD and gamma-0.6
LAUNCH PLAN entries quote acct1/acct2/acct3 as CFG assignment strings;
a same-day terminology sweep (scripts_provenance.md Part K; ledger
section 10) renamed those exact strings inside the notebook files
themselves to slice1/slice2/slice3, without this plan being reconciled
against it at the time. Neither entry was rewritten (both remain accurate
historical record of what those runs were assigned); a naming addendum was
appended after the LAUNCH PLAN entry instead, giving the acct-to-slice
identity mapping and narrowing the still-open question (whether the old
strings also appear in the resume registry or committed-CSV metadata) to
scripts_provenance.md item K-a. The §6 note flagging scripts_provenance.md's
05-Aug-vs-08-Aug currency gap was updated to record that both files were
further extended 09 Aug; this plan itself was not reconciled against either
extension until this pass. Owner: Adi.
Repo: ci-cd-patchtst.
Main certified at c04d3cb (29 pass, 0 warn, 0 fail); work happens on
revision/reviewer-yc7L (tip fd3a863, confirmed 05 Aug by `git rev-parse HEAD`
= fd3a8635fd42feba609a9ac16a0fab588a85e931; 777c462 was the B1 commit and is
now three commits behind). Docs policy 03 Aug:
this plan and its three sibling docs are maintained LOCAL-ONLY from v2.2
onward; the committed v2.1 snapshots are frozen (see ledger).

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
  A1 FALSIFIED 08 Aug: a second review (reviewer MoZG) arrived, accepting
  the accuracy and cost claims and requesting two additions — a closed-form
  theoretical-optimal-MSE derivation (no new training runs) and a full
  accessibility rewrite of the abstract, introduction, and terminology.
  Neither item is scheduled against the writing track yet (retro action A1,
  due 11 Aug). Whether a third review is still pending is itself unchecked
  since 02 Aug — re-run the V1 forum check (§12) before treating the review
  set as final.
- Response draft exists with [DECIDE] stubs; Critical 1 resolves to option B
  (evidence-primary) pending B1 numbers; the variate-token reimplementation
  remains declined on scope, quoting the submission. (03 Aug: the v2 draft
  is currently lost locally, recoverable from the project chat — recovery
  owed, see ledger §4.) [RESOLVED 03 Aug: recovered byte-faithfully;
  superseded by v3-v5 — v5 is canonical, fingerprint-verified 04 Aug.]
- Dry-run measurements (torch 2.13, Kaggle T4, AMP):
  lf_c21: CD 38.1 s/epoch, CD_Block 9.2 s/epoch, both batch 128.
  ar1_c84: CD 512.6 s/epoch peak 6.13 GiB, CD_Block 69.1 s/epoch peak
  6.21 GiB, both batch 128.
- Environment finding: committed CSVs record CD batch 1 at C=84 and batch 8
  at C=21; current torch holds batch 128 at both. (Step count at C=84 CD is
  7,433 for seeds {42,123,456} and 8,033 for seeds {789,1011} — the grid ran
  on two series lengths; see §12 OQ5. Earlier wording here gave 8,033 as if
  uniform.)
  Cause: fused SDPA no longer materialises the (C*N)^2 score matrix. Memory
  and batch claims are environment-bound; the quadratic-in-C compute
  asymmetry survives (38 -> 513 s/epoch for C 21 -> 84 at fixed batch).
- Boundary protocol: P=2 uses stride 1 (stated in main.tex; N=511,
  C*N=10,731). I1 CONFIRMED (03 Aug): main.tex at the branch tip states the
  sweep uses S = P/2 and gives the P=4 arithmetic C*N = 21x255 = 5355, so
  P=4 uses stride 2 and its CD cost is approximately the measured ar1_c84
  ~513 s/epoch. (No train_boundary notebook exists in the repo — the README
  discloses this — so the paper text is the confirming source.)
  [SUPERSEDED 04 Aug: the original boundary training notebooks were
  retrieved from Kaggle and committed under notebooks/original/ with
  content-based names, alongside the identity-free folder record
  notebooks/original/README.md; see ledger §6 and provenance Part I. Four of
  the six are byte-identical to the retrieved files; two carry a same-day
  comment reword, so the folder README's hashes and Part I's differ for those
  two (Part I §6). The paper text remains the confirming source for the
  protocol as published.] Stage 0
  correction 03 Aug: per-STEP confirmed (8.7 vs 8.1 s/step at matched C*N);
  per-EPOCH is 915.7 s, 1.67x the ar1 figure, because boundary epochs carry
  13,393 train windows (13,392 consumed per epoch at batch 8, drop_last;
  pinned 04 Aug from the retrieved notebooks) vs ar1's 8,033 — see §6 B2 G0 application.
- ECL protocol (from train_ecl.ipynb): proportional split 15840/5256/5256 of
  26352 rows (reference constants, applied proportionally; realized
  15,811/26,304 on the actual file — provenance CSV 7), z-score fit on train only, ddof=0. Test windows at H=96: 4649.
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

Status 03 Aug (v2.3): first half complete (lf_c21 and ar1_c84, CD and
CD_Block, batch 128; results in the ledger). Extension DELIVERED as a
sibling script dryrun_stage0.py (deliberate deviation from "extend": the
committed probe stays untouched); critiqued and smoke-tested. Uploaded and
run 03 Aug as a block-attn-dryrun dataset version; boundary and lf pricing
in the ledger; ecl_cd OOM at batch 128 recorded, batch-64 rung still
running (outcome pending, kill-decision by GPU-utilization check) -- G0
open for B6 only. [SUPERSEDED late 03 Aug: the rung printed; ECL priced;
G0 FULLY CLOSED — ledger dryrun_stage0 entry.] The lf batch-8 cells are cross-checks only — B1's live
run measured them on real data (CD ~820 s/run, CD_Block ~340 s/run at
batch 8).

Original extension spec (implemented 03 Aug as the sibling script;
retained for the record). Extend dryrun_block_attention.py with: CI mode;
lf_c21 CD_Block at batch 8
(closes the B1 gap); boundary_p4_c21 (stride 2 per I1) and boundary_p2_c21
(stride 1) for CD and CI; ecl_cd and ecl_cd_block (C=321, contiguous groups
of 3, ECL window counts). Output per (config, mode): feasible batch, warmup
and steady s/epoch, peak GiB, projected hours per converged run. G0 output:
the finalised run matrix below with measured costs and any triggered
fallbacks.

## 6. Run matrix (priority order; owner; estimate until G0 replaces it)

- B1 Leader-follower three-arm: COMPLETE 03 Aug. CD_Block + CI + CD, gamma
  {0,.3,.6,.9} x 5 seeds, committed lf protocol; measured ~8.4 h total (vs
  the ~10-20 h estimate). 60/60 runs; results and B5 diagnostics in the
  ledger. Outcome: block attention narrows but does not close the CD
  penalty (Blk/CI 1.0045-1.0056 vs CD/CI 1.0065-1.0082 at gamma>0);
  gamma=0 control passed. Served Critical 1, Recommended 2, P1 lf baseline,
  and B5 as planned. Canonical analysis: analyze_block_attention.py
  (oracle PASS local 03 Aug; commit owed at G2).
- B2 Boundary P=4 completion: CD + CI, gamma grid x 5 seeds, boundary
  protocol, batch 128 per P3. Owner: collaborator C1. Estimate via I1: ~35
  converged epochs x ~513 s x 20 CD runs plus cheap CI runs, ceiling
  60-100 h; split across accounts by gamma if OQ3 requires. Numeric trigger,
  applied twice and allowed to improve but never worsen mid-block:
  (a) at G0 on the 35-epoch ceiling, (b) after the first seed's runs reveal
  realistic early-stopped convergence. Tiers: total <= 70 collaborator-hours
  -> full grid; 70-100 -> gamma {0,.6,.9}; > 100 -> existence-proof runs at
  one gamma, remainder argued on the environment-dependence ground.
  Pre-G0 note (03 Aug, ledger-consulted): committed boundary CD best_epoch
  max is 12, so realistic convergence is ~22 epochs, not 35; the ceiling
  drops to ~22 x 513 s x 20 ~ 63 h < 70, provisionally the full-grid tier.
  Formal application still happens at G0 with the probe's measured s/epoch.
  G0 APPLICATION (a), 03 Aug, measured 915.7 s/epoch at batch 128:
  20 CD x 5.60 h = 112.0 h + 20 CI x 1.22 h = 24.4 h -> ~136 h total.
  Middle tier also fails (gamma {0,.6,.9}: ~102 h > 100); CD alone
  exceeds 100 h. TIER: > 100 -> existence-proof runs at one gamma
  (5 CD + 5 CI ~ 34 h), remainder argued on the environment-dependence
  ground. Trigger (b) remains open but cannot plausibly improve the
  tier (middle tier needs <= 21 converged epochs vs 22 projected; full
  grid needs ~9). Caveat: 22-epoch projection borrows P=8 CD
  best_epoch max 12 + patience; B1 calibration suggests projections
  understate live cost by a per-epoch val/checkpoint overhead that is
  proportionally small at this epoch length.
  PROPOSED 03 Aug (assistant proposal, NOT yet decided; decide by
  Friday 07 Aug): revise the tier thresholds by plan amendment. The
  70/100 h thresholds encoded a cross-block allocation cap that pricing
  has since collapsed (B3: ~220-234 h demand -> ~30 h fallback; B6
  vanilla-CD: ~215 h/run x 1-2 dismissed as runs), freeing roughly 200 h
  of assumed demand. This is a threshold revision, NOT a trigger-(b)
  application — (b) improves only via measured convergence and, at the
  22-epoch projection, cannot lift the tier under the written thresholds.
  Sequencing gammas 0.9 -> 0 -> 0.6 -> 0.3 makes every stopping point a
  pre-declared tier either way; trigger (b) is applied Friday 07 Aug on
  measured gamma-0.9 convergence against whatever thresholds then stand.
  If adopted, record the amendment here; the tier-3 verdict above remains
  the standing application-(a) result until then. Account provenance
  caveat: if the extra accounts are Adi-operated rather than C1/C2,
  multi-accounting violates Kaggle ToS and a mid-block ban strands
  checkpoints (R3-class risk, flagged in-session; §7's compliance line
  stands).
  TRIGGER (b) INPUT, 05 Aug — MEASURED, NOT YET APPLIED. Three of five
  gamma-0.9 CD runs are complete. Epochs to stop (best_epoch + patience 10)
  are 28 / 20 / 23 at seeds 42 / 123 / 456, mean 23.7 against the 22-epoch
  projection (+7.7%). At the G0-measured 915.7 s/epoch that is 7.12 / 5.09 /
  5.85 h, total 18.06 h, mean 6.02 h/run against the 5.60 h projection
  (+7.5%) — the direction and size the B1 calibration caveat predicted.
  Scaling the measured mean to five seeds gives ~30.1 h of CD; the CI arm
  prices at ~3.3 h for five seeds (87.7 s/epoch over the committed
  gamma-0.9 CI best_epochs + patience). One gamma therefore costs ~33.4 h,
  against the ~34 h projected at G0. Extrapolated: three gammas ~100.2 h,
  four gammas ~133.6 h. Under the written thresholds the full grid fails
  clearly and the middle tier fails by 0.2% — a margin measurement cannot
  separate from the threshold, so trigger (b) as written neither lifts nor
  confirms the tier at any useful confidence, and the PROPOSED threshold
  revision remains the only lever that changes the outcome. Apply (b)
  formally on Friday 07 Aug against whatever thresholds then stand; the two
  remaining CD seeds should land first, since the middle-tier margin turns
  on them. Caveats: the CI figure is projected from committed best_epochs,
  not measured in this environment; the extrapolation assumes other gammas
  converge like gamma 0.9, which is untested.
  SCHEDULING NOTE 05 Aug: only ~17.6 h of the gamma-0.9 tranche remains
  (CD 789 ~7.4 h, CD 1011 ~6.9 h, five CI runs ~3.3 h), which fits inside
  one account's weekly quota. Cross-account resume was verified end to end
  on 05 Aug (registry and checkpoint travel as a private dataset; the
  receiving session bootstrapped and resumed mid-run), so the assignment is
  no longer constrained to the launching account.
-  CONVERGENCE RE-PRICING GATE, 07 Aug — applies to the gamma {0, 0.3, 0.6}
  tranche if the threshold revision (the "PROPOSED 03 Aug" threshold-
  revision paragraph above) is adopted. The "TRIGGER (b) INPUT, 05 Aug"
  paragraph above already flags that the trigger-(b) extrapolation
  assumes other gammas converge like gamma 0.9 at P=4, untested. The
  already-committed P=8/P=16 boundary data (results_boundary.csv, CD mode,
  5 seeds/cell) speaks to that assumption and argues against it:
    best_epoch mean, CD:       gamma0   gamma0.3   gamma0.6   gamma0.9
      P=8                       7.0      7.4        5.2        4.2
      P=16                      6.8      6.0        6.0        5.8
  At P=8, gamma 0 and gamma 0.3 converge ~1.4-1.8x slower than gamma 0.9; at
  P=16 the effect is weaker but the same direction. No cell shows a lower
  gamma converging faster than gamma 0.9. Applying the measured gamma-0.9
  per-run cost (6.02 h/run, the "TRIGGER (b) INPUT" paragraph above)
  uniformly to the remaining gammas therefore likely UNDERSTATES gamma-0
  and gamma-0.3 cost — which matters because the three-gamma total these
  gammas roll up to is already ~100.2 h against the original 100 h
  middle-tier cap (the "G0 APPLICATION (a)" tier table above, a 0.2% miss),
  the total the revised threshold needs to comfortably clear for the
  revision to actually unblock the grid. If the revised cap gives less
  headroom than the ~200 h freed in the "PROPOSED 03 Aug" paragraph above
  suggests, this pattern would erode it.
  UPDATE 08 Aug, SUPERSEDED same day by the full row (see below): partial
  n=2 gamma-0 data first suggested the direction was confirmed but the
  magnitude (~1.14-1.23x) was well under the 1.4-1.8x this section
  extrapolated from P=8/P=16. That n=2 read is now known to have undersold
  the real variance — retained here only as the audit trail for what was
  believed before the full row landed.
  UPDATE 08 Aug, FULL ROW: the complete 5-seed gamma-0 CD row shows
  per-seed convergence-ratio variance of 0.57x-2.50x against gamma-0.9 —
  not the clean multiplier either the P=8/P=16 extrapolation (1.4-1.8x) or
  the n=2 partial read (~1.14-1.23x) implied. Mean-of-ratios (1.24x) and
  ratio-of-means (1.03x) disagree substantially, so no single number
  re-prices the remaining tranches without a stated choice of which
  summary statistic to trust. See the B2 LAUNCH RECORD FALSIFIABLE-
  PREDICTION CHECK above for the full figures.
  HARD STOP — DATA GAP CLOSED, RE-PRICING DECISION STILL OPEN: gamma-0
  (running first, per the planned 0.9->0->0.6->0.3 sequencing stated in
  the "PROPOSED 03 Aug" paragraph above) has completed at 5/5 seeds, so
  the data this gate was waiting on is now in hand. The gate does NOT
  advance itself to a decision: given the real per-seed scatter above, a
  single re-pricing multiplier is not well defined, and Adi has not yet
  chosen an approach (conservative multiplier vs. a variance band vs.
  waiting for more data — retro task list item 2, 2026-08-09). Do not
  schedule gamma 0.6 or gamma 0.3 until that choice is made and applied.
  The tier-3 existence-proof fallback (one gamma, ~33.4 h measured vs
  ~34 h projected, per the tier table and "TRIGGER (b) INPUT" paragraphs
  above) remains available regardless of which re-pricing approach is
  chosen, if the re-priced total exceeds it.
  Caveat: the P=8/P=16 signal is suggestive, not a P=4-specific
  measurement; n=5 seeds/cell, no significance test run on the gamma
  effect. The P=4 gamma-0 row itself is now n=5 and still shows no clean
  multiplier, so the caveat about small-n unreliability applies less to
  "is P=4 slower" and more to "by how much, uniformly."
- LAUNCH RECORD, 07 Aug — gamma-0 CD+CI tranche launched across three
  accounts in parallel (deviates from §7's shared-config-dataset
  distribution model; CFG embedded per notebook copy instead of pulled
  from an attached dataset — see §7 exception note). Notebook variants
  held locally at
  C:\Users\adime\Documents\ci-cd-patchtst\notebooks\train_boundary_p4_gamma_0\
  (train_boundary_p4_gamma0_a1/_a2/_a3.ipynb), not yet committed.
    acct1 (assignment acct1-tranche2-gamma0-CD-a, CFG sha256[:16]
      5ed9b8adc93851ab): budget 10.5h, CD seeds 42, 123.
    acct2 (assignment acct2-tranche2-gamma0-CI-plus-CD-c, CFG sha256[:16]
      12e22d4ac7490369): budget 8.3h, CI seeds 42/123/456/789/1011, then
      CD seed 1011.
    acct3 (assignment acct3-tranche2-gamma0-CD-b, CFG sha256[:16]
      32bbe85a15e2f697): budget 11.5h, CD seeds 456, 789.
  Seed coverage check: CD seeds across accounts = {42,123} + {1011} +
  {456,789} = {42,123,456,789,1011}, the full canonical set, no
  duplicates or gaps. CI: all 5 seeds on acct2 alone.
  Cross-account consistency verified from session logs: models.py and
  models_cd_block.py sha256[:16] identical across all three accounts
  (1eecbddbdf130528 / 8703aeed74877b1a), confirming a single
  b1-block-attn version in use. Protocol tripwires (windows 13393,
  spe 104 at b128, C*N=5355) also identical across all three, as
  expected (gamma-independent).
  STATUS 08 Aug (supersedes the 07 Aug "all three in-flight, no completed
  runs yet" line, which was accurate only at launch): FULL ROW COMPLETE —
  5 of 5 gamma-0 CD seeds DONE, all resumes verified clean end to end.
    acct1: seed 42 DONE mse=0.9883, best_epoch 21, 31 epochs total, 8.22 h
      (epoch time 948-956 s). Seed 123: resumed clean from checkpoint
      ([resume] log line matched the prior stopping point exactly) and
      finished this session; per-seed final numbers are recorded in the
      ledger's gamma-0 entry (§9), not restated here to avoid a second
      copy of the same figures drifting out of sync.
    acct2: all 5 CI seeds DONE (results match the 05 Aug ledger figures
      exactly). CD seed 1011: resumed clean from checkpoint ([resume] log
      line verified) and finished (epoch time ~892-896 s, fastest of the
      three accounts); final numbers in ledger §9.
    acct3: seed 456 DONE mse=0.9607, best_epoch 16, 26 epochs total,
      6.64 h (epoch time 908-922 s). Seed 789: resumed clean from
      checkpoint ([resume] log line verified) and finished; final numbers
      in ledger §9.
  All resumes were verified clean (Kaggle bootstrap correctly seeded the
  registry and checkpoint from each notebook's own prior-version output,
  skipped completed runs, and the [resume] log line matched the prior
  session's stopping point exactly) — not merely attached and re-launched
  from scratch.
  FALSIFIABLE-PREDICTION CHECK, UPDATED 08 Aug on the full 5-seed row
  (supersedes the n=2 partial read below): gamma-0 best_epoch IS higher
  than gamma-0.9's on every seed, confirming the predicted direction, but
  per-seed convergence-ratio variance is far larger than the n=2 read
  suggested — 0.57x-2.50x across the five seeds, not a clean multiplier.
  Mean-of-ratios (1.24x) and ratio-of-means (1.03x) disagree substantially,
  which the n=2 partial read (seed 42: 1.17x; seed 456: 1.23x, both inside
  a narrow ~1.14-1.23x band) could not have shown. CD/CI ratio at gamma=0
  is 0.9997, matching the expected internal-control result (three modes
  should coincide when gamma=0 removes the coupling gamma is meant to
  probe). Superseded n=2 read, retained for the audit trail: "for the two
  now-complete matched seeds, gamma-0 best_epoch IS higher than
  gamma-0.9's — seed 42: 21 vs 18 (ratio 1.17x); seed 456: 16 vs 13 (ratio
  1.23x)... measured per-run wall-clock ratio is ~1.14-1.16x... not
  1.4-1.8x." That partial read undersold the real variance — see the
  Insights section of retro_paper10341-revision_20260808.md, which names
  this the second occurrence of that pattern (the first being the
  P=8/P=16-based projection this gate itself extrapolated from).
  Precise per-seed test_mse/test_mae/best_epoch values for seeds 123, 789,
  1011 are not restated in this plan; see ledger §9, which is this
  project's single source for per-run numbers (P1: numbers traceable to a
  committed CSV, never duplicated by hand across documents).
  Ledger and scripts_provenance updates: ledger §9 added this pass;
  scripts_provenance.md still carries a currency gap (dated content stops
  05 Aug against this plan's 08 Aug) — reconciliation owed, unchanged by
  this pass since scripts_provenance.md was not a critique target today.
  UPDATE 09 Aug: both files have since been extended further —
  scripts_provenance.md gained Part K (comment/terminology/tone sweeps
  across all working-tree .py/.ipynb files) and the ledger gained section
  10 (same subject, ledger-level summary). This plan itself was not
  reconciled against either at that time; the acct1/acct2/acct3 staleness
  that update introduced into this section is addressed by the note
  immediately below rather than folded in here, to avoid two notes saying
  the same thing in one entry.
  OPEN (undecided): config_boundary_p4.json still describes tranche 1
  (assignment acct2-tranche1-gamma09) and is unused by this tranche;
  whether it stays as a historical record or gets annotated superseded
  is undecided — see G3 scope. Same open item is independently tracked in
  the ledger (§6/§7 b1-block-attn dataset entries) with no pointer between
  the two; resolve once, here or there, and point the other at it.
- GAMMA-0.6 LAUNCH PLAN, DECIDED 09 Aug — pending execution (measure-then-
  commit step between the HARD STOP data-gap closure above and any
  gamma-0.3 decision). Same three-account structure as the gamma-0 launch,
  seed continuity preserved:
    acct1 (assignment acct1-tranche3-gamma06-CD-a): CD seeds 42, 123.
      22:20h available this week.
    acct2 (assignment acct2-tranche3-gamma06-CI-plus-CD-c): CI seeds
      42/123/456/789/1011, then CD seed 1011. 27:20h available.
    acct3 (assignment acct3-tranche3-gamma06-CD-b): CD seeds 456, 789.
      28:30h available.
  Total available this week: 78:10h (78.17h). Session self-budget stays at
  11.0h per notebook, unchanged from every prior tranche — that cap guards
  the 12h Kaggle kill per session, not the weekly quota above; a seed
  running past 11h resumes into a second session on the same verified
  engine, same as seed 1011 did at gamma-0.9. CFG embedded per notebook
  copy, continuing the gamma-0 exception (§7) rather than reverting to the
  shared-config-dataset model.
  CI ARM IS NOT OPTIONAL, STATED EXPLICITLY so it is never later assumed
  skippable: gamma=0.6 already has CI-only rows in the committed, original-
  environment results_boundary_p4_ci.csv (old-env half-width 0.0268,
  already wired into analyze_boundary_p4.py's reproducibility oracle). That
  file cannot substitute for this tranche's CI arm — P1 forbids pairing a
  new-environment CD row against an old-environment CI row, since the two
  ran on different torch/attention-kernel environments (documented in §1's
  environment finding). This launch's own CI seeds are what the paired
  CD-CI contrast requires; the committed old rows remain what they always
  were, a separate cross-environment reproducibility check, never pooled
  into the paired test.
  BUDGET CHECK, re-priced off the gamma-0 row (ledger §9) rather than the
  original P=8/P=16 extrapolation: applying the gamma-0 tranche's two
  summary ratios to the gamma-0.9 measured cost (~33.4h/gamma: ~30.1h CD +
  ~3.3h CI, five seeds, per §6) gives ~34.4h (ratio-of-means, 1.03x) to
  ~41.4h (mean-of-ratios, 1.24x) for one additional gamma. Both bracket
  ends leave comfortable headroom against 78:10h available — even the
  high end leaves ~37h free this week for B4 (~6-12h estimate) alongside
  it. This bracket, not a single number, is the honest output of the
  0.57x-2.50x per-seed scatter already on record; treating either end as
  the answer would repeat the partial-sample-read pattern this project has
  now corrected twice.
  PER-ACCOUNT RISK, not covered by the tranche-level budget check above:
  the 0.57x-2.50x scatter is per-seed, so a tranche-level average can still
  mask an account-level shortfall. acct1 carries the lowest weekly budget
  of the three (22:20h) — this is the risk driver, not its baseline cost
  (acct1's two gamma-0.9 seeds, 42 and 123, total 12.21h, actually less
  than acct3's 456+789 pair). Even with the lower baseline, acct1's
  smaller cushion means a single seed landing at the observed 2.50x
  outlier alone pushes its pair to ~22.9h — already just past its 22:20h
  ceiling on its own, before accounting for anything else; both seeds
  landing there simultaneously (~30.5h) would exceed it by about 37%. This
  is a real, quantified risk, not a hypothetical one, since gamma-0
  already produced a 2.50x outlier somewhere in its own row. Mitigation:
  the cross-account resume path is verified twice now (05 Aug, and again
  this session for gamma-0) — if acct1 threatens to exceed its budget
  mid-run, the registry and checkpoint travel to acct2 or acct3 (both with
  more headroom) rather than the run stalling. No seed/account
  reassignment is being forced ahead of the fact.
  SEQUENCING: this tranche's own measured cost, not the bracket above,
  is what re-prices gamma-0.3 — same measure-then-commit principle the
  HARD STOP gate already applies between gamma-0.9 and gamma-0. No
  gamma-0.3 launch decision is made by this entry.
  Kaggle quota resets Saturday morning; nothing above assumes this
  tranche must finish before then; slower-than-projected gamma-0.6 runs
  simply continue into the refreshed quota via the resume engine, the
  same way multi-session runs always have.
  NAMING ADDENDUM, 09 Aug (not a correction to either entry above — both
  stand as written; this is a forward pointer, per this project's own
  discipline of never rewriting historical record in place): a
  comment/terminology sweep across all working-tree .py/.ipynb files
  (scripts_provenance.md Part K; ledger section 10) renamed the CFG
  assignment strings inside the notebook files themselves — acct1/acct2/
  acct3 became slice1/slice2/slice3 — in both the gamma-0 tranche notebooks
  (train_boundary_p4_gamma0_a1/_a2/_a3.ipynb, LAUNCH RECORD above) and the
  gamma-0.6 tranche notebooks this LAUNCH PLAN entry describes. Every
  acct1/acct2/acct3 reference in both entries above (LAUNCH RECORD and
  GAMMA-0.6 LAUNCH PLAN, including the CFG sha256 fingerprints, which are
  unaffected by the string rename and remain valid as recorded) describes
  what those runs are and were actually assigned; the notebooks now
  self-report the identical assignments under the new names. Mapping is
  identity, name only: acct1 = slice1, acct2 = slice2, acct3 = slice3.
  Anyone cross-referencing this section against the current notebook files
  by assignment name needs this mapping. Whether the literal strings
  acct1/acct2/acct3 also appear in the resume registry or any committed
  CSV's row-level metadata (as opposed to this plan's and the notebooks'
  own prose, both now accounted for) remains an open question — tracked as
  scripts_provenance.md Part K item K-a, not resolved by this addendum.
- B3 Boundary P=2 partial: CD + CI at gamma {.6,.9} x 5 seeds if priced in;
  else one gamma as an existence proof that the cell completes under fused
  attention. Owner: C2. Gradient checkpointing flag available if the probe
  shows P=2 needs it.
  G0 APPLICATION, 03 Aug: P=2 CD measures 22.0-23.4 h/run -> the
  priced-in option (2 gammas x 5 seeds: ~220 h at batch 8, ~234 h at
  the standardised batch 64) is dismissed either way; fallback stands:
  one-gamma existence proof, single CD run spans 2-3 sessions on the
  resume engine, labelled single-seed in any reported number. Protocol
  consequence per P3: CI OOMs at the fixed batch 128, so the P=2 family
  standardises on measured-feasible batch 64 for both modes, stated in
  the text. Gradient-checkpointing flag DISMISSED: peak 6.18 GiB at
  batch 64 fits; the binding constraint is time, which checkpointing
  worsens.
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
  G0 NOTE, 03 Aug: ecl_cd OOM at batch 128 is itself a measured datum for
  the current-torch infeasibility status; the batch-64 rung (running) and
  ecl_cd_block pricing complete the picture. The p2-scaled bound
  (>= ~4.3 h/epoch at batch 64, ~215 h for a 50-epoch worst case, single
  estimate) suggests the vanilla-CD item is satisfied argumentatively by
  the probe rather than by a run; decide when the rung prints.

## 7. Engineering spec (one engine, all notebooks)

- Atomic per-epoch checkpointing to /kaggle/working: model, optimiser,
  scaler, epoch, and RNG states (torch, cuda, numpy, python).
- Session resume reconstructing mid-run state; safe against the 12 h cap;
  multi-session runs (B3, B6 CD) depend on it.
- Idempotent run registry: CSV of (cell, mode, seed, status, best_val,
  checkpoint path); a restarted notebook skips completed runs; results
  appended via temp-write-and-rename.
- One canonical registry filename per block, carried across accounts rather
  than kept per-account: the notebook globs the exact registry name in its
  attached inputs and asserts a single match, so the registry and its
  checkpoints travel with the work. Verified end to end 05 Aug on B2 across
  two accounts. Per-session download copies are tagged by resume point
  (<registry>_<MODE>_s<seed>.csv) and are strict supersets of their
  predecessors; the highest-tagged file is the record. Static block
  assignment remains the coordination mechanism; Adi merges across blocks
  manually and the merge script checks for duplicate (cell, mode, seed) keys.
  (Earlier wording said "registry is per-account", which the B2 engine does
  not do and never did.)
- Distribution: one shared private Kaggle dataset (code + configs),
  collaborators added as collaborators; results return as notebook outputs.
  Exception: the gamma-0 P=4 tranche launched 07 Aug embeds CFG per
  notebook copy instead of a shared config dataset — see B2 LAUNCH RECORD.
- Memory levers: AMP plus fused SDPA (automatic); gradient checkpointing
  behind a flag, off unless the probe shows P=2 needs it (03 Aug: probe
  shows it does not — see B3 G0 application).
- Compliance: each collaborator operates their own account personally. This
  resolves the R3-class multi-accounting caveat in the B2 "PROPOSED 03 Aug"
  paragraph's account-provenance note (extra
  accounts are C1/C2-operated, not Adi-operated); no further action pending
  on that caveat.
 
Rationale: the B2 "PROPOSED 03 Aug" paragraph's account-provenance caveat
says "§7's compliance line
stands" without stating that the compliance line actually answers it.
Nothing operational changes; this only closes an apparent open item that
isn't open.

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
  constraints". (03 Aug: Stage 0 adds the P=2 CI OOM-at-128 and the
  ecl_cd OOM-at-128 as concrete current-torch data points.)
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
  triggered or dismissed; I1 confirmed or corrected. Status 03 Aug:
  CLOSED for lf and boundary (B2/B3 applications recorded in §6); OPEN
  for ECL (B6 unpriced until the running rung prints). [04 Aug: CLOSED
  for ECL too — FULLY CLOSED; ledger.]
- G1 B1 analysed -> Critical 1 response wording frozen. Status 03 Aug
  (v2.3): analyze_block_attention.py oracle PASS on Adi's machine (22/22,
  exit 0) — the P4 precondition is met; the freeze itself executes at W6,
  since the gate's object is the response wording, which lands with the
  recovered response v2.
- G2 All planned CSVs committed; analyze scripts with passing oracle checks;
  certification script extended to run the new analyses; certification
  green on the branch.
- G3 Branch merged to main; certification green on main; anonymity sweep
  (name, GitHub URL, Kaggle usernames, local paths) passes; single mirror
  cut; Phase 2 submission. Named sweep items so far: the as-ran
  notebooks/train_block_attention.ipynb (hard-coded INPUT_ROOT embeds the
  Kaggle username; restore Path("/kaggle/input") in the mirror copy); the
  docs/ directory (local-only from 03 Aug; the stale v2.1 snapshots
  committed at 8451b5d ship scrubbed or docs/ is excluded from the mirror);
  and (added 05 Aug) the B2 notebook's cross-session bootstrap, whose three
  print statements emit full attached-input paths and so expose two Kaggle
  usernames in cell output — the dataset owner's and the prior session's.
  Printing `.name` instead of the path makes that leak structurally
  impossible rather than merely stripped, and should be applied before the
  next session rather than at the mirror. Committed notebooks currently
  carry no outputs, so nothing has leaked yet; that safety depends on
  remembering to strip, which is what the fix removes.
  DOCS ITEM CLOSED 05 Aug: all three tracked docs untracked via
  `git rm --cached` (`--sparse` needed for two, which carried the
  skip-worktree bit). The `/docs/` ignore rule was already at HEAD and had
  been inert throughout, since ignore rules do not apply to tracked paths —
  `git rm --cached` was always the operative step, not the rule. A
  full-history grep of docs/ for Kaggle usernames, kaggle.com URLs and the
  local Windows path returned empty at every commit, so the frozen v2.1
  snapshots are stale rather than identifying and no history rewrite is
  warranted. Commit execution and push are UNVERIFIED (ledger §7).
  NEW SWEEP ITEM 05 Aug — commit author metadata. Every commit carries a
  full personal name and a personal email. No file-level sweep reaches
  this. The mirror must therefore be built by `git init` on a copy of the
  working tree under an anonymous identity, never by pushing or cloning
  this history; P6's "one mirror cut" now means one fresh repository, not
  one filtered push. Whether the current origin is public is OPEN and
  decides whether this is a future or a present exposure.

## 10. Risks and mitigations

- R1 Estimates off by factor two -> P2 gates every block on measurement;
  fallbacks pre-declared per block. (03 Aug instance: pre-G0 B2 estimate
  missed 1.79x; the gate worked as designed — no block had started.)
- R2 Session death mid-run -> checkpoint/resume engine; registry
  idempotence; multi-session runs designed for it.
- R3 Collaborator unavailability or account trouble -> static assignment
  makes any block reassignable; checkpoints travel via dataset if needed.
- R4 A1 falsifies (new review arrives) -> V1 monitors; re-triage before
  submission; spent compute remains useful as robustness evidence.
  TRIGGERED 08 Aug: reviewer MoZG's review arrived. It accepts the
  accuracy and cost claims (no new critical items against the GPU track)
  and adds two writing-track items (theoretical-optimal-MSE, accessibility
  rewrite) — spent and in-flight compute remains fully useful, consistent
  with the mitigation. Re-triage action: confirm via V1 (§12) whether a
  third review is still pending before treating the set as final.
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
- Status note 08-09 Aug: gamma-0 P=4 CD+CI tranche complete at 5/5 seeds
  (§6 LAUNCH RECORD); HARD STOP gate's data gap closed, re-pricing
  decision still OPEN. Second review (reviewer MoZG) arrived 08 Aug (§1,
  R4); its two writing-track items (theoretical-optimal-MSE,
  accessibility rewrite) are not yet sequenced against W1-W6 above —
  unscheduled as of this pass, retro action A1 due 11 Aug. The Week 4
  buffer above predates MoZG's added scope; whether the original 3-4 week
  Phase 1 ETA still holds is an open question, not yet re-assessed.
- Status note 05 Aug: B2 continuing on a third account after verified
  cross-account resume; three of five gamma-0.9 CD runs complete; ~17.6 h
  remaining. The W-track boundary-DGP disclosure is EXECUTED — four edits
  applied to main.tex adding tab:synth_protocols over the four synthetic
  families, scoping the gamma=0 anchor, retracting the per-seed-spread
  explanation of the P=16 boundary/sweep gap, and scoping the grid protocol
  sentence — so Friday 07 Aug now carries only trigger (b), the PROPOSED
  thresholds, and the mirror exclusion list. B4 amendment: its analysis
  script receives a FULL oracle on the G1 pattern (analyze_block_attention.py
  scale, 22/22), not the scoped narrowing PROPOSED earlier the same day. The
  narrowing was rejected because it would put a coverage caveat into the
  provenance chain behind a response whose Critical 1 is itself about whether
  the controls are what the paper claims. Coverage list in ledger §7.
- Status note 04 Aug: B2 gamma-0.9 tranche launched (tier 3; session 1 on
  a collaborator account, tripwires green); boundary notebooks retrieved
  and COMMITTED 04 Aug at 2b0a885 (the earlier "commit prepared" wording
  predated execution); README rebuilt (v2).
- Status note 03 Aug: Week 1 items complete except Stage 0's ECL cell;
  Week 2's B2/B3 scale and assignment follow the §6 G0 applications and
  the pending §6 PROPOSED decision (Friday 07 Aug).

## 12. Remaining open items

- OQ1' RESOLVED 03 Aug: I1 confirmed from main.tex at the branch tip
  (S = P/2; C*N = 21x255 = 5355 stated for P=4).
- OQ3 Collaborator availability windows (shapes B2/B3 scheduling and the
  gamma-split decision). Scope shrunk 03 Aug: the B2 tier-3 outcome cuts the
  collaborator ask from ~136 h to ~34 h (plus ~30 h for B3). LARGELY CLOSED
  05 Aug: measurement puts the gamma-0.9 tranche at ~33.4 h with ~17.6 h
  remaining, which fits one account's weekly quota, and verified
  cross-account resume means a stranded tranche can be moved rather than
  restarted. What remains open is only B3's assignment and the availability
  question for whichever account takes it.
- OQ5 NEW 05 Aug: the AR(1) grid ran on two series lengths — seeds
  {42,123,456} on 13,400 usable timesteps (7,433 training windows) and seeds
  {789,1011} on 14,400 (8,033). Recovered exactly from recorded
  steps_per_epoch and confirmed independently by the gamma=0 cross-experiment
  anchor, which reproduces to 4 dp where the window counts match and diverges
  where they do not. Paired per-seed statistics are unaffected; absolute
  Table 3 cell means mix two series lengths, and main.tex tab:batch_sizes
  reports only the 7,433 family as if it were the protocol. Blocks: the
  tab:batch_sizes disclosure decision. Full record in provenance Part J §2
  and ledger §7.
- OQ4 NEW 03 Aug: ecl batch-64 rung outcome (session running). RESOLVED
  late 03 Aug: the rung printed (ledger dryrun_stage0 measurements); B6
  priced; G0 fully closed.
- V1 RESOLVED 02 Aug: single reviewer confirmed on the form's reader list
  at Phase-1 posting; A1 stood at that check. SUPERSEDED 08 Aug: A1 has
  since falsified (§1, R4) — the second review (MoZG) arrived. The forum
  re-check this line deferred to "before Phase 2 submission" is now the
  live question of whether a third review is also pending (retro task
  list item 4, 2026-08-09); not yet run as of this pass.

## 13. Deliverables checklist (endgame)

- [x] Phase 1 official comment posted 02 Aug (writing-only items + ETA)
- [~] results_block_attention.csv COMMITTED 03 Aug (777c462);
      analyze_block_attention.py DELIVERED 03 Aug, oracle PASS local
      (22/22); script commit + cert extension + README row + test smoke
      owed at G2
- [~] results_boundary_p4.csv — the canonical registry name, which the
      notebook's bootstrap glob depends on; per-session download copies are
      tagged results_boundary_p4_<MODE>_s<seed>.csv and the highest-tagged
      file is the record. Scope tracks the §6 B2 tier decision; under
      standing tier 3 this is the one-gamma existence-proof CSV. UPDATED
      08 Aug: gamma-0.9 CD+CI complete (5/5 seeds, ledger §8); gamma-0
      CD+CI also complete (5/5 seeds, ledger §9) — the tranche now covers
      two gammas, ahead of the standing tier-3 one-gamma floor, pending
      the HARD STOP re-pricing decision (§6) before gamma {0.6, 0.3} are
      scheduled; P=2 decided infeasible by wall-clock, not run; boundary
      analysis update (skeleton delivered per P4, oracle PASS on the
      gamma-0.9 row per ledger §8)
- [ ] results_etth1_rerun.csv + etth1_coupling_partition.csv + subgroup
      analyze script
- [ ] results_ecl_block.csv + ECL vanilla-CD current-torch status note
- [ ] B5 instrumentation CSV + mechanism panel + Section 5 text
- [ ] main.tex: W1-W5 integrated
- [ ] README, tests, and certify_clean_clone.ps1 extended to cover every
      new script and CSV
- [ ] response_to_reviewer_yc7L final, stubs resolved, reviewer order
- [ ] NEW 08 Aug: reviewer MoZG's two items — theoretical-optimal-MSE
      derivation (tractability confirmed for the AR(1)-grid and
      leader-follower families per scripts_provenance Part B constants;
      the boundary family's iid-innovation DGP is unscoped, see retro A5)
      and the accessibility rewrite of abstract/introduction/terminology —
      not yet sequenced into W1-W6 above (retro action A1, due 11 Aug)
- [ ] G3: merge, certification 0, anonymity sweep, mirror cut, submission