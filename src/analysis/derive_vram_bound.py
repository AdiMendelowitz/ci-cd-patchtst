"""Materialised-attention VRAM bound for PatchTST-CD on ECL, worked arithmetic.

Reads nothing: every number is derived analytically from committed
architecture constants (C, seq_len, patch_size, stride, num_heads,
num_layers, dtype). No results CSV is read or required, the same
provenance posture as derive_theoretical_bounds.py (B7) -- this is a
derivation, not a measurement.

Method
------
Flattening every variate's patches into one CD attention sequence gives
C * N tokens (main.tex, Sec. 2, Table 1). A naive (non-fused) attention
implementation materialises the full score matrix of shape
(B, num_heads, C*N, C*N) and keeps it resident for the backward pass,
one copy per encoder layer under standard (non-checkpointed) autograd.
That is the O(C^2 N^2) term the paper's cost section already names; this
script turns it into bytes for the reviewer's requested C = 321 (ECL)
worked example.

    bytes_per_layer = batch * num_heads * (C*N)**2 * bytes_per_element
    bytes_all_layers = bytes_per_layer * num_layers

This is a bound on the materialised-attention term alone, not a full
training-memory estimate (it excludes parameters, gradients, optimizer
state, patch-embedding and FFN activations). It is deliberately narrow
so it isolates the one term the reviewer asked about.

Cross-check against measured anchors
-------------------------------------
Two real Stage-0 probe measurements exist for ecl_cd/ecl_cd_block under
the current (fused-attention) PyTorch environment on a real T4
(dryrun_stage0.py, account REDACTED, block-attn-dryrun dataset,
12 Aug run -- corrected architecture: seq_len=96, d_model=128, 16 heads,
matching Table 1). A first probe run (03 Aug) measured 11.67 GiB / OOM at
this same nominal token count, but that run silently used the synthetic-
family architecture (seq_len=512, d_model=64, 8 heads, C*N=20223) instead
of ECL's own -- a real bug, fixed in dryrun_stage0.py, and NOT the anchor
used below. The anchors below are from the corrected, architecture-
verified run only.

    batch 128: peak 8.134 GiB, no OOM (892.9 s/epoch)
    batch   8: peak 0.534 GiB, no OOM

Fused / flash-attention kernels never materialise the (C*N) x (C*N)
score matrix (they use online softmax with O(C*N) memory per head), so
the naive bound computed here is expected to sit far above these
measured totals. That gap is the arithmetic support for the claim in
response_to_reviewer_yc7L_v6.md: current-environment feasibility comes
from kernel fusion removing this term, not from the term itself having
shrunk, and the O(C^2 N^2) compute cost (unaffected by fusion) is what
still makes a full ECL-CD run close to (not dramatically beyond) a single
Kaggle session's wall-clock budget.

Writes
------
results/vram_bound.csv -- machine-readable, human-gated (a paper
decision, not this script's, whether or how any row is used in the
response or main.tex).
"""

from pathlib import Path

import pandas as pd

# ECL CD architecture constants (main.tex Table 1; train_ecl.ipynb /
# ecl_ci_cd_train.ipynb CONFIG; N derived from seq_len/patch_size/stride).
C = 321
SEQ_LEN = 96
PATCH_SIZE = 16
STRIDE = 8
NUM_HEADS = 16
NUM_LAYERS = 3
BYTES_PER_ELEMENT = 2  # FP16

GIB = 1024 ** 3

# Measured 12 Aug anchors, corrected (real ECL) architecture, current
# (fused-attention) environment, single T4 (dryrun_stage0.py; ledger Â§13).
MEASURED_ANCHORS = {
    128: {"peak_gib": 8.134, "sec_per_epoch": 892.9, "oom": False},
    8: {"peak_gib": 0.534, "sec_per_epoch": 894.9, "oom": False},
}

T4_VRAM_GIB = 16.0


def num_patches(seq_len: int, patch_size: int, stride: int) -> int:
    return (seq_len - patch_size) // stride + 1


def materialized_attention_bytes(batch: int, tokens: int, num_heads: int,
                                  num_layers: int, bytes_per_element: int) -> tuple[float, float]:
    """(bytes for one layer, bytes for num_layers layers retained simultaneously)."""
    per_layer = batch * num_heads * (tokens ** 2) * bytes_per_element
    return per_layer, per_layer * num_layers


def crossover_batch(tokens: int, num_heads: int, num_layers: int,
                     bytes_per_element: int, budget_gib: float) -> float:
    """Largest real-valued batch at which num_layers layers' materialised
    attention alone would consume budget_gib."""
    budget_bytes = budget_gib * GIB
    per_batch_element = num_heads * (tokens ** 2) * bytes_per_element * num_layers
    return budget_bytes / per_batch_element


def main() -> int:
    N = num_patches(SEQ_LEN, PATCH_SIZE, STRIDE)
    tokens = C * N
    print(f"C={C}  seq_len={SEQ_LEN}  patch_size={PATCH_SIZE}  stride={STRIDE}  N={N}")
    print(f"C*N={tokens}  num_heads={NUM_HEADS}  num_layers={NUM_LAYERS}  "
          f"dtype=FP16 ({BYTES_PER_ELEMENT} bytes/elem)\n")

    rows = []
    print("=== Materialised-attention bound: O(C^2 N^2) score matrix, all heads ===")
    for batch in (8, 128):
        per_layer_b, all_layers_b = materialized_attention_bytes(
            batch, tokens, NUM_HEADS, NUM_LAYERS, BYTES_PER_ELEMENT
        )
        per_layer_gib = per_layer_b / GIB
        all_layers_gib = all_layers_b / GIB
        anchor = MEASURED_ANCHORS.get(batch, {})
        measured_gib = anchor.get("peak_gib")
        measured_str = (f"{measured_gib:.3f} GiB measured (real T4, fused)" if measured_gib is not None
                         else "OOM measured (fused)" if anchor.get("oom") else "no anchor")
        ratio_str = (f"  ratio(3L/measured)={all_layers_gib / measured_gib:.1f}x"
                     if measured_gib is not None else "")
        print(f"  batch={batch:>3}: 1 layer = {per_layer_gib:8.2f} GiB   "
              f"{NUM_LAYERS} layers = {all_layers_gib:8.2f} GiB   vs {measured_str}{ratio_str}")
        rows.append({
            "batch": batch, "tokens_C_N": tokens, "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS, "bound_1layer_GiB": per_layer_gib,
            "bound_alllayers_GiB": all_layers_gib,
            "measured_fused_GiB": measured_gib,
            "measured_sec_per_epoch": anchor.get("sec_per_epoch"),
            "measured_fused_OOM": anchor.get("oom", False),
        })

    xover_1layer = crossover_batch(tokens, NUM_HEADS, 1, BYTES_PER_ELEMENT, T4_VRAM_GIB)
    xover_alllayers = crossover_batch(tokens, NUM_HEADS, NUM_LAYERS, BYTES_PER_ELEMENT, T4_VRAM_GIB)
    print(f"\nCrossover batch size at which the materialised bound alone reaches "
          f"{T4_VRAM_GIB:.0f} GiB (T4 budget):")
    print(f"  1 layer:              batch ~= {xover_1layer:.1f}")
    print(f"  all {NUM_LAYERS} layers retained:  batch ~= {xover_alllayers:.1f}")

    b128 = next(r for r in rows if r["batch"] == 128)
    print(f"\nSummary: at batch=128, one encoder layer's materialised score matrix alone "
          f"({b128['bound_1layer_GiB']:.1f} GiB) already exceeds the entire T4 budget "
          f"({T4_VRAM_GIB:.0f} GiB), and all {NUM_LAYERS} layers together would need "
          f"{b128['bound_alllayers_GiB']:.1f} GiB -- roughly "
          f"{b128['bound_alllayers_GiB'] / MEASURED_ANCHORS[128]['peak_gib']:.1f}x the "
          f"{MEASURED_ANCHORS[128]['peak_gib']:.2f} GiB actually measured on a real T4 under "
          "fused attention at the same batch size, on the correct ECL architecture. The gap "
          "is attributable to fused/flash-attention kernels never forming the (C*N) x (C*N) "
          "matrix in memory (O(C*N) working memory per head via online softmax). This "
          "environment is memory-feasible at every batch tried, including the largest "
          "(128); the binding constraint is wall-clock, not VRAM.")

    out_path = Path("results/vram_bound.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists, refusing to overwrite silently")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nwrote {len(rows)} rows to {out_path}")

    # Real check, not an unconditional claim: the whole argument rests on
    # the materialised bound exceeding every measured anchor. Verify it
    # rather than asserting PASS by print alone.
    failures = [
        r for r in rows
        if r["measured_fused_GiB"] is not None and r["bound_alllayers_GiB"] <= r["measured_fused_GiB"]
    ]
    if failures:
        raise AssertionError(
            f"materialised bound did not exceed the measured anchor for "
            f"{len(failures)} row(s): {failures} -- the kernel-fusion argument "
            f"requires bound > measured at every batch; investigate before trusting "
            f"either number."
        )
    print("\nRESULT: PASS -- materialised bound exceeds every measured fused-attention "
          "anchor at the same batch size, as expected from the kernel-fusion argument, "
          "on the architecture the paper actually uses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
