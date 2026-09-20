"""T4 feasibility dry run for block-wise cross-variate attention CD.

Measures, per requested configuration: the largest feasible batch size, peak
CUDA memory at that batch, mean wall-clock seconds per epoch over a short timed
run, and a projected wall-clock for a full training run. Data are random
tensors shaped exactly like the real experiments; memory and speed do not
depend on data values, so no generator import is needed and the script is
self-contained for a Kaggle cell, with models_cd_block.py uploaded beside it
(the only import outside the standard stack):

    %run dryrun_block_attention.py --config lf_c21 --config ar1_c84
    %run dryrun_block_attention.py --config lf_c21 --modes CD_Block --epochs 3

The first epoch carries CUDA kernel autotune and allocator warmup, so it is
timed separately and excluded from the steady-state mean and the projection
whenever more than one epoch runs.

The training step mirrors the paper's engine: AdamW, AMP autocast with
GradScaler, and gradient-norm clipping at 1.0, so measured memory matches what
a real run would allocate. Nothing is written to disk; results print as a
summary table.
"""

import argparse
import gc
import time

import torch
import torch.nn as nn

from models_cd_block import PatchTST_CD_Block, contiguous_groups, leader_follower_groups, num_patches

# Architecture and schedule constants, identical to the training notebooks.
_D_MODEL = 64
_N_HEADS = 8
_N_LAYERS = 3
_DROPOUT = 0.2
_LR = 1e-4
_WEIGHT_DECAY = 1e-4
_GRAD_CLIP = 1.0
_SEQ_LEN = 512
_PRED_LEN = 96
_PATCH = 16
_STRIDE = 8
# Windows in the synthetic training split: 8640 - 512 - 96 + 1.
_N_TRAIN_WINDOWS = 8033
_FULL_RUN_EPOCHS = 35  # typical CI-style run length under patience 10.

# name -> (C, partition builder). Partition choice is part of the design:
# pair-preserving for leader-follower, generator-aligned blocks of 7 for C=84.
_CONFIGS = {
    "lf_c21": (21, leader_follower_groups),
    "ar1_c84": (84, lambda: contiguous_groups(84, 7)),
}
_BATCH_LADDER = (128, 64, 32, 16, 8, 4, 2, 1)


class _VanillaCD(nn.Module):
    """Flattened-token CD, reproduced verbatim from the training notebook for a
    same-session baseline measurement."""

    def __init__(self, num_variates: int) -> None:
        super().__init__()
        n = num_patches(_SEQ_LEN, _PATCH, _STRIDE)
        self.num_variates = num_variates
        self.proj = nn.Linear(_PATCH, _D_MODEL)
        self.embed_dropout = nn.Dropout(_DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=_D_MODEL, nhead=_N_HEADS, dim_feedforward=_D_MODEL * 4, dropout=_DROPOUT, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=_N_LAYERS)
        self.head = nn.Linear(n * _D_MODEL, _PRED_LEN)
        self._n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, Cv = x.shape
        xp = x.permute(0, 2, 1).reshape(B * Cv, L)
        p = xp.unfold(-1, _PATCH, _STRIDE)
        emb = self.embed_dropout(self.proj(p)).reshape(B, Cv, self._n, -1)
        seq = emb.reshape(B, Cv * self._n, -1)
        enc = self.encoder(seq).reshape(B * Cv, -1)
        return self.head(enc).reshape(B, Cv, -1).permute(0, 2, 1)


def _build(mode: str, C: int, groups: list[list[int]]) -> nn.Module:
    if mode == "CD":
        return _VanillaCD(C)
    if mode == "CD_Block":
        return PatchTST_CD_Block(
            groups=groups,
            num_variates=C,
            seq_len=_SEQ_LEN,
            pred_len=_PRED_LEN,
            patch_size=_PATCH,
            stride=_STRIDE,
            d_model=_D_MODEL,
            n_heads=_N_HEADS,
            n_layers=_N_LAYERS,
            dropout=_DROPOUT,
        )
    raise ValueError(f"Unknown mode: {mode}")


def _try_epochs(mode: str, C: int, groups: list[list[int]], batch: int, epochs: int, device: torch.device) -> dict:
    """Run `epochs` timed epochs at a fixed batch size.

    Returns a record with seconds per epoch and peak memory, or raises the
    original RuntimeError on OOM so the caller can descend the ladder.
    """
    torch.manual_seed(0)
    model = _build(mode, C, groups).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.MSELoss()

    steps = (_N_TRAIN_WINDOWS + batch - 1) // batch
    x = torch.randn(batch, _SEQ_LEN, C, device=device)
    y = torch.randn(batch, _PRED_LEN, C, device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    epoch_times: list[float] = []
    model.train()
    for _ in range(epochs):
        t0 = time.perf_counter()
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), _GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_times.append(time.perf_counter() - t0)

    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    del model, opt, scaler, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # First epoch includes CUDA autotune and allocator warmup; exclude it from
    # the steady-state figure whenever a later epoch exists.
    steady = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
    sec = sum(steady) / len(steady)
    return {
        "mode": mode,
        "C": C,
        "batch": batch,
        "steps_per_epoch": steps,
        "warmup_sec": epoch_times[0],
        "sec_per_epoch": sec,
        "peak_gib": peak_gib,
        "projected_hours_full_run": sec * _FULL_RUN_EPOCHS / 3600.0,
    }


def _descend_ladder(
    mode: str, C: int, groups: list[list[int]], start: int, epochs: int, device: torch.device
) -> dict | None:
    """Find the largest feasible batch on the ladder at or below `start`."""
    for batch in (b for b in _BATCH_LADDER if b <= start):
        try:
            return _try_epochs(mode, C, groups, batch, epochs, device)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
        # Handled outside the except block: with the exception cleared, the
        # traceback no longer pins _try_epochs' frame, so gc can actually
        # release the failed attempt's tensors before the next, smaller one.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {mode} C={C}: OOM at batch {batch}, descending.")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Block-attention CD feasibility dry run.")
    parser.add_argument("--config", action="append", choices=sorted(_CONFIGS), required=True)
    parser.add_argument("--modes", nargs="+", default=["CD", "CD_Block"], choices=["CD", "CD_Block"])
    parser.add_argument("--epochs", type=int, default=2, help="Timed epochs per measurement.")
    parser.add_argument("--start-batch", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA unavailable: timings will not reflect the T4 and peak memory is not measured.")

    records: list[dict] = []
    for name in args.config:
        C, make_groups = _CONFIGS[name]
        groups = make_groups()
        sizes = sorted({len(g) for g in groups})
        print(f"\n=== {name}: C={C}, {len(groups)} groups, sizes {sizes} ===")
        for mode in args.modes:
            rec = _descend_ladder(mode, C, groups, args.start_batch, args.epochs, device)
            if rec is None:
                print(f"  {mode}: infeasible at every batch size down to 1.")
                records.append({"config": name, "mode": mode, "batch": 0})
                continue
            rec["config"] = name
            print(
                f"  {mode}: batch {rec['batch']}, {rec['steps_per_epoch']} steps/epoch, "
                f"warmup {rec['warmup_sec']:.1f} s, steady {rec['sec_per_epoch']:.1f} s/epoch, "
                f"peak {rec['peak_gib']:.2f} GiB, "
                f"projected {rec['projected_hours_full_run']:.2f} h for {_FULL_RUN_EPOCHS} epochs"
            )
            records.append(rec)

    print("\n=== SUMMARY (largest feasible batch per mode) ===")
    header = f"{'config':>10} {'mode':>9} {'batch':>6} {'s/epoch':>9} {'peak GiB':>9} {'proj h':>7}"
    print(header)
    print("-" * len(header))
    for rec in records:
        if rec["batch"] == 0:
            print(f"{rec['config']:>10} {rec['mode']:>9} {'OOM@1':>6}")
            continue
        print(
            f"{rec['config']:>10} {rec['mode']:>9} {rec['batch']:>6} {rec['sec_per_epoch']:>9.1f} "
            f"{rec['peak_gib']:>9.2f} {rec['projected_hours_full_run']:>7.2f}"
        )


if __name__ == "__main__":
    main()
