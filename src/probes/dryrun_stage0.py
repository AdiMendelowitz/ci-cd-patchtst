"""Stage 0 measurement probe.

Extends the committed dry run (dryrun_block_attention.py) to every
configuration in the run matrix:

    lf_c21           CI ladder; CD and CD_Block at the committed batch 8
                     (CD@8 calibrates CD_Block@8)
    boundary_p4_c21  P=4 stride 2 (S = P/2, confirmed from main.tex):
                     N=255, C*N=5355; CD and CI
    boundary_p2_c21  P=2 stride 1: N=511, C*N=10731; CD and CI. If CD OOMs
                     at batch 8, that is the signal that the protocol needs the
                     gradient-checkpointing flag.
    ecl_cd           ECL C=321 vanilla CD (C*N=3531): feasibility under
                     the current PyTorch
    ecl_cd_block     ECL C=321, contiguous groups of 3 (neutral partition)

Per (config, mode, batch) cell it reports: feasibility, warmup and steady
s/epoch, peak GiB, and projected hours for a converged run, where the
converged epoch count comes from the committed CSVs (max best_epoch +
patience 10, capped at MAX_EPOCHS 50) rather than a blanket assumption;
unmeasured families fall back to the 50-epoch worst case and say so.

Measurement engine is the committed probe's, unchanged: random tensors
shaped like the real data (memory and speed do not depend on values), AdamW
+ AMP autocast + GradScaler + clip at 1.0, first epoch reported separately
as warmup. CD-family modes are additionally measured at the committed
protocol batch 8 even when a larger batch fits, because protocol identity
fixes the batch of any new CD arm.

Run on a T4 with models_cd_block.py beside it:

    %run <path>/dryrun_stage0.py                      # all configs
    %run <path>/dryrun_stage0.py --config lf_c21 --config boundary_p4_c21
    %run <path>/dryrun_stage0.py --epochs 3 --csv /kaggle/working/stage0.csv

Without CUDA, timing and peak-memory figures are not representative of the
T4, but every cell still runs, which is enough to check the script itself
before spending GPU time on it.

Prints a summary table; writes a CSV only if --csv is given.
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# ── Locate the AMP-fixed committed module (beside this file, or anywhere under
# /kaggle/input — Kaggle currently mounts datasets at
# /kaggle/input/datasets/<user>/<dataset>/, previously /kaggle/input/<dataset>/).
_AMP_FIX_MARKER = "Allocated from the first encoder output"
try:
    import models_cd_block  # noqa: F401  (works when cwd holds the dataset copy)
except ImportError:
    _repo_src = Path(__file__).resolve().parent.parent  # src/, this repo's layout
    if (_repo_src / "models_cd_block.py").exists():
        sys.path.insert(0, str(_repo_src))
    elif Path("/kaggle/input").exists():
        _hits = sorted(Path("/kaggle/input").rglob("models_cd_block.py"))
        if not _hits:
            raise ImportError("models_cd_block.py not found beside this script, in the repo's src/, or under /kaggle/input.")
        sys.path.insert(0, str(_hits[0].parent))
    else:
        raise ImportError("models_cd_block.py not found beside this script, in the repo's src/, or under /kaggle/input.")
    import models_cd_block  # noqa: F401

_module_path = Path(models_cd_block.__file__)
assert _AMP_FIX_MARKER in _module_path.read_text(), (
    f"{_module_path} lacks the AMP-fix marker: a pre-fix module version is on the path."
)
from models_cd_block import PatchTST_CD_Block, contiguous_groups, leader_follower_groups, num_patches

# Schedule constants, shared by every family.
_N_LAYERS = 3            # main.tex Table 2: encoder layers is 3 for both configurations
_DROPOUT = 0.2           # main.tex Table 2: dropout is 0.2 for both configurations
_LR = 1e-4
_WEIGHT_DECAY = 1e-4
_GRAD_CLIP = 1.0
_PRED_LEN = 96
_MAX_EPOCHS = 50
_PATIENCE = 10
_CD_FAMILY_BATCH = 8     # committed protocol batch for CD and CD_Block arms
_CI_BATCH = 128          # committed protocol batch for CI arms (synthetic families)
_BATCH_LADDER = (128, 64, 32, 16, 8, 4, 2, 1)

# Architecture presets, main.tex Table 2 ("Architectural hyperparameters").
# Synthetic and ETTh1 share one configuration; ECL uses a wider model at a
# shorter sequence length to accommodate 321 variates. Every _CONFIGS entry
# below sets one of these explicitly -- there is no module-level default, so
# a family that forgets to set one fails on a missing dict key instead of
# silently inheriting the wrong architecture.
_ARCH_SYNTH = dict(seq_len=512, d_model=64, n_heads=8)
_ARCH_ECL = dict(seq_len=96, d_model=128, n_heads=16)

_ECL_C = 321

# Per-family train-window counts.
# lf: 8640 - 512 - 96 + 1 = 8033 (committed lf CSV: CI 63 steps @128, ceil).
# boundary: 13392, the unique count consistent with BOTH committed step counts
#   (CI 104 @128 and CD 1674 @8) under drop_last=True; the probe's own loop
#   uses ceil, so committed step counts are pinned per cell below instead.
# ecl: ECLDataset applies the iTransformer split (15840/26352 train
#   proportion) to the real file's row count. The live training run
#   (results/ecl_ci_cd_train_resumable_v3_stdout.txt) logs the actual split as
#   train_end=15813 rows; at seq_len=96, pred_len=96 that gives
#   15813 - 96 - 96 + 1 = 15622 windows. This also matches the same log's
#   CI steps/epoch of 1953 at batch 8 (ceil(15622 / 8) = 1953), which is an
#   independent cross-check on the window count.
_CONFIGS: dict[str, dict] = {
    "lf_c21": dict(
        C=21, patch=16, stride=8, windows=8033, family="lf",
        groups=leader_follower_groups,
        cells=[("CI", None), ("CD", _CD_FAMILY_BATCH), ("CD_Block", _CD_FAMILY_BATCH)],
        **_ARCH_SYNTH,
    ),
    "boundary_p4_c21": dict(
        C=21, patch=4, stride=2, windows=13392, family="boundary",
        groups=None,
        cells=[("CI", None), ("CI", _CI_BATCH), ("CD", None), ("CD", _CD_FAMILY_BATCH)],
        **_ARCH_SYNTH,
    ),
    "boundary_p2_c21": dict(
        C=21, patch=2, stride=1, windows=13392, family="boundary",
        groups=None,
        cells=[("CI", None), ("CI", _CI_BATCH), ("CD", None), ("CD", _CD_FAMILY_BATCH)],
        **_ARCH_SYNTH,
    ),
    "ecl_cd": dict(
        C=_ECL_C, patch=16, stride=8, windows=15622, family="ecl",
        groups=None,
        cells=[("CD", None), ("CD", _CD_FAMILY_BATCH)],
        **_ARCH_ECL,
    ),
    "ecl_cd_block": dict(
        C=_ECL_C, patch=16, stride=8, windows=15622, family="ecl",
        groups=lambda: contiguous_groups(_ECL_C, 3),
        cells=[("CD_Block", None), ("CD_Block", _CD_FAMILY_BATCH)],
        **_ARCH_ECL,
    ),
}

# Committed steps/epoch, pinned where a committed CSV states them (loader
# drop_last behaviour differs across families; measurement must match the
# protocol's step count, not this script's ceil).
_STEPS_PINNED: dict[tuple[str, str, int], int] = {
    ("boundary_p4_c21", "CI", 128): 104,   # results_boundary_p4_ci.csv
    ("boundary_p2_c21", "CI", 128): 104,   # results_boundary.csv (P=2 rows)
    ("boundary_p4_c21", "CD", 8): 1674,    # equals ceil(13392/8); pinned as documentation
    ("boundary_p2_c21", "CD", 8): 1674,    # of protocol identity with results_boundary.csv CD
}

# Converged-run epoch counts for projections: committed max best_epoch +
# patience, capped at MAX_EPOCHS. Families without committed evidence use the
# 50-epoch worst case, flagged in the provenance string.
_PROJ_EPOCHS: dict[tuple[str, str], tuple[int, str]] = {
    ("lf", "CI"):             (41, "lf CSV: max best_epoch 31 + 10"),
    ("lf", "CD"):             (17, "lf CSV: max best_epoch 7 + 10"),
    ("lf", "CD_Block"):       (17, "assumed = lf CD family (unmeasured)"),
    ("boundary", "CI"):       (50, "boundary CSV: max best_epoch 48 + 10, capped 50"),
    ("boundary", "CD"):       (22, "boundary CSV: max best_epoch 12 + 10"),
    ("ecl", "CD"):            (50, "worst case (no committed ECL CD runs)"),
    ("ecl", "CD_Block"):      (50, "worst case (no committed ECL CD_Block runs)"),
}


def _is_oom(exc: RuntimeError) -> bool:
    """True if exc is an allocation failure, CUDA or CPU.

    CUDA raises "CUDA out of memory. Tried to allocate ...". The CPU
    allocator raises a different message, "DefaultCPUAllocator: can't
    allocate memory: ..." (verified against the installed torch build) --
    matching on "out of memory" alone misses it and lets a CPU allocation
    failure crash the run instead of falling back down the batch ladder.
    """
    msg = str(exc).lower()
    return "out of memory" in msg or "can't allocate memory" in msg or "defaultcpuallocator" in msg


class _VanillaCD(nn.Module):
    """Flattened-token CD, reproduced from the training notebooks; patch,
    stride, and architecture parameterised so every config shares one
    implementation."""

    def __init__(self, num_variates: int, patch: int, stride: int,
                 seq_len: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        n = num_patches(seq_len, patch, stride)
        self.num_variates, self._n, self._patch, self._stride = num_variates, n, patch, stride
        self.proj = nn.Linear(patch, d_model)
        self.embed_dropout = nn.Dropout(_DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=_DROPOUT, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=_N_LAYERS)
        self.head = nn.Linear(n * d_model, _PRED_LEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, Cv = x.shape
        xp = x.permute(0, 2, 1).reshape(B * Cv, L)
        p = xp.unfold(-1, self._patch, self._stride)
        emb = self.embed_dropout(self.proj(p)).reshape(B, Cv, self._n, -1)
        seq = emb.reshape(B, Cv * self._n, -1)
        enc = self.encoder(seq).reshape(B * Cv, -1)
        return self.head(enc).reshape(B, Cv, -1).permute(0, 2, 1)


class _VanillaCI(nn.Module):
    """Channel-independent PatchTST, reproduced from the training notebooks."""

    def __init__(self, patch: int, stride: int,
                 seq_len: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        n = num_patches(seq_len, patch, stride)
        self._n, self._patch, self._stride = n, patch, stride
        self.proj = nn.Linear(patch, d_model)
        self.embed_dropout = nn.Dropout(_DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=_DROPOUT, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=_N_LAYERS)
        self.head = nn.Linear(n * d_model, _PRED_LEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, Cv = x.shape
        xp = x.permute(0, 2, 1).reshape(B * Cv, L)
        p = xp.unfold(-1, self._patch, self._stride)
        enc = self.encoder(self.embed_dropout(self.proj(p)))
        return self.head(enc.reshape(B * Cv, -1)).reshape(B, Cv, -1).permute(0, 2, 1)


def _build(mode: str, cfg: dict) -> nn.Module:
    if mode == "CI":
        return _VanillaCI(cfg["patch"], cfg["stride"], cfg["seq_len"], cfg["d_model"], cfg["n_heads"])
    if mode == "CD":
        return _VanillaCD(cfg["C"], cfg["patch"], cfg["stride"], cfg["seq_len"], cfg["d_model"], cfg["n_heads"])
    if mode == "CD_Block":
        assert cfg["groups"] is not None, "CD_Block cell in a config without a partition"
        return PatchTST_CD_Block(
            groups=cfg["groups"](), num_variates=cfg["C"], seq_len=cfg["seq_len"], pred_len=_PRED_LEN,
            patch_size=cfg["patch"], stride=cfg["stride"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], n_layers=_N_LAYERS, dropout=_DROPOUT,
        )
    raise ValueError(f"Unknown mode: {mode}")


def _steps_for(name: str, mode: str, batch: int, windows: int) -> int:
    return _STEPS_PINNED.get((name, mode, batch), (windows + batch - 1) // batch)


def _try_epochs(name: str, mode: str, cfg: dict, batch: int, epochs: int, device: torch.device,
                max_steps: int = 60) -> dict:
    """Run `epochs` timed passes of min(steps, max_steps) training steps at a
    fixed batch size, scaling the per-epoch figure back up; peak memory is
    unaffected by the cap. Raises RuntimeError on OOM."""
    torch.manual_seed(0)
    model = _build(mode, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=_LR, weight_decay=_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = nn.MSELoss()

    steps = _steps_for(name, mode, batch, cfg["windows"])
    steps_timed = min(steps, max_steps)
    x = torch.randn(batch, cfg["seq_len"], cfg["C"], device=device)
    y = torch.randn(batch, _PRED_LEN, cfg["C"], device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    epoch_times: list[float] = []
    model.train()
    for _ in range(epochs):
        t0 = time.perf_counter()
        for _ in range(steps_timed):
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
        epoch_times.append((time.perf_counter() - t0) * (steps / steps_timed))

    peak_gib = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else float("nan")
    del model, opt, scaler, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    steady = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
    sec = sum(steady) / len(steady)
    proj_epochs, proj_src = _PROJ_EPOCHS[(cfg["family"], mode)]
    return {
        "config": name, "mode": mode, "batch": batch, "steps_per_epoch": steps,
        "steps_timed": steps_timed,
        "warmup_sec": round(epoch_times[0], 2), "sec_per_epoch": round(sec, 2),
        "peak_gib": round(peak_gib, 3),
        "proj_epochs": proj_epochs, "proj_epochs_src": proj_src,
        "projected_hours_converged_run": round(sec * proj_epochs / 3600.0, 3),
    }


def _measure(name: str, mode: str, cfg: dict, batch: int | None, epochs: int, device: torch.device,
             max_steps: int = 60) -> dict | None:
    """Fixed batch when given; otherwise descend the ladder from its top."""
    ladder = [batch] if batch is not None else list(_BATCH_LADDER)
    for b in ladder:
        try:
            return _try_epochs(name, mode, cfg, b, epochs, device, max_steps)
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
        # Outside the except block so the cleared traceback releases the frame.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"  {mode}: OOM at batch {b}" + ("" if batch is None else " (fixed protocol batch)"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 measurement probe.")
    parser.add_argument("--config", action="append", choices=sorted(_CONFIGS), default=None,
                        help="Repeatable; default: all configs in plan order.")
    parser.add_argument("--epochs", type=int, default=2, help="Timed epochs per cell (first is warmup).")
    parser.add_argument("--max-steps", type=int, default=60,
                        help="Timed steps per pass; per-epoch time is scaled from these.")
    parser.add_argument("--csv", type=str, default=None, help="Optional output CSV path.")
    args = parser.parse_args()
    names = args.config or list(_CONFIGS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[WARN] CUDA unavailable: timings do not reflect the T4; peak memory not measured.")

    records: list[dict] = []
    for name in names:
        cfg = _CONFIGS[name]
        n = num_patches(cfg["seq_len"], cfg["patch"], cfg["stride"])
        print(f"\n=== {name}: C={cfg['C']}, P={cfg['patch']}, S={cfg['stride']}, L={cfg['seq_len']}, "
              f"N={n}, C*N={cfg['C'] * n}, d_model={cfg['d_model']}, heads={cfg['n_heads']}, "
              f"windows={cfg['windows']} ===")
        seen: set[tuple[str, int]] = set()
        for mode, batch in cfg["cells"]:
            if batch is not None and (mode, batch) in seen:
                continue     # ladder already landed on the protocol batch
            rec = _measure(name, mode, cfg, batch, args.epochs, device, args.max_steps)
            if rec is None:
                b = "ladder" if batch is None else batch
                records.append({"config": name, "mode": mode, "batch": 0,
                                "note": f"infeasible ({b})"})
                print(f"  {mode}: INFEASIBLE " + ("at every ladder batch." if batch is None
                                                  else f"at the fixed protocol batch {batch}."))
                continue
            seen.add((rec["mode"], rec["batch"]))
            records.append(rec)
            print(f"  {mode}: batch {rec['batch']}, {rec['steps_per_epoch']} steps/epoch, "
                  f"warmup {rec['warmup_sec']:.1f} s, steady {rec['sec_per_epoch']:.1f} s/epoch, "
                  f"peak {rec['peak_gib']:.2f} GiB, "
                  f"projected {rec['projected_hours_converged_run']:.2f} h "
                  f"for {rec['proj_epochs']} epochs [{rec['proj_epochs_src']}]")

    print("\n=== SUMMARY ===")
    header = (f"{'config':>16} {'mode':>9} {'batch':>6} {'steps':>6} {'s/epoch':>9} "
              f"{'peak GiB':>9} {'ep':>4} {'proj h/run':>10}")
    print(header)
    print("-" * len(header))
    for rec in records:
        if rec["batch"] == 0:
            print(f"{rec['config']:>16} {rec['mode']:>9} {'OOM':>6}  <- {rec['note']}")
            continue
        print(f"{rec['config']:>16} {rec['mode']:>9} {rec['batch']:>6} {rec['steps_per_epoch']:>6} "
              f"{rec['sec_per_epoch']:>9.1f} {rec['peak_gib']:>9.2f} {rec['proj_epochs']:>4} "
              f"{rec['projected_hours_converged_run']:>10.2f}")
    print("\ntrigger arithmetic: 20 CD runs x proj h/run at boundary_p4 CD@8; "
          "tiers fire at >70 h (gamma {0,.6,.9}) and >100 h (reassess).")

    if args.csv:
        import csv as _csv
        keys = ["config", "mode", "batch", "steps_per_epoch", "steps_timed", "warmup_sec", "sec_per_epoch",
                "peak_gib", "proj_epochs", "proj_epochs_src", "projected_hours_converged_run", "note"]
        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for rec in records:
                w.writerow({k: rec.get(k, "") for k in keys})
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()