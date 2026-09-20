# ── P=2 feasibility probe (current environment) ───────────────────────────────
# Paste as a cell into the boundary notebook AFTER its definition cells (the ones
# that define: generate, split_and_normalise, WindowDataset, M (import models),
# LOOKBACK, PRED_LEN, N_TOTAL, D_MODEL, N_HEADS, N_LAYERS, DROPOUT, DEVICE).
# It measures peak CUDA memory, per-step compute time, and confirms
# steps_per_epoch for a P=2 CD run at batch 64 before a full run is launched.
# Console only; nothing is trained to completion and no file is written.
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

assert torch.cuda.is_available(), "probe needs a CUDA GPU (run on the T4 accelerator)"
assert N_TOTAL == 21, f"expected C=21, notebook has N_TOTAL={N_TOTAL}"

_P2, _STRIDE2, _BATCH2 = 2, 1, 64          # P=2, S=P//2=1, standard P=2 batch
_N_PATCHES2 = (LOOKBACK - _P2) // _STRIDE2 + 1
_WARMUP, _TIMED = 3, 12
_MAX_EPOCHS, _TYPICAL_EPOCHS = 50, 30      # schedule ceiling; P=4-like early-stop point
_N_CD, _N_CI = 5, 5                         # declared run list in config_boundary_p2.json
_T4_USABLE_GIB = 14.8                       # ~16 GB card less driver/context headroom

# Real data + real model code; only patch size and batch differ from the run.
_tr, _va, _te = split_and_normalise(generate(0.9, 42))
_dl = DataLoader(WindowDataset(_tr, LOOKBACK, PRED_LEN),
                 batch_size=_BATCH2, shuffle=True, drop_last=True)
_spe = len(_dl)
print(f"P=2 probe: n_patches={_N_PATCHES2}  C*N={N_TOTAL * _N_PATCHES2}  "
      f"batch={_BATCH2}  steps_per_epoch={_spe}")
assert _spe == 209, f"spe {_spe} != expected 209 — update config_boundary_p2.json expected_spe"


def _build(mode):
    m = M.build_model(mode, seq_len=LOOKBACK, pred_len=PRED_LEN, num_variates=N_TOTAL,
                      patch_size=_P2, stride=_STRIDE2, d_model=D_MODEL,
                      n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT).to(DEVICE)
    assert m.head.in_features == _N_PATCHES2 * D_MODEL, f"{mode} head wrong: {m.head}"
    return m


def _time_arm(mode):
    """Return (peak_alloc_GiB, peak_reserved_GiB, median_step_s) or None on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        m = _build(mode)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
        crit = nn.MSELoss()
        scaler = torch.amp.GradScaler("cuda")
        it = iter(_dl)
        samples = []
        for k in range(_WARMUP + _TIMED):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(_dl)
                xb, yb = next(it)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                loss = crit(m(xb), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            if k >= _WARMUP:
                samples.append(time.perf_counter() - t0)
        peak_a = torch.cuda.max_memory_allocated() / 2 ** 30
        peak_r = torch.cuda.max_memory_reserved() / 2 ** 30
        samples.sort()
        med = samples[len(samples) // 2]
        del m, opt, scaler
        torch.cuda.empty_cache()
        return peak_a, peak_r, med
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


_epoch_h = {}
for _mode in ("CD", "CI"):
    _res = _time_arm(_mode)
    if _res is None:
        print(f"  {_mode}: OUT OF MEMORY at batch {_BATCH2} — not feasible as configured; "
              f"reduce batch or declare P=2 {_mode} infeasible.")
        _epoch_h[_mode] = None
        continue
    _pa, _pr, _med = _res
    _eh = _spe * _med / 3600.0
    _epoch_h[_mode] = _eh
    _fit = ("FITS" if _pr < _T4_USABLE_GIB - 1.0
            else "TIGHT" if _pr <= _T4_USABLE_GIB else "LIKELY OOM")
    print(f"  {_mode}: peak {_pa:5.2f} GiB alloc / {_pr:5.2f} GiB reserved [{_fit}]  |  "
          f"{_med * 1000:6.0f} ms/step  |  {_eh:4.2f} h/epoch  |  "
          f"~{_TYPICAL_EPOCHS * _eh:5.1f} h/seed (best-epoch ~{_TYPICAL_EPOCHS}), "
          f"{_MAX_EPOCHS * _eh:5.1f} h ceiling")

if _epoch_h.get("CD") is not None and _epoch_h.get("CI") is not None:
    _typ = _N_CD * _TYPICAL_EPOCHS * _epoch_h["CD"] + _N_CI * _TYPICAL_EPOCHS * _epoch_h["CI"]
    _ceil = _N_CD * _MAX_EPOCHS * _epoch_h["CD"] + _N_CI * _MAX_EPOCHS * _epoch_h["CI"]
    print(f"  run list ({_N_CD} CD + {_N_CI} CI, gamma 0.9): "
          f"~{_typ:5.1f} h typical, {_ceil:5.1f} h ceiling "
          f"(per-slice weekly quota is 30 h — trim seeds or split across slices "
          f"if this exceeds it).")
    print("  note: per-step is compute-only; real epochs add data-loading and a "
          "per-epoch validation pass, so treat these hours as a lower bound.")
