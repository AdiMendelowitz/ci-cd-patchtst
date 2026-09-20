"""Unit tests for the generators, paired statistics, and models.

Generator and statistics tests run anywhere numpy and scipy are present. Model
tests exercise src/models.py and require torch; they are skipped automatically
when torch is unavailable, so the suite still passes on a CPU-only machine
without the deep-learning stack.

Run from the repository root:
    python -m pytest src/tests/ -q
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import generate_block_cov as gbc
import paired_stats as ps

_SEQ_LEN = 512
_PRED_LEN = 96
_PATCH = 16
_STRIDE = 8


# ── Generator ───────────────────────────────────────────────────────────────


def test_generator_shapes_and_split():
    data = gbc.generate(phi=0.8, rho_in=0.9, C=21, group_size=7, seed=42)
    assert data.shape == (gbc.T_TOTAL, 21)
    train, val, test = gbc.split_and_normalise(data)
    assert train.shape == (8640, 21)
    assert val.shape == (2880, 21)
    assert test.shape == (2880, 21)


def test_generator_deterministic_and_seed_sensitive():
    a = gbc.generate(phi=0.8, rho_in=0.9, C=21, group_size=7, seed=42)
    b = gbc.generate(phi=0.8, rho_in=0.9, C=21, group_size=7, seed=42)
    c = gbc.generate(phi=0.8, rho_in=0.9, C=21, group_size=7, seed=123)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generator_block_structure():
    data = gbc.generate(phi=0.8, rho_in=0.9, C=84, group_size=7, seed=42)
    train, _, _ = gbc.split_and_normalise(data)
    summary = gbc.empirical_block_summary(train, 84, 7, 0.9, 0.0)
    within = summary.loc[summary["pairs"] == "within_group", "mean_abs_r"].iloc[0]
    between = summary.loc[summary["pairs"] == "between_group", "mean_abs_r"].iloc[0]
    assert abs(within - 0.9) < 0.02
    assert between < 0.05


def test_generator_granger_non_causal():
    # Diagonal transition implies the optimal one-step predictor ignores other
    # channels' lags. Check the conditional regression coefficient is near zero.
    data = gbc.generate(phi=0.8, rho_in=0.9, C=21, group_size=7, seed=42)
    y = data[1:, 0]
    design = np.column_stack([np.ones(len(y)), data[:-1, 0], data[:-1, 1]])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    assert abs(beta[1] - 0.8) < 0.05  # own lag near phi
    assert abs(beta[2]) < 0.05  # other channel's lag near zero


def test_covariance_guards():
    with pytest.raises(ValueError):
        gbc.make_block_covariance(C=85, group_size=7, rho_in=0.9)
    with pytest.raises(ValueError):
        gbc.make_block_covariance(C=84, group_size=7, rho_in=1.0)
    with pytest.raises(ValueError):
        gbc.make_block_covariance(C=84, group_size=7, rho_in=0.5, rho_out=0.5)


# ── Paired statistics ─────────────────────────────────────────────────────────


def _toy_results() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for gamma in (0.0, 0.6):
        for seed in (42, 123, 456, 789, 1011):
            ci = 1.0 + rng.normal(0, 0.01)
            cd = ci + 0.002 + rng.normal(0, 0.003)
            rows += [
                {"gamma": gamma, "seed": seed, "mode": "CI", "test_mse": ci},
                {"gamma": gamma, "seed": seed, "mode": "CD", "test_mse": cd},
            ]
    return pd.DataFrame(rows)


def test_make_diff_frame_sign_convention():
    df = _toy_results()
    diff = ps.make_diff_frame(df, ["gamma"], "CI", "CD")
    assert set(["gamma", "seed", "_base", "_alt", "diff"]).issubset(diff.columns)
    assert np.allclose(diff["diff"], diff["_alt"] - diff["_base"])
    assert len(diff) == 10


def test_paired_differences_matches_manual():
    df = _toy_results()
    diff = ps.make_diff_frame(df, ["gamma"], "CI", "CD")
    paired = ps.paired_differences(diff, ["gamma"])
    cell = diff[diff["gamma"] == 0.0]["diff"].to_numpy()
    mean = cell.mean()
    se = cell.std(ddof=1) / np.sqrt(len(cell))
    tcrit = stats.t.ppf(0.975, df=len(cell) - 1)
    row = paired[paired["gamma"] == 0.0].iloc[0]
    assert np.isclose(row["mean_diff"], mean)
    assert np.isclose(row["ci_lo"], mean - tcrit * se)
    assert np.isclose(row["ci_hi"], mean + tcrit * se)


def test_half_width_and_grand_mean():
    df = _toy_results()
    diff = ps.make_diff_frame(df, ["gamma"], "CI", "CD")
    hw = ps.compute_ci_half_width(diff, ["gamma"])
    assert hw > 0
    gm = ps.grand_mean_diff(diff)
    assert gm["n"] == 10
    assert gm["ci_lo"] < gm["mean"] < gm["ci_hi"]


def test_make_diff_frame_missing_mode_raises():
    df = _toy_results()
    with pytest.raises(ValueError):
        ps.make_diff_frame(df, ["gamma"], "CI", "CD_Head")


# ── Models (require torch) ─────────────────────────────────────────────────────


def _models():
    pytest.importorskip("torch")
    import models

    return models


def test_num_patches():
    m = _models()
    assert m.num_patches(_SEQ_LEN, _PATCH, _STRIDE) == 63


@pytest.mark.parametrize("mode", ["CI", "CD", "CD_Head", "DLinear"])
def test_model_output_shape(mode):
    m = _models()
    import torch
    model = m.build_model(mode, _SEQ_LEN, _PRED_LEN, num_variates=21,
                          patch_size=_PATCH, stride=_STRIDE, d_model=64, n_heads=8, n_layers=3, dropout=0.2)
    model.eval()
    x = torch.zeros(2, _SEQ_LEN, 21)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, _PRED_LEN, 21)


def test_head_dimensions():
    m = _models()
    n = m.num_patches(_SEQ_LEN, _PATCH, _STRIDE)
    cd = m.build_model("CD", _SEQ_LEN, _PRED_LEN, 21, _PATCH, _STRIDE, 64, 8, 3, 0.2)
    head = m.build_model("CD_Head", _SEQ_LEN, _PRED_LEN, 21, _PATCH, _STRIDE, 64, 8, 3, 0.2)
    assert cd.head.in_features == n * 64
    assert head.head.in_features == n * 64 + 64  # plus the cross-variate context vector
    assert cd.head.out_features == _PRED_LEN
    assert head.head.out_features == _PRED_LEN


def test_cd_head_encoder_matches_cd():
    # The encoder must be untouched relative to CD so the variant isolates the head.
    m = _models()
    cd = m.build_model("CD", _SEQ_LEN, _PRED_LEN, 21, _PATCH, _STRIDE, 64, 8, 3, 0.2)
    head = m.build_model("CD_Head", _SEQ_LEN, _PRED_LEN, 21, _PATCH, _STRIDE, 64, 8, 3, 0.2)
    cd_enc = sum(p.numel() for p in cd.encoder.parameters())
    head_enc = sum(p.numel() for p in head.encoder.parameters())
    assert cd_enc == head_enc


def test_build_model_unknown_mode_raises():
    m = _models()
    with pytest.raises(ValueError):
        m.build_model("CI_CD", _SEQ_LEN, _PRED_LEN, 21, _PATCH, _STRIDE, 64, 8, 3, 0.2)


def test_model_deterministic_in_eval():
    m = _models()
    import torch
    torch.manual_seed(0)
    model = m.build_model("CD_Head", _SEQ_LEN, _PRED_LEN, 21, _PATCH, _STRIDE, 64, 8, 3, 0.2)
    model.eval()
    x = torch.randn(2, _SEQ_LEN, 21)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    assert torch.allclose(a, b)