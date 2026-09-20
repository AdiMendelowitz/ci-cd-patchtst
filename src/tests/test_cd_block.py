"""Unit tests for the block-wise cross-variate attention CD variant.

Run from the directory holding models_cd_block.py:
    python -m pytest test_cd_block.py -q
"""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import models_cd_block as mcb  # noqa: E402  (import gated on torch availability)

_SEQ_LEN = 512
_PRED_LEN = 96
_PATCH = 16
_STRIDE = 8
_D_MODEL = 64
_N_HEADS = 8
_N_LAYERS = 3
_DROPOUT = 0.2


def _build(groups: list[list[int]], num_variates: int) -> mcb.PatchTST_CD_Block:
    return mcb.PatchTST_CD_Block(
        groups=groups,
        num_variates=num_variates,
        seq_len=_SEQ_LEN,
        pred_len=_PRED_LEN,
        patch_size=_PATCH,
        stride=_STRIDE,
        d_model=_D_MODEL,
        n_heads=_N_HEADS,
        n_layers=_N_LAYERS,
        dropout=_DROPOUT,
    )


def test_partition_helpers():
    groups = mcb.contiguous_groups(84, 7)
    assert len(groups) == 12
    assert groups[0] == list(range(7))
    lf = mcb.leader_follower_groups()
    assert len(lf) == 11
    assert lf[0] == [0, 10]
    assert lf[-1] == [20]
    assert sorted(i for g in lf for i in g) == list(range(21))


def test_partition_guards():
    with pytest.raises(ValueError):
        mcb.contiguous_groups(85, 7)
    with pytest.raises(ValueError):
        _build([[0, 1], [1, 2]], 3)  # duplicate channel
    with pytest.raises(ValueError):
        _build([[0], [2]], 3)  # missing channel
    with pytest.raises(ValueError):
        _build([[0], [1], [2], []], 3)  # empty group


def test_constructor_patching_guard():
    with pytest.raises(ValueError):
        mcb.PatchTST_CD_Block(
            groups=[[0]], num_variates=1, seq_len=8, pred_len=4,
            patch_size=16, stride=8, d_model=8, n_heads=2, n_layers=1, dropout=0.0,
        )


def test_forward_channel_mismatch_raises():
    model = _build(mcb.leader_follower_groups(), 21)
    model.eval()
    with pytest.raises(ValueError):
        with torch.no_grad():
            model(torch.zeros(1, _SEQ_LEN, 7))


def test_output_shape_leader_follower():
    model = _build(mcb.leader_follower_groups(), 21)
    model.eval()
    x = torch.zeros(2, _SEQ_LEN, 21)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, _PRED_LEN, 21)


def test_deterministic_in_eval():
    torch.manual_seed(0)
    model = _build(mcb.leader_follower_groups(), 21)
    model.eval()
    x = torch.randn(2, _SEQ_LEN, 21)
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))


def test_cross_group_independence():
    # Perturbing channels of one group must not change any other group's output.
    torch.manual_seed(0)
    model = _build(mcb.leader_follower_groups(), 21)
    model.eval()
    x = torch.randn(1, _SEQ_LEN, 21)
    y = x.clone()
    y[:, :, 0] += 10.0  # leader 0: group [0, 10]
    y[:, :, 10] -= 5.0
    with torch.no_grad():
        a, b = model(x), model(y)
    changed = [0, 10]
    unchanged = [c for c in range(21) if c not in changed]
    assert not torch.allclose(a[:, :, changed], b[:, :, changed])
    assert torch.allclose(a[:, :, unchanged], b[:, :, unchanged], atol=1e-6)


def test_within_group_dependence():
    # The follower's output must respond to its leader's input.
    torch.manual_seed(0)
    model = _build(mcb.leader_follower_groups(), 21)
    model.eval()
    x = torch.randn(1, _SEQ_LEN, 21)
    y = x.clone()
    y[:, :, 3] += 10.0  # leader 3: group [3, 13]
    with torch.no_grad():
        a, b = model(x), model(y)
    assert not torch.allclose(a[:, :, 13], b[:, :, 13])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast dtype policy not reproducible on CPU")
def test_forward_under_cuda_autocast():
    # Regression: the group scatter must tolerate autocast's mixed dtypes
    # (fp16 embedding output, fp32 LayerNorm-terminated encoder output).
    model = _build(mcb.leader_follower_groups(), 21).cuda()
    model.train()
    x = torch.randn(2, _SEQ_LEN, 21, device="cuda")
    with torch.autocast("cuda"):
        out = model(x)
    assert out.shape == (2, _PRED_LEN, 21)


def test_parameter_parity_with_vanilla_cd():
    # Same encoder config, shared across groups: parameter count must equal a
    # vanilla CD built from the same pieces (embed + encoder + shared head).
    torch.manual_seed(0)
    block = _build(mcb.leader_follower_groups(), 21)

    proj = nn.Linear(_PATCH, _D_MODEL)
    layer = nn.TransformerEncoderLayer(
        d_model=_D_MODEL, nhead=_N_HEADS, dim_feedforward=_D_MODEL * 4, dropout=_DROPOUT, batch_first=True
    )
    encoder = nn.TransformerEncoder(layer, num_layers=_N_LAYERS)
    head = nn.Linear(mcb.num_patches(_SEQ_LEN, _PATCH, _STRIDE) * _D_MODEL, _PRED_LEN)
    vanilla_total = sum(p.numel() for m in (proj, encoder, head) for p in m.parameters())

    block_total = sum(p.numel() for p in block.parameters())
    assert block_total == vanilla_total


def test_singleton_groups_match_ci_scope():
    # All-singleton partition means no cross-channel attention at all: changing
    # one channel leaves every other channel's output untouched.
    torch.manual_seed(0)
    model = _build([[c] for c in range(7)], 7)
    model.eval()
    x = torch.randn(1, _SEQ_LEN, 7)
    y = x.clone()
    y[:, :, 2] += 10.0
    with torch.no_grad():
        a, b = model(x), model(y)
    others = [c for c in range(7) if c != 2]
    assert torch.allclose(a[:, :, others], b[:, :, others], atol=1e-6)