"""PatchTST models for experiment 1 (head-bottleneck control).

The channel-independent (CI) and channel-dependent (CD) classes reproduce the
keystone notebook train_leader_follower exactly: a Linear patch embedding with no
positional encoding, patches taken by unfold with no padding so that
N = (seq_len - patch_size) // stride + 1, an nn.TransformerEncoder built from
nn.TransformerEncoderLayer (post-norm, ReLU feed-forward, dim_feedforward =
4 * d_model), and a shared per-variate head Linear(N * d_model, pred_len). The CD
encoder attends jointly over all C * N tokens; the head still projects each
variate independently.

PatchTST_CD_Head answers one objection: the per-variate head may discard the
cross-variate information the CD encoder mixes, so the RQ2 null could be a head
artefact rather than an architectural fact. It adds an explicit cross-variate
pool at the head and changes nothing else. After the unchanged CD encoder, each
variate's N tokens are mean-pooled to one summary vector, a single multi-head
attention over the C summary tokens produces a per-variate context, and that
context is concatenated to the variate's flattened encoder output before the
shared projection. The encoder is identical to PatchTST_CD, so any difference in
results is attributable to the head alone.

This deliberately avoids the global head Linear(C * N * D, pred_len * C) that
failed catastrophically in earlier runs (it could not train and stopped at epoch
one); the pooled-context head keeps the parameter count near the canonical head
and remains trainable.

All three PatchTST classes take (B, seq_len, C) and return (B, pred_len, C).
"""

import torch
import torch.nn as nn

__all__ = ["PatchEmbedding", "PatchTST_CI", "PatchTST_CD", "PatchTST_CD_Head", "TrueDLinear", "build_model"]

_MLP_RATIO: int = 4
_DLINEAR_KERNEL: int = 25


def num_patches(seq_len: int, patch_size: int, stride: int) -> int:
    """Patch count for the no-padding unfold used throughout this project.

    Args:
        seq_len: Input sequence length.
        patch_size: Timesteps per patch.
        stride: Stride between patches.

    Returns:
        N = (seq_len - patch_size) // stride + 1.
    """
    return (seq_len - patch_size) // stride + 1


class PatchEmbedding(nn.Module):
    """Linear patch projection with dropout and no positional encoding.

    Matches the keystone notebook: positional information is not added, which is
    part of the flattened-token CD behaviour under study.
    """

    def __init__(self, patch_size: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(patch_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project patches of shape (..., patch_size) to (..., d_model)."""
        return self.dropout(self.proj(x))


def _build_encoder(d_model: int, n_heads: int, n_layers: int, dropout: float) -> nn.TransformerEncoder:
    """Construct the post-norm Transformer encoder used by every PatchTST mode."""
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=d_model * _MLP_RATIO,
        dropout=dropout,
        batch_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=n_layers)


class PatchTST_CI(nn.Module):
    """Channel-independent PatchTST. Input (B, L, C) -> (B, pred_len, C)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_variates: int,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.n_patches = num_patches(seq_len, patch_size, stride)
        self.embed = PatchEmbedding(patch_size, d_model, dropout)
        self.encoder = _build_encoder(d_model, n_heads, n_layers, dropout)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forecast each variate independently with shared weights."""
        b, length, c = x.shape
        flat = x.permute(0, 2, 1).reshape(b * c, length)
        patches = flat.unfold(-1, self.patch_size, self.stride)  # (B*C, N, P)
        encoded = self.encoder(self.embed(patches))  # (B*C, N, D)
        out = self.head(encoded.reshape(b * c, -1))  # (B*C, pred_len)
        return out.reshape(b, c, -1).permute(0, 2, 1)


class PatchTST_CD(nn.Module):
    """Channel-dependent PatchTST with per-variate head. Input (B, L, C) -> (B, pred_len, C)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_variates: int,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.n_patches = num_patches(seq_len, patch_size, stride)
        self.embed = PatchEmbedding(patch_size, d_model, dropout)
        self.encoder = _build_encoder(d_model, n_heads, n_layers, dropout)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode all variates jointly, then project each variate independently."""
        b, length, c = x.shape
        flat = x.permute(0, 2, 1).reshape(b * c, length)
        patches = flat.unfold(-1, self.patch_size, self.stride)  # (B*C, N, P)
        emb = self.embed(patches).reshape(b, c, self.n_patches, self.d_model)  # (B, C, N, D)
        seq = emb.reshape(b, c * self.n_patches, self.d_model)  # (B, C*N, D)
        encoded = self.encoder(seq).reshape(b * c, -1)  # (B*C, N*D)
        out = self.head(encoded)  # (B*C, pred_len)
        return out.reshape(b, c, -1).permute(0, 2, 1)


class PatchTST_CD_Head(nn.Module):
    """Channel-dependent PatchTST with an explicit cross-variate head.

    The encoder is identical to PatchTST_CD. The head additionally pools each
    variate's encoder output to a summary vector, attends across the C summaries
    so every variate receives a context shaped by the others, and concatenates
    that per-variate context to the flattened encoder output before projecting.
    Input (B, L, C) -> (B, pred_len, C).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_variates: int,
        patch_size: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.n_patches = num_patches(seq_len, patch_size, stride)
        self.embed = PatchEmbedding(patch_size, d_model, dropout)
        self.encoder = _build_encoder(d_model, n_heads, n_layers, dropout)
        self.summary_norm = nn.LayerNorm(d_model)
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.head = nn.Linear(self.n_patches * d_model + d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode jointly, pool a per-variate cross-channel context, then project."""
        b, length, c = x.shape
        flat = x.permute(0, 2, 1).reshape(b * c, length)
        patches = flat.unfold(-1, self.patch_size, self.stride)  # (B*C, N, P)
        emb = self.embed(patches).reshape(b, c, self.n_patches, self.d_model)  # (B, C, N, D)
        seq = emb.reshape(b, c * self.n_patches, self.d_model)  # (B, C*N, D)
        encoded = self.encoder(seq).reshape(b, c, self.n_patches, self.d_model)  # (B, C, N, D)

        summary = self.summary_norm(encoded.mean(dim=2))  # (B, C, D): one token per variate
        context, _ = self.channel_attn(summary, summary, summary)  # (B, C, D): cross-variate pool

        per_variate = encoded.reshape(b, c, self.n_patches * self.d_model)  # (B, C, N*D)
        fused = torch.cat([per_variate, context], dim=-1)  # (B, C, N*D + D)
        out = self.head(fused.reshape(b * c, -1))  # (B*C, pred_len)
        return out.reshape(b, c, -1).permute(0, 2, 1)


class TrueDLinear(nn.Module):
    """DLinear (Zeng et al., 2023): moving-average trend plus remainder, two independent linears.

    Input (B, L, C) -> (B, pred_len, C). Branches are not weight-shared.
    """

    def __init__(self, seq_len: int, pred_len: int) -> None:
        super().__init__()
        pad = (_DLINEAR_KERNEL - 1) // 2
        self.seq_len = seq_len
        self.avg_pool = nn.AvgPool1d(kernel_size=_DLINEAR_KERNEL, stride=1, padding=pad)
        self.linear_trend = nn.Linear(seq_len, pred_len)
        self.linear_remainder = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decompose into trend and remainder, project each, and sum."""
        b, length, c = x.shape
        xf = x.permute(0, 2, 1).reshape(b * c, 1, length)
        trend = self.avg_pool(xf).reshape(b * c, length)[:, :length]
        remainder = xf.reshape(b * c, length) - trend
        out = self.linear_trend(trend) + self.linear_remainder(remainder)
        return out.reshape(b, c, -1).permute(0, 2, 1)


def build_model(
    mode: str,
    seq_len: int,
    pred_len: int,
    num_variates: int,
    patch_size: int = 16,
    stride: int = 8,
    d_model: int = 64,
    n_heads: int = 8,
    n_layers: int = 3,
    dropout: float = 0.2,
) -> nn.Module:
    """Construct a model by mode name.

    Args:
        mode: One of "CI", "CD", "CD_Head", "DLinear".
        seq_len: Input sequence length.
        pred_len: Forecast horizon.
        num_variates: Number of channels C.
        patch_size: Timesteps per patch (PatchTST modes).
        stride: Patch stride (PatchTST modes).
        d_model: Transformer model dimension.
        n_heads: Attention heads.
        n_layers: Encoder layers.
        dropout: Dropout rate.

    Returns:
        The constructed nn.Module.

    Raises:
        ValueError: If mode is unknown.
    """
    kwargs = dict(
        seq_len=seq_len,
        pred_len=pred_len,
        num_variates=num_variates,
        patch_size=patch_size,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout,
    )
    if mode == "CI":
        return PatchTST_CI(**kwargs)
    if mode == "CD":
        return PatchTST_CD(**kwargs)
    if mode == "CD_Head":
        return PatchTST_CD_Head(**kwargs)
    if mode == "DLinear":
        return TrueDLinear(seq_len=seq_len, pred_len=pred_len)
    raise ValueError(f"Unknown mode: {mode}")
