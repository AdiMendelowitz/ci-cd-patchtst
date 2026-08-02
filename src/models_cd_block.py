"""Block-wise cross-variate attention variant of channel-dependent PatchTST.

Purpose
-------
Vanilla CD flattens all channels into one token sequence of length C * N and
pays an attention cost of O((C * N)^2), which collapses the feasible batch size
as C grows. This variant partitions the channels into groups and runs the same
Transformer encoder within each group independently, so the score-matrix cost
falls to O(sum_g (|g| * N)^2). With G equal groups of size b this is a factor
b / C of the vanilla CD cost. No attention mask is used anywhere: masking a
full (C*N, C*N) score matrix would still allocate it, so the saving is obtained
structurally, by never forming cross-group scores.

Semantics
---------
Channels in the same group attend to each other exactly as in vanilla CD;
channels in different groups never interact inside the encoder. A singleton
group reproduces CI behaviour for that channel. The encoder weights are shared
across groups, so the parameter count equals vanilla CD with the same
configuration and any accuracy difference is attributable to the attention
scope, not to capacity.

Partition choice is part of the experiment design, not a tuning knob. For the
leader-follower cell the partition must keep each leader with its follower
(leader_follower_groups), otherwise the variant is structurally unable to
express the lag-1 coupling and the ablation tests nothing. For block-covariance
or AR(1) grids, contiguous groups (contiguous_groups) align with, or are
neutral to, the generative structure.

The class is self-contained (no notebook globals); the training notebook's
constants are passed to the constructor.
"""

import torch
import torch.nn as nn


def num_patches(seq_len: int, patch_size: int, stride: int) -> int:
    """Number of patches produced by unfold(seq_len, patch_size, stride)."""
    return (seq_len - patch_size) // stride + 1


def contiguous_groups(num_variates: int, group_size: int) -> list[list[int]]:
    """Partition channels 0..C-1 into contiguous groups of equal size.

    Args:
        num_variates: Total channel count C.
        group_size: Channels per group; must divide C.

    Returns:
        List of index lists, group j holding channels j*b .. (j+1)*b - 1.

    Raises:
        ValueError: If group_size does not divide num_variates.
    """
    if num_variates % group_size != 0:
        raise ValueError(f"C={num_variates} is not divisible by group_size={group_size}.")
    return [list(range(j, j + group_size)) for j in range(0, num_variates, group_size)]


def leader_follower_groups(n_pairs: int = 10, isolate_idx: int = 20) -> list[list[int]]:
    """Pair-preserving partition for the leader-follower C=21 layout.

    Leader k occupies index k and its follower index k + n_pairs; the isolate is
    a singleton group and therefore behaves as CI, matching its role as the
    internal negative control.

    Args:
        n_pairs: Number of leader-follower pairs.
        isolate_idx: Index of the isolate channel.

    Returns:
        n_pairs groups of size two plus one singleton, covering 0..2*n_pairs.
    """
    groups = [[k, n_pairs + k] for k in range(n_pairs)]
    groups.append([isolate_idx])
    return groups


def _validate_partition(groups: list[list[int]], num_variates: int) -> None:
    """Require groups to cover 0..C-1 exactly once."""
    flat = [i for g in groups for i in g]
    if sorted(flat) != list(range(num_variates)):
        raise ValueError(f"Groups must partition 0..{num_variates - 1} exactly once; got {sorted(flat)}.")
    if any(len(g) == 0 for g in groups):
        raise ValueError("Empty group in partition.")


class PatchTST_CD_Block(nn.Module):
    """CD PatchTST with block-wise cross-variate attention. (B, L, C) -> (B, pred_len, C).

    Embedding, encoder configuration, and the shared per-variate head are
    identical to the notebook's PatchTST_CD; only the attention scope differs.
    """

    def __init__(
        self,
        groups: list[list[int]],
        num_variates: int,
        seq_len: int,
        pred_len: int,
        patch_size: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if patch_size <= 0 or stride <= 0 or seq_len < patch_size:
            raise ValueError(f"Invalid patching: seq_len={seq_len}, patch_size={patch_size}, stride={stride}.")
        _validate_partition(groups, num_variates)
        self.num_variates = num_variates
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        self.n_patches = num_patches(seq_len, patch_size, stride)

        self.proj = nn.Linear(patch_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patches * d_model, pred_len)

        # Groups of equal size are batched into one encoder call. Index tensors
        # are buffers so .to(device) carries them along.
        by_size: dict[int, list[list[int]]] = {}
        for g in groups:
            by_size.setdefault(len(g), []).append(sorted(g))
        self._sizes: list[int] = sorted(by_size)
        for size in self._sizes:
            idx = torch.tensor(by_size[size], dtype=torch.long)  # (n_groups, size)
            self.register_buffer(f"_idx_{size}", idx, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, Cv = x.shape
        if Cv != self.num_variates:
            raise ValueError(f"Expected {self.num_variates} channels, got {Cv}.")
        n, d = self.n_patches, self.d_model

        xp = x.permute(0, 2, 1).reshape(B * Cv, L)
        p = xp.unfold(-1, self.patch_size, self.stride)  # (B*C, N, P)
        emb = self.embed_dropout(self.proj(p)).reshape(B, Cv, n, d)

        out = emb.new_empty(B, Cv, n, d)
        for size in self._sizes:
            idx: torch.Tensor = getattr(self, f"_idx_{size}")  # (n_groups, size)
            n_groups = idx.shape[0]
            g = emb[:, idx.reshape(-1)]  # (B, n_groups*size, N, D)
            g = g.reshape(B * n_groups, size * n, d)
            enc = self.encoder(g).reshape(B, n_groups * size, n, d)
            out[:, idx.reshape(-1)] = enc

        return self.head(out.reshape(B * Cv, n * d)).reshape(B, Cv, -1).permute(0, 2, 1)
