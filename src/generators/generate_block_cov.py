"""Block-diagonal-covariance AR(1) data generator for experiment 3.

Purpose
-------
Tests whether the RQ1 null (no CD advantage from purely instantaneous
correlation) survives a covariance family other than compound symmetry.
Compound symmetry is treated as the configuration most hostile to CD; this
generator supplies the second covariance family needed to test that claim.

Design
------
The process is a vector AR(1) with a diagonal transition matrix:

    x_t = phi * x_{t-1} + eps_t,    eps_t ~ N(0, Sigma)

with phi applied on the diagonal only (transition is phi * I). Because the
transition operator is diagonal, Granger non-causality holds exactly across all
channel pairs at every lag, identical to the compound-symmetry AR(1) grid. The
only structural change is Sigma: instead of one global off-diagonal value, the
channels are partitioned into equal groups and correlation is placed within
groups (rho_in) and, optionally, between groups (rho_out, default 0).

This isolates the same single variable as the original grid, instantaneous
correlation, while changing its *shape* from flat (compound symmetry) to
block-structured. A diagonal transition keeps the RQ1 assumption intact: any CD
advantage here would have to come from instantaneous correlation alone, because
no lagged cross-channel signal exists by construction.

RNG and protocol
----------------
The simulation path is identical to generate_leader_follower.generate: a single
numpy Generator (np.random.default_rng(seed)), a per-step recurrence, and
innovations drawn one step at a time as L @ standard_normal(C) where L is the
Cholesky factor of Sigma. Burn-in, length, the 60/20/20 split, and per-channel
z-score fit on the training split only all match the canonical grid. Numbers are
not expected to reproduce the compound-symmetry grid (a different covariance is a
different process); the requirement is internal CI-versus-CD comparability, which
holds because both modes receive the identical data array per seed.

Output schema
-------------
generate() returns a float64 array of shape (T, C) with channels grouped in
index order (group 0 occupies indices 0..group_size-1, and so on). Callers split,
normalise, and save. A saved CSV should record rho_in, rho_out, and group_size so
runs are distinguishable from the compound-symmetry grid.

Usage
-----
    from generate_block_cov import generate, make_block_covariance

    data = generate(phi=0.8, rho_in=0.9, C=84, group_size=7, rho_out=0.0,
                    T=14400, burn_in=1000, seed=42)
"""

import argparse

import numpy as np
import pandas as pd

# Protocol constants, identical to the compound-symmetry AR(1) grid.
T_TOTAL: int = 14_400
BURN_IN: int = 1_000
TRAIN_FRAC: float = 0.6
VAL_FRAC: float = 0.2
DEFAULT_GROUP_SIZE: int = 7


def make_transition_matrix(phi: float, C: int) -> np.ndarray:
    """Construct the diagonal transition matrix phi * I.

    Args:
        phi: Autoregressive coefficient, shared across channels.
        C: Number of channels.

    Returns:
        Float64 array of shape (C, C) with phi on the diagonal and zeros
        elsewhere, giving Granger non-causality across all channel pairs.
    """
    return np.eye(C, dtype=np.float64) * phi


def make_block_covariance(C: int, group_size: int, rho_in: float, rho_out: float = 0.0) -> np.ndarray:
    """Construct a block-structured innovation covariance matrix.

    Channels are partitioned into C // group_size equal contiguous groups.
    Within a group every off-diagonal entry is rho_in; between groups every
    entry is rho_out. Diagonals are unit variance.

    Args:
        C: Number of channels. Must be divisible by group_size.
        group_size: Number of channels per group.
        rho_in: Within-group contemporaneous correlation, in [0, 1).
        rho_out: Between-group contemporaneous correlation, in [0, 1).
            Defaults to 0.0 (independent groups).

    Returns:
        Float64 array of shape (C, C).

    Raises:
        ValueError: If C is not divisible by group_size, if rho_in or rho_out is
            outside [0, 1), if rho_out >= rho_in, or if the matrix is not
            positive definite.
    """
    if C % group_size != 0:
        raise ValueError(f"C={C} must be divisible by group_size={group_size}.")
    if not 0.0 <= rho_in < 1.0:
        raise ValueError(f"rho_in must be in [0, 1), got {rho_in}.")
    if not 0.0 <= rho_out < 1.0:
        raise ValueError(f"rho_out must be in [0, 1), got {rho_out}.")
    if rho_out >= rho_in:
        raise ValueError(
            f"rho_out ({rho_out}) must be strictly less than rho_in ({rho_in}); "
            f"block structure requires stronger within-group correlation."
        )

    n_groups = C // group_size
    group_id = np.repeat(np.arange(n_groups), group_size)
    same_group = group_id[:, None] == group_id[None, :]

    sigma = np.where(same_group, rho_in, rho_out).astype(np.float64)
    np.fill_diagonal(sigma, 1.0)

    min_eig = float(np.linalg.eigvalsh(sigma).min())
    if min_eig <= 0.0:
        raise ValueError(
            f"Block covariance is not positive definite for rho_in={rho_in}, "
            f"rho_out={rho_out}, group_size={group_size} (min eigenvalue={min_eig:.6f})."
        )
    return sigma


def generate(
    phi: float,
    rho_in: float,
    C: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    rho_out: float = 0.0,
    T: int = T_TOTAL,
    burn_in: int = BURN_IN,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a block-diagonal-covariance AR(1) process.

    Args:
        phi: Autoregressive coefficient (spectral radius phi < 1 for stability).
        rho_in: Within-group contemporaneous correlation.
        C: Number of channels.
        group_size: Channels per group. Defaults to DEFAULT_GROUP_SIZE.
        rho_out: Between-group correlation. Defaults to 0.0.
        T: Number of timesteps returned after burn-in.
        burn_in: Number of initial steps discarded.
        seed: Seed for numpy's default_rng.

    Returns:
        Float64 array of shape (T, C).
    """
    rng = np.random.default_rng(seed)
    A = make_transition_matrix(phi, C)
    L = np.linalg.cholesky(make_block_covariance(C, group_size, rho_in, rho_out))

    total_steps = T + burn_in
    X = np.zeros((total_steps, C), dtype=np.float64)
    for t in range(1, total_steps):
        X[t] = A @ X[t - 1] + L @ rng.standard_normal(C)
    return X[burn_in:]


def split_and_normalise(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split (T, C) data 60/20/20 and apply per-channel z-score.

    Statistics are fit on the training split only, matching the canonical grid
    protocol and preventing leakage from validation or test.

    Args:
        data: Float64 array of shape (T, C).

    Returns:
        Tuple of (train, val, test) arrays, each per-channel z-scored.
    """
    T = len(data)
    n_train = int(T * TRAIN_FRAC)
    n_val = int(T * VAL_FRAC)

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)

    return (train - mean) / std, (val - mean) / std, (test - mean) / std


def empirical_block_summary(data: np.ndarray, C: int, group_size: int, rho_in: float, rho_out: float) -> pd.DataFrame:
    """Compare empirical within-group and between-group absolute correlation to nominal values.

    Verifies the data-generating process produced the intended block structure,
    consistent with the grid's empirical-correlation validation protocol.

    Args:
        data: Float64 array of shape (T, C); use the training split.
        C: Number of channels.
        group_size: Channels per group.
        rho_in: Nominal within-group correlation.
        rho_out: Nominal between-group correlation.

    Returns:
        DataFrame with two rows (within-group, between-group) reporting the mean
        absolute Pearson correlation and the nominal target.
    """
    corr = np.corrcoef(data.T)
    group_id = np.repeat(np.arange(C // group_size), group_size)
    iu, ju = np.triu_indices(C, k=1)
    same = group_id[iu] == group_id[ju]
    abs_r = np.abs(corr[iu, ju])

    return pd.DataFrame(
        [
            {"pairs": "within_group", "mean_abs_r": float(abs_r[same].mean()), "nominal": rho_in},
            {"pairs": "between_group", "mean_abs_r": float(abs_r[~same].mean()), "nominal": rho_out},
        ]
    )


def main() -> None:
    """Generate one configuration and print structural diagnostics.

    Sanity-checks the generator before a Kaggle run. Does not write to disk; the
    training notebook owns I/O.
    """
    parser = argparse.ArgumentParser(description="Generate a block-covariance AR(1) sample and print diagnostics.")
    parser.add_argument("--phi", type=float, default=0.8)
    parser.add_argument("--rho-in", type=float, default=0.9)
    parser.add_argument("--rho-out", type=float, default=0.0)
    parser.add_argument("--C", type=int, default=84)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(
        f"Generating: phi={args.phi}, rho_in={args.rho_in}, rho_out={args.rho_out}, "
        f"C={args.C}, group_size={args.group_size}, n_groups={args.C // args.group_size}, seed={args.seed}"
    )

    data = generate(
        phi=args.phi,
        rho_in=args.rho_in,
        C=args.C,
        group_size=args.group_size,
        rho_out=args.rho_out,
        seed=args.seed,
    )
    train, val, test = split_and_normalise(data)
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    A = make_transition_matrix(args.phi, args.C)
    spec_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    print(f"Transition spectral radius: {spec_radius:.6f}  (must be < 1)")

    summary = empirical_block_summary(train, args.C, args.group_size, args.rho_in, args.rho_out)
    print("\nEmpirical lag-0 correlation (training split):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
