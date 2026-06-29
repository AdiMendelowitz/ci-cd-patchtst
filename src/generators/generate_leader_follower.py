"""Leader-follower VAR(1) data generator for the coupled synthetic experiment.

Design
------
Channels are partitioned into three groups within a C=21 multivariate process:
  - Leaders   (indices 0–9):   pure AR(1), no cross-channel dependence.
  - Followers (indices 10–19): each follows the corresponding leader with lag-1
                                coupling of strength gamma.
  - Isolate   (index 20):      pure AR(1), no cross-channel dependence.
                                Serves as an internal negative control.

The transition matrix A(gamma) is:

    A[i, i]          = phi           for all i          (autoregressive term)
    A[j, leader(j)]  = gamma         for j in followers  (leader->follower coupling)
    all other entries = 0

where leader(j) = j - 10 for j in {10, ..., 19}.

Stability
---------
A(gamma) is lower-triangular (leaders before followers in index order). Its
eigenvalues are phi repeated C times, giving spectral radius phi = 0.8 < 1
for all finite gamma. The process is therefore stationary for any gamma value
used here, without requiring a stability constraint on gamma.

Granger causality
-----------------
By construction, leader i Granger-causes follower i+10 (the coupling is
explicit in the transition), but the reverse does not hold (follower's past
does not appear in leader's transition equation). Leaders and the isolate
variate are Granger non-causal with respect to all other channels.
This structure is verified analytically and empirically by validate_granger.py.

Innovation covariance
---------------------
Innovations use compound-symmetry covariance with off-diagonal value rho,
identical to the original AR(1) grid. This means contemporaneous correlation
is present alongside lagged coupling, matching realistic data conditions.

Output schema
-------------
The generated array has shape (T, C) with channels in the order described
above. Callers are responsible for splitting, normalising, and saving.
The saved CSV must include a 'gamma' column to distinguish runs from the
original AR(1) grid (which has gamma=0 implicitly).

Usage
-----
Run from time-series-forecasting/ or import generate() directly:

    from generate_leader_follower import generate, make_transition_matrix

    data = generate(phi=0.8, gamma=0.3, rho=0.5, T=14400, burn_in=1000, seed=42)
"""

import argparse

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

N_LEADERS:   int = 10
N_FOLLOWERS: int = 10
N_ISOLATE:   int = 1
C_TOTAL:     int = N_LEADERS + N_FOLLOWERS + N_ISOLATE  # 21

# Index boundaries
_FOLLOWER_SLICE: slice = slice(N_LEADERS, N_LEADERS + N_FOLLOWERS)
_ISOLATE_IDX:    int   = N_LEADERS + N_FOLLOWERS  # 20

# Training configuration matching the original AR(1) grid
T_TOTAL:  int = 14_400
BURN_IN:  int = 1_000
TRAIN_FRAC: float = 0.6
VAL_FRAC:   float = 0.2


# ── Core functions ────────────────────────────────────────────────────────────

def make_transition_matrix(phi: float, gamma: float) -> np.ndarray:
    """Construct the C_TOTAL x C_TOTAL transition matrix A(phi, gamma).

    Args:
        phi:   Autoregressive coefficient (same for all channels).
        gamma: Leader-to-follower coupling strength.

    Returns:
        Float64 array of shape (C_TOTAL, C_TOTAL).
    """
    A = np.zeros((C_TOTAL, C_TOTAL), dtype=np.float64)
    # Diagonal: autoregressive terms for all channels.
    np.fill_diagonal(A, phi)
    # Off-diagonal: leader -> follower coupling.
    for k in range(N_FOLLOWERS):
        follower_idx = N_LEADERS + k
        leader_idx   = k
        A[follower_idx, leader_idx] = gamma
    return A


def make_innovation_covariance(rho: float) -> np.ndarray:
    """Compound-symmetry covariance matrix for innovations.

    Sigma[i, i] = 1, Sigma[i, j] = rho for i != j.
    Identical to the original AR(1) grid covariance.

    Args:
        rho: Off-diagonal correlation strength in [0, 1).

    Returns:
        Float64 array of shape (C_TOTAL, C_TOTAL).

    Raises:
        ValueError: If rho is outside [0, 1) or the resulting matrix is not
                    positive definite.
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(
            f"rho must be in [0, 1), got {rho}. "
            f"Negative rho (negative contemporaneous correlation) is not used "
            f"in this paper's experiments and is excluded to keep the guard "
            f"simple. Remove this check if negative rho is needed."
        )
    Sigma = np.full((C_TOTAL, C_TOTAL), rho, dtype=np.float64)
    np.fill_diagonal(Sigma, 1.0)
    # Positive definiteness: compound-symmetry is PD iff rho > -1/(C-1).
    # With rho >= 0 this always holds, but check the eigenvalue for safety.
    min_eig = np.linalg.eigvalsh(Sigma).min()
    if min_eig <= 0:
        raise ValueError(
            f"Innovation covariance is not positive definite for rho={rho} "
            f"(min eigenvalue={min_eig:.6f})."
        )
    return Sigma


def generate(
    phi: float,
    gamma: float,
    rho: float,
    T: int = T_TOTAL,
    burn_in: int = BURN_IN,
    seed: int = 42,
) -> np.ndarray:
    """Simulate a leader-follower VAR(1) process.

    Args:
        phi:     Autoregressive coefficient (spectral radius of A = phi < 1).
        gamma:   Leader-to-follower coupling strength. The process is stable
                 for all finite gamma because A(gamma) is lower-triangular with
                 eigenvalues all equal to phi.
        rho:     Off-diagonal contemporaneous correlation in innovations.
        T:       Number of timesteps to return (after burn-in discard).
        burn_in: Number of initial steps to discard.
        seed:    NumPy random seed for reproducibility.

    Returns:
        Float64 array of shape (T, C_TOTAL).
    """
    rng   = np.random.default_rng(seed)
    A     = make_transition_matrix(phi, gamma)
    Sigma = make_innovation_covariance(rho)
    L     = np.linalg.cholesky(Sigma)  # for efficient multivariate normal sampling

    total_steps = T + burn_in
    X = np.zeros((total_steps, C_TOTAL), dtype=np.float64)
    # The per-step loop is intentional. The VAR(1) recurrence is inherently
    # sequential (X[t] depends on X[t-1]), and the noise is drawn one step at
    # a time to match the RNG draw order used historically. Pre-drawing all
    # noise as (total_steps-1, C_TOTAL) and applying L via matmul would be
    # numerically equivalent and seed-compatible, but the gain at T=15,400
    # and C=21 is negligible since generate() is called once per training run.
    for t in range(1, total_steps):
        eps  = L @ rng.standard_normal(C_TOTAL)
        X[t] = A @ X[t - 1] + eps

    return X[burn_in:]  # shape (T, C_TOTAL)


def split_and_normalise(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split (T, C) data into train/val/test and apply per-channel z-score.

    Normalisation statistics are fit on the training split only, identical
    to the original AR(1) grid protocol.

    Args:
        data: Float64 array of shape (T, C).

    Returns:
        Tuple of (train, val, test) arrays, each normalised.
    """
    T = len(data)
    n_train = int(T * TRAIN_FRAC)
    n_val   = int(T * VAL_FRAC)

    train = data[:n_train]
    val   = data[n_train: n_train + n_val]
    test  = data[n_train + n_val:]

    mean = train.mean(axis=0, keepdims=True)
    std  = train.std(axis=0, keepdims=True)
    std  = np.where(std == 0, 1.0, std)  # guard against constant channels

    return (train - mean) / std, (val - mean) / std, (test - mean) / std


# ── Empirical correlation check ───────────────────────────────────────────────

def empirical_correlation_summary(data: np.ndarray, rho: float) -> pd.DataFrame:
    """Compute empirical lag-0 cross-correlation and compare to nominal rho.

    Useful for verifying that the data-generating process behaved as intended,
    consistent with the original AR(1) grid's validation protocol.

    Args:
        data: Float64 array of shape (T, C) — use the training split.
        rho:  Nominal innovation correlation.

    Returns:
        DataFrame with one row per channel-group (leaders, followers, isolate)
        summarising mean and median absolute off-diagonal Pearson correlation.
    """
    # Group-level summaries (per-group off-diagonal absolute correlations).
    # The global off-diagonal values are not returned; group-level breakdown
    # is more informative for diagnosing whether leaders, followers, and the
    # isolate have the expected within-group correlation structure.
    groups = {
        "leaders":   list(range(N_LEADERS)),
        "followers": list(range(N_LEADERS, N_LEADERS + N_FOLLOWERS)),
        "isolate":   [_ISOLATE_IDX],
    }
    rows = []
    for name, idxs in groups.items():
        if len(idxs) < 2:
            rows.append({"group": name, "mean_abs_r": float("nan"),
                         "median_abs_r": float("nan"), "nominal_rho": rho})
            continue
        sub = np.corrcoef(data[:, idxs].T)
        si, sj = np.triu_indices(len(idxs), k=1)
        vals = np.abs(sub[si, sj])
        rows.append({"group": name, "mean_abs_r": float(vals.mean()),
                     "median_abs_r": float(np.median(vals)), "nominal_rho": rho})
    return pd.DataFrame(rows)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """Generate and validate one dataset configuration, print a summary.

    Used for quick sanity-checking the generator before running Kaggle jobs.
    Does not save to disk; the Kaggle notebook is responsible for I/O.
    """
    parser = argparse.ArgumentParser(
        description="Generate a leader-follower VAR(1) sample and print diagnostics."
    )
    parser.add_argument("--phi",   type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--rho",   type=float, default=0.5)
    parser.add_argument("--seed",  type=int,   default=42)
    args = parser.parse_args()

    print(f"Generating: phi={args.phi}, gamma={args.gamma}, rho={args.rho}, "
          f"seed={args.seed}, T={T_TOTAL}, burn_in={BURN_IN}")

    data = generate(phi=args.phi, gamma=args.gamma, rho=args.rho,
                    T=T_TOTAL, burn_in=BURN_IN, seed=args.seed)
    train, val, test = split_and_normalise(data)

    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    A = make_transition_matrix(args.phi, args.gamma)
    spec_radius = np.max(np.abs(np.linalg.eigvals(A)))
    print(f"Transition matrix spectral radius: {spec_radius:.6f}  (must be < 1)")

    summary = empirical_correlation_summary(train, args.rho)
    print("\nEmpirical lag-0 correlation by group (training split):")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()