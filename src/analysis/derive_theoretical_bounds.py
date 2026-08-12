"""Closed-form theoretical MSE bounds for CI vs CD across the synthetic families.

Reads nothing: every number is derived analytically from committed protocol
constants (phi, gamma, rho, C, pred_len). No results CSV is read or
required: the benchmark is derivable without new training runs.

Method
------
For a stationary VAR(1) x_t = A x_(t-1) + eps_t, eps_t ~ N(0, Sigma), the
population-optimal h-step forecast given full history is A^h x_t0, with
error covariance Sigma_h = sum_{k=0}^{h-1} A^k Sigma (A^k)^T. [Sigma_h]_ii is
channel i's unrestricted (CD) ceiling.

Lemma: when A is diagonal, the channel-independent (own-history-only)
optimal predictor attains the same [Sigma_h]_ii exactly, regardless of
Sigma's off-diagonal structure: innovations are iid across time, so no
other channel's past reveals anything about channel i's future innovations.
Diagonal A implies CI bound = CD bound exactly.

Normalisation: test_mse in the committed CSVs is reported on z-scored data
(per-channel, train-split statistics). Dividing Sigma_h's diagonal by each
channel's stationary variance gives the bound in the same units. For a
diagonal-A family this reduces to the closed form 1 - phi**(2h),
independent of rho, C, and group structure.

Leader-follower has a non-diagonal A (lag-1 leader->follower coupling), so
the lemma does not apply to followers directly. Leaders and the isolate
channel receive no incoming coupling (A is diagonal for their own rows), so
their bound is the exact diagonal ceiling at every horizon, same as every
other diagonal family; reported as single h=96 rows, consistent with the
rest of the table. Followers get two bounds that sandwich the true CI
ceiling: CD_bound <= CI_bound <= naive_bound = 1 (conditioning on strictly
more information never increases MSE). The sandwich width (naive_bound -
CD_bound) is an upper bound on CI_bound - CD_bound, not the gap itself; it
is exact only where the width is already ~0. Followers are reported across
the full horizon sweep (h in {1,5,10,50,96}, all four gamma) rather than a
single h=96 point, since the sandwich only tightens to near-zero width close
to h=96; at shorter horizons the available signal is far from negligible
and varies with coupling strength.

Coverage
--------
AR(1) grid, block-covariance, boundary sweep, leader-follower. Also covers
results_cd_head.csv and results_block_attention.csv (same leader-follower
DGP, different model arm), and both cells of results_equal_compute.csv
(same AR(1)-grid and leader-follower DGPs, matched-update-budget training
variant). The bound is a property of the DGP, not of training budget or
model architecture, so no additional derivation is needed for any of these.

Excludes ETTh1 and ECL: real data has no known closed-form DGP.

Protocol constants
-------------------
phi = 0.8 and pred_len = 96 for every family. Verified directly against
results_grid.csv (all 135 rows: pred_len=96) and train_block_cov.ipynb
(PRED_LEN=96); the leader-follower and boundary-sweep notebooks confirm the
same pred_len. The AR(1) grid has no standalone generator script in this
repo (a pre-revision artifact); its structural form (diagonal transition,
compound-symmetry Sigma, phi=0.8) is reconstructed below rather than
imported.

Dependencies
------------
Reuses make_transition_matrix / make_innovation_covariance from
generate_leader_follower.py and make_transition_matrix / make_block_covariance
from generate_block_cov.py rather than re-deriving matrix structure that
already has a canonical definition. Both live in src/generators/, a
separate directory from this script's own src/analysis/, so that directory
is added to sys.path explicitly rather than relying on same-directory
auto-import.

Writes
------
results/Revision/theoretical_bounds.csv.

Row count: 42 (9 AR(1) grid + 4 block-covariance + 1 boundary sweep + 8
leader-follower leader/isolate rows [2 x 4 gamma] + 20 leader-follower
follower-sweep rows [5 horizons x 4 gamma]).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "generators"))

import generate_block_cov as bc
import generate_leader_follower as lf

PHI = 0.8
H = 96
SWEEP_H = (1, 5, 10, 50, 96)

AR_GRID_C = (7, 21, 84)
AR_GRID_RHO = (0.1, 0.5, 0.9)

BLOCK_COV_C = (21, 84)
BLOCK_COV_RHO_IN = (0.5, 0.9)
BLOCK_COV_RHO_OUT = 0.0
BLOCK_COV_GROUP_SIZE = 7

LF_GAMMA = (0.0, 0.3, 0.6, 0.9)
LF_RHO = 0.5
LF_ISOLATE_IDX = lf.N_LEADERS + lf.N_FOLLOWERS  # avoids depending on lf._ISOLATE_IDX (private)

TOL = 1e-9
OUT_PATH = Path("results/Revision/theoretical_bounds.csv")


def diagonal_ceiling(phi: float, h: int) -> float:
    """Exact closed-form CI=CD bound for any diagonal-transition family, in
    z-scored units. Independent of Sigma's off-diagonal structure."""
    return 1.0 - phi ** (2 * h)


def sigma_h_diag(A: np.ndarray, Sigma: np.ndarray, h: int) -> np.ndarray:
    """diag(sum_{k=0}^{h-1} A^k Sigma (A^k)^T), the raw (un-normalised)
    h-step forecast error variance per channel."""
    C = A.shape[0]
    total = np.zeros((C, C))
    A_power = np.eye(C)
    for _ in range(h):
        total += A_power @ Sigma @ A_power.T
        A_power = A @ A_power
    return np.diag(total)


def stationary_var_diag(A: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """diag(X), X solving the discrete Lyapunov equation X = A X A^T + Sigma."""
    return np.diag(solve_discrete_lyapunov(A, Sigma))


def normalized_bound(A: np.ndarray, Sigma: np.ndarray, h: int) -> np.ndarray:
    """h-step forecast MSE per channel, in z-scored (stationary-variance) units."""
    return sigma_h_diag(A, Sigma, h) / stationary_var_diag(A, Sigma)


def sandwich_width(cd_bound: float) -> float:
    """naive_bound (1.0) minus cd_bound, clipped at zero.

    A negative raw value is a float64 noise-floor artefact (accumulated
    rounding error from h iterated matrix products, ~1e-15 scale), not a
    real violation; CD_bound <= naive_bound always, by the conditioning
    monotonicity argument in the module docstring.
    """
    return max(0.0, 1.0 - cd_bound)


def compound_symmetry(rho: float, C: int) -> np.ndarray:
    """Compound-symmetry covariance: unit diagonal, rho off-diagonal.

    No standalone AR(1)-grid generator exists in this repo; this
    reconstructs its structural form directly. Formula matches
    generate_leader_follower.make_innovation_covariance, generalised over C.
    """
    Sigma = np.full((C, C), rho, dtype=np.float64)
    np.fill_diagonal(Sigma, 1.0)
    return Sigma


def verify_diagonal_family(name: str, A: np.ndarray, Sigma: np.ndarray) -> float:
    """Cross-check the general matrix computation against the closed form
    for a diagonal-A family. Raises if they disagree."""
    got = normalized_bound(A, Sigma, H)
    want = diagonal_ceiling(PHI, H)
    if not np.allclose(got, want, atol=TOL):
        raise AssertionError(
            f"{name}: matrix computation {got} disagrees with closed form "
            f"{want} beyond tolerance {TOL}"
        )
    return want


def main() -> int:
    rows = []

    print(f"phi={PHI}  h={H}  phi**(2h)={PHI ** (2 * H):.3e}\n")

    print("=== AR(1) grid ===")
    for C in AR_GRID_C:
        for rho in AR_GRID_RHO:
            A = PHI * np.eye(C)
            Sigma = compound_symmetry(rho, C)
            bound = verify_diagonal_family(f"AR(1) grid C={C} rho={rho}", A, Sigma)
            print(f"  C={C:>2}  rho={rho}: CI=CD bound = {bound:.10f}")
            rows.append({"family": "AR(1) grid", "cell": f"C={C}, rho={rho}", "h": H,
                         "bound_type": "CI=CD (exact)", "bound": bound, "sandwich_width": np.nan})

    print("\n=== Block-covariance ===")
    for C in BLOCK_COV_C:
        for rho_in in BLOCK_COV_RHO_IN:
            A = bc.make_transition_matrix(PHI, C)
            Sigma = bc.make_block_covariance(C, BLOCK_COV_GROUP_SIZE, rho_in, BLOCK_COV_RHO_OUT)
            bound = verify_diagonal_family(f"block-cov C={C} rho_in={rho_in}", A, Sigma)
            print(f"  C={C:>2}  rho_in={rho_in}: CI=CD bound = {bound:.10f}")
            rows.append({"family": "block-covariance", "cell": f"C={C}, rho_in={rho_in}", "h": H,
                         "bound_type": "CI=CD (exact)", "bound": bound, "sandwich_width": np.nan})

    print("\n=== Boundary sweep (all P, all gamma tranches) ===")
    print("  gamma/rho columns are inert metadata for this family.")
    bound = diagonal_ceiling(PHI, H)
    print(f"  CI=CD bound = {bound:.10f}  (independent of P, gamma, seed)")
    rows.append({"family": "boundary sweep", "cell": "all P, all gamma", "h": H,
                 "bound_type": "CI=CD (exact)", "bound": bound, "sandwich_width": np.nan})

    print("\n=== Leader-follower: leader / isolate (diagonal ceiling, h=96) ===")
    for gamma in LF_GAMMA:
        A = lf.make_transition_matrix(PHI, gamma)
        Sigma = lf.make_innovation_covariance(LF_RHO)
        bound = normalized_bound(A, Sigma, H)

        leader_bound = bound[0]
        isolate_bound = bound[LF_ISOLATE_IDX]
        for label, val in (("leader", leader_bound), ("isolate", isolate_bound)):
            if not np.isclose(val, diagonal_ceiling(PHI, H), atol=TOL):
                raise AssertionError(f"gamma={gamma}: {label} bound {val} != diagonal ceiling")
            print(f"  gamma={gamma}  {label}: CI=CD bound = {val:.10f}")
            rows.append({"family": "leader-follower", "cell": f"gamma={gamma}, {label}", "h": H,
                         "bound_type": "CI=CD (exact)", "bound": float(val), "sandwich_width": np.nan})

        follower_block = bound[lf.N_LEADERS:lf.N_LEADERS + lf.N_FOLLOWERS]
        if not np.allclose(follower_block, follower_block[0], atol=TOL):
            raise AssertionError(
                f"gamma={gamma}: followers disagree with each other {follower_block}"
            )

    print("\n=== Leader-follower: follower sandwich, full horizon sweep ===")
    print("  width upper-bounds the true CI-CD gap; equals it only where already ~0.")
    for gamma in LF_GAMMA:
        A = lf.make_transition_matrix(PHI, gamma)
        Sigma = lf.make_innovation_covariance(LF_RHO)
        stat_var = stationary_var_diag(A, Sigma)
        for h in SWEEP_H:
            follower_bound = float(sigma_h_diag(A, Sigma, h)[lf.N_LEADERS] / stat_var[lf.N_LEADERS])
            width = sandwich_width(follower_bound)
            print(f"  gamma={gamma}  h={h:>3}: follower CD bound = {follower_bound:.10f}  "
                  f"CI in [{follower_bound:.10f}, 1.0]  (sandwich width {width:.3e})")
            rows.append({"family": "leader-follower", "cell": f"gamma={gamma}, follower", "h": h,
                         "bound_type": "CD (sandwich upper=1.0)", "bound": follower_bound,
                         "sandwich_width": width})

    print(f"\nSummary: at h={H}, phi={PHI}, the theoretical CI=CD ceiling is indistinguishable")
    print(f"from 1.0 to float64 precision (phi**(2h)={PHI ** (2 * H):.3e}) for every")
    print("diagonal-transition family and cell, and for leader-follower's followers up to")
    print("the sandwich's floating-point noise floor. No model, CI or CD, has room to")
    print("exploit real predictive signal at this horizon.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        raise FileExistsError(f"{OUT_PATH} already exists, refusing to overwrite silently")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {len(rows)} rows to {OUT_PATH}")
    assert len(rows) == 42, f"expected 42 rows, got {len(rows)}: row-count contract broken"

    print("\nRESULT: PASS: all diagonal-family cross-checks agree with the closed form, 42/42 rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())