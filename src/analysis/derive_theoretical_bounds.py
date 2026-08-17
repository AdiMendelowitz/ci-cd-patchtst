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
rest of the table.

Followers are reported three ways across the full horizon sweep:

  1. CD bound: the unrestricted ceiling [Sigma_h]_ii / stationary variance.
  2. CI bound: the own-history-only ceiling, computed exactly (below).
  3. Sandwich width: 1.0 - CD bound, retained from the earlier revision of
     this script for continuity with material already produced from it.

The sandwich width upper-bounds the CI-CD gap but does not estimate it. Its
upper limit of 1.0 is the no-signal ceiling, not the CI-restricted ceiling,
and at short horizons that limit is loose because own-history autoregression
alone predicts well. The decisive demonstration is gamma=0, where no coupling
exists by construction: the true CI-CD gap is identically zero at every
horizon, yet the sandwich width at h=1 is 0.64. Width is therefore reported
as a diagnostic only, and the CI-CD gap is read from bounds 1 and 2.

CI-restricted bound for followers
---------------------------------
A follower observes only its own history, but its own past partially reveals
its leader's past, so the own-history-optimal predictor is not the marginal
AR(1) predictor. Treating the joint process as a linear-Gaussian state-space
model with the follower as the sole observed coordinate, the steady-state
Kalman predictor is optimal among all own-history predictors, and its h-step
error covariance follows by propagating the filtered covariance through the
dynamics. The observation is noiseless; the discrete algebraic Riccati solver
requires a positive-definite observation-noise term, so a negligible value is
supplied and the result is verified insensitive to it over four orders of
magnitude (KALMAN_OBS_NOISE, checked in verify_kalman_conditioning).

Two exact checks anchor this computation. At gamma=0 the follower is a pure
AR(1) channel, so its CI bound must equal 1 - phi**(2h) at every horizon.
At every gamma and horizon the ordering CD <= CI <= 1 must hold, since
conditioning on strictly more information cannot increase the
population-optimal forecast error. Both are asserted, not assumed.

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
results/Revision/theoretical_bounds.csv, resolved against the repository
root so the script produces the same file from any working directory.

Row count: 62 (9 AR(1) grid + 4 block-covariance + 1 boundary sweep + 8
leader-follower leader/isolate rows [2 x 4 gamma] + 20 follower CD rows
[5 horizons x 4 gamma] + 20 follower CI rows [5 horizons x 4 gamma]).

An existing output file is not overwritten. Because the computation is
deterministic and reads no data, a rerun over an existing file is a
reproducibility check: the freshly computed table is compared against the
file on disk row by row, and the script reports agreement and exits 0, or
reports the first disagreements and exits 1. Regenerating after an
intentional change requires removing the file explicitly.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src" / "generators"))

import generate_block_cov as bc  # noqa: E402
import generate_leader_follower as lf  # noqa: E402

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

# Stand-in for a noiseless observation in the Riccati solve. The solver requires
# a positive-definite observation-noise term; this value is small enough that the
# filtered covariance is indistinguishable from the noiseless limit at float64
# precision. verify_kalman_conditioning checks that conclusion rather than
# assuming it.
KALMAN_OBS_NOISE = 1e-10
KALMAN_OBS_NOISE_PROBES = (1e-8, 1e-10, 1e-12)

EXPECTED_ROWS = 62
OUT_PATH = _ROOT / "results" / "Revision" / "theoretical_bounds.csv"

SCHEMA = ["family", "cell", "h", "bound_type", "bound", "sandwich_width"]
MERGE_KEYS = ["family", "cell", "h", "bound_type"]

BOUND_TYPE_EXACT = "CI=CD (exact)"
BOUND_TYPE_CD = "CD (sandwich upper=1.0)"
BOUND_TYPE_CI = "CI (own-history optimal)"


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


def own_history_bound(A: np.ndarray, Sigma: np.ndarray, idx: int, h: int,
                      obs_noise: float = KALMAN_OBS_NOISE) -> float:
    """Optimal h-step forecast MSE for channel idx given only its own history,
    in z-scored units.

    The joint process is a linear-Gaussian state-space model observed through
    the single coordinate idx. The steady-state Kalman predictor is optimal
    over all measurable functions of that channel's own past, so its error
    covariance is the CI-restricted ceiling. The filtered covariance is
    propagated h steps through the dynamics, accumulating innovation
    covariance at each step.
    """
    C = A.shape[0]
    obs = np.zeros((1, C))
    obs[0, idx] = 1.0
    noise = np.array([[obs_noise]])

    predicted = solve_discrete_are(A.T, obs.T, Sigma, noise)
    gain = predicted @ obs.T @ np.linalg.inv(obs @ predicted @ obs.T + noise)
    filtered = predicted - gain @ obs @ predicted

    error = filtered
    for _ in range(h):
        error = A @ error @ A.T + Sigma
    return float(error[idx, idx] / stationary_var_diag(A, Sigma)[idx])


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
    """Cross-check the general matrix computation against the closed form for
    a diagonal-A family. Raises if they disagree.

    Returns the closed-form value, which the two agree on within TOL. The
    analytic expression is the one recorded, so the table carries no
    accumulated rounding from the iterated matrix product.
    """
    got = normalized_bound(A, Sigma, H)
    want = diagonal_ceiling(PHI, H)
    if not np.allclose(got, want, atol=TOL):
        raise AssertionError(
            f"{name}: matrix computation {got} disagrees with closed form "
            f"{want} beyond tolerance {TOL}"
        )
    return want


def verify_kalman_conditioning() -> None:
    """Confirm the own-history bound does not depend on the observation-noise
    stand-in, over four orders of magnitude, at the most strongly coupled cell.

    A dependence here would mean KALMAN_OBS_NOISE acts as a real measurement
    noise rather than as a regulariser for the noiseless limit.
    """
    A = lf.make_transition_matrix(PHI, max(LF_GAMMA))
    Sigma = lf.make_innovation_covariance(LF_RHO)
    values = [own_history_bound(A, Sigma, lf.N_LEADERS, 5, obs_noise=probe)
              for probe in KALMAN_OBS_NOISE_PROBES]
    if not np.allclose(values, values[0], atol=TOL):
        raise AssertionError(
            f"own-history bound varies with the observation-noise stand-in "
            f"{KALMAN_OBS_NOISE_PROBES}: {values}"
        )


def verify_follower_ordering(gamma: float, h: int, cd: float, ci: float) -> None:
    """Assert CD <= CI <= 1 for a follower cell.

    Conditioning on strictly more information cannot increase the
    population-optimal forecast error, so the unrestricted ceiling cannot
    exceed the own-history ceiling, and neither can exceed the no-signal
    value of 1. A violation beyond TOL indicates an error in the derivation
    rather than a property of the process.
    """
    if not (cd <= ci + TOL and ci <= 1.0 + TOL):
        raise AssertionError(
            f"gamma={gamma}, h={h}: ordering CD <= CI <= 1 violated "
            f"(CD={cd!r}, CI={ci!r})"
        )


def verify_zero_coupling(ci_bounds: dict[int, float]) -> None:
    """At gamma=0 the follower is a pure AR(1) channel, so its own-history
    ceiling must equal the closed form at every horizon."""
    for h, value in ci_bounds.items():
        want = diagonal_ceiling(PHI, h)
        if not np.isclose(value, want, atol=TOL):
            raise AssertionError(
                f"gamma=0, h={h}: own-history bound {value} != closed form {want}"
            )


def compare_with_existing(rows: list[dict], path: Path) -> int:
    """Compare a freshly computed table against the file already on disk.

    The computation is deterministic and reads no data, so a rerun over an
    existing file is a reproducibility check rather than a regeneration.
    Returns a process exit code, printing any disagreement.
    """
    fresh = pd.DataFrame(rows)[SCHEMA]
    stored = pd.read_csv(path)[SCHEMA]

    if len(fresh) != len(stored):
        print(f"\nFAIL: {path} holds {len(stored)} rows, computation produced {len(fresh)}")
        return 1

    merged = fresh.merge(stored, on=MERGE_KEYS, how="outer",
                         suffixes=("_new", "_old"), indicator=True)
    unmatched = merged[merged["_merge"] != "both"]
    if not unmatched.empty:
        print(f"\nFAIL: key sets differ between computation and {path}:")
        print(unmatched[MERGE_KEYS + ["_merge"]].to_string(index=False))
        return 1

    for column in ("bound", "sandwich_width"):
        new, old = merged[f"{column}_new"], merged[f"{column}_old"]
        differing = merged[~np.isclose(new, old, atol=TOL, equal_nan=True)]
        if not differing.empty:
            print(f"\nFAIL: {column} differs from {path} in {len(differing)} row(s):")
            columns = MERGE_KEYS + [f"{column}_old", f"{column}_new"]
            print(differing[columns].head(10).to_string(index=False))
            return 1

    print(f"\nRESULT: PASS: recomputation reproduces {path} exactly "
          f"({len(fresh)} rows, tolerance {TOL})")
    return 0


def build_rows() -> list[dict]:
    """Derive every table row, printing the human-readable report as it goes."""
    rows: list[dict] = []

    print(f"phi={PHI}  h={H}  phi**(2h)={PHI ** (2 * H):.3e}\n")

    print("=== AR(1) grid ===")
    for C in AR_GRID_C:
        for rho in AR_GRID_RHO:
            A = PHI * np.eye(C)
            Sigma = compound_symmetry(rho, C)
            bound = verify_diagonal_family(f"AR(1) grid C={C} rho={rho}", A, Sigma)
            print(f"  C={C:>2}  rho={rho}: CI=CD bound = {bound:.10f}")
            rows.append({"family": "AR(1) grid", "cell": f"C={C}, rho={rho}", "h": H,
                         "bound_type": BOUND_TYPE_EXACT, "bound": bound,
                         "sandwich_width": np.nan})

    print("\n=== Block-covariance ===")
    for C in BLOCK_COV_C:
        for rho_in in BLOCK_COV_RHO_IN:
            A = bc.make_transition_matrix(PHI, C)
            Sigma = bc.make_block_covariance(C, BLOCK_COV_GROUP_SIZE, rho_in, BLOCK_COV_RHO_OUT)
            bound = verify_diagonal_family(f"block-cov C={C} rho_in={rho_in}", A, Sigma)
            print(f"  C={C:>2}  rho_in={rho_in}: CI=CD bound = {bound:.10f}")
            rows.append({"family": "block-covariance", "cell": f"C={C}, rho_in={rho_in}", "h": H,
                         "bound_type": BOUND_TYPE_EXACT, "bound": bound,
                         "sandwich_width": np.nan})

    print("\n=== Boundary sweep (all P, all gamma tranches) ===")
    print("  gamma/rho columns are inert metadata for this family.")
    bound = diagonal_ceiling(PHI, H)
    print(f"  CI=CD bound = {bound:.10f}  (independent of P, gamma, seed)")
    rows.append({"family": "boundary sweep", "cell": "all P, all gamma", "h": H,
                 "bound_type": BOUND_TYPE_EXACT, "bound": bound, "sandwich_width": np.nan})

    print("\n=== Leader-follower: leader / isolate (diagonal ceiling, h=96) ===")
    for gamma in LF_GAMMA:
        A = lf.make_transition_matrix(PHI, gamma)
        Sigma = lf.make_innovation_covariance(LF_RHO)
        bound_vector = normalized_bound(A, Sigma, H)

        for label, value in (("leader", bound_vector[0]),
                             ("isolate", bound_vector[LF_ISOLATE_IDX])):
            if not np.isclose(value, diagonal_ceiling(PHI, H), atol=TOL):
                raise AssertionError(f"gamma={gamma}: {label} bound {value} != diagonal ceiling")
            print(f"  gamma={gamma}  {label}: CI=CD bound = {value:.10f}")
            rows.append({"family": "leader-follower", "cell": f"gamma={gamma}, {label}", "h": H,
                         "bound_type": BOUND_TYPE_EXACT, "bound": float(value),
                         "sandwich_width": np.nan})

        follower_block = bound_vector[lf.N_LEADERS:lf.N_LEADERS + lf.N_FOLLOWERS]
        if not np.allclose(follower_block, follower_block[0], atol=TOL):
            raise AssertionError(
                f"gamma={gamma}: followers disagree with each other {follower_block}"
            )

    verify_kalman_conditioning()

    print("\n=== Leader-follower: follower ceilings, full horizon sweep ===")
    print("  CD is the unrestricted ceiling, CI the own-history-only ceiling;")
    print("  gap = CI - CD is the cross-channel opportunity available to CD.")
    print("  width = 1 - CD is the earlier sandwich diagnostic, retained for continuity;")
    print("  it upper-bounds the gap and is loose wherever the gap is small.")
    for gamma in LF_GAMMA:
        A = lf.make_transition_matrix(PHI, gamma)
        Sigma = lf.make_innovation_covariance(LF_RHO)
        stat_var = stationary_var_diag(A, Sigma)
        ci_by_h: dict[int, float] = {}

        for h in SWEEP_H:
            cd_bound = float(sigma_h_diag(A, Sigma, h)[lf.N_LEADERS] / stat_var[lf.N_LEADERS])
            ci_bound = own_history_bound(A, Sigma, lf.N_LEADERS, h)
            verify_follower_ordering(gamma, h, cd_bound, ci_bound)
            ci_by_h[h] = ci_bound

            width = sandwich_width(cd_bound)
            gap = ci_bound - cd_bound
            relative = 100.0 * gap / ci_bound if ci_bound > 0 else 0.0
            print(f"  gamma={gamma}  h={h:>3}: CD = {cd_bound:.10f}  CI = {ci_bound:.10f}  "
                  f"gap = {gap:+.6f} ({relative:5.1f}% of CI)  width = {width:.3e}")

            rows.append({"family": "leader-follower", "cell": f"gamma={gamma}, follower", "h": h,
                         "bound_type": BOUND_TYPE_CD, "bound": cd_bound,
                         "sandwich_width": width})
            rows.append({"family": "leader-follower", "cell": f"gamma={gamma}, follower", "h": h,
                         "bound_type": BOUND_TYPE_CI, "bound": ci_bound,
                         "sandwich_width": np.nan})

        if gamma == 0.0:
            verify_zero_coupling(ci_by_h)

    return rows


def main() -> int:
    rows = build_rows()

    if len(rows) != EXPECTED_ROWS:
        raise AssertionError(
            f"expected {EXPECTED_ROWS} rows, got {len(rows)}: row-count contract broken"
        )

    print(f"\nSummary: at h={H}, phi={PHI}, the theoretical CI=CD ceiling is indistinguishable")
    print(f"from 1.0 to float64 precision (phi**(2h)={PHI ** (2 * H):.3e}) for every")
    print("diagonal-transition family and cell, and the follower CI and CD ceilings coincide")
    print("to the same precision. No model, CI or CD, has room to exploit real predictive")
    print("signal at this horizon. The cross-channel opportunity is non-zero only at short")
    print("horizons and has vanished by h=50.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUT_PATH.exists():
        return compare_with_existing(rows, OUT_PATH)

    pd.DataFrame(rows)[SCHEMA].to_csv(OUT_PATH, index=False)
    print(f"\nwrote {len(rows)} rows to {OUT_PATH}")
    print(f"\nRESULT: PASS: all cross-checks agree with the closed form, "
          f"{len(rows)}/{EXPECTED_ROWS} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())