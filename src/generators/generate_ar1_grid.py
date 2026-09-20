"""Compound-symmetry AR(1) generator for the factorial (C, rho) grid.

This is the generator that produced the AR(1) grid series for seeds
{42, 123, 456} in results/results_grid.csv. Seeds {789, 1011} were generated
by the same process with 14,400 usable timesteps (burn-in simulated on top of
the returned length rather than inside it); the paper's protocol table records
the two usable lengths and the resulting training-window counts.

Two properties are controlled independently:
    C:   number of variates
    rho: uniform off-diagonal contemporaneous correlation

No cross-channel lagged coupling is introduced. All channels share the AR
coefficient phi and are correlated only through contemporaneous innovation, so
Granger non-causality holds exactly across every channel pair at every lag.
Stationarity requires |phi| < 1.

The process is a vector AR(1) with a diagonal transition matrix:
    x_t = phi * x_{t-1} + epsilon_t,    epsilon_t ~ N(0, Sigma)

Sigma is a uniform-correlation (compound-symmetry) matrix:
    Sigma_ij = rho for i != j, else 1

which is positive definite for rho in the open interval (-1/(C-1), 1) and
singular at the boundary values.

Because the transition is phi * I, the stationary covariance satisfies
P = phi^2 * P + Sigma, hence P = Sigma / (1 - phi^2). The initial state x_0 is
drawn from N(0, I) rather than the stationary distribution; the burn-in reduces
the effect of that initialisation geometrically at rate phi^t.

Split protocol matches the ETTh1 fractions (60 / 20 / 20 of usable timesteps):
    usable = total_len - _BURN_IN          (default: 14400 - 1000 = 13400)
    train  = usable * 8640 // 14400        (default: 8040)
    val    = usable * 2880 // 14400        (default: 2680)
    test   = usable * 2880 // 14400        (default: 2680)

Normalisation: per-channel z-score with mean and standard deviation fit on the
training split and applied to all three splits. The original module used
scikit-learn's StandardScaler for this step; it is replaced here by the
equivalent NumPy computation (population standard deviation, zero-variance
channels left unscaled) so the module needs no dependency outside the project.

Usage:
    from generate_ar1_grid import SyntheticARDataset

    ds = SyntheticARDataset(C=7, rho=0.5, phi=0.8, seq_len=512, pred_len=96, split="train", seed=42)
    x, y = ds[0]  # x: (512, 7), y: (96, 7)

The array-level functions build_covariance, generate_ar1 and split_normalise
need only NumPy; the Dataset class needs torch.
"""

from typing import Literal

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object

_BURN_IN: int = 1000

# ETTh1 split row counts used as fraction numerators over 14400 total rows,
# applied proportionally to the usable series length after burn-in.
_TRAIN_NUM: int = 8640
_VAL_NUM: int = 2880
_DENOM: int = 14400

type Split = Literal["train", "val", "test"]


def build_covariance(C: int, rho: float) -> np.ndarray:
    """Construct a uniform off-diagonal (compound-symmetry) correlation matrix.

    The matrix has ones on the diagonal and rho elsewhere:
        Sigma_ij = rho for i != j, else 1
    Equivalent to rho * ones(C, C) + (1 - rho) * eye(C).

    Args:
        C: number of variates, must be >= 2.
        rho: off-diagonal correlation coefficient.

    Returns:
        Float64 array of shape (C, C).

    Raises:
        ValueError: if C < 2 or rho is outside the positive-definite range.
    """
    if C < 2:
        raise ValueError(f"C must be >= 2, got {C}")
    lower = -1.0 / (C - 1)
    if not (lower < rho < 1.0):
        raise ValueError(
            f"rho={rho} is outside the valid range ({lower:.6f}, 1.0) for C={C}. "
            "The covariance matrix would not be positive definite."
        )
    diag = (1.0 - rho) * np.eye(C, dtype=np.float64)
    off_diag = rho * np.ones((C, C), dtype=np.float64)
    return off_diag + diag


def generate_ar1(T: int, C: int, phi: float, cov: np.ndarray, seed: int) -> np.ndarray:
    """Generate a multivariate AR(1) series and discard the burn-in steps.

    Recurrence: x_t = phi * x_{t-1} + epsilon_t with epsilon_t ~ N(0, cov) and
    x_0 ~ N(0, I). All innovations are drawn in one vectorised call; the
    recurrence loop is the only sequential step.

    Args:
        T: total timesteps to simulate before discarding burn-in, must be > _BURN_IN.
        C: number of variates.
        phi: scalar AR(1) coefficient, |phi| < 1 for stationarity.
        cov: positive-definite covariance matrix of shape (C, C).
        seed: integer seed for a local NumPy generator; global RNG state is untouched.

    Returns:
        Float64 array of shape (T - _BURN_IN, C).

    Raises:
        ValueError: if T <= _BURN_IN.
    """
    if T <= _BURN_IN:
        raise ValueError(f"T={T} must be > _BURN_IN={_BURN_IN}")

    rng = np.random.default_rng(seed)

    x = np.empty((T, C), dtype=np.float64)
    x[0] = rng.standard_normal(C)
    noise = rng.multivariate_normal(np.zeros(C), cov, size=T - 1)  # (T-1, C)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + noise[t - 1]
    return x[_BURN_IN:]


def split_bounds(usable: int) -> tuple[int, int]:
    """Return (train_end, val_end) row indices for a usable series length."""
    train_end = usable * _TRAIN_NUM // _DENOM
    val_end = train_end + usable * _VAL_NUM // _DENOM
    return train_end, val_end


def split_normalise(series: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split chronologically 60/20/20 and z-score every split with training statistics.

    Standard deviation is the population value (ddof=0); channels with zero
    training variance are left unscaled.

    Args:
        series: float array of shape (usable, C).

    Returns:
        (train, val, test) float32 arrays.
    """
    train_end, val_end = split_bounds(len(series))
    train = series[:train_end]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    normalised = ((series - mean) / std).astype(np.float32)
    return normalised[:train_end], normalised[train_end:val_end], normalised[val_end:]


class SyntheticARDataset(Dataset):
    """Sliding-window dataset over a synthetic multivariate AR(1) series.

    Generates the full series at construction time, applies the chronological
    60/20/20 split, normalises with training-split statistics only, and exposes
    (input, target) windows through __getitem__.

    Args:
        C: number of variates; the grid uses {7, 21, 84}.
        rho: off-diagonal correlation; the grid uses {0.1, 0.5, 0.9}.
        phi: AR(1) coefficient, fixed at 0.8 in the grid.
        seq_len: input timesteps per sample.
        pred_len: target timesteps immediately following the input.
        split: one of "train", "val" or "test".
        seed: integer seed for the series.
        total_len: timesteps to simulate before the burn-in discard. Default
            14,400 gives 13,400 usable timesteps, split 8040 / 2680 / 2680.
    """

    def __init__(
        self,
        C: int,
        rho: float,
        phi: float,
        seq_len: int,
        pred_len: int,
        split: Split,
        seed: int,
        total_len: int = 14400,
    ) -> None:
        if torch is None:
            raise ImportError("SyntheticARDataset requires torch")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be one of 'train', 'val' or 'test', got {split}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if pred_len < 1:
            raise ValueError(f"pred_len must be >= 1, got {pred_len}")

        cov = build_covariance(C, rho)
        series = generate_ar1(total_len, C, phi, cov, seed)  # (usable, C)
        train, val, test = split_normalise(series)

        match split:
            case "train":
                self._data = train
            case "val":
                self._data = val
            case "test":
                self._data = test

        window = seq_len + pred_len
        if len(self._data) < window:
            raise ValueError(
                f"Split {split} has {len(self._data)} rows but seq_len + pred_len = {window}. "
                "Reduce seq_len or pred_len, or increase total_len."
            )

        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self._data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        x = self._data[idx : idx + self.seq_len]  # (seq_len, C)
        y = self._data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)
