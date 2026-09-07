"""
data_generation.py
-------------------
Builds a single fixed training batch (X, Y) from a target function.

X is sampled uniformly from x_range in each input dimension.
Y = target_fn(X), optionally corrupted with Gaussian noise.
"""

from typing import Callable, Tuple
import torch


def generate_dataset(
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    n_samples: int = 200,
    input_dim: int = 1,
    x_range: Tuple[float, float] = (-3.0, 3.0),
    noise_std: float = 0.0,
    seed: int = 0,
    is_random: bool=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a single training batch (X, Y).

    Parameters
    ----------
    target_fn  : callable mapping (N, input_dim) -> (N, output_dim)
    n_samples  : number of points in the (single) training batch
    input_dim  : dimensionality of X
    x_range    : (low, high) — X is sampled uniformly from this interval
                 independently in every input dimension
    noise_std  : standard deviation of optional Gaussian label noise
    seed       : RNG seed, for reproducibility

    Returns
    -------
    X : Tensor (n_samples, input_dim)
    Y : Tensor (n_samples, output_dim)
    """
    low, high = x_range
    if is_random:
        g = torch.Generator().manual_seed(0)
        X = torch.rand(n_samples, input_dim, generator=g) * (high - low) + low

        with torch.no_grad():
            Y = target_fn(X)

        if noise_std > 0:
            Y = Y + noise_std * torch.randn(Y.shape, generator=g)
    else:
        X=torch.linspace(low,high, n_samples)
        with torch.no_grad():
            Y=target_fn(X)

    return X, Y

def train_val_split(X, Y, ratio=0.2, shuffle=True, random_seed=42):
    n_samples = X.shape[0]
    n_val = int(n_samples * ratio)
    if shuffle:
        torch.manual_seed(random_seed)
        indices = torch.randperm(n_samples)
        X, Y = X[indices], Y[indices]
    train_X, val_X = X[n_val:], X[:n_val]
    train_Y, val_Y = Y[n_val:], Y[:n_val]     # ← Y, not X
    return train_X, val_X, train_Y, val_Y
