"""
data_generation.py
-------------------
Builds a single fixed training batch (X, Y) from a target function.

X is sampled uniformly from x_range in each input dimension.
Y = target_fn(X), optionally corrupted with Gaussian noise.
"""

from typing import Callable, Tuple
import torch
import os 

from target_fn_attention import make_target_fn_attention, iter_target_functions_attention 
INPUT_FILE = "intresting_function_attention.jsonl"
OUTPUT_DIR = "generated_datasets"
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
DATASET_DIR = os.path.join(OUTPUT_DIR, "datasets")

NUMBER_OF_DATA_POINT = 400
import matplotlib.pyplot as plt


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
        g = torch.Generator()
        g.manual_seed(random_seed)          # fixed seed, kept — split stays reproducible
        indices = torch.randperm(n_samples, generator=g)   # local generator, no global side effect
        X, Y = X[indices], Y[indices]
    train_X, val_X = X[n_val:], X[:n_val]
    train_Y, val_Y = Y[n_val:], Y[:n_val]
    return train_X, val_X, train_Y, val_Y


def plot_data_and_save_data(target_fn,n_samples,input_dim,x_range,noise_std,seed): 
    fig=plt.figure(figsize=(7,5))
    X, Y = generate_dataset(
        target_fn, n_samples=n_samples, input_dim=input_dim,
        x_range=x_range, noise_std=noise_std, seed=seed, is_random=True
    )
    X_train,X_val,Y_train,Y_val=train_val_split(X,Y)
    print(X.sum().item())
    x_plot = torch.linspace(x_range[0], x_range[1], 300).unsqueeze(1)
    y_plot=y_true    = target_fn(x_plot)
    plt.plot(x_plot.squeeze(), y_true.squeeze(),
                  label="Target function", color="grey", linewidth=1)
    plt.scatter(X_train.squeeze(), Y_train.squeeze(), s=10, color="blue", alpha=0.8, label="Training data")
    plt.scatter(X_val.squeeze(), Y_val.squeeze(), s=10, color="red", alpha=0.8, label="Validation  data")
    return fig
if __name__=="__main__":
    for  score, target_params,seq_len,embed_dim in iter_target_functions_attention(INPUT_FILE):
        target_fn = make_target_fn_attention(**target_params)
        fig=plot_data_and_save_data(target_fn,  input_dim=1,n_samples=200,x_range=(-1.0, 1.0),noise_std=0.0,seed=None)
        plt.legend()
        plt.show()
