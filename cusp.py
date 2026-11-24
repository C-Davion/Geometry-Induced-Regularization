'''
cusp.py

Look at how the output of a trained R -> R network, evaluated on the training
inputs, moves when only the feature-map parameters theta^f (the parameters of
of the feature transformation, i.e. every parameter whose name starts with "rho.") are perturbed.

    f(theta, X) in R^N        output of the network on the N training points
    P in R^(N x m)            Gaussian projection matrix, entries ~ N(0, sigma^2)
    d_1, ..., d_n             unit vectors, Gaussian then normalised, supported on theta^f only
                              (0 on every component of theta not in theta^f)

and, for each direction d_i and t in [t_min, t_max], the curve

    t  ->  P^T f(theta + t * d_i, X)   in R^m      (m = --proj-dim)

is plotted: left, its m coordinates as functions of t; right (m >= 2), the planar
curve of the first two coordinates, coloured by t, whose corners/cusps are the
interesting points. The t's at which the activation pattern of the hidden
layer (on X) changes are marked with thin grey vertical lines: a kink of the
curve can only happen there.

    python cusp.py BASE_DIR RUN_FOLDER DATASET_PATH --seed 0 --sigma 1.0 -n 4 --t-range -0.5 0.5

BASE_DIR/RUN_FOLDER/model_weights.pt is rebuilt with build_network, and X is the
training input of DATASET_PATH (load_dataset). Adjust the two imports below to
where those functions live in your project.
'''

import argparse
import copy
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from buildnetwork import build_network, DEFAULT_DATASET_PATH        
from helper import load_dataset            
from compute_ranks import _resolve_dataset_path

RHO_PREFIX = "rho."       # parameters of the feature map: theta^f


@contextmanager
def _default_dtype(dtype):
    """
    Temporarily change torch's default dtype: the feature-attention forward
    builds torch.ones(seq_len) in the default dtype, which would clash with
    float64 inputs. (Harmless if that forward already uses dtype=x.dtype.)
    """
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)

# ============================================================
# Data
# ============================================================

def load_X(dataset_path):
    """
    Training inputs X of shape (N, 1) from load_dataset(dataset_path). Accepts
    the usual return shapes: X directly, a dict with "X_train", or a tuple whose
    first element is one of those (e.g. (datas, hyperparams) or (X_train, Y_train, ...)).
    """
    (X_train,_,_,_),_ = load_dataset(_resolve_dataset_path(dataset_path))
    return X_train
    
# ============================================================
# The network as a function of theta^f
# ============================================================

def make_eval_fn(net, X, dtype=torch.float64, device="cpu"):
    """
    Wrap `net` as a function of theta^f only. Works on a deep copy (net is not modified).

    Returns (eval_fn, theta0, n_params):
        theta0     : numpy vector, the current values of theta^f (all "rho.*" parameters, flattened)
        eval_fn(theta_f) -> (f, pattern): sets theta^f = theta_f in the copy (everything else
                     unchanged), returns f = network output on X as a flat numpy vector (N * output_dim,)
                     and the 0/1 activation pattern of the hidden layer on X, shape (N, hidden_dim)
    """
    model = copy.deepcopy(net).to(device=device, dtype=dtype)
    Xc = X.to(device=device, dtype=dtype)
    rho = [(n, p) for n, p in model.named_parameters() if n.startswith(RHO_PREFIX)]
    if not rho:
        raise ValueError(f"No parameter named '{RHO_PREFIX}*' in the network.")

    theta0 = np.concatenate([p.detach().cpu().numpy().ravel() for _, p in rho]).astype(np.float64)

    def eval_fn(theta_f):
        i = 0
        with torch.no_grad():
            for _, p in rho:
                k = p.numel()
                p.copy_(torch.as_tensor(theta_f[i:i + k].reshape(tuple(p.shape)), dtype=dtype, device=device))
                i += k
            f = model(Xc).reshape(-1).cpu().numpy()
            pattern = (model.activation_pattern(Xc) > 0).cpu().numpy()
        return f, pattern

    return eval_fn, theta0, theta0.size

# ============================================================
# Sampling and curves
# ============================================================

def sample_projection_and_directions(n_out, n_params, proj_dim, n_dirs, sigma, seed):
    """
    For a given seed: P (n_out, proj_dim) with entries N(0, sigma^2), then n_dirs
    unit vectors in R^n_params (Gaussian, normalised), in that order.
    """
    rng = np.random.default_rng(seed)
    P = rng.normal(0.0, sigma, size=(n_out, proj_dim))
    dirs = rng.normal(size=(n_dirs, n_params))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return P, dirs


def curves_along_directions(eval_fn, theta0, dirs, P, ts):
    """
    For every direction d and t in ts: P^T f(theta + t d, X).

    Returns (coords, changes, proj0):
        coords  (n_dirs, len(ts), m)  the projected curves
        changes (n_dirs, len(ts))     True at the t's where the activation pattern differs from the previous t
        proj0   (m,)                  P^T f at t = 0 (the unperturbed network)
    """
    f0, _ = eval_fn(theta0)
    proj0 = P.T @ f0

    coords = np.empty((len(dirs), len(ts), P.shape[1]))
    changes = np.zeros((len(dirs), len(ts)), dtype=bool)
    for i, d in enumerate(dirs):
        prev = None
        for k, t in enumerate(ts):
            f, pattern = eval_fn(theta0 + t * d)
            coords[i, k] = P.T @ f
            if prev is not None:
                changes[i, k] = not np.array_equal(pattern, prev)
            prev = pattern
    return coords, changes, proj0

# ============================================================
# Plot
# ============================================================

def plot_cusps(ts, coords, changes, proj0, title=""):
    """One row per direction: coordinates vs t (left), planar curve coloured by t (right, if m >= 2)."""
    n_dirs, _, m = coords.shape
    n_cols = 2 if m >= 2 else 1
    fig, axes = plt.subplots(n_dirs, n_cols, figsize=(6 * n_cols, 3.2 * n_dirs), squeeze=False)

    for i in range(n_dirs):
        ax = axes[i, 0]
        for j in range(m):
            ax.plot(ts, coords[i, :, j], lw=1.2, label=f"proj {j + 1}")
        lo, hi = ax.get_ylim()
        if changes[i].any():
            ax.vlines(ts[changes[i]], lo, hi, color="gray", alpha=0.25, lw=0.6)
        ax.axvline(0.0, color="k", lw=0.6, ls=":")
        ax.set_ylim(lo, hi)
        ax.set_xlabel("t")
        ax.set_ylabel(r"$P^\top f(\theta+t\,d,X)$")
        ax.set_title(f"direction {i}: {int(changes[i].sum())} activation-pattern change(s)")
        if m > 1:
            ax.legend(fontsize=8)

        if n_cols == 2:
            ax2 = axes[i, 1]
            c = coords[i]
            ax2.plot(c[:, 0], c[:, 1], color="lightgray", lw=0.8, zorder=1)
            sc = ax2.scatter(c[:, 0], c[:, 1], c=ts, s=6, cmap="coolwarm", zorder=2)
            ax2.plot(proj0[0], proj0[1], marker="*", color="k", ms=1, ls="none", zorder=3, label="t = 0")
            ax2.set_aspect("equal", adjustable="datalim")
            ax2.set_xlabel("proj 1")
            ax2.set_ylabel("proj 2")
            ax2.set_title("planar curve (coloured by t)")
            ax2.legend(fontsize=8)
            fig.colorbar(sc, ax=ax2, label="t")

    fig.suptitle(title)
    fig.tight_layout()
    return fig

# ============================================================
# Command line
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Project the network output when only theta^f (rho) is perturbed.")
    p.add_argument("base_dir")
    p.add_argument("run_folder", help="folder inside base_dir containing model_weights.pt")
    p.add_argument("--dataset_path",type=str, default=DEFAULT_DATASET_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=1.0, help="std of the Gaussian projection matrix")
    p.add_argument("-n", "--n-dirs", type=int, default=4, help="number of random directions")
    p.add_argument("--proj-dim", type=int, default=2, help="columns of the projection matrix (2 gives the planar curve)")
    p.add_argument("--t-range", type=float, nargs=2, default=(-0.5, 0.5), metavar=("LOW", "HIGH"))
    p.add_argument("--n-t", type=int, default=801, help="number of t values")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--save", default=None, help="save the figure to this path")
    p.add_argument("--no-show", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    weights_path = Path(args.base_dir) / args.run_folder / "model_weights.pt"
    net = build_network(weights_path)
    X = load_X(args.dataset_path)

    with _default_dtype(dtype):
        eval_fn, theta0, n_params = make_eval_fn(net, X, dtype=dtype, device=args.device)
        n_out = eval_fn(theta0)[0].size

        P, dirs = sample_projection_and_directions(
            n_out, n_params, args.proj_dim, args.n_dirs, args.sigma, args.seed)
        ts = np.linspace(args.t_range[0], args.t_range[1], args.n_t)

        print(f"run: {weights_path}")
        print(f"N = {len(X)} training points (f lives in R^{n_out}),  |theta^f| = {n_params} parameters, "
              f"||theta^f|| = {np.linalg.norm(theta0):.3f}")
        print(f"seed={args.seed}  sigma={args.sigma}  directions={args.n_dirs}  proj_dim={args.proj_dim}  "
              f"t in [{ts[0]}, {ts[-1]}] ({args.n_t} points)")

        coords, changes, proj0 = curves_along_directions(eval_fn, theta0, dirs, P, ts)

    for i in range(args.n_dirs):
        print(f"  direction {i}: {int(changes[i].sum())} activation-pattern change(s) along t")

    fig = plot_cusps(ts, coords, changes, proj0,
                     title=f"{args.run_folder}   seed={args.seed}  sigma={args.sigma}")
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved to {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()