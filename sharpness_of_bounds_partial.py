'''
partial_rank_ratio_attention.py

Same pipeline as rank_ratio_attention.py (runs from several directories,
grad-norm filter, everything recomputed in float64 on X_train from
model_weights.pt), but for the PARTIAL Jacobians of theta -> f(theta, X_train):

    J = [ J_rest | J_rho ]

    J_rho  : derivative w.r.t. rho.parameters() only          (n_f columns)
    J_rest : derivative w.r.t. every parameter BUT rho's      (W1, b1, W2, b2)

One histogram per partial Jacobian. By default each rank is normalised
between a lower and an upper bound, in the spirit of
t = (jac_rank - lower) / (upper - lower):

    upper = upper_rest + n_f,   upper_rest = output_dim * (seq_len*embed_dim + 1) * pattern_count

    J_rest :  lower <= rank(J_rest) <= upper_rest
              (lower bound: the b1-columns of the Jacobian are D @ diag(W2),
               whose rank is lower)
              t_rest = (rank(J_rest) - lower) / (upper_rest - lower)
    J_rho  :  0  <= rank(J_rho)  <= min(n_f,n)      (number of columns,number of rows)
              t_rho  = rank(J_rho) / n_f



Note on the upper bound: upper_rest (like upper) has no term for the b2 column,
which can add output_dim to the rank in tight cases, so with
upper_mode="p_rank" t_rest can exceed 1 by a hair when the bound is tight.
upper_mode="n_distinct" (|A(theta, X)|, as in the training plots) does not
have that issue. Out-of-range values are reported, not hidden.
'''
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from torch.func import functional_call, jacrev

from expstats_attention import (
    BASE_DIR,
    GRAD_THRESHOLD,
    DEVICE,
    build_network,
    load_run_config,
)
from sharpness_of_bounds import (
    gather_runs_across_dirs,
    load_training_inputs,
    compute_upper,
    compute_lower,
    _default_dtype,
    _METHOD_SPECS
)
from helper import pattern_stats

# The two experiment directories (siblings of each other by default).
BASE_DIRS = [
    #BASE_DIR,
    BASE_DIR.parent / "reg_experiments_attention_init",
]

# Applied inside every directory in BASE_DIRS.
FOLDER_PATTERN = "run_47.77_*"

# ============================================================
# Jacobian blocks and their ranks
# ============================================================

RHO_PREFIX = "rho."      # named_parameters() of OneHiddenLayerReLUNet: "rho.E", "rho.B", "rho.attention.*"


def _jacobian_blocks(model, X):
    """
    Stacked Jacobian of theta -> f(theta, X), one block per parameter.

    Same computation as helper.stacked_jacobian_rank, but the blocks are
    kept apart instead of being concatenated and reduced to a rank.
    Returns {param_name: np.ndarray of shape (N * output_dim, numel)}.
    (Dtype is whatever model / X are: call inside _default_dtype(float64)
    with a float64 model and X.)
    """
    params = dict(model.named_parameters())

    def f_batch(p: dict) -> torch.Tensor:
        return functional_call(model, p, X).flatten()      # (N * output_dim,)

    jac_dict = jacrev(f_batch)(params)
    return {
        name: j.flatten(start_dim=1).detach().cpu().numpy()
        for name, j in jac_dict.items()
    }


def _rank_of(mats, rtol=None, stable=False):
    """
    Rank of the horizontal concatenation of the given 2D arrays (0 if none).
      rtol=None   : default tolerance max(m, n) * eps(dtype) * sigma_max
                    (same as torch.linalg.matrix_rank's default)
      rtol=r      : singular values <= r * sigma_max count as zero
      stable=True : stable rank ||M||_F^2 / ||M||_2^2 (a float) instead
    """
    if not mats:
        return 0.0 if stable else 0
    M = np.concatenate(mats, axis=1)
    s = np.linalg.svd(M, compute_uv=False)
    if s.size == 0 or s[0] == 0:
        return 0.0 if stable else 0
    if stable:
        return float((s ** 2).sum() / s[0] ** 2)
    if rtol is None:
        rtol = max(M.shape) * np.finfo(s.dtype).eps
    return int((s > rtol * s[0]).sum())


def _ranks_from_blocks(blocks, rho_prefix=RHO_PREFIX, rtol=None, stable=False):
    """
    (rank(J_rho), rank(J_rest), rank(J_full)) from per-parameter blocks:
    J_rho = blocks of parameters named "rho.*", J_rest = all the others.
    """
    rho = [b for name, b in blocks.items() if name.startswith(rho_prefix)]
    rest = [b for name, b in blocks.items() if not name.startswith(rho_prefix)]
    return (_rank_of(rho, rtol, stable),
            _rank_of(rest, rtol, stable),
            _rank_of(rho + rest, rtol, stable))


def compute_partial_ranks(model, X, upper_mode, lower_mode, rank_method="float64",
                          device=DEVICE, w2_tol=0.0):
    """
    Same as before, plus `method` in {"float32", "float32_rtol", "float64",
    "stable_rank"} (see _METHOD_SPECS). Works on a deep copy: `model` is
    never modified. Ranks are floats for "stable_rank".
    Returns a dict with keys rank_rho, rank_rest, rank_full, p_rank,
    n_distinct, lower.
    """
    if rank_method not in _METHOD_SPECS:
        raise ValueError(f"method must be one of {list(_METHOD_SPECS)}, got {rank_method!r}")
    dtype, rtol, stable = _METHOD_SPECS[rank_method]

    with _default_dtype(dtype):
        net = copy.deepcopy(model).to(device=device, dtype=dtype)
        Xc = X.to(device=device, dtype=dtype)

        blocks = _jacobian_blocks(net, Xc)
        p_rank, n_distinct = pattern_stats(net, Xc, rtol=rtol, stable=stable)
        lower = compute_lower(net, Xc, lower_mode, device=device, w2_tol=w2_tol)

    rank_rho, rank_rest, rank_full = _ranks_from_blocks(blocks, rtol=rtol, stable=stable)
    return {
        "rank_rho": rank_rho,
        "rank_rest": rank_rest,
        "rank_full": rank_full,
        "p_rank": p_rank,
        "n_distinct": n_distinct,
        "lower": lower,
    }
# ============================================================
# Per-run table
# ============================================================

def compute_partial_rank_table(base_dirs=BASE_DIRS, folder_pattern=FOLDER_PATTERN,
                                threshold=GRAD_THRESHOLD, device=DEVICE,
                                w2_tol=0.0, upper_mode="p_rank", lower_mode="p_rank", rank_method="float64", verbose=True):
    """
    One row per run (across every directory in base_dirs) passing the
    grad-norm threshold, with the partial-Jacobian ranks and their
    normalised versions t_rho and t_rest (see the module docstring).

    upper_mode: "p_rank" (default) or "n_distinct" — which pattern count goes
    into upper_rest (p_rank <= n_distinct).
    """

    runs_df = gather_runs_across_dirs(
        base_dirs=base_dirs, folder_pattern=folder_pattern,
        threshold=threshold, verbose=verbose,
    )

    if len(runs_df) == 0:
        print("No runs passed the threshold.")
        return runs_df
    total=0
    rows = []
    for _, r in runs_df.iterrows():
        if total %50==0:
            print(total)
        run_dir = Path(r["base_dir"]) / r["run"]

        config = load_run_config(run_dir)
        output_dim = config["output_dim"]
        seq_len = config["seq_len"]
        embed_dim = config["embed_dim"]

        model = build_network(run_dir / "model_weights.pt", device=device)
        X_train = load_training_inputs(config)

        res = compute_partial_ranks(model, X_train,upper_mode,lower_mode, rank_method=rank_method, device=device,   w2_tol=w2_tol)

        pattern_count = res["p_rank"] if upper_mode == "p_rank" else res["n_distinct"]
        n_f = sum(p.numel() for p in model.rho.parameters())
        upper_weight = compute_upper(output_dim, seq_len, embed_dim, pattern_count, model.rho, upper_mode,X_train, keep_feature=False)

        lower = res["lower"]
        t_rest = (res["rank_rest"] - lower) / (upper_weight - lower) if upper_weight != lower else np.nan
        t_rho = res["rank_rho"] / min(n_f,X_train.shape[0]) if n_f > 0 else np.nan

        rows.append({
            "run": r["run"],
            "base_dir": r["base_dir"],
            "n_train": int(X_train.shape[0]),
            "output_dim": output_dim,
            "rank_rho": res["rank_rho"],
            "rank_rest": res["rank_rest"],
            "rank_full": res["rank_full"],
            "p_rank": res["p_rank"],
            "n_patterns": res["n_distinct"],
            "lower": lower,
            "upper_weight":upper_weight,
            "n_f": n_f,
            "t_rest": t_rest,
            "t_rho": t_rho,
        })
        total+=1

    table = pd.DataFrame(rows)

    if len(table) == 0:
        return table

    # --- sanity checks (reported, never hidden) ----------------------
    sub = table[table["rank_full"] > table["rank_rho"] + table["rank_rest"]]
    sup = table[table["rank_full"] < table[["rank_rho", "rank_rest"]].max(axis=1)]
    low = table[table["rank_rest"] < table["lower"]]
    print("\nSanity checks on the recomputed ranks:")
    print(f"  rank_full > rank_rho + rank_rest : {len(sub)}/{len(table)} runs (must be 0)")
    print(f"  rank_full < max(rank_rho, rank_rest) : {len(sup)}/{len(table)} runs (must be 0)")
    print(f"  rank_rest < lower                   : {len(low)}/{len(table)} runs (must be 0)")

    eps = 1e-9
    out = table[(table["t_rest"] < -eps) | (table["t_rest"] > 1 + eps)]
    if len(out):
        print(f"\nWarning: {len(out)}/{len(table)} runs have t_rest outside [0, 1]:")
        print(out[["run", "rank_rest", "lower", "p_rank", "n_patterns", "upper_rest",
                   "t_rest"]].to_string())

    return table

# ============================================================
# Histograms
# ============================================================

def _histogram(values, xlabel, title, values_stats, bins, figsize, degenerate_pad,text_loc=(0,98,0,98)):
    """One histogram of `values` (NaNs dropped). Returns (fig, ax) or (None, None)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        print(f"Nothing to plot for: {title}")
        return None, None
    text = (f"mean:       {values_stats['mean']:.4f}\n"
                f"min:   {values_stats['min']:.4f}\n"
                f"25%:   {values_stats['25%']:.4f}\n"
                f"75%:   {values_stats['75%']:.4f}\n"
                f"max:   {values_stats['max']:.4f}")

    lo, hi = v.min(), max(v.max(),0.1)
    if hi - lo < 1e-12:                      # all runs identical: give the bar some width
        lo, hi = lo - degenerate_pad, hi + degenerate_pad

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(v, bins=np.linspace(lo, hi, bins + 1),
            color="tab:blue", edgecolor="black", alpha=0.85)
    ax.text(text_loc[0],text_loc[1], text,transform=ax.transAxes,fontsize=10,verticalalignment="top",horizontalalignment="right",bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="white",
                  edgecolor="gray",
                  alpha=0.9))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("number of runs")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


def plot_partial_rank_histograms(base_dirs=BASE_DIRS, folder_pattern=FOLDER_PATTERN,
                                  threshold=GRAD_THRESHOLD, bins=20, device=DEVICE,
                                  w2_tol=0.0, upper_mode="n_distinct", lower_mode="n_distinct", rank_method="float64", normalize=True,
                                  verbose=True, figsize=(7, 5)):
    """
    Two histograms over every run passing the grad-norm threshold, across
    all directories in base_dirs:
      1. partial Jacobian w.r.t. rho.parameters() only
      2. partial Jacobian w.r.t. every parameter but rho's

    normalize=True  : t_rho  = rank(J_rho) / n_f
                      t_rest = (rank(J_rest) - lower) / (upper_rest - lower)
    normalize=False : the raw ranks.

    Returns (table, (fig_rho, ax_rho), (fig_rest, ax_rest)).
    """
    if rank_method not in _METHOD_SPECS:
                raise ValueError(f"method must be one of {list(_METHOD_SPECS)}, got {rank_method!r}")
    if upper_mode not in ("p_rank", "n_distinct"):
        raise ValueError("upper_mode must be 'p_rank' or 'n_distinct'")
    if lower_mode not in ("p_rank", "n_distinct"):
        raise ValueError("upper_mode must be 'p_rank' or 'n_distinct'")

    table = compute_partial_rank_table(
        base_dirs=base_dirs, folder_pattern=folder_pattern, threshold=threshold,
        device=device, w2_tol=w2_tol, upper_mode=upper_mode, lower_mode=lower_mode,rank_method=rank_method, verbose=verbose,
    )

    if len(table) == 0:
        print("No runs to plot.")
        return table, (None, None), (None, None)

    if normalize:
        fig_rho, ax_rho = _histogram(
            table["t_rho"], None,
            rf"Distribution of $t_{{\theta^f}}$ across {len(table)} runs", table.t_rho.describe(), bins, figsize, 0.5e-3,text_loc=(0.21,0.98))
        fig_rest, ax_rest = _histogram(
            table["t_rest"],None,
            rf"Distribution $t_{{par\_rank}}$ across {len(table)} runs", table.t_rest.describe(), bins, figsize, 0.5e-3,text_loc=(0.98,0.98))
    else:
        fig_rho, ax_rho = _histogram(
            table["rank_rho"], "rank(J_rho)",
            "Partial Jacobian w.r.t. rho parameters only", bins, figsize, 0.5)
        fig_rest, ax_rest = _histogram(
            table["rank_rest"], "rank(J_rest)",
            "Partial Jacobian w.r.t. all parameters but rho's", bins, figsize, 0.5)

    return table, (fig_rho, ax_rho), (fig_rest, ax_rest)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    table, _, _ = plot_partial_rank_histograms(upper_mode='p_rank',lower_mode='p_rank',rank_method="float64")
    #print(table)
    print("rank_rho max")
    print(table.loc[table['rank_rho'] == table['rank_rho'].max(),['run','rank_rho', 'rank_rest']])
    print("rank_rho min")
    print(table.loc[table['rank_rho'] == table['rank_rho'].min(),['run','rank_rho', 'rank_rest']])

    # Row(s) with maximum rank_rest
    print("rank_rest max")
    print(table.loc[table['rank_rest'] == table['rank_rest'].max(),['run', 'rank_rho', 'rank_rest']])
    print("rank_rest min")
    print(table.loc[table['rank_rest'] == table['rank_rest'].min(),['run', 'rank_rho', 'rank_rest']])
    plt.show()