'''
jacobian_ranks_2d.py

For every row of a DataFrame (one column with the run folder, one with the
path to model_weights.pt):

  1. build the trained network with build_network_2d,
  2. load that run's training inputs X_train from the dataset .pt whose path
     is stored in the run's config.json,
  3. compute, from ONE Jacobian of theta -> f(theta, X_train) (float64 by
     default), the numerical rank of
        - the full Jacobian                           -> rank_full
        - the partial Jacobian w.r.t. rho's parameters  -> rank_rho
        - the partial Jacobian w.r.t. every parameter
          but rho's (W1, b1, W2, b2)                  -> rank_rest
  4. add them (plus n_train and rank_error) to the DataFrame.

The Jacobian helpers are the same as in the earlier partial-rank code
(_jacobian_blocks, _rank_of, _ranks_from_blocks, _default_dtype,
_METHOD_SPECS); they are repeated here so the script is self-contained.

Methods (as in compute_rank): "float64" (default, default tolerance),
"float32", "float32_rtol" (rtol = 2^-37), "stable_rank". For any method other
than "float64" the column names get a "_<method>" suffix, so several methods
can be added to the same DataFrame.

Usage (in a session):
    df = add_jacobian_ranks(df, folder_col="run_folder", weights_col="weights_path")
Usage (command line):
    python jacobian_ranks_2d.py --df runs.pkl --out runs_with_ranks.pkl
Add
Diagnostic for rank_rho > rank_full type inconsistencies.
 
    from rank_diagnostic import rank_spectra
    res = jacobian(net, X, rank_method="float64", device="cpu", eval_mode=False)
    rank_spectra(res["J"], res["layout"], rho_prefix=RHO_PREFIX)
 
It computes the singular values of J_full, J_rho, J_rest, then shows them on a log scale together
with
  * each matrix's OWN threshold  max(m, n) * eps * sigma_max(matrix)   (what the rank code uses),
  * ONE COMMON threshold         max(m, n) * eps * sigma_max(full)     (guarantees rank_rho <= rank_full).
'''



import torch
from torch.func import functional_call, jacrev,vmap
import argparse
import copy
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd

from typing import Dict, List, Optional, Sequence, Tuple, Union

from buildnetwork import build_network, load_run_config, DEFAULT_DATASET_PATH
from helper import pattern_stats, rank_spectra
RHO_PREFIX = "rho."      # named_parameters(): "rho.E", "rho.B", "rho.attention.*"

_METHOD_SPECS = {
    # method:        (dtype,          rtol,       stable)
    "float64":       (torch.float64,  None,       False), #n_1=10 rtol=2e-13,n_1=100 rtol=2e-11
    "stable_rank":   (torch.float64,  None,       True),
    "float641e-12":  (torch.float64, 1e-12,      False), #use this one maybe to 1e-11
    "float641e-6":   (torch.float64,1e-6,        False)
} 
Keys = Union[str, Sequence[str]]

# ============================================================
# Jacobian ranks (same code as the earlier partial-rank module)
# ============================================================
"""
One function that merges `compute_jacobian` (jacobian_study.py) and `jacobian_ranks`.

    res = jacobian(model, X, keys="full", jac_method="batched",
                   rank_method=None, ...)

returns {"J": tensor (N*output_dim, p_sel) or None, "layout": {key: (col_start, col_stop)},
         "ranks": {"rank_full", "rank_rho", "rank_rest"} or None}

Equivalent calls
----------------
Old A:  J = compute_jacobian(model, X, keys, method="per_sample", chunk_size=c)
New  :  J = jacobian(model, X, keys, jac_method="per_sample", chunk_size=c)["J"]
        (add copy_model=False to also reproduce A's in-place model.eval())

Old B:  r = jacobian_ranks(net, X, method="float64", device="cuda")
New  :  r = jacobian(net, X, rank_method="float64", device="cuda",
                     return_jacobian=False, eval_mode=False)["ranks"]
        (eval_mode=False reproduces B exactly; the default eval_mode=True only differs
         if the net contains BatchNorm / dropout)

Both at once:
        res = jacobian(net, X, keys=["W1", "b1"], rank_method="float64", device="cuda")
        res["J"], res["ranks"], res["layout"]
"""
def old_jacobian(
    model: torch.nn.Module,
    X: torch.Tensor,
    keys: Keys = "full",
    *,
    jac_method: str = "batched",
    chunk_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    rank_method: Optional[str] = None,
    rho_prefix: str = RHO_PREFIX,
    return_jacobian: bool = True,
    copy_model: bool = True,
    eval_mode: bool = True,
) -> dict:
    """
    Stacked Jacobian J[(i, o), j] = d f(theta, X)_{i,o} / d theta_j of theta -> f(theta, X),
    restricted to the parameters `keys`, and (optionally) its ranks.

    Arguments
    ---------
    model, X        : the network and the batched input (N, ...). The model may return
                      output_dim >= 1 values per sample; rows are ordered sample-major
                      (row = i * output_dim + o), as out.flatten() would give.
    keys            : "full" (all parameters) or names of *parameters* in the state_dict
                      (buffers are rejected). Columns of J follow the order of `keys`, each
                      tensor flattened row-major; see res["layout"].
    jac_method      : "batched"    -> jacrev of the batched map (exact, valid even if samples
                                      interact; ~N backward passes through the whole batch).
                      "per_sample" -> vmap(jacrev) of the single-sample map (O(N); equals the
                                      batched Jacobian only if samples do not interact).
    chunk_size      : forwarded to jacrev / vmap to limit memory.
    dtype, device   : cast the (copied) model and X. dtype=None keeps the current one, unless
                      `rank_method` is given, in which case its spec's dtype is used.
                      The torch default dtype is set to `dtype` during the computation
                      (needed e.g. when a forward builds torch.ones(seq_len) in the default dtype).
                      X is cast only if it is floating-point.
    rank_method     : None -> no rank computation. Otherwise a key of _METHOD_SPECS giving
                      (dtype, rtol, stable):  rtol=None -> tolerance max(m, n)*eps*sigma_max,
                      rtol=r -> singular values <= r*sigma_max are zero, stable=True -> stable
                      rank ||M||_F^2 / ||M||_2^2 (a float).
    rho_prefix      : parameters whose name starts with it form J_rho; the others J_rest.
    return_jacobian : False -> res["J"] is None (saves memory when only the ranks matter).
    copy_model      : True -> work on a deep copy, `model` is left untouched.
                      False -> `model` is modified in place (eval mode, dtype/device).
    eval_mode       : call model.eval() (no dropout, BatchNorm uses running statistics).

    common_tol      : Due to the difference in size, sometimes the rank of the sub-matrix can exceed the one of the full matrix
                    common_tol= True thresolds all singular values of all matrix with simga_max(A)*max(A.shape)*eps with A the biggest matrix
                    'full'
    """
    # ---- 0. rank spec / dtype --------------------------------------------------------------
    rtol, stable = None, False
    if rank_method is not None:
        if rank_method not in _METHOD_SPECS:
            raise ValueError(f"rank_method must be one of {list(_METHOD_SPECS)}, got {rank_method!r}")
        spec_dtype, rtol, stable = _METHOD_SPECS[rank_method]
        dtype = dtype if dtype is not None else spec_dtype

    # ---- 1. working copy of model and data -----------------------------------------------
    if copy_model:
        model = copy.deepcopy(model)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    if eval_mode:
        model.eval()
    X = X.to(next(model.parameters()).device)
    if dtype is not None and X.is_floating_point():
        X = X.to(dtype)

    # ---- 2. keys and column layout ---------------------------------------------------------
    named = dict(model.named_parameters())
    if keys is None or keys == "full" or list(keys) == ["full"]:
        keys = list(named)
    else:
        keys = [keys] if isinstance(keys, str) else list(keys)
        for k in keys:
            if k not in named:
                if k in model.state_dict():
                    raise ValueError(f"'{k}' is a buffer, not a trainable parameter.")
                raise KeyError(f"'{k}' not found. Available parameters: {list(named)}")
    layout, offset = {}, 0
    for k in keys:
        layout[k] = (offset, offset + named[k].numel())
        offset += named[k].numel()

    # ---- 3. the Jacobian -------------------------------------------------------------------
    sel = {k: named[k].detach() for k in keys}
    n = X.shape[0]
    old_default = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)
    try:
        if jac_method == "batched":
            def f_batch(p):
                out = functional_call(model, p, (X,))
                if out.numel() % n != 0:
                    raise ValueError(f"Output has {out.numel()} entries, not a multiple of N={n}.")
                return out.reshape(-1)                      # (N * output_dim,)

            jac = jacrev(f_batch, chunk_size=chunk_size)(sel)       # key -> (N*od, *shape)
        elif jac_method == "per_sample":
            def f_single(p, x):
                return functional_call(model, p, (x.unsqueeze(0),)).reshape(-1)   # (od,)

            jac = vmap(jacrev(f_single), in_dims=(None, 0), chunk_size=chunk_size)(sel)  # key -> (N, od, *shape)
        else:
            raise ValueError("jac_method must be 'batched' or 'per_sample'")
    finally:
        torch.set_default_dtype(old_default)

    J = torch.cat([jac[k].reshape(-1, named[k].numel()) for k in keys], dim=1)  # (N*od, p_sel)
    del jac

    # ---- 4. ranks --------------------------------------------------------------------------
    ranks = None
    if rank_method is not None:
        M = J.detach().cpu().numpy()
        empty = np.empty(0, dtype=int)
        rho_idx = np.concatenate([np.arange(a, b) for k, (a, b) in layout.items()
                                  if k.startswith(rho_prefix)] or [empty])
        rest_idx = np.concatenate([np.arange(a, b) for k, (a, b) in layout.items()
                                   if not k.startswith(rho_prefix)] or [empty])

        def rank_of(mats):
            if not mats:
                return 0.0 if stable else 0
            A = np.concatenate(mats, axis=1)
            s = np.linalg.svd(A, compute_uv=False)
            if s.size == 0 or s[0] == 0:
                return 0.0 if stable else 0
            if stable:
                return float((s ** 2).sum() / s[0] ** 2)
            tol = rtol if rtol is not None else max(A.shape) * np.finfo(s.dtype).eps
            return int((s > tol * s[0]).sum())
        rho = [M[:, rho_idx]] if rho_idx.size else []
        rest = [M[:, rest_idx]] if rest_idx.size else []
        ranks = {"rank_full": rank_of(rho + rest), "rank_rho": rank_of(rho), "rank_rest": rank_of(rest)}
        del M

    return {"J": J if return_jacobian else None, "layout": layout, "ranks": ranks}



def ranks_from_matrix(M, layout, rho_prefix=RHO_PREFIX, rtol=None, stable=False, common_tol=False):
    """
    Ranks of J_full = M, J_rho (columns of the keys starting with rho_prefix) and J_rest (the other
    columns), from ONE SVD per matrix. Pure numpy (no torch). Returns (ranks, svals):
        ranks = {"rank_full", "rank_rho", "rank_rest"}   svals = {"full": s, "rho": s, "rest": s}
 
    Threshold used for a matrix A with singular values s:
        common_tol=False : tol_A = t(A) * s[0]            (each matrix has its own threshold)
        common_tol=True  : tol   = t(M) * sigma_max(M)    (ONE absolute threshold, from the full matrix)
      with t(A) = rtol if rtol is not None else max(A.shape) * eps(dtype).
    With one common threshold, rank_rho <= rank_full and rank_rest <= rank_full always hold.
    stable=True returns the stable rank ||A||_F^2 / ||A||_2^2 instead (no threshold, common_tol ignored).
    """
    empty = np.empty(0, dtype=int)
    rho_idx = np.concatenate([np.arange(a, b) for k, (a, b) in layout.items() if k.startswith(rho_prefix)] or [empty])
    rest_idx = np.concatenate([np.arange(a, b) for k, (a, b) in layout.items() if not k.startswith(rho_prefix)] or [empty])
    blocks = {"full": M}
    if rho_idx.size:
        blocks["rho"] = M[:, rho_idx]
    if rest_idx.size:
        blocks["rest"] = M[:, rest_idx]
    svals = {k: np.linalg.svd(A, compute_uv=False) for k, A in blocks.items()}   # one SVD per matrix
 
    def t_of(shape, dtype):
        return rtol if rtol is not None else max(shape) * np.finfo(dtype).eps
 
    def rank_of(name, abs_tol=None):
        if name not in blocks:
            return 0.0 if stable else 0
        s = svals[name]
        if s.size == 0 or s[0] == 0:
            return 0.0 if stable else 0
        if stable:
            return float((s ** 2).sum() / s[0] ** 2)
        if abs_tol is None:
            abs_tol = t_of(blocks[name].shape, s.dtype) * s[0]
        return int((s > abs_tol).sum())
 
    abs_tol = None
    if common_tol and not stable:
        s_full = svals["full"]
        abs_tol = t_of(M.shape, s_full.dtype) * (s_full[0] if s_full.size else 0.0)
    ranks = {"rank_full": rank_of("full", abs_tol), "rank_rho": rank_of("rho", abs_tol),
             "rank_rest": rank_of("rest", abs_tol)}
    return ranks, svals
 
 
def jacobian(
    model: torch.nn.Module,
    X: torch.Tensor,
    keys: Keys = "full",
    *,
    jac_method: str = "batched",
    chunk_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    rank_method: Optional[str] = None,
    rho_prefix: str = RHO_PREFIX,
    return_jacobian: bool = True,
    copy_model: bool = True,
    eval_mode: bool = True,
    common_tol: bool = False,
) -> dict:
    """
    Stacked Jacobian J[(i, o), j] = d f(theta, X)_{i,o} / d theta_j of theta -> f(theta, X),
    restricted to the parameters `keys`, and (optionally) its ranks.
 
    Arguments
    ---------
    model, X        : the network and the batched input (N, ...). The model may return
                      output_dim >= 1 values per sample; rows are ordered sample-major
                      (row = i * output_dim + o), as out.flatten() would give.
    keys            : "full" (all parameters) or names of *parameters* in the state_dict
                      (buffers are rejected). Columns of J follow the order of `keys`, each
                      tensor flattened row-major; see res["layout"].
    jac_method      : "batched"    -> jacrev of the batched map (exact, valid even if samples
                                      interact; ~N backward passes through the whole batch).
                      "per_sample" -> vmap(jacrev) of the single-sample map (O(N); equals the
                                      batched Jacobian only if samples do not interact).
    chunk_size      : forwarded to jacrev / vmap to limit memory.
    dtype, device   : cast the (copied) model and X. dtype=None keeps the current one, unless
                      `rank_method` is given, in which case its spec's dtype is used.
                      The torch default dtype is set to `dtype` during the computation
                      (needed e.g. when a forward builds torch.ones(seq_len) in the default dtype).
                      X is cast only if it is floating-point.
    rank_method     : None -> no rank computation. Otherwise a key of _METHOD_SPECS giving
                      (dtype, rtol, stable):  rtol=None -> tolerance max(m, n)*eps*sigma_max,
                      rtol=r -> singular values <= r*sigma_max are zero, stable=True -> stable
                      rank ||M||_F^2 / ||M||_2^2 (a float).
    rho_prefix      : parameters whose name starts with it form J_rho; the others J_rest.
    return_jacobian : False -> res["J"] is None (saves memory when only the ranks matter).
    copy_model      : True -> work on a deep copy, `model` is left untouched.
                      False -> `model` is modified in place (eval mode, dtype/device).
    eval_mode       : call model.eval() (no dropout, BatchNorm uses running statistics).
    common_tol      : False -> each matrix (full / rho / rest) uses its OWN threshold (see ranks_from_matrix);
                      this can give rank_rho > rank_full.
                      True  -> ONE absolute threshold computed from the full matrix, applied to all three;
                      guarantees rank_rho <= rank_full and rank_rest <= rank_full. Uses rtol if the
                      rank_method gives one, else max(m, n) * eps. Ignored for the stable rank.
    Returns {"J", "layout", "ranks", "svals"}; "svals" = singular values of full / rho / rest (or None),
    so that other tolerances can be tested without recomputing any SVD.
    """
    # ---- 0. rank spec / dtype --------------------------------------------------------------
    rtol, stable = None, False
    if rank_method is not None:
        if rank_method not in _METHOD_SPECS:
            raise ValueError(f"rank_method must be one of {list(_METHOD_SPECS)}, got {rank_method!r}")
        spec_dtype, rtol, stable = _METHOD_SPECS[rank_method]
        dtype = dtype if dtype is not None else spec_dtype
 
    # ---- 1. working copy of model and data -----------------------------------------------
    if copy_model:
        model = copy.deepcopy(model)
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    if eval_mode:
        model.eval()
    X = X.to(next(model.parameters()).device)
    if dtype is not None and X.is_floating_point():
        X = X.to(dtype)
 
    # ---- 2. keys and column layout ---------------------------------------------------------
    named = dict(model.named_parameters())
    if keys is None or keys == "full" or list(keys) == ["full"]:
        keys = list(named)
    else:
        keys = [keys] if isinstance(keys, str) else list(keys)
        for k in keys:
            if k not in named:
                if k in model.state_dict():
                    raise ValueError(f"'{k}' is a buffer, not a trainable parameter.")
                raise KeyError(f"'{k}' not found. Available parameters: {list(named)}")
    layout, offset = {}, 0
    for k in keys:
        layout[k] = (offset, offset + named[k].numel())
        offset += named[k].numel()
 
    # ---- 3. the Jacobian -------------------------------------------------------------------
    sel = {k: named[k].detach() for k in keys}
    n = X.shape[0]
    old_default = torch.get_default_dtype()
    if dtype is not None:
        torch.set_default_dtype(dtype)
    try:
        if jac_method == "batched":
            def f_batch(p):
                out = functional_call(model, p, (X,))
                if out.numel() % n != 0:
                    raise ValueError(f"Output has {out.numel()} entries, not a multiple of N={n}.")
                return out.reshape(-1)                      # (N * output_dim,)
 
            jac = jacrev(f_batch, chunk_size=chunk_size)(sel)       # key -> (N*od, *shape)
        elif jac_method == "per_sample":
            def f_single(p, x):
                return functional_call(model, p, (x.unsqueeze(0),)).reshape(-1)   # (od,)
 
            jac = vmap(jacrev(f_single), in_dims=(None, 0), chunk_size=chunk_size)(sel)  # key -> (N, od, *shape)
        else:
            raise ValueError("jac_method must be 'batched' or 'per_sample'")
    finally:
        torch.set_default_dtype(old_default)
 
    J = torch.cat([jac[k].reshape(-1, named[k].numel()) for k in keys], dim=1)  # (N*od, p_sel)
    del jac
 
    # ---- 4. ranks --------------------------------------------------------------------------
    ranks, svals = None, None
    if rank_method is not None:
        M = J.detach().cpu().numpy()
        ranks, svals = ranks_from_matrix(M, layout, rho_prefix, rtol, stable, common_tol)
        del M
 
    return {"J": J if return_jacobian else None, "layout": layout, "ranks": ranks, "svals": svals}
 

# ============================================================
# Training inputs
# ============================================================

def _resolve_dataset_path(dataset_path):
    """The path as given; if it does not exist and is relative, also try it relative to the run folder."""
    p = Path(dataset_path)
    if p.exists() or p.is_absolute():
        return p
    q = Path('generated_datasets')/'datasets' / p
    return q if q.exists() else p


def _load_X_train(dataset_path, cache):
    """X_train from the dataset .pt (cached: many runs share the same dataset)."""
    p = _resolve_dataset_path(dataset_path)
    key = str(p)
    if key not in cache:
        data = torch.load(p, map_location="cpu", weights_only=True)
        cache[key] = data["X_train"]
    return cache[key]

# ============================================================
# DataFrame driver
# ============================================================

def add_jacobian_ranks(df, folder_col="run_folder", base_dir_col='base_dir', weights_col="weight_path",
                       rank_method="float64", device="cpu", dataset_key="dataset_path", compute_initial=False,
                       default_dataset_path=DEFAULT_DATASET_PATH, verbose_every=25,diagnostic=False):
    """
    Add Jacobian-rank columns to `df` (in place; `df` is also returned):

        rank_full, rank_rho, rank_rest   (suffix "_<method>" if method != "float64")
        n_train                          number of training points the ranks were computed on
        rank_error                       None if the row went fine, else the error message

    A row that fails (missing file, bad checkpoint, ...) gets missing values
    and its error message in rank_error; the loop goes on.

    folder_col  : column with the run folder (config.json lives there)
    weights_col : column with the path to model_weights.pt
    dataset_key : key of config.json holding the dataset path
                  (default_dataset_path is used if the key is absent)
    compute_initial : if True, add prefix for the names so to distinguish between the initial_weights and the weights at critical point.
    device      : "cpu" by default (a float64 Jacobian of this size is cheap on CPU)
    diagnositc : if true calls the rank_spectra function in helper that plots the svd, truncated svd and rebuild and re svd on log scale, use to find 
    the noise thresolds
    
    """
    if rank_method not in _METHOD_SPECS:
        raise ValueError(f"method must be one of {list(_METHOD_SPECS)}, got {rank_method!r}")
    suffix = "" if rank_method == "float64" else f"_{rank_method}"
    prefix="" if not compute_initial else "init_"
    stable = _METHOD_SPECS[rank_method][2]
    rtol=_METHOD_SPECS[rank_method][1]
    if compute_initial:
        assert weights_col=="init_weight_path", "Mismatched between compute_initial and weight_col. This will compute the ranks at critical point but will be saved as init_."

    cache, results, n_rows, t0 = {}, [], len(df), time.time()
    for k, (_, row) in enumerate(df.iterrows(), start=1):
        base_dir= Path(row[base_dir_col])
        run_dir= base_dir / row[folder_col]
        try:
            config = load_run_config(run_dir)
            dataset_path = config.get(dataset_key, default_dataset_path)
            X_train = _load_X_train(dataset_path, cache)
            net = build_network(row[weights_col], device=device)
            pattern_rank,n_distinct=pattern_stats(net,X_train)
            if not diagnostic:
                res=jacobian(net,X_train,rank_method=rank_method,device=device,return_jacobian=False,eval_mode=True,common_tol=True)["ranks"]
                res.update(num_distinct_pat=n_distinct,pattern_rank=pattern_rank,n_train=int(X_train.shape[0]), rank_error=None)
                print('is computing initial', compute_initial,res)
            else: #for diagnostic the thresolds, so we also output the jacobian
                res=jacobian(net,X_train,rank_method=rank_method,device=device,return_jacobian=True,eval_mode=True,common_tol=True) #need common cutoff
                rank_spectra(res["J"], res["layout"], rho_prefix=RHO_PREFIX, same_rtol=rtol, show=True,title=run_dir.name)
                res=res["ranks"]
                res.update(num_distinct_pat=n_distinct,pattern_rank=pattern_rank,n_train=int(X_train.shape[0]), rank_error=None)
                print('is computing initial', compute_initial,res)
        except Exception as e:                                   # keep going on a bad row
            print(f"[{k}/{n_rows}] {run_dir.name}: FAILED ({type(e).__name__}: {e})")
            res = {"rank_full": None, "rank_rho": None, "rank_rest": None, "num_distinct_pat": None, "pattern_rank": None,
                   "n_train": None, "rank_error": f"{type(e).__name__}: {e}"}
        results.append(res)

        if verbose_every and k % verbose_every == 0:
            print(f"[{k}/{n_rows}] {time.time() - t0:.0f}s elapsed")

    for name in ("rank_full", "rank_rho", "rank_rest","num_distinct_pat","pattern_rank"):
        values = [r[name] for r in results]
        df[prefix+name + suffix] = (pd.Series(values, index=df.index, dtype="float64") if stable
                             else pd.Series(values, index=df.index, dtype="Int64"))
    df["n_train"] = pd.Series([r["n_train"] for r in results], index=df.index, dtype="Int64")
    df["rank_error"] = pd.Series([r["rank_error"] for r in results], index=df.index, dtype="object")

    n_failed = sum(r["rank_error"] is not None for r in results)
    print(f"Done: {n_rows - n_failed}/{n_rows} rows ok, {n_failed} failed, "
          f"{time.time() - t0:.0f}s.")
    return df

# ============================================================
# Command line
# ============================================================

def _read_df(path):
    path = Path(path)
    return pd.read_csv(path) if path.suffix == ".csv" else pd.read_pickle(path)


def _write_df(df, path):
    path = Path(path)
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_pickle(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Add Jacobian-rank columns to a DataFrame of runs.")
    p.add_argument("--df", required=True, help="DataFrame file (.pkl or .csv)")
    p.add_argument("--out", default=None, help="output file (default: <df>_with_ranks.<ext>)")
    p.add_argument("--folder-col", default="run_folder")
    p.add_argument("--weights-col", default="weights_path")
    p.add_argument("--rank_method", choices=list(_METHOD_SPECS), default="float64")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dataset-key", default="dataset_path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = _read_df(args.df)
    df = add_jacobian_ranks(df, folder_col=args.folder_col, weights_col=args.weights_col,
                            rank_method=args.rank_method, device=args.device, dataset_key=args.dataset_key)
    src = Path(args.df)
    out = args.out or str(src.with_name(f"{src.stem}_with_ranks{src.suffix}"))
    _write_df(df, out)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()