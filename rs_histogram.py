'''
rank_ratio_attention.py

For every run passing the grad-norm threshold, across several experiment
directories (e.g. reg_experiments_attention and
reg_experiments_attention_init), RECOMPUTES from model_weights.pt, on the
exact training inputs X_train (same train_val_split as in training), in
float64:

    jac_rank = rank of the stacked Jacobian of theta -> f(theta, X_train)
    p_rank   = rank of the activation-pattern matrix
    n_distinct = number of distinct activation patterns
    lower = rank of D[:, A], where
           D = activation-pattern matrix (n_train x hidden_dim, 0/1) with
               duplicate rows removed,
           A = hidden units whose output weight W2 is non-zero
    upper = output_dim * (seq_len * embed_dim + 1) * pattern_count + (# params of rho)
         (pattern_count = 2*p_rank by default, or n_distinct: see upper_mode)
    t  = (jac_rank - lower) / (upper - lower)

and histograms t over all collected runs.

Why recompute everything instead of reading the logged ranks: for a
single-output network the b1-columns of the Jacobian are exactly
D @ diag(W2), whose rank is lower, so jac_rank >= lower must hold (t >= 0). A
t < 0 therefore means the quantities were not computed on the same
weights / data / precision. Recomputing all of them here, with one code
path, removes that source of inconsistency. The logged values are kept in
the table (*_logged columns) and compared against the recomputed ones.

Reads from each run folder:
    config.json        -> hidden_dim, output_dim, seq_len, embed_dim (+ dataset settings)
    training_log.json  -> only used for the grad-norm filter and the *_logged columns
    model_weights.pt   -> the trained network (via expstats_attention.build_network)
'''
import torch
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import copy 
import matplotlib.pyplot as plt
import expstats_attention as ea
from expstats_attention import (
    BASE_DIR,
    DEVICE,
    FIELDS,
    build_network,
    load_run_config,
    _read_json_or_jsonl,
    reconstruct_target_fn_attention,
)
from data_generation import generate_dataset, train_val_split
from helper import stacked_jacobian_rank, pattern_stats

# ============================================================
# Configuration
# ============================================================

# The two experiment directories (siblings of each other by default).
BASE_DIRS = [
    #BASE_DIR,
    #BASE_DIR.parent / "reg_experiments_attention_init",
    BASE_DIR.parent/ "reg_experiments_randomsampling/1104" 
]

# Applied inside every directory in BASE_DIRS.
# One (glob_pattern, label) pair per histogram. The label is what shows up in
# the legend. Add as many pairs as you like: all of them end up stacked in ONE
# figure, each with its own colour.
# None is allowed: a None pattern (or a None entry) is skipped, and a None
# label falls back to the pattern string.
FOLDER_PATTERN = [
    ("run_47.77_h10_*", r"$n_1=10$"),
    ("run_47.77_h140_*", r"$n_1=140$"),
]
GRAD_THRESHOLD=1.5e-5
# expstats_attention.load_filtered_runs only keeps the training_log.json
# fields listed in its FIELDS; p_rank_at_min_grad_norm isn't one of them
# by default, so register it (same list object, so load_filtered_runs sees
# it). Only used for the *_logged comparison columns.
if "p_rank_at_min_grad_norm" not in ea.FIELDS:
    ea.FIELDS.append("p_rank_at_min_grad_norm")

# ============================================================
# Gather runs across directories
# ============================================================

def gather_runs_across_dirs(base_dirs=BASE_DIRS, folder_pattern=FOLDER_PATTERN,
                             threshold=GRAD_THRESHOLD, verbose=True,init_scale_filter=1.0,init_scheme_filter='uniform',use_init_filter=False,random_sampling=False):
    """
    Apply the usual grad-norm filter (expstats_attention) inside every
    directory in base_dirs and concatenate the results. Adds a "base_dir"
    column, since a run's folder name alone no longer says where it lives.
    """
    dfs = []
    for base_dir in base_dirs:
        base_dir = Path(base_dir)
        if verbose:
            print(f"\n=== {base_dir} ===")
        if not base_dir.is_dir():
            print(f"Warning: {base_dir} does not exist, skipping.")
            continue

        df = _gather_filtered_runs(
            folder_pattern, base_dir=base_dir, threshold=threshold, 
            verbose=verbose,init_scale_filter=init_scale_filter, init_scheme_filter=init_scheme_filter, use_init_filter=use_init_filter,random_sampling=random_sampling
        )
        if len(df):
            df = df.copy()
            df["base_dir"] = str(base_dir)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
def _gather_filtered_runs(folder_patterns, base_dir=BASE_DIR,
                           threshold=GRAD_THRESHOLD, verbose=True,init_scale_filter=1.0, init_scheme_filter='uniform',use_init_filter=False,random_sampling=False):
    """
    Collect every run passing the grad-norm threshold, across one or more
    folder patterns — just the logged training_log.json fields (FIELDS),
    no model/network loading. Shared by anything that only needs those
    logged metrics.
    """
    if isinstance(folder_patterns, str):
        folder_patterns = [folder_patterns]
 
    dfs = []
    for pattern in folder_patterns:
        df = load_filtered_runs(base_dir=base_dir, folder_pattern=pattern,
                                 threshold=threshold, verbose=verbose, 
                                 init_scale_filter=init_scale_filter,init_scheme_filter=init_scheme_filter, use_init_filter=use_init_filter,random_sampling=random_sampling)
        if len(df):
            df = df.copy()
            df["folder_pattern"] = pattern
            dfs.append(df)
 
    if not dfs:
        if verbose:
            print("No runs passed the threshold.")
        return pd.DataFrame()
 
    return pd.concat(dfs, ignore_index=True)

def load_filtered_runs(base_dir=BASE_DIR, folder_pattern=FOLDER_PATTERN, rank_method="float64",
                        threshold=GRAD_THRESHOLD, device=DEVICE, verbose=True,init_scale_filter=1.0,init_scheme_filter='uniform',use_init_filter=True, random_sampling=False): #same as the above function
    """
    Scan base_dir for folders matching folder_pattern and keep the ones
    whose training_log.json reports min_normalized_grad_norm < threshold.
    """
    init_schemes={
        "uniform": "weight:he_uniform,bias:uniform,Attention:xavier_uniform,embedding:normal", #1.0 3.0 6.0
        "normal": "weight:he_normal,bias:normal,Attention:xavier_normal,embedding:normal" , #TEMP #3.0
        "uniform_bias": "weight:he_uniform,bias:uniform,Attention:xavier_uniform,embedding:normal, embedding_bias:normal" #1.0 6.0 4.5
    }
    if random_sampling:
        threshold=float('inf') #no training occured.
    if rank_method not in _METHOD_SPECS:
            raise ValueError(f"method must be one of {list(_METHOD_SPECS)}, got {rank_method!r}")
    base_dir = Path(base_dir)
    run_dirs = sorted(base_dir.glob(folder_pattern))

    if verbose:
        print(f"Found {len(run_dirs)} matching directories")

    records = []

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue
        log_file = run_dir / "training_log.json"
        config = load_run_config(run_dir)
        hidden_dim = config["hidden_dim"]
        init_scheme=config["initialisation_type"]
        init_scale=config["initalisation_scale"]
        

        entries = _read_json_or_jsonl(log_file)
        found_good_entry = False
        for data in entries:
            if not isinstance(data, dict):
                continue
            if random_sampling:
                norm=10 #just to move it away from the non type, because there was no training for random sampling
            else:
                norm = data.get("min_normalized_grad_norm")
            if use_init_filter: 
                if norm is not None and  norm < threshold and  init_scheme== init_schemes[init_scheme_filter] and init_scale==init_scale_filter : #change this for filtering with init scheme.
                    record = {"run": run_dir.name}
                    for field in FIELDS:
                        record[field] = data.get(field)
                    records.append(record)
                    found_good_entry = True
                    break  # one summary record per run
            else:
                if norm is not None and  norm < threshold: #change this for filtering with init scheme.
                    print("here")
                    record = {"run": run_dir.name}
                    for field in FIELDS:
                        record[field] = data.get(field)
                    records.append(record)
                    found_good_entry = True
                    break  # one summary record per run

        if verbose and not found_good_entry:
            print(f"{run_dir.name}: no entry passed threshold")

    df = pd.DataFrame(records)

    if verbose:
        print(f"\nKept {len(df)} runs")

    return df

# ============================================================
# Training inputs
# ============================================================

def load_training_inputs(config):
    """
    Rebuild the run's TRAINING inputs X_train exactly as in the training
    script: target function from config, generate_dataset (deterministic,
    its generator is hard-coded), then train_val_split (deterministic:
    fixed seed, local generator). Returns X_train, shape (n_train, input_dim).
    """
    target_fn = reconstruct_target_fn_attention(config)
    X, Y = generate_dataset(
        target_fn,
        n_samples=config["n_samples"],
        input_dim=config.get("input_dim", 1),
        x_range=tuple(config["x_range"]),
        noise_std=config.get("noise_std", 0.0),
        seed=config.get("seed", None),
        is_random=True,
    )
    X_train, _, _, _ = train_val_split(X, Y)
    return X_train

# ============================================================
# lower: rank of the (deduplicated, restricted) activation-pattern matrix
# ============================================================

def _lower_from_pattern(pattern, W2, w2_tol=0.0):
    """
    Pure-numpy core of lower.

    pattern: (n_samples, hidden_dim) 0/1 activation matrix
    W2:      (output_dim, hidden_dim) output weights

    1. Remove duplicate rows of `pattern`            -> D
    2. A = hidden units with a non-zero output weight (|W2| > w2_tol for
       at least one output)
    3. lower = rank of D restricted to the columns in A

    Returns (lower, n_distinct_patterns) where n_distinct_patterns = D.shape[0].
    """
    P = np.asarray(pattern).astype(np.int64)
    D = np.unique(P, axis=0)
    lower=0
    W2 = np.atleast_2d(np.asarray(W2))
    for k in range(W2.shape[0]):
        A = np.where(np.any(np.abs(W2[k,:]) > w2_tol, axis=0))[0]

        if len(A) == 0:
            lower+=0
        else:
            lower += int(np.linalg.matrix_rank(D[:, A].astype(np.float64))) #(160 x n_1) matrix, so is stable.
    return lower, D.shape[0]


def compute_lower(model, X,lower_mode, device=DEVICE, w2_tol=0.0,):
    """
    lower for one trained network: activation patterns of `model` on the
    training inputs X (see _lower_from_pattern for the definition).

    Returns (lower, n_distinct_patterns).
    """
    with torch.no_grad():
        pattern = model.activation_pattern(X.to(device))
    pattern = (pattern > 0).cpu().numpy()
    W2 = model.W2.detach().cpu().numpy()
    lower, Card_A=_lower_from_pattern(pattern, W2, w2_tol=w2_tol)
    return lower if lower_mode=='p_rank' else W2.shape[0]*np.ceil(np.log2(Card_A))

# ============================================================
# Recompute every rank, in float64, at exactly the saved weights
# ============================================================

@contextmanager
def _default_dtype(dtype):
    """
    Temporarily change torch's default dtype. Needed because
    relu_network.LinearFeatureAttention.forward builds
    torch.ones(seq_len) in the default dtype, which would clash with a
    float64 input. (Alternative: in that forward, use
    torch.ones(self.seq_len, dtype=x.dtype, device=x.device).)
    """
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)

_METHOD_SPECS = {
    # method:        (dtype,          rtol,       stable)
    "float32":       (torch.float32,  None,       False),   # model's own precision, default tolerance
    "float32_rtol":  (torch.float32,  2.0 ** -37, False),   # same, but rtol = 2^-37
    "float64":       (torch.float64,  None,       False),   # float64, default tolerance
    "float64_half":   (torch.float64, 2.0**-37,   False),    #float64, custom tolerance
    "stable_rank":   (torch.float64,  None,       True),    # ||J||_F^2 / ||J||_2^2, no tolerance
}


def compute_rank(model, X, lower_mode, method="float64", device=DEVICE, w2_tol=0.0):
    """
    Same as compute_ranks_float64, plus `method` in
    {"float32", "float32_rtol", "float64", "stable_rank"} (see _METHOD_SPECS).
    Returns (jac_rank, p_rank, n_distinct, lower); jac_rank and p_rank are
    floats for "stable_rank". Works on a deep copy: `model` is never modified.
    """
    if method not in _METHOD_SPECS:
        raise ValueError(f"method must be one of {list(_METHOD_SPECS)}, got {method!r}")
    dtype, rtol, stable = _METHOD_SPECS[method]

    with _default_dtype(dtype):
        net = copy.deepcopy(model).to(device=device, dtype=dtype)
        Xc = X.to(device=device, dtype=dtype)

        jac_rank = stacked_jacobian_rank(net, Xc, rtol=rtol, stable=stable)
        p_rank, n_distinct = pattern_stats(net, Xc, rtol=rtol, stable=stable)
        lower = compute_lower(net, Xc, lower_mode, device=device, w2_tol=w2_tol)

    return jac_rank, p_rank, n_distinct, lower



# ============================================================
# upper
# ============================================================

def compute_upper(output_dim, seq_len, embed_dim, pattern_count, rho,upper_mode,X_train,keep_feature=True):
    """
    upper = output_dim * (seq_len * embed_dim + 1) * pattern_count + min(# parameters in rho, number of training samples)

    i.e. n_2 * (n_0 + 1) * pattern_count + min (n_f,n), with pattern_count being
    2*p_rank (rank of the pattern matrix) or n_distinct (number of distinct
    patterns, |A(theta, X)|) — see compute_rank_ratio_table's upper_mode.

    keep_feature=True: add the constant n_f, if False, drop it
    """
    n_rho_params = sum(p.numel() for p in rho.parameters()) if keep_feature else 0
    return output_dim * (seq_len * embed_dim + 1) * pattern_count + min(n_rho_params,X_train.shape[0]) if upper_mode=="n_distinct" else \
        output_dim * (2*(seq_len * embed_dim + 1) * pattern_count+1) + min(n_rho_params,X_train.shape[0])
   

# ============================================================
# Per-run table of lower, upper, t
# ============================================================

def compute_rank_ratio_table(base_dirs=BASE_DIRS, folder_pattern=FOLDER_PATTERN,
                              threshold=GRAD_THRESHOLD, device=DEVICE,
                              w2_tol=0.0, upper_mode="p_rank", lower_mode="p_rank",rank_method="float64", 
                              init_scale_filter=1.0,init_scheme_filter='uniform',use_init_filter=False, verbose=True,random_sampling=False):
    """
    One row per run (across every directory in base_dirs) passing the
    grad-norm threshold. jac_rank, p_rank, n_patterns and lower are all
    RECOMPUTED (float64, X_train, saved weights); the values from
    training_log.json are kept in the *_logged columns for comparison.

        t = (jac_rank - lower) / (upper - lower)

    upper_mode and lower mode: "p_rank" (default) which is rank(D) or "n_distinct" which is |A(θ,Χ)|— which pattern count goes
    into upper (p_rank <= n_distinct, so "n_distinct" gives a larger upper).
    """
    
    runs_df = gather_runs_across_dirs( ####change here. TODO: pass into every functions?
        base_dirs=base_dirs, folder_pattern=folder_pattern,
        threshold=threshold, verbose=verbose, init_scale_filter=init_scale_filter,init_scheme_filter=init_scheme_filter,use_init_filter=use_init_filter,random_sampling=random_sampling
    )

    if len(runs_df) == 0:
        print("No runs passed the threshold.")
        return runs_df

    rows = []
    total=0
    for _, r in runs_df.iterrows():
        if total %50==0:
            print(total)
        run_dir = Path(r["base_dir"]) / r["run"]

        config = load_run_config(run_dir)
        hidden_dim = config["hidden_dim"]
        output_dim = config["output_dim"]
        seq_len = config["seq_len"]
        embed_dim = config["embed_dim"]

        model = build_network(run_dir / "model_weights.pt", device=device)
        if model.W2.shape[1] != hidden_dim:
            print(f"Warning: {run_dir.name}: config hidden_dim={hidden_dim} "
                  f"but W2 has {model.W2.shape[1]} hidden units.")

        X_train = load_training_inputs(config)

        jac_rank, p_rank, n_patterns, lower = compute_rank(
            model, X_train, device=device, lower_mode=lower_mode, method=rank_method, w2_tol=w2_tol,
        )


        pattern_count = p_rank if upper_mode == "p_rank" else n_patterns
        upper = compute_upper(output_dim, seq_len, embed_dim, pattern_count, model.rho,upper_mode,X_train)
        if not random_sampling:
            t = (jac_rank - lower) / (upper - lower) if upper != lower else np.nan #for sharpness of bounds
        else:
            t = jac_rank  #for random  sampling

        rows.append({
            "run": r["run"],
            "base_dir": r["base_dir"],
            "n_train": int(X_train.shape[0]),
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "seq_len": seq_len,
            "embed_dim": embed_dim,
            "jac_rank": jac_rank,
            "p_rank": p_rank,
            "n_patterns": n_patterns,
            "lower": lower,
            "upper": upper,
            "t": t,
            "jac_rank_logged": r.get("jac_rank_at_min_grad_norm", np.nan),
            "p_rank_logged": r.get("p_rank_at_min_grad_norm", np.nan),
            "n_patterns_logged": r.get("ndistinct_pattern_at_min_grad_norm", np.nan),
        })
        total+=1

    table = pd.DataFrame(rows)

    if len(table) == 0:
        return table

    # --- diagnostics: recomputed vs logged ---------------------------
    print(f"\nRecomputed ({rank_method}, X_train) vs logged in training_log.json:")
    for name in ("jac_rank", "p_rank", "n_patterns"):
        logged = f"{name}_logged"
        comparable = table[logged].notna()
        differs = comparable & (table[name] != table[logged])
        print(f"  {name:<11}: {int(differs.sum())}/{int(comparable.sum())} runs differ")

    n_neg_logged = int((table["jac_rank_logged"] < table["lower"]).sum())
    n_neg_recomp = int((table["jac_rank"] < table["lower"]).sum())
    print(f"  runs with jac_rank < lower (i.e. t < 0): "
          f"{n_neg_logged} with the logged jac_rank, {n_neg_recomp} with the recomputed one")

    eps = 1e-9
    out_of_range = table[(table["t"] < -eps) | (table["t"] > 1 + eps)]
    if len(out_of_range):
        print(f"\nWarning: {len(out_of_range)}/{len(table)} runs have t outside [0, 1]:")
        print(out_of_range[["run", "jac_rank", "jac_rank_logged", "lower", "p_rank",
                            "upper", "t"]].to_string())

    return table

# ============================================================
# Histogram of t
# ============================================================

def _normalize_folder_patterns(folder_patterns):
    """
    Turn FOLDER_PATTERN into a list of (glob_pattern, label) pairs.
    Accepts a list of (pattern, label) pairs, a single (pattern, label) pair,
    a list of plain pattern strings, or a single pattern string (label = pattern).

    None is allowed:
      - a None pattern (or a None entry, or FOLDER_PATTERN = None) is skipped,
      - a None label falls back to the pattern string itself.
    """
    if folder_patterns is None:
        return []
    if isinstance(folder_patterns, str):
        return [(folder_patterns, folder_patterns)]
    if (isinstance(folder_patterns, tuple) and len(folder_patterns) == 2
            and all(x is None or isinstance(x, str) for x in folder_patterns)):
        folder_patterns = [folder_patterns]
    pairs = []
    for item in folder_patterns:
        if item is None:
            continue
        if isinstance(item, str):
            pairs.append((item, item))
        else:
            pattern, label = item
            if pattern is None:
                continue
            pairs.append((pattern, pattern if label is None else label))
    return pairs

def plot_rank_ratio_histogram(base_dirs=BASE_DIRS, folder_pattern=FOLDER_PATTERN,
                               threshold=GRAD_THRESHOLD, bins=20, device=DEVICE,
                               w2_tol=0.0, upper_mode="p_rank", lower_mode="p_rank", rank_method="float64", verbose=True, 
                               init_scale_filter=1.0, init_scheme_filter='uniform',use_init_filter=False, random_sampling=False,
                               figsize=(7, 5), unit_range=False):
    """
    Histogram of t = (jac_rank - lower) / (upper - lower) over every run passing
    the grad-norm threshold, across all directories in base_dirs (all
    ranks recomputed, see compute_rank_ratio_table).

    By default the bins span the range of the observed t values. With
    unit_range=True the x-range is forced to [0, 1] (widened only if some
    run falls outside it).

    folder_pattern is a list of (glob_pattern, label) pairs (see FOLDER_PATTERN).
    Each pattern is processed separately (compute_rank_ratio_table is unchanged)
    and its t values form one histogram; all histograms are stacked, with
    shared bins, in a single figure. The returned table is the concatenation of
    the per-pattern tables, with extra "label" and "folder_pattern" columns.

    Returns (table, fig, ax).
    """
    if rank_method not in _METHOD_SPECS:
                raise ValueError(f"method must be one of {list(_METHOD_SPECS)}, got {rank_method!r}")
    if upper_mode not in ("p_rank", "n_distinct"):
            raise ValueError("upper_mode must be 'p_rank' or 'n_distinct'")
    if lower_mode not in ("p_rank", "n_distinct"):
        raise ValueError("lower_mode must be 'p_rank' or 'n_distinct'")
    patterns = _normalize_folder_patterns(folder_pattern)

    # One table per pattern, computed exactly as before.
    tables = []
    for pattern, label in patterns:
        if verbose:
            print(f"\n##### {label}  ({pattern}) #####")
        tbl = compute_rank_ratio_table(
            base_dirs=base_dirs, folder_pattern=pattern,
            threshold=threshold, device=device, w2_tol=w2_tol,
            upper_mode=upper_mode, lower_mode=lower_mode, rank_method=rank_method, verbose=verbose,
            init_scale_filter=init_scale_filter,init_scheme_filter=init_scheme_filter,use_init_filter=use_init_filter,random_sampling=random_sampling
        )
        if len(tbl) == 0:
            print(f"No runs for '{label}' ({pattern}), skipping it.")
            continue
        tbl = tbl.copy()
        tbl["label"] = label
        tbl["folder_pattern"] = pattern
        tables.append((label, tbl))

    if not tables:
        print("No runs to plot.")
        return pd.DataFrame(), None, None

    table = pd.concat([tbl for _, tbl in tables], ignore_index=True)

    # Per-pattern t values (+ summary statistics).
    groups = []   # (label, t values, describe())
    for label, tbl in tables:
        t_g = tbl["t"].dropna().to_numpy(dtype=float)
        if len(t_g) == 0:
            print(f"t is undefined (upper == lower) for every run of '{label}', skipping it.")
            continue
        groups.append((label, t_g, tbl["t"].describe()))

    if not groups:
        print("t is undefined (upper == lower) for every run, nothing to plot.")
        return table, None, None

    t = np.concatenate([g[1] for g in groups])   # all runs, used for the shared bins

    if unit_range:
        lo, hi = min(0.0, t.min()), max(1.0, t.max())
    else:
        lo, hi = t.min(), max(t.max(),0.1)
        if hi - lo < 1e-12:                    # all runs identical: give the bar some width
            lo, hi = lo - 0.5e-3, hi + 0.5e-3

    legend_labels = []
    for label, t_g, stats in groups:
        legend_labels.append(f"{label} (N={len(t_g)}) "
                             f"mean: {stats['mean']:.1f} "
                             f"min: {stats['min']:.1f} "
                             f"max: {stats['max']:.1f}")
    
    latex_labels={
        "p_rank": r"$t_{rank}$",
        "n_distinct": r"$t_{|\mathcal{A}|}$"
    }
    
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(groups))]

    fig, ax = plt.subplots(figsize=figsize)
    # One histogram per pattern, same bins, stacked on top of each other.
    ax.hist([g[1] for g in groups], bins=np.linspace(lo, hi, bins + 1), stacked=True,
            color=colors, label=legend_labels,
            edgecolor="black", alpha=0.85)
    ax.set_ylabel("number of runs")
    if not random_sampling:
        ax.set_title(rf"Distribution of {latex_labels[upper_mode]} across (N={len(t)}) runs")
        ax.legend(loc='upper left')
    else:
        ax.set_title(rf"Distribution of local dimension across (N={len(t)}) runs")
        ax.axvline(x=112, color="red", linestyle="--",label="Average for Adam_Uniform")
        ax.axvline(x=102, color="green", linestyle="--",label="Average for SGD_Uniform")
        ax.legend(loc='upper left')
    fig.tight_layout()

    return table, fig, ax

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    table, fig, ax = plot_rank_ratio_histogram(upper_mode='p_rank',lower_mode='p_rank',rank_method="float64",init_scale_filter=4.5,init_scheme_filter='uniform_bias',use_init_filter=True,random_sampling=True) #chage here to switch the mode between rank(D): p_rank and |A|: n_distinct, and rank_method for float32, float32 custom, float64, stable rank.
    print(table[(table.p_rank != table.p_rank_logged) | (table.n_patterns != table.n_patterns_logged)])
    print(table.t.describe())
    plt.show()