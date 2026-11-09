

import matplotlib.pyplot as plt
from pathlib import Path
import json
from expstats_attention import build_network
from sharpness_of_bounds import load_run_config,load_training_inputs, compute_rank,_METHOD_SPECS,DEVICE
FIELDS = ["jac_rank_at_min_grad_norm", "p_rank_at_min_grad_norm","ndistinct_pattern_at_min_grad_norm"]
BASE_DIR= Path(__file__).resolve().parent / "reg_experiments_attention_init" #change here to swtich between SGD and init for Adam
GRAD_THRESHOLD=1.5e-5
FOLDER_PATTERN = "run_47.77_hxx_date_*"
import pandas as pd
import numpy as np
def _read_json_or_jsonl(path): #rewrite this function here instead of importing from expstats_attention.py because of the torch pandas import order DLL conflict.
    """
    Read a file that's either one JSON object/list, or JSONL (one JSON
    object per line). Always returns a list of dicts.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        entries = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries


def load_filtered_runs(base_dir=BASE_DIR, folder_pattern=FOLDER_PATTERN, rank_method="float64",
                        threshold=GRAD_THRESHOLD, device=DEVICE, verbose=True): #same as the above function
    """
    Scan base_dir for folders matching folder_pattern and keep the ones
    whose training_log.json reports min_normalized_grad_norm < threshold.
    """
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
        

        entries = _read_json_or_jsonl(log_file)
        found_good_entry = False
        for data in entries:
            if not isinstance(data, dict):
                continue

            norm = data.get("min_normalized_grad_norm")
            temp=data.get("critical_point_reached")
            if norm is not None and  norm < threshold and  init_scheme== "weight:he_normal,bias:normal,Attention:he_normal,embedding:normal" and temp: #change this for filtering with init scheme.
                record = {"run": run_dir.name}
                model = build_network(run_dir / "model_weights.pt", device=device)
                if model.W2.shape[1] != hidden_dim:
                    print(f"Warning: {run_dir.name}: config hidden_dim={hidden_dim} "
                            f"but W2 has {model.W2.shape[1]} hidden units.")
                
                X_train = load_training_inputs(config)
        
                quantities = compute_rank(model, X_train, device=device, lower_mode='p_rank', method=rank_method, w2_tol=0.0)
                for (i,quantity) in enumerate(quantities[:3]): #we don't want lower
                    record[FIELDS[i]]=quantity
                records.append(record)
                found_good_entry = True
                break  # one summary record per run

        if verbose and not found_good_entry:
            print(f"{run_dir.name}: no entry passed threshold")

    df = pd.DataFrame(records)

    if verbose:
        print(f"\nKept {len(df)} runs")

    return df

def collect(patterns,rank_method,verbose):
    """patterns: list of glob patterns, e.g. ["run_47.77_h_4_0929_*", "run_47.77_h_8_0929_*", ...]
       Returns one row per kept run, with n_1 parsed from the folder name."""
    frames = []
    for pat in patterns:
        df = load_filtered_runs(folder_pattern=pat, rank_method=rank_method,verbose=verbose)
        if not df.empty:
            frames.append(df)
    runs = pd.concat(frames, ignore_index=True)
    print(runs)
    runs["n_1"] = runs["run"].str.extract(r"_h(\d+)_")[0].astype(int)
    return runs

def aggregate(runs:pd.DataFrame, fields=FIELDS):
    """For each n_1: mean, variance and number of runs of each field."""
    return runs.groupby("n_1")[fields].agg(["mean", "var", "count", "min", "max"]).sort_index()





def plot(runs: pd.DataFrame, title, fields=FIELDS, logx=False, figsize=(10, 5),
             q=(0.25, 0.75), show_minmax=True):
    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 4))
    axes = np.atleast_1d(axes)
    labels={'jac_rank_at_min_grad_norm': rf"$\mathrm{{rank}}\mathbf{{D}}f(\theta,X)$",
            'p_rank_at_min_grad_norm':rf"$\mathrm{{rank}}(D)$",
            'ndistinct_pattern_at_min_grad_norm':rf"$|\mathcal{{A}}(\theta,X)|$"}
    g = runs.groupby("n_1")
    n1 = g.size().index.values
    q_lo, q_hi = q

    for ax, f in zip(axes, fields):
        label = labels[f]
        med = g[f].median().values
        lo  = g[f].quantile(q_lo).values
        hi  = g[f].quantile(q_hi).values

        line, = ax.plot(n1, med, "o-", label=label + " (median)")
        c = line.get_color()
        ax.fill_between(n1, lo, hi, color=c, alpha=0.2,
                        label=labels[f] + f" ({int(q_lo*100)}-{int(q_hi*100)}%)")

        if show_minmax:
            ax.plot(n1, g[f].min().values, "--", color=c, lw=1, alpha=0.8, label=labels[f] + " (min)")
            ax.plot(n1, g[f].max().values, ":",  color=c, lw=1, alpha=0.8, label=labels[f] + " (max)")
        ax.axhline(y = 160, color = 'b', linestyle='--',label='# training samples')
        ax.set_xlabel(r"$n_1$")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)

        ax.legend(ncol=2, fontsize=9, loc="best")
        if logx:
            ax.set_xscale("log")
    fig.suptitle(title)
    fig.tight_layout()
    return fig, ax

if __name__ == "__main__":
    rank_method="float64_half" #between float64, float32, float32_rtol and stable_rank
    hidden_dims=[10,20,40,60,80,100,200,180,140,220,160,120,240,480,420,360,300,240] #prevent writing them in order
    #hidden_dims=[12,14,16,18,3,4,5,6,7,8,9,10] #smaller n
    hidden_dims=list(set(hidden_dims))
    list.sort(hidden_dims)
    patterns = [f"run_47.77_h{n}_*" for n in [10]] #200 is above the 160 training point
    runs = collect(patterns,rank_method=rank_method, verbose=True)
    print(runs.groupby("n_1").size())
    runs = runs.groupby("n_1", group_keys=False).apply(lambda g: g.sample(n=min(len(g), 10), random_state=0))
    print(runs.groupby("n_1").size())
    #runs.boxplot(column=FIELDS[0], by="n_1") #do this for small n_1, because of problem of dead relus and such.
    agg  = aggregate(runs)
    print(agg)      # check the counts per n_1
    #fig,ax=plot_quantiles_for_small_n1(runs, 'uniform')
    plot(runs,title=None)
    plt.show()