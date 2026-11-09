'''
expstats_attention_init.py

same thing as expsstats_attention.py, but for runs with the initialization scheme changed

Root layout expected:

    reg_experiments_attention/
        run_47.77_h20_0921_.../
            model_weights.pt
            config.json
            training_log.json
        ...


For each run directory matching a folder pattern like
"run_47.77_hxx_date_*", this file:

  1. Reads training_log.json and keeps only runs whose
     min_normalized_grad_norm < GRAD_THRESHOLD and with the wanted initlisation scheme
  2. For each kept run:
       - Reads config.json's "target_params" 
       - Builds a dataframe to log the local dimension rank(Df(θ,X)) and the number of distinct pattern |A(θ,X)| for each run
       - Then plot, with respect to n1 (the hidden dimension), the average of both quantities
'''




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
FIELDS = ["jac_rank_at_min_grad_norm", "ndistinct_pattern_at_min_grad_norm"]
BASE_DIR= Path(__file__).resolve().parent / "reg_experiments_attention_init"
GRAD_THRESHOLD=1.5e-5
FOLDER_PATTERN = "run_47.77_hxx_date_*"
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
def load_filtered_runs(base_dir=BASE_DIR, folder_pattern=FOLDER_PATTERN, fields=FIELDS,
                        threshold=GRAD_THRESHOLD, verbose=True): #same as the above function
    """
    Scan base_dir for folders matching folder_pattern and keep the ones
    whose training_log.json reports min_normalized_grad_norm < threshold.
    """
    base_dir = Path(base_dir)
    run_dirs = sorted(base_dir.glob(folder_pattern))

    if verbose:
        print(f"Found {len(run_dirs)} matching directories")

    records = []

    for run_dir in run_dirs:
        if not run_dir.is_dir():
            continue

        log_file = run_dir / "training_log.json"
        config_file = run_dir / "config.json"
        weights_file = run_dir / "model_weights.pt"

        missing = [p.name for p in (log_file, config_file, weights_file) if not p.exists()]
        if missing:
            if verbose:
                print(f"{run_dir.name}: missing {missing}")
            continue

        entries = _read_json_or_jsonl(log_file)
        config_entries=_read_json_or_jsonl(config_file)
        init_scheme=config_entries[0].get("initialisation_type")
        found_good_entry = False
        for data in entries:
            if not isinstance(data, dict):
                continue

            norm = data.get("min_normalized_grad_norm")
            temp=data.get("critical_point_reached")
            if norm is None:
                continue
            if norm < threshold and  init_scheme== "weight:he_uniform,bias:uniform,Attention:xavier_uniform,embedding:normal, embedding_bias:normal"  : #change this for filtering with init scheme.
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

def collect(patterns,verbose):
    """patterns: list of glob patterns, e.g. ["run_47.77_h4_0929_*", "run_47.77_h8_0929_*", ...]
       Returns one row per kept run, with n_1 parsed from the folder name."""
    frames = []
    for pat in patterns:
        df = load_filtered_runs(folder_pattern=pat,verbose=verbose)
        if not df.empty:
            frames.append(df)
    runs = pd.concat(frames, ignore_index=True)
    print(runs)
    runs["n_1"] = runs["run"].str.extract(r"_h(\d+)_")[0].astype(int)
    return runs

def aggregate(runs:pd.DataFrame, fields=FIELDS):
    """For each n_1: mean, variance and number of runs of each field."""
    return runs.groupby("n_1")[fields].agg(["mean", "var", "count", "min", "max"]).sort_index()

def plot(agg:pd.DataFrame,title, fields=FIELDS, logx=False,show_minmax=True):
    fig, axes = plt.subplots(1,len(FIELDS),figsize=(6*len(FIELDS), 4))
    axes = np.atleast_1d(axes)
    labels={'jac_rank_at_min_grad_norm': rf"$\mathrm{{rank}}\mathbf{{D}}f(\theta,X)$",
                'ndistinct_pattern_at_min_grad_norm':rf"$|\mathcal{{A}}(\theta,X)|$"}
    n1 = agg.index.values
    for ax, f in zip(axes, fields):
        label=labels[f]
        m = agg[(f, "mean")].values
        s = np.sqrt(np.nan_to_num(agg[(f, "var")].values))   # band = mean ± std
        line, =ax.plot(n1, m, "o-")
        ax.fill_between(n1, m - s, m + s, alpha=0.2)
        ax.set_xlabel(r"$n_1$"); ax.set_ylabel(label); ax.grid(alpha=0.3);ax.set_title(title)
        c=line.get_color()
        if show_minmax:
                    ax.plot(n1, agg[(f, "min")].values, "--", color=c, lw=1, alpha=0.8,
                            label=label + " (min)")
                    ax.plot(n1, agg[(f, "max")].values, ":", color=c, lw=1, alpha=0.8,
                            label=label + " (max)")
        if logx: ax.set_xscale("log")
        ax.axhline(y = 160, color = 'b', linestyle='--',label='# training samples') #hardcoded 16-, in config is the samples, then 80% traning split
    fig.tight_layout()
    return fig
def plot_quantiles_for_small_n1(runs: pd.DataFrame, title, fields=FIELDS, logx=False, show_minmax=True,
         q=(0.25, 0.75), n_train=160):
    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 4))
    axes = np.atleast_1d(axes)
    labels = {'jac_rank_at_min_grad_norm': r"$\mathrm{rank}\,\mathbf{D}f(\theta,X)$",
              'ndistinct_pattern_at_min_grad_norm': r"$|\mathcal{A}(\theta,X)|$"}

    g = runs.groupby("n_1")          # no column selection here, g[f] does it
    n1 = g.size().index.values
    q_lo, q_hi = q

    for ax, f in zip(axes, fields):
        label = labels[f]
        med = g[f].median().values
        lo  = g[f].quantile(q_lo).values
        hi  = g[f].quantile(q_hi).values

        line, = ax.plot(n1, med, "o-", label="median")
        c = line.get_color()
        ax.fill_between(n1, lo, hi, color=c, alpha=0.2,
                        label=f"{int(q_lo*100)}-{int(q_hi*100)}%")

        if show_minmax:
            ax.plot(n1, g[f].min().values, "--", color=c, lw=1, alpha=0.8, label="min")
            ax.plot(n1, g[f].max().values, ":",  color=c, lw=1, alpha=0.8, label="max")

        ax.axhline(y=n_train, color='b', linestyle='--', label='# training samples')

        ax.set_xlabel(r"$n_1$")
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        if logx:
            ax.set_xscale("log")

    fig.tight_layout()
    return fig, axes

def plot1fig(agg: pd.DataFrame,title, fields=FIELDS, logx=False, labels=None, figsize=(10, 5),show_minmax=True):
    fig, ax = plt.subplots(figsize=figsize)
    n1 = agg.index.values
    labels={'jac_rank_at_min_grad_norm': rf"$\mathrm{{rank}}\mathbf{{D}}f(\theta,X)$",
            'ndistinct_pattern_at_min_grad_norm':rf"$|\mathcal{{A}}(\theta,X)|$"} # latex labels for the plot
    for f in fields:
        m = agg[(f, "mean")].values
        s = np.sqrt(np.nan_to_num(agg[(f, "var")].values))   # band = mean ± std
        label = labels[f]
        line, = ax.plot(n1, m, "o-", label=label)             # color picked from the default cycle
        c=line.get_color()
        ax.fill_between(n1, m - s, m + s, color=c, alpha=0.2)
        if show_minmax:
                    ax.plot(n1, agg[(f, "min")].values, "--", color=c, lw=1, alpha=0.8,
                            label=labels[f] + " (min)")
                    ax.plot(n1, agg[(f, "max")].values, ":", color=c, lw=1, alpha=0.8,
                            label=labels[f] + " (max)")
    #ax.axhline(y = 160, color = 'b', linestyle='--',label='# training samples') #hardcoded 16-, in config is the samples, then 80% traning split
    ax.set_xlabel(r"$n_1$")
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend()
    if logx:
        ax.set_xscale("log")

    fig.tight_layout()
    return fig, ax
def plot1fig_quantiles_for_small_n1(runs: pd.DataFrame, title, fields=FIELDS, logx=False, figsize=(10, 5),
             q=(0.25, 0.75), show_minmax=True):
    fig, ax = plt.subplots(figsize=figsize)
    labels = {'jac_rank_at_min_grad_norm': r"$\mathrm{rank}\,\mathbf{D}f(\theta,X)$",
              'ndistinct_pattern_at_min_grad_norm': r"$|\mathcal{A}(\theta,X)|$"}
    g = runs.groupby("n_1")
    n1 = g.size().index.values
    q_lo, q_hi = q

    for f in fields:
        med = g[f].median().values
        lo  = g[f].quantile(q_lo).values
        hi  = g[f].quantile(q_hi).values

        line, = ax.plot(n1, med, "o-", label=labels[f] + " (median)")
        c = line.get_color()
        ax.fill_between(n1, lo, hi, color=c, alpha=0.2,
                        label=labels[f] + f" ({int(q_lo*100)}-{int(q_hi*100)}%)")

        if show_minmax:
            ax.plot(n1, g[f].min().values, "--", color=c, lw=1, alpha=0.8, label=labels[f] + " (min)")
            ax.plot(n1, g[f].max().values, ":",  color=c, lw=1, alpha=0.8, label=labels[f] + " (max)")

    ax.set_xlabel(r"$n_1$")
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=9)
    if logx:
        ax.set_xscale("log")

    fig.tight_layout()
    return fig, ax

if __name__ == "__main__":
    patterns = [f"run_47.77_h10_1106_*" ] #200 is above the 160 training point
    runs = collect(patterns,verbose=True)
    print(set(runs["n_1"].values))
    print("a total of  ", len(runs))
    
    # runs = runs.groupby("n_1", group_keys=False).apply(lambda g: g.sample(n=min(len(g), 10), random_state=0))
    # print(runs.groupby("n_1").size())
    # #runs.boxplot(column=FIELDS[0], by="n_1") #do this for small n_1, because of problem of dead relus and such.
    # agg  = aggregate(runs)
    # print(agg)      # check the counts per n_1
    # fig,ax=plot_quantiles_for_small_n1(runs, 'uniform')
    # #plot1fig(agg,title='uniform')
    # plt.show()