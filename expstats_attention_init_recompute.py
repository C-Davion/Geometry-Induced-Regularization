
from loadrun import gather_runs
from compute_ranks import add_jacobian_ranks, _write_df, _read_df
from pathlib import Path

import pandas as pd
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PLOT_FIELDS=['rank_full','num_distinct_pat','pattern_rank'] 

INIT_PLOT_FIELDS=['init_rank_full','init_num_distinct_pat','init_pattern_rank']

#Colors and NAME
COLOR_CRITICAL = "tab:blue"
COLOR_INIT = "tab:orange"
COLOR_TRAIN = "k"
LABEL_OLD = "critical point"
LABEL_NEW = "initialisation"
# --- Legend -----------------------------------------------------------------
# "shared_bottom" : 2 compact legends under all subplots (colour = series,
#                   line style = statistic)  -> recommended for slides
# "shared_top"    : same, above the subplots
# "per_axes"      : one full legend inside each subplot (like your original)
# "outside_right" : one legend column on the right of the figure
LEGEND_MODE = "shared_bottom"
LEGEND_FONTSIZE = 11
LEGEND_NCOL_SERIES = 2             # columns of the "series" legend
LEGEND_NCOL_STATS = 5              # columns of the "statistics" legend
LEGEND_GAP = 0.02                  # horizontal gap between the 2 legends (fig. fraction)
PER_AXES_LOC = ["lower right", "lower right", "upper left"]   # used in "per_axes"
BOTTOM_MARGIN = 0.14               # room reserved for the shared legend

def stat_handles(color="0.25"):
    """Neutral handles explaining line styles (used by the shared legend)."""
    return [
        Line2D([], [], color=color, lw=1.8, marker="o", ms=5, label="median"),
        Patch(facecolor=color, alpha=0.2, label="25-75%"),
        Line2D([], [], color=color, ls="--", lw=1.2, label="min"),
        Line2D([], [], color=color, ls=":", lw=1.2, label="max"),
        Line2D([], [], color=COLOR_TRAIN, ls="--", lw=1.5, label="# training samples"),
    ]


def series_handles():
    return [
        Line2D([], [], color=COLOR_CRITICAL, lw=3, label=LABEL_OLD),
        Line2D([], [], color=COLOR_INIT, lw=3, label=LABEL_NEW),
    ]


def add_legends(fig, axes):
    fs = LEGEND_FONTSIZE
    if LEGEND_MODE in ("shared_bottom", "shared_top"):
        bottom = LEGEND_MODE == "shared_bottom"
        y = 0.01 if bottom else 0.99
        loc = "lower " if bottom else "upper "
        # two legends side by side, centred as a block around x = 0.5
        l1 = fig.legend(handles=series_handles(), ncol=LEGEND_NCOL_SERIES,
                        loc=loc + "right", bbox_to_anchor=(0.5 - LEGEND_GAP / 2, y),
                        fontsize=fs, title="Series", title_fontsize=fs,
                        frameon=True)
        fig.legend(handles=stat_handles(), ncol=LEGEND_NCOL_STATS,
                   loc=loc + "left", bbox_to_anchor=(0.5 + LEGEND_GAP / 2, y),
                   fontsize=fs, title="Statistics", title_fontsize=fs,
                   frameon=True)
    elif LEGEND_MODE == "per_axes":
        for ax, loc in zip(axes, PER_AXES_LOC):
            ax.legend(handles=series_handles() + stat_handles(), loc=loc,
                      ncol=2, fontsize=fs - 3, framealpha=0.9, columnspacing=1.0)
    elif LEGEND_MODE == "outside_right":
        fig.legend(handles=series_handles() + stat_handles(), loc="center left",
                   bbox_to_anchor=(0.985, 0.5), fontsize=fs, frameon=True)
    else:
        raise ValueError(f"Unknown LEGEND_MODE: {LEGEND_MODE}")





def build_and_save_df_for_plot(base_dirs,patterns,random_sampling=False,init_scheme_filter='uniform',init_scale_filter=1.0,rank_method='float64'\
                                        ,verbose=True,out_path='exp_run.csv'):
    runs=gather_runs(base_dirs,patterns,random_sampling=random_sampling,init_scale_filter=init_scale_filter,init_scheme_filter=init_scheme_filter,verbose=verbose)
    runs=add_jacobian_ranks(runs,folder_col='run',base_dir_col='base_dir',weights_col='weight_path',device='cpu')
    runs = add_jacobian_ranks(runs, folder_col='run',base_dir_col='base_dir',weights_col='init_weight_path',device='cpu',compute_initial=True)
    _write_df(runs,out_path)
    return runs



def aggregate(runs:pd.DataFrame, fields=PLOT_FIELDS):
    """For each n_1: mean, variance and number of runs of each field."""
    return runs.groupby("n_1")[fields].agg(["mean", "var", "count", "min", "max"]).sort_index()





def plot(runs: pd.DataFrame=pd.DataFrame(), dataframe_path:str| None =None , title='Quantities as a function of n_1', rank_method: str='float64', fields=PLOT_FIELDS, logx=False,
             q=(0.25, 0.75), show_minmax=True):
    if runs.empty:
        if dataframe_path is None:
            raise ValueError("Got an empty dataframe and no path to read one from, specify path or pass from_scratch as True in CLI")
        else:
            runs=_read_df(dataframe_path)

    suffix = "" if rank_method == "float64" else f"_{rank_method}"
    fig, axes = plt.subplots(1, len(fields), figsize=(6 * len(fields), 4))
    axes = np.atleast_1d(axes)
    labels={'rank_full'+suffix: rf"$\mathrm{{rank}}\mathbf{{D}}f(\theta,X)$",
            'pattern_rank'+suffix:rf"$\mathrm{{rank}}(D)$",
            'num_distinct_pat'+suffix:rf"$|\mathcal{{A}}(\theta,X)|$"}
    g = runs.groupby("hidden_dim")
    n1 = g.size().index.values
    q_lo, q_hi = q

    for ax, f in zip(axes, fields):
        #PLOT CRITICAL POINT
        label_crit = labels[f+suffix]
        med_crit = g[f+suffix].median().values
        lo_crit  = g[f+suffix].quantile(q_lo).values
        hi_crit  = g[f+suffix].quantile(q_hi).values

        line, = ax.plot(n1, med_crit, "o-",color=COLOR_CRITICAL)#l,bel=label_crit + " (median)")
        c = line.get_color()
        ax.fill_between(n1, lo_crit, hi_crit, color=c, alpha=0.2)
                       # label=labels[f+suffix] + f" ({int(q_lo*100)}-{int(q_hi*100)}%)")

        if show_minmax:
            ax.plot(n1, g[f+suffix].min().values, "--", color=c, lw=1, alpha=0.8)# label=labels[f+suffix] + " (min)")
            ax.plot(n1, g[f+suffix].max().values, ":",  color=c, lw=1, alpha=0.8)# label=labels[f+suffix] + " (max)")
        #PLOT INITIAL VALUES
        label_init = labels[f+suffix]
        med_init = g['init_'+f+suffix].median().values
        lo_init  = g['init_'+f+suffix].quantile(q_lo).values
        hi_init  = g['init_'+f+suffix].quantile(q_hi).values

        line, = ax.plot(n1, med_init,"o-",color=COLOR_INIT,)#label=label_init + " (median)")
        c = line.get_color()
        ax.fill_between(n1, lo_init, hi_init, color=c, alpha=0.2)
                        #label=labels[f+suffix] + f" ({int(q_lo*100)}-{int(q_hi*100)}%)")

        if show_minmax:
            ax.plot(n1, g['init_'+f+suffix].min().values, "--", color=c, lw=1, alpha=0.8)# label=labels[f+suffix] + " (min)")
            ax.plot(n1, g['init_'+f+suffix].max().values, ":",  color=c, lw=1, alpha=0.8) #label=labels[f+suffix] + " (max)")
        ax.axhline(y = 160, color = 'b', linestyle='--')#label='# training samples')
        ax.set_xlabel(r"$n_1$")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)

        ax.legend(ncol=2, fontsize=9, loc="best")
        if logx:
            ax.set_xscale("log")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, BOTTOM_MARGIN, 1, 1))
    add_legends(fig,ax)
    return fig, ax

def main():
    parser = argparse.ArgumentParser(
        description="Build and save a dataframe for plotting experiments."
    )
    
    parser.add_argument("--base_dirs",nargs="+",default=["reg_experiments_attention_init"],help="One or more base directories.")

    parser.add_argument("--hidden_dims",nargs="+",type=int,default=[10, 20, 40, 60, 80, 100, 120, 140, 
                    160, 180,200, 220, 240, 300, 360, 420, 480],help="Hidden dimensions to include.")

    parser.add_argument("--random_sampling",action="store_true",help="Filters the runs issued of random sampling")

    parser.add_argument("--init_scheme_filter",type=str,default="uniform_bias")

    parser.add_argument("--init_scale_filter",type=float,default=1.0)

    parser.add_argument("--rank_method",type=str,choices=["float64", "float32", "float32_rtol", "stable_rank"],default="float64")

    parser.add_argument("--verbose",action=argparse.BooleanOptionalAction,default=True)

    parser.add_argument("--out_path",type=str,default="exp_run.csv")

    parser.add_argument("--from_scratch",action=argparse.BooleanOptionalAction,
            default=True,help="recompute the dataframe from sratch, if false, need to sepcify dataframe_path")

    parser.add_argument("--dataframe_path",type=str,default=None,help="Path to the already computed dataframe")

    #ADD PLOT ARGS

    args = parser.parse_args()

    if args.from_scratch:
    # Construct patterns from hidden dimensions
        patterns = [f"run_47.77_h{n}_*" for n in sorted(set(args.hidden_dims))]

        runs=build_and_save_df_for_plot(base_dirs=args.base_dirs,patterns=patterns,random_sampling=args.random_sampling,
            init_scheme_filter=args.init_scheme_filter,init_scale_filter=args.init_scale_filter,
            rank_method=args.rank_method,verbose=args.verbose,out_path=args.out_path,)

    else:
       
        runs=_read_df(args.dataframe_path)

    #code to plot
    print(runs.columns)
    fig,ax=plot(runs,rank_method=args.rank_method)
    plt.show()

if __name__ == "__main__":
    # '''
    # When refactoring this file, save everything to a dataframe and plot from the dataframe.
    # '''

    # base_dirs=['reg_experiments_attention_init']

    # rank_method="float64" #between float64, float32, float32_rtol and stable_rank
    # hidden_dims=[10,20,40,60,80,100,200,180,140,220,160,120,240,480,420,360,300,240] #prevent writing them in order
    # #hidden_dims=[12,14,16,18,3,4,5,6,7,8,9,10] #smaller n
    # hidden_dims=list(set(hidden_dims))
    # list.sort(hidden_dims)
    # patterns = [f"run_47.77_h{n}_*" for n in [10]] #200 is above the 160 training point
    # runs = build_and_save_df_for_plot(base_dirs,patterns,random_sampling=False,init_scale_filter=4.5,init_scheme_filter='uniform_bias',
    #                                         rank_method=rank_method, verbose=True)
    # print(runs.groupby("n_1").size())
    # runs = runs.groupby("n_1", group_keys=False).apply(lambda g: g.sample(n=min(len(g), 10), random_state=0))
    # print(runs.groupby("n_1").size())
    # #runs.boxplot(column=FIELDS[0], by="n_1") #do this for small n_1, because of problem of dead relus and such.
    # agg  = aggregate(runs)
    # print(agg)      # check the counts per n_1
    # #fig,ax=plot_quantiles_for_small_n1(runs, 'uniform')
    # plot(runs,title=None)
    # plt.show()

    #CLI python expstats_attention_init_recompute.py --base_dirs reg_experiments_attention_init 
    # --init_scale_filter 4.5 --init_scheme_filter uniform_bias
    main()