
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from buildnetwork import build_network


def load_csvs(csv_paths):
    """
    Load and concatenate multiple CSV files.
    Checks that all files have the same columns.
    """
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]

    dfs = []

    for path in csv_paths:
        df = pd.read_csv(path)
        dfs.append(df)

    if not dfs:
        raise ValueError("No CSV files provided.")

    reference_columns = set(dfs[0].columns)

    for i, df in enumerate(dfs[1:], start=1):
        if set(df.columns) != reference_columns:
            raise ValueError(
                f"CSV files have different columns: {csv_paths[i]}"
            )

    # Ensure identical column ordering before concatenation
    dfs = [df[dfs[0].columns] for df in dfs]

    return pd.concat(dfs, ignore_index=True)


def plot_rank_histograms(csv_paths,rank_type="rank_full",lower_col="n_train",bins=30,figsize=(12, 5),output_path=None,):
    """
    Plot histograms of normalized rank values.

    Parameters
    ----------
    csv_paths : str, Path, or list
        One or multiple CSV file paths.

    rank_type : str
        One of:
            - "rank_full"
            - "rank_rest"
            - "rank_rho"

    lower_col : str
        Column used as the lower bound (default: n_train).

    bins : int
        Number of histogram bins.

    figsize : tuple
        Figure size.

    output_path : str or Path, optional
        If provided, save the figure to this path.
    """

    valid_rank_types = ["rank_full", "rank_rest", "rank_rho"]

    if rank_type not in valid_rank_types:
        raise ValueError(f"rank_type must be one of {valid_rank_types}")

    df = load_csvs(csv_paths)

    # Determine which upper-bound modes to plot
    modes = ["p_rank"] if rank_type == "rank_rho" else ["p_rank", "n_distinct"]

    upper_columns = {"p_rank": "pattern_rank","n_distinct": "num_distinct_pat",}

    required_columns = { rank_type, lower_col, *[upper_columns[m] for m in modes],}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in CSV files: {missing}")

    fig, axes = plt.subplots( 1, len(modes), figsize=figsize, squeeze=False,)

    axes = axes.ravel()

    for ax, mode in zip(axes, modes):

        upper_col = upper_columns[mode]

        # Extract relevant columns
        data = df[[rank_type, upper_col, lower_col]].copy()

        # Convert to numeric
        data = data.apply(pd.to_numeric, errors="coerce")

        # Remove invalid values
        data = data.dropna()

        # Avoid division by zero
        denominator = data[upper_col] - data[lower_col]
        valid = denominator != 0

        data = data.loc[valid]
        denominator = denominator.loc[valid]

        # Compute normalized rank
        t = (
            data[rank_type] - data[upper_col]
        ) / denominator

        t = t.replace([np.inf, -np.inf], np.nan).dropna()

        if len(t) == 0:
            ax.set_title(f"{mode}: no valid data")
            continue

        ax.hist(t,bins=bins,edgecolor="black",alpha=0.75,)

        ax.set_title(f"{mode} (upper={upper_col})")
        ax.set_xlabel("t")
        ax.set_ylabel("Frequency")
        ax.grid(alpha=0.25)

        ax.axvline(t.mean(),linestyle="--",linewidth=1.5,label=f"Mean = {t.mean():.3f}",)

        ax.legend()

    fig.suptitle(
        f"Normalized {rank_type} distribution",
        fontsize=14,
    )

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, df

if __name__=="__main__":

    parser = argparse.ArgumentParser(
            description="Build and save a dataframe for plotting and histogram experiments.")
    parser.add_argument("--dataframe_paths",nargs="+",type=str,required=True,help="path to the dataframe to concatenate")

    parser.add_argument("--rank_type",type=str,defaut='rank_full',choices=['rank_full','rank_rest','rank_rho'],help="which type of ranks to test the bounds.")

    args=parser.parse_args()

    print(args.dataframe_paths)