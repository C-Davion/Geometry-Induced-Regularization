"""
Generate and save a dataframe containing Jacobian rank statistics for
neural network experiments.

This script gathers experiment runs from one or more base directories,
filters them according to the initialization scheme, initialization scale,
and sampling strategy (just the initial weights or not), and computes the ranks of the corresponding
Jacobians; full jacobian, only with respect to feature transformation parameters and only with respect to model weights.

The resulting dataframe is saved as a CSV file, with the rank computation
method appended to the output filename.

Main workflow:
    1. Gather experiment runs matching the specified directory patterns.
    2. Filter runs according to the initialization configuration.
    3. Compute Jacobian ranks using the selected numerical precision/method.
    4. Optionally enable diagnostic information during rank computation: This prints the singular value with a log scales
    5. Save the resulting dataframe to a CSV file.

Command-line arguments:
    --base_dirs:
        One or more directories containing experiment runs.

    --hidden_dims:
        Hidden dimensions to include in the analysis. These are used to
        construct the run directory matching patterns.

    --random_sampling:
        If specified, filters runs associated with random sampling.

    --init_scheme_filter:
        Initialization scheme used to filter experiment runs.
        Defaults to "uniform_bias".

    --init_scale_filter:
        Initialization scale used to filter experiment runs.
        Defaults to 1.0.

    --rank_method:
        Numerical method used to compute Jacobian ranks. Available methods
        are defined in `_METHOD_SPECS`. Defaults to "float64".

    --verbose:
        Enable or disable verbose output during run gathering.

    --diagnostic:
        Enable or disable diagnostic information during Jacobian rank
        computation.

    --out_path:
        Base name of the output CSV file. The selected rank method is
        automatically appended to the filename.

Output:
    A CSV file containing the gathered experiment information and computed
    Jacobian rank statistics. The output filename follows the convention:

        <out_path_without_extension>__<rank_method>.csv

Example:
    python expstats_attention_init_recompute.py \\
        --base_dirs reg_experiments_attention_init \\
        --init_scale_filter 4.5 \\
        --init_scheme_filter uniform_bias \\
        --rank_method float64

    To analyze a selected set of hidden dimensions:

    python expstats_attention_init_recompute.py \\
        --base_dirs reg_experiments_attention_init \\
        --hidden_dims 10 20 40 60 80 \\
        --rank_method float64 \\
        --diagnostic

Main functions:
    build_and_save_df_for_plot:
        Gathers experiment runs, computes their Jacobian ranks, and saves
        the resulting dataframe.

    main:
        Parses command-line arguments, constructs run matching patterns,
        and executes the dataframe generation pipeline.
"""


from loadrun import gather_runs
from compute_ranks import add_jacobian_ranks, _write_df,_METHOD_SPECS
from pathlib import Path

import pandas as pd
import argparse



def build_and_save_df_for_plot(base_dirs,patterns,random_sampling=False,init_scheme_filter='uniform',init_scale_filter=1.0,rank_method='float64'\
                                        ,verbose=True,diagnostic=False,out_path='exp_run.csv'):
    runs=gather_runs(base_dirs,patterns,random_sampling=random_sampling,init_scale_filter=init_scale_filter,init_scheme_filter=init_scheme_filter,verbose=verbose)
    runs=add_jacobian_ranks(runs,folder_col='run',base_dir_col='base_dir',weights_col='weight_path',rank_method=rank_method,device='cpu',diagnostic=diagnostic)
    #runs = add_jacobian_ranks(runs, folder_col='run',base_dir_col='base_dir',weights_col='init_weight_path',rank_method=rank_method,device='cpu',compute_initial=True,diagnostic=diagnostic)
    out_path=out_path.split('.')[0]
    out_path=out_path+'__'+rank_method+'.csv'
    _write_df(runs,out_path)
    return runs



def main():
    parser = argparse.ArgumentParser(
        description="Build and save a dataframe for plotting and histogram experiments."
    )
    
    parser.add_argument("--base_dirs",nargs="+",default=["reg_experiments_attention_init"],help="One or more base directories.")

    parser.add_argument("--hidden_dims",nargs="+",type=int,default=[10, 20, 40, 60, 80, 100, 120, 140, 
                    160, 180,200, 220, 240, 300, 360, 420, 480],help="Hidden dimensions to include.")

    parser.add_argument("--random_sampling",action="store_true",help="Filters the runs issued of random sampling")

    parser.add_argument("--init_scheme_filter",type=str,default="uniform_bias")

    parser.add_argument("--init_scale_filter",type=float,default=1.0)

    parser.add_argument("--rank_method",type=str,choices=list(_METHOD_SPECS.keys()),default="float64")

    parser.add_argument("--verbose",action=argparse.BooleanOptionalAction,default=True)

    parser.add_argument("--diagnostic",action=argparse.BooleanOptionalAction,default=False)

    parser.add_argument("--out_path",type=str,default="exp_run.csv",help="name of the output path, need to add .csv")

    args=parser.parse_args()

    patterns = [f"run_47.77_h{n}_*" for n in sorted(set(args.hidden_dims))]

    runs=build_and_save_df_for_plot(base_dirs=args.base_dirs,patterns=patterns,random_sampling=args.random_sampling,
        init_scheme_filter=args.init_scheme_filter,init_scale_filter=args.init_scale_filter,
        rank_method=args.rank_method,verbose=args.verbose,diagnostic=args.diagnostic,out_path=args.out_path,)
    print(runs.columns)
    print('Succesfully generated')
if __name__=='__main__':
      #CLI python expstats_attention_init_recompute.py --base_dirs reg_experiments_attention_init 
    # --init_scale_filter 4.5 --init_scheme_filter uniform_bias
    main()