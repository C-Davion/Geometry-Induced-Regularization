from loadrun import gather_runs
from compute_ranks import add_jacobian_ranks, _write_df,_METHOD_SPECS
from pathlib import Path

import pandas as pd
import argparse



def build_and_save_df_for_plot(base_dirs,patterns,random_sampling=False,init_scheme_filter='uniform',init_scale_filter=1.0,rank_method='float64'\
                                        ,verbose=True,diagnostic=False,out_path='exp_run.csv'):
    runs=gather_runs(base_dirs,patterns,random_sampling=random_sampling,init_scale_filter=init_scale_filter,init_scheme_filter=init_scheme_filter,verbose=verbose)
    runs=add_jacobian_ranks(runs,folder_col='run',base_dir_col='base_dir',weights_col='weight_path',rank_method=rank_method,device='cpu',diagnostic=diagnostic)
    #runs = add_jacobian_ranks(runs, folder_col='run',base_dir_col='base_dir',weights_col='init_weight_path',rank_method=rank_method,device='cpu',compute_initial=True)
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