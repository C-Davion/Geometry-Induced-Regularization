f'''
Using the run diagnostic on a single run.
Accept a folder of the form

folder/run_score_hxx_date_runid

Inside it should contain
    -config.json
    -2 .pmg
    -initial_weights.pt
    -model_weights.pt
    -training_log.json
'''
import torch
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from expstats_attention import build_network, plot_run,DEVICE
from sharpness_of_bounds import compute_rank, compute_upper,load_training_inputs,load_run_config,_default_dtype
from sharpness_of_bounds_partial import compute_partial_ranks

BASE_DIR = Path(__file__).resolve().parent 




def get_model_traininginput_config_from_rundir(run_dir,device=DEVICE):
    config=load_run_config(run_dir)
    model = build_network(run_dir / "model_weights.pt", device=device)
    X_train=load_training_inputs(config)
    return model,X_train,config
    
def compute_jac_rank(model,X_train,config, rank_method="float64",lower_mode='p_rank',upper_mode='p_rank',w2_tol=0.0,device=DEVICE):
    '''
    Wrapper for compute rank
    '''
    hidden_dim = config["hidden_dim"]
    output_dim = config["output_dim"]
    seq_len = config["seq_len"]
    embed_dim = config["embed_dim"]

    if model.W2.shape[1] != hidden_dim:
        print(f"Warning: config hidden_dim={hidden_dim} "
                f"but W2 has {model.W2.shape[1]} hidden units.")


    jac_rank, p_rank, n_patterns, lower = compute_rank(model, X_train, device=device, lower_mode=lower_mode, method=rank_method, w2_tol=w2_tol,)
    pattern_count = p_rank if upper_mode == "p_rank" else n_patterns
    upper = compute_upper(output_dim, seq_len, embed_dim, pattern_count, model.rho,upper_mode,X_train)


    return jac_rank,p_rank,n_patterns,lower,upper

def compute_jac_rank_partial(model,X_train,config,rank_method="float64",lower_mode='prank',upper_mode='prank',w2_tol=0.0,device=DEVICE):
    'Wrapper for compute_rank_partial'
    res = compute_partial_ranks(model, X_train,upper_mode,lower_mode, rank_method=rank_method, device=device,   w2_tol=w2_tol)

    return res['rank_rest'],res['rank_rho']

def compute_grad_norm(model,config, device=DEVICE):
    """
    Load model_weights.pt (saved in float32), convert to float64,
    and compute the gradient L2 norm divided by sqrt(number of parameters).
    """
    
    X_train,Y_train=load_training_inputs(config,return_Y=True)
    with _default_dtype(torch.float64):
            model = model.to(device=device, dtype=torch.float64)
    
            model.zero_grad(set_to_none=True)
            X_train=X_train.to(device,dtype=torch.float64)
            Y_train=Y_train.to(device,dtype=torch.float64)
            loss_fn=torch.nn.MSELoss()
            # Compute loss
            output = model(X_train)
            loss = loss_fn(output,Y_train)
            loss.backward()
            grads = torch.cat([
                        p.grad.flatten()
                        for p in model.parameters()
                        if p.grad is not None
                    ])
            
            P = grads.numel()                     # number of trainable scalars actually receiving grads
            norm = torch.sqrt((grads ** 2).sum()).item()
            normalized_norm = norm / (P ** 0.5)   # ||grad|| / sqrt(P): per-parameter-averaged magnitude

    return normalized_norm


def display_matrix(matrice, precision=2, name="Matrix"):
    """
    Display a matrix in a human-readable format.

    Args:
        matrice: PyTorch tensor, NumPy array, or array-like object.
        precision: Number of decimal places to display.
        name: Optional matrix name.
    """
    if isinstance(matrice, torch.Tensor):
        matrice = matrice.detach().cpu().numpy()
    else:
        matrice = np.asarray(matrice)

    if matrice.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {matrice.shape}")

    # Disable NumPy truncation
    np.set_printoptions(threshold=np.inf)

    # Format each entry
    formatted = [
        [f"{value:.{precision}f}" for value in row]
        for row in matrice
    ]

    # Calculate column widths
    col_widths = [
        max(len(formatted[i][j]) for i in range(len(formatted)))
        for j in range(matrice.shape[1])
    ]

    print(f"\n{name} {matrice.shape}:")
    print("[")

    for row in formatted:
        print("  [" + "  ".join(
            value.rjust(col_widths[j])
            for j, value in enumerate(row)
        ) + "]")

    print("]")



if __name__=="__main__":
    run_dir= BASE_DIR/ "reg_experiments_attention_init"/"run_47.77_h3_1006_17"
    model,X_train,config=get_model_traininginput_config_from_rundir(run_dir)
    print('normalized_grad=',compute_grad_norm(model,config))
    jac_rank,p_rank,n_distinct,_,_=compute_jac_rank(model,X_train,config)
    print(f'jac_rank={jac_rank}, p_rank={p_rank},n_distinct={n_distinct}')
    rank_rest,rank_rho=compute_jac_rank_partial(model,X_train,config)
    print(f'rank_rest={rank_rest},rank_rho={rank_rho}')
    fig,ax=plot_run(run_dir)
    plt.show()
    