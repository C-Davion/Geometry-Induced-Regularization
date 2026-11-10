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
from expstats_attention import build_network, plot_run
from sharpness_of_bounds import compute_rank, compute_lower, compute_upper,load_training_inputs
from sharpness_of_bounds_partial import compute_partial_ranks

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
BASE_DIR = Path(__file__).resolve().parent / 


def plot_network_with_pattern(model):
'''
Wrapper for plot run
'''
#Assert model weight is in the filepath
#Passit to plot run
    pass

def compute_jac_rank(model, rank_method="float64",lower_mode='prank',uppper_mode='prank'):
    '''
    Wrapper for compute rank
    '''
#Assert model weight is in the filepath
#Pass it to compute_rank
# if yes, also compute lower and upper and t
    pass

def compute_jac_rank_partial(model,rank_method="float64",lower_mode='prank',upper_mode='prank'):
    'Wrapper for compute_rank_partial'
    #Assert model weight is in the filepath
    #Pass it to compute_rank
    # if yes, also compute lower and upper and t
    pass

def compute_grad_norm(model):
    '''
    Given model_weights.pt, compute the normalized gradient norm in float64
    '''
    pass


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
    filepath= "reg_experiments_attention"/ " run_47.77_h10_*"
    
    print("hello world")