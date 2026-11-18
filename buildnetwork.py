'''
Has functions that 

-reads a model_weight.pt files and builds the network

-Rebuilds the target function from the config

-plot the network with the activation pattern and the training dataset
'''
import json
from pathlib import Path
from posix import truncate

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from relu_network import OneHiddenLayerReLUNet, LinearFeatureAttention
from target_fn_attention import make_target_fn_attention
from data_generation import load_dataset
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

DEFAULT_DATASET_PATH=Path('generated_dataset')/'dataset'/'score_47.77.pt'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# ============================================================
# Helper class to draw in the plot legend
# ============================================================
class MulticolorLineHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle,
        xdescent, ydescent, width, height,
        fontsize, trans
    ):
        colors = orig_handle["colors"]

        artists = []
        n = len(colors)

        for i, color in enumerate(colors):
            x0 = xdescent + width * i / n
            x1 = xdescent + width * (i + 1) / n

            line = Line2D(
                [x0, x1],
                [ydescent + height / 2, ydescent + height / 2],
                color=color,
                linewidth=2.2,
                solid_capstyle="butt",
                transform=trans,
            )
            artists.append(line)

        return artists
# ============================================================
# Build the trained network from model_weights.pt
# ============================================================

def build_network(weights_path, device=DEVICE):
    """
    Load model_weights.pt and reconstruct the OneHiddenLayerReLUNet that was
    trained, with rho = relu_network.LinearFeatureAttention.

    hidden_dim / feature_dim come from W1's shape. seq_len / embed_dim (the
    attention rho's own shape) are read straight off the saved rho.E
    tensor, so no extra config is needed to know the network's
    architecture — just the checkpoint itself.
    """
    state_dict = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )

    W1 = state_dict["W1"]
    b1 = state_dict["b1"]
    W2 = state_dict["W2"]
    b2 = state_dict["b2"]
   

    feature_dim = W1.shape[1]
    seq_len, embed_dim = state_dict["rho.E"].shape

    if seq_len * embed_dim != feature_dim:
        raise ValueError(
            f"rho.E shape ({seq_len}, {embed_dim}) doesn't match "
            f"W1's feature_dim ({feature_dim})"
        )

    rho = LinearFeatureAttention(seq_len=seq_len, embed_dim=embed_dim).to(
        device, dtype=DTYPE
    )
    model = OneHiddenLayerReLUNet(W1, b1, W2, b2, rho=rho).to(
        device, dtype=DTYPE
    )

    # W1/b1/W2/b2 are already set from the constructor call above; this
    # load_state_dict also fills in rho's own weights (E, B, attention.*).
    model.load_state_dict(state_dict)

    model.eval()

    return model

# ============================================================
# Rebuild the true target function from config.json
# ============================================================

def load_run_config(run_dir):
    with open(Path(run_dir) / "config.json", "r") as f:
        return json.load(f)


def _build_target_rho(tp, dtype=DTYPE):
    """
    Build a relu_network.LinearFeatureAttention (the same class used inside
    the trained network) from a flat dict shaped like
    LinearFeatureAttention.get_parameters(), but with rho's own weights
    renamed to avoid colliding with the branch coefficients A/A1/B/B1:
        {"Embed": [...], "Bias": [...], "attention": {...}}
    (i.e. "Embed" == rho.E, "Bias" == rho.B). seq_len / embed_dim come
    from Embed's own shape.
    """
    E = np.array(tp["Embed"])
    seq_len, embed_dim = E.shape

    rho = LinearFeatureAttention(seq_len=seq_len, embed_dim=embed_dim)

    att = tp["attention"]
    with torch.no_grad():
        rho.E.copy_(torch.tensor(E, dtype=dtype))
        rho.B.copy_(torch.tensor(np.array(tp["Bias"]), dtype=dtype))
        rho.attention.in_proj_weight.copy_(
            torch.tensor(att["in_proj_weight"], dtype=dtype)
        )
        rho.attention.in_proj_bias.copy_(
            torch.tensor(att["in_proj_bias"], dtype=dtype)
        )
        rho.attention.out_proj.weight.copy_(
            torch.tensor(att["out_proj_weight"], dtype=dtype)
        )
        rho.attention.out_proj.bias.copy_(
            torch.tensor(att["out_proj_bias"], dtype=dtype)
        )

    rho.eval()
    return rho


def reconstruct_target_fn_attention(config, dtype=DTYPE):
    """
    Rebuild the teacher/target function from config["target_params"].

    target_params holds the branch coefficients A, A1, B, B1 (each
    (output_dim, 1)) AND rho's own weights, flat, at the same dict level
    — not nested under a "rho" key. rho's own weights are stored under
    "Embed"/"Bias" (renamed from E/B) specifically so they don't collide
    with the branch coefficients A/A1/B/B1.
    """
    tp = config["target_params"]

    rho = _build_target_rho(tp, dtype=dtype)

    A = torch.tensor(tp["A"], dtype=dtype).reshape(-1, 1)
    A1 = torch.tensor(tp["A1"], dtype=dtype).reshape(-1, 1)
    B = torch.tensor(tp["B"], dtype=dtype).reshape(-1, 1)
    B1 = torch.tensor(tp["B1"], dtype=dtype).reshape(-1, 1)

    return make_target_fn_attention(A=A, A1=A1, B=B, B1=B1, rho=rho)


# ============================================================
# Plot: true function (band) + training data + network output (by pattern)
# ============================================================

def plot_run(run_dir,dataset_path=DEFAULT_DATASET_PATH, out_dir=None,
             n_plot_points=400, band_width_frac=0.01, device=DEVICE,
             jac_rank=None, ax=None,scatter_training=True):
    """
    One figure for one run:
      - light-grey band for the true target function
      - scatter of the training data (X, Y)
      - network output, colored segment by segment according to which
        hidden-layer activation pattern is active over that segment.
 
    Pass an existing `ax` to draw into it instead of creating a new
    figure — used by plot_cluster_representatives to put every
    representative on one shared window. When ax is given, out_dir is
    ignored (saving the whole composite figure is the caller's job).
    """
    if isinstance(run_dir,str):
        run_dir=Path(run_dir)
    if isinstance(dataset_path,str):
        dataset_path=Path(dataset_path)
    run_name=run_dir.name
    
 
    config = load_run_config(run_dir)
    target_fn = reconstruct_target_fn_attention(config)
 
    x_range = tuple(config["x_range"])
    (X_train,Y_train,_,_),_=load_dataset(dataset_path=dataset_path)
 
    model = build_network(run_dir / "model_weights.pt", device=device)
 
    x_plot = torch.linspace(x_range[0], x_range[1], n_plot_points,
                             device=device, dtype=DTYPE).reshape(-1, 1)
    with torch.no_grad():
        y_true = target_fn(x_plot.cpu()).cpu().squeeze(-1).numpy()
        y_pred = model(x_plot).cpu().squeeze(-1).numpy()
        pattern = model.activation_pattern(x_plot).cpu()
 
    x_np = x_plot.cpu().squeeze(-1).numpy()
 
    # Assign an integer id to each distinct activation pattern, in order
    # of first appearance.
    pattern_tuples = [tuple(r.tolist()) for r in pattern]
    unique_patterns, pattern_ids = [], []
    for pt in pattern_tuples:
        if pt not in unique_patterns:
            unique_patterns.append(pt)
        pattern_ids.append(unique_patterns.index(pt))
    pattern_ids = np.array(pattern_ids)
    n_patterns = len(unique_patterns)
    cmap = cm.get_cmap("tab10" if n_patterns <= 10 else "tab20")
 
    fig = ax.figure if ax is not None else None
    owns_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
 
    # True function as a light grey band.
    band = band_width_frac * (y_true.max() - y_true.min() + 1e-8)
    ax.fill_between(x_np, y_true - band, y_true + band,
                     color="lightgray", alpha=0.8, zorder=1,
                     label="target function")
 
    # Training samples used for this run.
    if scatter_training:
        ax.scatter(X_train.squeeze(-1).numpy(), Y.train.(-1).numpy(),
               s=12, color="black", alpha=0.5, zorder=2,
               label="training samples")
 
    # Network output, one color per contiguous activation-pattern segment.
    change_points = np.where(np.diff(pattern_ids) != 0)[0] + 1
    bounds = [0] + change_points.tolist() + [len(x_np)]
    for i in range(len(bounds) - 1):
        s, e = bounds[i], min(bounds[i + 1] + 1, len(x_np))  # +1 overlap so segments connect
        pid = pattern_ids[s]
        ax.plot(x_np[s:e], y_pred[s:e], color=cmap(pid % cmap.N),
                linewidth=2.2, zorder=3)
 
    legend_title = f"# perceived patterns: {n_patterns}"
    if jac_rank is not None:
        legend_title += rf" | $\mathrm{{rank}}(\mathbf{{D}}f(\theta,X))$: {jac_rank}"
    ax.legend(title=legend_title, fontsize=8, loc="best")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(rf"Network output for $n_1$={config['hidden_dim']}")
 
    if owns_fig:
        fig.tight_layout()
 
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{run_name}.png"
            fig.savefig(out_path, dpi=150)
            print(f"Saved: {out_path}")
 
    return fig, ax

if __name__=="__main__":
    run_dir='' #to fill
    fig,ax=plot_run(run_dir,dataset_path=DEFAULT_DATASET_PATH)
    plt.show()
