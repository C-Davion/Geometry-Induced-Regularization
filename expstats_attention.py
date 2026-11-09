'''
expstats_attention.py

Same purpose as expstats.py, but for runs trained with the attention-based
feature map (relu_network.LinearFeatureAttention) instead of
QuadraticFeatureMap / PolynomialFeatureMap.

Root layout expected:

    reg_experiments_attention/
        run_47.77_h20_0921_.../
            model_weights.pt
            config.json
            training_log.json
        ...

Based on the example config.json actually seen:
  - rho no longer has an "A"/Updim upscale — it's the same
    relu_network.LinearFeatureAttention for BOTH the target (teacher)
    function and the trained network. Its own weights are stored under
    "Embed" (== rho.E) and "Bias" (== rho.B).
  - config["target_params"] stores the branch coefficients A, A1, B, B1
    (each (output_dim, 1)) and rho's own Embed/Bias/attention flat, all at
    the same dict level (no nested "rho" key). Renaming rho's own E/B to
    Embed/Bias was specifically done to stop them colliding with the
    branch coefficients A/A1/B/B1 — an earlier config version used plain
    "E"/"B" for both, which silently overwrote the branch-B coefficient.
    That's resolved now; see _build_target_rho().

For each run directory matching a folder pattern like
"run_47.77_hxx_date_*", this file:

  1. Reads training_log.json and keeps only runs whose
     min_normalized_grad_norm < GRAD_THRESHOLD.
  2. For each kept run:
       - Reads config.json's "target_params" and rebuilds the true target
         function via target_fn_attention.make_target_fn_attention.
       - Reads model_weights.pt and rebuilds the trained network
         (relu_network.OneHiddenLayerReLUNet with
         rho=relu_network.LinearFeatureAttention) via build_network().
       - Plots, over x_range (from config.json):
           * the true target function as a light grey band
           * the training dataset (X, Y) as a scatter
           * the network's output, colored segment-by-segment according
             to which hidden-layer activation pattern is active.

This intentionally leaves out clustering / representative-run selection
and multi-axes grid plotting (cluster_runs_by_loss, select_representative_runs,
render_run_on_ax, plot_pattern_grid, plot_overall_grid, ...) from
expstats.py — those are for later.
'''

import json
from pathlib import Path

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from relu_network import OneHiddenLayerReLUNet, LinearFeatureAttention
from target_fn_attention import make_target_fn_attention
from data_generation import generate_dataset
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage as sch_linkage, dendrogram as sch_dendrogram
from scipy.spatial.distance import pdist
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent / "reg_experiments_attention"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
GRAD_THRESHOLD = 1.5e-5

# e.g. "run_47.77_h20_0921_*" — adjust to the runs you want to look at.
FOLDER_PATTERN = "run_47.77_hxx_date_*"

# Extra fields (besides min_normalized_grad_norm) to keep around for
# reference in the filtered DataFrame; purely informational here.
FIELDS = [
    "min_normalized_grad_norm",
    "loss_at_min_grad_norm",
    "jac_rank_at_min_grad_norm",
    "ndistinct_pattern_at_min_grad_norm",
]
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
# Load & filter training logs
# ============================================================

def _read_json_or_jsonl(path):
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


def load_filtered_runs(base_dir=BASE_DIR, folder_pattern=FOLDER_PATTERN, use_init_scale_filter=False, init_scale_filter=1.0,
                        threshold=GRAD_THRESHOLD, verbose=True):
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
        init_scale=config_entries[0].get('initalisation_scale')
        found_good_entry = False
        for data in entries:
            if not isinstance(data, dict):
                continue

            norm = data.get("min_normalized_grad_norm")
            if norm is None:
                continue
            if use_init_scale_filter:
                if norm < threshold and init_scale==init_scale_filter: 
                    record = {"run": run_dir.name}
                    for field in FIELDS:
                        record[field] = data.get(field)
                    records.append(record)
                    found_good_entry = True
                    break  # one summary record per run
            else:
                if norm < threshold :
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
# Plot: true function (band) + training data + network output (by pattern)
# ============================================================

def plot_run(run_name, base_dir=BASE_DIR, out_dir=None,
             n_plot_points=400, band_width_frac=0.01, device=DEVICE,
             jac_rank=None, ax=None):
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
    run_dir = Path(base_dir) / run_name
 
    config = load_run_config(run_dir)
    target_fn = reconstruct_target_fn_attention(config)
 
    x_range = tuple(config["x_range"])
    n_samples = config["n_samples"]
    input_dim = config.get("input_dim", 1)
    noise_std = config.get("noise_std", 0.0)
    seed = config.get("seed", None)
 
    # Same call as training (is_random=True), so this reproduces the exact
    # training set regardless of the run's own weight-init seed.
    X, Y = generate_dataset(
        target_fn, n_samples=n_samples, input_dim=input_dim,
        x_range=x_range, noise_std=noise_std, seed=seed, is_random=True,
    )
 
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
    ax.scatter(X.squeeze(-1).numpy(), Y.squeeze(-1).numpy(),
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
# ============================================================
# Cluster runs by network output on their own training data
# ============================================================
 
def compute_run_output_vector(run_name, base_dir=BASE_DIR, device=DEVICE):
    """
    f(theta, X) for one run, flattened to shape (n_samples * output_dim,).
 
    X is the run's own training inputs (rebuilt the same way plot_run
    does). We sort by x before flattening so that runs whose training
    inputs landed at slightly different locations (e.g. different random
    seeds) are still compared roughly position-by-position rather than at
    arbitrary, unrelated indices.
    """
    run_dir = Path(base_dir) / run_name
    config = load_run_config(run_dir)
    target_fn = reconstruct_target_fn_attention(config)
 
    x_range = tuple(config["x_range"])
    n_samples = config["n_samples"]
    input_dim = config.get("input_dim", 1)
    noise_std = config.get("noise_std", 0.0)
    seed = config.get("seed", None)
 
    X, _ = generate_dataset(
        target_fn, n_samples=n_samples, input_dim=input_dim,
        x_range=x_range, noise_std=noise_std, seed=seed, is_random=True,
    )
    order = torch.argsort(X[:, 0])
    X_sorted = X[order]
 
    model = build_network(run_dir / "model_weights.pt", device=device)
    with torch.no_grad():
        Y_pred = model(X_sorted.to(device)).cpu()
 
    return Y_pred.reshape(-1).numpy()  # (n_samples * output_dim,)
def _gather_filtered_runs(folder_patterns, base_dir=BASE_DIR,
                           threshold=GRAD_THRESHOLD, verbose=True,init_scale_filter=1.0,use_init_scale_filter=False):
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
                                 threshold=threshold, verbose=verbose, init_scale_filter=init_scale_filter, use_init_scale_filter=use_init_scale_filter)
        if len(df):
            df = df.copy()
            df["folder_pattern"] = pattern
            dfs.append(df)
 
    if not dfs:
        if verbose:
            print("No runs passed the threshold.")
        return pd.DataFrame()
 
    return pd.concat(dfs, ignore_index=True)
 
 
def _gather_run_output_vectors(folder_patterns, base_dir=BASE_DIR,
                                threshold=GRAD_THRESHOLD, verbose=True):
    """
    Collect every run passing the grad-norm threshold, across one or more
    folder patterns (the network architecture is assumed identical across
    patterns, per the caller's guarantee — only n_samples * output_dim
    needs to match for the vectors to be directly comparable, and that's
    checked below).
 
    Returns (runs_df, X_features) with X_features.shape == (n_runs, D).
    """
    if isinstance(folder_patterns, str):
        folder_patterns = [folder_patterns]
 
    dfs = []
    for pattern in folder_patterns:
        df = load_filtered_runs(base_dir=base_dir, folder_pattern=pattern,
                                 threshold=threshold, verbose=verbose)
        if len(df):
            df = df.copy()
            df["folder_pattern"] = pattern
            dfs.append(df)
 
    if not dfs:
        if verbose:
            print("No runs passed the threshold.")
        return pd.DataFrame(), np.empty((0, 0))
 
    runs_df = pd.concat(dfs, ignore_index=True)
 
    vectors, dims = [], set()
    for run_name in runs_df["run"]:
        vec = compute_run_output_vector(run_name, base_dir=base_dir)
        vectors.append(vec)
        dims.add(vec.shape[0])
 
    if len(dims) > 1:
        raise ValueError(
            f"Runs don't all have the same output-vector length {sorted(dims)} "
            "(n_samples * output_dim differs across runs/folder patterns) "
            "— clustering needs directly comparable vectors. Filter to "
            "runs with matching n_samples/output_dim first."
        )
 
    X_features = np.stack(vectors, axis=0)
    return runs_df, X_features
 
 
def _default_distance_threshold(X_features):
    """
    Rough placeholder distance_threshold: 10% of the median pairwise
    Euclidean distance among the feature vectors. This is NOT a
    principled choice — call plot_cluster_dendrogram() and pick a real
    value by looking at where the merge heights actually jump.
    """
    d = pdist(X_features, metric="euclidean")
    return 0.1 * np.median(d)
def _cluster_runs_by_output_with_features(folder_patterns, base_dir=BASE_DIR,
                                           threshold=GRAD_THRESHOLD,
                                           distance_threshold=None, verbose=True):
    """
    Does the actual clustering work for cluster_runs_by_output, but also
    returns the feature matrix X_features (needed to pick a representative
    run per cluster) and the distance_threshold actually used.
 
    Returns (runs_df, X_features, distance_threshold).
    """
    runs_df, X_features = _gather_run_output_vectors(
        folder_patterns, base_dir=base_dir, threshold=threshold, verbose=verbose,
    )
 
    if len(runs_df) == 0:
        return runs_df, X_features, distance_threshold
 
    if len(runs_df) == 1:
        runs_df = runs_df.copy()
        runs_df["cluster"] = 0
        return runs_df, X_features, distance_threshold
 
    if distance_threshold is None:
        distance_threshold = _default_distance_threshold(X_features)
        if verbose:
            print(
                f"No distance_threshold given; using a rough default of "
                f"{distance_threshold:.4g} — inspect a dendrogram "
                "(plot_cluster_dendrogram) and pass your own value for "
                "anything that matters."
            )
 
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage="ward",
    )
    labels = clustering.fit_predict(X_features)
 
    runs_df = runs_df.copy()
    runs_df["cluster"] = labels
 
    if verbose:
        n_clusters = len(set(labels))
        sizes = pd.Series(labels).value_counts().sort_index()
        print(f"{n_clusters} clusters from {len(runs_df)} runs "
              f"(distance_threshold={distance_threshold:.4g})")
        print(sizes.to_string())
 
    return runs_df, X_features, distance_threshold
def _cluster_quality(members, n_samples):
    """
    The requested per-cluster quality measure, for a candidate cluster
    whose feature vectors are `members` (shape (m, D)):
 
        quality(C) = (1 / sqrt(|C| * n)) * sum_{i in C} || f_C - f(theta_i, X) ||
 
    where f_C is the barycenter (mean) of the cluster's vectors and n is
    the number of training samples each vector was evaluated on
    (n_samples; D = n_samples * output_dim). Lower is tighter.
    """
    centroid = members.mean(axis=0)
    dists = np.linalg.norm(members - centroid, axis=1)
    return dists.sum() / np.sqrt(len(members) * n_samples)
 
 
def _grow_qt_candidate(seed_idx, available, X_features, n_samples, quality_threshold):
    """
    Greedily grow one QT-style candidate cluster starting at seed_idx,
    only considering indices in `available` (points not yet claimed by
    an already-extracted cluster). At each step, add whichever remaining
    point keeps the resulting quality (_cluster_quality) lowest; stop as
    soon as no remaining point can be added without quality reaching
    quality_threshold.
    """
    members = [seed_idx]
    pool = [j for j in available if j != seed_idx]
 
    while pool:
        cur = X_features[members]                                   # (m, D)
        cand = X_features[pool]                                      # (p, D)
        m = len(members)
 
        # Centroid of {members} + each single candidate j, one per row.
        new_centroids = (cur.sum(axis=0)[None, :] + cand) / (m + 1)   # (p, D)
 
        # Sum of distances from the existing members to that new centroid...
        d_existing = np.linalg.norm(
            cur[None, :, :] - new_centroids[:, None, :], axis=2
        ).sum(axis=1)                                                  # (p,)
        # ...plus the candidate point's own distance to it.
        d_new = np.linalg.norm(cand - new_centroids, axis=1)            # (p,)
 
        qualities = (d_existing + d_new) / np.sqrt((m + 1) * n_samples)
 
        best = int(np.argmin(qualities))
        if qualities[best] >= quality_threshold:
            break
 
        members.append(pool.pop(best))
 
    return members
 
 
def _qt_cluster(X_features, n_samples, quality_threshold, verbose=True):
    """
    Quality-Threshold (QT) clustering — Heyer, L. J., Kruglyak, S., &
    Yooseph, S. (1999). "Exploring Expression Data: Identification and
    Analysis of Coexpressed Genes." Genome Research, 9(11), 1106-1115
    (https://doi.org/10.1101/gr.9.11.1106).
 
    The original algorithm: for every point, greedily grow a candidate
    cluster around it, adding whichever remaining point keeps the
    cluster's "quality" (originally: max pairwise diameter) below a
    fixed threshold, for as long as possible; accept the LARGEST
    candidate found across all starting points as a real cluster; remove
    its members; repeat on what's left until every point is assigned.
    The number of clusters is never chosen up front — it falls out of
    repeatedly extracting the largest group that still satisfies the
    quality bound, which is exactly the property requested here.
 
    This adapts QT to use the caller's own quality function
    (_cluster_quality — the normalized sum of distances to the cluster's
    barycenter) instead of the original's max-pairwise-diameter
    criterion. Caveat: diameter is provably monotonic non-decreasing as
    points are added, which is what makes the original's greedy growth
    exact; this centroid-based quality is NOT guaranteed monotonic
    (adding a point shifts the centroid and can lower the average
    distance to it), so this is a greedy heuristic generalization of QT,
    not a literal same-guarantees reimplementation.
 
    Cost: each candidate-growth call is a handful of vectorized numpy
    passes, but it's tried from every remaining point, every round — up
    to O(n_runs^2) candidate-growth calls in the worst case. Fine at the
    scale of an experiment sweep (tens to a few hundred runs); expect it
    to get slow well before a few thousand.
 
    Returns integer cluster labels, shape (n_runs,), 0-indexed in the
    order clusters were extracted (cluster 0 was the first/largest found).

    the normalized sum of distances to the cluster's is defined as
    let f_C(θ,Χ) be the barycenter of cluster C, and n the number of training samples. 
    Then we want
    sum_{i in C} |f_C(θ,X)-f(θ_i,X)|< sqrt{n*C}* quality_thresolds.
    """
    n_runs = X_features.shape[0]
    remaining = list(range(n_runs))
    labels = np.full(n_runs, -1, dtype=int)
    next_label = 0
 
    while remaining:
        best_candidate = None
        for seed in remaining:
            candidate = _grow_qt_candidate(
                seed, remaining, X_features, n_samples, quality_threshold,
            )
            if best_candidate is None or len(candidate) > len(best_candidate):
                best_candidate = candidate
 
        for idx in best_candidate:
            labels[idx] = next_label
 
        if verbose:
            q = _cluster_quality(X_features[best_candidate], n_samples)
            print(f"  cluster {next_label}: {len(best_candidate)} runs, "
                  f"quality={q:.4g}")
 
        next_label += 1
        claimed = set(best_candidate)
        remaining = [j for j in remaining if j not in claimed]
 
    return labels
 
 
def _default_quality_threshold(X_features, n_samples):
    """
    Rough placeholder quality_threshold, derived from what a 2-point
    cluster {a, b} of the two closest runs would score: its quality is
    exactly ||a - b|| / sqrt(2 * n_samples) (barycenter is the midpoint,
    both points sit at half the pairwise distance from it). Using the
    median pairwise distance as a "typical" scale gives a loose starting
    guess. NOT a principled choice — plot_cluster_dendrogram gives a
    rough sense of the data's scale, but tune quality_threshold by hand
    for anything that matters.
    """
    d = pdist(X_features, metric="euclidean")
    return 0.1 * np.median(d) / np.sqrt(2 * n_samples) 
def _cluster_runs_by_output_with_features(folder_patterns, base_dir=BASE_DIR,
                                           threshold=GRAD_THRESHOLD,
                                           quality_threshold=None, verbose=True):
    """
    Does the actual clustering work for cluster_runs_by_output, but also
    returns the feature matrix X_features (needed to pick a representative
    run per cluster) and the quality_threshold actually used.
 
    Returns (runs_df, X_features, quality_threshold).
    """
    runs_df, X_features = _gather_run_output_vectors(
        folder_patterns, base_dir=base_dir, threshold=threshold, verbose=verbose,
    )
 
    if len(runs_df) == 0:
        return runs_df, X_features, quality_threshold
 
    if len(runs_df) == 1:
        runs_df = runs_df.copy()
        runs_df["cluster"] = 0
        return runs_df, X_features, quality_threshold
 
    # n = number of training samples each vector was evaluated on. Vector
    # length is already required to match across runs (checked in
    # _gather_run_output_vectors), so any run's own config gives the
    # right n.
    n_samples = load_run_config(Path(base_dir) / runs_df["run"].iloc[0])["n_samples"]
 
    if quality_threshold is None:
        quality_threshold = _default_quality_threshold(X_features, n_samples)
        if verbose:
            print(
                f"No quality_threshold given; using a rough default of "
                f"{quality_threshold:.4g} — inspect a dendrogram "
                "(plot_cluster_dendrogram) for the data's rough scale and "
                "pass your own value for anything that matters."
            )
 
    if verbose:
        print(f"Running QT clustering (quality_threshold={quality_threshold:.4g})...")
 
    labels = _qt_cluster(X_features, n_samples, quality_threshold, verbose=verbose)
 
    runs_df = runs_df.copy()
    runs_df["cluster"] = labels
 
    if verbose:
        n_clusters = len(set(labels))
        sizes = pd.Series(labels).value_counts().sort_index()
        print(f"{n_clusters} clusters from {len(runs_df)} runs "
              f"(quality_threshold={quality_threshold:.4g})")
        print(sizes.to_string())
 
    return runs_df, X_features, quality_threshold
def cluster_runs_by_output(folder_patterns, base_dir=BASE_DIR,
                            threshold=GRAD_THRESHOLD,
                            quality_threshold=None, verbose=True):
    """
    Cluster runs (from one or more folder patterns, all passing the
    min_normalized_grad_norm threshold) by f(theta, X): the network's
    output over its own training inputs, flattened to
    (n_samples * output_dim,).
 
    Clustering: Quality-Threshold (QT) clustering (see _qt_cluster's
    docstring for the full explanation and citation), which enforces,
    for every returned cluster C:
 
        (1 / sqrt(|C| * n)) * sum_{i in C} || f_C(theta, X) - f(theta_i, X) || < quality_threshold
 
    where f_C(theta, X) is the cluster's barycenter and n is the number
    of training samples. Every returned cluster satisfies this bound by
    construction; the number of clusters is never chosen up front, it's
    whatever falls out of repeatedly extracting the largest group that
    still satisfies it.
 
    If quality_threshold is None, a rough data-dependent default is used
    (see _default_quality_threshold) — treat it as a placeholder, not a
    real choice. Call plot_cluster_dendrogram() first for a rough sense
    of scale.
    """
    runs_df, _, _ = _cluster_runs_by_output_with_features(
        folder_patterns, base_dir=base_dir, threshold=threshold,
        quality_threshold=quality_threshold, verbose=verbose,
    )
    return runs_df
 
def plot_cluster_dendrogram(folder_patterns, base_dir=BASE_DIR,
                             threshold=GRAD_THRESHOLD, verbose=True):
    """
    Plot a Ward-linkage dendrogram of every run passing the threshold, to
    help pick a distance_threshold for cluster_runs_by_output(): a good
    cut is a horizontal line drawn through a tall, sparsely-crossed gap
    in the merge heights (y-axis).
    """
    runs_df, X_features = _gather_run_output_vectors(
        folder_patterns, base_dir=base_dir, threshold=threshold, verbose=verbose,
    )
 
    if len(runs_df) < 2:
        print("Need at least 2 runs to draw a dendrogram.")
        return None
 
    Z = sch_linkage(X_features, method="ward")
 
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(runs_df)), 5))
    sch_dendrogram(Z, labels=runs_df["run"].tolist(), leaf_rotation=90, ax=ax)
    ax.set_ylabel("Ward linkage distance")
    ax.set_title("Run clustering dendrogram (by network output on training data)")
    fig.tight_layout()
 
    return fig, ax, Z
def select_cluster_representatives(folder_patterns, base_dir=BASE_DIR,
                                    threshold=GRAD_THRESHOLD,
                                    quality_threshold=None, verbose=True):
    """
    Cluster runs (cluster_runs_by_output) and pick one representative run
    per cluster: the run whose output vector is closest (Euclidean) to
    its cluster's centroid — i.e. the medoid, the run most typical of
    that cluster's function shape.
 
    Returns (runs_df, representatives) where representatives is a dict
    {cluster_id: run_name}.
    """
    runs_df, X_features, _ = _cluster_runs_by_output_with_features(
        folder_patterns, base_dir=base_dir, threshold=threshold,
        quality_threshold=quality_threshold, verbose=verbose,
    )
 
    if len(runs_df) == 0:
        return runs_df, {}
 
    labels = runs_df["cluster"].to_numpy()
    representatives = {}
    for cluster_id in sorted(runs_df["cluster"].unique()):
        idx = np.where(labels == cluster_id)[0]
        centroid = X_features[idx].mean(axis=0)
        dists = np.linalg.norm(X_features[idx] - centroid, axis=1)
        rep_idx = idx[np.argmin(dists)]
        representatives[cluster_id] = runs_df.iloc[rep_idx]["run"]
 
    return runs_df, representatives
def plot_cluster_representatives(folder_patterns, base_dir=BASE_DIR,
                                  threshold=GRAD_THRESHOLD,
                                  quality_threshold=None, verbose=True,
                                  n_cols=3, subplot_size=(5, 4)):
    """
    Cluster runs by their output on the training data, then plot one
    representative run per cluster (its medoid) using plot_run — all in
    one shared figure/window, arranged in a grid (n_cols columns).
 
    Returns (runs_df, fig, axes).
    """
    runs_df, representatives = select_cluster_representatives(
        folder_patterns, base_dir=base_dir, threshold=threshold,
        quality_threshold=quality_threshold, verbose=verbose,
    )
 
    if len(runs_df) == 0:
        print("No runs to plot.")
        return runs_df, None, None
 
    cluster_ids = sorted(representatives.keys())
    n_clusters = len(cluster_ids)
    n_cols = max(1, min(n_cols, n_clusters))
    n_rows = int(np.ceil(n_clusters / n_cols))
 
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(subplot_size[0] * n_cols, subplot_size[1] * n_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()
 
    for i, cluster_id in enumerate(cluster_ids):
        run_name = representatives[cluster_id]
        row = runs_df.loc[runs_df["run"] == run_name].iloc[0]
        ax = axes_flat[i]
 
        plot_run(run_name, base_dir=base_dir,
                 jac_rank=row["jac_rank_at_min_grad_norm"], ax=ax)
 
        n_in_cluster = int((runs_df["cluster"] == cluster_id).sum())
        ax.set_title(ax.get_title() + f" — cluster {cluster_id} (|C|={n_in_cluster})")
 
    # Hide any unused grid cells (n_clusters doesn't perfectly fill the grid).
    for ax in axes_flat[n_clusters:]:
        ax.axis("off")
 
    fig.tight_layout()
 
    return runs_df, fig, axes
 
 
def plot_cluster_jac_rank_scatter(folder_patterns, base_dir=BASE_DIR,
                                   threshold=GRAD_THRESHOLD,
                                   quality_threshold=None, verbose=True,
                                   jitter=0.08, figsize=(8, 6)):
    """
    Cluster runs by their output on the training data, then scatter every
    run at (cluster_id, jac_rank_at_min_grad_norm): runs in the same
    cluster all sit on the same vertical line (x = cluster_id), with a
    small random horizontal jitter (set jitter=0 to disable) so
    overlapping points stay visible. Each cluster's mean
    jac_rank_at_min_grad_norm is marked with a large '*'.
    """
    runs_df = cluster_runs_by_output(
        folder_patterns, base_dir=base_dir, threshold=threshold,
        quality_threshold=quality_threshold, verbose=verbose,
    )
 
    if len(runs_df) == 0:
        print("No runs to plot.")
        return None
 
    if runs_df["jac_rank_at_min_grad_norm"].isna().any():
        missing = runs_df.loc[
            runs_df["jac_rank_at_min_grad_norm"].isna(), "run"
        ].tolist()
        print(f"Warning: jac_rank_at_min_grad_norm missing for: {missing}")
 
    fig, ax = plt.subplots(figsize=figsize)
 
    rng = np.random.default_rng(0)
    cluster_ids = sorted(runs_df["cluster"].unique())
    cmap = cm.get_cmap("tab10" if len(cluster_ids) <= 10 else "tab20")
 
    for cluster_id in cluster_ids:
        sub = runs_df[runs_df["cluster"] == cluster_id]
        color = cmap(cluster_id % cmap.N)
 
        x_jitter = cluster_id + rng.uniform(-jitter, jitter, size=len(sub))
        ax.scatter(x_jitter, sub["jac_rank_at_min_grad_norm"],
                   color=color, alpha=0.8, s=40,
                   label=f"cluster {cluster_id} (n={len(sub)})", zorder=2)
 
        mean_rank = sub["jac_rank_at_min_grad_norm"].mean()
        ax.scatter([cluster_id], [mean_rank], marker="*", s=280,
                   color=color, edgecolor="black", linewidth=1.2, zorder=3)
 
    ax.set_xticks(cluster_ids)
    ax.set_xlabel("cluster")
    ax.set_ylabel("jac_rank_at_min_grad_norm")
    ax.set_title("Jacobian rank at min grad norm, by cluster")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
 
    return fig, ax
 
# ============================================================
# Cluster runs by network output on their own training data
# ============================================================
def scatter_runs(df:pd.DataFrame):
    import matplotlib.ticker as ticker
    losses=[]
    ndistinct_patterns=[]
    local_dims=[]
    for row in df.iterrows():
        losses.append(row[1][FIELDS[1]])
        local_dims.append(row[1][FIELDS[2]])
        ndistinct_patterns.append(row[1][FIELDS[3]])
    # Figure 1: local dimension vs loss
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.scatter(losses, local_dims)
    ax1.set_xlabel("Training Loss")
    ax1.set_ylabel(rf"$\mathrm{{rank}}(\mathbb{{D}}f(\theta,X))$")
    ax1.set_title("Local dimension vs Loss")
    fig1.tight_layout()
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    for x in ax1.get_xticks():
        ax1.axvline(x, linestyle="--", alpha=0.4, linewidth=0.8)
    fig1.tight_layout()


    # Figure 2: number of distinct patterns vs loss
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.scatter(losses, ndistinct_patterns)
    ax2.set_xlabel("Loss")
    ax2.set_ylabel(rf"$|\mathcal{{A}}(\theta,X)|$")
    ax2.set_title("Number of distinct patterns vs Loss")
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    for x in ax2.get_xticks():
        ax2.axvline(x, linestyle="--", alpha=0.4, linewidth=0.8)
    fig2.tight_layout()

    return fig1, fig2
    
# Main
# ============================================================
FIELDS = [
    "min_normalized_grad_norm",
    "loss_at_min_grad_norm",
    "jac_rank_at_min_grad_norm",
    "ndistinct_pattern_at_min_grad_norm",
]
if __name__ == "__main__":
    folder_patterns=["run_47.77_*"]
    dfs=[]
    for pattern in folder_patterns:
        df = load_filtered_runs(

            base_dir=BASE_DIR,
            folder_pattern=pattern,
            threshold=24+GRAD_THRESHOLD,
            verbose=True,  
        )
        dfs.append(df)
    if not dfs:
            print("No runs passed the threshold.")
    dfs= pd.concat(dfs, ignore_index=True)
    print("a total of " ,len(dfs))
    #fig1,fig2=scatter_runs(dfs)
    # plt.show()
    # # --- cluster with QT, using your own bound ---------------------------
    # clustered = cluster_runs_by_output(folder_patterns, quality_threshold=0.15)
    # print(clustered[["run", "folder_pattern", "cluster", "jac_rank_at_min_grad_norm"]])

    # --- just the medoid run names per cluster, no plotting ---------------
    # runs_df, reps = select_cluster_representatives(folder_patterns, quality_threshold=0.15)
    # print(reps)   # {cluster_id: run_name}

    # --- one plot per cluster (medoid run), all in one window -------------
    # runs_df, fig, axes = plot_cluster_representatives(folder_patterns, quality_threshold=0.15)
    # plt.show()

    # --- jac_rank scatter, grouped by cluster ------------------------------
    # plot_cluster_jac_rank_scatter(folder_patterns, quality_threshold=0.15)
    # plt.show()

   