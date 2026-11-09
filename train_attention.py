"""
train.py
--------
Trains a OneHiddenLayerReLUNet to approximate a target function via
full-batch gradient descent (a single fixed training batch, no minibatching).

During training, logs:
  - the training loss (every epoch)
  - three structural quantities, every `log_every` epochs:
      1. Jacobian rank    (rank of the stacked Jacobian of θ -> f_θ(X) over
                            the whole training batch X)
      2. Pattern rank      (rank of the activation-pattern matrix over X)
      3. # distinct patterns (number of distinct rows in that matrix)

After training, produces:
  - loss_and_quantities.png : loss (left) and the three quantities (right),
                               side by side
  - learned_vs_initial.png  : target function, initial network (light color),
                               and learned network, overlaid
                               (only drawn when input_dim == output_dim == 1)

Everything is saved to a folder whose name is built from a user-supplied
format string, e.g. "run_h{hidden_dim}_{dateofrun}_{runnumber}".
Try with the muon optimizer
"""

import json
from copy import deepcopy
from pathlib import Path
import glob
from typing import Callable, Optional, Tuple
from datetime import datetime
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev
import matplotlib.pyplot as plt
import time
from relu_network import OneHiddenLayerReLUNet, LinearFeatureAttention
from data_generation import generate_dataset,train_val_split
from helper import _jsonify_param,stacked_jacobian_rank,pattern_stats,adam_step_norm,Muon,maybe_warm_restart




# ── Main training routine ────────────────────────────────────────────────────

def train_network(
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    input_dim: int = 1,
    seq_len: int =8,
    embed_dim: int=32,
    hidden_dim: int = 20,
    output_dim: int = 1,
    n_samples: int = 200,
    x_range: Tuple[float, float] = (-3.0, 3.0),
    noise_std: float = 0.0,
    lr: float = 1e-2,
    epochs: int = 2000,
    log_every: int = 10,
    seed: int = 0,
    timestamp: str = None,
    score: Optional[float] = None,
    target_params: Optional[dict] = None,
    out_root: str = "reg_experiments_attention",
    rho: LinearFeatureAttention = None,
    grad_tol: float = 1.5e-5,
):
    """
    Train a OneHiddenLayerReLUNet on a fixed single batch to approximate
    target_fn, tracking loss and structural quantities, then save all
    artefacts to out_root/folder_template.format(**config).

    Parameters
    ----------
    target_fn       : callable (N, input_dim) -> (N, output_dim); the function
                       to learn. Imported from a separate module by the caller.
    input_dim, hidden_dim, output_dim : network architecture
    n_samples       : number of points in the single training batch
    x_range         : range from which X is sampled
    noise_std       : optional Gaussian noise added to labels
    lr              : Adam learning rate
    epochs          : number of full-batch gradient steps
    log_every       : how often (in epochs) to log the 3 structural quantities
    seed            : RNG seed (data + weight init)
    target_params   : optional dict of the target function's own fixed
                       parameters (independent of the network's trainable
                       parameters), e.g. {"A": A, "A1": A1, "B": B, "B1": B1}.
                       Tensor values are stringified (e.g. [1,2,-1] -> "1_2_-1")
                       and made available to folder_template under their key
                       names; raw values are saved in config.json.
    out_root        : parent directory for the experiment folder
    rho             : optional feature-transformation module (see
                       OneHiddenLayerReLUNet); defaults to identity

    Returns
    -------
    net, initial_net, out_dir
    """
    start=time.perf_counter()
    if seed is not None:
        torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # ── 1. Build the (single) training batch ───────────────────────────────
    X, Y = generate_dataset(
        target_fn, n_samples=n_samples, input_dim=input_dim,
        x_range=x_range, noise_std=noise_std, seed=seed, is_random=True
    )
    X_train,X_val,Y_train,Y_val=train_val_split(X,Y)
    X_train=X_train.to(device,dtype=torch.float32)
    X_val=X_val.to(device,dtype=torch.float32)
    Y_train=Y_train.to(device,dtype=torch.float32)
    Y_val=Y_val.to(device,dtype=torch.float32)
    # print(torch.allclose(Y_train.cpu(), X_train.cpu(), atol=1e-2))
    # print((Y_train - X_train).abs().max().item())

     # ── 2. Initialise the network ───────────────────────────────────────────
    init_scale = 3.0     # tune: bigger → more spread across x_range, "more random"
    rho=rho(seq_len,embed_dim,init_scale).to(device,dtype=torch.float32)
    feature_dim = rho.output_dim if rho is not None else input_dim
    feature_param_count=rho.get_parameter_count()
    W1 = torch.nn.init.kaiming_uniform_(torch.empty(hidden_dim, feature_dim, device=device)) * init_scale
    b1 = (torch.rand(hidden_dim, device=device) * 2 - 1) * init_scale     # was: torch.zeros(...)
    W2 = torch.nn.init.kaiming_uniform_(torch.empty(output_dim, hidden_dim, device=device)) * init_scale
    b2 = torch.zeros(output_dim, device=device)
    net = OneHiddenLayerReLUNet(W1, b1, W2, b2, rho=rho).to(device,dtype=torch.float32)

    initial_net = deepcopy(net)   # frozen snapshot, for the "before" plot
    print("X sum:", X.sum().item())              # should be identical every run
    print("W1 sum:", net.W1.sum().item()) 
    

    # ── 3. Full-batch training loop ─────────────────────────────────────────
    losses = []
    val_losses=[]
    log_epochs, jac_ranks, pattern_ranks, n_distinct_list = [], [], [], []
    pattern_hashes=[]
    min_grad_norm=float("inf")
    min_normalized_grad_norm=float("inf")
    jac_rank_at_best_state=10
    p_rank_at_best_state=10
    n_dist_at_best_state=10
    best_state=None    
    critical_point_reached=False
    epoch_reached=None
    last_restart_epoch=-float('inf')
    def gradient_norm():
        net.zero_grad()

        pred = net(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()

        grads = torch.cat([
            p.grad.flatten()
            for p in net.parameters()
            if p.grad is not None
        ])

        P = grads.numel()                     # number of trainable scalars actually receiving grads
        norm = torch.sqrt((grads ** 2).sum()).item()
        normalized_norm = norm / (P ** 0.5)   # ||grad|| / sqrt(P): per-parameter-averaged magnitude
        max_component = grads.max().item()
        min_component = grads.min().item()

        net.zero_grad()   # leave state clean — this is a pure read, not a training step

        return norm, normalized_norm, max_component, min_component
    def closure():
            optimizer.zero_grad()
            pred=net(X_train)
            loss=loss_fn(pred,Y_train)
            loss.backward()
            return loss
    def muon_closure():
        opt_muon.zero_grad()
        opt_bias.zero_grad()
        pred = net(X_train)
        loss = loss_fn(pred, Y_train)
        loss.backward()
        return loss
    def validation():
        net.eval()
        with torch.no_grad():
            val_out=net(X_val)
            val_loss=loss_fn(val_out,Y_val)
            return val_loss
    
    total_epochs = 0
    use_muon=False
    if not use_muon:
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000,threshold=1e-5, threshold_mode='abs', min_lr=1e-8,)
        loss_fn = nn.MSELoss()
        optimizer_config=f"Adam with {scheduler.__class__.__name__}"
    else:
        opt_muon = Muon([net.W1, net.W2], lr=0.02, momentum=0.95)
        opt_bias = torch.optim.Adam([net.b1, net.b2], lr=lr)
        scheduler_weight = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_muon, mode='min', factor=0.5, patience=1000,threshold=1e-5, threshold_mode='abs', min_lr=1e-8,)
        scheduler_bias = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_bias, mode='min', factor=0.5, patience=1000,threshold=1e-5, threshold_mode='abs', min_lr=1e-8,)
        loss_fn=nn.MSELoss()
        #scheduler_weight=torch.optim.lr_scheduler.CosineAnnealingLR(opt_muon, T_max=epochs)
        #scheduler_bias =torch.optim.lr_scheduler.CosineAnnealingLR(opt_bias, T_max=epochs)
        optimizer_config=f"Adam for bias, muon for weight matrices with {scheduler_weight.__class__.__name__}"
    for _ in range(epochs):

        if not use_muon:
            net.train()
            loss = closure()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            grad_norm,_,_,_=gradient_norm() #for LRon plateau scheduler
            scheduler.step()
            losses.append(loss.item())
            val_loss = validation()
            val_losses.append(val_loss.item())
        else:
            net.train()
            loss = muon_closure()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt_muon.step()
            opt_bias.step()
            grad_norm,_,_,_=gradient_norm() #for LRon plateau scheduler
            losses.append(loss.item())
            val_loss=validation()
            val_losses.append(val_loss.item())
            scheduler_weight.step(grad_norm)
            scheduler_bias.step(grad_norm)
        with torch.no_grad():
                pat = net.activation_pattern(X_train)
                pat_hash = hash(tuple(pat.flatten().tolist()))     # add this
                pattern_hashes.append(pat_hash)                     # and this

        if total_epochs % log_every == 0:
            jr = stacked_jacobian_rank(net, X_train)
            p_rank, n_dist = pattern_stats(net, X_train)
            log_epochs.append(total_epochs)
            jac_ranks.append(jr)
            pattern_ranks.append(output_dim * p_rank)
            n_distinct_list.append(output_dim * (feature_dim + 1) * n_dist+feature_param_count)

        grad_norm, normalized_grad_norm,max_component,min_component = gradient_norm()
        #a_norm = adam_step_norm(optimizer)
        # if total_epochs %10==0 and total_epochs>30000:
        #     print(f"epoch {total_epochs}: grad_norm={grad_norm:.3e}  adam_step_norm={a_norm:.3e}")
        if normalized_grad_norm < min_normalized_grad_norm:
            min_normalized_grad_norm = normalized_grad_norm
            min_grad_norm = grad_norm
            max_coord_at_min=max_component
            min_coord_at_min=min_component
            epoch_reached = total_epochs
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            loss_at_best_state=loss.item()
            val_loss_at_best_state=val_loss.item()
            jr = stacked_jacobian_rank(net, X_train)
            p_rank, n_dist = pattern_stats(net, X_train)
            jac_rank_at_best_state=jr
            p_rank_at_best_state=p_rank
            n_dist_at_best_state=n_dist
        if use_muon:
            restarted, last_restart_epoch = maybe_warm_restart(opt_muon, scheduler_weight, opt_bias, scheduler_bias,total_epochs, epoch_reached, last_restart_epoch,patience_for_restart=5000, restart_factor=0.1,)
            if restarted:
                print(f"epoch {total_epochs}: warm restart — lr reset "
                f"(muon={opt_muon.param_groups[0]['lr']:.2e}, bias={opt_bias.param_groups[0]['lr']:.2e})")
        if normalized_grad_norm<= grad_tol:
            print("reached critical point, stopping training")
            critical_point_reached = True
            break
        total_epochs += 1
    if best_state is not None:
        net.load_state_dict(best_state)   
    with torch.no_grad():
        true_mse = loss_fn(net(X_train), Y_train).item()
    print("true MSE:", true_mse)
    if use_muon:
        print("opt_muon_lr",opt_muon.param_groups[0]['lr'])
        print("opt_bias_lr",opt_bias.param_groups[0]['lr'])
    end=time.perf_counter()
    # tail = pattern_hashes[-200:]
    # print("distinct patterns in tail:", len(set(tail)))
    # print("tail sequence:", tail[-20:], "...")
   
    # ── 4. Resolve output folder name ───────────────────────────────────────
    target_params = target_params or {}

    config = dict(
    input_dim=input_dim,
    seq_len=seq_len,
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_samples=n_samples,
    x_range=x_range,
    noise_std=noise_std,
    lr=lr,
    epochs=epochs,
    log_every=log_every,
    seed=seed,
    timestamp=timestamp,
    total_epochs=total_epochs,
    grad_tol=grad_tol,
    initalisation_scale=init_scale
    )
    max_jacobian_rank = max(jac_ranks) if jac_ranks else 0

    monthday = datetime.now().strftime("%m%d")

    prefix = f"run_{score:.2f}_h{hidden_dim}_{monthday}_"

    existing = sorted(Path(out_root).glob(prefix + "*"))
    run_idx = len(existing)

    folder_name = f"{prefix}{run_idx}"

    out_dir = Path(out_root) / folder_name
    out_dir.mkdir(parents=True, exist_ok=False)
    # ── 5. Plot: loss (left) and 3 quantities (right), side by side ────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(losses, color="tab:red",label="training loss")
    axes[0].plot(val_losses,color="tab:blue",label="validation_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_yscale("log")
    axes[0].set_title("Training loss")
    axes[0].legend()

    axes[1].plot(log_epochs, jac_ranks, label=rf"$\mathrm{{rank}}(\mathbf{{D}}f(\theta,X))$", marker="o", markersize=3)
    axes[1].plot(log_epochs,pattern_ranks, label=rf"$n_2\mathrm{{rank}}D$", marker="s", markersize=3)
    axes[1].plot(log_epochs, n_distinct_list, label=rf"$n_2(n_0+1)|\mathcal{{A}}(\theta,X)|+n_f$", marker="^", markersize=3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("value")
    axes[1].set_title("Structural quantities during training")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "loss_and_quantities.png", dpi=150)
    plt.close(fig)

    # ── 6. Plot: learned vs initial function (only for 1D -> 1D) ───────────
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    if input_dim == 1 and output_dim == 1:
        x_plot = torch.linspace(x_range[0], x_range[1], 300).unsqueeze(1).to(device)
        with torch.no_grad():
            pat_train = net.activation_pattern(X_train)
            pat_plot  = net.activation_pattern(x_plot)
            print("distinct patterns on training data:", pat_train.unique(dim=0).shape[0])
            print("distinct patterns on plot range:   ", pat_plot.unique(dim=0).shape[0])
        # print("x_plot:", x_plot.device)
        # print("net:", next(net.parameters()).device)
        # print("initial:", next(initial_net.parameters()).device)
        with torch.no_grad():
            y_initial = initial_net(x_plot)
            y_learned = net(x_plot)
            y_true    = target_fn(x_plot.cpu())
        print("y_learned range:", y_learned.min().item(), y_learned.max().item())
        print("y_true    range:", y_true.min().item(), y_true.max().item())
        x_plot = x_plot.cpu()
        y_true = y_true.cpu()
        y_initial = y_initial.cpu()
        y_learned = y_learned.cpu()
        ax2.set_ylim(y_true.min().item() - 1, y_true.max().item() + 1)
        ax2.plot(x_plot.squeeze(), y_true.squeeze(),
                  label="Target function", color="black", linewidth=2)
        ax2.plot(x_plot.squeeze(), y_initial.squeeze(),
                  label="Initial network", color="lightskyblue", linewidth=2, alpha=0.8)
        ax2.plot(x_plot.squeeze(), y_learned.squeeze(),
                  label="Learned network", color="darkblue", linewidth=2)
        X_plot=X.cpu()
        Y_plot=Y.cpu()
        ax2.scatter(X_plot.squeeze(), Y_plot.squeeze(), s=10, color="gray", alpha=0.4, label="Training data")
        ax2.legend()
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title("Learned vs. initial function")
    else:
        ax2.text(
            0.5, 0.5,
            f"Function plot only supported for input_dim=output_dim=1\n"
            f"(here input_dim={input_dim}, output_dim={output_dim})",
            ha="center", va="center", wrap=True,
        )
        ax2.axis("off")

    fig2.tight_layout()
    fig2.savefig(out_dir / "learned_vs_initial.png", dpi=150)
    plt.close(fig2)

    # ── 7. Save weights and config ──────────────────────────────────────────\
    torch.save(net.state_dict(), out_dir / "model_weights.pt")
    torch.save(initial_net.state_dict(), out_dir / "initial_weights.pt")
    attention_param=target_params["rho"].get_parameters() #target_params["rho"] is a LinearAttention class object, so can't be json serialized
    target_params.pop("rho") #Therefore, we get the parameters from the attention model in a separate dict, remove the key-value pair from target param
    all_params=target_params|attention_param #and merge the dictionnaries
    #Caution, key changed in get_parameters method of LinearFeatureAttention, E->Embed, B->Bias, however script still runs bc
    # iter_target_param reads from intresting_attention.jsonl, which still have E, B, and rho as a LinearFeatureAttention object.
    # and since target_params["rho"] is a LinearFeatureAttention object, the get_parameters methods returns with the new keys, and its log in config
    config_to_save = dict(config)
    config_to_save["target_params"] = {
        k: _jsonify_param(v) for k, v in all_params.items()
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    with open(out_dir / "training_log.json", "w") as f:
        json.dump(
            dict(
            losses=losses,
            val_losses=val_losses,
            log_epochs=log_epochs,
            jacobian_rank=jac_ranks,
            pattern_rank=pattern_ranks,
            n_distinct_patterns=n_distinct_list,
            max_jacobian_rank=max_jacobian_rank,
            final_gradient_norm=grad_norm,
            critical_point_reached=critical_point_reached,
            total_epochs=total_epochs,
            min_grad_norm=min_grad_norm,
            min_normalized_grad_norm=min_normalized_grad_norm,
            epoch_reached=epoch_reached,
            max_component_at_min_grad_norm=max_coord_at_min,
            min_component_at_min_grad_norm=min_coord_at_min,
            last_val_loss=val_losses[-1],
            min_val_loss=min(val_losses),
            loss_at_min_grad_norm=loss_at_best_state,
            val_loss_at_min_grad_norm=val_loss_at_best_state,
            jac_rank_at_min_grad_norm=jac_rank_at_best_state,
            p_rank_at_min_grad_norm=p_rank_at_best_state,
            ndistinct_pattern_at_min_grad_norm=n_dist_at_best_state,
            optimizer_config=optimizer_config,
            time_taken=f"Training took {end-start} second",
            rho_formula=rho.formula,
            ),
            f,
        indent=2,
        )

    print(f"Saved all artefacts to: {out_dir}")
    return net, initial_net, out_dir



# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from target_fn_attention import make_target_fn_attention, iter_target_functions_attention
    jsonl_path ="intresting_function_attention.jsonl"

    for score, target_params,seq_len,embed_dim in iter_target_functions_attention(jsonl_path):
        for hidden_dim in [10]:
            for epoch in [1000]:
                for i in range(1):
                    target_fn = make_target_fn_attention(**target_params)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    net, initial_net, out_dir = train_network(
                        target_fn=target_fn,
                        input_dim=1,
                        seq_len=seq_len,
                        embed_dim=embed_dim,
                        hidden_dim=hidden_dim,
                        output_dim=1,
                        n_samples=200,
                        x_range=(-1.0, 1.0),
                        noise_std=0.0,
                        lr=1e-3,
                        timestamp=timestamp,
                        epochs=epoch,
                        log_every=500,
                        seed=None,
                        target_params=target_params,
                        score=score,
                        rho=LinearFeatureAttention,
                    )
                    print(f"run {i,hidden_dim,epoch} ended")
        print("all run ended")