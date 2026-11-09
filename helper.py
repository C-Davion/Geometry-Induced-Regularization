'''
Some function used to diagnosis the training,  put them here to not clog the train file.
'''
import json
from copy import deepcopy
from pathlib import Path
import glob
from typing import Callable, Optional, Tuple, Union
from datetime import datetime
import torch
import torch.nn as nn
from torch.func import functional_call, jacrev
import matplotlib.pyplot as plt
import time
from relu_network import OneHiddenLayerReLUNet, QuadraticFeatureMap
from data_generation import generate_dataset,train_val_split
#torch.set_default_dtype(torch.float64)
def _jsonify_param(value):
    """Turn a target-function parameter into a JSON-serializable value."""
    if torch.is_tensor(value):
        return value.tolist()
    return value


# ── Structural-quantity helpers (batch versions) ────────────────────────────


def stable_rank(M: torch.Tensor) -> float:
    """
    Stable rank  ||M||_F^2 / ||M||_2^2 = sum_i sigma_i^2 / sigma_max^2.
    Real-valued, in [1, rank(M)] (0 for the zero matrix), needs no tolerance.
    """
    s = torch.linalg.svdvals(M)
    if s.numel() == 0 or s[0] == 0:
        return 0.0
    return float((s ** 2).sum() / s[0] ** 2)


def stacked_jacobian(net, X: torch.Tensor) -> torch.Tensor:
    """Jacobian of θ -> [f(θ,x1); ...; f(θ,xN)] w.r.t. all parameters, shape (N*output_dim, P)."""
    params = dict(net.named_parameters())

    def f_batch(p: dict) -> torch.Tensor:
        out = functional_call(net, p, X)   # (N, output_dim)
        return out.flatten()               # (N*output_dim,)

    jac_dict = jacrev(f_batch)(params)
    return torch.cat([j.flatten(start_dim=1) for j in jac_dict.values()], dim=1)


def stacked_jacobian_rank(net, X: torch.Tensor, rtol=None,
                          stable: bool = False) -> Union[int, float]:
    """rtol=None -> torch's default max(m,n)*eps(dtype). stable=True -> stable rank (float)."""
    J = stacked_jacobian(net, X)
    if stable:
        return stable_rank(J)
    return int(torch.linalg.matrix_rank(J, rtol=rtol))


def pattern_stats(net, X: torch.Tensor, rtol=None,
                  stable: bool = False) -> Tuple[Union[int, float], int]:
    """Return (rank of pattern matrix, number of distinct pattern rows)."""
    with torch.no_grad():
        pat = net.activation_pattern(X)          # (N, hidden_dim)
    P = pat.to(X.dtype)                          # was .float(); identical for float32 X
    rank = stable_rank(P) if stable else int(torch.linalg.matrix_rank(P, rtol=rtol))
    n_distinct = len(set(tuple(row.tolist()) for row in pat))
    return rank, n_distinct
def adam_step_norm(optimizer):
    """
    Norm of the update Adam would actually apply on its NEXT .step() call,
    using its current momentum (m) and variance (v) estimates — as opposed
    to the raw instantaneous gradient norm.
    """
    total_sq = 0.0
    for group in optimizer.param_groups:
        lr, eps = group['lr'], group['eps']
        beta1, beta2 = group['betas']
        for p in group['params']:
            state = optimizer.state.get(p)
            if not state or 'exp_avg' not in state:
                continue
            t = state['step']
            t = t.item() if torch.is_tensor(t) else t
            if t == 0:
                continue
            m_hat = state['exp_avg'] / (1 - beta1 ** t)
            v_hat = state['exp_avg_sq'] / (1 - beta2 ** t)
            update = lr * m_hat / (v_hat.sqrt() + eps)
            total_sq += update.pow(2).sum().item()
    return total_sq ** 0.5

def hidden_unit_roots(net, x_range):
    """
    For each hidden unit i, with z_i(x) = a_i*x^2 + b_i*x + c_i
    (a_i = W1[i,1], b_i = W1[i,0], c_i = b1[i]),
    return real root(s) in x where z_i(x) = 0 — the neuron's ReLU boundary.
    Only roots inside x_range are kept, since only those affect training data.
    """
    low, high = x_range
    with torch.no_grad():
        a = net.W1[:, 1].detach().cpu()
        b = net.W1[:, 0].detach().cpu()
        c = net.b1.detach().cpu()

    roots_per_unit = []
    for i in range(net.hidden_dim):
        ai, bi, ci = a[i].item(), b[i].item(), c[i].item()
        roots = []
        if abs(ai) < 1e-10:
            if abs(bi) > 1e-10:
                roots = [-ci / bi]
        else:
            disc = bi**2 - 4 * ai * ci
            if disc >= 0:
                s = disc ** 0.5
                roots = sorted([(-bi + s) / (2 * ai), (-bi - s) / (2 * ai)])
        roots_per_unit.append([r for r in roots if low <= r <= high])
    return roots_per_unit
def find_kinks(target_fn, x_range=(-3.0, 3.0), n_points=20000, rel_thresh=0.05):
    """
    Numerically detect kink locations of a scalar target_fn(x) over x_range,
    by looking for jumps in the discrete first derivative (slope). Works for
    any callable target_fn: (N,1) -> (N,1), regardless of how it's built
    internally (min/max composition, splines, etc.) — no access to its
    internal parameters is needed.
    """
    x = torch.linspace(x_range[0], x_range[1], n_points, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        y = target_fn(x).squeeze(-1).double()

    dx = (x[1, 0] - x[0, 0]).item()
    slopes = (y[1:] - y[:-1]) / dx                      # discrete first derivative
    slope_jumps = (slopes[1:] - slopes[:-1]).abs()       # change in slope, point to point

    # a true kink is one grid point with a much bigger slope-jump than its
    # neighbours (which just reflect smooth within-piece curvature)
    thresh = rel_thresh * slope_jumps.max().item()
    idx = torch.nonzero(slope_jumps > thresh).flatten()

    # merge adjacent grid indices that belong to the same kink
    kink_locations = []
    if len(idx) > 0:
        groups = [[idx[0].item()]]
        for i in idx[1:].tolist():
            (groups[-1].append(i) if i - groups[-1][-1] <= 2 else groups.append([i]))
        for g in groups:
            center = int(sum(g) / len(g)) + 1
            kink_locations.append(round(x[center, 0].item(), 4))

    return kink_locations
def iter_target_functions(jsonl_path):
    """
    Iterate over a JSONL file whose lines have the form

    {
        "score": ...,
        "A": [...],
        "A1": [...],
        "B": [...],
        "B1": [...]
    }
    """
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            target_params = {
                "A": torch.tensor(obj["A"], dtype=torch.float32).reshape(-1, 1),
                "A1": torch.tensor(obj["A1"], dtype=torch.float32).reshape(-1, 1),
                "B": torch.tensor(obj["B"], dtype=torch.float32).reshape(-1, 1),
                "B1": torch.tensor(obj["B1"], dtype=torch.float32).reshape(-1, 1),
            }

            yield obj["score"], target_params
            break #do just one
def iter_target_functions_attention(jsonl_path):
    """
    Iterate over a JSONL file whose lines have the form
 
    {
        "score": ...,
        "rho": {...},   # LinearFeatureAttention weights, see build_rho()
        "A": [...],
        "A1": [...],
        "B": [...],
        "B1": [...]
    }
 
    Yields (score, target_params) pairs, where target_params is a dict with
    keys "A", "A1", "B", "B1", "rho" — matching make_target_fn's parameter
    names exactly, so it can be unpacked straight into it:
 
        for score, target_params in iter_target_functions(path):
            target_fn = make_target_fn(**target_params)
    """
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
 
            obj = json.loads(line)
 
            target_params = {
                "A": torch.tensor(obj["A"], dtype=torch.float32).reshape(-1, 1),
                "A1": torch.tensor(obj["A1"], dtype=torch.float32).reshape(-1, 1),
                "B": torch.tensor(obj["B"], dtype=torch.float32).reshape(-1, 1),
                "B1": torch.tensor(obj["B1"], dtype=torch.float32).reshape(-1, 1),
                "rho": build_rho(obj["rho"]),
            }
 
            yield obj["score"], target_params
            break  # do just one
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Orthogonalize a 2D matrix's update via a quintic Newton-Schulz iteration."""
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.double()
    X = X / (X.norm() + eps)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum = group['lr'], group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if group['weight_decay'] != 0:
                    g = g.add(p, alpha=group['weight_decay'])

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if group['nesterov'] else buf

                assert g.ndim == 2, "Muon is only defined for 2D matrix parameters"
                g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                g = g * max(g.size(0), g.size(1)) ** 0.5   # scale to match typical update size

                p.add_(g, alpha=-lr)
def maybe_warm_restart(opt_muon, scheduler_weight, opt_bias, scheduler_bias,
                        total_epochs, epoch_reached, last_restart_epoch,
                        patience_for_restart=5000, restart_factor=0.1):
    epochs_since_improvement = total_epochs - epoch_reached
    epochs_since_restart = total_epochs - last_restart_epoch

    if epochs_since_improvement > patience_for_restart and epochs_since_restart > patience_for_restart:
        for g in opt_muon.param_groups:
            g['lr'] = opt_muon.defaults['lr'] * restart_factor
        for g in opt_bias.param_groups:
            g['lr'] = opt_bias.defaults['lr'] * restart_factor   # reset from its OWN base, not multiply current

        scheduler_weight.cooldown_counter = 0
        scheduler_weight.num_bad_epochs = 0
        scheduler_bias.cooldown_counter = 0
        scheduler_bias.num_bad_epochs = 0
        return True, total_epochs
    return False, last_restart_epoch