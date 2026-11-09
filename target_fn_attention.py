'''
Builds the teacher (target) function via a `make_target_fn` closure factory,
using rho = LinearFeatureAttention (the same attention mechanism as in
ploteacher_attention.py) instead of the fixed polynomial rho, then plots it
the same way ploteacher_attention.py does.

Reads from the same .jsonl format as ploteacher_attention.py:
{
    "score": <float>,
    "rho": {
        "Updim": [[...]],           # (seq_len, 1)
        "E":     [[...], ...],      # (seq_len, embed_dim)
        "B":     [[...], ...],      # (seq_len, embed_dim)
        "attention": {
            "in_proj_weight":  [[...], ...],
            "in_proj_bias":    [...],
            "out_proj_weight": [[...], ...],
            "out_proj_bias":   [...],
        },
    },
    "A": [...], "A1": [...], "B": [...], "B1": [...],   # (seq_len*embed_dim,)
}
'''

import json

import numpy as np
import matplotlib.pyplot as plt
import torch
from relu_network import LinearFeatureAttention


LOG_FILE = "intresting_function_attention.jsonl"





def build_rho(rho_params: dict) -> LinearFeatureAttention:
    """
    Reconstruct the LinearFeatureAttention module used as rho for one
    teacher function, loading its trained weights from the jsonl record.
    """
    E = np.array(rho_params["E"])
    B = np.array(rho_params["B"])

    seq_len = E.shape[0]
    embed_dim = E.shape[1]

    model = LinearFeatureAttention(seq_len=seq_len, embed_dim=embed_dim)

    att = rho_params["attention"]
    with torch.no_grad():
        model.E.copy_(torch.tensor(E, dtype=torch.float32))
        model.B.copy_(torch.tensor(B, dtype=torch.float32))

        model.attention.in_proj_weight.copy_(
            torch.tensor(att["in_proj_weight"], dtype=torch.float32)
        )
        model.attention.in_proj_bias.copy_(
            torch.tensor(att["in_proj_bias"], dtype=torch.float32)
        )
        model.attention.out_proj.weight.copy_(
            torch.tensor(att["out_proj_weight"], dtype=torch.float32)
        )
        model.attention.out_proj.bias.copy_(
            torch.tensor(att["out_proj_bias"], dtype=torch.float32)
        )

    model.eval()
    return model,seq_len, embed_dim
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
    keys "A", "A1", "B", "B1", "rho" — matching make_target_fn_attention's parameter
    names exactly, so it can be unpacked straight into it:
 
        for score, target_params in iter_target_functions(path):
            target_fn = make_target_fn_attention(**target_params)
    """
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
 
            obj = json.loads(line)
            rho,seq_len,embed_dim=build_rho(obj["rho"])
 
            target_params = {
                "A": torch.tensor(obj["A"], dtype=torch.float32).reshape(-1, 1),
                "A1": torch.tensor(obj["A1"], dtype=torch.float32).reshape(-1, 1),
                "B": torch.tensor(obj["B"], dtype=torch.float32).reshape(-1, 1),
                "B1": torch.tensor(obj["B1"], dtype=torch.float32).reshape(-1, 1),
                "rho": rho,
            }
 
            yield obj["score"], target_params,seq_len,embed_dim
            break  # do just


def make_target_fn_attention(A: torch.Tensor, A1: torch.Tensor,
                    B: torch.Tensor, B1: torch.Tensor,
                    rho: LinearFeatureAttention):
    """
    Build and return a target_fn(x) closing over fixed parameters
    A, A1, B, B1 — each of shape (rho.output_dim,) — and a fitted
    rho = LinearFeatureAttention feature map (built via build_rho).

    Functional form (1D input -> 1D output):
        X = rho(x).reshape(N, -1)              # (N, rho.output_dim)
        branch_A = min(X @ A,  X @ A1)          # (N,)
        branch_B = min(X @ B,  X @ B1)          # (N,)
        y = max(branch_A, branch_B)             # (N,)

    Unlike the polynomial version's rho (a fixed, hand-written [1, x, x^2]
    map), rho here is the trained LinearFeatureAttention module
    (x -> Ax -> per-feature embeddings -> self-attention), so it can't be
    written inline: it must be reconstructed from the jsonl record via
    build_rho() and passed in already built.
    """
    def target_fn(x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1)
        with torch.no_grad():
            X = rho(x)                              # (N, seq_len, embed_dim)
            X = torch.reshape(X, (X.shape[0], -1))   # (N, seq_len*embed_dim)
            y = torch.maximum(
                torch.minimum(X @ A, X @ A1),
                torch.minimum(X @ B, X @ B1),
            )
        return y

    return target_fn


if __name__ == "__main__":
    x_attention = torch.linspace(-1, 1, 800)
    x_input = x_attention.reshape(-1, 1)

    with open(LOG_FILE) as file:
        for line in file:
            r = json.loads(line)

            rho = build_rho(r["rho"])
            A = torch.tensor(r["A"], dtype=torch.float32)
            A1 = torch.tensor(r["A1"], dtype=torch.float32)
            B = torch.tensor(r["B"], dtype=torch.float32)
            B1 = torch.tensor(r["B1"], dtype=torch.float32)

            target_fn = make_target_fn_attention(A, A1, B, B1, rho)
            y = target_fn(x_input)

            # target_fn only returns y (per the requested signature), so
            # recompute the four branch values here just to color the plot
            # by which branch (A / A1 / B / B1) is active, same as
            # ploteacher_attention.py.
            with torch.no_grad():
                X = rho(x_input)
                X = torch.reshape(X, (X.shape[0], -1))
                Ax, A1x, Bx, B1x = X @ A, X @ A1, X @ B, X @ B1

            x = x_attention.numpy()
            y_np = y.numpy()
            Ax_np, A1x_np, Bx_np, B1x_np = (
                t.numpy() for t in (Ax, A1x, Bx, B1x)
            )

            use_A = np.minimum(Ax_np, A1x_np) >= np.minimum(Bx_np, B1x_np)

            labels = np.empty(len(x), dtype=int)
            labels[use_A & (Ax_np <= A1x_np)] = 0
            labels[use_A & (Ax_np > A1x_np)] = 1
            labels[~use_A & (Bx_np <= B1x_np)] = 2
            labels[~use_A & (Bx_np > B1x_np)] = 3

            plt.figure(figsize=(7, 4))
            plt.plot(x, np.where(labels == 0, y_np, np.nan), lw=3, label="A")
            plt.plot(x, np.where(labels == 1, y_np, np.nan), lw=3, label="A1")
            plt.plot(x, np.where(labels == 2, y_np, np.nan), lw=3, label="B")
            plt.plot(x, np.where(labels == 3, y_np, np.nan), lw=3, label="B1")

            title = "teacher_function (target_fn)"
            if "score" in r:
                title += f" (score={r['score']:.3f})"
            plt.title(title)
            plt.legend()
            plt.show()

    print("All ploted")
