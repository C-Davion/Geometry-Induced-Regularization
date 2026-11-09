'''
Plots the teacher function (true Y) by reading the .jsonl file generated for the
attention-based teacher function (LinearFeatureAttention used as the feature map rho).

Requires prior knowledge of the feature transformation (rho) used to construct the
teacher function, as well as the general shape of the function.

Expected jsonl record shape (one line per teacher function):
{
    "score": <float>,
    "rho": {
        "Updim": [[...]],           # (seq_len, 1)
        "E":     [[...], ...],      # (seq_len, embed_dim)
        "B":     [[...], ...],      # (seq_len, embed_dim)
        "attention": {
            "in_proj_weight":  [[...], ...],   # (3*embed_dim, embed_dim)
            "in_proj_bias":    [...],          # (3*embed_dim,)
            "out_proj_weight": [[...], ...],   # (embed_dim, embed_dim)
            "out_proj_bias":   [...],          # (embed_dim,)
        },
    },
    "A":  [...],   # (seq_len*embed_dim,)
    "A1": [...],   # (seq_len*embed_dim,)
    "B":  [...],   # (seq_len*embed_dim,)
    "B1": [...],   # (seq_len*embed_dim,)
}
'''

import json

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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
    return model


def f(A, A1, B, B1, X):
    Ax = X @ A
    A1x = X @ A1
    Bx = X @ B
    B1x = X @ B1

    mA = torch.minimum(Ax, A1x)
    mB = torch.minimum(Bx, B1x)

    y = torch.maximum(mA, mB)

    return y, Ax, A1x, Bx, B1x


if __name__ == "__main__":
    x_attention = torch.linspace(-1, 1, 800)
    x_input = x_attention.reshape(-1, 1)

    with open(LOG_FILE) as file:
        for line in file:
            r = json.loads(line)

            rho = build_rho(r["rho"])

            with torch.no_grad():
                X = rho(x_input)
                X = torch.reshape(X, (X.shape[0], -1))

                A = torch.tensor(r["A"], dtype=torch.float32)
                A1 = torch.tensor(r["A1"], dtype=torch.float32)
                B = torch.tensor(r["B"], dtype=torch.float32)
                B1 = torch.tensor(r["B1"], dtype=torch.float32)

                y, Ax, A1x, Bx, B1x = f(A, A1, B, B1, X)

            x = x_attention.numpy()
            y_np = y.numpy()
            Ax_np = Ax.numpy()
            A1x_np = A1x.numpy()
            Bx_np = Bx.numpy()
            B1x_np = B1x.numpy()

            use_A = np.minimum(Ax_np, A1x_np) >= np.minimum(Bx_np, B1x_np)

            labels = np.empty(len(x), dtype=int)
            labels[use_A & (Ax_np <= A1x_np)] = 0
            labels[use_A & (Ax_np > A1x_np)] = 1
            labels[~use_A & (Bx_np <= B1x_np)] = 2
            labels[~use_A & (Bx_np > B1x_np)] = 3

            plt.figure(figsize=(7, 4))
            plt.plot(x, np.where(labels == 0, y_np, np.nan), lw=3, label="M")
            plt.plot(x, np.where(labels == 1, y_np, np.nan), lw=3, label=r"$M_1$")
            plt.plot(x, np.where(labels == 2, y_np, np.nan), lw=3, label="N")
            plt.plot(x, np.where(labels == 3, y_np, np.nan), lw=3, label=r"$N_1$")

            title = "teacher function"
            if "score" in r:
                title += f" (score={r['score']:.3f})"
            plt.title(title)
            plt.legend()
            plt.show()

    print("All ploted")
