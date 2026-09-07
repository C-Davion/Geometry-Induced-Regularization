"""
target_function.py
-------------------
Defines the ground-truth function the network should learn.

Replace `target_fn` with any function you like. Contract:
    input  : Tensor of shape (N, input_dim)
    output : Tensor of shape (N, output_dim)

The example below is a 1D -> 1D function so it can be visualised directly,
but nothing else in the pipeline assumes 1D — only the final comparison
plot special-cases input_dim == output_dim == 1.
"""

import torch
import matplotlib.pyplot as plt

def target_fn1(x: torch.Tensor) -> torch.Tensor:
    """
    Example target: y = sin(3x) + 0.5 * x

    x : Tensor (N, 1)
    returns : Tensor (N, 1)
    """
    return torch.sin(3.0 * x) + 0.5 * x

def make_target_fn(A: torch.Tensor, A1: torch.Tensor,
                    B: torch.Tensor, B1: torch.Tensor):
    """
    Build and return a target_fn(x) closing over fixed parameters
    A, A1, B, B1 — each of shape (3, 1) in this example.
 
    Example functional form (1D input -> 1D output):
        features_1 = ReLU(x @ A1.T)      # (N, 3)
        features_2 = sin(x @ B1.T)       # (N, 3)
        y = features_1 @ A + features_2 @ B     # (N, 1)
    """
    def target_fn(x)-> torch.Tensor:
        def rho(x):
            # x: (N, 1)
            return torch.cat((torch.ones_like(x),x,x**2), dim=1)   # (N, 3)
        X=rho(x)
        return torch.maximum(torch.minimum(X@A,X@A1),torch.minimum(X@B,X@B1))

 
    return target_fn



if __name__ == "__main__":
    x = torch.linspace(-2, 2, 100).unsqueeze(1)  # (100,1)
    A = torch.Tensor([1, 2, -1]).reshape([3,1])
    A1 =torch.Tensor([0, 1, -1]).reshape([3,1])
    B = torch.Tensor([0.5, -1, 0.5]).reshape([3,1])
    B1 =torch.Tensor([1, 3, -0.5]).reshape([3,1])
    fn=make_target_fn(A,A1,B,B1)
    y=fn(x)
    plt.plot(x.cpu().numpy(),y.cpu().numpy())
    plt.show()