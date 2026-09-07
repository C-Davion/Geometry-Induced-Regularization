import torch
import torch.nn as nn
from torch.func import functional_call, jacrev
from typing import Optional


# ── Default identity feature map ──────────────────────────────────────────────

class IdentityFeatureMap(nn.Module):
    """
    Trivial rho(x) = x.  Used as the default when no feature map is supplied.
    Has no trainable parameters; output_dim equals input_dim.
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = input_dim   # d = n0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


# ── Network ───────────────────────────────────────────────────────────────────

class OneHiddenLayerReLUNet(nn.Module):
    """
    A one-hidden-layer ReLU network with an optional feature transformation rho:

        f(x) = W2 @ ReLU(W1 @ rho(x) + b1) + b2

    rho is an nn.Module that maps R^{input_dim} → R^{feature_dim}.
    Its parameters (if any) are registered as part of the network and
    are included in the Jacobian computation.

    When rho is omitted, the identity map is used and the network reduces to
    the original  f(x) = W2 @ ReLU(W1 @ x + b1) + b2.

    Parameters
    ----------
    W1  : Tensor of shape (hidden_dim, feature_dim)
          feature_dim must equal rho.output_dim (or input_dim if rho is None).
    b1  : Tensor of shape (hidden_dim,)
    W2  : Tensor of shape (output_dim, hidden_dim)
    b2  : Tensor of shape (output_dim,)
    rho : nn.Module with attribute output_dim, or None for the identity map.
          Must accept inputs of shape (..., input_dim) and return
          (..., feature_dim).
    """

    def __init__(
        self,
        W1:  torch.Tensor,
        b1:  torch.Tensor,
        W2:  torch.Tensor,
        b2:  torch.Tensor,
        rho: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        hidden_dim, feature_dim = W1.shape
        output_dim, hidden_dim2 = W2.shape

        if b1.shape != (hidden_dim,):
            raise ValueError(
                f"b1 must have shape ({hidden_dim},), got {tuple(b1.shape)}"
            )
        if hidden_dim2 != hidden_dim:
            raise ValueError(
                f"W2 hidden dim ({hidden_dim2}) must match W1 hidden dim ({hidden_dim})"
            )
        if b2.shape != (output_dim,):
            raise ValueError(
                f"b2 must have shape ({output_dim},), got {tuple(b2.shape)}"
            )

        # Resolve feature map
        if rho is None:
            rho = IdentityFeatureMap(feature_dim)

        # Verify that rho's declared output dimension matches W1's input dimension
        if not hasattr(rho, "output_dim"):
            raise ValueError(
                "rho must expose an `output_dim` attribute so we can verify "
                "it is compatible with W1."
            )
        if rho.output_dim != feature_dim:
            raise ValueError(
                f"rho.output_dim ({rho.output_dim}) must match "
                f"W1's feature_dim ({feature_dim})."
            )

        # Store dimensions
        self.input_dim   = rho.input_dim if hasattr(rho, "input_dim") else None
        self.feature_dim = feature_dim
        self.hidden_dim  = hidden_dim
        self.output_dim  = output_dim

        # Learnable parameters of the linear layers
        self.W1 = nn.Parameter(W1.float())
        self.b1 = nn.Parameter(b1.float())
        self.W2 = nn.Parameter(W2.float())
        self.b2 = nn.Parameter(b2.float())

        # Feature map (may itself contain learnable parameters)
        self.rho = rho

        self.relu = nn.ReLU()

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rho to x.  Shape: (..., feature_dim)."""
        return self.rho(x)

    def _pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pre-activation values of the hidden layer: W1 @ rho(x) + b1.
        Shape: (..., hidden_dim).
        """
        return self._features(x) @ self.W1.T + self.b1

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute f(x) = W2 @ ReLU(W1 @ rho(x) + b1) + b2.

        Accepts a single input (..., input_dim) or a batch (batch_size, input_dim).
        Returns a tensor of shape (..., output_dim).
        """
        h = self.relu(self._pre_activations(x))   # (..., hidden_dim)
        return h @ self.W2.T + self.b2             # (..., output_dim)

    # ------------------------------------------------------------------
    # Activation pattern
    # ------------------------------------------------------------------

    def activation_pattern(self, x: torch.Tensor) -> torch.Tensor:
        """
        Binary activation pattern of the hidden layer given input x.

        Returns a float tensor of shape (..., hidden_dim):
          1  if the neuron's pre-activation > 0 (ReLU open),
          0  otherwise (ReLU clamped).

        Works for single inputs and batches alike.
        """
        pre = self._pre_activations(x)
        return (pre > 0).float()

    # ------------------------------------------------------------------
    # Jacobian w.r.t. ALL parameters (W1, b1, W2, b2, and rho's params)
    # ------------------------------------------------------------------

    def parameter_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Jacobian of f w.r.t. all trainable parameters θ at a single input x.

        θ includes W1, b1, W2, b2, and any parameters inside rho.

        Returns
        -------
        J : Tensor of shape (output_dim, P)
            P = total number of scalar parameters across W1, b1, W2, b2, rho.
        """
        if x.dim() != 1:
            raise ValueError(
                f"parameter_jacobian expects a single 1-D input, got shape {tuple(x.shape)}"
            )

        params = dict(self.named_parameters())   # includes rho.* if rho has params

        def f_of_params(p: dict) -> torch.Tensor:
            return functional_call(self, p, x)   # (output_dim,)

        jac_dict = jacrev(f_of_params)(params)

        J = torch.cat(
            [j.flatten(start_dim=1) for j in jac_dict.values()],
            dim=1,
        )
        return J   # (output_dim, P)
    def parameter_jacobian_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: (N, input_dim)

        Returns:
            J: (N * output_dim, P)
        """
        if X.dim() != 2 or X.shape[1] != self.input_dim:
            raise ValueError("X must have shape (N, input_dim)")

        params = dict(self.named_parameters())

        def f_of_params(p):
            return functional_call(self, p, X)   # (N, output_dim)

        jac_dict = jacrev(f_of_params)(params)

        J = torch.cat(
            [
                j.reshape(-1, p.numel())
                for j, p in zip(jac_dict.values(), params.values())
            ],
            dim=1,
        )

        return J

    def jacobian_rank(self, x: torch.Tensor, tol: Optional[float] = None) -> int:
        """
        Rank of the parameter Jacobian at a single input x.
        """
        J = self.parameter_jacobian(x)
        kwargs = {} if tol is None else {"atol": tol, "rtol": tol}
        return int(torch.linalg.matrix_rank(J, **kwargs))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        rho_name = type(self.rho).__name__
        return (
            f"OneHiddenLayerReLUNet("
            f"input_dim={self.input_dim}, "
            f"feature_dim={self.feature_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"output_dim={self.output_dim}, "
            f"rho={rho_name})"
        )


# ── Example feature maps ──────────────────────────────────────────────────────

class LinearFeatureMap(nn.Module):
    """
    Trainable linear feature map: rho(x) = A @ x + c
    Maps R^{input_dim} → R^{feature_dim}.
    """
    def __init__(self, input_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = feature_dim
        self.A = nn.Parameter(torch.randn(feature_dim, input_dim) * 0.1)
        self.c = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.A.T + self.c


class PolynomialFeatureMap(nn.Module):
    """
    Fixed (non-trainable) degree-2 polynomial feature map.
    For input x ∈ R^{n}, produces [x, x^2] ∈ R^{2n}.
    No trainable parameters.
    """
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = 3 * input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.ones_like(x),x, x ** 2], dim=-1)

class QuadraticFeatureMap(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim         # was 3 * input_dim
    def forward(self, x):
        return torch.cat([x, x**2], dim=-1)     # (N, 2) — no constant column
    formula="x->[x,x**2]"


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    input_dim  = 4
    hidden_dim = 6
    output_dim = 2

    # ── 1. No rho (identity) — original behaviour ─────────────────────
    print("=" * 60)
    print("1. Identity feature map (rho = None)")
    W1 = torch.randn(hidden_dim, input_dim)
    b1 = torch.zeros(hidden_dim)
    W2 = torch.randn(output_dim, hidden_dim)
    b2 = torch.zeros(output_dim)

    net_id = OneHiddenLayerReLUNet(W1, b1, W2, b2)
    print(net_id)

    x = torch.randn(input_dim)
    print(f"  output : {net_id(x)}")
    print(f"  act.pat: {net_id.activation_pattern(x)}")
    print(f"  J shape: {tuple(net_id.parameter_jacobian(x).shape)}")
    print(f"  J rank : {net_id.jacobian_rank(x)}")

    # ── 2. Trainable linear feature map ───────────────────────────────
    print("\n" + "=" * 60)
    feature_dim = 8
    print(f"2. LinearFeatureMap  ({input_dim} → {feature_dim})")
    rho_lin = LinearFeatureMap(input_dim, feature_dim)
    W1_lin  = torch.randn(hidden_dim, feature_dim)
    net_lin = OneHiddenLayerReLUNet(W1_lin, b1, W2, b2, rho=rho_lin)
    print(net_lin)

    print(f"  output : {net_lin(x)}")
    print(f"  act.pat: {net_lin.activation_pattern(x)}")
    J_lin = net_lin.parameter_jacobian(x)
    P_lin = sum(p.numel() for p in net_lin.parameters())
    print(f"  total params P : {P_lin}")
    print(f"  J shape        : {tuple(J_lin.shape)}")
    print(f"  J rank         : {net_lin.jacobian_rank(x)}")

    # ── 3. Fixed polynomial feature map ───────────────────────────────
    print("\n" + "=" * 60)
    print(f"3. PolynomialFeatureMap  ({input_dim} → {2*input_dim}, no trainable params)")
    rho_poly = PolynomialFeatureMap(input_dim)
    W1_poly  = torch.randn(hidden_dim, rho_poly.output_dim)
    net_poly = OneHiddenLayerReLUNet(W1_poly, b1, W2, b2, rho=rho_poly)
    print(net_poly)

    print(f"  output : {net_poly(x)}")
    print(f"  act.pat: {net_poly.activation_pattern(x)}")
    J_poly = net_poly.parameter_jacobian(x)
    P_poly = sum(p.numel() for p in net_poly.parameters())
    print(f"  total params P : {P_poly}")
    print(f"  J shape        : {tuple(J_poly.shape)}")
    print(f"  J rank         : {net_poly.jacobian_rank(x)}")

    # ── 4. Batch forward & activation pattern ─────────────────────────
    print("\n" + "=" * 60)
    print("4. Batch forward (all three nets)")
    X_batch = torch.randn(5, input_dim)
    for name, net in [("identity", net_id), ("linear", net_lin), ("poly", net_poly)]:
        y  = net(X_batch)
        ap = net.activation_pattern(X_batch)
        print(f"  {name:8s}  output {tuple(y.shape)}  act.pat {tuple(ap.shape)}")