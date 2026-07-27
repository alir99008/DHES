import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rms_norm_fn(x: torch.Tensor, scale: torch.Tensor,
                 eps: float = 1e-6) -> torch.Tensor:
    """RMS normalisation with a learnable per-dimension scale — Eq. (3)."""
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * rms * scale


class DHESAdapter(nn.Module):
    """
    Wraps one frozen nn.Linear and injects the DHES residual correction.

        z_t   = A x_t                                   # Eq. (1)
        z~_t  = G(theta) z_t                            # Eq. (2)  Givens rotation
        z^_t  = RMSNorm_gamma(z~_t)                     # Eq. (3)
        u_t   = z^_t (I + Phi)                          # Eq. (4)  rank mixing
        s_t   = SiLU(u_t)                               # Eq. (5)  per-token gate
        zbar  = mean_t(z_t);  tau = clamp(exp(l),.05,5) # Eq. (6)
        g     = sigmoid((w * zbar) / tau)               # Eq. (7)  context gate
        h_t   = W x_t + (alpha/r) B (g * s_t)           # Eq. (8)

    B is zero-initialised, so the adapter is an exact identity at step 0.
    No state is carried across forward calls (see Section 3.3.2).
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 16,
                 alpha: float = 16.0, norm_eps: float = 1e-6):
        super().__init__()
        assert rank % 2 == 0, "rank must be even for pairwise Givens rotation"
        self.base_layer = base_layer
        for p in self.base_layer.parameters():      # freeze the pretrained weight
            p.requires_grad = False

        d_in  = base_layer.in_features
        d_out = base_layer.out_features
        self.rank        = rank
        self.lora_scale  = alpha / rank
        self._norm_eps   = norm_eps

        # --- low-rank projections -------------------------------------------
        self.A = nn.Linear(d_in, rank, bias=False)
        self.B = nn.Linear(rank, d_out, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)               # identity at initialisation

        # --- rank-space components ------------------------------------------
        self.theta        = nn.Parameter(torch.zeros(rank // 2))   # rotation
        self.norm_scale   = nn.Parameter(torch.ones(rank))         # gamma
        self.Phi          = nn.Parameter(torch.zeros(rank, rank))  # mixing
        self.W_gate       = nn.Parameter(torch.randn(rank) * 0.02) # gate weights
        self.log_gate_tau = nn.Parameter(                          # log temperature
            torch.full((rank,), math.log(0.25)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)               # frozen W x
        Ax       = self.A(x)                        # Eq. (1)  (B, T, r)

        # Eq. (2) — exact orthogonal rotation over adjacent rank pairs
        cos_t, sin_t = torch.cos(self.theta), torch.sin(self.theta)
        x_even, x_odd = Ax[..., 0::2], Ax[..., 1::2]
        rot = torch.stack([cos_t * x_even - sin_t * x_odd,
                           sin_t * x_even + cos_t * x_odd],
                          dim=-1).flatten(-2)

        # Eqs. (3)-(4) — RMSNorm then residual rank mixing
        normed   = _rms_norm_fn(rot, self.norm_scale, self._norm_eps)
        combined = normed + normed @ self.Phi

        # Eq. (5) — per-token SiLU gate
        gate_seq = F.silu(combined)

        # Eqs. (6)-(7) — per-sequence context gate with learnable temperature
        gate_tau   = torch.exp(self.log_gate_tau).clamp(min=0.05, max=5.0)
        gate_logit = self.W_gate * Ax.mean(dim=1, keepdim=True)   # (B, 1, r)
        gate_ctx   = torch.sigmoid(gate_logit / gate_tau)

        # Eq. (8)
        delta = self.B(gate_ctx * gate_seq) * self.lora_scale
        return base_out + delta


def inject_dhes(model: nn.Module, cfg: dict) -> nn.Module:
    """Replace every targeted nn.Linear with a DHESAdapter wrapping it."""
    targets = set(cfg["dhes_targets"])              # e.g. {"q_proj", "v_proj"}
    for full_name, module in list(model.named_modules()):
        leaf = full_name.split(".")[-1]
        if leaf in targets and isinstance(module, nn.Linear):
            parent_path = ".".join(full_name.split(".")[:-1])
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, leaf,
                    DHESAdapter(module,
                                rank=cfg["dhes_rank"],
                                alpha=cfg["dhes_alpha"]))
    return model