import torch
from torch import nn

from . import functional


class RMSNorm(nn.Module):
    """Drop-in replacement for a last-dimension, FP32-accumulating RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6, *, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.rms_norm(x, self.weight, self.eps)



class Linear(nn.Linear):
    """nn.Linear-compatible parameters; HGEMM forward where supported."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.linear(x, self.weight, self.bias)
