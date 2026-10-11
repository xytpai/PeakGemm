from pathlib import Path

import torch


torch.ops.load_library(str(Path(__file__).with_name("libPeakGemm.so")))

# Split-K workspaces are reused only on the same device and stream.
_workspaces = {}


def hgemm(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    split_k: int = 1,
    bias: torch.Tensor | None = None,
    *,
    config: int = 3,
) -> torch.Tensor:
    """Write A @ B.T + bias into C using a precompiled configuration (0..3).

    On ROCm, config=3 uses HTI: M/N multiples of 256, K a multiple of 128,
    and split_k=1. Configs 0..2 also support split-K.
    """
    semaphore = signal = None
    if split_k > 1 and a.is_cuda:
        stream = torch.cuda.current_stream(a.device)
        key = (a.device.index, stream.cuda_stream)
        if key not in _workspaces:
            _workspaces[key] = tuple(
                torch.zeros(256, dtype=torch.uint32, device=a.device)
                for _ in range(2)
            )
        semaphore, signal = _workspaces[key]
    torch.ops.PeakGemm.hgemm(c, a, b, split_k, bias, config, semaphore, signal)
    return c


# Register differentiable operators after loading the shared library.
from . import functional
from .functional import rms_norm
from .nn import Linear, RMSNorm

__all__ = ["hgemm", "rms_norm", "functional", "RMSNorm", "Linear"]
