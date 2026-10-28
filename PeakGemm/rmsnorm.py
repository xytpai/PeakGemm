"""ModelX-compatible RMSNorm, using the same C++ kernel on CUDA and HIP."""
from __future__ import annotations

import functools
import hashlib
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.autograd.function import once_differentiable

_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@functools.lru_cache(maxsize=None)
def _extension(arch: str):
    from torch.utils.cpp_extension import load, _get_build_directory

    root = Path(__file__).resolve().parent
    include = root.parent / "include"
    if not (include / "peak_gemm/kernel/rmsnorm.hpp").exists():
        include = root / "include"  # Headers shipped in wheels.
    source = root / "csrc/rmsnorm.cu"
    digest = hashlib.sha256()
    for path in [Path(__file__), source, include / "peak_gemm/kernel/rmsnorm.hpp",
                 include / "peak_gemm/backend/runtime.hpp", include / "peak_gemm/core/config.hpp"]:
        digest.update(path.read_bytes())
    digest.update(f"{torch.__version__}:{torch.version.hip}:{arch}".encode())
    flags = ["-O3", "-std=c++17"]
    if torch.version.hip:
        # Existing runtime.hpp requires ROCM_TARGET for its wave-size abstraction.
        flags += [f"-DROCM_TARGET={arch.removeprefix('gfx')}",
                  f"--offload-arch={arch}", "-ffp-contract=off",
                  "-U__HIP_NO_HALF_OPERATORS__", "-U__HIP_NO_HALF_CONVERSIONS__"]
    else:
        capability = arch.removeprefix("sm")
        flags += [f"-gencode=arch=compute_{capability},code=sm_{capability}",
                  "--expt-relaxed-constexpr", "--fmad=false",
                  "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                  "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"]
    name = f"peak_gemm_rmsnorm_{digest.hexdigest()[:16]}"
    build_dir = Path(_get_build_directory(name, verbose=False))
    # hipify writes sibling *_hip.hpp/.hip files. Stage only our dependencies
    # in the JIT cache, never into a source checkout or installed site-packages.
    # Atomic replace makes concurrent first-use staging safe.
    import os
    import tempfile

    def stage(path, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        if destination.exists() and destination.read_bytes() == data:
            return
        fd, tmp = tempfile.mkstemp(dir=destination.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, destination)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    staged_include = build_dir / "include"
    for relative in ("kernel/rmsnorm.hpp", "backend/runtime.hpp", "core/config.hpp"):
        stage(include / "peak_gemm" / relative, staged_include / "peak_gemm" / relative)
    staged_source = build_dir / "src" / "rmsnorm.cu"
    stage(source, staged_source)
    return load(
        name=name, build_directory=str(build_dir),
        sources=[str(staged_source)], extra_include_paths=[str(staged_include)],
        extra_cflags=["-O3", "-std=c++17"], extra_cuda_cflags=flags,
        with_cuda=True, verbose=False,
    )


def _get_extension(device: torch.device):
    if torch.version.hip:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        if arch not in ("gfx942", "gfx950", "gfx1250"):
            raise RuntimeError(f"PeakGemm's HIP Warp backend does not support {arch}")
    else:
        major, minor = torch.cuda.get_device_capability(device)
        if major < 8:
            raise RuntimeError("PeakGemm RMSNorm requires CUDA SM80 or newer")
        arch = f"sm{major}{minor}"
    # load() is cached. Call once before CUDA/HIP graph capture.
    return _extension(arch)


def _check(x: Tensor, weight: Tensor, eps: float) -> None:
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("RMSNorm requires CUDA/HIP tensors")
    if x.device != weight.device:
        raise ValueError("RMSNorm tensors must be on the same device")
    if x.dtype not in _DTYPES or weight.dtype not in _DTYPES:
        raise TypeError("RMSNorm supports fp16, bf16 and fp32")
    if x.ndim < 1 or weight.ndim != 1 or x.shape[-1] == 0 or x.shape[-1] != weight.numel():
        raise ValueError("RMSNorm expects x[..., N] and weight[N], N > 0")
    if not math.isfinite(eps) or eps < 0 or eps > torch.finfo(torch.float32).max:
        raise ValueError("RMSNorm eps must be finite, nonnegative and representable in fp32")


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ext = _get_extension(x.device)
        save = any(ctx.needs_input_grad[:2])
        y, rstd = ext.forward(x, weight, eps, save)
        if save:
            ctx.save_for_backward(x, weight, rstd)
            ctx.ext = ext
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        need_dx, need_dw = ctx.needs_input_grad[:2]
        dx, dw = ctx.ext.backward(dy.contiguous(), x, weight, rstd, need_dx, need_dw)
        return dx if need_dx else None, dw if need_dw else None, None


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize the last dimension with FP32 math and cast the result to x.dtype.

    Supports first-order autograd, mixed input/weight dtypes and arbitrary rank.
    Noncontiguous tensors are materialized before the GPU kernel. No residual,
    quantization or Gemma (1 + weight) offset is applied.
    """
    eps = float(eps)
    _check(x, weight, eps)
    x, weight = x.contiguous(), weight.contiguous()
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return _RMSNorm.apply(x, weight, eps)
    return _get_extension(x.device).forward(x, weight, eps, False)[0]


class RMSNorm(nn.Module):
    """Drop-in replacement for modelx/model/transformer.py::RMSNorm."""

    def __init__(self, dim: int, eps: float, *, device=None, dtype=None) -> None:
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("dim must be a positive integer")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)
