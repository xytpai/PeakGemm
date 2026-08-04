from __future__ import annotations

import hashlib
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch


ext_base_name = os.path.splitext(os.path.basename(__file__))[0]


_RUNNERS = {}
_WORKSPACES = {}


@dataclass(frozen=True)
class RuntimeParams:
    block_m: int
    block_n: int
    block_k: int
    block_m_warps: int
    block_n_warps: int
    stages: int
    swizzle_m: int


@dataclass(frozen=True)
class ConstexprParams:
    block_m: int
    block_n: int
    block_k: int
    block_m_warps: int
    block_n_warps: int
    stages: int
    swizzle_m: int
    has_bias: bool
    is_split_k: bool


class Compiler:
    def __init__(self, params: ConstexprParams):
        self.constexpr_params = params
    
    def get_ext_name(self) -> str:
        key = (torch.__version__, self.constexpr_params)
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        return f"{ext_base_name}_{digest}"

    def get_source(self) -> str:
        p = self.constexpr_params
        is_split_k = "true" if p.is_split_k else "false"
        has_bias = "true" if p.has_bias else "false"
        ext_name = self.get_ext_name()
        return f"""
#include <cstdint>
#include <limits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "peak_gemm/backend/arch_sm80.hpp"
#include "peak_gemm/kernel/hgemm_sm80.hpp"

namespace {{

template <typename scalar_t>
inline void launch(
    torch::Tensor &a,
    torch::Tensor &b,
    torch::Tensor &c,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t split_k,
    uint32_t *semaphore,
    uint32_t *signal,
    gpuStream_t stream) {{
    peak_gemm::kernel::hgemm_template<
        scalar_t,
        {p.block_m},
        {p.block_n},
        {p.block_k},
        {p.block_m_warps},
        {p.block_n_warps},
        {p.stages},
        {p.swizzle_m},
        {has_bias},
        {is_split_k}>(
        reinterpret_cast<const scalar_t *>(a.data_ptr()),
        reinterpret_cast<const scalar_t *>(b.data_ptr()),
        reinterpret_cast<scalar_t *>(c.data_ptr()),
        m,
        n,
        k,
        split_k,
        semaphore,
        signal,
        static_cast<const scalar_t *>(nullptr),
        ßstream);
}}

void hgemm(
    torch::Tensor c,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor semaphore,
    torch::Tensor signal) {{
    TORCH_CHECK(
        c.is_cuda() && a.is_cuda() && b.is_cuda(),
        "PeakGemm JIT expects CUDA tensors");
    TORCH_CHECK(
        c.device() == a.device() && a.device() == b.device(),
        "PeakGemm JIT tensors must be on the same device");
    TORCH_CHECK(
        c.scalar_type() == a.scalar_type() &&
            a.scalar_type() == b.scalar_type(),
        "PeakGemm JIT tensors must have the same dtype");
    TORCH_CHECK(
        c.is_contiguous() && a.is_contiguous() && b.is_contiguous(),
        "PeakGemm JIT tensors must be contiguous");
    TORCH_CHECK(
        c.dim() == 2 && a.dim() == 2 && b.dim() == 2,
        "PeakGemm JIT expects matrices");

    const auto m64 = a.size(0);
    const auto n64 = b.size(0);
    const auto k64 = a.size(1);
    TORCH_CHECK(
        b.size(1) == k64 && c.size(0) == m64 && c.size(1) == n64,
        "PeakGemm JIT tensor shapes do not match");
    TORCH_CHECK(
        m64 <= std::numeric_limits<std::uint32_t>::max() &&
            n64 <= std::numeric_limits<std::uint32_t>::max() &&
            k64 <= std::numeric_limits<std::uint32_t>::max(),
        "PeakGemm JIT dimensions exceed uint32");
    TORCH_CHECK(
        semaphore.device() == a.device() && signal.device() == a.device() &&
            semaphore.numel() >= 256 && signal.numel() >= 256,
        "PeakGemm JIT workspace is invalid");

    c10::DeviceGuard guard(a.device());
    const auto m = static_cast<std::uint32_t>(m64);
    const auto n = static_cast<std::uint32_t>(n64);
    const auto k = static_cast<std::uint32_t>(k64);
    auto *semaphore_ptr = semaphore.data_ptr<std::uint32_t>();
    auto *signal_ptr = signal.data_ptr<std::uint32_t>();
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

    if (a.scalar_type() == at::kHalf) {{
        launch<__half>(
            c, a, b, m, n, k, semaphore_ptr, signal_ptr, stream);
    }} else if (a.scalar_type() == at::kBFloat16) {{
        launch<__bfloat16>(
            c, a, b, m, n, k, semaphore_ptr, signal_ptr, stream);
    }} else {{
        TORCH_CHECK(false, "PeakGemm JIT supports fp16 and bf16");
    }}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}

}} // namespace

TORCH_LIBRARY({ext_name}, module) {{
    module.def(
        "hgemm(Tensor(a!) c, Tensor a, Tensor b, "
        "Tensor(b!) semaphore, Tensor(c!) signal) -> ()");
    module.impl("hgemm", &hgemm);
}}
"""

    def compile(params: RuntimeParams, cache_dir: str):
        """Compile one params, optionally under cache_dir, and return a callable."""
        if not isinstance(params, RuntimeParams):
            raise TypeError("params must be RuntimeParams")

        cache_path = Path(cache_dir).expanduser().resolve()
        ext_name = self.get_ext_name()
        cached = _RUNNERS.get(namespace, None)
        if cached is not None:
            return cached

        from torch.utils.cpp_extension import load_inline

        include_dir = Path(__file__).resolve().parents[1] / "include"
        build_dir = cache_path / namespace if cache_path is not None else None
        if build_dir is not None:
            build_dir.mkdir(parents=True, exist_ok=True)
        original_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        peakgemm_arch = os.environ.get("PEAKGEMM_CUDA_ARCH_LIST")
        if original_arch is None and peakgemm_arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = peakgemm_arch
        try:
            load_inline(
                name=namespace,
                cpp_sources="",
                cuda_sources=_cuda_source(namespace, config),
                extra_include_paths=[str(include_dir)],
                extra_cuda_cflags=[
                    "-O3",
                    "--std=c++20",
                    "--expt-relaxed-constexpr",
                ],
                build_directory=(
                    str(build_dir) if build_dir is not None else None
                ),
                with_cuda=True,
                is_python_module=False,
                verbose=verbose,
            )
        finally:
            if original_arch is None and peakgemm_arch:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)

        op = getattr(getattr(torch.ops, namespace), "hgemm")
        def run(
            c: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
        ) -> torch.Tensor:
            if not a.is_cuda:
                raise ValueError("PeakGemm JIT expects CUDA tensors")
            stream = torch.cuda.current_stream(a.device)
            device_index = (
                torch.cuda.current_device()
                if a.device.index is None
                else a.device.index
            )
            key = (device_index, int(stream.cuda_stream))
            if key not in _WORKSPACES:
                _WORKSPACES[key] = (
                    torch.zeros(256, dtype=torch.uint32, device=a.device),
                    torch.zeros(256, dtype=torch.uint32, device=a.device),
                )
            semaphore, signal = _WORKSPACES[key]
            op(c, a, b, semaphore, signal)
            return c

        _RUNNERS[namespace] = run
        return run
