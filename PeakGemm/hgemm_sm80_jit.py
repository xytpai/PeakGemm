from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch


_EXT_BASE_NAME = Path(__file__).stem
_SOURCE_VERSION = 2
_RUNNERS: dict[str, Callable] = {}
_WORKSPACES: dict[
    tuple[int, int], tuple[torch.Tensor, torch.Tensor]
] = {}
_EXTRA_CUDA_CFLAGS = [
    "-O3",
    "--std=c++17",
    "--expt-relaxed-constexpr",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]


def _resolve_cache_dir(
    cache_dir: str | os.PathLike[str] | None,
) -> Path:
    path = Path.cwd() / "temp" if cache_dir is None else Path(cache_dir)
    return path.expanduser().resolve()


@dataclass(frozen=True)
class ConstexprParams:
    block_m: int
    block_n: int
    block_k: int
    block_m_warps: int
    block_n_warps: int
    stages: int
    swizzle_m: int = 8

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                or value > 0xFFFFFFFF
            ):
                raise ValueError(f"{name} must be a positive uint32")
        if self.stages < 2:
            raise ValueError("stages must be at least 2")


class Compiler:
    def __init__(self, params: ConstexprParams):
        if not isinstance(params, ConstexprParams):
            raise TypeError("params must be ConstexprParams")
        self.constexpr_params = params

    def get_ext_name(self, cache_dir: Path | None = None) -> str:
        arch = os.environ.get(
            "TORCH_CUDA_ARCH_LIST",
            os.environ.get("PEAKGEMM_CUDA_ARCH_LIST", ""),
        )
        key = (
            _SOURCE_VERSION,
            torch.__version__,
            torch.version.cuda,
            arch,
            self.constexpr_params,
            str(cache_dir) if cache_dir is not None else "",
        )
        digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
        return f"{_EXT_BASE_NAME}_{digest}"

    def get_source(self, ext_name: str | None = None) -> str:
        p = self.constexpr_params
        ext_name = ext_name or self.get_ext_name()
        return f"""
#include <cstdint>
#include <limits>
#include <optional>

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "peak_gemm/backend/arch_sm80.hpp"
#include "peak_gemm/kernel/hgemm_sm80.hpp"

namespace {{

template <typename scalar_t, bool HAS_BIAS, bool IS_SPLIT_K>
inline void launch(
    torch::Tensor &c,
    torch::Tensor &a,
    torch::Tensor &b,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    std::uint32_t split_k,
    std::uint32_t *semaphore,
    std::uint32_t *signal,
    const scalar_t *bias,
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
        HAS_BIAS,
        IS_SPLIT_K>(
            reinterpret_cast<const scalar_t *>(a.data_ptr()),
            reinterpret_cast<const scalar_t *>(b.data_ptr()),
            reinterpret_cast<scalar_t *>(c.data_ptr()),
            m,
            n,
            k,
            split_k,
            semaphore,
            signal,
            bias,
            stream);
}}

template <typename scalar_t>
inline void dispatch(
    torch::Tensor &c,
    torch::Tensor &a,
    torch::Tensor &b,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    std::uint32_t split_k,
    std::uint32_t *semaphore,
    std::uint32_t *signal,
    const scalar_t *bias,
    gpuStream_t stream) {{
    if (split_k > 1U) {{
        if (bias != nullptr) {{
            launch<scalar_t, true, true>(
                c, a, b, m, n, k, split_k,
                semaphore, signal, bias, stream);
        }} else {{
            launch<scalar_t, false, true>(
                c, a, b, m, n, k, split_k,
                semaphore, signal, bias, stream);
        }}
    }} else if (bias != nullptr) {{
        launch<scalar_t, true, false>(
            c, a, b, m, n, k, split_k,
            semaphore, signal, bias, stream);
    }} else {{
        launch<scalar_t, false, false>(
            c, a, b, m, n, k, split_k,
            semaphore, signal, bias, stream);
    }}
}}

void hgemm(
    torch::Tensor c,
    torch::Tensor a,
    torch::Tensor b,
    std::int64_t split_k,
    std::optional<torch::Tensor> bias,
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
    TORCH_CHECK(
        split_k > 0 && split_k <= 65535,
        "PeakGemm JIT split_k must be between 1 and 65535");

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
        semaphore.is_cuda() && signal.is_cuda() &&
            semaphore.device() == a.device() &&
            signal.device() == a.device() &&
            semaphore.numel() >= 256 && signal.numel() >= 256,
        "PeakGemm JIT workspace is invalid");

    const bool has_bias = bias.has_value() && bias->defined();
    if (has_bias) {{
        TORCH_CHECK(
            bias->is_cuda() && bias->device() == a.device() &&
                bias->scalar_type() == a.scalar_type() &&
                bias->is_contiguous() && bias->numel() == n64,
            "PeakGemm JIT bias must be a contiguous CUDA tensor of size N");
    }}

    c10::DeviceGuard guard(a.device());
    const auto m = static_cast<std::uint32_t>(m64);
    const auto n = static_cast<std::uint32_t>(n64);
    const auto k = static_cast<std::uint32_t>(k64);
    const auto split = static_cast<std::uint32_t>(split_k);
    auto *semaphore_ptr = semaphore.data_ptr<std::uint32_t>();
    auto *signal_ptr = signal.data_ptr<std::uint32_t>();
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

    if (a.scalar_type() == at::kHalf) {{
        const auto *bias_ptr = has_bias
            ? reinterpret_cast<const __half *>(bias->data_ptr())
            : nullptr;
        dispatch<__half>(
            c, a, b, m, n, k, split,
            semaphore_ptr, signal_ptr, bias_ptr, stream);
    }} else if (a.scalar_type() == at::kBFloat16) {{
        const auto *bias_ptr = has_bias
            ? reinterpret_cast<const __bfloat16 *>(bias->data_ptr())
            : nullptr;
        dispatch<__bfloat16>(
            c, a, b, m, n, k, split,
            semaphore_ptr, signal_ptr, bias_ptr, stream);
    }} else {{
        TORCH_CHECK(false, "PeakGemm JIT supports fp16 and bf16");
    }}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}}

}} // namespace

TORCH_LIBRARY({ext_name}, module) {{
    module.def(
        "hgemm(Tensor(a!) c, Tensor a, Tensor b, int split_k, "
        "Tensor? bias, Tensor(b!) semaphore, Tensor(c!) signal) -> ()");
    module.impl("hgemm", &hgemm);
}}
"""

    def compile(
        self,
        cache_dir: str | os.PathLike[str] | None = None,
        *,
        verbose: bool = False,
    ) -> Callable:
        cache_path = _resolve_cache_dir(cache_dir)
        ext_name = self.get_ext_name(cache_path)
        cached = _RUNNERS.get(ext_name)
        if cached is not None:
            return cached

        from torch.utils.cpp_extension import load_inline

        include_dir = Path(__file__).resolve().parents[1] / "include"
        build_dir = cache_path / ext_name
        build_dir.mkdir(parents=True, exist_ok=True)

        original_arch = os.environ.get("TORCH_CUDA_ARCH_LIST")
        peakgemm_arch = os.environ.get("PEAKGEMM_CUDA_ARCH_LIST")
        if original_arch is None and peakgemm_arch:
            os.environ["TORCH_CUDA_ARCH_LIST"] = peakgemm_arch
        try:
            load_inline(
                name=ext_name,
                cpp_sources="",
                cuda_sources=self.get_source(ext_name),
                extra_include_paths=[str(include_dir)],
                extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
                build_directory=str(build_dir),
                with_cuda=True,
                is_python_module=False,
                verbose=verbose,
            )
        finally:
            if original_arch is None and peakgemm_arch:
                os.environ.pop("TORCH_CUDA_ARCH_LIST", None)

        op = getattr(getattr(torch.ops, ext_name), "hgemm")

        def run(
            c: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
            *,
            split_k: int = 1,
            bias: torch.Tensor | None = None,
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
            op(c, a, b, split_k, bias, semaphore, signal)
            return c

        _RUNNERS[ext_name] = run
        return run


def compile_hgemm(
    params: ConstexprParams,
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    verbose: bool = False,
) -> Callable:
    return Compiler(params).compile(cache_dir, verbose=verbose)
