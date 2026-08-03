from __future__ import annotations

import hashlib
import itertools
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch


_SOURCE_VERSION = 2
_RUNNERS: dict[str, Callable] = {}
_WORKSPACES: dict[
    tuple[int, int], tuple[torch.Tensor, torch.Tensor]
] = {}


@dataclass(frozen=True)
class HgemmConfig:
    block_m: int
    block_n: int
    block_k: int
    block_m_warps: int
    block_n_warps: int
    stages: int
    swizzle_m: int = 8
    split_k: int = 1

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
        if self.split_k > 65535:
            raise ValueError("split_k exceeds the CUDA grid Y limit")

    @property
    def template_args(self) -> tuple[int, ...]:
        return (
            self.block_m,
            self.block_n,
            self.block_k,
            self.block_m_warps,
            self.block_n_warps,
            self.stages,
            self.swizzle_m,
        )


def enumerate_hgemm_configs(
    *,
    block_m: Iterable[int],
    block_n: Iterable[int],
    block_k: Iterable[int],
    block_m_warps: Iterable[int],
    block_n_warps: Iterable[int],
    stages: Iterable[int],
    swizzle_m: Iterable[int] = (8,),
    split_k: Iterable[int] = (1,),
) -> Iterator[HgemmConfig]:
    """Yield the Cartesian product; each config is compiled separately."""
    axes = (
        block_m,
        block_n,
        block_k,
        block_m_warps,
        block_n_warps,
        stages,
        swizzle_m,
        split_k,
    )
    for values in itertools.product(*(tuple(axis) for axis in axes)):
        yield HgemmConfig(*values)


def _extension_name(
    config: HgemmConfig,
    cache_dir: Path | None = None,
) -> str:
    arch = os.environ.get(
        "TORCH_CUDA_ARCH_LIST",
        os.environ.get("PEAKGEMM_CUDA_ARCH_LIST", ""),
    )
    key = (
        _SOURCE_VERSION,
        config,
        torch.__version__,
        torch.version.cuda,
        arch,
        str(cache_dir) if cache_dir is not None else "",
    )
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    return f"peakgemm_jit_{digest}"


def _cuda_source(namespace: str, config: HgemmConfig) -> str:
    template_args = ", ".join(map(str, config.template_args))
    is_split_k = "true" if config.split_k > 1 else "false"
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
void launch(
    torch::Tensor &c,
    torch::Tensor &a,
    torch::Tensor &b,
    std::uint32_t m,
    std::uint32_t n,
    std::uint32_t k,
    std::uint32_t *semaphore,
    std::uint32_t *signal,
    gpuStream_t stream) {{
    peak_gemm::kernel::hgemm_template<
        scalar_t, {template_args}, false, {is_split_k}>(
            reinterpret_cast<const scalar_t *>(a.data_ptr()),
            reinterpret_cast<const scalar_t *>(b.data_ptr()),
            reinterpret_cast<scalar_t *>(c.data_ptr()),
            m,
            n,
            k,
            {config.split_k}U,
            semaphore,
            signal,
            static_cast<const scalar_t *>(nullptr),
            stream);
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

TORCH_LIBRARY({namespace}, module) {{
    module.def(
        "hgemm(Tensor(a!) c, Tensor a, Tensor b, "
        "Tensor(b!) semaphore, Tensor(c!) signal) -> ()");
    module.impl("hgemm", &hgemm);
}}
"""


def compile_hgemm(
    config: HgemmConfig,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    verbose: bool = False,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Compile one config, optionally under cache_dir, and return a callable."""
    if not isinstance(config, HgemmConfig):
        raise TypeError("config must be HgemmConfig")

    cache_path = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else None
    )
    namespace = _extension_name(config, cache_path)
    cached = _RUNNERS.get(namespace)
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
                "--std=c++17",
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
