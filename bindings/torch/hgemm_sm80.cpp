#include "ops.hpp"

#include <cstdint>
#include <limits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceGuard.h>

#include "peak_gemm/backend/arch_sm80.hpp"
#include "peak_gemm/kernel/hgemm_sm80.hpp"

void gemm_peak(
    Tensor &c, Tensor &a, Tensor &b,
    int64_t m, int64_t n, int64_t k,
    Tensor &semaphore, Tensor &signal) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "gemm_peak requires CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "gemm_peak requires contiguous tensors");
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(), "gemm_peak tensors must be on the same device");
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(), "gemm_peak tensors must have the same dtype");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16, "gemm_peak supports fp16 and bf16");
    TORCH_CHECK(m > 0 && n > 0 && k > 0, "gemm_peak dimensions must be positive");
    TORCH_CHECK(
        m <= std::numeric_limits<std::uint32_t>::max() && n <= std::numeric_limits<std::uint32_t>::max() && k <= std::numeric_limits<std::uint32_t>::max(),
        "gemm_peak dimensions exceed uint32");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "gemm_peak tensors must be matrices");
    TORCH_CHECK(
        a.size(0) == m && a.size(1) == k && b.size(0) == n && b.size(1) == k && c.size(0) == m && c.size(1) == n,
        "gemm_peak tensor shapes do not match M, N and K");
    TORCH_CHECK(semaphore.device() == a.device() && signal.device() == a.device(), "gemm_peak workspace must be on the input device");
    TORCH_CHECK(semaphore.is_contiguous() && signal.is_contiguous(), "gemm_peak workspace must be contiguous");
    TORCH_CHECK(
        semaphore.numel() >= peak_gemm::kernel::kSemaphoreCount && signal.numel() >= peak_gemm::kernel::kSemaphoreCount,
        "gemm_peak workspace is too small");

    c10::DeviceGuard device_guard(a.device());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto *semaphore_ptr = semaphore.data_ptr<std::uint32_t>();
    auto *signal_ptr = signal.data_ptr<std::uint32_t>();
    const auto m32 = static_cast<std::uint32_t>(m);
    const auto n32 = static_cast<std::uint32_t>(n);
    const auto k32 = static_cast<std::uint32_t>(k);
    if (a.scalar_type() == at::kHalf) {
        peak_gemm::kernel::hgemm_gpu<__half>(
            reinterpret_cast<const __half *>(a.data_ptr<c10::Half>()),
            reinterpret_cast<const __half *>(b.data_ptr<c10::Half>()),
            reinterpret_cast<__half *>(c.data_ptr<c10::Half>()),
            m32, n32, k32, semaphore_ptr, signal_ptr,
            static_cast<const __half *>(nullptr), stream);
    } else {
        peak_gemm::kernel::hgemm_gpu<__bfloat16>(
            reinterpret_cast<const __bfloat16 *>(a.data_ptr<c10::BFloat16>()),
            reinterpret_cast<const __bfloat16 *>(b.data_ptr<c10::BFloat16>()),
            reinterpret_cast<__bfloat16 *>(c.data_ptr<c10::BFloat16>()),
            m32, n32, k32, semaphore_ptr, signal_ptr,
            static_cast<const __bfloat16 *>(nullptr), stream);
    }
}
