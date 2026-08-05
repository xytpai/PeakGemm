#pragma once

#include <cstdint>

#include "peak_gemm/backend/runtime.hpp"
#include "peak_gemm/core/math.hpp"

namespace peak_gemm::kernel {

template <typename scalar_t>
__global__ void gemm_naive_kernel(
    const scalar_t *a, const scalar_t *b, scalar_t *c, const scalar_t *bias,
    uint32_t m_size, uint32_t n_size, uint32_t k_size) {
    const uint32_t m = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= m_size || n >= n_size) return;
    float value = 0.0F;
    for (uint32_t k = 0; k < k_size; ++k) {
        value += static_cast<float>(a[m * k_size + k]) * static_cast<float>(b[n * k_size + k]);
    }
    if (bias != nullptr) value += static_cast<float>(bias[n]);
    c[m * n_size + n] = static_cast<scalar_t>(value);
}

template <typename scalar_t>
void gemm_naive_gpu(
    const scalar_t *a, const scalar_t *b, scalar_t *c,
    uint32_t m_size, uint32_t n_size, uint32_t k_size,
    const scalar_t *bias = nullptr, gpuStream_t stream = nullptr) {
    const dim3 block(16, 16);
    const dim3 grid(core::ceil_div(n_size, 16U), core::ceil_div(m_size, 16U));
    gemm_naive_kernel<scalar_t><<<grid, block, 0, stream>>>(
        a, b, c, bias, m_size, n_size, k_size);
}

} // namespace peak_gemm::kernel
