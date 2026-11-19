#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "peak_gemm/backend/runtime.hpp"

namespace peak_gemm::kernel {

template <typename input_t, typename scalar_t>
PEAKGEMM_DEVICE_INLINE float moe_warp_dot(const input_t *x, const scalar_t *weight, uint32_t size) {
    using Warp = backend::Warp;
    float value = 0.0F;
    for (uint32_t k = threadIdx.x % Warp::size; k < size; k += Warp::size) {
        value += static_cast<float>(x[k]) * static_cast<float>(weight[k]);
    }
#pragma unroll
    for (uint32_t offset = Warp::size / 2; offset > 0; offset /= 2) {
        value += Warp::shuffle_xor(value, offset);
    }
    return value;
}

// One CTA owns one token, so all stages need only block-local synchronization.
// Contiguous row-major tensors: T = gridDim.x, H = hidden_size,
// I = intermediate_size, E = num_experts. Expert slice E is the shared expert.
template <typename scalar_t>
__global__ void fused_moe_kernel(
    const scalar_t *x,      // [T, H] input token activations
    const scalar_t *router, // [E, H] router projection weights
    const scalar_t *gate,   // [E+1, I, H] gate projection weights
    const scalar_t *up,     // [E+1, I, H] up projection weights
    const scalar_t *down,   // [E+1, H, I] down projection weights
    scalar_t *y,            // [T, H] weighted routed outputs + shared expert output
    uint32_t hidden_size, uint32_t intermediate_size, uint32_t num_experts, uint32_t top_k) {
    using Warp = backend::Warp;
    const uint32_t thread = threadIdx.x;
    const uint32_t warp = thread / Warp::size;
    const uint32_t lane = thread % Warp::size;
    const uint32_t warp_count = blockDim.x / Warp::size;
    const auto *token = x + static_cast<std::size_t>(blockIdx.x) * hidden_size;
    extern __shared__ float shared[];
    float *output = shared;
    float *activation = output + hidden_size;
    float *logits = activation + intermediate_size;
    float *weights = logits + num_experts;
    auto *ids = reinterpret_cast<uint32_t *>(weights + top_k);

    for (uint32_t h = thread; h < hidden_size; h += blockDim.x) {
        output[h] = 0.0F;
    }
    for (uint32_t expert = warp; expert < num_experts; expert += warp_count) {
        const float value = moe_warp_dot(token, router + static_cast<std::size_t>(expert) * hidden_size, hidden_size);
        if (lane == 0) logits[expert] = value;
    }
    __syncthreads();

    if (thread == 0) {
        // softmax(logits).topk(K), renormalized, equals softmax(topk(logits)).
        // Select before exponentiation to avoid unnecessary work and underflow.
        float maximum = 0.0F;
        float sum = 0.0F;
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            uint32_t best = 0;
            for (uint32_t expert = 1; expert < num_experts; ++expert) {
                if (logits[expert] > logits[best]) best = expert;
            }
            const float value = logits[best];
            ids[slot] = best;
            logits[best] = -INFINITY;
            if (slot == 0) maximum = value;
            weights[slot] = expf(value - maximum);
            sum += weights[slot];
        }
        for (uint32_t slot = 0; slot < top_k; ++slot) {
            weights[slot] /= sum;
        }
    }
    __syncthreads();

    // The last weight slice is the shared expert, with routing weight one.
    const std::size_t expert_stride = static_cast<std::size_t>(hidden_size) * intermediate_size;
    for (uint32_t slot = 0; slot <= top_k; ++slot) {
        const uint32_t expert = slot == top_k ? num_experts : ids[slot];
        const float weight = slot == top_k ? 1.0F : weights[slot];
        const auto *expert_gate = gate + expert * expert_stride;
        const auto *expert_up = up + expert * expert_stride;
        const auto *expert_down = down + expert * expert_stride;
        for (uint32_t i = warp; i < intermediate_size; i += warp_count) {
            const std::size_t row = static_cast<std::size_t>(i) * hidden_size;
            const float g = moe_warp_dot(token, expert_gate + row, hidden_size);
            const float u = moe_warp_dot(token, expert_up + row, hidden_size);
            if (lane == 0) activation[i] = (g / (1.0F + expf(-g))) * u;
        }
        __syncthreads();
        for (uint32_t h = warp; h < hidden_size; h += warp_count) {
            const float value = moe_warp_dot(activation, expert_down + static_cast<std::size_t>(h) * intermediate_size, intermediate_size);
            if (lane == 0) output[h] += weight * value;
        }
        // All warps must finish reading activation before the next expert.
        __syncthreads();
    }
    for (uint32_t h = thread; h < hidden_size; h += blockDim.x) {
        y[static_cast<std::size_t>(blockIdx.x) * hidden_size + h] = static_cast<scalar_t>(output[h]);
    }
}

// Contiguous row-major tensors:
// x/y [T,H], router [E,H], gate/up [E+1,I,H], down [E+1,H,I].
// Slice E is the shared expert. No bias, EP, workspace or auxiliary launches.
// Arithmetic and intermediates are FP32; only the final output is cast.
// Inputs must produce finite logits/activations. Equal logits prefer lower ids.
// y must not overlap any input. T == 0 is a no-op (null pointers are allowed).
template <typename scalar_t>
void fused_moe_gpu(
    const scalar_t *x, const scalar_t *router, const scalar_t *gate, const scalar_t *up, const scalar_t *down, scalar_t *y,
    uint32_t token_count, uint32_t hidden_size, uint32_t intermediate_size, uint32_t num_experts, uint32_t top_k,
    gpuStream_t stream = nullptr) {
    static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __bfloat16>,
                  "Fused MoE supports fp32, fp16 and bf16");
    if (hidden_size == 0 || intermediate_size == 0 || num_experts == 0 || top_k == 0 || top_k > num_experts) {
        throw std::invalid_argument("Fused MoE requires positive dimensions and 1 <= top_k <= num_experts");
    }
    const std::size_t shared_bytes =
        (static_cast<std::size_t>(hidden_size) + intermediate_size + num_experts + top_k) * sizeof(float) + top_k * sizeof(uint32_t);
    // Stay within the default per-block shared-memory budget on CUDA and HIP.
    if (shared_bytes > 48U * 1024U) {
        throw std::invalid_argument("Fused MoE shared memory exceeds 48 KiB");
    }
    if (token_count > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("Fused MoE token count exceeds the grid X limit");
    }
    if (token_count == 0) return;
    if (x == nullptr || router == nullptr || gate == nullptr || up == nullptr || down == nullptr || y == nullptr) {
        throw std::invalid_argument("Fused MoE tensor pointers must not be null");
    }
    fused_moe_kernel<scalar_t><<<token_count, 256, shared_bytes, stream>>>(
        x, router, gate, up, down, y, hidden_size, intermediate_size, num_experts, top_k);
}

} // namespace peak_gemm::kernel
