#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "peak_gemm/backend/runtime.hpp"
#include "peak_gemm/core/math.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::kernel {

template <typename input_t, typename scalar_t>
PEAKGEMM_DEVICE_INLINE float moe_warp_dot(const input_t *x, const scalar_t *weight, uint32_t size) {
    using Warp = backend::Warp;
    constexpr uint32_t VEC_SIZE = 16 / sizeof(scalar_t);
    constexpr uint32_t INPUT_VEC_SIZE = 16 / sizeof(input_t);
    static_assert(VEC_SIZE % INPUT_VEC_SIZE == 0);
    const uint32_t lane = threadIdx.x % Warp::size;
    float value = 0.0F;
    for (uint32_t k = lane * VEC_SIZE; k < size; k += Warp::size * VEC_SIZE) {
        core::Vector<scalar_t, VEC_SIZE> w;
        w.load(weight + k);
        // FP32 activation + 16-bit weights: two 16B input loads per weight vector.
#pragma unroll
        for (uint32_t part = 0; part < VEC_SIZE; part += INPUT_VEC_SIZE) {
            core::Vector<input_t, INPUT_VEC_SIZE> v;
            v.load(x + k + part);
#pragma unroll
            for (uint32_t j = 0; j < INPUT_VEC_SIZE; ++j) {
                value += static_cast<float>(v[j]) * static_cast<float>(w[part + j]);
            }
        }
    }
#pragma unroll
    for (uint32_t offset = Warp::size / 2; offset > 0; offset /= 2) {
        value += Warp::shuffle_xor(value, offset);
    }
    return value;
}

// All lanes of one warp participate. Scores/weights/ids reside in shared memory.
// CACHED_EXPERTS must cover num_experts, or be zero for a warp-parallel shared
// scan. All scores/biases are finite, scores >= 0, and 1 <= top_k <= num_experts.
// The caller synchronizes the block before other warps consume weights/ids.
template <uint32_t CACHED_EXPERTS>
PEAKGEMM_DEVICE_INLINE void moe_warp_topk(
    float *scores, const float *bias, float *weights, uint32_t *ids, uint32_t num_experts, uint32_t top_k) {
    using Warp = backend::Warp;
    constexpr uint32_t ITEMS = CACHED_EXPERTS == 0 ? 1 : core::ceil_div(CACHED_EXPERTS, Warp::size);
    const uint32_t lane = threadIdx.x % Warp::size;
    float candidates[ITEMS];
    if constexpr (CACHED_EXPERTS != 0) {
#pragma unroll
        for (uint32_t i = 0; i < ITEMS; ++i) {
            const uint32_t expert = lane + i * Warp::size;
            candidates[i] = expert < num_experts ? scores[expert] + bias[expert] : -INFINITY;
        }
    }

    float sum = 0.0F;
    for (uint32_t slot = 0; slot < top_k; ++slot) {
        float best_value = -INFINITY;
        uint32_t best_id = UINT32_MAX;
        if constexpr (CACHED_EXPERTS != 0) {
#pragma unroll
            for (uint32_t i = 0; i < ITEMS; ++i) {
                if (candidates[i] > best_value) {
                    best_value = candidates[i];
                    best_id = lane + i * Warp::size;
                }
            }
        } else {
            for (uint32_t expert = lane; expert < num_experts; expert += Warp::size) {
                const float value = scores[expert] + bias[expert];
                if (value > best_value) {
                    best_value = value;
                    best_id = expert;
                }
            }
        }
#pragma unroll
        for (uint32_t offset = Warp::size / 2; offset > 0; offset /= 2) {
            const float value = Warp::shuffle_xor(best_value, offset);
            const uint32_t id = Warp::shuffle_xor(best_id, offset);
            if (value > best_value || (value == best_value && id < best_id)) {
                best_value = value;
                best_id = id;
            }
        }
        // Broadcast the unbiased score, not the corrected selection value.
        float weight = 0.0F;
        if (lane == 0) {
            weight = scores[best_id];
            weights[slot] = weight;
            ids[slot] = best_id;
        }
        sum += Warp::shuffle(weight, 0);
        if constexpr (CACHED_EXPERTS != 0) {
#pragma unroll
            for (uint32_t i = 0; i < ITEMS; ++i) {
                if (lane + i * Warp::size == best_id) candidates[i] = -INFINITY;
            }
        } else {
            if (lane == 0) scores[best_id] = -INFINITY;
            Warp::barrier();
        }
    }
    // Make lane 0's stores visible before other lanes normalize the weights.
    Warp::barrier();
    for (uint32_t slot = lane; slot < top_k; slot += Warp::size) {
        weights[slot] = weights[slot] / (sum + 1.0e-20F) * 1.5F;
    }
}

// DeepSeek-V4.1-Flash text MoE (also DSpark), with all experts on one device.
// One CTA owns one token, so all stages need only block-local synchronization.
// Contiguous row-major tensors: T = gridDim.x, H = hidden_size,
// I = intermediate_size, E = num_experts. Expert slice E is the shared expert.
// All tensor pointers and H/I row sizes in bytes must be 16-byte aligned.
template <typename scalar_t>
__global__ void fused_moe_kernel(
    const scalar_t *x,        // [T, H] input token activations
    const scalar_t *router,   // [E, H] router projection weights
    const float *router_bias, // [E] FP32 expert score correction bias, selection only
    const scalar_t *gate,     // [E+1, I, H] gate projection weights
    const scalar_t *up,       // [E+1, I, H] up projection weights
    const scalar_t *down,     // [E+1, H, I] down projection weights
    scalar_t *y,              // [T, H] weighted routed outputs + shared expert output
    uint32_t hidden_size, uint32_t intermediate_size, uint32_t num_experts, uint32_t top_k) {
    using Warp = backend::Warp;
    const uint32_t thread = threadIdx.x;
    const uint32_t warp = thread / Warp::size;
    const uint32_t lane = thread % Warp::size;
    const uint32_t warp_count = blockDim.x / Warp::size;
    const auto *token = x + static_cast<std::size_t>(blockIdx.x) * hidden_size;
    extern __shared__ __align__(16) float shared[];
    float *output = shared;
    float *activation = output + hidden_size;
    float *scores = activation + intermediate_size;
    float *weights = scores + num_experts;
    auto *ids = reinterpret_cast<uint32_t *>(weights + top_k);

    for (uint32_t h = thread; h < hidden_size; h += blockDim.x) {
        output[h] = 0.0F;
    }
    for (uint32_t expert = warp; expert < num_experts; expert += warp_count) {
        const float value = moe_warp_dot(token, router + static_cast<std::size_t>(expert) * hidden_size, hidden_size);
        // Stable sqrt(softplus(value)); correction bias is not part of the score.
        if (lane == 0) scores[expert] = sqrtf(fmaxf(value, 0.0F) + log1pf(expf(-fabsf(value))));
    }
    __syncthreads();

    if (warp == 0) {
        if (num_experts <= 384) {
            moe_warp_topk<384>(scores, router_bias, weights, ids, num_experts, top_k);
        } else {
            moe_warp_topk<0>(scores, router_bias, weights, ids, num_experts, top_k);
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
            // Clamp before SiLU: gate has no lower bound; up is clipped on both sides.
            const float g = fminf(moe_warp_dot(token, expert_gate + row, hidden_size), 10.0F);
            const float u = fminf(fmaxf(moe_warp_dot(token, expert_up + row, hidden_size), -10.0F), 10.0F);
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
    constexpr uint32_t VEC_SIZE = 16 / sizeof(scalar_t);
    constexpr uint32_t OUTPUT_VEC_SIZE = 16 / sizeof(float);
    auto *token_y = y + static_cast<std::size_t>(blockIdx.x) * hidden_size;
    for (uint32_t h = thread * VEC_SIZE; h < hidden_size; h += blockDim.x * VEC_SIZE) {
        core::Vector<scalar_t, VEC_SIZE> result;
#pragma unroll
        for (uint32_t part = 0; part < VEC_SIZE; part += OUTPUT_VEC_SIZE) {
            core::Vector<float, OUTPUT_VEC_SIZE> values;
            values.load(output + h + part);
#pragma unroll
            for (uint32_t j = 0; j < OUTPUT_VEC_SIZE; ++j) {
                result[part + j] = static_cast<scalar_t>(values[j]);
            }
        }
        result.store(token_y + h);
    }
}

// Contiguous row-major tensors:
// x/y [T,H], router [E,H], router_bias [E] (FP32), gate/up [E+1,I,H], down [E+1,H,I].
// Slice E is the shared expert. No linear projection bias, EP, workspace or auxiliary launches.
// Routing: sqrtsoftplus -> top-k(score + bias) -> unbiased normalization (eps 1e-20) * 1.5.
// SwiGLU: silu(min(gate, 10)) * clamp(up, -10, 10), for routed and shared experts.
// This is the unquantized text MoE; packed FP4/FP8 weights must be dequantized by the caller.
// Arithmetic and intermediates are FP32; only the final output is cast.
// Bias values and computed logits/activations must be finite. Equal corrected scores prefer lower ids.
// All tensor pointers must be 16-byte aligned; H/I must be multiples of 16 / sizeof(scalar_t).
// y must not overlap any input. T == 0 is a no-op (null pointers are allowed).
template <typename scalar_t>
void fused_moe_gpu(
    const scalar_t *x, const scalar_t *router, const float *router_bias, const scalar_t *gate, const scalar_t *up, const scalar_t *down, scalar_t *y,
    uint32_t token_count, uint32_t hidden_size, uint32_t intermediate_size, uint32_t num_experts, uint32_t top_k,
    gpuStream_t stream = nullptr) {
    static_assert(std::is_same_v<scalar_t, float> || std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __bfloat16>,
                  "Fused MoE supports fp32, fp16 and bf16");
    if (hidden_size == 0 || intermediate_size == 0 || num_experts == 0 || top_k == 0 || top_k > num_experts) {
        throw std::invalid_argument("Fused MoE requires positive dimensions and 1 <= top_k <= num_experts");
    }
    constexpr uint32_t VEC_SIZE = 16 / sizeof(scalar_t);
    if (hidden_size % VEC_SIZE != 0 || intermediate_size % VEC_SIZE != 0) {
        throw std::invalid_argument("Fused MoE H/I must be aligned to 16-byte vectors");
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
    if (x == nullptr || router == nullptr || router_bias == nullptr || gate == nullptr || up == nullptr || down == nullptr || y == nullptr) {
        throw std::invalid_argument("Fused MoE tensor pointers must not be null");
    }
    const uintptr_t addresses = reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(router) | reinterpret_cast<uintptr_t>(router_bias) | reinterpret_cast<uintptr_t>(gate)
                                | reinterpret_cast<uintptr_t>(up) | reinterpret_cast<uintptr_t>(down) | reinterpret_cast<uintptr_t>(y);
    if (addresses % 16U != 0) {
        throw std::invalid_argument("Fused MoE tensor pointers must be 16-byte aligned");
    }
    fused_moe_kernel<scalar_t><<<token_count, 256, shared_bytes, stream>>>(
        x, router, router_bias, gate, up, down, y, hidden_size, intermediate_size, num_experts, top_k);
}

} // namespace peak_gemm::kernel
