#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "peak_gemm/data.hpp"
#include "peak_gemm/kernel/fused_moe.hpp"

// Exercise the exact device helper used by the fused kernel and export routes.
__global__ void moe_topk_test_kernel(
    const float *input, const float *bias, float *weights, uint32_t *ids, uint32_t experts, uint32_t top_k) {
    using Warp = peak_gemm::backend::Warp;
    extern __shared__ float shared[];
    float *scores = shared;
    float *selected_weights = scores + experts;
    auto *selected_ids = reinterpret_cast<uint32_t *>(selected_weights + top_k);
    for (uint32_t e = threadIdx.x; e < experts; e += Warp::size) {
        scores[e] = input[blockIdx.x * experts + e];
    }
    __syncthreads();
    if (experts <= 384) {
        peak_gemm::kernel::moe_warp_topk<384>(scores, bias, selected_weights, selected_ids, experts, top_k);
    } else {
        peak_gemm::kernel::moe_warp_topk<0>(scores, bias, selected_weights, selected_ids, experts, top_k);
    }
    __syncthreads();
    for (uint32_t k = threadIdx.x; k < top_k; k += Warp::size) {
        weights[blockIdx.x * top_k + k] = selected_weights[k];
        ids[blockIdx.x * top_k + k] = selected_ids[k];
    }
}

void check_topk(uint32_t experts, uint32_t top_k) {
    using peak_gemm::Data;
    constexpr auto cpu = peak_gemm::DataDevice::cpu;
    constexpr auto gpu = peak_gemm::DataDevice::gpu;
    constexpr uint32_t ROWS = 7;
    auto scores = Data<float>::uniform({ROWS, experts}, 0.0F, 2.0F, cpu, 2026);
    auto bias = Data<float>::uniform({experts}, -1.0F, 1.0F, cpu, 2027);
    for (uint32_t row = 1; row < ROWS; ++row) {
        for (uint32_t e = 0; e < experts; ++e) {
            float value;
            switch (row) {
            case 1: value = 2.0F; break;                                                  // Ties within and across lanes.
            case 2: value = 0.0F; break;                                                  // All scores zero.
            case 3: value = e % peak_gemm::backend::Warp::size == 0 ? 4.0F : 0.0F; break; // Repeated winners from one lane.
            case 4: value = e + 1 == experts ? 8.0F : 0.0F; break;                        // Last valid candidate, including partial lane groups.
            case 5: value = static_cast<float>(e % 4); break;
            default: value = 1.0e-22F; break; // Normalization epsilon matters.
            }
            scores[row * experts + e] = value;
        }
    }
    // Random corrected scores, then exact ties/negative corrected scores.
    for (uint32_t mode = 0; mode < 3; ++mode) {
        if (mode == 1) bias.fill(0.0F);
        if (mode == 2) {
            for (uint32_t e = 0; e < experts; ++e) bias[e] = -static_cast<float>(e % 4) - 16.0F;
        }
        Data<float> expected_weights({ROWS, top_k});
        Data<uint32_t> expected_ids({ROWS, top_k});
        std::vector<uint32_t> order(experts);
        for (uint32_t row = 0; row < ROWS; ++row) {
            const auto *values = scores.data() + row * experts;
            std::iota(order.begin(), order.end(), 0U);
            std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return values[a] + bias[a] > values[b] + bias[b]; });
            float sum = 0.0F;
            for (uint32_t k = 0; k < top_k; ++k) sum += values[order[k]];
            for (uint32_t k = 0; k < top_k; ++k) {
                expected_ids[row * top_k + k] = order[k];
                expected_weights[row * top_k + k] = values[order[k]] / (sum + 1.0e-20F) * 1.5F;
            }
        }
        auto device_scores = scores.copy_to(gpu);
        auto device_bias = bias.copy_to(gpu);
        Data<float> device_weights({ROWS, top_k}, gpu);
        Data<uint32_t> device_ids({ROWS, top_k}, gpu);
        device_weights.fill(NAN);
        device_ids.fill(UINT32_MAX);
        for (uint32_t repeat = 0; repeat < 2; ++repeat) {
            moe_topk_test_kernel<<<ROWS, peak_gemm::backend::Warp::size, (experts + 2 * top_k) * sizeof(float)>>>(
                device_scores.data(), device_bias.data(), device_weights.data(), device_ids.data(), experts, top_k);
            if (gpuGetLastError() != gpuSuccess || gpuDeviceSynchronize() != gpuSuccess) {
                throw std::runtime_error("MoE top-k test kernel failed");
            }
            const auto ids = device_ids.copy_to(cpu);
            const auto weights = device_weights.copy_to(cpu);
            for (std::size_t i = 0; i < ids.size(); ++i) {
                const float diff = std::abs(weights[i] - expected_weights[i]);
                if (ids[i] != expected_ids[i] || !std::isfinite(diff) || diff > 2.0e-6F) {
                    throw std::runtime_error("MoE top-k ids/weights mismatch");
                }
            }
        }
    }
    std::cout << "E=" << experts << " K=" << top_k << " top-k ids/weights: pass\n";
}

int main() {
    const uint32_t cases[][2] = {
        {1, 1},
        {31, 3},
        {32, 32},
        {33, 6},
        {63, 6},
        {64, 64},
        {65, 65},
        {128, 3},
        {257, 257},
        {384, 6},
        {384, 384},
        {385, 6},
        {385, 385},
        {1024, 8},
    };
    for (const auto &shape : cases) check_topk(shape[0], shape[1]);
    std::cout << "ok\n";
}
