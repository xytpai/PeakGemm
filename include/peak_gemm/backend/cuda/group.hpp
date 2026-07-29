#pragma once

#include <cstdint>

#include "peak_gemm/core/config.hpp"

namespace peak_gemm::backend::cuda {

struct Warp {
    static constexpr std::uint32_t size = 32;

    template <typename T>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static T shuffle(T value, int source_lane) {
        return __shfl_sync(0xffffffffU, value, source_lane, size);
    }

    template <typename T>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static T shuffle_xor(T value, int lane_mask) {
        return __shfl_xor_sync(0xffffffffU, value, lane_mask, size);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static void barrier() {
        __syncwarp();
    }
};

} // namespace peak_gemm::backend::cuda
