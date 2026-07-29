#pragma once

#include <cstdint>

#include "peak_gemm/core/config.hpp"

namespace peak_gemm::backend::hip {

struct Wave {
    static constexpr std::uint32_t size = 64;

    template <typename T>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static T shuffle(T value, int source_lane) {
        return __shfl(value, source_lane, size);
    }

    template <typename T>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static T shuffle_xor(T value, int lane_mask) {
        return __shfl_xor(value, lane_mask, size);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static void barrier() {
        __builtin_amdgcn_wave_barrier();
    }
};

} // namespace peak_gemm::backend::hip
