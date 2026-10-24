#pragma once

#include <cstdint>

#include "peak_gemm/core/config.hpp"

namespace peak_gemm::core {

template <uint32_t SwizzleBits = 3, uint32_t BaseBits = 3, uint32_t ShiftBits = 3>
struct XorSwizzle {
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE uint32_t operator()(uint32_t address) const {
        constexpr uint32_t swizzle_mask = ((1U << SwizzleBits) - 1U) << BaseBits;
        return ((address >> ShiftBits) & swizzle_mask) ^ address;
    }
};

struct IdentitySwizzle {
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE uint32_t operator()(uint32_t address) const {
        return address;
    }
};

} // namespace peak_gemm::core
