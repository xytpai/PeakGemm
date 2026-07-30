#pragma once

#include <cstdint>

#include "peak_gemm/core/config.hpp"

namespace peak_gemm::core {

struct BlockCoordinate {
    std::uint32_t m;
    std::uint32_t n;
};

template <std::uint32_t GroupM>
PEAKGEMM_HOST_DEVICE constexpr BlockCoordinate block_swizzle(
    std::uint32_t block,
    std::uint32_t blocks_m,
    std::uint32_t blocks_n) {
    static_assert(GroupM > 0, "Block swizzle group must be positive");
    const std::uint32_t group_size = GroupM * blocks_n;
    const std::uint32_t first_m = block / group_size * GroupM;
    const std::uint32_t actual_group_m =
        blocks_m - first_m < GroupM ? blocks_m - first_m : GroupM;
    const std::uint32_t block_in_group = block % group_size;
    return {
        first_m + block_in_group % actual_group_m,
        block_in_group / actual_group_m};
}

} // namespace peak_gemm::core
