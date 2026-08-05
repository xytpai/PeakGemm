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
    std::uint32_t m_blocks,
    std::uint32_t n_blocks) {
    static_assert(GroupM >= 0);
    const std::uint32_t group_size = GroupM * n_blocks;
    const std::uint32_t first_m = block / group_size * GroupM;
    const std::uint32_t actual_group_m =
        m_blocks - first_m < GroupM ? m_blocks - first_m : GroupM;
    const std::uint32_t block_in_group = block % group_size;
    return {
        first_m + block_in_group % actual_group_m,
        block_in_group / actual_group_m};
}

} // namespace peak_gemm::core
