#pragma once

#include <cstddef>

#include "peak_gemm/core/config.hpp"

namespace peak_gemm::core {

template <typename Integer>
PEAKGEMM_HOST_DEVICE constexpr Integer ceil_div(
    Integer value,
    Integer divisor) {
    return (value + divisor - 1) / divisor;
}

template <std::size_t Value>
struct Log2 {
    static_assert(Value > 0, "Log2 requires a positive value");
    static constexpr std::size_t value =
        1 + Log2<(Value >> 1)>::value;
};

template <>
struct Log2<1> {
    static constexpr std::size_t value = 0;
};

} // namespace peak_gemm::core
