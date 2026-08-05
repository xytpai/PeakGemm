#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "peak_gemm/core/config.hpp"

namespace peak_gemm::core {

template <uint32_t... Extents>
struct Shape {
    static_assert(sizeof...(Extents) > 0, "A shape must have at least one dimension");
    static_assert(((Extents > 0) && ...), "Shape extents must be positive");

    using extent_type = uint32_t;
    using coordinate_type = std::array<extent_type, sizeof...(Extents)>;

    static constexpr std::size_t dim = sizeof...(Extents);

    PEAKGEMM_HOST_DEVICE static constexpr extent_type extent(
        std::size_t axis) {
        return coordinate_type{Extents...}[axis];
    }

    template <std::size_t Axis>
    PEAKGEMM_HOST_DEVICE static constexpr extent_type get() {
        static_assert(Axis < dim, "Shape axis is out of bounds");
        return extent(Axis);
    }
};

} // namespace peak_gemm::core
