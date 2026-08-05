#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "peak_gemm/core/config.hpp"

namespace peak_gemm::core {

template <std::size_t Dim, typename Index = std::int64_t>
struct StridedLayout {
    static_assert(Dim > 0, "A layout must have at least one dimension");

    using index_type = Index;
    using coordinate_type = std::array<index_type, Dim>;

    coordinate_type strides;

    PEAKGEMM_HOST_DEVICE constexpr index_type operator()(
        const coordinate_type &coordinates) const {
        index_type offset = 0;
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
        for (std::size_t axis = 0; axis < Dim; ++axis) {
            offset += coordinates[axis] * strides[axis];
        }
        return offset;
    }

    template <typename... Coordinates>
    PEAKGEMM_HOST_DEVICE constexpr index_type operator()(
        Coordinates... coordinates) const {
        static_assert(
            sizeof...(Coordinates) == Dim,
            "The coordinate count must match the layout dimension");
        return (*this)(
            coordinate_type{static_cast<index_type>(coordinates)...});
    }

    PEAKGEMM_HOST_DEVICE constexpr index_type stride(
        std::size_t axis) const {
        return strides[axis];
    }
};

template <typename Layout>
struct TensorView {
    void *data;
    Layout layout;
};

} // namespace peak_gemm::core
