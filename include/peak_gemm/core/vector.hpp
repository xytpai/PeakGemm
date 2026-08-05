#pragma once

#include <cstddef>
#include <type_traits>

#include "peak_gemm/core/config.hpp"

namespace peak_gemm::core {

template <typename T, std::size_t Size>
struct alignas(sizeof(T) * Size) Vector {
    static_assert(Size > 0, "Vector must contain at least one element");

    T values[Size];

    PEAKGEMM_HOST_DEVICE PEAKGEMM_FORCEINLINE T &operator[](std::size_t index) {
        return values[index];
    }

    PEAKGEMM_HOST_DEVICE PEAKGEMM_FORCEINLINE const T &operator[](std::size_t index) const {
        return values[index];
    }

    PEAKGEMM_HOST_DEVICE PEAKGEMM_FORCEINLINE void fill(T value) {
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
        for (std::size_t index = 0; index < Size; ++index) {
            values[index] = value;
        }
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load(const T *pointer) {
        *this = *reinterpret_cast<const Vector *>(pointer);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void store(T *pointer) const {
        *reinterpret_cast<Vector *>(pointer) = *this;
    }

    template <typename Input>
    PEAKGEMM_HOST_DEVICE PEAKGEMM_FORCEINLINE void convert_from(
        const Vector<Input, Size> &source,
        float scale = 1.0F) {
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
        for (std::size_t index = 0; index < Size; ++index) {
            if constexpr (std::is_same_v<T, Input>) {
                values[index] = source[index];
            } else {
                values[index] = static_cast<T>(static_cast<float>(source[index]) / scale);
            }
        }
    }

    PEAKGEMM_HOST_DEVICE PEAKGEMM_FORCEINLINE Vector &operator+=(
        const Vector &other) {
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
        for (std::size_t index = 0; index < Size; ++index) {
            values[index] += other.values[index];
        }
        return *this;
    }
};

} // namespace peak_gemm::core
