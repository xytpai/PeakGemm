#pragma once

#include <cstdint>
#include <type_traits>

#include "peak_gemm/core/config.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::backend {

using floatx4_t = float __attribute__((__vector_size__(4 * sizeof(float))));
using fp16x8_t = __fp16 __attribute__((__vector_size__(8 * sizeof(__fp16))));
using bf16x8_t = __bf16 __attribute__((__vector_size__(8 * sizeof(__bf16))));

template <uint32_t SwizzleBits = 3, uint32_t BaseBits = 3, uint32_t ShiftBits = 3>
struct Gfx950Swizzle {
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE uint32_t operator()(uint32_t address) const {
        constexpr uint32_t swizzle_mask = ((1U << SwizzleBits) - 1U) << BaseBits;
        return ((address >> ShiftBits) & swizzle_mask) ^ address;
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE uint32_t operator()(uint32_t, uint32_t column) const {
        return (*this)(column);
    }
};

struct Gfx950IdentitySwizzle {
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE uint32_t operator()(uint32_t address) const {
        return address;
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE uint32_t operator()(uint32_t, uint32_t column) const {
        return column;
    }
};

template <typename Scalar, typename Accumulator>
struct MfmaM16N16K32 {
    static constexpr uint32_t m = 16, n = 16, k = 32;
    enum : uint32_t { M = m,
                      N = n,
                      K = k };
    using FragmentAT = core::Vector<Scalar, 8>;
    using FragmentBT = core::Vector<Scalar, 8>;
    using FragmentCT = core::Vector<Accumulator, 4>;
    using ComputeT = Scalar;

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void init(uint32_t lane) {
        lane_ = lane;
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void operator()(
        FragmentCT &destination,
        const FragmentAT &lhs,
        const FragmentBT &rhs,
        const FragmentCT &accumulator) const {
        const auto *a = reinterpret_cast<const uint32_t *>(&lhs);
        const auto *b = reinterpret_cast<const uint32_t *>(&rhs);
        const auto *c = reinterpret_cast<const Accumulator *>(&accumulator);
        auto *d = reinterpret_cast<Accumulator *>(&destination);
        if constexpr (std::is_same_v<Scalar, __half>) {
            *reinterpret_cast<floatx4_t *>(d) =
                __builtin_amdgcn_mfma_f32_16x16x32_f16(
                    *reinterpret_cast<const fp16x8_t *>(a),
                    *reinterpret_cast<const fp16x8_t *>(b),
                    *reinterpret_cast<const floatx4_t *>(c), 0, 0, 0);
        } else {
            *reinterpret_cast<floatx4_t *>(d) =
                __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                    *reinterpret_cast<const bf16x8_t *>(a),
                    *reinterpret_cast<const bf16x8_t *>(b),
                    *reinterpret_cast<const floatx4_t *>(c), 0, 0, 0);
        }
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void reset_fragment_c(
        FragmentCT &fragment, Accumulator value = 0) const {
        fragment.fill(value);
    }

    template <typename Fragment, typename Swizzle>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix(
        Fragment &fragment,
        Scalar *base,
        uint32_t row,
        uint32_t column,
        uint32_t stride,
        const Swizzle &swizzle) const {
        const auto target_row = row + lane_ % 16;
        const auto offset = swizzle(target_row * stride + column + lane_ / 16 * 8);
        fragment = *reinterpret_cast<Fragment *>(base + offset);
    }

    template <typename Swizzle>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix_a(
        FragmentAT &fragment, Scalar *base, uint32_t row,
        uint32_t column, uint32_t stride, const Swizzle &swizzle) const {
        load_matrix(fragment, base, row, column, stride, swizzle);
    }

    template <typename Swizzle>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix_b(
        FragmentBT &fragment, Scalar *base, uint32_t row,
        uint32_t column, uint32_t stride, const Swizzle &swizzle) const {
        load_matrix(fragment, base, row, column, stride, swizzle);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void store_matrix(
        Scalar *pointer, uint32_t stride, const FragmentCT &fragment) const {
        const auto x = lane_ % 16;
        const auto y = lane_ / 16 * 4;
#pragma unroll
        for (uint32_t index = 0; index < 4; ++index) {
            pointer[(y + index) * stride + x] =
                static_cast<Scalar>(fragment[index]);
        }
    }

private:
    uint32_t lane_ = 0;
};

template <typename Scalar, typename Accumulator>
using WmmaDefault = MfmaM16N16K32<Scalar, Accumulator>;

} // namespace peak_gemm::backend
