#pragma once

#include <cstdint>
#include <type_traits>

#include "peak_gemm/core/config.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::backend {

using floatx4_t = float __attribute__((__vector_size__(4 * sizeof(float))));
using fp16x2_t = __fp16 __attribute__((__vector_size__(2 * sizeof(__fp16))));
using bf16x2_t = __bf16 __attribute__((__vector_size__(2 * sizeof(__bf16))));
using fp16x4_t = __fp16 __attribute__((__vector_size__(4 * sizeof(__fp16))));
using bf16x4_t = __bf16 __attribute__((__vector_size__(4 * sizeof(__bf16))));
using fp16x8_t = __fp16 __attribute__((__vector_size__(8 * sizeof(__fp16))));
using bf16x8_t = __bf16 __attribute__((__vector_size__(8 * sizeof(__bf16))));

PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE floatx4_t mfma_bf16_m16n16k32(
    bf16x8_t &lhs, bf16x8_t &rhs, floatx4_t &accumulator) {
    asm volatile(
        "v_mfma_f32_16x16x32_bf16 %0, %1, %2, %0"
        : "+a"(accumulator)
        : "v"(lhs), "v"(rhs));
    return accumulator;
}

template <
    typename Scalar,
    typename Accumulator,
    bool UseSwizzle = true,
    std::uint32_t KBlocks16 = 0>
struct MfmaM16N16K32 {
    static constexpr std::uint32_t m = 16, n = 16, k = 32;
    enum : std::uint32_t { M = m,
                           N = n,
                           K = k };
    using FragmentAT = core::Vector<Scalar, 8>;
    using FragmentBT = core::Vector<Scalar, 8>;
    using FragmentCT = core::Vector<Accumulator, 4>;
    using ComputeT = Scalar;

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void init(std::uint32_t lane) {
        lane_ = lane;
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void operator()(
        FragmentCT &destination,
        const FragmentAT &lhs,
        const FragmentBT &rhs,
        const FragmentCT &accumulator) const {
        const auto *a = reinterpret_cast<const std::uint32_t *>(&lhs);
        const auto *b = reinterpret_cast<const std::uint32_t *>(&rhs);
        const auto *c = reinterpret_cast<const Accumulator *>(&accumulator);
        auto *d = reinterpret_cast<Accumulator *>(&destination);
        if constexpr (std::is_same_v<Scalar, __half>) {
            *reinterpret_cast<floatx4_t *>(d) =
                __builtin_amdgcn_mfma_f32_16x16x32_f16(
                    *reinterpret_cast<const fp16x8_t *>(a),
                    *reinterpret_cast<const fp16x8_t *>(b),
                    *reinterpret_cast<const floatx4_t *>(c), 0, 0, 0);
        } else {
            auto lhs_vector = *reinterpret_cast<const bf16x8_t *>(a);
            auto rhs_vector = *reinterpret_cast<const bf16x8_t *>(b);
            auto accumulator_vector = *reinterpret_cast<const floatx4_t *>(c);
            *reinterpret_cast<floatx4_t *>(d) =
                mfma_bf16_m16n16k32(lhs_vector, rhs_vector, accumulator_vector);
        }
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void reset_fragment_c(
        FragmentCT &fragment, Accumulator value = 0) const {
        fragment.fill(value);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE std::uint32_t swizzle(
        std::uint32_t row, std::uint32_t column) const {
        return (column * sizeof(Scalar) ^ row % KBlocks16 * 16) / sizeof(Scalar);
    }

    template <typename Fragment>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix(
        Fragment &fragment,
        Scalar *base,
        std::uint32_t row,
        std::uint32_t column,
        std::uint32_t stride) const {
        const auto target_row = row + lane_ % 16;
        auto target_column = column + lane_ / 16 * 8;
        if constexpr (UseSwizzle) target_column = swizzle(target_row, target_column);
        fragment = *reinterpret_cast<Fragment *>(
            base + target_row * stride + target_column);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix_a(
        FragmentAT &fragment, Scalar *base, std::uint32_t row,
        std::uint32_t column, std::uint32_t stride) const {
        load_matrix(fragment, base, row, column, stride);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix_b(
        FragmentBT &fragment, Scalar *base, std::uint32_t row,
        std::uint32_t column, std::uint32_t stride) const {
        load_matrix(fragment, base, row, column, stride);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void store_matrix(
        Scalar *pointer, std::uint32_t stride, const FragmentCT &fragment) const {
        const auto x = lane_ % 16;
        const auto y = lane_ / 16 * 4;
#pragma unroll
        for (std::uint32_t index = 0; index < 4; ++index) {
            pointer[(y + index) * stride + x] =
                static_cast<Scalar>(fragment[index]);
        }
    }

private:
    std::uint32_t lane_ = 0;
};

template <
    typename Scalar,
    typename Accumulator,
    bool UseSwizzle = true,
    std::uint32_t KBlocks16 = 0>
struct MfmaM16N16K16 {
    static constexpr std::uint32_t m = 16, n = 16, k = 16;
    enum : std::uint32_t { M = m,
                           N = n,
                           K = k };
    using FragmentAT = core::Vector<Scalar, 4>;
    using FragmentBT = core::Vector<Scalar, 4>;
    using FragmentCT = core::Vector<Accumulator, 4>;
    using ComputeT = Scalar;

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void init(std::uint32_t lane) {
        lane_ = lane;
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void operator()(
        FragmentCT &destination,
        const FragmentAT &lhs,
        const FragmentBT &rhs,
        const FragmentCT &accumulator) const {
        const auto *a = reinterpret_cast<const std::uint32_t *>(&lhs);
        const auto *b = reinterpret_cast<const std::uint32_t *>(&rhs);
        const auto *c = reinterpret_cast<const Accumulator *>(&accumulator);
        auto *d = reinterpret_cast<Accumulator *>(&destination);
        if constexpr (std::is_same_v<Scalar, __half>) {
            *reinterpret_cast<floatx4_t *>(d) =
                __builtin_amdgcn_mfma_f32_16x16x16f16(
                    *reinterpret_cast<const fp16x4_t *>(a),
                    *reinterpret_cast<const fp16x4_t *>(b),
                    *reinterpret_cast<const floatx4_t *>(c), 0, 0, 0);
        } else {
            *reinterpret_cast<floatx4_t *>(d) =
                __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                    *reinterpret_cast<const bf16x4_t *>(a),
                    *reinterpret_cast<const bf16x4_t *>(b),
                    *reinterpret_cast<const floatx4_t *>(c), 0, 0, 0);
        }
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void reset_fragment_c(
        FragmentCT &fragment, Accumulator value = 0) const {
        fragment.fill(value);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE std::uint32_t swizzle(
        std::uint32_t row, std::uint32_t column) const {
        return (column * sizeof(Scalar) ^ row % KBlocks16 * 16) / sizeof(Scalar);
    }

    template <typename Fragment>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix(
        Fragment &fragment,
        Scalar *base,
        std::uint32_t row,
        std::uint32_t column,
        std::uint32_t stride) const {
        const auto target_row = row + lane_ % 16;
        auto target_column = column + lane_ / 16 * 4;
        if constexpr (UseSwizzle) target_column = swizzle(target_row, target_column);
        fragment = *reinterpret_cast<Fragment *>(
            base + target_row * stride + target_column);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix_a(
        FragmentAT &fragment, Scalar *base, std::uint32_t row,
        std::uint32_t column, std::uint32_t stride) const {
        load_matrix(fragment, base, row, column, stride);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix_b(
        FragmentBT &fragment, Scalar *base, std::uint32_t row,
        std::uint32_t column, std::uint32_t stride) const {
        load_matrix(fragment, base, row, column, stride);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void store_matrix(
        Scalar *pointer, std::uint32_t stride, const FragmentCT &fragment) const {
        const auto x = lane_ % 16;
        const auto y = lane_ / 16 * 4;
#pragma unroll
        for (std::uint32_t index = 0; index < 4; ++index) {
            pointer[(y + index) * stride + x] =
                static_cast<Scalar>(fragment[index]);
        }
    }

private:
    std::uint32_t lane_ = 0;
};

template <typename Scalar, typename Accumulator, bool UseSwizzle = true>
using WmmaDefault = MfmaM16N16K32<Scalar, Accumulator, UseSwizzle>;

} // namespace peak_gemm::backend
