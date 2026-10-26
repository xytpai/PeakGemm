#pragma once

#include <cstdint>
#include <type_traits>

#include "peak_gemm/core/config.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::backend {

// CDNA5 / gfx1250 is wave32 and issues WMMA, not MFMA. v_wmma_f32_16x16x32_*
// reads 16 source elements per lane. Lane group 0 owns K {0..7, 16..23} and
// lane group 1 owns K {8..15, 24..31}; each VGPR holds two consecutive K.
#if defined(__gfx1250__) && defined(__HIP_DEVICE_COMPILE__)
using fp32x8_t = float __attribute__((ext_vector_type(8)));
using fp16x16_t = _Float16 __attribute__((ext_vector_type(16)));
using bf16x16_t = __bf16 __attribute__((ext_vector_type(16)));

PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE fp32x8_t wmma_f32_16x16x32_f16(
    fp16x16_t lhs, fp16x16_t rhs, fp32x8_t accumulator) {
    return __builtin_amdgcn_wmma_f32_16x16x32_f16(
        false, lhs, false, rhs, static_cast<short>(0), accumulator, false, false);
}

PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE fp32x8_t wmma_f32_16x16x32_bf16(
    bf16x16_t lhs, bf16x16_t rhs, fp32x8_t accumulator) {
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(
        false, lhs, false, rhs, static_cast<short>(0), accumulator, false, false);
}

// 5.03 PFLOPS is one 16x16x32 WMMA every 8 cycles on each of 1024 SIMDs at
// 2.4 GHz. The whole trip count stays in this block so the compiler cannot
// copy the accumulators between batches. Four independent accumulators cover
// the result latency, and SCHED_MODE[2] lets the wave issue the next WMMA
// before the previous one retires.
constexpr int kWmmaPeakBatch = 32;

#define PEAKGEMM_WMMA4(op) \
    #op " %[c0], %[a], %[b], 0\n\t" #op " %[c1], %[a], %[b], 0\n\t" #op " %[c2], %[a], %[b], 0\n\t" #op " %[c3], %[a], %[b], 0\n\t"

template <bool IsHalf>
PEAKGEMM_DEVICE void wmma_peak_asm(
    std::conditional_t<IsHalf, fp16x16_t, bf16x16_t> lhs,
    std::conditional_t<IsHalf, fp16x16_t, bf16x16_t> rhs,
    fp32x8_t &c0, fp32x8_t &c1, fp32x8_t &c2, fp32x8_t &c3, uint32_t trips) {
    uint32_t cnt = 0;
    if constexpr (IsHalf) {
        asm volatile(
            "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1\n\t"
            "s_mov_b32 %[cnt], %[trips]\n\t"
            "%=:\n\t" PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_f16)
                PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_f16)
                    PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_f16)
                        PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_f16)
                            PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_f16)
                                PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_f16)
                                    PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_f16)
                                        PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_f16) "s_sub_u32 %[cnt], %[cnt], 1\n\t"
                                                                                "s_cmp_eq_u32 %[cnt], 0\n\t"
                                                                                "s_cbranch_scc0 %=b\n\t"
            : [c0] "=v"(c0), [c1] "=v"(c1), [c2] "=v"(c2), [c3] "=v"(c3), [cnt] "=&s"(cnt)
            : [a] "v"(lhs), [b] "v"(rhs), [trips] "s"(trips)
            : "scc");
    } else {
        asm volatile(
            "s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1\n\t"
            "s_mov_b32 %[cnt], %[trips]\n\t"
            "%=:\n\t" PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_bf16)
                PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_bf16)
                    PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_bf16)
                        PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_bf16)
                            PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_bf16)
                                PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_bf16)
                                    PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_bf16)
                                        PEAKGEMM_WMMA4(v_wmma_f32_16x16x32_bf16) "s_sub_u32 %[cnt], %[cnt], 1\n\t"
                                                                                 "s_cmp_eq_u32 %[cnt], 0\n\t"
                                                                                 "s_cbranch_scc0 %=b\n\t"
            : [c0] "=v"(c0), [c1] "=v"(c1), [c2] "=v"(c2), [c3] "=v"(c3), [cnt] "=&s"(cnt)
            : [a] "v"(lhs), [b] "v"(rhs), [trips] "s"(trips)
            : "scc");
    }
}

#undef PEAKGEMM_WMMA4
#endif

template <typename Scalar, typename Accumulator>
struct WmmaM16N16K32 {
    static constexpr uint32_t m = 16, n = 16, k = 32;
    enum : uint32_t { M = m,
                      N = n,
                      K = k };
    using FragmentAT = core::Vector<Scalar, 16>;
    using FragmentBT = core::Vector<Scalar, 16>;
    using FragmentCT = core::Vector<Accumulator, 8>;
    using ComputeT = Scalar;

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void init(uint32_t lane) {
        lane_ = lane; // 0, 1, ..., warp_size - 1.
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void operator()(
        FragmentCT &destination,
        const FragmentAT &lhs,
        const FragmentBT &rhs,
        const FragmentCT &accumulator) const {
        static_assert(std::is_same_v<Accumulator, float>, "gfx1250 f16/bf16 WMMA accumulates f32");
#if defined(__gfx1250__) && defined(__HIP_DEVICE_COMPILE__)
        const auto c = *reinterpret_cast<const fp32x8_t *>(&accumulator);
        fp32x8_t d;
        if constexpr (std::is_same_v<Scalar, __half>) {
            d = wmma_f32_16x16x32_f16(
                *reinterpret_cast<const fp16x16_t *>(&lhs),
                *reinterpret_cast<const fp16x16_t *>(&rhs),
                c);
        } else {
            d = wmma_f32_16x16x32_bf16(
                *reinterpret_cast<const bf16x16_t *>(&lhs),
                *reinterpret_cast<const bf16x16_t *>(&rhs),
                c);
        }
        *reinterpret_cast<fp32x8_t *>(&destination) = d;
#else
        destination = accumulator;
        (void)lhs;
        (void)rhs;
#endif
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void reset_fragment_c(
        FragmentCT &fragment, Accumulator value = 0) const {
        fragment.fill(value);
    }

    template <typename Fragment, typename Swizzle>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix(
        Fragment &fragment,
        Scalar *base,
        uint32_t warp_row,
        uint32_t warp_column,
        uint32_t row_stride,
        const Swizzle &swizzle) const {
        using LoadT = core::Vector<Scalar, 8>;
        static_assert(sizeof(LoadT) == 16);
        auto *fragment_parts = reinterpret_cast<LoadT *>(&fragment);
        const auto target_row = warp_row + lane_ % 16;
        const auto target_column = warp_column + lane_ / 16 * 8;
        // Two 16B-aligned loads; swizzle must preserve each 8-element block.
#pragma unroll
        for (uint32_t index = 0; index < 2; ++index) {
            const auto offset = swizzle(target_row * row_stride + target_column + index * 16);
            fragment_parts[index] = *reinterpret_cast<const LoadT *>(base + offset);
        }
    }

    template <typename Swizzle>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix_a(
        FragmentAT &fragment, Scalar *base, uint32_t mma_m_begin,
        uint32_t mma_k_begin, uint32_t m_stride, const Swizzle &swizzle) const {
        load_matrix(fragment, base, mma_m_begin, mma_k_begin, m_stride, swizzle);
    }

    template <typename Swizzle>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix_b(
        FragmentBT &fragment, Scalar *base, uint32_t mma_n_begin,
        uint32_t mma_k_begin, uint32_t n_stride, const Swizzle &swizzle) const {
        load_matrix(fragment, base, mma_n_begin, mma_k_begin, n_stride, swizzle);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void store_matrix(
        Scalar *pointer, uint32_t m_stride, const FragmentCT &fragment) const {
        const auto x = lane_ % 16;
        const auto y = lane_ / 16 * 8;
#pragma unroll
        for (uint32_t index = 0; index < 8; ++index) {
            pointer[(y + index) * m_stride + x] =
                static_cast<Scalar>(fragment[index]);
        }
    }

private:
    uint32_t lane_ = 0;
};

template <typename Scalar, typename Accumulator>
using WmmaDefault = WmmaM16N16K32<Scalar, Accumulator>;

// Perform exactly Loop matrix instructions. Loop must be a multiple of
// kWmmaPeakBatch. C is the inline constant 0 either way; ZeroInputs selects
// whether A and B are zero registers or the loaded fragments.
template <typename Scalar, int Loop, bool ZeroInputs>
PEAKGEMM_DEVICE void wmma_peak_loop(
    const core::Vector<Scalar, 16> &a_frag,
    const core::Vector<Scalar, 16> &b_frag,
    core::Vector<float, 8> &c_frag) {
#if defined(__gfx1250__) && defined(__HIP_DEVICE_COMPILE__)
    static_assert(Loop % kWmmaPeakBatch == 0, "peak loop length must be a multiple of 32");
    constexpr bool is_half = std::is_same_v<Scalar, __half>;
    using input_vec_t = std::conditional_t<is_half, fp16x16_t, bf16x16_t>;
    input_vec_t a{};
    input_vec_t b{};
    if constexpr (!ZeroInputs) {
        a = *reinterpret_cast<const input_vec_t *>(&a_frag);
        b = *reinterpret_cast<const input_vec_t *>(&b_frag);
    }
    fp32x8_t c0{};
    fp32x8_t c1{};
    fp32x8_t c2{};
    fp32x8_t c3{};
    wmma_peak_asm<is_half>(a, b, c0, c1, c2, c3, static_cast<uint32_t>(Loop / kWmmaPeakBatch));
    *reinterpret_cast<fp32x8_t *>(&c_frag) = c0 + c1 + c2 + c3;
#else
    (void)a_frag;
    (void)b_frag;
    (void)c_frag;
#endif
}

} // namespace peak_gemm::backend
