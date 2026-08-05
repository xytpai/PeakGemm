#pragma once

#include <cstdint>
#include <type_traits>

#include "peak_gemm/core/config.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::backend {

struct AsyncCopy {
    template <typename T>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static void copy(T *destination, const T *source) {
        constexpr int bytes = sizeof(T);
        const auto shared_destination =
            static_cast<uint32_t>(__cvta_generic_to_shared(destination));
        const auto global_source = reinterpret_cast<std::uint64_t>(source);
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], %2;\n" ::
                "r"(shared_destination),
            "l"(global_source), "n"(bytes));
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static void commit() {
        asm volatile("cp.async.commit_group;\n" ::);
    }

    template <int PendingGroups>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static void wait() {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(PendingGroups));
    }
};

template <uint32_t SwizzleBits = 3, uint32_t BaseBits = 3, uint32_t ShiftBits = 3>
struct Sm80Swizzle {
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE uint32_t operator()(uint32_t address) const {
        constexpr uint32_t swizzle_mask = ((1U << SwizzleBits) - 1U) << BaseBits;
        return ((address >> ShiftBits) & swizzle_mask) ^ address;
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE uint32_t operator()(uint32_t, uint32_t column) const {
        return (*this)(column);
    }
};

struct IdentitySwizzle {
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE uint32_t operator()(uint32_t address) const {
        return address;
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE uint32_t operator()(uint32_t, uint32_t column) const {
        return column;
    }
};

template <typename Scalar, typename Accumulator>
struct MmaM16N8K16 {
    static constexpr uint32_t m = 16, n = 8, k = 16;
    enum : uint32_t { M = m,
                      N = n,
                      K = k };
    using FragmentAT = core::Vector<Scalar, 8>;
    using FragmentBT = core::Vector<Scalar, 4>;
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
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                  "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
                  "f"(c[2]), "f"(c[3]));
        } else {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                  "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]),
                  "f"(c[2]), "f"(c[3]));
        }
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void reset_fragment_c(
        FragmentCT &fragment, Accumulator value = 0) const {
        fragment.fill(value);
    }

    template <typename Swizzle>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix_a(
        FragmentAT &fragment,
        Scalar *base,
        uint32_t row,
        uint32_t column,
        uint32_t stride,
        const Swizzle &swizzle) const {
        auto *registers = reinterpret_cast<uint32_t *>(&fragment);
        const auto offset = swizzle((row + lane_ % 16) * stride + column + lane_ / 16 * 8);
        const auto address =
            static_cast<uint32_t>(__cvta_generic_to_shared(base + offset));
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(registers[0]), "=r"(registers[1]),
              "=r"(registers[2]), "=r"(registers[3])
            : "r"(address));
    }

    template <typename Swizzle>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void load_matrix_b(
        FragmentBT &fragment,
        Scalar *base,
        uint32_t row,
        uint32_t column,
        uint32_t stride,
        const Swizzle &swizzle) const {
        const auto y = column + lane_ % 4 * 2;
        const auto x = row + lane_ / 4;
        const auto offset0 = swizzle(x * stride + y);
        const auto offset1 = swizzle(x * stride + y + 8);
        fragment.values[0] = base[offset0];
        fragment.values[1] = base[offset0 + 1];
        fragment.values[2] = base[offset1];
        fragment.values[3] = base[offset1 + 1];
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void store_matrix(
        Scalar *pointer, uint32_t stride, const FragmentCT &fragment) const {
        const auto y = lane_ / 4;
        const auto x = lane_ % 4 * 2;
        using Pair = core::Vector<Scalar, 2>;
        const Pair row0{{static_cast<Scalar>(fragment[0]), static_cast<Scalar>(fragment[1])}};
        const Pair row1{{static_cast<Scalar>(fragment[2]), static_cast<Scalar>(fragment[3])}};
        *reinterpret_cast<Pair *>(&pointer[y * stride + x]) = row0;
        *reinterpret_cast<Pair *>(&pointer[(y + 8) * stride + x]) = row1;
    }

private:
    uint32_t lane_ = 0;
};

template <typename Scalar, typename Accumulator>
using WmmaDefault = MmaM16N8K16<Scalar, Accumulator>;

} // namespace peak_gemm::backend
