#pragma once

#include "device_common.h"

using namespace kernel_utils;

namespace wmma_utils {

#ifdef __CUDACC__

template <typename scalar_t, typename acc_t, bool USE_SWIZZLE = true>
struct WMMA_M16N8K16 {
    enum {
        M = 16,
        N = 8,
        K = 16,
    };
    using FragmentAT = aligned_array<scalar_t, 8>;
    using FragmentBT = aligned_array<scalar_t, 4>;
    using FragmentCT = aligned_array<acc_t, 4>;
    using ComputeT = scalar_t;

    __device__ __forceinline__ WMMA_M16N8K16() {
    }

    __device__ __forceinline__ void init(int w_tid_) {
        w_tid = w_tid_;
    }

    __device__ __forceinline__ void operator()(
        FragmentCT &d,
        FragmentAT const &a,
        FragmentBT const &b,
        FragmentCT const &c) {
        uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
        uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
        acc_t const *C = reinterpret_cast<acc_t const *>(&c);
        acc_t *D = reinterpret_cast<acc_t *>(&d);
        if constexpr (std::is_same_v<scalar_t, __half>) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                "{%10,%11,%12,%13};\n"
                : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                  "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
        } else {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                "{%10,%11,%12,%13};\n"
                : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]),
                  "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
        }
    }

    __device__ __forceinline__ void reset_fragment_c(FragmentCT &c, acc_t val = 0) {
        c.val[0] = val;
        c.val[1] = val;
        c.val[2] = val;
        c.val[3] = val;
    }

    template <uint32_t VEC_BITS = 3>
    __device__ __forceinline__ uint32_t swizzle(uint32_t addr) {
        constexpr uint32_t COL_BITS = 7 - 4; // 32*4B (7bits) - 16B (4bits)
        constexpr uint32_t COL_MASK = ((1 << COL_BITS) - 1) << VEC_BITS;
        return ((addr >> VEC_BITS) & COL_MASK) ^ addr;
    }

    __device__ __forceinline__ void load_matrix_a(FragmentAT &a, scalar_t *base_ptr, int soffset, int stride) {
        auto A = reinterpret_cast<uint32_t *>(&a);
        uint32_t offset_ = soffset + (w_tid % 16) * stride + (w_tid / 16) * 8;
        if constexpr (USE_SWIZZLE) {
            offset_ = swizzle(offset_);
        }
        auto addr = (uint32_t)__cvta_generic_to_shared(base_ptr + offset_);
        asm volatile(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
            : "r"(addr));
    }

    __device__ __forceinline__ void load_matrix_b(FragmentBT &b, scalar_t *base_ptr, int soffset, int stride) {
        auto B = reinterpret_cast<uint32_t *>(&b);
        auto y = w_tid % 4 * 2;
        auto x = w_tid / 4;
        uint32_t offset0 = soffset + x * stride + y;
        uint32_t offset1 = soffset + x * stride + y + 8;
        if constexpr (USE_SWIZZLE) {
            offset0 = swizzle(offset0);
            offset1 = swizzle(offset1);
        }
        auto ptr0 = base_ptr + offset0;
        auto ptr1 = base_ptr + offset1;
        b.val[0] = ptr0[0];
        b.val[1] = ptr0[1];
        b.val[2] = ptr1[0];
        b.val[3] = ptr1[1];
    }

    __device__ __forceinline__ void store_matrix(scalar_t *ptr, int stride, FragmentCT const &c) {
        auto y = w_tid / 4;
        auto x = w_tid % 4 * 2;
        using vec_t = aligned_array<scalar_t, 2>;
        vec_t vec0, vec1;
        vec0.val[0] = (acc_t)c.val[0];
        vec0.val[1] = (acc_t)c.val[1];
        vec1.val[0] = (acc_t)c.val[2];
        vec1.val[1] = (acc_t)c.val[3];
        *reinterpret_cast<vec_t *>(&ptr[y * stride + x]) = vec0;
        *reinterpret_cast<vec_t *>(&ptr[(y + 8) * stride + x]) = vec1;
    }

public:
    int w_tid;
};

#elif defined(__HIPCC__)

typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;
typedef __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16 fp16x8_t;
typedef __attribute__((__vector_size__(8 * sizeof(__bf16)))) __bf16 bf16x8_t;

template <typename scalar_t, typename acc_t, bool USE_SWIZZLE = true>
struct WMMA_M16N16K32 {
    enum {
        M = 16,
        N = 16,
        K = 32,
    };
    using FragmentAT = aligned_array<scalar_t, 8>;
    using FragmentBT = aligned_array<scalar_t, 8>;
    using FragmentCT = aligned_array<acc_t, 4>;
    using ComputeT = scalar_t;

    __device__ __forceinline__ WMMA_M16N16K32() {
    }

    __device__ __forceinline__ void init(int w_tid_) {
        w_tid = w_tid_;
    }

    __device__ __forceinline__ void operator()(
        FragmentCT &d,
        FragmentAT const &a,
        FragmentBT const &b,
        FragmentCT const &c) {
        uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
        uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
        acc_t const *C = reinterpret_cast<acc_t const *>(&c);
        acc_t *D = reinterpret_cast<acc_t *>(&d);
        if constexpr (std::is_same_v<scalar_t, __half>) {
            *(floatx4_t *)D = __builtin_amdgcn_mfma_f32_16x16x32_f16(
                (*(fp16x8_t *)A),
                (*(fp16x8_t *)B),
                *(floatx4_t *)C,
                0, 0, 0);
        } else {
            *(floatx4_t *)D = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
                (*(bf16x8_t *)A),
                (*(bf16x8_t *)B),
                *(floatx4_t *)C,
                0, 0, 0);
        }
    }

    __device__ __forceinline__ void reset_fragment_c(FragmentCT &c, acc_t val = 0) {
        c.val[0] = val;
        c.val[1] = val;
        c.val[2] = val;
        c.val[3] = val;
    }

    template <uint32_t VEC_BITS = 3>
    __device__ __forceinline__ uint32_t swizzle(uint32_t addr) {
        // constexpr uint32_t COL_BITS = 7 - 4; // 32*4B (7bits) - 16B (4bits)
        constexpr uint32_t COL_BITS = 4; // 32*4B (7bits) - 16B (4bits)
        constexpr uint32_t COL_MASK = ((1 << COL_BITS) - 1) << VEC_BITS;
        return ((addr >> COL_BITS) & COL_MASK) ^ addr;
    }

    __device__ __forceinline__ void load_matrix_a(FragmentAT &a, scalar_t *base_ptr, int soffset, int stride) {
        auto x = w_tid / 16 * 8;
        auto y = w_tid % 16;
        uint32_t offset_ = soffset + y * stride + x;
        if constexpr (USE_SWIZZLE) {
            offset_ = swizzle(offset_);
        }
        a = *reinterpret_cast<FragmentAT *>(base_ptr + offset_);
    }

    __device__ __forceinline__ void load_matrix_b(FragmentBT &b, scalar_t *base_ptr, int soffset, int stride) {
        auto x = w_tid / 16 * 8;
        auto y = w_tid % 16;
        uint32_t offset_ = soffset + y * stride + x;
        if constexpr (USE_SWIZZLE) {
            offset_ = swizzle(offset_);
        }
        b = *reinterpret_cast<FragmentBT *>(base_ptr + offset_);
    }

    __device__ __forceinline__ void store_matrix(scalar_t *ptr, int stride, FragmentCT const &c) {
        auto x = w_tid % 16;
        auto y_begin = w_tid / 16 * 4;
        ptr[(y_begin + 0) * stride + x] = (scalar_t)c.val[0];
        ptr[(y_begin + 1) * stride + x] = (scalar_t)c.val[1];
        ptr[(y_begin + 2) * stride + x] = (scalar_t)c.val[2];
        ptr[(y_begin + 3) * stride + x] = (scalar_t)c.val[3];
    }

public:
    int w_tid;
};

#endif

} // namespace wmma_utils
