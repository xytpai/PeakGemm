#pragma once

#include "device_common.h"

using namespace kernel_utils;
using namespace wmma_utils;

namespace hgemm {

template <typename T>
__device__ __forceinline__ void atomic_pack_add_scalar(T *pk_dst, T *pk_src) {
    // static_assert(false);
}

#ifdef __CUDACC__
#define LAUNCH_CONFIG __launch_bounds__(BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE, 2)
template <>
__device__ __forceinline__ void atomic_pack_add_scalar<__bfloat16>(__bfloat16 *pk_dst, __bfloat16 *pk_src) {
    atomicAdd(&pk_dst[0], pk_src[0]);
    atomicAdd(&pk_dst[1], pk_src[1]);
}
template <>
__device__ __forceinline__ void atomic_pack_add_scalar<__half>(__half *pk_dst, __half *pk_src) {
    atomicAdd(&pk_dst[0], pk_src[0]);
    atomicAdd(&pk_dst[1], pk_src[1]);
}
__device__ __forceinline__ void sched_barrier() {
}
#endif

#ifdef __HIPCC__
#define LAUNCH_CONFIG __attribute__((amdgpu_waves_per_eu(2, 2), amdgpu_flat_work_group_size(BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE, BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE)))
__device__ __forceinline__ void sched_barrier() {
    __builtin_amdgcn_sched_barrier(0);
}
template <>
__device__ __forceinline__ void atomic_pack_add_scalar<__bfloat16>(__bfloat16 *pk_dst, __bfloat16 *pk_src) {
    auto dst = reinterpret_cast<bf16x2_t *>(pk_dst);
    bf16x2_t val = *reinterpret_cast<bf16x2_t *>(pk_src);
    __builtin_amdgcn_global_atomic_fadd_v2bf16(dst, val);
}
template <>
__device__ __forceinline__ void atomic_pack_add_scalar<__half>(__half *pk_dst, __half *pk_src) {
    auto dst = reinterpret_cast<fp16x2_t *>(pk_dst);
    fp16x2_t val = *reinterpret_cast<fp16x2_t *>(pk_src);
    __builtin_amdgcn_global_atomic_fadd_v2f16(dst, val);
}
template <uint32_t P>
__device__ __forceinline__ void hip_s_setprio() {
    asm volatile("s_setprio %0" ::"n"(P));
}
__device__ __forceinline__ void hip_s_barrier() {
    asm volatile("s_barrier");
}
#endif

#define SPLIT_K_SEMAPHORE_MAX_LEN 256

template <
    typename scalar_t,
    typename WMMAT,
    uint32_t WARP_SIZE,
    uint32_t BLOCK_K,
    uint32_t BLOCK_M_WARPS,
    uint32_t BLOCK_N_WARPS,
    uint32_t BLOCK_K_WARPS,
    uint32_t WARP_M_STEPS,
    uint32_t WARP_N_STEPS>
struct BlockTile {
    using FragmentAT = typename WMMAT::FragmentAT;
    using FragmentBT = typename WMMAT::FragmentBT;
    using FragmentCT = typename WMMAT::FragmentCT;
    enum {
        WARP_MASK = WARP_SIZE - 1,
        WARP_SHIFT = Log2<WARP_SIZE>::VALUE,
        WARP_ATOM_M = WMMAT::M,
        WARP_ATOM_N = WMMAT::N,
        WARP_ATOM_K = WMMAT::K,
        WARP_GROUP_K = BLOCK_K_WARPS * WARP_ATOM_K,
        WARP_K_STEPS = BLOCK_K / WARP_GROUP_K,
        K_SLICE = BLOCK_K / BLOCK_K_WARPS,
        LDG_VEC_SIZE = 16 / sizeof(scalar_t),
        BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE,
        BLOCK_MN_WARPS = BLOCK_M_WARPS * BLOCK_N_WARPS,
        WARP_M = WARP_M_STEPS * WARP_ATOM_M,
        WARP_N = WARP_N_STEPS * WARP_ATOM_N,
        BLOCK_M = BLOCK_M_WARPS * WARP_M,
        BLOCK_N = BLOCK_N_WARPS * WARP_N,
        BLOCK_MK_SIZE = BLOCK_M * BLOCK_K,
        BLOCK_NK_SIZE = BLOCK_N * BLOCK_K,
        LDG_A_X_THREADS = BLOCK_K / LDG_VEC_SIZE,
        LDG_B_X_THREADS = BLOCK_K / LDG_VEC_SIZE,
        LDG_REG_A_COUNT = BLOCK_MK_SIZE / LDG_VEC_SIZE / BLOCK_THREADS,
        LDG_REG_B_COUNT = BLOCK_NK_SIZE / LDG_VEC_SIZE / BLOCK_THREADS,
        DMA_BYTES = 16,
        BLOCK_DMA_STRIDE = BLOCK_THREADS * DMA_BYTES,
    };
    static_assert(LDG_REG_A_COUNT >= 1 && LDG_REG_B_COUNT >= 1);
    static_assert(BLOCK_K % WARP_GROUP_K == 0 && WARP_K_STEPS >= 1 && K_SLICE % WARP_ATOM_K == 0);
    using ldg_vec_t = aligned_array<scalar_t, LDG_VEC_SIZE>;

    __device__ __forceinline__ BlockTile(uint32_t tid, uint32_t a_stride, uint32_t b_stride) :
        tid(tid), wid(tid >> WARP_SHIFT), w_tid(tid & WARP_MASK),
        ldg_a_vec_idx(tid % LDG_A_X_THREADS),
        ldg_b_vec_idx(tid % LDG_B_X_THREADS) {
        wmma.init(w_tid);
#pragma unroll
        for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
#pragma unroll
            for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                wmma.reset_fragment_c(fo[mi][ni]);
            }
        }
        wid_mn = wid % BLOCK_MN_WARPS;
        wid_k = wid / BLOCK_MN_WARPS;
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_A_COUNT; ++i) {
            uint32_t col = ldg_a_vec_idx * LDG_VEC_SIZE;
            uint32_t row = (BLOCK_THREADS * i + tid) / LDG_A_X_THREADS;
            swizzle_cache_a[i] = row * a_stride + wmma.swizzle(row, col);
        }
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_B_COUNT; ++i) {
            uint32_t col = ldg_b_vec_idx * LDG_VEC_SIZE;
            uint32_t row = (BLOCK_THREADS * i + tid) / LDG_B_X_THREADS;
            swizzle_cache_b[i] = row * b_stride + wmma.swizzle(row, col);
        }
    }

#ifdef __CUDACC__
    __device__ __forceinline__ void ldg_copy_async(
        scalar_t *as, scalar_t *bs,
        const scalar_t *a, uint32_t a_stride, const scalar_t *b, uint32_t b_stride) {
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_A_COUNT; i++) {
            uint32_t tid_ = BLOCK_THREADS * i + tid;
            uint32_t soffset = wmma.swizzle(tid_ * LDG_VEC_SIZE);
            CopyAsync::add(
                reinterpret_cast<ldg_vec_t *>(as + soffset),
                &(reinterpret_cast<ldg_vec_t *>(
                    const_cast<scalar_t *>(a) + (tid_ / LDG_A_X_THREADS) * a_stride)[ldg_a_vec_idx]));
        }
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_B_COUNT; i++) {
            uint32_t tid_ = BLOCK_THREADS * i + tid;
            auto soffset = wmma.swizzle(tid_ * LDG_VEC_SIZE);
            CopyAsync::add(
                reinterpret_cast<ldg_vec_t *>(bs + soffset),
                &(reinterpret_cast<ldg_vec_t *>(
                    const_cast<scalar_t *>(b) + (tid_ / LDG_B_X_THREADS) * b_stride)[ldg_b_vec_idx]));
        }
    }
#endif

#ifdef __HIPCC__

    __device__ __forceinline__ void init_hip(scalar_t *as, scalar_t *bs) {
        as_ = __builtin_amdgcn_readfirstlane(reinterpret_cast<uintptr_t>(as) + (wid * WARP_SIZE * DMA_BYTES));
        bs_ = __builtin_amdgcn_readfirstlane(reinterpret_cast<uintptr_t>(bs) + (wid * WARP_SIZE * DMA_BYTES));
    }

    __device__ __forceinline__ void ldg_copy_async(
        uint32_t as_offset, uint32_t bs_offset,
        i32x4 &a_rsrc, uint32_t a_begin, uint32_t a_stride, i32x4 &b_rsrc, uint32_t b_begin, uint32_t b_stride,
        uint32_t m_bound, uint32_t n_bound, uint32_t k_bound) {
        uint32_t as_warp_ = as_ + as_offset * sizeof(scalar_t);
        uint32_t bs_warp_ = bs_ + bs_offset * sizeof(scalar_t);
        uint32_t a_col = ldg_a_vec_idx * LDG_VEC_SIZE;
        uint32_t b_col = ldg_b_vec_idx * LDG_VEC_SIZE;
        a_col = a_col < k_bound ? a_col : 0;
        b_col = b_col < k_bound ? b_col : 0;
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_A_COUNT; i++) {
            uint32_t row = (BLOCK_THREADS * i + tid) / LDG_A_X_THREADS;
            uint32_t global_offset = a_begin + (row < m_bound ? row : 0) * a_stride + wmma.swizzle(row, a_col);
            llvm_amdgcn_raw_buffer_load_lds(
                a_rsrc,
                (as3_uint32_ptr) static_cast<uintptr_t>(as_warp_ + i * BLOCK_DMA_STRIDE),
                DMA_BYTES,
                global_offset * sizeof(scalar_t),
                0,
                0,
                0);
        }
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_B_COUNT; i++) {
            uint32_t row = (BLOCK_THREADS * i + tid) / LDG_B_X_THREADS;
            uint32_t global_offset = b_begin + (row < n_bound ? row : 0) * b_stride + wmma.swizzle(row, b_col);
            llvm_amdgcn_raw_buffer_load_lds(
                b_rsrc,
                (as3_uint32_ptr) static_cast<uintptr_t>(bs_warp_ + i * BLOCK_DMA_STRIDE),
                DMA_BYTES,
                global_offset * sizeof(scalar_t),
                0,
                0,
                0);
        }
    }

    template <uint32_t i>
    __device__ __forceinline__ void ldg_copy_async_a(uint32_t as_offset, i32x4 &a_rsrc, uint32_t a_begin, uint32_t a_stride) {
        uint32_t as_warp_ = as_ + as_offset * sizeof(scalar_t);
        uint32_t global_offset = a_begin + swizzle_cache_a[i];
        llvm_amdgcn_raw_buffer_load_lds(
            a_rsrc,
            (as3_uint32_ptr) static_cast<uintptr_t>(as_warp_ + i * BLOCK_DMA_STRIDE),
            DMA_BYTES,
            global_offset * sizeof(scalar_t),
            0,
            0,
            0);
    }

    template <uint32_t i>
    __device__ __forceinline__ void ldg_copy_async_b(uint32_t bs_offset, i32x4 &b_rsrc, uint32_t b_begin, uint32_t b_stride) {
        uint32_t bs_warp_ = bs_ + bs_offset * sizeof(scalar_t);
        uint32_t global_offset = b_begin + swizzle_cache_b[i];
        llvm_amdgcn_raw_buffer_load_lds(
            b_rsrc,
            (as3_uint32_ptr) static_cast<uintptr_t>(bs_warp_ + i * BLOCK_DMA_STRIDE),
            DMA_BYTES,
            global_offset * sizeof(scalar_t),
            0,
            0,
            0);
    }

#endif

    template <uint32_t S = 0>
    __device__ __forceinline__ void wait() {
        CopyAsync::wait<S>();
    }

    __device__ __forceinline__ void commit() {
        CopyAsync::commit();
    }

    template <bool C_SHUFFLE = false, bool USE_ATOMIC = false>
    __device__ __forceinline__ void store_matrix(scalar_t *ptr, scalar_t (&cs)[BLOCK_K_WARPS][BLOCK_M * BLOCK_N], uint32_t block_m_idx, uint32_t block_n_idx, uint32_t m, uint32_t n) {
        uint32_t warp_m_begin = wid_mn / BLOCK_N_WARPS * WARP_M;
        uint32_t warp_n_begin = wid_mn % BLOCK_N_WARPS * WARP_N;
        if constexpr (!C_SHUFFLE) {
            uint32_t warp_m_begin = wid_mn / BLOCK_N_WARPS * WARP_M;
            uint32_t warp_n_begin = wid_mn % BLOCK_N_WARPS * WARP_N;
#pragma unroll
            for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
                uint32_t warp_atom_offset_m = warp_m_begin + mi * WARP_ATOM_M;
                uint32_t m_global_idx = block_m_idx * BLOCK_M + warp_atom_offset_m;
#pragma unroll
                for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                    uint32_t warp_atom_offset_n = warp_n_begin + ni * WARP_ATOM_N;

                    uint32_t n_global_idx = block_n_idx * BLOCK_N + warp_atom_offset_n;

                    auto dst_ptr = ptr + m_global_idx * n + n_global_idx;
                    wmma.store_matrix(dst_ptr, n, fo[mi][ni]);
                }
            }
        } else {
            __syncthreads();
#pragma unroll
            for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
                uint32_t warp_atom_offset_m = warp_m_begin + mi * WARP_ATOM_M;
#pragma unroll
                for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                    uint32_t warp_atom_offset_n = warp_n_begin + ni * WARP_ATOM_N;
                    auto ptr_ = &cs[wid_k][warp_atom_offset_m * BLOCK_N + warp_atom_offset_n];
                    wmma.store_matrix(ptr_, BLOCK_N, fo[mi][ni]);
                }
            }
            __syncthreads();
            constexpr uint32_t LDG_REG_C_COUNT = BLOCK_M * BLOCK_N / (BLOCK_THREADS * LDG_VEC_SIZE);
            constexpr uint32_t LDG_C_X_THREADS = BLOCK_N / LDG_VEC_SIZE;
#pragma unroll
            for (uint32_t i = 0; i < LDG_REG_C_COUNT; ++i) {
                uint32_t global_tid = BLOCK_THREADS * i + tid;
                uint32_t m_local_idx = global_tid / LDG_C_X_THREADS;
                uint32_t n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE;
                uint32_t m_global_idx = block_m_idx * BLOCK_M + m_local_idx;
                uint32_t n_global_idx = block_n_idx * BLOCK_N + n_local_idx;
                if (m_global_idx < m && n_global_idx < n) {
                    auto src = *reinterpret_cast<ldg_vec_t *>(&cs[wid_k][m_local_idx * BLOCK_N + n_local_idx]);
                    if constexpr (BLOCK_K_WARPS > 1) {
#pragma unroll
                        for (uint32_t k_warp = 1; k_warp < BLOCK_K_WARPS; ++k_warp) {
                            src += *reinterpret_cast<ldg_vec_t *>(&cs[k_warp][m_local_idx * BLOCK_N + n_local_idx]);
                        }
                    }
                    auto dst_ptr = ptr + m_global_idx * n + n_global_idx;
                    if constexpr (USE_ATOMIC) {
#pragma unroll
                        for (uint32_t pk_idx = 0; pk_idx < LDG_VEC_SIZE; pk_idx += 2) {
                            atomic_pack_add_scalar<scalar_t>(&dst_ptr[pk_idx], &src[pk_idx]);
                        }
                    } else {
                        *reinterpret_cast<ldg_vec_t *>(dst_ptr) = src;
                    }
                }
            }
        }
    }

    __device__ __forceinline__ void compute_tile_streaming(scalar_t *as, scalar_t *bs) {
        uint32_t warp_m_begin = wid_mn / BLOCK_N_WARPS * WARP_M;
        uint32_t warp_n_begin = wid_mn % BLOCK_N_WARPS * WARP_N;
        uint32_t warp_k_slice_base = wid_k * K_SLICE;
#pragma unroll
        for (uint32_t ki = 0; ki < WARP_K_STEPS; ++ki) {
            uint32_t k_col = warp_k_slice_base + ki * WARP_ATOM_K;
            FragmentBT b_frag[WARP_N_STEPS];
            FragmentAT a_frag[WARP_M_STEPS];
#pragma unroll
            for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                uint32_t warp_atom_offset_n = warp_n_begin + ni * WARP_ATOM_N;
                wmma.load_matrix_b(
                    b_frag[ni],
                    bs,
                    warp_atom_offset_n,
                    k_col,
                    BLOCK_K);
            }
            sched_barrier();
#pragma unroll
            for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
                uint32_t warp_atom_offset_m = warp_m_begin + mi * WARP_ATOM_M;
                wmma.load_matrix_a(
                    a_frag[mi],
                    as,
                    warp_atom_offset_m,
                    k_col,
                    BLOCK_K);
            }
            sched_barrier();
#pragma unroll
            for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
#pragma unroll
                for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                    wmma(
                        fo[mi][ni],
                        a_frag[mi],
                        b_frag[ni],
                        fo[mi][ni]);
                }
            }
        }
    }

    __device__ __forceinline__ void ldg_compute_tile_streaming(
        scalar_t *as, scalar_t *bs,
        uint32_t as_offset, uint32_t bs_offset,
        i32x4 &a_rsrc, uint32_t a_begin, uint32_t a_stride, i32x4 &b_rsrc, uint32_t b_begin, uint32_t b_stride,
        uint32_t m_bound, uint32_t n_bound, uint32_t k_bound) {
        uint32_t as_warp_ = as_ + as_offset * sizeof(scalar_t);
        uint32_t bs_warp_ = bs_ + bs_offset * sizeof(scalar_t);
        uint32_t a_col = ldg_a_vec_idx * LDG_VEC_SIZE;
        uint32_t b_col = ldg_b_vec_idx * LDG_VEC_SIZE;
        a_col = a_col < k_bound ? a_col : 0;
        b_col = b_col < k_bound ? b_col : 0;
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_A_COUNT; i++) {
            uint32_t row = (BLOCK_THREADS * i + tid) / LDG_A_X_THREADS;
            uint32_t global_offset = a_begin + (row < m_bound ? row : 0) * a_stride + wmma.swizzle(row, a_col);
            llvm_amdgcn_raw_buffer_load_lds(
                a_rsrc,
                (as3_uint32_ptr) static_cast<uintptr_t>(as_warp_ + i * BLOCK_DMA_STRIDE),
                DMA_BYTES,
                global_offset * sizeof(scalar_t),
                0,
                0,
                0);
        }
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_B_COUNT; i++) {
            uint32_t row = (BLOCK_THREADS * i + tid) / LDG_B_X_THREADS;
            uint32_t global_offset = b_begin + (row < n_bound ? row : 0) * b_stride + wmma.swizzle(row, b_col);
            llvm_amdgcn_raw_buffer_load_lds(
                b_rsrc,
                (as3_uint32_ptr) static_cast<uintptr_t>(bs_warp_ + i * BLOCK_DMA_STRIDE),
                DMA_BYTES,
                global_offset * sizeof(scalar_t),
                0,
                0,
                0);
        }
        uint32_t warp_m_begin = wid_mn / BLOCK_N_WARPS * WARP_M;
        uint32_t warp_n_begin = wid_mn % BLOCK_N_WARPS * WARP_N;
        uint32_t warp_k_slice_base = wid_k * K_SLICE;
#pragma unroll
        for (uint32_t ki = 0; ki < WARP_K_STEPS; ++ki) {
            uint32_t k_col = warp_k_slice_base + ki * WARP_ATOM_K;
            FragmentBT b_frag[WARP_N_STEPS];
            FragmentAT a_frag[WARP_M_STEPS];
#pragma unroll
            for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                uint32_t warp_atom_offset_n = warp_n_begin + ni * WARP_ATOM_N;
                wmma.load_matrix_b(
                    b_frag[ni],
                    bs,
                    warp_atom_offset_n,
                    k_col,
                    BLOCK_K);
            }
            sched_barrier();
#pragma unroll
            for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
                uint32_t warp_atom_offset_m = warp_m_begin + mi * WARP_ATOM_M;
                wmma.load_matrix_a(
                    a_frag[mi],
                    as,
                    warp_atom_offset_m,
                    k_col,
                    BLOCK_K);
            }
            sched_barrier();
#pragma unroll
            for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
#pragma unroll
                for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                    wmma(
                        fo[mi][ni],
                        a_frag[mi],
                        b_frag[ni],
                        fo[mi][ni]);
                }
            }
        }
    }

    __device__ __forceinline__ void ldg_compute_tile_streaming_ex(
        scalar_t *as, scalar_t *bs,
        uint32_t as_offset, uint32_t bs_offset,
        i32x4 &a_rsrc, uint32_t a_begin, uint32_t a_stride,
        i32x4 &b_rsrc, uint32_t b_begin, uint32_t b_stride) {
        constexpr uint32_t M_HALF_STEPS = WARP_M_STEPS / 2;
        constexpr uint32_t N_HALF_STEPS = WARP_N_STEPS / 2;
        static_assert(M_HALF_STEPS >= 1);
        static_assert(N_HALF_STEPS >= 1);
        // static_assert(LDG_REG_A_COUNT == 4);
        // static_assert(LDG_REG_B_COUNT == 4);
        uint32_t warp_m_begin = wid_mn / BLOCK_N_WARPS * WARP_M;
        uint32_t warp_n_begin = wid_mn % BLOCK_N_WARPS * WARP_N;
        uint32_t warp_k_slice_base = wid_k * K_SLICE;

#pragma unroll
        for (uint32_t ki = 0; ki < WARP_K_STEPS; ++ki) {
            uint32_t k_col = warp_k_slice_base + ki * WARP_ATOM_K;
            FragmentBT b0_frag[N_HALF_STEPS];
            FragmentBT b1_frag[N_HALF_STEPS];
            FragmentAT a0_frag[M_HALF_STEPS];
            FragmentAT a1_frag[M_HALF_STEPS];
            if (ki == 0) {
                ldg_copy_async_b<0>(bs_offset, b_rsrc, b_begin, b_stride);
                ldg_copy_async_b<1>(bs_offset, b_rsrc, b_begin, b_stride);
            }
#pragma unroll
            for (uint32_t ni = 0; ni < N_HALF_STEPS; ++ni) {
                wmma.load_matrix_b(b0_frag[ni], bs, warp_n_begin + ni * WARP_ATOM_N, k_col, BLOCK_K);
            }
#pragma unroll
            for (uint32_t mi = 0; mi < M_HALF_STEPS; ++mi) {
                wmma.load_matrix_a(a0_frag[mi], as, warp_m_begin + mi * WARP_ATOM_M, k_col, BLOCK_K);
            }
            sched_barrier();
            hip_s_setprio<1>();
#pragma unroll
            for (uint32_t mi = 0; mi < M_HALF_STEPS; ++mi) {
#pragma unroll
                for (uint32_t ni = 0; ni < N_HALF_STEPS; ++ni) {
                    wmma(
                        fo[mi][ni],
                        a0_frag[mi],
                        b0_frag[ni],
                        fo[mi][ni]);
                }
            }
            sched_barrier();
            hip_s_setprio<0>();
            if (ki == 0) {
                ldg_copy_async_a<0>(as_offset, a_rsrc, a_begin, a_stride);
                ldg_copy_async_a<1>(as_offset, a_rsrc, a_begin, a_stride);
            }
#pragma unroll
            for (uint32_t ni = 0; ni < N_HALF_STEPS; ++ni) {
                wmma.load_matrix_b(b1_frag[ni], bs, warp_n_begin + (N_HALF_STEPS + ni) * WARP_ATOM_N, k_col, BLOCK_K);
            }
            sched_barrier();
            hip_s_setprio<1>();
#pragma unroll
            for (uint32_t mi = 0; mi < M_HALF_STEPS; ++mi) {
#pragma unroll
                for (uint32_t ni = 0; ni < N_HALF_STEPS; ++ni) {
                    wmma(
                        fo[mi][N_HALF_STEPS + ni],
                        a0_frag[mi],
                        b1_frag[ni],
                        fo[mi][N_HALF_STEPS + ni]);
                }
            }
            sched_barrier();
            hip_s_setprio<0>();
            if (ki == 0) {
                ldg_copy_async_b<2>(bs_offset, b_rsrc, b_begin, b_stride);
                ldg_copy_async_b<3>(bs_offset, b_rsrc, b_begin, b_stride);
            }
#pragma unroll
            for (uint32_t mi = 0; mi < M_HALF_STEPS; ++mi) {
                wmma.load_matrix_a(a1_frag[mi], as, warp_m_begin + (M_HALF_STEPS + mi) * WARP_ATOM_M, k_col, BLOCK_K);
            }
            sched_barrier();
            hip_s_setprio<1>();
#pragma unroll
            for (uint32_t mi = 0; mi < M_HALF_STEPS; ++mi) {
#pragma unroll
                for (uint32_t ni = 0; ni < N_HALF_STEPS; ++ni) {
                    wmma(
                        fo[M_HALF_STEPS + mi][ni],
                        a1_frag[mi],
                        b0_frag[ni],
                        fo[M_HALF_STEPS + mi][ni]);
                }
            }

            if (ki == 0) {
                sched_barrier();
                hip_s_setprio<0>();
                ldg_copy_async_a<2>(as_offset, a_rsrc, a_begin, a_stride);
                ldg_copy_async_a<3>(as_offset, a_rsrc, a_begin, a_stride);
                sched_barrier();
                hip_s_setprio<1>();
            }
#pragma unroll
            for (uint32_t mi = 0; mi < M_HALF_STEPS; ++mi) {
#pragma unroll
                for (uint32_t ni = 0; ni < N_HALF_STEPS; ++ni) {
                    wmma(
                        fo[M_HALF_STEPS + mi][N_HALF_STEPS + ni],
                        a1_frag[mi],
                        b1_frag[ni],
                        fo[M_HALF_STEPS + mi][N_HALF_STEPS + ni]);
                }
            }
            if (ki == 0) {
                sched_barrier();
                hip_s_setprio<0>();
            }
        }
    }

private:
    uint32_t tid;
    uint32_t wid;
    uint32_t w_tid;
    uint32_t ldg_a_vec_idx;
    uint32_t ldg_b_vec_idx;
    uint32_t wid_mn;
    uint32_t wid_k;
    WMMAT wmma;
    FragmentCT fo[WARP_M_STEPS][WARP_N_STEPS];
    uint32_t swizzle_cache_a[LDG_REG_A_COUNT];
    uint32_t swizzle_cache_b[LDG_REG_B_COUNT];
#ifdef __HIPCC__
    uint32_t as_;
    uint32_t bs_;
#endif
};

template <uint32_t CNT = 0, bool BARRIER = true>
__device__ __forceinline__ void __barrier() {
#ifdef __HIPCC__
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(%0)" ::"n"(CNT));
    if constexpr (BARRIER) {
        __builtin_amdgcn_s_barrier();
    }
    __builtin_amdgcn_sched_barrier(0);
#else
    __syncthreads();
#endif
}

template <uint32_t BLOCK_M, uint32_t BLOCK_N, bool L2_SW = false>
__device__ __forceinline__ void get_tile_mn(uint32_t m, uint32_t n, uint32_t &mi, uint32_t &ni) {
    uint32_t bn = (n + BLOCK_N - 1) / BLOCK_N;
#ifdef __CUDACC__
    mi = blockIdx.x / bn;
    ni = blockIdx.x % bn;
#endif
#ifdef __HIPCC__
    if constexpr (L2_SW) {
        uint32_t bm = (m + BLOCK_M - 1) / BLOCK_M;
        uint32_t pid = blockIdx.x;
        constexpr uint32_t NUM_XCDS = 8;
        constexpr uint32_t NUM_PID_M_IN_GROUP = 4;
        constexpr uint32_t CU_NUM = 256;
        // if (gridDim.x % NUM_XCDS != 0) {
        if (true) {
            mi = pid / bn;
            ni = pid % bn;
            return;
        }
        uint32_t intra_xcd_id = pid / NUM_XCDS;
        uint32_t xcd_id = pid % NUM_XCDS;
        uint32_t num_pids_in_xcd = (gridDim.x + NUM_XCDS - 1) / NUM_XCDS;
        uint32_t swizzled_pid = xcd_id * num_pids_in_xcd + intra_xcd_id;
        uint32_t num_pid_in_group = NUM_PID_M_IN_GROUP * bn;
        uint32_t group_id = swizzled_pid / num_pid_in_group;
        uint32_t intra_group_id = swizzled_pid % num_pid_in_group;
        uint32_t first_pid_m = group_id * NUM_PID_M_IN_GROUP;
        uint32_t group_size_m = min(bm - first_pid_m, NUM_PID_M_IN_GROUP);
        ni = intra_group_id / group_size_m;
        mi = first_pid_m + intra_group_id % group_size_m;
    } else {
        mi = blockIdx.x / bn;
        ni = blockIdx.x % bn;
    }
#endif
}

template <typename scalar_t, uint32_t STAGES, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t BLOCK_K_WARPS>
union SharedStorage {
    struct {
        scalar_t as[STAGES][BLOCK_M * BLOCK_K];
        scalar_t bs[STAGES][BLOCK_N * BLOCK_K];
    };
    scalar_t cs[BLOCK_K_WARPS][BLOCK_M * BLOCK_N];
};

template <uint32_t STAGES, uint32_t COPY_INSTS_PER_STAGE, uint32_t S, typename bt_t, typename sm_t>
__device__ void run_epilogue_stage(bt_t &block_tile, sm_t &smem, uint32_t &current_stage) {
    __barrier<(STAGES - 2 - S) * COPY_INSTS_PER_STAGE>();
    block_tile.compute_tile_streaming(
        smem.as[current_stage],
        smem.bs[current_stage]);
    current_stage = (current_stage + 1) % STAGES;
}

template <uint32_t STAGES, uint32_t COPY_INSTS_PER_STAGE, uint32_t S, uint32_t END, typename bt_t, typename sm_t>
__device__ void run_epilogue_stages(bt_t &block_tile, sm_t &smem, uint32_t &current_stage) {
    if constexpr (S < END) {
        run_epilogue_stage<STAGES, COPY_INSTS_PER_STAGE, S, bt_t, sm_t>(block_tile, smem, current_stage);
        run_epilogue_stages<STAGES, COPY_INSTS_PER_STAGE, S + 1, END, bt_t, sm_t>(block_tile, smem, current_stage);
    }
}

template <
    typename scalar_t,
    typename WMMAT,
    uint32_t WARP_SIZE,
    uint32_t BLOCK_K,
    uint32_t BLOCK_M_WARPS,
    uint32_t BLOCK_N_WARPS,
    uint32_t BLOCK_K_WARPS,
    uint32_t WARP_M_STEPS,
    uint32_t WARP_N_STEPS,
    uint32_t STAGES,
    uint32_t SPLIT_K>
LAUNCH_CONFIG __global__ void hgemm_kernel(
    scalar_t *c,
    const scalar_t *a,
    const scalar_t *b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k,
    uint32_t *semaphore,
    uint32_t *signal) {
    using BlockTileT = BlockTile<scalar_t, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_M_STEPS, WARP_N_STEPS>;
    constexpr uint32_t BLOCK_M = BlockTileT::BLOCK_M;
    constexpr uint32_t BLOCK_N = BlockTileT::BLOCK_N;
    constexpr bool IS_SPLIT_K = SPLIT_K > 1;

    uint32_t tid = threadIdx.x;
    uint32_t ks = (k + SPLIT_K - 1) / SPLIT_K;
    uint32_t ks_idx = blockIdx.y;
    uint32_t ks_begin = ks_idx * ks;
    uint32_t mi, ni;
    get_tile_mn<BLOCK_M, BLOCK_N, !IS_SPLIT_K>(m, n, mi, ni);
    uint32_t m_offset = mi * BLOCK_M;
    uint32_t n_offset = ni * BLOCK_N;
    uint32_t m_remain = (m_offset < m) ? (m - m_offset) : 0;
    uint32_t n_remain = (n_offset < n) ? (n - n_offset) : 0;
    uint32_t signal_idx;
    if constexpr (IS_SPLIT_K) {
        signal_idx = blockIdx.x;
    }

    __shared__ SharedStorage<scalar_t, STAGES, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_K_WARPS> smem;

    BlockTileT block_tile(tid, k, k);
    uint32_t current_stage = 0;
    uint32_t a_begin = m_offset * k + ks_begin;
    uint32_t b_begin = n_offset * k + ks_begin;
    uint32_t a_end = std::min(a_begin + ks, a_begin + k - ks_begin);
    uint32_t k_remain = std::min(BLOCK_K, a_end - a_begin);

    if constexpr (IS_SPLIT_K) {
        if (ks_idx == 0) {
            // zero c
            constexpr uint32_t LDG_VEC_SIZE = BlockTileT::LDG_VEC_SIZE;
            constexpr uint32_t BLOCK_THREADS = BlockTileT::BLOCK_THREADS;
            constexpr uint32_t LDG_REG_C_COUNT = BLOCK_M * BLOCK_N / (BLOCK_THREADS * LDG_VEC_SIZE);
            constexpr uint32_t LDG_C_X_THREADS = BLOCK_N / LDG_VEC_SIZE;
            using ldg_vec_t = aligned_array<scalar_t, LDG_VEC_SIZE>;
            ldg_vec_t zeros;
#pragma unroll
            for (int i = 0; i < LDG_VEC_SIZE; ++i) {
                zeros.val[i] = 0;
            }
#pragma unroll
            for (int i = 0; i < LDG_REG_C_COUNT; ++i) {
                uint32_t global_tid = BLOCK_THREADS * i + tid;
                uint32_t m_local_idx = global_tid / LDG_C_X_THREADS;
                uint32_t n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE;
                uint32_t row_idx = m_offset + m_local_idx;
                uint32_t col_idx = n_offset + n_local_idx;
                if (row_idx < m && col_idx < n) {
                    *reinterpret_cast<ldg_vec_t *>(&c[row_idx * n + col_idx]) = zeros; // bypass l2 write
                }
            }
            __threadfence();
            __syncthreads();
            // trigger signal when zeroc is done by the first block
            if (tid == 0) {
                signal[signal_idx] = 1;
                __threadfence();
            }
            __syncthreads();
        }
    }

#ifdef __CUDACC__

#pragma unroll
    for (uint32_t s = 0; s < STAGES - 1; ++s) {
        block_tile.ldg_copy_async(smem.as[s], smem.bs[s], &a[a_begin + s * BLOCK_K], k, &b[b_begin + s * BLOCK_K], k);
        block_tile.commit();
    }
    for (; a_begin < a_end; a_begin += BLOCK_K, b_begin += BLOCK_K) {
        block_tile.template wait<STAGES - 2>();
        __syncthreads();
        block_tile.load_matrix(smem.as[current_stage], smem.bs[current_stage]);
        block_tile();
        if (a_begin + (STAGES - 1) * BLOCK_K < a_end) {
            uint32_t write_stage = (current_stage + STAGES - 1) % STAGES;
            block_tile.ldg_copy_async(
                smem.as[write_stage],
                smem.bs[write_stage],
                &a[a_begin + (STAGES - 1) * BLOCK_K], k,
                &b[b_begin + (STAGES - 1) * BLOCK_K], k);
        }
        block_tile.commit();
        current_stage = (current_stage + 1) % STAGES;
    }

#endif

#ifdef __HIPCC__

    auto a_rsrc = make_srsrc(a, /*range_bytes*/ 0xFFFFFFFFu);
    auto b_rsrc = make_srsrc(b, /*range_bytes*/ 0xFFFFFFFFu);
    block_tile.init_hip(&smem.as[0][0], &smem.bs[0][0]);

    constexpr uint32_t COPY_INSTS_PER_STAGE = BlockTileT::LDG_REG_A_COUNT + BlockTileT::LDG_REG_B_COUNT;
#pragma unroll
    for (uint32_t s = 0; s < STAGES - 1; ++s) {
        block_tile.ldg_copy_async(
            s * BLOCK_M * BLOCK_K,
            s * BLOCK_N * BLOCK_K,
            a_rsrc, a_begin + s * BLOCK_K, k,
            b_rsrc, b_begin + s * BLOCK_K, k,
            m_remain, n_remain, k_remain);
    }
    for (; a_begin < a_end - (STAGES - 1) * BLOCK_K; a_begin += BLOCK_K, b_begin += BLOCK_K) {
        __barrier<(STAGES - 2) * COPY_INSTS_PER_STAGE>();
        uint32_t write_stage = (current_stage + STAGES - 1) % STAGES;
        if constexpr (BLOCK_M == 256 && BLOCK_N == 256 && BLOCK_K == 64) {
            block_tile.ldg_compute_tile_streaming_ex(
                smem.as[current_stage], smem.bs[current_stage],
                write_stage * BLOCK_M * BLOCK_K,
                write_stage * BLOCK_N * BLOCK_K,
                a_rsrc, a_begin + (STAGES - 1) * BLOCK_K, k,
                b_rsrc, b_begin + (STAGES - 1) * BLOCK_K, k);
        } else {
            block_tile.ldg_compute_tile_streaming(
                smem.as[current_stage], smem.bs[current_stage],
                write_stage * BLOCK_M * BLOCK_K,
                write_stage * BLOCK_N * BLOCK_K,
                a_rsrc, a_begin + (STAGES - 1) * BLOCK_K, k,
                b_rsrc, b_begin + (STAGES - 1) * BLOCK_K, k,
                m_remain, n_remain, k_remain);
        }
        current_stage = (current_stage + 1) % STAGES;
    }
    run_epilogue_stages<STAGES, COPY_INSTS_PER_STAGE, 0, STAGES - 1>(block_tile, smem, current_stage);

#endif

    if constexpr (IS_SPLIT_K) {
        // spin-wait until signal triggered
        if (tid == 0) {
            while (*reinterpret_cast<uint32_t volatile *>(&signal[signal_idx]) == 0) {}
        }
        __syncthreads();
        // clean semaphore and signal if this is the last block within split-k group
        if (tid == 0) {
            uint32_t arrive_idx = atomicAdd(&semaphore[signal_idx], static_cast<uint32_t>(1));
            if (arrive_idx == SPLIT_K - 1) {
                semaphore[signal_idx] = 0;
                signal[signal_idx] = 0;
            }
        }
        block_tile.template store_matrix<true, true>(c, smem.cs, mi, ni, m, n);
    } else {
        block_tile.template store_matrix<true, false>(c, smem.cs, mi, ni, m, n);
    }
}

template <
    typename scalar_t,
    typename WMMAT,
    uint32_t WARP_SIZE,
    uint32_t BLOCK_K,
    uint32_t BLOCK_M,
    uint32_t BLOCK_N,
    uint32_t BLOCK_M_WARPS,
    uint32_t BLOCK_N_WARPS>
struct BlockTileHT {
    using FragmentAT = typename WMMAT::FragmentAT;
    using FragmentBT = typename WMMAT::FragmentBT;
    using FragmentCT = typename WMMAT::FragmentCT;

    enum {
        WARP_MASK = WARP_SIZE - 1,
        WARP_SHIFT = Log2<WARP_SIZE>::VALUE,
        HALF_BLOCK_M = BLOCK_M / 2,
        HALF_BLOCK_N = BLOCK_N / 2,
        WARP_ATOM_M = WMMAT::M,
        WARP_ATOM_N = WMMAT::N,
        WARP_ATOM_K = WMMAT::K,
        WARP_M_STEPS = HALF_BLOCK_M / BLOCK_M_WARPS / WARP_ATOM_M,
        WARP_N_STEPS = HALF_BLOCK_N / BLOCK_N_WARPS / WARP_ATOM_N,
        WARP_K_STEPS = BLOCK_K / WARP_ATOM_K,
        WARP_M = WARP_M_STEPS * WARP_ATOM_M,
        WARP_N = WARP_N_STEPS * WARP_ATOM_N,
        A_FRAGS_LEN = WARP_K_STEPS * WARP_M_STEPS,
        B_FRAGS_LEN = WARP_K_STEPS * WARP_N_STEPS,
        A_FRAG_VALUES = sizeof(FragmentAT) / sizeof(scalar_t),
        B_FRAG_VALUES = sizeof(FragmentBT) / sizeof(scalar_t),
        HALF_BLOCK_MK_SIZE = HALF_BLOCK_M * BLOCK_K,
        HALF_BLOCK_NK_SIZE = HALF_BLOCK_N * BLOCK_K,
        LDG_VEC_SIZE = 16 / sizeof(scalar_t),
        LDG_X_THREADS = BLOCK_K / LDG_VEC_SIZE,
        BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE,
        LDG_REG_A_COUNT = HALF_BLOCK_MK_SIZE / BLOCK_THREADS / LDG_VEC_SIZE,
        LDG_REG_B_COUNT = HALF_BLOCK_NK_SIZE / BLOCK_THREADS / LDG_VEC_SIZE,
        DMA_BYTES = 16,
        BLOCK_DMA_STRIDE = BLOCK_THREADS * DMA_BYTES,
    };
    using ldg_vec_t = aligned_array<scalar_t, LDG_VEC_SIZE>;
    static_assert(LDG_REG_A_COUNT >= 1 && LDG_REG_B_COUNT >= 1);

    __device__ __forceinline__ BlockTileHT(uint32_t tid, uint32_t k) :
        tid(tid), wid(tid >> WARP_SHIFT), w_tid(tid & WARP_MASK),
        ldg_vec_idx(tid % LDG_X_THREADS), k(k) {
        wmma.init(w_tid);
#pragma unroll
        for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
#pragma unroll
            for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                wmma.reset_fragment_c(fo[0][0][mi][ni]);
                wmma.reset_fragment_c(fo[0][1][mi][ni]);
                wmma.reset_fragment_c(fo[1][0][mi][ni]);
                wmma.reset_fragment_c(fo[1][1][mi][ni]);
            }
        }
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_A_COUNT; ++i) {
            uint32_t col = ldg_vec_idx * LDG_VEC_SIZE;
            uint32_t row = (BLOCK_THREADS * i + tid) / LDG_X_THREADS;
            swizzle_cache_a[i] = row * k + wmma.swizzle(row, col);
        }
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_B_COUNT; ++i) {
            uint32_t col = ldg_vec_idx * LDG_VEC_SIZE;
            uint32_t row = (BLOCK_THREADS * i + tid) / LDG_X_THREADS;
            swizzle_cache_b[i] = row * k + wmma.swizzle(row, col);
        }
    }

    __device__ __forceinline__ void ldmatrix_a(scalar_t *as) {
        uint32_t warp_m_begin = wid / BLOCK_N_WARPS * WARP_M;
#pragma unroll
        for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
            uint32_t warp_atom_offset_m = warp_m_begin + mi * WARP_ATOM_M;
#pragma unroll
            for (uint32_t ki = 0; ki < WARP_K_STEPS; ++ki) {
                uint32_t row = warp_atom_offset_m;
                uint32_t col = ki * WARP_ATOM_K;
                wmma.load_matrix_a(fa[ki][mi], as, row, col, BLOCK_K);
            }
        }
    }

    template <uint32_t buffer_id>
    __device__ __forceinline__ void ldmatrix_b(scalar_t *bs) {
        uint32_t warp_n_begin = wid % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
            uint32_t warp_atom_offset_n = warp_n_begin + ni * WARP_ATOM_N;
#pragma unroll
            for (uint32_t ki = 0; ki < WARP_K_STEPS; ++ki) {
                uint32_t row = warp_atom_offset_n;
                uint32_t col = ki * WARP_ATOM_K;
                wmma.load_matrix_b(fb[buffer_id][ki][ni], bs, row, col, BLOCK_K);
            }
        }
    }

    template <uint32_t m_, uint32_t n_>
    __device__ __forceinline__ void consume() {
#pragma unroll
        for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
#pragma unroll
            for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
#pragma unroll
                for (uint32_t ki = 0; ki < WARP_K_STEPS; ++ki) {
                    wmma(fo[m_][n_][mi][ni], fa[ki][mi], fb[n_][ki][ni], fo[m_][n_][mi][ni]);
                }
            }
        }
    }

#ifdef __HIPCC__

    __device__ __forceinline__ void init_hip(scalar_t *as, scalar_t *bs) {
        as_ = __builtin_amdgcn_readfirstlane(reinterpret_cast<uintptr_t>(as) + (wid * WARP_SIZE * DMA_BYTES));
        bs_ = __builtin_amdgcn_readfirstlane(reinterpret_cast<uintptr_t>(bs) + (wid * WARP_SIZE * DMA_BYTES));
    }

    __device__ __forceinline__ void ldg_copy_async_a(uint32_t as_offset, i32x4 &a_rsrc, uint32_t a_begin) {
        uint32_t as_warp_ = as_ + as_offset * sizeof(scalar_t);
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_A_COUNT; ++i) {
            uint32_t global_offset = a_begin + swizzle_cache_a[i];
            llvm_amdgcn_raw_buffer_load_lds(
                a_rsrc,
                (as3_uint32_ptr) static_cast<uintptr_t>(as_warp_ + i * BLOCK_DMA_STRIDE),
                DMA_BYTES,
                global_offset * sizeof(scalar_t),
                0,
                0,
                0);
        }
    }

    __device__ __forceinline__ void ldg_copy_async_b(uint32_t bs_offset, i32x4 &b_rsrc, uint32_t b_begin) {
        uint32_t bs_warp_ = bs_ + bs_offset * sizeof(scalar_t);
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_B_COUNT; ++i) {
            uint32_t global_offset = b_begin + swizzle_cache_b[i];
            llvm_amdgcn_raw_buffer_load_lds(
                b_rsrc,
                (as3_uint32_ptr) static_cast<uintptr_t>(bs_warp_ + i * BLOCK_DMA_STRIDE),
                DMA_BYTES,
                global_offset * sizeof(scalar_t),
                0,
                0,
                0);
        }
    }

#endif

    template <bool C_SHUFFLE = false>
    __device__ __forceinline__ void store_matrix(scalar_t *ptr, scalar_t (&cs)[BLOCK_M][BLOCK_N], uint32_t block_m_idx, uint32_t block_n_idx, uint32_t m, uint32_t n) {
        uint32_t warp_m_begin = wid / BLOCK_N_WARPS * WARP_M;
        uint32_t warp_n_begin = wid % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (uint32_t m_ = 0; m_ < 2; ++m_) {
#pragma unroll
            for (uint32_t n_ = 0; n_ < 2; ++n_) {
                if constexpr (!C_SHUFFLE) {
#pragma unroll
                    for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
                        uint32_t warp_atom_offset_m = warp_m_begin + mi * WARP_ATOM_M;
                        uint32_t m_global_idx = block_m_idx * BLOCK_M + m_ * HALF_BLOCK_M + warp_atom_offset_m;
#pragma unroll
                        for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                            uint32_t warp_atom_offset_n = warp_n_begin + ni * WARP_ATOM_N;
                            uint32_t n_global_idx = block_n_idx * BLOCK_N + n_ * HALF_BLOCK_N + warp_atom_offset_n;
                            auto dst_ptr = ptr + m_global_idx * n + n_global_idx;
                            wmma.store_matrix(dst_ptr, n, fo[m_][n_][mi][ni]);
                        }
                    }
                } else {
#pragma unroll
                    for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
                        uint32_t warp_atom_offset_m = warp_m_begin + mi * WARP_ATOM_M;
#pragma unroll
                        for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                            uint32_t warp_atom_offset_n = warp_n_begin + ni * WARP_ATOM_N;
                            auto ptr_ = &cs[m_ * HALF_BLOCK_M + warp_atom_offset_m][n_ * HALF_BLOCK_N + warp_atom_offset_n];
                            wmma.store_matrix(ptr_, BLOCK_N, fo[m_][n_][mi][ni]);
                        }
                    }
                }
            }
        }

        if constexpr (C_SHUFFLE) {
            __syncthreads();
            constexpr uint32_t LDG_REG_C_COUNT = BLOCK_M * BLOCK_N / (BLOCK_THREADS * LDG_VEC_SIZE);
            constexpr uint32_t LDG_C_X_THREADS = BLOCK_N / LDG_VEC_SIZE;
#pragma unroll
            for (uint32_t i = 0; i < LDG_REG_C_COUNT; ++i) {
                uint32_t global_tid = BLOCK_THREADS * i + tid;
                uint32_t m_local_idx = global_tid / LDG_C_X_THREADS;
                uint32_t n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE;
                uint32_t m_global_idx = block_m_idx * BLOCK_M + m_local_idx;
                uint32_t n_global_idx = block_n_idx * BLOCK_N + n_local_idx;
                if (m_global_idx < m && n_global_idx < n) {
                    auto src = *reinterpret_cast<ldg_vec_t *>(&cs[m_local_idx][n_local_idx]);
                    auto dst_ptr = ptr + m_global_idx * n + n_global_idx;
                    *reinterpret_cast<ldg_vec_t *>(dst_ptr) = src;
                }
            }
        }
    }

    template <uint32_t MPart, uint32_t NPart>
    __device__ __forceinline__ void store_matrix_to_lds_mi(scalar_t (&cs)[BLOCK_M][BLOCK_N], uint32_t mi_step) {
        uint32_t warp_m_begin = wid / BLOCK_N_WARPS * WARP_M;
        uint32_t warp_n_begin = wid % BLOCK_N_WARPS * WARP_N;
        uint32_t warp_atom_offset_m = warp_m_begin + mi_step * WARP_ATOM_M;
#pragma unroll
        for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
            uint32_t warp_atom_offset_n = warp_n_begin + ni * WARP_ATOM_N;
            auto ptr_ = &cs[MPart * HALF_BLOCK_M + warp_atom_offset_m][NPart * HALF_BLOCK_N + warp_atom_offset_n];
            wmma.store_matrix(ptr_, BLOCK_N, fo[MPart][NPart][mi_step][ni]);
        }
    }

    template <uint32_t MPart, uint32_t NPart>
    __device__ __forceinline__ void store_matrix_from_lds_mi(
        scalar_t *ptr,
        scalar_t (&cs)[BLOCK_M][BLOCK_N],
        uint32_t block_m_idx,
        uint32_t block_n_idx,
        uint32_t m,
        uint32_t n,
        uint32_t mi_step) {
        constexpr uint32_t LDG_REG_C_COUNT =
            BLOCK_M_WARPS * WARP_ATOM_M * HALF_BLOCK_N / (BLOCK_THREADS * LDG_VEC_SIZE);
        constexpr uint32_t LDG_C_X_THREADS = HALF_BLOCK_N / LDG_VEC_SIZE;
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_C_COUNT; ++i) {
            uint32_t global_tid = BLOCK_THREADS * i + tid;
            uint32_t m_band_idx = global_tid / LDG_C_X_THREADS;
            uint32_t n_local_idx = global_tid % LDG_C_X_THREADS * LDG_VEC_SIZE;
            uint32_t warp_m_band = m_band_idx / WARP_ATOM_M;
            uint32_t atom_m_idx = m_band_idx % WARP_ATOM_M;
            uint32_t m_local_idx = MPart * HALF_BLOCK_M + warp_m_band * WARP_M + mi_step * WARP_ATOM_M + atom_m_idx;
            uint32_t n_local_idx_full = NPart * HALF_BLOCK_N + n_local_idx;
            uint32_t m_global_idx = block_m_idx * BLOCK_M + m_local_idx;
            uint32_t n_global_idx = block_n_idx * BLOCK_N + n_local_idx_full;
            if (m_global_idx < m && n_global_idx < n) {
                auto src = *reinterpret_cast<ldg_vec_t *>(&cs[m_local_idx][n_local_idx_full]);
                auto dst_ptr = ptr + m_global_idx * n + n_global_idx;
                *reinterpret_cast<ldg_vec_t *>(dst_ptr) = src;
            }
        }
    }

private:
    uint32_t tid;
    uint32_t wid;
    uint32_t w_tid;
    uint32_t ldg_vec_idx;
    uint32_t k;
    WMMAT wmma;
    FragmentAT fa[WARP_K_STEPS][WARP_M_STEPS];
    FragmentBT fb[2][WARP_K_STEPS][WARP_N_STEPS];
    FragmentCT fo[2][2][WARP_M_STEPS][WARP_N_STEPS];
    uint32_t swizzle_cache_a[LDG_REG_A_COUNT];
    uint32_t swizzle_cache_b[LDG_REG_B_COUNT];
#ifdef __HIPCC__
    uint32_t as_;
    uint32_t bs_;
#endif
};

template <typename scalar_t, uint32_t HALF_BLOCK_M, uint32_t HALF_BLOCK_N, uint32_t BLOCK_K>
union SharedHTStorage {
    struct {
        scalar_t as[2][2][HALF_BLOCK_M][BLOCK_K];
        scalar_t bs[2][2][HALF_BLOCK_N][BLOCK_K];
    };
    scalar_t cs[2 * HALF_BLOCK_M][2 * HALF_BLOCK_N];
};

template <
    typename scalar_t,
    typename WMMAT,
    uint32_t WARP_SIZE,
    uint32_t BLOCK_K,
    uint32_t BLOCK_M,
    uint32_t BLOCK_N,
    uint32_t BLOCK_M_WARPS,
    uint32_t BLOCK_N_WARPS>
__attribute__((amdgpu_waves_per_eu(2, 2), amdgpu_flat_work_group_size(BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE, BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE)))
__global__ void
hgemm_ht_kernel(
    scalar_t *c,
    const scalar_t *a,
    const scalar_t *b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
    using BlockTileT = BlockTileHT<scalar_t, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M, BLOCK_N, BLOCK_M_WARPS, BLOCK_N_WARPS>;
    constexpr uint32_t HALF_BLOCK_M = BlockTileT::HALF_BLOCK_M;
    constexpr uint32_t HALF_BLOCK_N = BlockTileT::HALF_BLOCK_N;
    constexpr uint32_t LDG_REG_A_COUNT = BlockTileT::LDG_REG_A_COUNT;
    constexpr uint32_t LDG_REG_B_COUNT = BlockTileT::LDG_REG_B_COUNT;
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid / WARP_SIZE;
    uint32_t mi, ni;
    get_tile_mn<BLOCK_M, BLOCK_N, true>(m, n, mi, ni);
    uint32_t m_offset = mi * BLOCK_M;
    uint32_t n_offset = ni * BLOCK_N;

    __shared__ SharedHTStorage<scalar_t, HALF_BLOCK_M, HALF_BLOCK_N, BLOCK_K> smem;

    BlockTileT block_tile(tid, k);
    uint32_t a_begin = m_offset * k;
    uint32_t b_begin = n_offset * k;
    uint32_t a_end = a_begin + k;

    auto a_rsrc = make_srsrc(a, /*range_bytes*/ 0xFFFFFFFFu);
    auto b_rsrc = make_srsrc(b, /*range_bytes*/ 0xFFFFFFFFu);
    block_tile.init_hip(&smem.as[0][0][0][0], &smem.bs[0][0][0][0]);

#define LDG_ASYNC_A(M_, K_, F_) block_tile.ldg_copy_async_a((K_ * 2 + M_) * HALF_BLOCK_M * BLOCK_K, a_rsrc, a_begin + M_ * HALF_BLOCK_M * k + (F_ + K_) * BLOCK_K)
#define LDG_ASYNC_B(N_, K_, F_) block_tile.ldg_copy_async_b((K_ * 2 + N_) * HALF_BLOCK_N * BLOCK_K, b_rsrc, b_begin + N_ * HALF_BLOCK_N * k + (F_ + K_) * BLOCK_K)
#define LDMAT_A(M_, K_) block_tile.ldmatrix_a(&smem.as[K_][M_][0][0])
#define LDMAT_B(N_, K_) block_tile.template ldmatrix_b<N_>(&smem.bs[K_][N_][0][0])
#define CONSUME(M_, N_, EMIT_SB)                \
    {                                           \
        block_tile.template consume<M_, N_>();  \
        hip_s_barrier();                        \
        if constexpr (EMIT_SB) sched_barrier(); \
    }

    LDG_ASYNC_B(0, 0, 0);
    LDG_ASYNC_A(0, 0, 0);
    LDG_ASYNC_B(1, 0, 0);
    LDG_ASYNC_A(1, 0, 0);

    if (wid / 4 == 1)
        hip_s_barrier();
    hip_s_barrier();

    LDG_ASYNC_B(0, 1, 0);
    LDG_ASYNC_A(0, 1, 0);
    LDG_ASYNC_B(1, 1, 0);
    // 4b3a
    __barrier<1 * LDG_REG_B_COUNT + 1 * LDG_REG_A_COUNT>();
    for (; a_begin < a_end - 2 * BLOCK_K; a_begin += 2 * BLOCK_K, b_begin += 2 * BLOCK_K) {
        // 0
        LDMAT_B(0, 0);
        LDMAT_A(0, 0);
        LDG_ASYNC_A(1, 1, 0);
        hip_s_barrier();
        CONSUME(0, 0, true);
        LDMAT_B(1, 0);
        LDG_ASYNC_B(0, 0, 2);
        hip_s_barrier();
        CONSUME(0, 1, false);
        LDMAT_A(1, 0);
        LDG_ASYNC_A(0, 0, 2);
        hip_s_barrier();
        CONSUME(1, 0, true);
        LDMAT_B(0, 1);
        LDG_ASYNC_B(1, 0, 2);
        __barrier<2 * LDG_REG_B_COUNT + 1 * LDG_REG_A_COUNT>();
        CONSUME(1, 1, false);
        // 1
        LDMAT_A(0, 1);
        LDG_ASYNC_A(1, 0, 2);
        hip_s_barrier();
        CONSUME(0, 0, true);
        LDMAT_B(1, 1);
        LDG_ASYNC_B(0, 1, 2);
        hip_s_barrier();
        CONSUME(0, 1, false);
        LDMAT_A(1, 1);
        LDG_ASYNC_A(0, 1, 2);
        hip_s_barrier();
        CONSUME(1, 0, true);
        LDG_ASYNC_B(1, 1, 2);
        __barrier<1 * LDG_REG_B_COUNT + 1 * LDG_REG_A_COUNT>();
        CONSUME(1, 1, false);
    }
    // 0
    LDMAT_B(0, 0);
    LDMAT_A(0, 0);
    LDG_ASYNC_A(1, 1, 0);
    hip_s_barrier();
    CONSUME(0, 0, true);
    LDMAT_B(1, 0);
    hip_s_barrier();
    CONSUME(0, 1, false);
    LDMAT_A(1, 0);
    hip_s_barrier();
    CONSUME(1, 0, true);
    LDMAT_B(0, 1);
    hip_s_barrier();
    CONSUME(1, 1, false);
    // 1
    __barrier<0>();
    LDMAT_A(0, 1);
    hip_s_barrier();
    CONSUME(0, 0, false);
    LDMAT_B(1, 1);
    hip_s_barrier();
    CONSUME(0, 1, false);
    LDMAT_A(1, 1);
    hip_s_barrier();
    CONSUME(1, 0, false);
#pragma unroll
    for (uint32_t mi_step = 0; mi_step < BlockTileT::WARP_M_STEPS; ++mi_step) {
        block_tile.template store_matrix_to_lds_mi<0, 0>(smem.cs, mi_step);
    }
#pragma unroll
    for (uint32_t mi_step = 0; mi_step < BlockTileT::WARP_M_STEPS; ++mi_step) {
        block_tile.template store_matrix_to_lds_mi<0, 1>(smem.cs, mi_step);
    }
    hip_s_barrier();
#pragma unroll
    for (uint32_t mi_step = 0; mi_step < BlockTileT::WARP_M_STEPS; ++mi_step) {
        block_tile.template store_matrix_from_lds_mi<0, 0>(c, smem.cs, mi, ni, m, n, mi_step);
    }
#pragma unroll
    for (uint32_t mi_step = 0; mi_step < BlockTileT::WARP_M_STEPS; ++mi_step) {
        block_tile.template store_matrix_from_lds_mi<0, 1>(c, smem.cs, mi, ni, m, n, mi_step);
    }
    CONSUME(1, 1, false);
#pragma unroll
    for (uint32_t mi_step = 0; mi_step < BlockTileT::WARP_M_STEPS; ++mi_step) {
        block_tile.template store_matrix_to_lds_mi<1, 0>(smem.cs, mi_step);
    }
#pragma unroll
    for (uint32_t mi_step = 0; mi_step < BlockTileT::WARP_M_STEPS; ++mi_step) {
        block_tile.template store_matrix_to_lds_mi<1, 1>(smem.cs, mi_step);
    }
    hip_s_barrier();
#pragma unroll
    for (uint32_t mi_step = 0; mi_step < BlockTileT::WARP_M_STEPS; ++mi_step) {
        block_tile.template store_matrix_from_lds_mi<1, 0>(c, smem.cs, mi, ni, m, n, mi_step);
    }
#pragma unroll
    for (uint32_t mi_step = 0; mi_step < BlockTileT::WARP_M_STEPS; ++mi_step) {
        block_tile.template store_matrix_from_lds_mi<1, 1>(c, smem.cs, mi, ni, m, n, mi_step);
    }

#undef LDG_ASYNC_A
#undef LDG_ASYNC_B
#undef LDMAT_A
#undef LDMAT_B
#undef CONSUME
}

std::tuple<dim3, uint32_t> get_grid(uint32_t m, uint32_t n, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t SPLIT_K) {
    uint32_t bm = (m + BLOCK_M - 1) / BLOCK_M;
    uint32_t bn = (n + BLOCK_N - 1) / BLOCK_N;
    uint32_t grid_dim_x = bm * bn;
    return {dim3(grid_dim_x, SPLIT_K, 1), grid_dim_x};
}

#ifdef __CUDACC__

#define GET_HGEMM_WMMA_M16N8K16_IMPL_NAME(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_SIZE, STAGES, SPLIT_K) \
    hgemm_wmma_m16n8k16_##BLOCK_M##x##BLOCK_N##x##BLOCK_K##_spk##SPLIT_K##_w##BLOCK_M_WARPS##x##BLOCK_N_WARPS##x##BLOCK_K_WARPS##x##WARP_SIZE##_s##STAGES##_

#define REGISTER_HGEMM_WMMA_M16N8K16_IMPL(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_SIZE, STAGES, SPLIT_K)   \
    void GET_HGEMM_WMMA_M16N8K16_IMPL_NAME(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_SIZE, STAGES, SPLIT_K)( \
        short *c, const short *a, const short *b, const uint32_t m, const uint32_t n, const uint32_t k, const bool is_bf16,                     \
        uint32_t *semaphore, uint32_t *signal, gpuStream_t stream) {                                                                            \
        constexpr uint32_t VEC_SIZE = 8;                                                                                                        \
        assert(n % VEC_SIZE == 0);                                                                                                              \
        assert(k % VEC_SIZE == 0);                                                                                                              \
        auto gr = get_grid(m, n, BLOCK_M, BLOCK_N, SPLIT_K);                                                                                    \
        dim3 grid = std::get<0>(gr);                                                                                                            \
        constexpr uint32_t BLOCK_SIZE = BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE;                                              \
        dim3 block(BLOCK_SIZE);                                                                                                                 \
        constexpr uint32_t WARP_M_STEPS = BLOCK_M / BLOCK_M_WARPS / 16;                                                                         \
        constexpr uint32_t WARP_N_STEPS = BLOCK_N / BLOCK_N_WARPS / 8;                                                                          \
        if (is_bf16 == false) {                                                                                                                 \
            using T = __half;                                                                                                                   \
            using WMMAT = WMMA_M16N8K16<T, float>;                                                                                              \
            hgemm_kernel<T, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_M_STEPS, WARP_N_STEPS,                 \
                         STAGES, SPLIT_K><<<grid, block, 0, stream>>>((T *)c, (T *)a, (T *)b, m, n, k, semaphore, signal);                      \
        } else {                                                                                                                                \
            using T = __bfloat16;                                                                                                               \
            using WMMAT = WMMA_M16N8K16<T, float>;                                                                                              \
            hgemm_kernel<T, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_M_STEPS, WARP_N_STEPS,                 \
                         STAGES, SPLIT_K><<<grid, block, 0, stream>>>((T *)c, (T *)a, (T *)b, m, n, k, semaphore, signal);                      \
        }                                                                                                                                       \
    }

REGISTER_HGEMM_WMMA_M16N8K16_IMPL(/*BLOCK_M*/ 16, /*BLOCK_N*/ 64, /*BLOCK_K*/ 64, /*BLOCK_M_WARPS*/ 1, /*BLOCK_N_WARPS*/ 1, /*BLOCK_K_WARPS*/ 2, /*WARP_SIZE*/ 32, /*STAGES*/ 2, /*SPLIT_K*/ 4)
REGISTER_HGEMM_WMMA_M16N8K16_IMPL(/*BLOCK_M*/ 128, /*BLOCK_N*/ 128, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 4, /*BLOCK_K_WARPS*/ 1, /*WARP_SIZE*/ 32, /*STAGES*/ 4, /*SPLIT_K*/ 1)

void hgemm_peak(
    short *c,
    const short *a,
    const short *b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k,
    const bool is_bf16,
    uint32_t *semaphore,
    uint32_t *signal,
    gpuStream_t stream) {
    if (m <= 256) {
        GET_HGEMM_WMMA_M16N8K16_IMPL_NAME(/*BLOCK_M*/ 16, /*BLOCK_N*/ 64, /*BLOCK_K*/ 64, /*BLOCK_M_WARPS*/ 1, /*BLOCK_N_WARPS*/ 1, /*BLOCK_K_WARPS*/ 2, /*WARP_SIZE*/ 32, /*STAGES*/ 2, /*SPLIT_K*/ 4)
        (c, a, b, m, n, k, is_bf16, semaphore, signal, stream);
    } else {
        GET_HGEMM_WMMA_M16N8K16_IMPL_NAME(/*BLOCK_M*/ 128, /*BLOCK_N*/ 128, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 4, /*BLOCK_K_WARPS*/ 1, /*WARP_SIZE*/ 32, /*STAGES*/ 4, /*SPLIT_K*/ 1)
        (c, a, b, m, n, k, is_bf16, semaphore, signal, stream);
    }
}

#endif

#ifdef __HIPCC__

#define GET_HGEMM_WMMA_M16N16K32_IMPL_NAME(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_SIZE, STAGES, SPLIT_K) \
    hgemm_wmma_m16n16k32_##BLOCK_M##x##BLOCK_N##x##BLOCK_K##_spk##SPLIT_K##_w##BLOCK_M_WARPS##x##BLOCK_N_WARPS##x##BLOCK_K_WARPS##x##WARP_SIZE##_s##STAGES##_

#define REGISTER_HGEMM_WMMA_M16N16K32_IMPL(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_SIZE, STAGES, SPLIT_K)   \
    void GET_HGEMM_WMMA_M16N16K32_IMPL_NAME(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_SIZE, STAGES, SPLIT_K)( \
        short *c, const short *a, const short *b, const uint32_t m, const uint32_t n, const uint32_t k, const bool is_bf16,                      \
        uint32_t *semaphore, uint32_t *signal, gpuStream_t stream) {                                                                             \
        constexpr uint32_t VEC_SIZE = 8;                                                                                                         \
        assert(n % VEC_SIZE == 0);                                                                                                               \
        assert(k % VEC_SIZE == 0);                                                                                                               \
        auto gr = get_grid(m, n, BLOCK_M, BLOCK_N, SPLIT_K);                                                                                     \
        dim3 grid = std::get<0>(gr);                                                                                                             \
        constexpr uint32_t BLOCK_SIZE = BLOCK_M_WARPS * BLOCK_N_WARPS * BLOCK_K_WARPS * WARP_SIZE;                                               \
        dim3 block(BLOCK_SIZE);                                                                                                                  \
        constexpr uint32_t WARP_M = BLOCK_M_WARPS * 16;                                                                                          \
        constexpr uint32_t WARP_N = BLOCK_N_WARPS * 16;                                                                                          \
        constexpr uint32_t WARP_M_STEPS = (BLOCK_M + WARP_M - 1) / WARP_M;                                                                       \
        constexpr uint32_t WARP_N_STEPS = (BLOCK_N + WARP_N - 1) / WARP_N;                                                                       \
        if (SPLIT_K > 1) {                                                                                                                       \
            assert(std::get<1>(gr) < SPLIT_K_SEMAPHORE_MAX_LEN);                                                                                 \
            assert(k % (SPLIT_K * BLOCK_K) == 0);                                                                                                \
        }                                                                                                                                        \
        if (is_bf16 == false) {                                                                                                                  \
            using T = __half;                                                                                                                    \
            using WMMAT = WMMA_M16N16K32<T, float, true, BLOCK_K * 2 / 16>;                                                                      \
            hgemm_kernel<T, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_M_STEPS, WARP_N_STEPS,                  \
                         STAGES, SPLIT_K><<<grid, block, 0, stream>>>((T *)c, (T *)a, (T *)b, m, n, k, semaphore, signal);                       \
        } else {                                                                                                                                 \
            using T = __bfloat16;                                                                                                                \
            using WMMAT = WMMA_M16N16K32<T, float, true, BLOCK_K * 2 / 16>;                                                                      \
            hgemm_kernel<T, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, BLOCK_K_WARPS, WARP_M_STEPS, WARP_N_STEPS,                  \
                         STAGES, SPLIT_K><<<grid, block, 0, stream>>>((T *)c, (T *)a, (T *)b, m, n, k, semaphore, signal);                       \
        }                                                                                                                                        \
    }

REGISTER_HGEMM_WMMA_M16N16K32_IMPL(/*BLOCK_M*/ 16, /*BLOCK_N*/ 256, /*BLOCK_K*/ 64, /*BLOCK_M_WARPS*/ 1, /*BLOCK_N_WARPS*/ 2, /*BLOCK_K_WARPS*/ 1, /*WARP_SIZE*/ 64, /*STAGES*/ 4, /*SPLIT_K*/ 8)
REGISTER_HGEMM_WMMA_M16N16K32_IMPL(/*BLOCK_M*/ 256, /*BLOCK_N*/ 256, /*BLOCK_K*/ 64, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 4, /*BLOCK_K_WARPS*/ 1, /*WARP_SIZE*/ 64, /*STAGES*/ 2, /*SPLIT_K*/ 1)

void hgemm_peak(
    short *c,
    const short *a,
    const short *b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k,
    const bool is_bf16,
    uint32_t *semaphore,
    uint32_t *signal,
    gpuStream_t stream) {
    assert(n % 8 == 0 && k % 8 == 0);
    if (m <= 256) {
        GET_HGEMM_WMMA_M16N16K32_IMPL_NAME(/*BLOCK_M*/ 16, /*BLOCK_N*/ 256, /*BLOCK_K*/ 64, /*BLOCK_M_WARPS*/ 1, /*BLOCK_N_WARPS*/ 2, /*BLOCK_K_WARPS*/ 1, /*WARP_SIZE*/ 64, /*STAGES*/ 4, /*SPLIT_K*/ 8)
        (c, a, b, m, n, k, is_bf16, semaphore, signal, stream);
    } else {
        // GET_HGEMM_WMMA_M16N16K32_IMPL_NAME(/*BLOCK_M*/ 256, /*BLOCK_N*/ 256, /*BLOCK_K*/ 64, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 4, /*BLOCK_K_WARPS*/ 1, /*WARP_SIZE*/ 64, /*STAGES*/ 2, /*SPLIT_K*/ 1)
        // (c, a, b, m, n, k, is_bf16, semaphore, signal, stream);
        constexpr uint32_t BLOCK_M = 256;
        constexpr uint32_t BLOCK_N = 256;
        constexpr uint32_t BLOCK_K = 64;
        constexpr uint32_t BLOCK_M_WARPS = 2;
        constexpr uint32_t BLOCK_N_WARPS = 4;
        constexpr uint32_t WARP_SIZE = 64;
        constexpr uint32_t VEC_SIZE = 8;
        assert(n % VEC_SIZE == 0);
        assert(k % VEC_SIZE == 0);
        auto gr = get_grid(m, n, BLOCK_M, BLOCK_N, 1);
        dim3 grid = std::get<0>(gr);
        constexpr uint32_t BLOCK_SIZE = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE;
        dim3 block(BLOCK_SIZE);
        if (is_bf16 == false) {
            using T = __half;
            using WMMAT = WMMA_M16N16K32<T, float, true, BLOCK_K * 2 / 16>;
            hgemm_ht_kernel<T, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M, BLOCK_N, BLOCK_M_WARPS, BLOCK_N_WARPS><<<grid, block, 0, stream>>>((T *)c, (T *)a, (T *)b, m, n, k);
        } else {
            using T = __bfloat16;
            using WMMAT = WMMA_M16N16K32<T, float, true, BLOCK_K * 2 / 16>;
            hgemm_ht_kernel<T, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M, BLOCK_N, BLOCK_M_WARPS, BLOCK_N_WARPS><<<grid, block, 0, stream>>>((T *)c, (T *)a, (T *)b, m, n, k);
        }
    }
}

#endif

} // namespace hgemm
