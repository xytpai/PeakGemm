#pragma once

#include "device_common.h"

using namespace kernel_utils;
using namespace wmma_utils;

namespace hgemm {

template <
    typename scalar_t,
    typename WMMAT,
    uint32_t WARP_SIZE,
    uint32_t BLOCK_K,
    uint32_t BLOCK_M_WARPS,
    uint32_t BLOCK_N_WARPS,
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
        WARP_K_STEPS = BLOCK_K / WARP_ATOM_K,
        LDG_VEC_SIZE = 16 / sizeof(scalar_t),
        BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE,
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
    };
    static_assert(LDG_REG_A_COUNT >= 1 && LDG_REG_B_COUNT >= 1);
    using ldg_vec_t = aligned_array<scalar_t, LDG_VEC_SIZE>;

    __device__ __forceinline__ BlockTile(uint32_t tid) :
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
        i32x4 &a_rsrc, uint32_t a_begin, uint32_t a_stride, i32x4 &b_rsrc, uint32_t b_begin, uint32_t b_stride) {
        uint32_t as_warp_ = as_ + as_offset * sizeof(scalar_t);
        uint32_t bs_warp_ = bs_ + bs_offset * sizeof(scalar_t);
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_A_COUNT; i++) {
            uint32_t tid_ = BLOCK_THREADS * i + tid;
            uint32_t row = tid_ / LDG_A_X_THREADS;
            uint32_t col = ldg_a_vec_idx * LDG_VEC_SIZE;
            col = wmma.swizzle(row, col);
            uint32_t global_offset = a_begin + row * a_stride + col;
            llvm_amdgcn_raw_buffer_load_lds(
                a_rsrc,
                (as3_uint32_ptr) static_cast<uintptr_t>(as_warp_),
                DMA_BYTES,
                global_offset * sizeof(scalar_t),
                0,
                0,
                0);
            as_warp_ += BLOCK_THREADS * DMA_BYTES;
        }
#pragma unroll
        for (uint32_t i = 0; i < LDG_REG_B_COUNT; i++) {
            uint32_t tid_ = BLOCK_THREADS * i + tid;
            uint32_t row = tid_ / LDG_B_X_THREADS;
            uint32_t col = ldg_b_vec_idx * LDG_VEC_SIZE;
            col = wmma.swizzle(row, col);
            uint32_t global_offset = b_begin + row * b_stride + col;
            llvm_amdgcn_raw_buffer_load_lds(
                b_rsrc,
                (as3_uint32_ptr) static_cast<uintptr_t>(bs_warp_),
                DMA_BYTES,
                global_offset * sizeof(scalar_t),
                0,
                0,
                0);
            bs_warp_ += BLOCK_THREADS * DMA_BYTES;
        }
    }
#endif

    template <uint32_t S = 0>
    __device__ __forceinline__ void wait() {
        CopyAsync::wait<S>();
    }

    __device__ __forceinline__ void commit() {
        CopyAsync::commit();
    }

    __device__ __forceinline__ void load_matrix(scalar_t *as, scalar_t *bs) {
        uint32_t warp_m_begin = wid / BLOCK_N_WARPS * WARP_M;
        uint32_t warp_n_begin = wid % BLOCK_N_WARPS * WARP_N;
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
#pragma unroll
        for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
            uint32_t warp_atom_offset_n = warp_n_begin + ni * WARP_ATOM_N;
#pragma unroll
            for (uint32_t ki = 0; ki < WARP_K_STEPS; ++ki) {
                uint32_t row = warp_atom_offset_n;
                uint32_t col = ki * WARP_ATOM_K;
                wmma.load_matrix_b(fb[ki][ni], bs, row, col, BLOCK_K);
            }
        }
    }

    __device__ __forceinline__ void store_matrix(scalar_t *ptr, uint32_t block_m_idx, uint32_t block_n_idx, uint32_t m, uint32_t n) {
        uint32_t warp_m_begin = block_m_idx * BLOCK_M + wid / BLOCK_N_WARPS * WARP_M;
        uint32_t warp_n_begin = block_n_idx * BLOCK_N + wid % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
            uint32_t warp_atom_offset_m = warp_m_begin + mi * WARP_ATOM_M;
#pragma unroll
            for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
                uint32_t warp_atom_offset_n = warp_n_begin + ni * WARP_ATOM_N;
                if (warp_atom_offset_m < m && warp_atom_offset_n < n) {
                    auto ptr_ = ptr + warp_atom_offset_m * n + warp_atom_offset_n;
                    wmma.store_matrix(ptr_, n, fo[mi][ni]);
                    __syncthreads();
                }
            }
        }
    }

    __device__ __forceinline__ void operator()() {
#pragma unroll
        for (uint32_t mi = 0; mi < WARP_M_STEPS; ++mi) {
#pragma unroll
            for (uint32_t ni = 0; ni < WARP_N_STEPS; ++ni) {
#pragma unroll
                for (uint32_t ki = 0; ki < WARP_K_STEPS; ++ki) {
                    wmma(fo[mi][ni], fa[ki][mi], fb[ki][ni], fo[mi][ni]);
                }
            }
        }
    }

private:
    uint32_t tid;
    uint32_t wid;
    uint32_t w_tid;
    uint32_t ldg_a_vec_idx;
    uint32_t ldg_b_vec_idx;
    ldg_vec_t ldg_a_reg[LDG_REG_A_COUNT];
    ldg_vec_t ldg_b_reg[LDG_REG_B_COUNT];
    WMMAT wmma;
    FragmentAT fa[WARP_K_STEPS][WARP_M_STEPS];
    FragmentBT fb[WARP_K_STEPS][WARP_N_STEPS];
    FragmentCT fo[WARP_M_STEPS][WARP_N_STEPS];
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

template <
    typename scalar_t,
    typename WMMAT,
    uint32_t WARP_SIZE,
    uint32_t BLOCK_K,
    uint32_t BLOCK_M_WARPS,
    uint32_t BLOCK_N_WARPS,
    uint32_t WARP_M_STEPS,
    uint32_t WARP_N_STEPS,
    uint32_t STAGES = 3>
__global__ void hgemm_kernel(
    scalar_t *c,
    const scalar_t *a,
    const scalar_t *b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k) {
    using BlockTileT = BlockTile<scalar_t, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_M_STEPS, WARP_N_STEPS>;
    constexpr uint32_t BLOCK_M = BlockTileT::BLOCK_M;
    constexpr uint32_t BLOCK_N = BlockTileT::BLOCK_N;

    uint32_t tid = threadIdx.x;
    uint32_t mi = blockIdx.y;
    uint32_t ni = blockIdx.x;

    constexpr uint32_t BLOCK_MK = BLOCK_M * BLOCK_K;
    constexpr uint32_t BLOCK_NK = BLOCK_N * BLOCK_K;
    __shared__ scalar_t as[STAGES][BLOCK_MK];
    __shared__ scalar_t bs[STAGES][BLOCK_NK];

    BlockTileT block_tile(tid);
    uint32_t current_stage = 0;
    uint32_t a_begin = mi * BLOCK_M * k;
    uint32_t b_begin = ni * BLOCK_N * k;
    uint32_t a_end = a_begin + k;

#ifdef __CUDACC__

#pragma unroll
    for (uint32_t s = 0; s < STAGES - 1; ++s) {
        block_tile.ldg_copy_async(as[s], bs[s], &a[a_begin + s * BLOCK_K], k, &b[b_begin + s * BLOCK_K], k);
        block_tile.commit();
    }
    for (; a_begin < a_end; a_begin += BLOCK_K, b_begin += BLOCK_K) {
        block_tile.template wait<STAGES - 2>();
        __syncthreads();
        block_tile.load_matrix(as[current_stage], bs[current_stage]);
        block_tile();
        if (a_begin + (STAGES - 1) * BLOCK_K < a_end) {
            uint32_t write_stage = (current_stage + STAGES - 1) % STAGES;
            block_tile.ldg_copy_async(as[write_stage], bs[write_stage],
                                      &a[a_begin + (STAGES - 1) * BLOCK_K], k,
                                      &b[b_begin + (STAGES - 1) * BLOCK_K], k);
        }
        block_tile.commit();
        current_stage = (current_stage + 1) % STAGES;
    }

#elif defined(__HIPCC__)

    static_assert(STAGES == 2);
    auto a_rsrc = make_srsrc(a, /*range_bytes*/ 0xFFFFFFFFu);
    auto b_rsrc = make_srsrc(b, /*range_bytes*/ 0xFFFFFFFFu);
    block_tile.init_hip(&as[0][0], &bs[0][0]);
    block_tile.ldg_copy_async(0, 0, a_rsrc, a_begin, k, b_rsrc, b_begin, k);
    __barrier();
    for (; a_begin < a_end - BLOCK_K; a_begin += BLOCK_K, b_begin += BLOCK_K) {
        uint32_t write_stage = (current_stage + 1) % STAGES;
        block_tile.ldg_copy_async(write_stage * BLOCK_M * BLOCK_K, write_stage * BLOCK_N * BLOCK_K,
                                  a_rsrc, a_begin + BLOCK_K, k, b_rsrc, b_begin + BLOCK_K, k);
        block_tile.load_matrix(as[current_stage], bs[current_stage]);
        block_tile();
        current_stage = (current_stage + 1) % STAGES;
        __barrier();
    }
    block_tile.load_matrix(as[current_stage], bs[current_stage]);
    block_tile();

#endif

    block_tile.store_matrix(c, mi, ni, m, n);
}

std::tuple<dim3, uint32_t> get_grid(uint32_t m, uint32_t n, uint32_t BLOCK_M, uint32_t BLOCK_N) {
    uint32_t bm = (m + BLOCK_M - 1) / BLOCK_M;
    uint32_t bn = (n + BLOCK_N - 1) / BLOCK_N;
    uint32_t raster_factor = 1;
    return {dim3(bn, bm, 1), raster_factor};
}

#ifdef __CUDACC__

#define GET_HGEMM_WMMA_M16N8K16_IMPL_NAME(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_SIZE, STAGES) \
    hgemm_wmma_m16n8k16_##BLOCK_M##x##BLOCK_N##x##BLOCK_K##_w##BLOCK_M_WARPS##x##BLOCK_N_WARPS##x##WARP_SIZE##_s##STAGES##_

#define REGISTER_HGEMM_WMMA_M16N8K16_IMPL(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_SIZE, STAGES)                             \
    void GET_HGEMM_WMMA_M16N8K16_IMPL_NAME(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_SIZE, STAGES)(                           \
        short *c, const short *a, const short *b, const uint32_t m, const uint32_t n, const uint32_t k, const bool is_bf16, gpuStream_t stream) { \
        constexpr uint32_t VEC_SIZE = 8;                                                                                                          \
        assert(n % VEC_SIZE == 0);                                                                                                                \
        assert(k % VEC_SIZE == 0);                                                                                                                \
        auto gr = get_grid(m, n, BLOCK_M, BLOCK_N);                                                                                               \
        dim3 grid = std::get<0>(gr);                                                                                                              \
        constexpr uint32_t BLOCK_SIZE = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE;                                                                \
        dim3 block(BLOCK_SIZE);                                                                                                                   \
        constexpr uint32_t WARP_M_STEPS = BLOCK_M / BLOCK_M_WARPS / 16;                                                                           \
        constexpr uint32_t WARP_N_STEPS = BLOCK_N / BLOCK_N_WARPS / 8;                                                                            \
        if (is_bf16 == false) {                                                                                                                   \
            using T = __half;                                                                                                                     \
            using WMMAT = WMMA_M16N8K16<T, float>;                                                                                                \
            hgemm_kernel<T, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_M_STEPS, WARP_N_STEPS,                                  \
                         STAGES><<<grid, block, 0, stream>>>((T *)c, (T *)a, (T *)b, m, n, k);                                                    \
        } else {                                                                                                                                  \
            using T = __bfloat16;                                                                                                                 \
            using WMMAT = WMMA_M16N8K16<T, float>;                                                                                                \
            hgemm_kernel<T, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_M_STEPS, WARP_N_STEPS,                                  \
                         STAGES><<<grid, block, 0, stream>>>((T *)c, (T *)a, (T *)b, m, n, k);                                                    \
        }                                                                                                                                         \
    }

REGISTER_HGEMM_WMMA_M16N8K16_IMPL(/*BLOCK_M*/ 128, /*BLOCK_N*/ 128, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 4, /*WARP_SIZE*/ 32, /*STAGES*/ 4)

void hgemm_peak(
    short *c,
    const short *a,
    const short *b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k,
    const bool is_bf16,
    gpuStream_t stream) {
    GET_HGEMM_WMMA_M16N8K16_IMPL_NAME(/*BLOCK_M*/ 128, /*BLOCK_N*/ 128, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 4, /*WARP_SIZE*/ 32, /*STAGES*/ 4)
    (c, a, b, m, n, k, is_bf16, stream);
}

#elif defined(__HIPCC__)

#define GET_HGEMM_WMMA_M16N16K32_IMPL_NAME(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_SIZE, STAGES) \
    hgemm_wmma_m16n16k32_##BLOCK_M##x##BLOCK_N##x##BLOCK_K##_w##BLOCK_M_WARPS##x##BLOCK_N_WARPS##x##WARP_SIZE##_s##STAGES##_

#define REGISTER_HGEMM_WMMA_M16N16K32_IMPL(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_SIZE, STAGES)                            \
    void GET_HGEMM_WMMA_M16N16K32_IMPL_NAME(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_SIZE, STAGES)(                          \
        short *c, const short *a, const short *b, const uint32_t m, const uint32_t n, const uint32_t k, const bool is_bf16, gpuStream_t stream) { \
        constexpr uint32_t VEC_SIZE = 8;                                                                                                          \
        assert(n % VEC_SIZE == 0);                                                                                                                \
        assert(k % VEC_SIZE == 0);                                                                                                                \
        auto gr = get_grid(m, n, BLOCK_M, BLOCK_N);                                                                                               \
        dim3 grid = std::get<0>(gr);                                                                                                              \
        constexpr uint32_t BLOCK_SIZE = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE;                                                                \
        dim3 block(BLOCK_SIZE);                                                                                                                   \
        constexpr uint32_t WARP_M_STEPS = BLOCK_M / BLOCK_M_WARPS / 16;                                                                           \
        constexpr uint32_t WARP_N_STEPS = BLOCK_N / BLOCK_N_WARPS / 16;                                                                           \
        if (is_bf16 == false) {                                                                                                                   \
            using T = __half;                                                                                                                     \
            using WMMAT = WMMA_M16N16K32<T, float, true, BLOCK_K * 2 / 16>;                                                                       \
            hgemm_kernel<T, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_M_STEPS, WARP_N_STEPS,                                  \
                         STAGES><<<grid, block, 0, stream>>>((T *)c, (T *)a, (T *)b, m, n, k);                                                    \
        } else {                                                                                                                                  \
            using T = __bfloat16;                                                                                                                 \
            using WMMAT = WMMA_M16N16K32<T, float, true, BLOCK_K * 2 / 16>;                                                                       \
            hgemm_kernel<T, WMMAT, WARP_SIZE, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_M_STEPS, WARP_N_STEPS,                                  \
                         STAGES><<<grid, block, 0, stream>>>((T *)c, (T *)a, (T *)b, m, n, k);                                                    \
        }                                                                                                                                         \
    }

REGISTER_HGEMM_WMMA_M16N16K32_IMPL(/*BLOCK_M*/ 128, /*BLOCK_N*/ 128, /*BLOCK_K*/ 64, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 2, /*WARP_SIZE*/ 64, /*STAGES*/ 2)

void hgemm_peak(
    short *c,
    const short *a,
    const short *b,
    const uint32_t m,
    const uint32_t n,
    const uint32_t k,
    const bool is_bf16,
    gpuStream_t stream) {
    GET_HGEMM_WMMA_M16N16K32_IMPL_NAME(/*BLOCK_M*/ 128, /*BLOCK_N*/ 128, /*BLOCK_K*/ 64, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 2, /*WARP_SIZE*/ 64, /*STAGES*/ 2)
    (c, a, b, m, n, k, is_bf16, stream);
}

#endif

} // namespace hgemm
