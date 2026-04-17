#pragma once

#include "device_common.h"

using namespace kernel_utils;

namespace sgemm {

template <typename scalar_t>
__global__ void sgemm_naive_kernel(
    scalar_t *c,
    const scalar_t *a,
    const scalar_t *b,
    const int m,
    const int n,
    const int k) {
    int mi = blockIdx.y * 32 + threadIdx.y;
    int ni = blockIdx.x * 32 + threadIdx.x;
    if (mi < m && ni < n) {
        float acc = 0.f;
        for (int ki = 0; ki < k; ki++) {
            acc += (float)a[mi * k + ki] * (float)b[ni * k + ki];
        }
        c[mi * n + ni] = (scalar_t)(acc);
    }
}

template <typename scalar_t>
void sgemm_naive(
    scalar_t *c,
    const scalar_t *a,
    const scalar_t *b,
    const int m,
    const int n,
    const int k,
    gpuStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((n + 32 - 1) / 32, (m + 32 - 1) / 32);
    sgemm_naive_kernel<scalar_t><<<grid, block, 0, stream>>>(c, a, b, m, n, k);
}

template <typename scalar_t, int VEC_A, int VEC_B>
struct mma_reg_t {
    using vec_a_t = aligned_array<scalar_t, VEC_A>;
    using vec_b_t = aligned_array<scalar_t, VEC_B>;
    union {
        vec_a_t a_vec;
        scalar_t a[VEC_A];
    };
    union {
        vec_b_t b_vec;
        scalar_t b[VEC_B];
    };
};

template <
    typename scalar_t,
    int BLOCK_K,
    int WARP_M_THREADS,
    int WARP_N_THREADS,
    int VEC_M,
    int VEC_N,
    int KSTRIDE_A,
    int KSTRIDE_B>
struct WarpTile {
    __device__ __forceinline__ void operator()(scalar_t *o, scalar_t *as, scalar_t *bs, int wy, int wx, int w_tid) {
        using a_vec_t = aligned_array<scalar_t, VEC_M>;
        using b_vec_t = aligned_array<scalar_t, VEC_N>;
        int th_y = wy + w_tid / WARP_N_THREADS * VEC_M;
        int th_x = wx + w_tid % WARP_N_THREADS * VEC_N;
        mma_reg_t<scalar_t, VEC_M, VEC_N> reg;
#pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            reg.a_vec = *reinterpret_cast<a_vec_t *>(&as[k * KSTRIDE_A + th_y]);
            reg.b_vec = *reinterpret_cast<b_vec_t *>(&bs[k * KSTRIDE_B + th_x]);
#pragma unroll
            for (int i = 0; i < VEC_M; ++i) {
#pragma unroll
                for (int j = 0; j < VEC_N; ++j) {
                    o[i * VEC_N + j] += reg.a[i] * reg.b[j];
                }
            }
        }
    }
};

template <
    typename scalar_t,
    int BLOCK_K,
    int BLOCK_M_WARPS,
    int BLOCK_N_WARPS,
    int WARP_M_STEPS,
    int WARP_N_STEPS,
    int WARP_M_THREADS,
    int WARP_N_THREADS,
    int VEC_M,
    int VEC_N,
    int WARP_SIZE>
struct BlockTile {
    enum {
        WARP_MASK = WARP_SIZE - 1,
        WARP_SHIFT = Log2<WARP_SIZE>::VALUE,
        LDG_VEC_SIZE = 16 / sizeof(scalar_t),
        BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE,
        WARP_ATOM_M = WARP_M_THREADS * VEC_M,
        WARP_ATOM_N = WARP_N_THREADS * VEC_N,
        WARP_M = WARP_M_STEPS * WARP_ATOM_M,
        WARP_N = WARP_N_STEPS * WARP_ATOM_N,
        BLOCK_M = BLOCK_M_WARPS * WARP_M,
        BLOCK_N = BLOCK_N_WARPS * WARP_N,
        BLOCK_KM_SIZE = BLOCK_K * BLOCK_M,
        BLOCK_KN_SIZE = BLOCK_K * BLOCK_N,
        LDG_A_X_THREADS = BLOCK_K / LDG_VEC_SIZE,
        LDG_B_X_THREADS = BLOCK_K / LDG_VEC_SIZE,
        LDG_REG_A_COUNT = BLOCK_KM_SIZE / LDG_VEC_SIZE / BLOCK_THREADS,
        LDG_REG_B_COUNT = BLOCK_KN_SIZE / LDG_VEC_SIZE / BLOCK_THREADS,
#ifdef __HIPCC__
        PAD = 0,
#elif defined(__CUDACC__)
        PAD = LDG_VEC_SIZE, // swizzle is not a good idea for sgemm
#endif
    };
    static_assert(WARP_M_THREADS * WARP_N_THREADS == WARP_SIZE);
    static_assert(LDG_REG_A_COUNT >= 1 && LDG_REG_B_COUNT >= 1);
    using ldg_vec_t = aligned_array<scalar_t, LDG_VEC_SIZE>;

    __device__ __forceinline__ BlockTile(int tid) :
        tid(tid), wid(tid >> WARP_SHIFT), w_tid(tid & WARP_MASK),
        ldg_a_vec_idx(tid % LDG_A_X_THREADS),
        ldg_b_vec_idx(tid % LDG_B_X_THREADS) {
    }

    __device__ __forceinline__ void ldg(const scalar_t *a, int a_stride, const scalar_t *b, int b_stride) {
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++)
            ldg_a_reg[i] = reinterpret_cast<ldg_vec_t *>(
                const_cast<scalar_t *>(a) + ((BLOCK_THREADS * i + tid) / LDG_A_X_THREADS) * a_stride)[ldg_a_vec_idx];
#pragma unroll
        for (int i = 0; i < LDG_REG_B_COUNT; i++)
            ldg_b_reg[i] = reinterpret_cast<ldg_vec_t *>(
                const_cast<scalar_t *>(b) + ((BLOCK_THREADS * i + tid) / LDG_B_X_THREADS) * b_stride)[ldg_b_vec_idx];
    }

    __device__ __forceinline__ void sts(scalar_t *as, scalar_t *bs) {
#pragma unroll
        for (int i = 0; i < LDG_REG_A_COUNT; i++) {
            int y = (BLOCK_THREADS * i + tid) / LDG_A_X_THREADS;
#pragma unroll
            for (int j = 0; j < LDG_VEC_SIZE; j++) {
                int x = ldg_a_vec_idx * LDG_VEC_SIZE + j;
                as[x * (BLOCK_M + PAD) + y] = ldg_a_reg[i].val[j];
            }
        }
#pragma unroll
        for (int i = 0; i < LDG_REG_B_COUNT; i++) {
            int y = (BLOCK_THREADS * i + tid) / LDG_B_X_THREADS;
#pragma unroll
            for (int j = 0; j < LDG_VEC_SIZE; j++) {
                int x = ldg_b_vec_idx * LDG_VEC_SIZE + j;
                bs[x * (BLOCK_N + PAD) + y] = ldg_b_reg[i].val[j];
            }
        }
    }

    __device__ __forceinline__ void operator()(scalar_t (*o)[VEC_M * VEC_N], scalar_t *as, scalar_t *bs) {
        int warp_y = wid / BLOCK_N_WARPS * WARP_M;
        int warp_x = wid % BLOCK_N_WARPS * WARP_N;
        WarpTile<scalar_t, BLOCK_K, WARP_M_THREADS, WARP_N_THREADS, VEC_M, VEC_N, BLOCK_M + PAD, BLOCK_N + PAD> warp_tile;
#pragma unroll
        for (int i = 0; i < WARP_M_STEPS; ++i) {
            int warp_atom_offset_y = warp_y + i * WARP_ATOM_M;
#pragma unroll
            for (int j = 0; j < WARP_N_STEPS; ++j) {
                int warp_atom_offset_x = warp_x + j * WARP_ATOM_N;
                warp_tile(o[i * WARP_N_STEPS + j], as, bs, warp_atom_offset_y, warp_atom_offset_x, w_tid);
            }
        }
    }

private:
    int tid;
    int wid;
    int w_tid;
    int ldg_a_vec_idx;
    int ldg_b_vec_idx;
    ldg_vec_t ldg_a_reg[LDG_REG_A_COUNT];
    ldg_vec_t ldg_b_reg[LDG_REG_B_COUNT];
};

template <
    typename scalar_t,
    int BLOCK_K,
    int BLOCK_M_WARPS,
    int BLOCK_N_WARPS,
    int WARP_M_STEPS,
    int WARP_N_STEPS,
    int WARP_M_THREADS,
    int WARP_N_THREADS,
    int VEC_M,
    int VEC_N,
    int WARP_SIZE = 32,
    int STAGES = 2>
__global__ void sgemm_kernel(
    scalar_t *c,
    const scalar_t *a,
    const scalar_t *b,
    const int m,
    const int n,
    const int k) {
    constexpr int WARP_M = WARP_M_STEPS * WARP_M_THREADS * VEC_M;
    constexpr int WARP_N = WARP_N_STEPS * WARP_N_THREADS * VEC_N;
    constexpr int BLOCK_M = BLOCK_M_WARPS * WARP_M;
    constexpr int BLOCK_N = BLOCK_N_WARPS * WARP_N;
    using BlockTileT = BlockTile<scalar_t, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS,
                                 WARP_M_STEPS, WARP_N_STEPS, WARP_M_THREADS, WARP_N_THREADS, VEC_M, VEC_N, WARP_SIZE>;

    // get idx
    int tid = threadIdx.x;
    int wid = tid >> BlockTileT::WARP_SHIFT;
    int w_tid = tid & BlockTileT::WARP_MASK;
    int mi = blockIdx.y;
    int ni = blockIdx.x;

    // get slm
    constexpr int AS_STRIDE = BLOCK_M + BlockTileT::PAD;
    constexpr int BS_STRIDE = BLOCK_N + BlockTileT::PAD;
    __shared__ scalar_t as[STAGES][BLOCK_K * AS_STRIDE];
    __shared__ scalar_t bs[STAGES][BLOCK_K * BS_STRIDE];

    // init regs
    scalar_t o_reg[WARP_M_STEPS * WARP_N_STEPS][VEC_M * VEC_N] = {{(scalar_t)0}};
    BlockTileT block_tile(tid);
    int current_stage = 0;
    int a_begin = mi * BLOCK_M * k;
    int b_begin = ni * BLOCK_N * k;
    int a_end = a_begin + k;

    for (; a_begin < a_end; a_begin += BLOCK_K, b_begin += BLOCK_K) {
        block_tile.ldg(&a[a_begin], k, &b[b_begin], k);
        block_tile.sts(as[current_stage], bs[current_stage]);
        __syncthreads();
        block_tile(o_reg, as[current_stage], bs[current_stage]);
        current_stage = (current_stage + 1) % STAGES;
    }

    { // write back
        using stg_vec_t = aligned_array<scalar_t, VEC_N>;
        int out_warp_y = mi * BLOCK_M + wid / BLOCK_N_WARPS * WARP_M;
        int out_warp_x = ni * BLOCK_N + wid % BLOCK_N_WARPS * WARP_N;
        constexpr int WARP_ATOM_M = WARP_M / WARP_M_STEPS;
        constexpr int WARP_ATOM_N = WARP_N / WARP_N_STEPS;
#pragma unroll
        for (int lm = 0; lm < WARP_M_STEPS; lm++) {
#pragma unroll
            for (int ln = 0; ln < WARP_N_STEPS; ln++) {
                int out_thread_y = out_warp_y + lm * WARP_ATOM_M + w_tid / WARP_N_THREADS * VEC_M;
                int out_thread_x = out_warp_x + ln * WARP_ATOM_N + w_tid % WARP_N_THREADS * VEC_N;
#pragma unroll
                for (int i = 0; i < VEC_M; i++) {
                    int y = out_thread_y + i;
                    if (y < m && out_thread_x < n) {
                        stg_vec_t vec;
#pragma unroll
                        for (int j = 0; j < VEC_N; j++) {
                            vec.val[j] = o_reg[lm * WARP_N_STEPS + ln][i * VEC_N + j];
                        }
                        *reinterpret_cast<stg_vec_t *>(c + y * n + out_thread_x) = vec;
                    }
                }
            }
        }
    }
}

std::tuple<dim3, int> get_grid(int m, int n, int BLOCK_M, int BLOCK_N) {
    int bm = (m + BLOCK_M - 1) / BLOCK_M;
    int bn = (n + BLOCK_N - 1) / BLOCK_N;
    int raster_factor = 1;
    return {dim3(bn, bm, 1), raster_factor};
}

#define GET_SGEMM_IMPL_NAME(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_SIZE, STAGES) \
    sgemm_##BLOCK_M##x##BLOCK_N##x##BLOCK_K##_w##BLOCK_M_WARPS##x##BLOCK_N_WARPS##x##WARP_SIZE##_s##STAGES##_

#define REGISTER_SGEMM_IMPL(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_SIZE, STAGES)                                          \
    void GET_SGEMM_IMPL_NAME(BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_SIZE, STAGES)(                                        \
        float *c, const float *a, const float *b, const int m, const int n, const int k, gpuStream_t stream) {                                   \
        constexpr int VEC_SIZE = 16 / sizeof(float);                                                                                             \
        assert(n % VEC_SIZE == 0);                                                                                                               \
        assert(k % VEC_SIZE == 0);                                                                                                               \
        auto gr = get_grid(m, n, BLOCK_M, BLOCK_N);                                                                                              \
        dim3 grid = std::get<0>(gr);                                                                                                             \
        constexpr int BLOCK_SIZE = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE;                                                                    \
        dim3 block(BLOCK_SIZE);                                                                                                                  \
        constexpr int WARP_M_THREADS = WARP_SIZE / 8;                                                                                            \
        constexpr int WARP_N_THREADS = 8;                                                                                                        \
        constexpr int WARP_M_STEPS = BLOCK_M / BLOCK_M_WARPS / WARP_M_THREADS / 4;                                                               \
        constexpr int WARP_N_STEPS = BLOCK_N / BLOCK_N_WARPS / WARP_N_THREADS / 4;                                                               \
        sgemm_kernel<float, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_M_STEPS, WARP_N_STEPS,                                                   \
                     WARP_M_THREADS, WARP_N_THREADS, /*VEC_M*/ 4, /*VEC_N*/ 4, WARP_SIZE, STAGES><<<grid, block, 0, stream>>>(c, a, b, m, n, k); \
    }

#ifdef __CUDACC__

REGISTER_SGEMM_IMPL(/*BLOCK_M*/ 64, /*BLOCK_N*/ 64, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 4, /*BLOCK_N_WARPS*/ 2, /*WARP_SIZE*/ 32, /*STAGES*/ 2)
REGISTER_SGEMM_IMPL(/*BLOCK_M*/ 32, /*BLOCK_N*/ 64, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 2, /*WARP_SIZE*/ 32, /*STAGES*/ 2)
REGISTER_SGEMM_IMPL(/*BLOCK_M*/ 128, /*BLOCK_N*/ 128, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 4, /*BLOCK_N_WARPS*/ 2, /*WARP_SIZE*/ 32, /*STAGES*/ 2)
REGISTER_SGEMM_IMPL(/*BLOCK_M*/ 64, /*BLOCK_N*/ 128, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 2, /*WARP_SIZE*/ 32, /*STAGES*/ 2)

void sgemm_peak(
    float *c,
    const float *a,
    const float *b,
    const int m,
    const int n,
    const int k,
    gpuStream_t stream) {
    GET_SGEMM_IMPL_NAME(/*BLOCK_M*/ 128, /*BLOCK_N*/ 128, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 4, /*BLOCK_N_WARPS*/ 2, /*WARP_SIZE*/ 32, /*STAGES*/ 2)
    (c, a, b, m, n, k, stream);
}

#elif defined(__HIPCC__)

REGISTER_SGEMM_IMPL(/*BLOCK_M*/ 128, /*BLOCK_N*/ 128, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 2, /*WARP_SIZE*/ 64, /*STAGES*/ 2)

void sgemm_peak(
    float *c,
    const float *a,
    const float *b,
    const int m,
    const int n,
    const int k,
    gpuStream_t stream) {
    GET_SGEMM_IMPL_NAME(/*BLOCK_M*/ 128, /*BLOCK_N*/ 128, /*BLOCK_K*/ 16, /*BLOCK_M_WARPS*/ 2, /*BLOCK_N_WARPS*/ 2, /*WARP_SIZE*/ 64, /*STAGES*/ 2)
    (c, a, b, m, n, k, stream);
}

#endif

} // namespace sgemm
