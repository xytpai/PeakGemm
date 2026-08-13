#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "peak_gemm/backend/arch_sm100.hpp"
#include "peak_gemm/core/block_swizzle.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::kernel {

namespace sm100_hgemm {

constexpr uint32_t MMA_K = 16;
constexpr uint32_t OUTPUT_STRIPE_N = 64;

template <typename scalar_t, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t STAGES>
struct SharedStorage {
    static constexpr uint32_t B_ROWS = BLOCK_N / 2;

    static_assert(BLOCK_M == 128, "SM100 2CTA Layout-A epilogue requires BLOCK_M == 128");
    static_assert(BLOCK_N >= OUTPUT_STRIPE_N && BLOCK_N <= 256 && BLOCK_N % OUTPUT_STRIPE_N == 0 && (BLOCK_N & (BLOCK_N - 1)) == 0,
                  "SM100 BLOCK_N must be 64, 128, or 256");
    static_assert(BLOCK_K == 64, "SM100 SW128 mainloop requires BLOCK_K == 64");
    static_assert(STAGES >= 2, "SM100 pipeline requires at least 2 stages");

    union {
        struct {
            alignas(1024) scalar_t a[STAGES][BLOCK_M * BLOCK_K];
            alignas(1024) scalar_t b[STAGES][B_ROWS * BLOCK_K];
        } mainloop;
        struct {
            alignas(1024) scalar_t output[2][BLOCK_M * OUTPUT_STRIPE_N];
            scalar_t bias[BLOCK_N];
        } epilogue;
    };
    struct {
        alignas(8) uint64_t tma[STAGES];
        alignas(8) uint64_t mma[STAGES];
        alignas(8) uint64_t mainloop_done;
        uint32_t tmem_base;
    } synchronization;
};

template <typename scalar_t, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t STAGES, bool HAS_BIAS>
PEAKGEMM_DEVICE_INLINE void store_tmem_stripe(
    SharedStorage<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K, STAGES> &storage, uint32_t rank, uint32_t warp, uint32_t stripe, uint32_t segment) {
    namespace arch = backend::sm100;
    using vector_t = core::Vector<scalar_t, 8>;

    const uint32_t local_row = threadIdx.x;
    const uint32_t tmem_row = rank * BLOCK_M + warp * 32;
    const uint32_t column = stripe * OUTPUT_STRIPE_N + segment * 16;
    const uint32_t tmem_address = (tmem_row << 16) + storage.synchronization.tmem_base + column;

    float accumulator[16];
    arch::tmem_load_x16(accumulator, tmem_address);

    vector_t low;
    vector_t high;
#pragma unroll
    for (uint32_t element = 0; element < 8; ++element) {
        float low_value = accumulator[element];
        float high_value = accumulator[element + 8];
        if constexpr (HAS_BIAS) {
            low_value += static_cast<float>(storage.epilogue.bias[column + element]);
            high_value += static_cast<float>(storage.epilogue.bias[column + element + 8]);
        }
        low[element] = static_cast<scalar_t>(low_value);
        high[element] = static_cast<scalar_t>(high_value);
    }

    const uint32_t buffer = stripe & 1U;
    const uint32_t low_chunk = arch::swizzle_128b_chunk(local_row, segment * 2);
    const uint32_t high_chunk = arch::swizzle_128b_chunk(local_row, segment * 2 + 1);
    low.store(storage.epilogue.output[buffer] + low_chunk * 8);
    high.store(storage.epilogue.output[buffer] + high_chunk * 8);
}

template <typename scalar_t, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t STAGES, uint32_t SWIZZLE_M, bool HAS_BIAS>
__global__ __launch_bounds__((BLOCK_M / 32 + 2) * 32, 1) void hgemm_kernel(
    const __grid_constant__ CUtensorMap a_map, const __grid_constant__ CUtensorMap b_map, const __grid_constant__ CUtensorMap c_map,
    const scalar_t *bias, uint32_t m_blocks, uint32_t n_blocks, uint32_t k_tiles) {
    namespace arch = backend::sm100;
    constexpr uint32_t CLUSTER_M = 2 * BLOCK_M;
    constexpr uint32_t B_ROWS = BLOCK_N / 2;
    constexpr uint32_t EPILOGUE_WARPS = BLOCK_M / 32;
    constexpr uint32_t TMA_WARP = EPILOGUE_WARPS;
    constexpr uint32_t MMA_WARP = EPILOGUE_WARPS + 1;
    static_assert(std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __bfloat16>);
    arch::require_sm100a();

    extern __shared__ __align__(1024) char shared_bytes[];
    auto &storage = *reinterpret_cast<SharedStorage<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K, STAGES> *>(shared_bytes);

    const uint32_t thread = threadIdx.x;
    const uint32_t warp = thread / 32;
    const uint32_t lane = thread % 32;
    const uint32_t rank = arch::cluster_rank();
    const uint32_t cluster = blockIdx.x / 2;
    const auto tile = core::block_swizzle<SWIZZLE_M>(cluster, m_blocks, n_blocks);
    const uint32_t row = tile.m * CLUSTER_M + rank * BLOCK_M;
    const uint32_t column = tile.n * BLOCK_N;

    if (thread == 0) {
#pragma unroll
        for (uint32_t stage = 0; stage < STAGES; ++stage) {
            arch::mbarrier_init(&storage.synchronization.tma[stage], 2);
            arch::mbarrier_init(&storage.synchronization.mma[stage], 1);
        }
        arch::mbarrier_init(&storage.synchronization.mainloop_done, 1);
        arch::mbarrier_init_fence();
    }
    __syncthreads();
    arch::cluster_sync();

    if (warp == MMA_WARP) {
        arch::tmem_allocate(&storage.synchronization.tmem_base, BLOCK_N);
    }
    __syncthreads();

    if (warp == TMA_WARP) {
        if (lane == 0) {
            uint32_t stage = 0;
            uint32_t mma_phase = 1;
            for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                arch::mbarrier_wait(&storage.synchronization.mma[stage], mma_phase);
                arch::tma_load_3d_2cta(storage.mainloop.a[stage], &a_map, 0, row, k_tile, &storage.synchronization.tma[stage]);
                arch::tma_load_3d_2cta(storage.mainloop.b[stage], &b_map, 0, column + rank * B_ROWS, k_tile, &storage.synchronization.tma[stage]);
                arch::mbarrier_arrive_expect_tx(&storage.synchronization.tma[stage], (BLOCK_M + B_ROWS) * BLOCK_K * sizeof(scalar_t));

                stage = (stage + 1) % STAGES;
                if (stage == 0) mma_phase ^= 1U;
            }
        }
    } else if (warp == MMA_WARP) {
        __syncwarp();
        if (rank == 0 && lane == 0) {
            uint32_t stage = 0;
            uint32_t tma_phase = 0;
            const uint32_t instruction = arch::mma_instruction_descriptor<scalar_t, CLUSTER_M, BLOCK_N>();

            for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                arch::mbarrier_wait(&storage.synchronization.tma[stage], tma_phase);
                arch::tcgen05_fence();

                uint64_t descriptor_a = arch::make_smem_descriptor(storage.mainloop.a[stage]);
                uint64_t descriptor_b = arch::make_smem_descriptor(storage.mainloop.b[stage]);
#pragma unroll
                for (uint32_t k_step = 0; k_step < BLOCK_K / MMA_K; ++k_step) {
                    arch::mma_f16_2cta(storage.synchronization.tmem_base, descriptor_a, descriptor_b, instruction, k_tile != 0 || k_step != 0);
                    descriptor_a += 2;
                    descriptor_b += 2;
                }
                arch::mma_commit_multicast(&storage.synchronization.mma[stage], 0x3);

                stage = (stage + 1) % STAGES;
                if (stage == 0) tma_phase ^= 1U;
            }

            arch::mma_commit_multicast(&storage.synchronization.mainloop_done, 0x3);
        }
    } else if (warp < EPILOGUE_WARPS) {
        if (warp == 0 && lane == 0) {
            arch::mbarrier_wait(&storage.synchronization.mainloop_done, 0);
        }
        arch::named_barrier(1, EPILOGUE_WARPS * 32);
        arch::tcgen05_fence();

        if constexpr (HAS_BIAS) {
            for (uint32_t index = thread; index < BLOCK_N; index += EPILOGUE_WARPS * 32) {
                storage.epilogue.bias[index] = bias[column + index];
            }
            arch::named_barrier(1, EPILOGUE_WARPS * 32);
        }

#pragma unroll
        for (uint32_t stripe = 0; stripe < BLOCK_N / OUTPUT_STRIPE_N; ++stripe) {
            if (stripe >= 2) {
                if (warp == 0 && lane == 0) arch::tma_store_wait_read<1>();
                arch::named_barrier(1, EPILOGUE_WARPS * 32);
            }

#pragma unroll
            for (uint32_t segment = 0; segment < 4; ++segment) {
                store_tmem_stripe<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K, STAGES, HAS_BIAS>(storage, rank, warp, stripe, segment);
            }
            arch::fence_proxy_async_shared();
            arch::named_barrier(1, EPILOGUE_WARPS * 32);

            if (warp == 0 && lane == 0) {
                const uint32_t buffer = stripe & 1U;
                arch::tma_store_2d(&c_map, column + stripe * OUTPUT_STRIPE_N, row, storage.epilogue.output[buffer]);
                arch::tma_store_commit();
            }
            arch::named_barrier(1, EPILOGUE_WARPS * 32);
        }

        if (warp == 0 && lane == 0) arch::tma_store_wait_read<0>();
        arch::named_barrier(1, EPILOGUE_WARPS * 32);
    }

    __syncthreads();
    arch::cluster_sync();
    if (warp == MMA_WARP) {
        arch::tmem_deallocate(storage.synchronization.tmem_base, BLOCK_N);
        arch::tmem_relinquish();
    }
}

} // namespace sm100_hgemm

template <
    typename scalar_t,
    uint32_t BLOCK_M,
    uint32_t BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t STAGES,
    uint32_t SWIZZLE_M,
    bool HAS_BIAS,
    bool IS_SPLIT_K>
void hgemm_template(
    const scalar_t *a,
    const scalar_t *b,
    scalar_t *c,
    uint32_t m,
    uint32_t n,
    uint32_t k,
    uint32_t split_k,
    uint32_t *semaphore,
    uint32_t *signal,
    const scalar_t *bias = nullptr,
    gpuStream_t stream = nullptr) {
    namespace impl = sm100_hgemm;
    using SharedStorageT =
        impl::SharedStorage<
            scalar_t,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            STAGES>;
    constexpr uint32_t CLUSTER_M = 2 * BLOCK_M;
    constexpr uint32_t B_ROWS = BLOCK_N / 2;
    constexpr uint32_t BLOCK_THREADS =
        (BLOCK_M / 32 + 2) * 32;
    static_assert(
        std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __bfloat16>);
    static_assert(!IS_SPLIT_K, "SM100 HGEMM does not support split-K");
    static_assert(
        sizeof(SharedStorageT) <= 227 * 1024,
        "SM100 HGEMM tile exceeds B200 shared-memory capacity");
    static_cast<void>(semaphore);
    static_cast<void>(signal);

    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument(
            "SM100 HGEMM tensor pointers must not be null");
    }
    if constexpr (HAS_BIAS) {
        if (bias == nullptr) {
            throw std::invalid_argument(
                "SM100 HGEMM bias specialization requires bias");
        }
    }
    if (split_k != 1) {
        throw std::invalid_argument(
            "SM100 HGEMM requires split_k == 1");
    }
    if (m == 0 || n == 0 || k == 0 || m % CLUSTER_M != 0 || n % BLOCK_N != 0 || k % BLOCK_K != 0) {
        throw std::invalid_argument(
            "SM100 HGEMM dimensions must be tile aligned");
    }

    const uint64_t a_dimensions[3] = {
        BLOCK_K,
        m,
        k / BLOCK_K};
    const uint64_t b_dimensions[3] = {
        BLOCK_K,
        n,
        k / BLOCK_K};
    const uint64_t input_strides[2] = {
        static_cast<uint64_t>(k) * sizeof(scalar_t),
        BLOCK_K * sizeof(scalar_t)};
    const uint32_t a_box[3] = {
        BLOCK_K,
        BLOCK_M,
        1};
    const uint32_t b_box[3] = {
        BLOCK_K,
        B_ROWS,
        1};
    const uint32_t input_element_strides[3] = {1, 1, 1};

    const uint64_t c_dimensions[2] = {n, m};
    const uint64_t c_strides[1] = {
        static_cast<uint64_t>(n) * sizeof(scalar_t)};
    const uint32_t c_box[2] = {
        impl::OUTPUT_STRIPE_N,
        BLOCK_M};
    const uint32_t c_element_strides[2] = {1, 1};

    const backend::sm100::TensorMap<scalar_t, 3> a_map(
        a,
        a_dimensions,
        input_strides,
        a_box,
        input_element_strides);
    const backend::sm100::TensorMap<scalar_t, 3> b_map(
        b,
        b_dimensions,
        input_strides,
        b_box,
        input_element_strides);
    const backend::sm100::TensorMap<scalar_t, 2> c_map(
        c,
        c_dimensions,
        c_strides,
        c_box,
        c_element_strides);

    const uint32_t m_blocks = m / CLUSTER_M;
    const uint32_t n_blocks = n / BLOCK_N;
    const uint32_t clusters = m_blocks * n_blocks;
    const uint32_t shared_bytes = sizeof(SharedStorageT);
    const auto kernel =
        impl::hgemm_kernel<
            scalar_t,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            STAGES,
            SWIZZLE_M,
            HAS_BIAS>;

    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes);

    cudaLaunchAttribute cluster_attribute{};
    cluster_attribute.id = cudaLaunchAttributeClusterDimension;
    cluster_attribute.val.clusterDim = {2, 1, 1};

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(clusters * 2);
    config.blockDim = dim3(BLOCK_THREADS);
    config.dynamicSmemBytes = shared_bytes;
    config.stream = stream;
    config.attrs = &cluster_attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(
        &config,
        kernel,
        a_map.descriptor(),
        b_map.descriptor(),
        c_map.descriptor(),
        bias,
        m_blocks,
        n_blocks,
        k / BLOCK_K);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

} // namespace peak_gemm::kernel
