#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "peak_gemm/backend/arch_sm100.hpp"
#include "peak_gemm/core/block_swizzle.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::kernel {

namespace sm100_hgemm {

constexpr uint32_t ClusterM = 256;
constexpr uint32_t ClusterN = 256;
constexpr uint32_t BlockM = 128;
constexpr uint32_t BlockK = 64;
constexpr uint32_t MmaK = 16;
constexpr uint32_t Stages = 6;
constexpr uint32_t Threads = 192;
constexpr uint32_t EpilogueWarps = 4;
constexpr uint32_t TmaWarp = 4;
constexpr uint32_t MmaWarp = 5;
constexpr uint32_t OutputStripeN = 64;
constexpr uint32_t TmemColumns = ClusterN;
constexpr uint32_t SwizzleM = 8;

template <typename scalar_t>
struct SharedStorage {
    union {
        struct {
            alignas(1024) scalar_t a[Stages][BlockM * BlockK];
            alignas(1024) scalar_t b[Stages][BlockM * BlockK];
        } mainloop;
        struct {
            alignas(1024)
                scalar_t output[2][BlockM * OutputStripeN];
            scalar_t bias[ClusterN];
        } epilogue;
    };
    struct {
        alignas(8) uint64_t tma[Stages];
        alignas(8) uint64_t mma[Stages];
        alignas(8) uint64_t mainloop_done;
        uint32_t tmem_base;
    } synchronization;
};

template <typename scalar_t>
PEAKGEMM_DEVICE_INLINE void store_tmem_stripe(
    SharedStorage<scalar_t> &storage,
    uint32_t rank,
    uint32_t warp,
    uint32_t stripe,
    uint32_t segment,
    bool has_bias) {
    namespace arch = backend::sm100;
    using vector_t = core::Vector<scalar_t, 8>;

    const uint32_t local_row = threadIdx.x;
    const uint32_t tmem_row = rank * BlockM + warp * 32;
    const uint32_t column =
        stripe * OutputStripeN + segment * 16;
    const uint32_t tmem_address =
        (tmem_row << 16) + storage.synchronization.tmem_base + column;

    float accumulator[16];
    arch::tmem_load_x16(accumulator, tmem_address);

    vector_t low;
    vector_t high;
#pragma unroll
    for (uint32_t element = 0; element < 8; ++element) {
        float low_value = accumulator[element];
        float high_value = accumulator[element + 8];
        if (has_bias) {
            low_value += static_cast<float>(
                storage.epilogue.bias[column + element]);
            high_value += static_cast<float>(
                storage.epilogue.bias[column + element + 8]);
        }
        low[element] = static_cast<scalar_t>(low_value);
        high[element] = static_cast<scalar_t>(high_value);
    }

    const uint32_t buffer = stripe & 1U;
    const uint32_t low_chunk =
        arch::swizzle_128b_chunk(local_row, segment * 2);
    const uint32_t high_chunk =
        arch::swizzle_128b_chunk(local_row, segment * 2 + 1);
    low.store(storage.epilogue.output[buffer] + low_chunk * 8);
    high.store(storage.epilogue.output[buffer] + high_chunk * 8);
}

template <typename scalar_t, bool HasBias>
__global__ __launch_bounds__(Threads, 1) void hgemm_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ CUtensorMap c_map,
    const scalar_t *bias,
    uint32_t m_blocks,
    uint32_t n_blocks,
    uint32_t k_tiles) {
    namespace arch = backend::sm100;
    static_assert(
        std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __bfloat16>);
    arch::require_sm100a();

    extern __shared__ __align__(1024) char shared_bytes[];
    auto &storage =
        *reinterpret_cast<SharedStorage<scalar_t> *>(shared_bytes);

    const uint32_t thread = threadIdx.x;
    const uint32_t warp = thread / 32;
    const uint32_t lane = thread % 32;
    const uint32_t rank = arch::cluster_rank();
    const uint32_t cluster = blockIdx.x / 2;
    const auto tile =
        core::block_swizzle<SwizzleM>(cluster, m_blocks, n_blocks);
    const uint32_t row =
        tile.m * ClusterM + rank * BlockM;
    const uint32_t column = tile.n * ClusterN;

    if (thread == 0) {
#pragma unroll
        for (uint32_t stage = 0; stage < Stages; ++stage) {
            arch::mbarrier_init(
                &storage.synchronization.tma[stage],
                2);
            arch::mbarrier_init(
                &storage.synchronization.mma[stage],
                1);
        }
        arch::mbarrier_init(
            &storage.synchronization.mainloop_done,
            1);
        arch::mbarrier_init_fence();
    }
    __syncthreads();
    arch::cluster_sync();

    if (warp == MmaWarp) {
        arch::tmem_allocate(
            &storage.synchronization.tmem_base,
            TmemColumns);
    }
    __syncthreads();

    if (warp == TmaWarp) {
        if (lane == 0) {
            uint32_t stage = 0;
            uint32_t mma_phase = 1;
            for (uint32_t k_tile = 0; k_tile < k_tiles; ++k_tile) {
                arch::mbarrier_wait(
                    &storage.synchronization.mma[stage],
                    mma_phase);

                arch::tma_load_3d_2cta(
                    storage.mainloop.a[stage],
                    &a_map,
                    0,
                    row,
                    k_tile,
                    &storage.synchronization.tma[stage]);
                arch::tma_load_3d_2cta(
                    storage.mainloop.b[stage],
                    &b_map,
                    0,
                    column + rank * BlockM,
                    k_tile,
                    &storage.synchronization.tma[stage]);
                arch::mbarrier_arrive_expect_tx(
                    &storage.synchronization.tma[stage],
                    2 * BlockM * BlockK * sizeof(scalar_t));

                stage = (stage + 1) % Stages;
                if (stage == 0) {
                    mma_phase ^= 1U;
                }
            }
        }
    } else if (warp == MmaWarp) {
        __syncwarp();
        if (rank == 0 && lane == 0) {
            uint32_t stage = 0;
            uint32_t tma_phase = 0;
            const uint32_t instruction =
                arch::mma_instruction_descriptor<scalar_t>();

            for (uint32_t k_tile = 0;
                 k_tile < k_tiles;
                 ++k_tile) {
                arch::mbarrier_wait(
                    &storage.synchronization.tma[stage],
                    tma_phase);
                arch::tcgen05_fence();

                uint64_t descriptor_a =
                    arch::make_smem_descriptor(
                        storage.mainloop.a[stage]);
                uint64_t descriptor_b =
                    arch::make_smem_descriptor(
                        storage.mainloop.b[stage]);
#pragma unroll
                for (uint32_t k_step = 0;
                     k_step < BlockK / MmaK;
                     ++k_step) {
                    arch::mma_f16_2cta(
                        storage.synchronization.tmem_base,
                        descriptor_a,
                        descriptor_b,
                        instruction,
                        k_tile != 0 || k_step != 0);
                    descriptor_a += 2;
                    descriptor_b += 2;
                }
                arch::mma_commit_multicast(
                    &storage.synchronization.mma[stage],
                    0x3);

                stage = (stage + 1) % Stages;
                if (stage == 0) {
                    tma_phase ^= 1U;
                }
            }

            arch::mma_commit_multicast(
                &storage.synchronization.mainloop_done,
                0x3);
        }
    } else if (warp < EpilogueWarps) {
        if (warp == 0 && lane == 0) {
            arch::mbarrier_wait(
                &storage.synchronization.mainloop_done,
                0);
        }
        arch::named_barrier(1, EpilogueWarps * 32);
        arch::tcgen05_fence();

        if constexpr (HasBias) {
            for (uint32_t index = thread;
                 index < ClusterN;
                 index += EpilogueWarps * 32) {
                storage.epilogue.bias[index] =
                    bias[column + index];
            }
            arch::named_barrier(1, EpilogueWarps * 32);
        }

#pragma unroll
        for (uint32_t stripe = 0;
             stripe < ClusterN / OutputStripeN;
             ++stripe) {
            if (stripe >= 2) {
                if (warp == 0 && lane == 0) {
                    arch::tma_store_wait_read<1>();
                }
                arch::named_barrier(1, EpilogueWarps * 32);
            }

#pragma unroll
            for (uint32_t segment = 0; segment < 4; ++segment) {
                store_tmem_stripe(
                    storage,
                    rank,
                    warp,
                    stripe,
                    segment,
                    HasBias);
            }
            arch::fence_proxy_async_shared();
            arch::named_barrier(1, EpilogueWarps * 32);

            if (warp == 0 && lane == 0) {
                const uint32_t buffer = stripe & 1U;
                arch::tma_store_2d(
                    &c_map,
                    column + stripe * OutputStripeN,
                    row,
                    storage.epilogue.output[buffer]);
                arch::tma_store_commit();
            }
            arch::named_barrier(1, EpilogueWarps * 32);
        }

        if (warp == 0 && lane == 0) {
            arch::tma_store_wait_read<0>();
        }
        arch::named_barrier(1, EpilogueWarps * 32);
    }

    __syncthreads();
    arch::cluster_sync();
    if (warp == MmaWarp) {
        arch::tmem_deallocate(
            storage.synchronization.tmem_base,
            TmemColumns);
        arch::tmem_relinquish();
    }
}

} // namespace sm100_hgemm

template <
    typename scalar_t,
    bool HasBias,
    bool IsSplitK = false>
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
    static_assert(
        std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __bfloat16>);
    static_assert(!IsSplitK, "SM100 HGEMM does not support split-K");
    static_cast<void>(semaphore);
    static_cast<void>(signal);

    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument(
            "SM100 HGEMM tensor pointers must not be null");
    }
    if constexpr (HasBias) {
        if (bias == nullptr) {
            throw std::invalid_argument(
                "SM100 HGEMM bias specialization requires bias");
        }
    }
    if (split_k != 1) {
        throw std::invalid_argument(
            "SM100 HGEMM requires split_k == 1");
    }
    if (m == 0 || n == 0 || k == 0 || m % impl::ClusterM != 0 || n % impl::ClusterN != 0 || k % impl::BlockK != 0) {
        throw std::invalid_argument(
            "SM100 HGEMM requires M,N % 256 and K % 64");
    }

    const uint64_t a_dimensions[3] = {impl::BlockK, m, k / impl::BlockK};
    const uint64_t b_dimensions[3] = {impl::BlockK, n, k / impl::BlockK};
    const uint64_t input_strides[2] = {
        static_cast<uint64_t>(k) * sizeof(scalar_t),
        impl::BlockK * sizeof(scalar_t)};
    const uint32_t input_box[3] = {
        impl::BlockK,
        impl::BlockM,
        1};
    const uint32_t input_element_strides[3] = {1, 1, 1};

    const uint64_t c_dimensions[2] = {n, m};
    const uint64_t c_strides[1] = {
        static_cast<uint64_t>(n) * sizeof(scalar_t)};
    const uint32_t c_box[2] = {
        impl::OutputStripeN,
        impl::BlockM};
    const uint32_t c_element_strides[2] = {1, 1};

    const backend::sm100::TensorMap<scalar_t, 3> a_map(
        a,
        a_dimensions,
        input_strides,
        input_box,
        input_element_strides);
    const backend::sm100::TensorMap<scalar_t, 3> b_map(
        b,
        b_dimensions,
        input_strides,
        input_box,
        input_element_strides);
    const backend::sm100::TensorMap<scalar_t, 2> c_map(
        c,
        c_dimensions,
        c_strides,
        c_box,
        c_element_strides);

    const uint32_t m_blocks = m / impl::ClusterM;
    const uint32_t n_blocks = n / impl::ClusterN;
    const uint32_t clusters = m_blocks * n_blocks;
    const uint32_t shared_bytes =
        sizeof(impl::SharedStorage<scalar_t>);
    const auto kernel =
        impl::hgemm_kernel<scalar_t, HasBias>;

    cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_bytes);

    cudaLaunchAttribute cluster_attribute{};
    cluster_attribute.id = cudaLaunchAttributeClusterDimension;
    cluster_attribute.val.clusterDim = {2, 1, 1};

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(clusters * 2);
    config.blockDim = dim3(impl::Threads);
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
        k / impl::BlockK);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

} // namespace peak_gemm::kernel
