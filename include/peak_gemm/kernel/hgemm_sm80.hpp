#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "peak_gemm/backend/runtime.hpp"
#include "peak_gemm/core/block_swizzle.hpp"
#include "peak_gemm/core/math.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::kernel {

inline constexpr uint32_t kSemaphoreCount = 256;

template <typename scalar_t, typename Wmma, uint32_t BlockK, uint32_t BlockMWarps, uint32_t BlockNWarps, uint32_t BlockKWarps,
          uint32_t WarpMSteps, uint32_t WarpNSteps>
class HgemmBlockTile {
public:
    using FragmentAT = typename Wmma::FragmentAT;
    using FragmentBT = typename Wmma::FragmentBT;
    using FragmentCT = typename Wmma::FragmentCT;
    static_assert(std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __bfloat16>, "CUDA HGEMM supports fp16 and bf16 inputs");
    enum : uint32_t {
        WarpSize = 32,
        WarpMask = WarpSize - 1,
        WarpShift = core::Log2<WarpSize>::value,
        WarpAtomM = Wmma::M,
        WarpAtomN = Wmma::N,
        WarpAtomK = Wmma::K,
        WarpGroupK = BlockKWarps * WarpAtomK,
        WarpKSteps = BlockK / WarpGroupK,
        KSlice = BlockK / BlockKWarps,
        VectorSize = 16 / sizeof(scalar_t),
        BlockThreads = BlockMWarps * BlockNWarps * BlockKWarps * WarpSize,
        BlockMNWarps = BlockMWarps * BlockNWarps,
        WarpM = WarpMSteps * WarpAtomM,
        WarpN = WarpNSteps * WarpAtomN,
        BlockM = BlockMWarps * WarpM,
        BlockN = BlockNWarps * WarpN,
        LoadAXThreads = BlockK / VectorSize,
        LoadBXThreads = BlockK / VectorSize,
        LoadARegisters = BlockM * BlockK / VectorSize / BlockThreads,
        LoadBRegisters = BlockN * BlockK / VectorSize / BlockThreads
    };
    using Vector = core::Vector<scalar_t, VectorSize>;

    static_assert(LoadARegisters >= 1 && LoadBRegisters >= 1);
    static_assert(BlockK % WarpGroupK == 0);
    static_assert(WarpKSteps >= 1 && KSlice % WarpAtomK == 0);

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE explicit HgemmBlockTile(uint32_t thread) : thread_(thread), warp_(thread >> WarpShift), lane_(thread & WarpMask), load_a_vector_(thread % LoadAXThreads), load_b_vector_(thread % LoadBXThreads),
                                                                                    warp_mn_(warp_ % BlockMNWarps), warp_k_(warp_ / BlockMNWarps) {
        wmma_.init(lane_);
#pragma unroll
        for (uint32_t m = 0; m < WarpMSteps; ++m) {
#pragma unroll
            for (uint32_t n = 0; n < WarpNSteps; ++n) {
                wmma_.reset_fragment_c(output_[m][n]);
            }
        }
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void copy_async(scalar_t *shared_a, scalar_t *shared_b, const scalar_t *a, uint32_t stride_a, const scalar_t *b,
                                                         uint32_t stride_b) {
#pragma unroll
        for (uint32_t i = 0; i < LoadARegisters; ++i) {
            const uint32_t thread = BlockThreads * i + thread_;
            const uint32_t shared_offset = wmma_.swizzle(thread * VectorSize);
            const auto *source = a + thread / LoadAXThreads * stride_a + load_a_vector_ * VectorSize;
            backend::AsyncCopy::copy(reinterpret_cast<Vector *>(shared_a + shared_offset), reinterpret_cast<const Vector *>(source));
        }
#pragma unroll
        for (uint32_t i = 0; i < LoadBRegisters; ++i) {
            const uint32_t thread = BlockThreads * i + thread_;
            const uint32_t shared_offset = wmma_.swizzle(thread * VectorSize);
            const auto *source = b + thread / LoadBXThreads * stride_b + load_b_vector_ * VectorSize;
            backend::AsyncCopy::copy(reinterpret_cast<Vector *>(shared_b + shared_offset), reinterpret_cast<const Vector *>(source));
        }
    }

    template <int PendingGroups>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void wait() {
        backend::AsyncCopy::template wait<PendingGroups>();
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void commit() {
        backend::AsyncCopy::commit();
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void compute(scalar_t *shared_a, scalar_t *shared_b) {
        const uint32_t warp_m = warp_mn_ / BlockNWarps * WarpM;
        const uint32_t warp_n = warp_mn_ % BlockNWarps * WarpN;
        const uint32_t warp_k = warp_k_ * KSlice;
#pragma unroll
        for (uint32_t k = 0; k < WarpKSteps; ++k) {
            const uint32_t column = warp_k + k * WarpAtomK;
            FragmentAT fragment_a[WarpMSteps];
            FragmentBT fragment_b[WarpNSteps];
#pragma unroll
            for (uint32_t n = 0; n < WarpNSteps; ++n) {
                wmma_.load_matrix_b(fragment_b[n], shared_b, warp_n + n * WarpAtomN, column, BlockK);
            }
#pragma unroll
            for (uint32_t m = 0; m < WarpMSteps; ++m) {
                wmma_.load_matrix_a(fragment_a[m], shared_a, warp_m + m * WarpAtomM, column, BlockK);
            }
#pragma unroll
            for (uint32_t m = 0; m < WarpMSteps; ++m) {
#pragma unroll
                for (uint32_t n = 0; n < WarpNSteps; ++n) {
                    wmma_(output_[m][n], fragment_a[m], fragment_b[n], output_[m][n]);
                }
            }
        }
    }

    template <bool UseAtomic>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void store(scalar_t *c, scalar_t (&shared_c)[BlockKWarps][BlockM * BlockN], uint32_t block_m,
                                                    uint32_t block_n, uint32_t m_size, uint32_t n_size, const scalar_t *bias) {
        const uint32_t warp_m = warp_mn_ / BlockNWarps * WarpM;
        const uint32_t warp_n = warp_mn_ % BlockNWarps * WarpN;
        __syncthreads();
#pragma unroll
        for (uint32_t m = 0; m < WarpMSteps; ++m) {
#pragma unroll
            for (uint32_t n = 0; n < WarpNSteps; ++n) {
                auto *destination = &shared_c[warp_k_][(warp_m + m * WarpAtomM) * BlockN + warp_n + n * WarpAtomN];
                wmma_.store_matrix(destination, BlockN, output_[m][n]);
            }
        }
        __syncthreads();
        constexpr uint32_t StoreRegisters = BlockM * BlockN / (BlockThreads * VectorSize);
        constexpr uint32_t StoreXThreads = BlockN / VectorSize;
#pragma unroll
        for (uint32_t i = 0; i < StoreRegisters; ++i) {
            const uint32_t global_thread = BlockThreads * i + thread_;
            const uint32_t local_m = global_thread / StoreXThreads;
            const uint32_t local_n = global_thread % StoreXThreads * VectorSize;
            const uint32_t global_m = block_m * BlockM + local_m;
            const uint32_t global_n = block_n * BlockN + local_n;
            if (global_m >= m_size || global_n >= n_size)
                continue;
            auto value = *reinterpret_cast<Vector *>(&shared_c[0][local_m * BlockN + local_n]);
#pragma unroll
            for (uint32_t k = 1; k < BlockKWarps; ++k) {
                value += *reinterpret_cast<Vector *>(&shared_c[k][local_m * BlockN + local_n]);
            }
            if constexpr (!UseAtomic) {
                if (bias != nullptr) {
                    Vector bias_value;
                    bias_value.load(bias + global_n);
                    value += bias_value;
                }
            }
            auto *destination = c + global_m * n_size + global_n;
            if constexpr (UseAtomic) {
#pragma unroll
                for (uint32_t element = 0; element < VectorSize; element += 2) {
                    backend::atomic_pair_add(destination + element, &value[element]);
                }
            } else {
                value.store(destination);
            }
        }
    }

private:
    uint32_t thread_;
    uint32_t warp_;
    uint32_t lane_;
    uint32_t load_a_vector_;
    uint32_t load_b_vector_;
    uint32_t warp_mn_;
    uint32_t warp_k_;
    Wmma wmma_;
    FragmentCT output_[WarpMSteps][WarpNSteps];
};

template <typename scalar_t, uint32_t Stages, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t BlockKWarps>
union HgemmSharedStorage {
    struct {
        scalar_t a[Stages][BlockM * BlockK];
        scalar_t b[Stages][BlockN * BlockK];
    };
    scalar_t c[BlockKWarps][BlockM * BlockN];
};

template <typename scalar_t, typename Wmma, uint32_t BlockK, uint32_t BlockMWarps, uint32_t BlockNWarps, uint32_t BlockKWarps,
          uint32_t WarpMSteps, uint32_t WarpNSteps, uint32_t Stages, uint32_t SplitK, uint32_t SwizzleM>
__global__ __launch_bounds__(BlockMWarps * BlockNWarps * BlockKWarps * 32,
                             2) void hgemm_kernel(scalar_t *c, const scalar_t *a, const scalar_t *b, uint32_t m_size, uint32_t n_size,
                                                  uint32_t k_size, uint32_t *semaphore, uint32_t *signal, const scalar_t *bias) {
    using BlockTile = HgemmBlockTile<scalar_t, Wmma, BlockK, BlockMWarps, BlockNWarps, BlockKWarps, WarpMSteps, WarpNSteps>;
    constexpr uint32_t BlockM = BlockTile::BlockM;
    constexpr uint32_t BlockN = BlockTile::BlockN;
    constexpr bool IsSplitK = SplitK > 1;
    const uint32_t blocks_m = core::ceil_div(m_size, BlockM);
    const uint32_t blocks_n = core::ceil_div(n_size, BlockN);
    const auto block = core::block_swizzle<SwizzleM>(blockIdx.x, blocks_m, blocks_n);
    const uint32_t block_m = block.m;
    const uint32_t block_n = block.n;
    const uint32_t thread = threadIdx.x;
    const uint32_t partition_k = k_size / SplitK;
    const uint32_t partition = blockIdx.y;
    const uint32_t begin_k = partition * partition_k;
    uint32_t a_begin = block_m * BlockM * k_size + begin_k;
    uint32_t b_begin = block_n * BlockN * k_size + begin_k;
    const uint32_t a_end = a_begin + partition_k;
    __shared__ HgemmSharedStorage<scalar_t, Stages, BlockM, BlockN, BlockK, BlockKWarps> shared;
    BlockTile block_tile(thread);

    if constexpr (IsSplitK) {
        if (partition == 0) {
            constexpr uint32_t VectorSize = BlockTile::VectorSize;
            constexpr uint32_t StoreRegisters = BlockM * BlockN / (BlockTile::BlockThreads * VectorSize);
            constexpr uint32_t StoreXThreads = BlockN / VectorSize;
            core::Vector<scalar_t, VectorSize> initial;
#pragma unroll
            for (uint32_t i = 0; i < StoreRegisters; ++i) {
                const uint32_t global_thread = BlockTile::BlockThreads * i + thread;
                const uint32_t local_m = global_thread / StoreXThreads;
                const uint32_t local_n = global_thread % StoreXThreads * VectorSize;
                const uint32_t global_m = block_m * BlockM + local_m;
                const uint32_t global_n = block_n * BlockN + local_n;
                if (global_m < m_size && global_n < n_size) {
                    if (bias != nullptr) {
                        initial.load(bias + global_n);
                    } else {
                        initial.fill(static_cast<scalar_t>(0));
                    }
                    initial.store(c + global_m * n_size + global_n);
                }
            }
            __threadfence();
            __syncthreads();
            if (thread == 0) {
                backend::atomic_exchange(&signal[blockIdx.x], 1U);
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (uint32_t stage = 0; stage < Stages - 1; ++stage) {
        block_tile.copy_async(shared.a[stage], shared.b[stage], a + a_begin + stage * BlockK, k_size, b + b_begin + stage * BlockK, k_size);
        block_tile.commit();
    }

    uint32_t current_stage = 0;
    for (; a_begin < a_end; a_begin += BlockK, b_begin += BlockK) {
        block_tile.template wait<Stages - 2>();
        __syncthreads();
        block_tile.compute(shared.a[current_stage], shared.b[current_stage]);
        if (a_begin + (Stages - 1) * BlockK < a_end) {
            const uint32_t write_stage = (current_stage + Stages - 1) % Stages;
            block_tile.copy_async(shared.a[write_stage], shared.b[write_stage], a + a_begin + (Stages - 1) * BlockK, k_size,
                                  b + b_begin + (Stages - 1) * BlockK, k_size);
        }
        block_tile.commit();
        current_stage = (current_stage + 1) % Stages;
    }

    if constexpr (IsSplitK) {
        if (thread == 0) {
            while (backend::atomic_add(&signal[blockIdx.x], 0U) == 0) {
            }
        }
        __syncthreads();
        if (thread == 0) {
            const uint32_t arrival =
                backend::atomic_add(&semaphore[blockIdx.x], 1U);
            if (arrival == SplitK - 1) {
                semaphore[blockIdx.x] = 0;
                signal[blockIdx.x] = 0;
            }
        }
        block_tile.template store<true>(c, shared.c, block_m, block_n, m_size, n_size, bias);
    } else {
        block_tile.template store<false>(c, shared.c, block_m, block_n, m_size, n_size, bias);
    }
}

template <typename scalar_t, uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, uint32_t BlockMWarps, uint32_t BlockNWarps,
          uint32_t BlockKWarps, uint32_t Stages, uint32_t SplitK, uint32_t SwizzleM = 8>
void launch_hgemm(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t m, uint32_t n, uint32_t k, uint32_t *semaphore,
                  uint32_t *signal, const scalar_t *bias = nullptr, gpuStream_t stream = nullptr) {
    using Wmma = backend::WmmaDefault<scalar_t, float, true>;
    constexpr uint32_t WarpMSteps = BlockM / BlockMWarps / Wmma::M;
    constexpr uint32_t WarpNSteps = BlockN / BlockNWarps / Wmma::N;
    constexpr uint32_t BlockThreads = BlockMWarps * BlockNWarps * BlockKWarps * 32;
    static_assert(BlockM % (BlockMWarps * Wmma::M) == 0);
    static_assert(BlockN % (BlockNWarps * Wmma::N) == 0);
    const uint32_t blocks_m = core::ceil_div(m, BlockM);
    const uint32_t blocks_n = core::ceil_div(n, BlockN);
    if (m == 0 || n == 0 || k == 0) {
        throw std::invalid_argument("CUDA HGEMM dimensions must be positive");
    }
    if (m % BlockM != 0 || n % BlockN != 0) {
        throw std::invalid_argument("CUDA HGEMM M and N must be block-tile aligned");
    }
    if (k % (SplitK * BlockK) != 0 || k / SplitK < (Stages - 1) * BlockK) {
        throw std::invalid_argument("CUDA HGEMM K does not satisfy pipeline alignment");
    }
    if constexpr (SplitK > 1) {
        if (blocks_m * blocks_n > kSemaphoreCount) {
            throw std::invalid_argument("CUDA HGEMM split-K workspace is too small");
        }
        if (semaphore == nullptr || signal == nullptr) {
            throw std::invalid_argument("CUDA HGEMM split-K requires workspace");
        }
    }
    const dim3 grid(blocks_m * blocks_n, SplitK);
    hgemm_kernel<scalar_t, Wmma, BlockK, BlockMWarps, BlockNWarps, BlockKWarps, WarpMSteps, WarpNSteps, Stages, SplitK, SwizzleM>
        <<<grid, BlockThreads, 0, stream>>>(c, a, b, m, n, k, semaphore, signal, bias);
}

template <typename scalar_t>
void hgemm_gpu(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t m, uint32_t n, uint32_t k, uint32_t *semaphore,
               uint32_t *signal, const scalar_t *bias = nullptr, gpuStream_t stream = nullptr) {
    if (m <= 256) {
        launch_hgemm<scalar_t, 16, 64, 64, 1, 1, 2, 2, 4>(a, b, c, m, n, k, semaphore, signal, bias, stream);
    } else {
        launch_hgemm<scalar_t, 128, 128, 32, 2, 4, 1, 3, 1>(a, b, c, m, n, k, semaphore, signal, bias, stream);
    }
}

} // namespace peak_gemm::kernel
