#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "peak_gemm/backend/runtime.hpp"
#include "peak_gemm/core/block_swizzle.hpp"
#include "peak_gemm/core/math.hpp"
#include "peak_gemm/core/vector.hpp"

using Warp = peak_gemm::backend::Warp;
constexpr uint32_t WARP_SIZE = Warp::size;

namespace peak_gemm::kernel {

constexpr uint32_t kSemaphoreCount = 256;

template <
    typename scalar_t,
    typename wmma_t,
    uint32_t BLOCK_K,
    uint32_t BLOCK_M_WARPS,
    uint32_t BLOCK_N_WARPS,
    uint32_t WARP_M_STEPS,
    uint32_t WARP_N_STEPS,
    uint32_t STAGES,
    bool HAS_BIAS,
    bool IS_SPLIT_K>
class HgemmBlockTile {
public:
    using FragmentAT = typename wmma_t::FragmentAT;
    using FragmentBT = typename wmma_t::FragmentBT;
    using FragmentCT = typename wmma_t::FragmentCT;
    static_assert(std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __bfloat16>, "CUDA HGEMM supports fp16 and bf16 inputs");

    enum : uint32_t {
        WARP_MASK = WARP_SIZE - 1,
        WARP_SHIFT = core::Log2<WARP_SIZE>::value,
        WARP_ATOM_M = wmma_t::M,
        WARP_ATOM_N = wmma_t::N,
        WARP_ATOM_K = wmma_t::K,
        WARP_K_STEPS = BLOCK_K / WARP_ATOM_K,
        VEC_SIZE = 16 / sizeof(scalar_t),
        BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE,
        WARP_M = WARP_M_STEPS * WARP_ATOM_M,
        WARP_N = WARP_N_STEPS * WARP_ATOM_N,
        BLOCK_M = BLOCK_M_WARPS * WARP_M,
        BLOCK_N = BLOCK_N_WARPS * WARP_N,
        LDG_X_THREADS = BLOCK_K / VEC_SIZE,
        NUM_A_LOADS = BLOCK_M * BLOCK_K / BLOCK_THREADS / VEC_SIZE,
        NUM_B_LOADS = BLOCK_N * BLOCK_K / BLOCK_THREADS / VEC_SIZE,
        STG_X_THREADS = BLOCK_N / VEC_SIZE,
        STG_ITERS = BLOCK_M * BLOCK_N / (BLOCK_THREADS * VEC_SIZE),
    };

    using vec_t = core::Vector<scalar_t, VEC_SIZE>;
    using SharedA = scalar_t[STAGES][BLOCK_M * BLOCK_K];
    using SharedB = scalar_t[STAGES][BLOCK_N * BLOCK_K];

    static_assert(NUM_A_LOADS >= 1 && NUM_B_LOADS >= 1);
    static_assert(VEC_SIZE * BLOCK_THREADS * NUM_A_LOADS == BLOCK_M * BLOCK_K);
    static_assert(VEC_SIZE * BLOCK_THREADS * NUM_B_LOADS == BLOCK_N * BLOCK_K);
    static_assert(LDG_X_THREADS >= 1 && BLOCK_K % VEC_SIZE == 0);
    static_assert(BLOCK_THREADS % LDG_X_THREADS == 0);
    static_assert(WARP_K_STEPS >= 1 && BLOCK_K % WARP_ATOM_K == 0);
    static_assert(STG_X_THREADS >= 1 && BLOCK_N % VEC_SIZE == 0);
    static_assert(IS_SPLIT_K || BLOCK_THREADS % STG_X_THREADS == 0);
    static_assert(STG_ITERS >= 1 && BLOCK_M * BLOCK_N % (BLOCK_THREADS * VEC_SIZE) == 0);

    PEAKGEMM_DEVICE_INLINE explicit HgemmBlockTile(
        uint32_t thread, SharedA &shared_a, SharedB &shared_b, uint32_t *semaphore, uint32_t *signal, scalar_t *c, const scalar_t *bias) :
        thread_(thread), warp_(thread >> WARP_SHIFT), lane_(thread & WARP_MASK), shared_a_(shared_a), shared_b_(shared_b), semaphore_(semaphore), signal_(signal),
        c_(c), bias_(bias) {
    }

    PEAKGEMM_DEVICE_INLINE void init(uint32_t partition, uint32_t block_m_idx, uint32_t block_n_idx, uint32_t m_size, uint32_t n_size) {
        wmma_.init(lane_);
#pragma unroll
        for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
#pragma unroll
            for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
                wmma_.reset_fragment_c(output_[m][n]);
            }
        }
        if constexpr (IS_SPLIT_K) {
            if (partition == 0) {
                vec_t initial;
                initial.fill(static_cast<scalar_t>(0));
#pragma unroll
                for (uint32_t i = 0; i < STG_ITERS; ++i) {
                    const uint32_t global_thread = BLOCK_THREADS * i + thread_;
                    const uint32_t local_m = global_thread / STG_X_THREADS;
                    const uint32_t local_n = global_thread % STG_X_THREADS * VEC_SIZE;
                    const uint32_t global_m = block_m_idx * BLOCK_M + local_m;
                    const uint32_t global_n = block_n_idx * BLOCK_N + local_n;
                    if (global_m < m_size && global_n < n_size) {
                        if constexpr (HAS_BIAS) {
                            initial.load(bias_ + global_n);
                        }
                        initial.store(c_ + global_m * n_size + global_n);
                    }
                }
                __threadfence();
                __syncthreads();
                if (thread_ == 0) {
                    backend::atomic_exchange(&signal_[blockIdx.x], 1U);
                }
                __syncthreads();
            }
        }
    }

    PEAKGEMM_DEVICE_INLINE void copy_async(uint32_t stage, const scalar_t *a, uint32_t stride_a, const scalar_t *b, uint32_t stride_b) {
        const uint32_t x_vector = thread_ % LDG_X_THREADS;
#pragma unroll
        for (uint32_t i = 0; i < NUM_A_LOADS; ++i) {
            const uint32_t thread = BLOCK_THREADS * i + thread_;
            const uint32_t shared_offset = wmma_.swizzle(thread * VEC_SIZE);
            const auto *source = a + thread / LDG_X_THREADS * stride_a + x_vector * VEC_SIZE;
            backend::AsyncCopy::copy(reinterpret_cast<vec_t *>(shared_a_[stage] + shared_offset), reinterpret_cast<const vec_t *>(source));
        }
#pragma unroll
        for (uint32_t i = 0; i < NUM_B_LOADS; ++i) {
            const uint32_t thread = BLOCK_THREADS * i + thread_;
            const uint32_t shared_offset = wmma_.swizzle(thread * VEC_SIZE);
            const auto *source = b + thread / LDG_X_THREADS * stride_b + x_vector * VEC_SIZE;
            backend::AsyncCopy::copy(reinterpret_cast<vec_t *>(shared_b_[stage] + shared_offset), reinterpret_cast<const vec_t *>(source));
        }
    }

    template <int PENDING_GROUPS = 0>
    PEAKGEMM_DEVICE_INLINE void wait() {
        backend::AsyncCopy::template wait<PENDING_GROUPS>();
    }

    PEAKGEMM_DEVICE_INLINE void commit() {
        backend::AsyncCopy::commit();
    }

    PEAKGEMM_DEVICE_INLINE void compute(uint32_t stage) {
        const uint32_t warp_m = warp_ / BLOCK_N_WARPS * WARP_M;
        const uint32_t warp_n = warp_ % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (uint32_t k = 0; k < WARP_K_STEPS; ++k) {
            const uint32_t column = k * WARP_ATOM_K;
            FragmentAT fragment_a[WARP_M_STEPS];
            FragmentBT fragment_b[WARP_N_STEPS];
#pragma unroll
            for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
                wmma_.load_matrix_b(fragment_b[n], shared_b_[stage], warp_n + n * WARP_ATOM_N, column, BLOCK_K);
            }
#pragma unroll
            for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
                wmma_.load_matrix_a(fragment_a[m], shared_a_[stage], warp_m + m * WARP_ATOM_M, column, BLOCK_K);
            }
#pragma unroll
            for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
#pragma unroll
                for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
                    wmma_(output_[m][n], fragment_a[m], fragment_b[n], output_[m][n]);
                }
            }
        }
    }

    PEAKGEMM_DEVICE_INLINE void store(
        scalar_t (&shared_c)[BLOCK_M * BLOCK_N],
        uint32_t block_m_idx,
        uint32_t block_n_idx,
        uint32_t m_size,
        uint32_t n_size,
        uint32_t split_k) {
        const uint32_t warp_m = warp_ / BLOCK_N_WARPS * WARP_M;
        const uint32_t warp_n = warp_ % BLOCK_N_WARPS * WARP_N;
        __syncthreads();
#pragma unroll
        for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
#pragma unroll
            for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
                auto *destination = &shared_c[(warp_m + m * WARP_ATOM_M) * BLOCK_N + warp_n + n * WARP_ATOM_N];
                wmma_.store_matrix(destination, BLOCK_N, output_[m][n]);
            }
        }

        if constexpr (IS_SPLIT_K) {
            if (thread_ == 0) {
                while (backend::atomic_add(&signal_[blockIdx.x], 0U) == 0) {
                }
            }
            __syncthreads();
            if (thread_ == 0) {
                const uint32_t arrival =
                    backend::atomic_add(&semaphore_[blockIdx.x], 1U);
                if (arrival == split_k - 1) {
                    semaphore_[blockIdx.x] = 0;
                    signal_[blockIdx.x] = 0;
                }
            }
            __syncthreads();
#pragma unroll
            for (uint32_t i = 0; i < STG_ITERS; ++i) {
                const uint32_t global_thread = BLOCK_THREADS * i + thread_;
                const uint32_t local_m = global_thread / STG_X_THREADS;
                const uint32_t local_n = global_thread % STG_X_THREADS * VEC_SIZE;
                const uint32_t global_m = block_m_idx * BLOCK_M + local_m;
                const uint32_t global_n = block_n_idx * BLOCK_N + local_n;
                if (global_m < m_size && global_n < n_size) {
                    auto value = *reinterpret_cast<vec_t *>(&shared_c[local_m * BLOCK_N + local_n]);
                    auto *destination = c_ + global_m * n_size + global_n;
#pragma unroll
                    for (uint32_t element = 0; element < VEC_SIZE; element += 2) {
                        backend::atomic_pair_add(destination + element, &value[element]);
                    }
                }
            }
        } else {
            __syncthreads();
            const uint32_t local_n = thread_ % STG_X_THREADS * VEC_SIZE;
            const uint32_t global_n = block_n_idx * BLOCK_N + local_n;
            vec_t bias_value;
            if constexpr (HAS_BIAS) {
                bias_value.load(bias_ + global_n);
            }
#pragma unroll
            for (uint32_t i = 0; i < STG_ITERS; ++i) {
                const uint32_t global_thread = BLOCK_THREADS * i + thread_;
                const uint32_t local_m = global_thread / STG_X_THREADS;
                const uint32_t global_m = block_m_idx * BLOCK_M + local_m;
                if (global_m < m_size && global_n < n_size) {
                    auto value = *reinterpret_cast<vec_t *>(&shared_c[local_m * BLOCK_N + local_n]);
                    if constexpr (HAS_BIAS) {
                        value += bias_value;
                    }
                    auto *destination = c_ + global_m * n_size + global_n;
                    value.store(destination);
                }
            }
        }
    }

private:
    uint32_t thread_;
    uint32_t warp_;
    uint32_t lane_;
    wmma_t wmma_;
    SharedA &shared_a_;
    SharedB &shared_b_;
    uint32_t *semaphore_;
    uint32_t *signal_;
    scalar_t *c_;
    const scalar_t *bias_;
    FragmentCT output_[WARP_M_STEPS][WARP_N_STEPS];
};

template <typename scalar_t, uint32_t STAGES, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K>
union HgemmSharedStorage {
    struct {
        scalar_t a[STAGES][BLOCK_M * BLOCK_K];
        scalar_t b[STAGES][BLOCK_N * BLOCK_K];
    };
    scalar_t c[BLOCK_M * BLOCK_N];
};

template <
    typename scalar_t,
    typename wmma_t,
    uint32_t BLOCK_K,
    uint32_t BLOCK_M_WARPS,
    uint32_t BLOCK_N_WARPS,
    uint32_t WARP_M_STEPS,
    uint32_t WARP_N_STEPS,
    uint32_t STAGES,
    uint32_t SWIZZLE_M,
    bool HAS_BIAS,
    bool IS_SPLIT_K>
__global__ __launch_bounds__(BLOCK_M_WARPS *BLOCK_N_WARPS *Warp::size, 2) void hgemm_kernel(
    scalar_t *c, const scalar_t *a, const scalar_t *b, uint32_t m_size, uint32_t n_size, uint32_t k_size, uint32_t split_k,
    uint32_t *semaphore, uint32_t *signal, const scalar_t *bias) {
    using BlockTile =
        HgemmBlockTile<scalar_t, wmma_t, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_M_STEPS, WARP_N_STEPS, STAGES, HAS_BIAS, IS_SPLIT_K>;
    constexpr uint32_t BLOCK_M = BlockTile::BLOCK_M;
    constexpr uint32_t BLOCK_N = BlockTile::BLOCK_N;
    const uint32_t m_blocks = core::ceil_div(m_size, BLOCK_M);
    const uint32_t n_blocks = core::ceil_div(n_size, BLOCK_N);
    const auto block_idx = core::block_swizzle<SWIZZLE_M>(blockIdx.x, m_blocks, n_blocks);
    const uint32_t block_m_idx = block_idx.m;
    const uint32_t block_n_idx = block_idx.n;
    const uint32_t thread = threadIdx.x;
    const uint32_t partition_k = k_size / split_k;
    const uint32_t partition = blockIdx.y;
    const uint32_t begin_k = partition * partition_k;
    uint32_t a_begin = block_m_idx * BLOCK_M * k_size + begin_k;
    uint32_t b_begin = block_n_idx * BLOCK_N * k_size + begin_k;
    const uint32_t a_end = a_begin + partition_k;
    __shared__ HgemmSharedStorage<scalar_t, STAGES, BLOCK_M, BLOCK_N, BLOCK_K> shared;
    BlockTile block_tile(thread, shared.a, shared.b, semaphore, signal, c, bias);

    block_tile.init(partition, block_m_idx, block_n_idx, m_size, n_size);

#pragma unroll
    for (uint32_t stage = 0; stage < STAGES - 1; ++stage) {
        block_tile.copy_async(stage, a + a_begin + stage * BLOCK_K, k_size, b + b_begin + stage * BLOCK_K, k_size);
        block_tile.commit();
    }

    uint32_t current_stage = 0;
    for (; a_begin < a_end; a_begin += BLOCK_K, b_begin += BLOCK_K) {
        block_tile.template wait<STAGES - 2>();
        __syncthreads();
        block_tile.compute(current_stage);
        if (a_begin + (STAGES - 1) * BLOCK_K < a_end) {
            const uint32_t write_stage = (current_stage + STAGES - 1) % STAGES;
            block_tile.copy_async(write_stage, a + a_begin + (STAGES - 1) * BLOCK_K, k_size, b + b_begin + (STAGES - 1) * BLOCK_K, k_size);
        }
        block_tile.commit();
        current_stage = (current_stage + 1) % STAGES;
    }

    block_tile.store(shared.c, block_m_idx, block_n_idx, m_size, n_size, split_k);
}

template <
    typename scalar_t,
    uint32_t BLOCK_M,
    uint32_t BLOCK_N,
    uint32_t BLOCK_K,
    uint32_t BLOCK_M_WARPS,
    uint32_t BLOCK_N_WARPS,
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
    using wmma_t = backend::WmmaDefault<scalar_t, float, true>;
    constexpr uint32_t WARP_M_STEPS = BLOCK_M / BLOCK_M_WARPS / wmma_t::M;
    constexpr uint32_t WARP_N_STEPS = BLOCK_N / BLOCK_N_WARPS / wmma_t::N;
    static_assert(WARP_M_STEPS >= 1 && WARP_M_STEPS <= 4);
    static_assert(WARP_N_STEPS >= 1 && WARP_N_STEPS <= 4);
    static_assert(BLOCK_M % (BLOCK_M_WARPS * wmma_t::M) == 0);
    static_assert(BLOCK_N % (BLOCK_N_WARPS * wmma_t::N) == 0);
    constexpr uint32_t BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * Warp::size;
    if (m == 0 || n == 0 || k == 0) {
        throw std::invalid_argument("CUDA HGEMM dimensions must be positive");
    }
    constexpr uint64_t MAX_FLAT_INDEX = std::numeric_limits<uint32_t>::max();
    if (static_cast<uint64_t>(m) * k > MAX_FLAT_INDEX || static_cast<uint64_t>(n) * k > MAX_FLAT_INDEX || static_cast<uint64_t>(m) * n > MAX_FLAT_INDEX) {
        throw std::invalid_argument("CUDA HGEMM flattened tensor indices exceed uint32");
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("CUDA HGEMM tensor pointers must not be null");
    }
    if (reinterpret_cast<uintptr_t>(a) % 16U != 0U || reinterpret_cast<uintptr_t>(b) % 16U != 0U || reinterpret_cast<uintptr_t>(c) % 16U != 0U) {
        throw std::invalid_argument("CUDA HGEMM tensors must be 16-byte aligned");
    }
    if (split_k == 0 || split_k > 65535U) {
        throw std::invalid_argument("CUDA HGEMM split_k exceeds the CUDA grid Y limit");
    }
    if constexpr (IS_SPLIT_K) {
        if (split_k <= 1) {
            throw std::invalid_argument("CUDA HGEMM split-K specialization requires split_k > 1");
        }
    } else if (split_k != 1) {
        throw std::invalid_argument("CUDA HGEMM non-split specialization requires split_k == 1");
    }
    if constexpr (HAS_BIAS) {
        if (bias == nullptr) {
            throw std::invalid_argument("CUDA HGEMM bias specialization requires bias");
        }
        if (reinterpret_cast<uintptr_t>(bias) % 16U != 0U) {
            throw std::invalid_argument("CUDA HGEMM bias must be 16-byte aligned");
        }
    }
    if (m % BLOCK_M != 0 || n % BLOCK_N != 0) {
        throw std::invalid_argument("CUDA HGEMM M and N must be block-tile aligned");
    }
    const uint64_t k_alignment = static_cast<uint64_t>(split_k) * BLOCK_K;
    if (k % k_alignment != 0 || k / split_k < (STAGES - 1) * BLOCK_K) {
        throw std::invalid_argument("CUDA HGEMM K does not satisfy pipeline alignment");
    }
    const uint32_t m_blocks = core::ceil_div(m, BLOCK_M);
    const uint32_t n_blocks = core::ceil_div(n, BLOCK_N);
    if (static_cast<uint64_t>(SWIZZLE_M) * n_blocks > MAX_FLAT_INDEX) {
        throw std::invalid_argument("CUDA HGEMM block swizzle index exceeds uint32");
    }
    if constexpr (IS_SPLIT_K) {
        if (static_cast<uint64_t>(m_blocks) * n_blocks > kSemaphoreCount) {
            throw std::invalid_argument("CUDA HGEMM split-K workspace is too small");
        }
        if (semaphore == nullptr || signal == nullptr) {
            throw std::invalid_argument("CUDA HGEMM split-K requires workspace");
        }
    }
    const dim3 grid(m_blocks * n_blocks, split_k);
    hgemm_kernel<scalar_t, wmma_t, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_M_STEPS, WARP_N_STEPS, STAGES, SWIZZLE_M, HAS_BIAS, IS_SPLIT_K>
        <<<grid, BLOCK_THREADS, 0, stream>>>(c, a, b, m, n, k, split_k, semaphore, signal, bias);
}

} // namespace peak_gemm::kernel
