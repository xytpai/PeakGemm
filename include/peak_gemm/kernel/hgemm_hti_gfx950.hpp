#pragma once

#include "peak_gemm/backend/arch_gfx950.hpp"
#include "peak_gemm/kernel/hgemm_gfx950.hpp"

namespace peak_gemm::kernel {

PEAKGEMM_DEVICE_INLINE void s_barrier() {
    asm volatile("s_barrier");
}

template <uint32_t CNT>
PEAKGEMM_DEVICE_INLINE void wait_barrier() {
    asm volatile("s_waitcnt vmcnt(%0)\n\ts_barrier" ::"n"(CNT));
}

template <typename scalar_t, typename wmma_t, typename swizzle_t, uint32_t BLOCK_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_M_WARPS,
          uint32_t BLOCK_N_WARPS, bool HAS_BIAS>
class HgemmBlockTileHti {
public:
    using FragmentAT = typename wmma_t::FragmentAT;
    using FragmentBT = typename wmma_t::FragmentBT;
    using FragmentCT = typename wmma_t::FragmentCT;

    enum : uint32_t {
        WARP_MASK = WARP_SIZE - 1,
        WARP_SHIFT = core::Log2<WARP_SIZE>::value,
        HALF_BLOCK_M = BLOCK_M / 2,
        HALF_BLOCK_N = BLOCK_N / 2,
        WARP_ATOM_M = wmma_t::M,
        WARP_ATOM_N = wmma_t::N,
        WARP_ATOM_K = wmma_t::K,
        WARP_M_STEPS = HALF_BLOCK_M / BLOCK_M_WARPS / WARP_ATOM_M,
        WARP_N_STEPS = HALF_BLOCK_N / BLOCK_N_WARPS / WARP_ATOM_N,
        WARP_K_STEPS = BLOCK_K / WARP_ATOM_K,
        WARP_M = WARP_M_STEPS * WARP_ATOM_M,
        WARP_N = WARP_N_STEPS * WARP_ATOM_N,
        VEC_SIZE = 16 / sizeof(scalar_t),
        LDG_X_THREADS = BLOCK_K / VEC_SIZE,
        BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE,
        NUM_A_LOADS = HALF_BLOCK_M * BLOCK_K / BLOCK_THREADS / VEC_SIZE,
        NUM_B_LOADS = HALF_BLOCK_N * BLOCK_K / BLOCK_THREADS / VEC_SIZE,
        DMA_BYTES = 16,
        BLOCK_DMA_STRIDE = BLOCK_THREADS * DMA_BYTES,
        STG_X_THREADS = HALF_BLOCK_N / VEC_SIZE,
        STG_ITERS = HALF_BLOCK_M * HALF_BLOCK_N / (BLOCK_THREADS * VEC_SIZE),
    };

    using vec_t = core::Vector<scalar_t, VEC_SIZE>;

    static_assert(std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __bfloat16>, "ROCm HTI HGEMM supports fp16 and bf16 inputs");
    static_assert(BLOCK_M % 2 == 0 && BLOCK_N % 2 == 0);
    static_assert(BLOCK_M_WARPS == 2 && BLOCK_N_WARPS == 4, "ROCm HTI schedule requires a 2x4 wave layout");
    static_assert(BLOCK_THREADS == 512);
    static_assert(WARP_M_STEPS >= 1 && HALF_BLOCK_M % (BLOCK_M_WARPS * WARP_ATOM_M) == 0);
    static_assert(WARP_N_STEPS >= 1 && HALF_BLOCK_N % (BLOCK_N_WARPS * WARP_ATOM_N) == 0);
    static_assert(WARP_K_STEPS == 2 && BLOCK_K % WARP_ATOM_K == 0, "ROCm HTI schedule requires two MFMA K steps");
    static_assert(NUM_A_LOADS == 2 && NUM_B_LOADS == 2, "ROCm HTI schedule requires two A and B load instructions");
    static_assert(VEC_SIZE * BLOCK_THREADS * NUM_A_LOADS == HALF_BLOCK_M * BLOCK_K);
    static_assert(VEC_SIZE * BLOCK_THREADS * NUM_B_LOADS == HALF_BLOCK_N * BLOCK_K);
    static_assert(LDG_X_THREADS >= 1 && BLOCK_K % VEC_SIZE == 0);
    static_assert(BLOCK_THREADS % LDG_X_THREADS == 0);
    static_assert(STG_X_THREADS >= 1 && HALF_BLOCK_N % VEC_SIZE == 0);
    static_assert(BLOCK_THREADS % STG_X_THREADS == 0);
    static_assert(STG_ITERS >= 1 && HALF_BLOCK_M * HALF_BLOCK_N % (BLOCK_THREADS * VEC_SIZE) == 0);

    PEAKGEMM_DEVICE_INLINE HgemmBlockTileHti(
        uint32_t thread, scalar_t *shared_a, scalar_t *shared_b) :
        thread_(thread), warp_(thread >> WARP_SHIFT) {
        const wmma_t wmma;
#pragma unroll
        for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
#pragma unroll
            for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
                wmma.reset_fragment_c(output_[0][0][m][n]);
                wmma.reset_fragment_c(output_[0][1][m][n]);
                wmma.reset_fragment_c(output_[1][0][m][n]);
                wmma.reset_fragment_c(output_[1][1][m][n]);
            }
        }

        const uint32_t shared_a_address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(shared_a));
        const uint32_t shared_b_address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(shared_b));
        shared_a_address_ = __builtin_amdgcn_readfirstlane(shared_a_address + warp_ * WARP_SIZE * DMA_BYTES);
        shared_b_address_ = __builtin_amdgcn_readfirstlane(shared_b_address + warp_ * WARP_SIZE * DMA_BYTES);
    }

    PEAKGEMM_DEVICE_INLINE void copy_async_a(
        uint32_t half_m, uint32_t k_buffer, BufferResource &resource, uint32_t source_begin, uint32_t k_size, const swizzle_t &swizzle) {
        const uint32_t shared_offset = (k_buffer * 2 + half_m) * HALF_BLOCK_M * BLOCK_K;
        const uint32_t shared_wave = shared_a_address_ + shared_offset * sizeof(scalar_t);
#pragma unroll
        for (uint32_t i = 0; i < NUM_A_LOADS; ++i) {
            const uint32_t global_thread = BLOCK_THREADS * i + thread_;
            const uint32_t swizzled_offset = swizzle(global_thread * VEC_SIZE);
            const uint32_t source_offset = swizzled_offset / BLOCK_K * k_size + swizzled_offset % BLOCK_K;
            raw_buffer_load_lds(resource, (SharedPointer) static_cast<uintptr_t>(shared_wave + i * BLOCK_DMA_STRIDE), DMA_BYTES,
                                (source_begin + source_offset) * sizeof(scalar_t), 0, 0, 0);
        }
    }

    PEAKGEMM_DEVICE_INLINE void copy_async_b(
        uint32_t half_n, uint32_t k_buffer, BufferResource &resource, uint32_t source_begin, uint32_t k_size, const swizzle_t &swizzle) {
        const uint32_t shared_offset = (k_buffer * 2 + half_n) * HALF_BLOCK_N * BLOCK_K;
        const uint32_t shared_wave = shared_b_address_ + shared_offset * sizeof(scalar_t);
#pragma unroll
        for (uint32_t i = 0; i < NUM_B_LOADS; ++i) {
            const uint32_t global_thread = BLOCK_THREADS * i + thread_;
            const uint32_t swizzled_offset = swizzle(global_thread * VEC_SIZE);
            const uint32_t source_offset = swizzled_offset / BLOCK_K * k_size + swizzled_offset % BLOCK_K;
            raw_buffer_load_lds(resource, (SharedPointer) static_cast<uintptr_t>(shared_wave + i * BLOCK_DMA_STRIDE), DMA_BYTES,
                                (source_begin + source_offset) * sizeof(scalar_t), 0, 0, 0);
        }
    }

    PEAKGEMM_DEVICE_INLINE void load_matrix_a(scalar_t *shared, const swizzle_t &swizzle) {
        wmma_t wmma;
        wmma.init(thread_ & WARP_MASK);
        const uint32_t warp_m = warp_ / BLOCK_N_WARPS * WARP_M;
#pragma unroll
        for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
#pragma unroll
            for (uint32_t k = 0; k < WARP_K_STEPS; ++k) {
                wmma.load_matrix_a(fragment_a_[k][m], shared, warp_m + m * WARP_ATOM_M, k * WARP_ATOM_K, BLOCK_K, swizzle);
            }
        }
    }

    template <uint32_t Buffer>
    PEAKGEMM_DEVICE_INLINE void load_matrix_b(scalar_t *shared, const swizzle_t &swizzle) {
        wmma_t wmma;
        wmma.init(thread_ & WARP_MASK);
        const uint32_t warp_n = warp_ % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
#pragma unroll
            for (uint32_t k = 0; k < WARP_K_STEPS; ++k) {
                wmma.load_matrix_b(fragment_b_[Buffer][k][n], shared, warp_n + n * WARP_ATOM_N, k * WARP_ATOM_K, BLOCK_K, swizzle);
            }
        }
    }

    template <uint32_t HalfM, uint32_t HalfN>
    PEAKGEMM_DEVICE_INLINE void consume() {
        const wmma_t wmma;
#pragma unroll
        for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
#pragma unroll
            for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
#pragma unroll
                for (uint32_t k = 0; k < WARP_K_STEPS; ++k) {
                    wmma(output_[HalfM][HalfN][m][n], fragment_a_[k][m], fragment_b_[HalfN][k][n], output_[HalfM][HalfN][m][n]);
                }
            }
        }
    }

    template <uint32_t HalfM, uint32_t HalfN>
    PEAKGEMM_DEVICE_INLINE void store_matrix_to_shared(scalar_t (&shared_c)[BLOCK_M][BLOCK_N]) {
        wmma_t wmma;
        wmma.init(thread_ & WARP_MASK);
        const uint32_t warp_m = warp_ / BLOCK_N_WARPS * WARP_M;
        const uint32_t warp_n = warp_ % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
#pragma unroll
            for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
                auto *destination =
                    &shared_c[HalfM * HALF_BLOCK_M + warp_m + m * WARP_ATOM_M][HalfN * HALF_BLOCK_N + warp_n + n * WARP_ATOM_N];
                wmma.store_matrix(destination, BLOCK_N, output_[HalfM][HalfN][m][n]);
            }
        }
    }

    template <uint32_t HalfM, uint32_t HalfN>
    PEAKGEMM_DEVICE_INLINE void store_matrix_from_shared(
        scalar_t *c, scalar_t (&shared_c)[BLOCK_M][BLOCK_N], const scalar_t *bias, uint32_t block_m_idx, uint32_t block_n_idx, uint32_t m_size,
        uint32_t n_size) {
#pragma unroll
        for (uint32_t i = 0; i < STG_ITERS; ++i) {
            const uint32_t global_thread = BLOCK_THREADS * i + thread_;
            const uint32_t local_m = HalfM * HALF_BLOCK_M + global_thread / STG_X_THREADS;
            const uint32_t local_n = HalfN * HALF_BLOCK_N + global_thread % STG_X_THREADS * VEC_SIZE;
            const uint32_t global_m = block_m_idx * BLOCK_M + local_m;
            const uint32_t global_n = block_n_idx * BLOCK_N + local_n;
            if (global_m < m_size && global_n < n_size) {
                vec_t value;
                value.load(&shared_c[local_m][local_n]);
                if constexpr (HAS_BIAS) {
                    vec_t bias_value;
                    bias_value.load(bias + global_n);
                    value += bias_value;
                }
                value.store(c + global_m * n_size + global_n);
            }
        }
    }

private:
    uint32_t thread_;
    uint32_t warp_;
    uint32_t shared_a_address_;
    uint32_t shared_b_address_;
    FragmentAT fragment_a_[WARP_K_STEPS][WARP_M_STEPS];
    FragmentBT fragment_b_[2][WARP_K_STEPS][WARP_N_STEPS];
    FragmentCT output_[2][2][WARP_M_STEPS][WARP_N_STEPS];
};

template <typename scalar_t, uint32_t HALF_BLOCK_M, uint32_t HALF_BLOCK_N, uint32_t BLOCK_K>
union alignas(16) HgemmSharedStorageHti {
    struct {
        scalar_t a[2][2][HALF_BLOCK_M][BLOCK_K];
        scalar_t b[2][2][HALF_BLOCK_N][BLOCK_K];
    };
    scalar_t c[2 * HALF_BLOCK_M][2 * HALF_BLOCK_N];
};

template <typename scalar_t, typename wmma_t, typename swizzle_t, uint32_t BLOCK_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_M_WARPS,
          uint32_t BLOCK_N_WARPS, uint32_t SWIZZLE_M, bool HAS_BIAS>
__attribute__((amdgpu_waves_per_eu(2, 2),
               amdgpu_flat_work_group_size(BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE, BLOCK_M_WARPS * BLOCK_N_WARPS * WARP_SIZE))) __global__ void
hgemm_hti_kernel(
    scalar_t *c, const scalar_t *a, const scalar_t *b, uint32_t m_size, uint32_t n_size, uint32_t k_size, const scalar_t *bias) {
    using BlockTile = HgemmBlockTileHti<scalar_t, wmma_t, swizzle_t, BLOCK_K, BLOCK_M, BLOCK_N, BLOCK_M_WARPS, BLOCK_N_WARPS, HAS_BIAS>;
    constexpr uint32_t HALF_BLOCK_M = BlockTile::HALF_BLOCK_M;
    constexpr uint32_t HALF_BLOCK_N = BlockTile::HALF_BLOCK_N;
    constexpr uint32_t BLOCK_THREADS = BlockTile::BLOCK_THREADS;
    const uint32_t m_blocks = core::ceil_div(m_size, BLOCK_M);
    const uint32_t n_blocks = core::ceil_div(n_size, BLOCK_N);
    const auto block_idx = core::block_swizzle<SWIZZLE_M>(blockIdx.x, m_blocks, n_blocks);
    const uint32_t block_m_idx = block_idx.m;
    const uint32_t block_n_idx = block_idx.n;
    const uint32_t thread = threadIdx.x;
    const uint32_t warp = thread / WARP_SIZE;
    uint32_t a_begin = block_m_idx * BLOCK_M * k_size;
    uint32_t b_begin = block_n_idx * BLOCK_N * k_size;
    const uint32_t a_end = a_begin + k_size;
    __shared__ HgemmSharedStorageHti<scalar_t, HALF_BLOCK_M, HALF_BLOCK_N, BLOCK_K> shared;
    const swizzle_t swizzle;
    BlockTile block_tile(thread, &shared.a[0][0][0][0], &shared.b[0][0][0][0]);
    auto a_resource = make_buffer_resource(a);
    auto b_resource = make_buffer_resource(b);

#define HTI_LDG_A(M, K, FETCH) \
    block_tile.copy_async_a(M, K, a_resource, a_begin + (M) * HALF_BLOCK_M * k_size + ((FETCH) + (K)) * BLOCK_K, k_size, swizzle)
#define HTI_LDG_B(N, K, FETCH) \
    block_tile.copy_async_b(N, K, b_resource, b_begin + (N) * HALF_BLOCK_N * k_size + ((FETCH) + (K)) * BLOCK_K, k_size, swizzle)
#define HTI_LDMAT_A(M, K) block_tile.load_matrix_a(&shared.a[K][M][0][0], swizzle)
#define HTI_LDMAT_B(N, K) block_tile.template load_matrix_b<N>(&shared.b[K][N][0][0], swizzle)
#define HTI_CONSUME(M, N, SCHEDULE)                 \
    {                                               \
        block_tile.template consume<M, N>();        \
        s_barrier();                                \
        if constexpr (SCHEDULE) schedule_barrier(); \
    }

    HTI_LDG_B(0, 0, 0);
    HTI_LDG_A(0, 0, 0);
    HTI_LDG_B(1, 0, 0);
    HTI_LDG_A(1, 0, 0);

    schedule_barrier();
    if (warp / BLOCK_N_WARPS == 1) s_barrier();
    schedule_barrier();
    s_barrier();
    schedule_barrier();

    HTI_LDG_B(0, 1, 0);
    HTI_LDG_A(0, 1, 0);
    HTI_LDG_B(1, 1, 0);
    schedule_barrier();
    wait_barrier<BlockTile::NUM_B_LOADS + BlockTile::NUM_A_LOADS>();

    for (; a_begin < a_end - 2 * BLOCK_K; a_begin += 2 * BLOCK_K, b_begin += 2 * BLOCK_K) {
        // 0
        HTI_LDMAT_B(0, 0);
        HTI_LDMAT_A(0, 0);
        HTI_LDG_A(1, 1, 0);
        s_barrier();
        HTI_CONSUME(0, 0, true);
        HTI_LDMAT_B(1, 0);
        HTI_LDG_B(0, 0, 2);
        s_barrier();
        HTI_CONSUME(0, 1, false);
        HTI_LDMAT_A(1, 0);
        HTI_LDG_A(0, 0, 2);
        s_barrier();
        HTI_CONSUME(1, 0, true);
        HTI_LDMAT_B(0, 1);
        HTI_LDG_B(1, 0, 2);
        wait_barrier<2 * BlockTile::NUM_B_LOADS + BlockTile::NUM_A_LOADS>();
        HTI_CONSUME(1, 1, false);
        // 1
        HTI_LDMAT_A(0, 1);
        HTI_LDG_A(1, 0, 2);
        s_barrier();
        HTI_CONSUME(0, 0, true);
        HTI_LDMAT_B(1, 1);
        HTI_LDG_B(0, 1, 2);
        s_barrier();
        HTI_CONSUME(0, 1, false);
        HTI_LDMAT_A(1, 1);
        HTI_LDG_A(0, 1, 2);
        s_barrier();
        HTI_CONSUME(1, 0, true);
        HTI_LDG_B(1, 1, 2);
        wait_barrier<BlockTile::NUM_B_LOADS + BlockTile::NUM_A_LOADS>();
        HTI_CONSUME(1, 1, false);
    }
    // 0
    HTI_LDMAT_B(0, 0);
    HTI_LDMAT_A(0, 0);
    HTI_LDG_A(1, 1, 0);
    s_barrier();
    HTI_CONSUME(0, 0, true);
    HTI_LDMAT_B(1, 0);
    s_barrier();
    HTI_CONSUME(0, 1, false);
    HTI_LDMAT_A(1, 0);
    s_barrier();
    HTI_CONSUME(1, 0, true);
    HTI_LDMAT_B(0, 1);
    s_barrier();
    HTI_CONSUME(1, 1, false);
    // 1
    wait_barrier<0>();
    HTI_LDMAT_A(0, 1);
    s_barrier();
    HTI_CONSUME(0, 0, false);
    HTI_LDMAT_B(1, 1);
    s_barrier();
    HTI_CONSUME(0, 1, false);
    HTI_LDMAT_A(1, 1);
    s_barrier();
    block_tile.template consume<1, 0>();
    wait_barrier<0>();
    block_tile.template store_matrix_to_shared<0, 0>(shared.c);
    block_tile.template store_matrix_to_shared<0, 1>(shared.c);
    s_barrier();
    block_tile.template consume<1, 1>();
    wait_barrier<0>();
    block_tile.template store_matrix_from_shared<0, 0>(c, shared.c, bias, block_m_idx, block_n_idx, m_size, n_size);
    block_tile.template store_matrix_from_shared<0, 1>(c, shared.c, bias, block_m_idx, block_n_idx, m_size, n_size);
    wait_barrier<0>();
    block_tile.template store_matrix_to_shared<1, 0>(shared.c);
    block_tile.template store_matrix_to_shared<1, 1>(shared.c);
    wait_barrier<0>();
    block_tile.template store_matrix_from_shared<1, 0>(c, shared.c, bias, block_m_idx, block_n_idx, m_size, n_size);
    block_tile.template store_matrix_from_shared<1, 1>(c, shared.c, bias, block_m_idx, block_n_idx, m_size, n_size);

#undef HTI_LDG_A
#undef HTI_LDG_B
#undef HTI_LDMAT_A
#undef HTI_LDMAT_B
#undef HTI_CONSUME
}

template <typename scalar_t, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K, uint32_t BLOCK_M_WARPS, uint32_t BLOCK_N_WARPS,
          uint32_t SWIZZLE_M, bool HAS_BIAS>
void hgemm_hti_template(
    const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t m, uint32_t n, uint32_t k, const scalar_t *bias = nullptr, gpuStream_t stream = nullptr) {
    using wmma_t = backend::WmmaDefault<scalar_t, float, true>;
    using swizzle_t = backend::Gfx950Swizzle<3, 3, 3>;
    using BlockTile = HgemmBlockTileHti<scalar_t, wmma_t, swizzle_t, BLOCK_K, BLOCK_M, BLOCK_N, BLOCK_M_WARPS, BLOCK_N_WARPS, HAS_BIAS>;
    constexpr uint32_t BLOCK_THREADS = BlockTile::BLOCK_THREADS;
    static_assert(wmma_t::M == 16 && wmma_t::N == 16 && wmma_t::K == 32);
    if (m == 0 || n == 0 || k == 0) {
        throw std::invalid_argument("ROCm HTI HGEMM dimensions must be positive");
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("ROCm HTI HGEMM tensor pointers must not be null");
    }
    if (reinterpret_cast<uintptr_t>(a) % 16U != 0U || reinterpret_cast<uintptr_t>(b) % 16U != 0U || reinterpret_cast<uintptr_t>(c) % 16U != 0U) {
        throw std::invalid_argument("ROCm HTI HGEMM tensors must be 16-byte aligned");
    }
    if constexpr (HAS_BIAS) {
        if (bias == nullptr || reinterpret_cast<uintptr_t>(bias) % 16U != 0U) {
            throw std::invalid_argument("ROCm HTI HGEMM bias must be present and 16-byte aligned");
        }
    }
    if (m % BLOCK_M != 0 || n % BLOCK_N != 0) {
        throw std::invalid_argument("ROCm HTI HGEMM M and N must be block-tile aligned");
    }
    if (k < 2 * BLOCK_K || k % (2 * BLOCK_K) != 0) {
        throw std::invalid_argument("ROCm HTI HGEMM K must be aligned to two K tiles");
    }
    constexpr uint64_t MAX_FLAT_INDEX = std::numeric_limits<uint32_t>::max() / sizeof(scalar_t);
    if (static_cast<uint64_t>(m) * k > MAX_FLAT_INDEX || static_cast<uint64_t>(n) * k > MAX_FLAT_INDEX
        || static_cast<uint64_t>(m) * n > MAX_FLAT_INDEX) {
        throw std::invalid_argument("ROCm HTI HGEMM flattened tensor byte offsets exceed uint32");
    }
    const uint32_t m_blocks = core::ceil_div(m, BLOCK_M);
    const uint32_t n_blocks = core::ceil_div(n, BLOCK_N);
    if (static_cast<uint64_t>(m_blocks) * n_blocks > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("ROCm HTI HGEMM grid size exceeds uint32");
    }
    const dim3 grid(m_blocks * n_blocks);
    hgemm_hti_kernel<scalar_t, wmma_t, swizzle_t, BLOCK_K, BLOCK_M, BLOCK_N, BLOCK_M_WARPS, BLOCK_N_WARPS, SWIZZLE_M, HAS_BIAS>
        <<<grid, BLOCK_THREADS, 0, stream>>>(c, a, b, m, n, k, bias);
}

} // namespace peak_gemm::kernel
