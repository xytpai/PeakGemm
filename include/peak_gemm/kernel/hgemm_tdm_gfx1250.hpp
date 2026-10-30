#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "peak_gemm/backend/runtime.hpp"
#include "peak_gemm/backend/arch_gfx1250.hpp"
#include "peak_gemm/core/block_swizzle.hpp"
#include "peak_gemm/core/math.hpp"
#include "peak_gemm/core/swizzle.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::kernel {

template <typename scalar_t, typename wmma_t, typename swizzle_t, uint32_t BLOCK_K, uint32_t BLOCK_M_WARPS, uint32_t BLOCK_N_WARPS, uint32_t WARP_M_STEPS,
          uint32_t WARP_N_STEPS, bool HAS_BIAS>
class HgemmBlockTileTdm {
public:
    using FragmentAT = typename wmma_t::FragmentAT;
    using FragmentBT = typename wmma_t::FragmentBT;
    using FragmentCT = typename wmma_t::FragmentCT;
    static_assert(std::is_same_v<scalar_t, __half> || std::is_same_v<scalar_t, __bfloat16>, "ROCm HGEMM supports fp16 and bf16 inputs");

    enum : uint32_t {
        WARP_SIZE = backend::Warp::size,
        STAGES = 2,
        LDS_PAD = 8,
        LDS_STRIDE = BLOCK_K + LDS_PAD,
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
        STG_X_THREADS = BLOCK_N / VEC_SIZE,
        STG_ITERS = BLOCK_M * BLOCK_N / (BLOCK_THREADS * VEC_SIZE),
    };

    using vec_t = core::Vector<scalar_t, VEC_SIZE>;
    using SharedA = scalar_t[STAGES][BLOCK_M * LDS_STRIDE];
    using SharedB = scalar_t[STAGES][BLOCK_N * LDS_STRIDE];

    static_assert(WARP_SIZE == 32);
    static_assert(BLOCK_K >= 32 && BLOCK_K <= 512 && (BLOCK_K & (BLOCK_K - 1)) == 0,
                  "TDM row padding requires power-of-two BLOCK_K in [32, 512]");
    static_assert(BLOCK_M % (BLOCK_M_WARPS * BLOCK_N_WARPS) == 0);
    static_assert(BLOCK_N % (BLOCK_M_WARPS * BLOCK_N_WARPS) == 0);
    static_assert(BLOCK_M / (BLOCK_M_WARPS * BLOCK_N_WARPS) <= 65535);
    static_assert(BLOCK_N / (BLOCK_M_WARPS * BLOCK_N_WARPS) <= 65535);
    static_assert(STG_X_THREADS >= 1 && BLOCK_N % VEC_SIZE == 0);
    static_assert(BLOCK_THREADS % STG_X_THREADS == 0);
    static_assert(STG_ITERS >= 1 && BLOCK_M * BLOCK_N % (BLOCK_THREADS * VEC_SIZE) == 0);

    PEAKGEMM_DEVICE_INLINE explicit HgemmBlockTileTdm(
        uint32_t thread, uint32_t k_size, SharedA &shared_a, SharedB &shared_b,
        const swizzle_t &swizzle, scalar_t *c, const scalar_t *bias) :
        thread_(thread), warp_(thread >> WARP_SHIFT), lane_(thread & WARP_MASK),
        shared_a_(shared_a), shared_b_(shared_b), swizzle_(swizzle),
        k_size_(k_size), c_(c), bias_(bias) {
    }

    PEAKGEMM_DEVICE_INLINE void init() {
        wmma_.init(lane_);
#pragma unroll
        for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
#pragma unroll
            for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
                wmma_.reset_fragment_c(output_[m][n]);
            }
        }
    }

    PEAKGEMM_DEVICE_INLINE void copy_async(
        uint32_t stage, const scalar_t *a, uint32_t a_begin, const scalar_t *b, uint32_t b_begin) {
        constexpr uint32_t COPY_WAVES = BLOCK_M_WARPS * BLOCK_N_WARPS;
        // Every wave owns disjoint rows of both operands.
        const uint32_t wave = __builtin_amdgcn_readfirstlane(warp_);
        tdm_copy<BLOCK_N / COPY_WAVES>(
            b + b_begin + wave * (BLOCK_N / COPY_WAVES) * k_size_,
            shared_b_[stage] + wave * (BLOCK_N / COPY_WAVES) * LDS_STRIDE);
        tdm_copy<BLOCK_M / COPY_WAVES>(
            a + a_begin + wave * (BLOCK_M / COPY_WAVES) * k_size_,
            shared_a_[stage] + wave * (BLOCK_M / COPY_WAVES) * LDS_STRIDE);
    }

    template <uint32_t Rows>
    PEAKGEMM_DEVICE_INLINE void tdm_copy(const scalar_t *src, scalar_t *dst) {
        using Group0 = uint32_t __attribute__((ext_vector_type(4)));
        using Group1 = uint32_t __attribute__((ext_vector_type(8)));
        const uintptr_t address = reinterpret_cast<uintptr_t>(src);
        const uint32_t lds_address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst));
        const Group0 group0 = {
            1, lds_address, static_cast<uint32_t>(address),
            static_cast<uint32_t>(address >> 32) | 0x80000000U};
        // dim0 is contiguous K; stride is in elements. LDS padding is encoded
        // in dwords: interval = log2(BLOCK_K / 2) - 1, amount = LDS_PAD / 2 - 1.
        constexpr uint32_t interval = core::Log2<BLOCK_K>::value - 2;
        const Group1 group1 = {
            (1U << 16) | (1U << 20) | (interval << 22) | ((LDS_PAD / 2 - 1) << 25),
            BLOCK_K << 16, Rows << 16, BLOCK_K << 16, Rows, k_size_, 0, 0};
        __builtin_amdgcn_tensor_load_to_lds(group0, group1, Group0{}, Group0{}, Group1{}, 0);
    }

    PEAKGEMM_DEVICE_INLINE void wait() {
        // tensorcnt is per wave. Every producer must wait before the block
        // barrier lets any other wave consume (or recycle) the shared tile.
        asm volatile("s_wait_tensorcnt 0" ::: "memory");
        __syncthreads();
    }

    PEAKGEMM_DEVICE_INLINE static void schedule_barrier() {
        __builtin_amdgcn_sched_barrier(0);
    }

    PEAKGEMM_DEVICE_INLINE void compute(uint32_t stage) {
        const uint32_t warp_m = warp_ / BLOCK_N_WARPS * WARP_M;
        const uint32_t warp_n = warp_ % BLOCK_N_WARPS * WARP_N;
#pragma unroll
        for (uint32_t k = 0; k < WARP_K_STEPS; ++k) {
            const uint32_t column = k * WARP_ATOM_K;
            FragmentBT fragment_b[WARP_N_STEPS];
#pragma unroll
            for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
                wmma_.load_matrix_b(fragment_b[n], shared_b_[stage], warp_n + n * WARP_ATOM_N, column, LDS_STRIDE, swizzle_);
            }
            schedule_barrier();
#pragma unroll
            for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
                FragmentAT fragment_a;
                wmma_.load_matrix_a(fragment_a, shared_a_[stage], warp_m + m * WARP_ATOM_M, column, LDS_STRIDE, swizzle_);
                schedule_barrier();
#pragma unroll
                for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
                    wmma_(output_[m][n], fragment_a, fragment_b[n], output_[m][n]);
                }
            }
        }
    }

    PEAKGEMM_DEVICE_INLINE void store(scalar_t (&shared_c)[BLOCK_M * BLOCK_N], uint32_t block_m_idx, uint32_t block_n_idx, uint32_t m_size, uint32_t n_size) {
        const uint32_t warp_m = warp_ / BLOCK_N_WARPS * WARP_M;
        const uint32_t warp_n = warp_ % BLOCK_N_WARPS * WARP_N;

        __syncthreads();
#pragma unroll
        for (uint32_t m = 0; m < WARP_M_STEPS; ++m) {
#pragma unroll
            for (uint32_t n = 0; n < WARP_N_STEPS; ++n) {
                auto *destination =
                    &shared_c[(warp_m + m * WARP_ATOM_M) * BLOCK_N + warp_n + n * WARP_ATOM_N];
                wmma_.store_matrix(destination, BLOCK_N, output_[m][n]);
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
                if constexpr (HAS_BIAS) {
                    vec_t bias_value;
                    bias_value.load(bias_ + global_n);
                    value += bias_value;
                }
                auto *destination = c_ + global_m * n_size + global_n;
                value.store(destination);
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
    const swizzle_t &swizzle_;
    uint32_t k_size_;
    scalar_t *c_;
    const scalar_t *bias_;
    FragmentCT output_[WARP_M_STEPS][WARP_N_STEPS];
};

template <typename scalar_t, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K>
union alignas(16) HgemmSharedStorageTdm {
    struct {
        scalar_t a[2][BLOCK_M * (BLOCK_K + 8)];
        scalar_t b[2][BLOCK_N * (BLOCK_K + 8)];
    };
    scalar_t c[BLOCK_M * BLOCK_N];
};

template <typename scalar_t, typename wmma_t, typename swizzle_t, uint32_t BLOCK_K,
          uint32_t BLOCK_M_WARPS, uint32_t BLOCK_N_WARPS, uint32_t WARP_M_STEPS,
          uint32_t WARP_N_STEPS, uint32_t SWIZZLE_M, bool HAS_BIAS>
__attribute__((amdgpu_flat_work_group_size(BLOCK_M_WARPS * BLOCK_N_WARPS * 32, BLOCK_M_WARPS * BLOCK_N_WARPS * 32)))
__global__ void hgemm_tdm_kernel(
    scalar_t *c, const scalar_t *a, const scalar_t *b,
    uint32_t m_size, uint32_t n_size, uint32_t k_size, const scalar_t *bias) {
    using BlockTile = HgemmBlockTileTdm<scalar_t, wmma_t, swizzle_t, BLOCK_K,
                                      BLOCK_M_WARPS, BLOCK_N_WARPS, WARP_M_STEPS, WARP_N_STEPS, HAS_BIAS>;
    constexpr uint32_t BLOCK_M = BlockTile::BLOCK_M;
    constexpr uint32_t BLOCK_N = BlockTile::BLOCK_N;
    using SharedStorage = HgemmSharedStorageTdm<scalar_t, BLOCK_M, BLOCK_N, BLOCK_K>;
    static_assert(sizeof(SharedStorage) <= 320 * 1024, "gfx1250 TDM tile exceeds 320 KiB LDS");

    const uint32_t m_blocks = core::ceil_div(m_size, BLOCK_M);
    const uint32_t n_blocks = core::ceil_div(n_size, BLOCK_N);
    const auto block_idx = core::block_swizzle<SWIZZLE_M>(blockIdx.x, m_blocks, n_blocks);
    const uint32_t block_m_idx = block_idx.m;
    const uint32_t block_n_idx = block_idx.n;
    uint32_t a_begin = block_m_idx * BLOCK_M * k_size;
    uint32_t b_begin = block_n_idx * BLOCK_N * k_size;
    __shared__ SharedStorage shared;
    const swizzle_t swizzle;
    BlockTile block_tile(threadIdx.x, k_size, shared.a, shared.b, swizzle, c, bias);

    // Permit back-to-back WMMA issue (WAVE_SCHED_MODE.DISABLE_XDL_ARB_STALL).
    asm volatile("s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 2, 1), 1" ::: "memory");
    block_tile.init();
    block_tile.copy_async(0, a, a_begin, b, b_begin);
    BlockTile::schedule_barrier();

    // Expand one complete double-buffer ring so LDS stage offsets are constant.
    const uint32_t main_iterations = k_size / BLOCK_K - 1;
    uint32_t iteration = 0;
    for (; iteration + 2 <= main_iterations;
         iteration += 2, a_begin += 2 * BLOCK_K, b_begin += 2 * BLOCK_K) {
        block_tile.wait();
        block_tile.copy_async(1, a, a_begin + BLOCK_K, b, b_begin + BLOCK_K);
        BlockTile::schedule_barrier();
        block_tile.compute(0);
        block_tile.wait();
        block_tile.copy_async(0, a, a_begin + 2 * BLOCK_K, b, b_begin + 2 * BLOCK_K);
        BlockTile::schedule_barrier();
        block_tile.compute(1);
    }
    uint32_t current_stage = 0;
    for (; iteration < main_iterations; ++iteration, a_begin += BLOCK_K, b_begin += BLOCK_K) {
        block_tile.wait();
        block_tile.copy_async((current_stage + 1) % 2, a, a_begin + BLOCK_K, b, b_begin + BLOCK_K);
        BlockTile::schedule_barrier();
        block_tile.compute(current_stage);
        current_stage = (current_stage + 1) % 2;
    }
    block_tile.wait();
    block_tile.compute(current_stage);
    block_tile.store(shared.c, block_m_idx, block_n_idx, m_size, n_size);
}

// A[M,K], B[N,K], C[M,N], all contiguous. Full tiles only, two LDS stages,
// no split-K. Like hgemm_template, bias is added after rounding C to scalar_t.
template <typename scalar_t, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t BLOCK_M_WARPS, uint32_t BLOCK_N_WARPS, uint32_t SWIZZLE_M, bool HAS_BIAS>
void hgemm_tdm_template(
    const scalar_t *a, const scalar_t *b, scalar_t *c,
    uint32_t m, uint32_t n, uint32_t k, const scalar_t *bias = nullptr, gpuStream_t stream = nullptr) {
    using wmma_t = backend::WmmaDefault<scalar_t, float>;
    using swizzle_t = core::IdentitySwizzle;
    static_assert(BLOCK_M_WARPS > 0 && BLOCK_N_WARPS > 0);
    static_assert(wmma_t::M == 16 && wmma_t::N == 16 && wmma_t::K == 32);
    constexpr uint32_t WARP_M_STEPS = BLOCK_M / BLOCK_M_WARPS / wmma_t::M;
    constexpr uint32_t WARP_N_STEPS = BLOCK_N / BLOCK_N_WARPS / wmma_t::N;
    static_assert(WARP_M_STEPS >= 1 && WARP_M_STEPS <= 8);
    static_assert(WARP_N_STEPS >= 1 && WARP_N_STEPS <= 8);
    static_assert(BLOCK_M % (BLOCK_M_WARPS * wmma_t::M) == 0);
    static_assert(BLOCK_N % (BLOCK_N_WARPS * wmma_t::N) == 0);
    constexpr uint32_t BLOCK_THREADS = BLOCK_M_WARPS * BLOCK_N_WARPS * 32;
    static_assert(BLOCK_THREADS <= 1024);
    if (m == 0 || n == 0 || k == 0) {
        throw std::invalid_argument("TDM HGEMM dimensions must be positive");
    }
    constexpr uint64_t MAX_FLAT_INDEX = std::numeric_limits<uint32_t>::max() / sizeof(scalar_t);
    if (static_cast<uint64_t>(m) * k > MAX_FLAT_INDEX || static_cast<uint64_t>(n) * k > MAX_FLAT_INDEX
        || static_cast<uint64_t>(m) * n > MAX_FLAT_INDEX) {
        throw std::invalid_argument("TDM HGEMM flattened tensor byte offsets exceed uint32");
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        throw std::invalid_argument("TDM HGEMM tensor pointers must not be null");
    }
    if (reinterpret_cast<uintptr_t>(a) % 16U != 0U || reinterpret_cast<uintptr_t>(b) % 16U != 0U
        || reinterpret_cast<uintptr_t>(c) % 16U != 0U) {
        throw std::invalid_argument("TDM HGEMM tensors must be 16-byte aligned");
    }
    if constexpr (HAS_BIAS) {
        if (bias == nullptr || reinterpret_cast<uintptr_t>(bias) % 16U != 0U) {
            throw std::invalid_argument("TDM HGEMM bias must be present and 16-byte aligned");
        }
    }
    if (m % BLOCK_M != 0 || n % BLOCK_N != 0 || k % BLOCK_K != 0) {
        throw std::invalid_argument("TDM HGEMM dimensions must be block-tile aligned");
    }
    const uint32_t m_blocks = core::ceil_div(m, BLOCK_M);
    const uint32_t n_blocks = core::ceil_div(n, BLOCK_N);
    if (static_cast<uint64_t>(SWIZZLE_M) * n_blocks > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("TDM HGEMM block swizzle index exceeds uint32");
    }
    hgemm_tdm_kernel<scalar_t, wmma_t, swizzle_t, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS,
                     WARP_M_STEPS, WARP_N_STEPS, SWIZZLE_M, HAS_BIAS>
        <<<dim3(m_blocks * n_blocks), BLOCK_THREADS, 0, stream>>>(c, a, b, m, n, k, bias);
}

} // namespace peak_gemm::kernel
