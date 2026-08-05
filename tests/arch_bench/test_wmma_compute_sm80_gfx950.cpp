#include "peak_gemm/peak_gemm.hpp"

#if defined(__CUDACC__)
#include "peak_gemm/backend/arch_sm80.hpp"
#elif defined(__HIPCC__)
#include "peak_gemm/backend/arch_gfx950.hpp"
#endif

#include <climits>
#include <cmath>
#include <cstdlib>
#include <typeinfo>

template <typename scalar_t, typename WMMAT, int BLOCK_WARPS, int WARP_SIZE, int LOOP>
__global__ void wmma_loop_kernel(float *c, scalar_t *a, scalar_t *b) {
    constexpr int WM = WMMAT::M;
    constexpr int WN = WMMAT::N;
    constexpr int WK = WMMAT::K;
    int wid = threadIdx.x / WARP_SIZE;
    int w_tid = threadIdx.x % WARP_SIZE;
    int wmma_batch_id = blockIdx.x * BLOCK_WARPS + wid;
    auto c_ = c + wmma_batch_id * WM * WN;
    auto a_ = a + wmma_batch_id * WM * WK;
    auto b_ = b + wmma_batch_id * WN * WK;
    __shared__ scalar_t cs[BLOCK_WARPS * WM * WN];
    __shared__ scalar_t as[BLOCK_WARPS * WM * WK];
    __shared__ scalar_t bs[BLOCK_WARPS * WN * WK];
    auto cs_ = cs + wid * WM * WN;
    auto as_ = as + wid * WM * WK;
    auto bs_ = bs + wid * WN * WK;
    if (threadIdx.x % WARP_SIZE == 0) {
        for (int i = 0; i < WM * WK; ++i) {
            as_[i] = a_[i];
        }
        for (int i = 0; i < WN * WK; ++i) {
            bs_[i] = b_[i];
        }
    }
    __syncthreads();
    typename WMMAT::FragmentCT c_frag;
    typename WMMAT::FragmentAT a_frag;
    typename WMMAT::FragmentBT b_frag;
    WMMAT wmma;
    wmma.init(w_tid);
    wmma.reset_fragment_c(c_frag);
#if defined(__CUDACC__)
    using SwizzleT = peak_gemm::backend::IdentitySwizzle;
#else
    using SwizzleT = peak_gemm::backend::Gfx950IdentitySwizzle;
#endif
    const SwizzleT swizzle;
    wmma.load_matrix_a(a_frag, as_, 0, 0, WK, swizzle);
    wmma.load_matrix_b(b_frag, bs_, 0, 0, WK, swizzle);
    for (int i = 0; i < LOOP; ++i) {
        wmma(c_frag, a_frag, b_frag, c_frag);
    }
    wmma.store_matrix(cs_, WN, c_frag);
    __syncthreads();
    if (threadIdx.x % WARP_SIZE == 0) {
        for (int i = 0; i < WM * WN; ++i) {
            c_[i] = cs_[i];
        }
    }
}

template <typename scalar_t, typename WMMAT, int BLOCK_WARPS, int WARP_SIZE, int LOOP, int NBLOCKS, bool VALID>
float wmma_test() {
    constexpr int BLOCK_SIZE = BLOCK_WARPS * WARP_SIZE;
    dim3 threads_per_block(BLOCK_SIZE);
    dim3 num_blocks(NBLOCKS);
    constexpr int WM = WMMAT::M;
    constexpr int WN = WMMAT::N;
    constexpr int WK = WMMAT::K;
    constexpr int BATCH_SIZE = NBLOCKS * BLOCK_WARPS;
    constexpr int LEN_C = BATCH_SIZE * WM * WN;
    constexpr int LEN_A = BATCH_SIZE * WM * WK;
    constexpr int LEN_B = BATCH_SIZE * WN * WK;
    auto cpu_a = new scalar_t[LEN_A];
    auto cpu_b = new scalar_t[LEN_B];
    auto cpu_c = new float[LEN_C];
    auto ref_c = new float[LEN_C];
    for (int i = 0; i < LEN_A; ++i) {
        cpu_a[i] = static_cast<scalar_t>(
            2.0F * (rand() / static_cast<float>(INT_MAX) - 0.5F));
    }
    for (int i = 0; i < LEN_B; ++i) {
        cpu_b[i] = static_cast<scalar_t>(
            2.0F * (rand() / static_cast<float>(INT_MAX) - 0.5F));
    }
    for (int i = 0; i < LEN_C; ++i) {
        cpu_c[i] = 2.0F * (rand() / static_cast<float>(INT_MAX) - 0.5F);
        ref_c[i] = cpu_c[i];
    }
    if constexpr (VALID) {
        assert(LOOP == 1);
        for (int bi = 0; bi < BATCH_SIZE; ++bi) {
            auto ref_c_ = ref_c + bi * WM * WN;
            auto a_ = cpu_a + bi * WM * WK;
            auto b_ = cpu_b + bi * WN * WK;
            for (int mi = 0; mi < WM; ++mi) {
                for (int ni = 0; ni < WN; ++ni) {
                    float acc = 0.0F;
                    for (int ki = 0; ki < WK; ++ki) {
                        acc += static_cast<float>(a_[mi * WK + ki]) * static_cast<float>(b_[ni * WK + ki]);
                    }
                    ref_c_[mi * WN + ni] = acc;
                }
            }
        }
    }
    scalar_t *gpu_a;
    scalar_t *gpu_b;
    float *gpu_c;
    gpuMalloc(&gpu_c, LEN_C * sizeof(float));
    gpuMalloc(&gpu_a, LEN_A * sizeof(scalar_t));
    gpuMalloc(&gpu_b, LEN_B * sizeof(scalar_t));
    gpuMemcpy(gpu_c, cpu_c, LEN_C * sizeof(float), gpuMemcpyHostToDevice);
    gpuMemcpy(gpu_a, cpu_a, LEN_A * sizeof(scalar_t), gpuMemcpyHostToDevice);
    gpuMemcpy(gpu_b, cpu_b, LEN_B * sizeof(scalar_t), gpuMemcpyHostToDevice);

    gpuEvent_t start, stop;
    gpuEventCreate(&start);
    gpuEventCreate(&stop);
    gpuEventRecord(start);
    wmma_loop_kernel<scalar_t, WMMAT, BLOCK_WARPS, WARP_SIZE, LOOP>
        <<<num_blocks, threads_per_block>>>(gpu_c, gpu_a, gpu_b);
    gpuDeviceSynchronize();
    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float ms = 0.0F;
    gpuEventElapsedTime(&ms, start, stop);
    gpuMemcpy(cpu_c, gpu_c, LEN_C * sizeof(float), gpuMemcpyDeviceToHost);

    if constexpr (VALID) {
        float max_diff = -1.0F;
        for (int i = 0; i < LEN_C; ++i) {
            max_diff = std::max(max_diff, std::abs(ref_c[i] - cpu_c[i]));
        }
        std::cout << "maxdiff:" << max_diff << '\n';
    }

    gpuEventDestroy(start);
    gpuEventDestroy(stop);
    gpuFree(gpu_a);
    gpuFree(gpu_b);
    gpuFree(gpu_c);
    delete[] cpu_a;
    delete[] cpu_b;
    delete[] cpu_c;
    delete[] ref_c;
    return static_cast<double>(2 * WM * WN * WK) * LOOP * BATCH_SIZE / (ms / 1000.0) * 1.0e-12;
}

template <typename scalar_t>
void run_type() {
    using WMMAT = peak_gemm::backend::WmmaDefault<scalar_t, float>;
    constexpr int warp_size = peak_gemm::backend::Warp::size;
    constexpr int accuracy_loop = 1;
    constexpr int accuracy_blocks = 4;
    constexpr int accuracy_block_warps = 8;
    constexpr int loop = 1000000;
    constexpr int blocks = 4096;
    constexpr int block_warps = 8;
    std::cout << "======== " << typeid(WMMAT).name()
              << ", WARP_SIZE=" << warp_size << " ========\n";
    wmma_test<typename WMMAT::ComputeT, WMMAT, accuracy_block_warps, warp_size,
              accuracy_loop, accuracy_blocks, true>();
    for (int i = 0; i < 3; ++i) {
        const auto tflops = wmma_test<typename WMMAT::ComputeT, WMMAT,
                                      block_warps, warp_size, loop, blocks, false>();
        std::cout << tflops << " TFLOPS\n";
    }
}

int main() {
    run_type<__half>();
    run_type<__bfloat16>();
    std::cout << "ok\n";
    return 0;
}
