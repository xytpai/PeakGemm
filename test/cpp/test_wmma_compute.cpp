#include "device_common.h"

using namespace kernel_utils;
using namespace wmma_utils;
using namespace std;

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
    wmma.load_matrix_a(a_frag, as_, 0, 0, WK);
    wmma.load_matrix_b(b_frag, bs_, 0, 0, WK);
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
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(NBLOCKS);
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
        cpu_a[i] = (scalar_t)(2.f * ((rand() / (float)INT_MAX) - 0.5f));
    }
    for (int i = 0; i < LEN_B; ++i) {
        cpu_b[i] = (scalar_t)(2.f * ((rand() / (float)INT_MAX) - 0.5f));
    }
    for (int i = 0; i < LEN_C; ++i) {
        cpu_c[i] = (2.f * ((rand() / (float)INT_MAX) - 0.5f));
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
                    float acc = 0.0;
                    for (int ki = 0; ki < WK; ++ki) {
                        acc += (float)a_[mi * WK + ki] * (float)b_[ni * WK + ki];
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

    wmma_loop_kernel<scalar_t, WMMAT, BLOCK_WARPS, WARP_SIZE, LOOP><<<numBlocks, threadsPerBlock>>>(gpu_c, gpu_a, gpu_b);
    gpuDeviceSynchronize();

    gpuEventRecord(stop);
    gpuEventSynchronize(stop);
    float ms = 0;
    gpuEventElapsedTime(&ms, start, stop);

    gpuMemcpy(cpu_c, gpu_c, LEN_C * sizeof(float), gpuMemcpyDeviceToHost);

    if constexpr (VALID) {
        float maxdiff = -1.0;
        for (int i = 0; i < LEN_C; ++i) {
            float diff = std::abs(ref_c[i] - cpu_c[i]);
            maxdiff = std::max(maxdiff, diff);
            // std::cout << "ref:" << ref_c[i] << ", out:" << cpu_c[i] << "\n";
        }
        std::cout << "maxdiff:" << maxdiff << "\n";
    }

    gpuFree(gpu_a);
    gpuFree(gpu_b);
    gpuFree(gpu_c);
    delete[] cpu_a;
    delete[] cpu_b;
    delete[] cpu_c;
    delete[] ref_c;
    auto tflops = ((double)2 * WM * WN * WK) * LOOP * BATCH_SIZE / (ms / 1000) * 1e-12;
    return tflops;
}

int main() {
    constexpr int ACC_TEST_LOOP = 1;
    constexpr int ACC_TEST_NBLOCKS = 4;
    constexpr int ACC_TEST_BLOCK_WARPS = 8;

    constexpr int LOOP = 1000000;
    constexpr int NBLOCKS = 4096;
    constexpr int BLOCK_WARPS = 8;

#ifdef __CUDACC__

    {
        constexpr int WARP_SIZE = 32;
        using WMMAT = WMMA_M16N8K16<__half, float, false>;
        std::cout << "======== " << typeid(WMMAT).name() << ", WARP_SIZE=" << WARP_SIZE << " ========\n";
        wmma_test<typename WMMAT::ComputeT, WMMAT, ACC_TEST_BLOCK_WARPS, WARP_SIZE, ACC_TEST_LOOP, ACC_TEST_NBLOCKS, true>();
        for (int i = 0; i < 3; i++) {
            auto tflops = wmma_test<typename WMMAT::ComputeT, WMMAT, BLOCK_WARPS, WARP_SIZE, LOOP, NBLOCKS, false>();
            std::cout << tflops << " TFLOPS" << std::endl;
        }
    }

    {
        constexpr int WARP_SIZE = 32;
        using WMMAT = WMMA_M16N8K16<__bfloat16, float, false>;
        std::cout << "======== " << typeid(WMMAT).name() << ", WARP_SIZE=" << WARP_SIZE << " ========\n";
        wmma_test<typename WMMAT::ComputeT, WMMAT, ACC_TEST_BLOCK_WARPS, WARP_SIZE, ACC_TEST_LOOP, ACC_TEST_NBLOCKS, true>();
        for (int i = 0; i < 3; i++) {
            auto tflops = wmma_test<typename WMMAT::ComputeT, WMMAT, BLOCK_WARPS, WARP_SIZE, LOOP, NBLOCKS, false>();
            std::cout << tflops << " TFLOPS" << std::endl;
        }
    }

#elif defined(__HIPCC__)

    {
        constexpr int WARP_SIZE = 64;
        using WMMAT = WMMA_M16N16K32<__half, float, false>;
        std::cout << "======== " << typeid(WMMAT).name() << ", WARP_SIZE=" << WARP_SIZE << " ========\n";
        wmma_test<typename WMMAT::ComputeT, WMMAT, ACC_TEST_BLOCK_WARPS, WARP_SIZE, ACC_TEST_LOOP, ACC_TEST_NBLOCKS, true>();
        for (int i = 0; i < 3; i++) {
            auto tflops = wmma_test<typename WMMAT::ComputeT, WMMAT, BLOCK_WARPS, WARP_SIZE, LOOP, NBLOCKS, false>();
            std::cout << tflops << " TFLOPS" << std::endl;
        }
    }

    {
        constexpr int WARP_SIZE = 64;
        using WMMAT = WMMA_M16N16K32<__bfloat16, float, false>;
        std::cout << "======== " << typeid(WMMAT).name() << ", WARP_SIZE=" << WARP_SIZE << " ========\n";
        wmma_test<typename WMMAT::ComputeT, WMMAT, ACC_TEST_BLOCK_WARPS, WARP_SIZE, ACC_TEST_LOOP, ACC_TEST_NBLOCKS, true>();
        for (int i = 0; i < 3; i++) {
            auto tflops = wmma_test<typename WMMAT::ComputeT, WMMAT, BLOCK_WARPS, WARP_SIZE, LOOP, NBLOCKS, false>();
            std::cout << tflops << " TFLOPS" << std::endl;
        }
    }

    {
        constexpr int WARP_SIZE = 64;
        using WMMAT = WMMA_M16N16K16<__half, float, false>;
        std::cout << "======== " << typeid(WMMAT).name() << ", WARP_SIZE=" << WARP_SIZE << " ========\n";
        wmma_test<typename WMMAT::ComputeT, WMMAT, ACC_TEST_BLOCK_WARPS, WARP_SIZE, ACC_TEST_LOOP, ACC_TEST_NBLOCKS, true>();
        for (int i = 0; i < 3; i++) {
            auto tflops = wmma_test<typename WMMAT::ComputeT, WMMAT, BLOCK_WARPS, WARP_SIZE, LOOP, NBLOCKS, false>();
            std::cout << tflops << " TFLOPS" << std::endl;
        }
    }

    {
        constexpr int WARP_SIZE = 64;
        using WMMAT = WMMA_M16N16K16<__bfloat16, float, false>;
        std::cout << "======== " << typeid(WMMAT).name() << ", WARP_SIZE=" << WARP_SIZE << " ========\n";
        wmma_test<typename WMMAT::ComputeT, WMMAT, ACC_TEST_BLOCK_WARPS, WARP_SIZE, ACC_TEST_LOOP, ACC_TEST_NBLOCKS, true>();
        for (int i = 0; i < 3; i++) {
            auto tflops = wmma_test<typename WMMAT::ComputeT, WMMAT, BLOCK_WARPS, WARP_SIZE, LOOP, NBLOCKS, false>();
            std::cout << tflops << " TFLOPS" << std::endl;
        }
    }

#endif

    return 0;
}
