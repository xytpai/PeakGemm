#if (defined(__HIPCC__) && (!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx950__))) || \
    (defined(__CUDACC__) && !defined(__HIPCC__) && (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800))

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

#include "peak_gemm/peak_gemm.hpp"

#if defined(__HIPCC__)
#include "peak_gemm/backend/arch_gfx950.hpp"
using SwizzleT = peak_gemm::backend::Gfx950IdentitySwizzle;
#else
#include "peak_gemm/backend/arch_sm80.hpp"
using SwizzleT = peak_gemm::backend::IdentitySwizzle;
#endif

template <typename scalar_t>
__global__ void wmma_kernel(
    const scalar_t *a,
    const scalar_t *b,
    scalar_t *c,
    uint32_t m_size,
    uint32_t n_size,
    uint32_t k_size) {
    using WmmaT = peak_gemm::backend::WmmaDefault<scalar_t, float>;
    __shared__ scalar_t tile_a[WmmaT::M * WmmaT::K];
    __shared__ scalar_t tile_b[WmmaT::N * WmmaT::K];
    __shared__ scalar_t tile_c[WmmaT::M * WmmaT::N];
    const uint32_t tile_m = blockIdx.y * WmmaT::M;
    const uint32_t tile_n = blockIdx.x * WmmaT::N;
    WmmaT wmma;
    typename WmmaT::FragmentAT fragment_a;
    typename WmmaT::FragmentBT fragment_b;
    typename WmmaT::FragmentCT fragment_c;
    wmma.init(threadIdx.x);
    wmma.reset_fragment_c(fragment_c);
    const SwizzleT swizzle;

    for (uint32_t tile_k = 0; tile_k < k_size; tile_k += WmmaT::K) {
        for (uint32_t i = threadIdx.x; i < WmmaT::M * WmmaT::K; i += blockDim.x) {
            const uint32_t m = tile_m + i / WmmaT::K;
            const uint32_t k = tile_k + i % WmmaT::K;
            tile_a[i] = m < m_size && k < k_size ? a[m * k_size + k] : scalar_t(0.0F);
        }
        for (uint32_t i = threadIdx.x; i < WmmaT::N * WmmaT::K; i += blockDim.x) {
            const uint32_t n = tile_n + i / WmmaT::K;
            const uint32_t k = tile_k + i % WmmaT::K;
            tile_b[i] = n < n_size && k < k_size ? b[n * k_size + k] : scalar_t(0.0F);
        }
        __syncthreads();
        wmma.load_matrix_a(fragment_a, tile_a, 0, 0, WmmaT::K, swizzle);
        wmma.load_matrix_b(fragment_b, tile_b, 0, 0, WmmaT::K, swizzle);
        wmma(fragment_c, fragment_a, fragment_b, fragment_c);
        __syncthreads();
    }

    wmma.store_matrix(tile_c, WmmaT::N, fragment_c);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < WmmaT::M * WmmaT::N; i += blockDim.x) {
        const uint32_t m = tile_m + i / WmmaT::N;
        const uint32_t n = tile_n + i % WmmaT::N;
        if (m < m_size && n < n_size) c[m * n_size + n] = tile_c[i];
    }
}

template <typename scalar_t>
void wmma_gpu(
    const scalar_t *a,
    const scalar_t *b,
    scalar_t *c,
    uint32_t m_size,
    uint32_t n_size,
    uint32_t k_size,
    gpuStream_t stream = nullptr) {
    using WmmaT = peak_gemm::backend::WmmaDefault<scalar_t, float>;
    using Warp = peak_gemm::backend::Warp;
    dim3 grid(
        (n_size + WmmaT::N - 1) / WmmaT::N,
        (m_size + WmmaT::M - 1) / WmmaT::M);

    wmma_kernel<scalar_t><<<grid, Warp::size, 0, stream>>>(
        a, b, c, m_size, n_size, k_size);
}

int main() {
#if defined(__HIPCC__)
    int device = 0;
    hipDeviceProp_t properties{};
    auto status = hipGetDevice(&device);
    if (status == hipSuccess) status = hipGetDeviceProperties(&properties, device);
    if (status != hipSuccess) {
        std::fprintf(stderr, "GPU query failed: %s\n", hipGetErrorString(status));
        return 1;
    }
    if (!((std::strncmp(properties.gcnArchName, "gfx950", 6) == 0 && (properties.gcnArchName[6] == '\0' || properties.gcnArchName[6] == ':')))) {
        std::cout << "[skip] unsupported GPU architecture\n";
        return 0;
    }
#else
    int device = 0;
    cudaDeviceProp properties{};
    auto status = cudaGetDevice(&device);
    if (status == cudaSuccess) status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "GPU query failed: %s\n", cudaGetErrorString(status));
        return 1;
    }
    if (!(properties.major >= 8)) {
        std::cout << "[skip] unsupported GPU architecture\n";
        return 0;
    }
#endif
    using Data = peak_gemm::Data<__half>;
    constexpr auto cpu = peak_gemm::DataDevice::cpu;
    constexpr auto gpu = peak_gemm::DataDevice::gpu;
    struct TestCase {
        uint32_t m, n, k;
    };
    constexpr TestCase cases[] = {
        {1, 1, 1},
        {15, 7, 15},
        {16, 16, 16},
        {17, 19, 33},
        {35, 53, 71},
        {64, 64, 128},
        {256, 256, 256},
    };
    uint64_t seed = 2026;

    for (const auto [m_size, n_size, k_size] : cases) {
        auto a = Data::uniform({m_size, k_size}, -1.0F, 1.0F, cpu, seed++);
        auto b = Data::uniform({n_size, k_size}, -1.0F, 1.0F, cpu, seed++);
        peak_gemm::Data<float> expected({m_size, n_size}, cpu);
        for (uint32_t m = 0; m < m_size; ++m) {
            for (uint32_t n = 0; n < n_size; ++n) {
                expected[m * n_size + n] = 0.0F;
                for (uint32_t k = 0; k < k_size; ++k) {
                    expected[m * n_size + n] +=
                        static_cast<float>(a[m * k_size + k]) * static_cast<float>(b[n * k_size + k]);
                }
            }
        }

        auto device_a = a.copy_to(gpu);
        auto device_b = b.copy_to(gpu);
        Data device_c({m_size, n_size}, gpu);
        wmma_gpu(
            device_a.data(), device_b.data(), device_c.data(), m_size, n_size, k_size);
        auto c = device_c.copy_to(cpu);
        float max_diff = 0.0F;
        for (uint32_t i = 0; i < m_size * n_size; ++i) {
            max_diff = std::max(
                max_diff,
                std::abs(static_cast<float>(c[i]) - expected[i]));
        }
        assert(max_diff <= 0.01);
        std::cout << "M=" << m_size << ", N=" << n_size << ", K=" << k_size
                  << ", max_diff=" << max_diff << '\n';
    }
    std::cout << "ok\n";
    return 0;
}

#else

#include <iostream>

int main() {
    std::cout << "[skip] unsupported GPU architecture\n";
    return 0;
}

#endif
