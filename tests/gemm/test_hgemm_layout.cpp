#include <cmath>
#include <iostream>
#include <vector>

#include "peak_gemm/data.hpp"

#if defined(__HIPCC__)
#include "peak_gemm/backend/arch_gfx950.hpp"
#include "peak_gemm/kernel/hgemm_gfx950.hpp"
#include "peak_gemm/kernel/hgemm_hti_gfx950.hpp"
constexpr uint32_t N_WARPS = 1, SWIZZLE_M = 0;
#else
#include "peak_gemm/backend/arch_sm80.hpp"
#include "peak_gemm/kernel/hgemm_sm80.hpp"
constexpr uint32_t N_WARPS = 2, SWIZZLE_M = 8;
#endif

using peak_gemm::Data;
namespace kernel = peak_gemm::kernel;
constexpr auto cpu = peak_gemm::DataDevice::cpu;
constexpr auto gpu = peak_gemm::DataDevice::gpu;

template <typename T, bool B_TRANSPOSE, bool HAS_BIAS, bool SPLIT_K, bool LARGE>
void test_case(uint32_t m, uint32_t n, uint32_t k) {
    auto a = Data<T>::uniform({m, k}, -0.5F, 0.5F, cpu, 2026);
    auto b = Data<T>::uniform({k, n}, -0.5F, 0.5F, cpu, 2027);
    auto bias = Data<T>::uniform({n}, -1.0F, 1.0F, cpu, 2028);
    Data<T> stored_b({k * n}, cpu);
    std::vector<float> expected(m * n);
    for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            stored_b[B_TRANSPOSE ? j * k + i : i * n + j] = b[i * n + j];
        }
    }
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            float sum = 0;
            for (uint32_t p = 0; p < k; ++p) {
                sum += static_cast<float>(a[i * k + p]) * static_cast<float>(b[p * n + j]);
            }
            expected[i * n + j] = sum + (HAS_BIAS ? static_cast<float>(bias[j]) : 0.0F);
        }
    }

    auto da = a.copy_to(gpu), db = stored_b.copy_to(gpu), dbias = bias.copy_to(gpu);
    Data<T> c({m, n}, gpu);
    Data<uint32_t> semaphore({kernel::kSemaphoreCount}, gpu), signal({kernel::kSemaphoreCount}, gpu);
    semaphore.fill(0U);
    signal.fill(0U);
    gpuStream_t stream;
    if (gpuStreamCreate(&stream) != gpuSuccess) throw std::runtime_error("stream creation failed");
    for (int repeat = 0; repeat < 3; ++repeat) {
        const auto *bias_ptr = HAS_BIAS ? dbias.data() : nullptr;
        if constexpr (LARGE) {
#if defined(__HIPCC__)
            kernel::hgemm_hti_template<T, 256, 256, 64, 2, 4, 0, HAS_BIAS, B_TRANSPOSE>(
                da.data(), db.data(), c.data(), m, n, k, bias_ptr, stream);
#else
            kernel::hgemm_template<T, 128, 128, 32, 2, 4, 3, 8, HAS_BIAS, false, B_TRANSPOSE>(
                da.data(), db.data(), c.data(), m, n, k, 1, semaphore.data(), signal.data(), bias_ptr, stream);
#endif
        } else {
            kernel::hgemm_template<T, 16, 64, 64, 1, N_WARPS, 2, SWIZZLE_M, HAS_BIAS, SPLIT_K, B_TRANSPOSE>(
                da.data(), db.data(), c.data(), m, n, k, SPLIT_K ? 4 : 1, semaphore.data(), signal.data(), bias_ptr, stream);
        }
    }
    if (gpuGetLastError() != gpuSuccess || gpuStreamSynchronize(stream) != gpuSuccess) {
        throw std::runtime_error("GEMM launch failed");
    }
    const double tol = (std::is_same_v<T, __half> ? 2e-3 : 2e-2) * std::sqrt(k);
    const double error = peak_gemm::check_allclose(c.copy_to(cpu), expected, tol, 0, "gemm");
    if (gpuStreamDestroy(stream) != gpuSuccess) throw std::runtime_error("stream destruction failed");
    std::cout << (B_TRANSPOSE ? "NT" : "NN") << " M/N/K=" << m << '/' << n << '/' << k
              << " bias=" << HAS_BIAS << " split=" << SPLIT_K << " large=" << LARGE << " max_diff=" << error << '\n';
}

template <typename T, bool B_TRANSPOSE, bool HAS_BIAS>
void test_layout() {
    test_case<T, B_TRANSPOSE, HAS_BIAS, false, false>(32, 192, 64);
    test_case<T, B_TRANSPOSE, HAS_BIAS, false, false>(64, 192, 256);
    test_case<T, B_TRANSPOSE, HAS_BIAS, true, false>(32, 192, 256);
    for (uint32_t k : {128U, 384U, 1024U}) {
        test_case<T, B_TRANSPOSE, HAS_BIAS, false, true>(512, 768, k);
    }
}

template <typename T>
void test_type() {
    test_layout<T, true, false>();
    test_layout<T, true, true>();
    test_layout<T, false, false>();
    test_layout<T, false, true>();
}

int main() {
    test_type<__half>();
    test_type<__bfloat16>();
    std::cout << "ok\n";
}
