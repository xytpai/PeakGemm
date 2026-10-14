#include <iostream>

#include "peak_gemm/data.hpp"

#if defined(__HIPCC__)
#include "peak_gemm/backend/arch_gfx950.hpp"
#include "peak_gemm/core/math.hpp"

using peak_gemm::Data;
namespace backend = peak_gemm::backend;
constexpr auto cpu = peak_gemm::DataDevice::cpu;
constexpr auto gpu = peak_gemm::DataDevice::gpu;

template <typename T, uint32_t COLS>
__global__ void load_b_fragments(T *output, uint32_t row, uint32_t column, bool raw_bits = false) {
    // Same NN layout used by the standard and HTI kernels.
    using swizzle_t = backend::Gfx950Swizzle<2, 4, peak_gemm::core::Log2<COLS>::value - 4>;
    const swizzle_t swizzle;
    __shared__ T shared[64 * COLS];
    for (uint32_t i = threadIdx.x; i < 64 * COLS; i += 64) {
        if (raw_bits) {
            // Include signed zero, infinities, NaNs and subnormal bit patterns.
            const uint16_t bits[] = {0x0000, 0x8000, 0x0001, 0x7c00, 0x7f80, 0x7fc1, 0x7fff, 0xffff};
            *reinterpret_cast<uint16_t *>(&shared[swizzle(i)]) = bits[(i / COLS + i % COLS) % 8];
        } else {
            shared[swizzle(i)] = static_cast<T>(static_cast<float>(i % 251));
        }
    }
    __syncthreads();
    backend::WmmaDefault<T, float> wmma;
    wmma.init(threadIdx.x);
    typename decltype(wmma)::FragmentBT fragment;
    wmma.template load_matrix_b<swizzle_t, false>(fragment, shared, row, column, COLS, swizzle);
    *reinterpret_cast<decltype(fragment) *>(output + threadIdx.x * 8) = fragment;
}

template <typename T, uint32_t COLS>
void test_layout() {
    Data<T> actual({64, 8}, gpu);
    Data<float> expected({64, 8}, cpu);
    for (uint32_t n = 0; n < COLS; n += 16) {
        for (uint32_t k : {0U, 32U}) {
            load_b_fragments<T, COLS><<<1, 64>>>(actual.data(), n, k);
            if (gpuGetLastError() != gpuSuccess) throw std::runtime_error("TR16 launch failed");
            for (uint32_t lane = 0; lane < 64; ++lane) {
                for (uint32_t i = 0; i < 8; ++i) {
                    const uint32_t offset = (k + lane / 16 * 8 + i) * COLS + n + lane % 16;
                    expected[lane * 8 + i] = static_cast<float>(offset % 251);
                }
            }
            peak_gemm::check_allclose(actual.copy_to(cpu), expected, 0, 0, "TR16 fragment");
        }
    }
    std::cout << "[pass] TR16 NN fragments: cols=" << COLS << ", all lanes and K slices\n";
}

template <typename T>
void test_bit_preservation() {
    constexpr uint32_t COLS = 128;
    const uint16_t bits[] = {0x0000, 0x8000, 0x0001, 0x7c00, 0x7f80, 0x7fc1, 0x7fff, 0xffff};
    Data<uint16_t> actual({64, 8}, gpu), expected({64, 8}, cpu);
    load_b_fragments<T, COLS><<<1, 64>>>(reinterpret_cast<T *>(actual.data()), 16, 32, true);
    if (gpuGetLastError() != gpuSuccess) throw std::runtime_error("TR16 launch failed");
    for (uint32_t lane = 0; lane < 64; ++lane) {
        for (uint32_t i = 0; i < 8; ++i) {
            const uint32_t k = 32 + lane / 16 * 8 + i;
            const uint32_t n = 16 + lane % 16;
            expected[lane * 8 + i] = bits[(k + n) % 8];
        }
    }
    peak_gemm::check_allclose(actual.copy_to(cpu), expected, 0, 0, "TR16 bits");
    std::cout << "[pass] TR16 preserves all tested 16-bit encodings\n";
}

int main() {
    test_bit_preservation<__half>();
    test_bit_preservation<__bfloat16>();
    test_layout<__half, 64>();
    test_layout<__half, 128>();
    test_layout<__half, 256>();
    test_layout<__bfloat16, 64>();
    test_layout<__bfloat16, 128>();
    test_layout<__bfloat16, 256>();
    std::cout << "ok\n";
}
#else
int main() {
    std::cout << "skip: LDS TR16 requires gfx950\n";
}
#endif
