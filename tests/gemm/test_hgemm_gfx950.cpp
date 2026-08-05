#include "peak_gemm/gemm_bench_infra.hpp"

#if defined(__HIPCC__)

#include "peak_gemm/backend/arch_gfx950.hpp"
#include "peak_gemm/kernel/hgemm_gfx950.hpp"
#include "peak_gemm/kernel/hgemm_hti_gfx950.hpp"

using peak_gemm::Data;
namespace bench = peak_gemm::bench;
namespace kernel = peak_gemm::kernel;
constexpr auto gpu = peak_gemm::DataDevice::gpu;

struct HgemmGfx950Launch {
    mutable Data<uint32_t> semaphore;
    mutable Data<uint32_t> signal;

    HgemmGfx950Launch() :
        semaphore({kernel::kSemaphoreCount}, gpu), signal({kernel::kSemaphoreCount}, gpu) {
        semaphore.fill(0U);
        signal.fill(0U);
    }

    template <typename scalar_t>
    void operator()(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t m, uint32_t n, uint32_t k, const scalar_t *bias) const {
        if (m <= 256) {
            if (bias == nullptr) {
                kernel::hgemm_template<scalar_t, 16, 64, 64, 1, 1, 2, 0, false, true>(
                    a, b, c, m, n, k, 4, semaphore.data(), signal.data(), bias);
            } else {
                kernel::hgemm_template<scalar_t, 16, 64, 64, 1, 1, 2, 0, true, true>(
                    a, b, c, m, n, k, 4, semaphore.data(), signal.data(), bias);
            }
        } else if (bias == nullptr) {
            if (k % 128 != 0) {
                kernel::hgemm_template<scalar_t, 256, 256, 64, 4, 4, 2, 0, false, false>(
                    a, b, c, m, n, k, 1, semaphore.data(), signal.data(), bias);
            } else {
                kernel::hgemm_hti_template<scalar_t, 256, 256, 64, 2, 4, 0, false>(a, b, c, m, n, k, bias);
            }
        } else {
            if (k % 128 != 0) {
                kernel::hgemm_template<scalar_t, 256, 256, 64, 4, 4, 2, 0, true, false>(
                    a, b, c, m, n, k, 1, semaphore.data(), signal.data(), bias);
            } else {
                kernel::hgemm_hti_template<scalar_t, 256, 256, 64, 2, 4, 0, true>(a, b, c, m, n, k, bias);
            }
        }
    }
};

int main() {
    const std::vector<bench::GemmShape> mnks{
        {16, 64, 256},
        {32, 128, 512},
        {256, 256, 1024},
        {512, 256, 64},
        {512, 256, 128},
        {512, 512, 2048},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {4096, 4096, 8192},
        {8192, 8192, 8192},
        {16384, 16384, 16384},
    };
    HgemmGfx950Launch launch;
    const bench::GpuNaiveGemmReference reference;
    std::cout << "gfx950 hgemm\n";
    bench::run<__half>(mnks, launch, reference, true);
    bench::run<__bfloat16>(mnks, launch, reference, true);
    bench::run<__half>(mnks, launch, reference, false);
    bench::run<__bfloat16>(mnks, launch, reference, false);
    std::cout << "ok\n";
    return 0;
}

#else

int main() {
    std::cout << "skip: gfx950 requires rocm\n";
    return 0;
}

#endif
