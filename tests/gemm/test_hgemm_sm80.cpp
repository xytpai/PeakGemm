#include "peak_gemm/gemm_bench_infra.hpp"

#if defined(__CUDACC__)

#include "peak_gemm/backend/arch_sm80.hpp"
#include "peak_gemm/kernel/hgemm_sm80.hpp"

using peak_gemm::Data;
namespace bench = peak_gemm::bench;
namespace kernel = peak_gemm::kernel;
constexpr auto gpu = peak_gemm::DataDevice::gpu;

struct HgemmSm80Launch {
    mutable Data<uint32_t> semaphore;
    mutable Data<uint32_t> signal;

    HgemmSm80Launch() :
        semaphore({kernel::kSemaphoreCount}, gpu), signal({kernel::kSemaphoreCount}, gpu) {
        semaphore.fill(0U);
        signal.fill(0U);
    }

    template <typename scalar_t>
    void operator()(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t m, uint32_t n, uint32_t k, const scalar_t *bias) const {
        kernel::hgemm_gpu(a, b, c, m, n, k, semaphore.data(), signal.data(), bias);
    }
};

int main() {
    const std::vector<bench::GemmShape> mnks{
        {16, 64, 256},
        {32, 128, 512},
        {256, 256, 1024},
        {512, 128, 128},
        {512, 512, 2048},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {4096, 4096, 8192},
        {8192, 8192, 8192},
        {16384, 16384, 16384},
    };
    HgemmSm80Launch launch;
    const bench::GpuNaiveGemmReference reference;
    std::cout << "sm80 hgemm\n";
    bench::run<__half>(mnks, launch, reference, true);
    bench::run<__bfloat16>(mnks, launch, reference, true);
    bench::run<__half>(mnks, launch, reference, false);
    bench::run<__bfloat16>(mnks, launch, reference, false);
    std::cout << "ok\n";
    return 0;
}

#else

int main() {
    std::cout << "skip: sm80 requires cuda\n";
    return 0;
}

#endif
