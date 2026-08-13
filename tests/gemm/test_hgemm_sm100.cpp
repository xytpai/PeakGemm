#include "peak_gemm/gemm_bench_infra.hpp"

#if defined(__CUDACC__) && !defined(__HIPCC__) && (__CUDACC_VER_MAJOR__ >= 13)

#include "peak_gemm/kernel/hgemm_sm100.hpp"

using peak_gemm::Data;
namespace bench = peak_gemm::bench;
namespace kernel = peak_gemm::kernel;
constexpr auto gpu = peak_gemm::DataDevice::gpu;

struct HgemmSm100Launch {
    mutable Data<uint32_t> semaphore;
    mutable Data<uint32_t> signal;

    HgemmSm100Launch() :
        semaphore({256}, gpu), signal({256}, gpu) {
        semaphore.fill(0U);
        signal.fill(0U);
    }

    template <typename scalar_t>
    void operator()(
        const scalar_t *a,
        const scalar_t *b,
        scalar_t *c,
        uint32_t m,
        uint32_t n,
        uint32_t k,
        const scalar_t *bias) const {
        if (bias == nullptr) {
            kernel::hgemm_template<
                scalar_t,
                128,
                256,
                64,
                6,
                8,
                false,
                false>(
                a,
                b,
                c,
                m,
                n,
                k,
                1,
                semaphore.data(),
                signal.data());
        } else {
            kernel::hgemm_template<
                scalar_t,
                128,
                256,
                64,
                6,
                8,
                true,
                false>(
                a,
                b,
                c,
                m,
                n,
                k,
                1,
                semaphore.data(),
                signal.data(),
                bias);
        }
    }
};

int main() {
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    if (properties.major != 10 || properties.minor != 0) {
        std::cout << "skip: sm100 hgemm requires B200\n";
        return 0;
    }

    const std::vector<bench::GemmShape> mnks{
        {256, 256, 64},
        {256, 256, 384},
        {256, 512, 448},
        {512, 512, 128},
        {512, 512, 2048},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        {4096, 4096, 8192},
        {8192, 8192, 8192},
        {16384, 16384, 16384},
    };

    HgemmSm100Launch launch;
    const bench::GpuNaiveGemmReference reference;

    std::cout << "sm100 2CTA TMA hgemm\n";
    bench::run<__half>(mnks, launch, reference, false);
    bench::run<__half>(mnks, launch, reference, true);
    bench::run<__bfloat16>(mnks, launch, reference, false);
    bench::run<__bfloat16>(mnks, launch, reference, true);
    std::cout << "ok\n";
    return 0;
}

#else

int main() {
    std::cout << "skip: sm100 hgemm requires CUDA 13+\n";
    return 0;
}

#endif
