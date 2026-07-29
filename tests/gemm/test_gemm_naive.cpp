#include "peak_gemm/gemm_bench_infra.hpp"
#include "peak_gemm/kernel/gemm_naive.hpp"

template <typename scalar_t>
struct GemmNaiveLaunch {
    void operator()(
        const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t m, uint32_t n, uint32_t k,
        const scalar_t *bias) const {
        peak_gemm::kernel::gemm_naive_gpu(a, b, c, m, n, k, bias);
    }
};

int main() {
    const std::vector<peak_gemm::bench::GemmShape> mnks{
        {1, 1, 1},
        {15, 15, 15},
        {16, 16, 16},
        {17, 17, 17},
        {31, 33, 65},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {512, 1024, 2048},
    };
    const peak_gemm::bench::DefaultCpuGemmReference reference;
    peak_gemm::bench::run<float>(mnks, GemmNaiveLaunch<float>{}, reference, false);
    peak_gemm::bench::run<float>(mnks, GemmNaiveLaunch<float>{}, reference, true);
    peak_gemm::bench::run<__half>(mnks, GemmNaiveLaunch<__half>{}, reference, false);
    peak_gemm::bench::run<__half>(mnks, GemmNaiveLaunch<__half>{}, reference, true);
    peak_gemm::bench::run<__bfloat16>(mnks, GemmNaiveLaunch<__bfloat16>{}, reference, false);
    peak_gemm::bench::run<__bfloat16>(mnks, GemmNaiveLaunch<__bfloat16>{}, reference, true);
    std::cout << "ok\n";
    return 0;
}
