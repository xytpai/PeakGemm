#include <cassert>
#include <iostream>

#include "peak_gemm/backend/runtime.hpp"
#include "peak_gemm/data.hpp"

using peak_gemm::Data;
namespace backend = peak_gemm::backend;
constexpr auto cpu = peak_gemm::DataDevice::cpu;
constexpr auto gpu = peak_gemm::DataDevice::gpu;

__global__ void scalar_atomic_kernel(uint32_t *counter, uint32_t *exchange_value, uint32_t *exchange_old) {
    backend::atomic_add(counter, 1U);
    if (threadIdx.x == 0) exchange_old[0] = backend::atomic_exchange(exchange_value, 42U);
}

template <typename scalar_t>
__global__ void pair_atomic_kernel(scalar_t *destination, const scalar_t *source) {
    backend::atomic_pair_add(destination, source);
}

void test_scalar_atomics() {
    Data<uint32_t> counter({1}, gpu), exchange_value({1}, gpu), exchange_old({1}, gpu);
    counter.fill(0U);
    exchange_value.fill(7U);
    exchange_old.fill(0U);
    scalar_atomic_kernel<<<1, 256>>>(counter.data(), exchange_value.data(), exchange_old.data());
    gpuDeviceSynchronize();
    const auto counter_cpu = counter.copy_to(cpu);
    const auto exchange_value_cpu = exchange_value.copy_to(cpu);
    const auto exchange_old_cpu = exchange_old.copy_to(cpu);
    assert(counter_cpu[0] == 256U && exchange_value_cpu[0] == 42U && exchange_old_cpu[0] == 7U);
    std::cout << "[pass] atomic add and exchange\n";
}

template <typename scalar_t>
void test_pair_atomic(const char *type_name) {
    Data<scalar_t> source({2});
    source[0] = static_cast<scalar_t>(1.0F);
    source[1] = static_cast<scalar_t>(2.0F);
    auto source_gpu = source.copy_to(gpu);
    Data<scalar_t> destination({2}, gpu);
    destination.fill(static_cast<scalar_t>(0.0F));
    pair_atomic_kernel<scalar_t><<<1, 64>>>(destination.data(), source_gpu.data());
    gpuDeviceSynchronize();
    const auto result = destination.copy_to(cpu);
    assert(static_cast<float>(result[0]) == 64.0F && static_cast<float>(result[1]) == 128.0F);
    std::cout << "[pass] " << type_name << " atomic pair add\n";
}

int main() {
    test_scalar_atomics();
    test_pair_atomic<__half>("fp16");
    test_pair_atomic<__bfloat16>("bf16");
    std::cout << "ok\n";
    return 0;
}
