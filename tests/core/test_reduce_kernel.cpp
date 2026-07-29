#include "peak_gemm/peak_gemm.hpp"

using Warp = peak_gemm::backend::Warp;

__global__ void reduce_kernel(const float *in, float *out, uint32_t reduce_size) {
    uint32_t batch = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t wid = tid / Warp::size;
    uint32_t warp_count = (blockDim.x + Warp::size - 1) / Warp::size;
    const auto row = in + batch * reduce_size;
    extern __shared__ float warp_sums[];

    float value = 0.0;
    for (uint32_t index = tid; index < reduce_size; index += blockDim.x) {
        value += row[index];
    }

#pragma unroll
    for (uint32_t offset = Warp::size >> 1; offset > 0; offset >>= 1) {
        value += Warp::shuffle_xor(value, offset);
    }

    if (tid % Warp::size == 0) {
        warp_sums[wid] = value;
    }
    __syncthreads();

    if (tid == 0) {
        float block_sum = 0.0F;
        for (uint32_t index = 0; index < warp_count; ++index) {
            block_sum += warp_sums[index];
        }
        out[batch] = block_sum;
    }
}

void reduce_gpu(const float *in, float *out, uint32_t batch_size, uint32_t reduce_size, gpuStream_t stream = nullptr) {
    reduce_kernel<<<batch_size, 1024, 1024 / Warp::size * sizeof(float), stream>>>(in, out, reduce_size);
}

int main() {
    using Data = peak_gemm::Data<float>;
    constexpr auto cpu = peak_gemm::DataDevice::cpu;
    constexpr auto gpu = peak_gemm::DataDevice::gpu;
    struct TestCase {
        uint32_t batch_size, reduce_size;
    };
    constexpr TestCase cases[] = {
        {1, 1},
        {3, Warp::size - 1},
        {7, Warp::size},
        {17, Warp::size + 13},
        {31, 1024},
        {65, 1024 + 17},
        {11, 4097},
    };

    uint64_t seed = 2026;
    for (const auto [batch_size, reduce_size] : cases) {
        auto host_x = Data::uniform(
            {batch_size, reduce_size}, -1.0F, 1.0F, cpu, seed++);
        Data expected_y({batch_size}, cpu);
        expected_y.fill(0.0F);
        for (uint32_t batch = 0; batch < batch_size; ++batch) {
            for (uint32_t index = 0; index < reduce_size; ++index) {
                expected_y[batch] += host_x[batch * reduce_size + index];
            }
        }

        auto device_x = host_x.copy_to(gpu);
        Data device_y({batch_size}, gpu);
        reduce_gpu(device_x.data(), device_y.data(), batch_size, reduce_size);
        auto host_y = device_y.copy_to(cpu);
        float max_diff = 0.0F;
        for (uint32_t batch = 0; batch < batch_size; ++batch) {
            max_diff = std::max(max_diff, std::abs(host_y[batch] - expected_y[batch]));
        }
        assert(max_diff <= 0.1);

        std::cout << "batch=" << batch_size << ", reduce_size=" << reduce_size
                  << ", max_diff=" << max_diff << '\n';
    }
    return 0;
}
