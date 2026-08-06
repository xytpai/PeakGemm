#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#if defined(__CUDACC__) && !defined(__HIPCC__) && (__CUDACC_VER_MAJOR__ >= 13)

#include "peak_gemm/backend/arch_sm100.hpp"

namespace sm100 = peak_gemm::backend::sm100;

using scalar_t = __bfloat16;

constexpr int TileM = 256;
constexpr int TileN = 256;
constexpr int TileK = 64;
constexpr int CtaM = TileM / 2;
constexpr int TileBytes = CtaM * TileK * sizeof(scalar_t);

struct SharedStorage {
    alignas(1024) scalar_t a[CtaM * TileK];
    alignas(1024) scalar_t b[CtaM * TileK];
    alignas(8) uint64_t tma_barrier;
    alignas(8) uint64_t mma_barrier;
    uint32_t tmem_base;
};

__global__ void tcgen05_smoke_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    float *result) {
    sm100::require_sm100a();
    extern __shared__ __align__(1024) char storage_bytes[];
    auto &storage = *reinterpret_cast<SharedStorage *>(storage_bytes);

    const uint32_t lane = threadIdx.x;
    const uint32_t rank = sm100::cluster_rank();

    if (lane == 0) {
        sm100::mbarrier_init(&storage.tma_barrier, 2);
        sm100::mbarrier_init(&storage.mma_barrier, 1);
        sm100::mbarrier_init_fence();
    }
    __syncwarp();
    sm100::cluster_sync();

    sm100::tmem_allocate(&storage.tmem_base, TileN);
    __syncwarp();

    if (lane == 0) {
        sm100::tma_load_3d_2cta(
            storage.a,
            &a_map,
            0,
            rank * CtaM,
            0,
            &storage.tma_barrier);
        sm100::tma_load_3d_2cta(
            storage.b,
            &b_map,
            0,
            rank * CtaM,
            0,
            &storage.tma_barrier);
        sm100::mbarrier_arrive_expect_tx(
            &storage.tma_barrier,
            2 * TileBytes);
    }

    if (rank == 0 && lane == 0) {
        sm100::mbarrier_wait(&storage.tma_barrier, 0);
    }
    __syncwarp();
    sm100::cluster_sync();

    if (rank == 0 && lane == 0) {
        const uint64_t descriptor_a =
            sm100::make_smem_descriptor(storage.a);
        const uint64_t descriptor_b =
            sm100::make_smem_descriptor(storage.b);
        sm100::tcgen05_fence();
        sm100::mma_f16_2cta(
            storage.tmem_base,
            descriptor_a,
            descriptor_b,
            sm100::mma_instruction_descriptor<scalar_t>(),
            false);
        sm100::mma_commit_multicast(&storage.mma_barrier, 0x3);
    }

    if (lane == 0) {
        sm100::mbarrier_wait(&storage.mma_barrier, 0);
    }
    __syncwarp();
    sm100::tcgen05_fence();

    float values[16];
    const uint32_t tmem_row = rank * CtaM;
    sm100::tmem_load_x16(
        values,
        (tmem_row << 16) + storage.tmem_base);
    for (int column = 0; column < 16; ++column) {
        result[(rank * 32 + lane) * 16 + column] = values[column];
    }

    sm100::cluster_sync();
    sm100::tmem_deallocate(storage.tmem_base, TileN);
    sm100::tmem_relinquish();
}

int main() {
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    if (properties.major != 10 || properties.minor != 0) {
        std::cout << "skip: TCGEN05 requires B200\n";
        return 0;
    }

    std::vector<scalar_t> host_a(TileM * TileK, scalar_t(1.0f));
    std::vector<scalar_t> host_b(TileN * TileK, scalar_t(1.0f));
    std::vector<float> host_result(2 * 32 * 16);

    scalar_t *device_a;
    scalar_t *device_b;
    float *device_result;
    cudaMalloc(&device_a, host_a.size() * sizeof(scalar_t));
    cudaMalloc(&device_b, host_b.size() * sizeof(scalar_t));
    cudaMalloc(&device_result, host_result.size() * sizeof(float));
    cudaMemcpy(
        device_a,
        host_a.data(),
        host_a.size() * sizeof(scalar_t),
        cudaMemcpyHostToDevice);
    cudaMemcpy(
        device_b,
        host_b.data(),
        host_b.size() * sizeof(scalar_t),
        cudaMemcpyHostToDevice);

    const uint64_t dimensions_a[3] = {TileK, TileM, 1};
    const uint64_t strides_a[2] = {
        TileK * sizeof(scalar_t),
        TileK * sizeof(scalar_t)};
    const uint32_t box[3] = {TileK, CtaM, 1};
    const uint32_t element_strides[3] = {1, 1, 1};

    sm100::TensorMap<scalar_t, 3> a_map(
        device_a,
        dimensions_a,
        strides_a,
        box,
        element_strides);
    sm100::TensorMap<scalar_t, 3> b_map(
        device_b,
        dimensions_a,
        strides_a,
        box,
        element_strides);

    cudaFuncSetAttribute(
        tcgen05_smoke_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(SharedStorage));

    cudaLaunchAttribute cluster_attribute{};
    cluster_attribute.id = cudaLaunchAttributeClusterDimension;
    cluster_attribute.val.clusterDim = {2, 1, 1};

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(2);
    config.blockDim = dim3(32);
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.attrs = &cluster_attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(
        &config,
        tcgen05_smoke_kernel,
        a_map.descriptor(),
        b_map.descriptor(),
        device_result);
    cudaDeviceSynchronize();

    cudaMemcpy(
        host_result.data(),
        device_result,
        host_result.size() * sizeof(float),
        cudaMemcpyDeviceToHost);

    const bool correct = std::all_of(
        host_result.begin(),
        host_result.end(),
        [](float value) { return std::abs(value - 16.0f) < 1e-3f; });

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_result);

    std::cout << (correct ? "TCGEN05 2CTA smoke test passed\n" : "TCGEN05 2CTA smoke test failed\n");
    return correct ? 0 : 1;
}

#else

int main() {
    std::cout << "skip: TCGEN05 requires CUDA 13+\n";
    return 0;
}

#endif
