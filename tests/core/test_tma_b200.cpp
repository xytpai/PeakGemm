#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

#if defined(__CUDACC__) && !defined(__HIPCC__) && (__CUDACC_VER_MAJOR__ >= 13)

#include "peak_gemm/backend/arch_sm100.hpp"
#include "peak_gemm/data.hpp"

namespace {

namespace sm100 = peak_gemm::backend::sm100;

using scalar_t = __half;

constexpr int MatrixWidth = 64;
constexpr int MatrixHeight = 256;
constexpr int TileWidth = 64;
constexpr int TileHeight = 128;
constexpr int TileElements = TileWidth * TileHeight;
constexpr int TileBytes = TileElements * sizeof(scalar_t);

struct SharedStorage {
    alignas(1024) scalar_t tile[TileElements];
    alignas(8) uint64_t barrier;
};

__global__ void tma_2cta_round_trip(
    const __grid_constant__ CUtensorMap source,
    const __grid_constant__ CUtensorMap destination) {
    sm100::require_sm100a();
    extern __shared__ __align__(1024) char storage_bytes[];
    auto &storage = *reinterpret_cast<SharedStorage *>(storage_bytes);

    const uint32_t rank = sm100::cluster_rank();
    const int32_t row = rank * TileHeight;

    if (threadIdx.x == 0) {
        sm100::mbarrier_init(&storage.barrier, 2);
        sm100::mbarrier_init_fence();
    }
    __syncthreads();
    sm100::cluster_sync();

    if (threadIdx.x == 0) {
        sm100::tma_load_2d_2cta(
            storage.tile,
            &source,
            0,
            row,
            &storage.barrier);
        sm100::mbarrier_arrive_expect_tx(
            &storage.barrier,
            TileBytes);
    }

    if (rank == 0 && threadIdx.x == 0) {
        sm100::mbarrier_wait(&storage.barrier, 0);
    }
    __syncthreads();
    sm100::cluster_sync();

    for (uint32_t index = threadIdx.x;
         index < TileElements;
         index += blockDim.x) {
        const uint32_t logical_row = index / TileWidth;
        const uint32_t logical_column = index % TileWidth;
        const uint32_t chunk = logical_column / 8;
        const uint32_t physical =
            sm100::swizzle_128b_chunk(logical_row, chunk) * 8 + logical_column % 8;
        storage.tile[physical] =
            scalar_t(static_cast<float>(storage.tile[physical]) + 1.0f);
    }

    sm100::fence_proxy_async_shared();
    __syncthreads();

    if (threadIdx.x == 0) {
        sm100::tma_store_2d(
            &destination,
            0,
            row,
            storage.tile);
        sm100::tma_store_commit();
        sm100::tma_store_wait<0>();
    }
}

void test_tma() {
    constexpr auto cpu = peak_gemm::DataDevice::cpu;
    constexpr auto gpu = peak_gemm::DataDevice::gpu;

    peak_gemm::Data<scalar_t> source({MatrixHeight, MatrixWidth});
    for (uint32_t index = 0; index < source.size(); ++index) {
        source[index] = scalar_t(static_cast<float>(index % 128));
    }

    auto source_gpu = source.copy_to(gpu);
    peak_gemm::Data<scalar_t> destination_gpu(
        {MatrixHeight, MatrixWidth},
        gpu);
    destination_gpu.fill(scalar_t(-1.0f));

    constexpr uint64_t dimensions[2] = {
        MatrixWidth,
        MatrixHeight};
    constexpr uint64_t strides[1] = {
        MatrixWidth * sizeof(scalar_t)};
    constexpr uint32_t box[2] = {
        TileWidth,
        TileHeight};
    constexpr uint32_t element_strides[2] = {1, 1};

    const sm100::TensorMap<scalar_t, 2> source_map(
        source_gpu.data(),
        dimensions,
        strides,
        box,
        element_strides);
    const sm100::TensorMap<scalar_t, 2> destination_map(
        destination_gpu.data(),
        dimensions,
        strides,
        box,
        element_strides);

    cudaFuncSetAttribute(
        tma_2cta_round_trip,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(SharedStorage));

    cudaLaunchAttribute cluster_attribute{};
    cluster_attribute.id = cudaLaunchAttributeClusterDimension;
    cluster_attribute.val.clusterDim = {2, 1, 1};

    cudaLaunchConfig_t config{};
    config.gridDim = dim3(2);
    config.blockDim = dim3(128);
    config.dynamicSmemBytes = sizeof(SharedStorage);
    config.attrs = &cluster_attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(
        &config,
        tma_2cta_round_trip,
        source_map.descriptor(),
        destination_map.descriptor());
    gpuDeviceSynchronize();

    const auto destination = destination_gpu.copy_to(cpu);
    for (uint32_t index = 0; index < source.size(); ++index) {
        assert(
            std::abs(
                static_cast<float>(destination[index]) - static_cast<float>(source[index]) - 1.0f)
            < 1e-3f);
    }
    std::cout << "[pass] B200 2CTA TMA peer-barrier round trip\n";
}

} // namespace

int main() {
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    if (properties.major != 10 || properties.minor != 0) {
        std::cout << "[skip] B200 TMA requires SM100\n";
        return 0;
    }
    test_tma();
    return 0;
}

#else

int main() {
    std::cout << "[skip] B200 TMA requires CUDA 13+\n";
    return 0;
}

#endif
