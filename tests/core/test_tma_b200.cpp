#include <cassert>
#include <cstdint>
#include <iostream>

#if defined(__CUDACC__) && !defined(__HIPCC__) && (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 8))

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cudaTypedefs.h>

#include "peak_gemm/data.hpp"

namespace {

constexpr int MatrixWidth = 48;
constexpr int MatrixHeight = 20;
constexpr int TileWidth = 32;
constexpr int TileHeight = 8;
constexpr int TileElements = TileWidth * TileHeight;

constexpr uint64_t MatrixDimensions[] = {MatrixWidth, MatrixHeight};
constexpr uint64_t MatrixStrides[] = {
    MatrixWidth * sizeof(int32_t)};
constexpr uint32_t TileDimensions[] = {TileWidth, TileHeight};
constexpr uint32_t ElementStrides[] = {1, 1};

using TmaBarrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

template <uint32_t Dim>
class TmaTensorMap {
public:
    TmaTensorMap(
        int32_t *data,
        const uint64_t (&dimensions)[Dim],
        const uint64_t *strides,
        const uint32_t (&box)[Dim],
        const uint32_t (&element_strides)[Dim]) {
        static_assert(Dim >= 1 && Dim <= 5);
        static const auto encode = get_encoder();
        const auto result = encode(
            &descriptor_,
            CU_TENSOR_MAP_DATA_TYPE_INT32,
            Dim,
            data,
            dimensions,
            strides,
            box,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        assert(result == CUDA_SUCCESS);
    }

    const CUtensorMap &descriptor() const {
        return descriptor_;
    }

private:
    static auto get_encoder() {
        void *entry = nullptr;
        const auto result = cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled",
            &entry,
            12000,
            cudaEnableDefault,
            nullptr);
        assert(result == cudaSuccess && entry != nullptr);
        return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(entry);
    }

    CUtensorMap descriptor_{};
};

__global__ void tma_round_trip(
    const __grid_constant__ CUtensorMap source,
    const __grid_constant__ CUtensorMap destination) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    __shared__ alignas(128) int32_t tile[TileHeight][TileWidth];
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ TmaBarrier barrier;

    if (threadIdx.x == 0) init(&barrier, 1);
    __syncthreads();

    const int32_t coordinates[2] = {
        static_cast<int32_t>(blockIdx.x * TileWidth),
        static_cast<int32_t>(blockIdx.y * TileHeight),
    };
    if (threadIdx.x == 0) {
        ptx::cp_async_bulk_tensor(
            ptx::space_shared,
            ptx::space_global,
            &tile,
            &source,
            coordinates,
            cuda::device::barrier_native_handle(barrier));
        barrier.wait(cuda::device::barrier_arrive_tx(
            barrier,
            1,
            sizeof(tile)));
    }
    __syncthreads();

    const int index = threadIdx.x;
    ++tile[index / TileWidth][index % TileWidth];

    ptx::fence_proxy_async(ptx::space_shared);
    __syncthreads();
    if (threadIdx.x == 0) {
        ptx::cp_async_bulk_tensor(
            ptx::space_global,
            ptx::space_shared,
            &destination,
            coordinates,
            &tile);
        ptx::cp_async_bulk_commit_group();
        ptx::cp_async_bulk_wait_group(ptx::n32_t<0>());
        (&barrier)->~TmaBarrier();
    }
#endif
}

void test_tma() {
    constexpr auto cpu = peak_gemm::DataDevice::cpu;
    constexpr auto gpu = peak_gemm::DataDevice::gpu;

    peak_gemm::Data<int32_t> source({MatrixHeight, MatrixWidth});
    for (uint32_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<int32_t>(index);
    }
    auto source_gpu = source.copy_to(gpu);
    peak_gemm::Data<int32_t> destination_gpu(
        {MatrixHeight, MatrixWidth},
        gpu);
    destination_gpu.fill(-1);

    const dim3 grid(
        (MatrixWidth + TileWidth - 1) / TileWidth,
        (MatrixHeight + TileHeight - 1) / TileHeight);
    const TmaTensorMap source_map(
        source_gpu.data(),
        MatrixDimensions,
        MatrixStrides,
        TileDimensions,
        ElementStrides);
    const TmaTensorMap destination_map(
        destination_gpu.data(),
        MatrixDimensions,
        MatrixStrides,
        TileDimensions,
        ElementStrides);
    tma_round_trip<<<grid, TileElements>>>(
        source_map.descriptor(),
        destination_map.descriptor());
    gpuDeviceSynchronize();

    const auto destination = destination_gpu.copy_to(cpu);
    for (uint32_t index = 0; index < source.size(); ++index) {
        assert(destination[index] == source[index] + 1);
    }
    std::cout << "[pass] B200 2D TMA round trip\n";
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
    std::cout << "[skip] B200 TMA requires CUDA 12.8+\n";
    return 0;
}

#endif
