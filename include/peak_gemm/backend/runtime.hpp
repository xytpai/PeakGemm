#pragma once

#include <cstdint>

#include "peak_gemm/core/config.hpp"

#if defined(__HIPCC__)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace peak_gemm::backend {

struct Warp {
    static constexpr uint32_t size = 64;

    template <typename T>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static T shuffle(T value, int source_lane) {
        return __shfl(value, source_lane, size);
    }

    template <typename T>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static T shuffle_xor(T value, int lane_mask) {
        return __shfl_xor(value, lane_mask, size);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static void barrier() {
        __builtin_amdgcn_wave_barrier();
    }
};

template <typename scalar_t>
PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE scalar_t atomic_add(scalar_t *destination, scalar_t value) {
    return atomicAdd(destination, value);
}

template <typename scalar_t>
PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE scalar_t atomic_exchange(scalar_t *destination, scalar_t value) {
    return atomicExch(destination, value);
}

template <typename scalar_t>
PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void atomic_pair_add(scalar_t *destination, const scalar_t *source);

using AtomicFp16x2 = __fp16 __attribute__((__vector_size__(4)));
using AtomicBf16x2 = __bf16 __attribute__((__vector_size__(4)));

template <>
PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void atomic_pair_add(__half *destination, const __half *source) {
    auto *packed_destination = reinterpret_cast<AtomicFp16x2 *>(destination);
    const auto packed_source = *reinterpret_cast<const AtomicFp16x2 *>(source);
    __builtin_amdgcn_global_atomic_fadd_v2f16(packed_destination, packed_source);
}

template <>
PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void atomic_pair_add(__hip_bfloat16 *destination, const __hip_bfloat16 *source) {
    auto *packed_destination = reinterpret_cast<AtomicBf16x2 *>(destination);
    const auto packed_source = *reinterpret_cast<const AtomicBf16x2 *>(source);
    __builtin_amdgcn_global_atomic_fadd_v2bf16(packed_destination, packed_source);
}

} // namespace peak_gemm::backend

#define gpuSuccess hipSuccess
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemset hipMemset
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuMemcpyPeerAsync hipMemcpyPeerAsync
#define gpuDeviceCanAccessPeer hipDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess hipDeviceEnablePeerAccess

#define gpuEvent_t hipEvent_t
#define gpuEventCreate hipEventCreate
#define gpuEventDestroy hipEventDestroy
#define gpuEventRecord hipEventRecord
#define gpuEventSynchronize hipEventSynchronize
#define gpuEventElapsedTime hipEventElapsedTime

#define gpuStream_t hipStream_t
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize

#define gpuFuncAttributes hipFuncAttributes
#define gpuFuncGetAttributes hipFuncGetAttributes
#define gpuDeviceGetAttribute hipDeviceGetAttribute
#define gpuDevAttrMaxRegistersPerBlock \
    hipDeviceAttributeMaxRegistersPerBlock
#define gpuDevAttrMultiProcessorCount \
    hipDeviceAttributeMultiprocessorCount

#define __bfloat16 __hip_bfloat16
#define __bfloat16_raw __hip_bfloat16_raw

#define gpuIpcMemHandle_t hipIpcMemHandle_t
#define gpuIpcGetMemHandle hipIpcGetMemHandle
#define gpuIpcOpenMemHandle hipIpcOpenMemHandle
#define gpuIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define gpuPointerGetAttribute hipPointerGetAttribute
#define GPU_POINTER_ATTRIBUTE_RANGE_START_ADDR \
    HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR
#define gpuDeviceptr_t hipDeviceptr_t
#define gpuStreamCaptureStatus hipStreamCaptureStatus
#define gpuStreamIsCapturing hipStreamIsCapturing
#define gpuStreamCaptureStatusActive hipStreamCaptureStatusActive

#elif defined(__CUDACC__)

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace peak_gemm::backend {

struct Warp {
    static constexpr uint32_t size = 32;

    template <typename T>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static T shuffle(T value, int source_lane) {
        return __shfl_sync(0xffffffffU, value, source_lane, size);
    }

    template <typename T>
    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static T shuffle_xor(T value, int lane_mask) {
        return __shfl_xor_sync(0xffffffffU, value, lane_mask, size);
    }

    PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE static void barrier() {
        __syncwarp();
    }
};

template <typename scalar_t>
PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE scalar_t atomic_add(scalar_t *destination, scalar_t value) {
    return atomicAdd(destination, value);
}

template <typename scalar_t>
PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE scalar_t atomic_exchange(scalar_t *destination, scalar_t value) {
    return atomicExch(destination, value);
}

template <typename scalar_t>
PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void atomic_pair_add(scalar_t *destination, const scalar_t *source);

template <>
PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void atomic_pair_add(__half *destination, const __half *source) {
    atomicAdd(&destination[0], source[0]);
    atomicAdd(&destination[1], source[1]);
}

template <>
PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE void atomic_pair_add(__nv_bfloat16 *destination, const __nv_bfloat16 *source) {
    atomicAdd(&destination[0], source[0]);
    atomicAdd(&destination[1], source[1]);
}

} // namespace peak_gemm::backend

#define gpuSuccess cudaSuccess
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemset cudaMemset
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuMemcpyPeerAsync cudaMemcpyPeerAsync
#define gpuDeviceCanAccessPeer cudaDeviceCanAccessPeer
#define gpuDeviceEnablePeerAccess cudaDeviceEnablePeerAccess

#define gpuEvent_t cudaEvent_t
#define gpuEventCreate cudaEventCreate
#define gpuEventDestroy cudaEventDestroy
#define gpuEventRecord cudaEventRecord
#define gpuEventSynchronize cudaEventSynchronize
#define gpuEventElapsedTime cudaEventElapsedTime

#define gpuStream_t cudaStream_t
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize

#define gpuFuncAttributes cudaFuncAttributes
#define gpuFuncGetAttributes cudaFuncGetAttributes
#define gpuDeviceGetAttribute cudaDeviceGetAttribute
#define gpuDevAttrMaxRegistersPerBlock cudaDevAttrMaxRegistersPerBlock
#define gpuDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount

#define __bfloat16 __nv_bfloat16
#define __bfloat16_raw __nv_bfloat16_raw

#define gpuIpcMemHandle_t cudaIpcMemHandle_t
#define gpuIpcGetMemHandle cudaIpcGetMemHandle
#define gpuIpcOpenMemHandle cudaIpcOpenMemHandle
#define gpuIpcMemLazyEnablePeerAccess cudaIpcMemLazyEnablePeerAccess
#define gpuPointerGetAttribute cuPointerGetAttribute
#define GPU_POINTER_ATTRIBUTE_RANGE_START_ADDR \
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR
#define gpuDeviceptr_t CUdeviceptr
#define gpuStreamCaptureStatus cudaStreamCaptureStatus
#define gpuStreamIsCapturing cudaStreamIsCapturing
#define gpuStreamCaptureStatusActive cudaStreamCaptureStatusActive

#else
#error "Compile PeakGemm with nvcc or hipcc"
#endif
