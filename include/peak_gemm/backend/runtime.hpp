#pragma once

#if defined(__HIPCC__)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "peak_gemm/backend/hip/group.hpp"

namespace peak_gemm::backend {
using Warp = hip::Wave;
}

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

#include "peak_gemm/backend/cuda/group.hpp"

namespace peak_gemm::backend {
using Warp = cuda::Warp;
}

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
