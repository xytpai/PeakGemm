#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define PEAKGEMM_HOST_DEVICE __host__ __device__
#define PEAKGEMM_DEVICE __device__
#define PEAKGEMM_HOST __host__
#define PEAKGEMM_FORCEINLINE __forceinline__
#elif defined(__HIPCC__)
#define PEAKGEMM_HOST_DEVICE __host__ __device__
#define PEAKGEMM_DEVICE __device__
#define PEAKGEMM_HOST __host__
#define PEAKGEMM_FORCEINLINE inline __attribute__((always_inline))
#else
#define PEAKGEMM_HOST_DEVICE
#define PEAKGEMM_DEVICE
#define PEAKGEMM_HOST
#define PEAKGEMM_FORCEINLINE inline __attribute__((always_inline))
#endif

#define PEAKGEMM_DEVICE_INLINE PEAKGEMM_DEVICE PEAKGEMM_FORCEINLINE

namespace peak_gemm {

enum class Backend : std::uint8_t {
    cuda,
    hip,
};

enum class DataType : std::uint8_t {
    fp32,
    fp16,
    bf16,
};

} // namespace peak_gemm
