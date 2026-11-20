#pragma once

#include "peak_gemm/backend/runtime.hpp"

#if defined(__CUDACC__) && !defined(__HIPCC__)
#include <cuda/atomic>
#endif

namespace peak_gemm::backend {

// Inter-GPU flags require coherent, system-atomic-capable memory and P2P support.
// On HIP use fine-grained memory, not an ordinary coarse-grained hipMalloc flag.
PEAKGEMM_DEVICE_INLINE uint32_t load_acquire_system(uint32_t *address) {
#if defined(__HIPCC__)
    return __hip_atomic_load(address, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> value(*address);
    return value.load(cuda::memory_order_acquire);
#endif
}

PEAKGEMM_DEVICE_INLINE void store_release_system(uint32_t *address, uint32_t value) {
#if defined(__HIPCC__)
    __hip_atomic_store(address, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> destination(*address);
    destination.store(value, cuda::memory_order_release);
#endif
}

} // namespace peak_gemm::backend
