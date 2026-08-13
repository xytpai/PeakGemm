#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include <cuda.h>
#include <cudaTypedefs.h>

#include "peak_gemm/backend/runtime.hpp"
#include "peak_gemm/core/config.hpp"

#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || (defined(__CUDA_ARCH_SPECIFIC__) && __CUDA_ARCH_SPECIFIC__ == 1000)
#define PEAKGEMM_ARCH_SM100A 1
#else
#define PEAKGEMM_ARCH_SM100A 0
#endif

namespace peak_gemm::backend::sm100 {

constexpr uint32_t WarpSize = 32;
constexpr uint32_t CtaGroupSize = 2;
constexpr uint32_t PeerBarrierMask = 0xFEFFFFFFU;

template <typename scalar_t>
constexpr CUtensorMapDataType tensor_map_data_type() {
    if constexpr (std::is_same_v<scalar_t, __half>) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if constexpr (std::is_same_v<scalar_t, __bfloat16>) {
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if constexpr (std::is_same_v<scalar_t, float>) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    } else {
        static_assert(std::is_same_v<scalar_t, int32_t>);
        return CU_TENSOR_MAP_DATA_TYPE_INT32;
    }
}

inline auto tensor_map_encoder() {
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

template <typename scalar_t, uint32_t Dim>
class TensorMap {
public:
    TensorMap(
        const scalar_t *data,
        const uint64_t (&dimensions)[Dim],
        const uint64_t *strides,
        const uint32_t (&box)[Dim],
        const uint32_t (&element_strides)[Dim],
        CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_128B) {
        static_assert(Dim >= 1 && Dim <= 5);
        static const auto encode = tensor_map_encoder();
        const auto result = encode(
            &descriptor_,
            tensor_map_data_type<scalar_t>(),
            Dim,
            const_cast<scalar_t *>(data),
            dimensions,
            strides,
            box,
            element_strides,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            swizzle,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        assert(result == CUDA_SUCCESS);
    }

    const CUtensorMap &descriptor() const {
        return descriptor_;
    }

private:
    CUtensorMap descriptor_{};
};

PEAKGEMM_DEVICE_INLINE uint32_t shared_address(const void *pointer) {
    return static_cast<uint32_t>(
        __cvta_generic_to_shared(const_cast<void *>(pointer)));
}

PEAKGEMM_DEVICE_INLINE void require_sm100a() {
#if !PEAKGEMM_ARCH_SM100A
    asm volatile("trap;");
#endif
}

PEAKGEMM_DEVICE_INLINE uint32_t cluster_rank() {
#if PEAKGEMM_ARCH_SM100A
    uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
#else
    return 0;
#endif
}

PEAKGEMM_DEVICE_INLINE void cluster_sync() {
#if PEAKGEMM_ARCH_SM100A
    asm volatile(
        "barrier.cluster.arrive.relaxed.aligned;\n"
        "barrier.cluster.wait.acquire.aligned;\n" ::
            : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void named_barrier(uint32_t id, uint32_t count) {
#if PEAKGEMM_ARCH_SM100A
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void mbarrier_init(uint64_t *barrier, uint32_t count) {
#if PEAKGEMM_ARCH_SM100A
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;" ::
            "r"(address),
        "r"(count) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void mbarrier_init_fence() {
#if PEAKGEMM_ARCH_SM100A
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void mbarrier_wait(
    const uint64_t *barrier,
    uint32_t phase) {
#if PEAKGEMM_ARCH_SM100A
    const uint32_t address = shared_address(barrier);
    constexpr uint32_t ticks = 0x989680U;
    asm volatile(
        "{\n\t"
        ".reg .pred complete;\n\t"
        "WAIT%=:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
        "complete, [%0], %1, %2;\n\t"
        "@complete bra.uni DONE%=;\n\t"
        "bra.uni WAIT%=;\n\t"
        "DONE%=:\n\t"
        "}" ::
            "r"(address),
        "r"(phase),
        "r"(ticks) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void mbarrier_arrive_expect_tx(
    uint64_t *barrier,
    uint32_t bytes) {
#if PEAKGEMM_ARCH_SM100A
    const uint32_t address =
        shared_address(barrier) & PeerBarrierMask;
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 "
        "_, [%0], %1;" ::
            "r"(address),
        "r"(bytes) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void tma_load_2d_2cta(
    void *destination,
    const CUtensorMap *tensor_map,
    int32_t coordinate0,
    int32_t coordinate1,
    uint64_t *barrier) {
#if PEAKGEMM_ARCH_SM100A
    const uint32_t destination_address = shared_address(destination);
    const uint32_t barrier_address =
        shared_address(barrier) & PeerBarrierMask;
    asm volatile(
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4}], [%2];" ::
            "r"(destination_address),
        "l"(tensor_map),
        "r"(barrier_address),
        "r"(coordinate0),
        "r"(coordinate1) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void tma_load_3d_2cta(
    void *destination,
    const CUtensorMap *tensor_map,
    int32_t coordinate0,
    int32_t coordinate1,
    int32_t coordinate2,
    uint64_t *barrier) {
#if PEAKGEMM_ARCH_SM100A
    const uint32_t destination_address = shared_address(destination);
    const uint32_t barrier_address =
        shared_address(barrier) & PeerBarrierMask;
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {%3, %4, %5}], [%2];" ::
            "r"(destination_address),
        "l"(tensor_map),
        "r"(barrier_address),
        "r"(coordinate0),
        "r"(coordinate1),
        "r"(coordinate2) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void tma_store_2d(
    const CUtensorMap *tensor_map,
    int32_t coordinate0,
    int32_t coordinate1,
    const void *source) {
#if PEAKGEMM_ARCH_SM100A
    const uint32_t source_address = shared_address(source);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
        "[%0, {%1, %2}], [%3];" ::
            "l"(tensor_map),
        "r"(coordinate0),
        "r"(coordinate1),
        "r"(source_address) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void tma_store_commit() {
#if PEAKGEMM_ARCH_SM100A
    asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#endif
}

template <int PendingGroups>
PEAKGEMM_DEVICE_INLINE void tma_store_wait() {
#if PEAKGEMM_ARCH_SM100A
    asm volatile(
        "cp.async.bulk.wait_group %0;" ::"n"(PendingGroups) : "memory");
#endif
}

template <int PendingGroups>
PEAKGEMM_DEVICE_INLINE void tma_store_wait_read() {
#if PEAKGEMM_ARCH_SM100A
    asm volatile(
        "cp.async.bulk.wait_group.read %0;" ::"n"(PendingGroups) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void fence_proxy_async_shared() {
#if PEAKGEMM_ARCH_SM100A
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE constexpr uint64_t descriptor_encode(
    uint64_t address) {
    return (address & 0x3FFFFULL) >> 4ULL;
}

PEAKGEMM_DEVICE_INLINE uint64_t make_smem_descriptor(
    const void *matrix) {
    const uint64_t address = shared_address(matrix);
    constexpr uint64_t stride_byte_offset = 8ULL * 128ULL;
    return descriptor_encode(address) | (descriptor_encode(stride_byte_offset) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
}

PEAKGEMM_DEVICE_INLINE void tmem_allocate(
    uint32_t *destination,
    uint32_t columns) {
#if PEAKGEMM_ARCH_SM100A
    const uint32_t address = shared_address(destination);
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 "
        "[%0], %1;" ::
            "r"(address),
        "r"(columns) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void tmem_deallocate(
    uint32_t address,
    uint32_t columns) {
#if PEAKGEMM_ARCH_SM100A
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::
            "r"(address),
        "r"(columns) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void tmem_relinquish() {
#if PEAKGEMM_ARCH_SM100A
    asm volatile(
        "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::
            : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void tcgen05_fence() {
#if PEAKGEMM_ARCH_SM100A
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#endif
}

template <
    typename scalar_t,
    uint32_t M,
    uint32_t N>
PEAKGEMM_DEVICE_INLINE constexpr uint32_t mma_instruction_descriptor() {
    static_assert(M == 128 || M == 256);
    static_assert(N >= 16 && N <= 256 && N % 16 == 0);
    constexpr uint32_t input_type =
        std::is_same_v<scalar_t, __bfloat16> ? 1U : 0U;
    return (1U << 4U) | (input_type << 7U) | (input_type << 10U) | ((N >> 3U) << 17U) | ((M >> 4U) << 24U);
}

PEAKGEMM_DEVICE_INLINE void mma_f16_2cta(
    uint32_t tmem,
    uint64_t descriptor_a,
    uint64_t descriptor_b,
    uint32_t instruction_descriptor,
    bool accumulate) {
#if PEAKGEMM_ARCH_SM100A
    uint32_t masks[8] = {};
    const uint32_t accumulate_value = accumulate;
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 "
        "[%0], %1, %2, %3, "
        "{%5, %6, %7, %8, %9, %10, %11, %12}, p;\n\t"
        "}" ::
            "r"(tmem),
        "l"(descriptor_a),
        "l"(descriptor_b),
        "r"(instruction_descriptor),
        "r"(accumulate_value),
        "r"(masks[0]),
        "r"(masks[1]),
        "r"(masks[2]),
        "r"(masks[3]),
        "r"(masks[4]),
        "r"(masks[5]),
        "r"(masks[6]),
        "r"(masks[7]) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void mma_commit_multicast(
    uint64_t *barrier,
    uint16_t cta_mask) {
#if PEAKGEMM_ARCH_SM100A
    const uint32_t address = shared_address(barrier);
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one."
        "shared::cluster.multicast::cluster.b64 [%0], %1;" ::
            "r"(address),
        "h"(cta_mask) : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE void tmem_load_x16(
    float (&values)[16],
    uint32_t address) {
#if PEAKGEMM_ARCH_SM100A
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15}, [%16];\n\t"
        "tcgen05.wait::ld.sync.aligned;"
        : "=f"(values[0]), "=f"(values[1]),
          "=f"(values[2]), "=f"(values[3]),
          "=f"(values[4]), "=f"(values[5]),
          "=f"(values[6]), "=f"(values[7]),
          "=f"(values[8]), "=f"(values[9]),
          "=f"(values[10]), "=f"(values[11]),
          "=f"(values[12]), "=f"(values[13]),
          "=f"(values[14]), "=f"(values[15])
        : "r"(address)
        : "memory");
#endif
}

PEAKGEMM_DEVICE_INLINE uint32_t swizzle_128b_chunk(
    uint32_t row,
    uint32_t chunk) {
    return row * 8U + (chunk ^ (row & 7U));
}

} // namespace peak_gemm::backend::sm100
