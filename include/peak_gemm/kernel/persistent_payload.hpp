#pragma once

#include <cstddef>

#include "peak_gemm/backend/system_atomic.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::kernel {

// SPSC channel on the consumer GPU; zero before use and enable producer P2P access.
// HIP requires fine-grained memory. Epoch 0 means unused.
template <uint32_t WIDTH, uint32_t SLOTS = 2>
struct alignas(64) PayloadChannel {
    static_assert(WIDTH > 0 && WIDTH % 4 == 0, "Payload rows must be 16-byte aligned");
    static_assert(SLOTS > 0);
    alignas(16) float payload[SLOTS][WIDTH];
    alignas(64) uint32_t ready[SLOTS];
    alignas(64) uint32_t consumed[SLOTS];
    alignas(64) uint32_t cancelled;
};

// CTA-wide wait: leader polls; all threads acquire on success (conservative).
PEAKGEMM_DEVICE_INLINE bool payload_wait(uint32_t *flag, uint32_t epoch, uint32_t *cancelled, uint32_t max_polls) {
    __shared__ uint32_t success;
    if (threadIdx.x == 0) {
        success = 0;
        for (uint32_t poll = 0; poll < max_polls; ++poll) {
            if (backend::load_acquire_system(cancelled)) break;
            if (backend::load_acquire_system(flag) == epoch) {
                success = 1;
                break;
            }
        }
        if (!success) backend::store_release_system(cancelled, 1U);
    }
    __syncthreads();
    const bool acquired = success != 0;
    if (acquired) (void)backend::load_acquire_system(flag);
    __syncthreads();
    return acquired;
}

// All threads fence before the leader publishes completion.
PEAKGEMM_DEVICE_INLINE void payload_publish(uint32_t *flag, uint32_t epoch) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) backend::store_release_system(flag, epoch);
    __syncthreads();
}

// Demo: one 1D CTA per GPU. Launch both endpoints before waiting for either.
// Producer input / consumer output: [tiles, WIDTH], 16B aligned, no aliasing.
// Epochs start at 1, continue across calls, and must not wrap.
// Both kernels must finish before relaunch/reset/free. Cancellation invalidates
// output; reset before retrying. max_polls is an iteration budget, not a timer.
template <bool PRODUCER, uint32_t WIDTH, uint32_t SLOTS = 2>
__global__ void persistent_payload_kernel(
    const float *input, float *output, PayloadChannel<WIDTH, SLOTS> *channel,
    uint32_t tiles, uint32_t first_epoch, uint32_t max_polls) {
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        const uint32_t epoch = first_epoch + tile;
        const uint32_t slot = epoch % SLOTS;
        if constexpr (PRODUCER) {
            // Wait until this slot's previous epoch is consumed.
            const uint32_t previous = epoch > SLOTS ? epoch - SLOTS : 0;
            if (!payload_wait(&channel->consumed[slot], previous, &channel->cancelled, max_polls)) return;
            for (uint32_t h = threadIdx.x * 4; h < WIDTH; h += blockDim.x * 4) {
                core::Vector<float, 4> values;
                values.load(input + static_cast<std::size_t>(tile) * WIDTH + h);
                values.store(channel->payload[slot] + h);
            }
            // Publish after all payload stores.
            payload_publish(&channel->ready[slot], epoch);
        } else {
            // Acquire before reading payload.
            if (!payload_wait(&channel->ready[slot], epoch, &channel->cancelled, max_polls)) return;
            for (uint32_t h = threadIdx.x * 4; h < WIDTH; h += blockDim.x * 4) {
                core::Vector<float, 4> values;
                values.load(channel->payload[slot] + h);
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) values[j] += 1.0F; // Demo compute.
                values.store(output + static_cast<std::size_t>(tile) * WIDTH + h);
            }
            // Allow reuse after all payload reads.
            payload_publish(&channel->consumed[slot], epoch);
        }
    }
}

} // namespace peak_gemm::kernel
