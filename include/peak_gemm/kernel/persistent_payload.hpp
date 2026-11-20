#pragma once

#include <cstddef>

#include "peak_gemm/backend/system_atomic.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::kernel {

// One producer GPU -> one consumer GPU. Allocate on the consumer, enable producer
// P2P access, and zero once before BOTH launches. HIP requires fine-grained memory.
// A slot may be overwritten only after its previous epoch has been consumed.
template <uint32_t WIDTH, uint32_t SLOTS = 2>
struct alignas(64) PayloadChannel {
    static_assert(WIDTH > 0 && WIDTH % 4 == 0, "Payload rows must be 16-byte aligned");
    static_assert(SLOTS > 0);
    alignas(16) float payload[SLOTS][WIDTH];
    alignas(64) uint32_t ready[SLOTS];
    alignas(64) uint32_t consumed[SLOTS];
    alignas(64) uint32_t cancelled;
};

// Collective over the CTA. Only the leader polls. Conservatively let every
// participant acquire the publication directly before accessing the slot;
// this does not rely on forwarding a leader-only acquire through the CTA barrier.
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

// All payload-writing/reading threads participate. Drain every thread's accesses
// before the leader publishes; a leader-only fence cannot drain other writers.
PEAKGEMM_DEVICE_INLINE void payload_publish(uint32_t *flag, uint32_t epoch) {
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) backend::store_release_system(flag, epoch);
    __syncthreads();
}

// Teaching example, NOT an MoE/all-to-all or a throughput-tuned transport.
// Launch exactly ONE 1D CTA on each GPU, on separate device streams:
//   producer: input [tiles, WIDTH] local to producer, output == nullptr;
//   consumer: input == nullptr, output [tiles, WIDTH] local to consumer.
// Inputs/outputs are 16B aligned and do not alias channel storage.
// first_epoch starts at 1, advances by tiles per invocation, and must not wrap.
// Relaunch/reset/free only after BOTH kernels finish; do not overlap invocations.
// max_polls bounds a missing/slow peer (not a wall-clock timeout); on cancellation,
// discard output, synchronize both GPUs, then reset the channel before retrying.
template <bool PRODUCER, uint32_t WIDTH, uint32_t SLOTS = 2>
__global__ void persistent_payload_kernel(
    const float *input, float *output, PayloadChannel<WIDTH, SLOTS> *channel,
    uint32_t tiles, uint32_t first_epoch, uint32_t max_polls) {
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        const uint32_t epoch = first_epoch + tile;
        const uint32_t slot = (epoch - 1) % SLOTS;
        if constexpr (PRODUCER) {
            // 1. Backpressure: don't overwrite data the peer is still reading.
            const uint32_t previous = epoch > SLOTS ? epoch - SLOTS : 0;
            if (!payload_wait(&channel->consumed[slot], previous, &channel->cancelled, max_polls)) return;
            // 2. Vector stores target the consumer's device memory over P2P.
            for (uint32_t h = threadIdx.x * 4; h < WIDTH; h += blockDim.x * 4) {
                core::Vector<float, 4> values;
                values.load(input + static_cast<std::size_t>(tile) * WIDTH + h);
                values.store(channel->payload[slot] + h);
            }
            // 3. Publish ready only AFTER the entire payload is visible.
            payload_publish(&channel->ready[slot], epoch);
        } else {
            // 4. Acquire ready BEFORE reading payload. No host event per tile.
            if (!payload_wait(&channel->ready[slot], epoch, &channel->cancelled, max_polls)) return;
            for (uint32_t h = threadIdx.x * 4; h < WIDTH; h += blockDim.x * 4) {
                core::Vector<float, 4> values;
                values.load(channel->payload[slot] + h);
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) values[j] += 1.0F; // Stand-in for tile compute.
                values.store(output + static_cast<std::size_t>(tile) * WIDTH + h);
            }
            // 5. Release the slot only AFTER all threads have consumed it.
            payload_publish(&channel->consumed[slot], epoch);
        }
    }
}

} // namespace peak_gemm::kernel
