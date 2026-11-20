#include <algorithm>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

#include "peak_gemm/data.hpp"
#include "peak_gemm/kernel/persistent_payload.hpp"

using peak_gemm::Data;
constexpr auto cpu = peak_gemm::DataDevice::cpu;
constexpr auto gpu = peak_gemm::DataDevice::gpu;

template <typename Error>
void check_gpu(Error error) {
    if (error != gpuSuccess) throw std::runtime_error(gpuGetErrorString(error));
}

bool enable_channel_access(int producer, int consumer) {
    int access = 0, native_atomics = 0;
    check_gpu(gpuDeviceCanAccessPeer(&access, producer, consumer));
    if (!access) return false;
#if defined(__HIPCC__)
    int fine_grained = 0;
    check_gpu(hipDeviceGetAttribute(&fine_grained, hipDeviceAttributeFineGrainSupport, consumer));
    if (!fine_grained) return false;
    check_gpu(hipDeviceGetP2PAttribute(&native_atomics, hipDevP2PAttrNativeAtomicSupported, producer, consumer));
#else
    check_gpu(cudaDeviceGetP2PAttribute(&native_atomics, cudaDevP2PAttrNativeAtomicSupported, producer, consumer));
#endif
    if (!native_atomics) return false;
    check_gpu(gpuSetDevice(producer));
    auto error = gpuDeviceEnablePeerAccess(consumer, 0);
#if defined(__HIPCC__)
    if (error == hipErrorPeerAccessAlreadyEnabled) (void)gpuGetLastError();
#else
    if (error == cudaErrorPeerAccessAlreadyEnabled) (void)gpuGetLastError();
#endif
    else
        check_gpu(error);
    return true;
}

// Only the communication channel needs special allocation. Inputs/outputs reuse Data.
template <typename Channel>
struct ChannelStorage {
    Channel *pointer = nullptr;
    int device;

    explicit ChannelStorage(int device_index) : device(device_index) {
        check_gpu(gpuSetDevice(device));
#if defined(__HIPCC__)
        check_gpu(hipExtMallocWithFlags(reinterpret_cast<void **>(&pointer), sizeof(Channel), hipDeviceMallocFinegrained));
#else
        check_gpu(gpuMalloc(&pointer, sizeof(Channel)));
#endif
        check_gpu(gpuMemset(pointer, 0, sizeof(Channel)));
    }

    ~ChannelStorage() {
        (void)gpuSetDevice(device);
        (void)gpuFree(pointer);
    }

    ChannelStorage(const ChannelStorage &) = delete;
    ChannelStorage &operator=(const ChannelStorage &) = delete;

    Channel snapshot() const {
        Channel result;
        check_gpu(gpuSetDevice(device));
        check_gpu(gpuMemcpy(&result, pointer, sizeof(Channel), gpuMemcpyDeviceToHost));
        return result;
    }
};

template <uint32_t WIDTH, uint32_t SLOTS>
void check_channel(int producer, int consumer) {
    using Channel = peak_gemm::kernel::PayloadChannel<WIDTH, SLOTS>;
    ChannelStorage<Channel> storage(consumer);
    gpuStream_t send_stream, recv_stream;
    check_gpu(gpuSetDevice(producer));
    check_gpu(gpuStreamCreate(&send_stream));
    check_gpu(gpuSetDevice(consumer));
    check_gpu(gpuStreamCreate(&recv_stream));
    uint32_t epoch = 1;
    uint32_t expected[SLOTS]{};
    uint32_t invocation = 0;
    // Reuse without zeroing flags: empty call, partial ring, wraps, many wraps.
    for (uint32_t tiles : {0U, 1U, SLOTS, SLOTS + 1, 257U, 3U}) {
        const uint32_t rows = std::max(tiles, 1U);
        auto host = Data<float>::uniform({rows, WIDTH}, -10.0F, 10.0F, cpu, 2026 + invocation);
        auto input = host.copy_to(gpu, producer);
        Data<float> output({static_cast<std::size_t>(rows) * WIDTH + 8}, gpu, consumer);
        output.fill(-12345.0F);
        constexpr uint32_t MAX_POLLS = 10000000;
        auto send = [&] {
            check_gpu(gpuSetDevice(producer));
            peak_gemm::kernel::persistent_payload_kernel<true, WIDTH, SLOTS><<<1, 256, 0, send_stream>>>(
                input.data(), nullptr, storage.pointer, tiles, epoch, MAX_POLLS);
            check_gpu(gpuGetLastError());
        };
        auto recv = [&] {
            check_gpu(gpuSetDevice(consumer));
            peak_gemm::kernel::persistent_payload_kernel<false, WIDTH, SLOTS><<<1, 256, 0, recv_stream>>>(
                nullptr, output.data() + 4, storage.pointer, tiles, epoch, MAX_POLLS);
            check_gpu(gpuGetLastError());
        };
        // Skew either endpoint. Never synchronize a waiting kernel before launching its peer.
        if (invocation % 2 == 0) {
            send();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            recv();
        } else {
            recv();
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            send();
        }
        check_gpu(gpuSetDevice(producer));
        check_gpu(gpuStreamSynchronize(send_stream));
        check_gpu(gpuSetDevice(consumer));
        check_gpu(gpuStreamSynchronize(recv_stream));
        const auto state = storage.snapshot();
        if (state.cancelled) throw std::runtime_error("Payload channel timed out/cancelled");
        const auto actual = output.copy_to(cpu);
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const bool valid = i >= 4 && i - 4 < static_cast<std::size_t>(tiles) * WIDTH;
            const float value = valid ? host[i - 4] + 1.0F : -12345.0F;
            if (actual[i] != value) throw std::runtime_error("Stale/partial payload or output guard overwritten");
        }
        for (uint32_t t = 0; t < tiles; ++t) expected[(epoch + t) % SLOTS] = epoch + t;
        for (uint32_t s = 0; s < SLOTS; ++s) {
            if (state.ready[s] != expected[s] || state.consumed[s] != expected[s]) {
                throw std::runtime_error("Payload epoch/ack mismatch");
            }
        }
        epoch += tiles;
        ++invocation;
    }
    check_gpu(gpuSetDevice(producer));
    check_gpu(gpuStreamDestroy(send_stream));
    check_gpu(gpuSetDevice(consumer));
    check_gpu(gpuStreamDestroy(recv_stream));
    std::cout << "[pass] GPU" << producer << " -> GPU" << consumer << " WIDTH=" << WIDTH << " SLOTS=" << SLOTS
              << " repeated launches / delayed peers / slot reuse\n";
}

void check_missing_peer(int producer, int consumer) {
    using Channel = peak_gemm::kernel::PayloadChannel<5120>;
    ChannelStorage<Channel> storage(consumer);
    Data<float> input({3, 5120}, gpu, producer), output({3, 5120}, gpu, consumer);
    input.fill(1.0F);
    // Epochs 1/2 fill slots 1/0; epoch 3 must wait for slot 1's ack.
    check_gpu(gpuSetDevice(producer));
    peak_gemm::kernel::persistent_payload_kernel<true, 5120><<<1, 256>>>(
        input.data(), nullptr, storage.pointer, 3, 1, 128);
    check_gpu(gpuGetLastError());
    check_gpu(gpuDeviceSynchronize());
    auto state = storage.snapshot();
    if (!state.cancelled || state.ready[0] != 2 || state.ready[1] != 1 || state.consumed[0] != 0 || state.consumed[1] != 0) {
        throw std::runtime_error("Producer backpressure/cancellation failed");
    }
    // A late consumer observes cancellation and must not consume partial/stale work.
    output.fill(-12345.0F);
    peak_gemm::kernel::persistent_payload_kernel<false, 5120><<<1, 256>>>(
        nullptr, output.data(), storage.pointer, 3, 1, 128);
    check_gpu(gpuGetLastError());
    check_gpu(gpuDeviceSynchronize());
    const auto unchanged = output.copy_to(cpu);
    for (std::size_t i = 0; i < unchanged.size(); ++i) {
        if (unchanged[i] != -12345.0F) throw std::runtime_error("Consumer ignored cancellation");
    }
    // Reset only with both kernels finished. A consumer with no sender must also exit.
    check_gpu(gpuSetDevice(consumer));
    check_gpu(gpuMemset(storage.pointer, 0, sizeof(Channel)));
    peak_gemm::kernel::persistent_payload_kernel<false, 5120><<<1, 256>>>(
        nullptr, output.data(), storage.pointer, 1, 1, 128);
    check_gpu(gpuGetLastError());
    check_gpu(gpuDeviceSynchronize());
    state = storage.snapshot();
    if (!state.cancelled || state.consumed[1] != 0) throw std::runtime_error("Consumer cancellation failed");
    std::cout << "[pass] missing peer / backpressure / cancellation\n";
}

int main() {
    int count = 0, original = 0;
    check_gpu(gpuGetDeviceCount(&count));
    if (count < 2) {
        std::cout << "[skip] persistent payload demo requires two GPUs\n";
        return 0;
    }
    check_gpu(gpuGetDevice(&original));
    if (!enable_channel_access(0, 1) || !enable_channel_access(1, 0)) {
        std::cout << "[skip] requires P2P native atomics and coherent device memory\n";
        check_gpu(gpuSetDevice(original));
        return 0;
    }
    check_channel<4, 2>(0, 1);
    check_channel<5120, 1>(0, 1);
    check_channel<5120, 2>(0, 1);
    check_channel<5120, 2>(1, 0);
    check_channel<5120, 3>(0, 1);
    check_channel<8200, 2>(0, 1);
    check_missing_peer(0, 1);
    check_gpu(gpuSetDevice(original));
    std::cout << "ok\n";
}
