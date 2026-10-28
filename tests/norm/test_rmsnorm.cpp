#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "peak_gemm/data.hpp"
#include "peak_gemm/kernel/rmsnorm.hpp"

using peak_gemm::Data;
constexpr auto cpu = peak_gemm::DataDevice::cpu;
constexpr auto gpu = peak_gemm::DataDevice::gpu;

void check_gpu() {
    auto err = gpuGetLastError();
    if (err != gpuSuccess) {
        throw std::runtime_error(gpuGetErrorString(err));
    }
}

void close(float got, float ref, float tolerance, const char *what) {
    if (!std::isfinite(got) || std::abs(got - ref) > tolerance * (1.0F + std::abs(ref))) {
        throw std::runtime_error(std::string(what) + " mismatch: got " +
                                 std::to_string(got) + ", ref " + std::to_string(ref));
    }
}

template <typename T, typename W>
void test(int rows, int cols) {
    constexpr float eps = 1e-6F;
    constexpr float tol = std::is_same_v<T, float> ? 3e-4F : 1e-2F;
    auto hx = Data<T>::uniform({size_t(rows), size_t(cols)}, -1.0F, 1.0F, cpu, 41);
    auto hw = Data<W>::uniform({size_t(cols)}, -1.0F, 1.0F, cpu, 42);
    auto hg = Data<T>::uniform({size_t(rows), size_t(cols)}, -1.0F, 1.0F, cpu, 43);
    auto x = hx.copy_to(gpu);
    auto w = hw.copy_to(gpu);
    auto g = hg.copy_to(gpu);
    Data<T> y({size_t(rows), size_t(cols)}, gpu), dx({size_t(rows), size_t(cols)}, gpu);
    Data<W> dw({size_t(cols)}, gpu);
    Data<float> inv({size_t(rows)}, gpu);
    const int parts = peak_gemm::kernel::rmsnorm_backward_partitions(rows);
    Data<float> partial({size_t(parts), size_t(cols)}, gpu);
    gpuStream_t stream;
    if (gpuStreamCreate(&stream) != gpuSuccess) {
        throw std::runtime_error("stream creation failed");
    }
    peak_gemm::kernel::rmsnorm_forward(x.data(), w.data(), y.data(), inv.data(), rows, cols, eps, stream);
    check_gpu();
    peak_gemm::kernel::rmsnorm_backward(g.data(), x.data(), w.data(), inv.data(),
                                      dx.data(), dw.data(), partial.data(), rows, cols, stream);
    check_gpu();
    auto status = gpuStreamSynchronize(stream);
    auto destroy_status = gpuStreamDestroy(stream);
    if (destroy_status != gpuSuccess) {
        throw std::runtime_error(gpuGetErrorString(destroy_status));
    }
    if (status != gpuSuccess) {
        throw std::runtime_error(gpuGetErrorString(status));
    }
    auto hy = y.copy_to(cpu);
    auto hdx = dx.copy_to(cpu);
    auto hdw = dw.copy_to(cpu);
    auto hi = inv.copy_to(cpu);
    std::vector<double> refdw(cols, 0.0);
    for (int row = 0; row < rows; ++row) {
        double square = 0.0, dot = 0.0;
        for (int col = 0; col < cols; ++col) {
            const int idx = row * cols + col;
            square += double(float(hx[idx])) * float(hx[idx]);
            dot += double(float(hg[idx])) * float(hw[col]) * float(hx[idx]);
        }
        const double r = 1.0 / std::sqrt(square / cols + eps);
        close(hi[row], float(r), 3e-5F, "rstd");
        for (int col = 0; col < cols; ++col) {
            const int idx = row * cols + col;
            const double value = float(hx[idx]);
            close(float(hy[idx]), float(T(float(value * r * float(hw[col])))), tol, "y");
            close(float(hdx[idx]), float(T(float(r * (float(hg[idx]) * float(hw[col]) -
                  value * dot / cols * r * r)))), tol, "dx");
            refdw[col] += float(hg[idx]) * value * r;
        }
    }
    for (int col = 0; col < cols; ++col) {
        close(float(hdw[col]), float(W(float(refdw[col]))), 1e-2F, "dw");
    }
    // The inference entry also accepts no rstd output.
    peak_gemm::kernel::rmsnorm_forward(x.data(), w.data(), y.data(), nullptr, rows, cols, eps);
    check_gpu();
}

int main() {
    const int cols[] = {1, 31, 64, 127, 128, 255, 256, 257, 1024, 1025,
                       2048, 2049, 4096, 4097, 6144, 8192, 8193, 16385};
    for (int n : cols) {
        test<float, float>(17, n);
        test<__half, __half>(17, n);
        test<__bfloat16, __bfloat16>(17, n);
        test<__bfloat16, float>(17, n);
    }
    // Data deliberately rejects empty extents; test the raw zero-row API.
    Data<float> empty_dw({128}, gpu);
    Data<float> empty_partial({128}, gpu);
    peak_gemm::kernel::rmsnorm_forward<float, float>(
        nullptr, nullptr, nullptr, nullptr, 0, 128, 1e-6F);
    peak_gemm::kernel::rmsnorm_backward<float, float>(
        nullptr, nullptr, nullptr, nullptr, nullptr, empty_dw.data(),
        empty_partial.data(), 0, 128);
    auto empty_host = empty_dw.copy_to(cpu);
    for (int col = 0; col < 128; ++col) {
        close(empty_host[col], 0.0F, 0.0F, "empty dw");
    }
    test<float, float>(4097, 128);
    if (gpuDeviceSynchronize() != gpuSuccess) {
        throw std::runtime_error("GPU synchronization failed");
    }
    std::cout << "RMSNorm forward/backward: 74 CUDA/HIP cases passed\n";
}
