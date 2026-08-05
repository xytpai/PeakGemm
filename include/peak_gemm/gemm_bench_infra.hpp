#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "peak_gemm/backend/runtime.hpp"
#include "peak_gemm/data.hpp"
#include "peak_gemm/kernel/gemm_naive.hpp"

namespace peak_gemm::bench {

inline constexpr auto cpu = DataDevice::cpu;
inline constexpr auto gpu = DataDevice::gpu;
using GemmShape = std::array<uint32_t, 3>;

struct GemmBenchResult {
    double median_ms;
    double tflops;
};

struct DefaultCpuGemmReference {
    template <typename scalar_t>
    void operator()(
        const scalar_t *a,
        const scalar_t *b,
        scalar_t *c,
        uint32_t m_size,
        uint32_t n_size,
        uint32_t k_size,
        const scalar_t *bias) const {
        for (uint32_t m = 0; m < m_size; ++m) {
            for (uint32_t n = 0; n < n_size; ++n) {
                float value = bias == nullptr ? 0.0F : static_cast<float>(bias[n]);
                for (uint32_t k = 0; k < k_size; ++k) {
                    value += static_cast<float>(a[m * k_size + k]) * static_cast<float>(b[n * k_size + k]);
                }
                c[m * n_size + n] = static_cast<scalar_t>(value);
            }
        }
    }
};

struct GpuNaiveGemmReference {
    template <typename scalar_t>
    void operator()(
        const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t m,
        uint32_t n, uint32_t k, const scalar_t *bias) const {
        using DataT = Data<scalar_t>;
        DataT host_a({m, k}), host_b({n, k}), host_bias({n});
        std::copy_n(a, host_a.size(), host_a.data());
        std::copy_n(b, host_b.size(), host_b.data());
        if (bias != nullptr) {
            std::copy_n(bias, host_bias.size(), host_bias.data());
        } else {
            host_bias.fill(static_cast<scalar_t>(0));
        }
        auto device_a = host_a.copy_to(gpu);
        auto device_b = host_b.copy_to(gpu);
        auto device_bias = host_bias.copy_to(gpu);
        DataT device_c({m, n}, gpu);
        peak_gemm::kernel::gemm_naive_gpu(
            device_a.data(), device_b.data(), device_c.data(), m, n, k,
            device_bias.data());
        gpuDeviceSynchronize();
        const auto result = device_c.copy_to(cpu);
        std::copy_n(result.data(), result.size(), c);
    }
};

template <typename scalar_t>
constexpr float default_base_tolerance();

template <>
constexpr float default_base_tolerance<float>() {
    return 2.0e-6F;
}

template <>
constexpr float default_base_tolerance<__half>() {
    return 2.0e-3F;
}

template <>
constexpr float default_base_tolerance<__bfloat16>() {
    return 2.0e-2F;
}

template <typename scalar_t>
constexpr const char *default_type_name();

template <>
constexpr const char *default_type_name<float>() {
    return "fp32";
}

template <>
constexpr const char *default_type_name<__half>() {
    return "fp16";
}

template <>
constexpr const char *default_type_name<__bfloat16>() {
    return "bf16";
}

template <typename scalar_t>
class GemmBench {
public:
    using DataT = Data<scalar_t>;

    GemmBench(uint32_t m, uint32_t n, uint32_t k, bool use_bias = true) :
        m_(m),
        n_(n),
        k_(k),
        use_bias_(use_bias),
        a_(DataT::uniform({m, k}, -1.0F, 1.0F, cpu, 2026)),
        b_(DataT::uniform({n, k}, -1.0F, 1.0F, cpu, 2027)),
        bias_(DataT::uniform({n}, 10.0F, 20.0F, cpu, 2028)),
        device_a_(a_.copy_to(gpu)),
        device_b_(b_.copy_to(gpu)),
        device_bias_(bias_.copy_to(gpu)),
        device_c_({m, n}, gpu) {
    }

    template <typename Launch, typename Reference>
    float check_accuracy(const Launch &launch, const Reference &reference) {
        launch_kernel(launch);
        gpuDeviceSynchronize();
        auto actual = device_c_.copy_to(cpu);
        DataT expected({m_, n_}, cpu);
        reference(a_.data(), b_.data(), expected.data(), m_, n_, k_, use_bias_ ? bias_.data() : nullptr);
        float max_diff = 0.0F;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const float diff = std::abs(static_cast<float>(actual[i]) - static_cast<float>(expected[i]));
            if (!std::isfinite(diff)) {
                return std::numeric_limits<float>::infinity();
            }
            max_diff = std::max(max_diff, diff);
        }
        return max_diff;
    }

    template <typename Launch>
    GemmBenchResult benchmark(const Launch &launch, uint32_t warmup, uint32_t measurements, uint32_t iterations) {
        for (uint32_t i = 0; i < warmup; ++i)
            launch_kernel(launch);
        gpuDeviceSynchronize();
        gpuEvent_t start, stop;
        gpuEventCreate(&start);
        gpuEventCreate(&stop);
        std::vector<double> latencies;
        latencies.reserve(measurements);
        for (uint32_t measurement = 0; measurement < measurements; ++measurement) {
            gpuEventRecord(start);
            for (uint32_t i = 0; i < iterations; ++i) {
                launch_kernel(launch);
            }
            gpuEventRecord(stop);
            gpuEventSynchronize(stop);
            float elapsed_ms = 0.0F;
            gpuEventElapsedTime(&elapsed_ms, start, stop);
            latencies.push_back(elapsed_ms / iterations);
        }
        gpuEventDestroy(start);
        gpuEventDestroy(stop);
        std::sort(latencies.begin(), latencies.end());
        const std::size_t middle = latencies.size() / 2;
        const double median_ms = latencies.size() % 2 == 0 ? (latencies[middle - 1] + latencies[middle]) / 2.0 : latencies[middle];
        const double tflops = 2.0 * m_ * n_ * k_ / (median_ms * 1.0e9);
        return {median_ms, tflops};
    }

    bool use_bias() const {
        return use_bias_;
    }

    float tolerance(float base_tolerance) const {
        float scale = std::sqrt(static_cast<float>(k_));
        if (use_bias_) {
            scale += 80.0F;
        }
        return base_tolerance * scale;
    }

private:
    template <typename Launch>
    void launch_kernel(const Launch &launch) {
        launch(device_a_.data(), device_b_.data(), device_c_.data(), m_, n_, k_, use_bias_ ? device_bias_.data() : nullptr);
    }

    uint32_t m_;
    uint32_t n_;
    uint32_t k_;
    bool use_bias_;
    DataT a_;
    DataT b_;
    DataT bias_;
    DataT device_a_;
    DataT device_b_;
    DataT device_bias_;
    DataT device_c_;
};

template <typename scalar_t, typename Launch, typename Reference>
void run(
    const char *type_name,
    float base_tolerance,
    const std::vector<GemmShape> &shapes,
    const Launch &launch,
    const Reference &reference,
    bool use_bias = true,
    uint32_t warmup = 10,
    uint32_t measurements = 8,
    uint32_t iterations = 20) {
    std::cout << "\n[" << type_name << "]\n"
              << std::right << std::setw(8) << "m"
              << std::setw(8) << "n"
              << std::setw(8) << "k"
              << std::setw(8) << "bias"
              << std::setw(8) << "acc"
              << std::setw(16) << "tol"
              << std::setw(16) << "max_diff"
              << std::setw(16) << "median_ms"
              << std::setw(16) << "tflops" << '\n';
    bool all_passed = true;
    for (const auto &[m, n, k] : shapes) {
        GemmBench<scalar_t> bench(m, n, k, use_bias);
        const float max_diff = bench.check_accuracy(launch, reference);
        const float tolerance = bench.tolerance(base_tolerance);
        const bool passed = max_diff <= tolerance;
        all_passed &= passed;
        const auto result = bench.benchmark(launch, warmup, measurements, iterations);
        std::cout << std::setw(8) << m
                  << std::setw(8) << n
                  << std::setw(8) << k
                  << std::setw(8) << bench.use_bias()
                  << std::setw(8) << (passed ? "pass" : "fail")
                  << std::scientific << std::setprecision(4)
                  << std::setw(16) << tolerance
                  << std::setw(16) << max_diff
                  << std::fixed << std::setprecision(6)
                  << std::setw(16) << result.median_ms
                  << std::defaultfloat << std::setprecision(6)
                  << std::setw(16) << result.tflops << '\n';
    }
    if (!all_passed) {
        throw std::runtime_error("GEMM accuracy check failed");
    }
}

template <typename scalar_t, typename Launch, typename Reference>
void run(
    const char *type_name,
    const std::vector<GemmShape> &shapes,
    const Launch &launch,
    const Reference &reference,
    bool use_bias = true,
    uint32_t warmup = 10,
    uint32_t measurements = 8,
    uint32_t iterations = 20) {
    run<scalar_t>(
        type_name,
        default_base_tolerance<scalar_t>(),
        shapes,
        launch,
        reference,
        use_bias,
        warmup,
        measurements,
        iterations);
}

template <typename scalar_t, typename Launch, typename Reference>
void run(
    const std::vector<GemmShape> &shapes,
    const Launch &launch,
    const Reference &reference,
    bool use_bias = true,
    uint32_t warmup = 10,
    uint32_t measurements = 8,
    uint32_t iterations = 20) {
    run<scalar_t>(
        default_type_name<scalar_t>(),
        default_base_tolerance<scalar_t>(),
        shapes,
        launch,
        reference,
        use_bias,
        warmup,
        measurements,
        iterations);
}

} // namespace peak_gemm::bench
