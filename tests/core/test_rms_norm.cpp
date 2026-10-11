#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include <type_traits>

#include "peak_gemm/data.hpp"
#include "peak_gemm/kernel/rms_norm.hpp"

using peak_gemm::Data;
namespace kernel = peak_gemm::kernel;
constexpr auto cpu = peak_gemm::DataDevice::cpu;
constexpr auto gpu = peak_gemm::DataDevice::gpu;

template <typename T>
void check_close(const Data<T> &actual, const std::vector<double> &expected, const char *name) {
    const double tolerance = std::is_same_v<T, __bfloat16> ? 1e-2 : (std::is_same_v<T, __half> ? 2e-3 : 2e-5);
    double max_diff = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double value = static_cast<float>(actual[i]);
        const double diff = std::abs(value - expected[i]);
        max_diff = std::max(max_diff, diff);
        if (!std::isfinite(value) || diff > tolerance * (1 + std::abs(expected[i]))) {
            std::cerr << name << "[" << i << "]: " << value << " != " << expected[i] << '\n';
            assert(false);
        }
    }
    std::cout << " " << name << "_max_diff=" << max_diff;
}

template <typename input_t, typename weight_t>
void test_case(uint32_t rows, uint32_t dim, float eps, bool zeros = false, float scale = 1.0F, float shift = 0.0F) {
    auto x = Data<input_t>::uniform({rows, dim}, -2, 2, cpu, 2026);
    auto weight = Data<weight_t>::uniform({dim}, -1, 1, cpu, 2027);
    auto dy = Data<input_t>::uniform({rows, dim}, -1, 1, cpu, 2028);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<input_t>(static_cast<float>(x[i]) * scale + shift);
    }
    if (zeros) x.fill(static_cast<input_t>(0.0F));
    std::vector<double> expected_y(x.size()), expected_dx(x.size()), expected_dw(dim, 0), expected_rstd(rows);
    for (uint32_t row = 0; row < rows; ++row) {
        double sum = 0, dot = 0;
        for (uint32_t col = 0; col < dim; ++col) {
            const auto index = static_cast<std::size_t>(row) * dim + col;
            const double value = static_cast<float>(x[index]);
            sum += value * value;
            dot += static_cast<float>(dy[index]) * static_cast<double>(static_cast<float>(weight[col])) * value;
        }
        const double r = 1 / std::sqrt(sum / dim + eps);
        expected_rstd[row] = r;
        for (uint32_t col = 0; col < dim; ++col) {
            const auto index = static_cast<std::size_t>(row) * dim + col;
            const double value = static_cast<float>(x[index]);
            const double grad = static_cast<float>(dy[index]);
            const double w = static_cast<float>(weight[col]);
            expected_y[index] = w * value * r;
            expected_dx[index] = r * (grad * w - value * r * r * dot / dim);
            expected_dw[col] += grad * value * r;
        }
    }

    auto device_x = x.copy_to(gpu), device_dy = dy.copy_to(gpu);
    auto device_weight = weight.copy_to(gpu);
    Data<input_t> y({rows, dim}, gpu), dx({rows, dim}, gpu);
    Data<weight_t> dw({dim}, gpu);
    Data<float> rstd({rows}, gpu), partial_dw({kernel::rms_norm_groups(rows, dim), dim}, gpu);
    gpuStream_t stream;
    assert(gpuStreamCreate(&stream) == gpuSuccess);
    // Repeated calls must overwrite scratch/output, not accumulate stale values.
    for (int repeat = 0; repeat < 3; ++repeat) {
        kernel::rms_norm_forward(device_x.data(), device_weight.data(), y.data(), rstd.data(), rows, dim, eps, stream);
        kernel::rms_norm_backward(device_dy.data(), device_x.data(), device_weight.data(), rstd.data(),
                                  dx.data(), dw.data(), partial_dw.data(), rows, dim, stream);
    }
    assert(gpuGetLastError() == gpuSuccess);
    assert(gpuStreamSynchronize(stream) == gpuSuccess);
    assert(gpuStreamDestroy(stream) == gpuSuccess);
    std::cout << "rows=" << rows << " dim=" << dim << " eps=" << eps;
    check_close(y.copy_to(cpu), expected_y, "y");
    check_close(rstd.copy_to(cpu), expected_rstd, "rstd");
    check_close(dx.copy_to(cpu), expected_dx, "dx");
    check_close(dw.copy_to(cpu), expected_dw, "dw");
    std::cout << '\n';
}

template <typename input_t, typename weight_t>
void test_types(const char *name) {
    std::cout << "[test] " << name << '\n';
    for (const uint32_t dim : {8U, 24U, 32U, 56U, 64U, 72U, 248U, 256U, 264U, 1024U, 4104U, 8192U, 65536U}) {
        test_case<input_t, weight_t>(3, dim, 1e-6F);
    }
    test_case<input_t, weight_t>(513, 264, 1e-5F);
    test_case<input_t, weight_t>(8193, 128, 1e-6F);
    test_case<input_t, weight_t>(3, 4096, 1e-6F, false, 100.0F, 10000.0F);
    test_case<input_t, weight_t>(3, 4096, 1e-6F, false, 1e-5F, 1e-4F);
    test_case<input_t, weight_t>(1025, 1024, 0.1F);
    test_case<input_t, weight_t>(3, 128, 1e-6F, true);
    test_case<input_t, weight_t>(3, 64, 0.0F);
}


template <typename Function>
void expect_invalid(Function function) {
    bool thrown = false;
    try {
        function();
    } catch (const std::invalid_argument &) {
        thrown = true;
    }
    assert(thrown);
}

void test_validation() {
    constexpr uint32_t rows = 3, dim = 128;
    Data<__bfloat16> x({rows * dim + 8}, gpu), y({rows * dim + 8}, gpu);
    Data<float> weight({dim + 8}, gpu), rstd({rows + 1}, gpu), dw({dim + 8}, gpu);
    Data<float> partial_dw({rows * dim + 8}, gpu);
    // Both dtype vector widths must divide dim, even for zero rows.
    for (uint32_t invalid_dim : {0U, 1U, 4U, 7U, 127U, 129U, 65537U}) {
        expect_invalid([&] {
            kernel::rms_norm_forward(x.data(), weight.data(), y.data(), rstd.data(), rows, invalid_dim, 1e-6F);
        });
        expect_invalid([&] {
            kernel::rms_norm_backward(x.data(), x.data(), weight.data(), rstd.data(),
                                      y.data(), dw.data(), partial_dw.data(), rows, invalid_dim);
        });
    }
    // Check every accessed pointer before launch, including output/workspace.
    for (int which = 0; which < 4; ++which) {
        expect_invalid([&] {
            kernel::rms_norm_forward(
                x.data() + (which == 0), weight.data() + (which == 1 ? 4 : 0),
                y.data() + (which == 2), which == 3 ? nullptr : rstd.data(), rows, dim, 1e-6F);
        });
    }
    for (int which = 0; which < 7; ++which) {
        expect_invalid([&] {
            kernel::rms_norm_backward(
                x.data() + (which == 0), x.data() + (which == 1),
                weight.data() + (which == 2 ? 4 : 0), which == 3 ? nullptr : rstd.data(),
                y.data() + (which == 4), dw.data() + (which == 5),
                partial_dw.data() + (which == 6 ? 4 : 0), rows, dim);
        });
    }
    for (float eps : {-1.0F, std::numeric_limits<float>::infinity()}) {
        expect_invalid([&] {
            kernel::rms_norm_forward(x.data(), weight.data(), y.data(), rstd.data(), rows, dim, eps);
        });
    }
    // Empty batches have no input accesses, but still overwrite dweight.
    dw.fill(5.0F);
    kernel::rms_norm_forward<float, float>(nullptr, nullptr, nullptr, nullptr, 0, dim, 1e-6F);
    kernel::rms_norm_backward<float, float>(nullptr, nullptr, nullptr, nullptr, nullptr, dw.data(), nullptr, 0, dim);
    auto actual = dw.copy_to(cpu);
    for (uint32_t i = 0; i < dim; ++i) assert(actual[i] == 0.0F);
    for (uint32_t i = dim; i < dw.size(); ++i) assert(actual[i] == 5.0F);
    assert(gpuGetLastError() == gpuSuccess);
    std::cout << "[pass] shape/alignment rejection and empty batch\n";
}


// Same GPU event timing style as the standalone architecture benchmarks.
template <typename Function>
float time_kernel(Function launch) {
    for (int i = 0; i < 10; ++i) launch();
    gpuEvent_t start, stop;
    assert(gpuEventCreate(&start) == gpuSuccess);
    assert(gpuEventCreate(&stop) == gpuSuccess);
    assert(gpuEventRecord(start) == gpuSuccess);
    for (int i = 0; i < 100; ++i) launch();
    assert(gpuEventRecord(stop) == gpuSuccess);
    assert(gpuEventSynchronize(stop) == gpuSuccess);
    float milliseconds = 0;
    assert(gpuEventElapsedTime(&milliseconds, start, stop) == gpuSuccess);
    assert(gpuEventDestroy(start) == gpuSuccess);
    assert(gpuEventDestroy(stop) == gpuSuccess);
    return milliseconds * 1000 / 100;
}

void benchmark() {
    constexpr uint32_t rows = 4096, dim = 4096;
    auto x = Data<__bfloat16>::uniform({rows, dim}, -1, 1, gpu, 2026);
    auto dy = Data<__bfloat16>::uniform({rows, dim}, -1, 1, gpu, 2027);
    auto weight = Data<float>::uniform({dim}, -1, 1, gpu, 2028);
    Data<__bfloat16> y({rows, dim}, gpu), dx({rows, dim}, gpu);
    Data<float> rstd({rows}, gpu), dw({dim}, gpu), partial_dw({kernel::rms_norm_groups(rows, dim), dim}, gpu);
    const float forward_us = time_kernel([&] {
        kernel::rms_norm_forward(x.data(), weight.data(), y.data(), rstd.data(), rows, dim, 1e-6F);
    });
    const float backward_us = time_kernel([&] {
        kernel::rms_norm_backward(dy.data(), x.data(), weight.data(), rstd.data(),
                                  dx.data(), dw.data(), partial_dw.data(), rows, dim);
    });
    assert(gpuGetLastError() == gpuSuccess);
    std::cout << "[bench] bf16 x / fp32 weight, rows=" << rows << " dim=" << dim
              << " forward_us=" << forward_us << " backward_us=" << backward_us << '\n';
}

int main() {
    test_types<float, float>("fp32 x / fp32 weight");
    test_types<__half, float>("fp16 x / fp32 weight");
    test_types<__bfloat16, float>("bf16 x / fp32 weight");
    test_types<__half, __half>("fp16 x / fp16 weight");
    test_types<__bfloat16, __bfloat16>("bf16 x / bf16 weight");
    test_case<float, float>(3, 4, 1e-6F);
    test_validation();
    benchmark();
    std::cout << "ok\n";
}
