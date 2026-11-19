#include <numeric>

#include "peak_gemm/gemm_bench_infra.hpp"
#include "peak_gemm/kernel/fused_moe.hpp"

using peak_gemm::Data;
using peak_gemm::bench::cpu;
using peak_gemm::bench::gpu;

struct MoeShape {
    uint32_t tokens, hidden, intermediate, experts, top_k;
};

enum class InputCase {
    random,
    zero_input,
    tied_router,
    large_logits,
    shared_only,
    routed_only,
};

template <typename scalar_t>
Data<float> to_float(const Data<scalar_t> &input) {
    Data<float> output(input.shape());
    for (std::size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<float>(input[i]);
    }
    return output;
}

// Independent expert-major reference: full softmax, top-k, then weighted sum.
// Reuse the existing CPU GEMM for all projections, with FP32 intermediates.
template <typename scalar_t>
Data<float> moe_reference(
    const Data<scalar_t> &x, const Data<scalar_t> &router, const Data<scalar_t> &gate,
    const Data<scalar_t> &up, const Data<scalar_t> &down, MoeShape shape) {
    const auto [tokens, hidden, intermediate, experts, top_k] = shape;
    const auto xf = to_float(x);
    const auto rf = to_float(router);
    const auto gf = to_float(gate);
    const auto uf = to_float(up);
    const auto df = to_float(down);
    const peak_gemm::bench::DefaultCpuGemmReference gemm;
    Data<float> logits({tokens, experts}), weights({tokens, experts});
    Data<float> g({tokens, intermediate}), u({tokens, intermediate}), expert_y({tokens, hidden}), output({tokens, hidden});
    weights.fill(0.0F);
    output.fill(0.0F);
    gemm(xf.data(), rf.data(), logits.data(), tokens, experts, hidden, static_cast<const float *>(nullptr));
    std::vector<uint32_t> ids(experts);
    for (uint32_t t = 0; t < tokens; ++t) {
        auto *row = logits.data() + static_cast<std::size_t>(t) * experts;
        const float maximum = *std::max_element(row, row + experts);
        float sum = 0.0F;
        for (uint32_t e = 0; e < experts; ++e) {
            row[e] = std::exp(row[e] - maximum);
            sum += row[e];
        }
        for (uint32_t e = 0; e < experts; ++e) {
            row[e] /= sum;
        }
        std::iota(ids.begin(), ids.end(), 0U);
        std::stable_sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
        float selected_sum = 0.0F;
        for (uint32_t k = 0; k < top_k; ++k) {
            selected_sum += row[ids[k]];
        }
        for (uint32_t k = 0; k < top_k; ++k) {
            weights[static_cast<std::size_t>(t) * experts + ids[k]] = row[ids[k]] / selected_sum;
        }
    }
    const std::size_t stride = static_cast<std::size_t>(hidden) * intermediate;
    for (uint32_t e = 0; e <= experts; ++e) {
        gemm(xf.data(), gf.data() + e * stride, g.data(), tokens, intermediate, hidden, static_cast<const float *>(nullptr));
        gemm(xf.data(), uf.data() + e * stride, u.data(), tokens, intermediate, hidden, static_cast<const float *>(nullptr));
        for (std::size_t i = 0; i < g.size(); ++i) {
            g[i] = (g[i] / (1.0F + std::exp(-g[i]))) * u[i];
        }
        gemm(g.data(), df.data() + e * stride, expert_y.data(), tokens, hidden, intermediate, static_cast<const float *>(nullptr));
        for (uint32_t t = 0; t < tokens; ++t) {
            const float weight = e == experts ? 1.0F : weights[static_cast<std::size_t>(t) * experts + e];
            for (uint32_t h = 0; h < hidden; ++h) {
                const std::size_t index = static_cast<std::size_t>(t) * hidden + h;
                output[index] += weight * expert_y[index];
            }
        }
    }
    return output;
}

template <typename Error>
void check_gpu(Error error) {
    if (error != gpuSuccess) throw std::runtime_error(gpuGetErrorString(error));
}

template <typename scalar_t>
void check_case(MoeShape shape, InputCase input_case, gpuStream_t stream, uint32_t pointer_offset = 0) {
    const auto [tokens, hidden, intermediate, experts, top_k] = shape;
    using DataT = Data<scalar_t>;
    const float input_scale = 1.0F / std::sqrt(static_cast<float>(hidden));
    const float down_scale = 1.0F / std::sqrt(static_cast<float>(intermediate));
    auto x = DataT::uniform({tokens, hidden}, -1.0F, 1.0F, cpu, 2026);
    auto router = DataT::uniform({experts, hidden}, -input_scale, input_scale, cpu, 2027);
    auto gate = DataT::uniform({experts + 1, intermediate, hidden}, -input_scale, input_scale, cpu, 2028);
    auto up = DataT::uniform({experts + 1, intermediate, hidden}, -input_scale, input_scale, cpu, 2029);
    auto down = DataT::uniform({experts + 1, hidden, intermediate}, -down_scale, down_scale, cpu, 2030);
    if (input_case == InputCase::zero_input) x.fill(static_cast<scalar_t>(0));
    if (input_case == InputCase::tied_router) router.fill(static_cast<scalar_t>(0));
    if (input_case == InputCase::large_logits) {
        x.fill(static_cast<scalar_t>(1));
        for (uint32_t e = 0; e < experts; ++e) {
            for (uint32_t h = 0; h < hidden; ++h) {
                router[static_cast<std::size_t>(e) * hidden + h] =
                    static_cast<scalar_t>((static_cast<float>(e) - experts / 2.0F) * 1000.0F / hidden);
            }
        }
    }
    const std::size_t stride = static_cast<std::size_t>(hidden) * intermediate;
    if (input_case == InputCase::shared_only) {
        std::fill_n(down.data(), experts * stride, static_cast<scalar_t>(0));
    }
    if (input_case == InputCase::routed_only) {
        std::fill_n(down.data() + experts * stride, stride, static_cast<scalar_t>(0));
    }
    // Offset views exercise the scalar fallback even with aligned H/I.
    const auto copy_input = [pointer_offset](const DataT &host) {
        DataT device({host.size() + pointer_offset}, gpu);
        check_gpu(gpuMemcpy(device.data() + pointer_offset, host.data(), host.bytes(), gpuMemcpyHostToDevice));
        return device;
    };
    auto device_x = copy_input(x);
    auto device_router = copy_input(router);
    auto device_gate = copy_input(gate);
    auto device_up = copy_input(up);
    auto device_down = copy_input(down);
    const uint32_t guard = 16 / sizeof(scalar_t) + pointer_offset;
    DataT device_y({static_cast<std::size_t>(tokens) * hidden + 2 * guard}, gpu);
    device_y.fill(static_cast<scalar_t>(NAN));
    const auto expected = moe_reference(x, router, gate, up, down, shape);
    float max_diff = 0.0F;
    // Repeated launches must overwrite output without relying on a zeroed buffer.
    for (uint32_t iteration = 0; iteration < 2; ++iteration) {
        peak_gemm::kernel::fused_moe_gpu(
            device_x.data() + pointer_offset, device_router.data() + pointer_offset, device_gate.data() + pointer_offset,
            device_up.data() + pointer_offset, device_down.data() + pointer_offset, device_y.data() + guard,
            tokens, hidden, intermediate, experts, top_k, stream);
        check_gpu(gpuGetLastError());
        check_gpu(gpuStreamSynchronize(stream));
        const auto actual = device_y.copy_to(cpu);
        const float tolerance = peak_gemm::bench::default_base_tolerance<scalar_t>();
        for (uint32_t i = 0; i < guard; ++i) {
            if (!std::isnan(static_cast<float>(actual[i])) || !std::isnan(static_cast<float>(actual[guard + expected.size() + i]))) {
                throw std::runtime_error("Fused MoE overwrote an output guard");
            }
        }
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const float diff = std::abs(static_cast<float>(actual[guard + i]) - expected[i]);
            if (!std::isfinite(diff) || diff > 2.0e-6F + tolerance * std::abs(expected[i])) {
                throw std::runtime_error("Fused MoE accuracy check failed at index " + std::to_string(i));
            }
            max_diff = std::max(max_diff, diff);
        }
    }
    std::cout << peak_gemm::bench::default_type_name<scalar_t>()
              << " T=" << tokens << " H=" << hidden << " I=" << intermediate << " E=" << experts << " K=" << top_k
              << " offset=" << pointer_offset << " case=" << static_cast<int>(input_case) << " max_diff=" << max_diff << '\n';
}

void check_arguments() {
    auto launch = [](MoeShape shape) {
        peak_gemm::kernel::fused_moe_gpu<float>(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            shape.tokens, shape.hidden, shape.intermediate, shape.experts, shape.top_k);
    };
    launch({0, 1, 1, 1, 1});
    const MoeShape invalid[] = {
        {1, 1, 1, 1, 1}, // Null tensor pointers.
        {0, 0, 1, 1, 1},
        {0, 1, 0, 1, 1},
        {0, 1, 1, 0, 1},
        {0, 1, 1, 1, 0},
        {0, 1, 1, 1, 2},
        {0, 12288, 1, 1, 1}, // Shared-memory limit.
        {0, 1, UINT32_MAX, 1, 1},
        {UINT32_MAX, 1, 1, 1, 1},
    };
    for (const auto shape : invalid) {
        bool rejected = false;
        try {
            launch(shape);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("Fused MoE accepted invalid arguments");
    }
    check_gpu(gpuGetLastError());
}

template <typename scalar_t>
void run(gpuStream_t stream) {
    const MoeShape shapes[] = {
        {1, 1, 1, 1, 1},
        {5, 128, 256, 8, 2},
        {3, 520, 264, 9, 3},
        {2, 2056, 264, 5, 2},
        {3, 12, 20, 5, 2},
        {3, 7, 9, 5, 2},
        {3, 511, 519, 5, 2},
        {3, 31, 17, 5, 1},
        {7, 65, 33, 5, 2},
        {2, 17, 65, 7, 7},
        {4, 257, 513, 17, 4},
        {3, 65, 129, 257, 8},
        {2, 17, 33, 257, 257},
        {257, 33, 65, 5, 2},
        {1, 17, 12268, 1, 1},
        {1, 12268, 17, 1, 1},
    };
    for (const auto shape : shapes) {
        check_case<scalar_t>(shape, InputCase::random, stream);
    }
    for (uint32_t offset = 1; offset < 16 / sizeof(scalar_t); ++offset) {
        check_case<scalar_t>({3, 128, 256, 5, 2}, InputCase::random, stream, offset);
        check_case<scalar_t>({3, 65, 33, 5, 2}, InputCase::random, stream, offset);
    }
    for (const auto input_case : {InputCase::zero_input, InputCase::tied_router, InputCase::large_logits, InputCase::shared_only, InputCase::routed_only}) {
        check_case<scalar_t>({3, 33, 65, 5, 3}, input_case, stream);
    }
}

int main() {
    check_arguments();
    gpuStream_t stream;
    check_gpu(gpuStreamCreate(&stream));
    run<float>(stream);
    run<__half>(stream);
    run<__bfloat16>(stream);
    check_case<float>({2, 17, 33, 3, 2}, InputCase::random, nullptr);
    check_gpu(gpuStreamDestroy(stream));
    std::cout << "ok\n";
    return 0;
}
