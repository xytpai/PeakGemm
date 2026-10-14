// Torch may disable native half operators; the kernels need them.
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__

#include <cstdint>
#include <limits>
#include <optional>

#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#if defined(__HIPCC__)
#include <c10/hip/HIPStream.h>
#include "peak_gemm/backend/arch_gfx950.hpp"
#include "peak_gemm/kernel/hgemm_gfx950.hpp"
#include "peak_gemm/kernel/hgemm_hti_gfx950.hpp"
#else
#include <c10/cuda/CUDAStream.h>
#include "peak_gemm/backend/arch_sm80.hpp"
#include "peak_gemm/kernel/hgemm_sm80.hpp"
#endif

namespace {

#if defined(__HIPCC__)
constexpr uint32_t N_WARPS_64 = 1, N_WARPS_128 = 2, SWIZZLE_M = 0;
#else
constexpr uint32_t N_WARPS_64 = 2, N_WARPS_128 = 4, SWIZZLE_M = 8;
#endif

struct GemmArgs {
    const void *a;
    const void *b;
    void *c;
    const void *bias;
    uint32_t m, n, k, split_k;
    uint32_t *semaphore;
    uint32_t *signal;
    gpuStream_t stream;
};

template <typename scalar_t, bool HAS_BIAS, bool SPLIT_K, bool B_TRANSPOSE>
void launch(const GemmArgs &args, int64_t config) {
    const auto *a = static_cast<const scalar_t *>(args.a);
    const auto *b = static_cast<const scalar_t *>(args.b);
    auto *c = static_cast<scalar_t *>(args.c);
    const auto *bias = static_cast<const scalar_t *>(args.bias);
    using peak_gemm::kernel::hgemm_template;

    // Only these four tile configurations are instantiated at build time.
    switch (config) {
    case 0:
        hgemm_template<scalar_t, 16, 64, 64, 1, N_WARPS_64, 2, SWIZZLE_M, HAS_BIAS, SPLIT_K, B_TRANSPOSE>(
            a, b, c, args.m, args.n, args.k, args.split_k, args.semaphore, args.signal, bias, args.stream);
        break;
    case 1:
        hgemm_template<scalar_t, 64, 64, 32, 2, N_WARPS_64, 3, SWIZZLE_M, HAS_BIAS, SPLIT_K, B_TRANSPOSE>(
            a, b, c, args.m, args.n, args.k, args.split_k, args.semaphore, args.signal, bias, args.stream);
        break;
    case 2:
        hgemm_template<scalar_t, 64, 128, 32, 2, N_WARPS_128, 3, SWIZZLE_M, HAS_BIAS, SPLIT_K, B_TRANSPOSE>(
            a, b, c, args.m, args.n, args.k, args.split_k, args.semaphore, args.signal, bias, args.stream);
        break;
    case 3:
#if defined(__HIPCC__)
        // Large shapes: HTI with a fixed 2x4 (M, N) wave layout.
        if constexpr (SPLIT_K) {
            TORCH_CHECK(false, "PeakGemm HTI config=3 requires split_k=1; use config=0..2 for split-K");
        } else {
            peak_gemm::kernel::hgemm_hti_template<scalar_t, 256, 256, 64, 2, 4, 0, HAS_BIAS, B_TRANSPOSE>(
                a, b, c, args.m, args.n, args.k, bias, args.stream);
        }
#else
        hgemm_template<scalar_t, 128, 128, 32, 2, N_WARPS_128, 3, SWIZZLE_M, HAS_BIAS, SPLIT_K, B_TRANSPOSE>(
            a, b, c, args.m, args.n, args.k, args.split_k, args.semaphore, args.signal, bias, args.stream);
#endif
        break;
    }
}

template <typename scalar_t, bool B_TRANSPOSE>
void dispatch(const GemmArgs &args, int64_t config) {
    if (args.split_k > 1) {
        if (args.bias) {
            launch<scalar_t, true, true, B_TRANSPOSE>(args, config);
        } else {
            launch<scalar_t, false, true, B_TRANSPOSE>(args, config);
        }
    } else if (args.bias) {
        launch<scalar_t, true, false, B_TRANSPOSE>(args, config);
    } else {
        launch<scalar_t, false, false, B_TRANSPOSE>(args, config);
    }
}


template <typename scalar_t>
void dispatch_layout(const GemmArgs &args, int64_t config, bool transpose_b) {
    if (transpose_b) {
        dispatch<scalar_t, true>(args, config);
    } else {
        dispatch<scalar_t, false>(args, config);
    }
}

void hgemm(
    at::Tensor c, const at::Tensor &a, const at::Tensor &b, int64_t split_k,
    const std::optional<at::Tensor> &bias, int64_t config,
    const std::optional<at::Tensor> &semaphore, const std::optional<at::Tensor> &signal, bool transpose_b = true) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "PeakGemm expects GPU tensors");
    TORCH_CHECK(a.device() == b.device() && a.device() == c.device(), "PeakGemm tensors must be on the same device");
    TORCH_CHECK(a.scalar_type() == b.scalar_type() && a.scalar_type() == c.scalar_type(), "PeakGemm tensors must have the same dtype");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16, "PeakGemm supports fp16 and bf16");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "PeakGemm expects matrices");
    TORCH_CHECK(a.is_contiguous() && c.is_contiguous(), "PeakGemm A and C must be contiguous");
    TORCH_CHECK(b.is_contiguous() || b.transpose(0, 1).is_contiguous(),
                "PeakGemm B must be contiguous or a transpose of a contiguous matrix");
    TORCH_CHECK(config >= 0 && config <= 3, "PeakGemm config must be between 0 and 3");
    TORCH_CHECK(split_k > 0 && split_k <= 65535, "PeakGemm split_k must be between 1 and 65535");

    const auto m = a.size(0), n = b.size(transpose_b ? 0 : 1), k = a.size(1);
    TORCH_CHECK(b.size(transpose_b ? 1 : 0) == k && c.size(0) == m && c.size(1) == n, "PeakGemm tensor shapes do not match");
    // A transpose view changes the logical operation, not the underlying data.
    const bool b_transpose = b.is_contiguous() ? transpose_b : !transpose_b;
    TORCH_CHECK(m <= std::numeric_limits<uint32_t>::max() && n <= std::numeric_limits<uint32_t>::max()
                    && k <= std::numeric_limits<uint32_t>::max(), "PeakGemm dimensions exceed uint32");
    if (bias) {
        TORCH_CHECK(bias->device() == a.device() && bias->scalar_type() == a.scalar_type()
                        && bias->is_contiguous() && bias->numel() == n,
                    "PeakGemm bias must be contiguous, have N elements, and match input device/dtype");
    }
    if (split_k > 1) {
        for (const auto *workspace : {&semaphore, &signal}) {
            TORCH_CHECK(workspace->has_value() && (*workspace)->device() == a.device()
                            && (*workspace)->scalar_type() == at::kUInt32 && (*workspace)->is_contiguous()
                            && (*workspace)->numel() >= peak_gemm::kernel::kSemaphoreCount,
                        "PeakGemm split-K workspace must be contiguous uint32 on the input device with at least 256 elements");
        }
    }

    c10::DeviceGuard guard(a.device());
#if defined(__HIPCC__)
    auto stream = c10::hip::getCurrentHIPStream(a.get_device()).stream();
#else
    auto stream = c10::cuda::getCurrentCUDAStream(a.get_device()).stream();
#endif
    const GemmArgs args{
        a.data_ptr(), b.data_ptr(), c.data_ptr(), bias ? bias->data_ptr() : nullptr,
        static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k), static_cast<uint32_t>(split_k),
        split_k > 1 ? semaphore->data_ptr<uint32_t>() : nullptr,
        split_k > 1 ? signal->data_ptr<uint32_t>() : nullptr, stream};
    if (a.scalar_type() == at::kHalf) {
        dispatch_layout<__half>(args, config, b_transpose);
    } else {
        dispatch_layout<__bfloat16>(args, config, b_transpose);
    }
    const auto error = gpuGetLastError();
    TORCH_CHECK(error == gpuSuccess, "PeakGemm kernel launch failed: ", gpuGetErrorString(error));
}


// Select only among precompiled, non-split configurations. No padding/JIT.
int linear_config(int64_t m, int64_t n, int64_t k) {
    if (m <= 0 || n <= 0 || k <= 0) return -1;
#if defined(__HIPCC__)
    constexpr int64_t max_elements = std::numeric_limits<uint32_t>::max() / 2;
    if (m > max_elements / k || n > max_elements / k || m > max_elements / n) return -1;
    if (m % 256 == 0 && n % 256 == 0 && k % 128 == 0) return 3; // HTI, waves 2x4
#else
    constexpr int64_t max_elements = std::numeric_limits<uint32_t>::max();
    if (m > max_elements / k || n > max_elements / k || m > max_elements / n) return -1;
    if (m % 128 == 0 && n % 128 == 0 && k >= 64 && k % 32 == 0) return 3;
#endif
    if (m % 64 == 0 && k >= 64 && k % 32 == 0) {
        if (n % 128 == 0) return 2;
        if (n % 64 == 0) return 1;
    }
    if (m % 16 == 0 && n % 64 == 0 && k % 64 == 0) return 0;
    return -1;
}

at::Tensor linear(const at::Tensor &x, const at::Tensor &weight) {
    TORCH_CHECK(x.is_cuda() && weight.device() == x.device(), "PeakGemm linear expects tensors on the same GPU");
    TORCH_CHECK(x.dim() >= 1 && weight.dim() == 2 && x.size(-1) == weight.size(1),
                "PeakGemm linear expects x[..., K] and weight[N, K]");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "PeakGemm linear tensors must have the same dtype");
    c10::DeviceGuard guard(x.device());
    const auto k = weight.size(1), n = weight.size(0);
    const auto m = k == 0 ? 0 : x.numel() / k;
    const int config = linear_config(m, n, k);
    if (config < 0 || (x.scalar_type() != at::kHalf && x.scalar_type() != at::kBFloat16)) {
        return at::linear(x, weight);
    }
    auto a = x.reshape({m, k}).contiguous();
    // A weight.T view can be consumed by the NN specialization without a copy.
    auto b = weight.is_contiguous() || weight.transpose(0, 1).is_contiguous() ? weight : weight.contiguous();
    // A contiguous tensor can still have an unaligned storage offset.
    if (reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 || reinterpret_cast<uintptr_t>(b.data_ptr()) % 16) {
        return at::linear(x, weight);
    }
    auto c = at::empty({m, n}, x.options());
    hgemm(c, a, b, 1, std::nullopt, config, std::nullopt, std::nullopt);
    auto shape = x.sizes().vec();
    shape.back() = n;
    return c.view(shape);
}

at::Tensor linear_autocast(const at::Tensor &x, const at::Tensor &weight) {
    c10::impl::ExcludeDispatchKeyGuard guard(c10::DispatchKey::AutocastCUDA);
    const auto dtype = at::autocast::get_autocast_dtype(at::kCUDA);
    // Redispatch through autograd after casting, just like native Linear.
    static auto op = c10::Dispatcher::singleton()
        .findSchemaOrThrow("PeakGemm::linear", "")
        .typed<at::Tensor(const at::Tensor &, const at::Tensor &)>();
    return op.call(at::autocast::cached_cast(dtype, x), at::autocast::cached_cast(dtype, weight));
}

} // namespace

TORCH_LIBRARY(PeakGemm, module) {
    module.def("linear(Tensor x, Tensor weight) -> Tensor");
    module.def("hgemm(Tensor(a!) c, Tensor a, Tensor b, int split_k, Tensor? bias, int config, "
               "Tensor(b!)? semaphore, Tensor(c!)? signal, bool transpose_b=True) -> ()", &hgemm);
}

TORCH_LIBRARY_IMPL(PeakGemm, CUDA, module) {
    module.impl("linear", &linear);
}

TORCH_LIBRARY_IMPL(PeakGemm, AutocastCUDA, module) {
    module.impl("linear", &linear_autocast);
}
