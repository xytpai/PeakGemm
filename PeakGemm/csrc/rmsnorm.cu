#include <cmath>
#include <limits>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include "peak_gemm/kernel/rmsnorm.hpp"

namespace {
bool supported(at::ScalarType type) {
    return type == at::kFloat || type == at::kHalf || type == at::kBFloat16;
}

void check(const at::Tensor &x, const at::Tensor &weight) {
    TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "RMSNorm requires CUDA/HIP tensors");
    TORCH_CHECK(x.device() == weight.device(), "RMSNorm tensors must be on the same device");
    TORCH_CHECK(supported(x.scalar_type()) && supported(weight.scalar_type()),
                "RMSNorm supports fp16, bf16 and fp32");
    TORCH_CHECK(x.dim() >= 1 && weight.dim() == 1, "RMSNorm expects x[..., N], weight[N]");
    TORCH_CHECK(x.size(-1) > 0 && x.size(-1) == weight.numel(), "RMSNorm weight size mismatch");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "RMSNorm C++ inputs must be contiguous");
    TORCH_CHECK(x.size(-1) <= INT32_MAX && x.numel() / x.size(-1) <= INT32_MAX,
                "RMSNorm dimensions exceed INT32_MAX");
}

// Dispatch independently: ModelX starts with FP32 weight, but .bf16() casts it.
template <typename F>
void dispatch(at::ScalarType type, F &&f) {
    switch (type) {
    case at::kHalf: f(__half{}); break;
    case at::kBFloat16: f(__bfloat16{}); break;
    case at::kFloat: f(float{}); break;
    default: TORCH_CHECK(false, "unsupported RMSNorm dtype");
    }
}

std::vector<at::Tensor> forward(at::Tensor x, at::Tensor weight, double eps, bool save_rstd) {
    check(x, weight);
    TORCH_CHECK(std::isfinite(eps) && eps >= 0.0 && eps <= std::numeric_limits<float>::max(),
                "RMSNorm eps must be finite, nonnegative and representable in fp32");
    c10::DeviceGuard guard(x.device());
    const int cols = int(x.size(-1));
    const int64_t rows = x.numel() / cols;
    auto y = at::empty_like(x);
    auto rstd = at::empty({save_rstd ? rows : 0}, x.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    dispatch(x.scalar_type(), [&](auto t) {
        using T = decltype(t);
        dispatch(weight.scalar_type(), [&](auto w) {
            using W = decltype(w);
            peak_gemm::kernel::rmsnorm_forward<T, W>(
                reinterpret_cast<const T *>(x.data_ptr()), reinterpret_cast<const W *>(weight.data_ptr()), reinterpret_cast<T *>(y.data_ptr()),
                save_rstd ? rstd.data_ptr<float>() : nullptr, rows, cols, float(eps), stream);
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, rstd};
}

std::vector<at::Tensor> backward(
    at::Tensor dy, at::Tensor x, at::Tensor weight, at::Tensor rstd, bool need_dx, bool need_dw) {
    check(x, weight);
    const int cols = int(x.size(-1));
    const int64_t rows = x.numel() / cols;
    TORCH_CHECK(dy.device() == x.device() && dy.sizes() == x.sizes() &&
                dy.scalar_type() == x.scalar_type() && dy.is_contiguous(), "invalid RMSNorm dy");
    TORCH_CHECK(rstd.device() == x.device() && rstd.scalar_type() == at::kFloat &&
                rstd.dim() == 1 && rstd.numel() == rows && rstd.is_contiguous(), "invalid RMSNorm rstd");
    c10::DeviceGuard guard(x.device());
    auto dx = need_dx ? at::empty_like(x) : at::empty({0}, x.options());
    auto dw = need_dw ? at::empty_like(weight) : at::empty({0}, weight.options());
    const int parts = peak_gemm::kernel::rmsnorm_backward_partitions(rows);
    auto partial = at::empty({need_dw ? int64_t(parts) * cols : 0}, x.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
    dispatch(x.scalar_type(), [&](auto t) {
        using T = decltype(t);
        dispatch(weight.scalar_type(), [&](auto w) {
            using W = decltype(w);
            peak_gemm::kernel::rmsnorm_backward<T, W>(
                reinterpret_cast<const T *>(dy.data_ptr()), reinterpret_cast<const T *>(x.data_ptr()), reinterpret_cast<const W *>(weight.data_ptr()), rstd.data_ptr<float>(),
                need_dx ? reinterpret_cast<T *>(dx.data_ptr()) : nullptr, need_dw ? reinterpret_cast<W *>(dw.data_ptr()) : nullptr,
                need_dw ? partial.data_ptr<float>() : nullptr, rows, cols, stream);
        });
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {dx, dw};
}
} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward);
    m.def("backward", &backward);
}
