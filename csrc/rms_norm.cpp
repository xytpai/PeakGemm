// Native FP16/BF16 conversions are needed by the backend-neutral kernels.
#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__

#include "peak_gemm/kernel/rms_norm.hpp"

#include <limits>
#include <tuple>

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>

#if defined(__HIPCC__)
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAStream.h>
#endif

namespace {

gpuStream_t current_stream(const at::Tensor &x) {
#if defined(__HIPCC__)
    return c10::hip::getCurrentHIPStream(x.get_device()).stream();
#else
    return c10::cuda::getCurrentCUDAStream(x.get_device()).stream();
#endif
}

template <typename Function>
void dispatch_dtype(at::ScalarType dtype, Function function) {
    switch (dtype) {
    case at::kFloat: function.template operator()<float>(); break;
    case at::kHalf: function.template operator()<__half>(); break;
    case at::kBFloat16: function.template operator()<__bfloat16>(); break;
    default: TORCH_CHECK(false, "RMSNorm supports fp32, fp16 and bf16");
    }
}

void check_inputs(const at::Tensor &x, const at::Tensor &weight) {
    TORCH_CHECK(x.is_cuda() && weight.device() == x.device(), "RMSNorm expects tensors on the same GPU");
    TORCH_CHECK(x.dim() >= 1 && weight.dim() == 1 && weight.size(0) == x.size(-1),
                "RMSNorm expects x[..., dim] and weight[dim]");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "RMSNorm tensors must be contiguous");
    TORCH_CHECK(x.size(-1) > 0 && x.size(-1) <= peak_gemm::kernel::kRmsNormMaxDim,
                "RMSNorm dim must be between 1 and 65536");
    TORCH_CHECK(x.numel() / x.size(-1) <= 0x7FFFFFFF, "RMSNorm rows exceed the GPU grid limit");
}

void check_launch() {
    const auto error = gpuGetLastError();
    TORCH_CHECK(error == gpuSuccess, "RMSNorm kernel launch failed: ", gpuGetErrorString(error));
}

std::tuple<at::Tensor, at::Tensor> rms_norm_forward(const at::Tensor &x, const at::Tensor &weight, double eps) {
    check_inputs(x, weight);
    TORCH_CHECK(std::isfinite(eps) && eps >= 0 && eps <= std::numeric_limits<float>::max(),
                "RMSNorm eps must be finite, non-negative and representable in fp32");
    c10::DeviceGuard guard(x.device());
    const auto dim = static_cast<uint32_t>(x.size(-1));
    const auto rows = static_cast<uint32_t>(x.numel() / dim);
    auto y = at::empty(x.sizes(), x.options());
    auto rstd = at::empty(x.sizes().slice(0, x.dim() - 1), x.options().dtype(at::kFloat));
    dispatch_dtype(x.scalar_type(), [&]<typename input_t>() {
        dispatch_dtype(weight.scalar_type(), [&]<typename weight_t>() {
            peak_gemm::kernel::rms_norm_forward(
                static_cast<const input_t *>(x.data_ptr()), static_cast<const weight_t *>(weight.data_ptr()),
                static_cast<input_t *>(y.data_ptr()), rstd.data_ptr<float>(), rows, dim, static_cast<float>(eps), current_stream(x));
        });
    });
    check_launch();
    return {y, rstd};
}

std::tuple<at::Tensor, at::Tensor> rms_norm_backward(
    const at::Tensor &dy, const at::Tensor &x, const at::Tensor &weight, const at::Tensor &rstd) {
    check_inputs(x, weight);
    TORCH_CHECK(dy.device() == x.device() && dy.scalar_type() == x.scalar_type()
                    && dy.sizes() == x.sizes() && dy.is_contiguous(), "RMSNorm dy must match contiguous x");
    TORCH_CHECK(rstd.device() == x.device() && rstd.scalar_type() == at::kFloat
                    && rstd.sizes() == x.sizes().slice(0, x.dim() - 1) && rstd.is_contiguous(),
                "RMSNorm rstd must be contiguous fp32 with shape x.shape[:-1]");
    c10::DeviceGuard guard(x.device());
    const auto dim = static_cast<uint32_t>(x.size(-1));
    const auto rows = static_cast<uint32_t>(x.numel() / dim);
    auto dx = at::empty(x.sizes(), x.options());
    auto dw = at::empty(weight.sizes(), weight.options());
    auto partial_dw = at::empty({peak_gemm::kernel::rms_norm_groups(rows, dim), dim}, x.options().dtype(at::kFloat));
    dispatch_dtype(x.scalar_type(), [&]<typename input_t>() {
        dispatch_dtype(weight.scalar_type(), [&]<typename weight_t>() {
            peak_gemm::kernel::rms_norm_backward(
                static_cast<const input_t *>(dy.data_ptr()), static_cast<const input_t *>(x.data_ptr()),
                static_cast<const weight_t *>(weight.data_ptr()), rstd.data_ptr<float>(),
                static_cast<input_t *>(dx.data_ptr()), static_cast<weight_t *>(dw.data_ptr()), partial_dw.data_ptr<float>(),
                rows, dim, current_stream(x));
        });
    });
    check_launch();
    return {dx, dw};
}

} // namespace

TORCH_LIBRARY_FRAGMENT(PeakGemm, module) {
    module.def("rms_norm_forward(Tensor x, Tensor weight, float eps) -> (Tensor, Tensor)");
    module.def("rms_norm_backward(Tensor dy, Tensor x, Tensor weight, Tensor rstd) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(PeakGemm, CUDA, module) {
    module.impl("rms_norm_forward", &rms_norm_forward);
    module.impl("rms_norm_backward", &rms_norm_backward);
}
