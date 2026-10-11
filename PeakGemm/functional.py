"""Differentiable fused operators, in the style of torch.nn.functional."""

import torch


__all__ = ["rms_norm", "linear"]


@torch.library.register_fake("PeakGemm::rms_norm_forward")
def _rms_norm_forward_fake(x, weight, eps):
    return x.new_empty(x.shape), x.new_empty(x.shape[:-1], dtype=torch.float32)


@torch.library.register_fake("PeakGemm::rms_norm_backward")
def _rms_norm_backward_fake(grad_output, x, weight, rstd):
    return x.new_empty(x.shape), weight.new_empty(weight.shape)


def _rms_norm_setup_context(ctx, inputs, output):
    x, weight, _ = inputs
    _, rstd = output
    ctx.save_for_backward(x, weight, rstd)
    ctx.mark_non_differentiable(rstd)
    ctx.set_materialize_grads(False)


def _rms_norm_backward(ctx, grad_output, _):
    if grad_output is None:
        return None, None, None
    if torch.is_grad_enabled():
        raise RuntimeError("PeakGemm RMSNorm supports first-order gradients only")
    x, weight, rstd = ctx.saved_tensors
    grad_input, grad_weight = torch.ops.PeakGemm.rms_norm_backward(
        grad_output.contiguous(), x, weight, rstd
    )
    return grad_input, grad_weight, None


torch.library.register_autograd(
    "PeakGemm::rms_norm_forward",
    _rms_norm_backward,
    setup_context=_rms_norm_setup_context,
)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize x[..., dim] with weight[dim], accumulating in FP32.

    CUDA/ROCm; FP32, FP16 or BF16 x/weight (independent dtypes).
    Output/dx follow x.dtype, dweight follows weight.dtype.
    Requires contiguous, vector-aligned x/weight and 1 <= dim <= 65536.
    dim must be divisible by both 16/x.element_size() and 16/weight.element_size().
    Invalid layouts raise an error; no scalar fallback or implicit input copies.
    """
    return torch.ops.PeakGemm.rms_norm_forward(x, weight, eps)[0]


@torch.library.register_fake("PeakGemm::linear")
def _linear_forward_fake(x, weight):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def _linear_setup_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _linear_backward(ctx, grad_output):
    x, weight = ctx.saved_tensors
    # The current HGEMM only accepts contiguous A @ B.T and low-precision output.
    # Native GEMMs handle transposed gradients without materializing both inputs.
    with torch.autocast("cuda", enabled=False):
        grad_input = grad_output.matmul(weight) if ctx.needs_input_grad[0] else None
        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_output_2d = grad_output.flatten(0, -2) if grad_output.dim() > 1 else grad_output.unsqueeze(0)
            input_2d = x.flatten(0, -2) if x.dim() > 1 else x.unsqueeze(0)
            grad_weight = grad_output_2d.T.matmul(input_2d)
    return grad_input, grad_weight


torch.library.register_autograd(
    "PeakGemm::linear",
    _linear_backward,
    setup_context=_linear_setup_context,
)


def linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear with precompiled HGEMM forward and native PyTorch backward.

    Aligned FP16/BF16 GPU shapes use PeakGemm; other shapes/dtypes, CPU and
    biased layers use torch.nn.functional.linear. Autocast uses PyTorch's
    cached parameter casts, retaining FP32 master parameters/gradients.
    """
    if not x.is_cuda or bias is not None:
        return torch.nn.functional.linear(x, weight, bias)
    return torch.ops.PeakGemm.linear(x, weight)
