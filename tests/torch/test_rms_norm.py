import pytest
import torch

import PeakGemm
from PeakGemm import functional as F


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def reference(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * y).to(x.dtype)


def assert_close(actual, expected):
    tolerance = {
        torch.float32: 2e-4,
        torch.float16: 3e-3,
        torch.bfloat16: 3e-2,
    }[actual.dtype]
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [
    (8,), (24,), (3, 32), (3, 64), (3, 72), (2, 3, 264),
    (513, 120), (2049, 128), (1025, 1024), (7, 4104), (3, 8192), (2, 65536), (0, 128),
])
@pytest.mark.parametrize("eps", [1e-6, 0.1])
def test_accuracy(dtype, weight_dtype, shape, eps):
    torch.manual_seed(2026)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.randn(shape[-1], device="cuda", dtype=weight_dtype, requires_grad=True)
    dy = torch.randn_like(x)
    y = F.rms_norm(x, weight, eps)
    expected = reference(x, weight, eps)
    dx, dw = torch.autograd.grad(y, (x, weight), dy)
    ref_dx, ref_dw = torch.autograd.grad(expected, (x, weight), dy)
    assert y.dtype == dx.dtype == x.dtype
    assert dw.dtype == weight.dtype
    assert_close(y, expected)
    assert_close(dx, ref_dx)
    assert_close(dw, ref_dw)


def test_zero_input():
    x = torch.zeros(2, 3, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(128, device="cuda", requires_grad=True)
    y = F.rms_norm(x, weight)
    dx, dw = torch.autograd.grad(y, (x, weight), torch.ones_like(y))
    assert_close(y, torch.zeros_like(x))
    assert_close(dx, torch.full_like(x, 1000))
    assert_close(dw, torch.zeros_like(weight))


@pytest.mark.parametrize("which", ["x", "weight"])
def test_noncontiguous_rejected(which):
    x = torch.randn(2, 128, 3, device="cuda").transpose(1, 2)
    weight = torch.ones(256, device="cuda")[::2]
    if which == "x":
        weight = weight.contiguous()
    else:
        x = x.contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        F.rms_norm(x, weight)


def test_broadcast_gradient():
    x = torch.randn(3, 128, device="cuda", requires_grad=True)
    weight = torch.ones(128, device="cuda", requires_grad=True)
    dx, dw = torch.autograd.grad(F.rms_norm(x, weight).sum(), (x, weight))
    rx, rw = torch.autograd.grad(reference(x, weight, 1e-6).sum(), (x, weight))
    assert_close(dx, rx)
    assert_close(dw, rw)


@pytest.mark.parametrize("train_x,train_weight", [(True, True), (True, False), (False, True)])
def test_training(train_x, train_weight):
    layer = PeakGemm.RMSNorm(512, 1e-6, device="cuda")
    weight_ref = layer.weight.detach().clone().requires_grad_(train_weight)
    layer.weight.requires_grad_(train_weight)
    x = torch.randn(7, 512, device="cuda", dtype=torch.bfloat16, requires_grad=train_x)
    ref_x = x.detach().clone().requires_grad_(train_x)
    # Two backwards exercise accumulation into FP32 parameter gradients.
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            layer(x).float().square().mean().backward()
            reference(ref_x, weight_ref, layer.eps).float().square().mean().backward()
    if train_x:
        assert_close(x.grad, ref_x.grad)
    if train_weight:
        assert layer.weight.grad.dtype == torch.float32
        assert_close(layer.weight.grad, weight_ref.grad)
        opt = torch.optim.SGD(layer.parameters(), lr=0.1)
        old = layer.weight.detach().clone()
        opt.step()
        assert_close(layer.weight, old - 0.1 * weight_ref.grad)
    assert list(layer.state_dict()) == ["weight"]


def test_finite_difference():
    x = torch.randn(2, 8, device="cuda", requires_grad=True)
    weight = torch.randn(8, device="cuda", requires_grad=True)
    grad = torch.randn_like(x)
    dx, dw = torch.autograd.grad(F.rms_norm(x, weight), (x, weight), grad)
    with torch.no_grad():
        for parameter, analytic in [(x, dx), (weight, dw)]:
            direction = torch.randn_like(parameter)
            step = 1e-3
            parameter.add_(direction, alpha=step)
            plus = (F.rms_norm(x, weight) * grad).sum()
            parameter.add_(direction, alpha=-2 * step)
            minus = (F.rms_norm(x, weight) * grad).sum()
            parameter.add_(direction, alpha=step)
            torch.testing.assert_close(
                (plus - minus) / (2 * step), (analytic * direction).sum(),
                atol=2e-3, rtol=2e-3,
            )


def test_stream_and_graph():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(513, 264, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(264, device="cuda", requires_grad=True)
        dy = torch.randn_like(x)
        for _ in range(3):
            y = F.rms_norm(x, weight)
            dx, dw = torch.autograd.grad(y, (x, weight), dy)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = F.rms_norm(x, weight)
        dx, dw = torch.autograd.grad(y, (x, weight), dy)
    with torch.cuda.stream(stream):
        for _ in range(3):
            with torch.no_grad():
                x.normal_()
            graph.replay()
            expected = reference(x, weight, 1e-6)
            ref_dx, ref_dw = torch.autograd.grad(expected, (x, weight), dy)
            assert_close(y, expected)
            assert_close(dx, ref_dx)
            assert_close(dw, ref_dw)
    stream.synchronize()


def test_compile():
    fn = torch.compile(F.rms_norm, fullgraph=True, dynamic=True)
    for rows in [3, 7]:
        x = torch.randn(rows, 264, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(264, device="cuda", requires_grad=True)
        dy = torch.randn_like(x)
        y = fn(x, weight)
        dx, dw = torch.autograd.grad(y, (x, weight), dy)
        expected = reference(x, weight, 1e-6)
        ref_dx, ref_dw = torch.autograd.grad(expected, (x, weight), dy)
        assert_close(y, expected)
        assert_close(dx, ref_dx)
        assert_close(dw, ref_dw)


def test_opcheck():
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(64, device="cuda", requires_grad=True)
    torch.library.opcheck(torch.ops.PeakGemm.rms_norm_forward.default, (x, weight, 1e-6))
    with torch.no_grad():
        _, rstd = torch.ops.PeakGemm.rms_norm_forward(x, weight, 1e-6)
    torch.library.opcheck(
        torch.ops.PeakGemm.rms_norm_backward.default,
        (torch.randn_like(x), x.detach(), weight.detach(), rstd),
    )


@pytest.mark.parametrize("case", ["dim", "weight_shape", "dtype", "device", "eps"])
def test_invalid(case):
    x = torch.empty(3, 64, device="cuda")
    weight = torch.ones(64, device="cuda")
    eps = 1e-6
    if case == "dim":
        x = x[:, :0]
        weight = weight[:0]
    elif case == "weight_shape":
        weight = weight[:32]
    elif case == "dtype":
        x = x.double()
    elif case == "device":
        weight = weight.cpu()
    elif case == "eps":
        eps = -1
    with pytest.raises(RuntimeError, match="RMSNorm"):
        F.rms_norm(x, weight, eps)


def test_no_double_backward():
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    weight = torch.ones(64, device="cuda", requires_grad=True)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(F.rms_norm(x, weight).sum(), x, create_graph=True)


@pytest.mark.parametrize("case", ["dy_shape", "dy_dtype", "rstd_dtype", "rstd_shape"])
def test_backward_invalid(case):
    x = torch.randn(3, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(64, device="cuda")
    _, rstd = torch.ops.PeakGemm.rms_norm_forward(x, weight, 1e-6)
    dy = torch.randn_like(x)
    if case == "dy_shape":
        dy = dy[:2]
    elif case == "dy_dtype":
        dy = dy.float()
    elif case == "rstd_dtype":
        rstd = rstd.to(x.dtype)
    elif case == "rstd_shape":
        rstd = rstd.reshape(1, 3)
    with pytest.raises(RuntimeError, match="RMSNorm"):
        torch.ops.PeakGemm.rms_norm_backward(dy, x, weight, rstd)


def test_saved_tensor_version():
    x = torch.randn(3, 64, device="cuda", requires_grad=True)
    weight = torch.ones(64, device="cuda", requires_grad=True)
    y = F.rms_norm(x, weight)
    with torch.no_grad():
        weight.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dim", [1, 3, 4, 5, 7, 9, 127, 129, 4097])
def test_nonvectorizable_dim_rejected(dtype, weight_dtype, dim):
    x = torch.randn(3, dim, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.ones(dim, device="cuda", dtype=weight_dtype, requires_grad=True)
    if dim % (16 // x.element_size()) == 0 and dim % (16 // weight.element_size()) == 0:
        # FP32/FP32 dim=4 is a complete vector, not an error.
        y = F.rms_norm(x, weight)
        assert_close(y, reference(x, weight, 1e-6))
        y.sum().backward()
        return
    with pytest.raises((RuntimeError, ValueError), match="VEC_SIZE"):
        F.rms_norm(x, weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("which", ["x", "weight", "dy"])
def test_unaligned_pointers_rejected(dtype, which):
    def make(shape, dtype, offset):
        count = 1
        for size in shape:
            count *= size
        return torch.randn(count + offset, device="cuda", dtype=dtype)[offset:].view(shape)

    x = make((3, 128), dtype, int(which == "x")).requires_grad_()
    weight = make((128,), torch.float32, int(which == "weight")).requires_grad_()
    dy = make((3, 128), dtype, int(which == "dy"))
    assert x.is_contiguous() and weight.is_contiguous() and dy.is_contiguous()
    with pytest.raises((RuntimeError, ValueError), match="vector-aligned"):
        F.rms_norm(x, weight).backward(dy)


def test_mixed_dtype_weight_requires_32b_alignment():
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(132, device="cuda")[4:]
    assert weight.data_ptr() % 16 == 0 and weight.data_ptr() % 32 != 0
    with pytest.raises((RuntimeError, ValueError), match="vector-aligned"):
        F.rms_norm(x, weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale", [1e-12, 1e-4, 1.0, 1e4, 1e12])
@pytest.mark.parametrize("distribution", ["shifted", "constant", "outlier"])
@pytest.mark.parametrize("eps", [0.0, 1e-6])
def test_unnormalized_inputs_against_fp64(dtype, scale, distribution, eps):
    torch.manual_seed(2026)
    # No assumption of zero mean, unit variance, or a narrow dynamic range.
    x = torch.randn(7, 4096, device="cuda")
    if distribution == "shifted":
        x = (x * 0.01 + 1) * scale
    elif distribution == "constant":
        x.fill_(scale)
    else:
        x *= scale * 1e-3
        x[:, 0] = scale
    x = x.to(dtype).requires_grad_()
    weight = torch.randn(4096, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = F.rms_norm(x, weight, eps)
    dx, dw = torch.autograd.grad(y, (x, weight), dy)

    # Convert the already-quantized input to FP64: don't conflate BF16 input
    # quantization with reduction accuracy.
    x64 = x.detach().double().requires_grad_()
    w64 = weight.detach().double().requires_grad_()
    y64 = w64 * x64 * torch.rsqrt(x64.square().mean(-1, keepdim=True) + eps)
    dx64, dw64 = torch.autograd.grad(y64, (x64, w64), dy.double())
    for actual, expected in [(y, y64), (dx, dx64), (dw, dw64)]:
        assert torch.isfinite(actual).all()
        tolerance = 1e-5 if actual.dtype == torch.float32 else 5e-3
        delta = (actual.double() - expected).abs()
        assert delta.max() <= tolerance * expected.abs().max() + 1e-30
        error = delta.norm()
        assert error <= tolerance * expected.norm() + 1e-30, (
            distribution, scale, eps, actual.dtype, error.item(), expected.norm().item()
        )
    # The original autograd expression can overflow its r^3 intermediate
    # for tiny x with eps=0. FP64 above is the oracle for that boundary.
    expected = reference(x, weight, eps)
    ref_dx, ref_dw = torch.autograd.grad(expected, (x, weight), dy)
    assert_close(y, expected)
    if eps > 0 or scale >= 1e-4:
        assert_close(dx, ref_dx)
    assert_close(dw, ref_dw)
