"""Run: python -m pytest tests/torch/test_rmsnorm.py -q."""
import pytest
import torch

from PeakGemm import RMSNorm, rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP GPU required")
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def reference(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + eps)
    return (weight.float() * y).to(x.dtype)


def close(a, b):
    if a.dtype == torch.float32:
        torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-4)
    elif a.dtype == torch.float16:
        torch.testing.assert_close(a, b, rtol=3e-3, atol=3e-3)
    else:
        torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("weight_dtype", DTYPES)
@pytest.mark.parametrize("shape", [
    (1,), (7, 1), (3, 31), (2, 3, 64), (2, 3, 5, 128), (17, 255),
    (3, 256), (3, 257), (5, 1023), (3, 1024), (3, 1025), (3, 2048),
    (3, 2049), (7, 3072), (3, 4096), (3, 4097), (7, 6144), (3, 8192), (3, 8193),
    (3, 16385), (513, 128), (4097, 17), (0, 128), (2, 0, 64),
])
def test_forward_backward(shape, dtype, weight_dtype):
    torch.manual_seed(123)
    x = torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(shape[-1], device="cuda", dtype=weight_dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    dy = torch.randn_like(x)
    y = rms_norm(x, w, 1e-6)
    yr = reference(xr, wr, 1e-6)
    close(y, yr)
    y.backward(dy)
    yr.backward(dy)
    close(x.grad, xr.grad)
    close(w.grad, wr.grad)
    assert y.dtype == x.dtype and y.shape == x.shape
    assert x.grad.dtype == x.dtype and w.grad.dtype == w.dtype


@pytest.mark.parametrize("which", ["x", "weight", "both", "none"])
def test_noncontiguous_and_partial_gradients(which):
    # Non-unit column stride and noncontiguous weight; copies remain in autograd.
    xbase = torch.randn(4, 7, 256, device="cuda", requires_grad=which in ("x", "both"))
    wbase = torch.randn(256, device="cuda", requires_grad=which in ("weight", "both"))
    xr = xbase.detach().clone().requires_grad_(xbase.requires_grad)
    wr = wbase.detach().clone().requires_grad_(wbase.requires_grad)
    x, w = xbase.transpose(0, 1)[..., ::2], wbase[::2]
    y = rms_norm(x, w)
    yr = reference(xr.transpose(0, 1)[..., ::2], wr[::2], 1e-6)
    close(y, yr)
    # Expanded stride-zero grad output.
    if which != "none":
        y.sum().backward()
        yr.sum().backward()
    if xbase.requires_grad:
        close(xbase.grad, xr.grad)
    if wbase.requires_grad:
        close(wbase.grad, wr.grad)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("value", [0.0, 1e-4, 100.0])
@pytest.mark.parametrize("eps", [0.0, 1e-6, 0.1])
def test_values(dtype, value, eps):
    x = torch.full((5, 128), value, device="cuda", dtype=dtype)
    w = torch.linspace(-2, 2, 128, device="cuda", dtype=torch.float32)
    a, b = rms_norm(x, w, eps), reference(x, w, eps)
    torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-2, equal_nan=True)


def test_module_training_state_dict():
    module = RMSNorm(128, 1e-6, device="cuda")
    assert list(module.state_dict()) == ["weight"]
    assert module.weight.dtype == torch.float32
    torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
    x = torch.randn(2, 3, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.01)
    before = module.weight.detach().clone()
    module(x).float().square().mean().backward()
    optimizer.step()
    assert not torch.equal(before, module.weight)
    other = RMSNorm(128, 1e-6, device="cuda")
    other.load_state_dict(module.state_dict())
    with torch.inference_mode():
        close(module(x), other(x))
    module.bfloat16()
    assert module(x).dtype == torch.bfloat16


def test_finite_difference():
    # The op intentionally performs FP32 math, so double gradcheck is unsuitable.
    torch.manual_seed(32)
    x = torch.randn(2, 7, device="cuda", requires_grad=True)
    w = torch.randn(7, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    (rms_norm(x, w) * dy).sum().backward()
    h = 0.001
    for tensor in (x, w):
        for index in range(tensor.numel()):
            plus, minus = tensor.detach().clone(), tensor.detach().clone()
            plus.view(-1)[index] += h
            minus.view(-1)[index] -= h
            if tensor is x:
                fplus, fminus = rms_norm(plus, w.detach()), rms_norm(minus, w.detach())
            else:
                fplus, fminus = rms_norm(x.detach(), plus), rms_norm(x.detach(), minus)
            numerical = ((fplus - fminus) * dy).sum() / (2 * h)
            torch.testing.assert_close(tensor.grad.view(-1)[index], numerical, rtol=5e-3, atol=5e-3)


def test_side_stream_and_graph():
    # Build/warm up outside capture; use the current stream rather than default.
    x = torch.randn(17, 6144, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(6144, device="cuda")
    rms_norm(x, w)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        y = rms_norm(x, w)
    torch.cuda.current_stream().wait_stream(stream)
    close(y, reference(x, w, 1e-6))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = rms_norm(x, w)
    x.normal_()
    graph.replay()
    close(out, reference(x, w, 1e-6))


def test_backward_graph():
    x = torch.randn(17, 128, device="cuda", requires_grad=True)
    w = torch.randn(128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    for _ in range(2):
        rms_norm(x, w).backward(dy)
        x.grad = w.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = rms_norm(x, w)
        y.backward(dy)
    x.detach().normal_()
    graph.replay()
    xr = x.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    reference(xr, wr, 1e-6).backward(dy)
    close(y, reference(x, w, 1e-6))
    close(x.grad, xr.grad)
    close(w.grad, wr.grad)


def test_deterministic_weight_grad():
    x = torch.randn(4097, 128, device="cuda", requires_grad=True)
    w = torch.randn(128, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    rms_norm(x, w).backward(dy)
    dx, dw = x.grad.clone(), w.grad.clone()
    x.grad = w.grad = None
    rms_norm(x, w).backward(dy)
    assert torch.equal(dx, x.grad) and torch.equal(dw, w.grad)


@pytest.mark.parametrize("eps", [-1.0, float("inf"), float("nan"), 1e100])
def test_bad_eps(eps):
    with pytest.raises(ValueError):
        rms_norm(torch.ones(2, 128, device="cuda"), torch.ones(128, device="cuda"), eps)


def test_bad_inputs():
    with pytest.raises(ValueError, match="CUDA/HIP"):
        rms_norm(torch.ones(2, 128), torch.ones(128))
    for shape in [(), (2, 0), (2, 127)]:
        with pytest.raises(ValueError):
            rms_norm(torch.ones(shape, device="cuda"), torch.ones(128, device="cuda"))
    with pytest.raises(ValueError):
        rms_norm(torch.ones(2, 128, device="cuda"), torch.ones(1, 128, device="cuda"))
    with pytest.raises(TypeError):
        rms_norm(torch.ones(2, 128, device="cuda", dtype=torch.float64), torch.ones(128, device="cuda"))
    with pytest.raises(ValueError):
        RMSNorm(0, 1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("cols", [1024, 6144])
def test_contiguous_but_unaligned(dtype, cols):
    # contiguous() alone does not fix nonzero storage-offset alignment.
    xb = torch.randn(3 * cols + 1, device="cuda", dtype=dtype, requires_grad=True)
    wb = torch.randn(cols + 1, device="cuda", dtype=dtype, requires_grad=True)
    xr = xb.detach().clone().requires_grad_()
    wr = wb.detach().clone().requires_grad_()
    x, w = xb[1:].view(3, cols), wb[1:]
    assert x.is_contiguous() and w.is_contiguous()
    y = rms_norm(x, w)
    yr = reference(xr[1:].view(3, cols), wr[1:], 1e-6)
    close(y, yr)
    y.sum().backward()
    yr.sum().backward()
    close(xb.grad, xr.grad)
    close(wb.grad, wr.grad)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two visible GPUs required")
def test_device_guard():
    x = torch.randn(7, 128, device="cuda:1", requires_grad=True)
    w = torch.randn(128, device="cuda:1", requires_grad=True)
    with torch.cuda.device(0):
        y = rms_norm(x, w)
        y.sum().backward()
    close(y, reference(x, w, 1e-6))
    assert y.device == x.device and w.grad.device == w.device


def test_explicit_no_grad():
    x = torch.randn(7, 128, device="cuda", requires_grad=True)
    w = torch.randn(128, device="cuda", requires_grad=True)
    with torch.no_grad():
        y = rms_norm(x, w)
    assert not y.requires_grad
    close(y, reference(x, w, 1e-6))
