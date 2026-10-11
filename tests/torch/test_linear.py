import pytest
import torch
from torch.nn import functional as native
from torch.profiler import ProfilerActivity, profile

from PeakGemm import Linear, functional as F


gpu_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def compare(x, weight, bias=None, autocast=False):
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=autocast):
        actual = F.linear(x, weight, bias)
        expected = native.linear(x, weight, bias)
    tol = {torch.float32: 2e-5, torch.float64: 1e-10, torch.float16: 2e-2, torch.bfloat16: 0.15}[actual.dtype]
    torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    assert actual.dtype == expected.dtype
    params = [p for p in (x, weight, bias) if p is not None and p.requires_grad]
    if params:
        dy = torch.randn_like(actual)
        dx = torch.autograd.grad(actual, params, dy)
        ref = torch.autograd.grad(expected, params, dy)
        for a, b, p in zip(dx, ref, params):
            assert a.dtype == p.dtype
            # Native backward should agree exactly up to backend reduction choices.
            torch.testing.assert_close(a, b, atol=tol, rtol=tol)


@gpu_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("m,n,k", [
    (16, 64, 64), (64, 64, 96), (64, 128, 96),
    (256, 256, 128), (512, 256, 512),
    (17, 65, 127), (1, 64, 256),
    (0, 64, 64), (16, 0, 64), (16, 64, 0),
])
def test_linear_accuracy(dtype, m, n, k):
    x = torch.randn(m, k, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(n, k, device="cuda", dtype=dtype, requires_grad=True)
    compare(x, w)


@gpu_only
@pytest.mark.parametrize("shape,n", [((256,), 128), ((2, 128, 256), 512), ((2, 17, 127), 65)])
@pytest.mark.parametrize("bias", [False, True])
def test_linear_autocast(shape, n, bias):
    x = torch.randn(shape, device="cuda", requires_grad=True)
    w = torch.randn(n, shape[-1], device="cuda", requires_grad=True)
    b = torch.randn(n, device="cuda", requires_grad=True) if bias else None
    compare(x, w, b, autocast=True)


@gpu_only
@pytest.mark.parametrize("layout", ["transpose", "offset"])
def test_linear_layout(layout):
    if layout == "transpose":
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16).T
        w = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16).T
    else:
        x = torch.randn(256 * 128 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(256, 128)
        w = torch.randn(256 * 128 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(256, 128)
    compare(x.requires_grad_(), w.requires_grad_())


@gpu_only
@pytest.mark.parametrize("train_x,train_w", [(True, False), (False, True), (True, True)])
def test_linear_training(train_x, train_w):
    layer = Linear(256, 256, bias=False, device="cuda")
    layer.weight.requires_grad_(train_w)
    ref = layer.weight.detach().clone().requires_grad_(train_w)
    x = torch.randn(256, 256, device="cuda", requires_grad=train_x)
    ref_x = x.detach().clone().requires_grad_(train_x)
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y, expected = layer(x), native.linear(ref_x, ref)
        # Fixed upstream gradients isolate gradient accumulation from forward rounding.
        dy = torch.randn_like(y)
        y.backward(dy)
        expected.backward(dy)
    for p, q in [(x, ref_x), (layer.weight, ref)]:
        if p.requires_grad:
            assert p.grad.dtype == torch.float32
            torch.testing.assert_close(p.grad, q.grad, atol=0.1, rtol=0.03)
    if train_w:
        optimizer = torch.optim.SGD(layer.parameters(), lr=1e-3)
        optimizer.step()
        torch.testing.assert_close(layer.weight, ref - ref.grad * 1e-3, atol=1e-4, rtol=1e-3)


@gpu_only
def test_linear_dispatch():
    x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(256, 256, device="cuda", dtype=x.dtype, requires_grad=True)
    F.linear(x, w).sum().backward()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        F.linear(x, w).sum().backward()
        torch.cuda.synchronize()
    keys = [event.key for event in prof.key_averages()]
    kernel = "hgemm_hti_kernel" if torch.version.hip else "hgemm_kernel"
    assert any(kernel in key for key in keys), keys


@gpu_only
def test_linear_compile_and_opcheck():
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(256, 256, device="cuda", dtype=x.dtype, requires_grad=True)
    torch.library.opcheck(torch.ops.PeakGemm.linear.default, (x, w))
    compiled = torch.compile(F.linear, fullgraph=True, dynamic=True)
    for rows in [256, 17]:
        x = torch.randn(rows, 256, device="cuda", requires_grad=True)
        weight = torch.randn(256, 256, device="cuda", requires_grad=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = compiled(x, weight)
            ref = native.linear(x, weight)
        torch.testing.assert_close(y, ref, atol=0.15, rtol=0.03)
        dy = torch.randn_like(y)
        dx, dw = torch.autograd.grad(y, (x, weight), dy)
        rx, rw = torch.autograd.grad(ref, (x, weight), dy)
        torch.testing.assert_close(dx, rx, atol=0.1, rtol=0.03)
        torch.testing.assert_close(dw, rw, atol=0.1, rtol=0.03)


@gpu_only
def test_linear_graph():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        w = torch.randn(256, 256, device="cuda", dtype=x.dtype, requires_grad=True)
        dy = torch.randn_like(x)
        for _ in range(3):
            y = F.linear(x, w)
            torch.autograd.grad(y, (x, w), dy)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = F.linear(x, w)
        dx, dw = torch.autograd.grad(y, (x, w), dy)
    with torch.cuda.stream(stream):
        for _ in range(3):
            with torch.no_grad():
                x.normal_()
            graph.replay()
            ref = native.linear(x, w)
            rx, rw = torch.autograd.grad(ref, (x, w), dy)
            torch.testing.assert_close(y, ref, atol=0.15, rtol=0.03)
            torch.testing.assert_close(dx, rx, atol=0.1, rtol=0.03)
            torch.testing.assert_close(dw, rw, atol=0.1, rtol=0.03)
    stream.synchronize()


def test_linear_cpu_fallback():
    x = torch.randn(2, 3, 7, requires_grad=True)
    w = torch.randn(5, 7, requires_grad=True)
    compare(x, w)


@gpu_only
def test_linear_autocast_cache(monkeypatch):
    # The same weight is used twice in one autocast region, as in recurrent use.
    layer = Linear(256, 256, bias=False, device="cuda")
    x = torch.randn(256, 256, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
        y1, y2 = layer(x), layer(x)
    (y1.float().sum() + y2.float().sum()).backward()
    assert layer.weight.grad.dtype == torch.float32
    assert torch.isfinite(layer.weight.grad).all()
