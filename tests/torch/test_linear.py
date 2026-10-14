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
            # FP32 accumulation followed by the same output dtype; reduction order may differ.
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
        # Gradients were compared above. Check SGD uses the computed FP32
        # gradient exactly, without imposing a tighter native-gradient tolerance.
        expected_update = layer.weight.detach() - layer.weight.grad * 1e-3
        optimizer = torch.optim.SGD(layer.parameters(), lr=1e-3)
        optimizer.step()
        torch.testing.assert_close(layer.weight, expected_update, atol=1e-7, rtol=1e-6)


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


@gpu_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("train_x,train_w", [(True, False), (False, True), (True, True)])
def test_linear_backward_uses_peakgemm(dtype, train_x, train_w):
    torch.manual_seed(2026)
    x = torch.randn(2, 256, 256, device="cuda", dtype=dtype, requires_grad=train_x)
    w = torch.randn(256, 256, device="cuda", dtype=dtype, requires_grad=train_w)
    params = [p for p in (x, w) if p.requires_grad]
    dy = torch.randn(2, 256, 256, device="cuda", dtype=dtype).transpose(0, 1).contiguous().transpose(0, 1)
    # Keep forward outside the profiler: these launches must come from backward.
    y = F.linear(x, w)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        grads = torch.autograd.grad(y, params, dy)
        torch.cuda.synchronize()
    kernel = "hgemm_hti_kernel" if torch.version.hip else "hgemm_kernel"
    launches = sum(kernel in event.name for event in prof.events())
    assert launches == int(train_x) + int(train_w), launches
    expected = torch.autograd.grad(native.linear(x, w), params, dy)
    tolerance = 0.15 if dtype == torch.bfloat16 else 0.02
    for actual, ref, parameter in zip(grads, expected, params):
        assert actual.dtype == parameter.dtype
        torch.testing.assert_close(actual, ref, atol=tolerance, rtol=0.03)


@gpu_only
@pytest.mark.parametrize("m,n,k,expected_launches", [
    (256, 256, 256, 2),  # dx and dweight both supported
    (17, 256, 256, 0),   # dx M not tile aligned; dweight K not pipeline aligned
    (256, 256, 65, 0),   # dx/dweight output N not tile aligned
    (16, 256, 256, 1),   # dx uses the small tile; dweight K=16 falls back
    (32, 256, 256, 1),   # dweight K=32 is below the pipeline minimum
])
def test_linear_backward_independent_fallback(m, n, k, expected_launches):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(n, k, device="cuda", dtype=x.dtype, requires_grad=True)
    dy = torch.randn(m, n, device="cuda", dtype=x.dtype)
    y = F.linear(x, w)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        grads = torch.autograd.grad(y, (x, w), dy)
        torch.cuda.synchronize()
    launches = sum("hgemm_kernel" in event.name or "hgemm_hti_kernel" in event.name for event in prof.events())
    assert launches == expected_launches, launches
    expected = torch.autograd.grad(native.linear(x, w), (x, w), dy)
    for actual, ref in zip(grads, expected):
        torch.testing.assert_close(actual, ref, atol=0.15, rtol=0.03)


@gpu_only
def test_linear_backward_inside_autocast_keeps_fp32():
    # Autocast during backward must not downcast a forward that ran in FP32.
    x = torch.randn(256, 256, device="cuda", requires_grad=True)
    w = torch.randn(256, 256, device="cuda", requires_grad=True)
    dy = torch.randn_like(x)
    y = F.linear(x, w)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        grads = torch.autograd.grad(y, (x, w), dy)
    expected = torch.autograd.grad(native.linear(x, w), (x, w), dy)
    for actual, ref in zip(grads, expected):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-5)


@gpu_only
def test_linear_transposed_weight_no_copy():
    x = torch.randn(512, 384, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    storage = torch.randn(384, 768, device="cuda", dtype=x.dtype, requires_grad=True)
    weight = storage.T
    F.linear(x, weight)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        y = F.linear(x, weight)
        torch.cuda.synchronize()
    names = [event.name for event in prof.events()]
    assert not any(name in {"aten::clone", "aten::copy_", "aten::contiguous"} for name in names), names
    ref = native.linear(x, weight)
    torch.testing.assert_close(y, ref, atol=0.15, rtol=0.03)
    dy = torch.randn_like(y)
    grads = torch.autograd.grad(y, (x, storage), dy)
    expected = torch.autograd.grad(ref, (x, storage), dy)
    for actual, expected_grad in zip(grads, expected):
        torch.testing.assert_close(actual, expected_grad, atol=0.2, rtol=0.03)


@gpu_only
@pytest.mark.parametrize("which,expected_clones", [("x", 0), ("weight", 1)])
def test_linear_backward_transpose_copy_count(which, expected_clones):
    x = torch.randn(512, 256, device="cuda", dtype=torch.bfloat16, requires_grad=which == "x")
    w = torch.randn(256, 256, device="cuda", dtype=x.dtype, requires_grad=which == "weight")
    y = F.linear(x, w)
    dy = torch.randn_like(y)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        y.backward(dy)
        torch.cuda.synchronize()
    names = [event.name for event in prof.events()]
    assert sum(name == "aten::clone" for name in names) == expected_clones, names



@gpu_only
@pytest.mark.parametrize("m,n,k", [
    (512, 768, 1024),
    (4096, 4096, 4096),
    (4096, 1024, 4096),
    (4096, 12288, 4096),
])
def test_linear_native_benchmark(m, n, k):
    from statistics import median

    torch.manual_seed(2026)
    x = (torch.randn(m, k, device="cuda") * 0.1).requires_grad_()
    weight = (torch.randn(n, k, device="cuda") * 0.1).requires_grad_()
    dy = torch.randn(m, n, device="cuda", dtype=torch.bfloat16) * 0.1

    def forward(fn):
        # Exit the autocast region each iteration, as in a training step.
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return fn(x, weight)

    pg_y, native_y = forward(F.linear), forward(native.linear)
    pg_grad = torch.autograd.grad(pg_y, (x, weight), dy, retain_graph=True)
    native_grad = torch.autograd.grad(native_y, (x, weight), dy, retain_graph=True)
    torch.testing.assert_close(pg_y, native_y, atol=0.05, rtol=0.03)
    for actual, expected in zip(pg_grad, native_grad):
        assert actual.dtype == expected.dtype == torch.float32
        torch.testing.assert_close(actual, expected, atol=0.05, rtol=0.03)
    del pg_grad, native_grad

    def backward(y):
        return torch.autograd.grad(y, (x, weight), dy, retain_graph=True)

    def forward_backward(fn):
        return torch.autograd.grad(forward(fn), (x, weight), dy)

    functions = {
        "PG forward": lambda: forward(F.linear),
        "native forward": lambda: forward(native.linear),
        "PG backward": lambda: backward(pg_y),
        "native backward": lambda: backward(native_y),
        "PG fwd+bwd": lambda: forward_backward(F.linear),
        "native fwd+bwd": lambda: forward_backward(native.linear),
    }
    for fn in functions.values():
        for _ in range(10):
            fn()
    torch.cuda.synchronize()
    times = {name: [] for name in functions}
    for repeat in range(5):
        order = list(functions.items())
        if repeat % 2:
            order.reverse()
        for name, fn in order:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                fn()
            stop.record()
            stop.synchronize()
            times[name].append(start.elapsed_time(stop) * 1000 / 20)
    print(f"\nLinear FP32 params + BF16 autocast, M/N/K={m}/{n}/{k}")
    for phase in ["forward", "backward", "fwd+bwd"]:
        pg_us, native_us = median(times[f"PG {phase}"]), median(times[f"native {phase}"])
        print(f"  {phase}: PG={pg_us:.1f} us, native={native_us:.1f} us, native/PG={native_us/pg_us:.2f}x")
