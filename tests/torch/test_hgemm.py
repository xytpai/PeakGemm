import math
from dataclasses import dataclass

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import PeakGemm


ROTARY_INPUTS_TARGET_BYTES = 8 * 1024**3
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required"
)


@dataclass
class _TestArgs:
    dtype: torch.dtype
    m: int
    n: int
    k: int
    config: int
    SPLIT_K: int
    HAS_BIAS: bool


def create_inputs(args: _TestArgs):
    a = torch.empty((args.m, args.k), dtype=args.dtype, device="cuda")
    b = torch.empty((args.n, args.k), dtype=args.dtype, device="cuda")
    a.uniform_(-1, 1)
    b.uniform_(-1, 1)
    if args.HAS_BIAS:
        bias = torch.empty((args.n,), dtype=args.dtype, device="cuda")
        bias.uniform_(10, 20)
    else:
        bias = None
    return a, b, bias


def create_outputs(args: _TestArgs):
    return (torch.randn((args.m, args.n), dtype=args.dtype, device="cuda"),)


def ref_func(a, b, bias, c):
    if bias is None:
        torch.mm(a, b.t(), out=c)
    else:
        torch.addmm(bias, a, b.t(), out=c)


def func(a, b, bias, c, args: _TestArgs):
    PeakGemm.hgemm(c, a, b, args.SPLIT_K, bias, config=args.config)


def tensor_nbytes(tensors):
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


def get_rotary_inputs(sample_inputs, sample_outputs):
    slot_bytes = 2 * (tensor_nbytes(sample_inputs) + tensor_nbytes(sample_outputs))
    return max(1, ROTARY_INPUTS_TARGET_BYTES // slot_bytes)


@torch.inference_mode()
def check_acc(args: _TestArgs):
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    maxdiff_out = []
    tolerance = {
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
    }[args.dtype] * math.sqrt(args.k)

    for _ in range(5):
        func(*(inputs + outputs + (args,)))
        ref_func(*(inputs + ref_outputs))
        for output, ref_output in zip(outputs, ref_outputs):
            maxdiff = (output.float() - ref_output.float()).abs().max().item()
            maxdiff_out.append(maxdiff)
            print(maxdiff, flush=True)
            torch.testing.assert_close(
                output,
                ref_output,
                atol=tolerance,
                rtol=tolerance,
                check_dtype=True,
            )
    print(f"\n{args}\nmaxdiff_out:{maxdiff_out}")


@torch.inference_mode()
def benchmark(args: _TestArgs, warmup: int = 500, niters: int = 600):
    sample_inputs = create_inputs(args)
    sample_outputs = create_outputs(args)
    rotary_inputs = get_rotary_inputs(sample_inputs, sample_outputs)
    inputs = [sample_inputs] + [create_inputs(args) for _ in range(rotary_inputs - 1)]
    ref_inputs = [create_inputs(args) for _ in range(rotary_inputs)]
    outputs = [sample_outputs] + [
        create_outputs(args) for _ in range(rotary_inputs - 1)
    ]
    ref_outputs = [create_outputs(args) for _ in range(rotary_inputs)]
    print(
        f"rotary_inputs:{rotary_inputs}, target_bytes:{ROTARY_INPUTS_TARGET_BYTES}, "
        f"warmup:{warmup}, niters:{niters}"
    )

    def run_ref(idx):
        ref_func(*(ref_inputs[idx] + ref_outputs[idx]))

    def run_peak_gemm(idx):
        func(*(inputs[idx] + outputs[idx] + (args,)))

    print("===================== [INTERLEAVED] =====================")
    for i in range(warmup):
        idx = i % rotary_inputs
        if i % 2 == 0:
            run_ref(idx)
            run_peak_gemm(idx)
        else:
            run_peak_gemm(idx)
            run_ref(idx)
        torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(warmup, niters):
            idx = i % rotary_inputs
            if i % 2 == 0:
                run_ref(idx)
                run_peak_gemm(idx)
            else:
                run_peak_gemm(idx)
                run_ref(idx)
            torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("config,split_k", [
    (config, split_k)
    for config in range(4)
    for split_k in [1, 4]
    if not (torch.version.hip and config == 3 and split_k > 1)
])
@pytest.mark.parametrize("has_bias", [False, True])
def test_hgemm_acc(dtype, config, split_k, has_bias):
    check_acc(_TestArgs(dtype, 256, 256, 1024, config, split_k, has_bias))


@pytest.mark.parametrize("config,m,n,k", [
    (0, 16, 64, 64),
    (1, 64, 64, 64),
    (2, 64, 128, 64),
    (3, 256, 256, 128) if torch.version.hip else (3, 128, 128, 64),
])
def test_hgemm_small_acc(config, m, n, k):
    check_acc(_TestArgs(torch.float16, m, n, k, config, 1, True))


def test_hgemm_default_config():
    a, b, bias = create_inputs(_TestArgs(torch.float16, 256, 256, 128, 3, 1, True))
    c = torch.empty((256, 256), device=a.device, dtype=a.dtype)
    assert PeakGemm.hgemm(c, a, b, bias=bias) is c
    torch.testing.assert_close(c, a @ b.T + bias, atol=0.03, rtol=0.01)


@pytest.mark.parametrize("config,split_k", [(0, 4), (3, 1)])
def test_hgemm_streams(config, split_k):
    # Each stream needs its own split-K workspace.
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    results = []
    for stream in streams:
        with torch.cuda.stream(stream):
            args = _TestArgs(torch.float16, 256, 256, 1024, config, split_k, True)
            a, b, bias = create_inputs(args)
            c, = create_outputs(args)
            for _ in range(3):
                PeakGemm.hgemm(c, a, b, split_k, bias, config=config)
            results.append((c, a @ b.T + bias))
    for stream in streams:
        stream.synchronize()
    for c, expected in results:
        torch.testing.assert_close(c, expected, atol=0.1, rtol=0.03)


@pytest.mark.parametrize("config,split_k", [(0, 4), (3, 1)])
def test_hgemm_graph(config, split_k):
    args = _TestArgs(torch.float16, 256, 256, 1024, config, split_k, True)
    a, b, bias = create_inputs(args)
    c, = create_outputs(args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        PeakGemm.hgemm(c, a, b, split_k, bias, config=config)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        PeakGemm.hgemm(c, a, b, split_k, bias, config=config)
    for _ in range(3):
        a.uniform_(-1, 1)
        graph.replay()
        torch.testing.assert_close(c, a @ b.T + bias, atol=0.1, rtol=0.03)


@pytest.mark.parametrize("case,match", [
    ("config", "config"),
    ("split_k", "split_k"),
    ("dtype", "fp16 and bf16"),
    ("shape", "shapes"),
    ("layout", "contiguous"),
    ("bias", "bias"),
    ("m_alignment", "block-tile aligned"),
    ("k_alignment", "pipeline alignment"),
    ("workspace_limit", "workspace is too small"),
])
def test_hgemm_invalid(case, match):
    a = torch.empty((256, 1024), device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)
    c = torch.empty((256, 256), device="cuda", dtype=a.dtype)
    kwargs = {"config": 0}
    if case == "config":
        kwargs["config"] = 4
    elif case == "split_k":
        kwargs["split_k"] = 0
    elif case == "dtype":
        a, b, c = a.float(), b.float(), c.float()
    elif case == "shape":
        c = c[:128]
    elif case == "layout":
        a = a.T.contiguous().T
    elif case == "bias":
        kwargs["bias"] = torch.empty(128, device="cuda", dtype=a.dtype)
    elif case == "m_alignment":
        a, c = a[:17], c[:17]
    elif case == "k_alignment":
        a, b = a[:, :96].contiguous(), b[:, :96].contiguous()
    elif case == "workspace_limit":
        a = torch.empty((2048, 1024), device="cuda", dtype=a.dtype)
        c = torch.empty((2048, 256), device="cuda", dtype=a.dtype)
        kwargs["split_k"] = 4
    with pytest.raises((RuntimeError, ValueError), match=match):
        PeakGemm.hgemm(c, a, b, **kwargs)


def test_hgemm_rejects_cpu():
    a = torch.empty((128, 128), dtype=torch.float16)
    with pytest.raises(RuntimeError, match="GPU tensors"):
        PeakGemm.hgemm(torch.empty_like(a), a, a)



@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("m,n,k", [
    (512, 256, 128),
    (1024, 1024, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
])
def test_hgemm_large_acc(dtype, has_bias, m, n, k):
    check_acc(_TestArgs(dtype, m, n, k, 3, 1, has_bias))


@pytest.mark.skipif(not torch.version.hip, reason="ROCm HTI only")
@pytest.mark.parametrize("m,n,k,split_k,match", [
    (256, 256, 1024, 4, "requires split_k=1"),
    (128, 256, 128, 1, "block-tile aligned"),
    (256, 128, 128, 1, "block-tile aligned"),
    (256, 256, 64, 1, "two K tiles"),
    (256, 256, 192, 1, "two K tiles"),
])
def test_hgemm_hti_invalid(m, n, k, split_k, match):
    args = _TestArgs(torch.float16, m, n, k, 3, split_k, False)
    a, b, bias = create_inputs(args)
    c, = create_outputs(args)
    with pytest.raises((RuntimeError, ValueError), match=match):
        PeakGemm.hgemm(c, a, b, split_k, bias)


@pytest.mark.skipif(not torch.version.hip, reason="ROCm HTI only")
def test_hgemm_hti_dispatch():
    args = _TestArgs(torch.bfloat16, 1024, 1024, 1024, 3, 1, True)
    a, b, bias = create_inputs(args)
    c, = create_outputs(args)
    PeakGemm.hgemm(c, a, b, bias=bias)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        PeakGemm.hgemm(c, a, b, bias=bias)
        torch.cuda.synchronize()
    assert any("hgemm_hti_kernel" in event.key for event in prof.key_averages())


def test_hgemm_benchmark():
    benchmark(_TestArgs(torch.bfloat16, 8192, 8192, 8192, 3, 1, True))
