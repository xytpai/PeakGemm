import functools
import math
from dataclasses import dataclass

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import PeakGemm


ROTARY_INPUTS_TARGET_BYTES = 8 * 1024**3
CACHE_DIR = "temp"


@dataclass
class _TestArgs:
    dtype: torch.dtype
    m: int
    n: int
    k: int
    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    BLOCK_M_WARPS: int
    BLOCK_N_WARPS: int
    STAGES: int
    SWIZZLE_M: int
    SPLIT_K: int
    HAS_BIAS: bool


@functools.lru_cache(maxsize=None)
def get_hgemm(params):
    return PeakGemm.compile_hgemm(params, CACHE_DIR)


def get_params(args: _TestArgs):
    return PeakGemm.ConstexprParams(
        args.BLOCK_M,
        args.BLOCK_N,
        args.BLOCK_K,
        args.BLOCK_M_WARPS,
        args.BLOCK_N_WARPS,
        args.STAGES,
        args.SWIZZLE_M,
    )


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
    get_hgemm(get_params(args))(c, a, b, args.SPLIT_K, bias)


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
@pytest.mark.parametrize(
    "m, n, k, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, STAGES, SWIZZLE_M, SPLIT_K, HAS_BIAS",
    [
        (16, 64, 256, 16, 64, 64, 1, 2, 2, 8, 1, False),
        (16, 64, 256, 16, 64, 64, 1, 2, 2, 8, 1, True),
        (16, 64, 256, 16, 64, 64, 1, 2, 2, 8, 4, False),
        (16, 64, 256, 16, 64, 64, 1, 2, 2, 8, 4, True),
        (256, 256, 1024, 16, 64, 64, 1, 2, 2, 8, 4, False),
        (256, 256, 1024, 16, 64, 64, 1, 2, 2, 8, 4, True),
        (512, 512, 2048, 128, 128, 32, 2, 4, 3, 8, 1, False),
        (512, 512, 2048, 128, 128, 32, 2, 4, 3, 8, 1, True),
        (128, 256, 1024, 32, 64, 32, 1, 2, 2, 4, 1, False),
        (256, 256, 1024, 64, 64, 32, 2, 2, 3, 2, 4, True),
        (256, 512, 1024, 64, 128, 32, 2, 4, 3, 4, 1, True),
        (512, 256, 1024, 128, 64, 32, 4, 2, 3, 4, 4, False),
        (512, 512, 1024, 128, 128, 16, 2, 4, 4, 1, 1, False),
        (64, 512, 1024, 16, 128, 64, 1, 4, 2, 8, 4, True),
    ],
)
def test_hgemm_acc(
    dtype,
    m,
    n,
    k,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    BLOCK_M_WARPS,
    BLOCK_N_WARPS,
    STAGES,
    SWIZZLE_M,
    SPLIT_K,
    HAS_BIAS,
):
    check_acc(
        _TestArgs(
            dtype,
            m,
            n,
            k,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            BLOCK_M_WARPS,
            BLOCK_N_WARPS,
            STAGES,
            SWIZZLE_M,
            SPLIT_K,
            HAS_BIAS,
        )
    )


# =========================================== benchmark ===========================================


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "m, n, k, BLOCK_M, BLOCK_N, BLOCK_K, BLOCK_M_WARPS, BLOCK_N_WARPS, STAGES, SWIZZLE_M, SPLIT_K, HAS_BIAS",
    [
        (8192, 8192, 8192, 128, 128, 32, 2, 4, 3, 8, 1, True),
    ],
)
def test_hgemm_benchmark(
    dtype,
    m,
    n,
    k,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    BLOCK_M_WARPS,
    BLOCK_N_WARPS,
    STAGES,
    SWIZZLE_M,
    SPLIT_K,
    HAS_BIAS,
):
    benchmark(
        _TestArgs(
            dtype,
            m,
            n,
            k,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            BLOCK_M_WARPS,
            BLOCK_N_WARPS,
            STAGES,
            SWIZZLE_M,
            SPLIT_K,
            HAS_BIAS,
        )
    )
