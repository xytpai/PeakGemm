import time
import torch
import argparse
import functools
import numpy as np
import warnings
warnings.simplefilter('once')
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from dataclasses import dataclass
from abc import ABC, abstractmethod

import PeakGemm


@dataclass
class Args:
    dtype: torch.dtype
    m: int
    n: int
    k: int


def create_inputs(args):
    a = torch.empty((args.m, args.k), dtype=args.dtype, device='cuda')
    a.uniform_(-1, 1)
    b = torch.empty((args.n, args.k), dtype=args.dtype, device='cuda')
    b.uniform_(-1, 1)
    return (a, b)


def create_outputs(args):
    c = torch.randn((args.m, args.n), dtype=args.dtype, device='cuda')
    return (c,)


def ref_func(a, b, c):
    F.linear(a, b, out=c)


def func(a, b, c):
    if a.dtype == torch.float:
        warnings.warn('NOTE: The SGEMM has not been optimized. It\'s treated as a reference path.')
        PeakGemm.sgemm_peak(c, a, b)


def benchmark(args, func, ref_func, warmup=10, niters=50, sole_inputs=False):
    inputs = create_inputs(args)
    outputs = create_outputs(args)
    ref_outputs = create_outputs(args)
    inouts = inputs + outputs
    ref_inouts = inputs + ref_outputs
    for i in range(5):
        func(*inouts)
        ref_func(*ref_inouts)
        for output, ref_output in zip(outputs, ref_outputs):
            is_allclose = torch.allclose(output, ref_output, atol=1e-2, rtol=1e-2)
            # print(output)
            # print(ref_output)
            maxdiff_out = (output - ref_output).abs().max()
            print(f"maxdiff_out:{maxdiff_out}")
            # assert is_allclose == True

    niters_ = niters if not sole_inputs else 1
    inputs = [create_inputs(args) for i in range(niters_)]
    ref_inputs = [create_inputs(args) for i in range(niters_)]
    outputs = [create_outputs(args) for i in range(niters_)]
    ref_outputs = [create_outputs(args) for i in range(niters_)]

    # get ref_func perf
    print("===================== [REF] =====================")
    for i in range(warmup):
        idx = i % niters_
        ref_func(*(ref_inputs[idx] + ref_outputs[idx]))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(warmup, niters):
            idx = i % niters_
            ref_func(*(ref_inputs[idx] + ref_outputs[idx]))
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

    # get func perf
    print("===================== [PeakGemm] =====================")
    for i in range(warmup):
        idx = i % niters_
        func(*(inputs[idx] + outputs[idx]))
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    with profile(activities=[ProfilerActivity.CUDA], ) as prof:
        for i in range(warmup, niters):
            idx = i % niters_
            func(*(inputs[idx] + outputs[idx]))
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Examples")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--dtype", type=str, required=True)
    args = parser.parse_args()
    print(f"run: {__file__}, args: {args}")
    dtype_convert = {'f32': torch.float, 'f16': torch.half, 'bf16': torch.bfloat16}
    args.dtype = dtype_convert[args.dtype]
    args = Args(**vars(args))
    benchmark(args, func, ref_func)
    # python3 test/test_gemm.py --m=2048 --n=2048 --k=2048 --dtype=f32
    # python3 test/test_gemm.py --m=4096 --n=4096 --k=4096 --dtype=f32
    # python3 test/test_gemm.py --m=8192 --n=8192 --k=8192 --dtype=f32
    # python3 test/test_gemm.py --m=32 --n=384 --k=7168 --dtype=f32
    # python3 test/test_gemm.py --m=32 --n=7168 --k=2048 --dtype=f32
    # python3 test/test_gemm.py --m=32 --n=384 --k=16384 --dtype=f32
