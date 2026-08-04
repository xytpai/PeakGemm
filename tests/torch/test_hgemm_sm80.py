import argparse
import functools
import math

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import PeakGemm


ACCURACY_SHAPES = (
    (16, 64, 256),
    (256, 256, 1024),
    (512, 512, 2048),
)
BASE_TOLERANCE = {
    torch.float16: 2.0e-3,
    torch.bfloat16: 2.0e-2,
}
SMALL_PARAMS = PeakGemm.ConstexprParams(16, 64, 64, 1, 2, 2, 8)
LARGE_PARAMS = PeakGemm.ConstexprParams(128, 128, 32, 2, 4, 3, 8)
CACHE_DIR = "temp"


@functools.lru_cache(maxsize=None)
def get_hgemm(params):
    return PeakGemm.compile_hgemm(params, CACHE_DIR)


def get_hgemm_for_shape(m):
    return (
        (get_hgemm(SMALL_PARAMS), 4)
        if m <= 256
        else (get_hgemm(LARGE_PARAMS), 1)
    )


def make_inputs(m, n, k, dtype, device):
    a = torch.empty((m, k), dtype=dtype, device=device).uniform_(-1.0, 1.0)
    b = torch.empty((n, k), dtype=dtype, device=device).uniform_(-1.0, 1.0)
    return a, b


@torch.inference_mode()
def check_accuracy(device):
    torch.manual_seed(2026)
    print("accuracy")
    for dtype in (torch.float16, torch.bfloat16):
        for m, n, k in ACCURACY_SHAPES:
            a, b = make_inputs(m, n, k, dtype, device)
            hgemm, split_k = get_hgemm_for_shape(m)
            for has_bias in (False, True):
                bias = (
                    torch.randn((n,), dtype=dtype, device=device)
                    if has_bias
                    else None
                )
                actual = torch.empty((m, n), dtype=dtype, device=device)
                hgemm(
                    actual, a, b, split_k=split_k, bias=bias)
                torch.cuda.synchronize(device)
                expected = torch.mm(a, b.T)
                if bias is not None:
                    expected += bias
                torch.cuda.synchronize(device)
                max_diff = (
                    actual.float() - expected.float()
                ).abs().max().item()
                tolerance = BASE_TOLERANCE[dtype] * math.sqrt(k)
                if max_diff > tolerance:
                    raise AssertionError(
                        f"{dtype} {m}x{n}x{k} bias={has_bias}: "
                        f"max_diff={max_diff:.6g}, "
                        f"tolerance={tolerance:.6g}")
                print(
                    f"  {str(dtype).removeprefix('torch.'):8s} "
                    f"{m:5d} {n:5d} {k:5d}  bias={has_bias}  "
                    f"max_diff={max_diff:.6g}  "
                    f"tolerance={tolerance:.6g}")


def device_time_us(event):
    value = getattr(event, "device_time_total", 0.0)
    if value == 0.0:
        value = getattr(event, "cuda_time_total", 0.0)
    return float(value)


@torch.inference_mode()
def profile_performance(m, n, k, dtype, device, warmup, iterations, trace):
    a, b = make_inputs(m, n, k, dtype, device)
    peak_output = torch.empty((m, n), dtype=dtype, device=device)
    native_output = torch.empty_like(peak_output)
    hgemm, split_k = get_hgemm_for_shape(m)

    def peak_gemm():
        hgemm(peak_output, a, b, split_k=split_k)

    def native_gemm():
        torch.mm(a, b.T, out=native_output)

    for _ in range(warmup):
        peak_gemm()
        native_gemm()
    torch.cuda.synchronize(device)

    with profile(
        activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
        record_shapes=False,
        profile_memory=False,
    ) as profiler:
        for iteration in range(iterations):
            calls = (
                (("peak_gemm", peak_gemm), ("torch_native", native_gemm))
                if iteration % 2 == 0
                else (("torch_native", native_gemm), ("peak_gemm", peak_gemm))
            )
            for label, operation in calls:
                with record_function(label):
                    operation()
        torch.cuda.synchronize(device)

    events = {event.key: event for event in profiler.key_averages()}
    peak_us = device_time_us(events["peak_gemm"]) / iterations
    native_us = device_time_us(events["torch_native"]) / iterations
    if peak_us <= 0.0 or native_us <= 0.0:
        raise RuntimeError("torch.profiler did not record CUDA time")

    operations = 2.0 * m * n * k
    peak_tflops = operations / (peak_us * 1.0e6)
    native_tflops = operations / (native_us * 1.0e6)
    print("\nperformance (torch.profiler CUDA time)")
    print(f"  shape:  {m} x {n} x {k}")
    print(f"  dtype:  {str(dtype).removeprefix('torch.')}")
    print(f"  peak:   {peak_us / 1000.0:.6f} ms  {peak_tflops:.3f} TFLOPS")
    print(f"  native: {native_us / 1000.0:.6f} ms  {native_tflops:.3f} TFLOPS")
    print(f"  ratio:  {native_us / peak_us:.4f}x")

    if trace:
        profiler.export_chrome_trace(trace)
        print(f"  trace:  {trace}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check PeakGemm accuracy and profile it against torch.mm")
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trace", help="optional Chrome trace output path")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    major, minor = torch.cuda.get_device_capability(device)
    if major < 8:
        raise RuntimeError("SM80 or newer GPU is required")
    print(
        f"device: {torch.cuda.get_device_name(device)} "
        f"(sm{major}{minor})")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    if not args.skip_accuracy:
        check_accuracy(device)
    if not args.skip_profile:
        profile_performance(
            args.m, args.n, args.k, dtype, device,
            args.warmup, args.iterations, args.trace)


if __name__ == "__main__":
    main()
