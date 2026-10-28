"""C++ CUDA/HIP RMSNorm GPU benchmark (run from the repository root).

python -m tests.torch.bench_rmsnorm is not required; use:
PYTHONPATH=. python tests/torch/bench_rmsnorm.py --rows 1 128 4096 --cols 128 6144

Optional --aiter compares the standard (not Gemma) aiter C++ RMSNorm forward.
"""
import argparse
import csv
import statistics
import sys

import torch

from PeakGemm.rmsnorm import _get_extension


def reference(x, w, eps):
    y = x.float()
    return (w.float() * (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps))).to(x.dtype)


def bench(fn, nodes, samples):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            fn()
    for _ in range(3):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    values = []
    for _ in range(samples):
        start.record()
        for _ in range(10):
            graph.replay()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000 / (nodes * 10))
    return statistics.median(values), min(values), max(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 512, 1024, 4096, 8192])
    parser.add_argument("--cols", type=int, nargs="+", default=[128, 2048, 3072, 4096])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--aiter", action="store_true")
    args = parser.parse_args()
    if args.nodes <= 0 or args.samples <= 0 or any(m < 1 for m in args.rows) or any(n < 1 for n in args.cols):
        parser.error("rows, cols, samples and nodes must be positive")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(1)
    torch.set_num_threads(4)
    print(f"# GPU={torch.cuda.get_device_name()} arch={torch.cuda.get_device_properties(0)}", file=sys.stderr)
    print(f"# torch={torch.__version__} hip={torch.version.hip}; warm-cache HIP/CUDA graph; GPU events", file=sys.stderr)
    writer = csv.writer(sys.stdout)
    writer.writerow(["M", "N", "dtype", "operation", "median_us", "min_us", "max_us"])
    ext = _get_extension(torch.device("cuda"))
    for m in args.rows:
        for n in args.cols:
            x = torch.randn(m, n, device="cuda", dtype=dtype)
            w = torch.randn(n, device="cuda", dtype=dtype)
            dy = torch.randn_like(x)
            y, rstd = ext.forward(x, w, 1e-6, True)
            tol = 2e-2 if dtype == torch.bfloat16 else 3e-3 if dtype == torch.float16 else 2e-4
            torch.testing.assert_close(y, reference(x, w, 1e-6), rtol=tol, atol=tol)
            functions = [
                ("forward", lambda: ext.forward(x, w, 1e-6, False)),
                ("forward_save_rstd", lambda: ext.forward(x, w, 1e-6, True)),
                ("backward_dx_dw", lambda: ext.backward(dy, x, w, rstd, True, True)),
            ]
            if args.aiter:
                from aiter.ops.rmsnorm import rmsnorm
                out = torch.empty_like(x)
                functions.append(("aiter_cpp_forward", lambda: rmsnorm(out, x, w, 1e-6)))
            for name, fn in functions:
                result = bench(fn, args.nodes, args.samples)
                writer.writerow([m, n, args.dtype, name, *(f"{v:.3f}" for v in result)])
                sys.stdout.flush()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
