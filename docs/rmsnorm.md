# RMSNorm: unified CUDA/HIP implementation

Implements standard RMSNorm from [ModelX](https://github.com/xytpai/modelx/blob/f45e8c626524849dca5d47c7a5ccec9de146e6b0/model/transformer.py#L14):

    z = x.float()
    z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + eps)
    y = (weight.float() * z).to(x.dtype)

Not Gemma RMSNorm: no `1 + weight`, residual add, or quantization.
No aiter or Triton dependency.

## Python / ModelX

Install with `python -m pip install -e .` or build a wheel with
`python -m pip wheel . --no-deps`. The first GPU call JIT-compiles using
PyTorch's CUDA/ROCm toolchain, Ninja and a C++ compiler. Import does not compile.
HIP-generated source/header files are staged in the torch extension cache,
not written into the checkout or installed package.

    import torch
    from PeakGemm import RMSNorm, rms_norm

    norm = RMSNorm(3072, eps=1e-6, device="cuda")
    x = torch.randn(2, 128, 3072, device="cuda",
                    dtype=torch.bfloat16, requires_grad=True)
    y = norm(x)
    y.float().square().mean().backward()
    y = rms_norm(x, norm.weight, eps=1e-6)

In ModelX, replace the local RMSNorm class definition with
`from PeakGemm import RMSNorm`. Positional constructor `(dim, eps)`, `eps`,
ones-initialized `weight`, and state-dict keys match the original.
Optional device/dtype constructor arguments are additions.
`.to(...)`, `.bfloat16()` and ModelX `Transformer.bf16()` work normally.
This change does not modify ModelX itself.

### Contract and limits

- GPU only: CUDA SM80+, HIP gfx942/gfx950/gfx1250.
- `x[..., N]`, `weight[N]`, N > 0, arbitrary rank and empty leading dimensions.
  Rows and columns each fit INT32_MAX; address arithmetic uses 64 bits.
- Input and weight independently support FP16/BF16/FP32.
  Output shape/dtype matches input; gradients match their corresponding inputs.
- FP32 statistics, multiplication and gradient reductions. Reduction ordering
  may differ from eager PyTorch; agreement is tolerance-based, not bitwise.
- First-order autograd for dx/dweight, including either gradient alone.
  Higher-order derivatives, forward-mode AD, vmap and
  `torch.compile(fullgraph=True)` are not supported by this binding.
- Noncontiguous inputs/grad output are copied to contiguous buffers.
- eps must be finite, nonnegative and representable in FP32.
  eps=0 with a zero row follows the reference's NaN behavior.
- Current device/stream and CUDA/HIP graph capture are supported.
  Warm up once before capture to finish JIT compilation.

## Standalone C++

Header: `include/peak_gemm/kernel/rmsnorm.hpp`, also included by
`peak_gemm/peak_gemm.hpp`. The kernel header has no PyTorch dependency.

    peak_gemm::kernel::rmsnorm_forward<T, W>(
        x, weight, y, rstd, M, N, eps, stream);

    int parts = peak_gemm::kernel::rmsnorm_backward_partitions(M);
    peak_gemm::kernel::rmsnorm_backward<T, W>(
        dy, x, weight, rstd, dx, dw, workspace, M, N, stream);

T/W: float, __half, __bfloat16. All tensors are contiguous.
rstd is M FP32 values, or nullptr for inference.
Workspace is parts*N FP32 values and is needed only when dw is requested.
dx/dw may independently be nullptr. Inputs/outputs must not overlap.
Caller owns device/stream selection, buffers, and launch-error checks.
Empty rows are a forward no-op and yield zero dw in backward.

Both backends instantiate the same templates:
- N <= 256: one warp/wave per row, multiple rows/block.
- Aligned N <= 8192: four-element vector loads/stores, 256-thread block/row.
- Odd/unaligned dimensions: masked scalar register-resident path.
- Wider dimensions: bounded-register two-pass fallback.
- Backward: block/row for dx; deterministic two-stage FP32 dw reduction,
  no atomics, at most 128*N*sizeof(float) workspace.

CUDA uses 32-lane warps. AMD device compilation selects its actual wave size,
including multi-architecture builds. Backward is a portable baseline and is
not a fully tuned training kernel.

## Tests and benchmark

    bash build_single.sh tests/norm/test_rmsnorm.cpp && ./a.out

On NVIDIA:

    BACKEND=cuda ARCH=sm_80 bash build_single.sh tests/norm/test_rmsnorm.cpp
    ./a.out

PyTorch and optional upstream ModelX integration:

    python -m pytest tests/torch/test_rmsnorm.py -q
    MODELX_ROOT=/path/to/modelx python -m pytest tests/torch/test_rmsnorm_modelx.py -q

GPU timing (optional --aiter compares standard C++ RMSNorm forward):

    PYTHONPATH=. python tests/torch/bench_rmsnorm.py --rows 1 128 4096 8192 --cols 128 2048 3072 4096
    PYTHONPATH=. python tests/torch/bench_rmsnorm.py --aiter

test_all.sh includes tests/norm/*.cpp.

### Validation

Host: gfx950, 256 CUs, physical GPU 7; PyTorch 2.10.0 + ROCm 7.2.4.

- Python operator + ModelX integration: 266 passed, 1 skipped
  (requires a second visible GPU).
- Standalone C++: 74 cases passed.
- Existing test_core, test_data, test_reduce_kernel: passed;
  test_data's multi-GPU subtest skipped with one visible GPU.
- gfx1250 wave32: standalone test compiled, not executed.
- Wheel installed in an isolated directory: BF16 forward/backward passed
  outside the checkout; sources and headers are included.
- Upstream ModelX two-layer Transformer FP32/BF16: logits and all parameter
  gradients agree with the original under test tolerances.
- **CUDA compilation/execution unavailable on this host. NVIDIA validation
  remains outstanding.** The algorithm and API are shared.

## Performance snapshot

BF16 input/weight, eps=1e-6; GPU event times in microseconds, warm-cache
graph replay, 64 calls/graph, 10 replays/sample, median of 9 samples.
Allocation, Python dispatch, JIT and stride-copy costs are excluded.
Not an end-to-end training measurement. aiter is standard non-Gemma RMSNorm.

| M | N | Forward | Forward + rstd | Backward dx+dw | aiter C++ forward |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 1.98 | 2.01 | 5.07 | 1.72 |
| 1 | 2048 | 2.38 | 2.40 | 6.49 | 1.78 |
| 1 | 3072 | 2.62 | 2.64 | 7.23 | 1.82 |
| 1 | 4096 | 2.93 | 2.94 | 7.93 | 1.83 |
| 1 | 6144 | 3.45 | 3.47 | 9.43 | 1.90 |
| 128 | 128 | 2.25 | 2.28 | 8.56 | 1.93 |
| 128 | 2048 | 2.62 | 2.75 | 10.15 | 1.92 |
| 128 | 3072 | 2.96 | 3.11 | 11.11 | 2.03 |
| 128 | 4096 | 3.20 | 3.35 | 11.66 | 2.04 |
| 128 | 6144 | 3.84 | 3.89 | 13.07 | 2.21 |
| 4096 | 128 | 2.54 | 2.62 | 37.65 | 2.51 |
| 4096 | 2048 | 7.13 | 7.15 | 58.67 | 5.91 |
| 4096 | 3072 | 10.51 | 10.51 | 65.37 | 9.65 |
| 4096 | 4096 | 13.87 | 13.88 | 80.20 | 12.73 |
| 4096 | 6144 | 18.57 | 18.62 | 95.56 | 17.84 |
| 8192 | 128 | 3.08 | 3.17 | 43.68 | 3.47 |
| 8192 | 2048 | 13.94 | 13.96 | 80.31 | 12.77 |
| 8192 | 3072 | 18.39 | 18.52 | 93.10 | 17.45 |
| 8192 | 4096 | 23.34 | 23.36 | 113.63 | 22.07 |
| 8192 | 6144 | 32.54 | 32.52 | 158.05 | 31.40 |

This portable baseline is **not faster than aiter on every shape**. Small-batch latency and backward retain tuning headroom. Re-run the included benchmark on the deployment GPU.
