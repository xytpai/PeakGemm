# PeakGemm

PeakGemm is a small, header-first GPU GEMM library focused on kernel
experimentation and peak throughput.

The current optimized path is an SM80-style Tensor Core HGEMM kernel for
NVIDIA GPUs. It supports FP16 and BF16 inputs, FP32 accumulation, optional
kernel-side bias, asynchronous global-to-shared copies, split-K for small M,
and grouped block swizzling for L2 reuse. The PyTorch API currently launches
the kernel through a JIT-compiled extension without bias.

## Requirements

- Linux
- Python 3.10+
- A CUDA-enabled PyTorch installation
- CUDA Toolkit with `nvcc`
- NVIDIA GPU with SM80 or newer Tensor Core instructions

The PyTorch binding is currently CUDA-only. HIP backend primitives and the
GFX950 architecture layer are present, but the optimized GFX950 HGEMM entry
point is not implemented yet.

## Install

Install the Python package from the repository root:

```bash
python3 -m pip install -v -e . --no-build-isolation
# -i https://pypi.tuna.tsinghua.edu.cn/simple
```

No CUDA extension is built during installation. `compile_hgemm` invokes
PyTorch's JIT extension builder for the requested config. For
cross-compilation, set
`PEAKGEMM_CUDA_ARCH_LIST` explicitly, for example `8.9+PTX` for an RTX 4090
binary with a forward-compatible PTX fallback.

## PyTorch API

The callable returned by `compile_hgemm` computes:

```text
C[M, N] = A[M, K] @ B[N, K].T
```

Inputs and output must be contiguous, 16-byte-aligned CUDA tensors with the
same FP16 or BF16 dtype. Flattened A, B, and C element counts must each fit in
`uint32`. The output tensor is supplied by the caller and updated in place.

```python
import torch
import PeakGemm

params = PeakGemm.ConstexprParams(
    block_m=128,
    block_n=128,
    block_k=32,
    block_m_warps=2,
    block_n_warps=4,
    stages=3,
    swizzle_m=8,
)
gemm = PeakGemm.compile_hgemm(params, cache_dir="./temp")

m = n = k = 4096
a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

gemm(c, a, b, split_k=1)
torch.cuda.synchronize()

reference = a @ b.T
torch.testing.assert_close(c, reference, atol=1.0, rtol=1e-2)
```

The callable keeps split-K workspace per CUDA device and stream. One extension
embeds all four `bias/no-bias × split/non-split` kernel variants. Therefore
`ConstexprParams` contains only tile parameters; `split_k` and optional `bias`
are runtime arguments:

```python
bias = torch.randn((n,), device="cuda", dtype=a.dtype)
gemm(c, a, b, split_k=4, bias=bias)
```

`cache_dir` is required and must be supplied by the caller. Each compiled
extension is stored under `<cache_dir>/<extension-hash>` and reused by PyTorch
when requested again.

## Shape constraints

For selected constexpr parameters, M and N must be divisible by `block_m` and `block_n`.
K must be divisible by `split_k * block_k`, and each split must contain enough
K tiles to fill the configured pipeline. Unsupported shapes raise an
exception instead of silently selecting another config.

## Python accuracy and performance test

Run the parameterized accuracy suite:

```bash
pytest -s tests/torch/test_hgemm_sm80.py -k acc
```

Run the interleaved PeakGemm/PyTorch benchmark:

```bash
pytest -s tests/torch/test_hgemm_sm80.py -k benchmark
```

The benchmark rotates through approximately 8 GiB of inputs, alternates
PeakGemm and native PyTorch launches, and prints CUDA activity collected by
`torch.profiler`.

## C++ tests and benchmarks

Compile and run one test:

```bash
bash build_single.sh tests/gemm/test_hgemm_sm80.cpp
./a.out
```

For a specific CUDA architecture:

```bash
ARCH=sm_89 bash build_single.sh tests/gemm/test_hgemm_sm80.cpp
./a.out
```

Run the complete CUDA test suite:

```bash
bash test_all.sh cuda
```

The GEMM benchmarks perform an accuracy check before timing, followed by
warmup and repeated measurements. They report median latency and TFLOPS for
each shape.

## Repository layout

```text
include/peak_gemm/
  backend/          CUDA/HIP runtime and architecture primitives
  core/             vectors, layouts, shapes, math, and block swizzling
  kernel/           naive GEMM and architecture-specific HGEMM kernels
PeakGemm/           Python JIT API
tests/              core, bandwidth, compute, and GEMM tests
```

## License

See [LICENSE](LICENSE).
