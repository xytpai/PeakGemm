# PeakGemm

PeakGemm is a small, header-first GPU GEMM library focused on kernel
experimentation and peak throughput.

The current optimized path is an SM80-style Tensor Core HGEMM kernel for
NVIDIA GPUs. It supports FP16 and BF16 inputs, FP32 accumulation, optional
kernel-side bias, asynchronous global-to-shared copies, split-K for small M,
and grouped block swizzling for L2 reuse. The PyTorch API currently launches
the kernel without bias.

## Requirements

- Linux
- Python 3.10+
- CMake 3.18+
- A CUDA-enabled PyTorch installation
- CUDA Toolkit with `nvcc`
- NVIDIA GPU with SM80 or newer Tensor Core instructions

The PyTorch binding is currently CUDA-only. HIP backend primitives and the
GFX950 architecture layer are present, but the optimized GFX950 HGEMM entry
point is not implemented yet.

## Install

Build the PyTorch extension from the repository root:

```bash
rm -rf build libPeakGemm.so libPeakGemm_device.so
python3 -m pip install -v -e . --no-build-isolation
```

The build detects every visible GPU capability. For cross-compilation, set
`PEAKGEMM_CUDA_ARCH_LIST` explicitly, for example `8.9+PTX` for an RTX 4090
binary with a forward-compatible PTX fallback.

The build is defined entirely under `bindings/torch/`; the Python package
loads the resulting `libPeakGemm.so`.

## PyTorch API

`gemm_peak` computes:

```text
C[M, N] = A[M, K] @ B[N, K].T
```

Inputs and output must be contiguous CUDA tensors with the same FP16 or BF16
dtype. The output tensor is supplied by the caller and updated in place.

```python
import torch
import PeakGemm

m = n = k = 4096
a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

PeakGemm.gemm_peak(c, a, b)
torch.cuda.synchronize()

reference = a @ b.T
torch.testing.assert_close(c, reference, atol=1.0, rtol=1e-2)
```

The wrapper keeps split-K semaphore storage per CUDA device and stream.

## Shape constraints

The current dispatcher uses two kernel configurations:

- `M <= 256`: M must be divisible by 16, N by 64, and K by 256.
- `M > 256`: M and N must be divisible by 128; K must be at least 48 and
  divisible by 16.

Unsupported shapes raise an exception instead of silently using a fallback.

## Python accuracy and performance test

Run the accuracy suite and compare PeakGemm directly with `torch.mm` using
`torch.profiler` CUDA activity:

```bash
python3 tests/torch/test_hgemm_sm80.py
```

Select a shape, dtype, or export a Chrome trace:

```bash
python3 tests/torch/test_hgemm_sm80.py \
    --m 8192 --n 8192 --k 8192 --dtype bf16 \
    --iterations 20 --trace hgemm_trace.json
```

The profile alternates PeakGemm and native PyTorch launches in one profiler
session, then reports average CUDA latency, TFLOPS, and their performance
ratio.

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
bindings/torch/     PyTorch operator and standalone CMake build
PeakGemm/           Python API
tests/              core, bandwidth, compute, and GEMM tests
```

## License

See [LICENSE](LICENSE).
