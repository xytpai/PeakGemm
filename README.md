# PeakGemm

GPU GEMM and fused RMSNorm for NVIDIA SM80-compatible GPUs and AMD GFX950.
Header-based C++ kernels, with a CMake-built PyTorch package. No runtime JIT.

## Install

Requires Linux, Python 3.10+, PyTorch 2.4+, and a matching CUDA/ROCm toolkit
with a C++20-capable compiler.

```bash
pip install "cmake>=3.21" ninja
pip install -v -e . --no-build-isolation
```

The build selects CUDA or HIP from your PyTorch installation and uses
Release (`-O3`). Import loads the compiled library through `torch.ops.load_library`.

- Default targets: SM80 + PTX on CUDA; GFX950 on ROCm.
- Set `MAX_JOBS` for build parallelism (default 4).
- Override architecture with `CMAKE_ARGS`, not `TORCH_CUDA_ARCH_LIST`:

```bash
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" pip install -v -e . --no-build-isolation
```

## PyTorch training

```python
import torch
import PeakGemm as pg

norm = pg.RMSNorm(256, eps=1e-6, device="cuda")
linear = pg.Linear(256, 256, bias=False, device="cuda")
x = torch.randn(2, 128, 256, device="cuda", requires_grad=True)

with torch.autocast("cuda", dtype=torch.bfloat16):
    y = linear(norm(x))
y.float().square().mean().backward()
```

To reuse existing module parameters:

```python
from PeakGemm import functional as F

y = F.rms_norm(x, weight, eps=1e-6)
y = F.linear(x, weight, bias=None)
```

- **Linear:** supported FP16/BF16 forward and backward GEMMs use PG; other
  shapes/dtypes, CPU, and biased layers use native PyTorch. Large aligned ROCm
  shapes use HTI with **2×4 M/N waves**. NN/NT read B directly, including
  transpose views; backward may still copy transposed A. Speedups are not guaranteed.
- **RMSNorm:** GPU only, FP32 accumulation; FP32/FP16/BF16 input and weight may
  have different dtypes. Inputs must be contiguous and vector-aligned, with
  `1 <= dim <= 65536` divisible by both dtype vector widths (`16 / sizeof(T)`).
  BF16 input + FP32 weight requires 32B weight alignment. Invalid layouts raise
  errors—no padding or scalar fallback. Use positive `eps` for training;
  double backward is unsupported.
- Autograd, autocast, `torch.compile`, and graph capture are supported.
  Module parameter names match the native equivalents; FP32 master parameters
  retain FP32 gradients.

## Explicit HGEMM

In-place **inference/out API**, without autograd: NT (`A @ B.T`, default) or NN (`A @ B`), plus optional bias.

```python
import torch
import PeakGemm as pg

a = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn(256, 512, device="cuda", dtype=a.dtype)
c = torch.empty(256, 256, device="cuda", dtype=a.dtype)
pg.hgemm(c, a, b, config=3, split_k=1)  # NT; optional bias=[N]
pg.hgemm(c, a, b.T, transpose_b=False)  # NN; transpose view, no copy
```

Four configurations are compiled in [csrc/hgemm.cpp](csrc/hgemm.cpp):

| config | CUDA tile M/N/K | ROCm tile M/N/K | Stages / schedule |
| ---: | --- | --- | --- |
| 0 | 16/64/64 | 16/64/64 | 2 |
| 1 | 64/64/32 | 64/64/32 | 3 |
| 2 | 64/128/32 | 64/128/32 | 3 |
| 3 (default) | 128/128/32 | 256/256/64 | CUDA: 3; ROCm: HTI 2×4 |

A/C must be contiguous; B may be contiguous or a contiguous matrix's transpose
view. All must be 16B-aligned, on the same GPU and FP16/BF16 dtype.
`transpose_b=True` takes B[N,K]; `False` takes B[K,N]. Both use the same kernel
template (compile-time B layout), with no separate transpose kernel.
C must not overlap inputs. M/N must be tile multiples. Standard kernels require
K divisible by `split_k * block_k`, with at least `stages - 1` blocks per partition.
Split-K accepts 1–65535 partitions and at most 256 output tiles when enabled.
ROCm HTI requires K divisible by 128 and `split_k=1`. Invalid shapes raise errors.
Warm up split-K on the intended stream before graph capture.

## Tests

```bash
# C++: auto-selects nvcc, otherwise hipcc; BACKEND=cuda/rocm overrides.
bash build_single.sh tests/core/test_rms_norm.cpp && ./a.out
bash build_single.sh tests/gemm/test_hgemm_gfx950.cpp && ./a.out
# CUDA GEMM: use tests/gemm/test_hgemm_sm80.cpp instead.
bash test_all.sh

# Python accuracy, autograd and integration checks; then GEMM benchmark.
pytest -q tests/torch -k "not benchmark"
pytest -s tests/torch/test_hgemm.py -k benchmark
```

Sources: [kernels](include/peak_gemm/kernel), [GPU runtime](include/peak_gemm/backend/runtime.hpp),
[PyTorch API](PeakGemm/functional.py). License: [MIT](LICENSE).
