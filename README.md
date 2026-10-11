# PeakGemm

PeakGemm is a compact, header-first GPU GEMM library for studying low-level
kernel design and approaching hardware peak throughput on NVIDIA CUDA and AMD
ROCm.

It currently provides:

- FP16 and BF16 inputs with FP32 accumulation.
- Optional fused bias: `C = A @ B.T + bias`.
- Pipelined global-to-shared copies, shared-memory swizzling, and split-K.
- An SM80-style Tensor Core path for NVIDIA GPUs.
- Standard and hand-scheduled HTI MFMA paths for AMD GFX950, including AGPR
  accumulation.
- Fused RMSNorm forward/backward for CUDA and ROCm, with PyTorch autograd.
- Standalone C++ tests/benchmarks and a CMake-built PyTorch API.

## Supported targets

| Target | Compiler | Optimized HGEMM | C++ | PyTorch |
| --- | --- | --- | :---: | :---: |
| NVIDIA SM80-compatible | `nvcc` | `hgemm_sm80.hpp` | Yes | Yes |
| AMD GFX950 | `hipcc` | `hgemm_gfx950.hpp`, `hgemm_hti_gfx950.hpp` | Yes | Yes |

The SM80 path is tested on an RTX 4090 (SM89). Backend and architecture
selection are compile-time: PeakGemm does not contain a runtime device
dispatcher or central kernel registry.

## HGEMM performance

An `8192 x 8192 x 8192` snapshot:

| GPU | Kernel | Type | No bias | Bias |
| --- | --- | --- | ---: | ---: |
| NVIDIA RTX 4090 | SM80 | FP16 | 163.6 TFLOPS | 164.0 TFLOPS |
| NVIDIA RTX 4090 | SM80 | BF16 | 164.9 TFLOPS | 165.8 TFLOPS |
| AMD Instinct MI355X | GFX950 HTI | FP16 | 1528.7 TFLOPS | 1510.5 TFLOPS |
| AMD Instinct MI355X | GFX950 HTI | BF16 | 1609.2 TFLOPS | 1590.8 TFLOPS |

These results were measured on 2026-08-05 with the C++ GEMM benchmarks. Each
case first passed an accuracy check, then ran 10 warmups and 8 measurements of
20 iterations; the reported value is the median and uses
`TFLOPS = 2 * M * N * K / time`. Results are single-system measurements rather
than vendor specifications.

## Build and test

Requirements:

- Linux.
- A C++20-capable `nvcc` or `hipcc`.
- A CUDA toolkit for the SM80 path, or ROCm for the GFX950 path.
- Python 3.10+, GPU-enabled PyTorch, CMake 3.21+, and Ninja for the Python API.

Run the complete C++ suite:

```bash
bash test_all.sh
```

Build and run one benchmark:

```bash
# RTX 4090
bash build_single.sh tests/gemm/test_hgemm_sm80.cpp
./a.out

# MI355X
bash build_single.sh tests/gemm/test_hgemm_gfx950.cpp
./a.out
```

With no backend argument, `build_single.sh` selects `nvcc` first and otherwise
falls back to `hipcc`. The standalone C++ tests do not use CMake.

### GFX950 thread trace

After building the GFX950 benchmark, collect an instruction-level Advanced
Thread Trace (ATT) with:

```bash
rocprofv3 --att=true --att-library-path /opt/rocm/lib -d att_out -- ./a.out
```

The `./` prefix is required because the current directory is normally not in
`PATH`. The generated `att_out` directory contains the raw trace, instruction
statistics, and `ui_output_agent_*_dispatch_*` data for ROCprof Compute Viewer.

## PyTorch API

`setup.py` invokes CMake to compile `PeakGemm/libPeakGemm.so`. Importing
`PeakGemm` loads it with `torch.ops.load_library`; no JIT compiler, generated
source, or runtime build cache is needed. CUDA PyTorch builds the SM80 path;
ROCm PyTorch builds the standard GFX950 path for small tiles and the HTI
schedule for large shapes (default `config=3`, M/N wave layout **2x4**).

Install the matching GPU-enabled PyTorch and toolkit first, then build:

```bash
python3 -m pip install "cmake>=3.21" ninja
python3 -m pip install -v -e . --no-build-isolation

# Optional: target a specific CUDA architecture (default: SM80 + PTX).
CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89" python3 -m pip install -v -e . --no-build-isolation
# ROCm defaults to gfx950.
```

`MAX_JOBS` controls build parallelism (default: 4). Use CMake's architecture
settings rather than `TORCH_CUDA_ARCH_LIST`. A non-editable install or wheel
also includes the compiled library.

```python
import torch
import PeakGemm

m = n = k = 4096
a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
bias = torch.randn((n,), device="cuda", dtype=a.dtype)
c = torch.empty((m, n), device="cuda", dtype=a.dtype)

PeakGemm.hgemm(c, a, b, bias=bias)  # config=3, split_k=1 by default
torch.cuda.synchronize()
torch.testing.assert_close(c, a @ b.T + bias, atol=1.0, rtol=1e-2)

# Small M, using a precompiled split-K specialization:
a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
b = torch.randn((64, 256), device="cuda", dtype=a.dtype)
c = torch.empty((16, 64), device="cuda", dtype=a.dtype)
PeakGemm.hgemm(c, a, b, split_k=4, config=0)
```

There are exactly four tile configurations, defined in `csrc/hgemm.cpp`:

| config | Backend | Block M/N/K | M/N warps | Stages / schedule |
| ---: | --- | --- | --- | --- |
| 0 | CUDA / ROCm | 16 / 64 / 64 | 1 / 2, 1 / 1 | 2 |
| 1 | CUDA / ROCm | 64 / 64 / 32 | 2 / 2, 2 / 1 | 3 |
| 2 | CUDA / ROCm | 64 / 128 / 32 | 2 / 4, 2 / 2 | 3 |
| 3 (default) | CUDA | 128 / 128 / 32 | 2 / 4 | 3 |
| 3 (default) | ROCm | **256 / 256 / 64** | **2 / 4** | **HTI** |

Swizzle M is 8 on CUDA and 0 on ROCm. Each backend has four configurations,
compiled ahead of time for FP16/BF16 and bias/no-bias. Standard configurations
also compile split-K/non-split variants. `config` only selects an already
compiled variant; changing the tiles requires rebuilding.

On ROCm, `config=3` always uses HTI, with M/N multiples of 256, K a positive
multiple of 128, and `split_k=1`. Unsupported shapes or split-K raise an
exception rather than silently falling back. Use `config=0..2` for small
shapes or split-K.

`A[M,K]`, `B[N,K]`, `C[M,N]`, and optional bias must be contiguous, 16-byte
aligned, on the same GPU, and use the same FP16 or BF16 dtype. Bias has N
elements. `C` is updated in place and returned; it must not overlap inputs.
Dimensions must be positive. M/N must be multiples of the selected block M/N.
For standard configurations, K must be divisible by `split_k * block_k`,
and each K partition must contain
at least `stages - 1` blocks. `split_k` is between 1 and 65535; split-K supports
at most 256 output tiles. Unsupported shapes raise an exception.

Split-K workspaces are cached per device and stream. Warm up on the intended
stream before CUDA/HIP graph capture. This is an inference/out API; it does not
provide autograd.

The old `Compiler`, `ConstexprParams`, and `compile_hgemm` JIT API is removed.
Use `PeakGemm.hgemm(..., config=...)` instead.

Run the accuracy tests and profiler benchmark with:

```bash
pytest -s tests/torch/test_hgemm.py -k "not benchmark"
pytest -s tests/torch/test_hgemm.py -k benchmark
```

## Training-compatible Linear

`PeakGemm.functional.linear(x, weight, bias=None)` follows the
`torch.nn.functional.linear` interface. `PeakGemm.Linear` is an `nn.Linear`
subclass with the same constructor and parameter names.

```python
from PeakGemm import functional as F

# In an existing module, keep the same parameters/checkpoint:
y = F.linear(x, self.weight, self.bias)
```

- Supported FP16/BF16 GPU forwards select among the four precompiled HGEMM
  tiles. ROCm uses the HTI 2x4-wave configuration when M/N are 256-aligned and
  K is 128-aligned, otherwise it tries the standard small tiles.
- Other shapes, FP32/FP64, unaligned storage offsets, CPU and biased layers
  use native PyTorch Linear. Empty dimensions also work. No padding or split-K.
- Autocast uses PyTorch's cached casts; FP32 master parameters keep FP32 `.grad`.
- Backward reuses the same dispatcher: `dx = linear(dy, weight.T)` and
  `dweight = linear(dy.T, x.T)` after flattening batch dimensions. Each gradient
  independently uses HGEMM/HTI when its shape/dtype is supported, otherwise
  native PyTorch. Only requested gradients are computed.
- The current kernels accept contiguous `A @ B.T`, so supported backward
  multiplications materialize transposed operands. This adds memory and copy
  overhead; using HGEMM is not a guarantee of faster end-to-end training.
  Accumulation is FP32, with FP16/BF16 GEMM outputs converted back to FP32 master
  parameter gradients by autograd's cast backward, matching autocast dtypes.
- Autograd, fake tensors/`torch.compile`, and graph capture are registered and
  tested. The raw `hgemm(c, a, b, ...)` remains an inference/out API.

```bash
pytest -q tests/torch/test_linear.py
```

## Fused RMSNorm (training)

`PeakGemm.functional.rms_norm` matches the following computation, including FP32
accumulation and casting the final output back to the input dtype:

```python
y = x.float()
y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
y = (weight.float() * y).to(x.dtype)
```

The kernel uses only `backend/runtime.hpp` (no MMA or architecture headers).
The same source compiles with CUDA and HIP. Forward is one kernel; backward
fuses `dx` and grouped FP32 partial `dweight` into one kernel, then reduces the
partial gradients in a second kernel. No global atomic accumulation is used.

Each kernel derives `constexpr VEC_SIZE = 16 / sizeof(T)`: 4 FP32 elements
or 8 FP16/BF16 elements. Forward/backward use the input dtype; the final
weight-gradient reduction uses the weight dtype. Aligned input/output vectors
use direct `*reinterpret_cast<const vec_t *>(ptr)` loads and
`*reinterpret_cast<vec_t *>(ptr) = value` stores. Mixed-dtype operands use vectors
with the same element count, so their byte size may differ. Alignment is checked
against `alignof(vec_t)` in the launcher. Unsupported dimensions or pointer
alignment raise an error before launch; there are no scalar/tail kernels or
load/store helper branches.

**Numerics:** RMSNorm reduces `mean(x^2)`, not centered variance
`mean(x^2) - mean(x)^2`; it does not require zero-mean/unit-variance input.
Welford's protection against cancellation in centered variance is therefore
not needed here. Squaring, sums and gradients use FP32. The backward uses
`dx = r * (dy*w - x_hat * mean(dy*w*x_hat))`, with `x_hat=x*r`, avoiding
unnormalized intermediate dot products and explicit `r^2`/`r^3`.

This is still FP32 arithmetic: extreme inputs can overflow the square/sum,
and very small values can underflow. Welford is not a general remedy for that;
handling the entire BF16/FP32 exponent range would require scaled sum-of-squares
or a wider accumulator. `eps > 0` is recommended for training; all-zero rows
with `eps=0` are undefined. Tests compare nonzero-mean, constant, and outlier
inputs at scales from `1e-12` to `1e12` against FP64; this is not a guarantee of
long-run convergence or arbitrary-range numerical stability.

Use the functional API like `torch.nn.functional`, without monkey-patching
PyTorch:

```python
from PeakGemm import functional as F

# Inside an existing nn.Module; keep self.weight and the checkpoint unchanged:
def forward(self, x):
    return F.rms_norm(x, self.weight, self.eps)
```

Or replace the whole module:

```python
import torch
from PeakGemm import RMSNorm

norm = RMSNorm(4096, eps=1e-6, device="cuda")  # weight defaults to FP32
x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
y = norm(x)
y.float().square().mean().backward()
# x.grad is BF16; norm.weight.grad is FP32.
```

- `x` has shape `[..., dim]`, weight has shape `[dim]`, on the same GPU.
- FP32/FP16/BF16 input and weight dtypes can differ. Output and `dx` match
  `x.dtype`; `dweight` matches `weight.dtype`.
- `1 <= dim <= 65536`, divisible by **both** `16 / sizeof(input_t)` and
  `16 / sizeof(weight_t)` (4 for FP32, 8 for FP16/BF16). Empty batches work.
- Input and weight must already be contiguous and vector-aligned; the public
  function does not copy, pad, or fall back. Input/output vectors require 16B
  alignment; the weight loads require `sizeof(weight_t) * (16 / sizeof(input_t))`
  alignment (32B for BF16 input + FP32 weight). Launcher checks also cover
  gradient buffers and workspace. Invalid arguments raise an error.
- Autograd materializes noncontiguous upstream gradients (e.g. `sum()`'s
  broadcast gradient) before calling the same vector-only backward kernel.
  Contiguous but misaligned upstream gradients are rejected.
- `eps` is a finite, non-negative FP32-representable scalar (default `1e-6`).
- Supports first-order autograd, FP32 parameter gradient accumulation,
  autocast, graph capture, and `torch.compile`. Double backward is explicitly
  unsupported. Requires PyTorch 2.4+.
- `RMSNorm(dim, eps)` retains the original `weight` state-dict key, so existing
  weights/checkpoints can be loaded unchanged.

C++ reference checks and event benchmark (automatically included in `test_all.sh`):

```bash
bash build_single.sh tests/core/test_rms_norm.cpp
./a.out
pytest -q tests/torch/test_rms_norm.py
```

## Architecture

PeakGemm deliberately separates portable runtime support, ISA primitives, and
GEMM scheduling:

```text
core/*                       backend-neutral layouts, vectors, and math
backend/runtime.hpp          CUDA/HIP runtime, warp, atomics, events, streams
backend/arch_<target>.hpp    target ISA primitives, swizzles, WMMA/MFMA
kernel/hgemm_<target>.hpp    target-specific GEMM pipeline and launchers
tests or csrc/hgemm.cpp       explicitly include and instantiate one target
```

`runtime.hpp` selects CUDA or HIP through `__CUDACC__` / `__HIPCC__`. It is the
shared layer for non-GEMM code that only needs allocation, copies, streams,
events, warp operations, or atomics. It does **not** contain SM80/GFX950 tensor
instructions.

Architecture primitives are added separately in `arch_*.hpp`. GEMM
implementations are then appended one target at a time as
`kernel/hgemm_<target>.hpp`, with a matching benchmark. Callers explicitly
include the required pair; for example:

```cpp
#include "peak_gemm/backend/arch_sm80.hpp"
#include "peak_gemm/kernel/hgemm_sm80.hpp"
```

The umbrella header `peak_gemm.hpp` intentionally includes core utilities,
`runtime.hpp`, and `data.hpp`, but not architecture or GEMM headers.

### Adding a new target

1. **Reuse or extend the runtime.** A new CUDA or HIP architecture normally
   needs no `runtime.hpp` change. A new GPU vendor must add a compiler branch,
   warp/atomic implementations, runtime API mappings, and a compiler case in
   `build_single.sh`.
2. **Add ISA primitives.** Create `backend/arch_<target>.hpp` with the target
   swizzle/copy primitives and a WMMA/MFMA type exposing `M`, `N`, `K`,
   fragment types, load/store methods, accumulation, and `WmmaDefault`.
3. **Append the GEMM implementation.** Create
   `kernel/hgemm_<target>.hpp`; bind the new `WmmaDefault` and swizzle, then
   implement its block tile, pipeline, epilogue, validation, and host launcher.
4. **Add coverage.** Add target-specific WMMA tests/benchmarks when the ISA
   changes, plus `tests/gemm/test_hgemm_<target>.cpp`. `test_all.sh` picks up
   new C++ test files automatically.
5. **Expose Python only if needed.** Add compile-time target selection and
   fixed kernel configurations in `csrc/hgemm.cpp` and update CMake as needed.

For a second schedule on an existing target, follow the GFX950 pattern:
keep the shared ISA layer in `arch_gfx950.hpp` and add a separate kernel such
as `hgemm_hti_gfx950.hpp`.

## Repository layout

The complete tracked source tree is:

```text
PeakGemm/
├── include/peak_gemm/
│   ├── peak_gemm.hpp                 umbrella: core + runtime + Data
│   ├── data.hpp                      host/device buffers, copies, check_allclose
│   ├── gemm_bench_infra.hpp          accuracy, timing, and TFLOPS harness
│   ├── backend/
│   │   ├── runtime.hpp               shared CUDA/HIP runtime abstraction
│   │   ├── arch_sm80.hpp             cp.async, swizzle, m16n8k16 MMA
│   │   └── arch_gfx950.hpp           swizzle, m16n16k16/k32 MFMA, AGPR
│   ├── core/
│   │   ├── config.hpp                compile macros and basic enums
│   │   ├── block_swizzle.hpp         grouped GEMM grid traversal
│   │   ├── layout.hpp                compile-time strided layouts
│   │   ├── math.hpp                  ceil_div and compile-time Log2
│   │   ├── shape.hpp                 static shape metadata
│   │   └── vector.hpp                aligned vector load/store wrapper
│   └── kernel/
│       ├── rms_norm.hpp              portable fused RMSNorm forward/backward
│       ├── gemm_naive.hpp            portable reference GPU GEMM
│       ├── hgemm_sm80.hpp            CUDA HGEMM, bias, split-K
│       ├── hgemm_gfx950.hpp          ROCm HGEMM, bias, split-K
│       └── hgemm_hti_gfx950.hpp      hand-scheduled GFX950 HTI HGEMM
├── PeakGemm/
│   ├── __init__.py                   shared-library loading and Python API
│   ├── functional.py                 RMSNorm autograd/fake registration
│   └── nn.py                         RMSNorm module
├── csrc/
│   ├── hgemm.cpp                     PyTorch GEMM operator and fixed configurations
│   └── rms_norm.cpp                  PyTorch RMSNorm forward/backward operators
├── tests/
│   ├── core/
│   │   ├── test_atomic.cpp           scalar and packed GPU atomics
│   │   ├── test_core.cpp             shape/layout/vector/math units
│   │   ├── test_data.cpp             allocation, copy, allclose, validation
│   │   ├── test_rms_norm.cpp         RMSNorm CPU reference, gradients, timing
│   │   ├── test_reduce_kernel.cpp    warp shuffle reduction
│   │   └── test_wmma_kernel_sm80_gfx950.cpp  WMMA/MFMA correctness
│   ├── arch_bench/
│   │   ├── test_fma_compute.cpp      FP32 FMA throughput
│   │   ├── test_global_bandwidth.cpp vectorized copy bandwidth
│   │   └── test_wmma_compute_sm80_gfx950.cpp WMMA/MFMA peak throughput
│   ├── gemm/
│   │   ├── test_gemm_naive.cpp       reference GEMM accuracy/performance
│   │   ├── test_hgemm_sm80.cpp       SM80 HGEMM dispatch and benchmark
│   │   └── test_hgemm_gfx950.cpp     GFX950 standard/HTI dispatch
│   └── torch/
│       ├── test_hgemm.py             PyTorch accuracy and profiler benchmark
│       └── test_rms_norm.py          autograd, mixed dtype, graph/compile checks
├── build_single.sh                   compile one C++ source with nvcc/hipcc
├── test_all.sh                       build and run every C++ test
├── format.sh                         clang-format all C/C++ sources
├── CMakeLists.txt                    Torch shared-library build
├── MANIFEST.in                       C++ sources/headers for source distributions
├── setup.py                          Python packaging and CMake build_ext
├── pyproject.toml                    Python build-system metadata
├── .clang-format                     C/C++ formatting rules
├── .gitignore                        generated-file exclusions
├── LICENSE
└── README.md
```

Generated binaries, build directories, Python caches, and profiler output
are not part of the source tree.

## License

See [LICENSE](LICENSE).
