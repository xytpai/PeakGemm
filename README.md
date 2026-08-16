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
- Standalone C++ tests/benchmarks and a CUDA-only PyTorch JIT API.

## Supported targets

| Target | Compiler | Optimized HGEMM | C++ | PyTorch JIT |
| --- | --- | --- | :---: | :---: |
| NVIDIA SM80-compatible | `nvcc` | `hgemm_sm80.hpp` | Yes | Yes |
| AMD GFX950 | `hipcc` | `hgemm_gfx950.hpp`, `hgemm_hti_gfx950.hpp` | Yes | No |

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
- Python, PyTorch, and Ninja only when using the Python JIT API.

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
falls back to `hipcc`. There is no CMake build.

### GFX950 thread trace

After building the GFX950 benchmark, collect an instruction-level Advanced
Thread Trace (ATT) with:

```bash
rocprofv3 --att=true --att-library-path /opt/rocm/lib -d att_out -- ./a.out
```

The `./` prefix is required because the current directory is normally not in
`PATH`. The generated `att_out` directory contains the raw trace, instruction
statistics, and `ui_output_agent_*_dispatch_*` data for ROCprof Compute Viewer.

## PyTorch JIT API

The Python package currently exposes the CUDA SM80 path. Installation only
installs Python code; `compile_hgemm` builds and caches the requested extension
on first use.

```bash
python3 -m pip install -v -e . --no-build-isolation
```

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
bias = torch.randn((n,), device="cuda", dtype=a.dtype)
c = torch.empty((m, n), device="cuda", dtype=a.dtype)

gemm(c, a, b, split_k=1, bias=bias)
torch.cuda.synchronize()

torch.testing.assert_close(c, a @ b.T + bias, atol=1.0, rtol=1e-2)
```

`A`, `B`, `C`, and optional bias must be contiguous, on the same CUDA device,
and use the same FP16 or BF16 dtype. `C` is updated in place. The selected tile
configuration determines the M/N/K divisibility constraints; unsupported
shapes raise an exception. For JIT cross-compilation, use PyTorch's
`TORCH_CUDA_ARCH_LIST`, for example `TORCH_CUDA_ARCH_LIST=8.9`.

Run the Python accuracy and profiler benchmarks with:

```bash
pytest -s tests/torch/test_hgemm_sm80.py -k acc
pytest -s tests/torch/test_hgemm_sm80.py -k benchmark
```

## Architecture

PeakGemm deliberately separates portable runtime support, ISA primitives, and
GEMM scheduling:

```text
core/*                       backend-neutral layouts, vectors, and math
backend/runtime.hpp          CUDA/HIP runtime, warp, atomics, events, streams
backend/arch_<target>.hpp    target ISA primitives, swizzles, WMMA/MFMA
kernel/hgemm_<target>.hpp    target-specific GEMM pipeline and launchers
tests or Python JIT          explicitly include and instantiate one target
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
5. **Expose Python only if needed.** Add a target JIT module and export it from
   `PeakGemm/__init__.py`; the current JIT source is intentionally SM80-only.

For a second schedule on an existing target, follow the GFX950 pattern:
keep the shared ISA layer in `arch_gfx950.hpp` and add a separate kernel such
as `hgemm_hti_gfx950.hpp`.

## Repository layout

The complete tracked source tree is:

```text
PeakGemm/
├── include/peak_gemm/
│   ├── peak_gemm.hpp                 umbrella: core + runtime + Data
│   ├── data.hpp                      host/device RAII buffer and copies
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
│       ├── gemm_naive.hpp            portable reference GPU GEMM
│       ├── hgemm_sm80.hpp            CUDA HGEMM, bias, split-K
│       ├── hgemm_gfx950.hpp          ROCm HGEMM, bias, split-K
│       └── hgemm_hti_gfx950.hpp      hand-scheduled GFX950 HTI HGEMM
├── PeakGemm/
│   ├── __init__.py                   public Python exports
│   └── hgemm_sm80_jit.py             PyTorch CUDA source generation/cache
├── tests/
│   ├── core/
│   │   ├── test_atomic.cpp           scalar and packed GPU atomics
│   │   ├── test_core.cpp             shape/layout/vector/math units
│   │   ├── test_data.cpp             allocation, copy, and validation
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
│       └── test_hgemm_sm80.py        JIT accuracy and profiler benchmark
├── build_single.sh                   compile one C++ source with nvcc/hipcc
├── test_all.sh                       build and run every C++ test
├── format.sh                         clang-format all C/C++ sources
├── setup.py                          Python package metadata/dependencies
├── pyproject.toml                    Python build-system metadata
├── .clang-format                     C/C++ formatting rules
├── .gitignore                        generated-file exclusions
├── LICENSE
└── README.md
```

Generated binaries, JIT caches, Python caches, and profiler output are not part
of the source tree.

## License

See [LICENSE](LICENSE).
