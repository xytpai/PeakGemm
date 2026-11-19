# PeakGemm

PeakGemm is a cross-platform library of performance-critical GPU kernels, with support for:

| Operator | Backend | Arch | Layout | DType |
| --- | --- | --- | --- | --- |
| hgemm | rocm | gfx950 | nn/nt, optional bias | fp16/bf16 |
| hgemm | rocm | gfx1250 | nn/nt, optional bias | fp16/bf16 |
| hgemm | cuda | sm80 | nn/nt, optional bias | fp16/bf16 |

## Performance

An `8192 x 8192 x 8192` snapshot:

| GPU | Arch | Type | No bias | Bias |
| --- | --- | --- | ---: | ---: |
| NVIDIA RTX 4090 | SM80 | FP16 | 163.6 TFLOPS | 164.0 TFLOPS |
| NVIDIA RTX 4090 | SM80 | BF16 | 164.9 TFLOPS | 165.8 TFLOPS |
| AMD Instinct MI355X | GFX950 | FP16 | 1528.7 TFLOPS | 1510.5 TFLOPS |
| AMD Instinct MI355X | GFX950 | BF16 | 1609.2 TFLOPS | 1590.8 TFLOPS |
| AMD Instinct MI455X | GFX1250 | FP16 | 2449.7 TFLOPS | 2329.6 TFLOPS |
| AMD Instinct MI455X | GFX1250 | BF16 | 2565.6 TFLOPS | 2539.4 TFLOPS |

## Project structure

| Path | Description |
| --- | --- |
| `include/peak_gemm/backend/arch_*.hpp` | Wrappers for the MMA units of each platform architecture |
| `include/peak_gemm/backend/runtime.hpp` | Macros that unify basic CUDA and ROCm operations, plus wrappers for warp and atomic operations |
| `include/peak_gemm/core/*.hpp` | Core utilities |
| `include/peak_gemm/kernel/*.hpp` | Kernel templates |
| `include/peak_gemm/data.hpp` | Wrappers for test data units |
| `include/peak_gemm/gemm_bench_infra.hpp` | Shared framework for GEMM tests |
| `include/peak_gemm/peak_gemm.hpp` | Public include entry point, without MMA architecture headers |
| `tests/arch_bench/test_wmma_compute_*.cpp` | WMMA compute FLOPS on a single GPU |
| `tests/common_bench/test_fma_compute.cpp` | FP32 compute FLOPS on a single GPU |
| `tests/common_bench/test_global_bandwidth.cpp` | Device-memory copy bandwidth on a single GPU |
| `tests/core/*.cpp` | Tests of GPU features and utilities |
| `tests/gemm/*.cpp` | GEMM correctness tests and benchmarks |
| `tests/moe/*.cpp` | Fused MoE correctness tests |

## Test

To run all cpp tests:

```bash
bash test_all.sh
```

For single cpp test:

```bash
bash build_single.sh tests/common_bench/test_global_bandwidth.cpp && ./a.out
```

## ROCM thread trace

After building the ROCM benchmark, collect an instruction-level Advanced
Thread Trace (ATT) with:

```bash
rocprofv3 --att=true --att-library-path /opt/rocm/lib -d att_out -- ./a.out
```

The `./` prefix is required because the current directory is normally not in
`PATH`. The generated `att_out` directory contains the raw trace, instruction
statistics, and `ui_output_agent_*_dispatch_*` data for ROCprof Compute Viewer.

## Minimal fused MoE

`include/peak_gemm/kernel/fused_moe.hpp` implements the single-device equivalent
of `MoE` / `EPMoE` in `skills/kernel_optimization/mega_kernel/transformer_example.py`:

```text
router -> normalized top-k -> sum(weight * down(silu(gate(x)) * up(x))) + shared(x)
```

`peak_gemm::kernel::fused_moe_gpu` performs the entire operation in **one GPU
kernel launch**, including the router and shared expert. There is no EP
communication, bias, external workspace, or separate dispatch/combine kernel.

```cpp
#include "peak_gemm/kernel/fused_moe.hpp"

// Device pointers, contiguous row-major tensors:
// x/y: [T,H], router: [E,H]
// gate/up: [E+1,I,H], down: [E+1,H,I]
// The last expert slice (index E) contains the shared expert's weights.
peak_gemm::kernel::fused_moe_gpu(
    x, router, gate, up, down, y, T, H, I, E, K, stream);
```

This is a correctness-first baseline: one CTA per token, warp-level dot products,
and sequential selected experts. Projection inputs/weights and the final output
use 16-byte loads/stores via `core::Vector` when row addresses are aligned;
unaligned rows and incomplete vector tails use scalar accesses. Mixed FP32
activations / FP16 or BF16 weights are read in separate 16-byte chunks.
It uses the common CUDA/HIP runtime, not
architecture-specific HGEMM tiles or tensor cores. It is not intended to match
grouped-GEMM throughput. FP32, FP16 and BF16 tensors are supported; accumulation,
routing and intermediate activations use FP32, with one final cast to the output
dtype. Thus low-precision results need not be bitwise identical to PyTorch's
per-operation rounding.

Requirements: positive H/I/E, `1 <= K <= E`, finite logits/activations, and
`4 * (H + I + E + 2*K) <= 48 KiB` shared memory per CTA. No dimension alignment
is required. Equal router logits select lower expert ids first (PyTorch does
not guarantee the same tie order). Output must not overlap any input.
`T == 0` is a no-op with otherwise valid dimensions. The caller supplies
flattened `[T,H]` tensors and restores any batch/sequence dimensions.

```bash
bash build_single.sh tests/moe/test_fused_moe.cpp && ./a.out
```
