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

## Project structure

| Path | Description |
| --- | --- |
| `include/peak_gemm/backend/arch_*.hpp` | Wrappers for the MMA units of each platform architecture |
| `include/peak_gemm/backend/runtime.hpp` | Macros that unify basic CUDA and ROCm operations, plus wrappers for warp and atomic operations |
| `include/peak_gemm/core/*.hpp` | Core utilities |
| `include/peak_gemm/kernel/*.hpp` | Kernel templates |
| `include/peak_gemm/data.hpp` | Wrappers for test data units |
| `include/peak_gemm/gemm_bench_infra.hpp` | Shared framework for GEMM tests |
| `include/peak_gemm/peak_gemm.hpp` | Public include entry point |
| `tests/arch_bench/test_fma_compute.cpp` | FP32 compute FLOPS on a single GPU |
| `tests/arch_bench/test_global_bandwidth.cpp` | Device-memory copy bandwidth on a single GPU |
| `tests/arch_bench/test_wmma_compute_*.cpp` | WMMA compute FLOPS on a single GPU |
| `tests/core/*.cpp` | Tests of GPU features and utilities |
| `tests/kernel/*.cpp` | Kernel tests |

## Test

To run all tests:

```bash
bash test_all.sh
```

For single target test:

```bash
bash build_single.sh tests/arch_bench/test_global_bandwidth.cpp ; ./a.out
```
