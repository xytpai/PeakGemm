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

## Single-device DeepSeek-V4.1-Flash MoE

`include/peak_gemm/kernel/fused_moe.hpp` implements the unquantized **text MoE**
of DeepSeek-V4.1-Flash, including its DSpark expert configuration. It started
from `skills/kernel_optimization/mega_kernel/transformer_example.py`; the router
and activation now follow the DSV4.1 model rather than that example's softmax.

```text
scores  = sqrt(softplus(x @ router.T))
ids     = topk(scores + router_bias, K)
weights = scores[ids] / (sum(scores[ids]) + 1e-20) * 1.5
expert  = down(silu(min(gate(x), 10)) * clamp(up(x), -10, 10))
output  = sum(weights * routed_expert(x)) + shared_expert(x)
```

The FP32 correction bias affects **selection only**, not the mixture weights.
The clamp applies to both routed and shared experts; the shared expert is not
multiplied by the routed scaling factor.

`peak_gemm::kernel::fused_moe_gpu` performs the entire operation in **one GPU
kernel launch**, including the router and shared expert. All experts reside on
one device: no EP communication, linear projection bias, external workspace,
or separate dispatch/combine kernel.

```cpp
#include "peak_gemm/kernel/fused_moe.hpp"

// Contiguous, 16-byte-aligned device tensors:
// x/y: [T,H], router: [E,H], router_bias: [E] (float)
// gate/up: [E+1,I,H], down: [E+1,H,I]
// Slice E holds the shared expert weights; all other tensors use scalar_t.
peak_gemm::kernel::fused_moe_gpu(
    x, router, router_bias, gate, up, down, y, T, H, I, E, K, stream);
```

| Configuration | H | I | Routed experts E | Top-K |
| --- | ---: | ---: | ---: | ---: |
| DSV4.1-Flash main model | 5120 | 2304 | 384 | 6 |
| DSpark draft | 5120 | 2304 | 128 | 3 |

For the main model, gate/up are `[385,2304,5120]` and down is
`[385,5120,2304]`; each CTA uses 31,280 bytes of shared memory.
T is the number of token rows in this invocation, not session concurrency.

This is a correctness-first baseline: one CTA per token, warp-level dot products,
and sequential selected experts. Top-K uses one warp for repeated argmax with
expert-id tie breaking via the existing warp shuffle helpers. Up to 384
corrected scores are distributed across lane registers (covering main/draft
models); larger expert sets use warp-parallel scans of shared scores.
Projection inputs/weights and final output
use unconditional 16-byte loads/stores via the existing `core::Vector`; no
unaligned or scalar-tail fallback is provided. It uses the common CUDA/HIP
runtime, not tensor cores or architecture-specific HGEMM tiles.

FP32, FP16 and BF16 tensors are supported with **FP32 accumulation/intermediates**
and a final output cast. The checkpoint's packed FP4/FP8 tensors and scales
are **not** accepted: dequantize and pack the logical weights before calling.
This does not reproduce per-operation low-precision rounding or quantization,
and is not a full-model/checkpoint loader. Vision-specific routing, original
V4 hash-routing layers, attention, mHC and RMSNorm are outside this kernel.

Requirements: positive H/I/E, `1 <= K <= E`, finite logits/bias/activations, and
`4 * (H + I + E + 2*K) <= 48 KiB` shared memory per CTA. All tensor pointers,
including router_bias, must be 16-byte aligned; H/I must be multiples of
`16 / sizeof(scalar_t)` (4 for FP32, 8 for FP16/BF16).
The host launcher rejects unsupported alignment before execution.
Equal corrected scores prefer lower expert ids. Output must not overlap any
input. `T == 0` is a no-op with otherwise valid dimensions.

```bash
bash build_single.sh tests/moe/test_fused_moe.cpp && ./a.out
```

Semantic references:
[official DSV4.1 Gate/Expert/MoE](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py),
[model config](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json).

## Minimal persistent P2P payload handshake

`kernel/persistent_payload.hpp` is a two-GPU SPSC demo, not EP8 MoE or a tuned
transport. One persistent CTA sends rows; the other adds one and writes output.
Payload accesses reuse `core::Vector<float, 4>` (16B).

Epochs start at 1; `slot = epoch % SLOTS`. Two slots allow sending the next row
while the peer consumes the previous one. A single slot also works.

```text
producer: wait consumed[slot] == max(epoch - SLOTS, 0)
          write payload -> release ready[slot] = epoch
consumer: acquire ready[slot] == epoch
          read / compute -> release consumed[slot] = epoch
```

- Zero the channel before first use; epoch 0 means unused. Continue epochs
  across calls without uint32 wrap. Wait for both kernels before relaunch,
  reset or free; no concurrent invocations on the same channel.
- Launch both endpoints before synchronizing either. Each GPU runs one 1D CTA.
- Allocate the channel on the consumer GPU with producer P2P access and native
  peer atomics. HIP needs fine-grained allocation for the whole channel
  (`hipExtMallocWithFlags`); ordinary inputs/outputs use `Data`.
- All threads fence before the leader publishes. The demo conservatively uses
  per-thread acquire after leader polling. See `backend/system_atomic.hpp`.
- `max_polls` bounds polling iterations, not elapsed time. Cancellation makes
  output invalid: finish both kernels, reset the channel, restart at epoch 1.
  This is a debug escape, not GPU-failure recovery.

```bash
bash build_single.sh tests/core/test_persistent_payload.cpp && ./a.out
```

The test covers slot reuse, delayed/missing peers, both transfer directions and
output guards; unsupported GPU pairs are skipped. No performance claim.
