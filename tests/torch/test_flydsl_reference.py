"""Optional benchmark of PyTorch's vendored FlyDSL reference, PG and native.

USE_FLYDSL_REFERENCE=1 pytest -s tests/torch/test_flydsl_reference.py
Requires FlyDSL and a PyTorch checkout/install containing the reference.
No reference code is patched, and JIT compilation is excluded from timing.
"""

import os
from statistics import median

import pytest
import torch

import PeakGemm


pytestmark = pytest.mark.skipif(
    os.environ.get("USE_FLYDSL_REFERENCE") != "1"
    or not torch.cuda.is_available() or not torch.version.hip,
    reason="opt-in FlyDSL reference benchmark on ROCm",
)


@pytest.mark.parametrize("m,n,k", [
    (512, 768, 1024), (4096, 4096, 4096),
    (8192, 8192, 8192), (4096, 1024, 4096),
])
@pytest.mark.parametrize("transpose_b", [False, True], ids=["NN", "NT"])
def test_flydsl_reference_benchmark(m, n, k, transpose_b):
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("GFX950 required")
    flyc = pytest.importorskip("flydsl.compiler")
    ref = pytest.importorskip(
        "torch._inductor.kernel.vendored_templates.flydsl.kernels.gemm_gfx950"
    )

    torch.manual_seed(2026)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.25
    b = torch.randn(k, n, device="cuda", dtype=a.dtype) * 0.25
    if transpose_b:
        b = b.T.contiguous().T
    out = torch.empty(m, n, device="cuda", dtype=a.dtype)
    expected = a @ b
    stream = torch.cuda.current_stream().cuda_stream
    functions = {}
    for group_m in (0, 4):  # PyTorch's two default large-shape HTI candidates.
        param = ref.make_gemm_gfx950_param(
            dtype_id=ref.GEMM_DTYPE_BF16, tile_m=256, tile_n=256, tile_k=64,
            stages=2, m_waves=2, n_waves=4, group_m=group_m,
            use_half_tile_interleaved=True, b_is_transposed=transpose_b,
        )
        compiled = flyc.compile(
            ref.gemm_gfx950,
            *(flyc.from_torch_tensor(t).mark_layout_dynamic() for t in (out, a, b)),
            param, stream,
        )
        functions[f"FlyDSL gm{group_m}"] = (
            lambda compiled=compiled, param=param: compiled(out, a, b, param, stream)
        )
    functions["PG"] = lambda: PeakGemm.hgemm(out, a, b, transpose_b=False)
    functions["native"] = lambda: torch.mm(a, b, out=out)
    for fn in functions.values():
        fn()
        torch.testing.assert_close(out, expected, atol=0.15, rtol=0.03)
        for _ in range(10):
            fn()
    torch.cuda.synchronize()

    times = {name: [] for name in functions}
    for repeat in range(5):
        order = list(functions.items())
        if repeat % 2:
            order.reverse()
        for name, fn in order:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                fn()
            stop.record()
            stop.synchronize()
            times[name].append(start.elapsed_time(stop) * 1000 / 20)
    print(f"\nBF16 {'NT' if transpose_b else 'NN'} M/N/K={m}/{n}/{k}: " + ", ".join(
        f"{name}={median(values):.1f} us" for name, values in times.items()
    ))
