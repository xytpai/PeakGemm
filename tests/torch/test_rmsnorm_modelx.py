"""Optional integration: MODELX_ROOT=/path/to/modelx python -m pytest ..."""
import os
import sys

import pytest
import torch

from PeakGemm import RMSNorm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/HIP GPU required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_modelx_transformer(dtype):
    root = os.environ.get("MODELX_ROOT")
    if not root:
        pytest.skip("set MODELX_ROOT to run the upstream ModelX integration test")
    sys.path.insert(0, root)
    try:
        import model.transformer as model
        from model.config import ModelArgs
    finally:
        sys.path.pop(0)
    torch.manual_seed(12)
    args = ModelArgs(dim=256, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=64,
                     vocab_size=256, max_seq_len=32, ffn_type="dense", intermediate_size=512)
    original = model.RMSNorm
    reference = model.Transformer(args).cuda()
    try:
        model.RMSNorm = RMSNorm
        actual = model.Transformer(args).cuda()
    finally:
        model.RMSNorm = original
    actual.load_state_dict(reference.state_dict())
    if dtype == torch.bfloat16:
        actual.bf16()
        reference.bf16()
    tokens = torch.randint(0, 256, (2, 16), device="cuda")
    y, yr = actual(tokens), reference(tokens)
    tol = 0.025 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(y, yr, rtol=tol, atol=tol)
    y.float().square().mean().backward()
    yr.float().square().mean().backward()
    for (name, param), (ref_name, ref_param) in zip(actual.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        torch.testing.assert_close(
            param.grad, ref_param.grad,
            rtol=0.05 if dtype == torch.bfloat16 else 5e-4,
            atol=0.002 if dtype == torch.bfloat16 else 1e-5,
        )
