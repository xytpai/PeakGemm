import os
import functools
import torch


this_dir = os.path.dirname(__file__)
package_name = os.path.basename(this_dir)
filename = os.path.join(os.path.dirname(this_dir), f"lib{package_name}.so")
print("Loading extension from:", filename)
torch.ops.load_library(filename)
ops = getattr(torch.ops, package_name)


@functools.lru_cache(maxsize=128)
def get_semaphore(device, stream):
    semaphore = torch.zeros((256,), dtype=torch.uint32, device=device)
    signal = torch.zeros((256,), dtype=torch.uint32, device=device)
    return semaphore, signal


def _gemm_arguments(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    assert a.dtype in (torch.half, torch.bfloat16)
    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert b.shape[1] == k
    c = c.view(-1, n)
    assert c.shape[0] == m
    semaphore, signal = get_semaphore(
        a.device, torch.cuda.current_stream(a.device))
    return c, a, b, m, n, k, semaphore, signal


def gemm_peak(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    args = _gemm_arguments(c, a, b)
    ops.gemm_peak(*args)
