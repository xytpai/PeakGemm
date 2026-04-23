import os
import ctypes
import functools
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import Tuple
from contextlib import contextmanager


this_dir = os.path.dirname(__file__)
package_name = os.path.basename(this_dir)
filename = os.path.join(os.path.dirname(this_dir), f"lib{package_name}.so")
print("Loading extension from:", filename)
torch.ops.load_library(filename)
prefix = f"torch.ops.{package_name}"


gemm_peak_ = eval(f"{prefix}.gemm_peak")


@functools.lru_cache(maxsize=128)
def get_semaphore(device):
    semaphore = torch.zeros((128*3,), dtype=torch.uint32, device=device)
    signal_state = torch.zeros((128,), dtype=torch.uint32, device=device)
    return semaphore, signal_state


def gemm_peak(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    assert a.dtype in [torch.float, torch.half, torch.bfloat16]
    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert b.shape[1] == k
    c = c.view(-1, n)
    assert c.shape[0] == m
    semaphore, signal_state = get_semaphore(a.device)
    gemm_peak_(c, a, b, m, n, k, semaphore, signal_state)
