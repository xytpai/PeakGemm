import os
import ctypes
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


sgemm_peak_ = eval(f"{prefix}.sgemm_peak")
hgemm_f16_peak_ = eval(f"{prefix}.hgemm_f16_peak")


def sgemm_peak(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    assert a.dtype == torch.float
    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert b.shape[1] == k
    c = c.view(-1, n)
    assert c.shape[0] == m
    sgemm_peak_(c, a, b, m, n, k)


def hgemm_f16_peak(
    c: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
):
    assert a.dtype == torch.half
    k = a.shape[-1]
    a = a.view(-1, k)
    m = a.shape[0]
    n = b.shape[0]
    assert b.shape[1] == k
    c = c.view(-1, n)
    assert c.shape[0] == m
    hgemm_f16_peak_(c, a, b, m, n, k)
