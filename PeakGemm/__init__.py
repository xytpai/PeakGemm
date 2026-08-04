import os
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import Tuple
from contextlib import contextmanager


# this_dir = os.path.dirname(__file__)
# package_name = os.path.basename(this_dir)
# filename = os.path.join(os.path.dirname(this_dir), f"lib{package_name}.so")
# print("Loading extension from:", filename)
# torch.ops.load_library(filename)


from .hgemm_jit import (
    HgemmConfig,
    compile_hgemm,
    enumerate_hgemm_configs,
)


__all__ = [
    "HgemmConfig",
    "compile_hgemm",
    "enumerate_hgemm_configs",
]
