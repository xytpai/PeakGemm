import importlib.util
import sys
import types
import unittest
from pathlib import Path


try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    torch.__version__ = "test"
    torch.version = types.SimpleNamespace(cuda=None)
    sys.modules["torch"] = torch


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "PeakGemm"
    / "hgemm_jit.py"
)
SPEC = importlib.util.spec_from_file_location("peakgemm_jit_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
JIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = JIT
SPEC.loader.exec_module(JIT)


class HgemmJitTest(unittest.TestCase):
    def test_enumerates_the_cartesian_product(self):
        configs = list(
            JIT.enumerate_hgemm_configs(
                block_m=(16, 32),
                block_n=(64,),
                block_k=(32,),
                block_m_warps=(1,),
                block_n_warps=(2,),
                stages=(2,),
                split_k=(1, 4),
            )
        )

        self.assertEqual(len(configs), 4)

    def test_rejects_invalid_scalar_values(self):
        with self.assertRaises(ValueError):
            JIT.HgemmConfig(16, 64, 32, 1, 2, 1)
        with self.assertRaises(ValueError):
            JIT.HgemmConfig(16, 64, 32, 1, 2, 2, split_k=65536)

    def test_each_config_has_a_distinct_extension(self):
        first = JIT.HgemmConfig(16, 64, 32, 1, 2, 2, split_k=1)
        second = JIT.HgemmConfig(16, 64, 32, 1, 2, 2, split_k=4)

        self.assertNotEqual(
            JIT._extension_name(first),
            JIT._extension_name(second),
        )

    def test_cache_directories_have_distinct_extensions(self):
        config = JIT.HgemmConfig(16, 64, 32, 1, 2, 2)

        self.assertNotEqual(
            JIT._extension_name(config, Path("cache-a")),
            JIT._extension_name(config, Path("cache-b")),
        )

    def test_source_contains_only_one_config(self):
        config = JIT.HgemmConfig(16, 64, 32, 1, 2, 2, split_k=4)

        source = JIT._cuda_source("test_extension", config)

        self.assertIn(
            "scalar_t, 16, 64, 32, 1, 2, 2, 8, false, true",
            source,
        )
        self.assertIn("peak_gemm::kernel::hgemm_template<", source)
        self.assertIn("4U", source)
        self.assertIn("TORCH_LIBRARY(test_extension, module)", source)
        self.assertNotIn("config_id", source)


if __name__ == "__main__":
    unittest.main()
