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
    / "hgemm_sm80_jit.py"
)
SPEC = importlib.util.spec_from_file_location("hgemm_sm80_jit_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
JIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = JIT
SPEC.loader.exec_module(JIT)


class HgemmJitTest(unittest.TestCase):
    def test_constexpr_params_only_contains_tile_parameters(self):
        params = JIT.ConstexprParams(16, 64, 32, 1, 2, 2)

        self.assertNotIn("has_bias", params.__dict__)
        self.assertNotIn("is_split_k", params.__dict__)

    def test_rejects_invalid_constexpr_values(self):
        with self.assertRaises(ValueError):
            JIT.ConstexprParams(16, 64, 32, 1, 2, 1)

    def test_source_embeds_all_bias_and_split_variants(self):
        source = JIT.Compiler(
            JIT.ConstexprParams(16, 64, 32, 1, 2, 2)
        ).get_source("test_extension")

        for arguments in (
            "launch<scalar_t, true, true>",
            "launch<scalar_t, false, true>",
            "launch<scalar_t, true, false>",
            "launch<scalar_t, false, false>",
        ):
            self.assertIn(arguments, source)
        self.assertIn("int split_k", source)
        self.assertIn("Tensor? bias", source)
        self.assertIn("TORCH_LIBRARY(test_extension, module)", source)

    def test_extension_name_depends_on_constexpr_params(self):
        first = JIT.Compiler(JIT.ConstexprParams(16, 64, 32, 1, 2, 2))
        second = JIT.Compiler(JIT.ConstexprParams(32, 64, 32, 1, 2, 2))

        self.assertNotEqual(first.get_ext_name(), second.get_ext_name())

    def test_cuda_half_support_is_reenabled(self):
        for macro in (
            "__CUDA_NO_HALF_OPERATORS__",
            "__CUDA_NO_HALF_CONVERSIONS__",
            "__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "__CUDA_NO_HALF2_OPERATORS__",
        ):
            self.assertIn(f"-U{macro}", JIT._EXTRA_CUDA_CFLAGS)

    def test_default_cache_is_current_directory_temp(self):
        self.assertEqual(
            JIT._resolve_cache_dir(None),
            (Path.cwd() / "temp").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
