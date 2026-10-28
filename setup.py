from pathlib import Path
import shutil

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        # JIT kernels also work from installed wheels, not only source checkouts.
        shutil.copytree(
            Path(__file__).parent / "include" / "peak_gemm",
            Path(self.build_lib) / "PeakGemm" / "include" / "peak_gemm",
            dirs_exist_ok=True,
        )


setup(
    name="PeakGemm",
    version="0.1.0",
    packages=find_packages(),
    package_data={"PeakGemm": ["csrc/*.cu", "include/peak_gemm/**/*.hpp"]},
    cmdclass={"build_py": BuildPy},
    install_requires=["ninja", "torch"],
)
