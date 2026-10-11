import os
from pathlib import Path
import shlex
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
import torch


class CMakeBuild(build_ext):
    def get_ext_filename(self, ext_name):
        # This is a torch.ops library, not a Python/pybind11 module.
        return os.path.join(*ext_name.split(".")[:-1], "libPeakGemm.so")

    def build_extension(self, ext):
        build_dir = Path(self.build_temp).resolve()
        output_dir = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        torch_prefix = ";".join(str(Path(p) / "share/cmake") for p in torch.__path__)
        subprocess.check_call([
            "cmake", "-S", str(Path(__file__).resolve().parent),
            "-B", str(build_dir), "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_PREFIX_PATH={torch_prefix}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output_dir}",
            f"-DPEAKGEMM_BACKEND={'HIP' if torch.version.hip else 'CUDA'}",
            *shlex.split(os.environ.get("CMAKE_ARGS", "")),
        ])
        subprocess.check_call([
            "cmake", "--build", str(build_dir),
            "--parallel", os.environ.get("MAX_JOBS", "4"),
        ])


setup(
    name="PeakGemm",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[Extension("PeakGemm.libPeakGemm", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=["torch>=2.4"],
    python_requires=">=3.10",
)
