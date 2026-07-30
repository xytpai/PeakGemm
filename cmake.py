"Manages CMake."

import os
import sys
import torch
import sysconfig
from packaging.version import Version
from pathlib import Path
from subprocess import CalledProcessError, check_call, check_output
from typing import Any, cast


BUILD_DIR = 'build'


def cuda_arch_list() -> str:
    requested = os.environ.get("PEAKGEMM_CUDA_ARCH_LIST")
    if requested:
        return requested
    if torch.cuda.is_available():
        capabilities = {
            torch.cuda.get_device_capability(device)
            for device in range(torch.cuda.device_count())
        }
        if min(major for major, _ in capabilities) < 8:
            raise RuntimeError("PeakGemm requires SM80 or newer")
        return " ".join(
            f"{major}.{minor}"
            for major, minor in sorted(capabilities)
        )
    return "8.0+PTX"


def _mkdir_p(d: str) -> None:
    try:
        os.makedirs(d, exist_ok=True)
    except OSError as e:
        raise RuntimeError(
            f"Failed to create folder {os.path.abspath(d)}: {e.strerror}"
        ) from e


def which(thefile: str) -> str | None:
    path = os.environ.get("PATH", os.defpath).split(os.pathsep)
    for d in path:
        fname = os.path.join(d, thefile)
        if os.access(fname, os.F_OK | os.X_OK) and not os.path.isdir(fname):
            return fname
    return None


class CMake:
    "Manages cmake."

    def __init__(self, parallel_build=None, build_dir=BUILD_DIR) -> None:
        self._cmake_command = CMake._get_cmake_command()
        self.build_dir = build_dir
        self.parallel_build = parallel_build
        self.env = os.environ.copy()
        self.env["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list()

    @property
    def _cmake_cache_file(self) -> str:
        r"""Returns the path to CMakeCache.txt.

        Returns:
          string: The path to CMakeCache.txt.
        """
        return os.path.join(self.build_dir, "CMakeCache.txt")

    @staticmethod
    def _get_cmake_command() -> str:
        "Returns cmake command."
        cmake_command = which("cmake")
        cmake_version = CMake._get_version(cmake_command)
        _cmake_min_version = Version("3.18.0")
        if all(
            ver is None or ver < _cmake_min_version
            for ver in [cmake_version]
        ):
            raise RuntimeError("no cmake with version >= 3.18.0 found")
        return cmake_command

    @staticmethod
    def _get_version(cmd: str | None) -> Any:
        "Returns cmake version."
        if cmd is None:
            return None
        for line in check_output([cmd, "--version"]).decode("utf-8").split("\n"):
            if "version" in line:
                return Version(line.strip().split(" ")[2])
        raise RuntimeError("no version found")

    @staticmethod
    def defines(args, **kwargs) -> None:
        "Adds definitions to a cmake argument list."
        for key, value in sorted(kwargs.items()):
            if value is not None:
                args.append(f"-D{key}={value}")

    def run(self, args: list[str], env: dict[str, str]) -> None:
        "Executes cmake with arguments and an environment."
        command = [self._cmake_command] + args
        print(" ".join(command))
        try:
            check_call(command, cwd=self.build_dir, env=env)
        except (CalledProcessError, KeyboardInterrupt):
            # This error indicates that there was a problem with cmake, the
            # Python backtrace adds no signal here so skip over it by catching
            # the error and exiting manually
            sys.exit(1)
    
    def generate(self, rerun: bool, output_dir: str):
        if rerun and os.path.isfile(self._cmake_cache_file):
            os.remove(self._cmake_cache_file)
        source_dir = Path(__file__).resolve().parent / "bindings" / "torch"
        args = [str(source_dir)]
        _mkdir_p(self.build_dir)
        torch_dir = torch.__path__[0]
        CMake.defines(args, 
                      Torch_DIR=os.path.join(torch_dir, 'share/cmake/Torch'),
                      TORCH_CUDA_ARCH_LIST=self.env["TORCH_CUDA_ARCH_LIST"],
                      PYTHON_INCLUDE_DIR=sysconfig.get_paths()['include'],
                      CMAKE_LIBRARY_OUTPUT_DIRECTORY=output_dir)
        self.run(args, self.env)
    
    def build(self):
        if not self.parallel_build:
            self.run(['--build', '.'], self.env)
        elif isinstance(self.parallel_build, int):
            nworkers = str(self.parallel_build)
            self.run(['--build', '.', '-j', nworkers], self.env)
        else:
            self.run(['--build', '.', '--parallel'], self.env)
