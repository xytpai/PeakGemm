#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -eq 0 ]]; then
    echo "usage: bash build_single.sh <source> [compiler options]" >&2
    exit 1
fi
source_file=$1
shift

backend=${BACKEND:-}
if [[ -z "$backend" ]]; then
    if command -v nvcc >/dev/null 2>&1; then
        backend=cuda
    elif command -v hipcc >/dev/null 2>&1; then
        backend=rocm
    else
        echo "nvcc or hipcc is required" >&2
        exit 1
    fi
fi

case "$backend" in
    cuda)
        command -v nvcc >/dev/null 2>&1 || {
            echo "nvcc is required" >&2
            exit 1
        }
        ARCH=${ARCH:-native}
        # echo "Using CUDA (nvcc), arch=$ARCH"
        nvcc -x cu -O3 "$@" --std=c++20 --expt-relaxed-constexpr -arch ${ARCH} -Iinclude -diag-suppress 20012 "$source_file" -o a.out
        ;;
    rocm|hip)
        command -v hipcc >/dev/null 2>&1 || {
            echo "hipcc is required" >&2
            exit 1
        }
        # echo "Using ROCm (hipcc)"
        hipcc -Wno-unused-value -O3 "$@" --std=c++20 -Iinclude "$source_file" -o a.out
        ;;
    *)
        echo "unsupported BACKEND: $backend" >&2
        exit 1
        ;;
esac
