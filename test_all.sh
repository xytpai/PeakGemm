#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" =~ ^(cuda|rocm|hip)$ ]]; then
    export BACKEND=$1
    shift
fi

tests=(tests/core/*.cpp tests/bench/*.cpp)

for test_source in "${tests[@]}"; do
    echo "==> $test_source"
    bash build_single.sh "$test_source" "$@"
    ./a.out
done
