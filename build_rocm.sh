hipcc -Wno-unused-value -O3 ${@:2} --std=c++20 -Isrc/common -Isrc/impl -Rpass-analysis=kernel-resource-usage $1 -o a.out
