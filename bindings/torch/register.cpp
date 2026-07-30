#include <torch/extension.h>

#include "ops.hpp"

TORCH_LIBRARY(PeakGemm, module) {
    module.def("gemm_peak(Tensor c, Tensor a, Tensor b, SymInt m, SymInt n, SymInt k, Tensor semaphore, Tensor signal) -> ()");
    module.impl("gemm_peak", &gemm_peak);
}
