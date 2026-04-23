#include <torch/extension.h>
#include "ops.h"

TORCH_LIBRARY(PeakGemm, m) {
    m.def("gemm_peak(Tensor c, Tensor a, Tensor b, SymInt m, SymInt n, SymInt k, Tensor semaphore, Tensor signal_state) -> ()");
    m.impl("gemm_peak", &gemm_peak);
}
