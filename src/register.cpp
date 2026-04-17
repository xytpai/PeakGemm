#include <torch/extension.h>
#include "ops.h"

TORCH_LIBRARY(PeakGemm, m) {
    m.def("sgemm_peak(Tensor c, Tensor a, Tensor b, SymInt m, SymInt n, SymInt k) -> ()");
    m.impl("sgemm_peak", &sgemm_peak);
}
