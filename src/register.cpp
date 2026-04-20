#include <torch/extension.h>
#include "ops.h"

TORCH_LIBRARY(PeakGemm, m) {
    m.def("sgemm_peak(Tensor c, Tensor a, Tensor b, SymInt m, SymInt n, SymInt k) -> ()");
    m.impl("sgemm_peak", &sgemm_peak);
    m.def("hgemm_f16_peak(Tensor c, Tensor a, Tensor b, SymInt m, SymInt n, SymInt k) -> ()");
    m.impl("hgemm_f16_peak", &hgemm_f16_peak);
}
