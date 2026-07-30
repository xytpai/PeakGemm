#pragma once

#include <torch/extension.h>

using torch::Tensor;

void gemm_peak(Tensor &c, Tensor &a, Tensor &b, int64_t m, int64_t n, int64_t k, Tensor &semaphore, Tensor &signal);
