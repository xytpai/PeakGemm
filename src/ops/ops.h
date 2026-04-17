#pragma once

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

#ifdef __HIPCC__
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#define GET_CURRENT_STREAM c10::hip::getCurrentHIPStreamMasqueradingAsCUDA().stream()
#endif

#ifdef __CUDACC__
#include <ATen/cuda/CUDAContext.h>
#define GET_CURRENT_STREAM at::cuda::getCurrentCUDAStream()
#endif
