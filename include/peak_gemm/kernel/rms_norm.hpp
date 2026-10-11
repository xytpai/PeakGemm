#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "peak_gemm/backend/runtime.hpp"
#include "peak_gemm/core/vector.hpp"

namespace peak_gemm::kernel {

constexpr uint32_t kRmsNormThreads = 256;
constexpr uint32_t kRmsNormMaxDim = 65536;
constexpr uint32_t kRmsNormMaxGroups = 4096;

namespace rms_norm_detail {

template <typename T>
inline uint32_t threads_for(uint32_t dim) {
    constexpr uint32_t VEC_SIZE = 16 / sizeof(T);
    const uint32_t packs = dim / VEC_SIZE;
    const uint32_t warp = backend::Warp::size;
    return std::min(kRmsNormThreads, ((packs + warp - 1) / warp) * warp);
}

// All threads participate. Works with CUDA's 32 lanes and HIP's 64 lanes.
PEAKGEMM_DEVICE_INLINE float block_sum(float value, float *shared) {
    using Warp = backend::Warp;
    const uint32_t lane = threadIdx.x % Warp::size;
    const uint32_t warp = threadIdx.x / Warp::size;
#pragma unroll
    for (uint32_t offset = Warp::size / 2; offset > 0; offset /= 2) {
        value += Warp::shuffle_xor(value, offset);
    }
    if (blockDim.x == Warp::size) return value;
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = lane < blockDim.x / Warp::size ? shared[lane] : 0.0F;
#pragma unroll
        for (uint32_t offset = Warp::size / 2; offset > 0; offset /= 2) {
            value += Warp::shuffle_xor(value, offset);
        }
    }
    // Finish all shared reads before overwriting shared[0].
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = value;
    __syncthreads();
    return shared[0];
}

template <typename input_t, typename weight_t>
__global__ void forward_kernel(
    const input_t *x, const weight_t *weight, input_t *y, float *rstd,
    uint32_t dim, float eps) {
    constexpr uint32_t VEC_SIZE = 16 / sizeof(input_t);
    using vec_t = core::Vector<input_t, VEC_SIZE>;
    using weight_vec_t = core::Vector<weight_t, VEC_SIZE>;
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * dim;
    __shared__ float shared[kRmsNormThreads / backend::Warp::size];
    float sum = 0.0F;
    for (uint32_t col = threadIdx.x * VEC_SIZE; col < dim; col += blockDim.x * VEC_SIZE) {
        const auto values = *reinterpret_cast<const vec_t *>(x + base + col);
#pragma unroll
        for (uint32_t i = 0; i < VEC_SIZE; ++i) {
            const float value = static_cast<float>(values[i]);
            sum += value * value;
        }
    }
    const float r = rsqrtf(block_sum(sum, shared) / dim + eps);
    if (threadIdx.x == 0) rstd[blockIdx.x] = r;
    for (uint32_t col = threadIdx.x * VEC_SIZE; col < dim; col += blockDim.x * VEC_SIZE) {
        const auto values = *reinterpret_cast<const vec_t *>(x + base + col);
        const auto weights = *reinterpret_cast<const weight_vec_t *>(weight + col);
        vec_t output;
#pragma unroll
        for (uint32_t i = 0; i < VEC_SIZE; ++i) {
            const float normalized = static_cast<float>(values[i]) * r;
            output[i] = static_cast<input_t>(static_cast<float>(weights[i]) * normalized);
        }
        *reinterpret_cast<vec_t *>(y + base + col) = output;
    }
}

// A block owns a disjoint group of rows. Fuse dx and partial dweight, keeping
// dweight in FP32 registers across rows; no atomics or zeroed workspace needed.
template <typename input_t, typename weight_t, uint32_t ITEMS>
__global__ void backward_kernel(
    const input_t *dy, const input_t *x, const weight_t *weight, const float *rstd,
    input_t *dx, float *partial_dw, uint32_t rows, uint32_t dim) {
    constexpr uint32_t VEC_SIZE = 16 / sizeof(input_t);
    using vec_t = core::Vector<input_t, VEC_SIZE>;
    using weight_vec_t = core::Vector<weight_t, VEC_SIZE>;
    using acc_vec_t = core::Vector<float, VEC_SIZE>;
    __shared__ float shared[kRmsNormThreads / backend::Warp::size];
    acc_vec_t dw[ITEMS] = {};
    for (uint32_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const std::size_t base = static_cast<std::size_t>(row) * dim;
        const float r = rstd[row];
        float dot = 0.0F;
        for (uint32_t col = threadIdx.x * VEC_SIZE; col < dim; col += blockDim.x * VEC_SIZE) {
            const auto values = *reinterpret_cast<const vec_t *>(x + base + col);
            const auto grads = *reinterpret_cast<const vec_t *>(dy + base + col);
            const auto weights = *reinterpret_cast<const weight_vec_t *>(weight + col);
#pragma unroll
            for (uint32_t i = 0; i < VEC_SIZE; ++i) {
                const float normalized = static_cast<float>(values[i]) * r;
                dot += (static_cast<float>(grads[i]) * static_cast<float>(weights[i])) * normalized;
            }
        }
        // dx = r * (dy*w - x_hat * mean(dy*w*x_hat)).
        // Avoid forming r^2 or a dot product scaled by unnormalized x.
        const float correction = block_sum(dot, shared) / dim;
#pragma unroll
        for (uint32_t i = 0; i < ITEMS; ++i) {
            const uint32_t col = (threadIdx.x + i * blockDim.x) * VEC_SIZE;
            if (col < dim) {
                const auto values = *reinterpret_cast<const vec_t *>(x + base + col);
                const auto grads = *reinterpret_cast<const vec_t *>(dy + base + col);
                const auto weights = *reinterpret_cast<const weight_vec_t *>(weight + col);
                vec_t output;
#pragma unroll
                for (uint32_t j = 0; j < VEC_SIZE; ++j) {
                    const float normalized = static_cast<float>(values[j]) * r;
                    const float grad = static_cast<float>(grads[j]);
                    output[j] = static_cast<input_t>(
                        r * (grad * static_cast<float>(weights[j]) - normalized * correction));
                    dw[i][j] += grad * normalized;
                }
                *reinterpret_cast<vec_t *>(dx + base + col) = output;
            }
        }
        // Multi-warp blocks must finish shared reads before the next row.
        if (blockDim.x > backend::Warp::size) __syncthreads();
    }
#pragma unroll
    for (uint32_t i = 0; i < ITEMS; ++i) {
        const uint32_t col = (threadIdx.x + i * blockDim.x) * VEC_SIZE;
        if (col < dim) {
            *reinterpret_cast<acc_vec_t *>(partial_dw + static_cast<std::size_t>(blockIdx.x) * dim + col) = dw[i];
        }
    }
}

template <typename weight_t>
__global__ void weight_grad_kernel(const float *partial_dw, weight_t *dw, uint32_t groups, uint32_t dim) {
    constexpr uint32_t VEC_SIZE = 16 / sizeof(weight_t);
    using vec_t = core::Vector<weight_t, VEC_SIZE>;
    using acc_vec_t = core::Vector<float, VEC_SIZE>;
    // 8 adjacent output vectors x 32 group partitions.
    constexpr uint32_t COLS = 8, PARTS = kRmsNormThreads / COLS;
    const uint32_t lane = threadIdx.x % COLS, part = threadIdx.x / COLS;
    const uint32_t col = (blockIdx.x * COLS + lane) * VEC_SIZE;
    __shared__ acc_vec_t sums[PARTS][COLS];
    acc_vec_t sum = {};
    if (col < dim) {
        for (uint32_t group = part; group < groups; group += PARTS) {
            const auto values = *reinterpret_cast<const acc_vec_t *>(partial_dw + static_cast<std::size_t>(group) * dim + col);
#pragma unroll
            for (uint32_t i = 0; i < VEC_SIZE; ++i) sum[i] += values[i];
        }
    }
    sums[part][lane] = sum;
    __syncthreads();
    if (part == 0 && col < dim) {
        vec_t output;
#pragma unroll
        for (uint32_t i = 0; i < VEC_SIZE; ++i) {
            float value = 0.0F;
#pragma unroll
            for (uint32_t p = 0; p < PARTS; ++p) value += sums[p][lane][i];
            output[i] = static_cast<weight_t>(value);
        }
        *reinterpret_cast<vec_t *>(dw + col) = output;
    }
}

template <uint32_t ITEMS = 1, typename input_t, typename weight_t>
void launch_backward(
    const input_t *dy, const input_t *x, const weight_t *weight, const float *rstd,
    input_t *dx, float *partial_dw, uint32_t rows, uint32_t dim, uint32_t groups, gpuStream_t stream) {
    constexpr uint32_t VEC_SIZE = 16 / sizeof(input_t);
    const uint32_t threads = threads_for<input_t>(dim);
    if (dim <= ITEMS * threads * VEC_SIZE) {
        backward_kernel<input_t, weight_t, ITEMS><<<groups, threads, 0, stream>>>(
            dy, x, weight, rstd, dx, partial_dw, rows, dim);
    } else if constexpr (ITEMS * kRmsNormThreads * VEC_SIZE < kRmsNormMaxDim) {
        launch_backward<ITEMS * 2>(dy, x, weight, rstd, dx, partial_dw, rows, dim, groups, stream);
    }
}

template <typename input_t, typename weight_t>
inline void check_shape(uint32_t rows, uint32_t dim) {
    if (dim == 0 || dim > kRmsNormMaxDim) {
        throw std::invalid_argument("RMSNorm dim must be between 1 and 65536");
    }
    // Both the input vectors and the final dweight vectors must fit whole rows.
    if (dim % (16 / sizeof(input_t)) || dim % (16 / sizeof(weight_t))) {
        throw std::invalid_argument("RMSNorm dim must be divisible by input and weight VEC_SIZE");
    }
    if (rows > 0x7FFFFFFFU) {
        throw std::invalid_argument("RMSNorm rows exceed the GPU grid limit");
    }
}

inline void check_pointer(const void *ptr, std::size_t alignment) {
    if (!ptr || reinterpret_cast<uintptr_t>(ptr) % alignment) {
        throw std::invalid_argument("RMSNorm pointers must be non-null and vector-aligned");
    }
}

} // namespace rms_norm_detail

inline uint32_t rms_norm_groups(uint32_t rows, uint32_t dim) {
    // Q/K norms have many short rows: use more blocks to avoid long serial loops.
    return std::min(rows, dim <= kRmsNormThreads ? kRmsNormMaxGroups : 256U);
}

// Contiguous, non-overlapping tensors. dim must be divisible by both input and
// weight vector widths. Pointers must satisfy the vector types used in kernels.
// Unsupported shapes/alignment throw before launch; no scalar or tail path.
template <typename input_t, typename weight_t>
void rms_norm_forward(
    const input_t *x, const weight_t *weight, input_t *y, float *rstd,
    uint32_t rows, uint32_t dim, float eps, gpuStream_t stream = nullptr) {
    rms_norm_detail::check_shape<input_t, weight_t>(rows, dim);
    if (!std::isfinite(eps) || eps < 0.0F) {
        throw std::invalid_argument("RMSNorm eps must be finite and non-negative");
    }
    if (rows == 0) return;
    rms_norm_detail::check_pointer(x, 16);
    rms_norm_detail::check_pointer(y, 16);
    rms_norm_detail::check_pointer(weight, sizeof(weight_t) * (16 / sizeof(input_t)));
    rms_norm_detail::check_pointer(rstd, alignof(float));
    rms_norm_detail::forward_kernel<<<rows, rms_norm_detail::threads_for<input_t>(dim), 0, stream>>>(x, weight, y, rstd, dim, eps);
}

// partial_dw: [rms_norm_groups(rows, dim), dim], FP32 scratch (no initialization).
// dx and dw are overwritten, not accumulated. For rows=0, dw is filled with 0.
template <typename input_t, typename weight_t>
void rms_norm_backward(
    const input_t *dy, const input_t *x, const weight_t *weight, const float *rstd,
    input_t *dx, weight_t *dw, float *partial_dw, uint32_t rows, uint32_t dim, gpuStream_t stream = nullptr) {
    rms_norm_detail::check_shape<input_t, weight_t>(rows, dim);
    rms_norm_detail::check_pointer(dw, 16);
    const uint32_t groups = rms_norm_groups(rows, dim);
    if (rows > 0) {
        rms_norm_detail::check_pointer(x, 16);
        rms_norm_detail::check_pointer(dy, 16);
        rms_norm_detail::check_pointer(dx, 16);
        rms_norm_detail::check_pointer(weight, sizeof(weight_t) * (16 / sizeof(input_t)));
        rms_norm_detail::check_pointer(rstd, alignof(float));
        rms_norm_detail::check_pointer(partial_dw, sizeof(float) * std::max(16 / sizeof(input_t), 16 / sizeof(weight_t)));
        rms_norm_detail::launch_backward(dy, x, weight, rstd, dx, partial_dw, rows, dim, groups, stream);
    }
    constexpr uint32_t VEC_SIZE = 16 / sizeof(weight_t);
    constexpr uint32_t BLOCK_COLS = 8 * VEC_SIZE;
    rms_norm_detail::weight_grad_kernel<<<(dim + BLOCK_COLS - 1) / BLOCK_COLS, kRmsNormThreads, 0, stream>>>(
        partial_dw, dw, groups, dim);
}

} // namespace peak_gemm::kernel
