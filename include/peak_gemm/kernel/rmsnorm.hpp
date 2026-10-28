#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "peak_gemm/backend/runtime.hpp"

namespace peak_gemm::kernel {
namespace rmsnorm_detail {

using Warp = backend::Warp;

PEAKGEMM_DEVICE_INLINE float warp_sum(float value) {
#pragma unroll
    for (int offset = Warp::size / 2; offset > 0; offset /= 2) {
        value += Warp::shuffle_xor(value, offset);
    }
    return value;
}

// Every thread must participate. scratch holds THREADS / Warp::size floats.
template <int THREADS>
PEAKGEMM_DEVICE_INLINE float block_sum(float value, float *scratch) {
    value = warp_sum(value);
    if (threadIdx.x % Warp::size == 0) {
        scratch[threadIdx.x / Warp::size] = value;
    }
    __syncthreads();
    value = threadIdx.x < THREADS / Warp::size ? scratch[threadIdx.x] : 0.0F;
    value = warp_sum(value);
    if (threadIdx.x == 0) {
        scratch[0] = value;
    }
    __syncthreads();
    return scratch[0];
}

// Small head dimensions: a full warp per row, several independent rows/block.
template <typename T, typename W, int ITEMS>
__global__ __launch_bounds__(256) void forward_warp(
    const T *__restrict__ x, const W *__restrict__ weight, T *__restrict__ y, float *__restrict__ rstd,
    int64_t rows, int cols, float eps) {
    const int lane = threadIdx.x % Warp::size;
    const int64_t row = int64_t(blockIdx.x) * (256 / Warp::size) + threadIdx.x / Warp::size;
    if (row >= rows) {
        return; // Uniform within each warp, including CUDA's full-mask shuffles.
    }
    float values[ITEMS];
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int col = lane + i * Warp::size;
        values[i] = col < cols ? float(x[row * cols + col]) : 0.0F;
        sum += values[i] * values[i];
    }
    const float inv = rsqrtf(warp_sum(sum) / float(cols) + eps);
    if (rstd && lane == 0) {
        rstd[row] = inv;
    }
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int col = lane + i * Warp::size;
        if (col < cols) {
            y[row * cols + col] = T((values[i] * inv) * float(weight[col]));
        }
    }
}

template <typename T, typename W, int ITEMS>
__global__ __launch_bounds__(256) void forward_block(
    const T *__restrict__ x, const W *__restrict__ weight, T *__restrict__ y, float *__restrict__ rstd,
    int cols, float eps) {
    constexpr int THREADS = 256;
    const int64_t base = int64_t(blockIdx.x) * cols;
    __shared__ float scratch[THREADS / Warp::size];
    float values[ITEMS];
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int col = threadIdx.x + i * THREADS;
        values[i] = col < cols ? float(x[base + col]) : 0.0F;
        sum += values[i] * values[i];
    }
    const float inv = rsqrtf(block_sum<THREADS>(sum, scratch) / float(cols) + eps);
    if (rstd && threadIdx.x == 0) {
        rstd[blockIdx.x] = inv;
    }
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int col = threadIdx.x + i * THREADS;
        if (col < cols) {
            y[base + col] = T((values[i] * inv) * float(weight[col]));
        }
    }
}

template <typename T, int V>
struct alignas(sizeof(T) * V) Pack {
    T data[V];
};

template <typename T, typename W, int ITEMS>
__global__ __launch_bounds__(256) void forward_vector(
    const T *__restrict__ x, const W *__restrict__ weight,
    T *__restrict__ y, float *__restrict__ rstd, int cols, float eps) {
    constexpr int V = 4;
    constexpr int THREADS = 256;
    const int64_t base = int64_t(blockIdx.x) * (cols / V);
    const int col0 = threadIdx.x;
    __shared__ float scratch[THREADS / Warp::size];
    float values[ITEMS][V];
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int col = col0 + i * THREADS;
        Pack<T, V> v{};
        if (col < cols / V) {
            v = reinterpret_cast<const Pack<T, V> *>(x)[base + col];
        }
#pragma unroll
        for (int j = 0; j < V; ++j) {
            values[i][j] = float(v.data[j]);
            sum += values[i][j] * values[i][j];
        }
    }
    const float inv = rsqrtf(block_sum<THREADS>(sum, scratch) / float(cols) + eps);
    if (rstd && threadIdx.x == 0) {
        rstd[blockIdx.x] = inv;
    }
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const int col = col0 + i * THREADS;
        if (col < cols / V) {
            const auto w = reinterpret_cast<const Pack<W, V> *>(weight)[col];
            Pack<T, V> out;
#pragma unroll
            for (int j = 0; j < V; ++j) {
                out.data[j] = T((values[i][j] * inv) * float(w.data[j]));
            }
            reinterpret_cast<Pack<T, V> *>(y)[base + col] = out;
        }
    }
}

// Bounded register use for arbitrarily wide rows.
template <typename T, typename W>
__global__ __launch_bounds__(256) void forward_wide(
    const T *__restrict__ x, const W *__restrict__ weight, T *__restrict__ y, float *__restrict__ rstd, int cols, float eps) {
    __shared__ float scratch[256 / Warp::size];
    const int64_t base = int64_t(blockIdx.x) * cols;
    float sum = 0.0F;
    for (int64_t col = threadIdx.x; col < cols; col += 256) {
        const float v = float(x[base + col]);
        sum += v * v;
    }
    const float inv = rsqrtf(block_sum<256>(sum, scratch) / float(cols) + eps);
    if (rstd && threadIdx.x == 0) {
        rstd[blockIdx.x] = inv;
    }
    for (int64_t col = threadIdx.x; col < cols; col += 256) {
        y[base + col] = T((float(x[base + col]) * inv) * float(weight[col]));
    }
}

template <typename T, typename W>
__global__ __launch_bounds__(256) void backward_input(
    const T *__restrict__ dy, const T *__restrict__ x, const W *__restrict__ weight, const float *__restrict__ rstd, T *__restrict__ dx, int cols) {
    __shared__ float scratch[256 / Warp::size];
    const int64_t base = int64_t(blockIdx.x) * cols;
    const float inv = rstd[blockIdx.x];
    float dot = 0.0F;
    for (int64_t col = threadIdx.x; col < cols; col += 256) {
        dot += (float(dy[base + col]) * float(weight[col])) * float(x[base + col]);
    }
    const float correction = block_sum<256>(dot, scratch) / float(cols) * inv * inv;
    for (int64_t col = threadIdx.x; col < cols; col += 256) {
        const float g = float(dy[base + col]) * float(weight[col]);
        dx[base + col] = T(inv * (g - float(x[base + col]) * correction));
    }
}

// FP32 partial sums, with fixed reduction order and no atomics. Workspace is
// at most 128 * cols floats, independent of the number of token rows.
template <typename T>
__global__ __launch_bounds__(256) void backward_weight_partial(
    const T *__restrict__ dy, const T *__restrict__ x, const float *__restrict__ rstd, float *__restrict__ partial, int64_t rows, int cols) {
    const int64_t col = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (col >= cols) {
        return;
    }
    const int part = blockIdx.y;
    const int64_t begin = rows * part / gridDim.y;
    const int64_t end = rows * (part + 1) / gridDim.y;
    float sum = 0.0F;
    for (int64_t row = begin; row < end; ++row) {
        const int64_t idx = row * cols + col;
        sum += float(dy[idx]) * (float(x[idx]) * rstd[row]);
    }
    partial[int64_t(part) * cols + col] = sum;
}

template <typename W>
__global__ __launch_bounds__(256) void backward_weight_finish(const float *__restrict__ partial, W *__restrict__ dw, int cols, int parts) {
    const int64_t col = int64_t(blockIdx.x) * 256 + threadIdx.x;
    if (col >= cols) {
        return;
    }
    float sum = 0.0F;
    for (int part = 0; part < parts; ++part) {
        sum += partial[int64_t(part) * cols + col];
    }
    dw[col] = W(sum);
}

inline void check_shape(int64_t rows, int cols) {
    if (rows < 0 || rows > INT32_MAX || cols <= 0) {
        throw std::invalid_argument("RMSNorm requires 0 <= rows <= INT32_MAX and cols > 0");
    }
}

} // namespace rmsnorm_detail

// Contiguous [..., cols] flattened to [rows, cols]. T and W may differ.
// FP32 statistics and multiplication, output dtype T. rstd may be nullptr
// for inference, otherwise it must hold rows FP32 elements for backward.
// Inputs/outputs must not overlap. Caller owns device/stream selection and
// launch-error checks.
template <typename T, typename W = T>
inline void rmsnorm_forward(
    const T *__restrict__ x, const W *__restrict__ weight, T *__restrict__ y, float *__restrict__ rstd,
    int64_t rows, int cols, float eps, gpuStream_t stream = nullptr) {
    using namespace rmsnorm_detail;
    check_shape(rows, cols);
    if (!std::isfinite(eps) || eps < 0.0F) {
        throw std::invalid_argument("RMSNorm eps must be finite and nonnegative");
    }
    if (rows == 0) {
        return;
    }
    const bool vector_aligned = cols % 4 == 0 &&
        reinterpret_cast<uintptr_t>(x) % (4 * sizeof(T)) == 0 &&
        reinterpret_cast<uintptr_t>(y) % (4 * sizeof(T)) == 0 &&
        reinterpret_cast<uintptr_t>(weight) % (4 * sizeof(W)) == 0;
    if (cols > 256 && cols <= 8192 && vector_aligned) {
        if (cols <= 1024) {
            forward_vector<T, W, 1><<<uint32_t(rows), 256, 0, stream>>>(x, weight, y, rstd, cols, eps);
        } else if (cols <= 2048) {
            forward_vector<T, W, 2><<<uint32_t(rows), 256, 0, stream>>>(x, weight, y, rstd, cols, eps);
        } else if (cols <= 4096) {
            forward_vector<T, W, 4><<<uint32_t(rows), 256, 0, stream>>>(x, weight, y, rstd, cols, eps);
        } else if (cols <= 6144) {
            forward_vector<T, W, 6><<<uint32_t(rows), 256, 0, stream>>>(x, weight, y, rstd, cols, eps);
        } else {
            forward_vector<T, W, 8><<<uint32_t(rows), 256, 0, stream>>>(x, weight, y, rstd, cols, eps);
        }
    } else if (cols <= 256) {
        const auto blocks = uint32_t((rows + 256 / Warp::size - 1) / (256 / Warp::size));
        forward_warp<T, W, 256 / Warp::size><<<blocks, 256, 0, stream>>>(
            x, weight, y, rstd, rows, cols, eps);
    } else if (cols <= 1024) {
        forward_block<T, W, 4><<<uint32_t(rows), 256, 0, stream>>>(x, weight, y, rstd, cols, eps);
    } else if (cols <= 2048) {
        forward_block<T, W, 8><<<uint32_t(rows), 256, 0, stream>>>(x, weight, y, rstd, cols, eps);
    } else if (cols <= 4096) {
        forward_block<T, W, 16><<<uint32_t(rows), 256, 0, stream>>>(x, weight, y, rstd, cols, eps);
    } else if (cols <= 8192) {
        forward_block<T, W, 32><<<uint32_t(rows), 256, 0, stream>>>(x, weight, y, rstd, cols, eps);
    } else {
        forward_wide<T, W><<<uint32_t(rows), 256, 0, stream>>>(x, weight, y, rstd, cols, eps);
    }
}

inline int rmsnorm_backward_partitions(int64_t rows) {
    return int(std::min<int64_t>(128, std::max<int64_t>(1, (rows + 31) / 32)));
}

// dx / dw may independently be nullptr. For dw, partial must contain
// rmsnorm_backward_partitions(rows) * cols FP32 elements. Buffers must not
// overlap inputs/each other. Empty rows produce zero dw without reading x.
template <typename T, typename W = T>
inline void rmsnorm_backward(
    const T *__restrict__ dy, const T *__restrict__ x, const W *__restrict__ weight, const float *__restrict__ rstd,
    T *__restrict__ dx, W *__restrict__ dw, float *__restrict__ partial, int64_t rows, int cols, gpuStream_t stream = nullptr) {
    using namespace rmsnorm_detail;
    check_shape(rows, cols);
    if (dx && rows > 0) {
        backward_input<T, W><<<uint32_t(rows), 256, 0, stream>>>(dy, x, weight, rstd, dx, cols);
    }
    if (dw) {
        const int parts = rmsnorm_backward_partitions(rows);
        const auto blocks = uint32_t((int64_t(cols) + 255) / 256);
        backward_weight_partial<T><<<dim3(blocks, parts), 256, 0, stream>>>(
            dy, x, rstd, partial, rows, cols);
        backward_weight_finish<W><<<blocks, 256, 0, stream>>>(partial, dw, cols, parts);
    }
}

} // namespace peak_gemm::kernel
