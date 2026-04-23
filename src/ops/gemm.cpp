#include "ops.h"
#include "sgemm_impl.h"
#include "hgemm_wmma_impl.h"

void gemm_peak(Tensor &c, Tensor &a, Tensor &b, int64_t m, int64_t n, int64_t k, Tensor &semaphore, Tensor &signal_state) {
    TORCH_CHECK(c.is_contiguous() && a.is_contiguous() && b.is_contiguous());
    c10::DeviceGuard device_guard(a.device());
    auto stream = GET_CURRENT_STREAM;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        kHalf,
        kBFloat16,
        a.scalar_type(),
        "gemm_peak", [&] {
            if constexpr (std::is_same_v<scalar_t, c10::Half>) {
                hgemm::hgemm_peak(
                    (short *)c.data_ptr<c10::Half>(),
                    (short *)a.data_ptr<c10::Half>(),
                    (short *)b.data_ptr<c10::Half>(),
                    m,
                    n,
                    k,
                    false,
                    semaphore.data_ptr<uint32_t>(),
                    signal_state.data_ptr<uint32_t>(),
                    stream);
            } else if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
                hgemm::hgemm_peak(
                    (short *)c.data_ptr<c10::BFloat16>(),
                    (short *)a.data_ptr<c10::BFloat16>(),
                    (short *)b.data_ptr<c10::BFloat16>(),
                    m,
                    n,
                    k,
                    true,
                    semaphore.data_ptr<uint32_t>(),
                    signal_state.data_ptr<uint32_t>(),
                    stream);
            } else {
                sgemm::sgemm_peak(
                    c.data_ptr<float>(),
                    a.data_ptr<float>(),
                    b.data_ptr<float>(),
                    m,
                    n,
                    k,
                    stream);
            }
        });
}
