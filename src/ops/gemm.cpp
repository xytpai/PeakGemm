#include "ops.h"
#include "sgemm_impl.h"
#include "hgemm_wmma_impl.h"

void sgemm_peak(Tensor &c, Tensor &a, Tensor &b, int64_t m, int64_t n, int64_t k) {
    TORCH_CHECK(c.is_contiguous() && a.is_contiguous() && b.is_contiguous());
    c10::DeviceGuard device_guard(a.device());
    auto stream = GET_CURRENT_STREAM;
    sgemm::sgemm_peak(
        c.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        m,
        n,
        k,
        stream);
}

void hgemm_f16_peak(Tensor &c, Tensor &a, Tensor &b, int64_t m, int64_t n, int64_t k) {
    TORCH_CHECK(c.is_contiguous() && a.is_contiguous() && b.is_contiguous());
    c10::DeviceGuard device_guard(a.device());
    auto stream = GET_CURRENT_STREAM;
    hgemm::hgemm_f16_peak(
        (__half *)c.data_ptr<c10::Half>(),
        (__half *)a.data_ptr<c10::Half>(),
        (__half *)b.data_ptr<c10::Half>(),
        m,
        n,
        k,
        stream);
}
