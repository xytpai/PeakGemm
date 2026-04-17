#include "ops.h"
#include "sgemm_impl.h"

void sgemm_peak(Tensor &c, Tensor &a, Tensor &b, int64_t m, int64_t n, int64_t k) {
    TORCH_CHECK(c.is_contiguous() && a.is_contiguous() && b.is_contiguous());
    c10::DeviceGuard device_guard(a.device());
    auto stream = GET_CURRENT_STREAM;
    AT_DISPATCH_FLOATING_TYPES(
        a.scalar_type(),
        "sgemm", [&] {
            sgemm::sgemm_peak(
                c.data_ptr<float>(),
                a.data_ptr<float>(),
                b.data_ptr<float>(),
                m,
                n,
                k,
                stream);
        });
}
