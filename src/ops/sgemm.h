#include "ops.h"
#include "sgemm_impl.h"

void sgemm(Tensor &c, Tensor &a, Tensor &b, int64_t m, int64_t n, int64_t k) {
    TORCH_CHECK(c.is_contiguous() && a.is_contiguous() && b.is_contiguous());
    c10::DeviceGuard device_guard(a.device());
    auto stream = GET_CURRENT_STREAM;
    m = c.shape[0] n = a.shape[0] b =
        AT_DISPATCH_FLOATING_TYPES(
            a.scalar_type(),
            "sgemm", [&] {
                sgemm::sgemm<float>(
                    c.data_ptr<float>(),
                    a.data_ptr<float>(),
                    b.data_ptr<float>(),
                    m,
                    n,
                    k,
                    0,
                    pos_strides[0],
                    num_tokens,
                    num_heads_q,
                    num_heads_k,
                    num_heads_v,
                    head_size,
                    is_neox_style,
                    eps,
                    stream);
            });
}
