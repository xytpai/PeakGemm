#include "sgemm_impl.h"
#include "hgemm_wmma_impl.h"

namespace test {

template <typename T>
class CPUInputs {
public:
    int64_t m;
    int64_t n;
    int64_t k;
    T *c;
    T *a;
    T *b;
    T *c_dev;
    T *a_dev;
    T *b_dev;

    CPUInputs(
        int64_t m,
        int64_t n,
        int64_t k) :
        m(m),
        n(n), k(k) {
    }

    void allocate() {
        c = new T[m * n];
        a = new T[m * k];
        b = new T[n * k];
        gpuMalloc(&c_dev, m * n * sizeof(T));
        gpuMalloc(&a_dev, m * k * sizeof(T));
        gpuMalloc(&b_dev, n * k * sizeof(T));
        gpuDeviceSynchronize();
    }

    void reset() {
        for (int i = 0; i < m * n; ++i) {
            c[i] = (T)(2.f * ((rand() / (float)INT_MAX) - 0.5f));
        }
        for (int i = 0; i < m * k; ++i) {
            a[i] = (T)(2.f * ((rand() / (float)INT_MAX) - 0.5f));
        }
        for (int i = 0; i < n * k; ++i) {
            b[i] = (T)(2.f * ((rand() / (float)INT_MAX) - 0.5f));
        }
        gpuMemcpy(c_dev, c, m * n * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(a_dev, a, m * k * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(b_dev, b, n * k * sizeof(T), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    ~CPUInputs() {
        delete[] c;
        delete[] a;
        delete[] b;
        gpuFree(c_dev);
        gpuFree(a_dev);
        gpuFree(b_dev);
    }

    void operator()() {
        sgemm::sgemm_naive<T>(c_dev, a_dev, b_dev, m, n, k, 0);
        gpuDeviceSynchronize();
        gpuMemcpy(c, c_dev, m * n * sizeof(T), gpuMemcpyDeviceToHost);
        gpuDeviceSynchronize();
    }
};

template <typename T>
class GPUInputs {
public:
    int64_t m;
    int64_t n;
    int64_t k;
    T *c;
    T *a;
    T *b;

    GPUInputs(
        int64_t m,
        int64_t n,
        int64_t k) :
        m(m),
        n(n), k(k) {
    }

    void allocate() {
        gpuMalloc(&c, m * n * sizeof(T));
        gpuMalloc(&a, m * k * sizeof(T));
        gpuMalloc(&b, n * k * sizeof(T));
        gpuDeviceSynchronize();
    }

    void reset(CPUInputs<T> &inputs) {
        gpuMemcpy(c, inputs.c, m * n * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(a, inputs.a, m * k * sizeof(T), gpuMemcpyHostToDevice);
        gpuMemcpy(b, inputs.b, n * k * sizeof(T), gpuMemcpyHostToDevice);
        gpuDeviceSynchronize();
    }

    ~GPUInputs() {
        gpuFree(c);
        gpuFree(a);
        gpuFree(b);
        gpuDeviceSynchronize();
    }

    std::tuple<float, float, float> operator()() {
        gpuEvent_t start, stop;
        gpuEventCreate(&start);
        gpuEventCreate(&stop);
        gpuEventRecord(start);
        if constexpr (std::is_same_v<T, __half>) {
            hgemm::hgemm_peak((short *)c, (short *)a, (short *)b, m, n, k, false, 0);
        } else {
            hgemm::hgemm_peak((short *)c, (short *)a, (short *)b, m, n, k, true, 0);
        }
        gpuDeviceSynchronize();
        gpuEventRecord(stop);
        gpuEventSynchronize(stop);
        float ms = 0;
        gpuEventElapsedTime(&ms, start, stop);
        float input_bytes = (m * k + n * k) * sizeof(T);
        float output_bytes = (m * n) * sizeof(T);
        float gbps = (input_bytes + output_bytes) / 1000.0 / 1000.0 / ms;
        float tflops = ((float)2 * m * n * k) / (ms / 1000) * 1e-12;
        return {ms, gbps, tflops};
    }

    bool is_error(T out, T ref, float atol) {
        float out_f = (float)out;
        float ref_f = (float)ref;
        return std::isnan(out_f) || std::abs(out_f - ref_f) > atol;
    }

    bool validate(CPUInputs<T> &inputs, float atol) {
        auto out_cpu = new T[m * n];
        gpuMemcpy(out_cpu, c, m * n * sizeof(T), gpuMemcpyDeviceToHost);
        bool val = true;
        for (int i = 0; i < m * n; ++i) {
            if (is_error(out_cpu[i], inputs.c[i], atol)) {
                val = false;
                std::cout << "\n>>> out:" << (float)out_cpu[i] << ", ref_out:" << (float)inputs.c[i] << "\n";
                break;
            }
        }
        delete[] out_cpu;
        return val;
    }
};

template <typename T>
std::tuple<bool, float, float, float> runbench(
    int64_t m,
    int64_t n,
    int64_t k,
    float atol = 0.5) {
    CPUInputs<T> cpu_inputs(m, n, k);
    GPUInputs<T> gpu_inputs(m, n, k);
    cpu_inputs.allocate();
    gpu_inputs.allocate();
    cpu_inputs.reset();
    gpu_inputs.reset(cpu_inputs);
    cpu_inputs();
    auto r = gpu_inputs();
    bool val = gpu_inputs.validate(cpu_inputs, atol * k / 4096);
    return {val, std::get<0>(r), std::get<1>(r), std::get<2>(r)};
}

} // namespace test

int main() {
    std::vector<int> ms = {2048, 4096, 8192, 16384};
    std::vector<int> ns = {2048, 4096, 8192, 16384};
    std::vector<int> ks = {2048, 4096, 8192, 16384};
    for (int i = 0; i < ms.size(); ++i) {
        auto m = ms[i];
        auto n = ns[i];
        auto k = ks[i];
        std::cout << "m:" << m << ", n:" << n << ", k:" << k << ", dtype=__half";
        auto [val, ms, gbps, tflops] = test::runbench<__half>(m, n, k);
        std::cout << ", val:" << val << ", ms:" << ms << ", gbps:" << gbps << ", tflops:" << tflops << "\n";
    }
    for (int i = 0; i < ms.size(); ++i) {
        auto m = ms[i];
        auto n = ns[i];
        auto k = ks[i];
        std::cout << "m:" << m << ", n:" << n << ", k:" << k << ", dtype=__bfloat16";
        auto [val, ms, gbps, tflops] = test::runbench<__bfloat16>(m, n, k);
        std::cout << ", val:" << val << ", ms:" << ms << ", gbps:" << gbps << ", tflops:" << tflops << "\n";
    }
}
