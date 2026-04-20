## PeakGemm

Simple gemm kernel lib for peak performance on Nvidia & AMD GPUs.


### Performance

| OP | DEVICE | SHAPE | TIME | PERF | PEAK | RATIO |
|--|--|--|--|--|--|--|
| hgemm-wmma-bf16 | CUDA-RTX4090 | m=8192,n=8192,k=8192 | 6.7 ms | 163.5 TFLOPS | 165.2 TFLOPS | 99% |
| hgemm-wmma-bf16 | CUDA-H20 | m=8192,n=8192,k=8192 | 12.2 ms | 89.5 TFLOPS | 95 TFLOPS | 94% |

### Install

```bash
git clone https://github.com/xytpai/PeakGemm
cd PeakGemm
python3 -m pip install -e . --no-build-isolation
```

### Cpp Test

```bash
bash build_rocm.sh test/cpp/test_gemm_half.cpp ; ./a.out
```

```txt
m:2048, n:2048, k:2048, dtype=__half, val:1, ms:0.06532, gbps:385.27, tflops:263.011
m:4096, n:4096, k:4096, dtype=__half, val:1, ms:0.201441, gbps:499.716, tflops:682.279
m:8192, n:8192, k:8192, dtype=__half, val:1, ms:1.47965, gbps:272.127, tflops:743.089
m:16384, n:16384, k:16384, dtype=__half, val:1, ms:15.3889, gbps:104.661, tflops:571.587
m:2048, n:2048, k:2048, dtype=__bfloat16, val:1, ms:0.07468, gbps:336.982, tflops:230.046
m:4096, n:4096, k:4096, dtype=__bfloat16, val:1, ms:0.196121, gbps:513.271, tflops:700.786
m:8192, n:8192, k:8192, dtype=__bfloat16, val:1, ms:1.45917, gbps:275.947, tflops:753.518
m:16384, n:16384, k:16384, dtype=__bfloat16, val:1, ms:15.0604, gbps:106.944, tflops:584.056
```

### Python Test

```bash
python3 test/test_gemm.py --m=4096 --n=4096 --k=4096 --dtype=f32
```

```txt
run: /workspace/xyt/PeakGemm/test/test_gemm.py, args: Namespace(m=4096, n=4096, k=4096, dtype='f32')
/workspace/xyt/PeakGemm/test/test_gemm.py:43: UserWarning: NOTE: The SGEMM has not been optimized. It's treated as a reference path.
  warnings.warn('NOTE: The SGEMM has not been optimized. It\'s treated as a reference path.')
maxdiff_out:0.0
maxdiff_out:0.0
maxdiff_out:0.0
maxdiff_out:0.0
maxdiff_out:0.0
===================== [REF] =====================
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
sm80_xmma_gemm_f32f32_f32f32_f32_tn_n_tilesize64x64x...         0.00%       0.000us         0.00%       0.000us       0.000us     234.070ms       100.00%     234.070ms       5.852ms            40  
                                    cudaLaunchKernelExC         0.09%     204.768us         0.81%       1.899ms      47.475us       0.000us         0.00%       0.000us       0.000us            40  
                                Activity Buffer Request         0.72%       1.694ms         0.72%       1.694ms       1.694ms       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaDeviceSynchronize        99.19%     232.244ms        99.19%     232.244ms       5.664ms       0.000us         0.00%       0.000us       0.000us            41  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 234.143ms
Self CUDA time total: 234.070ms

===================== [PeakGemm] =====================
/workspace/xyt/PeakGemm/test/test_gemm.py:43: UserWarning: NOTE: The SGEMM has not been optimized. It's treated as a reference path.
  warnings.warn('NOTE: The SGEMM has not been optimized. It\'s treated as a reference path.')
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void sgemm::sgemm_kernel<float, 16, 4, 2, 2, 2, 4, 8...         0.00%       0.000us         0.00%       0.000us       0.000us     219.071ms       100.00%     219.071ms       5.477ms            40  
                                       cudaLaunchKernel         0.09%     186.927us         0.86%       1.897ms      47.434us       0.000us         0.00%       0.000us       0.000us            40  
                                Activity Buffer Request         0.78%       1.710ms         0.78%       1.710ms       1.710ms       0.000us         0.00%       0.000us       0.000us             1  
                                  cudaDeviceSynchronize        99.14%     217.459ms        99.14%     217.459ms       5.304ms       0.000us         0.00%       0.000us       0.000us            41  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 219.357ms
Self CUDA time total: 219.071ms
```
