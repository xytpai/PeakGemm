## PeakGemm

Simple gemm kernel lib for peak performance on Nvidia & AMD GPUs.

### Install

```bash
git clone https://github.com/xytpai/PeakGemm
cd PeakGemm
python3 setup.py develop
```

### Test

```bash
python3 test/test_gemm.py --m=8192 --n=8192 --k=8192 --dtype=f32
```

```txt
Loading extension from: /home/yuxu/PeakGemm/libPeakGemm.so
run: /home/yuxu/PeakGemm/test/test_gemm.py, args: Namespace(m=8192, n=8192, k=8192, dtype='f32')
maxdiff_out:0.000274658203125
maxdiff_out:0.000274658203125
maxdiff_out:0.000274658203125
maxdiff_out:0.000274658203125
maxdiff_out:0.000274658203125
===================== [REF] =====================
[W417 08:44:51.691941428 collection.cpp:1133] Warning: ROCTracer produced duplicate flow start: 4 (function operator())
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT256x256x...         0.00%       0.000us         0.00%       0.000us       0.000us     580.224ms       100.00%     580.224ms       7.253ms            80  
                            hipGetDevicePropertiesR0600         0.01%      56.900us         0.01%      56.900us       0.237us       0.000us         0.00%       0.000us       0.000us           240  
                               hipExtModuleLaunchKernel         0.06%     357.591us         0.06%     357.591us       4.470us       0.000us         0.00%       0.000us       0.000us            80  
                                   hipDeviceSynchronize        99.93%     580.805ms        99.93%     580.805ms       7.170ms       0.000us         0.00%       0.000us       0.000us            81  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 581.220ms
Self CUDA time total: 580.224ms

===================== [PeakGemm] =====================
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
void sgemm::sgemm_kernel<float, 16, 2, 2, 2, 2, 8, 8...         0.00%       0.000us         0.00%       0.000us       0.000us        1.332s       100.00%        1.332s      16.646ms            80  
                                        hipLaunchKernel         0.02%     285.506us         0.02%     285.506us       3.569us       0.000us         0.00%       0.000us       0.000us            80  
                                   hipDeviceSynchronize        99.98%        1.332s        99.98%        1.332s      16.449ms       0.000us         0.00%       0.000us       0.000us            81  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.333s
Self CUDA time total: 1.332s
```
