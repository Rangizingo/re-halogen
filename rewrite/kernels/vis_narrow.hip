// Rewrite of k_vis_narrow (f32 -> bf16 by truncation), codeobj_53000.s:63148-63181 (kernel at 0x26a200).
// _ZN7halogen12_GLOBAL__N_112k_vis_narrowEPKfPtl, kernarg: x@0 (f32), y@8 (bf16), n@16 (i64)
#include <hip/hip_runtime.h>

// 1D, i = blockIdx.x*blockDim.x + threadIdx.x as a 64-bit product (:63157), guard i < n (:63159).
//   y[i] = high 16 bits of x[i]: global_load_b32 then global_store_d16_hi_b16, no rounding (:63175-63177).
extern "C" __global__ void k_vis_narrow(const float* x, unsigned short* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = (unsigned short)(__float_as_uint(x[i]) >> 16);
}
