// Rewrite of k_vis_add, codeobj_53000.s:64866-64897 (kernel at 0x26c800).
// _ZN7halogen12_GLOBAL__N_19k_vis_addEPfPKfl, kernarg: a@0 (f32, in/out), b@8 (f32), n@16 (i64)
#include <hip/hip_runtime.h>

// 1D, i = blockIdx.x*blockDim.x + threadIdx.x as a 64-bit product (v_mad_u64_u32 :64875), guard i < n (:64877).
//   a[i] = b[i] + a[i]   (loads :64892-64893, v_add_f32 :64895, store to a :64896)
extern "C" __global__ void k_vis_add(float* a, const float* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  a[i] = b[i] + a[i];
}
