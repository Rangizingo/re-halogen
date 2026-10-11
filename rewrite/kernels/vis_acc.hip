// Rewrite of k_vis_acc, codeobj_53000.s:64758-64789 (kernel at 0x26c500).
// _ZN7halogen12_GLOBAL__N_19k_vis_accEPfPKfl, kernarg: a@0 (f32, in/out), b@8 (f32), n@16 (i64)
#include <hip/hip_runtime.h>

// The ISA is instruction-for-instruction the same as k_vis_add (:64761-64789 vs :64869-64897).
// 1D, i = blockIdx.x*blockDim.x + threadIdx.x as a 64-bit product (:64767), guard i < n (:64769).
//   a[i] = b[i] + a[i]   (loads :64784-64785, v_add_f32 :64787, store to a :64788)
extern "C" __global__ void k_vis_acc(float* a, const float* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  a[i] = b[i] + a[i];
}
