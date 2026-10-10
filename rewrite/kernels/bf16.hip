// Rewrite of k_bf16_to_f32 (codeobj_53000, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o bf16.co bf16.hip
// Listing: kernels-disasm/codeobj_53000.s
#include <hip/hip_runtime.h>

// halogen::(anonymous namespace)::k_bf16_to_f32(unsigned short const*, float*, long)
// _ZN7halogen12_GLOBAL__N_113k_bf16_to_f32EPKtPfl, kernarg: x@0, y@8, n@16
// i = blockIdx.x*blockDim.x + threadIdx.x as a 64-bit product (v_mad_u64_u32, :18564), guard i < n (:18566, undecoded .long compare, inferred),
// y[i] = x[i] << 16 (global_load_u16 :18583, v_lshlrev_b32 16 :18585, global_store_b32 :18586).
extern "C" __global__ void k_bf16_to_f32(const unsigned short* x, float* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = __uint_as_float((unsigned)x[i] << 16);
}
