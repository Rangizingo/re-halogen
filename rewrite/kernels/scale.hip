// Rewrites of the "scale" family from the vendor code objects (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o scale.co scale.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git; regenerate with llvm-objdump --mattr=+gfx1151).
#include <hip/hip_runtime.h>

// bf16 -> f32: the vendor does v_lshlrev_b32 16 (codeobj_53000.s:33087).
__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// f32 -> bf16, round to nearest even without a NaN special case: v_bfe_u32 v3,v2,16,1 ; v_add3_u32 v2,v2,v3,0x7fff
// then store the high half (codeobj_53000.s:33114-33117).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// halogen::(anonymous namespace)::k_scale_silu(unsigned short const*, unsigned short*, long, float)
// _ZN7halogen12_GLOBAL__N_112k_scale_siluEPKtPtlf, kernarg: x@0, y@8, n@16, scale@24
// 1D, i = blockIdx.x*blockDim.x + threadIdx.x as a 64-bit product (v_mad_u64_u32, :33071), guard i < n (:33072-33075).
//   v = bf16(x[i]) * scale                              (:33087-33089)
//   e = exp2(v * -1.442695)  (0xbfb8aa3b, v_exp_f32)    (:33090-33092)
//   y[i] = bf16_rne(v / (1 + e)), IEEE division         (:33093-33117, div_scale/rcp/fma/div_fmas/div_fixup)
extern "C" __global__ void k_scale_silu(const unsigned short* x, unsigned short* y, long n, float scale) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = bf2f(x[i]) * scale;
  float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
  y[i] = f2bf_rne(v / (1.0f + e));
}
