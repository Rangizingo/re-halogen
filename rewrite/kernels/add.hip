// Rewrites of the "add" family (bf16 adds) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o add.co add.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// bf16 -> f32: v_lshlrev_b32 16 (codeobj_53000.s:50801, 60702).
__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// f32 -> bf16 RNE, no NaN special case: v_bfe_u32 v3,v2,16,1 ; v_add3_u32 v2,v2,v3,0x7fff ; store high half
// (codeobj_53000.s:50806-50809, 60707-60709).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// halogen::(anonymous namespace)::k_add_bf16(unsigned short*, unsigned short const*, long)
// _ZN7halogen12_GLOBAL__N_110k_add_bf16EPtPKtl, kernarg: a@0 (in/out), b@8, n@16
// i = blockIdx.x*blockDim.x + threadIdx.x as 64-bit v_mad_u64_u32 (:50781); guard i < n (the .long at :50783 is a
// 64-bit compare, inferred); a[i] = bf16_rne(a[i] + b[i]) (loads :50798-50799, add :50805, store back to a :50809).
extern "C" __global__ void k_add_bf16_inplace(unsigned short* a, const unsigned short* b, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  a[i] = f2bf_rne(bf2f(a[i]) + bf2f(b[i]));
}

// k_add_bf16(unsigned short const*, unsigned short const*, unsigned short*, int)
// _ZN7halogen12_GLOBAL__N_110k_add_bf16EPKtS2_Pti, kernarg: a@0, b@8, out@16, n@24 (i32)
// i = blockIdx.x*blockDim.x + threadIdx.x, 32-bit guard i < n (:60676-60680), sign-extended index (:60682).
// out[i] = bf16_rne(a[i] + b[i]) (:60695-60709).
extern "C" __global__ void k_add_bf16_out(const unsigned short* a, const unsigned short* b, unsigned short* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = f2bf_rne(bf2f(a[i]) + bf2f(b[i]));
}
