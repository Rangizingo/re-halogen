// Rewrite of the "moe" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o moe.co moe.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// f32 -> bf16, round to nearest even without a NaN special case: v_bfe_u32 v6,v5,16,1 ; v_add3_u32 v5,v5,v6,0x7fff
// then store the high half (global_store_d16_hi_b16), codeobj_53000.s:50162-50166. The unrolled path does the
// same with v_and 1 / v_add3 (:50098-50104).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// halogen::(anonymous namespace)::k_moe_sum(unsigned short const*, unsigned short const*, float const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_19k_moe_sumEPKtS2_PKfPt, kernarg: a@0 (bf16), b@8 (bf16), w@16 (f32), out@24 (bf16)
// Block r = blockIdx.x (s2). Row = 2560 bf16 = 0x1400 B (s_mulk_i32 s2,0x1400 :50029, s_mul_hi_i32 :50027).
// wr = w[r] (s_load_b32 from w + r*4 :50032/:50036). Two code paths on blockDim.x == 1 (:50028-50030): an
// unrolled-by-2 loop for blockDim 1 and a general strided loop; both compute the same value:
//   out[r][j] = bf16_rne( fma(wr, bf2f(b[r][j]), bf2f(a[r][j])) )     v_fmac_f32 v5(a), s1(w), v4(b) :50158;
//   unrolled path v_fmac_f32 v10(a), s12(w), v11(b) :50091. a is arg0 (s4 base :50056), b is arg1 (s6 :50051).
// j runs tid, tid+blockDim.x, ... while j <= 0x9ff (v_cmp_lt_u32 0x9ff < j, :50147-50148).
extern "C" __global__ void k_moe_sum(const unsigned short* a, const unsigned short* b, const float* w,
                                     unsigned short* out) {
  long row = (long)blockIdx.x * 2560;
  float wr = w[blockIdx.x];
  for (unsigned j = threadIdx.x; j <= 0x9ffu; j += blockDim.x) {
    float v = __builtin_fmaf(wr, bf2f(b[row + j]), bf2f(a[row + j]));
    out[row + j] = f2bf_rne(v);
  }
}
