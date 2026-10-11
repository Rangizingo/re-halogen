// Rewrite of the "scatter" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o scatter.co scatter.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Lines that show ".long 0xD7..." plus a bogus
// v_cndmask are one VOP3 instruction (v_add_co_u32 / v_lshlrev_b64), decoded by hand.
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

// bf16 -> f32: v_lshlrev_b32 16 (codeobj_53000.s:49150).
__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// halogen::(anonymous namespace)::k_scatter_slots_dn(unsigned short const*, int, int const*, float const*, float*)
// _ZN7halogen12_GLOBAL__N_118k_scatter_slots_dnEPKtiPKiPKfPf, kernarg: x@0, n@8, slot@16, scale@24, y@32
// Block r = blockIdx.x (s2). Return if r >= n (s_cmp_ge_i32 :49095, signed).
// s = slot[r] (:49107). x row = 2560 bf16 = 0x1400 B at r*0x1400 (:49101-49102); y row = 2560 f32 = 0x2800 B
// at s*0x2800 (:49116, :49125). sc = scale[s] (:49120).
// do { y[s][j] = bf2f(x[r][j]) * sc; j += blockDim.x } while (j <= 0x9ff)   (:49144-49162; the first pass has
// no entry test, so threads must start below 2560; v_cmp_lt_u32 0x9ff < j at :49154, unsigned).
extern "C" __global__ void k_scatter_slots_dn(const unsigned short* x, int n, const int* slot, const float* scale,
                                              float* y) {
  int r = blockIdx.x;
  if (r >= n) return;
  int s = slot[r];
  float sc = scale[s];
  const unsigned short* xr = x + (long)r * 2560;
  float* yr = y + (long)s * 2560;
  unsigned j = threadIdx.x;
  do {
    yr[j] = sc * bf2f(xr[j]);
    j += blockDim.x;
  } while (j <= 0x9ffu);
}
