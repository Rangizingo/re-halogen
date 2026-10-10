// Rewrites of the "glu" family (SwiGLU) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o glu.co glu.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// out = silu(g) * u : e = exp2(g * -1.442695); s = g / (1 + e) IEEE division; r = s * u; bf16 RNE
// (codeobj_53000.s:49929-49959 for k_glu, :48840-48864 for k_glu_slots, :48929-48953 for k_glu_rows).
__device__ static inline unsigned short swiglu(unsigned short gate, unsigned short up) {
  float g = bf2f(gate);
  float e = __builtin_amdgcn_exp2f(g * -1.44269502162933349609375f);
  float s = g / (1.0f + e);
  return f2bf_rne(s * bf2f(up));
}

// k_glu(unsigned short const*, unsigned short const*, unsigned short*, long)
// _ZN7halogen12_GLOBAL__N_15k_gluEPKtS2_Ptl, kernarg: gate@0, up@8, out@16, n@24
// 64-bit index (:49910) and guard (:49912); gate is the first load (:49921, arg@0), up the second (:49926, arg@8).
extern "C" __global__ void k_glu(const unsigned short* gate, const unsigned short* up, unsigned short* out, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = swiglu(gate[i], up[i]);
}

// k_glu_rows(unsigned short const*, int, long, unsigned short*)
// _ZN7halogen12_GLOBAL__N_110k_glu_rowsEPKtilPt, kernarg: in@0, nrows@8 (i32), row0@16 (i64), out@24
// One block per row: row = blockIdx.x, exit if row >= nrows (:48878-48879).
// Input row stride 2560 bytes = 1280 bf16 (s_mulk 0xa00, :48889); gate = in[row*1280 + j], up = in[row*1280 + 640 + j]
// (loads at byte offset -1280 and 0 from a pointer advanced by 0x500, :48908, :48915-48916).
// Output row (row0 + row) of 1280 bytes = 640 bf16 (s_mul 0x500, :48895-48897). j runs tid, tid+blockDim, ... < 640
// (0x280 test :48881, loop bound 0x27f :48923). Result stored per :48953.
extern "C" __global__ void k_glu_rows(const unsigned short* in, int nrows, long row0, unsigned short* out) {
  int row = blockIdx.x;
  if (row >= nrows) return;
  const unsigned short* src = in + (long)row * 1280;
  unsigned short* dst = out + (row0 + row) * 640;
  for (int j = threadIdx.x; j < 640; j += blockDim.x) dst[j] = swiglu(src[j], src[640 + j]);
}

// k_glu_slots(unsigned short const*, int, int const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_111k_glu_slotsEPKtiPKiPt, kernarg: in@0, nrows@8, slot@16 (i32*), out@24
// Same as k_glu_rows but the output row is slot[row] (s_load_b32 of ids at :48803, scaled by 0x500 bytes :48816).
extern "C" __global__ void k_glu_slots(const unsigned short* in, int nrows, const int* slot, unsigned short* out) {
  int row = blockIdx.x;
  if (row >= nrows) return;
  const unsigned short* src = in + (long)row * 1280;
  unsigned short* dst = out + (long)slot[row] * 640;
  for (int j = threadIdx.x; j < 640; j += blockDim.x) dst[j] = swiglu(src[j], src[640 + j]);
}
