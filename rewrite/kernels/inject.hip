// Rewrites of the "inject" family (2*sigmoid gates, f32 output) from flash_serve, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o inject.co inject.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// Shared tail: s = 1 / (1 + exp2(v * -1.44269502)) with IEEE division, result doubled, f32 stored.
// v_mul_f32 0xbfb8aa3b ; v_exp_f32 ; v_add_f32 1.0 ; div_scale/rcp/fma/div_fmas/div_fixup ; v_add_f32 d,d
// (inject_gate :33207-33231, inject_sum :19177-19201).
__device__ static inline float two_sigmoid(float v) {
  float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
  float d = 1.0f / (1.0f + e);
  return d + d;
}

// halogen::(anonymous namespace)::k_inject_gate(unsigned short const*, float*, long, float)
// _ZN7halogen12_GLOBAL__N_113k_inject_gateEPKtPflf, kernarg: x@0, y@8, n@16, scale@24.
// i = blockIdx.x*blockDim.x + tid as 64-bit (:33186), guard i < n (:33188). v = bf16(x[i]) * scale (:33204-33206). y[i] = two_sigmoid(v) (:33232).
extern "C" __global__ void k_inject_gate(const unsigned short* x, float* y, long n, float scale) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = two_sigmoid(bf2f(x[i]) * scale);
}

// halogen::(anonymous namespace)::k_inject_sum(float const*, float*, long, float)
// _ZN7halogen12_GLOBAL__N_112k_inject_sumEPKfPflf, kernarg: x@0, y@8, n@16, scale@24.
// i as above (:19123), guard i < n (:19125). g = i >> 2, j = i & 3 (:19129, :19131).
// Four f32 loads at element g*16 + j + {0,4,8,12}: byte offsets (i&~3)*16 + (i&3)*4, +16, +32, and (i&3)*4+48 (:19134-19160).
// Sum ((0 + a0) + a1) + a2) + a3 (:19162-19170); round to bf16 RNE and widen back (bfe/add3/and 0xffff0000 :19171-19174);
// * scale (:19176); then the 2*sigmoid tail (:19177-19202).
extern "C" __global__ void k_inject_sum(const float* x, float* y, long n, float scale) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long g = i >> 2;
  int j = (int)(i & 3);
  const float* p = x + g * 16 + j;
  float s = 0.0f + p[0];
  s = s + p[4];
  s = s + p[8];
  s = s + p[12];
  unsigned u = __float_as_uint(s);
  u = (u + ((u >> 16) & 1u) + 0x7fffu) & 0xffff0000u;
  y[i] = two_sigmoid(__uint_as_float(u) * scale);
}
