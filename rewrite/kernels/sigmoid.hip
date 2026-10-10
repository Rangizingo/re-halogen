// Rewrites of the "sigmoid" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o sigmoid.co sigmoid.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// f32 -> bf16 RNE, no NaN special case (codeobj_53000.s:33171-33173).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// sigmoid(v) = 1 / (1 + exp2(v * -1.442695)), IEEE division (div_scale/rcp/fma/div_fmas/div_fixup).
__device__ static inline float sigm(float v) {
  float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);  // 0xbfb8aa3b, v_exp_f32
  return 1.0f / (1.0f + e);
}

// k_sigmoid(unsigned short const*, unsigned short*, long): x@0, y@8, n@16
// 64-bit index and guard (:33130-33133); y[i] = bf16_rne(sigm(bf16(x[i]))) (:33143-33173).
extern "C" __global__ void k_sigmoid(const unsigned short* x, unsigned short* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = f2bf_rne(sigm(bf2f(x[i])));
}

// k_sigmoid_f32(float const*, float*, long): x@0, y@8, n@16
// same sequence on f32 in/out, no bf16 rounding (codeobj_53000.s:49985-50011).
extern "C" __global__ void k_sigmoid_f32(const float* x, float* y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = sigm(x[i]);
}
