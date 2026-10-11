// Rewrites of the vis_bias family (two kernels), codeobj_53000.s.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3
#include <hip/hip_runtime.h>
// The vendor keeps each separate mul and add separate; every fused op below is an explicit fmaf.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// k_vis_bias (0x26a900), :63454-63664. _ZN7halogen12_GLOBAL__N_110k_vis_biasEPfPKtli
// kernarg: x@0 (f32, in/out), bias@8 (bf16), n@16 (i64), cols@24 (i32)
// 1D, i = blockIdx.x*blockDim.x + threadIdx.x (64-bit product :63460), guard i < n (:63462).
//   col = i % cols (signed 64-bit divide by the sign-extended int, :63467-63647; cols > 0 assumed)
//   x[i] = x[i] + bf16_to_f32(bias[col])   (u16 load, shl 16 :63661; f32 add :63662; store :63663)
extern "C" __global__ void k_vis_bias(float* x, const unsigned short* bias, long n, int cols) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long col = i % (long)cols;
  x[i] = x[i] + bf2f(bias[col]);
}

// k_vis_bias_gelu (0x26ae00), :63681-63948. _ZN7halogen12_GLOBAL__N_115k_vis_bias_geluEPKfPKtPtli
// kernarg: x@0 (f32), bias@8 (bf16), out@16 (bf16), n@24 (i64), cols@32 (i32)
// same indexing; u = x[i] + bf16(bias[i % cols])     (:63896-63898)
// gelu-tanh: t = 0.7978846 * (u + 0.044715*u^3) with the exact op order (:63899-63902):
//   v = u * 0x3d372713; v = u * v; v = fma(u, v, u); v = v * 0x3f4c422a
// tanh(t) as the device libm does it (:63903-63939):
//   ax = |t|; ax >= 0.625 (0x3f200000, :63903-63904):
//     arg = ax + ax (:63908); e = exp(arg) via x*log2e split (rndne, fma with 0x32a5705f correction), v_exp_f32, v_ldexp_f32;
//     e = 0 below -103.97 (0xc2ce8ed0), +inf above 88.72 (0x42b17218) (:63910-63923); r = fma(rcp(e + 1), -2, 1) (:63924-63926)
//   else: s = ax*ax; p = fmaak chain with 0xbbbac73d, 0x3ca908c9, 0xbd5c1c4e, 0x3e088382, 0xbeaaaa99 (:63929-63934);
//     p = p * ax (INFERRED: unknown VOP3 .long 0xd5080105 at :63935); r = fma(s, p, ax) (:63937)
//   t = copysign(r, t) (v_bfi_b32 :63939)
// out[i] = trunc_bf16( (0.5*u) * (1 + t) )   (:63942, :63946, global_store_d16_hi_b16 :63947)
__device__ static inline float tanh_ocml(float t) {
  float ax = fabsf(t);
  float r;
  if (ax >= 0.625f) {
    float x = ax + ax;
    float m = x * __uint_as_float(0x3fb8aa3bu);
    float rn = __builtin_rintf(m);
    float e0 = fmaf(__uint_as_float(0x3fb8aa3bu), x, -m);
    e0 = fmaf(x, __uint_as_float(0x32a5705fu), e0);
    float fr = (m - rn) + e0;
    float e = __builtin_amdgcn_exp2f(fr);
    e = __builtin_amdgcn_ldexpf(e, (int)rn);
    if (!(x >= __uint_as_float(0xc2ce8ed0u))) e = 0.0f;
    if (!(x <= __uint_as_float(0x42b17218u))) e = __uint_as_float(0x7f800000u);
    e = e + 1.0f;
    r = fmaf(__builtin_amdgcn_rcpf(e), -2.0f, 1.0f);
  } else {
    float s = ax * ax;
    float p = fmaf(__uint_as_float(0xbbbac73du), s, __uint_as_float(0x3ca908c9u));
    p = fmaf(s, p, __uint_as_float(0xbd5c1c4eu));
    p = fmaf(s, p, __uint_as_float(0x3e088382u));
    p = fmaf(s, p, __uint_as_float(0xbeaaaa99u));
    p = p * ax;
    r = fmaf(s, p, ax);
  }
  return __builtin_copysignf(r, t);
}

extern "C" __global__ void k_vis_bias_gelu(const float* x, const unsigned short* bias, unsigned short* out, long n, int cols) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long col = i % (long)cols;
  float u = x[i] + bf2f(bias[col]);
  float v = u * __uint_as_float(0x3d372713u);
  v = u * v;
  v = fmaf(u, v, u);
  v = v * __uint_as_float(0x3f4c422au);
  float t = tanh_ocml(v);
  float h = 0.5f * u;
  float g = 1.0f + t;
  out[i] = (unsigned short)(__float_as_uint(h * g) >> 16);
}
