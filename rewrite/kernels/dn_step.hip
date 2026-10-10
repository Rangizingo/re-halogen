// Rewrite of the "dn_step" family (k_dn_step_pre) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dn_step.co dn_step.hip
// Cites are addresses in codeobj_53000 (repo listing kernels-disasm/codeobj_53000.s, same addresses).
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);   // v_bfe_u32 16,1 ; v_add3_u32 0x7fff (@23C2C4-23C2DC)
}
__device__ static inline float silu_f(float v) {                      // @23C210-23C2A4
  float e = __builtin_amdgcn_exp2f(v * -1.44269502162933349609375f);
  return v / (1.0f + e);
}
// softplus as expanded at @23BF60-23BFF4 (see dn_gates.hip for the line-by-line).
__device__ static inline float softplus_v(float v) {
  if (!(20.0f < v)) {
    float e = __builtin_amdgcn_exp2f(v * 1.44269502162933349609375f);
    float s = 1.0f + e;
    bool small = __uint_as_float(0x800000u) > s;
    s = __builtin_amdgcn_ldexpf(s, small ? 32 : 0);
    float l = __builtin_amdgcn_logf(s);
    float p = l * __uint_as_float(0x3f317217u);
    bool fin = __uint_as_float(0x7f800000u) > __builtin_fabsf(l);
    float q = __builtin_fmaf(__uint_as_float(0x3f317217u), l, -p);
    q = __builtin_fmaf(l, __uint_as_float(0x3377d1cfu), q);
    float r = p + q;
    l = fin ? r : l;
    v = l - (small ? __uint_as_float(0x41b17218u) : 0.0f);
  }
  return v;
}

// k_dn_step_pre: _ZN7halogen12_GLOBAL__N_113k_dn_step_preEPKtS2_PtS3_S2_S2_PKfS5_PfS6_
// kernarg: x@0, w@8, state@16 (RW), y@24, a_in@32, b_in@40, A_log@48, dt_bias@56, beta@64, g@72.
// One launch, two jobs split on blockIdx.x (s2): s_cmp_gt_i32 s2, 39 (@23BF08).
//  blockIdx.x <= 39: conv step. c = blockIdx.x*blockDim.x + tid, guard c < 0x2800 (@23C0E4-23C0F8); blockDim from kernarg 0x5c.
//    reads state slots 1,2 (b32 at state+8c+2 @23C158) and 3 (d16 at +6 @23C168), x[c] (@23C178), w (b64 at w+8c @23C160);
//    acc = fma(s1,w0,0); fma(s2,w1); fma(s3,w2); fma(x,w3) (@23C1DC-23C210); y[c] = bf16_rne(silu(acc)) (@23C2A4-23C2F0);
//    new state {s1,s2,s3,x} stored at state+8c (global_store_b64 @23C2E8).
//  blockIdx.x >= 40: gates for the first 48 threads (v_cmpx_gt_u32 48, v0 @23BF18), j = tid:
//    g[j] = softplus(bf16(a_in[j]) + dt_bias[j]) * -exp2(A_log[j]*1.4426950216) (@23BF5C-23C01C), beta[j] = 1/(1+exp2(bf16(b_in[j])*-1.4426950216)) (@23C028-23C098)
//    stores: g -> arg 9 (s[14:15], @23C0A8), beta -> arg 8 (s[12:13], @23C0B0).
extern "C" __global__ void k_dn_step_pre(const unsigned short* x, const unsigned short* w, unsigned short* state,
                                         unsigned short* y, const unsigned short* a_in, const unsigned short* b_in,
                                         const float* A_log, const float* dt_bias, float* beta, float* g) {
  if (blockIdx.x > 39) {
    int j = threadIdx.x;
    if (j >= 48) return;
    float v = dt_bias[j] + bf2f(a_in[j]);
    float sp = softplus_v(v);
    float eA = __builtin_amdgcn_exp2f(A_log[j] * 1.44269502162933349609375f);
    g[j] = sp * (-eA);
    float e = __builtin_amdgcn_exp2f(bf2f(b_in[j]) * -1.44269502162933349609375f);
    beta[j] = 1.0f / (1.0f + e);
    return;
  }
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= 10240) return;
  unsigned short s1 = state[c * 4 + 1], s2 = state[c * 4 + 2], s3 = state[c * 4 + 3], xv = x[c];
  float acc = __builtin_fmaf(bf2f(s1), bf2f(w[c * 4 + 0]), 0.0f);
  acc = __builtin_fmaf(bf2f(s2), bf2f(w[c * 4 + 1]), acc);
  acc = __builtin_fmaf(bf2f(s3), bf2f(w[c * 4 + 2]), acc);
  acc = __builtin_fmaf(bf2f(xv), bf2f(w[c * 4 + 3]), acc);
  state[c * 4 + 0] = s1; state[c * 4 + 1] = s2; state[c * 4 + 2] = s3; state[c * 4 + 3] = xv;
  y[c] = f2bf_rne(silu_f(acc));
}
