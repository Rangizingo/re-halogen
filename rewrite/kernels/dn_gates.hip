// Rewrite of the "dn_gates" family (k_dn_gates) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dn_gates.co dn_gates.hip
// Cites are addresses in codeobj_53000 (repo listing kernels-disasm/codeobj_53000.s, same addresses).
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// softplus(v) as the vendor expands it (@23AB04-23AB98). v > 20 passes v through (v_cmpx_nlt_f32 0x41a00000: exec = !(20 < v)).
//   e = exp2(v*1.4426950216) (0x3fb8aa3b, v_exp_f32) ; s = 1 + e
//   flag = s < 2^-126 (0x800000 > s) ; s = ldexp(s, flag ? 32 : 0) ; l = v_log_f32(s)
//   p = l*0.6931471825 (0x3f317217) ; q = fma(0x3f317217, l, -p) ; q = fma(l, 0x3377d1cf, q) ; r = p + q ; l = |l| < inf ? r : l
//   result = l - (flag ? 22.18071 (0x41b17218) : 0)
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

// k_dn_gates: _ZN7halogen12_GLOBAL__N_110k_dn_gatesEPKtS2_PKfS4_PfS5_i
// kernarg: a_in@0 (bf16, s[4:5]), b_in@8 (bf16, s[6:7]), A_log@16 (f32[48], s[8:9]), dt_bias@24 (f32[48], s[10:11]),
//          beta@32 (f32 out, s[8:9] reloaded @23AADC), g@40 (f32 out, s[10:11]), rows@48 (s4 = rows*48 @23AA20).
// 1D grid, i = blockIdx.x*blockDim.x + tid, guard i < rows*48 (@23AA24-23AA38); j = i % 48 (mul_hi 0x2aaaaaab, @23AA3C-23AA70).
//   g[i]    = softplus(bf16(a_in[i]) + dt_bias[j]) * -exp2(A_log[j] * 1.4426950216)       (@23AB00, 23ABB4-23ABD4; v_mul v4, v5, -v4)
//   beta[i] = 1 / (1 + exp2(bf16(b_in[i]) * -1.4426950216))  IEEE division                (@23ABE0-23AC68)
extern "C" __global__ void k_dn_gates(const unsigned short* a_in, const unsigned short* b_in, const float* A_log,
                                      const float* dt_bias, float* beta, float* g, int rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * 48) return;
  int j = i % 48;
  float v = dt_bias[j] + bf2f(a_in[i]);
  float sp = softplus_v(v);
  float eA = __builtin_amdgcn_exp2f(A_log[j] * 1.44269502162933349609375f);
  g[i] = sp * (-eA);
  float e = __builtin_amdgcn_exp2f(bf2f(b_in[i]) * -1.44269502162933349609375f);
  beta[i] = 1.0f / (1.0f + e);
}
