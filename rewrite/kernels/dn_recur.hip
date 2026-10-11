// Rewrites of the "dn_recur" family from codeobj_53000 (flash_serve, gfx1151): k_dn_recur_step<M,B> and k_dn_recur_rows<a,b,c>.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dn_recur.co dn_recur.hip
// Cites are addresses in codeobj_53000 (repo listing kernels-disasm/codeobj_53000.s, same addresses).
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);   // v_bfe_u32 16,1 ; v_add3_u32 0x7fff (@8A90B8-8A90C0)
}

// rsqrt with the denormal guard the vendor emits (@8A8C74-8A8CF8): x < 2^-126 (0x800000 > x) scales x by 2^24 (0x4b800000)
// before v_rsq_f32 and the result by 2^12 (0x45800000).
__device__ static inline float rsq_guard(float x) {
  bool sc = __uint_as_float(0x800000u) > x;
  float r = __builtin_amdgcn_rsqf(sc ? x * 16777216.0f : x);
  return sc ? r * 4096.0f : r;
}

// softplus as expanded at @89CB30-89CBD0 (rows, in-kernel gates) and in k_dn_gates (see dn_gates.hip for the line-by-line).
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

// DnRows: 256 B by value. p[0..7] conv state write, p[8..15] conv state read (dn_conv.hip); p[16..23] = F2 (recurrent state write per row),
// p[24..31] = F3 (recurrent state read per row), cites in the rows kernels below.
struct DnRows { void* p[32]; };

// Tree sum over the 128 lanes of the block exactly as the LDS reduction does: strides 64,32,16,8,4,2,1,
// red[t] = red[t+stride] + red[t] for t < stride, a barrier between levels (@8A8928-8A8AA4).
__device__ static inline float tree128(float* red, float v) {
  int t = threadIdx.x;
  red[t] = v;
  __syncthreads();
  for (int s = 64; s >= 1; s >>= 1) {
    if (t < s) red[t] = red[t + s] + red[t];
    __syncthreads();
  }
  float r = red[0];
  __syncthreads();
  return r;
}

// Where the first (decay) pass reads the state from. S_A3: from the state buffer itself (arg 3).
// S_A11: from arg 11 (kernarg 88) unconditionally. S_A11_OR_A3: from arg 11 when non-null, else arg 3
// (s_cmp_eq_u64 s[42:43],0 / s_cselect @89F4B4-89F4BC in <2,true>). The write always goes to arg 3.
enum { S_A3 = 0, S_A11 = 1, S_A11_OR_A3 = 2 };

// k_dn_recur_step<M,B>: _ZN7halogen12_GLOBAL__N_115k_dn_recur_stepILi{M}ELb{B}EEEvPKtPKfS5_PfPtS6_S6_iS3_S3_S7_S5_
// kernarg: qkv@0 (bf16), g@8 (f32[48]), beta@16 (f32[48]), state@24 (f32 [48][128][128], RW), out@32 (bf16), kvout@40 (f32, nullable),
//          dlout@48 (f32, nullable), mode@56 (int), nw@64 (bf16 [128]), z@72 (bf16 [48*128]), nout@80 (bf16, nullable), state_in@88 (f32).
// 128 threads per block, blockIdx.x = value head h (0..47), thread d = column. Cites are from <0,false> (@8A8800), the loops are
// identical in the other instances (checked by compare.py).
template <int SRC>
__device__ static inline void dn_recur_step_body(const unsigned short* qkv, const float* g, const float* beta, float* state,
                                                 unsigned short* out, float* kvout, float* dlout, int mode,
                                                 const unsigned short* nw, const unsigned short* z, unsigned short* nout,
                                                 const float* state_in) {
  __shared__ float red[128];
  __shared__ float klds[128];
  __shared__ float qlds[128];
  const int d = threadIdx.x;
  const int h = blockIdx.x;
  const int kh = (mode != 2) ? h / 3 : h % 16;                    // @8A8810-8A8848 (mulhi 0x55555556 = /3, or mod 16)
  float q = bf2f(qkv[kh * 128 + d]);                               // @8A88D4
  float k = bf2f(qkv[2048 + kh * 128 + d]);                        // global_load_d16_b16 v2 at +0x1000 B (@8A88DC)
  float v = bf2f(qkv[4096 + h * 128 + d]);                         // @8A88E4
  float ssq_q = tree128(red, q * q);
  float ssq_k = tree128(red, k * k);
  float inv_q = rsq_guard(ssq_q + __uint_as_float(0x358637bdu));   // eps 1e-6 (@8A8C74)
  float inv_k = rsq_guard(ssq_k + __uint_as_float(0x358637bdu));
  float c128 = 128.0f;
  asm volatile("" : "+v"(c128));                                   // keep v_rsq_f32 of 128.0 a runtime op (@8A8C84)
  float rs128 = __builtin_amdgcn_rsqf(c128);
  float qh = rs128 * (inv_q * q);                                  // @8A8D10, 8A8D24
  float kh_ = inv_k * k;                                           // @8A8D34
  klds[d] = kh_;                                                   // ds_store at tid*4      (@8A8D4C)
  qlds[d] = qh;                                                    // ds_store at tid*4+512  (@8A8D44)
  __syncthreads();
  float decay = 1.0f;                                              // mode 4: skip (s_cmp_eq_u32 s28,4 @8A8D0C, s_cbranch_scc1 @8A8D64)
  if (mode != 4) decay = __builtin_amdgcn_exp2f(g[h] * 1.44269502162933349609375f);   // @8A8D84-8A8D8C
  float bt = beta[h];                                              // @8A8DA4
  float* S = state + (long)h * 16384 + d;                          // 64 KiB per head (s_lshl_b64 s[2:3],s[2:3],16 @8A8DA0)
  const float* R = S;
  if (SRC == S_A11 || (SRC == S_A11_OR_A3 && state_in)) R = state_in + (long)h * 16384 + d;
  float akv = 0.0f, aq = 0.0f;
  for (int i = 0; i < 128; ++i) {                                  // pass 1 (@8A8DD0-8A8F30): S' = decay * S ; kv += S'*k ; aq += S'*q
    float s = decay * R[i * 128];
    S[i * 128] = s;
    akv = __builtin_fmaf(s, klds[i], akv);
    aq = __builtin_fmaf(s, qlds[i], aq);
  }
  float delta = bt * (v - akv);                                    // @8A8F40-8A8F44
  float ao = 0.0f;
  for (int i = 0; i < 128; ++i) {                                  // pass 2 (@8A8F4C-8A9088): S'' = fma(delta, k, S') ; o += S''*q
    float s = __builtin_fmaf(delta, klds[i], S[i * 128]);
    S[i * 128] = s;
    ao = __builtin_fmaf(s, qlds[i], ao);
  }
  float o = (mode == 3) ? aq : ao;                                 // v_cndmask v8,v1,v2 on s_cmp_eq_u32 s28,3 (@8A9094-8A90A4)
  unsigned short ob = f2bf_rne(o);
  out[h * 128 + d] = ob;                                           // global_store_d16_hi_b16 (@8A90E8)
  if (nout) {                                                      // s_cmp_lg_u64 s[10:11],0 (@8A90E4): gated RMS norm
    float x = __uint_as_float((unsigned)ob << 16);                 // v_and 0xffff0000 (@8A90F4)
    float ss = tree128(red, x * x);
    float inv = rsq_guard(__builtin_fmaf(ss, 0.0078125f, __uint_as_float(0x358637bdu)));   // @8A92B8-8A92F8
    float xn = inv * x;                                            // @8A92FC
    float zz = bf2f(z[h * 128 + d]);                               // @8A930C
    float w = bf2f(nw[d]);                                         // @8A9314
    float e = __builtin_amdgcn_exp2f(zz * -1.44269502162933349609375f);
    float sig = 1.0f / (1.0f + e);                                 // @8A9340-8A93A0
    nout[h * 128 + d] = f2bf_rne((xn * w) * sig);                  // @8A9338, 8A93A8-8A93D8
  }
  if (kvout) kvout[h * 128 + d] = akv;                             // @8A9404
  if (dlout) dlout[h * 128 + d] = delta;                           // @8A9428
}

#define STEP_ARGS const unsigned short* qkv, const float* g, const float* beta, float* state, unsigned short* out, \
                  float* kvout, float* dlout, int mode, const unsigned short* nw, const unsigned short* z,        \
                  unsigned short* nout, const float* state_in
#define STEP_CALL qkv, g, beta, state, out, kvout, dlout, mode, nw, z, nout, state_in

extern "C" __global__ void k_dn_recur_step_s0(STEP_ARGS) { dn_recur_step_body<S_A3>(STEP_CALL); }
extern "C" __global__ void k_dn_recur_step_s1(STEP_ARGS) { dn_recur_step_body<S_A11>(STEP_CALL); }
extern "C" __global__ void k_dn_recur_step_s2(STEP_ARGS) { dn_recur_step_body<S_A11_OR_A3>(STEP_CALL); }

// ---------------------------------------------------------------------------------------------------------------
// k_dn_recur_rows<A,B,C>: _ZN7halogen12_GLOBAL__N_115k_dn_recur_rowsILb{A}ELb{B}ELb{C}EEEvPKtPKfS5_NS0_6DnRowsEPtS3_S3_S7_iS3_S3_S5_S5_
// kernarg: qkv@0 (bf16, row stride 10240), g@8 (f32 [row*48+h]), beta@16 (f32 [row*48+h]), rows@24 (DnRows, 256 B),
//          out@280 (bf16, row stride 6144), nw@288 (bf16 [128]), z@296 (bf16, row stride 6144), nout@304 (bf16, row stride 6144),
//          ntok@312 (int), a_in@320 (bf16 [row*48+h]), b_in@328 (bf16), A_log@336 (f32 [48]), dt_bias@344 (f32 [48]).
// Same recurrence as k_dn_recur_step<M,B> without the q-read pass, mode and optional outputs; blockIdx.x = head h, 128 threads, thread d.
// DnRows fields used (kernarg 24 + 0x80 + 8t = 152 + 8t, and 24 + 0xc0 + 8t): F2[t] = state written for row t, F3[t] = state read for row t.
//   A = false: the state is read from F2[t] as well (in place).  A = true: pass 1 reads F3[t] (s_load_b64 ...,0xc0 only in the A=true instances).
//   B = false: one row per block, t = blockIdx.y (s3, e.g. @89BC1C), F2[t] loaded at kernarg+8t+0x98 (@89BCF8). No ntok.
//   B = true : the block loops t = 0..ntok-1 (s_load_b32 0x138 @89D600, loop counter s16, s_cmp_eq_u32 s16,s26 @89D798).
//   C = true : g and beta are computed from a_in/b_in/A_log/dt_bias inside the kernel (s_load_b256 0x140 @89C824) instead of read from g/beta.
// The gated RMS norm of row t (out -> nout) runs after the state update of the same row; the vendor software-pipelines it into the next
// iteration (@89D76C-89D8AC), which does not change any value.
// Row body cites are from <1,1,0>: q,k,v loads @89D97C-89D98C; q/k norm @89DB1C-89DD9C; pass 1 @89DDCC-89DF18; pass 2 @89DF54-89E088;
// out store @89E0D8; norm @89D77x-89D8A4.
template <bool A, bool C>
__device__ static inline void dn_rows_row(int t, const unsigned short* qkv, const float* g, const float* beta, unsigned short* out,
                                          const unsigned short* nw, const unsigned short* z, unsigned short* nout,
                                          const unsigned short* a_in, const unsigned short* b_in, const float* A_log,
                                          const float* dt_bias, float* st_w, const float* st_r) {
  __shared__ float red[128];
  __shared__ float klds[128];
  __shared__ float qlds[128];
  const int d = threadIdx.x;
  const int h = blockIdx.x;
  const int kh = h / 3;
  const unsigned short* qr = qkv + (long)t * 10240;
  float q = bf2f(qr[kh * 128 + d]);
  float k = bf2f(qr[2048 + kh * 128 + d]);
  float v = bf2f(qr[4096 + h * 128 + d]);
  float decay, bt;
  if (C) {
    // in-kernel gates (@89CB10-89CBD0 softplus, @89D018 decay argument, @89D208-89D278 sigmoid)
    float eA = __builtin_amdgcn_exp2f(A_log[h] * 1.44269502162933349609375f);
    float sp = softplus_v(dt_bias[h] + bf2f(a_in[t * 48 + h]));
    decay = __builtin_amdgcn_exp2f((sp * eA) * -1.44269502162933349609375f);
    float e = __builtin_amdgcn_exp2f(bf2f(b_in[t * 48 + h]) * -1.44269502162933349609375f);
    bt = 1.0f / (1.0f + e);
  } else {
    decay = __builtin_amdgcn_exp2f(g[t * 48 + h] * 1.44269502162933349609375f);   // @89DCE0, 89DD7C
    bt = beta[t * 48 + h];
  }
  float ssq_q = tree128(red, q * q);
  float ssq_k = tree128(red, k * k);
  float inv_q = rsq_guard(ssq_q + __uint_as_float(0x358637bdu));
  float inv_k = rsq_guard(ssq_k + __uint_as_float(0x358637bdu));
  float c128 = 128.0f;
  asm volatile("" : "+v"(c128));
  float rs128 = __builtin_amdgcn_rsqf(c128);
  klds[d] = inv_k * k;
  qlds[d] = rs128 * (inv_q * q);
  __syncthreads();
  float* S = st_w + (long)h * 16384 + d;
  const float* R = (A ? st_r : (const float*)st_w) + (long)h * 16384 + d;
  float akv = 0.0f;
  for (int i = 0; i < 128; ++i) {
    float s = decay * R[i * 128];
    S[i * 128] = s;
    akv = __builtin_fmaf(s, klds[i], akv);
  }
  float delta = bt * (v - akv);
  float ao = 0.0f;
  for (int i = 0; i < 128; ++i) {
    float s = __builtin_fmaf(delta, klds[i], S[i * 128]);
    S[i * 128] = s;
    ao = __builtin_fmaf(s, qlds[i], ao);
  }
  unsigned short ob = f2bf_rne(ao);
  out[(long)t * 6144 + h * 128 + d] = ob;
  float x = __uint_as_float((unsigned)ob << 16);
  float ss = tree128(red, x * x);
  float inv = rsq_guard(__builtin_fmaf(ss, 0.0078125f, __uint_as_float(0x358637bdu)));
  float xn = inv * x;
  float zz = bf2f(z[(long)t * 6144 + h * 128 + d]);
  float w = bf2f(nw[d]);
  float e = __builtin_amdgcn_exp2f(zz * -1.44269502162933349609375f);
  float sig = 1.0f / (1.0f + e);
  nout[(long)t * 6144 + h * 128 + d] = f2bf_rne((xn * w) * sig);
  __syncthreads();
}

template <bool A, bool B, bool C>
__device__ static inline void dn_rows_body(const unsigned short* qkv, const float* g, const float* beta, DnRows rows,
                                           unsigned short* out, const unsigned short* nw, const unsigned short* z,
                                           unsigned short* nout, int ntok, const unsigned short* a_in,
                                           const unsigned short* b_in, const float* A_log, const float* dt_bias) {
  if (B) {
    for (int t = 0; t < ntok; ++t)
      dn_rows_row<A, C>(t, qkv, g, beta, out, nw, z, nout, a_in, b_in, A_log, dt_bias, (float*)rows.p[16 + t],
                        (const float*)rows.p[24 + t]);
  } else {
    int t = blockIdx.y;
    dn_rows_row<A, C>(t, qkv, g, beta, out, nw, z, nout, a_in, b_in, A_log, dt_bias, (float*)rows.p[16 + t],
                      (const float*)rows.p[24 + t]);
  }
}

#define ROWS_ARGS const unsigned short* qkv, const float* g, const float* beta, DnRows rows, unsigned short* out,       \
                  const unsigned short* nw, const unsigned short* z, unsigned short* nout, int ntok,                     \
                  const unsigned short* a_in, const unsigned short* b_in, const float* A_log, const float* dt_bias
#define ROWS_CALL qkv, g, beta, rows, out, nw, z, nout, ntok, a_in, b_in, A_log, dt_bias
#define ROWS_KERNEL(NAME, A, B, C) \
  extern "C" __global__ void NAME(ROWS_ARGS) { dn_rows_body<A, B, C>(ROWS_CALL); }

ROWS_KERNEL(k_dn_recur_rows_000, false, false, false)
ROWS_KERNEL(k_dn_recur_rows_001, false, false, true)
ROWS_KERNEL(k_dn_recur_rows_010, false, true, false)
ROWS_KERNEL(k_dn_recur_rows_011, false, true, true)
ROWS_KERNEL(k_dn_recur_rows_110, true, true, false)
ROWS_KERNEL(k_dn_recur_rows_111, true, true, true)
