// g16b extension of dnc_scan3.hip (g16, unchanged): the E>0 / F instances of k_dnc_scan3<A,B,C,D,E,F,G>, codeobj_1146000.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o x.co dnc_scan3_b.hip
// Cites: "dis:<n>" = line n of the ROCm re-decode /tmp/g16/dis/<kernel>.s on Ai-core with its 6 header lines removed (raw file line = n + 6);
// "@addr" = code-object address. Base math (g16): see dnc_scan3.hip header and listing line 154482 (@12bc00).
//
// What E and F are (proven by MATCH rows in tables/g16b.md, listing evidence below):
//   F is a bit mask of the four dot-product phases whose operands are rounded before the fma:
//     1 = pass 1   u = v - sum_j w[t][j]*S[j][col]       (w and S rounded)
//     2 = pass 2   sum_j q[t][j]*S[j][col]               (q and S rounded)
//     4 = pass 2   + sum_j' a[t][j']*u[j'][col]          (a and u rounded)
//     8 = state    S += sum_t (kk[t][j]*e1[t])*u[t][col] (the f32 product kk*e1 and u rounded)
//   Evidence for the bit assignment: conversion-instruction count per F in the E=1 listings is 384 / 48 / 3 / 192 for
//   F = 1 / 2 / 4 / 8 and 627 for F = 15 (= sum), and F=15 is the "all four" instance.
//   E selects the rounding: E in {1, 8}: fp16 round-to-nearest-even (v_cvt_f16_f32 then fma_mix_f32 / v_cvt_f32_f16,
//     dis:1110..1114 of scan3<..1,1,4>); E=2: bf16 RNE without NaN case (v_bfe_u32 16,1 / v_add3_u32 0x7fff / and 0xffff0000,
//     dis:1110..1118 of scan3<..1,2,4>); E in {10,12,..,20}: fp16 of x*2^(E-8), converted back and scaled by 2^-(E-8)
//     (s6, s7 constants, v_fma_mixlo_f16 x*s6 / v_fma_mix_f32 h*s7, dis:1109..1114 of scan3<..1,10,4>).
// The output o is always bf16 RNE of the f32 result, the accumulation chains keep the g16 order (j ascending, one accumulator).
#include <hip/hip_runtime.h>

#pragma clang fp contract(off)

__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float bf_round(float f) { return __uint_as_float((unsigned)f2bf_rne(f) << 16); }

// MODE: 0 none, 1 fp16, 2 bf16, 3 scaled fp16 (SH = E-8), 4 guarded fp16 (E=3). RMODE_BIAS is the bf16 RNE bias (negative control edits it).
template <int MODE, int SH>
__device__ static inline float rn(float x) {
  if (MODE == 1) return (float)(_Float16)x;
  if (MODE == 2) return bf_round(x);
  if (MODE == 4) {  // E=3: fp16 round only when 0x38800006 <= |x| <= 65504 (v_cmp_le/ge + v_cndmask, dis:1108..1124 of scan3<..1,3,4>)
    const float ax = __builtin_fabsf(x);
    return (ax >= __uint_as_float(0x38800006u) && ax <= __uint_as_float(0x477fe000u)) ? (float)(_Float16)x : x;
  }
  if (MODE == 3) {
    const float s6 = __uint_as_float((unsigned)(127 + SH) << 23), s7 = __uint_as_float((unsigned)(127 - SH) << 23);
    return (float)(_Float16)__builtin_fmaf(x, s6, 0.0f) * s7;
  }
  return x;
}

template <int A, int MODE, int SH, int FM>
__device__ static inline void scan3b_body(const float* __restrict__ q, const float* __restrict__ kk, const float* __restrict__ g,
                                          const float* __restrict__ v, const float* __restrict__ w, const float* __restrict__ a,
                                          float* __restrict__ st, const int* __restrict__ idx, long n, unsigned short* __restrict__ o,
                                          int k) {
  constexpr int CW = 128 / A, RT = A, SS = CW + 1;
  __shared__ float S[128 * SS];
  __shared__ float U[64 * SS];
  __shared__ float T[RT * 129];
  __shared__ float e1[64], e0[64];
  const int tid = threadIdx.x, head = blockIdx.x, y = blockIdx.y;
  const int r = tid / CW, col = tid % CW;
  float* sbase = st + n + (long)idx[0] * 0x2400000L + (long)head * 16384 + y * CW;
  for (int i = tid; i < 128 * CW; i += 128) S[(i / CW) * SS + (i % CW)] = sbase[(i / CW) * 128 + (i % CW)];
  __syncthreads();
  const int nch = (k > 0) ? ((k + 63) >> 6) : 0;
  for (int c = 0; c < nch; ++c) {
    const float glast = g[(long)c * 3072 + 3024 + head];
    if (tid < 64) {
      float gt = g[(long)(c * 64 + tid) * 48 + head];
      e1[tid] = expf(glast - gt);
      e0[tid] = expf(gt);
    }
    __syncthreads();
    for (int s10 = 0; s10 < 64; s10 += RT) {  // pass 1
      for (int i = 0; i < RT; ++i) T[i * 129 + tid] = w[(long)(c * 64 + s10 + i) * 6144 + head * 128 + tid];
      __syncthreads();
      float acc = v[(long)(c * 64 + s10 + r) * 6144 + head * 128 + y * CW + col];
      for (int j = 0; j < 128; ++j) {
        float x = T[r * 129 + j], s = S[j * SS + col];
        if (FM & 1) { x = rn<MODE, SH>(x); s = rn<MODE, SH>(s); }
        acc = __builtin_fmaf(-x, s, acc);
      }
      U[(s10 + r) * SS + col] = acc;
      __syncthreads();
    }
    for (int s10 = 0; s10 < 64; s10 += RT) {  // pass 2
      for (int i = 0; i < RT; ++i) T[i * 129 + tid] = q[(long)(c * 64 + s10 + i) * 6144 + head * 128 + tid];
      __syncthreads();
      const int tl = s10 + r, t = c * 64 + tl;
      if (t < k) {
        float acc = 0.0f;
        for (int j = 0; j < 128; ++j) {
          float x = T[r * 129 + j], s = S[j * SS + col];
          if (FM & 2) { x = rn<MODE, SH>(x); s = rn<MODE, SH>(s); }
          acc = __builtin_fmaf(x, s, acc);
        }
        float val = acc * e0[tl];
        const float* ar = a + (long)c * 196608 + head * 4096 + tl * 64;
        for (int jp = 0; jp <= tl; ++jp) {
          float x = ar[jp], u = U[jp * SS + col];
          if (FM & 4) { x = rn<MODE, SH>(x); u = rn<MODE, SH>(u); }
          val = __builtin_fmaf(x, u, val);
        }
        o[(long)t * 6144 + head * 128 + y * CW + col] = f2bf_rne(val);
      }
      __syncthreads();
    }
    const float gl = expf(glast);
    for (int i = tid; i < 128 * CW; i += 128) S[(i / CW) * SS + (i % CW)] *= gl;
    __syncthreads();
    for (int s10 = 0; s10 < 64; s10 += RT) {  // state update
      for (int i = 0; i < RT; ++i) T[i * 129 + tid] = kk[(long)(c * 64 + s10 + i) * 6144 + head * 128 + tid] * e1[s10 + i];
      __syncthreads();
      for (int i = tid; i < 128 * CW; i += 128) {
        const int j = i / CW, cc = i % CW;
        float s = S[j * SS + cc];
        for (int tt = 0; tt < RT; ++tt) {
          float x = T[tt * 129 + j], u = U[(s10 + tt) * SS + cc];
          if (FM & 8) { x = rn<MODE, SH>(x); u = rn<MODE, SH>(u); }
          s = __builtin_fmaf(x, u, s);
        }
        S[j * SS + cc] = s;
      }
      __syncthreads();
    }
  }
  for (int i = tid; i < 128 * CW; i += 128) sbase[(i / CW) * 128 + (i % CW)] = S[(i / CW) * SS + (i % CW)];
}

#define SCAN3B(NAME, AA, MODE, SH, FM)                                                                              \
  extern "C" __global__ __launch_bounds__(128) void NAME(                                                           \
      const unsigned short* keys, const float* q, const float* kk, const float* g, const float* v, const float* w,  \
      const float* a, float* st, const int* idx, long n, unsigned short* o, int k) {                                \
    scan3b_body<AA, MODE, SH, FM>(q, kk, g, v, w, a, st, idx, n, o, k);                                             \
  }
// base (E=0)
SCAN3B(scan3b_base, 8, 0, 0, 0)
SCAN3B(scan3b_a2, 2, 0, 0, 0)
SCAN3B(scan3b_a4, 4, 0, 0, 0)
SCAN3B(scan3b_a16, 16, 0, 0, 0)
SCAN3B(scan3b_a32, 32, 0, 0, 0)
#define SET(P, MODE, SH)                      \
  SCAN3B(P##_f1, 8, MODE, SH, 1)              \
  SCAN3B(P##_f2, 8, MODE, SH, 2)              \
  SCAN3B(P##_f4, 8, MODE, SH, 4)              \
  SCAN3B(P##_f8, 8, MODE, SH, 8)              \
  SCAN3B(P##_f15, 8, MODE, SH, 15)
SET(h16, 1, 0)   // E = 1, 8
SET(bf16, 2, 0)  // E = 2
SET(g16, 4, 0)   // E = 3
SET(sc2, 3, 2)   // E = 10
SET(sc4, 3, 4)   // E = 12
SET(sc6, 3, 6)   // E = 14
SET(sc8, 3, 8)   // E = 16
SET(sc10, 3, 10) // E = 18
SET(sc12, 3, 12) // E = 20

// ---- k_dnc_scan4<8,16,NA> (NA = last template arg 1,2,4,8): same math, 16-token groups and NA interleaved accumulators ----
// Evidence (decode of scan4<8,16,2>, /tmp/g16/dis): tokens per pass = 16 (v_lshrrev 4 on tid, loop stride 0x80, ds offsets 0..255 step 17);
// pass 1 u: chain A starts from v, chain B from 0, A takes even j, B odd j, u = A + B (dis:258..293);
// pass 2 q.S: two chains from 0, sum then * e0 (dis:735..766); a.u term stays one sequential chain (dis:773..783);
// state update: S is the start of chain 0 (tt even), chain 1 (tt odd) from 0, S = chain0 + chain1 (dis:980..1017).
// NA = 4, 8: same with j (or tt) mod NA, left fold ((a0+a1)+a2)+... (dis of scan4<8,16,4>:646..649, 1063..1069, 1320..1323).
template <int NA, bool BIG>
__device__ static inline void scan4b_body(const float* __restrict__ q, const float* __restrict__ kk, const float* __restrict__ g,
                                          const float* __restrict__ v, const float* __restrict__ w, const float* __restrict__ a,
                                          float* __restrict__ st, const int* __restrict__ idx, long n, unsigned short* __restrict__ o,
                                          int k) {
  constexpr int CW = 16, SS = CW + 1, TOK = 16, RR = 8;  // RR = thread rows (128 / CW)
  __shared__ float S[128 * SS];
  __shared__ float U[64 * SS];
  __shared__ float T[TOK * 129];
  __shared__ float e1[64], e0[64];
  const int tid = threadIdx.x & 127, head = blockIdx.x, y = blockIdx.y;
  const bool act = !BIG || threadIdx.x < 128;
  const int r = tid / CW, col = tid % CW;
  float* sbase = st + n + (long)idx[0] * 0x2400000L + (long)head * 16384 + y * CW;
  if (act) for (int i = tid; i < 128 * CW; i += 128) S[(i / CW) * SS + (i % CW)] = sbase[(i / CW) * 128 + (i % CW)];
  __syncthreads();
  const int nch = (k > 0) ? ((k + 63) >> 6) : 0;
  for (int c = 0; c < nch; ++c) {
    const float glast = g[(long)c * 3072 + 3024 + head];
    if (act && tid < 64) {
      float gt = g[(long)(c * 64 + tid) * 48 + head];
      e1[tid] = expf(glast - gt);
      e0[tid] = expf(gt);
    }
    __syncthreads();
    for (int s10 = 0; s10 < 64; s10 += TOK) {  // pass 1
      if (act) for (int i = 0; i < TOK; ++i) T[i * 129 + tid] = w[(long)(c * 64 + s10 + i) * 6144 + head * 128 + tid];
      __syncthreads();
      if (act) for (int rr = r; rr < TOK; rr += RR) {
        float acc[NA];
        acc[0] = v[(long)(c * 64 + s10 + rr) * 6144 + head * 128 + y * CW + col];
        for (int m = 1; m < NA; ++m) acc[m] = 0.0f;
        for (int j = 0; j < 128; ++j) acc[j % NA] = __builtin_fmaf(-T[rr * 129 + j], S[j * SS + col], acc[j % NA]);
        float u = acc[0];
        for (int m = 1; m < NA; ++m) u = u + acc[m];
        U[(s10 + rr) * SS + col] = u;
      }
      __syncthreads();
    }
    for (int s10 = 0; s10 < 64; s10 += TOK) {  // pass 2
      if (act) for (int i = 0; i < TOK; ++i) T[i * 129 + tid] = q[(long)(c * 64 + s10 + i) * 6144 + head * 128 + tid];
      __syncthreads();
      if (act) for (int rr = r; rr < TOK; rr += RR) {
        const int tl = s10 + rr, t = c * 64 + tl;
        if (t < k) {
          float acc[NA];
          for (int m = 0; m < NA; ++m) acc[m] = 0.0f;
          for (int j = 0; j < 128; ++j) acc[j % NA] = __builtin_fmaf(T[rr * 129 + j], S[j * SS + col], acc[j % NA]);
          float sum = acc[0];
          for (int m = 1; m < NA; ++m) sum = sum + acc[m];
          float val = sum * e0[tl];
          const float* ar = a + (long)c * 196608 + head * 4096 + tl * 64;
          for (int jp = 0; jp <= tl; ++jp) val = __builtin_fmaf(ar[jp], U[jp * SS + col], val);
          o[(long)t * 6144 + head * 128 + y * CW + col] = f2bf_rne(val);
        }
      }
      __syncthreads();
    }
    const float gl = expf(glast);
    if (act) for (int i = tid; i < 128 * CW; i += 128) S[(i / CW) * SS + (i % CW)] *= gl;
    __syncthreads();
    for (int s10 = 0; s10 < 64; s10 += TOK) {  // state update
      if (act) for (int i = 0; i < TOK; ++i) T[i * 129 + tid] = kk[(long)(c * 64 + s10 + i) * 6144 + head * 128 + tid] * e1[s10 + i];
      __syncthreads();
      if (act) for (int i = tid; i < 128 * CW; i += 128) {
        const int j = i / CW, cc = i % CW;
        float acc[NA];
        acc[0] = S[j * SS + cc];
        for (int m = 1; m < NA; ++m) acc[m] = 0.0f;
        for (int tt = 0; tt < TOK; ++tt) acc[tt % NA] = __builtin_fmaf(T[tt * 129 + j], U[(s10 + tt) * SS + cc], acc[tt % NA]);
        float s = acc[0];
        for (int m = 1; m < NA; ++m) s = s + acc[m];
        S[j * SS + cc] = s;
      }
      __syncthreads();
    }
  }
  if (act) for (int i = tid; i < 128 * CW; i += 128) sbase[(i / CW) * 128 + (i % CW)] = S[(i / CW) * SS + (i % CW)];
}
#define SCAN4B(NAME, NA, BIGB, LB)                                                                                           \
  extern "C" __global__ __launch_bounds__(LB) void NAME(                                                           \
      const unsigned short* keys, const float* q, const float* kk, const float* g, const float* v, const float* w,  \
      const float* a, float* st, const int* idx, long n, unsigned short* o, int k) {                                \
    scan4b_body<NA, BIGB>(q, kk, g, v, w, a, st, idx, n, o, k);                                                           \
  }
SCAN4B(scan4b_n1, 1, false, 128)
SCAN4B(scan4big_n1, 1, true, 1024)
SCAN4B(scan4b_n2, 2, false, 128)
SCAN4B(scan4big_n2, 2, true, 1024)
SCAN4B(scan4b_n4, 4, false, 128)
SCAN4B(scan4big_n4, 4, true, 1024)
SCAN4B(scan4b_n8, 8, false, 128)
SCAN4B(scan4big_n8, 8, true, 1024)
