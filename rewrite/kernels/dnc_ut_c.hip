// g24 NaN-sign copy of dnc_ut.hip (original untouched). Rewrite of k_dnc_ut (per-chunk triangular system and weighted sums), vendor code object gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_ut.co dnc_ut.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git), kernel at lines 1866-2455. The stock listing prints some
// 64-bit VOP3 words as ".long"; cites are raw-listing lines (those words were re-decoded with gfx1151 llvm-mc).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)  // each a*b+c in this file is an explicit __builtin_fmaf where the listing has v_fma/v_fmac

// g24: output-pass products with the vendor's operand order. AMD returns the first NaN operand (src0, src1, src2) of
// v_mul_f32 / v_fma_f32; hipcc commutes the multiplies, so tij * vm became vm * tij and a tij = -NaN (0xffc00000, made by
// inf*0 in the Gram dot) lost to a bf16 +NaN. Vendor listing (v2, codeobj_1146000.s, rotated unroll by 2, m even half):
//   :2103 `v_mul_f32 v31, v27, v31` (tij * kk[m]);  :2107 `v_mul_f32 v27, v27, v29` (tij * vm);
//   :2108 `v_mul_f32 v29, v11, v31` (beta[m] * (tij*kk));  :2113 `v_fmac_f32 v25, v12, v29` (o2 += eg[m] * that);
//   :2117 `v_fmac_f32 v24, v27, v11` (o1 += (tij*vm) * beta[m]);  :2080 `v_mul v24, v7, v8` (vm * beta[i]);
//   :2082 `v_mul v10, v8, v10` (beta[i] * kk);  :2084 `v_mul v25, v10, v9` (* eg[i]).
__device__ static inline float mulv(float a, float b) {
  float r;
  asm("v_mul_f32 %0, %1, %2" : "=v"(r) : "v"(a), "v"(b));
  return r;
}
__device__ static inline float fmav(float a, float b, float c) {
  float r;
  asm("v_fma_f32 %0, %1, %2, %3" : "=v"(r) : "v"(a), "v"(b), "v"(c));
  return r;
}

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// halogen::(anonymous namespace)::k_dnc_ut(unsigned short const*, float const*, float const*, float const*, float*, float*, int)
// _ZN7halogen12_GLOBAL__N_18k_dnc_utEPKtPKfS4_S4_PfS5_i
// kernarg: mirror@0 (u16), beta@8, kk@16, g@24, out1@32, out2@40, n@48 (:1869-1871)
// Block 256 threads, LDS 17408 B. blockIdx.x = chunk c (token base tb = 64c, :1873), blockIdx.y = head h (:1872).
// mirror: [token][10240] u16 (bf16); the v section starts at u16 index 4096 + h*128 (0x2000 B + h*256 B, :2237-2319).
// beta, g: [token][48] f32. kk, out1, out2: [token][48][128] f32.
//   LDS: T[64][65] floats at 0 (row stride 0x104 B), beta[64] at 16640, eg[64] at 16896, row scratch r[64] at 17152.
//   phase 0 (tid < 64, :1875-1919): t = tb + tid; beta[tid] = (t < n) ? beta_in[t*48+h] : 0 (signed compare, :1881-1893);
//     eg[tid] = expf(g[t*48+h]) with the full-precision expansion, g read with no bounds check (:1896-1919).
//   phase 1 (e = tid, tid+256, ... < 4096, :1946-2073): i = e>>6, j = e&63. j < i (strict, :1961):
//     dot = sum_{d=0..127} kk[tb+i][h][d] * kk[tb+j][h][d], one fma chain ascending d from 0, no bounds check (:1990-2029);
//     T[i][j] = (dot * -beta[i]) * expf(g[tb+i] - g[tb+j]);  else T[i][j] = 0.  (:2044-2073, v_mul_f32 with negated beta then times exp)
//   phase 2 (:2075-2205): for i = 1..63: threads j < i: acc = T[i][j]; for m = j+1 .. i-1 (ascending):
//     acc = fma(Told[i][m], T[m][j], acc), where Told[i][*] is row i as it was before this step (copied to the scratch at :2105-2108); T[i][j] = acc.
//   phase 3 (e = tid, tid+256, ... < 8192, :2299-2452): i = e>>7, d = e&127, t = tb+i.
//     v = (t < n) ? bf16(mirror[t*10240 + 4096 + h*128 + d]) : 0   (:2304-2322)
//     out1 = v*beta[i]; out2 = (beta[i]*kk[t][h][d]) * eg[i]  (:2336-2340; kk read with no bounds check)
//     i > 0: for m = 0..i-1 ascending (:2356-2383): vm = (tb+m < n) ? bf16(mirror[...m...]) : 0;
//        out1 = fma(T[i][m]*vm, beta[m], out1); out2 = fma(eg[m], beta[m]*(T[i][m]*kk[tb+m][h][d]), out2)
//     stores to out1[(t*48+h)*128+d] and out2 (same index) for every e (:2295-2296).
extern "C" __global__ void k_dnc_ut(const unsigned short* mirror, const float* beta_in, const float* kk, const float* g,
                                    float* out1, float* out2, int n) {
  __shared__ float T[64 * 65];
  __shared__ float beta[64], eg[64], rs[64];
  const int tid = threadIdx.x, c = blockIdx.x, h = blockIdx.y, tb = c * 64;
  if (tid < 64) {
    int t = tb + tid;
    beta[tid] = (t < n) ? beta_in[(size_t)t * 48 + h] : 0.0f;
    eg[tid] = expf(g[(size_t)t * 48 + h]);
  }
  __syncthreads();
  for (int e = tid; e < 4096; e += 256) {
    int i = e >> 6, j = e & 63;
    float v = 0.0f;
    if (j < i) {
      const float* ki = kk + ((size_t)(tb + i) * 48 + h) * 128;
      const float* kj = kk + ((size_t)(tb + j) * 48 + h) * 128;
      float acc = 0.0f;
      for (int d = 0; d < 128; d++) acc = __builtin_fmaf(ki[d], kj[d], acc);
      float ex = expf(g[(size_t)(tb + i) * 48 + h] - g[(size_t)(tb + j) * 48 + h]);
      v = (acc * -beta[i]) * ex;
    }
    T[i * 65 + j] = v;
  }
  __syncthreads();
  for (int i = 1; i < 64; i++) {
    if (tid < i) rs[tid] = T[i * 65 + tid];
    __syncthreads();
    if (tid < i) {
      float acc = rs[tid];
      for (int m = tid + 1; m < i; m++) acc = __builtin_fmaf(rs[m], T[m * 65 + tid], acc);
      T[i * 65 + tid] = acc;
    }
    __syncthreads();
  }
  for (int e = tid; e < 8192; e += 256) {
    int i = e >> 7, d = e & 127, t = tb + i;
    size_t oi = ((size_t)t * 48 + h) * 128 + d;
    float vi = (t < n) ? bf2f(mirror[(size_t)t * 10240 + 4096 + h * 128 + d]) : 0.0f;
    float o1 = mulv(vi, beta[i]);
    float o2 = mulv(mulv(beta[i], kk[oi]), eg[i]);
    for (int m = 0; m < i; m++) {
      int tm = tb + m;
      float vm = (tm < n) ? bf2f(mirror[(size_t)tm * 10240 + 4096 + h * 128 + d]) : 0.0f;
      float tkm = kk[((size_t)tm * 48 + h) * 128 + d];
      float tij = T[i * 65 + m];
      o1 = fmav(mulv(tij, vm), beta[m], o1);
      o2 = fmav(eg[m], mulv(beta[m], mulv(tij, tkm)), o2);
    }
    out1[oi] = o1;
    out2[oi] = o2;
  }
}
