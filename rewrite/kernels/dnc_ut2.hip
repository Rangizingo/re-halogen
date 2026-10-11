// Rewrite of k_dnc_ut2 (same system as k_dnc_ut, K and V staged in LDS), vendor code object gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dnc_ut2.co dnc_ut2.hip
// Listing: kernels-disasm/codeobj_1146000.s (not in git), kernel at lines 2651-3443. The stock listing prints some
// 64-bit VOP3 words as ".long"; cites are raw-listing lines (those words were re-decoded with gfx1151 llvm-mc).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)  // each a*b+c in this file is an explicit __builtin_fmaf where the listing has v_fma/v_fmac

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// halogen::(anonymous namespace)::k_dnc_ut2(unsigned short const*, float const*, float const*, float const*, float*, float*, int)
// _ZN7halogen12_GLOBAL__N_19k_dnc_ut2EPKtPKfS4_S4_PfS5_i
// kernarg: mirror@0, beta@8, kk@16, g@24, out1@32, out2@40, n@48 (:2654-2656). Same arithmetic as k_dnc_ut (dnc_ut.hip); only the staging differs.
// Block 256 threads, LDS 50432 B: tile[64][129] at 0 (K, later V), T[64][65] at 33024, beta[64] at 49664, eg[64] at 49920, row scratch at 50176.
// blockIdx.x = chunk c (tb = 64c, :2658), blockIdx.y = head h (:2657).
//   phase 0 (tid < 64, :2660-2704): beta[tid] = (tb+tid < n) ? beta_in : 0; eg[tid] = expf(g[(tb+tid)*48+h]) (g unguarded).
//   K tile (:2729-2750): tile[r][d] = kk[(tb+r)*48+h][d], r < 64, no bounds check.
//   Gram (:2766-2883): for j < i: acc = fma chain over d ascending from 0 of tile[i][d]*tile[j][d] read from LDS (:2795-2843);
//     T[i][j] = (acc * -beta[i]) * expf(g[tb+i] - g[tb+j]); else 0.
//   substitution (:2885-3016): identical to k_dnc_ut (row scratch at 50176, T rows at 33024).
//   out2 pass (:3017-3203, reads the K tile): o2 = (K_i*beta[i])*eg[i]; for m < i: o2 = fma((T[i][m]*K_m)*beta[m], eg[m], o2).
//   V tile (:3204-3266): tile[r][d] = (tb+r < n) ? bf16(mirror[(tb+r)*10240 + 4096 + h*128 + d]) : 0, after a barrier.
//   out1 pass (:3267-3439): o1 = V_i*beta[i]; for m < i: o1 = fma(T[i][m]*V_m, beta[m], o1). Stores to out1/out2[(t*48+h)*128+d].
extern "C" __global__ void k_dnc_ut2(const unsigned short* mirror, const float* beta_in, const float* kk, const float* g,
                                     float* out1, float* out2, int n) {
  __shared__ float tile[64 * 129];
  __shared__ float T[64 * 65];
  __shared__ float beta[64], eg[64], rs[64];
  const int tid = threadIdx.x, c = blockIdx.x, h = blockIdx.y, tb = c * 64;
  if (tid < 64) {
    int t = tb + tid;
    beta[tid] = (t < n) ? beta_in[(size_t)t * 48 + h] : 0.0f;
    eg[tid] = expf(g[(size_t)t * 48 + h]);
  }
  for (int e = tid; e < 8192; e += 256) {
    int r = e >> 7, d = e & 127;
    tile[r * 129 + d] = kk[((size_t)(tb + r) * 48 + h) * 128 + d];
  }
  __syncthreads();
  for (int e = tid; e < 4096; e += 256) {
    int i = e >> 6, j = e & 63;
    float v = 0.0f;
    if (j < i) {
      float acc = 0.0f;
      for (int d = 0; d < 128; d++) acc = __builtin_fmaf(tile[i * 129 + d], tile[j * 129 + d], acc);
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
    float o2 = (tile[i * 129 + d] * beta[i]) * eg[i];
    for (int m = 0; m < i; m++) o2 = __builtin_fmaf((T[i * 65 + m] * tile[m * 129 + d]) * beta[m], eg[m], o2);
    out2[((size_t)t * 48 + h) * 128 + d] = o2;
  }
  __syncthreads();
  for (int e = tid; e < 8192; e += 256) {
    int r = e >> 7, d = e & 127, t = tb + r;
    tile[r * 129 + d] = (t < n) ? bf2f(mirror[(size_t)t * 10240 + 4096 + h * 128 + d]) : 0.0f;
  }
  __syncthreads();
  for (int e = tid; e < 8192; e += 256) {
    int i = e >> 7, d = e & 127, t = tb + i;
    float o1 = tile[i * 129 + d] * beta[i];
    for (int m = 0; m < i; m++) o1 = __builtin_fmaf(T[i * 65 + m] * tile[m * 129 + d], beta[m], o1);
    out1[((size_t)t * 48 + h) * 128 + d] = o1;
  }
}
