// Rewrite of the "attn_sel" family: k_attn_sel (codeobj_53000, flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o attn_sel.co attn_sel.hip
// Listing: kernels-disasm/codeobj_53000.s:60713-61366 (not in git). Cites are file lines.
#include <hip/hip_runtime.h>
// No mul+add fusion: every fma below is written out explicitly (the vendor uses v_fmac where it fuses).
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// k_attn_sel(q@0, k@8, v@16, tab@24 (int), cnt@32 (int), out@40, int c0@48, int m@52, gate@56)
//   _ZN7halogen12_GLOBAL__N_110k_attn_selEPKtS2_S2_PKiS4_PtiiS2_, grid (bx = token, by = head 0..23), block 256, LDS 17440 B
//   (2052 scores, 2052 row ids, 256 reduction slots; so at most 2052 keys).
//   positions npos = c0 + bx + 1 (:60723,60726); tail0 = npos with the low two bits cleared (:60731-60734);
//   n = (npos - tail0) + 4*cnt[bx] (:60735-60739). Row of key j (:60782-60811, stored at LDS +8208 + 4j, :60774):
//     j < 4*cnt[bx]: 4*tab[bx*m + (j>>2)] + (j&3) ;  else tail0 + (j - 4*cnt[bx]).
//   scores for j = tid, tid+bd, ... < n (:60851-60937): K row = k + row*512 + g*256 elements, g = by/12 (:60814-60817, 60841);
//     s = 0.0625f * dot ; dot = fma(q[d], k[d], acc) over d = 0..255 in order starting at 0 (:60870-60926), q = q + (bx*24+by)*256.
//   M = max s (per-thread max over j = tid.., tree on +16416 with st = bd/2.., :60948-61032).
//   p_j = exp2((s_j - M) * 1.442695) (no inf test, :61044-61048), per-thread sum then the same tree (:61035-61089); inv = 1.0f / sum.
//   out for d = tid, tid+bd, ... < 256 (:61140-61147): acc = fma(p_j, v[row_j*512 + g*256 + d], acc) over j = 0..n-1 in order
//   (:61168-61260, unrolled by 8), o = inv * acc, bf16 RNE; gate != null: same sigmoid-gate step as k_attn.
//   out[bx*6144 + by*256 + d] (:61102-61116).
extern "C" __global__ void k_attn_sel(const unsigned short* q, const unsigned short* k, const unsigned short* v,
                                      const int* tab, const int* cnt, unsigned short* out, int c0, int m,
                                      const unsigned short* gate) {
  __shared__ float sc[2052];
  __shared__ int rows[2052];
  __shared__ float red[256];
  const int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x, bd = blockDim.x;
  const int g = by / 12;
  const float NEG_INF = __uint_as_float(0xff800000u);
  const float L2E = __uint_as_float(0x3fb8aa3bu);
  const float SCALE = __builtin_amdgcn_rsqf(256.0f);
  int npos = c0 + bx + 1;
  int tail0 = npos & ~3;
  int c4 = 4 * cnt[bx];
  int n = (npos - tail0) + c4;
  for (int j = tid; j < n; j += bd) rows[j] = (j < c4) ? tab[(long)bx * m + (j >> 2)] * 4 + (j & 3) : tail0 + (j - c4);
  __syncthreads();
  const unsigned short* qr = q + (long)(bx * 24 + by) * 256;
  for (int j = tid; j < n; j += bd) {
    const unsigned short* kr = k + (long)rows[j] * 512 + g * 256;
    float acc = 0.0f;
    for (int d = 0; d < 256; d++) acc = __builtin_fmaf(bf2f(qr[d]), bf2f(kr[d]), acc);
    sc[j] = SCALE * acc;
  }
  __syncthreads();
  float mx = NEG_INF;
  for (int j = tid; j < n; j += bd) mx = fmaxf(mx, sc[j]);
  red[tid] = mx;
  __syncthreads();
  for (int st = bd >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] = fmaxf(red[tid], red[tid + st]);
    __syncthreads();
  }
  float M = red[0];
  __syncthreads();
  float sum = 0.0f;
  for (int j = tid; j < n; j += bd) {
    float p = __builtin_amdgcn_exp2f((sc[j] - M) * L2E);
    sc[j] = p;
    sum = sum + p;
  }
  red[tid] = sum;
  __syncthreads();
  for (int st = bd >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] = red[tid + st] + red[tid];
    __syncthreads();
  }
  float inv = 1.0f / red[0];
  for (int d = tid; d < 256; d += bd) {
    float acc = 0.0f;
    for (int j = 0; j < n; j++) acc = __builtin_fmaf(sc[j], bf2f(v[(long)rows[j] * 512 + g * 256 + d]), acc);
    unsigned short r = f2bf_rne(inv * acc);
    long oi = (long)bx * 6144 + by * 256 + d;
    if (gate) {
      float gv = bf2f(gate[oi]);
      float e = __builtin_amdgcn_exp2f(gv * __uint_as_float(0xbfb8aa3bu));
      float sig = 1.0f / (1.0f + e);
      r = f2bf_rne(sig * bf2f(r));
    }
    out[oi] = r;
  }
}
