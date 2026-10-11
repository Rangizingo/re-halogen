// Rewrite of the "attn" family: k_attn (codeobj_53000, flash_serve, gfx1151), the legacy single-block attention.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o attn.co attn.hip
// Listing: kernels-disasm/codeobj_53000.s:61368-61920 (not in git). Cites are file lines.
#include <hip/hip_runtime.h>
// No mul+add fusion: every fma below is written out explicitly (the vendor uses v_fmac where it fuses).
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// k_attn(q@0, k@8, v@16, mask@24, out@32, int n@40, int c@44, gate@48)
//   _ZN7halogen12_GLOBAL__N_16k_attnEPKtS2_S2_S2_PtiiS2_, grid (bx = token, by = head 0..23), block 256 (reductions use
//   blockDim.x, assumed a power of two), LDS 17408 B = 4096 scores + 256 reduction slots.
//   g = by/12 (mulhi 0x2aaaaaab, :61373-61376). q row = q + (bx*24+by)*256 elements (:61394-61410); K/V row of key j =
//   k/v + j*512 elements + g*256 (key stride 1 KiB via lshl 10, :61445,:61850; head offset g<<9 bytes, :61414).
//   1. scores, j = tid, tid+bd, ... < n (:61424-61439):
//        valid iff j <= c + bx (v_cmpx_ge_i32 s19, v1; s19 = c + bx, :61389,:61439-61440) and (mask == null or
//        (mask[bx*n + j] & 0x7fff) != 0) (:61443-61458, mask index s18*bx + j); invalid -> -inf.
//        s = 0.0625f (v_rsq_f32 256.0, :61405) * dot ; dot = sequential fma over d = 0..255 in order, low half of
//        each dword first (:61493-61537), start 0.
//   2. M = max of scores: per-thread running max over j = tid.. step bd, then tree red[tid] = max(red[tid], red[tid+st]),
//      st = bd/2 ... 1, barrier per step (:61550-61625).
//   3. p_j = (s_j != -inf) ? exp2((s_j - M) * 1.442695) : 0 (:61639-61649), per-thread sum over j = tid.. step bd, then
//      the same tree with + (:61660-61699). sum = red[0].
//   4. inv = 1.0f / sum (IEEE division, :61709-61758). For d = tid, tid+bd, ... < 256 (:61761-61772): acc = fma(p_j, v_j[d], acc)
//      for j = 0..n-1 in order (:61796-61870, includes masked keys whose p is 0), o = inv * acc, bf16 RNE (:61873-61883);
//      gate != null: o = bf16( (1/(1+exp2(g*-1.442695))) * float(bf16 o) ), g = gate[same index] (:61886-61918).
//   out[bx*6144 + by*256 + d] (:61713-61722).
extern "C" __global__ void k_attn(const unsigned short* q, const unsigned short* k, const unsigned short* v,
                                  const unsigned short* mask, unsigned short* out, int n, int c,
                                  const unsigned short* gate) {
  __shared__ float sc[4096];
  __shared__ float red[256];
  const int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x, bd = blockDim.x;
  const int g = by / 12;
  const float NEG_INF = __uint_as_float(0xff800000u);
  const float L2E = __uint_as_float(0x3fb8aa3bu);
  const float SCALE = __builtin_amdgcn_rsqf(256.0f);
  const unsigned short* qr = q + (long)(bx * 24 + by) * 256;
  for (int j = tid; j < n; j += bd) {
    float s = NEG_INF;
    bool ok = (c + bx) >= j;
    if (ok && mask) ok = (mask[(long)n * bx + j] & 0x7fff) != 0;
    if (ok) {
      const unsigned short* kr = k + (long)j * 512 + g * 256;
      float acc = 0.0f;
      for (int d = 0; d < 256; d++) acc = __builtin_fmaf(bf2f(qr[d]), bf2f(kr[d]), acc);
      s = SCALE * acc;
    }
    sc[j] = s;
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
    float x = sc[j];
    float e = __builtin_amdgcn_exp2f((x - M) * L2E);
    float p = (x != NEG_INF) ? e : 0.0f;
    sc[j] = p;
    sum = sum + p;
  }
  red[tid] = sum;
  __syncthreads();
  for (int st = bd >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] = red[tid + st] + red[tid];
    __syncthreads();
  }
  float total = red[0];
  float inv = 1.0f / total;
  for (int d = tid; d < 256; d += bd) {
    float acc = 0.0f;
    for (int j = 0; j < n; j++) acc = __builtin_fmaf(sc[j], bf2f(v[(long)j * 512 + g * 256 + d]), acc);
    float o = inv * acc;
    unsigned short r = f2bf_rne(o);
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
