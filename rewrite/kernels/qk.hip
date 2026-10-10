// Rewrite of the "qk" family: k_qk_prep (codeobj_53000, flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o qk.co qk.hip
// Listing: kernels-disasm/codeobj_53000.s lines 61998-62371 (not in git). Cites below are "lq:<n>" with n the
// instruction's position in that listing region as read with the s_delay_alu/s_nop lines removed; the region start
// line is 61998 so the raw file line is within a few hundred lines of 61998+n.
#include <hip/hip_runtime.h>
// No mul+add fusion: the vendor fuses only where it emits v_fma/v_fmac, and every fma below is written out explicitly.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// ln(x) as the vendor expands it (same sequence as rope.hip vlog; :62280-62304 here): hipcc's logf differs by 1 ulp.
__device__ static inline float vlog(float x) {
  bool dn = x < 1.17549435e-38f;
  float l = __builtin_amdgcn_logf(dn ? x * 4294967296.0f : x);
  float p = l * __uint_as_float(0x3f317217u);
  float n = __builtin_fmaf(l, __uint_as_float(0x3f317217u), -p);
  n = __builtin_fmaf(l, __uint_as_float(0x3377d1cfu), n);
  float r = p + n;
  r = (__builtin_fabsf(l) < __uint_as_float(0x7f800000u)) ? r : l;
  return r - (dn ? __uint_as_float(0x41b17218u) : 0.0f);
}

// FdRows (160 B by value at kernarg +0x48): pointer arrays at +0 and +0x40 (s_load_b64 ... null / 0x40), ints at +0x80
// (s_load_b32 ... 0x80, minus the 4*i bias: address = base + 8*i - 4*i + 0x80). Same layout as kv.hip.
struct FdRows {
  unsigned short* p0[8];
  unsigned short* p1[8];
  int slot[8];
};

// k_qk_prep(q@0, k@8, v@16, qw@24, kw@32, a@40, b@44, c@48, theta@52, d@56, pos@64, FdRows@72)
//   _ZN7halogen12_GLOBAL__N_19k_qk_prepEPtS1_PKtS3_S3_iiifiPKiNS0_6FdRowsE, block 256, LDS 1536.
// Blocks (blockIdx.x = s2): [0, 24a) are Q heads (24 per token), [24a, 26a) are K heads (2 per token), [26a, 27a) copy
// one token's V row (:62008-62010 split at 26*a, :62116-62124 split at 24*a).
//   V block (idx = bx-26a): pool_v[p1[idx] + slot[idx]*512 + i] = v[idx*512 + i], i = tid..511 step blockDim
//     (src s8 + idx<<10 bytes :62042, dst p1[idx]+slot*1024 B :62028-62033, scalar u16 loop :62081-62110; the b64 path at
//     :62036-62073 only runs when blockDim.x == 1).
//   Q/K block: row x = q (k) + head*256 elements (s_lshl_b64 9, :62128-62130), weight = qw (kw) (K swaps s[10:11] and s[4:5]
//     with s[16:17], s[6:7] at :62124-62125), token = head / 24 (/ 2) (the s_mul_hi unsigned division, :62216-62246).
//   1. sumsq: for i = tid; i < b; i += blockDim: s = fma(x,x,s) (v_fmac :62155)
//   2. LDS tree: red[tid] = s ; barrier ; for (st = blockDim>>1; st > 0; st >>= 1) { if (tid < st) red[tid] = red[tid+st]+red[tid]; barrier }
//      (v_lshrrev_b16 start, :62161-62195)
//   3. mean = red[0] / (float)b, IEEE division (:62201-62220); m = mean + 9.99999997e-07f (0x358637bd, :62222);
//      rsq = v_rsq_f32 with the denormal scaling (m < 2^-126: m *= 2^24 (0x4b800000), rsq *= 2^12 (0x45800000), :62224-62235)
//   4. y = (1.0f + w[tid]) * (rsq * x[tid]) , bf16 RNE (:62239-62251) into LDS act[tid] (ds_store_b16_d16_hi, :62256)
//   5. tid < 32: rotate act[tid], act[tid+32] in LDS (NeoX layout, rope dim c, theta): p = pos ? pos[token] : token + d
//      (:62264-62275), inv = exp2(-1.4426950f * (ln(theta) * (float)(2*tid) / (float)c)) (:62276-62310, ocml logf expansion),
//      ang = ((float)p * inv) * 0.15915494f, v_sin/v_cos (:62312-62315); y0 = fma(co,x0,-(s*x1)), y1 = fma(co,x1,s*x0)
//      (:62318-62321); stored back as bf16 RNE (:62325-62330)
//   6. barrier ; out[tid] = act[tid] (threads < 64 reload from LDS :62337, others keep step 4's bf16) stored in place
//      (:62341); K blocks also store to pool_k[p0[token] + slot[token]*512 + (head&1)*256 + tid] (:62343-62370).
extern "C" __global__ void k_qk_prep(unsigned short* q, unsigned short* k, const unsigned short* v,
                                     const unsigned short* qw, const unsigned short* kw, int a, int b, int c,
                                     float theta, int d, const int* pos, FdRows r) {
  int bx = blockIdx.x, tid = threadIdx.x;
  if (bx >= 26 * a) {
    int idx = bx - 26 * a;
    const unsigned short* src = v + (long)idx * 512;
    unsigned short* dst = r.p1[idx] + (long)r.slot[idx] * 512;
    for (int i = tid; i < 512; i += blockDim.x) dst[i] = src[i];
    return;
  }
  __shared__ float red[256];
  __shared__ unsigned short act[256];
  bool isk = bx >= 24 * a;
  int head = isk ? bx - 24 * a : bx;
  int per = isk ? 2 : 24;
  unsigned short* x = (isk ? k : q) + (long)head * 256;
  const unsigned short* w = isk ? kw : qw;
  int token = head / per;
  float s = 0.0f;
  for (int i = tid; i < b; i += blockDim.x) {
    float xv = bf2f(x[i]);
    s = __builtin_fmaf(xv, xv, s);
  }
  red[tid] = s;
  __syncthreads();
  for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
    if (tid < st) red[tid] = red[tid + st] + red[tid];
    __syncthreads();
  }
  float mean = red[0] / (float)b;
  float m = mean + 9.99999997e-07f;
  bool small = m < 1.17549435e-38f;
  if (small) m = m * 16777216.0f;
  float rsq = __builtin_amdgcn_rsqf(m);
  if (small) rsq = rsq * 4096.0f;
  float xv = bf2f(x[tid]);
  float y = (1.0f + bf2f(w[tid])) * (rsq * xv);
  act[tid] = f2bf_rne(y);
  __syncthreads();
  if (tid < 32) {
    int p = pos ? pos[token] : token + d;
    float lnt = vlog(theta);
    float qq = (float)(2 * tid) / (float)c;
    float inv = __builtin_amdgcn_exp2f(-1.44269502162933349609375f * (lnt * qq));
    float ang = ((float)p * inv) * 0.15915494f;
    float sn = __builtin_amdgcn_sinf(ang), co = __builtin_amdgcn_cosf(ang);
    float x0 = bf2f(act[tid]), x1 = bf2f(act[tid + 32]);
    float t = sn * x1;
    float y0 = __builtin_fmaf(co, x0, -t);
    float y1 = __builtin_fmaf(co, x1, sn * x0);
    act[tid] = f2bf_rne(y0);
    act[tid + 32] = f2bf_rne(y1);
  }
  __syncthreads();
  unsigned short o = act[tid];
  x[tid] = o;
  if (isk) {
    unsigned short* kd = r.p0[token] + (long)r.slot[token] * 512 + (head & 1) * 256;
    kd[tid] = o;
  }
}
