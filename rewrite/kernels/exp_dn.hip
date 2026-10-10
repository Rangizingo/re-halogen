// Rewrites of the "exp_dn" family (routed-expert down projection, 10 slots summed with route weights), vendor code object 53000, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o exp_dn.co exp_dn.hip
// Listing: kernels-disasm/codeobj_53000.s (untracked; regenerate with llvm-objdump --mattr=+gfx1151). "L:n" below means that file, line n.
// `.long` words in the listing are 64-bit-operand VALU ops llvm-objdump could not print; they are read from the raw
// encoding (opcode field) and context, and the compare run is the check.
#include <hip/hip_runtime.h>

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float dot2(unsigned xp, unsigned wp, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, xp), __builtin_bit_cast(bf16x2, wp), acc, false);
}
// v_cvt_f32_f16 (L:2404653-2404660)
__device__ static inline float cvt_h(unsigned short h) {
  float d;
  unsigned s = h;
  asm("v_cvt_f32_f16 %0, %1" : "=v"(d) : "v"(s));
  return d;
}

// halogen::(anonymous namespace)::k_exp_dn_w<0>(unsigned short const* x, unsigned char const* W, int const* idx,
//   float const* rw, long a, long b, long c, unsigned short* out, unsigned short const* res, float const* rs, unsigned short* out2)
// kernarg x@0 W@8 idx@16 rw@24 a@32 b@40 c@48 out@56 res@64 rs@72 out2@80 (s_load_b512 s[4:19] L:2404396; s_load_b128 s[4:7], 0x40 L:2405523;
// s_load_b64 s[2:3], 0x50 L:2405511). Block of 256 threads, 8 rows per block, 32 lanes per row; grid (320, tokens).
//
// LDS: table T[256] at 0 (same construction as k_exp_gu_w: bf16(W_f32[t&15]) | bf16(W_f32[t>>4]) << 16, L:2404407-2404420);
// threads < 10 stage rw[tok*10 + i] at 0x400 + 4i and idx[tok*10 + i] at 0x430 + 4i (ds_store_2addr_b32 offset1:12, L:2404421-2404440,
// s_mul_i32 s21, tok, 10 at L:2404401).
//
// Row r = blockIdx.x*8 + (tid>>5), lane l = tid&31 (L:2404450). Each row has 10 slots x 20 groups of 32 elements (K=640 per slot,
// 320 weight bytes per slot-row). Lane l handles groups gi = l + 32*m, m = 0..6 (gi < 200; m=6 only for l < 8; L:2404657-2404693):
//   slot = gi / 20, h = gi % 20 (v_mad_i32_i24 by -20, L:2404499-2404504), e = idx[slot] (LDS 0x430),
//   weight bytes: W + a + (e*2560 + r)*320 + 16h (v_mad_u64_u32 0x140 with s[8:9] = W+a, L:2404497-2404505),
//   f16 scale:    W + b + (e*2560 + r)*c + 2h (v_mad_u64_u32 by s16, s[2:3] = W+b, L:2404510),
//   x:            x + (tok*10 + slot)*1280 + 64h, 4 b128 at +0,+16,+32,+48 (L:2404699-2404720, 0x500 = 1280).
// Group partial p = 16-step dot2 chain from 0 (x dword i of block q against T[byte i of weight dword q], as in k_exp_gu_w).
// acc = fma(f32(scale) * rw[slot], p, acc), acc starting at literal 0 (v_mul_f32 then v_fma_f32 ..., 0 / v_fmac_f32;
// L:2404807-2404809, 2404919-2404921, 2405465-2405467).
// Reduce over 32 lanes with xor 16,8,4,2,1 (ds_bpermute + v_add_f32, L:2405473-2405508). Lane 0:
//   s = reduced sum, o = bf16_rne(s) (L:2405516-2405520);
//   if out2 != null: o = bf16_rne(f32(o) + rs[tok] * f32(res[tok*2560 + r])) (v_fmac_f32 v2, s0, v3, L:2405535-2405544),
//   stored to out2; else o is stored to out (L:2405521-2405522 branch on s[2:3] == 0, store L:2405549). Index tok*2560 + r (L:2405514).
template <int P, int NR>
__device__ static inline void exp_dn_w_body(const unsigned short* x, const unsigned char* W, const int* idx, const float* rw, long a,
                                            long b, long c, unsigned short* out, const unsigned short* res, const float* rs,
                                            unsigned short* out2) {
  __shared__ unsigned T[256];
  __shared__ float rws[10];
  __shared__ int ids[10];
  int tid = threadIdx.x;
  int tok = blockIdx.y;
  {
    float lo = ((const float*)W)[tid & 15];
    float hi = ((const float*)W)[tid >> 4];
    T[tid] = (unsigned)f2bf_rne(lo) | ((unsigned)f2bf_rne(hi) << 16);
  }
  if (tid < 10) {
    rws[tid] = rw[tok * 10 + tid];
    ids[tid] = idx[tok * 10 + tid];
  }
  __syncthreads();
  int l = tid & 31;
  // NR rows per 32-lane group: NR = 1 gives r = blockIdx.x*8 + warp; NR = 2 (k_exp_dn_w2: grid x 160) gives rows 2*warp and
  // 2*warp + 1 of a 16-row block (v_and_or v72, tid>>4, 14, blockIdx<<4, L:38695; second store at offset:2, L:40392).
  for (int rr = 0; rr < NR; rr++) {
  int r = blockIdx.x * (8 * NR) + NR * (tid >> 5) + rr;
  float acc = 0.f;
  for (int m = 0; m < 7; m++) {
    int gi = l + 32 * m;
    if (gi >= 200) continue;
    int slot = gi / 20;
    int h = gi - 20 * slot;
    // P = 1 (DEC_PROBE=1): weight and scale addresses do not depend on the expert or the row; every row reads the same
    // 320 weight bytes and 40 scale bytes (global_load_b128 v, v2, s[8:9] with v2 = 16*h only; L:2403019-2403052).
    long row = (P == 1) ? 0 : (long)ids[slot] * 2560 + r;
    const unsigned char* wp = W + a + row * 320 + 16 * h;
    unsigned short sh = *(const unsigned short*)(W + b + row * c + 2 * h);
    const unsigned char* xp = (const unsigned char*)x + (long)(tok * 10 + slot) * 1280 + 64 * h;
    uint4 wq = *(const uint4*)wp;
    unsigned ws[4] = {wq.x, wq.y, wq.z, wq.w};
    float p = 0.f;
    if (P == 2) {
      // DEC_PROBE=2 variant: no x loads and no table; the partial is float(w0 ^ w1 ^ w2 ^ w3) (v_xor_b32, v_xor3_b32,
      // v_cvt_f32_u32; L:2404257-2404306, 2404253). Memory-bandwidth probe.
      p = (float)(ws[0] ^ ws[1] ^ ws[2] ^ ws[3]);
    } else {
#pragma unroll
      for (int q = 0; q < 4; q++) {
        uint4 xq = *(const uint4*)(xp + 16 * q);
        unsigned xd[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
        for (int i = 0; i < 4; i++) p = dot2(xd[i], T[(ws[q] >> (8 * i)) & 0xff], p);
      }
    }
    float k = cvt_h(sh) * rws[slot];
    acc = (m == 0) ? __builtin_fmaf(k, p, 0.0f) : __builtin_fmaf(k, p, acc);
  }
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) acc += __shfl_xor(acc, d, 32);
  if (l == 0) {
    unsigned short o = f2bf_rne(acc);
    long oi = (long)tok * 2560 + r;
    if (out2 != nullptr) {
      float rv = __uint_as_float((unsigned)res[oi] << 16);
      float t = __builtin_fmaf(rs[tok], rv, __uint_as_float((unsigned)o << 16));
      out2[oi] = f2bf_rne(t);
    } else {
      out[oi] = o;
    }
  }
  }
}
// P = 0 is the default decode instance. P = 1 has the same 112 dot2 and the same arithmetic as P = 0 in the listing
// (2402913-2403969: scalar-base loads, all seven groups issued up front); it is compared as a separate instance.
// P = 2 is the xor probe above (424 lines, no dot2).
#define EXP_DN(P)                                                                                                         \
  extern "C" __global__ void k_exp_dn_w_##P(const unsigned short* x, const unsigned char* W, const int* idx, const float* rw, \
                                            long a, long b, long c, unsigned short* out, const unsigned short* res,        \
                                            const float* rs, unsigned short* out2) {                                       \
    exp_dn_w_body<P, 1>(x, W, idx, rw, a, b, c, out, res, rs, out2);                                                       \
  }
EXP_DN(0)
EXP_DN(1)
EXP_DN(2)
// halogen::(anonymous namespace)::k_exp_dn_w2(...) same 11 arguments, two rows per lane group.
extern "C" __global__ void k_exp_dn_w2(const unsigned short* x, const unsigned char* W, const int* idx, const float* rw, long a,
                                       long b, long c, unsigned short* out, const unsigned short* res, const float* rs,
                                       unsigned short* out2) {
  exp_dn_w_body<0, 2>(x, W, idx, rw, a, b, c, out, res, rs, out2);
}

// halogen::(anonymous namespace)::k_exp_dn_aff(unsigned short const* x, unsigned char const* W, int const* idx, float const* rw,
//   AffDev, unsigned short* out, unsigned short const* res, float const* rs, unsigned short* out2)
// kernarg x@0 W@8 idx@16 rw@24 AffDev@32 (6 x u64) out@80 res@88 rs@96 out2@104 (s_load_b512 s[4:19], L:42929). Same 8-rows-per-block,
// 32-lane, slot x 20-group structure as k_exp_dn_w<0> (L:42973-43070), with the affine Q5 format:
//   nibbles  W + AffDev[0] + row*320 + 16h        (v_mad_u64_u32 0x140, s[8:9] = W + s12, L:43020, 43032)
//   bit plane W + AffDev[1] + row*80 + 4h          (v_mad_u64_u32 0x50, s[2:3] = W + s14, L:43025, 43043)
//   (d, dmin) f16 pair W + AffDev[2] + row*AffDev[3] + 4h (v_mad_u64_u32 v0, s18, s[6:7] = W + s16, L:43028, 43034)
// T table as k_exp_gu_aff (integer nibble values). Per group: Pq = dot2 chain, per x dword: dot2(x, T[byte]) then dot2(x, HB) on one
// accumulator (HB = bf16 pair of 16.0 per set plane bit, as in k_exp_gu_aff B=5; L:43306-43308, 43360-43467), Px = dot2 chain against (1,1)
// (L:43309-43314). acc = fma(f32(d)*rw[slot], Pq, acc); acc = fma(f32(dmin)*rw[slot], Px, acc) (v_mul_f32, v_fma_f32 ... 0 /
// v_fmac_f32; L:43520-43525, 43759-43764). Reduction and epilogue as k_exp_dn_w<0>.
struct AffDev { long f[6]; };
extern "C" __global__ void k_exp_dn_aff(const unsigned short* x, const unsigned char* W, const int* idx, const float* rw, AffDev ad,
                                        unsigned short* out, const unsigned short* res, const float* rs, unsigned short* out2) {
  __shared__ unsigned T[256];
  __shared__ float rws[10];
  __shared__ int ids[10];
  int tid = threadIdx.x;
  int tok = blockIdx.y;
  T[tid] = (unsigned)f2bf_rne((float)(tid & 15)) | ((unsigned)f2bf_rne((float)(tid >> 4)) << 16);
  if (tid < 10) {
    rws[tid] = rw[tok * 10 + tid];
    ids[tid] = idx[tok * 10 + tid];
  }
  __syncthreads();
  int r = blockIdx.x * 8 + (tid >> 5);
  int l = tid & 31;
  float acc = 0.f;
  for (int m = 0; m < 7; m++) {
    int gi = l + 32 * m;
    if (gi >= 200) continue;
    int slot = gi / 20;
    int h = gi - 20 * slot;
    long row = (long)ids[slot] * 2560 + r;
    uint4 wq = *(const uint4*)(W + ad.f[0] + row * 320 + 16 * h);
    unsigned pl = *(const unsigned*)(W + ad.f[1] + row * 80 + 4 * h);
    unsigned dd = *(const unsigned*)(W + ad.f[2] + row * ad.f[3] + 4 * h);
    const unsigned char* xp = (const unsigned char*)x + (long)(tok * 10 + slot) * 1280 + 64 * h;
    unsigned ws[4] = {wq.x, wq.y, wq.z, wq.w};
    float pq = 0.f, px = 0.f;
#pragma unroll
    for (int q = 0; q < 4; q++) {
      uint4 xq = *(const uint4*)(xp + 16 * q);
      unsigned xd[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int e2 = 8 * q + 2 * i;
        unsigned hb = (((pl >> e2) & 1u) ? 0x4180u : 0u) | (((pl >> (e2 + 1)) & 1u) ? 0x41800000u : 0u);
        pq = dot2(xd[i], T[(ws[q] >> (8 * i)) & 0xff], pq);
        pq = dot2(xd[i], hb, pq);
        px = dot2(xd[i], 0x3f803f80u, px);
      }
    }
    float w = rws[slot];
    float k1 = cvt_h((unsigned short)(dd & 0xffff)) * w;
    float k2 = cvt_h((unsigned short)(dd >> 16)) * w;
    acc = (m == 0) ? __builtin_fmaf(k1, pq, 0.0f) : __builtin_fmaf(k1, pq, acc);
    acc = __builtin_fmaf(k2, px, acc);
  }
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) acc += __shfl_xor(acc, d, 32);
  if (l == 0) {
    unsigned short o = f2bf_rne(acc);
    long oi = (long)tok * 2560 + r;
    if (out2 != nullptr) {
      float rv = __uint_as_float((unsigned)res[oi] << 16);
      float t = __builtin_fmaf(rs[tok], rv, __uint_as_float((unsigned)o << 16));
      out2[oi] = f2bf_rne(t);
    } else {
      out[oi] = o;
    }
  }
}

// halogen::(anonymous namespace)::k_exp_dn_w_q8(unsigned short const* x, unsigned char const* W, int const* idx, float const* rw,
//   unsigned short* out, unsigned short const* res, float const* rs, unsigned short* out2)
// kernarg x@0 W@8 idx@16 rw@24 out@32 res@40 rs@48 out2@56 (s_load_b512 s[4:19], L:40400). Q8S32 rows: 688 bytes = 640 int8
// followed by 20 f16 scales, one per 32 elements (v_mad_u64_u32 0x2b0, L:40492; global_load_d16_b16 ... offset:640, L:40598-40652).
// LDS: rw[tok*10+i] at 4i, idx[tok*10+i] at 48+4i (ds_store_2addr_b32 offset1:12, L:40424); no table. 256 threads, 8 rows per
// block, 32 lanes per row, grid (320, tokens). Items are 16-element chunks: gi = l + 32*m, m = 0..12, gi < 400
// (v_cmp_lt 39 / v_mul_u32_u24 0x667 = gi/40, L:40436, 40459): slot = gi/40, h = gi%40, e = idx[slot],
//   row = e*2560 + r, weights = W + row*688 + 16h, scale = f16 at W + row*688 + 640 + (h & ~1) (v_and -2, L:40490),
//   x = x + (tok*10 + slot)*1280 + 32h (two b128 at +0 and +16, L:40914-40915).
// Dequant: per int8 q: bf16_rne(float(q) * f32(scale)) (v_bfe_i32, v_cvt_f32_i32, v_mul_f32 scale*q, v_bfe_u32/v_add3_u32 0x7fff;
// L:40746-40762); pairs (q[2j], q[2j+1]) packed lo/hi (v_and_or_b32 0xffff0000, L:40765); 8 dot2 against x dwords 0..7 from acc 0
// (L:40783-40903); acc = fma(rw[slot], p, acc), first from literal 0 (v_fma_f32 v48, v49, v48, 0 / v_fmac_f32, L:40906, 41078).
// Reduction and epilogue as k_exp_dn_w<0> (L:41935-41986; out2 test s_cmp_eq_u64 s[18:19], 0 at L:42888).
extern "C" __global__ void k_exp_dn_w_q8(const unsigned short* x, const unsigned char* W, const int* idx, const float* rw,
                                         unsigned short* out, const unsigned short* res, const float* rs, unsigned short* out2) {
  __shared__ float rws[10];
  __shared__ int ids[10];
  int tid = threadIdx.x;
  int tok = blockIdx.y;
  if (tid < 10) {
    rws[tid] = rw[tok * 10 + tid];
    ids[tid] = idx[tok * 10 + tid];
  }
  __syncthreads();
  int r = blockIdx.x * 8 + (tid >> 5);
  int l = tid & 31;
  float acc = 0.f;
  for (int m = 0; m < 13; m++) {
    int gi = l + 32 * m;
    if (gi >= 400) continue;
    int slot = gi / 40;
    int h = gi - 40 * slot;
    long row = (long)ids[slot] * 2560 + r;
    const unsigned char* wr = W + row * 688;
    uint4 wq = *(const uint4*)(wr + 16 * h);
    unsigned short sh = *(const unsigned short*)(wr + 640 + (h & ~1));
    const unsigned char* xp = (const unsigned char*)x + (long)(tok * 10 + slot) * 1280 + 32 * h;
    uint4 x0 = *(const uint4*)xp;
    uint4 x1 = *(const uint4*)(xp + 16);
    unsigned xd[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    unsigned wd[4] = {wq.x, wq.y, wq.z, wq.w};
    float sc = cvt_h(sh);
    float p = 0.f;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      unsigned dw = wd[j >> 1];
      int q0 = (int)(signed char)((dw >> (16 * (j & 1))) & 0xff);
      int q1 = (int)(signed char)((dw >> (16 * (j & 1) + 8)) & 0xff);
      float f0 = sc * (float)q0;
      float f1 = sc * (float)q1;
      unsigned pk = (unsigned)f2bf_rne(f0) | ((unsigned)f2bf_rne(f1) << 16);
      p = dot2(xd[j], pk, p);
    }
    float w = rws[slot];
    acc = (m == 0) ? __builtin_fmaf(w, p, 0.0f) : __builtin_fmaf(w, p, acc);
  }
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) acc += __shfl_xor(acc, d, 32);
  if (l == 0) {
    unsigned short o = f2bf_rne(acc);
    long oi = (long)tok * 2560 + r;
    if (out2 != nullptr) {
      float rv = __uint_as_float((unsigned)res[oi] << 16);
      float t = __builtin_fmaf(rs[tok], rv, __uint_as_float((unsigned)o << 16));
      out2[oi] = f2bf_rne(t);
    } else {
      out[oi] = o;
    }
  }
}
