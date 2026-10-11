// Rewrites of the "exp_gu" family (routed-expert gate/up GEMV with silu*up epilogue), vendor code object 53000, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o exp_gu.co exp_gu.hip
// Listing: kernels-disasm/codeobj_53000.s (untracked; regenerate with llvm-objdump --mattr=+gfx1151). "L:n" below means that file, line n.
// `.long` words in the listing are 64-bit-operand VALU ops llvm-objdump could not print; they are read from the raw
// encoding (opcode field) and context, and the compare run is the check.
#include <hip/hip_runtime.h>

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
// v_dot2_f32_bf16 d, xpair, lutpair, acc (no clamp)
__device__ static inline float dot2(unsigned xp, unsigned wp, float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2, xp), __builtin_bit_cast(bf16x2, wp), acc, false);
}
// v_fma_mix_f32 d, scale_f16, partial_f32, acc  op_sel_hi:[1,0,0]; acc is the literal 0 for the first group (L:2401888, 2401891)
// and the running sum afterwards (L:2401892-2401926). Asm pins operand order and the +0 constant.
__device__ static inline float fma_mix0(unsigned short sc, float p) {
  float d;
  unsigned s = sc;
  asm("v_fma_mix_f32 %0, %1, %2, 0 op_sel_hi:[1,0,0]" : "=v"(d) : "v"(s), "v"(p));
  return d;
}
__device__ static inline float fma_mix(unsigned short sc, float p, float acc) {
  float d;
  unsigned s = sc;
  asm("v_fma_mix_f32 %0, %1, %2, %3 op_sel_hi:[1,0,0]" : "=v"(d) : "v"(s), "v"(p), "v"(acc));
  return d;
}

// halogen::(anonymous namespace)::k_exp_gu_w<R=16, K, E>(unsigned short const* x, unsigned char const* W, int const* idx,
//   long a, long b, long c, unsigned short* out)
// kernarg x@0 W@8 idx@16 a@24 b@32 c@40 out@48 (L:2400998, 2401044, 2401963). Block of 256 threads, 16 lanes per row, 16 rows per block.
//
// LDS table (1024 bytes): T[t] for t = tid (0..255) = bf16(W_f32[t & 15]) | bf16(W_f32[t >> 4]) << 16, where W_f32 is the 16-entry
// f32 LUT at W+0 (global_load_b32 at W + 4*(tid&15) and W + 4*(tid>>4), v_bfe/v_add3 0x7fff RNE, v_and_or 0xffff0000,
// ds_store_b32 tid*4; L:2401009-2401040), then s_barrier (L:2401046).
//
// Row mapping: r = blockIdx.x*16 + (tid>>4) (v_or3 of tid>>5 &14, blockIdx<<4, bit 4 of tid; L:2400999-2401011).
// p = r / 640 (mul_hi 0x66666667 >> 8 + sign, L:2401015-2401020), local = r - 640*p (L:2401049-2401051), token = r / 6400
// (mul_hi 0x51eb851f >> 11, L:2401042, 2401052-2401064). e = idx[p] (L:2401048). R0 = e*1280 + local (v_mad_i64_i32 0x500, L:2401057).
// Gate row = W + a + R0*1280 (v_mad_u64_u32 0x500, s[2:3] = W+a, L:2401058, 2401012); up row = gate row + 0xc8000 = 640 rows
// (L:2401069). Lane l = tid&15 reads 16 bytes at row + 256*j + 16*l for j = 0..4 (global_load_b128 offsets 0,256,...,1024,
// L:2401073-2401101). x pointer = x + token*5120 + 64*l (L:2401064-2401085), x block for pass j, quarter q at +1024*j + 16*q.
// Scales (f16): gate ptr = W + b + R0*c + 2*l, up ptr = gate ptr + 640*c; pass j at +32*j (global_load_d16_b16 offsets 0..128
// step 32, L:2401662-2401704).
//
// Per pass j and matrix: acc = 0; for q in 0..3, dword w of the 4 weight dwords in order (byte i of the dword):
//   for i in 0..3: acc = dot2(x_dword[4q+i], T[byte i of w_q], acc)   (L:2401142-2401167: bytes via (w & 0xff)<<2, (w>>6)&0x3fc,
//   (w>>14)&0x3fc, (w>>22)&0x3fc, ds_load_b32; one 16-dot2 chain per pass)
// then total = fma_mix(scale_j, acc_j, total), total starting at literal 0 (L:2401888-2401926).
// Reduction over the 16 lanes: v_xor 8,4,2,1 ds_bpermute + v_add_f32 (L:2401927-2401962). Lane 0 of each row finishes:
// g = reduced gate sum, u = reduced up sum; e = exp2(g * -1.4426950f) (0xbfb8aa3b), d = 1 + e, y = g / d (IEEE div,
// L:2401966-2401985), out[p*640 + local] = bf16_rne(u * y) (L:2401986-2402001; byte offset p*1280 + 2*local).
template <int R, int K, int E>
__device__ static inline void exp_gu_w(const unsigned short* x, const unsigned char* W, const int* idx, long a, long b, long c,
                                         unsigned short* out) {
  __shared__ unsigned T[256];
  int tid = threadIdx.x;
  {
    float lo = ((const float*)W)[tid & 15];
    float hi = ((const float*)W)[tid >> 4];
    T[tid] = (unsigned)f2bf_rne(lo) | ((unsigned)f2bf_rne(hi) << 16);
  }
  __syncthreads();
  int r = blockIdx.x * (256 / R) + (tid / R);
  int l = tid % R;
  int p = r / 640;
  int local = r - 640 * p;
  int token = r / 6400;
  long e = idx[p];
  long R0 = e * 1280 + local;
  const unsigned char* grow = W + a + R0 * 1280 + 16 * l;
  const unsigned char* urow = grow + 0xc8000;
  const unsigned char* xp = (const unsigned char*)x + (long)token * 5120 + 64 * l;
  const unsigned char* gsc = W + b + R0 * c + 2 * l;
  const unsigned char* usc = gsc + 640 * c;
  float gt = 0.f, ut = 0.f;
  // Group g = j*R + l (32 elements = 16 weight bytes = 64 x bytes = one f16 scale); 80 groups per row.
  // R=16: 5 passes (L:2401073-2401101); R=32: passes 0 and 1 for all lanes, pass 2 for lanes < 16 only
  // (v_cmpx_gt_u32 16, L:2394095); R=8: 10 passes, prefetch 128 and 256 bytes ahead (L:2396657-2396660, 2396941-2396942).
  for (int j = 0; j * R + l < 80; j++) {
    uint4 gw = *(const uint4*)(grow + 16 * R * j);
    uint4 uw = *(const uint4*)(urow + 16 * R * j);
    uint4 xq[4];
#pragma unroll
    for (int q = 0; q < 4; q++) xq[q] = *(const uint4*)(xp + 64 * R * j + 16 * q);
    unsigned gws[4] = {gw.x, gw.y, gw.z, gw.w};
    unsigned uws[4] = {uw.x, uw.y, uw.z, uw.w};
    float ga = 0.f, ua = 0.f;
#pragma unroll
    for (int q = 0; q < 4; q++) {
      unsigned xd[4] = {xq[q].x, xq[q].y, xq[q].z, xq[q].w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        ga = dot2(xd[i], T[(gws[q] >> (8 * i)) & 0xff], ga);
        ua = dot2(xd[i], T[(uws[q] >> (8 * i)) & 0xff], ua);
      }
    }
    unsigned short gs = *(const unsigned short*)(gsc + 2 * R * j);
    unsigned short us = *(const unsigned short*)(usc + 2 * R * j);
    gt = (j == 0) ? fma_mix0(gs, ga) : fma_mix(gs, ga, gt);
    ut = (j == 0) ? fma_mix0(us, ua) : fma_mix(us, ua, ut);
  }
  // xor butterfly over the R lanes of a row (R=16: 8,4,2,1 L:2401927-2401958; R=32 adds 16, L:2394126; R=8 starts at 4)
#pragma unroll
  for (int d = R / 2; d >= 1; d >>= 1) {
    gt += __shfl_xor(gt, d, R);
    ut += __shfl_xor(ut, d, R);
  }
  if (l == 0) {
    float ex = __builtin_amdgcn_exp2f(gt * -1.44269502162933349609375f);
    float y = gt / (1.0f + ex);
    out[(long)p * 640 + local] = f2bf_rne(ut * y);
  }
}
// K (prefetch depth) and E only change instruction scheduling in the listings (instance sizes differ, the dot2 chains do not).
#define EXP_GU(R, K, E)                                                                                                         \
  extern "C" __global__ void k_exp_gu_w_##R##_##K##_##E(const unsigned short* x, const unsigned char* W, const int* idx, long a, \
                                                        long b, long c, unsigned short* out) {                                \
    exp_gu_w<R, K, E>(x, W, idx, a, b, c, out);                                                                               \
  }
EXP_GU(8, 1, 0)
EXP_GU(8, 1, 1)
EXP_GU(8, 2, 0)
EXP_GU(8, 2, 1)
EXP_GU(8, 3, 0)
EXP_GU(8, 3, 1)
EXP_GU(16, 1, 0)
EXP_GU(16, 1, 1)
EXP_GU(16, 2, 0)
EXP_GU(16, 2, 1)
EXP_GU(16, 3, 0)
EXP_GU(16, 3, 1)
EXP_GU(32, 1, 0)
EXP_GU(32, 1, 1)
EXP_GU(32, 2, 0)
EXP_GU(32, 2, 1)
EXP_GU(32, 3, 0)
EXP_GU(32, 3, 1)

// ---- k_exp_gu_aff<R, K, B> (affine Q4A/Q5A gate/up, silu(gate)*up) ----
// halogen::(anonymous namespace)::k_exp_gu_aff<R,K,B>(unsigned short const* x, unsigned char const* W, int const* idx, AffDev, unsigned short* out)
// kernarg x@0 W@8 idx@16 AffDev@24 (6 x u64 = 48 bytes) out@72 (s_load_b256 s[4:11] at 0, s_load_b256 s[12:19] at 0x28,
// s_load_b64 s[2:3] at 0x48; L:2391373, 2391379, 2391427). AffDev[0] = weight data offset (s10:11), AffDev[1] is never read,
// AffDev[2], [3] = offset and row stride of the per-group byte pairs, AffDev[4], [5] = offset and row stride of the per-superblock
// f16 pairs (v_mad_u64_u32 v, s14, row, s[12:13] / s18, row, s[16:17]; L:2391464-2391491).
// Row mapping, weight addresses, x addresses, group g = j*R + l and the lane butterfly are identical to k_exp_gu_w.
// LDS table (1024 bytes): T[t] = bf16(float(t & 15)) | bf16(float(t >> 4)) << 16, i.e. the integer nibble values
// (v_cvt_f32_ubyte0 of tid&15 and tid>>4, L:2391375, 2391378, 2391404).
// Per group g (32 elements) and matrix: Pq = 16-step dot2 chain over T[byte] from 0, Px = 16-step dot2 chain of x against the
// bf16 pair (1.0, 1.0) = 0x3f803f80 (one chain shared by gate and up, L:2391638-2391720);
// (d, dmin) = u32 at W + AffDev[4] + row*AffDev[5] + 4*(g>>3) (one f16 pair per 256 elements);
// (sc, mn) = two bytes at W + AffDev[2] + row*AffDev[3] + 2*g (global_load_u16 ... offset:-1, L:2391526-2391527);
// kd = fma_mix(d, float(sc), -0), km = fma_mix(dmin, -float(mn), -0) (L:2391734-2391743);
// acc = fma(kd, Pq, acc); acc = fma(km, Px, acc) (v_fmac_f32 v23, v12, v10 ; v_fmac_f32 v23, v16, v9; L:2391745-2391751).
struct AffDev { long f[6]; };

__device__ static inline float fm_d(unsigned dd, float f) {  // fma_mix(lo f16 of dd, f, -0)
  float r;
  asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]" : "=v"(r) : "v"(dd), "v"(f));
  return r;
}
__device__ static inline float fm_m(unsigned dd, float f) {  // fma_mix(hi f16 of dd, -f, -0)
  float r;
  asm("v_fma_mix_f32 %0, %1, -%2, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]" : "=v"(r) : "v"(dd), "v"(f));
  return r;
}

template <int R, int K, int B>
__device__ static inline void exp_gu_aff_body(const unsigned short* x, const unsigned char* W, const int* idx, AffDev ad,
                                              unsigned short* out) {
  __shared__ unsigned T[256];
  int tid = threadIdx.x;
  T[tid] = (unsigned)f2bf_rne((float)(tid & 15)) | ((unsigned)f2bf_rne((float)(tid >> 4)) << 16);
  __syncthreads();
  int r = blockIdx.x * (256 / R) + (tid / R);
  int l = tid % R;
  int p = r / 640;
  int local = r - 640 * p;
  int token = r / 6400;
  long e = idx[p];
  long R0 = e * 1280 + local;
  const unsigned char* grow = W + ad.f[0] + R0 * 1280;
  const unsigned char* urow = grow + 0xc8000;
  const unsigned char* xp = (const unsigned char*)x + (long)token * 5120;
  const unsigned char* gb = W + ad.f[2] + R0 * ad.f[3];
  const unsigned char* ub = gb + 640 * ad.f[3];
  const unsigned char* gd = W + ad.f[4] + R0 * ad.f[5];
  const unsigned char* ud = gd + 640 * ad.f[5];
  float gt = 0.f, ut = 0.f;
  for (int j = 0; j * R + l < 80; j++) {
    int g = j * R + l;
    uint4 gw = *(const uint4*)(grow + 16 * g);
    uint4 uw = *(const uint4*)(urow + 16 * g);
    unsigned gws[4] = {gw.x, gw.y, gw.z, gw.w};
    unsigned uws[4] = {uw.x, uw.y, uw.z, uw.w};
    float gq = 0.f, uq = 0.f, px = 0.f;
    // B = 5: one more dword per matrix and group from the bit plane at W + AffDev[1] + row*320 + 4g (global_load_b32 v50 / v36,
    // L:2390747-2390748; s_add_u32 s8, s6, s12 and v_mad_u64_u32 0x140, L:2390714, 2390722). Bit e of the plane dword is the
    // fifth bit of element e: the pair for x dword i is bf16 (16.0 if bit 2i else 0) | (16.0 if bit 2i+1 else 0) << 16
    // (v_bfe_i32 bit,1 ; v_and 0x4180 ; v_and_or 0x41800000, L:2390847-2390915). It is a second dot2 on the same accumulator after
    // the nibble dot2 of that x dword (v_dot2 v70, v17, v70, 0 ; v_dot2 v70, v17, v83, v70; L:2391002-2391006).
    unsigned gpl = 0, upl = 0;
    if (B == 5) {
      gpl = *(const unsigned*)(W + ad.f[1] + R0 * 320 + 4 * g);
      upl = *(const unsigned*)(W + ad.f[1] + (R0 + 640) * 320 + 4 * g);
    }
#pragma unroll
    for (int q = 0; q < 4; q++) {
      uint4 xq = *(const uint4*)(xp + 64 * g + 16 * q);
      unsigned xd[4] = {xq.x, xq.y, xq.z, xq.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        gq = dot2(xd[i], T[(gws[q] >> (8 * i)) & 0xff], gq);
        if (B == 5) {
          int e2 = 8 * q + 2 * i;
          unsigned gh = (((gpl >> e2) & 1u) ? 0x4180u : 0u) | (((gpl >> (e2 + 1)) & 1u) ? 0x41800000u : 0u);
          gq = dot2(xd[i], gh, gq);
        }
        uq = dot2(xd[i], T[(uws[q] >> (8 * i)) & 0xff], uq);
        if (B == 5) {
          int e2 = 8 * q + 2 * i;
          unsigned uh = (((upl >> e2) & 1u) ? 0x4180u : 0u) | (((upl >> (e2 + 1)) & 1u) ? 0x41800000u : 0u);
          uq = dot2(xd[i], uh, uq);
        }
        px = dot2(xd[i], 0x3f803f80u, px);
      }
    }
    unsigned gdd = *(const unsigned*)(gd + 4 * (g >> 3));
    unsigned udd = *(const unsigned*)(ud + 4 * (g >> 3));
    unsigned short gbb = *(const unsigned short*)(gb + 2 * g);
    unsigned short ubb = *(const unsigned short*)(ub + 2 * g);
    float gkd = fm_d(gdd, (float)(gbb & 0xff));
    float gkm = fm_m(gdd, (float)(gbb >> 8));
    float ukd = fm_d(udd, (float)(ubb & 0xff));
    float ukm = fm_m(udd, (float)(ubb >> 8));
    gt = __builtin_fmaf(gkd, gq, gt);
    ut = __builtin_fmaf(ukd, uq, ut);
    gt = __builtin_fmaf(gkm, px, gt);
    ut = __builtin_fmaf(ukm, px, ut);
  }
#pragma unroll
  for (int d = R / 2; d >= 1; d >>= 1) {
    gt += __shfl_xor(gt, d, R);
    ut += __shfl_xor(ut, d, R);
  }
  if (l == 0) {
    float ex = __builtin_amdgcn_exp2f(gt * -1.44269502162933349609375f);
    float y = gt / (1.0f + ex);
    out[(long)p * 640 + local] = f2bf_rne(ut * y);
  }
}
#define EXP_GU_AFF(R, K, B)                                                                                                   \
  extern "C" __global__ void k_exp_gu_aff_##R##_##K##_##B(const unsigned short* x, const unsigned char* W, const int* idx, \
                                                          AffDev ad, unsigned short* out) {                               \
    exp_gu_aff_body<R, K, B>(x, W, idx, ad, out);                                                                          \
  }
EXP_GU_AFF(8, 1, 4)
EXP_GU_AFF(8, 2, 4)
EXP_GU_AFF(8, 3, 4)
EXP_GU_AFF(16, 1, 4)
EXP_GU_AFF(16, 2, 4)
EXP_GU_AFF(16, 3, 4)
EXP_GU_AFF(32, 1, 4)
EXP_GU_AFF(32, 2, 4)
EXP_GU_AFF(32, 3, 4)
EXP_GU_AFF(8, 1, 5)
EXP_GU_AFF(8, 2, 5)
EXP_GU_AFF(8, 3, 5)
EXP_GU_AFF(16, 1, 5)
EXP_GU_AFF(16, 2, 5)
EXP_GU_AFF(16, 3, 5)
EXP_GU_AFF(32, 1, 5)
EXP_GU_AFF(32, 2, 5)
EXP_GU_AFF(32, 3, 5)
