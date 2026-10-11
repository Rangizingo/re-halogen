// Rewrites of the "ht_gemv" family from the vendor code object codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ht_gemv.co ht_gemv.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). Cites are code-object addresses (@A84E00 = 0xa84e00) read from a
// ROCm llvm-objdump 23 re-decode (the repo listing prints some 8-byte VOP3 words as `.long` plus a bogus v_cndmask).
// HT weight format: analysis/closure/ht-decode.md (16x16 tiles, 128-row blocks, hash + byte sum decode).
#pragma clang fp contract(off)
#include <hip/hip_runtime.h>

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
// 1/sqrt(128) as the vendor literal 0x3db504f3 (v_mul_f32 0x3db504f3, @A8668C). Not the nearest f32 of 0.0883883506 (that is ...f4).
#define INV_SQRT128 __uint_as_float(0x3db504f3u)
struct HtItem { int expert, tile, begin, end; };  // htitem-r3.md: {expert, tile, begin, end}

// f32 -> bf16 RNE without NaN special case: v_bfe_u32 ; v_add3_u32 0x7fff ; store hi half (@A86694-A866F0).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// HT weight code: p = x * 0x83dcd12d (v_mul_lo_u32 @A8517C), S = v_sad_u8(p, 0, 0x6400), i.e. the f16 bit
// pattern 0x6400 + byte sum = 1024 + S. Weight = fma_f16(1024 + S, 0x1eee, 0xc931) (v_pk_fma_f16 ... s3=0x1eee, 0xc931).
__device__ static inline _Float16 ht_w(unsigned x16) {
  unsigned p = x16 * 0x83dcd12du;
  unsigned s = (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24) + 0x6400u;
  _Float16 hs = __builtin_bit_cast(_Float16, (unsigned short)s);
  _Float16 a = __builtin_bit_cast(_Float16, (unsigned short)0x1eee);
  _Float16 b = __builtin_bit_cast(_Float16, (unsigned short)0xc931);
  return __builtin_fmaf16(hs, a, b);
}

// Decode of one lane of a 16x16 tile (nw = 8*BITS words at tp). ht-decode.md "Unified decoder": the tile words are one little-endian
// bit string U read in reverse word order (last word = lowest bits); weight jw = 8t + c of lane t (row r = t%16, columns 8h + c)
// is the 16-bit window of U starting at bit BITS*(255 - jw) mod (256*BITS), circular. For 4 bit this is the vendor's
// pair {word[t-1], word[t]} >> 4*(7-c) (@A85154 v_lshrrev_b64 by 16 plus bfe 12/8/4/0, @A8512C-A85144); the 3-bit kernel uses
// 64-bit shifts by per-lane amounts (@A83680-A83694) and bfe 9/6/3/0. Weight column 8h + c.
template <int BITS>
__device__ static inline unsigned ht_win(const unsigned* tp, int pos) {
  constexpr int NW = 8 * BITS, NB = 32 * NW;
  unsigned v = 0;
  for (int i = 0; i < 16; i++) {
    int b = (pos + i) % NB;
    v |= ((tp[NW - 1 - (b >> 5)] >> (b & 31)) & 1u) << i;
  }
  return v;
}
template <int BITS>
__device__ static inline void ht_dec(const unsigned* tp, int t, _Float16 (&w)[8]) {
  for (int c = 0; c < 8; c++) w[c] = ht_w(ht_win<BITS>(tp, BITS * (255 - (8 * t + c))));
}

__device__ static inline float dot8(const _Float16 (&w)[8], const _Float16* xv, float acc) {
  for (int p = 0; p < 4; p++) {
    half2_t a = {w[2 * p], w[2 * p + 1]};
    half2_t b = {xv[2 * p], xv[2 * p + 1]};
    acc = __builtin_amdgcn_fdot2(a, b, acc, false);  // v_dot2acc_f32_f16, clamp off
  }
  return acc;
}

// 128-point Walsh-Hadamard: row = 16*j + r. j stages in registers (bit0, bit1, bit2: lower index first),
// then r stages by row_xmask 1,2,4,8: lane with the bit clear keeps x + p, lane with the bit set gets p - x (v_cndmask -x ; v_add).
#ifndef JO1
#define JO1 124
#endif
#ifndef JO2
#define JO2 124
#endif
__device__ static inline void wht_jstage(float (&s)[8], int bit) {
  float t[8];
  for (int i = 0; i < 8; i++) t[i] = (i & bit) ? s[i ^ bit] - s[i] : s[i] + s[i | bit];
  for (int i = 0; i < 8; i++) s[i] = t[i];
}
__device__ static inline void wht_rstages(float (&s)[8], int r) {
  for (int bit = 1; bit <= 8; bit <<= 1)
    for (int j = 0; j < 8; j++) {
      float p = __shfl_xor(s[j], bit, 16);
      s[j] = (r & bit) ? (p - s[j]) : (s[j] + p);
    }
}
// first stage already applied by the caller (bit 1 of j); remaining j stages in the order of JO2's last two digits
__device__ static inline void wht128_rest(const float (&t)[8], float (&s)[8], int r) {
  for (int i = 0; i < 8; i++) s[i] = t[i];
  wht_jstage(s, (JO2 / 10) % 10);
  wht_jstage(s, JO2 % 10);
  wht_rstages(s, r);
}
__device__ static inline void wht128(float (&s)[8], int r) {
  wht_jstage(s, JO1 / 100);
  wht_jstage(s, (JO1 / 10) % 10);
  wht_jstage(s, JO1 % 10);
  wht_rstages(s, r);
}

// k_ht_gemv_dn<BITS,8,1>(uint32 const* W, half const* x, HtItem const* items, half const* sv, ushort* out)
// kernarg W@0, x@8, items@16, sv@24, out@32. One workgroup (256 threads = 8 waves) per item (@A84E10: item = wg id * 16).
// Item: expert e, 128-row tile t of the 2560 x 640 down matrix, token range [begin, end). x row = min(begin, end-1) (@A84E24-A84E2C),
// 640 halves (0x500 B). W offset e*0xc8000 + t*0xa000 bytes (@A84E30-A84E44) = 819200 B per expert, 40960 B per 128-row tile.
// Wave w owns column tiles 5w..5w+4; lane = 16h + r holds weight row r of row tile j, columns 8h..8h+7.
// Acc j (8 row tiles) runs sequentially over the 5 column tiles with 4 dot2 each (pairs (c0,c1),(c2,c3),(c4,c5),(c6,c7)).
// Wave partials: own + permlanex16 (h1) -> LDS [w][j][r], wave 0 sums waves 0..7 in order starting from 0+,
// WHT-128 over rows 16j+r, times 1/sqrt(128) = 0x3db504f3, times sv (v_fma_mix ... neg(0)), bf16 RNE.
// Output: out[begin*2560 + t*128 + 16j + r], only when end > begin (v_cmp_gt_u32 exec, min(cnt,1), wave id).
template <int BITS>
__device__ static inline void ht_dn11(const unsigned* W, const _Float16* x, const HtItem* items, const _Float16* sv, unsigned short* out) {
  __shared__ float P[8][8][16];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, r = lane & 15, hh = lane >> 4;
  const HtItem it = items[blockIdx.x];
  const int xr = it.begin < it.end - 1 ? it.begin : it.end - 1;
  const _Float16* xrow = x + (long)xr * 640;
  constexpr int NW = 8 * BITS;  // words per 16x16 tile
  const unsigned* tb = W + ((long)it.expert * (2560 * 640 / 8 * BITS) + (long)it.tile * (128 * 640 / 8 * BITS)) / 4;
  float acc[8];
  for (int j = 0; j < 8; j++) acc[j] = 0.f;
  for (int i = 0; i < 5; i++) {
    const int ct = 5 * w + i;
    _Float16 xv[8];
    for (int c = 0; c < 8; c++) xv[c] = xrow[16 * ct + 8 * hh + c];
    for (int j = 0; j < 8; j++) {
      const unsigned* tp = tb + ct * (8 * NW) + j * NW;
      _Float16 wt[8];
      ht_dec<BITS>(tp, lane, wt);
      acc[j] = dot8(wt, xv, acc[j]);
    }
  }
  for (int j = 0; j < 8; j++) {
    float o = __shfl_xor(acc[j], 16, 32);
    if (hh == 0) P[w][j][r] = acc[j] + o;
  }
  __syncthreads();
  const int cnt = it.end - it.begin;
  if (w != 0 || cnt < 1) return;
  float s[8];
  for (int j = 0; j < 8; j++) {
    float a = 0.f + P[0][j][r];
    for (int q = 1; q < 8; q++) a = a + P[q][j][r];
    s[j] = a;
  }
  wht128(s, r);
  if (hh != 0) return;
  const _Float16* svp = sv + (long)it.expert * 2560 + it.tile * 128;
  unsigned short* op = out + (long)it.begin * 2560 + it.tile * 128;
  for (int j = 0; j < 8; j++) {
    float y = s[j] * INV_SQRT128;  // 0x3db504f3
    y = (float)svp[16 * j + r] * y;
    op[16 * j + r] = f2bf_rne(y);
  }
}

extern "C" __global__ void k_ht_gemv_dn_4_8_1(const unsigned* W, const _Float16* x, const HtItem* items, const _Float16* sv, unsigned short* out) {
  ht_dn11<4>(W, x, items, sv, out);
}

extern "C" __global__ void k_ht_gemv_dn_3_8_1(const unsigned* W, const _Float16* x, const HtItem* items, const _Float16* sv, unsigned short* out) {
  ht_dn11<3>(W, x, items, sv, out);
}

// ---------------------------------------------------------------------------------------------------------------------------
// k_ht_gemv_gu<BITS,8,1>(W, x, items, tok, sv, su, out): fused gate/up GEMV + SiLU + the next layer's input rotation.
// kernarg W@0, x@8, items@16, tok@24 (int), sv@32 (half, 1280 per expert), su@40 (half, 640 per expert), out@48 (half, 640 per token row).
// One workgroup (8 waves) per item {expert, tile b, begin, end}. x row = tok[min(begin,end-1)] (@A7FB44-A7FB6C), 2560 halves (0x1400 B).
// W expert stride 1280 rows * 2560 * BITS/8 (0x190000 at 4 bit, 0x12c000 at 3 bit), tile b = 256 rows = 128 gate rows b*128.. and 128 up
// rows 640 + b*128.. (stride 0x50000 / 0x3c000, @A7FC24-A7FCDC). Per column tile ct: 16 row tiles, gate first (k = g*8 + i)
// (global_load_b32 x16 at 128 B steps, @A7FD34-A7FDAC). Wave w owns ct 20w..20w+19.
// Wave reduction as in dn (own + permlanex16, LDS, wave 0 adds waves 0..7 from 0+). Then per half (gate / up): WHT-128 over rows
// 16j + r, * 0x3db504f3, * sv (gate sv[e*1280 + b*128 + 16j + r], up sv[... + 640]) giving G and U (f32).
// Z = U * silu(G), silu(G) = G / (1 + exp2(G * -1.442695)) with IEEE division (@A827D4-A827E4, v_div_scale/fmas/fixup).
// Z * su (su[e*640 + b*128 + 16j + r]) enters a second WHT-128; its first stage is fused: t[2m] = fma(Z[2m], su[2m], rn(Z[2m+1]*su[2m+1])),
// t[2m+1] = fma(Z[2m], su[2m], -rn(...)) (v_fma_mix_f32 ... op_sel_hi:[0,1,0], @A82E84-A82EE8). Then v_fma_mixlo_f16 with 0x3db504f3
// (f16 of the exact f32*f32 product, @A831FC-A83230). out[begin*640 + b*128 + 16j + r], only when end > begin.
template <int BITS>
__device__ static inline void ht_gu11(const unsigned* W, const _Float16* x, const HtItem* items, const int* tok, const _Float16* sv,
                                     const _Float16* su, _Float16* out) {
  __shared__ float P[8][16][16];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, r = lane & 15, hh = lane >> 4;
  const HtItem it = items[blockIdx.x];
  const int xi = it.begin < it.end - 1 ? it.begin : it.end - 1;
  const _Float16* xrow = x + (long)tok[xi] * 2560;
  constexpr int NW = 8 * BITS;
  const unsigned* tb = W + ((long)it.expert * (1280L * 2560 / 8 * BITS) + (long)it.tile * (256L * 2560 / 8 * BITS)) / 4;
  float acc[16];
  for (int k = 0; k < 16; k++) acc[k] = 0.f;
  for (int i = 0; i < 20; i++) {
    const int ct = 20 * w + i;
    _Float16 xv[8];
    for (int c = 0; c < 8; c++) xv[c] = xrow[16 * ct + 8 * hh + c];
    for (int k = 0; k < 16; k++) {
      const unsigned* tp = tb + ct * (16 * NW) + k * NW;
      _Float16 wt[8];
      ht_dec<BITS>(tp, lane, wt);
      acc[k] = dot8(wt, xv, acc[k]);
    }
  }
  for (int k = 0; k < 16; k++) {
    float o = __shfl_xor(acc[k], 16, 32);
    if (hh == 0) P[w][k][r] = acc[k] + o;
  }
  __syncthreads();
  const int cnt = it.end - it.begin;
  if (w != 0 || cnt < 1) return;
  float g[8], u[8];
  for (int k = 0; k < 16; k++) {
    float a = 0.f + P[0][k][r];
    for (int q = 1; q < 8; q++) a = a + P[q][k][r];
    if (k < 8) g[k] = a; else u[k - 8] = a;
  }
  wht128(g, r);
  wht128(u, r);
  const _Float16* svg = sv + (long)it.expert * 1280 + it.tile * 128;
  const _Float16* sus = su + (long)it.expert * 640 + it.tile * 128;
  float z[8], zs[8];
  for (int j = 0; j < 8; j++) {
    float G = (float)svg[16 * j + r] * (g[j] * INV_SQRT128);
    float U = (float)svg[640 + 16 * j + r] * (u[j] * INV_SQRT128);
    float e = __builtin_amdgcn_exp2f(G * -1.44269502162933349609375f);
    float q = G / (1.0f + e);
    z[j] = U * q;
    zs[j] = (float)sus[16 * j + r];
  }
  float t[8], o[8];
  for (int m = 0; m < 8; m += 2) {
    float rb = z[m + 1] * zs[m + 1];
    t[m] = __builtin_fmaf(z[m], zs[m], rb);
    t[m + 1] = __builtin_fmaf(z[m], zs[m], -rb);
  }
  wht128_rest(t, o, r);
  if (hh != 0) return;
  _Float16* op = out + (long)it.begin * 640 + it.tile * 128;
  for (int j = 0; j < 8; j++) {
    unsigned h;
    asm("v_fma_mixlo_f16 %0, %1, %2, 0" : "=v"(h) : "v"(o[j]), "v"(INV_SQRT128));
    op[16 * j + r] = __builtin_bit_cast(_Float16, (unsigned short)h);
  }
}

extern "C" __global__ void k_ht_gemv_gu_4_8_1(const unsigned* W, const _Float16* x, const HtItem* items, const int* tok, const _Float16* sv,
                                              const _Float16* su, _Float16* out) {
  ht_gu11<4>(W, x, items, tok, sv, su, out);
}
extern "C" __global__ void k_ht_gemv_gu_3_8_1(const unsigned* W, const _Float16* x, const HtItem* items, const int* tok, const _Float16* sv,
                                              const _Float16* su, _Float16* out) {
  ht_gu11<3>(W, x, items, tok, sv, su, out);
}
