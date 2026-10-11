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
// dn core: one 128-row x 640 tile (tb) against one x row; on return s[j] holds WHT row 16j + r (post sum, pre scale) for wave 0, lanes < 16.
// Returns true for the threads that own an output (wave 0, lane < 16, wr set).
template <int BITS>
__device__ static inline bool dn_core(const unsigned* tb, const _Float16* xrow, float (&s)[8], bool wr) {
  __shared__ float P[8][8][16];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, r = lane & 15, hh = lane >> 4;
  constexpr int NW = 8 * BITS;  // words per 16x16 tile
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
  if (w != 0 || !wr) return false;
  for (int j = 0; j < 8; j++) {
    float a = 0.f + P[0][j][r];
    for (int q = 1; q < 8; q++) a = a + P[q][j][r];
    s[j] = a;
  }
  wht128(s, r);
  return hh == 0;
}
__device__ static inline void dn_store(const float (&s)[8], const _Float16* svp, unsigned short* op) {
  const int r = threadIdx.x & 15;
  for (int j = 0; j < 8; j++) {
    float y = s[j] * INV_SQRT128;  // 0x3db504f3
    y = (float)svp[16 * j + r] * y;
    op[16 * j + r] = f2bf_rne(y);
  }
}
template <int BITS>
__device__ static inline void ht_dn11(const unsigned* W, const _Float16* x, const HtItem* items, const _Float16* sv, unsigned short* out) {
  const HtItem it = items[blockIdx.x];
  const int xr = it.begin < it.end - 1 ? it.begin : it.end - 1;
  const unsigned* tb = W + ((long)it.expert * (2560 * 640 / 8 * BITS) + (long)it.tile * (128 * 640 / 8 * BITS)) / 4;
  float s[8];
  if (dn_core<BITS>(tb, x + (long)xr * 640, s, it.end - it.begin >= 1))
    dn_store(s, sv + (long)it.expert * 2560 + it.tile * 128, out + (long)it.begin * 2560 + it.tile * 128);
}

// k_ht_gemv_dn_t<BITS,8,0,false>(W@0, x@8, ids@16 (int), sv@24, out@32, HtFold@40 unused here): token-indexed. wg = 20*pos + tile
// (@A65308 magic 0x66666667 >> 3, @A65398-A653D8), expert = ids[pos] (@A65340), x row pos (0x500 B, @A65358), W offset expert*0xc8000 +
// tile*0xa000 (@A653E0-A653F8), sv + expert*0x1400 + tile*256 (@A66AB4-A66AE0), out + pos*0x1400 + tile*256 (@A66BE0-A66C00). Always writes.
template <int BITS>
__device__ static inline void ht_dn_t0(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, unsigned short* out) {
  const int pos = blockIdx.x / 20, tile = blockIdx.x - 20 * pos, e = ids[pos];
  const unsigned* tb = W + ((long)e * (2560 * 640 / 8 * BITS) + (long)tile * (128 * 640 / 8 * BITS)) / 4;
  float s[8];
  if (dn_core<BITS>(tb, x + (long)pos * 640, s, true))
    dn_store(s, sv + (long)e * 2560 + tile * 128, out + (long)pos * 2560 + tile * 128);
}

struct HtFold { long raw[5]; };  // 40 bytes by value at kernarg 40 (layout: see the fold variants)
extern "C" __global__ void k_ht_gemv_dn_t_4_8_0_0(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, unsigned short* out, HtFold) {
  ht_dn_t0<4>(W, x, ids, sv, out);
}
extern "C" __global__ void k_ht_gemv_dn_t_3_8_0_0(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, unsigned short* out, HtFold) {
  ht_dn_t0<3>(W, x, ids, sv, out);
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
// One wave (global wave index gw = 0..7 of the 8 that cover K) accumulates 20 column tiles for the 16 row tiles; returns acc + other half (lanes < 16).
template <int BITS>
__device__ static inline void gu_wave_acc(const unsigned* tb, const _Float16* xrow, int gw, int lane, float (&acc)[16]) {
  const int hh = lane >> 4;
  constexpr int NW = 8 * BITS;
  for (int k = 0; k < 16; k++) acc[k] = 0.f;
  for (int i = 0; i < 20; i++) {
    const int ct = 20 * gw + i;
    _Float16 xv[8];
    for (int c = 0; c < 8; c++) xv[c] = xrow[16 * ct + 8 * hh + c];
    for (int k = 0; k < 16; k++) {
      const unsigned* tp = tb + ct * (16 * NW) + k * NW;
      _Float16 wt[8];
      ht_dec<BITS>(tp, lane, wt);
      acc[k] = dot8(wt, xv, acc[k]);
    }
  }
  for (int k = 0; k < 16; k++) acc[k] = acc[k] + __shfl_xor(acc[k], 16, 32);  // lanes < 16: own (h0) + other (h1)
}
// Wave 0 finish: P is [wave 0..7][k 0..15][r 0..15]; sums waves in order from 0 + P0, then WHT / SiLU / second WHT as documented above.
__device__ static inline void gu_finish(const volatile float* P, const _Float16* svg, const _Float16* sus, _Float16* op) {
  const int r = threadIdx.x & 15, hh = (threadIdx.x >> 4) & 1;
  float g[8], u[8];
  for (int k = 0; k < 16; k++) {
    float a = 0.f + P[k * 16 + r];
    for (int q = 1; q < 8; q++) a = a + P[(q * 16 + k) * 16 + r];
    if (k < 8) g[k] = a; else u[k - 8] = a;
  }
  wht128(g, r);
  wht128(u, r);
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
  for (int j = 0; j < 8; j++) {
    unsigned h;
    asm("v_fma_mixlo_f16 %0, %1, %2, 0" : "=v"(h) : "v"(o[j]), "v"(INV_SQRT128));
    op[16 * j + r] = __builtin_bit_cast(_Float16, (unsigned short)h);
  }
}
template <int BITS>
__device__ static inline void gu_core(const unsigned* tb, const _Float16* xrow, const _Float16* svg, const _Float16* sus, _Float16* op, bool wr) {
  __shared__ float P[8][16][16];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, r = lane & 15, hh = lane >> 4;
  float acc[16];
  gu_wave_acc<BITS>(tb, xrow, w, lane, acc);
  if (hh == 0) for (int k = 0; k < 16; k++) P[w][k][r] = acc[k];
  __syncthreads();
  if (w == 0 && wr) gu_finish(&P[0][0][0], svg, sus, op);
}
template <int BITS>
__device__ static inline void ht_gu11(const unsigned* W, const _Float16* x, const HtItem* items, const int* tok, const _Float16* sv,
                                     const _Float16* su, _Float16* out) {
  const HtItem it = items[blockIdx.x];
  const int xi = it.begin < it.end - 1 ? it.begin : it.end - 1;
  const unsigned* tb = W + ((long)it.expert * (1280L * 2560 / 8 * BITS) + (long)it.tile * (256L * 2560 / 8 * BITS)) / 4;
  gu_core<BITS>(tb, x + (long)tok[xi] * 2560, sv + (long)it.expert * 1280 + it.tile * 128, su + (long)it.expert * 640 + it.tile * 128,
                out + (long)it.begin * 640 + it.tile * 128, it.end - it.begin >= 1);
}

// k_ht_gemv_gu_t<BITS,8,T,U>(W@0, x@8, ids@16 (int), sv@24, su@32, out@40, HtSd@48 (88 B), int@136): slot-indexed. wg = 5*pos + tile
// (@A78908-A78920 magic 0x66666667 >> 1 = wg/5; tile = wg - 5*pos), x row = pos/10 (wg/50, 0x1400 B rows, @A78958-A7897C),
// expert = ids[pos] (@A7893C), sv + expert*0xa00 B + tile*256, su + expert*0x500 + tile*256 (@99B8AC-A99B8F8), out + pos*0x500 + tile*256.
template <int BITS>
__device__ static inline void ht_gu_t(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, const _Float16* su, _Float16* out) {
  const int pos = blockIdx.x / 5, tile = blockIdx.x - 5 * pos, e = ids[pos];
  const unsigned* tb = W + ((long)e * (1280L * 2560 / 8 * BITS) + (long)tile * (256L * 2560 / 8 * BITS)) / 4;
  gu_core<BITS>(tb, x + (long)(pos / 10) * 2560, sv + (long)e * 1280 + tile * 128, su + (long)e * 640 + tile * 128,
                out + (long)pos * 640 + tile * 128, true);
}
struct HtSd { long raw[11]; };  // 88 bytes by value at kernarg 48
extern "C" __global__ void k_ht_gemv_gu_t_4_8_0_0(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, const _Float16* su,
                                                  _Float16* out, HtSd, int) {
  ht_gu_t<4>(W, x, ids, sv, su, out);
}
extern "C" __global__ void k_ht_gemv_gu_t_3_8_0_0(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, const _Float16* su,
                                                  _Float16* out, HtSd, int) {
  ht_gu_t<3>(W, x, ids, sv, su, out);
}

extern "C" __global__ void k_ht_gemv_gu_4_8_1(const unsigned* W, const _Float16* x, const HtItem* items, const int* tok, const _Float16* sv,
                                              const _Float16* su, _Float16* out) {
  ht_gu11<4>(W, x, items, tok, sv, su, out);
}
extern "C" __global__ void k_ht_gemv_gu_3_8_1(const unsigned* W, const _Float16* x, const HtItem* items, const int* tok, const _Float16* sv,
                                              const _Float16* su, _Float16* out) {
  ht_gu11<3>(W, x, items, tok, sv, su, out);
}

// k_ht_gemv_gu_ts<BITS,S>(W@0, x@8, ids@16, sv@24, su@32, out@40, Pf@48 (float scratch), Pj@56 (uint counters)): the gu_t computation with the
// 8 waves of one (pos, tile) spread over S blocks of 8/S waves (block size 32*8/S = 128/64/32 threads). wg = (pos*5 + tile)*S + split
// (S=2: pos = wg/10, tile = wg/2 - 5*pos, @943108-943244). Wave gw = split*(8/S) + wave covers column tiles 20gw..20gw+19, as in gu.
// Each wave stores its 16x16 partials to Pf, thread 0 does atomic_add on Pj[pos*5 + tile] (@945414); the block that sees S-1 runs the gu finish
// on wave 0 over the 8 wave slots in order. Pf layout here is the rewrite's own ([pt][gw][k][r]); only `out` is compared.
template <int BITS, int S>
__device__ static inline void ht_gu_ts(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, const _Float16* su, _Float16* out,
                                      float* Pf, unsigned* Pj) {
  __shared__ int last;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, r = lane & 15, hh = lane >> 4;
  const int split = blockIdx.x % S, pt = blockIdx.x / S, pos = pt / 5, tile = pt - 5 * pos, e = ids[pos];
  const unsigned* tb = W + ((long)e * (1280L * 2560 / 8 * BITS) + (long)tile * (256L * 2560 / 8 * BITS)) / 4;
  float acc[16];
  const int gw = split * (8 / S) + w;
  gu_wave_acc<BITS>(tb, x + (long)(pos / 10) * 2560, gw, lane, acc);
  float* slot = Pf + ((long)pt * 8 + gw) * 256;
  if (hh == 0) for (int k = 0; k < 16; k++) slot[k * 16 + r] = acc[k];
  __threadfence();
  __syncthreads();
  if (tid == 0) last = (atomicAdd(Pj + pt, 1u) == (unsigned)(S - 1));
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (w == 0) gu_finish(Pf + (long)pt * 8 * 256, sv + (long)e * 1280 + tile * 128, su + (long)e * 640 + tile * 128, out + (long)pos * 640 + tile * 128);
}
#define GU_TS(B, S)                                                                                                                         \
  extern "C" __global__ void __launch_bounds__(32 * 8 / S) k_ht_gemv_gu_ts_##B##_##S(const unsigned* W, const _Float16* x, const int* ids,    \
                                                                                  const _Float16* sv, const _Float16* su, _Float16* out,   \
                                                                                  float* Pf, unsigned* Pj) {                                \
    ht_gu_ts<B, S>(W, x, ids, sv, su, out, Pf, Pj);                                                                                         \
  }
GU_TS(4, 2) GU_TS(4, 4) GU_TS(4, 8) GU_TS(3, 2) GU_TS(3, 4) GU_TS(3, 8)

// k_ht_gemv_gu_gs<BITS,8,G>(W@0, x@8, A@16 (HtItem), cnt@24 (int), tok@32 (int), sv@40, su@48, out@56, Pf@64, Pj@72): grouped GEMV, one wave per block,
// wg = 8*I + split, I < 5*(*cnt) (@903509-903534: wg/8 >= cnt*5 exits). Record A[I] = {expert, tile, begin, end}; window of G positions
// begin..begin+G-1 (x row tok[min(begin+g, end-1)], @903568-9035F4), n = min(end-begin, G) of them are real. Wave = split covers column tiles
// 20*split..+19 (@903694 s12*20) for all G tokens with the same decode. Partials go to slot ((begin+g)*5 + tile)*8 + split (@905A58-905BF0, 1024 B each);
// the block that brings the counter to 7 finishes each real token g (gu_finish) into out[(begin+g)*640 + tile*128 + ..].
template <int BITS, int G>
__device__ static inline void ht_gu_gs(const unsigned* W, const _Float16* x, const HtItem* A, const int* cnt, const int* tok, const _Float16* sv,
                                      const _Float16* su, _Float16* out, float* Pf, unsigned* Pj) {
  const int I = blockIdx.x >> 3, split = blockIdx.x & 7;
  if (I >= 5 * cnt[0]) return;
  const HtItem it = A[I];
  const int lane = threadIdx.x, r = lane & 15, hh = lane >> 4;
  const int n = (it.end - it.begin) < G ? (it.end - it.begin) : G;
  constexpr int NW = 8 * BITS;
  const unsigned* tb = W + ((long)it.expert * (1280L * 2560 / 8 * BITS) + (long)it.tile * (256L * 2560 / 8 * BITS)) / 4;
  const _Float16* xr[G];
  for (int g = 0; g < G; g++) {
    int p = it.begin + g; if (p > it.end - 1) p = it.end - 1;
    xr[g] = x + (long)tok[p] * 2560;
  }
  float acc[G][16];
  for (int g = 0; g < G; g++) for (int k = 0; k < 16; k++) acc[g][k] = 0.f;
  for (int i = 0; i < 20; i++) {
    const int ct = 20 * split + i;
    _Float16 xv[G][8];
    for (int g = 0; g < G; g++) for (int c = 0; c < 8; c++) xv[g][c] = xr[g][16 * ct + 8 * hh + c];
    for (int k = 0; k < 16; k++) {
      const unsigned* tp = tb + ct * (16 * NW) + k * NW;
      _Float16 wt[8];
      ht_dec<BITS>(tp, lane, wt);
      for (int g = 0; g < G; g++) acc[g][k] = dot8(wt, xv[g], acc[g][k]);
    }
  }
  for (int g = 0; g < G; g++) {
    float v[16];
    for (int k = 0; k < 16; k++) v[k] = acc[g][k] + __shfl_xor(acc[g][k], 16, 32);
    if (g < n && hh == 0) {
      float* slot = Pf + ((((long)(it.begin + g) * 5 + it.tile) * 8) + split) * 256;
      for (int k = 0; k < 16; k++) slot[k * 16 + r] = v[k];
    }
  }
  __threadfence();
  __syncthreads();
  unsigned old = 0;
  if (lane == 0) old = atomicAdd(Pj + I, 1u);
  old = __shfl(old, 0, 32);
  if (old != 7) return;
  __threadfence();
  for (int g = 0; g < n; g++) {
    const long pos = it.begin + g;
    gu_finish(Pf + (pos * 5 + it.tile) * 8 * 256, sv + (long)it.expert * 1280 + it.tile * 128, su + (long)it.expert * 640 + it.tile * 128,
              out + pos * 640 + it.tile * 128);
  }
}
#define GU_GS(B, G)                                                                                                                         \
  extern "C" __global__ void __launch_bounds__(32) k_ht_gemv_gu_gs_##B##_8_##G(                                                            \
      const unsigned* W, const _Float16* x, const HtItem* A, const int* cnt, const int* tok, const _Float16* sv, const _Float16* su,        \
      _Float16* out, float* Pf, unsigned* Pj) {                                                                                             \
    ht_gu_gs<B, G>(W, x, A, cnt, tok, sv, su, out, Pf, Pj);                                                                                 \
  }
GU_GS(4, 2) GU_GS(4, 4) GU_GS(3, 2) GU_GS(3, 4)


// ---------------------------------------------------------------------------------------------------------------------------
// g32 (const-audit review): HtSd dense path of k_ht_gemv_gu_t<B,8,T,U>, T = 1, 2, 4. Everything above this line is ht_gemv.hip unchanged.
// The vendor gu_t kernels send blocks wg < n (int at kernarg 136) to the token path above and the others to a dense job over the 88-byte HtSd at
// kernarg 48 (the signed-divide shifts 28/29 flagged by const_audit are in that block mapping). It is the job of k_ht_gemv_dn_ts / dn_g, so
// HtSdD, rot_chunk and ht_dense below are copied from ht_gemv_b.hip lines 245-286 and 288-370 (layout and listing cites are in that file).
struct HtSdD { const unsigned* w; const unsigned short* su; const _Float16* sv; const unsigned short* a; const unsigned short* b;
               unsigned short* out; float* part; unsigned* cnt; int m, stride, K, cpb, nsplit, pad; };
__device__ static inline float bf16f(unsigned short v) { return __uint_as_float((unsigned)v << 16); }
__device__ static inline void rot_chunk(const unsigned short* a, const unsigned short* b, const unsigned short* su, _Float16* dst, int lane) {
  float x[4];
  for (int k = 0; k < 4; k++) {
    float av = bf16f(a[4 * lane + k]);
    if (b) {
      float e = __builtin_amdgcn_exp2f(av * -1.44269502162933349609375f);
      float q = av / (1.0f + e);
      float y = q * bf16f(b[4 * lane + k]);
      av = __uint_as_float((unsigned)f2bf_rne(y) << 16);
    }
    x[k] = av;
  }
  float s0 = (float)__builtin_bit_cast(_Float16, su[4 * lane]), s1 = (float)__builtin_bit_cast(_Float16, su[4 * lane + 1]);
  float s2 = (float)__builtin_bit_cast(_Float16, su[4 * lane + 2]), s3 = (float)__builtin_bit_cast(_Float16, su[4 * lane + 3]);
  float p0 = x[0] * s0, p1 = x[1] * s1, p2 = x[2] * s2;
  float A = p0 + p1, B = p0 - p1, C = __builtin_fmaf(x[3], s3, p2), D = __builtin_fmaf(-x[3], s3, p2);
  float h[4] = {A + C, B + D, A - C, B - D};  // columns 4L + {0,1,2,3} = Hadamard bits (c1 c0) = 00, 01, 10, 11
  for (int bit = 1; bit <= 8; bit <<= 1)
    for (int k = 0; k < 4; k++) {
      float p = __shfl_xor(h[k], bit, 32);
      h[k] = (lane & bit) ? (p - h[k]) : (h[k] + p);
    }
  for (int k = 0; k < 4; k++) {
    float p = __shfl_xor(h[k], 16, 32);
    h[k] = (lane & 16) ? (p - h[k]) : (h[k] + p);
  }
  for (int k = 0; k < 4; k++) {
    unsigned hh;
    asm("v_fma_mixlo_f16 %0, %1, %2, 0" : "=v"(hh) : "v"(h[k]), "v"(INV_SQRT128));
    dst[4 * lane + k] = __builtin_bit_cast(_Float16, (unsigned short)hh);
  }
}
// One dense block for T activation rows (dn_ts template argument T = 1, 2, 4; m <= T rows are real, row t reads activation row min(t, m - 1),
// @95C8D4 / T=2 @9624E4 `v_mad_i64_i32 ..., s29, v11` and `v_min_i32 s47`). See the HtSd comment for the block mapping.
// Rotated rows are kept per token. Each wave w owns nch consecutive 16-column tiles [w*nch, (w+1)*nch) of the block's chunks (@95CD04-@95CD60:
// per-wave tile count = s45, base tile s33*8) and decodes each weight tile once for all T rows. Wave partials go to P[w][t][j][r]; element e = t*128 + 16 j + r
// of the block result is the sum over waves 0..7 from `0 + P0` (T=1 @95DF38-95DF74; T=2 @961F14-961F6C stride 1024 B = 2 tokens x 128 floats;
// T=4 @966310-966374 stride 2048 B, two passes e = tid, tid + 256).
// nsplit >= 2: the block stores its sums to part[(sp*T + t)*stride + rb*128 + (e & 127)] (T=1 @95DFB0-95DFE4; T=2 @961FA0-96203C `v_lshl_or_b32 s31, 1, v1`;
// T=4 @9663B4-9664B0), takes ticket old = atomicAdd(cnt[rb]) (@95E058, @9620B0, @96653C); the block with old == nsplit - 1 re-reads
// planes c = 0..nsplit-1 (plane c*T + t) in order summing from 0 (T=1 @95E0F4-95E13C, T=2 @962130-9621D0, T=4 @9665BC-966680) and resets cnt[rb] to 0
// (@95E168, @9621FC, @9666DC). Waves t < min(T, m) then finish token t: WHT-128, * 0x3db504f3, * sv[rb*128 + 16 j + r], bf16 RNE (dn_store),
// out[t*stride + rb*128 + ...] (T=2 @9625E4 `v_mad_i64_i32 ..., s29, v11` ; wave gate T=2 @962214 `v_cmp_gt_i32 s28, v11`, T=4 @9666F8).
template <int BITS, int T>
__device__ static inline void ht_dense(const HtSdD& S, int d) {
  __shared__ _Float16 R[T][20 * 128];
  __shared__ float P[8][T][8][16];
  __shared__ float Z[T * 128];
  __shared__ int last;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, r = lane & 15, hh = lane >> 4;
  const int rb = d / S.nsplit, sp = d - rb * S.nsplit, c0 = sp * S.cpb;
  const int kc = S.K / 128;
  const int nch = (kc < c0 + S.cpb ? kc : c0 + S.cpb) - c0;
  // Vendor bound: the rotated rows live in LDS at row*5120 B + chunk*256 B (T=2: `ds_load_b128 ... offset:5120` @960F48, T=4: offsets 5120/10240/15360
  // @965194-@965240; g22b i4r gl<2> `:1875202` confirms), so at most 5120/256 = 20 chunks per row fit. The host must keep cpb <= 20 (and K/128 per block);
  // the vendor code has no check, so an over-large chunk count is outside the kernel's contract. Trap instead of writing out of bounds.
  if (nch > 20) __builtin_trap();
  constexpr int NW = 8 * BITS;
  for (int idx = w; idx < nch * T; idx += 8) {
    const int t = idx / nch, ci = idx - t * nch;
    const int row = t < S.m - 1 ? t : S.m - 1;
    rot_chunk(S.a + (long)row * S.K + (c0 + ci) * 128, S.b ? S.b + (long)row * S.K + (c0 + ci) * 128 : nullptr, S.su + (c0 + ci) * 128, R[t] + ci * 128, lane);
  }
  __syncthreads();
  const unsigned* tb = S.w + ((long)rb * (S.K / 16) + c0 * 8) * (8 * NW);
  float acc[T][8];
  for (int t = 0; t < T; t++) for (int j = 0; j < 8; j++) acc[t][j] = 0.f;
  for (int i = 0; i < nch; i++) {
    const int ct = w * nch + i;
    _Float16 xv[T][8];
    for (int t = 0; t < T; t++) for (int c = 0; c < 8; c++) xv[t][c] = R[t][16 * ct + 8 * hh + c];
    for (int j = 0; j < 8; j++) {
      const unsigned* tp = tb + ct * (8 * NW) + j * NW;
      _Float16 wt[8];
      ht_dec<BITS>(tp, lane, wt);
      for (int t = 0; t < T; t++) acc[t][j] = dot8(wt, xv[t], acc[t][j]);
    }
  }
  for (int t = 0; t < T; t++)
    for (int j = 0; j < 8; j++) {
      float o = __shfl_xor(acc[t][j], 16, 32);
      if (hh == 0) P[w][t][j][r] = acc[t][j] + o;
    }
  __syncthreads();
  for (int e = tid; e < T * 128; e += 256) {
    const int t = e >> 7, j = (e >> 4) & 7, rr = e & 15;
    float sum = 0.f + P[0][t][j][rr];
    for (int q = 1; q < 8; q++) sum = sum + P[q][t][j][rr];
    Z[e] = sum;
  }
  __syncthreads();
  if (S.nsplit >= 2) {
    for (int e = tid; e < T * 128; e += 256)
      S.part[((long)sp * T + (e >> 7)) * S.stride + rb * 128 + (e & 127)] = Z[e];
    __threadfence();
    __syncthreads();
    if (tid == 0) last = (atomicAdd(S.cnt + rb, 1u) == (unsigned)(S.nsplit - 1));
    __syncthreads();
    if (!last) return;
    __threadfence();
    for (int e = tid; e < T * 128; e += 256) {
      float s = 0.f;
      for (int c = 0; c < S.nsplit; c++) s = s + __builtin_nontemporal_load(S.part + ((long)c * T + (e >> 7)) * S.stride + rb * 128 + (e & 127));
      Z[e] = s;
    }
    if (tid == 0) S.cnt[rb] = 0;
    __syncthreads();
  }
  if (w < T && w < S.m) {
    float s[8];
    for (int j = 0; j < 8; j++) s[j] = Z[w * 128 + 16 * j + r];
    wht128(s, r);
    if (hh == 0) dn_store(s, S.sv + rb * 128, S.out + (long)w * S.stride + rb * 128);
  }
}

template <int BITS, int T>
__device__ static inline void ht_gu_t_sd(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, const _Float16* su, _Float16* out,
                                         const HtSdD& S, int n) {
  if ((int)blockIdx.x < n) { ht_gu_t<BITS>(W, x, ids, sv, su, out); return; }
  ht_dense<4, T>(S, blockIdx.x - n);  // the dense matrix is 4-bit for both BITS (as in dn_ts)
}
#define GU_T_SD(B, T)                                                                                                                       \
  extern "C" __global__ void k_ht_gemv_gu_t_##B##_8_T##T(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv,           \
                                                         const _Float16* su, _Float16* out, HtSdD sd, int n) {                              \
    ht_gu_t_sd<B, T>(W, x, ids, sv, su, out, sd, n);                                                                                        \
  }
GU_T_SD(3, 1) GU_T_SD(3, 2) GU_T_SD(3, 4) GU_T_SD(4, 1) GU_T_SD(4, 2) GU_T_SD(4, 4)
