// ht_gemv_b_c.hip = ht_gemv_b.hip with the NaN-sign fix in dn_store (g14n, see rewrite/reports/g14n.md). Original header follows.
// Rewrites of the ht_gemv instances g14 left BLOCKED (g14b): dn_t with fold=true, dn_ts, dn_g. Code object codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ht_gemv_b.co ht_gemv_b.hip
// Cites: kernels-disasm-v2/codeobj_53000.s:<line> (llvm-objdump 23, 0 `.long`) or code-object addresses (@982500 = 0x982500).
// The helpers up to dn_store are copied unchanged from rewrite/kernels/ht_gemv.hip (g14, MATCH on dn / dn_t<false>); see that file for the cites.
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
    // sv is src0 of v_fma_mix_f32 ..., neg(0) in the vendor (kernels-disasm-v2/codeobj_53000.s:1151876); when sv and y are both NaN the
    // hardware returns src0, so sv must come first (g14n). The compiler folds fma(sv, y, -0) to a commuted v_mul_f32, so the vendor instruction is emitted by asm.
    unsigned sh = __builtin_bit_cast(unsigned short, svp[16 * j + r]);
    float o;
    asm("v_fma_mix_f32 %0, %1, %2, neg(0) op_sel_hi:[1,0,0]" : "=v"(o) : "v"(sh), "v"(y));
    y = o;
    op[16 * j + r] = f2bf_rne(y);
  }
}

// ---------------------------------------------------------------------------------------------------------------------------
// k_ht_gemv_dn_t<BITS,8,T,true>(W@0, x@8, ids@16, sv@24, out@32, HtFold@40): the dn_t partial (g14: out + pos*2560 + tile*128, pos = wg/20)
// followed by a fold of the 10 partial rows of a token. Cites are addresses in kernels-disasm-v2/codeobj_53000.s of dn_t<4,8,1,true> (@982500);
// the other T and the 3-bit kernels are covered by their own rows in tables/g14b.md.
// HtFold (40 B by value at kernarg 40, five pointers) from the kernarg loads `s_load_b512 s[4:19], 0` (@982500: s14:15 = kernarg 40, s16:17 = 48,
// s18:19 = 56) and `s_load_b128 s[4:7], 0x40` (@98369C: s4:5 = kernarg 64, s6:7 = 72):
//   q0@40 const float*  fw[tok*10 + k]   slot weight: s[10:11] = s14:15 + 40*tok (@98413C-@984140), global_load_b32 offsets 0..36 (@984148-@9843B8)
//   q1@48 const bf16*   resid[tok*2560 + e]  s16:17 + 2*(tok*2560 + e) (@9843DC-@9843E8, load @9843FC; v0 = e + tok*0xa00 from @984100)
//   q2@56 const float*  scal[tok]            s[8:9] = s18:19 + 4*tok (@984160-@984164), load @984404
//   q3@64 bf16*         fout[tok*2560 + e]   s4:5 + 2*(tok*2560 + e) (@984410-@984420), store @984500
//   q4@72 unsigned*     cnt[tok*20 + tile]   s20:21 = s6:7 + 4*(tok*20 + tile) (@984040-@984044), atomic add @984058, reset store @98452C
// tok = wg/200 (@983FC8 s_mul_hi_i32 0x51eb851f ; @983FE8 s_ashr 6), tile = wg - 20*(wg/20) (@98258C). After the partial every block adds 1 to
// cnt[tok*20 + tile] (thread 0, @984058 global_atomic_add_u32 ... glc); the block that gets old == 9 (@984078 v_cmp_eq_u32 9) runs the fold with
// threads 0..63 (@9840CC v_cmpx_gt_u32 64), 2 bf16 each (e = tile*128 + 2*tid, @9840E8), then thread 0 resets the counter to 0 (@98452C).
// Fold math per element pair (lo, hi) of the 10 slot rows at out[(10*tok + k)*2560 + e] (slot stride 0x1400 B, @984100-@98412C):
//   m_k = fw[k] * slot_k (v_mul_f32 on the bf16 widened by shift / mask; @984190-@984198 ...), f32 sums k = 0..9 as 0 + m_0, + m_1, ... + m_9
//   (v_add_f32 chain, lo @98440C-@98446C, hi @9841AC-@9843F8), each sum rounded to bf16 (v_bfe_u32 16,1 ; v_add3_u32 0x7fff ; and 0xffff0000,
//   hi @984418-@984450, lo @984478-@984490), then fma(scal[tok], resid, rounded) (v_fmac_f32 @9844B0 lo, @9844B4 hi), rounded to bf16 RNE
//   (@9844BC-@9844F0) and stored as one dword. The counters end at 0 (not compared: compare.py refuses an unchanged output).
struct HtFold { const float* fw; const unsigned short* resid; const float* scal; unsigned short* fout; unsigned* cnt; };

__device__ static inline float bf_round(float f) {  // sum rounded to bf16 and kept as f32 (@984418-984490)
  unsigned u = __float_as_uint(f);
  return __uint_as_float((u + ((u >> 16) & 1u) + 0x7fffu) & 0xffff0000u);
}
template <int BITS>
__device__ static inline void ht_dn_t_fold(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, unsigned short* out, HtFold F) {
  __shared__ int last;
  const int pos = blockIdx.x / 20, tile = blockIdx.x - 20 * pos, e = ids[pos];
  const unsigned* tb = W + ((long)e * (2560 * 640 / 8 * BITS) + (long)tile * (128 * 640 / 8 * BITS)) / 4;
  float s[8];
  if (dn_core<BITS>(tb, x + (long)pos * 640, s, true))
    dn_store(s, sv + (long)e * 2560 + tile * 128, out + (long)pos * 2560 + tile * 128);
  const int tok = blockIdx.x / 200, tid = threadIdx.x;
  __threadfence();
  __syncthreads();
  if (tid == 0) last = (atomicAdd(F.cnt + tok * 20 + tile, 1u) == 9u);
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (tid < 64) {
    const long el = (long)tok * 2560 + tile * 128 + 2 * tid;
    const volatile unsigned* part = (const volatile unsigned*)(out + (long)(10 * tok) * 2560 + tile * 128 + 2 * tid);
    float lo = 0.f, hi = 0.f;
    for (int k = 0; k < 10; k++) {
      unsigned v = part[(long)k * 1280];  // 2560 bf16 = 1280 dwords per slot row
      float w = F.fw[tok * 10 + k];
      float ml = w * __uint_as_float(v << 16), mh = w * __uint_as_float(v & 0xffff0000u);
      lo = lo + ml;
      hi = hi + mh;
    }
    lo = bf_round(lo);
    hi = bf_round(hi);
    const unsigned r = *(const unsigned*)(F.resid + el);
    const float sc = F.scal[tok];
    lo = __builtin_fmaf(sc, __uint_as_float(r << 16), lo);
    hi = __builtin_fmaf(sc, __uint_as_float(r & 0xffff0000u), hi);
    *(unsigned*)(F.fout + el) = (unsigned)f2bf_rne(lo) | ((unsigned)f2bf_rne(hi) << 16);
    if (tid == 0) F.cnt[tok * 20 + tile] = 0;
  }
}
#define DN_T_FOLD(B, T)                                                                                                                     \
  extern "C" __global__ void k_b_dn_t_##B##_8_##T##_1(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv,                  \
                                                      unsigned short* out, HtFold F) {                                                     \
    ht_dn_t_fold<B>(W, x, ids, sv, out, F);                                                                                                 \
  }
DN_T_FOLD(4, 0) DN_T_FOLD(4, 1) DN_T_FOLD(4, 2) DN_T_FOLD(4, 3) DN_T_FOLD(4, 4)
DN_T_FOLD(3, 0) DN_T_FOLD(3, 1) DN_T_FOLD(3, 2) DN_T_FOLD(3, 3) DN_T_FOLD(3, 4)

// ---------------------------------------------------------------------------------------------------------------------------
// k_ht_gemv_dn_ts<BITS,8,T,false>(W@0, x@8, ids@16, sv@24, out@32, HtSd@40 (88 B), int n@128, HtFold@136 (40 B)).
// Blocks wg < n take the token path (codeobj_53000 @95C424 s_cmp_lt_i32 s2, s3 ; @95C428 s_cbranch_scc1 to @95E728), the others a dense path
// over HtSd (see below). Token path @95E728-@960150: pos = wg/20 (@95E734 magic 0x66666667 >> 3), tile = wg - 20 pos, expert = ids[pos] (@95E75C-95E764),
// x row pos staged in LDS (0x500 B, @95E784-95E7AC), W + expert*0xc8000 + tile*0xa000 (@95E7F4-95E810), out + pos*0x1400 + tile*256
// (@95FFF8-960014), sv + expert*0x1400 + tile*256 (@95FECC-95FEE0), same WHT and 0x3db504f3 scale and bf16 RNE as dn_t (@95FF84 v_mul_f32 0x3db504f3).
// HtFold is not read on the token path (no s_load at kernarg 0x88 in @95E728-@960150).
template <int BITS>
__device__ static inline void ht_dn_ts_tok(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, unsigned short* out) {
  const int pos = blockIdx.x / 20, tile = blockIdx.x - 20 * pos, e = ids[pos];
  const unsigned* tb = W + ((long)e * (2560 * 640 / 8 * BITS) + (long)tile * (128 * 640 / 8 * BITS)) / 4;
  float s[8];
  if (dn_core<BITS>(tb, x + (long)pos * 640, s, true))
    dn_store(s, sv + (long)e * 2560 + tile * 128, out + (long)pos * 2560 + tile * 128);
}

// ---------------------------------------------------------------------------------------------------------------------------
// HtSd (88 B by value, kernarg 40 in dn_ts, 48 in dn_g): the "dense" job that rides in the same launch as extra blocks (wg >= n).
// Decoded from k_ht_gemv_dn_ts<4,8,1,false> (codeobj_53000 @95C400-@95E728, dense path; kernarg loads @95C404 s_load_b32 0x80, @95C40C s_load_b512 0,
// @95C42C s_load_b64 0x74, @95C43C s_load_b128 0x68, @95C444 s_load_b64 0x60, @95C44C s_load_b256 0x40). Field = register use:
//   +0  w     const unsigned*  HT weights [rb][K/16 col tiles][8 row tiles], tile = 8*BITS words (s22:23, @95CD64-@95CD68, tile base << 10 for 4 bit)
//   +8  su    const half*      per-column scale [K] (s24:25: global_load_b32 + d16 at +4 / +6, @95C95C-@95C970, @95C600)
//   +16 sv    const half*      per-row scale [N] (s26:27, @95E4D0-@95E4D8, offsets 32 apart = rows 16 j + r)
//   +24 a     const ushort*    activations, bf16 [M][K] (s36:37, @95C920, row stride s30 = K, row clamp min(row, m - 1) @95C8D4)
//   +32 b     const ushort*    second operand, bf16 [M][K], may be null (s38:39, null test @95C51C s_cmp_lg_u64 -> s48)
//   +40 out   ushort*          bf16 [M][N] (s40:41, @95E5D8-@95E5E4; stores @95E6E0-@95E718)
//   +48 part  float*           split partials [nsplit][stride] (s42:43, @95DFC8-@95DFE4 store, @95E0F4-@95E11C loads)
//   +56 cnt   unsigned*        split counters [N/128] (s34:35, @95E044 atomic add)
//   +64 m (s28), +68 stride (s29), +72 K (s30), +76 cpb = chunks of 128 columns per block (s8), +80 nsplit = blocks per row block (s9), +84 pad
// Dense block id d = wg - n (@95C4... s3 = s2 - s3): rb = d / nsplit (@95C460-@95C4CC integer division), sp = d - rb*nsplit (s31),
// first chunk c0 = sp*cpb (s33 = s31*s8), chunk count nch = min(K/128, c0 + cpb) - c0 (s45, @95C4F4-@95C504).
struct HtSdD { const unsigned* w; const unsigned short* su; const _Float16* sv; const unsigned short* a; const unsigned short* b;
               unsigned short* out; float* part; unsigned* cnt; int m, stride, K, cpb, nsplit, pad; };

// Rotation of one 128-column activation chunk (wave-wide, lane L owns columns 4L..4L+3): x = a (or bf16(silu(a) * b)), * su, 128-point WHT,
// * 0x3db504f3 stored as f16 (@95C614-@95C860, ds_store_b64 offset 9728 + chunk*256 + lane*8 = chunk-major f16 [128]).
// First stage (fused): p_k = x_k * su_k (v_fma_mix ... neg(0) = rounded product) for k = 0..2, q3 = x3 * su3 kept exact in a fma:
//   A = p0 + p1, B = p0 - p1, C = fma(x3, su3, p2), D = fma(-x3, su3, p2); h00 = A + C, h10 = A - C, h01 = B + D, h11 = B - D (@95C614-@95C63C).
// Then the lane stages 1,2,4,8 (row_xmask) and lane ^ 16 (permlanex16): lane with the bit set computes partner - own, else own + partner.
// silu(a) = a / (1 + exp2(a * -1.442695)) with IEEE division (@95C998-@95CB98), product with b, bf16 RNE without NaN handling (@95CBB4-@95CC08).
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
__device__ static inline void ht_dn_ts(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, unsigned short* out, const HtSdD& S, int n) {
  if (T == 0 || (int)blockIdx.x < n) { ht_dn_ts_tok<BITS>(W, x, ids, sv, out); return; }
  if constexpr (T >= 1) ht_dense<4, T>(S, blockIdx.x - n);  // the dense matrix is 4-bit for both BITS (3-bit kernel @94EBB0.. uses bfe 12/8/4/0 and 1024 B tile groups)
}
#define DN_TS(B, T)                                                                                                                         \
  extern "C" __global__ void k_b_dn_ts_##B##_8_##T(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv,                    \
                                                  unsigned short* out, HtSdD sd, int n, HtFold F) {                                        \
    ht_dn_ts<B, T>(W, x, ids, sv, out, sd, n);                                                                                              \
  }
DN_TS(4, 0) DN_TS(4, 1) DN_TS(4, 2) DN_TS(4, 4)
DN_TS(3, 0) DN_TS(3, 1) DN_TS(3, 2) DN_TS(3, 4)

// ---------------------------------------------------------------------------------------------------------------------------
// k_ht_gemv_dn_g<BITS,8,G,K>(W@0, x@8, A@16 (HtItem), cnt@24 (int), sv@32, out@40, HtSd@48 (88 B), int n@136): grouped down GEMV.
// K = 0: only the item path (codeobj_53000 dn_g<4,8,2,0> @92AB00-@92C788; the kernel ends at @92C788 s_endpgm when wg >= 20*cnt, no HtSd/int read).
// K = 2, 4: the same item path (blocks wg < n, @922120 s_cmp_lt_i32 s2, s3 with s3 = int@0x88 ; @92470C item path start, which repeats
// `wg >= 20*cnt[0]` -> exit at @924728-@924738) preceded by a dense path over HtSd for wg >= n.
// Item path (K = 0 listing): block wg < 20*cnt[0] (@92AB0C-@92AB24 s_load_b32 cnt ; s_mul_i32 20) reads A[wg] = {expert, tile, begin, end}
// (@92AB30-@92AB44 s_lshl_b64 4 ; s_load_b128), stages G x rows min(begin + g, end - 1) in LDS (640 halves each; @92AB50 v_cmpx 0xa0 = 160 threads x 16 B
// = 2 rows, g = v0 >= 80, row = min(begin + g, end - 1), @92AB68 s_add_i32 s15 -1, @92AB90 v_mad_i64_i32 0x500), W + expert*0xc8000 + tile*0xa000
// (@92AC18-@92AC34). Waves 5w..5w+4 column tiles as in dn, but the decoded weights feed G accumulator sets. Per-wave partials go to LDS
// ([w][g][j][r]); wave g < n = min(end - begin, G) (@92BF28-@92BF44 s_sub s15 s14 ; s_min_i32 2 ; v_cmpx_gt_i32 s0, v11) sums the 8 wave partials from 0 +
// (@92BF98 `v_add_f32 v5, 0, v5` then adds in ds_load order), runs the WHT-128 and dn_store for row begin + g.
template <int BITS, int G>
__device__ static inline void ht_dn_g_item(const unsigned* W, const _Float16* x, const HtItem* A, const int* cnt, const _Float16* sv, unsigned short* out) {
  __shared__ float P[8][G][8][16];
  if (blockIdx.x >= 20 * cnt[0]) return;
  const HtItem it = A[blockIdx.x];
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, r = lane & 15, hh = lane >> 4;
  constexpr int NW = 8 * BITS;
  const unsigned* tb = W + ((long)it.expert * (2560 * 640 / 8 * BITS) + (long)it.tile * (128 * 640 / 8 * BITS)) / 4;
  const _Float16* xr[G];
  for (int g = 0; g < G; g++) {
    int p = it.begin + g; if (p > it.end - 1) p = it.end - 1;
    xr[g] = x + (long)p * 640;
  }
  float acc[G][8];
  for (int g = 0; g < G; g++) for (int j = 0; j < 8; j++) acc[g][j] = 0.f;
  for (int i = 0; i < 5; i++) {
    const int ct = 5 * w + i;
    _Float16 xv[G][8];
    for (int g = 0; g < G; g++) for (int c = 0; c < 8; c++) xv[g][c] = xr[g][16 * ct + 8 * hh + c];
    for (int j = 0; j < 8; j++) {
      const unsigned* tp = tb + ct * (8 * NW) + j * NW;
      _Float16 wt[8];
      ht_dec<BITS>(tp, lane, wt);
      for (int g = 0; g < G; g++) acc[g][j] = dot8(wt, xv[g], acc[g][j]);
    }
  }
  for (int g = 0; g < G; g++)
    for (int j = 0; j < 8; j++) {
      float o = __shfl_xor(acc[g][j], 16, 32);
      if (hh == 0) P[w][g][j][r] = acc[g][j] + o;
    }
  __syncthreads();
  const int n = (it.end - it.begin) < G ? (it.end - it.begin) : G;
  if (w < n) {
    float s[8];
    for (int j = 0; j < 8; j++) {
      float a = 0.f + P[0][w][j][r];
      for (int q = 1; q < 8; q++) a = a + P[q][w][j][r];
      s[j] = a;
    }
    wht128(s, r);
    if (hh == 0) dn_store(s, sv + (long)it.expert * 2560 + it.tile * 128, out + (long)(it.begin + w) * 2560 + it.tile * 128);
  }
}
// K = 2, 4: blocks wg >= n run the dense job (HtSd layout identical to dn_ts: dn_g loads the same fields at kernarg 48 + offset, e.g. @922134
// s_load_b256 s[44:51], 0x60 = part, cnt, m, stride, K, cpb and @922124 s_load_b64 0x7c = cpb, nsplit; int n at 0x88 @922104), for K tokens.
template <int BITS, int G, int K>
__device__ static inline void ht_dn_g(const unsigned* W, const _Float16* x, const HtItem* A, const int* cnt, const _Float16* sv, unsigned short* out,
                                      const HtSdD& S, int n) {
  if (K == 0 || (int)blockIdx.x < n) { ht_dn_g_item<BITS, G>(W, x, A, cnt, sv, out); return; }
  if constexpr (K >= 1) ht_dense<4, K>(S, blockIdx.x - n);
}
#define DN_G(B, G, K)                                                                                                                       \
  extern "C" __global__ void k_b_dn_g_##B##_8_##G##_##K(const unsigned* W, const _Float16* x, const HtItem* A, const int* cnt,                   \
                                                       const _Float16* sv, unsigned short* out, HtSdD sd, int n) {                         \
    ht_dn_g<B, G, K>(W, x, A, cnt, sv, out, sd, n);                                                                                         \
  }
DN_G(4, 2, 0) DN_G(4, 2, 2) DN_G(4, 2, 4) DN_G(4, 4, 0) DN_G(4, 4, 2) DN_G(4, 4, 4)
DN_G(3, 2, 0) DN_G(3, 2, 2) DN_G(3, 2, 4) DN_G(3, 4, 0) DN_G(3, 4, 2) DN_G(3, 4, 4)
