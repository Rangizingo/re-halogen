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
    y = (float)svp[16 * j + r] * y;
    op[16 * j + r] = f2bf_rne(y);
  }
}

// ---------------------------------------------------------------------------------------------------------------------------
// k_ht_gemv_dn_t<BITS,8,T,true>(W@0, x@8, ids@16, sv@24, out@32, HtFold@40): the dn_t partial (g14: out + pos*2560 + tile*128, pos = wg/20)
// followed by a fold of 10 partial rows per token. HtFold (40 B, five pointers, decoded from the kernarg loads in dn_t<4,8,1,true>,
// codeobj_53000 kernels-disasm-v2 line 1217782 `s_load_b512 s[4:19], 0` and line 1218351 `s_load_b128 s[4:7], 0x40`):
//   q0@40 const float*  fw[tok*10 + k]   slot weight (line 1218790 `global_load_b32 v7, v6, s[10:11]`, s[10:11] = q0 + 40*tok at 1218760-1218762)
//   q1@48 const bf16*   resid[tok*2560 + e]  (line 1218796 `v_add_co_u32 v2, s16, v0`, s16:17 = kernarg 48)
//   q2@56 const float*  scal[tok]            (line 1218797 `global_load_b32 v3, v6, s[8:9]`, s[8:9] = s18:19 + 4*tok at 1218789-1218791)
//   q3@64 bf16*         fout[tok*2560 + e]   (line 1218802 `v_add_co_u32 v0, s4, v0`, s[4:5] = kernarg 64; store at line 1218927)
//   q4@72 unsigned*     cnt[tok*20 + tile]   (atomic add line 1218751, s[6:7] = kernarg 72, s20 = s[6:7] + 4*(tok*20 + tile))
// tok = wg/200 (@983FC8 mul_hi 0x51eb851f >> 6), tile = wg - 20*(wg/20). After its partial every block adds 1 to cnt[tok*20+tile] (thread 0,
// line 1218751 `global_atomic_add_u32 ... glc`); the block that sees old == 9 (@984078 v_cmp_eq 9) runs the fold with threads 0..63
// (@9840CC v_cmpx_gt_u32 64), 2 elements each (e = tile*128 + 2*tid), then thread 0 resets the counter to 0 (line 1218956).
// Fold math, per element pair (lo, hi) of the 10 slot rows at out[(10*tok + k)*2560 + e] (stride 0x1400 B, @984100-98412C):
//   m_k = fw[k] * slot_k (v_mul_f32, bf16 widened by shift/mask), sums run k = 0..9 as 0 + m_0, + m_1, ..., + m_9 in f32 (v_add_f32; line 1218790-1218920),
//   each sum rounded to bf16 (v_bfe_u32 16,1 ; v_add3_u32 0x7fff ; and 0xffff0000, @984418-984490), then fma(scal[tok], resid, rounded) (v_fmac_f32 @9844B0/9844B4),
//   rounded to bf16 RNE (@9844BC-9844F0) and stored as one dword.
// The weights stored by the partial pass are those of dn_t<false> (the T template argument does not change the math, g14).
struct HtFold { const float* fw; const unsigned short* resid; const float* scal; unsigned short* fout; unsigned* cnt; };

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
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
struct HtSd { long raw[11]; };
template <int BITS>
__device__ static inline void ht_dn_ts_tok(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv, unsigned short* out) {
  const int pos = blockIdx.x / 20, tile = blockIdx.x - 20 * pos, e = ids[pos];
  const unsigned* tb = W + ((long)e * (2560 * 640 / 8 * BITS) + (long)tile * (128 * 640 / 8 * BITS)) / 4;
  float s[8];
  if (dn_core<BITS>(tb, x + (long)pos * 640, s, true))
    dn_store(s, sv + (long)e * 2560 + tile * 128, out + (long)pos * 2560 + tile * 128);
}
#define DN_TS(B, T)                                                                                                                         \
  extern "C" __global__ void k_b_dn_ts_##B##_8_##T(const unsigned* W, const _Float16* x, const int* ids, const _Float16* sv,                    \
                                                  unsigned short* out, HtSd sd, int n, HtFold F) {                                         \
    ht_dn_ts_tok<B>(W, x, ids, sv, out);                                                                                                    \
  }
DN_TS(4, 0) DN_TS(4, 1) DN_TS(4, 2) DN_TS(4, 4)
DN_TS(3, 0) DN_TS(3, 1) DN_TS(3, 2) DN_TS(3, 4)
