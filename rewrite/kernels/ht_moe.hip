// Rewrites of the "ht_moe" family (k_ht_moe_*) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ht_moe.co ht_moe.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). HT format: analysis/closure/ht-decode.md, items: htitem-r3.md.
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;
typedef _Float16 half_t;
typedef half_t v16h __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

struct HtItem { int expert, tile, begin, end; };

__device__ static inline u16 f2bf(float f) {
  u32 u = __float_as_uint(f);
  return (u16)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
#define S3 __uint_as_float(0x3db504f3u)

// HT weight value as f16 (no bf16 rounding on this path: the f16 result of v_pk_fma_f16 is stored to LDS as is).
// p = x*0x83dcd12d ; S = bytesum(p) + 0x6400 ; w = fma_f16(f16bits(S), f16(0x1eee), f16(0xc931)).
// (dn<4,8,1> listing: codeobj_53000.s:1929881-1929921, hash 0x83dcd12d at :1929882, sad_u8 :1929899, pk_fma :1929917)
__device__ static inline half_t ht_w16(u32 x) {
  u32 p = x * 0x83dcd12du;
  u32 s = 0x6400u + (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24);
  return __builtin_fmaf16(__builtin_bit_cast(half_t, (u16)s), __builtin_bit_cast(half_t, (u16)0x1eee),
                          __builtin_bit_cast(half_t, (u16)0xc931));
}
// 16-bit window of a tile (8*B words) at bit pos of the reversed circular bit string (see ht_deq.hip).
template <int B>
__device__ static inline u32 tile_window(const u32* tw, int pos) {
  constexpr int W = 8 * B;
  int wi = (W - 1) - (pos >> 5), off = pos & 31;
  int wn = wi == 0 ? W - 1 : wi - 1;
  u64 pair = ((u64)tw[wn] << 32) | tw[wi];
  return (u32)(pair >> off) & 0xffffu;
}
// The 16 weights of row rr of a 16x16 tile, columns 0..15 (h = 0,1 lane halves of ht_deq).
template <int B>
__device__ static inline v16h tile_row16(const u32* tw, int rr) {
  v16h r;
#pragma unroll
  for (int h = 0; h < 2; h++)
#pragma unroll
    for (int c = 0; c < 8; c++) {
      int j = 128 * h + 8 * rr + c;
      r[8 * h + c] = ht_w16(tile_window<B>(tw, B * (255 - j)));
    }
  return r;
}

// Hadamard over 128 outputs of one token row held as 8 values per lane x 16 lanes:
// FWHT bit order j0, j1, j2 in registers (lo+hi, lo-hi), then lane xor 1, 2, 4, 8 (own bit set: partner - own).
// (dn<4,8,1>: stage 1 listing `v4=v66+v58, v5=v66-v58 ...`; lane stages `row_xmask:1/2/4/8`)
__device__ static inline void had_reg(const float* a1, float* c) {
  float a[8], b[8];
  for (int p = 0; p < 4; p++) { a[2 * p] = a1[2 * p] + a1[2 * p + 1]; a[2 * p + 1] = a1[2 * p] - a1[2 * p + 1]; }
  for (int q = 0; q < 8; q += 4)
    for (int d = 0; d < 2; d++) { b[q + d] = a[q + d] + a[q + d + 2]; b[q + d + 2] = a[q + d] - a[q + d + 2]; }
  for (int d = 0; d < 4; d++) { c[d] = b[d] + b[d + 4]; c[d + 4] = b[d] - b[d + 4]; }
}
// same, first stage with the f16 column scale folded in by fma_mix: p = fma(s0,h0,h1*s1), m = fma(s0,h0,-(h1*s1))
__device__ static inline void had_reg_scaled(const float* s, const float* h, float* c) {
  float a[8], b[8];
  for (int p = 0; p < 4; p++) {
    float t1 = __builtin_fmaf(h[2 * p + 1], s[2 * p + 1], -0.0f);
    a[2 * p] = __builtin_fmaf(h[2 * p], s[2 * p], t1);
    a[2 * p + 1] = __builtin_fmaf(h[2 * p], s[2 * p], -t1);
  }
  for (int q = 0; q < 8; q += 4)
    for (int d = 0; d < 2; d++) { b[q + d] = a[q + d] + a[q + d + 2]; b[q + d + 2] = a[q + d] - a[q + d + 2]; }
  for (int d = 0; d < 4; d++) { c[d] = b[d] + b[d + 4]; c[d + 4] = b[d] - b[d + 4]; }
}
__device__ static inline void had_lanes(float* x, int lane) {
  for (int k = 0; k < 8; k++) {
    float v = x[k];
    for (int bit = 1; bit <= 8; bit <<= 1) {
      float pr = __shfl_xor(v, bit);
      v = ((lane & bit) ? -v : v) + pr;
    }
    x[k] = v;
  }
}
__device__ static inline void had128(float* x, int lane) {
  float c[8];
  had_reg(x, c);
  for (int k = 0; k < 8; k++) x[k] = c[k];
  had_lanes(x, lane);
}

// k_ht_moe_dn<B, 8, G>(u32 const* w, half const* act, HtItem const* items, half const* sv, u16* out)
// kernarg w@0 act@8 items@16 sv@24 out@32. Block 256 = 8 waves; workgroup id = item index.
// Item {expert, tile, begin, end}: weights at w + expert*E + tile*T (E = 0xc8000/0x96000, T = 0xa000/0x7800 bytes for
// B = 4/3, :1929820-1929826 and DNG notes), tile = 128 output rows x K = 640. Wave w handles positions begin+16w .. +15
// (A rows, clamped to end-1), 8 WMMA 16x16x16 f16 per k step (one per 16-row subtile j), 40 k steps (K = 640), accumulate
// from zero in k order. Activation row stride 0x500 bytes (:1929941). Dequant tile (ks, rt) is at word (ks*8+rt)*8B.
// Epilogue per accumulator element i (token 2i + lane/16): had128, * 1/sqrt(128), * sv[expert][tile*128+16k+lane%16]
// (fma_mix with -0), bf16 RNE, store out[pos*2560 + tile*128 + 16k + lane%16] if pos < end (:1930308-1930393).
// G (third template arg) and the second (8) do not change the code (dn<4,8,1> and dn<4,8,8> listings are identical).
template <int B>
__device__ static inline void moe_dn(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  constexpr long EXP = (B == 4) ? 0xc8000 : 0x96000;
  constexpr long TIL = (B == 4) ? 0xa000 : 0x7800;
  HtItem it = items[blockIdx.x];
  int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const u32* wb = (const u32*)((const char*)w + (long)it.expert * EXP + (long)it.tile * TIL);
  int m = lane & 15;
  int pos0 = it.begin + 16 * wave;
  int arow = pos0 + m;
  if (arow > it.end - 1) arow = it.end - 1;
  const half_t* ap = act + (long)arow * 640;
  v8f acc[8];
  for (int j = 0; j < 8; j++) acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  bool active = pos0 < it.end;
  if (active) {
    for (int ks = 0; ks < 40; ks++) {
      v16h a;
      for (int k = 0; k < 16; k++) a[k] = ap[ks * 16 + k];
      for (int j = 0; j < 8; j++) {
        const u32* tw = wb + (long)(ks * 8 + j) * (8 * B);
        v16h bv = tile_row16<B>(tw, m);
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bv, acc[j]);
      }
    }
  }
  const half_t* svp = sv + (long)it.expert * 2560 + (long)it.tile * 128;
  for (int i = 0; i < 8; i++) {
    float x[8];
    for (int j = 0; j < 8; j++) x[j] = acc[j][i];
    had128(x, lane);
    int pos = pos0 + 2 * i + (lane >> 4);
    if (pos < it.end) {
      u16* op = out + (long)pos * 2560 + (long)it.tile * 128;
      for (int k = 0; k < 8; k++) {
        float y = x[k] * S3;
        op[16 * k + m] = f2bf(__builtin_fmaf((float)svp[16 * k + m], y, -0.0f));
      }
    }
  }
}

extern "C" __global__ void k_ht_moe_dn_b3(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  moe_dn<3>(w, act, items, sv, out);
}
extern "C" __global__ void k_ht_moe_dn_b4(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  moe_dn<4>(w, act, items, sv, out);
}

// k_ht_moe_gu<B, 8, G>(u32 const* w, half const* x, HtItem const* items, int const* ids, half const* sv, half const* suh, half* out)
// kernarg w@0 x@8 items@16 ids@24 sv@32 suh@40 out@48. Block 256 = 8 waves, workgroup id = item index.
// Weights: w + expert*E + tile*T (E = 0x190000/0x12c000, T = 0x50000/0x3c000 for B = 4/3; gu481 listing :1931807-1931819).
// A tile is 256 rows (gate rows 0..127 = subtiles j 0..7, up rows = j 8..15), K = 2560 (160 steps); dequant tile (ks, rt)
// is at word (ks*16+rt)*8B. A row = x + ids[min(pos, end-1)]*0x1400 (:1931955). 16 accumulators per wave, k order.
// Epilogue per accumulator element i: had128 on gate and on up (:1932227-1932494), * 1/sqrt(128) (v_mul),
// * sv (fma_mix, -0) with sv[expert*1280 + tile*128 + 16c + lane%16] (gate) and +640 (up) (loads :1932284-1932295),
// silu: x / (1 + exp2(x * -1.442695)) (v_exp_f32, IEEE division, :1932507-1932652), s_c = silu(g_c) * u_c,
// then the next layer's input rotation: t_c = suh[expert*640 + tile*128 + 16c + lane%16] * s_c folded into the first
// stage as fma (:1932699-1932716), 7-stage FWHT, fma_mixlo/hi_f16 with 1/sqrt(128) (+0) (:1932828-1932835),
// out[pos*640 + tile*128 + 16c + lane%16] (f16) if pos < end.
template <int B>
__device__ static inline void moe_gu(const u32* w, const half_t* x, const HtItem* items, const int* ids, const half_t* sv,
                                     const half_t* suh, half_t* out) {
  constexpr long EXP = (B == 4) ? 0x190000 : 0x12c000;
  constexpr long TIL = (B == 4) ? 0x50000 : 0x3c000;
  HtItem it = items[blockIdx.x];
  int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const u32* wb = (const u32*)((const char*)w + (long)it.expert * EXP + (long)it.tile * TIL);
  int m = lane & 15;
  int pos0 = it.begin + 16 * wave;
  int p = pos0 + m;
  if (p > it.end - 1) p = it.end - 1;
  const half_t* ap = x + (long)ids[p] * 2560;
  v8f acc[16];
  for (int j = 0; j < 16; j++) acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  if (pos0 < it.end) {
    for (int ks = 0; ks < 160; ks++) {
      v16h a;
      for (int k = 0; k < 16; k++) a[k] = ap[ks * 16 + k];
      for (int j = 0; j < 16; j++) {
        const u32* tw = wb + (long)(ks * 16 + j) * (8 * B);
        v16h bv = tile_row16<B>(tw, m);
        acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, bv, acc[j]);
      }
    }
  }
  const half_t* svp = sv + (long)it.expert * 1280 + (long)it.tile * 128;
  const half_t* sup = suh + (long)it.expert * 640 + (long)it.tile * 128;
  for (int i = 0; i < 8; i++) {
    float g[8], u[8];
    for (int j = 0; j < 8; j++) { g[j] = acc[j][i]; u[j] = acc[8 + j][i]; }
    had128(g, lane);
    had128(u, lane);
    float s[8], h[8];
    for (int c = 0; c < 8; c++) {
      float gs = __builtin_fmaf((float)svp[16 * c + m], g[c] * S3, -0.0f);
      float us = __builtin_fmaf((float)svp[640 + 16 * c + m], u[c] * S3, -0.0f);
      float e = __builtin_amdgcn_exp2f(gs * -1.44269502162933349609375f);
      float q = gs / (1.0f + e);
      s[c] = us * q;
      h[c] = (float)sup[16 * c + m];
    }
    float r[8];
    had_reg_scaled(s, h, r);
    had_lanes(r, lane);
    int pos = pos0 + 2 * i + (lane >> 4);
    if (pos < it.end) {
      half_t* op = out + (long)pos * 640 + (long)it.tile * 128;
      for (int c = 0; c < 8; c++) op[16 * c + m] = (half_t)__builtin_fmaf(r[c], S3, 0.0f);
    }
  }
}

// ---- k_ht_moe_gu3<B, 1, E>: same math inputs as gu, block of 512 threads (16 waves), up to 128 tokens per item. ----
// Wave w holds weight rows 16w..16w+15 of the 256-row tile (w < 8 gate, w >= 8 up) as the WMMA A operand, token tiles of
// 16 as B (listing :1466778-1466792: v_wmma v[57:64], v[157:164] (weights), v[93:100] (activations from LDS)). 8 token tiles,
// 160 k steps (gu3<4,1,0> loop s20 0..39 x 4 sub steps), accumulate from zero in k order.
// The epilogue regroups through LDS so a wave32 owns one token (4 gate rows + 4 up rows per lane, rows 4*lane+e) and uses the
// rot-style transform: 4 values in registers (bits 0,1) then lane xor 1,2,4,8 and 16 (permlanex16)  (:1466867-1466977).
__device__ static inline void lanes5(float* r, int lane, int n) {
  for (int k = 0; k < n; k++) {
    float v = r[k];
    v = ((lane & 1) ? -v : v) + __shfl_xor(v, 1);
    v = ((lane & 2) ? -v : v) + __shfl_xor(v, 2);
    v = ((lane & 4) ? -v : v) + __shfl_xor(v, 4);
    v = ((lane & 8) ? -v : v) + __shfl_xor(v, 8);
    v = ((lane & 16) ? -v : v) + __shfl_xor(v, 16);
    r[k] = v;
  }
}
__device__ static inline void reg4(const float* a, float* r) {
  float p0 = a[0] + a[1], m0 = a[0] - a[1], p1 = a[2] + a[3], m1 = a[2] - a[3];
  r[0] = p0 + p1; r[1] = m0 + m1; r[2] = p0 - p1; r[3] = m0 - m1;
}
__device__ static inline void reg4_scaled(const float* s, const float* h, float* r) {
  float t1 = __builtin_fmaf(h[1], s[1], -0.0f);
  float t3 = __builtin_fmaf(h[3], s[3], -0.0f);
  float p0 = __builtin_fmaf(h[0], s[0], t1), m0 = __builtin_fmaf(h[0], s[0], -t1);
  float p1 = __builtin_fmaf(h[2], s[2], t3), m1 = __builtin_fmaf(h[2], s[2], -t3);
  r[0] = p0 + p1; r[1] = m0 + m1; r[2] = p0 - p1; r[3] = m0 - m1;
}

// NW waves per block (row tiles per wave RT = 16/NW), GRPS groups of 64 tokens (MAXT = 64*GRPS). gu3: NW 16, GRPS 2; gu2: NW 8, GRPS 1.
template <int B, int NW, int GRPS>
__device__ static inline void moe_gut(const u32* w, const half_t* x, const HtItem* items, const int* ids, const half_t* sv,
                                      const half_t* suh, half_t* out) {
  constexpr long EXP = (B == 4) ? 0x190000 : 0x12c000;
  constexpr long TIL = (B == 4) ? 0x50000 : 0x3c000;
  constexpr int RT = 16 / NW, NT = 4 * GRPS, MAXT = 64 * GRPS;
  __shared__ float S[256 * 64];  // [row][token in group of 64]
  HtItem it = items[blockIdx.x];
  int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  int cnt = it.end - it.begin;
  if (cnt > MAXT) cnt = MAXT;
  const u32* wb = (const u32*)((const char*)w + (long)it.expert * EXP + (long)it.tile * TIL);
  int m = lane & 15;
  v8f acc[RT][NT];
  for (int r = 0; r < RT; r++)
    for (int t = 0; t < NT; t++) acc[r][t] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  const half_t* ap[NT];
  for (int t = 0; t < NT; t++) {
    int tk = 16 * t + m;
    if (tk > cnt - 1) tk = cnt - 1;
    ap[t] = x + (long)ids[it.begin + tk] * 2560;
  }
  int ntiles = (cnt + 15) >> 4;
  for (int ks = 0; ks < 160; ks++) {
    for (int r = 0; r < RT; r++) {
      const u32* tw = wb + (long)(ks * 16 + wave + NW * r) * (8 * B);
      v16h wa = tile_row16<B>(tw, m);
      for (int t = 0; t < ntiles; t++) {
        v16h bv;
        for (int k = 0; k < 16; k++) bv[k] = ap[t][ks * 16 + k];
        acc[r][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(wa, bv, acc[r][t]);
      }
    }
  }
  const half_t* svp = sv + (long)it.expert * 1280 + (long)it.tile * 128;
  const half_t* sup = suh + (long)it.expert * 640 + (long)it.tile * 128;
  for (int grp = 0; grp < GRPS; grp++) {
    __syncthreads();
    for (int r = 0; r < RT; r++)
      for (int tt = 0; tt < 4; tt++)
        for (int i = 0; i < 8; i++) S[(16 * (wave + NW * r) + 2 * i + (lane >> 4)) * 64 + 16 * tt + m] = acc[r][4 * grp + tt][i];
    __syncthreads();
    for (int iter = 0; iter < 64 / NW; iter++) {
      int tl = wave + NW * iter;           // token within the group of 64
      int t = 64 * grp + tl;
      if (t >= cnt) continue;
      float g4[4], u4[4];
      for (int e = 0; e < 4; e++) { g4[e] = S[(4 * lane + e) * 64 + tl]; u4[e] = S[(128 + 4 * lane + e) * 64 + tl]; }
      float g[4], u[4];
      reg4(g4, g);
      reg4(u4, u);
      lanes5(g, lane, 4);
      lanes5(u, lane, 4);
      float s[4], h[4];
      for (int e = 0; e < 4; e++) {
        float gs = __builtin_fmaf((float)svp[4 * lane + e], g[e] * S3, -0.0f);
        float us = __builtin_fmaf((float)svp[640 + 4 * lane + e], u[e] * S3, -0.0f);
        float ex = __builtin_amdgcn_exp2f(gs * -1.44269502162933349609375f);
        float q = gs / (1.0f + ex);
        s[e] = us * q;
        h[e] = (float)sup[4 * lane + e];
      }
      float r4[4];
      reg4_scaled(s, h, r4);
      lanes5(r4, lane, 4);
      half_t* op = out + (long)(it.begin + t) * 640 + (long)it.tile * 128;
      for (int e = 0; e < 4; e++) op[4 * lane + e] = (half_t)__builtin_fmaf(r4[e], S3, 0.0f);
    }
  }
}

extern "C" __global__ void k_ht_moe_gu2_b3(const u32* w, const half_t* x, const HtItem* items, const int* ids,
                                            const half_t* sv, const half_t* suh, half_t* out) {
  moe_gut<3, 8, 1>(w, x, items, ids, sv, suh, out);
}
extern "C" __global__ void k_ht_moe_gu2_b4(const u32* w, const half_t* x, const HtItem* items, const int* ids,
                                            const half_t* sv, const half_t* suh, half_t* out) {
  moe_gut<4, 8, 1>(w, x, items, ids, sv, suh, out);
}
extern "C" __global__ void k_ht_moe_gu3_b3_e0(const u32* w, const half_t* x, const HtItem* items, const int* ids,
                                              const half_t* sv, const half_t* suh, half_t* out) {
  moe_gut<3, 16, 2>(w, x, items, ids, sv, suh, out);
}
extern "C" __global__ void k_ht_moe_gu3_b4_e0(const u32* w, const half_t* x, const HtItem* items, const int* ids,
                                              const half_t* sv, const half_t* suh, half_t* out) {
  moe_gut<4, 16, 2>(w, x, items, ids, sv, suh, out);
}

// ---- k_ht_moe_dn3<B, 1, E>(w, act, items, sv, out): block 512, waves 0..7 handle tile `item.tile`, waves 8..15 tile+1
// (weights at tile*T via v_mad v6*40 KiB units, listing dn3<4,1,0> :1520913 with v6 = tile + wave/8). Each tile is 128 rows x
// K = 640; token tiles of 16 as WMMA B operand, rows as A. Epilogue as gu3 (wave per token, rows 4*lane+e, reg4 + 5 lane
// stages), then * 1/sqrt(128), * sv (fma_mix, -0), bf16 RNE, b64 stores at out + pos*0x1400 + tile*256 B (+256 for the
// second tile) (:1521429-1521430).
template <int B>
__device__ static inline void moe_dn3(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  constexpr long EXP = (B == 4) ? 0xc8000 : 0x96000;
  constexpr long TIL = (B == 4) ? 0xa000 : 0x7800;
  __shared__ float S[256 * 64];
  HtItem it = items[blockIdx.x];
  int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  int cnt = it.end - it.begin;
  if (cnt > 128) cnt = 128;
  int te = it.tile + (wave >> 3), rt = wave & 7;
  const u32* wb = (const u32*)((const char*)w + (long)it.expert * EXP + (long)te * TIL);
  int m = lane & 15;
  v8f acc[8];
  for (int t = 0; t < 8; t++) acc[t] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  const half_t* ap[8];
  for (int t = 0; t < 8; t++) {
    int tk = 16 * t + m;
    if (tk > cnt - 1) tk = cnt - 1;
    ap[t] = act + (long)(it.begin + tk) * 640;
  }
  int ntiles = (cnt + 15) >> 4;
  for (int ks = 0; ks < 40; ks++) {
    const u32* tw = wb + (long)(ks * 8 + rt) * (8 * B);
    v16h wa = tile_row16<B>(tw, m);
    for (int t = 0; t < ntiles; t++) {
      v16h bv;
      for (int k = 0; k < 16; k++) bv[k] = ap[t][ks * 16 + k];
      acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(wa, bv, acc[t]);
    }
  }
  const half_t* svp = sv + (long)it.expert * 2560;
  for (int grp = 0; grp < 2; grp++) {
    __syncthreads();
    for (int tt = 0; tt < 4; tt++)
      for (int i = 0; i < 8; i++) S[(128 * (wave >> 3) + 16 * rt + 2 * i + (lane >> 4)) * 64 + 16 * tt + m] = acc[4 * grp + tt][i];
    __syncthreads();
    for (int iter = 0; iter < 4; iter++) {
      int tl = wave + 16 * iter;
      int t = 64 * grp + tl;
      if (t >= cnt) continue;
      for (int half = 0; half < 2; half++) {
        float a4[4], r4[4];
        for (int e = 0; e < 4; e++) a4[e] = S[(128 * half + 4 * lane + e) * 64 + tl];
        reg4(a4, r4);
        lanes5(r4, lane, 4);
        u16* op = out + (long)(it.begin + t) * 2560 + (long)(it.tile + half) * 128;
        for (int e = 0; e < 4; e++) {
          float y = r4[e] * S3;
          op[4 * lane + e] = f2bf(__builtin_fmaf((float)svp[(it.tile + half) * 128 + 4 * lane + e], y, -0.0f));
        }
      }
    }
  }
}

// ---- k_ht_moe_dn2<B, 64, E>: block 256 (8 waves), one 128-row tile per block (weights at tile*T), up to 64 tokens. ----
template <int B>
__device__ static inline void moe_dn2(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  constexpr long EXP = (B == 4) ? 0xc8000 : 0x96000;
  constexpr long TIL = (B == 4) ? 0xa000 : 0x7800;
  __shared__ float S[128 * 64];
  HtItem it = items[blockIdx.x];
  int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  int cnt = it.end - it.begin;
  if (cnt > 64) cnt = 64;
  const u32* wb = (const u32*)((const char*)w + (long)it.expert * EXP + (long)it.tile * TIL);
  int m = lane & 15;
  v8f acc[4];
  for (int t = 0; t < 4; t++) acc[t] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  const half_t* ap[4];
  for (int t = 0; t < 4; t++) {
    int tk = 16 * t + m;
    if (tk > cnt - 1) tk = cnt - 1;
    ap[t] = act + (long)(it.begin + tk) * 640;
  }
  int ntiles = (cnt + 15) >> 4;
  for (int ks = 0; ks < 40; ks++) {
    const u32* tw = wb + (long)(ks * 8 + wave) * (8 * B);
    v16h wa = tile_row16<B>(tw, m);
    for (int t = 0; t < ntiles; t++) {
      v16h bv;
      for (int k = 0; k < 16; k++) bv[k] = ap[t][ks * 16 + k];
      acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(wa, bv, acc[t]);
    }
  }
  const half_t* svp = sv + (long)it.expert * 2560 + (long)it.tile * 128;
  __syncthreads();
  for (int tt = 0; tt < 4; tt++)
    for (int i = 0; i < 8; i++) S[(16 * wave + 2 * i + (lane >> 4)) * 64 + 16 * tt + m] = acc[tt][i];
  __syncthreads();
  for (int iter = 0; iter < 8; iter++) {
    int t = wave + 8 * iter;
    if (t >= cnt) continue;
    float a4[4], r4[4];
    for (int e = 0; e < 4; e++) a4[e] = S[(4 * lane + e) * 64 + t];
    reg4(a4, r4);
    lanes5(r4, lane, 4);
    u16* op = out + (long)(it.begin + t) * 2560 + (long)it.tile * 128;
    for (int e = 0; e < 4; e++) {
      float y = r4[e] * S3;
      op[4 * lane + e] = f2bf(__builtin_fmaf((float)svp[4 * lane + e], y, -0.0f));
    }
  }
}
// ---- k_ht_moe_dn2x<B, 64, P, E>: block 256, two consecutive 128-row tiles (tile, tile+1) per block, up to 64 tokens. ----
template <int B>
__device__ static inline void moe_dn2x(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  constexpr long EXP = (B == 4) ? 0xc8000 : 0x96000;
  constexpr long TIL = (B == 4) ? 0xa000 : 0x7800;
  __shared__ float S[256 * 64];
  HtItem it = items[blockIdx.x];
  int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  int cnt = it.end - it.begin;
  if (cnt > 64) cnt = 64;
  const u32* wb = (const u32*)((const char*)w + (long)it.expert * EXP + (long)it.tile * TIL);
  int m = lane & 15;
  v8f acc[2][4];
  for (int h = 0; h < 2; h++)
    for (int t = 0; t < 4; t++) acc[h][t] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
  const half_t* ap[4];
  for (int t = 0; t < 4; t++) {
    int tk = 16 * t + m;
    if (tk > cnt - 1) tk = cnt - 1;
    ap[t] = act + (long)(it.begin + tk) * 640;
  }
  int ntiles = (cnt + 15) >> 4;
  for (int ks = 0; ks < 40; ks++) {
    for (int h = 0; h < 2; h++) {
      const u32* tw = wb + (long)(h * (TIL / 4)) + (long)(ks * 8 + wave) * (8 * B);
      v16h wa = tile_row16<B>(tw, m);
      for (int t = 0; t < ntiles; t++) {
        v16h bv;
        for (int k = 0; k < 16; k++) bv[k] = ap[t][ks * 16 + k];
        acc[h][t] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(wa, bv, acc[h][t]);
      }
    }
  }
  const half_t* svp = sv + (long)it.expert * 2560;
  __syncthreads();
  for (int h = 0; h < 2; h++)
    for (int tt = 0; tt < 4; tt++)
      for (int i = 0; i < 8; i++) S[(128 * h + 16 * wave + 2 * i + (lane >> 4)) * 64 + 16 * tt + m] = acc[h][tt][i];
  __syncthreads();
  for (int iter = 0; iter < 8; iter++) {
    int t = wave + 8 * iter;
    if (t >= cnt) continue;
    for (int h = 0; h < 2; h++) {
      float a4[4], r4[4];
      for (int e = 0; e < 4; e++) a4[e] = S[(128 * h + 4 * lane + e) * 64 + t];
      reg4(a4, r4);
      lanes5(r4, lane, 4);
      u16* op = out + (long)(it.begin + t) * 2560 + (long)(it.tile + h) * 128;
      for (int e = 0; e < 4; e++) {
        float y = r4[e] * S3;
        op[4 * lane + e] = f2bf(__builtin_fmaf((float)svp[(it.tile + h) * 128 + 4 * lane + e], y, -0.0f));
      }
    }
  }
}
extern "C" __global__ void k_ht_moe_dn2x_b3(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  moe_dn2x<3>(w, act, items, sv, out);
}
extern "C" __global__ void k_ht_moe_dn2x_b4(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  moe_dn2x<4>(w, act, items, sv, out);
}

extern "C" __global__ void k_ht_moe_dn2_b3(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  moe_dn2<3>(w, act, items, sv, out);
}
extern "C" __global__ void k_ht_moe_dn2_b4(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  moe_dn2<4>(w, act, items, sv, out);
}

extern "C" __global__ void k_ht_moe_dn3_b3(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  moe_dn3<3>(w, act, items, sv, out);
}
extern "C" __global__ void k_ht_moe_dn3_b4(const u32* w, const half_t* act, const HtItem* items, const half_t* sv, u16* out) {
  moe_dn3<4>(w, act, items, sv, out);
}

extern "C" __global__ void k_ht_moe_gu_b3(const u32* w, const half_t* x, const HtItem* items, const int* ids,
                                          const half_t* sv, const half_t* suh, half_t* out) {
  moe_gu<3>(w, x, items, ids, sv, suh, out);
}
extern "C" __global__ void k_ht_moe_gu_b4(const u32* w, const half_t* x, const HtItem* items, const int* ids,
                                          const half_t* sv, const half_t* suh, half_t* out) {
  moe_gu<4>(w, x, items, ids, sv, suh, out);
}
