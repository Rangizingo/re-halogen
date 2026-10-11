// Rewrites of the remaining "i4r_rows" instances from codeobj_53000 (flash_serve, gfx1151): k_i4r_rows_fl_g<G> (HtItem group, down GEMV + fold),
// k_i4r_rows_gl_g, k_i4r_rows_gl, k_i4r_rows_gl_o. (g22b; g22's i4r_rows.hip is untouched.)
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o i4r_rows_b.co i4r_rows_b.hip
// Listing: kernels-disasm-v2/codeobj_53000.s (full decode, 0 .long; not in git). Every ":NNNN" below is a line of that file.
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ static inline float xstep(float v, int m) {
  float partner = __shfl_xor(v, m);
  return ((threadIdx.x & m) ? -v : v) + partner;
}
// Fold tail shared with k_i4r_fold / k_i4r_rows_fl (g22): 4-point transform in the lane, xor butterflies 1..16, * 0x3db504f3, then
// y_k = bf16_rne(fma(half(h[k]), v, -0.0)), out_k = bf16_rne(fma(z, bf16(r[k]), y_k)).
__device__ static inline void fold_tail(const float acc[4], const unsigned short* h, const unsigned short* r, float zt, unsigned short* o) {
  float p0 = acc[0] + acc[1], p1 = acc[0] - acc[1], p2 = acc[2] + acc[3], p3 = acc[2] - acc[3];
  float e[4] = {p0 + p2, p1 + p3, p0 - p2, p1 - p3};
  for (int m = 1; m <= 16; m <<= 1)
    for (int k = 0; k < 4; k++) e[k] = xstep(e[k], m);
  float C = __uint_as_float(0x3db504f3u);
  for (int k = 0; k < 4; k++) {
    float v = e[k] * C;
    float y = bf2f(f2bf_rne(__builtin_fmaf((float)__builtin_bit_cast(_Float16, h[k]), v, -0.0f)));
    o[k] = f2bf_rne(__builtin_fmaf(zt, bf2f(r[k]), y));
  }
}

// One 128-weight block of the int4 GEMV for one lane quad member: weights 16 B at wp (32 nibbles, low nibble first), f16 activations xq (32 halves).
// n - 8 as f16 pairs, v_dot2acc_f32_f16 ascending, accumulator starts at 0 (listing :1870356-1870473, v114..v118).
__device__ static inline float dot_block(uint4 wv, const _Float16* xq) {
  unsigned wd[4] = {wv.x, wv.y, wv.z, wv.w};
  float a = 0.0f;
  for (int w = 0; w < 4; w++)
    for (int k = 0; k < 4; k++) {
      unsigned byte = (wd[w] >> (8 * k)) & 0xffu;
      half2_t wh = {(_Float16)(int)((byte & 15u) - 8u), (_Float16)(int)((byte >> 4) - 8u)};
      half2_t xh = *(const half2_t*)(xq + 8 * w + 2 * k);
      a = __builtin_amdgcn_fdot2(wh, xh, a, false);
    }
  return a;
}

// halogen::(anonymous namespace)::k_i4r_rows_fl_g<G>(unsigned char const*, long, _Float16 const*, HtItem const*, int const*, int const*, float*,
//   float const*, _Float16 const*, unsigned short const*, float const*, unsigned short*, unsigned int*)
// kernarg: W@0, soff@8, x@16 (f16 [slot][640]), items@24 (HtItem {expert, tile, begin, end}, 5 records per item), ids@32 (int, slot id per position),
// cnt@40 (int, item count), y@48 (f32 [slot][2560]), wt@56 (f32 [token*10+m]), h@64 (f16 [2560]), r@72 (bf16 [token][2560]), z@80 (f32 [token]),
// out@88 (bf16 [token][2560]), ctr@96 (u32 [token*20 + blk/2]).
// Block: item = blockIdx / 40 (s_mul_hi_i32 0x66666667, ashr 4, :1869907-1869911 for G=2; :1868805-1868809 for G=4); blk = blockIdx - 40*item
// (s_mul_i32 -40, :1869984-1869986). Exits when item >= cnt[0] (s_cmp_ge_i32, :1869913). Record = items + item*5*16 B (:1869915-1869921);
// expert = rec[0] (s_load_b32 s14 :1869924), begin = rec[2], end = rec[3] (s_load_b64 +8 :1869923); n = end - begin, ns = min(n, G) (:1869927-1869929).
// Threads tid < G store ids[begin + tid] (0 when tid >= ns) to LDS +2560 (+5120 for G=4) (:1869930-1869946). x rows: for each slot i < ns, threads
// tid < 80 copy 16 B of x + sid[i]*1280 B into LDS slot i (0x500 B stride) (:1869974-1869982). If n <= 0 nothing else runs (:1869990).
// GEMV: as k_i4r_rows<640> with row = expert*2560 + blk*64 + 8*warp + ((tid>>2)&7) (v_mad_i64_i32 0xa00 :1870002); per slot i (loop s4 < s3, :1870343)
// the dot2 chain over the shared weights (:1870356-1870473), total = fma(s0, a0, 0), then fmac with scales 1..4 (:1870452-1870473), xor-1 and xor-2 adds
// (DPP, :1870474-1870482); lane quad 0 stores y[sid[i]*2560 + row%2560] (v_mad_i64_i32 0x2800, :1870486-1870487).
// Fold: after the stores (inv + barrier :1870493-1870500) threads tid < ns do old = atomicAdd(ctr + (sid/10)*20 + blk/2, 1) (:1870504-1870519),
// flag[tid] = (old == 19) (:1870521-1870523). Warp w < ns with flag[w] folds token sid[w]/10 columns (blk/2)*128 + 4*lane (:1870547): as k_i4r_rows_fl
// (y reads with glc, m = 0..9 ascending, f = f + wt*y), then lane 0 resets the counter (:1870932-1870940).
template <int G>
__device__ static inline void fl_g_body(const unsigned char* W, long soff, const _Float16* x, const int4* items, const int* ids, const int* cnt,
                                        float* y, const float* wt, const unsigned short* h, const unsigned short* r, const float* z,
                                        unsigned short* out, unsigned* ctr) {
  constexpr int K = 640, NB = 5, N = 2560;
  __shared__ _Float16 xs[G][K];
  __shared__ int sid[G];
  __shared__ int flag[G];
  int tid = threadIdx.x;
  int item = blockIdx.x / 40;
  int blk = blockIdx.x - 40 * item;
  if (item >= cnt[0]) return;
  int4 rec = items[item * 5];
  int expert = rec.x, begin = rec.z, end = rec.w;
  int n = end - begin;
  int ns = n < G ? n : G;
  if (tid < G) sid[tid] = (tid < ns) ? ids[begin + tid] : 0;
  __syncthreads();
  if (n <= 0) return;
  for (int i = 0; i < ns; i++)
    if (tid < 80) *(float4*)(&xs[i][8 * tid]) = *(const float4*)(x + (long)sid[i] * K + 8 * tid);
  __syncthreads();
  int q = tid & 3;
  int rr = blk * 64 + 8 * (tid >> 5) + ((tid >> 2) & 7);
  long row = (long)expert * N + rr;
  const unsigned char* wp = W + row * (K / 2) + 16 * q;
  const _Float16* sp = (const _Float16*)(W + soff + row * (NB * 2));
  uint4 wv[NB];
  for (int m = 0; m < NB; m++) wv[m] = *(const uint4*)(wp + 64 * m);
  for (int i = 0; i < ns; i++) {
    float acc[NB];
    for (int m = 0; m < NB; m++) acc[m] = dot_block(wv[m], &xs[i][128 * m + 32 * q]);
    float tot = __builtin_fmaf((float)sp[0], acc[0], 0.0f);
    for (int m = 1; m < NB; m++) tot = __builtin_fmaf((float)sp[m], acc[m], tot);
    tot = tot + __shfl_xor(tot, 1);
    tot = tot + __shfl_xor(tot, 2);
    if (q == 0) y[(long)sid[i] * N + rr] = tot;
  }
  __syncthreads();
  if (tid < ns) {
    __threadfence();
    unsigned old = atomicAdd(&ctr[(sid[tid] / 10) * 20 + (blk >> 1)], 1u);
    flag[tid] = (old == 19u);
  }
  __syncthreads();
  int w = tid >> 5, lane = tid & 31;
  if (w < ns && flag[w]) {
    __threadfence();
    int tok = sid[w] / 10;
    int col = (blk >> 1) * 128 + 4 * lane;
    float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int m = 0; m < 10; m++) {
      float wm = wt[tok * 10 + m];
      const volatile float* yp = (const volatile float*)(y + (long)(tok * 10 + m) * N + col);
      for (int k = 0; k < 4; k++) f[k] = f[k] + wm * yp[k];
    }
    fold_tail(f, h + col, r + (long)tok * N + col, z[tok], out + (long)tok * N + col);
    if (lane == 0) ctr[tok * 20 + (blk >> 1)] = 0u;
  }
}

extern "C" __global__ void k_i4r_rows_fl_g2(const unsigned char* W, long soff, const _Float16* x, const int4* items, const int* ids, const int* cnt,
                                            float* y, const float* wt, const unsigned short* h, const unsigned short* r, const float* z,
                                            unsigned short* out, unsigned* ctr) {
  fl_g_body<2>(W, soff, x, items, ids, cnt, y, wt, h, r, z, out, ctr);
}
extern "C" __global__ void k_i4r_rows_fl_g4(const unsigned char* W, long soff, const _Float16* x, const int4* items, const int* ids, const int* cnt,
                                            float* y, const float* wt, const unsigned short* h, const unsigned short* r, const float* z,
                                            unsigned short* out, unsigned* ctr) {
  fl_g_body<4>(W, soff, x, items, ids, cnt, y, wt, h, r, z, out, ctr);
}

// =================================================================================================================================
// k_i4r_rows_gl<R> family. One launch holds two independent jobs; blocks < nGemv (kernarg 160) run job P2, the rest run job P1.
//   P2 (listing :1872260-1874646 for gl<1>): routed-expert gate/up GEMV (the k_i4r_rows<2560> body, N = 1280 rows per expert) with the
//      k_i4r_gu_epi item pipeline fused through a ticket counter.
//   P1 (listing :1870954-1872258 for gl<1>): "dense HT" job. Prelude builds the rotated f16 activation in LDS from bf16 gate/up arrays,
//      then an HT-format (4 bit, 16x16 tiles) GEMV with split-K and the output Hadamard + sv scale.
// I4rSd (88 B by value at kernarg 72) fields, read off the P1 scalar loads (s_load_b512 s[12:27] at kernarg 0x40, :1870949; s_load_b64 0x94 :1870954;
// s_load_b128 0x88 :1870957; s_load_b64 0x80 :1870958):
//   +0 wht (HT tiles base, s14:15), +8 su (f16, s16:17), +16 sv (f16, s18:19), +24 ga (bf16 gate rows, s20:21), +32 up (bf16, s22:23, may be null),
//   +40 out (bf16 result, s24:25), +48 pf (f32 split-K partials, s26:27), +56 cnt (u32 tickets, s36:37), +64 nrows (s28), +68 ld (s29: Pf row stride
//   in floats and out row stride in elements), +72 K (s30), +80 nsplit (s11), +76 chunk (s10: 128-blocks per split), +84 unused.
struct I4rSd {
  const unsigned* wht; const _Float16* su; const _Float16* sv; const unsigned short* ga; const unsigned short* up;
  unsigned short* out; float* pf; unsigned* cnt;
  int nrows, ld, K, chunk, nsplit, pad;
};

// HT 4-bit weight decode (copy of the g14 rewrite in ht_gemv.hip, ht-decode.md "Unified decoder"): lane t of a 128 B tile holds word t; the 16-bit
// window for weight c is pair {word[t-1], word[t]} >> 4*(7-c); p = window * 0x83dcd12d; S = byte sum + 0x6400 (v_sad_u8); w = fma_f16(S, 0x1eee, 0xc931).
__device__ static inline _Float16 ht_w(unsigned x16) {
  unsigned p = x16 * 0x83dcd12du;
  unsigned s = (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24) + 0x6400u;
  _Float16 hs = __builtin_bit_cast(_Float16, (unsigned short)s);
  _Float16 a = __builtin_bit_cast(_Float16, (unsigned short)0x1eee);
  _Float16 b = __builtin_bit_cast(_Float16, (unsigned short)0xc931);
  return __builtin_fmaf16(hs, a, b);
}
__device__ static inline void ht_dec4(const unsigned* tp, int t, _Float16 (&w)[8]) {
  unsigned prev = tp[(t + 31) & 31], cur = tp[t];
  unsigned long long pr = ((unsigned long long)prev << 32) | cur;
  for (int c = 0; c < 8; c++) w[c] = ht_w((unsigned)((pr >> (4 * (7 - c))) & 0xffffu));
}
__device__ static inline float dot8(const _Float16 (&w)[8], const _Float16* xv, float acc) {
  for (int p = 0; p < 4; p++) {
    half2_t a = {w[2 * p], w[2 * p + 1]};
    half2_t b = {xv[2 * p], xv[2 * p + 1]};
    acc = __builtin_amdgcn_fdot2(a, b, acc, false);
  }
  return acc;
}
__device__ static inline float silu_g(float g) {
  float e = __builtin_amdgcn_exp2f(g * -1.44269502162933349609375f);
  return g / (1.0f + e);
}

// k_i4r_gu_epi item (slot row yrow of 1280 floats, item jj of 5, one wave; g22 i4r_gu.hip). y loads are glc (:1874258-1874265).
// a@40 (1280 halves: gate scales then up scales), bq@48 (640 halves), result f16 x4 per lane to orow + jj*128 + 4*lane (:1874634).
__device__ static inline void gl_epi(const float* yrow, const _Float16* a, const _Float16* bq, _Float16* orow, int jj, int lane) {
  int ix = jj * 128 + lane * 4;
  const volatile float* gr = (const volatile float*)(yrow + ix);
  float xg[4], xu[4];
  for (int k = 0; k < 4; k++) { xg[k] = gr[k]; xu[k] = gr[640 + k]; }
  float C = __uint_as_float(0x3db504f3u);
  float e[4], f[4];
  for (int pass = 0; pass < 2; pass++) {
    float* v = pass ? xu : xg;
    float* r = pass ? f : e;
    float p0 = v[0] + v[1], p1 = v[0] - v[1], p2 = v[2] + v[3], p3 = v[2] - v[3];
    r[0] = p0 + p2; r[2] = p0 - p2; r[1] = p1 + p3; r[3] = p1 - p3;
    for (int m = 1; m <= 16; m <<= 1)
      for (int k = 0; k < 4; k++) r[k] = xstep(r[k], m);
    for (int k = 0; k < 4; k++) r[k] = r[k] * C;
  }
  float prod[4];
  for (int k = 0; k < 4; k++) {
    float gs = __builtin_fmaf((float)a[ix + k], e[k], -0.0f);
    float us = __builtin_fmaf((float)a[640 + ix + k], f[k], -0.0f);
    prod[k] = us * silu_g(gs);
  }
  float bb[4];
  for (int k = 0; k < 4; k++) bb[k] = (float)bq[ix + k];
  float t1 = __builtin_fmaf(bb[1], prod[1], -0.0f), t3 = __builtin_fmaf(bb[3], prod[3], -0.0f);
  float s01 = __builtin_fmaf(prod[0], bb[0], t1), d01 = __builtin_fmaf(prod[0], bb[0], -t1);
  float s23 = __builtin_fmaf(prod[2], bb[2], t3), d23 = __builtin_fmaf(prod[2], bb[2], -t3);
  float hh[4] = {s01 + s23, d01 + d23, s01 - s23, d01 - d23};
  for (int m = 1; m <= 16; m <<= 1)
    for (int k = 0; k < 4; k++) hh[k] = xstep(hh[k], m);
  _Float16* o = orow + ix;
  for (int k = 0; k < 4; k++) o[k] = (_Float16)__builtin_fmaf(hh[k], C, 0.0f);
}

// ---- P2: gate/up GEMV + ticketed gu_epi item ----
// Block: slot = blockIdx / 20 (magic 0xcccccccd >> 4, :1872308-1872310), blk = blockIdx - 20*slot (:1872317-1872320), x row = blockIdx / 200
// (magic 0x51eb851f >> 6, :1872267-1872269), 320 threads' worth of 16 B staged to LDS (x row of 2560 halves, 0x1400 B per row, :1872271).
// rowmap[slot] (:1872319), row = rowmap*1280 + blk*64 + 8*warp + ((tid>>2)&7) (:1872324). GEMV as k_i4r_rows<2560>; y[slot*1280 + r] by quad 0
// (global_store_b32, :1874182, slot*0x1400 :1874175). Then fence + barrier (:1874185-1874194), ticket index slot*5 + ((blk%10)/2) (:1874192-1874205),
// thread 0 atomicAdd(+1), flag = (old == 3) (:1874228, four arrivals: two gate blocks, two up blocks). The last block runs the k_i4r_gu_epi item
// (slot, j) with its first 32 threads on y (glc loads :1874258-1874265), a@40, bq@48 (:1874276-1874284), writes out@56 + slot*1280 B + j*256 B + 8*lane
// (global_store_b64, :1874634), and thread 0 resets ctr[slot*5 + j] = 0 (:1874643).
// gl_o (sel != null): slot = sel[blockIdx / 20] replaces the slot everywhere except blk (listing gl_o<1> :1846355 region: s_load_b64 sel at kernarg 0xa8,
// global_load_b32 sel + 4*(blockIdx/20) -> s10, x row = s10 / 10 (signed magic 0x66666667 >> 2), rowmap[s10], y row s10*0x1400, ticket s10*5 + j,
// out row s10*0x500; blk = blockIdx - 20*(blockIdx/20)). The P1 half is byte-identical to gl.
__device__ static inline void gl_p2(const unsigned char* W, long soff, const _Float16* x, const int* rowmap, float* y, const _Float16* a,
                                   const _Float16* bq, _Float16* out, unsigned* ctr, const int* sel) {
  constexpr int K = 2560, NB = 20, N = 1280;
  __shared__ _Float16 xs[K];
  __shared__ int last;
  int tid = threadIdx.x;
  int slot0 = blockIdx.x / 20;
  int blk = blockIdx.x - 20 * slot0;
  int slot = sel ? sel[slot0] : slot0;
  const _Float16* xr = x + (long)(slot / 10) * K;
  for (int c = tid; c < K / 8; c += blockDim.x) *(float4*)(xs + 8 * c) = *(const float4*)(xr + 8 * c);
  __syncthreads();
  int q = tid & 3;
  int rr = blk * 64 + 8 * (tid >> 5) + ((tid >> 2) & 7);
  long row = (long)rowmap[slot] * N + rr;
  const unsigned char* wp = W + row * (K / 2) + 16 * q;
  const _Float16* sp = (const _Float16*)(W + soff + row * (NB * 2));
  float acc[NB];
  for (int m = 0; m < NB; m++) acc[m] = dot_block(*(const uint4*)(wp + 64 * m), xs + 128 * m + 32 * q);
  float tot = __builtin_fmaf((float)sp[0], acc[0], 0.0f);
  for (int m = 1; m < NB; m++) tot = __builtin_fmaf((float)sp[m], acc[m], tot);
  tot = tot + __shfl_xor(tot, 1);
  tot = tot + __shfl_xor(tot, 2);
  if (q == 0) y[(long)slot * N + rr] = tot;
  int jj = (blk % 10) >> 1;
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    unsigned old = atomicAdd(&ctr[slot * 5 + jj], 1u);
    last = (old == 3u);
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  if (tid >= 32) return;
  gl_epi(y + (long)slot * N, a, bq, out + (long)slot * 640, jj, tid);
  if (tid == 0) ctr[slot * 5 + jj] = 0u;
}

// ---- P1: dense HT job ----
// j = blockIdx - nGemv; rt = j / nsplit (output row tile of 128 rows), sp = j % nsplit (:1870955-1870997). nb = min(K/128, (sp+1)*chunk) - sp*chunk blocks
// (s39, :1870998-1871004). LDS activation xs[row][block][128 halves] with row stride 2560 halves (gl<2> :1875202: x' row 1 read at +5120 B; prelude row offset v_mul_lo_u32 0x1400, gl<1> :1871074).
// Prelude (warps, pairs p = warp, warp+8, ... < R*nb; row = p / nb, b = p % nb, lane covers elements 4*lane..+3 of 128-block (sp*chunk + b)):
//   gate/up bf16 b64 from ga/up at min(row, nrows-1)*K + el (:1871165-1871185), su f16 at el (:1871184-1871189, :1871050);
//   with up != null: y_k = bf16_rne(silu(g_k) * u_k) (v_mul 0xbfb8aa3b, v_exp, +1, IEEE div, v_mul by u, bf16 RNE via v_add3 0x7fff, :1871197-1871306),
//   else y_k = g_k; t0 = su0*y0, t1 = su1*y1, t2 = su2*y2 (v_fma_mix with -0.0), s01 = t0+t1, d01 = t0-t1, s23 = fma(y3, su3, t2), d23 = fma(-y3, su3, t2)
//   (:1871054-1871062); e0 = s01+s23, e1 = d01+d23, e2 = s01-s23, e3 = d01-d23; xor butterflies 1,2,4,8,16 (DPP / permlanex16, :1871064-1871130);
//   xs[el_local + k] = f16(fma(e_k, 0x3db504f3, 0)) via v_fma_mixlo/hi_f16 + ds_store_b64 (:1871133-1871140).
// GEMV: warp w handles column tiles [w*nb, w*nb + nb) of the split (3-bit counts s3 = (8*nb + 7) >> 3, :1871343-1871361); tile ct of row tile i is 128 B at
// wht + ((rt*(K/16) + sp*chunk*8 + ct)*8 + i)*128 (shl 10 per column tile, :1871351-1871368); lane decode as ht_dec4, acc_i = dot2acc over 4 pair dots (:1871492-1871821).
// Per-warp partial P[w][row][16*i + r] = acc_i[lane] + acc_i[lane^16] (v_permlanex16 + v_add, :1871879-1871896). Block sum of warps 0..7 starting from 0
// (:1871916-1871935). nsplit >= 2: write pf[(sp*R + row)*ld + rt*128 + t], fence, ticket cnt[rt]; last arrival (old == nsplit-1) re-sums all splits
// ascending from 0 (:1871937-1872026), thread 0 resets cnt[rt]. Epilogue (warp < R, row < nrows): 128-point WHT on the block totals (in-lane stages over
// i = bits 0,1,2, then xor 1,2,4,8; :1872053-1872165), * 0x3db504f3, * sv (f16, v_fma_mix with -0.0), bf16 RNE, out[row*ld + rt*128 + 16*i + r] (:1872214-1872257).
template <int R>
__device__ static inline void gl_p1(const I4rSd& sd, int nGemv) {
  constexpr int XROW = 2560;
  __shared__ _Float16 xs[R * XROW];
  __shared__ float part[8][R * 128];
  __shared__ float tot_s[R * 128];
  __shared__ int last;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int j = blockIdx.x - nGemv;
  const int rt = j / sd.nsplit, sp = j - rt * sd.nsplit;
  const int K = sd.K;
  int kb = (K + ((K >> 31) & 127)) >> 7;
  int hi = sp * sd.chunk + sd.chunk;
  const int nb = (kb < hi ? kb : hi) - sp * sd.chunk;
  const float C = __uint_as_float(0x3db504f3u);
  const int nrows = sd.nrows;
  // prelude
  for (int p = w; p < R * nb; p += 8) {
    int row = p / nb, b = p - row * nb;
    int el = (sp * sd.chunk + b) * 128 + 4 * lane;
    int ra = row < nrows - 1 ? row : nrows - 1;
    long off = (long)ra * K + el;
    float yv[4];
    for (int k = 0; k < 4; k++) {
      float g = bf2f(sd.ga[off + k]);
      if (sd.up) {
        float u = bf2f(sd.up[off + k]);
        float pr = silu_g(g) * u;
        yv[k] = bf2f(f2bf_rne(pr));
      } else {
        yv[k] = g;
      }
    }
    float s[4];
    for (int k = 0; k < 4; k++) s[k] = (float)sd.su[el + k];
    float t0 = __builtin_fmaf(s[0], yv[0], -0.0f), t1 = __builtin_fmaf(s[1], yv[1], -0.0f), t2 = __builtin_fmaf(s[2], yv[2], -0.0f);
    float s01 = t0 + t1, d01 = t0 - t1;
    float s23 = __builtin_fmaf(yv[3], s[3], t2), d23 = __builtin_fmaf(-yv[3], s[3], t2);
    float e[4] = {s01 + s23, d01 + d23, s01 - s23, d01 - d23};
    for (int m = 1; m <= 16; m <<= 1)
      for (int k = 0; k < 4; k++) e[k] = xstep(e[k], m);
    for (int k = 0; k < 4; k++) xs[row * XROW + b * 128 + 4 * lane + k] = (_Float16)__builtin_fmaf(e[k], C, 0.0f);
  }
  __syncthreads();
  // GEMV
  const int r16 = lane & 15, hh = lane >> 4;
  float acc[R][8];
  for (int rw = 0; rw < R; rw++) for (int i = 0; i < 8; i++) acc[rw][i] = 0.0f;
  const int nct = nb * 8;
  const int cpw = (nct + 7) >> 3;
  const int c0 = w * cpw;
  const int c1 = (c0 + cpw < nct) ? c0 + cpw : nct;
  for (int ct = c0; ct < c1; ct++) {
    const unsigned* tb = sd.wht + (((long)rt * (K / 16) + (long)sp * sd.chunk * 8 + ct) * 1024) / 4;
    _Float16 xv[R][8];
    for (int rw = 0; rw < R; rw++) for (int c = 0; c < 8; c++) xv[rw][c] = xs[rw * XROW + ct * 16 + 8 * hh + c];
    for (int i = 0; i < 8; i++) {
      _Float16 wt[8];
      ht_dec4(tb + i * 32, lane, wt);
      for (int rw = 0; rw < R; rw++) acc[rw][i] = dot8(wt, xv[rw], acc[rw][i]);
    }
  }
  __syncthreads();
  for (int rw = 0; rw < R; rw++)
    for (int i = 0; i < 8; i++) {
      float v = acc[rw][i];
      float o = __shfl_xor(v, 16);
      if (lane < 16) part[w][rw * 128 + 16 * i + r16] = v + o;
    }
  __syncthreads();
  if (tid < R * 128) {
    float t = 0.0f;
    for (int ww = 0; ww < 8; ww++) t = t + part[ww][tid];
    tot_s[tid] = t;
  }
  if (sd.nsplit >= 2) {
    __syncthreads();
    if (tid < R * 128) {
      int row = tid >> 7;
      sd.pf[(long)(sp * R + row) * sd.ld + rt * 128 + (tid & 127)] = tot_s[tid];
    }
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      unsigned old = atomicAdd(&sd.cnt[rt], 1u);
      last = (old == (unsigned)(sd.nsplit - 1));
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    if (tid < R * 128) {
      int row = tid >> 7;
      float t = 0.0f;
      for (int s2 = 0; s2 < sd.nsplit; s2++)
        t = t + ((const volatile float*)sd.pf)[(long)(s2 * R + row) * sd.ld + rt * 128 + (tid & 127)];
      tot_s[tid] = t;
    }
    __syncthreads();
    if (tid == 0) sd.cnt[rt] = 0u;
  }
  __syncthreads();
  if (w < R && w < nrows && tid < 64) {
    const int row = w;
    float X[8];
    for (int i = 0; i < 8; i++) X[i] = tot_s[row * 128 + 16 * i + r16];
    // in-lane stages over i bits 0,1,2
    for (int bit = 1; bit < 8; bit <<= 1) {
      float Y[8];
      for (int i = 0; i < 8; i++) {
        float p = X[i ^ bit];
        Y[i] = (i & bit) ? p - X[i] : X[i] + p;
      }
      for (int i = 0; i < 8; i++) X[i] = Y[i];
    }
    for (int m = 1; m <= 8; m <<= 1)
      for (int i = 0; i < 8; i++) {
        float partner = __shfl_xor(X[i], m);
        X[i] = ((lane & m) ? -X[i] : X[i]) + partner;
      }
    if (lane < 16) {
      for (int i = 0; i < 8; i++) {
        float svv = (float)sd.sv[rt * 128 + 16 * i + r16];
        float v = svv * (X[i] * C);
        sd.out[(long)row * sd.ld + rt * 128 + 16 * i + r16] = f2bf_rne(v);
      }
    }
  }
}

template <int R>
__device__ static inline void gl_body(const unsigned char* W, long soff, const _Float16* x, const int* rowmap, float* y, const _Float16* a,
                                      const _Float16* bq, _Float16* out, unsigned* ctr, const I4rSd& sd, int nGemv, const int* sel) {
  if ((int)blockIdx.x < nGemv) gl_p2(W, soff, x, rowmap, y, a, bq, out, ctr, sel);
  else gl_p1<R>(sd, nGemv);
}
extern "C" __global__ void __launch_bounds__(256) k_i4r_rows_gl_o1(const unsigned char* W, long soff, const _Float16* x, const int* rowmap, float* y,
                                                                   const _Float16* a, const _Float16* bq, _Float16* out, unsigned* ctr, I4rSd sd,
                                                                   int nGemv, const int* sel) {
  gl_body<1>(W, soff, x, rowmap, y, a, bq, out, ctr, sd, nGemv, sel);
}
extern "C" __global__ void __launch_bounds__(256) k_i4r_rows_gl_o2(const unsigned char* W, long soff, const _Float16* x, const int* rowmap, float* y,
                                                                   const _Float16* a, const _Float16* bq, _Float16* out, unsigned* ctr, I4rSd sd,
                                                                   int nGemv, const int* sel) {
  gl_body<2>(W, soff, x, rowmap, y, a, bq, out, ctr, sd, nGemv, sel);
}

// ---------------------------------------------------------------------------------------------------------------------------------
// k_i4r_rows_gl_g<R, G>(W@0, soff@8, x@16, items@24 (HtItem), ids@32, cnt@40, y@48, a@56, bq@64, out@72, ctr@80, I4rSd@88, nGemv@176).
// P1 is the gl<R> dense job with the sd fields at kernarg 0x58 (:1864202-1864207 for gl_g<1,2>); P2 is the item-grouped form of the gate/up job
// (listing gl_g<1,2> :1865527-1868798):
//   item = blockIdx / 20 (0x66666667 >> 3, :1865532-1865536), blk = blockIdx - 20*item (:1865728); exit if item >= cnt[0] (:1865538-1865541);
//   rec = items[item*5]: expert = rec.x, begin = rec.z, end = rec.w (:1865550-1865551); ns = min(end-begin, G) (:1865557); sid[tid] = ids[begin+tid] for
//   tid < G (:1865558-1865574, LDS +15360); n <= 0: nothing else (:1865584, :1865734). x rows: slot i's row sid[i]/10 (0x66666667 >> 2) staged to LDS
//   at i*0x1400 (:1865606-1865679). GEMV rows expert*1280 + blk*64 + 8*warp + ((tid>>2)&7) (:1865746), the weights decoded once and used for every
//   slot (scratch spills in the listing), y[sid[i]*1280 + r] by quad 0 (global_store_b32, 0x1400*sid :1868334-1868335).
//   Ticket: threads tid < ns atomicAdd(ctr + sid*5 + (blk%10)/2) (:1868361-1868371), flag = old == 3 (:1868373); warp w < ns with flag runs the
//   gu_epi item for slot sid[w] (:1868390-1868785) and lane 0 resets the ticket (:1868788-1868795).
template <int R, int G>
__device__ static inline void gl_g_p2(const unsigned char* W, long soff, const _Float16* x, const int4* items, const int* ids, const int* cnt,
                                     float* y, const _Float16* a, const _Float16* bq, _Float16* out, unsigned* ctr) {
  constexpr int K = 2560, NB = 20, N = 1280;
  __shared__ _Float16 xs[G][K];
  __shared__ int sid[G];
  __shared__ int flag[G];
  int tid = threadIdx.x;
  int item = blockIdx.x / 20;
  int blk = blockIdx.x - 20 * item;
  if (item >= cnt[0]) return;
  int4 rec = items[item * 5];
  int expert = rec.x, begin = rec.z, end = rec.w;
  int n = end - begin;
  int ns = n < G ? n : G;
  if (tid < G) sid[tid] = (tid < ns) ? ids[begin + tid] : 0;
  __syncthreads();
  if (n <= 0) return;
  for (int i = 0; i < ns; i++) {
    const _Float16* xr = x + (long)(sid[i] / 10) * K;
    for (int c = tid; c < K / 8; c += blockDim.x) *(float4*)(&xs[i][8 * c]) = *(const float4*)(xr + 8 * c);
  }
  __syncthreads();
  int q = tid & 3;
  int rr = blk * 64 + 8 * (tid >> 5) + ((tid >> 2) & 7);
  long row = (long)expert * N + rr;
  const unsigned char* wp = W + row * (K / 2) + 16 * q;
  const _Float16* sp = (const _Float16*)(W + soff + row * (NB * 2));
  uint4 wv[NB];
  for (int m = 0; m < NB; m++) wv[m] = *(const uint4*)(wp + 64 * m);
  for (int i = 0; i < ns; i++) {
    float acc[NB];
    for (int m = 0; m < NB; m++) acc[m] = dot_block(wv[m], &xs[i][128 * m + 32 * q]);
    float tot = __builtin_fmaf((float)sp[0], acc[0], 0.0f);
    for (int m = 1; m < NB; m++) tot = __builtin_fmaf((float)sp[m], acc[m], tot);
    tot = tot + __shfl_xor(tot, 1);
    tot = tot + __shfl_xor(tot, 2);
    if (q == 0) y[(long)sid[i] * N + rr] = tot;
  }
  int jj = (blk % 10) >> 1;
  __syncthreads();
  if (tid < ns) {
    __threadfence();
    unsigned old = atomicAdd(&ctr[sid[tid] * 5 + jj], 1u);
    flag[tid] = (old == 3u);
  }
  __syncthreads();
  int w = tid >> 5, lane = tid & 31;
  if (w < ns && flag[w]) {
    __threadfence();
    int slot = sid[w];
    gl_epi(y + (long)slot * N, a, bq, out + (long)slot * 640, jj, lane);
    if (lane == 0) ctr[slot * 5 + jj] = 0u;
  }
}

template <int R, int G>
__device__ static inline void gl_g_body(const unsigned char* W, long soff, const _Float16* x, const int4* items, const int* ids, const int* cnt,
                                        float* y, const _Float16* a, const _Float16* bq, _Float16* out, unsigned* ctr, const I4rSd& sd, int nGemv) {
  if ((int)blockIdx.x < nGemv) gl_g_p2<R, G>(W, soff, x, items, ids, cnt, y, a, bq, out, ctr);
  else gl_p1<R>(sd, nGemv);
}
#define GL_G(R_, G_)                                                                                                                         \
  extern "C" __global__ void __launch_bounds__(256) k_i4r_rows_gl_g##R_##G_(                                                                \
      const unsigned char* W, long soff, const _Float16* x, const int4* items, const int* ids, const int* cnt, float* y, const _Float16* a,  \
      const _Float16* bq, _Float16* out, unsigned* ctr, I4rSd sd, int nGemv) {                                                             \
    gl_g_body<R_, G_>(W, soff, x, items, ids, cnt, y, a, bq, out, ctr, sd, nGemv);                                                          \
  }
GL_G(2, 4) GL_G(1, 4) GL_G(2, 2) GL_G(1, 2)
extern "C" __global__ void __launch_bounds__(256) k_i4r_rows_gl1(const unsigned char* W, long soff, const _Float16* x, const int* rowmap, float* y,
                                                                 const _Float16* a, const _Float16* bq, _Float16* out, unsigned* ctr, I4rSd sd,
                                                                 int nGemv) {
  gl_body<1>(W, soff, x, rowmap, y, a, bq, out, ctr, sd, nGemv, nullptr);
}
extern "C" __global__ void __launch_bounds__(256) k_i4r_rows_gl2(const unsigned char* W, long soff, const _Float16* x, const int* rowmap, float* y,
                                                                 const _Float16* a, const _Float16* bq, _Float16* out, unsigned* ctr, I4rSd sd,
                                                                 int nGemv) {
  gl_body<2>(W, soff, x, rowmap, y, a, bq, out, ctr, sd, nGemv, nullptr);
}
