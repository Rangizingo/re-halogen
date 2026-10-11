// Rewrites of the "attn_qs" family: halogen::(anonymous)::k_attn_qs<A,B> and the qs_bt* variants (codeobj_1513000, flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o attn_qs.co attn_qs.hip
// Listing: kernels-disasm/codeobj_1513000.s (not in git). k_attn_qs<4,0> is lines 248233-249230 (work/kernels/..9k_attn_qsILi4ELi0E..s,
// line n of that slice = listing line 248232+n; hex addresses 0x175900-0x177018 are quoted as @addr).
#include <hip/hip_runtime.h>
// No mul+add fusion: every fma below is written out explicitly (the vendor uses v_fma/v_fmac where it fuses).
#pragma clang fp contract(off)

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
// bf16 RNE, no NaN special case: v_bfe_u32 x,v,16,1 ; v_add3_u32 v,v,x,0x7fff, high half kept (@176D18-176D28).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// expf(x) expansion (@176DB4-176E14 for the merge weights): t = x*log2e ; lo = fma(log2e,x,-t) ; r = rndne(t) ;
// f = (t - r) + fma(0x32a5705f, x, lo) ; e = ldexp(v_exp_f32(f), (int)r) ; x < -103.28 -> 0 ; x > 88.72 -> +inf.
__device__ static inline float expf_v(float x) {
  const float L2E = __uint_as_float(0x3fb8aa3bu);
  float t = x * L2E;
  float lo = __builtin_fmaf(L2E, x, -t);
  float r = __builtin_rintf(t);
  float f = t - r;
  lo = __builtin_fmaf(__uint_as_float(0x32a5705fu), x, lo);
  f = f + lo;
  float e = __builtin_amdgcn_ldexpf(__builtin_amdgcn_exp2f(f), (int)r);
  e = !(__uint_as_float(0xc2ce8ed0u) > x) ? e : 0.0f;
  e = !(__uint_as_float(0x42b17218u) < x) ? e : __uint_as_float(0x7f800000u);
  return e;
}
// expf(-g) expansion of the gate (@176BE0-176C98): t = g*-log2e ; lo = fma(-log2e,g,-t) ; r = rndne(t) ;
// f = (t - r) + fma(g, 0xb2a5705f, lo) ; ldexp(v_exp_f32(f), (int)r) ; g > 104.56 -> 0 ; g < -88.72 -> +inf.
__device__ static inline float expneg(float g) {
  const float NL2E = __uint_as_float(0xbfb8aa3bu);
  float t = g * NL2E;
  float lo = __builtin_fmaf(NL2E, g, -t);
  float r = __builtin_rintf(t);
  float f = t - r;
  lo = __builtin_fmaf(g, __uint_as_float(0xb2a5705fu), lo);
  f = f + lo;
  float e = __builtin_amdgcn_ldexpf(__builtin_amdgcn_exp2f(f), (int)r);
  e = !(__uint_as_float(0x42ce8ed0u) < g) ? e : 0.0f;
  e = !(__uint_as_float(0xc2b17218u) > g) ? e : __uint_as_float(0x7f800000u);
  return e;
}

#define NEG30 0xf149f2cau  // running-max init -1e30 (@175AE8)
#define NEG29 0xefa18f08u  // "m is valid" threshold -1e29 (@176DA4)

// Key index table (@175900-175AA0): len = sel[0] + bx + 1 ; bb = nblk[bx] ; nk = 4*bb + (len & 3) keys.
//   i < 4*bb      : 4*tbl[bx*tstride + (i>>2)] + (i&3)
//   i < nk        : i + (len & ~3) - 4*bb      (tail rows past the paged blocks)
//   nk <= i < nk+16: len - 1                   (padding, still read: P is 1.0 for all 16 keys of a tile in the probe kernels)
// Stored in LDS at 51200 + 4*i (@175A18), filled by all 128 threads in steps of 128 (@175A04-175A40).
struct KeyIdx {
  int len, bb, nk, tail;
};
__device__ static inline KeyIdx keyidx_setup(const int* sel, const int* nblk, int bx) {
  KeyIdx k;
  k.len = sel[0] + bx + 1;
  k.bb = nblk[bx];
  k.nk = (k.len & 3) + 4 * k.bb;
  k.tail = (k.len & ~3) - 4 * k.bb;
  return k;
}
__device__ static inline int keyidx_at(const KeyIdx& k, const int* tbl, int i) {
  if (i < 4 * k.bb) return tbl[i >> 2] * 4 + (i & 3);
  if (i < k.nk) return i + k.tail;
  return k.len - 1;
}

__shared__ int s_kidx[2304];
__shared__ float s_O[4][12][256];
__shared__ float s_ml[4][2][16];

// Wave merge + epilogue (@176908-177008). Each of the 4 waves has per-row running max m_w, sum l_w and O_w[row][d].
//   M = max(-1e30, m_0..m_3) ; w_i = (m_i > -1e29) ? expf(m_i - M) : 0
//   l = fma(w3,l3, fma(w2,l2, fma(w1,l1, fma(w0,l0,0)))) ; O likewise
//   out = bf16_rne( (O / l) / (1 + expneg(gate)) ) with IEEE divisions, gate at the same index as out (@176BD4-176D04)
__device__ static inline void qs_epilogue(unsigned short* O, const unsigned short* gate, int bx, int g) {
  for (int tid = threadIdx.x; tid < 3072; tid += 128) {
    int r = tid >> 8, d = tid & 255;
    float m[4], l[4], w[4];
    for (int i = 0; i < 4; i++) {
      m[i] = s_ml[i][0][r];
      l[i] = s_ml[i][1][r];
    }
    float M = fmaxf(fmaxf(fmaxf(m[0], __uint_as_float(NEG30)), m[1]), fmaxf(m[2], m[3]));
    for (int i = 0; i < 4; i++) w[i] = (m[i] > __uint_as_float(NEG29)) ? expf_v(m[i] - M) : 0.0f;
    float so = __builtin_fmaf(w[0], s_O[0][r][d], 0.0f);
    float sl = __builtin_fmaf(w[0], l[0], 0.0f);
    for (int i = 1; i < 4; i++) {
      so = __builtin_fmaf(w[i], s_O[i][r][d], so);
      sl = __builtin_fmaf(w[i], l[i], sl);
    }
    long oi = (long)bx * 6144 + (g * 12 + r) * 256 + d;
    float gv = bf2f(gate[oi]);
    float den = 1.0f + expneg(gv);
    float x = so / sl;
    O[oi] = f2bf_rne(x / den);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// k_attn_qs<4,0>: a timing-probe variant. S and softmax are skipped. Per wave w (tile t = w, w+4, ... < ceil(nk/16)):
//   P = 1.0 for every row/key (stores of 0x3f803f80, @1768DC-1768FC), alpha = 1.0 (@1768FC), m = 0 (v150 := 0 for lanes < 12, @175FBC),
//   l += float(min(16, nk - 16 t)) (@175FCC-175FF4),  O[c] = wmma(P, Vtile_c, O[c]) for the 16 d-chunks c (@176068-176390),
//   V row of key k = min(kidx[16 t + k], m_arg - 1) at v + row*stride + g*256 (@1763A8-176404, arg@60 = m_arg).
// Args: q@0 k@8 (unused here) v@16 stride@24 gate@32 o@40 sel@48 n@56(unused) m@60 tbl@64 nblk@72 tstride@80.
// Grid (tokens, 2 kv heads), block 128. Output o[bx*6144 + (g*12+head)*256 + d].
// ---------------------------------------------------------------------------------------------------------------
template <int A, int B>
__device__ static inline void qs_probe4(const unsigned short* Q, const unsigned short* K, const unsigned short* V, long stride,
                                      const unsigned short* gate, unsigned short* O, const int* sel, int n, int m,
                                      const int* tbl, const int* nblk, int tstride) {
  const int bx = blockIdx.x, g = blockIdx.y;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, col = lane & 15, half = lane >> 4;
  KeyIdx ki = keyidx_setup(sel, nblk, bx);
  const int* trow = tbl + (long)bx * tstride;
  for (int i = threadIdx.x; i < ki.nk + 16; i += 128) s_kidx[i] = keyidx_at(ki, trow, i);
  __syncthreads();
  const int ntiles = (ki.nk + 15) / 16;
  v8f Oacc[16];
  for (int c = 0; c < 16; c++)
    for (int i = 0; i < 8; i++) Oacc[c][i] = 0.0f;
  float mrun = __uint_as_float(NEG30), lrun = 0.0f;
  v16bf pa;
  for (int k = 0; k < 16; k++) pa[k] = __builtin_bit_cast(__bf16, (unsigned short)0x3f80);
  for (int t = w; t < ntiles; t += 4) {
    int rem = ki.nk - 16 * t;
    lrun = lrun + (float)(rem < 16 ? rem : 16);
    mrun = 0.0f;
    for (int c = 0; c < 16; c++) {
      v16bf vb;
      for (int k = 0; k < 16; k++) {
        int row = min(s_kidx[16 * t + k], m - 1);
        vb[k] = __builtin_bit_cast(__bf16, V[(long)row * stride + g * 256 + c * 16 + col]);
      }
      Oacc[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa, vb, Oacc[c]);
    }
  }
  for (int c = 0; c < 16; c++)
    for (int i = 0; i < 8; i++) {
      int r = 2 * i + half;
      if (r < 12) s_O[w][r][c * 16 + col] = Oacc[c][i];
    }
  if (lane < 12) {
    s_ml[w][0][lane] = mrun;
    s_ml[w][1][lane] = lrun;
  }
  __syncthreads();
  qs_epilogue(O, gate, bx, g);
}


__shared__ float s_sc[4][16][17];
__shared__ unsigned short s_pb[4][16][16];
__shared__ float s_al[4][16];

// ---------------------------------------------------------------------------------------------------------------
// k_attn_qs<0,0>: the full kernel (listing k_attn_qsILi0ELi0 at 0x17e800, slice line n = file line n+... ; cites below are @addr).
// Per wave w, tile t = w, w+4, ... < ceil(nk/16), 16 keys per tile, 12 q heads (rows 0..11 of the 16-row WMMA) of kv head g:
//   S = chain of 16 wmma(Qchunk_c, Kchunk_c) from 0 (@17f170-17f4e4 pattern, 16 chunks of 16 d), Q at q + bx*6144 + (g*12+row)*256,
//   K row of key k = min(kidx[16 t + k], m_arg - 1), K at k + row*stride + g*256 (@17f4f4-17f5c0)
//   s = S * 0.0625 (0x3d800000, @17f824), stored to LDS rows of stride 17 floats (@17fd28-17fd40)
//   cnt = min(nk - 16 t, 16) ; m_new = max(m_old, s_0..s_cnt-1) (@17fe08-17fe58) ; m_old init -1e30
//   p_j = expf(s_j - m_new) for j < cnt else 0 (@17fe78 ...) ; P_j = bf16_rne(p_j) ; l_tile = ((p_0+p_1)+p_2)+...+p_15 (@180aa4-180afc)
//   alpha = m_old > -1e29 ? expf(m_old - m_new) : 1.0 (@180a1c ... @17fd5c) ; l = fma(l_old, alpha, l_tile) (@180b04)
//   O_c = wmma(P, Vtile_c, O_c * alpha) (@17f0d4-17f4e4) ; end of tiles: m, l and O go through the wave merge in qs_epilogue.
// ---------------------------------------------------------------------------------------------------------------
// Probe bits of A (host env HALOGEN_ATTN_QS_PROBE, timing decomposition only): 1 = key row index masked with 0x1ff instead of
// clamped to m-1 (<1,0> @171514 v_and 0x1ff), 2 = V never staged to LDS (<2,0>/<3,0>, no ds_store_b16; output is stale LDS, BLOCKED),
// 4 = S and softmax skipped (qs_probe4), 8 = Q.K skipped, S stays zero (<8,0>: 16 wmma, 16+2 global loads, zero stores to the S tile).
#define QS_ROW(x) ((A & 1) ? ((x) & 0x1ff) : min((x), m - 1))
template <int A, int B>
__device__ static inline void qs_full(const unsigned short* Q, const unsigned short* K, const unsigned short* V, long stride,
                                      const unsigned short* gate, unsigned short* O, const int* sel, int n, int m,
                                      const int* tbl, const int* nblk, int tstride) {
  const int bx = blockIdx.x, g = blockIdx.y;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, col = lane & 15, half = lane >> 4;
  KeyIdx ki = keyidx_setup(sel, nblk, bx);
  const int* trow = tbl + (long)bx * tstride;
  for (int i = threadIdx.x; i < ki.nk + 16; i += 128) s_kidx[i] = keyidx_at(ki, trow, i);
  __syncthreads();
  const int ntiles = (ki.nk + 15) / 16;
  v8f Oacc[16];
  for (int c = 0; c < 16; c++)
    for (int i = 0; i < 8; i++) Oacc[c][i] = 0.0f;
  float mrun = __uint_as_float(NEG30), lrun = 0.0f;  // per row r = lane (lanes 0..11)
  const unsigned short* qrow = Q + (long)bx * 6144 + (long)(g * 12 + (col < 12 ? col : 0)) * 256;
  for (int t = w; t < ntiles; t += 4) {
    v8f S;
    for (int i = 0; i < 8; i++) S[i] = 0.0f;
    int krow = QS_ROW(s_kidx[16 * t + col]);
    const unsigned short* kp = K + (long)krow * stride + g * 256;
    if (!(A & 8))
      for (int kc = 0; kc < 16; kc++) {
        v16bf a = *(const v16bf*)(qrow + kc * 16);
        v16bf b = *(const v16bf*)(kp + kc * 16);
        S = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, S);
      }
    for (int i = 0; i < 8; i++) s_sc[w][2 * i + half][col] = S[i] * __uint_as_float(0x3d800000u);
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    int rem = ki.nk - 16 * t;
    if (lane < 12) {
      int cnt = rem < 16 ? rem : 16;
      float mo = mrun;
      float mn = mo;
      for (int j = 0; j < cnt; j++) mn = fmaxf(mn, s_sc[w][lane][j]);
      float pv[16];
      for (int j = 0; j < 16; j++) {
        pv[j] = 0.0f;
        if (j < rem) pv[j] = expf_v(s_sc[w][lane][j] - mn);
        s_pb[w][lane][j] = f2bf_rne(pv[j]);
      }
      float lt = pv[0] + pv[1];
      for (int j = 2; j < 16; j++) lt = lt + pv[j];
      float al = (mo > __uint_as_float(NEG29)) ? expf_v(mo - mn) : 1.0f;
      lrun = __builtin_fmaf(lrun, al, lt);
      mrun = mn;
      s_al[w][lane] = al;
    }
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    v16bf pa;
    for (int k = 0; k < 16; k++) pa[k] = __builtin_bit_cast(__bf16, s_pb[w][col][k]);
    for (int c = 0; c < 16; c++) {
      v16bf vb;
      for (int k = 0; k < 16; k++) {
        int row = QS_ROW(s_kidx[16 * t + k]);
        vb[k] = __builtin_bit_cast(__bf16, V[(long)row * stride + g * 256 + c * 16 + col]);
      }
      for (int i = 0; i < 8; i++) Oacc[c][i] = s_al[w][2 * i + half] * Oacc[c][i];
      Oacc[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa, vb, Oacc[c]);
    }
    __builtin_amdgcn_wave_barrier();
  }
  for (int c = 0; c < 16; c++)
    for (int i = 0; i < 8; i++) {
      int r = 2 * i + half;
      if (r < 12) s_O[w][r][c * 16 + col] = Oacc[c][i];
    }
  if (lane < 12) {
    s_ml[w][0][lane] = mrun;
    s_ml[w][1][lane] = lrun;
  }
  __syncthreads();
  qs_epilogue(O, gate, bx, g);
}

// ---------------------------------------------------------------------------------------------------------------
// k_attn_qs<0,1>: same math as <0,0> but with the softmax kept in the S accumulator registers (host env HALOGEN_ATTN_QS_RSM, 'register
// softmax'); listing 0x17bb00. Differences that change bits:
//   per row set i (row 2i+half) each lane holds key col : s = S[i]*0.0625 ; valid = col < nk - 16 t ; masked s = valid ? s : -1e30 (@17d0dc)
//   tile max by a 4-step xor butterfly (xmask 1,2,4,8, @17d0e8-17d134) ; m_new = max3(m_old, max) ; p = valid ? expf(s - m_new) : 0 (@17d14c-17d1b8)
//   alpha = m_old > -1e29 ? expf(m_old - m_new) : 0 (@17d1c8-17d24c : v175 starts 0, NOT 1.0)
//   the row sum is NOT updated per tile in a reduced form: each lane keeps l_lane = fma(l_lane, alpha, p) (@17c75c, @17c6fc-17c74c)
//   and the 16 lanes are summed once after the tile loop with the xor tree (1,2,4 then 8: add_f32_dpp @17dda8-17ddd0, @17dde4, @17de04)
// ---------------------------------------------------------------------------------------------------------------
template <int A, int B>
__device__ static inline void qs_rsm(const unsigned short* Q, const unsigned short* K, const unsigned short* V, long stride,
                                     const unsigned short* gate, unsigned short* O, const int* sel, int n, int m,
                                     const int* tbl, const int* nblk, int tstride) {
  const int bx = blockIdx.x, g = blockIdx.y;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, col = lane & 15, half = lane >> 4;
  KeyIdx ki = keyidx_setup(sel, nblk, bx);
  const int* trow = tbl + (long)bx * tstride;
  for (int i = threadIdx.x; i < ki.nk + 16; i += 128) s_kidx[i] = keyidx_at(ki, trow, i);
  __syncthreads();
  const int ntiles = (ki.nk + 15) / 16;
  v8f Oacc[16];
  for (int c = 0; c < 16; c++)
    for (int i = 0; i < 8; i++) Oacc[c][i] = 0.0f;
  float mrun[8], lrun[8];
  for (int i = 0; i < 8; i++) {
    mrun[i] = __uint_as_float(NEG30);
    lrun[i] = 0.0f;
  }
  const unsigned short* qrow = Q + (long)bx * 6144 + (long)(g * 12 + (col < 12 ? col : 0)) * 256;
  for (int t = w; t < ntiles; t += 4) {
    v8f S;
    for (int i = 0; i < 8; i++) S[i] = 0.0f;
    int krow = QS_ROW(s_kidx[16 * t + col]);
    const unsigned short* kp = K + (long)krow * stride + g * 256;
    for (int kc = 0; kc < 16; kc++) {
      v16bf a = *(const v16bf*)(qrow + kc * 16);
      v16bf b = *(const v16bf*)(kp + kc * 16);
      S = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, S);
    }
    const bool valid = col < ki.nk - 16 * t;
    float al[8];
    for (int i = 0; i < 8; i++) {
      float s = S[i] * __uint_as_float(0x3d800000u);
      float sm = valid ? s : __uint_as_float(NEG30);
      float mx = sm;
      mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
      mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
      mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
      mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
      float mo = mrun[i];
      float mn = fmaxf(fmaxf(mo, mx), mx);
      float p = valid ? expf_v(s - mn) : 0.0f;
      al[i] = (mo > __uint_as_float(NEG29)) ? expf_v(mo - mn) : 0.0f;
      lrun[i] = __builtin_fmaf(lrun[i], al[i], p);
      mrun[i] = mn;
      s_pb[w][2 * i + half][col] = f2bf_rne(p);
    }
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    v16bf pa;
    for (int k = 0; k < 16; k++) pa[k] = __builtin_bit_cast(__bf16, s_pb[w][col][k]);
    for (int c = 0; c < 16; c++) {
      v16bf vb;
      for (int k = 0; k < 16; k++) {
        int row = QS_ROW(s_kidx[16 * t + k]);
        vb[k] = __builtin_bit_cast(__bf16, V[(long)row * stride + g * 256 + c * 16 + col]);
      }
      for (int i = 0; i < 8; i++) Oacc[c][i] = al[i] * Oacc[c][i];
      Oacc[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa, vb, Oacc[c]);
    }
    __builtin_amdgcn_wave_barrier();
  }
  for (int c = 0; c < 16; c++)
    for (int i = 0; i < 8; i++) {
      int r = 2 * i + half;
      if (r < 12) s_O[w][r][c * 16 + col] = Oacc[c][i];
    }
  for (int i = 0; i < 8; i++) {
    float l = lrun[i];
    l = l + __shfl_xor(l, 1, 16);
    l = l + __shfl_xor(l, 2, 16);
    l = l + __shfl_xor(l, 4, 16);
    l = l + __shfl_xor(l, 8, 16);
    int r = 2 * i + half;
    if (col == 0 && r < 12) {
      s_ml[w][0][r] = mrun[i];
      s_ml[w][1][r] = l;
    }
  }
  __syncthreads();
  qs_epilogue(O, gate, bx, g);
}

template <int A, int B>
__device__ static inline void qs_body(const unsigned short* Q, const unsigned short* K, const unsigned short* V, long stride,
                                      const unsigned short* gate, unsigned short* O, const int* sel, int n, int m,
                                      const int* tbl, const int* nblk, int tstride) {
  if (A == 4) qs_probe4<A, B>(Q, K, V, stride, gate, O, sel, n, m, tbl, nblk, tstride);
  else if (B == 1) qs_rsm<A, B>(Q, K, V, stride, gate, O, sel, n, m, tbl, nblk, tstride);
  else qs_full<A, B>(Q, K, V, stride, gate, O, sel, n, m, tbl, nblk, tstride);
}

#define QS(A, B)                                                                                                    \
  extern "C" __global__ void k_attn_qs_##A##_##B(                                                                    \
      const unsigned short* Q, const unsigned short* K, const unsigned short* V, long stride, const unsigned short* gate, \
      unsigned short* O, const int* sel, int n, int m, const int* tbl, const int* nblk, int tstride) {              \
    qs_body<A, B>(Q, K, V, stride, gate, O, sel, n, m, tbl, nblk, tstride);                                         \
  }
QS(4, 0)
QS(0, 0)
QS(1, 0)
QS(8, 0)
QS(0, 1)

// ===============================================================================================================
// Paged variants (k_attn_qs_bt*): Q, K pool, V pool, gate, O, sel, int (unused), tbl, nblk, tstride.
//   args: q@0 kpool@8 vpool@16 gate@24 o@32 sel@40 n@48(unused) tbl@56 nblk@64 tstride@72
// Pool layouts come from k_kv_bt (rewrite/kernels/kv.hip): page p (4 rows) of kv head g
//   K: kp[p*2048 + g*1024 + (d>>4)*64 + r*16 + (d&15)]   (16-column chunks of 4 rows x 16)
//   V: vp[p*2048 + g*1024 + 4*d + r]                      ([column][row])
// Page list L[i] (i-th group of 4 keys): i < bb ? tbl[bx*tstride + i] : len/4  (@1b2a40-1b2ab4 for bt2<0,0>), nk = 4*bb + (len&3).
// ===============================================================================================================
__shared__ int s_pg[1024];

__device__ static inline int pg_at(int len, int bb, const int* trow, int i) { return i < bb ? trow[i] : (len >> 2); }

// ---------------------------------------------------------------------------------------------------------------
// k_attn_qs_bt2<0,0> (listing codeobj_1513000 0x1b2900; block 64 = 2 waves, wave w owns d in [128w, 128w+128)):
//   Q for the 12 heads of kv head g: q + bx*6144 + g*3072 + head*256 (staged to LDS, @1b2ac0-1b2b94)
//   per 16-key tile t (4 pages): S_w = chain of 8 wmma(Qchunk, Kchunk) from 0 over this wave's 8 d-chunks (@1b3450-1b34b4),
//   both partials exchanged through LDS, S = S_own + S_other (@1b34f8), s = S * 0.0625 (@1b350c) ;
//   then the register softmax of qs<0,1> (butterfly max, p, alpha = 0 when m_old invalid, l_lane = fma(l_lane, alpha, p)),
//   P as bf16 to LDS, O_c = alpha*O_c ; O_c = wmma(P, Vchunk_c, O_c) over the wave's 8 chunks (@1b3288-1b32dc).
//   end: l = xor-tree over the 16 lanes, inv = 1.0f/l, out = bf16( (O*inv) / (1 + expneg(gate)) ) (@1b44d8-1b4900).
// ---------------------------------------------------------------------------------------------------------------
__shared__ float s_s2[4][8][32];
__shared__ unsigned short s_pb2[4][16][16];

template <int A, int B, int NW, int SM = 0>
__device__ static inline void qs_bt_body(const unsigned short* Q, const unsigned short* Kp, const unsigned short* Vp,
                                          const unsigned short* gate, unsigned short* O, const int* sel, int n, const int* tbl,
                                          const int* nblk, int tstride) {
  const int bx = blockIdx.x, g = blockIdx.y;
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, col = lane & 15, half = lane >> 4;
  const int len = sel[0] + bx + 1;
  const int bb = nblk[bx];
  const int nk = (len & 3) + 4 * bb;
  const int* trow = tbl + (long)bx * tstride;
  const int ngrp = (nk + 3) / 4;
  for (int i = threadIdx.x; i < ((ngrp + 3) & ~3) + 4; i += 32 * NW) s_pg[i] = pg_at(len, bb, trow, i);
  __syncthreads();
  const int ntiles = (nk + 15) / 16;
  constexpr int CH = 16 / NW, DW = 256 / NW;
  v8f Oacc[CH];
  for (int c = 0; c < CH; c++)
    for (int i = 0; i < 8; i++) Oacc[c][i] = 0.0f;
  float mrun[8], lrun[8];
  for (int i = 0; i < 8; i++) {
    mrun[i] = __uint_as_float(NEG30);
    lrun[i] = 0.0f;
  }
  const unsigned short* qrow = Q + (long)bx * 6144 + (long)g * 3072 + (long)(col < 12 ? col : 0) * 256 + w * DW;
  for (int t = 0; t < ntiles; t++) {
    v8f S;
    for (int i = 0; i < 8; i++) S[i] = 0.0f;
    const int kpage = s_pg[4 * t + (col >> 2)], kr = col & 3;
    const unsigned short* kp = Kp + (long)kpage * 2048 + g * 1024 + kr * 16;
    for (int c = 0; c < CH; c++) {
      v16bf a = *(const v16bf*)(qrow + c * 16);
      v16bf b = *(const v16bf*)(kp + (CH * w + c) * 64);
      // probe 1 (bt2<1,0> @1932b8-193360 loads, @193a48 wmma v[137:144], A=v[145:152] (LDS Q), B=v[65:72]): the K fragments are loaded once
      // before the loop from pool pages 0..3 and reused for every tile; no K load inside the loop.
      if (A & 1) b = *(const v16bf*)(Kp + (long)(col >> 2) * 2048 + g * 1024 + (col & 3) * 16 + (CH * w + c) * 64);
      // probe 32 (bt2<32,0>, v2 listing @1a9f70-1aa0e0 region: s0 = tile counter, v_pk_add_u16 / v_and 0x70007 / v_or 0x3f803f80 and the v_mov_b16 shuffle):
      // no K loads; chunk c's B element k is the bf16 0x3f80 | ((t + c + (k&7)) & 7) in every lane (windows v[117:124],v[127:134],v[118:125],...).
      if (A & 32)
        for (int k = 0; k < 16; k++) b[k] = __builtin_bit_cast(__bf16, (unsigned short)(0x3f80 | ((t + c + (k & 7)) & 7)));
      S = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, S);
    }
    for (int i = 0; i < 8; i++) s_s2[w][i][lane] = S[i];
    // SM 1: every wave takes the full 16-chunk chain from zero; SM 2: two 8-chunk chains from zero, added (bt4x candidates).
    v8f Sf;
    for (int i = 0; i < 8; i++) Sf[i] = 0.0f;
    if (SM == 1) {
      const unsigned short* qfull = Q + (long)bx * 6144 + (long)g * 3072 + (long)(col < 12 ? col : 0) * 256;
      for (int c = 0; c < 16; c++) {
        v16bf a = *(const v16bf*)(qfull + c * 16);
        v16bf b = *(const v16bf*)(kp + c * 64);
        Sf = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, Sf);
      }
    }
    if (SM == 2) {
      const unsigned short* qfull = Q + (long)bx * 6144 + (long)g * 3072 + (long)(col < 12 ? col : 0) * 256;
      v8f S2;
      for (int i = 0; i < 8; i++) S2[i] = 0.0f;
      for (int c = 0; c < 8; c++) {
        v16bf a = *(const v16bf*)(qfull + c * 16);
        v16bf b = *(const v16bf*)(kp + c * 64);
        Sf = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, Sf);
      }
      for (int c = 8; c < 16; c++) {
        v16bf a = *(const v16bf*)(qfull + c * 16);
        v16bf b = *(const v16bf*)(kp + c * 64);
        S2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, S2);
      }
      for (int i = 0; i < 8; i++) Sf[i] = Sf[i] + S2[i];
    }
    __syncthreads();
    const bool valid = col < nk - 16 * t;
    float al[8];
    for (int i = 0; i < 8; i++) {
      float st = s_s2[0][i][lane];
      for (int j = 1; j < NW; j++) st = st + s_s2[j][i][lane];
      if (SM != 0) st = Sf[i];
      if (A & 8) st = S[i];  // probe 8: the cross-wave S exchange is skipped, each wave softmaxes its own partial (@19c7ac region, no ds_store_2addr)
      float s = st * __uint_as_float(0x3d800000u);
      float sm = valid ? s : __uint_as_float(NEG30);
      float mx = sm;
      mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
      mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
      mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
      mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
      float p;
      if (A & 4) {  // probe 4 (bt2<4,0> @19cdfc-19ce58): softmax skipped; p = valid ? S*0.0625 : 0, l += p, no alpha, no max
        p = valid ? s : 0.0f;
        al[i] = 1.0f;
        lrun[i] = lrun[i] + p;
      } else {
        float mo = mrun[i];
        float mn = fmaxf(fmaxf(mo, mx), mx);
        p = valid ? expf_v(s - mn) : 0.0f;
        al[i] = (mo > __uint_as_float(NEG29)) ? expf_v(mo - mn) : 0.0f;
        lrun[i] = __builtin_fmaf(lrun[i], al[i], p);
        mrun[i] = mn;
      }
      s_pb2[w][2 * i + half][col] = f2bf_rne(p);
    }
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    v16bf pa;
    for (int k = 0; k < 16; k++) pa[k] = __builtin_bit_cast(__bf16, s_pb2[w][col][k]);
    for (int c = 0; c < CH; c++) {
      const int dcol = DW * w + 16 * c + col;
      v16bf vb;
      for (int k = 0; k < 16; k++) {
        int vpg = s_pg[4 * t + (k >> 2)];
        // probe 2 (bt2<2,0> @197bf0-198114: v_add_co_u32 v65-v67 = 0x200/0x400/0x600 + (256g+col), then 32 global_load_b64 before the loop;
        // pointers Vp + 8*(((256g+col)|128w|16c|X)), X = 0,0x200,0x400,0x600): V fragments are loaded once from pool pages 0..3
        // (page j = k>>2, 4 rows r = k&3 per b64) and reused by every tile; the page list is not used for V.
        if (A & 2) vpg = k >> 2;
        vb[k] = __builtin_bit_cast(__bf16, Vp[(long)vpg * 2048 + g * 1024 + 4 * dcol + (k & 3)]);
      }
      for (int i = 0; i < 8; i++) Oacc[c][i] = al[i] * Oacc[c][i];
      Oacc[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa, vb, Oacc[c]);
    }
    __syncthreads();
  }
  for (int i = 0; i < 8; i++) {
    float l = lrun[i];
    l = l + __shfl_xor(l, 1, 16);
    l = l + __shfl_xor(l, 2, 16);
    l = l + __shfl_xor(l, 4, 16);
    l = l + __shfl_xor(l, 8, 16);
    const int r = 2 * i + half;
    const float inv = 1.0f / l;
    if (r < 12) {
      for (int c = 0; c < CH; c++) {
        const int d = DW * w + 16 * c + col;
        const long oi = (long)bx * 6144 + (g * 12 + r) * 256 + d;
        float x = Oacc[c][i] * inv;
        float gv = bf2f(gate[oi]);
        float den = 1.0f + expneg(gv);
        O[oi] = f2bf_rne(x / den);
      }
    }
  }
}

#define QSBT2(A, B)                                                                                                  \
  extern "C" __global__ void k_attn_qs_bt2_##A##_##B(                                                                \
      const unsigned short* Q, const unsigned short* Kp, const unsigned short* Vp, const unsigned short* gate,       \
      unsigned short* O, const int* sel, int n, const int* tbl, const int* nblk, int tstride) {                      \
    qs_bt_body<A, B, 2>(Q, Kp, Vp, gate, O, sel, n, tbl, nblk, tstride);                                              \
  }
QSBT2(0, 0)
QSBT2(1, 0)
QSBT2(2, 0)
QSBT2(32, 0)
QSBT2(64, 0)
QSBT2(4, 0)
QSBT2(8, 0)
QSBT2(16, 0)
QSBT2(0, 1)
QSBT2(0, 2)
QSBT2(0, 3)
QSBT2(0, 7)
QSBT2(0, 9)
QSBT2(0, 11)

// k_attn_qs_bt2w<OPT,WPE> (0x18a100 for <7,7>): same arithmetic as bt2<0,0>; the template args only change code generation
// (LDS 9216 vs 17616, page list built after the Q staging, waves-per-EU attribute from HALOGEN_ATTN_QS_BT2_WPE). Checked by MATCH rows.
#define QSBT2W(A, B)                                                                                                 \
  extern "C" __global__ void k_attn_qs_bt2w_##A##_##B(                                                               \
      const unsigned short* Q, const unsigned short* Kp, const unsigned short* Vp, const unsigned short* gate,       \
      unsigned short* O, const int* sel, int n, const int* tbl, const int* nblk, int tstride) {                      \
    qs_bt_body<0, 0, 2>(Q, Kp, Vp, gate, O, sel, n, tbl, nblk, tstride);                                              \
  }
QSBT2W(3, 6)
QSBT2W(7, 6)
QSBT2W(7, 7)
QSBT2W(7, 8)

// k_attn_qs_bt4<OPT> (0x1d2000): four waves, each owns a quarter of d (64 = 4 chunks): S partial = chain of 4 wmma from 0 (@1d24e8-1d2534),
// partials exchanged through LDS and summed in wave order ((p0+p1)+p2)+p3 (@1d25a8-1d2690), then 0.0625 (@1d25d8). Softmax, P, O update and
// epilogue are the same arithmetic as bt2 (the listing spreads rows 4w..4w+3 of the tile over the waves; per element the math is identical).
#define QSBT4(A)                                                                                                     \
  extern "C" __global__ void k_attn_qs_bt4_##A(                                                                      \
      const unsigned short* Q, const unsigned short* Kp, const unsigned short* Vp, const unsigned short* gate,       \
      unsigned short* O, const int* sel, int n, const int* tbl, const int* nblk, int tstride) {                      \
    qs_bt_body<0, 0, 4>(Q, Kp, Vp, gate, O, sel, n, tbl, nblk, tstride);                                            \
  }
QSBT4(0)
QSBT4(1)
QSBT4(3)
QSBT4(5)

// k_attn_qs_bt4x<OPT> (0x1db080): the bt4 arithmetic, software pipelined (S of tile t+1 is issued ahead of the O update of tile t), with a different
// S reduction: a pair of waves chains its 8 chunks through LDS (second wave loads the first wave's partial as the wmma C input, @1dbc34-1dbc84
// ds_load_2addr C then 4 wmma), the two pairs are added once (@1db630-1db644 v_add_f32 of 2 values). Net: S = chain8(chunks 0-7) + chain8(chunks 8-15),
// found by MATCH rows (SM=2); a plain 16-chunk chain (SM=1) and bt4's sequential sum of four 4-chunk chains both leave 2-19 of 18432 outputs wrong.
#ifndef SMX
#define SMX 2
#endif
#define QSBT4X(A)                                                                                                    \
  extern "C" __global__ void k_attn_qs_bt4x_##A(                                                                     \
      const unsigned short* Q, const unsigned short* Kp, const unsigned short* Vp, const unsigned short* gate,       \
      unsigned short* O, const int* sel, int n, const int* tbl, const int* nblk, int tstride) {                      \
    qs_bt_body<0, 0, 4, SMX>(Q, Kp, Vp, gate, O, sel, n, tbl, nblk, tstride);                                            \
  }
QSBT4X(0)
QSBT4X(1)

// ---------------------------------------------------------------------------------------------------------------
// k_attn_qs_bt (listing codeobj_1513000 0x25e00, block 128): one WAVE per token. Token tk = blockIdx.x*4 + wave, skipped when tk >= n (arg@48,
// @25e00-25e2c); kv head g = blockIdx.y. No barrier in the listing (every wave is independent). Same paged key list and register softmax as
// bt2/bt4, but the wave owns all of d: S = chain of 16 wmma from 0 (SM=1) or two 8-chunk chains added (SM=2, tried second), O has 16 chunks.
// ---------------------------------------------------------------------------------------------------------------
__shared__ int s_pgw[4][512];
__shared__ unsigned short s_pbw[4][16][16];

#ifndef SMBT
#define SMBT 1
#endif
__device__ static inline void qs_bt_wave(const unsigned short* Q, const unsigned short* Kp, const unsigned short* Vp,
                                         const unsigned short* gate, unsigned short* O, const int* sel, int n, const int* tbl,
                                         const int* nblk, int tstride) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, col = lane & 15, half = lane >> 4;
  const int tk = blockIdx.x * 4 + w, g = blockIdx.y;
  if (tk >= n) return;
  const int len = sel[0] + tk + 1;
  const int bb = nblk[tk];
  const int nk = (len & 3) + 4 * bb;
  const int* trow = tbl + (long)tk * tstride;
  const int ngrp = (nk + 3) / 4;
  for (int i = lane; i < ((ngrp + 3) & ~3) + 4; i += 32) s_pgw[w][i] = pg_at(len, bb, trow, i);
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
  const int ntiles = (nk + 15) / 16;
  v8f Oacc[16];
  for (int c = 0; c < 16; c++)
    for (int i = 0; i < 8; i++) Oacc[c][i] = 0.0f;
  float mrun[8], lrun[8];
  for (int i = 0; i < 8; i++) {
    mrun[i] = __uint_as_float(NEG30);
    lrun[i] = 0.0f;
  }
  const unsigned short* qrow = Q + (long)tk * 6144 + (long)g * 3072 + (long)(col < 12 ? col : 0) * 256;
  for (int t = 0; t < ntiles; t++) {
    const int kpage = s_pgw[w][4 * t + (col >> 2)], kr = col & 3;
    const unsigned short* kp = Kp + (long)kpage * 2048 + g * 1024 + kr * 16;
    v8f S, S2;
    for (int i = 0; i < 8; i++) {
      S[i] = 0.0f;
      S2[i] = 0.0f;
    }
    if (SMBT == 1) {
      for (int c = 0; c < 16; c++) {
        v16bf a = *(const v16bf*)(qrow + c * 16);
        v16bf b = *(const v16bf*)(kp + c * 64);
        S = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, S);
      }
    } else {
      for (int c = 0; c < 8; c++) {
        v16bf a = *(const v16bf*)(qrow + c * 16);
        v16bf b = *(const v16bf*)(kp + c * 64);
        S = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, S);
      }
      for (int c = 8; c < 16; c++) {
        v16bf a = *(const v16bf*)(qrow + c * 16);
        v16bf b = *(const v16bf*)(kp + c * 64);
        S2 = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, S2);
      }
      for (int i = 0; i < 8; i++) S[i] = S[i] + S2[i];
    }
    const bool valid = col < nk - 16 * t;
    float al[8];
    for (int i = 0; i < 8; i++) {
      float s = S[i] * __uint_as_float(0x3d800000u);
      float sm = valid ? s : __uint_as_float(NEG30);
      float mx = sm;
      mx = fmaxf(mx, __shfl_xor(mx, 1, 16));
      mx = fmaxf(mx, __shfl_xor(mx, 2, 16));
      mx = fmaxf(mx, __shfl_xor(mx, 4, 16));
      mx = fmaxf(mx, __shfl_xor(mx, 8, 16));
      float mo = mrun[i];
      float mn = fmaxf(fmaxf(mo, mx), mx);
      float p = valid ? expf_v(s - mn) : 0.0f;
      al[i] = (mo > __uint_as_float(NEG29)) ? expf_v(mo - mn) : 0.0f;
      lrun[i] = __builtin_fmaf(lrun[i], al[i], p);
      mrun[i] = mn;
      s_pbw[w][2 * i + half][col] = f2bf_rne(p);
    }
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    v16bf pa;
    for (int k = 0; k < 16; k++) pa[k] = __builtin_bit_cast(__bf16, s_pbw[w][col][k]);
    for (int c = 0; c < 16; c++) {
      const int dcol = 16 * c + col;
      v16bf vb;
      for (int k = 0; k < 16; k++) {
        int vpg = s_pgw[w][4 * t + (k >> 2)];
        vb[k] = __builtin_bit_cast(__bf16, Vp[(long)vpg * 2048 + g * 1024 + 4 * dcol + (k & 3)]);
      }
      for (int i = 0; i < 8; i++) Oacc[c][i] = al[i] * Oacc[c][i];
      Oacc[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa, vb, Oacc[c]);
    }
    __builtin_amdgcn_wave_barrier();
  }
  for (int i = 0; i < 8; i++) {
    float l = lrun[i];
    l = l + __shfl_xor(l, 1, 16);
    l = l + __shfl_xor(l, 2, 16);
    l = l + __shfl_xor(l, 4, 16);
    l = l + __shfl_xor(l, 8, 16);
    const int r = 2 * i + half;
    const float inv = 1.0f / l;
    if (r < 12) {
      for (int c = 0; c < 16; c++) {
        const int d = 16 * c + col;
        const long oi = (long)tk * 6144 + (g * 12 + r) * 256 + d;
        float x = Oacc[c][i] * inv;
        float gv = bf2f(gate[oi]);
        float den = 1.0f + expneg(gv);
        O[oi] = f2bf_rne(x / den);
      }
    }
  }
}

extern "C" __global__ void k_attn_qs_bt(const unsigned short* Q, const unsigned short* Kp, const unsigned short* Vp,
                                        const unsigned short* gate, unsigned short* O, const int* sel, int n, const int* tbl,
                                        const int* nblk, int tstride) {
  qs_bt_wave(Q, Kp, Vp, gate, O, sel, n, tbl, nblk, tstride);
}
