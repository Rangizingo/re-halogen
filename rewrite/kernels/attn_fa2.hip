// Rewrites of the "attn_fa2" family: halogen::(anonymous)::k_attn_fa2<A,B,C,OPT,X,Y> (codeobj_1513000, flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o attn_fa2.co attn_fa2.hip
// Listing: kernels-disasm/codeobj_1513000.s (not in git); the default decode/prefill instance <2,1,true,5,1,true> is lines
// 130254-139054. Cites are file lines of that listing.
#include <hip/hip_runtime.h>
// No mul+add fusion: every fma below is written out explicitly (the vendor uses v_fma/v_fmac where it fuses).
#pragma clang fp contract(off)

typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
// bf16 RNE, no NaN special case: v_bfe_u32 x,v,16,1 ; v_add3_u32 v,v,x,0x7fff, high half kept (:131575-131577).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// 2^x the way the kernel does it (:131563-131572): if x < -126 (0xc2fc0000) add 64 first, v_exp_f32, then v_ldexp_f32(-64).
__device__ static inline float exp2s(float x) {
  bool small = x < __uint_as_float(0xc2fc0000u);
  float y = small ? x + __uint_as_float(0x42800000u) : x;
  float e = __builtin_amdgcn_exp2f(y);
  return small ? __builtin_amdgcn_ldexpf(e, -64) : e;
}

// expf(x) as expanded in the X=0 instances (listing :211503-211520 for <2,1,true,0,0,false>): t = x*log2e ; lo = fma(log2e, x, -t) ;
// r = rndne(t) ; f = (t - r) + fma(0x32a5705f, x, lo) ; e = ldexp(v_exp_f32(f), (int)r) ; x < -103.28 -> 0 ; x > 88.72 -> +inf.
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

// expf(-g) as expanded in the epilogue (:132395-132412): t = g * -log2e ; lo = fma(-log2e, g, -t) ; r = rndne(t) ;
// f = (t - r) + fma(g, -1.9e-8 (0xb2a5705f), lo) ; e = ldexp(v_exp_f32(f), (int)r) ; g > 104.56 -> 0 ; g < -88.72 -> +inf.
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

struct FaSel {
  const unsigned* tilemask;  // +0x00, null in dense mode
  const int* ids;            // +0x08
  const int* cnt;            // +0x10
  unsigned words;            // +0x18
  unsigned stride;           // +0x1c
};

// One wave handles one 16-row query tile of one q head:
//   S = sum over 16 k-chunks of wmma(Q[:, 16kc..], K^T) (starting from 0, :130824-130929), s = S * 0x3db8aa3b (= 0.0625*log2e, :130931)
//   masked (key > row + sel[0], or key >= m) -> -inf
//   per row: m_new = max(m_old, s_0..s_15) ; p_j = exp2s(s_j - m_new) ; P_j = bf16_rne(p_j) ; l_tile = ((p_0+p_1)+p_2)+... ;
//   alpha = exp2s(m_old - m_new) if m_old > -1e29 else 0 (:132214-132229) ; l = fma(alpha, l_old, l_tile) ; O *= alpha (:132278-132341)
//   O += wmma(P, V tile) per 16-wide d chunk (:130660-130720), C input = alpha * O.
//   end: out = bf16_rne( (O * (1.0f / l)) / (1.0f + expneg(gate)) ) (:132359-132434), gate at row*12288 + head*512 + 256 + d.
__shared__ float sc[24][16][17];
__shared__ unsigned short pb[24][16][16];
__shared__ float mrow[24][16], lrow[24][16], arow[24][16];

template <int E, int MM = 0>
__device__ static inline void fa_core(const unsigned short* Q, const unsigned short* K, const unsigned short* V, long stride,
                                      const unsigned short* gate, unsigned short* O, int base, int n, int m, int g, int head,
                                      int row0, int w) {
  const int lane = threadIdx.x & 31;
  const int col = lane & 15, half = lane >> 4;
  // E = 1 (the X=1 instances): s = S * (0.0625*log2e) and exp2s ; E = 0: s = S * 0.0625 and expf_v (:210860, :211503).
  const float SC = E ? __uint_as_float(0x3db8aa3bu) : __uint_as_float(0x3d800000u);
  const float NEG_INF = __uint_as_float(0xff800000u);

  if (lane < 16) {
    mrow[w][lane] = __uint_as_float(0xf149f2cau);
    lrow[w][lane] = 0.0f;
  }
  v8f Oacc[16];
  for (int c = 0; c < 16; c++)
    for (int i = 0; i < 8; i++) Oacc[c][i] = 0.0f;
  const unsigned short* qrow = Q + (long)(row0 + col) * 6144 + head * 256;

  int lastpos = row0 + 15 + base;
  int ntiles = ((MM ? min(lastpos + 1, m) : lastpos + 1) + 15) / 16;
  for (int t = 0; t < ntiles; t++) {
    v8f S;
    for (int i = 0; i < 8; i++) S[i] = 0.0f;
    int kidx = min(16 * t + col, m - 1);
    const unsigned short* krow = K + (long)kidx * stride + g * 256;
    for (int kc = 0; kc < 16; kc++) {
      v16bf a = *(const v16bf*)(qrow + kc * 16);
      v16bf b = *(const v16bf*)(krow + kc * 16);
      S = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, S);
    }
    for (int i = 0; i < 8; i++) {
      int r = 2 * i + half;
      int key = 16 * t + col;
      float s = S[i] * SC;
      bool ok = (key <= row0 + r + base) && (!MM || key < m);
      sc[w][r][col] = ok ? s : NEG_INF;
    }
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    if (lane < 16) {
      float sv[16];
      for (int j = 0; j < 16; j++) sv[j] = sc[w][lane][j];
      float mo = mrow[w][lane];
      float mn = fmaxf(fmaxf(mo, sv[0]), sv[1]);
      for (int j = 2; j < 16; j += 2) mn = fmaxf(fmaxf(mn, sv[j]), sv[j + 1]);
      float lt = 0.0f;
      float pv[16];
      for (int j = 0; j < 16; j++) {
        pv[j] = E ? exp2s(sv[j] - mn) : expf_v(sv[j] - mn);
        pb[w][lane][j] = f2bf_rne(pv[j]);
      }
      lt = pv[0] + pv[1];
      for (int j = 2; j < 16; j++) lt = lt + pv[j];
      float al = (mo > __uint_as_float(0xefa18f08u)) ? (E ? exp2s(mo - mn) : expf_v(mo - mn)) : 0.0f;
      lrow[w][lane] = __builtin_fmaf(al, lrow[w][lane], lt);
      mrow[w][lane] = mn;
      arow[w][lane] = al;
    }
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_SEQ_CST, "wavefront");
    v16bf pa;
    for (int k = 0; k < 16; k++) pa[k] = __builtin_bit_cast(__bf16, pb[w][col][k]);
    for (int c = 0; c < 16; c++) {
      for (int i = 0; i < 8; i++) Oacc[c][i] = Oacc[c][i] * arow[w][2 * i + half];
      v16bf vb;
      for (int k = 0; k < 16; k++) {
        int vk = min(16 * t + k, m - 1);
        vb[k] = __builtin_bit_cast(__bf16, V[(long)vk * stride + g * 256 + c * 16 + col]);
      }
      Oacc[c] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pa, vb, Oacc[c]);
    }
  }
  __builtin_amdgcn_wave_barrier();
  for (int i = 0; i < 8; i++) {
    int r = 2 * i + half;
    int grow = row0 + r;
    float inv = 1.0f / lrow[w][r];
    if (grow < n) {
      for (int c = 0; c < 16; c++) {
        int d = c * 16 + col;
        float x = Oacc[c][i] * inv;
        float gv = bf2f(gate[(long)grow * 12288 + head * 512 + 256 + d]);
        float den = 1.0f + expneg(gv);
        O[(long)grow * 6144 + head * 256 + d] = f2bf_rne(x / den);
      }
    }
  }
}

template <int NSUB, int E>
__device__ static inline void fa2_body(const unsigned short* Q, const unsigned short* K, const unsigned short* V, long stride,
                                       const unsigned short* gate, unsigned short* O, const int* sel, int n, int m, int g,
                                       int tile) {
  const int w = threadIdx.x >> 5;
  const int sub = w / 12, hig = w % 12;
  if (sub >= NSUB) return;
  fa_core<E>(Q, K, V, stride, gate, O, sel[0], n, m, g, g * 12 + hig, (tile * NSUB + sub) * 16, w);
}

// Launch shapes (host, indexer-attn-host-r2.md section 1): A = q-tile multiplicity per block (block 384*A threads). The Y=true
// variants (OPT 5, X 1) use a flat 1-D grid (2*T, 1): tile group = blockIdx.x >> 1, kv head = blockIdx.x & 1; every other
// instance uses grid (T, 2): tile group = blockIdx.x, kv head = blockIdx.y.
template <int A, int B, int C, int D, int E, int F>
__device__ static inline void fa2_dispatch(const unsigned short* Q, const unsigned short* K, const unsigned short* V, long stride,
                                           const unsigned short* gate, unsigned short* O, const int* sel, int n, int m) {
  if (F) fa2_body<A, E>(Q, K, V, stride, gate, O, sel, n, m, blockIdx.x & 1, blockIdx.x >> 1);
  else fa2_body<A, E>(Q, K, V, stride, gate, O, sel, n, m, blockIdx.y, blockIdx.x);
}
#define FA2(A, B, C, D, E, F)                                                                                       \
  extern "C" __global__ void k_attn_fa2_##A##_##B##_##C##_##D##_##E##_##F(                                           \
      const unsigned short* Q, const unsigned short* K, const unsigned short* V, long stride,                       \
      const unsigned short* gate, unsigned short* O, const int* sel, int n, int m, FaSel fs) {                      \
    fa2_dispatch<A, B, C, D, E, F>(Q, K, V, stride, gate, O, sel, n, m);                                            \
  }
FA2(1, 1, 0, 0, 0, 0)
FA2(1, 0, 1, 0, 0, 0)
FA2(1, 0, 0, 0, 0, 0)
FA2(1, 1, 1, 5, 1, 1)
FA2(1, 1, 1, 5, 1, 0)
FA2(1, 1, 1, 1, 0, 0)
FA2(1, 1, 1, 2, 0, 0)
FA2(1, 1, 1, 3, 0, 0)
FA2(1, 1, 1, 4, 0, 0)
FA2(1, 1, 1, 5, 0, 0)
FA2(1, 1, 1, 6, 0, 0)
FA2(1, 1, 1, 7, 0, 0)
FA2(1, 1, 1, 0, 0, 0)
FA2(2, 1, 0, 0, 0, 0)
FA2(2, 0, 1, 0, 0, 0)
FA2(2, 0, 0, 0, 0, 0)
FA2(2, 1, 1, 5, 1, 1)
FA2(2, 1, 1, 5, 1, 0)
FA2(2, 1, 1, 1, 0, 0)
FA2(2, 1, 1, 2, 0, 0)
FA2(2, 1, 1, 3, 0, 0)
FA2(2, 1, 1, 4, 0, 0)
FA2(2, 1, 1, 5, 0, 0)
FA2(2, 1, 1, 6, 0, 0)
FA2(2, 1, 1, 7, 0, 0)
FA2(2, 1, 1, 0, 0, 0)
