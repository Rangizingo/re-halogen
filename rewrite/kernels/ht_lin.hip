// Rewrite of k_ht_lin<BITS,S,A,B,C,D> (codeobj_53000, gfx1151), HT-quantized linear.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o ht_lin.co ht_lin.hip
// Listing cites are addresses in kernels-disasm/codeobj_53000.s (k_ht_lin<4,1,0,0,0,0> starts at 0x478000,
// <4,2,0,0,0,0> at 0x481700). `.long` VOP3 pairs in the dump were re-decoded by hand (scratch tool).
// B, C, D do not change the math (B: neighbour word via global load instead of LDS; C: x*0x83dcd12d done as two
// u24 multiplies and a shift; D: not yet separated). A selects the output type: 0 = bf16 RNE, 1 = f32.
// Kernarg: w@0 x@8 a@16 K@20 N@24 d@28 su@32 sv@40 y@48 cnt@56 out@64 up@72.
//   a = tokens (<= S), K = inner dim, N = rows (multiple of 128), d = 128-column blocks per split-K slice.
//   grid (N/128, splits), block 256 (8 waves of 32).
#include <hip/hip_runtime.h>
#pragma clang fp contract(off)

typedef _Float16 h2 __attribute__((ext_vector_type(2)));

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
// bf16 round to nearest even, no NaN case: v_add3_u32 v,v,(v>>16)&1,0x7fff ; and 0xffff0000 (:478710, :478720)
__device__ static inline float rne_bf16(float f) {
  unsigned u = __float_as_uint(f);
  return __uint_as_float((u + ((u >> 16) & 1u) + 0x7fffu) & 0xffff0000u);
}
// v_sad_u8(p,0,0x6400) = 0x6400 + sum of the four bytes, read as f16 bits (:478BC4)
__device__ static inline unsigned hash_sad(unsigned w16) {
  unsigned p = w16 * 0x83dcd12du;  // :478B40
  return 0x6400u + (p & 255u) + ((p >> 8) & 255u) + ((p >> 16) & 255u) + (p >> 24);
}

// GROT: first butterfly level as k_ht_lin_grp does it (fused with the earlier element); ep(t,row,val) stores.
template <int BITS, int S, bool GROT, class EP>
__device__ static inline void ht_core(const unsigned* __restrict__ w, const unsigned short* x, int a, int K, int N,
                                      int d, const _Float16* su, float* y, unsigned* cnt, const unsigned short* up,
                                      const _Float16* sv, int rb, int split, int nsplit, EP ep) {
  constexpr int TW = 8 * BITS;  // words per 16x16 tile
  __shared__ __attribute__((aligned(16))) char smem[S * 5120 + S * 512 + 16];
  _Float16* xs = (_Float16*)smem;                  // [S][20*128] rotated x, f16
  float* red = (float*)smem;                       // [8][S][128], reuses xs after the barrier
  float* part = (float*)(smem + S * 5120);         // [S][128]
  unsigned* flag = (unsigned*)(smem + S * 5120 + S * 512);

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int kb = K >> 7;                           // :47805C
  const int s35 = d * split;
  const int s11 = min(kb, s35 + d) - s35;          // 128-col blocks in this slice (:478064)
  const float SC = __uint_as_float(0x3db504f3u);   // 1/sqrt(128) (:478120)

  // ---- phase 1: x -> optional silu*up -> *su -> 128-point WHT -> *1/sqrt(128) -> f16 in LDS (:478074-4783BC)
  for (int idx = wave; idx < S * s11; idx += 8) {
    int tok = idx / s11, kbl = idx - tok * s11;
    int t = min(tok, a - 1);
    int k0 = (s35 + kbl) * 128 + lane * 4;
    float q[4], sf[4];
    for (int j = 0; j < 4; j++) {
      float xv = bf2f(x[(size_t)t * K + k0 + j]);
      if (up) {
        float e = __builtin_amdgcn_exp2f(xv * -1.44269502162933349609375f);  // :478504
        float sl = xv / (1.0f + e);                                          // IEEE div :47851C-47866C
        q[j] = rne_bf16(sl * bf2f(up[(size_t)t * K + k0 + j]));
      } else {
        q[j] = xv;
      }
      sf[j] = (float)su[k0 + j];
    }
    float a0, a1, a2, a3;
    if (GROT) {  // k_ht_lin_grp: codeobj_53000 0x41BF3C-0x41BF70
      float e1 = sf[1] * q[1], e3 = sf[3] * q[3];
      a0 = fmaf(sf[0], q[0], e1), a1 = fmaf(sf[0], q[0], -e1);
      a2 = fmaf(sf[2], q[2], e3), a3 = fmaf(sf[2], q[2], -e3);
    } else {
      float e0 = sf[0] * q[0], e1 = sf[1] * q[1], e2 = sf[2] * q[2];
      a0 = e0 + e1, a1 = e0 - e1;                           // :478170-478188
      a2 = fmaf(q[3], sf[3], e2), a3 = fmaf(-q[3], sf[3], e2);
    }
    float v[4] = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};      // slot0..3
    for (int m = 1; m <= 16; m <<= 1)                       // DPP xmask 1,2,4,8 then permlanex16 (:478194-478378)
      for (int s = 0; s < 4; s++) {
        float p = __shfl_xor(v[s], m, 32);
        v[s] = ((lane & m) ? -v[s] : v[s]) + p;
      }
    _Float16* dst = xs + tok * 2560 + kbl * 128 + lane * 4;
    for (int s = 0; s < 4; s++) dst[s] = (_Float16)(v[s] * SC);  // v_fma_mix{lo,hi}_f16 (:478388-4783B4)
  }
  __syncthreads();

  // ---- phase 2: decode weights, dot2 against x (:478AA4-479864)
  float acc[S][8];
  for (int t = 0; t < S; t++)
    for (int i = 0; i < 8; i++) acc[t][i] = 0.0f;
  const int h = lane >> 4, r = lane & 15;
  const int c = s11 > 0 ? s11 : 0;
  const int tb = wave * c, te = min(8 * s11, tb + c);
  const h2 c1 = {__builtin_bit_cast(_Float16, (unsigned short)0x1eee), __builtin_bit_cast(_Float16, (unsigned short)0x1eee)};
  const h2 c2 = {__builtin_bit_cast(_Float16, (unsigned short)0xc931), __builtin_bit_cast(_Float16, (unsigned short)0xc931)};
  for (int ct = tb; ct < te; ct++) {
    const size_t ctg = (size_t)s35 * 8 + ct;
    const unsigned* slab = w + (((size_t)rb * (K >> 4) + ctg) * 8) * TW;
    h2 xp[S][4];
    for (int t = 0; t < S; t++)
      for (int p = 0; p < 4; p++) xp[t][p] = *(const h2*)&xs[t * 2560 + ct * 16 + 8 * h + 2 * p];
    for (int i = 0; i < 8; i++) {
      const unsigned* tile = slab + i * TW;
      for (int p = 0; p < 4; p++) {
        unsigned raw[2];
        for (int e = 0; e < 2; e++) {
          int j = 128 * h + 8 * r + 2 * p + e;
          int q = (BITS * (255 - j)) % (256 * BITS);
          int m = q >> 5;
          unsigned lo = tile[TW - 1 - m];
          unsigned hi = tile[TW - 1 - ((m + 1) % TW)];
          unsigned w16 = (unsigned)((((unsigned long long)hi << 32) | lo) >> (q & 31)) & 0xffffu;
          raw[e] = hash_sad(w16);
        }
        h2 v = __builtin_bit_cast(h2, raw[0] | (raw[1] << 16));
        h2 wq = __builtin_elementwise_fma(v, c1, c2);  // v_pk_fma_f16 (:478C78)
        for (int t = 0; t < S; t++) acc[t][i] = __builtin_amdgcn_fdot2(wq, xp[t][p], acc[t][i], false);
      }
    }
  }
  // own half + other half (:4799AC), per-wave partials to LDS
  __syncthreads();
  for (int t = 0; t < S; t++)
    for (int i = 0; i < 8; i++) {
      float o = __shfl_xor(acc[t][i], 16, 32);
      if (lane < 16) red[(wave * S + t) * 128 + 16 * i + r] = acc[t][i] + o;
    }
  __syncthreads();
  // sum over waves 0..7 starting at +0.0 (:479A58-479A98)
  for (int e = tid; e < S * 128; e += 256) {
    float s = 0.0f;
    for (int ww = 0; ww < 8; ww++) s += red[ww * S * 128 + e];
    part[e] = s;
  }
  __syncthreads();
  if (nsplit > 1) {
    // split-K: partials at y[(split*S+t)*N + rb*128 + row], last arriver sums splits 0..n-1 (:479AC8-479C98)
    for (int e = tid; e < S * 128; e += 256) {
      int t = e >> 7, row = e & 127;
      y[((size_t)split * S + t) * N + rb * 128 + row] = part[e];
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      unsigned old = atomicAdd(&cnt[rb], 1u);
      *flag = (old == (unsigned)(nsplit - 1)) ? 1u : 0u;
    }
    __syncthreads();
    if (*flag == 0) return;
    __threadfence();
    for (int e = tid; e < S * 128; e += 256) {
      int t = e >> 7, row = e & 127;
      float s = 0.0f;
      for (int sp = 0; sp < nsplit; sp++) s += __builtin_nontemporal_load(&y[((size_t)sp * S + t) * N + rb * 128 + row]);
      part[e] = s;
    }
    if (tid == 0) cnt[rb] = 0u;
    __syncthreads();
  }
  // ---- epilogue: 128-point WHT over rows, *1/sqrt(128), *sv, store (:479CC8-47A24C)
  if (wave < S && wave < a) {
    const int t = wave;
    float v[8];
    for (int i = 0; i < 8; i++) v[i] = part[t * 128 + 16 * i + r];
    float a_[8] = {v[0] + v[1], v[0] - v[1], v[2] + v[3], v[2] - v[3], v[4] + v[5], v[4] - v[5], v[6] + v[7], v[6] - v[7]};
    float b_[8] = {a_[0] + a_[2], a_[1] + a_[3], a_[0] - a_[2], a_[1] - a_[3], a_[4] + a_[6], a_[5] + a_[7], a_[4] - a_[6], a_[5] - a_[7]};
    float c_[8] = {b_[0] + b_[4], b_[1] + b_[5], b_[2] + b_[6], b_[3] + b_[7], b_[0] - b_[4], b_[1] - b_[5], b_[2] - b_[6], b_[3] - b_[7]};
    for (int m = 1; m <= 8; m <<= 1)
      for (int i = 0; i < 8; i++) {
        float p = __shfl_xor(c_[i], m, 32);
        c_[i] = ((lane & m) ? -c_[i] : c_[i]) + p;
      }
    if (lane < 16)
      for (int i = 0; i < 8; i++) {
        int row = rb * 128 + 16 * i + r;
        ep(t, row, (float)sv[row] * (c_[i] * SC));
      }
  }
}

template <int BITS, int S, bool F32>
__device__ static inline void ht_lin_body(const unsigned* w, const unsigned short* x, int a, int K, int N, int d,
                                          const _Float16* su, const _Float16* sv, float* y, unsigned* cnt, void* out,
                                          const unsigned short* up) {
  auto ep = [&](int t, int row, float val) {
    if (F32) ((float*)out)[(size_t)t * N + row] = val;
    else ((unsigned short*)out)[(size_t)t * N + row] = (unsigned short)(__float_as_uint(rne_bf16(val)) >> 16);
  };
  ht_core<BITS, S, false>(w, x, a, K, N, d, su, y, cnt, up, sv, blockIdx.x, blockIdx.y, gridDim.y, ep);
}

#define HT(BITS, S, A)                                                                                        \
  extern "C" __global__ __launch_bounds__(256) void ht_lin_b##BITS##_s##S##_a##A(                            \
      const unsigned* w, const unsigned short* x, int a, int K, int N, int d, const _Float16* su,            \
      const _Float16* sv, float* y, unsigned* cnt, void* out, const unsigned short* up) {                    \
    ht_lin_body<BITS, S, A>(w, x, a, K, N, d, su, sv, y, cnt, out, up);                                      \
  }
HT(3, 1, 0) HT(3, 1, 1) HT(3, 2, 0) HT(3, 2, 1) HT(3, 4, 0) HT(3, 4, 1) HT(3, 8, 0) HT(3, 8, 1)
HT(4, 1, 0) HT(4, 1, 1) HT(4, 2, 0) HT(4, 2, 1) HT(4, 4, 0) HT(4, 4, 1) HT(4, 8, 0) HT(4, 8, 1)

// ---------------------------------------------------------------------------------------------------------
// k_ht_lin_grp<BITS,S,b1,b2,b3,b4>(HtGrp, const u16* x, int a, int K, float* y, unsigned* cnt)
// Kernarg: HtGrp@0 (408 B), x@408, a@416, K@420, y@424, cnt@432. 1-D grid; up to 4 groups, each its own weights.
// Layout read from the listing (k_ht_lin_grp<4,1,0,0,0,0>, 0x41BE00): group g at 0x48*g:
//   +0 w, +8 su, +0x10 sv, +0x18 out, +0x20 out2, +0x28 N, +0x2c nsplit, +0x30 d, +0x34 first block,
//   +0x38 counter offset (u32 units), +0x40 y offset (floats, i64); 0x120 ngroups, 0x128 rot_su, 0x130 rot_out,
//   0x138 first rotation-only block. (0x41BE04, 0x41C1F8-0x41C2B8)
struct HtGrpG {
  const unsigned* w; const _Float16* su; const _Float16* sv; void* out; void* out2;
  int n, nsplit, d, start, cntoff, pad0; long long yoff;
};
struct HtGrp {
  HtGrpG g[4];
  int ngroups, pad1; const _Float16* rot_su; _Float16* rot_out; int rot_start, pad2;
  // fields read only by k_ht_lin_grp<...,b4=1> (0x3dba00): extra blocks past gstart
  const unsigned short* gmat; const float* gvec; float* gaout; int gstart, pad3;
  const char* gbase; long long goff1, goff2, pad4; const unsigned short* gx; float* gout; long long pad5;
};
static_assert(sizeof(HtGrpG) == 0x48 && sizeof(HtGrp) == 408, "HtGrp layout");

// rotation-only blocks: x (stride 2560, 20 blocks of 128 per token) -> f16 rotated copy (0x41BE3C-0x41C1D4)
__device__ static inline void grp_rot_only(const HtGrp& G, const unsigned short* x, int a) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int v1 = (blockIdx.x - G.rot_start) * 8 + wave;
  if (v1 >= a * 20) return;
  const int tok = v1 / 20, kbl = v1 - tok * 20, k = kbl * 128 + lane * 4;
  float q[4], sf[4];
  for (int j = 0; j < 4; j++) { q[j] = bf2f(x[(size_t)tok * 2560 + k + j]); sf[j] = (float)G.rot_su[k + j]; }
  float e1 = sf[1] * q[1], e3 = sf[3] * q[3];
  float a0 = fmaf(sf[0], q[0], e1), a1 = fmaf(sf[0], q[0], -e1), a2 = fmaf(sf[2], q[2], e3), a3 = fmaf(sf[2], q[2], -e3);
  float v[4] = {a0 + a2, a1 + a3, a0 - a2, a1 - a3};
  for (int m = 1; m <= 16; m <<= 1)
    for (int s = 0; s < 4; s++) { float p = __shfl_xor(v[s], m, 32); v[s] = ((lane & m) ? -v[s] : v[s]) + p; }
  const float SC = __uint_as_float(0x3db504f3u);
  for (int s = 0; s < 4; s++) G.rot_out[(size_t)tok * 2560 + k + s] = (_Float16)(v[s] * SC);
}

// b4=1 extra block j = blk - gstart. j < 64 (and j > 64): 8 rows of a bf16 [rows][2560] matrix dotted with an f32
// vector, 8 floats per block (0x3dba28-0x3dbd18 area: 8 fmac chains over j = tid + m*blockDim, LDS halving tree).
__device__ static inline void grp_job_a(const HtGrp& G) {
  __shared__ float sm[8 * 256];
  const int tid = threadIdx.x, bs = blockDim.x, j = blockIdx.x - G.gstart;
  const unsigned short* mat = G.gmat + (size_t)j * 8 * 2560;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int e = tid; e < 2560; e += bs) {
    float v = G.gvec[e];
    for (int r = 0; r < 8; r++) acc[r] = fmaf(v, bf2f(mat[(size_t)r * 2560 + e]), acc[r]);  // v_fmac_f32 (0x3dbc34-0x3dbc6c)
  }
  for (int r = 0; r < 8; r++) sm[r * 256 + tid] = acc[r];
  __syncthreads();
  for (int s = bs >> 1; s > 0; s >>= 1) {
    if (tid < s)
      for (int r = 0; r < 8; r++) sm[r * 256 + tid] = sm[r * 256 + tid + s] + sm[r * 256 + tid];
    __syncthreads();
  }
  if (tid < 8) G.gaout[j * 8 + tid] = sm[tid * 256];
}

// b4=1 extra block j == 64: sigmoid(dot) of a 2560-long row stored as 80 groups of 32 4-bit codes: 16 bytes of
// codes + one f16 scale per group; a byte indexes a 256-entry LUT of bf16 pairs built from a 16-entry f32 codebook
// (0x3dbe20-0x3dbf34 LUT, 0x3dbf48-0x3dc238 dot chain, 0x3dc358-0x3dc438 sum and sigmoid).
typedef short bf2v __attribute__((ext_vector_type(2)));
__device__ static inline void grp_job_b(const HtGrp& G) {
  __shared__ unsigned lut[256];
  __shared__ float wsum[8];
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const float* cb = (const float*)G.gbase;
  unsigned lo = (unsigned)(__float_as_uint(rne_bf16(cb[tid & 15])) >> 16);
  unsigned hi = __float_as_uint(rne_bf16(cb[(tid >> 4) & 15])) & 0xffff0000u;
  lut[tid] = hi | lo;
  __syncthreads();
  float acc = 0.0f;
  if (lane < 10) {
    const int g = wave * 10 + lane;
    const unsigned* codes = (const unsigned*)(G.gbase + G.goff1) + g * 4;
    const float sc = (float)((const _Float16*)(G.gbase + G.goff2))[g];
    const unsigned* xp = (const unsigned*)(G.gx + g * 32);
    float dot = 0.0f;
    for (int dw = 0; dw < 4; dw++)
      for (int b = 0; b < 4; b++) {
        unsigned code = (codes[dw] >> (8 * b)) & 0xffu;
        dot = __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf2v, xp[dw * 4 + b]), __builtin_bit_cast(bf2v, lut[code]), dot, false);
      }
    acc = fmaf(sc, dot, 0.0f);  // v_fmac_f32 v13,v15,v20 (0x3dc210)
  }
  for (int m = 16; m >= 1; m >>= 1) acc = acc + __shfl_xor(acc, m, 32);
  if (lane == 0) wsum[wave] = acc;
  __syncthreads();
  if (tid == 0) {
    float s = 0.0f;
    for (int w = 0; w < 8; w++) s += wsum[w];
    float e = __builtin_amdgcn_exp2f(s * -1.44269502162933349609375f);
    G.gout[0] = 1.0f / (1.0f + e);
  }
}

template <int BITS, int S, bool EXTRA>
__device__ static inline void ht_grp_body(const HtGrp& G, const unsigned short* x, int a, int K, float* y, unsigned* cnt) {
  const int blk = blockIdx.x;
  if (EXTRA && blk >= G.gstart) {
    if (blk - G.gstart == 64) grp_job_b(G); else grp_job_a(G);
    return;
  }
  if (G.rot_out) {
    if (blk >= G.rot_start) { grp_rot_only(G, x, a); return; }
  }
  int gi = 0;
  if (G.ngroups > 1 && blk >= G.g[1].start) gi = 1;
  if (G.ngroups > 2 && blk >= G.g[2].start) gi = 2;
  if (G.ngroups > 3 && blk >= G.g[3].start) gi = 3;
  const HtGrpG& g = G.g[gi];
  const int rel = blk - g.start, rb = rel / g.nsplit, split = rel - rb * g.nsplit;
  const int N = g.n;
  auto ep = [&](int t, int idx, float val) {
    unsigned short bits = (unsigned short)(__float_as_uint(rne_bf16(val)) >> 16);
    if (!g.out2) { ((unsigned short*)g.out)[(size_t)t * N + idx] = bits; return; }
    int half = N / 2, chunk = idx / 512, m = idx - chunk * 512;
    unsigned short* dst = (unsigned short*)(m < 256 ? g.out : g.out2);
    dst[(size_t)t * half + chunk * 256 + (m < 256 ? m : m - 256)] = bits;
  };
  ht_core<BITS, S, true>(g.w, x, a, K, N, g.d, g.su, y + g.yoff, cnt + g.cntoff, nullptr, g.sv, rb, split, g.nsplit, ep);
}

#define HG(BITS, S)                                                                                        \
  extern "C" __global__ __launch_bounds__(256) void ht_grp_b##BITS##_s##S(                                  \
      HtGrp G, const unsigned short* x, int a, int K, float* y, unsigned* cnt) {                            \
    ht_grp_body<BITS, S, false>(G, x, a, K, y, cnt);                                                        \
  }
HG(3, 1) HG(3, 2) HG(3, 4) HG(3, 8) HG(4, 1) HG(4, 2) HG(4, 4) HG(4, 8)
extern "C" __global__ __launch_bounds__(256) void ht_grp_b4_s1_x(HtGrp G, const unsigned short* x, int a, int K,
                                                                  float* y, unsigned* cnt) {
  ht_grp_body<4, 1, true>(G, x, a, K, y, cnt);
}
