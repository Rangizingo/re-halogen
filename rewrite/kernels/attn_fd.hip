// Rewrites of the "attn_fd" family (flash_serve, gfx1151, codeobj_53000): k_attn_fd_merge, k_attn_fd<false/true>.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o attn_fd.co attn_fd.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git); cites are lines of that file.
#include <hip/hip_runtime.h>
// No mul+add fusion: the vendor fuses only where it emits v_fma/v_fmac, and every fma below is written out explicitly.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }
// bf16 RNE without NaN special case (merge :62583-62586, :62623-62624).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// k_attn_fd_merge(const float* part@0, const float* ml@8, int ns@16, bf16* out@24, const bf16* gate@32)
//   _ZN7halogen12_GLOBAL__N_115k_attn_fd_mergeEPKfS2_iPtPKt, block 256, grid (token bx, by in 0..23). Listing :62443-62630.
//   by = g*12 + h (g = by/12 via mulhi 0x2aaaaaab, :62449-62458, h = by%12), idx = 2*bx + g (:62455,62461), base = idx*ns
//   (:62465-62468). Split k of (idx, h):  m = ml[((base+k)*12+h)*2], l = ml[...+1] (stride 0x60 B per split, 8 B per head,
//   :62474-62485, :62544), part[((base+k)*12+h)*256 + tid] (0x3000 B per split, 1024 B per head, :62521-62539).
//   M = max_k m_k from -inf (v_max_f32, :62480-62496)
//   w_k = exp2((m_k - M) * 1.442695 (0x3fb8aa3b)) (:62551-62554) ; L = fma(l_k, w_k, L) ; acc = fma(part_k, w_k, acc) (:62556-62558)
//   f = acc / L, IEEE division (:62561-62575) ; r = bf16_rne(f) (:62583-62586)
//   gate == null: out[bx*6144 + by*256 + tid] = r (index v_mad_i64_i32 0x1800*bx + (by<<8)+tid, :62576-62584, :62629)
//   gate != null: g = bf16 gate[same index] ; sig = 1.0f / (1.0f + exp2(g * -1.442695)) (IEEE division, :62597-62620) ;
//     out = bf16_rne(sig * float(r with low 16 bits cleared)) (:62596, :62621-62624)
extern "C" __global__ void k_attn_fd_merge(const float* part, const float* ml, int ns, unsigned short* out,
                                           const unsigned short* gate) {
  int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x;
  int g = by / 12, h = by - 12 * g;
  long base = (long)(2 * bx + g) * ns;
  float M = __uint_as_float(0xff800000u);
  for (int k = 0; k < ns; k++) M = fmaxf(M, ml[((base + k) * 12 + h) * 2]);
  float L = 0.0f, acc = 0.0f;
  for (int k = 0; k < ns; k++) {
    long row = (base + k) * 12 + h;
    float w = __builtin_amdgcn_exp2f((ml[row * 2] - M) * __uint_as_float(0x3fb8aa3bu));
    L = __builtin_fmaf(ml[row * 2 + 1], w, L);
    acc = __builtin_fmaf(part[row * 256 + tid], w, acc);
  }
  float f = acc / L;
  unsigned short r = f2bf_rne(f);
  long o = (long)bx * 6144 + by * 256 + tid;
  if (gate) {
    float gv = bf2f(gate[o]);
    float e = __builtin_amdgcn_exp2f(gv * __uint_as_float(0xbfb8aa3bu));
    float sig = 1.0f / (1.0f + e);
    r = f2bf_rne(sig * bf2f(r));
  }
  out[o] = r;
}

// ---------------------------------------------------------------------------------------------------------------
// k_attn_fd<false>: flash-decode, one query token per sequence, 12 q heads per KV head (GQA), head_dim 256.
// Listing: codeobj_53000.s:1201166-1205811 (work/kernels/..k_attn_fdILb0E...s is the same text, line n there = file line 1201165+n).
// Args: q@0 (bf16 [b][24][256]), FdRows@8 (p0[b] = K pool, p1[b] = V pool, c[b] = last position len, see struct), ns@184,
// part@192 (f32 [(((2b+g)*ns+k)*12+h)*256+d]), ml@200 (f32 pair (m,l) at the same row*2), a@168/b@176/x@208 unused here.
// Grid (k = split, g = kv head 0..1, b = sequence), block 256 = 4 key groups x 2 waves (wave32); the odd wave of a group
// takes q heads 6..11 (v144 = 6*bit5(tid), :105 of the slice), the even wave heads 0..5.
// ---------------------------------------------------------------------------------------------------------------
struct FdRows {
  unsigned short* p0[8];
  unsigned short* p1[8];
  int slot[8];
};

typedef __bf16 bf2v __attribute__((ext_vector_type(2)));
__device__ static inline float dot2(unsigned a, unsigned b, float c) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf2v, a), __builtin_bit_cast(bf2v, b), c, false);
}
#define LOG2E_BITS 0x3fb8aa3bu
__device__ static inline float xmax32(float v) {  // butterfly max, offsets 16,8,4,2,1 (ds_bpermute xor patterns v180..v184)
  v = fmaxf(v, __shfl_xor(v, 16));
  v = fmaxf(v, __shfl_xor(v, 8));
  v = fmaxf(v, __shfl_xor(v, 4));
  v = fmaxf(v, __shfl_xor(v, 2));
  v = fmaxf(v, __shfl_xor(v, 1));
  return v;
}
__device__ static inline float xsum32(float v) {  // butterfly sum, same order; own + partner each stage
  v = v + __shfl_xor(v, 16);
  v = v + __shfl_xor(v, 8);
  v = v + __shfl_xor(v, 4);
  v = v + __shfl_xor(v, 2);
  v = v + __shfl_xor(v, 1);
  return v;
}

// Row of key position pos in the pool. <false>: pos itself. <true> (listing :296-346 of the slice): pos < 4*bb[b]
// goes through the block table, row = 4*a[b*x208 + (pos>>2)] + (pos&3); otherwise row = (n & ~3) + (pos - 4*bb[b]).
template <bool T>
__device__ static inline long keyrow(int pos, int s34, int tail0, const int* a, int b, int x208) {
  if (!T) return pos;
  if (pos < s34) return (long)a[(long)b * x208 + (pos >> 2)] * 4 + (pos & 3);
  return (long)tail0 + (pos - s34);
}
template <bool T>
__device__ static inline void attn_fd_body(const unsigned short* q, FdRows r, const int* a, const int* bb, int ns,
                                         float* part, float* ml, int x208) {
  const int tid = threadIdx.x, lane = tid & 31, par = (tid >> 5) & 1, grp = tid >> 6;
  const int k = blockIdx.x, g = blockIdx.y, b = blockIdx.z;
  const float NEG_INF = __uint_as_float(0xff800000u);
  const float L2E = __uint_as_float(LOG2E_BITS);
  const float SCALE = __builtin_amdgcn_rsqf(256.0f);  // v_rsq_f32 v0, 256.0 (:181)
  // key range of this split and group (:21-87 of the slice)
  int len = r.slot[b];
  int n = len + 1;
  int tail0 = 0, s34 = 0;  // T: positions < 4*bb[b] go through the block table a, the rest are the n%4 newest rows
  if (T) {
    s34 = 4 * bb[b];
    tail0 = n & ~3;
    n = (n - tail0) + s34;
  }
  int chunk = (n + ns - 1) / ns;
  int start = chunk * k;
  int end = min(start + chunk, n);
  int cntb = max(end - start, 0);
  int pg = (cntb + 3) >> 2;
  int gs = start + pg * grp;
  int ge = min(gs + pg, end);
  int ng = max(ge - gs, 0);
  const unsigned short* Kb = r.p0[b] + (long)g * 256 + (long)lane * 8;
  const unsigned short* Vb = r.p1[b] + (long)g * 256 + (long)lane * 8;
  const unsigned short* qb = q + ((long)(24 * b + g * 12 + 6 * par) * 256) + lane * 8;
  uint4 qv[6];
  for (int h = 0; h < 6; h++) qv[h] = *(const uint4*)(qb + (long)h * 256);
  float m[6], l[6], acc[6][8];
  for (int h = 0; h < 6; h++) {
    m[h] = NEG_INF;
    l[h] = 0.0f;
    for (int e = 0; e < 8; e++) acc[h][e] = 0.0f;
  }
  for (int t0 = 0; t0 < ng; t0 += 16) {
    int cnt = min(16, ng - t0);
    float sc[6];
    for (int h = 0; h < 6; h++) sc[h] = NEG_INF;
    for (int j = 0; j < cnt; j++) {
      long row = keyrow<T>(gs + t0 + j, s34, tail0, a, b, x208);
      uint4 kv = *(const uint4*)(Kb + row * 512);
      for (int h = 0; h < 6; h++) {
        float d = dot2(kv.x, qv[h].x, 0.0f);
        d = dot2(kv.y, qv[h].y, d);
        d = dot2(kv.z, qv[h].z, d);
        d = dot2(kv.w, qv[h].w, d);
        d = d + __shfl_xor(d, 16);
        d = d + __shfl_xor(d, 8);
        d = d + __shfl_xor(d, 4);
        d = d + __shfl_xor(d, 2);
        d = d + __shfl_xor(d, 1);
        if (lane == j) sc[h] = SCALE * d;
      }
    }
    float pv[6];
    for (int h = 0; h < 6; h++) {
      float s = sc[h];
      float mx = xmax32(s);
      float mn = fmaxf(m[h], mx);
      float p = (lane < cnt) ? __builtin_amdgcn_exp2f((s - mn) * L2E) : 0.0f;
      float lt = xsum32(p);
      float alpha = __builtin_amdgcn_exp2f((m[h] - mn) * L2E);
      l[h] = __builtin_fmaf(l[h], alpha, lt);
      for (int e = 0; e < 8; e++) acc[h][e] = alpha * acc[h][e];
      m[h] = mn;
      pv[h] = p;
    }
    for (int j = 0; j < cnt; j++) {
      long row = keyrow<T>(gs + t0 + j, s34, tail0, a, b, x208);
      uint4 vv = *(const uint4*)(Vb + row * 512);
      unsigned vw[4] = {vv.x, vv.y, vv.z, vv.w};
      float vf[8];
      for (int w = 0; w < 4; w++) {
        vf[2 * w] = __uint_as_float(vw[w] << 16);
        vf[2 * w + 1] = __uint_as_float(vw[w] & 0xffff0000u);
      }
      for (int h = 0; h < 6; h++) {
        float pj = __shfl(pv[h], j);
        for (int e = 0; e < 8; e++) acc[h][e] = __builtin_fmaf(pj, vf[e], acc[h][e]);
      }
    }
  }
  // merge key groups 1..3 into group 0, one group at a time (:3512-4528 of the slice)
  __shared__ float sm_acc[3][2][6][256];
  __shared__ float sm_m[3][2][6], sm_l[3][2][6];
  for (int gi = 1; gi < 4; gi++) {
    if (grp == gi) {
      for (int h = 0; h < 6; h++) {
        for (int e = 0; e < 8; e++) sm_acc[gi - 1][par][h][lane * 8 + e] = acc[h][e];
        if (lane == 0) {
          sm_m[gi - 1][par][h] = m[h];
          sm_l[gi - 1][par][h] = l[h];
        }
      }
    }
    __syncthreads();
    if (grp == 0) {
      for (int h = 0; h < 6; h++) {
        float mg = sm_m[gi - 1][par][h];
        if (mg != NEG_INF) {
          float mn = fmaxf(m[h], mg);
          float wg = __builtin_amdgcn_exp2f((mg - mn) * L2E);
          float lg = wg * sm_l[gi - 1][par][h];
          float wc = __builtin_amdgcn_exp2f((m[h] - mn) * L2E);
          l[h] = __builtin_fmaf(l[h], wc, lg);
          for (int e = 0; e < 8; e++) {
            float ag = wg * sm_acc[gi - 1][par][h][lane * 8 + e];
            acc[h][e] = __builtin_fmaf(wc, acc[h][e], ag);
          }
          m[h] = mn;
        }
      }
    }
    __syncthreads();
  }
  if (grp == 0) {
    for (int h = 0; h < 6; h++) {
      long row = (((long)(2 * b + g) * ns + k) * 12) + par * 6 + h;
      float* o = part + row * 256 + lane * 8;
      for (int e = 0; e < 8; e++) o[e] = acc[h][e];
      if (lane == 0) {
        ml[row * 2] = m[h];
        ml[row * 2 + 1] = l[h];
      }
    }
  }
}

extern "C" __global__ void k_attn_fd_false(const unsigned short* q, FdRows r, const int* a, const int* bb, int ns,
                                           float* part, float* ml, int x208) {
  attn_fd_body<false>(q, r, a, bb, ns, part, ml, x208);
}
extern "C" __global__ void k_attn_fd_true(const unsigned short* q, FdRows r, const int* a, const int* bb, int ns,
                                          float* part, float* ml, int x208) {
  attn_fd_body<true>(q, r, a, bb, ns, part, ml, x208);
}
