// Shared helpers for the smpg_* rewrites (g23), codeobj_53000, gfx1151. Cites are lines of kernels-disasm/codeobj_53000.s.
// Not a family file: included by smpg_max.hip, smpg_level.hip, smpg_mass.hip, ...
#pragma once
#include <hip/hip_runtime.h>
// The vendor fuses mul+add only where it prints v_fma/v_fmac; keep the compiler from fusing anything else.
#pragma clang fp contract(off)

// SampleRow, 32 bytes (analysis/rebuild-guide.md:217): temperature, top_k, top_p, min_p, seed, position, stream tag.
struct SampleRow { float temp; int top_k; float top_p; float min_p; unsigned long long seed; unsigned pos; unsigned tag; };

// SmpScratch, 0x4048 bytes per row (s_mul_i32 s3, 0x4048, smpg_max :4888). Offsets as used by the kernels:
//   +0x0000 f32[nblk]   per-block row max (smpg_max :5105)
//   +0x0400 f32[nblk]   per-block sum of w (level0 :2407218)
//   +0x0800 f32[nblk]   per-block kept mass (mass, see smpg_mass.hip)
//   +0x1000 u32[4][256] digit counts per level (zeroed by block 0 of smpg_max :4925-4929)
//   +0x2000 u64[4][256] digit masses per level, 45-bit fixed point (zeroed :4928,4930-4932)
//   +0x4000 u32[3]      prefix after level 1,2,3 (+0,+4,+8)       (level1 :2407839)
//   +0x4010 u32[3]      carried count above the prefix (+0,+4,+8)
//   +0x4020 u64 at +0x4020, +0x4028, +0x4030 carried mass above the prefix
//   +0x4040 u32         Tkey (mass kernel, smpg_mass.hip)
struct SmpScratch { unsigned char b[0x4048]; };
#define SCR_ROW(scr, r) ((unsigned char*)(scr) + (long)(r) * 0x4048)

__device__ static inline float canon_max(float a, float b) { return __builtin_fmaxf(a, b); }

// Tree over 256 slots, offsets 128,64,...,1 (smpg_max :4984-5088). Slot tid takes op(slot tid, slot tid+off).
__device__ static inline void tree_max(float* lds) {
  int tid = threadIdx.x;
  for (int off = 128; off >= 1; off >>= 1) {
    if (tid < off) lds[tid] = __builtin_fmaxf(lds[tid], lds[tid + off]);
    __syncthreads();
  }
}
__device__ static inline void tree_sum(float* lds) {
  int tid = threadIdx.x;
  for (int off = 128; off >= 1; off >>= 1) {
    if (tid < off) { float hi = lds[tid + off], lo = lds[tid]; lds[tid] = hi + lo; }
    __syncthreads();
  }
}

// Row maximum of the per-block maxima: slot tid = tid < nblk ? scratch.f32[tid] : -inf (level0 :2406842-2406845).
__device__ static inline float row_max(const unsigned char* S, int nblk, float* lds) {
  int tid = threadIdx.x;
  float v = __int_as_float(0xff800000);
  if (nblk > tid) v = ((const float*)S)[tid];
  lds[tid] = v;
  __syncthreads();
  tree_max(lds);
  float r = lds[0];
  __syncthreads();
  return r;
}

// v_cvt_u32_f32 / v_cvt_u32_f64: truncate, negative and NaN give 0, saturate at 2^32-1.
__device__ static inline unsigned cvt_u32(float f) {
  if (!(f > 0.0f)) return 0u;
  if (f >= 4294967296.0f) return 0xffffffffu;
  return (unsigned)f;
}
__device__ static inline unsigned cvt_u32d(double f) {
  if (!(f > 0.0)) return 0u;
  if (f >= 4294967296.0) return 0xffffffffu;
  return (unsigned)f;
}

// Sortable key of a float: x < 0 ? ~x : x | 0x80000000 (level0 :2407065-2407070).
__device__ static inline unsigned skey(float x) {
  unsigned u = __float_as_uint(x);
  return ((int)u < 0) ? ~u : (u | 0x80000000u);
}

// w = exp2(((x - max) * (1/T)) * log2e) with the multiplies in this order (level0 :2407056-2407061).
__device__ static inline float wexp(float x, float mx, float invT) {
  float v = x - mx;
  v = invT * v;
  v = v * 1.44269502162933349609375f;  // 0x3fb8aa3b
  return __builtin_amdgcn_exp2f(v);
}

// 45-bit fixed point mass of w as two u32 halves: m = trunc(fma(2^45, w, 0.5)); hi = floor(m * 2^-32); lo = fma(-2^32, hi, m)
// (level0 :2407064-2407078). Packed as hi:lo.
__device__ static inline unsigned long long mass45(float w) {
  float m = __builtin_fmaf(35184372088832.0f, w, 0.5f);  // 0x56000000 = 2^45
  m = __builtin_truncf(m);
  float hi = __builtin_floorf(m * 2.3283064365386963e-10f);  // 0x2f800000 = 2^-32
  float lo = __builtin_fmaf(-4294967296.0f, hi, m);           // 0xcf800000
  return ((unsigned long long)cvt_u32(hi) << 32) | cvt_u32(lo);
}

// top_p target in 45-bit fixed point: trunc(top_p * Z * 2^45) as u64 (level1 :2407424-2407441).
__device__ static inline unsigned long long topp_target(float top_p, float Z) {
  double p = (double)top_p * (double)Z;
  p = __builtin_ldexp(p, 45);
  p = __builtin_trunc(p);
  double hi = __builtin_floor(__builtin_ldexp(p, -32));
  double lo = __builtin_fma(-4294967296.0, hi, p);
  return ((unsigned long long)cvt_u32d(hi) << 32) | cvt_u32d(lo);
}

__device__ static inline bool topk_active(const SampleRow& r, int n);
__device__ static inline bool topp_active(const SampleRow& r);

// Single-block Tkey selection used by the monolithic samplers (k_sample_f32, k_accept_prob_f32, k_resid_sample_f32).
// Four passes; pass LV (1..4): histogram of byte (4-LV) over keys whose higher bytes match prefix, then thread 0 scans
// digits 255..0 adding count and mass to the carry and stops at the first digit where
// (top_k active and top_k <= count) or (top_p active and mass >= target); carry before that digit and the new prefix are kept.
// Returns Tkey = prefix after pass 4. Must be called by all 256 threads of the block. Listing cites: k_sample_f32 in sample.hip.
__device__ static inline unsigned select_tkey(const float* xr, int n, float mx, float invT, const SampleRow& row, float Z) {
  __shared__ unsigned hc[256];
  __shared__ unsigned long long hm[256];
  __shared__ unsigned st_prefix, st_count;
  __shared__ unsigned long long st_mass;
  int tid = threadIdx.x;
  bool ka = topk_active(row, n), pa = topp_active(row);
  if (!(ka || pa)) return 0u;
  unsigned long long target = pa ? topp_target(row.top_p, Z) : 0ull;
  if (tid == 0) { st_prefix = 0; st_count = 0; st_mass = 0; }
  __syncthreads();
  for (int lv = 1; lv <= 4; lv++) {
    const int shift = 24 - 8 * (lv - 1);
    hc[tid] = 0u;
    hm[tid] = 0ull;
    __syncthreads();
    unsigned prefix = st_prefix;
    const unsigned lim = lv == 1 ? 0u : (1u << (32 - 8 * (lv - 1)));
    for (int i = tid; i < n; i += 256) {
      float xv = xr[i];
      unsigned key = skey(xv);
      if (lv > 1 && !((key ^ prefix) < lim)) continue;
      float w = wexp(xv, mx, invT);
      unsigned d = (key >> shift) & 0xffu;
      atomicAdd(&hc[d], 1u);
      atomicAdd(&hm[d], mass45(w));
    }
    __syncthreads();
    if (tid == 0) {
      unsigned c = st_count;
      unsigned long long m = st_mass;
      unsigned cb = c, d = 255;
      unsigned long long mb = m;
      for (int i = 255; i >= 0; i--) {
        cb = c; mb = m;
        c = c + hc[i];
        m = m + hm[i];
        d = i;
        bool stop = (ka && (unsigned)row.top_k <= c) || (pa && m >= target);
        if (stop) break;
      }
      st_prefix = (d << shift) | prefix;
      st_count = cb;
      st_mass = mb;
    }
    __syncthreads();
  }
  return st_prefix;
}

// Filter gates (level0 :2406971-2406985, mass :164-178 per sampler-math-r5.md): top_k active iff 0 < top_k < N,
// top_p active iff 0 < top_p < 1.0.
__device__ static inline bool topk_active(const SampleRow& r, int n) { return r.top_k > 0 && r.top_k < n; }
__device__ static inline bool topp_active(const SampleRow& r) { return r.top_p > 0.0f && r.top_p < 1.0f; }

// Uniform draw u in [0,1): splitmix64 style mix of seed, position and tag, top 24 bits, times 2^-24.
//   x1 = pos * 0xd1b54a32d192ed03   (:6248-6256), x2 = tag * 0x9e3779b97f4a7c15 (:6252-6257)
//   z = seed ^ x1 ^ x2 (:6257-6259); z += 0x9e3779b97f4a7c15 (:6260-6261)
//   z = (z ^ z>>30) * 0xbf58476d1ce4e5b9 (:6263-6271); z = (z ^ z>>27) * 0x94d049bb133111eb, only the high 32 bits
//   kept (:6273-6281); u24 = high32 >> 8 (:6283); u = float(u24) * 2^-24 (v_mul 0x33800000, :6297).
__device__ static inline float draw_u(const SampleRow& r) {
  unsigned long long z = r.seed ^ ((unsigned long long)r.pos * 0xd1b54a32d192ed03ull) ^ ((unsigned long long)r.tag * 0x9e3779b97f4a7c15ull);
  z += 0x9e3779b97f4a7c15ull;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  unsigned u24 = (unsigned)(z >> 40);
  return (float)u24 * 5.9604644775390625e-08f;  // 0x33800000 = 2^-24
}

// ln(v) - adj as the vendor computes it (:6657-6680): v < 2^-126 is scaled by 2^32 (adj = 32*ln2);
// v_log_f32 = log2; ln = c*l2 + correction (mul, fma, fmamk), used only when |l2| < inf.
__device__ static inline float ln_adj(float v) {
  bool den = (__uint_as_float(0x800000) > v);
  float vs = v * (den ? 4294967296.0f : 1.0f);
  float l2 = __builtin_amdgcn_logf(vs);
  const float C = __uint_as_float(0x3f317217);
  float hi = C * l2;
  float lo = __builtin_fmaf(C, l2, -hi);
  lo = __builtin_fmaf(l2, __uint_as_float(0x3377d1cf), lo);
  float ln = hi + lo;
  bool fin = (__uint_as_float(0x7f800000) > __builtin_fabsf(l2));
  float lnv = fin ? ln : l2;
  float adj = den ? __uint_as_float(0x41b17218) : 0.0f;
  return lnv - adj;
}

