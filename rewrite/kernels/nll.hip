// Rewrites of the "nll" family from codeobj_53000 (flash_serve, gfx1151): k_nll_init, k_nll_chunk<float>,
// k_nll_chunk<unsigned short> (bf16), k_nll_final.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o nll.co nll.hip
// Listing: kernels-disasm/codeobj_53000.s: init 50813-50849, final 50852-50914, chunk<float> 2406108-2406463,
// chunk<ushort> 2406466-2406822 (not in git). Cites below are into that file.
#include <hip/hip_runtime.h>
// The vendor uses explicit fma only where listed; stop the compiler from fusing separate mul and add.
#pragma clang fp contract(off)

// Online log-sum-exp over vocab chunks: m = running max, s = running sum of exp(x - m), t = target logit.

// k_nll_init(float* m, float* s, float* t, int n): kernarg m@0, s@8, t@16, n@24
// i = blockIdx.x*blockDim.x + tid, guard i < n (v_cmpx_gt_i32, :50823). m[i] = -inf, s[i] = 0, t[i] = -inf
// (stores :50846-50848, constants v6 = 0xff800000 and v7 = 0 :50830).
extern "C" __global__ void k_nll_init(float* m, float* s, float* t, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(n > i)) return;
  m[i] = __int_as_float(0xff800000);
  s[i] = 0.0f;
  t[i] = __int_as_float(0xff800000);
}

// Block tree over 256 LDS slots, offsets 128,64,...,1 (:2406161-2406281 for max, :2406337-2406413 for sum).
// Slot tid is updated by threads tid < off from slot tid+off.
template <int MODE>
__device__ static inline void tree(float* lds) {
  int tid = threadIdx.x;
  for (int off = 128; off >= 1; off >>= 1) {
    if (tid < off) {
      float hi = lds[tid + off], lo = lds[tid];
      lds[tid] = MODE == 0 ? __builtin_fmaxf(lo, hi) : hi + lo;
    }
    __syncthreads();
  }
}

// Shared body of k_nll_chunk. XB selects float (0) or bf16 (1: global_load_u16 then v_lshlrev_b32 16, e.g. :2406549 variant).
// kernarg: x@0, w@8 (chunk width, s22), base@12 (s23), tgt@16, m@24, s@32, t@40.  One block per chunk row c = blockIdx.x
// (s2), 256 threads. Row pointer x + c*w (s_mul + shift 2 for f32 / 1 for bf16, :2406113-2406122).
template <typename XT, bool BF16>
__device__ static inline float ld(const XT* p) {
  if (BF16) return __uint_as_float((unsigned)(*(const unsigned short*)p) << 16);
  return *(const float*)p;
}

template <typename XT, bool BF16>
__device__ static inline void nll_chunk(const XT* x, int w, int base, const int* tgt, float* m, float* s, float* t) {
  __shared__ float lds[256];
  int tid = threadIdx.x, c = blockIdx.x;
  const XT* row = x + (long)c * w;
  // pass 1: per-thread max, stride blockDim.x, start -inf (:2406111, loop :2406137-2406156, canonicalize + v_max_f32)
  float mx = __int_as_float(0xff800000);
  for (int i = tid; i < w; i += blockDim.x) mx = __builtin_fmaxf(mx, ld<XT, BF16>(row + i));
  lds[tid] = mx;
  __syncthreads();
  tree<0>(lds);
  float chunkmax = lds[0];
  __syncthreads();
  // old running max m[c] (:2406295); new max = max(m_old, chunk max) (:2406297-2406300)
  float mold = m[c];
  float gm = __builtin_fmaxf(__builtin_fmaxf(mold, mold), __builtin_fmaxf(chunkmax, chunkmax));
  // pass 2: per-thread sum of exp2((x - gm) * log2e), sequential in i (:2406318-2406329)
  float sum = 0.0f;
  for (int i = tid; i < w; i += blockDim.x) {
    float v = ld<XT, BF16>(row + i) - gm;
    v = v * 1.44269502162933349609375f;  // 0x3fb8aa3b :2406324
    sum = sum + __builtin_amdgcn_exp2f(v);
  }
  lds[tid] = sum;
  __syncthreads();
  tree<1>(lds);
  if (tid == 0) {
    float scaled = 0.0f;
    // old sum rescaled unless m_old == -inf (s_cmp_eq_f32 s24, 0xff800000 :2406420)
    if (!(mold == __int_as_float(0xff800000))) {
      float f = __builtin_amdgcn_exp2f((mold - gm) * 1.44269502162933349609375f);  // :2406424-2406428
      scaled = s[c] * f;                                                          // :2406431
    }
    s[c] = scaled + lds[0];  // :2406439-2406442
    m[c] = gm;               // :2406443
    int loc = tgt[c] - base;  // :2406440
    if (!(loc < 0 || loc >= w)) t[c] = ld<XT, BF16>(row + loc);  // :2406444-2406462
  }
}

extern "C" __global__ void k_nll_chunk_f32(const float* x, int w, int base, const int* tgt, float* m, float* s, float* t) {
  nll_chunk<float, false>(x, w, base, tgt, m, s, t);
}
extern "C" __global__ void k_nll_chunk_bf16(const unsigned short* x, int w, int base, const int* tgt, float* m, float* s, float* t) {
  nll_chunk<unsigned short, true>(x, w, base, tgt, m, s, t);
}

// k_nll_final(const float* m, const float* s, const float* t, float* out, int n): kernarg m@0, s@8, t@16, out@24, n@32
// i < n (:50862). sum = s[i], mx = m[i], tg = t[i]. out = -((tg - mx) - (ln(sum) - adj)) (:50892,50909-50912).
// ln(sum): if sum < 2^-126 (v_cmp_gt_f32 0x800000 > sum, :50886) scale sum by 2^32 (v_ldexp_f32, :50889) and
// subtract 32*ln2 (0x41b17218, :50900,50909). v_log_f32 = log2; ln = ln2*log2 with a two-term correction
// (mul, fma, fmamk with 0x3377d1cf, add :50895-50899). If log2 is not below +inf (inf or NaN) the raw log2 is kept
// (v_cmp_gt_f32 +inf > log2, select :50901-50904).
extern "C" __global__ void k_nll_final(const float* m, const float* s, const float* t, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(n > i)) return;
  float sum = s[i], mx = m[i], tg = t[i];
  bool den = (__uint_as_float(0x800000) > sum);
  float xs = den ? sum * 4294967296.0f : sum;
  float l2 = __builtin_amdgcn_logf(xs);
  const float LN2H = __uint_as_float(0x3f317217);
  float hi = __fmul_rn(LN2H, l2);
  float lo = __builtin_fmaf(LN2H, l2, -hi);
  lo = __builtin_fmaf(l2, __uint_as_float(0x3377d1cf), lo);
  float ln = hi + lo;
  bool fin = (__uint_as_float(0x7f800000) > l2);
  float lnv = fin ? ln : l2;
  float adj = den ? __uint_as_float(0x41b17218) : 0.0f;
  float d = tg - mx;
  float e = lnv - adj;
  float r = d - e;
  out[i] = -r;
}
