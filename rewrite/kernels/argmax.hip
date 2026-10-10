// Rewrites of the "argmax" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o argmax.co argmax.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
//
// Candidate rule used by k_argmax_f32, k_argmax_part, k_argmax_final and every reduction stage:
//   take (x, i) over (bx, bi) when x > bx, or x == bx and i < bi   (v_cmp_eq_f32 + v_cmp_lt_i32 + v_cmp_gt_f32,
//   e.g. codeobj_53000.s:51192-51198, 51257-51265). NaN is never taken (both compares false).
// The legacy kernel takes only on a strict x > bx (:51019).
// Start values: value -inf (0xff800000), index 0x7fffffff (v_bfrev_b32 -2); legacy index -1 (:50928).
#include <hip/hip_runtime.h>

// AmxMasks: 8 pointers read from the kernarg block at row*8 (s_load_b64 [kernarg + row*8], 0x20 / 0x18,
// codeobj_53000.s:51304-51310, 51106-51112). A null mask means every element is a candidate.
struct AmxMasks { const unsigned* p[8]; };

__device__ static inline bool take(float x, int i, float bv, int bi) { return x > bv || (x == bv && i < bi); }

// Mask test: base pointer with bit 0 cleared (s_and_b32 -2, :51316), bit (i & 31) of word i >> 5 (:51362-51376).
__device__ static inline const unsigned* mask_base(const unsigned* p) { return (const unsigned*)((unsigned long long)p & ~1ull); }
__device__ static inline bool in_mask(const unsigned* m, int i) { return (m[i >> 5] >> (i & 31)) & 1u; }

// Tree reduction over LDS, offsets blockDim/2 ... 1 (:51217-51270). Rule: other wins when it is better.
__device__ static inline void tree_reduce(float* sv, int* si, bool strict_only) {
  int tid = threadIdx.x;
  for (int s = (blockDim.x >> 1) & 0x7fff; s != 0; s >>= 1) {
    if (tid < s) {
      float ov = sv[tid + s];
      int oi = si[tid + s];
      bool t = strict_only ? (ov > sv[tid]) : take(ov, oi, sv[tid], si[tid]);
      if (t) { sv[tid] = ov; si[tid] = oi; }
    }
    __syncthreads();
  }
}

// k_argmax_f32(float const* x, int n, int* out, AmxMasks m)
// _ZN7halogen12_GLOBAL__N_112k_argmax_f32EPKfiPiNS_8AmxMasksE, kernarg: x@0, n@8, out@16, masks@24
// One block per row (s2 = blockIdx.x). Thread scan i = tid, tid+blockDim, ... < n (:51150), row base x + row*n (:51107-51120).
// Elements with a clear mask bit are skipped (:51154-51179). LDS: index at tid*4, value at 1024 + tid*4 (:51210).
// out[row] = best index, or -1 if it is still 0x7fffffff (:51284-51285).
extern "C" __global__ void k_argmax_f32(const float* x, int n, int* out, AmxMasks m) {
  __shared__ float sv[256];
  __shared__ int si[256];
  int tid = threadIdx.x, row = blockIdx.x;
  float bv = __int_as_float(0xff800000);
  int bi = 0x7fffffff;
  const unsigned* mk = mask_base(m.p[row & 7]);
  bool has = row < 8 && mk != nullptr;
  for (int i = tid; i < n; i += blockDim.x) {
    if (has && !in_mask(mk, i)) continue;
    float v = x[(long)row * n + i];
    if (take(v, i, bv, bi)) { bv = v; bi = i; }
  }
  sv[tid] = bv; si[tid] = bi;
  __syncthreads();
  tree_reduce(sv, si, false);
  if (tid == 0) out[row] = si[0] != 0x7fffffff ? si[0] : -1;
}

// k_argmax_f32_legacy(float const*, int, int*, AmxMasks)
// _ZN7halogen12_GLOBAL__N_119k_argmax_f32_legacyEPKfiPiNS_8AmxMasksE, same kernarg.
// Start index -1 (:50928); a candidate replaces the best only if strictly greater (v_cmpx_gt_f32, :51019); the tree
// stage also needs strictly greater (:51063), so ties keep the lower thread. out[row] = si[0] with no -1 mapping (:51085).
extern "C" __global__ void k_argmax_f32_legacy(const float* x, int n, int* out, AmxMasks m) {
  __shared__ float sv[256];
  __shared__ int si[256];
  int tid = threadIdx.x, row = blockIdx.x;
  float bv = __int_as_float(0xff800000);
  int bi = -1;
  const unsigned* mk = mask_base(m.p[row & 7]);
  bool has = row < 8 && mk != nullptr;
  for (int i = tid; i < n; i += blockDim.x) {
    if (has && !in_mask(mk, i)) continue;
    float v = x[(long)row * n + i];
    if (v > bv) { bv = v; bi = i; }
  }
  sv[tid] = bv; si[tid] = bi;
  __syncthreads();
  tree_reduce(sv, si, true);
  if (tid == 0) out[row] = si[0];
}

// k_argmax_part(float const* x, int n, int p, float* vals, int* idx, AmxMasks m)
// _ZN7halogen12_GLOBAL__N_113k_argmax_partEPKfiiPfPiNS_8AmxMasksE, kernarg: x@0, n@8, p@12, vals@16, idx@24, masks@32
// 256-thread blocks; grid (p, rows). part = blockIdx.x (s2), row = blockIdx.y (s3). Thread scan
// i = part*256 + tid, stride p*256 (s_lshl_b32 s12,s9,8, :51296, :51325), row base x + row*n (:51305-51308),
// mask pointer masks[row] (:51310). Tree stages 128..1 (:51409-51571). Thread 0 writes vals[row*p + part] and
// idx[row*p + part] (:51606-51619); the index is stored as is (0x7fffffff when nothing was a candidate).
extern "C" __global__ void k_argmax_part(const float* x, int n, int p, float* vals, int* idx, AmxMasks m) {
  __shared__ float sv[256];
  __shared__ int si[256];
  int tid = threadIdx.x, part = blockIdx.x, row = blockIdx.y;
  float bv = __int_as_float(0xff800000);
  int bi = 0x7fffffff;
  const unsigned* mk = mask_base(m.p[row & 7]);
  bool has = row < 8 && mk != nullptr;
  for (int i = part * 256 + tid; i < n; i += p * 256) {
    if (has && !in_mask(mk, i)) continue;
    float v = x[(long)row * n + i];
    if (take(v, i, bv, bi)) { bv = v; bi = i; }
  }
  sv[tid] = bv; si[tid] = bi;
  __syncthreads();
  tree_reduce(sv, si, false);
  if (tid == 0) {
    vals[row * p + part] = sv[0];
    idx[row * p + part] = si[0];
  }
}

// k_argmax_final(float const* vals, int const* idx, int p, int* out)
// _ZN7halogen12_GLOBAL__N_114k_argmax_finalEPKfPKiiPi, kernarg: vals@0, idx@8, p@16, out@24
// One block of 256 per row (s2 = blockIdx.x). Thread t < p loads vals[row*p + t] and idx[row*p + t] (:51630-51661);
// others start at -inf / 0x7fffffff. Tree stages 128..1 (:51670-51858). out[row] = idx, or -1 for 0x7fffffff (:51874-51876).
extern "C" __global__ void k_argmax_final(const float* vals, const int* idx, int p, int* out) {
  __shared__ float sv[256];
  __shared__ int si[256];
  int tid = threadIdx.x, row = blockIdx.x;
  float v = __int_as_float(0xff800000);
  int i = 0x7fffffff;
  if (tid < p) { v = vals[row * p + tid]; i = idx[row * p + tid]; }
  sv[tid] = v; si[tid] = i;
  __syncthreads();
  tree_reduce(sv, si, false);
  if (tid == 0) out[row] = si[0] != 0x7fffffff ? si[0] : -1;
}
