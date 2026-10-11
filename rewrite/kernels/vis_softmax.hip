// Rewrite of k_vis_softmax (row softmax, f32 in, bf16 hi/lo out), codeobj_53000.s:64495-64757 (kernel at 0x26c000).
// _ZN7halogen12_GLOBAL__N_113k_vis_softmaxEPKfPtS3_if
// kernarg: s@0 (f32 rows), hi@8 (bf16), lo@16 (bf16, may be null), n@24 (i32), scale@28 (f32)
#include <hip/hip_runtime.h>
// The vendor keeps each separate mul and add separate; every fused op below is an explicit fmaf.
#pragma clang fp contract(off)

// grid.x = row; 1D block, power of two, up to 256 (LDS 1024 bytes). row base = row*n elements (:64498-64499).
// Pass 1 (max): m = -inf (0xff800000, :64509); for i = tid; i < n; i += bdim: m = max(canon(m), scale * s[i]) (:64541-64544).
//   tree: for st = bdim/2; st > 0; st >>= 1: if tid < st: sm[tid] = max(canon(sm[tid]), canon(sm[tid+st])) (:64583-64589). max = sm[0].
// Pass 2 (sum): e = exp2( fma(scale, s[i], -max) * 0x3fb8aa3b ) via v_exp_f32; sum += e, same index order (:64615-64618);
//   tree sum sm[tid] = sm[tid+st] + sm[tid] (:64652-64656).
// Pass 3: inv = 1.0f / sum (IEEE, :64666-64677); p = inv * exp2(fma(scale, s[i], -max) * log2e) (:64701-64704);
//   hi[i] = hi16(p) (truncation, :64705); if lo != 0: lo[i] = hi16(p - (p & 0xffff0000)) (:64707-64712).
// Note pass 1 multiplies (scale * s) and passes 2/3 fuse (fma); both kept.
extern "C" __global__ void k_vis_softmax(const float* s, unsigned short* hi, unsigned short* lo, int n, float scale) {
  __shared__ float sm[256];
  const int tid = threadIdx.x, bd = blockDim.x;
  const float* sr = s + (long)blockIdx.x * n;
  unsigned short* hr = hi + (long)blockIdx.x * n;
  unsigned short* lr = lo ? lo + (long)blockIdx.x * n : lo;
  float m = __uint_as_float(0xff800000u);
  for (int i = tid; i < n; i += bd) m = fmaxf(m, scale * sr[i]);
  sm[tid] = m;
  __syncthreads();
  for (int st = bd >> 1; st > 0; st >>= 1) {
    if (tid < st) sm[tid] = fmaxf(sm[tid], sm[tid + st]);
    __syncthreads();
  }
  float mx = sm[0];
  __syncthreads();
  float sum = 0.0f;
  for (int i = tid; i < n; i += bd)
    sum = sum + __builtin_amdgcn_exp2f(fmaf(scale, sr[i], -mx) * __uint_as_float(0x3fb8aa3bu));
  sm[tid] = sum;
  __syncthreads();
  for (int st = bd >> 1; st > 0; st >>= 1) {
    if (tid < st) sm[tid] = sm[tid + st] + sm[tid];
    __syncthreads();
  }
  float inv = 1.0f / sm[0];
  for (int i = tid; i < n; i += bd) {
    float p = inv * __builtin_amdgcn_exp2f(fmaf(scale, sr[i], -mx) * __uint_as_float(0x3fb8aa3bu));
    hr[i] = (unsigned short)(__float_as_uint(p) >> 16);
    if (lr) lr[i] = (unsigned short)(__float_as_uint(p - __uint_as_float(__float_as_uint(p) & 0xffff0000u)) >> 16);
  }
}
