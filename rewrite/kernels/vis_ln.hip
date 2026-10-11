// Rewrite of k_vis_ln (LayerNorm, f32 in, bf16 out), codeobj_53000.s:63182-63403 (kernel at 0x26a300).
// _ZN7halogen12_GLOBAL__N_18k_vis_lnEPKfPKtS4_Ptif
// kernarg: x@0 (f32 rows), gamma@8 (bf16), beta@16 (bf16), out@24 (bf16), n@32 (i32), eps@36 (f32)
#include <hip/hip_runtime.h>
// The vendor keeps each separate mul and add separate; every fused op below is an explicit fmaf.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// grid.x = row; 1D block, power of two, up to 256 (LDS = 1024 bytes, kernel group_segment).
// row base = x + row*n (s_mul_i32/s_mul_hi_u32 :63189-63192).
// Pass 1: each thread sums x[i] for i = tid, tid+bdim, ... < n, in that order, acc starting at 0 (:63204-63222).
//   sm[tid] = acc (ds_store :63231); tree: for s = bdim/2; s > 0; s >>= 1: if tid < s: sm[tid] = sm[tid+s] + sm[tid] (:63236-63257); barrier each step.
//   mean = sm[0] / float(n)  (IEEE division, :63259-63273)
// Pass 2: acc2 = fma(d, d, acc2) with d = x[i] - mean over the same indices (:63291-63293); same tree reduction (:63299-63332).
//   var = sm[0] / float(n) (IEEE, :63336-63349); v = var + eps (:63351)
//   rs = rsqrt(v) with the denormal guard: if v < 0x00800000 then v *= 0x4b800000 (2^24), rs = v_rsq_f32(v), rs *= 0x45800000 (2^12) (:63352-63357)
// Pass 3: out[i] = trunc_bf16( fma(rs * (x[i] - mean), gamma[i], beta[i]) ) (:63394-63398)
//   gamma = arg1 (loaded to v9 -> v8, :63387,63394), beta = arg2 (v7, :63388,63395); store is global_store_d16_hi_b16.
extern "C" __global__ void k_vis_ln(const float* x, const unsigned short* gamma, const unsigned short* beta,
                                    unsigned short* out, int n, float eps) {
  __shared__ float sm[256];
  const int tid = threadIdx.x, bd = blockDim.x;
  const float* xr = x + (long)blockIdx.x * n;
  unsigned short* orow = out + (long)blockIdx.x * n;
  float acc = 0.0f;
  for (int i = tid; i < n; i += bd) acc = acc + xr[i];
  sm[tid] = acc;
  __syncthreads();
  for (int s = bd >> 1; s > 0; s >>= 1) {
    if (tid < s) sm[tid] = sm[tid + s] + sm[tid];
    __syncthreads();
  }
  float mean = sm[0] / (float)n;
  __syncthreads();
  float acc2 = 0.0f;
  for (int i = tid; i < n; i += bd) { float d = xr[i] - mean; acc2 = fmaf(d, d, acc2); }
  sm[tid] = acc2;
  __syncthreads();
  for (int s = bd >> 1; s > 0; s >>= 1) {
    if (tid < s) sm[tid] = sm[tid + s] + sm[tid];
    __syncthreads();
  }
  float var = sm[0] / (float)n;
  float v = var + eps;
  bool small = v < __uint_as_float(0x00800000u);
  if (small) v = v * __uint_as_float(0x4b800000u);
  float rs = __builtin_amdgcn_rsqf(v);
  if (small) rs = rs * __uint_as_float(0x45800000u);
  for (int i = tid; i < n; i += bd) {
    float t = rs * (xr[i] - mean);
    float r = fmaf(t, bf2f(gamma[i]), bf2f(beta[i]));
    orow[i] = (unsigned short)(__float_as_uint(r) >> 16);
  }
}
