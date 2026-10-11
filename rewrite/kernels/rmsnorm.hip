// Rewrites of the "rmsnorm" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o rmsnorm.co rmsnorm.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git).
#include <hip/hip_runtime.h>

// No implicit mul+add fusion: the vendor uses v_fmac only where noted.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// f32 -> bf16 RNE, no NaN special case (v_bfe_u32 ; v_add3_u32 0x7fff ; store high half), used by grouped and gated
// (codeobj_53000.s:29398-29400, 29738-29740).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// NaN-aware variant used by k_rmsnorm_gated_w: NaN -> (bits | 0x400000) >> 16 (inferred from the v_cmp_u_f32-style
// .long words at codeobj_53000.s:30002-30013 and checked by the NaN-heavy spec).
__device__ static inline unsigned short f2bf_rne_nan(float f) {
  unsigned u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u | 0x400000u) >> 16);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// 1 / sqrt(mean + 1e-6) as the vendor builds it:
//   v = ssq / (float)n  (IEEE division, v_div_scale/rcp/fma/div_fmas/div_fixup, :29322-29339)
//   v += 0x358637bd (1e-6f, :29344)
//   if v < 0x800000: v *= 2^24 (0x4b800000, :29349-29354); r = v_rsq_f32(v) (:29356); if small r *= 2^12 (0x45800000, :29357-29359)
__device__ static inline float inv_rms(float ssq, int n) {
  float v = ssq / (float)n;
  v = v + __uint_as_float(0x358637bdu);
  bool small = v < __uint_as_float(0x00800000u);
  if (small) v = v * __uint_as_float(0x4b800000u);
  float r = __builtin_amdgcn_rsqf(v);
  if (small) r = r * __uint_as_float(0x45800000u);
  return r;
}

// sigmoid(z) = 1 / (1 + exp2(z * -1.442695)), IEEE division (:29708-29736).
__device__ static inline float sigm(float z) {
  float e = __builtin_amdgcn_exp2f(z * -1.44269502162933349609375f);
  return 1.0f / (1.0f + e);
}

// Block reduction as the vendor: each thread stores its partial in LDS, then for s = blockDim/2, blockDim/4, ... 1:
// threads tid < s do lds[tid] = lds[tid + s] + lds[tid], barrier (:29305-29313, 29414-29435). Result in lds[0].
__device__ static inline float block_sum(float part, float* lds) {
  int tid = threadIdx.x;
  lds[tid] = part;
  __syncthreads();
  for (unsigned s = blockDim.x >> 1; s != 0; s >>= 1) {
    if (tid < (int)s) lds[tid] = lds[tid + s] + lds[tid];
    __syncthreads();
  }
  return lds[0];
}

// k_rmsnorm_grouped(unsigned short const* x, unsigned short const* w, unsigned short* out, int n, int groups)
// _ZN7halogen12_GLOBAL__N_117k_rmsnorm_groupedEPKtS2_Ptii, kernarg: x@0, w@8, out@16, n@24, groups@28
// Block b normalizes n elements x[b*n ..] (s_mul_i32 s12 = n*b, :29244) with weight row (b % groups) (the unsigned
// reciprocal division by s11 at :29225-29243, remainder :29340-29352, times n :29353).
// ssq = sum x^2 with v_fmac_f32 per element, thread stride blockDim (:29297); block_sum; out = bf16((r * x) * (1 + w)):
// v_mul r*x (:29390), v_add_f32 1.0 + w (:29391), v_mul (:29393), RNE (:29398-29400).
extern "C" __global__ void k_rmsnorm_grouped(const unsigned short* x, const unsigned short* w, unsigned short* out, int n, int groups) {
  __shared__ float lds[1024];
  long base = (long)blockIdx.x * n;
  int g = blockIdx.x % groups;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = bf2f(x[base + i]);
    acc = __builtin_fmaf(v, v, acc);
  }
  float ssq = block_sum(acc, lds);
  float r = inv_rms(ssq, n);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = bf2f(x[base + i]) ;
    float t = r * v;
    float ww = 1.0f + bf2f(w[(long)g * n + i]);
    out[base + i] = f2bf_rne(t * ww);
  }
}

// k_rmsnorm_gated(x, w, z, out, int n)
// _ZN7halogen12_GLOBAL__N_115k_rmsnorm_gatedEPKtS2_S2_Pti, kernarg: x@0, w@8, z@16, out@24, n@32
// Block b: row x[b*n ..] (:29571-29576). ssq as above (v_fmac, :29624). out = bf16(((r * x) * w) * sigmoid(z)):
// loads x from arg0 (:29703), w from arg8 (:29704), z from arg16 (:29692); products :29714-29717, :29736; RNE :29738-29740.
extern "C" __global__ void k_rmsnorm_gated(const unsigned short* x, const unsigned short* w, const unsigned short* z,
                                           unsigned short* out, int n) {
  __shared__ float lds[1024];
  long base = (long)blockIdx.x * n;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float v = bf2f(x[base + i]);
    acc = __builtin_fmaf(v, v, acc);
  }
  float ssq = block_sum(acc, lds);
  float r = inv_rms(ssq, n);
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    float t = r * bf2f(x[base + i]);
    t = t * bf2f(w[i]);
    float sg = sigm(bf2f(z[base + i]));
    out[base + i] = f2bf_rne(t * sg);
  }
}

// k_rmsnorm_gated_w(x, w, z, out, long rows, int n)
// _ZN7halogen12_GLOBAL__N_117k_rmsnorm_gated_wEPKtS2_S2_Ptli, kernarg: x@0, w@8, z@16, out@24, rows@32, n@40
// One 32-lane wave per row: row = blockIdx.x*8 + (tid >> 5) (:29774-29779), guarded by rows > row as unsigned 64-bit (:29784).
// Each lane handles 4 elements lane, lane+32, lane+64, lane+96 of the row (byte offsets 0/64/128/192, :29815-29818);
// the row stride is n (v_mad_u64_u32 row*n, :29796) and the mean divides by n (:29855, :29929), but only 128 elements are touched.
// ssq per lane = (e0^2 + e2^2) + (e1^2 + e3^2) with v_mul then v_add (:29826-29853). Wave sum by ds_bpermute with offsets
// 16, 8, 4, 2, 1 (lane reads itself when out of range; :29862-29927), broadcast from lane 0 (:29927).
// out_j = bf16(((r * x_j) * w_j) * sigmoid(z_j)) (:29989-30004), NaN-aware bf16 rounding (:30002-30028).
extern "C" __global__ void k_rmsnorm_gated_w(const unsigned short* x, const unsigned short* w, const unsigned short* z,
                                             unsigned short* out, long rows, int n) {
  long row = (long)blockIdx.x * 8 + (threadIdx.x >> 5);
  int lane = threadIdx.x & 31;
  if ((unsigned long)row >= (unsigned long)rows) return;
  long base = row * n;
  float xv[4], sq[4];
  for (int j = 0; j < 4; j++) {
    xv[j] = bf2f(x[base + lane + 32 * j]);
    sq[j] = xv[j] * xv[j];
  }
  float p = (sq[0] + sq[2]) + (sq[1] + sq[3]);
  p = p + __shfl_down(p, 16, 32);
  p = p + __shfl_down(p, 8, 32);
  p = p + __shfl_down(p, 4, 32);
  p = p + __shfl_down(p, 2, 32);
  p = p + __shfl_down(p, 1, 32);
  float ssq = __shfl(p, 0, 32);
  float r = inv_rms(ssq, n);
  for (int j = 0; j < 4; j++) {
    float t = r * xv[j];
    t = t * bf2f(w[lane + 32 * j]);
    float sg = sigm(bf2f(z[base + lane + 32 * j]));
    out[base + lane + 32 * j] = f2bf_rne_nan(t * sg);
  }
}
