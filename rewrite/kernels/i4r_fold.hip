// Rewrite of the "i4r_fold" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o i4r_fold.co i4r_fold.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). VOP3 instructions appear as ".long <dword1>" plus a
// mis-decoded second dword; they are decoded by hand.
// Cite note: kernel start/end lines in codeobj_53000.s are exact (see rewrite/reports/g22.md). Line numbers inside a kernel that are written
// as :NNNNN were read off a filtered re-listing; for the long unrolled kernels they can be off by tens of lines. Instruction names are exact.
#include <hip/hip_runtime.h>

// The vendor keeps v_mul_f32 and v_add_f32 separate; stop the compiler from fusing them into FMAs.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// f32 -> bf16, RNE without a NaN special case: v_bfe_u32 v,v,16,1 ; v_add3_u32 v,v,v,0x7fff, high half kept
// (k_i4r_fold :26346+ at v_bfe_u32 v13, v0, 16, 1 and the v_add3_u32 lines before the store).
__device__ static inline unsigned short f2bf_rne(float f) {
  unsigned u = __float_as_uint(f);
  return (unsigned short)((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

// Butterfly across lanes with xor distance m: x' = (lane has bit m ? -x : x) + x[lane ^ m]. The vendor uses
// v_cndmask_b32 (-x, x) + v_add_f32 with DPP row_xmask:m (m = 1,2,4,8) and v_permlanex16_b32 (m = 16).
__device__ static inline float xstep(float v, int m) {
  float partner = __shfl_xor(v, m);
  return ((threadIdx.x & m) ? -v : v) + partner;
}

// Shared tail of both kernels. Lane l of a wave holds acc[0..3] for elements 128*j + 4l + k of row t.
//  1. 4-point transform in the lane: p0=a0+a1, p1=a0-a1, p2=a2+a3, p3=a2-a3; e0=p0+p2, e2=p0-p2, e1=p1+p3, e3=p1-p3
//     (k_i4r_fold listing, the v_add/v_sub block after the last v_dual_add_f32 chain).
//  2. xstep for m = 1,2,4,8,16, then * 0x3db504f3 (1/sqrt(128)).
//  3. y_k = bf16_rne( half(h[k]) * e_k + (-0.0) )  (v_fma_mix_f32 ..., neg(0) op_sel_hi:[1,0,0]; the third operand is the
//     inline constant 0 with the neg modifier, read as -0.0), then out_k = bf16_rne( fma(z[t], bf16(r[t][k]), y_k) )
//     (v_fmac_f32 v5, v2, v12 with the bf16 widened by v_lshlrev 16 / v_and 0xffff0000).
__device__ static inline void fold_tail(float acc[4], const _Float16* h, const unsigned short* r, float zt,
                                        unsigned short* o, int ix) {
  float p0 = acc[0] + acc[1], p1 = acc[0] - acc[1], p2 = acc[2] + acc[3], p3 = acc[2] - acc[3];
  float e[4] = {p0 + p2, p1 + p3, p0 - p2, p1 - p3};
  for (int m = 1; m <= 16; m <<= 1)
    for (int k = 0; k < 4; k++) e[k] = xstep(e[k], m);
  float C = __uint_as_float(0x3db504f3u);
  for (int k = 0; k < 4; k++) {
    float v = e[k] * C;
    float y = bf2f(f2bf_rne(__builtin_fmaf((float)h[ix + k], v, -0.0f)));
    o[k] = f2bf_rne(__builtin_fmaf(zt, bf2f(r[k]), y));
  }
}

// halogen::(anonymous namespace)::k_i4r_fold(float const*, int, float const*, _Float16 const*, unsigned short const*,
//                                            float const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_110k_i4r_foldEPKfiS2_PKDF16_PKtS2_Pt  kernarg: x@0, n@8, sc@16, h@24, r@32, z@40, out@48
// One wave (32 lanes) per item w = blockIdx.x*8 + (tid>>5) (v_lshl_or_b32 v2, s2, 3, v1, codeobj_53000.s:26349-26350), guard
// w < n*20 (s_mul_i32 s3,s3,20, v_cmpx_gt_i32 :26352-26353). t = w/20 (magic 0x66666667, >>3), j = w - 20t (:26357-26364).
// ix = j*128 + 4*lane. x is [n][10][2560] floats: slab s at byte 0x2800*(10t+s) + 4*ix (v_mad_i64_i32 0x2800 :26381; the 20 float4 loads
// at base+0x0,0x2800,0x5000,... :26384-26470). sc[10t+s] scalar per slab (global_load_b32 offsets 0,4,..,36 from sc + 40t).
// m_s[k] = sc[10t+s] * x_s[k] (v_mul_f32), acc[k] = ((0 + m_0) + m_1) + ... + m_9 in ascending s, starting from +0.0
// (v_add_f32 v15, 0, v15 :26486 and the v_dual_add_f32 chain :26487-26510).
// h: 4 halves at h[ix..ix+3] (global_load_b64 from kernarg+24 + 2*ix). r: bf16 [n][2560] row stride 0x1400 B (v_mad_i64_i32 0x1400, s[8:9] :26471).
// z[t] one float (global_load_b32 from kernarg+40 + 4t). out: bf16 [n][2560].
extern "C" __global__ void k_i4r_fold(const float* x, int n, const float* sc, const _Float16* h, const unsigned short* r,
                                      const float* z, unsigned short* out) {
  int lane = threadIdx.x & 31;
  int w = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (!(n * 20 > w)) return;
  int t = w / 20, j = w - 20 * t;
  int ix = j * 128 + 4 * lane;
  const float* xb = x + (long)t * 10 * 2560 + ix;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int s = 0; s < 10; s++) {
    float scs = sc[t * 10 + s];
    float4 v = *(const float4*)(xb + (long)s * 2560);
    float vv[4] = {v.x, v.y, v.z, v.w};
    for (int k = 0; k < 4; k++) acc[k] = acc[k] + scs * vv[k];
  }
  fold_tail(acc, h, r + (long)t * 2560 + ix, z[t], out + (long)t * 2560 + ix, ix);
}

// halogen::(anonymous namespace)::k_i4r_fold_rows(unsigned short const*, int const*, int, float const*, _Float16 const*,
//                                                 unsigned short const*, float const*, unsigned short*)
// _ZN7halogen12_GLOBAL__N_115k_i4r_fold_rowsEPKtPKiiPKfPKDF16_S2_S6_Pt
// kernarg: x@0 (bf16 rows of 2560), idx@8, n@16, sc@24, h@32, r@40, z@48, out@56
// Same item mapping (guard n*20, :26704-26711). The 10 slabs come from rows: row_s = idx[10t+s] (global_load_b32 offsets 0..36 from idx + 40t,
// :26725, :26742...), a 4-element bf16 load (global_load_b64) at x + row_s*0x1400 B + 2*ix (v_mad_i64_i32 0x1400 :26728).
// m_s[k] = sc[10t+s] * bf16_to_f32(x) (v_lshlrev 16 / v_and 0xffff0000, v_mul_f32), accumulated as in k_i4r_fold from +0.0
// (v_dual_add_f32 v9, 0, v13 :26868).
extern "C" __global__ void k_i4r_fold_rows(const unsigned short* x, const int* idx, int n, const float* sc, const _Float16* h,
                                           const unsigned short* r, const float* z, unsigned short* out) {
  int lane = threadIdx.x & 31;
  int w = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (!(n * 20 > w)) return;
  int t = w / 20, j = w - 20 * t;
  int ix = j * 128 + 4 * lane;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int s = 0; s < 10; s++) {
    long row = idx[t * 10 + s];
    float scs = sc[t * 10 + s];
    const unsigned short* xp = x + row * 2560 + ix;
    for (int k = 0; k < 4; k++) acc[k] = acc[k] + scs * bf2f(xp[k]);
  }
  fold_tail(acc, h, r + (long)t * 2560 + ix, z[t], out + (long)t * 2560 + ix, ix);
}
