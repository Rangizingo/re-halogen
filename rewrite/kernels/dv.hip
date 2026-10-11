// Rewrites of the "dv" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dv.co dv.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). A ".long 0xD7.." line followed by a bogus
// v_cndmask/v_add_co_ci is one VOP3 instruction, decoded by hand.
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

// halogen::k_dv_scatter(float const*, int const*, float*, int)
// _ZN7halogen12k_dv_scatterEPKfPKiPfi, kernarg: x@0, idx@8, out@16, n@24
// i = blockIdx.x*blockDim.x + threadIdx.x (v_mad_u64_u32 :16135), guard (int)i < n (v_cmpx_gt_i32 :16137).
// k = idx[i] (:16152); out[i] = (k > -1) ? x[k] : -inf  (default 0xff800000 :16153; v_cmpx_lt_i32 -1 < k :16155;
// the load only runs under that mask :16165; store :16172).
extern "C" __global__ void k_dv_scatter(const float* x, const int* idx, float* out, int n) {
  int i = (int)((unsigned long long)blockIdx.x * blockDim.x + threadIdx.x);
  if (!(n > i)) return;
  int k = idx[i];
  float v = __uint_as_float(0xff800000u);
  if (-1 < k) v = x[k];
  out[i] = v;
}

// bf16 -> f32: v_lshlrev_b32 16 / v_and_b32 0xffff0000 (k_dv_quant8 listing, codeobj_53000.s:15100-15127).
__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// halogen::k_dv_quant8(unsigned short const*, int const*, int const*, int, unsigned char*, int)
// _ZN7halogen11k_dv_quant8EPKtPKiS3_iPhi  kernarg: x@0 (bf16 rows of A), dst@8 (int, destination row), src@16 (int, source row),
// n@24, q@32 (u8 table), A@40.   Layout and formulas also in analysis/closure/dv-quant-r3.md.
// tid = blockIdx.x*blockDim.x + threadIdx.x as a 64-bit product (v_mad_u64_u32, codeobj_53000.s:14872); nb = (A + bias) >> 5 blocks per row
// (s_ashr_i32 s9,s2,5, :14875); guard tid < nb*n (64-bit v_cmpx_gt_i64, :14880 region). r = tid / nb, b = tid - r*nb (64-bit divide).
// Source: 32 bf16 at x + src[r]*A + 32*b (four global_load_b128). Destination row = q + dst[r]*S, S = A + A/16 (r3 section 1).
// absmax = max |x| over the 32 values (v_max3_f32 chain from :15128); s = absmax / 127 (IEEE division, div_scale/rcp/fma/div_fmas/div_fixup
// with literal 0x42fe0000 = 127.0, :15189-15204); h = f16(s) (v_cvt_f16_f32, :15207) stored at dst row + A + 2*b (global_store_b16 :16096).
// If !(0x33000000 (2^-25) < s) (v_cmp_nlt_f32 as .long 0xd41e0000, :15208; NaN also takes this path) the 32 quant bytes are 0 (the store takes the high byte of
// the f16 register, which is 0, :15219). Otherwise d = float(h) (v_cvt_f32_f16, :15221) and q[k] = (int8) max(min(rne(x[k]/d), 127), -127)
// with IEEE division, v_rndne_f32, v_minmax_f32 (min with 127.0 held in s1, max with 0xc2fe0000 = -127.0), v_cvt_i32_f32 (:15258-15266, x32).
extern "C" __global__ void k_dv_quant8(const unsigned short* x, const int* dst, const int* src, int n, unsigned char* q, int A) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int nb = (A + (int)((unsigned)(A >> 31) >> 27)) >> 5;
  if (!((long)nb * n > tid)) return;
  long r = tid / nb;
  int b = (int)(tid - r * nb);
  int S = A + ((A + (int)((unsigned)(A >> 31) >> 28)) >> 4);
  const unsigned short* xp = x + (long)src[r] * A + 32 * b;
  unsigned char* row = q + (long)dst[r] * S;
  float v[32], am = 0.0f;
  for (int k = 0; k < 32; k++) { v[k] = bf2f(xp[k]); am = fmaxf(am, fabsf(v[k])); }
  float s = am / 127.0f;
  _Float16 h = (_Float16)s;
  *(_Float16*)(row + A + 2 * b) = h;
  if (!(0x1p-25f < s)) {
    for (int k = 0; k < 32; k++) row[32 * b + k] = 0;
  } else {
    float d = (float)h;
    for (int k = 0; k < 32; k++) {
      float t = rintf(v[k] / d);
      t = fmaxf(fminf(t, 127.0f), -127.0f);
      row[32 * b + k] = (unsigned char)(signed char)(int)t;
    }
  }
}
