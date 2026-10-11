// Rewrite of the "i4r_gu" family from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o i4r_gu.co i4r_gu.hip
// Listing: kernels-disasm/codeobj_53000.s (not in git). VOP3 instructions appear as ".long <dword1>" plus a
// mis-decoded second dword; they are decoded by hand.
// Cite note: kernel start/end lines in codeobj_53000.s are exact (see rewrite/reports/g22.md). Line numbers inside a kernel that are written
// as :NNNNN were read off a filtered re-listing; for the long unrolled kernels they can be off by tens of lines. Instruction names are exact.
#include <hip/hip_runtime.h>

// Keep float multiply and add separate unless an explicit __builtin_fmaf is written (hipcc fuses a*b+c by default).
#pragma clang fp contract(off)

// 128-point Walsh-Hadamard step shared by both transforms: butterfly across lanes with xor distance m:
// x' = (lane has bit m ? -x : x) + x[lane ^ m]. The vendor does v_cndmask_b32 (-x, x, pred) then v_add_f32 with a
// DPP row_xmask:m source (m = 1,2,4,8, e.g. codeobj_53000.s:22722-22900) and v_permlanex16_b32 for m = 16.
__device__ static inline float xstep(float v, int m) {
  float partner = __shfl_xor(v, m);
  return ((threadIdx.x & m) ? -v : v) + partner;
}

// halogen::(anonymous namespace)::k_i4r_gu_epi(float const*, int, _Float16 const*, _Float16 const*, _Float16*)
// _ZN7halogen12_GLOBAL__N_112k_i4r_gu_epiEPKfiPKDF16_S4_PDF16_  kernarg: g@0, n@8, a@16, b@24, out@32. Block of 256 =
// 8 waves of 32 lanes; one wave handles one item w = blockIdx.x*8 + (tid>>5) (v_lshl_or_b32 v1, s2, 3, tid>>5, :22609-22610),
// guard w < n*5 (s_mul_i32 s3,s3,5; v_cmpx_gt_i32, :22612-22614). t = w/5 (v_mul_hi_i32 0x66666667 magic), j = w - 5t.
// g row t = 1280 floats at 0x1400 B per row (v_mad_i64_i32 0x1400 :22632). Lane l reads 4 floats: gate g[t][j*128 + 4l..]
// and up g[t][640 + j*128 + 4l..] (two global_load_b128, second at +2560 B, :22640-22641).
// a (half): ag = a[j*128 + 4l + k], au = a[640 + j*128 + 4l + k] (global_load_b64 at +0 and +1280 B, :22662-22663);
// b (half): b[j*128 + 4l + k] (global_load_b64 :22668).
// Step 1 (gate and up each): 4-point transform in a lane: p0=x0+x1, p1=x0-x1, p2=x2+x3, p3=x2-x3 (:22645-22660),
//   e0=p0+p2, e2=p0-p2, e1=p1+p3, e3=p1-p3; then xstep for m = 1,2,4,8,16 (:22680-22790); then * 0x3db504f3 (1/sqrt(128)).
// Step 2: gs = ag * gate_h, us = au * up_h as v_fma_mix_f32 (half, f32, -0.0) (:22810-22835), silu = gs / (1 + exp2(gs * -1.442695))
//   with v_exp_f32 and the IEEE div_scale/rcp/fma/div_fmas/div_fixup sequence (:22840-22900); prod = us * silu.
// Step 3: pair (0,1) and (2,3): t = fma(b_hi, prod_hi, -0.0); s = fma(prod_lo, b_lo, t); d = fma(prod_lo, b_lo, -t)
//   (v_fma_mix_f32 ... op_sel_hi:[0,1,0], :22915-22921); e0 = s01+s23, e2 = s01-s23, e1 = d01+d23, e3 = d01-d23 (:22922-22927).
//   xstep for m = 1,2,4,8,16 (:22928-22990), then out[k] = f16(fma(e_k, 0x3db504f3, +0.0)) (v_fma_mixlo/hi_f16 ..., s2, 0 :23000-23008),
//   stored as one 8 B write to out + t*640 + j*128 + 4l (v_mad_i64_i32 0x500 :22968, global_store_b64 :23010).
extern "C" __global__ void k_i4r_gu_epi(const float* g, int n, const _Float16* a, const _Float16* b, _Float16* out) {
  int lane = threadIdx.x & 31;
  int w = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (!(n * 5 > w)) return;
  int t = w / 5, j = w - 5 * t;
  const float* gr = g + (long)t * 1280 + j * 128 + lane * 4;
  float4 gv = *(const float4*)gr;
  float4 uv = *(const float4*)(gr + 640);
  float x[4] = {gv.x, gv.y, gv.z, gv.w};
  float y[4] = {uv.x, uv.y, uv.z, uv.w};
  int ix = j * 128 + lane * 4;
  float C = __uint_as_float(0x3db504f3u);
  float e[4], f[4];
  for (int pass = 0; pass < 2; pass++) {
    float* v = pass ? y : x;
    float* r = pass ? f : e;
    float p0 = v[0] + v[1], p1 = v[0] - v[1], p2 = v[2] + v[3], p3 = v[2] - v[3];
    r[0] = p0 + p2; r[2] = p0 - p2; r[1] = p1 + p3; r[3] = p1 - p3;
    for (int m = 1; m <= 16; m <<= 1)
      for (int k = 0; k < 4; k++) r[k] = xstep(r[k], m);
    for (int k = 0; k < 4; k++) r[k] = r[k] * C;
  }
  float prod[4];
  for (int k = 0; k < 4; k++) {
    float gs = __builtin_fmaf((float)a[ix + k], e[k], -0.0f);
    float us = __builtin_fmaf((float)a[640 + ix + k], f[k], -0.0f);
    float ex = __builtin_amdgcn_exp2f(gs * -1.44269502162933349609375f);
    float silu = gs / (1.0f + ex);
    prod[k] = us * silu;
  }
  float bb[4];
  for (int k = 0; k < 4; k++) bb[k] = (float)b[ix + k];
  float t1 = __builtin_fmaf(bb[1], prod[1], -0.0f), t3 = __builtin_fmaf(bb[3], prod[3], -0.0f);
  float s01 = __builtin_fmaf(prod[0], bb[0], t1), d01 = __builtin_fmaf(prod[0], bb[0], -t1);
  float s23 = __builtin_fmaf(prod[2], bb[2], t3), d23 = __builtin_fmaf(prod[2], bb[2], -t3);
  float h[4] = {s01 + s23, d01 + d23, s01 - s23, d01 - d23};
  for (int m = 1; m <= 16; m <<= 1)
    for (int k = 0; k < 4; k++) h[k] = xstep(h[k], m);
  _Float16* o = out + (long)t * 640 + ix;
  for (int k = 0; k < 4; k++) o[k] = (_Float16)__builtin_fmaf(h[k], C, 0.0f);
}
