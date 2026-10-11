// Rewrite of halogen::k_dv_quant (NF4 draft-vocab quantizer) from codeobj_53000 (flash_serve, gfx1151).
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o dv_quant.co dv_quant.hip
// Listing: kernels-disasm-v2/codeobj_53000.s (full decode, 0 .long; not in git), kernel at lines 8421-13776. All ":NNNN" cites below are
// lines of that file.
#include <hip/hip_runtime.h>

// Float multiply followed by add stays separate unless an explicit __builtin_fmaf is written.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// halogen::k_dv_quant(unsigned short const*, int const*, int const*, int, unsigned char*, long, int)
// _ZN7halogen10k_dv_quantEPKtPKiS3_iPhli  kernarg: x@0 (bf16 rows of A), dst@8, src@16, n@24, q@32, soff@40 (long), A@48.
// Prologue as in k_dv_quant8: tid = blockIdx*blockDim + threadIdx (v_mad_u64_u32, :8431), nb = (A + bias) >> 5
// (s_ashr_i32 s10,s4,5, :8434), guard nb*n > tid (64-bit v_cmp_gt_i64 :8438), r = tid / nb, b = tid - r*nb (64-bit divide after :8438).
// Source row = x + src[r]*A + 32*b (global_load_b128 x4, :8656-8659, v_mad_i64_i32 src[r]*A :8645).
//
// LUT: 16 bf16-exact floats built from immediates and stored to the thread's 64 B LDS slice (ds_store_b128 x4, :8712-8716):
//   v46..v49 = -1.0, 0xbf320000, 0xbf060000, 0xbeca0000 (v_mov from :8618); v50..v53 = 0xbe920000, 0xbe3d0000, 0xbdba0000, 0;
//   v39..v42 = 0x3da30000, 0x3e250000, 0x3e7c0000, 0x3ead0000; v54..v57 = 0x3ee20000, 0x3f100000, 0x3f390000, 1.0.
// Midpoints (15 f32 literals, ascending; code index = number of midpoints m with m < t, v_cmp_lt_f32 m, t + v_cndmask 0/1 or 0/4
// summed into the LDS byte address, starting after :8790): see MIDB below.
//
// absmax: v_max3_f32 chain over |x| starting max3(|x0|, 0, |x1|) (:8667-8726) -> v33.
// Candidate loop, i = 0..7 (s5; exit test s_cmp_eq_u32 s5, 7 at :8730): s9 = 0xbd4ccccd = -0.05f (:8616);
//   f = fma(float(i), s9, 1.0f) (s_fmaak_f32, :8743), sc = f * absmax (v_mul_f32, :8744);
//   candidate skipped unless 2^-25 < sc (v_cmpx_lt_f32 0x33000000, :8747);
//   h = f16(sc) (v_cvt_f16_f32 :8749), d = float(h) (v_cvt_f32_f16 :8751);
//   t_k = x_k / d (IEEE: div_scale / rcp / fma / div_fmas / div_fixup, :8752-8790), idx_k as above, e_k = fma(-lut[idx_k], d, x_k)
//   (v_fma_mix_f32 -lut, h, x, op_sel_hi:[0,1,0]);
//   err = e1*e1 (v_mul_f32 v41, v42, v42, :8984), err = fma(e0, e0, err) (v_fmac, after :8984), then err = fma(e_k, e_k, err) for k = 2..31 ascending;
//   if err < best (v_cmp_lt_f32 vcc, v41, v38 :10971 and v_cndmask_b16 v0.l :10973) best = err, bh = h.
//   best starts at +inf (v38 = 0x7f800000, :8707); bh starts at f16 0 (v_mov_b16 v0.l, 0, :8711).
// After the loop, the scale f16 bh is stored to q + soff + dst[r]*(2*nb) + 2*b (global_store_b16, :13748; s_lshl_b32 s2,s10,1 :13738).
// If !(0 < bh) (v_cmp_nlt_f16 s0, 0, v0.l, :10989) all 16 code bytes are 0x77 (v_mov_b16 v0.h, 0x77, :10996, :11000);
// otherwise t_k = x_k / float(bh) again, idx_k, byte j = idx_{2j} | idx_{2j+1} << 4 (confirmed by the bitwise MATCH). Bytes go to
// q + dst[r]*(A/2) + 16*b + 64 + j (global_store_b8 / global_store_d16_hi_b8 at offsets 64..79: :10997, :11334, :11336 ... :13747).
__device__ static const unsigned LUTB[16] = {
  0xbf800000u, 0xbf320000u, 0xbf060000u, 0xbeca0000u, 0xbe920000u, 0xbe3d0000u, 0xbdba0000u, 0x00000000u,
  0x3da30000u, 0x3e250000u, 0x3e7c0000u, 0x3ead0000u, 0x3ee20000u, 0x3f100000u, 0x3f390000u, 0x3f800000u};
__device__ static const unsigned MIDB[15] = {
  0xbf590000u, 0xbf1c0000u, 0xbeeb0000u, 0xbeae0000u, 0xbe708000u, 0xbe0d0000u, 0xbd3a0000u, 0x3d230000u,
  0x3df68000u, 0x3e508000u, 0x3e958000u, 0x3ec78000u, 0x3f008000u, 0x3f248000u, 0x3f5c8000u};

__device__ static inline int nf4_idx(float t) {
  int c = 0;
  for (int m = 0; m < 15; m++) c += (__uint_as_float(MIDB[m]) < t) ? 1 : 0;
  return c;
}

extern "C" __global__ void k_dv_quant(const unsigned short* x, const int* dst, const int* src, int n, unsigned char* q, long soff, int A) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  int nb = (A + (int)((unsigned)(A >> 31) >> 27)) >> 5;
  if (!((long)nb * n > tid)) return;
  long r = tid / nb;
  int b = (int)(tid - r * nb);
  int dr = dst[r];
  const unsigned short* xp = x + (long)src[r] * A + 32 * b;
  float v[32], am = 0.0f;
  for (int k = 0; k < 32; k++) { v[k] = bf2f(xp[k]); am = fmaxf(am, fabsf(v[k])); }

  float best = __uint_as_float(0x7f800000u);
  _Float16 bh = (_Float16)0.0f;
  for (int i = 0; i < 8; i++) {
    float f = __builtin_fmaf((float)i, __uint_as_float(0xbd4ccccdu), 1.0f);
    float sc = f * am;
    if (!(0x1p-25f < sc)) continue;
    _Float16 h = (_Float16)sc;
    float d = (float)h;
    float err = 0.0f;
    for (int k = 0; k < 32; k++) {
      int kk = (k == 0) ? 1 : (k == 1 ? 0 : k);
      float e = __builtin_fmaf(-__uint_as_float(LUTB[nf4_idx(v[kk] / d)]), d, v[kk]);
      err = (k == 0) ? e * e : __builtin_fmaf(e, e, err);
    }
    if (err < best) { best = err; bh = h; }
  }

  int S = (A + (int)((unsigned)(A >> 31) >> 31)) >> 1;
  unsigned char* qp = q + (long)dr * S + 16 * b + 64;
  *(_Float16*)(q + soff + (long)dr * (2 * nb) + 2 * b) = bh;
  float d = (float)bh;
  if (!(0.0f < d)) {
    for (int j = 0; j < 16; j++) qp[j] = 0x77;
  } else {
    for (int j = 0; j < 16; j++) {
      int lo = nf4_idx(v[2 * j] / d), hi = nf4_idx(v[2 * j + 1] / d);
      qp[j] = (unsigned char)(lo | (hi << 4));
    }
  }
}
