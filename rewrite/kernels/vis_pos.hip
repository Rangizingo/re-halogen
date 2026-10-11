// Rewrite of k_vis_pos_add (bilinear 48x48 position-embedding add), codeobj_53000.s:63968-64175 (kernel at 0x26b400).
// _ZN7halogen12_GLOBAL__N_113k_vis_pos_addEPfPKtiiii, kernarg: x@0 (f32 rows of 1152), pos@8 (bf16 48x48x1152), nrows@16, gh@20, gw@24, mw@28
#include <hip/hip_runtime.h>
// The vendor keeps each separate mul and add separate; every fused op below is an explicit fmaf.
#pragma clang fp contract(off)

__device__ static inline float bf2f(unsigned short b) { return __uint_as_float((unsigned)b << 16); }

// grid.x = row s2; guard s2 >= nrows exits (:63971-63972).
// Row split (same as patchify, :63973-64039): a = s2%2, b = (s2/2)%2, c = (s2/4)%mw, d = s2/(4*mw);
//   ix = 2c + a, iy = 2d + b                                                  (s_lshl1_add_u32 :64034, :64039)
// fy = float(iy) * 47.0f / float(max(gh,2)-1), fx = float(ix) * 47.0f / float(max(gw,2)-1)
//   (s_cvt_f32_i32 :64040-64041, s_mul_f32 0x423c0000 :64045,64047, IEEE division :64049-64075)
// fl = floor(f): lo = clamp(int(fl), 0, 47) (v_med3_i32 :64087,64096), hi = min(max(int(fl),-1)+1, 47) (:64090-64105)
// weights (:64085-64117): fr = f - fl; w_lo = max(1 - |fr|, 0); w_hi = max(1 - |fr - 1|, 0)
// taps (bf16, tap index (y*48+x) * 1152 elements, 0x900 bytes :64122-64126) and weights (:64128-64134):
//   A(ylo,xlo)*wylo*wxlo, C(ylo,xhi)*wylo*wxhi, B(yhi,xlo)*wyhi*wxlo, D(yhi,xhi)*wyhi*wxhi
// per element e = tid; e < 1152 (0x480), e += blockDim (:64159-64160):
//   acc = fma(wA, A, 0); acc = fma(wC, C, acc); acc = fma(wB, B, acc); acc = fma(wD, D, acc)  (:64164-64167)
//   x[e] = acc + x[e]                                                          (:64168-64169)
// Tap register assignment (which of v8..v11 is which tap) is INFERRED from the address math; confirmed only by MATCH.
extern "C" __global__ void k_vis_pos_add(float* x, const unsigned short* pos, int nrows, int gh, int gw, int mw) {
  int s2 = blockIdx.x;
  if (s2 >= nrows) return;
  int a = s2 % 2, b = (s2 / 2) % 2, c = (s2 / 4) % mw, d = s2 / (4 * mw);
  int ix = 2 * c + a, iy = 2 * d + b;
  float hh = (float)(unsigned)(max(gh, 2) - 1), ww = (float)(unsigned)(max(gw, 2) - 1);
  float fy = ((float)iy * 47.0f) / hh;
  float fx = ((float)ix * 47.0f) / ww;
  float fly = floorf(fy), flx = floorf(fx);
  int iyf = (int)fly, ixf = (int)flx;
  int y0 = min(max(iyf, 0), 47), x0 = min(max(ixf, 0), 47);
  int y1 = min(max(iyf, -1) + 1, 47), x1 = min(max(ixf, -1) + 1, 47);
  float fry = fy - fly, frx = fx - flx;
  float wy0 = fmaxf(1.0f - fabsf(fry), 0.0f), wy1 = fmaxf(1.0f - fabsf(fry - 1.0f), 0.0f);
  float wx0 = fmaxf(1.0f - fabsf(frx), 0.0f), wx1 = fmaxf(1.0f - fabsf(frx - 1.0f), 0.0f);
  float wA = wy0 * wx0, wC = wy0 * wx1, wB = wy1 * wx0, wD = wy1 * wx1;
  const unsigned short* pA = pos + (long)(y0 * 48 + x0) * 1152;
  const unsigned short* pC = pos + (long)(y0 * 48 + x1) * 1152;
  const unsigned short* pB = pos + (long)(y1 * 48 + x0) * 1152;
  const unsigned short* pD = pos + (long)(y1 * 48 + x1) * 1152;
  float* row = x + (long)s2 * 1152;
  for (int e = threadIdx.x; e < 1152; e += blockDim.x) {
    float acc = fmaf(wA, bf2f(pA[e]), 0.0f);
    acc = fmaf(wC, bf2f(pC[e]), acc);
    acc = fmaf(wB, bf2f(pB[e]), acc);
    acc = fmaf(wD, bf2f(pD[e]), acc);
    row[e] = acc + row[e];
  }
}
