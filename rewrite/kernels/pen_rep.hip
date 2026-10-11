// Rewrite of k_pen_rep_rows_f32 (family pen_rep), codeobj_53000, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o pen_rep.co pen_rep.hip
// Listing: kernels-disasm/codeobj_53000.s lines 4720-4844 (not in git).
#include <hip/hip_runtime.h>
// x*f then +B are separate v_mul and v_add in the listing; keep the compiler from fusing them.
#pragma clang fp contract(off)

// _ZN7halogen18k_pen_rep_rows_f32EPfPKiPKfS4_S4_PKhS4_S2_iii
// kernarg: logits@0, ids@8, delta@16, A@24, B@32, flags(u8)@40, rep@48, counts@56, a@64, b@68, c@72
// Indexing is the same as pen_scatter: ridx = a + b*y (:4727-4729), guard i < counts[ridx] (:4741),
// idx = 0x3ca00*ridx + i (:4747), id = ids[idx], keep 0 <= id < c (:4756-4761), logit slot = logits + c*y + id (:4763-4781).
// rep = rep[ridx] (s_load_b32 from s16 + ridx*4, :4773-4777).
// rep == 1.0 (s_cmp_neq_f32 s0,1.0 / s_cbranch_scc0, :4783-4784): out = delta[idx] + logit (:4836-4839).
// else (NaN rep lands here, neq is true):
//   x = logit + A[idx]                                            (:4797)
//   if flags[idx] != 0 (global_load_d16_u8, v_cmpx_ne_u16, :4794,4799):
//     x = x * ((0 < x) ? 1/rep : rep)   IEEE 1/rep (div_scale/rcp/fma/div_fixup, :4801-4819), select :4817-4820, mul :4822
//   x = x + B[idx]                                                (:4830)
// store x to the logit slot (:4840).
extern "C" __global__ void k_pen_rep(float* logits, const int* ids, const float* delta, const float* A, const float* B,
                                     const unsigned char* flags, const float* rep, const int* counts,
                                     int a, int b, int c) {
  int y = blockIdx.y;
  int ridx = a + b * y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(counts[ridx] > i)) return;
  long idx = 248320L * ridx + i;
  int id = ids[idx];
  if (!(id >= 0 && id < c)) return;
  float* p = logits + (long)c * y + id;
  float r = rep[ridx];
  float x;
  if (!(r != 1.0f)) {
    float d = delta[idx];
    float l = *p;
    x = d + l;
  } else {
    float l = *p;
    float av = A[idx];
    x = l + av;
    if (flags[idx] != 0) {
      float f = (0.0f < x) ? (1.0f / r) : r;
      x = x * f;
    }
    x = x + B[idx];
  }
  *p = x;
}
