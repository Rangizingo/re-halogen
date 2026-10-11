// Rewrite of k_pen_scatter_rows_f32 (family pen_scatter), codeobj_53000, gfx1151.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o pen_scatter.co pen_scatter.hip
// Listing: kernels-disasm/codeobj_53000.s lines 4604-4669 (not in git).
#include <hip/hip_runtime.h>

// _ZN7halogen22k_pen_scatter_rows_f32EPfPKiPKfS2_iii
// kernarg: logits@0, ids@8, delta@16, counts@24, a@32, b@36, c@40
// grid (ceil(max_count/blockDim), rows). y = blockIdx.y, ridx = a + b*y (s_mul_i32 s13,s3 ; s_add_i32 s0,s0,s12, :4611-4613).
// i = blockIdx.x*blockDim.x + tid (v_mad_u64_u32, :4621). Guard i < counts[ridx] (v_cmpx_gt_i32 s10 > v0, :4625).
// idx = 0x3ca00*ridx + i (v_mad_i64_i32 with the literal 0x3ca00 = 248320, :4630); id = ids[idx] (:4631-4637).
// Keep only 0 <= id < c (v_cmp_lt_i32 -1 < id :4639 and v_cmp_gt_i32 c > id :4640-4642, exec &= both :4644).
// logits[c*y + id] = delta[idx] + logits[c*y + id] (row stride is c, :4647-4652, loads :4664-4665, add :4667, store :4668).
extern "C" __global__ void k_pen_scatter(float* logits, const int* ids, const float* delta, const int* counts,
                                         int a, int b, int c) {
  int y = blockIdx.y;
  int ridx = a + b * y;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (!(counts[ridx] > i)) return;
  long idx = 248320L * ridx + i;
  int id = ids[idx];
  if (!(id >= 0 && id < c)) return;
  float* p = logits + (long)c * y + id;
  float d = delta[idx];
  float l = *p;
  *p = d + l;
}
