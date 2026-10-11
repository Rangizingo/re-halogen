// Rewrite of k_vis_ohead_rows (head-major -> row-major copy), codeobj_53000.s:64792-64866 (kernel at 0x26c600).
// _ZN7halogen12_GLOBAL__N_116k_vis_ohead_rowsEPKfPfi, kernarg: in@0 (f32), out@8 (f32), rows@16 (i32)
#include <hip/hip_runtime.h>

// grid.x = row s2, grid.y = head s3, 1D block; guard s2 >= rows exits (:64795-64796); threads e = tid, e < 72 (0x48 :64798), e += blockDim.
//   src = in  + (head*rows + row) * 72 floats   (0x120 bytes: s_mul :64802-64812)
//   dst = out + row * 1152 floats + head * 72   (0x1200, 0x120 bytes :64805-64825)
//   dst[e] = src[e]   (global_load_b32 :64843, global_store_b32 :64850)
extern "C" __global__ void k_vis_ohead_rows(const float* in, float* out, int rows) {
  int row = blockIdx.x, head = blockIdx.y;
  if (row >= rows) return;
  const float* src = in + ((long)head * rows + row) * 72;
  float* dst = out + (long)row * 1152 + (long)head * 72;
  for (int e = threadIdx.x; e < 72; e += blockDim.x) dst[e] = src[e];
}
