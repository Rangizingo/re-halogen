// Rewrite of k_vis_qkv_split (packed f32 qkv -> bf16 q, k and transposed v), codeobj_53000.s:64380-64480 (kernel at 0x26bd00).
// _ZN7halogen12_GLOBAL__N_115k_vis_qkv_splitEPKfPtS3_S3_i, kernarg: qkv@0 (f32 rows of 3456), q@8, k@16, v@24 (bf16), n@32 (i32)
#include <hip/hip_runtime.h>

// grid.x = row s2, grid.y = head s3, 1D block; guard s2 >= n exits (:64383-64384); threads e = tid, e < 72 (0x48 :64386), e += blockDim (:64459,64463).
//   src = qkv + row*3456 + head*72 + e    (row stride 0x3600 :64399, head stride 0x120 :64400)
//   q = src[0], k = src[1152], v = src[2304]   (K base +0x1200 :64425, Q at K-4608 (offset -512 -0x1000 :64451,64457), V at K+4608 :64454,64458)
//   q[(head*n + row)*72 + e] = hi16(q)    (0x90 bytes per (head,row) :64397,64404; stores :64464)
//   k[(head*n + row)*72 + e] = hi16(k)    (:64466)
//   v[(head*72 + e)*n + row] = hi16(v)    (transposed, v_mad_u64_u32 by n :64421; store :64465)
// All stores are global_store_d16_hi_b16 (truncation).
extern "C" __global__ void k_vis_qkv_split(const float* qkv, unsigned short* q, unsigned short* k, unsigned short* v, int n) {
  int row = blockIdx.x, head = blockIdx.y;
  if (row >= n) return;
  const float* src = qkv + (long)row * 3456 + (long)head * 72;
  for (int e = threadIdx.x; e < 72; e += blockDim.x) {
    long o = ((long)head * n + row) * 72 + e;
    q[o] = (unsigned short)(__float_as_uint(src[e]) >> 16);
    k[o] = (unsigned short)(__float_as_uint(src[e + 1152]) >> 16);
    v[((long)head * 72 + e) * n + row] = (unsigned short)(__float_as_uint(src[e + 2304]) >> 16);
  }
}
