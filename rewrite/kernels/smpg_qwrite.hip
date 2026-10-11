// Rewrite of k_smpg_qwrite (family smpg_qwrite), codeobj_53000, gfx1151. Listing: codeobj_53000.s:7073-7315.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o smpg_qwrite.co smpg_qwrite.hip
#include "smpg_common.h"

// _ZN7halogen13k_smpg_qwriteEPKfPKNS_9SampleRowEPKNS_10SmpScratchEPfii
// kernarg: x@0, rows@8, scratch@16, q@24, n@32, nblk@36. grid (nblk, 1): the kernel never reads blockIdx.y, uses rows[0] and
// the first scratch row (s_load_b32 s6, s[6:7] :7081; scratch pointer used without a row offset :7085, 7214).
// Row max of the block maxima (:7082-7206); Z = serial sum of S.f32[0x100+b] (:7212-7230).
// q[i] = (1/Z) * exp2(((x[i] - mx) * (1/T)) * log2e) for i = bx*256+tid step nblk*256, i < n (:7242-7308), with
// IEEE 1/Z (v_div_scale at :7246) and 1/T (:7245). Mul order: v7*(x-mx) (:7303), *0x3fb8aa3b (:7304), exp2, then 1/Z * that (:7307).
extern "C" __global__ void k_smpg_qwrite(const float* x, const SampleRow* rows, const SmpScratch* scratch, float* q, int n, int nblk) {
  __shared__ float lds[256];
  __shared__ float zs;
  int tid = threadIdx.x, bx = blockIdx.x;
  float temp = rows[0].temp;
  const unsigned char* S = (const unsigned char*)scratch;
  float mx = row_max(S, nblk, lds);
  if (tid == 0) {
    float z = 0.0f;
    for (int b = 0; b < nblk; b++) z = z + ((const float*)(S + 0x400))[b];
    zs = z;
  }
  __syncthreads();
  float invZ = 1.0f / zs;
  float invT = 1.0f / temp;
  for (int i = bx * 256 + tid; i < n; i += nblk * 256) {
    float v = x[i] - mx;
    v = invT * v;
    v = v * 1.44269502162933349609375f;
    float w = __builtin_amdgcn_exp2f(v);
    q[i] = invZ * w;
  }
}
