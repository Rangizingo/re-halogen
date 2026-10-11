// Rewrite of k_smpg_max (family smpg_max), codeobj_53000, gfx1151. Listing: codeobj_53000.s:4885-5106.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o smpg_max.co smpg_max.hip
#include "smpg_common.h"

// _ZN7halogen10k_smpg_maxEPKfPNS_10SmpScratchEii   kernarg: x@0, scratch@8, n@16, nblk@20
// grid (nblk, rows), 256 threads. Row scratch S = scratch + blockIdx.y*0x4048 (:4888-4893).
// Block 0 of each row zeroes the histogram areas: u32 at +0x1000/+0x1400/+0x1800/+0x1c00 + 4*tid (:4925-4929)
// and u64 at +0x2000/+0x2800/+0x3000/+0x3800 + 8*tid (:4928,4930-4932).
// Every block: per-thread running max over i = bx*256+tid, stride nblk*256, i < n, start -inf (:4934-4974),
// canonicalizing v_max_f32 (= fmax); LDS tree 128..1 (:4984-5088); thread 0 writes S.f32[bx] (:5105).
// x row base = x + row*n (:4943-4951).
extern "C" __global__ void k_smpg_max(const float* x, SmpScratch* scratch, int n, int nblk) {
  __shared__ float lds[256];
  int tid = threadIdx.x, bx = blockIdx.x, row = blockIdx.y;
  unsigned char* S = SCR_ROW(scratch, row);
  if (bx == 0) {
    for (int k = 0; k < 4; k++) {
      ((unsigned*)(S + 0x1000 + 0x400 * k))[tid] = 0u;
      ((unsigned long long*)(S + 0x2000 + 0x800 * k))[tid] = 0ull;
    }
  }
  float mx = __int_as_float(0xff800000);
  const float* xr = x + (long)row * n;
  for (int i = bx * 256 + tid; i < n; i += nblk * 256) mx = __builtin_fmaxf(mx, xr[i]);
  lds[tid] = mx;
  __syncthreads();
  tree_max(lds);
  if (tid == 0) ((float*)S)[bx] = lds[0];
}
