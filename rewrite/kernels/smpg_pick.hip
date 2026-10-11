// Rewrite of k_smpg_pick (family smpg_pick), codeobj_53000, gfx1151. Listing: codeobj_53000.s:6067-6724.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o smpg_pick.co smpg_pick.hip
#include "smpg_common.h"

// _ZN7halogen11k_smpg_pickEPKfPKNS_9SampleRowEPKNS_10SmpScratchES1_S1_PiPfS9_ii
// kernarg: x@0, rows@8, scratch@16, scan@24, part@32, ids@40, lp_all@48, lp_kept@56, n@64, nblk@68
// grid (nblk, rows), 256 threads.
//  1. row max of the block maxima (:6087-6214); Z = serial sum of S.f32[0x100+b] (:6220-6242).
//  2. u = draw_u(row); Tkey = S+0x4040 (:6287).
//  3. thread 0: Zk = serial sum of S.f32[0x200+b] (:6300-6321); T = Zk*u (:6324). Walk blocks b = 0..nblk-1 with a running
//     f32 cum from 0: excl = !(cum <= T); base = cum when b == blockIdx.x; cum += kb;
//     chosen = b unless (!(kb > 0) or excl) (:6341-6359). Only the chosen block continues (:6377-6379).
//  4. each thread: p = part[row*65536 + idx]; cumb = base (+ scan[row*65536 + idx - 1] when tid != 0) (:6393-6407);
//     score = tid when p > 0 and cumb <= T else -1 (:6410-6419); max tree (:6420-6523); t* = (int)max (:6525).
//  5. thread t* only: walk its elements i = idx, idx+nblk*256, ... < n: keep when key >= Tkey and min_p <= w;
//     cum += w; the result is the index of the last kept element visited: the first one with cum > T (loop stops), or
//     the final kept element when cum never passes T; -1 when no element is kept (:6538-6629, branches at 0x2196F0,
//     0x219708 and 0x219710 all land on the block at 0x219640 that does v_mov v0, v2).
//  6. ids[row] = result (:6651); if lp_all != 0: lp_all[row] = s - ln_adj(Z); if lp_kept != 0: lp_kept[row] = s - ln_adj(Zk),
//     with s = (x[row*n + result] - rowmax) * (1/T) (:6649-6706). The x load happens even for result -1.
extern "C" __global__ void k_smpg_pick(const float* x, const SampleRow* rows, const SmpScratch* scratch, const float* scan,
                                       const float* part, int* ids, float* lp_all, float* lp_kept, int n, int nblk) {
  __shared__ float lds[256];
  __shared__ float zs, zks, bases;
  __shared__ int chosen;
  int tid = threadIdx.x, bx = blockIdx.x, r = blockIdx.y;
  SampleRow row = rows[r];
  const unsigned char* S = SCR_ROW(scratch, r);
  float mx = row_max(S, nblk, lds);
  float u = draw_u(row);
  float invT = 1.0f / row.temp;
  unsigned Tkey = *(const unsigned*)(S + 0x4040);
  if (tid == 0) {
    float z = 0.0f, zk = 0.0f;
    for (int b = 0; b < nblk; b++) z = z + ((const float*)(S + 0x400))[b];
    for (int b = 0; b < nblk; b++) zk = zk + ((const float*)(S + 0x800))[b];
    float T = zk * u;
    float cum = 0.0f, base = 0.0f;
    int ch = 0;
    for (int b = 0; b < nblk; b++) {
      float kb = ((const float*)(S + 0x800))[b];
      bool excl = !(cum <= T);
      if (b == bx) base = cum;
      bool ngt = !(kb > 0.0f);
      cum = cum + kb;
      if (!(ngt || excl)) ch = b;
    }
    zs = z; zks = zk; bases = base; chosen = ch;
  }
  __syncthreads();
  if (bx != chosen) return;
  float Z = zs, Zk = zks, base = bases;
  int idx = bx * 256 + tid;
  float p = part[(long)r * 65536 + idx];
  float cumb = base;
  if (tid != 0) cumb = base + scan[(long)r * 65536 + idx - 1];
  float T = u * Zk;
  bool bad = !(0.0f < p) || !(cumb <= T);
  lds[tid] = bad ? -1.0f : (float)tid;
  __syncthreads();
  tree_max(lds);
  int tstar = (int)lds[0];
  __syncthreads();
  if ((unsigned)tid != (unsigned)tstar) return;
  const float* xr = x + (long)r * n;
  int found = -1;
  float cum = cumb;
  for (int i = idx; i < n; i += nblk * 256) {
    float xv = xr[i];
    unsigned key = skey(xv);
    float w = wexp(xv, mx, invT);
    if (Tkey <= key && row.min_p <= w) {
      cum = cum + w;
      found = i;  // v_mov_b32 v0, v2 runs for every kept element, both branches reach it (:6582-6583, 6627, 6629)
      if (cum > T) break;
    }
  }
  ids[r] = found;
  float s = invT * (xr[found] - mx);
  if (lp_all) lp_all[r] = s - ln_adj(Z);
  if (lp_kept) lp_kept[r] = s - ln_adj(Zk);
}
