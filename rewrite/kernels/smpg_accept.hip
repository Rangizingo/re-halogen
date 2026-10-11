// Rewrite of k_smpg_accept_final (family smpg_accept), codeobj_53000, gfx1151. Listing: codeobj_53000.s:6725-7054.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o smpg_accept.co smpg_accept.hip
#include "smpg_common.h"

// _ZN7halogen19k_smpg_accept_finalEPKfPKiPKNS_9SampleRowEPKNS_10SmpScratchEPfSA_iii
// kernarg: x@0, ids@8, rows@16, scratch@24, out_acc@32, out_lp@40, n@48, mode@52, nblk@56. grid (1, rows), 256 threads.
// Row = blockIdx.y. Row max of the block maxima (:6744-6866), Z = serial sum S.f32[0x100+b] (:6877-6899),
// Zk = serial sum S.f32[0x200+b] (:6911-6932). Thread 0 then:
//   id = ids[row]; reject path when id < 0 (:6950), or !(0 < Zk), or id >= n (:6952-6958): p = 0, lp = -inf.
//   else xv = x[row*n + id]; key = sortable key (:6983-6985, 7014); lnZ = ln_adj(Z) (:6989-7009);
//   w = exp2(((invT*(xv - mx))*log2e)) with the mul order of :7000-7008; p = w / Zk (IEEE div, :7011-7032);
//   p = 0 when key < Tkey (Tkey at S+0x4040, :6967,7017) or !(min_p <= w) (:7030, 7034-7036);
//   lp = fma(invT, xv - mx, -lnZ) (:7033,7038).
// out_acc[row] = (mode != 3) ? p : 1.0 (:7042-7049); out_lp[row] = lp when out_lp != 0 (:7048-7053).
extern "C" __global__ void k_smpg_accept(const float* x, const int* ids, const SampleRow* rows, const SmpScratch* scratch,
                                         float* out_acc, float* out_lp, int n, int mode, int nblk) {
  __shared__ float lds[256];
  int tid = threadIdx.x, r = blockIdx.y;
  SampleRow row = rows[r];
  const unsigned char* S = SCR_ROW(scratch, r);
  float mx = row_max(S, nblk, lds);
  if (tid != 0) return;
  float Z = 0.0f, Zk = 0.0f;
  for (int b = 0; b < nblk; b++) Z = Z + ((const float*)(S + 0x400))[b];
  for (int b = 0; b < nblk; b++) Zk = Zk + ((const float*)(S + 0x800))[b];
  int id = ids[r];
  float p = 0.0f, lp = __int_as_float(0xff800000);
  if (!(id < 0) && !(!(0.0f < Zk) || id >= n)) {
    float invT = 1.0f / row.temp;
    unsigned Tkey = *(const unsigned*)(S + 0x4040);
    float xv = x[(long)r * n + id];
    unsigned key = skey(xv);
    float lnZ = ln_adj(Z);
    float d = xv - mx;
    float e = invT * d;
    e = e * 1.44269502162933349609375f;
    float w = __builtin_amdgcn_exp2f(e);
    float q = w / Zk;
    bool excl = (key < Tkey) || !(row.min_p <= w);
    p = excl ? 0.0f : q;
    lp = __builtin_fmaf(invT, d, -lnZ);
  }
  out_acc[r] = (mode != 3) ? p : 1.0f;
  if (out_lp) out_lp[r] = lp;
}
