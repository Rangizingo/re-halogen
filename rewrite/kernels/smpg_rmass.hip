// Rewrite of k_smpg_rmass (family smpg_rmass), codeobj_53000, gfx1151. Listing: codeobj_53000.s:7320-7756.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o smpg_rmass.co smpg_rmass.hip
#include "smpg_common.h"

// _ZN7halogen12k_smpg_rmassEPKfS1_PKNS_9SampleRowEPNS_10SmpScratchEPfS7_iii
// kernarg: p@0, q@8, rows@16, scratch@24, out_scan@32, out_part@40, n@48, nblk@52, mode@56. grid (nblk, 1), 256 threads.
// Single row: rows[0] and scratch row 0 (s_load_b32 s28,s[16:17] :7329, scratch used without row offset :7474, 7510).
// Row max of the block maxima (:7331-7466). Zk = serial sum of S.f32[0x200+b], b < nblk (:7472-7490).
// invZ = (0 < Zk) ? 1/Zk : 0 (IEEE 1/Zk :7506,7543, select :7549-7554); invT = 1/temp (:7507,7550); Tkey = S+0x4040 (:7510).
// Per thread over i = bx*256+tid step nblk*256, i < n (:7503, 7596):
//   w = exp2(((invT*(p[i]-mx))*log2e)) (:7569-7581, mul order v10*v14 then 0x3fb8aa3b)
//   pk = invZ*w, zeroed when key(p[i]) < Tkey (:7579) or !(min_p <= w) (:7582) (:7584-7589)
//   r = pk - q[i] (:7594); mode 2: r = pk; mode 1: r = |pk - q| (:7599-7601, s10 = mode==2, s31 = mode==1)
//   acc += r only when 0 < r (:7603-7605)
// Block inclusive scan of acc, Hillis-Steele offsets 1..128, new[t] = old[t-off] + old[t] (:7612-7729).
// out_scan[bx*256+tid] = scan, out_part[...] = acc (:7743-7744); thread 0 writes the block total scan[255] to S+0xC00+4*bx (:7752-7755).
extern "C" __global__ void k_smpg_rmass(const float* p, const float* q, const SampleRow* rows, SmpScratch* scratch,
                                        float* out_scan, float* out_part, int n, int nblk, int mode) {
  __shared__ float lds[256];
  __shared__ float zs;
  int tid = threadIdx.x, bx = blockIdx.x;
  SampleRow row = rows[0];
  unsigned char* S = (unsigned char*)scratch;
  float mx = row_max(S, nblk, lds);
  if (tid == 0) {
    float z = 0.0f;
    for (int b = 0; b < nblk; b++) z = z + ((const float*)(S + 0x800))[b];
    zs = z;
  }
  __syncthreads();
  float Zk = zs;
  float invZ = (0.0f < Zk) ? (1.0f / Zk) : 0.0f;
  float invT = 1.0f / row.temp;
  unsigned Tkey = *(const unsigned*)(S + 0x4040);
  float acc = 0.0f;
  for (int i = bx * 256 + tid; i < n; i += nblk * 256) {
    float pv = p[i], qv = q[i];
    unsigned key = skey(pv);
    float w = wexp(pv, mx, invT);
    bool excl = (key < Tkey) || !(row.min_p <= w);
    float pk = excl ? 0.0f : invZ * w;
    float r = pk - qv;
    if (mode == 2) r = pk;
    if (mode == 1) r = __builtin_fabsf(pk - qv);
    if (0.0f < r) acc = acc + r;
  }
  lds[tid] = acc;
  __syncthreads();
  for (int off = 1; off <= 128; off <<= 1) {
    float a = tid >= off ? lds[tid - off] : 0.0f;
    __syncthreads();
    lds[tid] = a + lds[tid];
    __syncthreads();
  }
  int idx = bx * 256 + tid;
  out_scan[idx] = lds[tid];
  out_part[idx] = acc;
  if (tid == 0) ((float*)(S + 0xC00))[bx] = lds[255];
}
