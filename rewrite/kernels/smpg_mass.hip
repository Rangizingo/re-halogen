// Rewrite of k_smpg_mass (family smpg_mass), codeobj_53000, gfx1151. Listing: codeobj_53000.s:5142-6066.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o smpg_mass.co smpg_mass.hip
#include "smpg_common.h"

// _ZN7halogen11k_smpg_massEPKfPKNS_9SampleRowEPNS_10SmpScratchEPfS7_ii
// kernarg: x@0, rows@8, scratch@16, out_scan@24, out_part@32, n@40, nblk@44. grid (nblk, rows), 256 threads.
//
// Phase 1, final byte select (:5303-5785). Same select as smpg_level<n>, one level further:
//   gates: top_k active iff 0 < top_k < n (:5305-5310), top_p active iff 0 < top_p < 1 (:5317-5322, :5357-5363).
//   neither active: Tkey = 0 (v5 stays 0 from :5299; the store at :5784 happens for block 0 thread 0).
//   else: Z = serial f32 sum of S.f32[0x100 + b], b < nblk (:5328-5347); target = trunc(top_p*Z*2^45) (:5369-5386);
//   state prefix/count/mass from S+0x4008, S+0x4018, S+0x4030 (:5390-5392); level-3 histogram S+0x1c00 / S+0x3800
//   (:5421-5422, offsets 0x1000+3072 and 0x3000+2048); suffix sums; digit d = highest i with
//   (top_k active and top_k <= C'[i]+C0) or (top_p active and M'[i]+M0 >= target) (:5654-5662); Tkey = prefix | d
//   (v_or_b32 v6, v8, v0 :5770). Block 0 thread 0 stores Tkey at S+0x4040 (:5784).
// Phase 2 (:5786-6025): per thread, over i = bx*256+tid step nblk*256 (< n), w = exp2(((x - rowmax)*invT)*log2e);
//   an element is dropped if key < Tkey (v_cmp_lt_u32 :5857) or !(min_p <= w) (v_cmp_nle_f32 :5861);
//   otherwise acc += w (:5863-5867). acc is written to out_part[row*65536 + bx*256 + tid] (:6014).
//   Block inclusive prefix scan of acc over tid, Hillis-Steele offsets 1,2,4,...,128, new[t] = old[t-off] + old[t]
//   (:5873-5997); the scan value goes to out_scan[same index] (:6013); thread 0 writes the block total (scan[255])
//   to S+0x800 + 4*bx (:6022-6025).
extern "C" __global__ void k_smpg_mass(const float* x, const SampleRow* rows, SmpScratch* scratch, float* out_scan,
                                       float* out_part, int n, int nblk) {
  __shared__ float lds[256];
  __shared__ unsigned sc[257];
  __shared__ unsigned long long sm[257];
  __shared__ float zs;
  __shared__ unsigned tkey;
  int tid = threadIdx.x, bx = blockIdx.x, r = blockIdx.y;
  SampleRow row = rows[r];
  unsigned char* S = SCR_ROW(scratch, r);
  float mx = row_max(S, nblk, lds);
  bool ka = topk_active(row, n), pa = topp_active(row);
  if (tid == 0) tkey = 0;
  if (ka || pa) {
    if (tid == 0) {
      float z = 0.0f;
      for (int b = 0; b < nblk; b++) z = z + ((const float*)(S + 0x400))[b];
      zs = z;
    }
    __syncthreads();
    unsigned long long target = pa ? topp_target(row.top_p, zs) : 0ull;
    if (tid == 0) {
      unsigned prefix = *(const unsigned*)(S + 0x4008);
      unsigned c0 = *(const unsigned*)(S + 0x4018);
      unsigned long long m0 = *(const unsigned long long*)(S + 0x4030);
      const unsigned* pc = (const unsigned*)(S + 0x1c00);
      const unsigned long long* pm = (const unsigned long long*)(S + 0x3800);
      sc[256] = 0; sm[256] = 0;
      for (int i = 255; i >= 0; i--) { sc[i] = sc[i + 1] + pc[i]; sm[i] = sm[i + 1] + pm[i]; }
      unsigned d = 0;
      for (int i = 0; i < 256; i++) {
        bool q = (ka && (unsigned)row.top_k <= sc[i] + c0) || (pa && sm[i] + m0 >= target);
        if (q) d = i;
      }
      tkey = d | prefix;
    }
  }
  __syncthreads();
  unsigned Tkey = tkey;
  if (bx == 0 && tid == 0) *(unsigned*)(S + 0x4040) = Tkey;
  float invT = 1.0f / row.temp;
  const float* xr = x + (long)r * n;
  float acc = 0.0f;
  for (int i = bx * 256 + tid; i < n; i += nblk * 256) {
    float xv = xr[i];
    unsigned key = skey(xv);
    float w = wexp(xv, mx, invT);
    bool excl = (key < Tkey) || !(row.min_p <= w);
    if (!excl) acc = acc + w;
  }
  lds[tid] = acc;
  __syncthreads();
  for (int off = 1; off <= 128; off <<= 1) {
    float a = tid >= off ? lds[tid - off] : 0.0f;
    __syncthreads();
    lds[tid] = a + lds[tid];
    __syncthreads();
  }
  long idx = (long)r * 65536 + bx * 256 + tid;
  out_scan[idx] = lds[tid];
  out_part[idx] = acc;
  if (tid == 0) ((float*)(S + 0x800))[bx] = lds[255];
}
