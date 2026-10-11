// Rewrite of k_accept_prob_f32 (family accept), codeobj_53000, gfx1151. Listing: codeobj_53000.s:1321-2490.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o accept.co accept.hip
#include "smpg_common.h"

// _ZN7halogen17k_accept_prob_f32EPKfPKiPKNS_9SampleRowEPfS7_ii
// kernarg: x@0, ids@8, rows@16, out_acc@24, out_lp@32, n@40, mode@44. One block of 256 threads per row, row = blockIdx.x (s2).
// Front half is the k_sample_f32 pipeline (listing :1324-2320): row max (:1344-1506), invT (:1508-1527), Z tree sum (:1530-1661),
// Tkey by four single-block passes (select_tkey, same as k_sample_f32), kept sum = per-thread sequential sum of w over kept elements
// then a tree sum -> Zk (lds[2048], read at :2382).
// Tail, thread 0 only (:2368-2464) is the same arithmetic as k_smpg_accept (see smpg_accept.hip):
//   id = ids[row]; reject (p = 0, lp = -inf) when id < 0 (:2376), id >= n or !(0 < Zk) (:2379-2388);
//   else w = exp2(((invT*(xv-mx))*log2e)), p = w/Zk (IEEE), p = 0 if key < Tkey or !(min_p <= w) (:2437-2446),
//   lp = fma(invT, xv-mx, -ln_adj(Z)) (:2447).
// out_acc[row] = (mode != 3) ? p : 1.0 (:2451-2458); out_lp[row] = lp when out_lp != 0 (:2457-2463).
extern "C" __global__ void k_accept(const float* x, const int* ids, const SampleRow* rows, float* out_acc, float* out_lp, int n, int mode) {
  __shared__ float lds[256];
  int tid = threadIdx.x, r = blockIdx.x;
  SampleRow row = rows[r];
  const float* xr = x + (long)r * n;
  float mx = __int_as_float(0xff800000);
  for (int i = tid; i < n; i += 256) mx = __builtin_fmaxf(mx, xr[i]);
  lds[tid] = mx;
  __syncthreads();
  tree_max(lds);
  mx = lds[0];
  __syncthreads();
  float invT = 1.0f / row.temp;
  float sum = 0.0f;
  for (int i = tid; i < n; i += 256) sum = sum + wexp(xr[i], mx, invT);
  lds[tid] = sum;
  __syncthreads();
  tree_sum(lds);
  float Z = lds[0];
  __syncthreads();
  unsigned Tkey = select_tkey(xr, n, mx, invT, row, Z);
  float acc = 0.0f;
  for (int i = tid; i < n; i += 256) {
    float xv = xr[i];
    unsigned key = skey(xv);
    float w = wexp(xv, mx, invT);
    bool excl = (key < Tkey) || !(row.min_p <= w);
    if (!excl) acc = acc + w;
  }
  lds[tid] = acc;
  __syncthreads();
  tree_sum(lds);
  float Zk = lds[0];
  if (tid != 0) return;
  int id = ids[r];
  float p = 0.0f, lp = __int_as_float(0xff800000);
  if (!(id < 0) && !(!(0.0f < Zk) || id >= n)) {
    float xv = xr[id];
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
