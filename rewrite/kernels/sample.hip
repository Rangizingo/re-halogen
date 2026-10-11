// Rewrite of k_sample_f32 (family sample), codeobj_53000, gfx1151. Listing: codeobj_53000.s:6-1320.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o sample.co sample.hip
#include "smpg_common.h"

// _ZN7halogen12k_sample_f32EPKfPKNS_9SampleRowEPiPfS6_i
// kernarg: x@0, rows@8, id@16, lp_all@24, lp_kept@32, n@40. One block of 256 threads per row, row = blockIdx.x (s2, :13).
// Same math as the smpg_* chain, done in one block:
//  A. row max: per-thread fmax over i = tid step 256 (:30-62), LDS tree (:66-205), mx.
//  B. invT = 1/temp (IEEE, :208-227); per-thread sum of w = exp2(((invT*(x-mx))*log2e)) (:257-263); tree sum -> Z (:273-360).
//  C. gates: top_k active iff 0 < top_k < n, top_p active iff 0 < top_p < 1 (:363-379). Neither: Tkey = 0 (v4 = 0, :362).
//     Else target = trunc(top_p*Z*2^45) as u64 (:385-403) and four histogram+select passes (:404-927):
//       pass LV (1..4): zero the LDS digit count and mass histograms; every element (pass 1) or every element with
//       (key ^ prefix) < 2^(32-8*LV)... i.e. 0x1000000, 0x10000, 0x100 (:587, :840) takes part; digit = (key >> (32-8*LV)) & 0xff
//       (:462, :590, bfe 8,8, and 0xff :843); count += 1 and mass += mass45(w) (:469-470, :607-608, :861-862).
//       Thread 0 then scans digits 255 down to 0 adding count and mass to the carried (count, mass); it stops at the first
//       digit where (top_k active and top_k <= count) or (top_p active and mass >= target) (v_cmp_ge_u64 :519, :645, :900),
//       or at digit 0 (:518-532). The carry before that digit and prefix |= digit << (32-8*LV) are kept (:533-538, :659-664, :914-918);
//       the last pass stores Tkey = prefix | digit (:914).
//  D. kept sum: per thread acc += w for elements with key >= Tkey and min_p <= w (:952-982); inclusive Hillis-Steele scan over
//     threads, offsets 1..128 (:986-1099) -> P[tid]; Zk = P[255] (:1150).
//  E. u = draw_u(row); T = u*Zk (:1153). Thread with T < P[tid] and T >= P[tid] - acc[tid] (:1155-1160) walks its elements from
//     i = tid step 256 with cum = P[tid] - acc: kept elements set result = i, cum += w, stop when cum > T (:1201-1228).
//  F. id[row] = result (:1250). s = invT*(x[row*n+result] - mx) (:1248-1254). lp_all != 0: s - ln_adj(Z) (:1256-1279);
//     lp_kept != 0: s - ln_adj(Zk) (:1282-1305). x is read even for result -1.
// If no thread qualifies in E nothing is written.
extern "C" __global__ void k_sample(const float* x, const SampleRow* rows, int* ids, float* lp_all, float* lp_kept, int n) {
  __shared__ float lds[256];
  __shared__ unsigned hc[256];
  __shared__ unsigned long long hm[256];
  __shared__ unsigned st_prefix, st_count, tkey_s;
  __shared__ unsigned long long st_mass;
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
  bool ka = topk_active(row, n), pa = topp_active(row);
  unsigned Tkey = 0;
  if (ka || pa) {
    unsigned long long target = pa ? topp_target(row.top_p, Z) : 0ull;
    if (tid == 0) { st_prefix = 0; st_count = 0; st_mass = 0; }
    __syncthreads();
    for (int lv = 1; lv <= 4; lv++) {
      const int shift = 24 - 8 * (lv - 1);
      hc[tid] = 0u;
      hm[tid] = 0ull;
      __syncthreads();
      unsigned prefix = st_prefix;
      const unsigned lim = lv == 1 ? 0u : (1u << (32 - 8 * (lv - 1)));
      for (int i = tid; i < n; i += 256) {
        float xv = xr[i];
        unsigned key = skey(xv);
        if (lv > 1 && !((key ^ prefix) < lim)) continue;
        float w = wexp(xv, mx, invT);
        unsigned d = (key >> shift) & 0xffu;
        atomicAdd(&hc[d], 1u);
        atomicAdd(&hm[d], mass45(w));
      }
      __syncthreads();
      if (tid == 0) {
        unsigned c = st_count;
        unsigned long long m = st_mass;
        unsigned cb = c, d = 255;
        unsigned long long mb = m;
        for (int i = 255; i >= 0; i--) {
          cb = c; mb = m;
          c = c + hc[i];
          m = m + hm[i];
          d = i;
          bool stop = (ka && (unsigned)row.top_k <= c) || (pa && m >= target);
          if (stop) break;
        }
        st_prefix = (d << shift) | prefix;
        st_count = cb;
        st_mass = mb;
      }
      __syncthreads();
    }
    Tkey = st_prefix;
  }
  // D: kept sum and scan
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
  for (int off = 1; off <= 128; off <<= 1) {
    float a = tid >= off ? lds[tid - off] : 0.0f;
    __syncthreads();
    lds[tid] = a + lds[tid];
    __syncthreads();
  }
  float P = lds[tid];
  float Zk = lds[255];
  float u = draw_u(row);
  float T = u * Zk;
  float ex = P - acc;
  bool mine = (T < P) && (T >= ex);
  if (!mine) return;
  int found = -1;
  float cum = ex;
  for (int i = tid; i < n; i += 256) {
    float xv = xr[i];
    unsigned key = skey(xv);
    float w = wexp(xv, mx, invT);
    if (key >= Tkey && row.min_p <= w) {
      cum = cum + w;
      found = i;
      if (cum > T) break;
    }
  }
  ids[r] = found;
  float s = invT * (xr[found] - mx);
  if (lp_all) lp_all[r] = s - ln_adj(Z);
  if (lp_kept) lp_kept[r] = s - ln_adj(Zk);
}
