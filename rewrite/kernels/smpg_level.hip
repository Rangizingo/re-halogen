// Rewrites of k_smpg_level<0..3> (family smpg_level), codeobj_53000, gfx1151.
// Listing: level0 codeobj_53000.s:2406825-2407221, level1 2407222-2407964, level2 2407965-2408730, level3 2408731-2409480.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o smpg_level.co smpg_level.hip
#include "smpg_common.h"

// kernarg (all four): x@0, rows(SampleRow*)@8, scratch@16, n@24, nblk@28. grid (nblk, rows), 256 threads.
// Row r = blockIdx.y: SampleRow = rows + r (shl 5, :2406831), scratch S = scratch + r*0x4048 (:2406837-2406840).
// invT = 1.0f / row.temp, IEEE division (div_scale/rcp/fma/div_fmas/div_fixup, :2406997-2407027).

// Per-block histogram pass shared by the four kernels. Elements i = bx*256+tid, stride nblk*256, i < n.
// A key takes part only when (key ^ prefix) < lim (level 1: 0x1000000, 2: 0x10000, 3: 0x100; level 0 has no test).
// Digit = (key >> shift) & 0xff. count += 1, mass += mass45(w) in LDS, then flushed with global atomic adds when non-zero
// (level0 :2407091-2407120; level1 :2407945-2407961).
template <int LV>
__device__ static inline void hist_pass(const float* xr, int n, int nblk, float mx, float invT, unsigned prefix,
                                        unsigned char* S, unsigned* hc, unsigned long long* hm) {
  int tid = threadIdx.x, bx = blockIdx.x;
  static_assert(LV >= 1, "level 0 has its own loop");
  const int shift = 24 - 8 * LV;
  const unsigned lim = 1u << (32 - 8 * LV);
  for (int i = bx * 256 + tid; i < n; i += nblk * 256) {
    float xv = xr[i];
    unsigned key = skey(xv);
    if (!((key ^ prefix) < lim)) continue;  // :2407918-2407920
    float w = wexp(xv, mx, invT);
    unsigned d = (key >> shift) & 0xffu;
    atomicAdd(&hc[d], 1u);
    atomicAdd(&hm[d], mass45(w));
  }
}

extern "C" __global__ void k_smpg_level0(const float* x, const SampleRow* rows, SmpScratch* scratch, int n, int nblk) {
  __shared__ float lds[256];
  __shared__ unsigned hc[256];
  __shared__ unsigned long long hm[256];
  int tid = threadIdx.x, bx = blockIdx.x, r = blockIdx.y;
  SampleRow row = rows[r];
  unsigned char* S = SCR_ROW(scratch, r);
  float mx = row_max(S, nblk, lds);                                  // :2406842-2406963
  bool active = topk_active(row, n) || topp_active(row);              // :2406971-2406985
  float invT = 1.0f / row.temp;
  hc[tid] = 0u;
  hm[tid] = 0ull;
  __syncthreads();
  float sum;
  // level 0 needs the per-thread sum even when the filter is off; the histogram is skipped then
  {
    float s = 0.0f;
    const float* xr = x + (long)r * n;
    for (int i = bx * 256 + tid; i < n; i += nblk * 256) {
      float xv = xr[i];
      float w = wexp(xv, mx, invT);
      s = s + w;
      if (active) {
        unsigned key = skey(xv);
        atomicAdd(&hc[key >> 24], 1u);
        atomicAdd(&hm[key >> 24], mass45(w));
      }
    }
    sum = s;
  }
  __syncthreads();
  if (active) {
    if (hc[tid] != 0u) atomicAdd((unsigned*)(S + 0x1000) + tid, hc[tid]);
    if (hm[tid] != 0ull) atomicAdd((unsigned long long*)(S + 0x2000) + tid, hm[tid]);
  }
  lds[tid] = sum;
  __syncthreads();
  tree_sum(lds);
  if (tid == 0) ((float*)(S + 0x400))[bx] = lds[0];                  // :2407218
}

// Levels 1..3. Steps:
//  1. no filter active: return (:2407250).
//  2. row max (as level 0); Z = sum over b < nblk of S.f32[0x100 + b], serial f32 adds from 0 (:2407391-2407408).
//  3. target = trunc(top_p*Z*2^45) as u64 when top_p is active, else 0 (:2407424-2407441).
//  4. state: level 1 starts from prefix 0, count 0, mass 0 (:2407445-2407450); levels 2,3 load prefix/count/mass from
//     S+0x4000+4*(LV-2), S+0x4010+4*(LV-2), S+0x4020+8*(LV-2) (level2 :156-158 of its listing, level3 likewise).
//  5. previous level histogram (count S+0x1000+0x400*(LV-1), mass S+0x2000+0x800*(LV-1)) goes through an inclusive
//     suffix sum (Hillis-Steele steps 1..128, :2407484-2407684): C'[i] = sum_{j>=i} count[j], M'[i] likewise (wrapping).
//  6. a digit i qualifies when (top_k active and top_k <= C'[i] + C0) or (top_p active and M'[i] + M0 >= target)
//     (:2407704-2407714, v_cmp_ge_u64 at :2407706). d = highest qualifying digit, 0 if none (max tree :2407715-2407801).
//  7. new state: prefix' = (d << (32-8*LV)) | prefix, C0' = C'[d+1] + C0, M0' = M'[d+1] + M0 (C'[256] = 0)
//     (:2407815-2407823). Block 0 thread 0 stores it to S+0x4000+4*(LV-1) (prefix), +0x4010+4*(LV-1) (count),
//     +0x4020+8*(LV-1) (mass) (:2407834-2407843).
//  8. histogram pass over keys whose higher bytes match prefix', digit = byte (3-LV) of the key; flush with atomics to
//     S+0x1000+0x400*LV and S+0x2000+0x800*LV.
template <int LV>
__device__ static inline void level_n(const float* x, const SampleRow* rows, SmpScratch* scratch, int n, int nblk) {
  __shared__ float lds[256];
  __shared__ unsigned hc[256];
  __shared__ unsigned long long hm[256];
  __shared__ unsigned sc[257];
  __shared__ unsigned long long sm[257];
  __shared__ unsigned st[3];  // prefix, count, (mass in sst)
  __shared__ unsigned long long sst;
  __shared__ float zs;
  __shared__ unsigned dsel;
  int tid = threadIdx.x, bx = blockIdx.x, r = blockIdx.y;
  SampleRow row = rows[r];
  unsigned char* S = SCR_ROW(scratch, r);
  bool ka = topk_active(row, n), pa = topp_active(row);
  if (!(ka || pa)) return;
  float mx = row_max(S, nblk, lds);
  if (tid == 0) {
    float z = 0.0f;
    for (int b = 0; b < nblk; b++) z = z + ((const float*)(S + 0x400))[b];
    zs = z;
  }
  __syncthreads();
  unsigned long long target = pa ? topp_target(row.top_p, zs) : 0ull;
  if (tid == 0) {
    if (LV == 1) { st[0] = 0; st[1] = 0; sst = 0; }
    else {
      st[0] = *(const unsigned*)(S + 0x4000 + 4 * (LV - 2));
      st[1] = *(const unsigned*)(S + 0x4010 + 4 * (LV - 2));
      sst = *(const unsigned long long*)(S + 0x4020 + 8 * (LV - 2));
    }
    // suffix sums of the previous level histogram
    const unsigned* pc = (const unsigned*)(S + 0x1000 + 0x400 * (LV - 1));
    const unsigned long long* pm = (const unsigned long long*)(S + 0x2000 + 0x800 * (LV - 1));
    sc[256] = 0; sm[256] = 0;
    for (int i = 255; i >= 0; i--) { sc[i] = sc[i + 1] + pc[i]; sm[i] = sm[i + 1] + pm[i]; }
    unsigned d = 0;
    for (int i = 0; i < 256; i++) {
      bool q = (ka && (unsigned)row.top_k <= sc[i] + st[1]) || (pa && sm[i] + sst >= target);
      if (q) d = i;
    }
    dsel = d;
    unsigned np = (d << (32 - 8 * LV)) | st[0];
    unsigned nc = sc[d + 1] + st[1];
    unsigned long long nm = sm[d + 1] + sst;
    st[0] = np; st[1] = nc; sst = nm;
    if (bx == 0) {
      *(unsigned*)(S + 0x4000 + 4 * (LV - 1)) = np;
      *(unsigned*)(S + 0x4010 + 4 * (LV - 1)) = nc;
      *(unsigned long long*)(S + 0x4020 + 8 * (LV - 1)) = nm;
    }
  }
  __syncthreads();
  unsigned prefix = st[0];
  float invT = 1.0f / row.temp;
  hc[tid] = 0u;
  hm[tid] = 0ull;
  __syncthreads();
  hist_pass<LV>(x + (long)r * n, n, nblk, mx, invT, prefix, S, hc, hm);
  __syncthreads();
  if (hc[tid] != 0u) atomicAdd((unsigned*)(S + 0x1000 + 0x400 * LV) + tid, hc[tid]);
  if (hm[tid] != 0ull) atomicAdd((unsigned long long*)(S + 0x2000 + 0x800 * LV) + tid, hm[tid]);
}

extern "C" __global__ void k_smpg_level1(const float* x, const SampleRow* rows, SmpScratch* s, int n, int nblk) { level_n<1>(x, rows, s, n, nblk); }
extern "C" __global__ void k_smpg_level2(const float* x, const SampleRow* rows, SmpScratch* s, int n, int nblk) { level_n<2>(x, rows, s, n, nblk); }
extern "C" __global__ void k_smpg_level3(const float* x, const SampleRow* rows, SmpScratch* s, int n, int nblk) { level_n<3>(x, rows, s, n, nblk); }
