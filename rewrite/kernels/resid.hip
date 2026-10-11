// Rewrite of k_resid_sample_f32 (family resid), codeobj_53000, gfx1151. Listing: codeobj_53000.s:2891-4603.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o resid.co resid.hip
#include "smpg_common.h"

// _ZN7halogen18k_resid_sample_f32EPKfS1_PKNS_9SampleRowEPiPfS5_ii
// kernarg: p@0, q@8, rows@16, id@24, lp@32, cnt@40, n@48, mode@52. One block of 256 threads, one row: rows[0] (s_load_b256 s[36:43],
// s[24:25] :2900, no block offset), outputs id[0], lp[0].
// Front half is the k_sample_f32 pipeline on p: row max (:2905-3075), invT (:3077-3096), Z tree sum (:3098-3229), Tkey by four
// single-block passes (select_tkey), kept sum Zk = per-thread sum over kept elements then tree sum (:3798-3938, lds[2048]).
// invZk = (0 < Zk) ? 1/Zk : 0 (:3940-3959).
// Residual values (per thread, over i = tid step 256, :3962-4020): w = exp2(((invT*(p-mx))*log2e)); kept = key >= Tkey and
// min_p <= w; pk = kept ? invZk*w : 0; r = pk - q[i]; mode 2: r = pk; mode 1: |pk - q|; acc += r only when 0 < r.
// Inclusive Hillis-Steele scan over threads (offsets 1..128, new[t] = old[t-off] + old[t], :4024-4144); R = scan[255].
// Residual path when 0 < R (s9, branch at :4163). Otherwise (R <= 0 or NaN, fallback): per thread acc = fma(invZk, w, acc) over kept
// elements (:4209), second scan (:4218-4340), total = its scan[255].
// Draw (:4341-4396): u = draw_u(rows[0]); T = u*total; excl = P[tid] - acc; the thread with T >= excl and T < P walks its elements
// from i = tid with cum = excl (:4390-4396):
//   residual path r as above, fallback path r = pk (:4450-4470, branch on s9); r > 0 only: cum += r, result = i, stop when cum > T
//   (result of the last visited r > 0 element, -1 if none; :4473-4482).
// id[0] = result (:4488). lp != 0: lp[0] = fma(invT, p[result]-mx, -ln_adj(Z)) (:4487-4520). cnt != 0 and fallback: cnt[0] += 1 (:4521-4540).
// Nothing is written when no thread qualifies.
extern "C" __global__ void k_resid(const float* p, const float* q, const SampleRow* rows, int* idout, float* lp, int* cnt, int n, int mode) {
  __shared__ float lds[256];
  int tid = threadIdx.x;
  SampleRow row = rows[0];
  const float* xr = p;
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
  __syncthreads();
  float invZk = (0.0f < Zk) ? (1.0f / Zk) : 0.0f;
  float racc = 0.0f;
  for (int i = tid; i < n; i += 256) {
    float xv = xr[i];
    unsigned key = skey(xv);
    float w = wexp(xv, mx, invT);
    bool excl = (key < Tkey) || !(row.min_p <= w);
    float pk = excl ? 0.0f : invZk * w;
    float d = pk - q[i];
    float r = d;
    if (mode == 2) r = pk;
    if (mode == 1) r = __builtin_fabsf(d);
    if (0.0f < r) racc = racc + r;
  }
  lds[tid] = racc;
  __syncthreads();
  for (int off = 1; off <= 128; off <<= 1) {
    float a = tid >= off ? lds[tid - off] : 0.0f;
    __syncthreads();
    lds[tid] = a + lds[tid];
    __syncthreads();
  }
  float R = lds[255];
  __syncthreads();
  bool resid = (0.0f < R);
  float useacc = racc;
  if (!resid) {
    float facc = 0.0f;
    for (int i = tid; i < n; i += 256) {
      float xv = xr[i];
      unsigned key = skey(xv);
      float w = wexp(xv, mx, invT);
      bool excl = (key < Tkey) || !(row.min_p <= w);
      if (!excl) facc = __builtin_fmaf(invZk, w, facc);
    }
    lds[tid] = facc;
    __syncthreads();
    for (int off = 1; off <= 128; off <<= 1) {
      float a = tid >= off ? lds[tid - off] : 0.0f;
      __syncthreads();
      lds[tid] = a + lds[tid];
      __syncthreads();
    }
    useacc = facc;
  }
  float P = lds[tid];
  float total = lds[255];
  float u = draw_u(row);
  float T = u * total;
  float ex = P - useacc;
  bool mine = (T >= ex) && (T < P);
  if (!mine) return;
  int found = -1;
  float cum = ex;
  for (int i = tid; i < n; i += 256) {
    float xv = xr[i];
    unsigned key = skey(xv);
    float w = wexp(xv, mx, invT);
    bool excl = (key < Tkey) || !(row.min_p <= w);
    float pk = excl ? 0.0f : invZk * w;
    float r = pk;
    if (resid) {
      float d = pk - q[i];
      r = d;
      if (mode == 2) r = pk;
      if (mode == 1) r = __builtin_fabsf(d);
    }
    if (0.0f < r) {
      cum = cum + r;
      found = i;
      if (cum > T) break;
    }
  }
  idout[0] = found;
  if (lp) {
    float d = xr[found] - mx;
    lp[0] = __builtin_fmaf(invT, d, -ln_adj(Z));
  }
  if (cnt && !resid) atomicAdd(cnt, 1);
}
