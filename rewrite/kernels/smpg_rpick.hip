// Rewrite of k_smpg_rpick (family smpg_rpick), codeobj_53000, gfx1151. Listing: codeobj_53000.s:7821-8739.
// Build: hipcc --genco --no-gpu-bundle-output --offload-arch=gfx1151 -O3 -o smpg_rpick.co smpg_rpick.hip
#include "smpg_common.h"

// _ZN7halogen12k_smpg_rpickEPKfS1_PKNS_9SampleRowEPKNS_10SmpScratchES1_S1_S1_S1_PiPfS8_iii
// kernarg: p@0, q@8, rows@16, scratch@24, scanP@32, partP@40, scanR@48, partR@56, id@64, lp@72, cnt@80, n@88, nblk@92, mode@96.
// Single row (rows[0], scratch row 0: s_load_b32 s34,s[20:21] :7830, scratch pointer s[22:23] used without a row offset).
//  1. row max of the block maxima (:7833-7960); Z = serial sum of S.f32[0x100+b] (:7964-7990, lds[1024]);
//     Zk = serial sum of S.f32[0x200+b] (:7994-8020, lds[1028]); R = serial sum of S.f32[0x300+b] (:8024-8051, lds[1032]).
//  2. u = draw_u(rows[0]) (hash :8050-8104, same as smpg_pick); residual path when 0 < R (v_cmp_lt_f32 :8099, s10).
//  3. residual path (0 < R): tot = R (serial sum of S+0xC00, :8111-8126); arrays scanR/partR (args @48, @56);
//     fallback path (R <= 0 or NaN): tot = Zk (serial sum of S+0x800, :8323-8338); arrays scanP/partP (@32, @40).
//     T = u*tot. Walk blocks exactly as smpg_pick (:8130-8155 and :8342-8371): chosen = last block with kb > 0 and
//     cum_before <= T, base = cum before this block. Only the chosen block continues (:8175, :8389).
//  4. thread pick as in smpg_pick: score tid when part > 0 and base+scan[idx-1] <= T (:8186-8207, :8400-8421); max tree;
//     t* = (int)max; only thread t* continues (:8538-8545). If no thread qualifies nothing is written.
//  5. thread t* walks i = idx, step nblk*256, i < n (:8566-8676). w = exp2(((invT*(p[i]-mx))*log2e));
//     kept = key >= Tkey and min_p <= w (:8639-8648).
//       residual path: r = invZk*w if kept else 0 (invZk = (0<Zk) ? 1/Zk : 0), then r = r - q[i]; mode 2: r = pk; mode 1: |pk - q|
//         (:8651-8660)
//       fallback path: r = kept ? w : 0 (:8663)
//     only r > 0 counts: cum += r, result = i, stop at the first cum > T (result of the last visited r > 0 element, :8667-8676,
//     the branches land on the v_mov v0,v2 block at :8611).
//  6. id[0] = result (:8686). lp != 0: lp[0] = fma(invT, p[result] - mx, -ln_adj(Z)) (:8688-8718). cnt != 0 and fallback path:
//     atomic add 1 to cnt[0] (:8719-8738).
extern "C" __global__ void k_smpg_rpick(const float* p, const float* q, const SampleRow* rows, const SmpScratch* scratch,
                                        const float* scanP, const float* partP, const float* scanR, const float* partR,
                                        int* idout, float* lp, int* cnt, int n, int nblk, int mode) {
  __shared__ float lds[256];
  __shared__ float zs, zks, rs, tots, bases;
  __shared__ int chosen;
  int tid = threadIdx.x, bx = blockIdx.x;
  SampleRow row = rows[0];
  const unsigned char* S = (const unsigned char*)scratch;
  float mx = row_max(S, nblk, lds);
  float u = draw_u(row);
  unsigned Tkey = *(const unsigned*)(S + 0x4040);
  if (tid == 0) {
    float z = 0.0f, zk = 0.0f, rr = 0.0f;
    for (int b = 0; b < nblk; b++) z = z + ((const float*)(S + 0x400))[b];
    for (int b = 0; b < nblk; b++) zk = zk + ((const float*)(S + 0x800))[b];
    for (int b = 0; b < nblk; b++) rr = rr + ((const float*)(S + 0xC00))[b];
    bool resid = (0.0f < rr);
    const float* kb_arr = (const float*)(S + (resid ? 0xC00 : 0x800));
    float tot = resid ? rr : zk;
    float T = u * tot;
    float cum = 0.0f, base = 0.0f;
    int ch = 0;
    for (int b = 0; b < nblk; b++) {
      float kb = kb_arr[b];
      bool excl = !(cum <= T);
      if (b == bx) base = cum;
      bool ngt = !(kb > 0.0f);
      cum = cum + kb;
      if (!(ngt || excl)) ch = b;
    }
    zs = z; zks = zk; rs = rr; tots = tot; bases = base; chosen = ch;
  }
  __syncthreads();
  if (bx != chosen) return;
  float Z = zs, Zk = zks, R = rs, tot = tots, base = bases;
  bool resid = (0.0f < R);
  const float* scan = resid ? scanR : scanP;
  const float* part = resid ? partR : partP;
  int idx = bx * 256 + tid;
  float pt = part[idx];
  float cumb = base;
  if (tid != 0) cumb = base + scan[idx - 1];
  float T = u * tot;
  bool bad = !(0.0f < pt) || !(cumb <= T);
  lds[tid] = bad ? -1.0f : (float)tid;
  __syncthreads();
  tree_max(lds);
  int tstar = (int)lds[0];
  __syncthreads();
  if ((unsigned)tid != (unsigned)tstar) return;
  float invT = 1.0f / row.temp;
  float invZk = (0.0f < Zk) ? (1.0f / Zk) : 0.0f;
  int found = -1;
  float cum = cumb;
  for (int i = idx; i < n; i += nblk * 256) {
    float pv = p[i];
    unsigned key = skey(pv);
    float w = wexp(pv, mx, invT);
    bool kept = (Tkey <= key) && (row.min_p <= w);
    float r;
    if (resid) {
      float pk = kept ? invZk * w : 0.0f;
      float d = pk - q[i];
      r = d;
      if (mode == 2) r = pk;
      if (mode == 1) r = __builtin_fabsf(d);
    } else {
      r = kept ? w : 0.0f;
    }
    if (0.0f < r) {
      cum = cum + r;
      found = i;
      if (cum > T) break;
    }
  }
  idout[0] = found;
  if (lp) {
    float d = p[found] - mx;
    lp[0] = __builtin_fmaf(invT, d, -ln_adj(Z));
  }
  if (cnt && !resid) atomicAdd(cnt, 1);
}
